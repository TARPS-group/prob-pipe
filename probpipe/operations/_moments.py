"""The distribution functionals: ``mean``, ``variance``, ``cov``, ``quantile``, and ``expectation``.

Each summarizes a distribution by a deterministic value. ``mean``, ``variance``,
``cov``, and ``quantile`` carry a capability route on their matching protocol,
``closed_form``, and a Monte Carlo fallback, ``monte_carlo``, on the event kinds
where the required averaging is defined. A moment of the event's kind keeps the
event's components and packaging and derives only its term specs, support
included.

``expectation(d, f)`` returns ``E[f(X)]`` for ``X ~ d``. The methods that can
compute it form a dispatch registry keyed on the distribution's type, and the
operation's one route delegates to it. The registry's selection order decides
which method runs:

1. ``exact``: the distribution's own ``_expectation``, for a law that claims
   :class:`~probpipe.distributions._capabilities.SupportsExpectation`, which in
   practice means finite support.
2. ``monte_carlo``: the average of ``f`` over independent draws, for any law
   that samples. It is the default approximate method.

Exact methods rank before approximate ones, so an exact method is taken
whenever one applies. Another method, such as quasi-Monte Carlo or quadrature,
joins by registering with ``expectation_method_registry.register``. A caller
selects it with ``method=``, and ``set_priorities`` makes it the default.

The draws of every Monte Carlo route are workflow-owned random events, and
their number is the ``n_broadcast_samples`` control.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..core._batch import Batch, BatchSpec
from ..core._broadcast_distributions import SAMPLE_LEVEL
from ..core._dispatch import (
    Feasibility,
    MathematicalDomainError,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from ..core._record_batch import _batch_class_for
from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec
from ..core._specs import OutputSpec
from ..core.constraints import (
    _Boolean,
    _IntegerInterval,
    _NonNegativeInteger,
    _Sphere,
    interval,
    non_negative,
    unit_interval,
)
from ..custom_types import PRNGKey
from ..distributions import _distribution as _base
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
    _capability_guard,
)
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import _raw_record
from ..functions import _broker, _descendants, function
from ..values import Function, FunctionSpec
from ._operation import ApplicabilityError, BoundCall, _workflow_draws, operation
from ._sample import _record_batch

__all__ = [
    "ExpectationMethod",
    "cov",
    "expectation",
    "expectation_method_registry",
    "mean",
    "quantile",
    "variance",
]


# ---------------------------------------------------------------------------
# The expectation method registry
# ---------------------------------------------------------------------------


class ExpectationMethod(UnaryDispatchMethod):
    """A method that computes ``E[f(X)]`` for ``X ~ d``.

    A subclass declares ``name``, ``exact``, a ``priority``, and the
    distribution types it admits, and implements ``check`` and ``execute``
    over ``(d, f, **options)``. The options are ``num_evaluations`` and ``key``,
    which a method that does not sample ignores.
    """

    def supported_types(self) -> tuple[type, ...]:
        """Every distribution; ``check`` decides which it applies to."""
        return (Distribution,)


class _ExactExpectation(ExpectationMethod):
    """The distribution's own exact ``_expectation``."""

    @property
    def name(self) -> str:
        return "exact"

    @property
    def exact(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return 100

    def check(self, d: Any, f: Callable[[Any], Any], /, **options: Any) -> Feasibility:
        """Feasible when *d* claims ``SupportsExpectation`` and its guard admits the call."""
        if not isinstance(d, SupportsExpectation):
            return Feasibility(False, f"{type(d).__name__} has no exact expectation")
        return _capability_guard(d, "_expectation")

    def execute(self, d: Any, f: Callable[[Any], Any], /, **options: Any) -> Any:
        """``d._expectation(f)``."""
        return d._expectation(f)


class _MonteCarloExpectation(ExpectationMethod):
    """The average of ``f`` over independent draws of the distribution.

    ``num_evaluations`` sets the number of draws, defaulting to
    ``probpipe.distributions._distribution.DEFAULT_NUM_EVALUATIONS``. Without a
    ``key``, the draws are workflow-owned random events, so a surrounding
    workflow's seed and replay govern them.
    """

    @property
    def name(self) -> str:
        return "monte_carlo"

    @property
    def exact(self) -> bool:
        return False

    @property
    def priority(self) -> int:
        return 50

    def check(self, d: Any, f: Callable[[Any], Any], /, **options: Any) -> Feasibility:
        """Feasible when *d* samples."""
        if not isinstance(d, SupportsSampling):
            return Feasibility(False, f"{type(d).__name__} does not sample")
        return _capability_guard(d, "_sample")

    def execute(
        self,
        d: Any,
        f: Callable[[Any], Any],
        /,
        *,
        num_evaluations: int | None = None,
        key: PRNGKey | None = None,
        **options: Any,
    ) -> Any:
        """The mean of ``f`` over ``num_evaluations`` draws of *d*.

        Raises
        ------
        TypeError
            If ``num_evaluations`` is not an integer.
        ValueError
            If ``num_evaluations`` is not positive.
        """
        n = _base.DEFAULT_NUM_EVALUATIONS if num_evaluations is None else num_evaluations
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"num_evaluations must be an integer; got {n!r}")
        if n <= 0:
            raise ValueError(f"num_evaluations must be positive; got {n!r}")
        if key is None:
            captured = _descendants.capture_stochastic_consumer(d)
            key = _broker._resolve_automatic_key(
                None,
                _broker._singleton_effect_plan(
                    operation_kind="expectation",
                    execution_mode="monte_carlo",
                    sample_shape=(n,),
                    record_path=captured.record_path,
                    descendant_descriptor=captured.descendant_descriptor,
                ),
            )
            draws = _descendants.sample_captured_consumer(captured, key, (n,))
        else:
            draws = d._sample(key, sample_shape=(n,))
        values = jax.vmap(f)(draws)
        return jax.tree.map(lambda v: jnp.mean(v, axis=0), values)


expectation_method_registry: UnaryDispatchRegistry[ExpectationMethod] = UnaryDispatchRegistry()
"""The methods that compute an expectation, in selection order."""

expectation_method_registry.register(_ExactExpectation())
expectation_method_registry.register(_MonteCarloExpectation())


@function(name="expectation")
def _expectation_function(
    dist: Distribution,
    f: Any,
    *,
    method: str | None = None,
    exact_only: bool = False,
    num_evaluations: int | None = None,
    key: PRNGKey | None = None,
) -> Any:
    """Compute ``E[f(X)]`` for ``X ~ dist``, with the method, budget, and key as arguments.

    The exact method runs when *dist* has one, and the Monte Carlo method
    otherwise; see :data:`expectation_method_registry` for the methods and their
    order. This is the form ``probpipe.expectation`` takes; the operation
    :data:`expectation` takes its controls through ``with_options`` and no key.

    Parameters
    ----------
    dist : Distribution
        The law to integrate against.
    f : callable
        Maps one draw to an array or a pytree of arrays.
    method : str, optional
        The name of a registered method to run instead of selecting one.
    exact_only : bool
        If ``True``, only exact methods are considered.
    num_evaluations : int, optional
        The number of draws a sampling method takes.
    key : PRNGKey, optional
        The key a sampling method draws with; workflow-owned when omitted.

    Returns
    -------
    Array or pytree of arrays
        ``E[f(X)]``, shaped as the output of *f*.

    Raises
    ------
    ResolutionError
        If no method applies under the controls, or *method* names one that is
        not registered or does not apply.
    """
    return expectation_method_registry.execute(
        dist,
        f,
        method=method,
        exact_only=exact_only,
        num_evaluations=num_evaluations,
        key=key,
    )


# ---------------------------------------------------------------------------
# The declarations of the moments
# ---------------------------------------------------------------------------


def _floating(dtype: Any) -> Any:
    """*dtype* when it is floating or unset, and otherwise the default floating dtype."""
    if dtype is None or jnp.issubdtype(dtype, jnp.inexact):
        return dtype
    return np.dtype(jnp.result_type(float))


def _hull(support: Any) -> Any:
    """The declared support that holds every average of points of *support*.

    A discrete support becomes the interval it spans, and a sphere, whose
    averages fill the ball, leaves the support undeclared.
    """
    if isinstance(support, _Boolean):
        return unit_interval
    if isinstance(support, _NonNegativeInteger):
        return non_negative
    if isinstance(support, _IntegerInterval):
        return interval(support.low, support.high)
    if isinstance(support, _Sphere):
        return None
    return support


def _mean_term(spec: TermSpec) -> TermSpec:
    """The term spec of a mean: floating leaves on the hull of their support."""
    if isinstance(spec, NumericArraySpec):
        return NumericArraySpec(spec.shape, _floating(spec.dtype), _hull(spec.support))
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _mean_term(child) for name, child in spec.children.items()})
    return spec


def _variance_term(spec: TermSpec) -> TermSpec:
    """The term spec of a variance: non-negative floating leaves."""
    if isinstance(spec, NumericArraySpec):
        return NumericArraySpec(spec.shape, _floating(spec.dtype), non_negative)
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _variance_term(child) for name, child in spec.children.items()})
    return spec


def _quantile_term(spec: TermSpec) -> TermSpec:
    """The term spec of a quantile: the event's shapes, with floating dtypes kept."""
    if isinstance(spec, NumericArraySpec):
        dtype = (
            spec.dtype if spec.dtype is not None and _floating(spec.dtype) == spec.dtype else None
        )
        return NumericArraySpec(spec.shape, dtype)
    if isinstance(spec, RecordSpec):
        return RecordSpec({name: _quantile_term(child) for name, child in spec.children.items()})
    return spec


def _array_leaves(spec: TermSpec) -> list[NumericArraySpec]:
    """The array leaves of a numeric spec, in the canonical flat order."""
    if isinstance(spec, NumericArraySpec):
        return [spec]
    if isinstance(spec, RecordSpec):
        return [leaf for child in spec.children.values() for leaf in _array_leaves(child)]
    return []


def _mean_result(d: DistributionSpec) -> OutputSpec:
    """A value of the event's kind, with the event's components and packaging.

    Its term specs are those of an average: floating, on the hull of the
    event's support, so the mean of a Bernoulli event lies in the unit interval.
    """
    return d.event_spec._with_spec(_mean_term(d.event_spec.spec))


def _variance_result(d: DistributionSpec) -> OutputSpec:
    """A value of the event's kind whose leaves are non-negative and floating."""
    return d.event_spec._with_spec(_variance_term(d.event_spec.spec))


def _cov_result(d: DistributionSpec) -> OutputSpec | None:
    """The dense covariance of the flattened draw, a ``(size, size)`` array.

    The size is the number of the event's coordinates; an event with free
    dimensions leaves it to the returned value.
    """
    leaves = _array_leaves(d.event_spec.spec)
    if any(leaf.free_dims for leaf in leaves):
        return OutputSpec(cov=None)
    size = sum(prod(leaf.shape) for leaf in leaves)
    dtypes = {_floating(leaf.dtype) for leaf in leaves}
    dtype = dtypes.pop() if len(dtypes) == 1 else None
    return OutputSpec(cov=NumericArraySpec((size, size), dtype))


def _quantile_result(d: DistributionSpec, q: TermSpec) -> OutputSpec:
    """One level returns the event's kind, and plural levels add a level named quantile.

    Raises
    ------
    ApplicabilityError
        If the levels are not numeric.
    """
    if not isinstance(q, NumericArraySpec):
        raise ApplicabilityError(f"quantile levels are a number or an array of numbers; got {q!r}")
    element = _quantile_term(d.event_spec.spec)
    if not q.shape:
        return d.event_spec._with_spec(element)
    return OutputSpec(quantile=BatchSpec(element, (q.shape,), ("quantile",)))


def _expectation_result(f: TermSpec) -> OutputSpec:
    """The kind the integrand's output declaration names, with an average's term specs.

    An integrand that declares no output leaves the declaration to the returned
    value.
    """
    if isinstance(f, FunctionSpec) and f.output_spec is not None and f.output_spec.spec is not None:
        return f.output_spec._with_spec(_mean_term(f.output_spec.spec))
    return OutputSpec(expectation=None)


def _numeric_event(d: DistributionSpec) -> bool:
    """The law draws a numeric value: an array or a record of arrays."""
    return isinstance(d.event_spec.spec, NumericSpec)


def _event_typed_variance(d: DistributionSpec) -> bool:
    """The event has an event-typed variance, which a measure-valued event lacks in general."""
    return not isinstance(d.event_spec.spec, DistributionSpec)


# ---------------------------------------------------------------------------
# The Monte Carlo fallbacks
# ---------------------------------------------------------------------------


def _can_sample(call: BoundCall, result: OutputSpec | None) -> Feasibility:
    """The law samples, and the moment is assumed to exist.

    Sampling alone does not establish that a moment exists, so the estimate
    assumes the moment is finite.
    """
    d = call.operands["d"]
    if not isinstance(d, SupportsSampling):
        return Feasibility(False, f"{type(d).__name__} does not sample")
    return _capability_guard(d, "_sample")


def _monte_carlo_draws(call: BoundCall, operation_kind: str) -> Any:
    """``n_broadcast_samples`` workflow-owned draws of the law, the draw axis leading."""
    return _workflow_draws(
        call.operands["d"],
        (call.controls["n_broadcast_samples"],),
        operation_kind=operation_kind,
        execution_mode="monte_carlo",
    )


def _empirical_of(call: BoundCall, draws: Any) -> EmpiricalDistribution:
    """The empirical law of the draws, whose moments the fallbacks report.

    Draws of an array event are its atoms along their leading axis. Draws of a
    record event are its atoms as the batch of records the law's event
    declaration calls for, whether they arrive as a nested mapping of raw
    columns or as a record of columns. A batch of records is taken as it is.
    """
    name = call.operation.name
    event = call.operands["d"].event_spec.spec
    if isinstance(draws, Batch) or not isinstance(event, RecordSpec):
        return EmpiricalDistribution(
            name, draws if isinstance(draws, Batch) else jnp.asarray(draws)
        )
    atoms = _batch_class_for(event)(name, _raw_record(draws), SAMPLE_LEVEL, element_spec=event)
    return EmpiricalDistribution(name, atoms)


def _mc_mean(call: BoundCall, result: OutputSpec | None) -> Any:
    """The coordinatewise average of the draws."""
    event = call.operands["d"].event_spec.spec
    if isinstance(event, NumericArraySpec):
        return jnp.mean(jnp.asarray(_monte_carlo_draws(call, "mean")), axis=0)
    if isinstance(event, NumericSpec):
        return _empirical_of(call, _monte_carlo_draws(call, "mean"))._mean()
    raise NotImplementedError("mean.monte_carlo: the average of function- and measure-valued draws")


def _mc_variance(call: BoundCall, result: OutputSpec | None) -> Any:
    """The coordinatewise sample variance of the draws."""
    event = call.operands["d"].event_spec.spec
    if isinstance(event, NumericArraySpec):
        return jnp.var(jnp.asarray(_monte_carlo_draws(call, "variance")), axis=0)
    if isinstance(event, NumericSpec):
        return _empirical_of(call, _monte_carlo_draws(call, "variance"))._variance()
    raise NotImplementedError("variance.monte_carlo: the pointwise variance of function draws")


def _dense(covariance: Any) -> Any:
    """A covariance as a dense array, whether an operator or an array holds it."""
    to_dense = getattr(covariance, "to_dense", None)
    return to_dense() if callable(to_dense) else jnp.asarray(covariance)


def _mc_cov(call: BoundCall, result: OutputSpec | None) -> Any:
    """The sample covariance of the flattened draws."""
    event = call.operands["d"].event_spec.spec
    draws = _monte_carlo_draws(call, "cov")
    if isinstance(event, NumericArraySpec):
        flat = jnp.asarray(draws).reshape((call.controls["n_broadcast_samples"], -1))
        return jnp.atleast_2d(jnp.cov(flat, rowvar=False))
    return _dense(_empirical_of(call, draws)._cov())


def _check_levels(q: Any) -> Any:
    """*q* as an array of levels in ``[0, 1]``.

    A traced *q*, as under ``jit``, is not checked.

    Raises
    ------
    MathematicalDomainError
        If a concrete level lies outside ``[0, 1]`` or is NaN.
    """
    levels = jnp.asarray(q)
    if not isinstance(levels, jax.core.Tracer) and bool(
        jnp.any((levels < 0) | (levels > 1) | jnp.isnan(levels))
    ):
        raise MathematicalDomainError(f"quantile levels must lie in [0, 1]; got {q!r}")
    return levels


def _mc_quantile(call: BoundCall, result: OutputSpec | None) -> Any:
    """The per-coordinate empirical quantiles of the draws, the level axes leading.

    A record event's quantiles at several levels are the declared batch of
    records.
    """
    levels = _check_levels(call.operands["q"])
    event = call.operands["d"].event_spec.spec
    draws = _monte_carlo_draws(call, "quantile")
    if isinstance(event, NumericArraySpec):
        return jnp.quantile(jnp.asarray(draws), levels, axis=0)
    return _record_batch(_empirical_of(call, draws)._quantile(levels), call, result)


# ---------------------------------------------------------------------------
# The operations
# ---------------------------------------------------------------------------


@operation(result=_mean_result)
def mean(d: Distribution):
    """The mean ``E[X]`` for ``X ~ d``, a value shaped like one draw.

    A record-drawing law's mean is a ``Record`` with the law's schema, a random
    function's is its mean function, and a random measure's is the
    marginalized law.

    Returns
    -------
    TrackedTerm
        The mean at the law's declared event kind.

    Raises
    ------
    ResolutionError
        If *d* has no closed-form mean and does not sample.
    """


mean.capability_route("closed_form", operand="d", protocol=SupportsMean, method="_mean", exact=True)
mean.fallback_route("monte_carlo", check=_can_sample, execute=_mc_mean, exact=False)


@operation(result=_variance_result, conditions=(_event_typed_variance,))
def variance(d: Distribution):
    """The variance of ``X ~ d``, a value shaped like one draw.

    Returns
    -------
    TrackedTerm
        The variance at the law's declared event kind, with non-negative leaves.

    Raises
    ------
    ApplicabilityError
        If the event is measure-valued, which has no event-typed variance in
        general.
    ResolutionError
        If *d* has no closed-form variance and does not sample.
    """


variance.capability_route(
    "closed_form", operand="d", protocol=SupportsVariance, method="_variance", exact=True
)
variance.fallback_route("monte_carlo", check=_can_sample, execute=_mc_variance, exact=False)


@operation(result=_cov_result, conditions=(_numeric_event,))
def cov(d: Distribution):
    """The covariance of the flattened draw of ``X ~ d``, a ``(size, size)`` array.

    Returns
    -------
    NumericArray
        The dense covariance over the event's coordinates in canonical order.

    Raises
    ------
    ApplicabilityError
        If the event is not numeric.
    ResolutionError
        If *d* has no closed-form covariance and does not sample.
    """


def _closed_form_cov(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d._cov()`` as a dense array."""
    return _dense(call.operands["d"]._cov())


cov.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsCovariance,
    method="_cov",
    exact=True,
    execute=_closed_form_cov,
)
cov.fallback_route("monte_carlo", check=_can_sample, execute=_mc_cov, exact=False)


@operation(result=_quantile_result, conditions=(_numeric_event,))
def quantile(d: Distribution, q: Any):
    """The per-coordinate quantiles of ``X ~ d`` at the levels *q*.

    Parameters
    ----------
    d : Distribution
        A law with a numeric event.
    q : float or array of float
        One level in ``[0, 1]``, or an array of them.

    Returns
    -------
    TrackedTerm
        For one level, a value of the event's kind; for several, the batch of
        those values on a level named ``quantile``.

    Raises
    ------
    ApplicabilityError
        If the event is not numeric, or *q* is not numeric.
    MathematicalDomainError
        If a concrete level lies outside ``[0, 1]``.
    ResolutionError
        If *d* has no closed-form quantiles and does not sample.
    """


def _closed_form_quantile(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d._quantile(q)`` at levels in ``[0, 1]``.

    A record event's quantiles at several levels are the declared batch of
    records.
    """
    quantiles = call.operands["d"]._quantile(_check_levels(call.operands["q"]))
    return _record_batch(quantiles, call, result)


quantile.capability_route(
    "closed_form",
    operand="d",
    protocol=SupportsQuantile,
    method="_quantile",
    exact=True,
    execute=_closed_form_quantile,
)
quantile.fallback_route("monte_carlo", check=_can_sample, execute=_mc_quantile, exact=False)


@operation(result=_expectation_result, roles={"f": (FunctionSpec,)})
def expectation(d: Distribution, f: Any):
    """The expectation ``E[f(X)]`` for ``X ~ d``, shaped by the output of *f*.

    The route delegates to :data:`expectation_method_registry`, whose exact
    method ranks first and whose Monte Carlo method is the default approximate
    one; ``with_options(method=...)`` selects a registered method by name.

    Parameters
    ----------
    d : Distribution
        The law to integrate against.
    f : callable or Function
        Maps one draw to a value; a ``Function``'s output declaration names the
        result's kind.

    Returns
    -------
    TrackedTerm
        ``E[f(X)]`` at the kind the integrand's output declaration names.

    Raises
    ------
    ApplicabilityError
        If *f* is not a callable.
    ResolutionError
        If no registered method applies under the controls.
    """


def _integration_arguments(call: BoundCall) -> tuple[Any, Any]:
    """The law and the integrand's raw callable."""
    f = call.operands["f"]
    return call.operands["d"], f.raw() if isinstance(f, Function) else f


def _integration_options(call: BoundCall) -> dict[str, Any]:
    """The sample count of the sampling methods."""
    return {"num_evaluations": call.controls["n_broadcast_samples"]}


expectation.registry_route(
    "methods",
    registry=expectation_method_registry,
    arguments=_integration_arguments,
    options=_integration_options,
    controls=(),
)
