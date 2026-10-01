"""The condition_on operation, its two stages, and the inference-method registry.

``condition_on(d, given)`` fixes fields of a distribution or of a conditional
distribution and returns the resulting law, normalized. ``given`` is
field-keyed, a ``Record`` or a mapping of field paths. Binding a given slot
applies the kernel, and binding a produced field conditions the law.

A call resolves in two stages. The **exact stage** computes the conditional: it
curries given slots, calls a conditioning capability, or, where no exact route
conditions a produced field, forms the **unnormalized conditional**, the law of
the unconditioned fields whose unnormalized log-density is the joint's at the
given values. The **normalization stage** returns a normalized result as it is,
and otherwise passes the result as the target to the inference-method registry,
whose selected method returns a normalized law. A kernel result is normalized
per value: it normalizes each law it yields once its last given is bound.

The routes, in selection order:

1. ``curry`` binds given slots of a kernel through its ``_condition_on``, which
   is exact unless the kernel claims ``SupportsApproximateConditioning``, and
   normalizes the result.
2. ``slice`` assembles the conditional from a factored law's normalized factors
   when the conditioned fields admit an exact slice.
3. ``exact_conditioning`` calls ``_condition_on`` on a law claiming
   ``SupportsExactConditioning``.
4. ``bayes`` curries any given slots the given names, forms the unnormalized
   conditional of the produced fields, and normalizes it.
5. ``approximate_conditioning`` calls ``_condition_on`` on a law claiming
   ``SupportsApproximateConditioning``.
6. ``unnormalized`` returns the exact stage's result, normalized or not; a caller
   selects it only by name, as ``method="unnormalized"``.

A route that normalizes through the registry is exact when its exact stage and
the selected method both are, so its exact methods rank with the exact routes
and its approximate methods with the approximate ones. No approximate route
therefore runs while an exact one applies, and ``exact_only`` excludes the
approximate conditioning capability and the approximate methods alike, which
raises ``ResolutionError`` for a call whose exact stage leaves an unnormalized
result. ``check`` reports both stages: its route is the exact stage's, and its
method the normalization's, which is ``None`` for a result that needs none.

The inference methods' own parameters, such as warmup lengths, are controls set
through ``with_options``. Each route declares the controls it reads: the routes
that normalize pass the parameters of the registered inference methods to the
selected method, ``approximate_conditioning`` passes an amortized posterior's
sample count and seed to its ``_condition_on`` as keyword options, and the other
routes read none. A control that no route declares raises ``TypeError`` at
``with_options``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar

from ..core._dispatch import (
    BaseDispatchRegistry,
    Feasibility,
    MethodInfo,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from ..core._empirical import RecordEmpiricalDistribution
from ..core._record_spec import RecordSpec
from ..core._spec_base import TermSpec
from ..core._specs import InputSpec, OutputSpec, _components_record
from ..core.provenance import Provenance
from ..core.record import Record
from ..distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalUnnormalizedLogProb,
    SupportsExactConditioning,
    SupportsUnnormalizedLogProb,
    _capability_guard,
    _capability_subclass,
    _guard_condition,
    _is_normalized,
    _kernel_is_normalized,
)
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import SupportsFactors
from ._convert import convert
from ._operation import (
    BoundCall,
    CallCheck,
    Operation,
    RouteSource,
    _Candidate,
    _CheckedRoute,
    _RegistryRoute,
    _workflow_draws,
    operation_registry,
)
from ._sample import _record_batch, _sample_result, _sample_shape, sample

__all__ = ["InferenceMethod", "condition_on", "inference_method_registry"]

_PATH_SEP = "/"

#: The name of the route that returns the exact stage's result.
_UNNORMALIZED = "unnormalized"


# ---------------------------------------------------------------------------
# The inference-method registry
# ---------------------------------------------------------------------------


class InferenceMethod(UnaryDispatchMethod):
    """Base class for registered inference methods; declares ``exact = False``.

    A method normalizes the target of ``condition_on``'s normalization stage: it
    takes the target alone, a law whose data are already bound, and returns a
    normalized law over the target's event. A subclass declares ``name``,
    ``supported_types``, ``check``, and ``execute``, and overrides ``priority``
    to take part in automatic selection. Its ``check`` reads the interface it
    requires of the target, such as an unnormalized density, a backend program,
    or the joint and the given values to simulate from.

    Notes
    -----
    Every inference method is approximate: a finite MCMC, SG-MCMC, slice,
    ABC, or variational output stands in for the conditional law, whatever
    its invariant target or asymptotic guarantee. Those guarantees are the
    method's own documentation, not its exactness. A method that returns a
    representation of the conditional law itself overrides ``exact``.
    """

    @property
    def exact(self) -> bool:
        return False


#: The builder of the target of a model and its observed data, which
#: ``probpipe.inference`` installs, since the way each model binds its data is
#: that package's.
_observed_target: Callable[[Any, Any], Any] | None = None


def _install_observed_target(builder: Callable[[Any, Any], Any]) -> None:
    """Install the builder of the target of a model and the data it is conditioned on."""
    global _observed_target
    _observed_target = builder


class _InferenceMethodRegistry(UnaryDispatchRegistry[UnaryDispatchMethod]):
    """The inference-method registry, whose methods take the target of the normalization stage.

    A call with a second positional argument passes a model and the data it is
    conditioned on, as ``probpipe.condition_on`` does, and dispatches on the
    target the installed builder forms from them.
    """

    def check(
        self, *args: Any, method: str | None = None, exact_only: bool = False, **kwargs: Any
    ) -> MethodInfo:
        """The report of the method a call would run; see :meth:`UnaryDispatchRegistry.check`."""
        return super().check(*self._targets(args), method=method, exact_only=exact_only, **kwargs)

    def execute(
        self, *args: Any, method: str | None = None, exact_only: bool = False, **kwargs: Any
    ) -> Any:
        """The selected method's result; see :meth:`UnaryDispatchRegistry.execute`."""
        return super().execute(*self._targets(args), method=method, exact_only=exact_only, **kwargs)

    @staticmethod
    def _targets(args: tuple[Any, ...]) -> tuple[Any, ...]:
        """*args* with a model and its observed data replaced by their target.

        Raises
        ------
        TypeError
            If a model and its data are passed before a builder is installed.
        """
        if len(args) != 2:
            return args
        if _observed_target is None:
            raise TypeError("the target of a model and its data needs probpipe.inference")
        return (_observed_target(*args),)


#: The registry of the normalization stage, keyed on the target's type; the
#: methods of ``probpipe.inference`` register here.
inference_method_registry: UnaryDispatchRegistry[UnaryDispatchMethod] = _InferenceMethodRegistry()


_MCMC_CONTROLS = ("init", "num_chains", "num_results", "num_warmup", "random_seed", "step_size")
_SGMCMC_CONTROLS = (
    "batch_size",
    "init",
    "num_results",
    "num_warmup",
    "random_seed",
    "step_size",
    "with_replacement",
)
_BACKEND_NUTS_CONTROLS = ("num_chains", "num_results", "num_warmup", "random_seed")

#: The parameters each inference method registered by :mod:`probpipe.inference`
#: reads, by method name; the routes that normalize declare every one of them.
_INFERENCE_METHOD_CONTROLS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "blackjax_nuts": (*_MCMC_CONTROLS, "num_integration_steps"),
        "blackjax_hmc": (*_MCMC_CONTROLS, "num_integration_steps"),
        "blackjax_rwmh": (*_MCMC_CONTROLS, "adapt", "n_windows", "proposal_cov"),
        "blackjax_elliptical_slice": (
            "init",
            "num_chains",
            "num_results",
            "num_warmup",
            "random_seed",
        ),
        "blackjax_sgld": _SGMCMC_CONTROLS,
        "blackjax_sghmc": (*_SGMCMC_CONTROLS, "alpha", "beta", "num_integration_steps"),
        "tfp_nuts": _MCMC_CONTROLS,
        "tfp_hmc": _MCMC_CONTROLS,
        "nutpie_nuts": _BACKEND_NUTS_CONTROLS,
        "cmdstan_nuts": _BACKEND_NUTS_CONTROLS,
        "pymc_nuts": (*_BACKEND_NUTS_CONTROLS, "cores"),
        "pymc_advi": ("num_iterations", "num_results", "random_seed", "vi_method"),
        "pyabc_smcabc": (
            "distance_fn",
            "eps",
            "eps_alpha",
            "max_populations",
            "max_total_nr_simulations",
            "max_walltime",
            "min_acceptance_rate",
            "minimum_epsilon",
            "n_particles",
            "random_seed",
            "sampler",
            "summary_fn",
            "transitions",
        ),
    }
)

#: Every parameter of the registered inference methods.
_NORMALIZATION_CONTROLS = frozenset(
    name for names in _INFERENCE_METHOD_CONTROLS.values() for name in names
)

#: The parameters an amortized posterior's ``_condition_on`` reads, as the
#: kernel :func:`~probpipe.inference.learn_amortized_posterior` returns does.
_AMORTIZED_CONDITIONING_CONTROLS = ("num_results", "random_seed")


# ---------------------------------------------------------------------------
# The given
# ---------------------------------------------------------------------------


def _given_keys(given: Any) -> tuple[str, ...] | None:
    """The field paths *given* names: a Record's fields or a mapping's keys, or None for neither."""
    if isinstance(given, Record):
        return tuple(given.fields)
    if isinstance(given, Mapping):
        return tuple(given)
    return None


def _given_values(given: Any) -> dict[str, Any]:
    """The values *given* holds, keyed by the paths :func:`_given_keys` names."""
    if isinstance(given, Record):
        return dict(given.children)
    return dict(given)


def _head(path: str) -> str:
    """The first segment of *path*: a given slot or an event component."""
    return path.split(_PATH_SEP, 1)[0]


def _slots_of(d: Any) -> frozenset[str]:
    """The given slots of *d*, none for a law."""
    return frozenset(d.given_spec) if isinstance(d, ConditionalDistribution) else frozenset()


# ---------------------------------------------------------------------------
# The targets of the normalization stage
# ---------------------------------------------------------------------------


def _unconditioned_event(law: Any, produced: Iterable[str]) -> OutputSpec:
    """The declaration of *law*'s components that *produced* leaves unconditioned.

    It is an exposed record of those components, in the order *law* declares
    them.
    """
    record = _components_record(law.event_spec)
    conditioned = set(produced)
    return OutputSpec(
        RecordSpec(
            {name: spec for name, spec in record.children.items() if name not in conditioned}
        )
    )


def _joint_value(value: Any, given: Record) -> Record:
    """The joint's value at *value*, a draw of the unconditioned fields, and at *given*."""
    fields = value.children if isinstance(value, Record) else value
    return Record("value", {**dict(fields), **dict(given.children)})


def _conditional_density(self: _UnnormalizedConditional, value: Any) -> Any:
    """The joint's unnormalized log-density at *value* and the given values.

    Field-keyed given values join *value* into one record of the joint's fields.
    Data the joint does not declare as fields pair with *value* as
    ``(value, data)``, the form a density over parameters and data takes.
    """
    given = self.given
    joint_value = _joint_value(value, given) if self.keyed else (value, given)
    return self.joint._unnormalized_log_prob(joint_value)


class _UnnormalizedConditional(Distribution):
    """The unnormalized conditional: the law of a joint's unconditioned fields at given values.

    Its unnormalized log-density at a value of the unconditioned fields is the
    joint's at that value and the given values. It carries the joint and the
    given values, so a method that simulates rather than evaluates a density
    reads them. It claims ``SupportsUnnormalizedLogProb`` when the joint claims
    an unnormalized density, and no normalized capability, so it is
    unnormalized, and ``condition_on`` passes it to the inference-method
    registry as the target.

    Parameters
    ----------
    joint : Distribution
        The law conditioned.
    given : Record or Any
        The values of the conditioned fields, keyed by their components; or,
        when *keyed* is false, the data of a joint that does not declare them
        as fields, such as a model conditioned through ``probpipe.condition_on``.
    event_spec : OutputSpec
        The declaration of the unconditioned fields.
    keyed : bool
        Whether *given* is keyed by the joint's fields.
    """

    _capability_table: ClassVar = {
        SupportsUnnormalizedLogProb: {"_unnormalized_log_prob": _conditional_density},
    }

    def __new__(
        cls, joint: Distribution, given: Any, event_spec: OutputSpec, *, keyed: bool = True
    ) -> Any:
        claimed = (
            (SupportsUnnormalizedLogProb,) if isinstance(joint, SupportsUnnormalizedLogProb) else ()
        )
        return object.__new__(_capability_subclass(_UnnormalizedConditional, claimed))

    def __init__(
        self, joint: Distribution, given: Any, event_spec: OutputSpec, *, keyed: bool = True
    ) -> None:
        super().__init__(joint.name, event_spec)
        self._joint = joint
        self._given = given
        self._keyed = keyed
        self.with_provenance(
            Provenance.create(
                "condition_on", parents=[joint], metadata={"stage": "exact", "route": "bayes"}
            )
        )

    @property
    def joint(self) -> Distribution:
        """The law conditioned."""
        return self._joint

    @property
    def given(self) -> Any:
        """The values of the conditioned fields, or the data of a joint without such fields."""
        return self._given

    @property
    def keyed(self) -> bool:
        """Whether the given values are keyed by the joint's fields."""
        return self._keyed


def _conditional_kernel_density(
    self: _UnnormalizedConditionalKernel, given: Record | Mapping[str, Any], value: Any
) -> Any:
    """The kernel's unnormalized log-density at *given*, *value*, and the conditioned values."""
    return self.kernel._conditional_unnormalized_log_prob(given, _joint_value(value, self.given))


class _UnnormalizedConditionalKernel(ConditionalDistribution):
    """The unnormalized conditional within each slice of a kernel's unmet givens.

    Binding the given slots yields the unnormalized conditional of the law the
    kernel yields there, at the values of the conditioned fields, or a kernel
    over the slots left. It claims ``SupportsConditionalUnnormalizedLogProb``
    when the kernel does.

    Parameters
    ----------
    kernel : ConditionalDistribution
        The kernel conditioned.
    given : Record
        The values of the conditioned fields, keyed by their components.
    event_spec : OutputSpec
        The declaration of the unconditioned fields, an exposed record.
    """

    _capability_table: ClassVar = {
        SupportsConditionalUnnormalizedLogProb: {
            "_conditional_unnormalized_log_prob": _conditional_kernel_density
        },
    }

    def __new__(cls, kernel: ConditionalDistribution, given: Record, event_spec: OutputSpec) -> Any:
        claimed = (
            (SupportsConditionalUnnormalizedLogProb,)
            if isinstance(kernel, SupportsConditionalUnnormalizedLogProb)
            else ()
        )
        return object.__new__(_capability_subclass(_UnnormalizedConditionalKernel, claimed))

    def __init__(
        self, kernel: ConditionalDistribution, given: Record, event_spec: OutputSpec
    ) -> None:
        super().__init__(kernel.name, kernel.given_spec, event_spec)
        self._kernel = kernel
        self._given = given

    @property
    def kernel(self) -> ConditionalDistribution:
        """The kernel conditioned."""
        return self._kernel

    @property
    def given(self) -> Record:
        """The values of the conditioned fields."""
        return self._given

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The unnormalized conditional at a value of every given slot, or a kernel over the rest."""
        return _unnormalized_conditional(self._kernel._condition_on(given, **kwargs), self._given)


def _unnormalized_conditional(law: Any, given: Record) -> Any:
    """The unnormalized conditional of *law* at *given*, a kernel's within each slice."""
    event_spec = _unconditioned_event(law, given.fields)
    if isinstance(law, ConditionalDistribution):
        return _UnnormalizedConditionalKernel(law, given, event_spec)
    return _UnnormalizedConditional(law, given, event_spec)


# ---------------------------------------------------------------------------
# The normalization stage
# ---------------------------------------------------------------------------


_NO_EXACT_METHOD = (
    "the exact stage's result is unnormalized, and no exact method normalizes it; "
    'method="unnormalized" returns that result'
)


def _packaged_alike(declared: OutputSpec, expected: OutputSpec) -> bool:
    """Whether *declared* packages its event as *expected* does, under the same components."""
    return declared.exposes_record == expected.exposes_record and tuple(
        declared.components
    ) == tuple(expected.components)


def _as_declared(source: Any, law: Any) -> EmpiricalDistribution:
    """The atoms and weights of the normalized *law* under *source*'s event declaration.

    The result is an ``EmpiricalDistribution`` carrying *law*'s provenance,
    since a posterior over a whole-term event draws a one-field record of it.
    """
    empirical = EmpiricalDistribution(
        source.name, law.draws(), law.weights, event_spec=source.event_spec
    )
    return empirical.with_provenance(law.provenance)


@dataclass(frozen=True)
class _Normalization:
    """How the normalization stage normalizes a target: the registry, the method, and its budgets.

    Attributes
    ----------
    registry : BaseDispatchRegistry
        The inference-method registry.
    method : str or None
        The method the caller named, or ``None`` for automatic selection.
    exact_only : bool
        Whether only exact methods may run.
    options : Mapping[str, Any]
        The method parameters the call sets.
    """

    registry: BaseDispatchRegistry[Any]
    method: str | None
    exact_only: bool
    options: Mapping[str, Any]

    def report(self, target: Any) -> Feasibility:
        """The registry's report for normalizing the law *target*.

        A report that no method applies names ``method="unnormalized"`` when
        ``exact_only`` excluded the approximate methods.
        """
        report = self.registry.check(
            target, method=self.method, exact_only=self.exact_only, **self.options
        )
        if report.feasible is False and self.exact_only:
            return replace(report, description=f"{_NO_EXACT_METHOD}: {report.description}")
        return report

    def normalize(self, law: Any) -> Any:
        """*law* as it is when it is normalized, and otherwise the selected method's result.

        The result carries *law*'s event declaration, as the result rule of
        currying requires.
        """
        if _is_normalized(law):
            return law
        posterior = self.registry.execute(
            law, method=self.method, exact_only=self.exact_only, **self.options
        )
        if _packaged_alike(posterior.event_spec, law.event_spec):
            return posterior
        return _as_declared(law, posterior)


def _per_value_sample(
    self: _PerValueNormalization,
    given: Record | Mapping[str, Any],
    key: Any,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """Draws of the normalized law at a value of every given slot."""
    return self._condition_on(given)._sample(key, sample_shape)


class _PerValueNormalization(ConditionalDistribution):
    """A kernel whose laws are another kernel's, each normalized once its last given is bound.

    It is the normalization stage's result for a kernel whose laws are
    unnormalized: binding every given slot yields the law the kernel yields
    there, normalized by the inference-method registry, and binding some yields
    another such kernel. Its laws sample, so it claims
    ``SupportsConditionalSampling``, and it claims
    ``SupportsApproximateConditioning`` unless only exact methods normalize it,
    since evaluating it then runs an approximate method.

    Parameters
    ----------
    kernel : ConditionalDistribution
        The kernel whose laws are normalized.
    normalization : _Normalization
        The registry, method, and budgets that normalize each law.
    """

    _capability_table: ClassVar = {SupportsApproximateConditioning: {}}

    def __new__(cls, kernel: ConditionalDistribution, normalization: _Normalization) -> Any:
        claimed = () if normalization.exact_only else (SupportsApproximateConditioning,)
        return object.__new__(_capability_subclass(_PerValueNormalization, claimed))

    def __init__(self, kernel: ConditionalDistribution, normalization: _Normalization) -> None:
        super().__init__(kernel.name, kernel.given_spec, kernel.event_spec)
        self._kernel = kernel
        self._normalization = normalization

    @property
    def kernel(self) -> ConditionalDistribution:
        """The kernel whose laws are normalized."""
        return self._kernel

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The normalized law at a value of every given slot, or a kernel over the rest."""
        return _normalized(self._kernel._condition_on(given, **kwargs), self._normalization)

    _conditional_sample = _per_value_sample


def _normalized(result: Any, normalization: _Normalization) -> Any:
    """*result* normalized: a law by the registry, and a kernel per value."""
    if isinstance(result, ConditionalDistribution):
        if _kernel_is_normalized(result):
            return result
        return _PerValueNormalization(result, normalization)
    return normalization.normalize(result)


# ---------------------------------------------------------------------------
# The exact stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExactStage:
    """One way the exact stage computes the conditional.

    Attributes
    ----------
    check : callable
        ``check(call)``, whether the stage applies, read from the declarations.
    compute : callable
        ``compute(call)``, the stage's result.
    exact : callable
        ``exact(call)``, whether the result is the conditional itself rather
        than a stand-in for it.
    normalized : callable
        ``normalized(call)``, whether the result is normalized, read from the
        declarations.
    """

    check: Callable[[BoundCall], Feasibility]
    compute: Callable[[BoundCall], Any]
    exact: Callable[[BoundCall], bool]
    normalized: Callable[[BoundCall], bool]


def _can_curry(call: BoundCall) -> Feasibility:
    """Every key of the given names a given slot of a conditional distribution.

    Raises
    ------
    NotImplementedError
        If a key names part of a structured slot.
    """
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    if not isinstance(d, ConditionalDistribution):
        return Feasibility(False, "route 'curry' declined: the conditioned object is not a kernel")
    if not keys:
        return Feasibility(False, "route 'curry' declined: the given names no field")
    if not {_head(key) for key in keys} <= _slots_of(d):
        return Feasibility(False, "route 'curry' declined: the given names a produced field")
    if any(_PATH_SEP in key for key in keys):
        raise NotImplementedError("condition_on.curry: binding part of a structured slot")
    return Feasibility(True)


def _curry(call: BoundCall) -> Any:
    """The kernel's ``_condition_on`` at the given slots.

    A kernel that claims ``SupportsApproximateConditioning`` also receives the
    budgets an amortized posterior reads that the call sets.
    """
    kernel = call.operands["d"]
    options: dict[str, Any] = {}
    if isinstance(kernel, SupportsApproximateConditioning):
        options = {
            name: call.controls[name]
            for name in _AMORTIZED_CONDITIONING_CONTROLS
            if name in call.controls
        }
    return kernel._condition_on(call.operands["given"], **options)


def _evaluation_is_exact(call: BoundCall) -> bool:
    """Evaluating the kernel is exact unless it claims ``SupportsApproximateConditioning``."""
    return not isinstance(call.operands["d"], SupportsApproximateConditioning)


def _curried_is_normalized(call: BoundCall) -> bool:
    """The kernel's laws are normalized, as its conditional capabilities declare."""
    return _kernel_is_normalized(call.operands["d"])


def _can_form_the_unnormalized_conditional(call: BoundCall) -> Feasibility:
    """The given names produced fields, each a component, and every other key is a given slot.

    At least one component must stay unconditioned, as the law of the
    unconditioned fields.
    """
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    if not keys:
        return Feasibility(False, "route 'bayes' declined: the given is not field-keyed")
    slots = _slots_of(d)
    produced = [key for key in keys if _head(key) not in slots]
    if not produced:
        return Feasibility(False, "route 'bayes' declined: the given names no produced field")
    components = set(d.event_spec.components)
    for key in produced:
        if _head(key) not in components:
            return Feasibility(
                False, f"route 'bayes' declined: {key!r} is neither a given slot nor an event path"
            )
        if key not in components:
            return Feasibility(
                False,
                f"route 'bayes' declined: conditioning the interior path {key!r} is not implemented",
            )
    if any(_PATH_SEP in key for key in keys if _head(key) in slots):
        return Feasibility(
            False, "route 'bayes' declined: binding part of a structured slot is not implemented"
        )
    if components <= set(produced):
        return Feasibility(
            False, "route 'bayes' declined: the given names every produced field, leaving no law"
        )
    return Feasibility(True)


def _bayes(call: BoundCall) -> Any:
    """The unnormalized conditional of the produced fields, the given slots curried first."""
    d, values = call.operands["d"], _given_values(call.operands["given"])
    slots = _slots_of(d)
    bound = {key: value for key, value in values.items() if _head(key) in slots}
    law = d._condition_on(bound) if bound else d
    produced = {key: value for key, value in values.items() if key not in bound}
    return _unnormalized_conditional(law, Record("given", produced))


def _always(call: BoundCall) -> bool:
    return True


def _never(call: BoundCall) -> bool:
    return False


_CURRY = _ExactStage(
    check=_can_curry,
    compute=_curry,
    exact=_evaluation_is_exact,
    normalized=_curried_is_normalized,
)
_BAYES = _ExactStage(
    check=_can_form_the_unnormalized_conditional,
    compute=_bayes,
    exact=_always,
    normalized=_never,
)


# ---------------------------------------------------------------------------
# The routes that normalize
# ---------------------------------------------------------------------------


class _NormalizingRoute(_RegistryRoute):
    """A route that runs one exact stage and normalizes its result through the registry.

    A result that is normalized is returned as it is, a law that is not is the
    target of the registry's selected method, and a kernel is normalized per
    value. The route's exactness is that of its exact stage combined with the
    selected method's, so it is ranked twice, as every registry route is: its
    exact candidate applies when the result needs no approximate step, and its
    approximate candidate otherwise. The report of a normalized result names no
    method, and that of a normalized target names the method that normalizes it.
    """

    def __init__(
        self,
        name: str,
        *,
        source: RouteSource,
        stage: _ExactStage,
        registry: BaseDispatchRegistry[Any],
    ) -> None:
        super().__init__(name, registry=registry, controls=_NORMALIZATION_CONTROLS)
        self.source = source
        self._stage = stage

    @property
    def condition(self) -> str:
        """The exact stage's condition, and the normalization's."""
        stage = _guard_condition(self._stage.check)
        return (
            f"{stage}; a result that is not normalized is the target of a method the "
            f"registry selects among {', '.join(self.registry.list_methods()) or 'none'}"
        )

    def _normalization(self, call: BoundCall, method: str | None, exact_only: bool) -> Any:
        return _Normalization(
            self.registry, method, exact_only, MappingProxyType(self.budgets(call))
        )

    def probe(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Feasibility:
        """The exact stage's report, then the normalization's, under the controls."""
        report = self._stage.check(call)
        if report.feasible is not True:
            return report
        exact = self._stage.exact(call)
        if exact_only and not exact:
            return Feasibility(
                False,
                f"route {self.name!r} declined: the kernel claims "
                f"SupportsApproximateConditioning, so evaluating it is approximate",
            )
        if self._stage.normalized(call):
            if method is not None:
                return Feasibility(
                    False,
                    f"route {self.name!r} declined: the exact stage's result is normalized, "
                    f"so no inference method runs on it",
                )
            return CallCheck(True, exact=exact)
        target = self._stage.compute(call)
        normalization = self._normalization(call, method, exact_only)
        if isinstance(target, ConditionalDistribution):
            return self._per_value_report(call, target, normalization, exact)
        info = normalization.report(target)
        if info.feasible is not True or exact or not isinstance(info, MethodInfo):
            return info
        return replace(info, exact=False)

    def _per_value_report(
        self,
        call: BoundCall,
        target: ConditionalDistribution,
        normalization: _Normalization,
        exact: bool,
    ) -> Feasibility:
        """The report of normalizing a kernel per value, whose method is selected once bound.

        Its laws are exact only when the caller requires exact methods, which
        then normalize each law.
        """
        if _kernel_is_normalized(target):
            return CallCheck(True, exact=exact)
        if normalization.exact_only and not call.controls["exact_only"]:
            return Feasibility(
                False,
                f"route {self.name!r} declined: the kernel's laws are normalized once its "
                f"last given is bound, by a method that may be approximate",
            )
        exact = exact and normalization.exact_only
        if normalization.method is not None:
            method = self.registry.get_method(normalization.method)
            return MethodInfo(True, method_name=normalization.method, exact=exact and method.exact)
        return CallCheck(True, exact=exact)

    def run(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Any:
        """The exact stage's result, normalized."""
        result = self._stage.compute(call)
        if self._stage.normalized(call):
            return result
        return _normalized(result, self._normalization(call, method, exact_only))


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


class _Conditioning(Operation):
    """``condition_on``'s operation: a named method selects among the routes that normalize.

    The routes that normalize share the inference-method registry, so a
    ``method=`` control naming one of its methods selects each of those routes
    with that method, in selection order, and the first whose exact stage
    applies runs. Every other control selects as for any operation.
    """

    def _candidates(self, controls: Mapping[str, Any]) -> list[_Candidate]:
        method = controls["method"]
        routes = list(self._route_table.routes)
        if method is None or any(route.name == method for route in routes):
            return super()._candidates(controls)
        holders = [
            _Candidate(route, None, index, method)
            for index, route in enumerate(routes)
            if isinstance(route, _RegistryRoute) and method in route.registry.list_methods()
        ]
        return holders or super()._candidates(controls)


def _condition_on_result(d: TermSpec, given: TermSpec) -> OutputSpec:
    """Binding given slots declares the kernel's law, or a kernel over the slots left.

    Conditioning on a produced field leaves the declaration to the law the
    selected route returns.
    """
    if isinstance(d, ConditionalDistributionSpec) and isinstance(given, RecordSpec):
        keys = tuple(given.children)
        if keys and all(key in d.given_spec for key in keys):
            left = {name: spec for name, spec in d.given_spec.items() if name not in keys}
            if not left:
                return OutputSpec(condition_on=DistributionSpec(d.event_spec))
            return OutputSpec(
                condition_on=ConditionalDistributionSpec(InputSpec(left), d.event_spec)
            )
    return OutputSpec(condition_on=None)


def _conditioning_operation(declaration: Callable[..., Any]) -> _Conditioning:
    """Declare ``condition_on`` as a :class:`_Conditioning` and register it."""
    op = _Conditioning(
        declaration,
        result=_condition_on_result,
        roles={"d": (DistributionSpec, ConditionalDistributionSpec), "given": (TermSpec,)},
    )
    operation_registry.register(op)
    return op


@_conditioning_operation
def condition_on(d: Distribution, given: Any):
    """Fix fields of *d* at the values *given* holds, and return the resulting law, normalized.

    The exact stage curries given slots, calls a conditioning capability, or
    forms the unnormalized conditional of the produced fields, and the
    normalization stage passes an unnormalized result to the inference-method
    registry. ``with_options(method="unnormalized")`` returns the exact stage's
    result as it is, and ``with_options(exact_only=True)`` raises rather than
    normalize by an approximate method.

    Parameters
    ----------
    d : Distribution or ConditionalDistribution
        The law or kernel to condition.
    given : Record or Mapping
        The values, keyed by field path: given slots of a kernel, or fields the
        law produces.

    Returns
    -------
    Distribution or ConditionalDistribution
        The conditional, normalized: an ordinary law once every given slot is
        bound, and otherwise a kernel over the slots left whose laws are
        normalized.

    Raises
    ------
    ApplicabilityError
        If *d* is neither distribution kind.
    ResolutionError
        If no route applies under the controls, including an exact stage whose
        result is unnormalized under ``exact_only``.
    """


def _slice_check(call: BoundCall, result: OutputSpec | None) -> Any:
    """The conditioned fields leave a conditional assembled from normalized factors.

    A field's slice is exact when the conditional is a product of available
    factors and exact local conditioning operations, which graph position alone
    does not establish.
    """
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    if not isinstance(d, SupportsFactors) or isinstance(d, ConditionalDistribution) or not keys:
        return False
    return Feasibility(
        False,
        "route 'slice' declined: assembling a conditional from the factors is not implemented",
    )


def _slice(call: BoundCall, result: OutputSpec | None) -> Any:
    """The conditional assembled from the factors the slice keeps."""
    raise NotImplementedError("condition_on.slice")


def _given_paths(d: Any, given: Any) -> tuple[str, ...]:
    """The paths *given* binds: its keys, or every component when the value is a whole draw."""
    keys = _given_keys(given)
    return keys if keys is not None else tuple(d.event_spec.components)


def _conditioning_guard(call: BoundCall, result: OutputSpec | None) -> Any:
    """The law's conditioning guard admits the given's paths, where its class defines one."""
    d = call.operands["d"]
    return _capability_guard(d, "_condition_on", _given_paths(d, call.operands["given"]))


def _exact_stage_by_name(call: BoundCall, result: OutputSpec | None) -> Any:
    """Selected only by name, as ``method="unnormalized"``; the exact stage alone then runs.

    The exact stage is the first of currying, exact conditioning, and the
    unnormalized conditional that applies.
    """
    if call.controls["method"] != _UNNORMALIZED:
        return Feasibility(
            False, f'route {_UNNORMALIZED!r} declined: it is selected only by method="unnormalized"'
        )
    reports = []
    for stage in (_CURRY, _EXACT_CONDITIONING, _BAYES):
        report = stage.check(call)
        if report.feasible is not False:
            return report if report.feasible is None else CallCheck(True, exact=stage.exact(call))
        reports.append(report.description)
    return Feasibility(False, f"route {_UNNORMALIZED!r} declined: {'; '.join(reports)}")


def _exact_stage_result(call: BoundCall, result: OutputSpec | None) -> Any:
    """The result of the exact stage that applies, normalized or not."""
    for stage in (_CURRY, _EXACT_CONDITIONING, _BAYES):
        if stage.check(call).feasible is True:
            return stage.compute(call)
    raise AssertionError("the exact stage ran with no stage that applies")


condition_on.register_route(
    _NormalizingRoute(
        "curry", source=RouteSource.STRUCTURAL, stage=_CURRY, registry=inference_method_registry
    )
)
condition_on.structural_route("slice", check=_slice_check, execute=_slice, exact=True)
_exact_conditioning_route = condition_on.capability_route(
    "exact_conditioning",
    operand="d",
    protocol=SupportsExactConditioning,
    method="_condition_on",
    exact=True,
    check=_conditioning_guard,
)
condition_on.capability_route(
    "approximate_conditioning",
    operand="d",
    protocol=SupportsApproximateConditioning,
    method="_condition_on",
    exact=False,
    check=_conditioning_guard,
    controls=_AMORTIZED_CONDITIONING_CONTROLS,
)
condition_on.register_route(
    _NormalizingRoute(
        "bayes", source=RouteSource.REGISTRY, stage=_BAYES, registry=inference_method_registry
    )
)
condition_on.register_route(
    _CheckedRoute(
        _UNNORMALIZED,
        source=RouteSource.STRUCTURAL,
        check=_exact_stage_by_name,
        execute=_exact_stage_result,
        exact=None,
    )
)


def _can_condition_exactly(call: BoundCall) -> Feasibility:
    """The law claims ``SupportsExactConditioning``, and its guard admits the given's paths."""
    return _exact_conditioning_route.check(call, None)


def _condition_exactly(call: BoundCall) -> Any:
    """The law's ``_condition_on`` at the given, which returns the conditional law."""
    return call.operands["d"]._condition_on(call.operands["given"])


#: Exact conditioning as a way the exact stage computes the conditional; its
#: result is the conditional law, which is normalized.
_EXACT_CONDITIONING = _ExactStage(
    check=_can_condition_exactly,
    compute=_condition_exactly,
    exact=_always,
    normalized=_always,
)


# ---------------------------------------------------------------------------
# The draws and conversions of an unnormalized law
# ---------------------------------------------------------------------------


class _NormalizingThen(_RegistryRoute):
    """A route of ``sample`` or ``convert`` that normalizes an unnormalized law, then answers.

    The law is the target of a method the inference-method registry selects,
    as in ``condition_on``'s normalization stage, and the call is answered on
    the normalized result. The route's exactness is the selected method's, and
    its budgets are the parameters of the registered methods.
    """

    def __init__(
        self,
        name: str,
        *,
        admits: Callable[[BoundCall], Feasibility],
        answer: Callable[[BoundCall, Any], Any],
    ) -> None:
        super().__init__(name, registry=inference_method_registry, controls=_NORMALIZATION_CONTROLS)
        self._admits = admits
        self._answer = answer

    @property
    def condition(self) -> str:
        """The law is unnormalized, the call is admitted, and a registered method applies."""
        return (
            f"The law is unnormalized, and a method the registry selects among "
            f"{', '.join(self.registry.list_methods()) or 'none'} normalizes it. "
            f"{_guard_condition(self._admits)}"
        )

    def probe(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Feasibility:
        """Whether the law is unnormalized and admitted, then the registry's report for it."""
        law = call.operands["d"]
        if not isinstance(law, Distribution) or _is_normalized(law):
            return Feasibility(False, f"route {self.name!r} declined: the law is normalized")
        admitted = self._admits(call)
        if admitted.feasible is not True:
            return admitted
        return self.registry.check(law, method=method, exact_only=exact_only, **self.budgets(call))

    def run(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Any:
        """The answer on the law the selected method returns."""
        normalized = self.registry.execute(
            call.operands["d"], method=method, exact_only=exact_only, **self.budgets(call)
        )
        return self._answer(call, normalized)


def _every_sample_shape(call: BoundCall) -> Feasibility:
    """Every sample shape is admitted."""
    return Feasibility(True)


def _declared_as_source(call: BoundCall, law: Any) -> Any:
    """The normalized *law* under the source's event declaration.

    A law that declares the source's event is returned as it is; otherwise its
    atoms and weights are declared as the source's event.
    """
    source = call.operands["d"]
    if law.event_spec == source.event_spec:
        return law
    return _as_declared(source, law)


def _draws_of(call: BoundCall, law: Any) -> Any:
    """Draws of the normalized *law*, as ``sample`` draws them from a law that samples."""
    draws = _workflow_draws(
        _declared_as_source(call, law),
        _sample_shape(call.operands["sample_shape"]),
        operation_kind="sample",
        execution_mode="sampled",
    )
    result = _sample_result(call.operands["d"].spec, call.operands["sample_shape"])
    return _record_batch(draws, call, result)


def _an_empirical_target(call: BoundCall) -> Feasibility:
    """The target is an empirical class, or a capability protocol an empirical law claims."""
    target = call.operands["target"]
    if issubclass(EmpiricalDistribution, target) or issubclass(RecordEmpiricalDistribution, target):
        return Feasibility(True)
    return Feasibility(
        False, f"route 'normalize' declined: an empirical law does not satisfy {target.__name__}"
    )


def _empirical_of(call: BoundCall, law: Any) -> Any:
    """The normalized *law* at the target, carrying the source's event declaration.

    Raises
    ------
    TypeError
        If the normalized law under the source's declaration is not the target.
    """
    source, target = call.operands["d"], call.operands["target"]
    for candidate in (law, _declared_as_source(call, law)):
        if isinstance(candidate, target) and candidate.event_spec == source.event_spec:
            return candidate
    empirical = EmpiricalDistribution(
        source.name, law.draws(), law.weights, event_spec=source.event_spec
    )
    if not isinstance(empirical, target):
        raise TypeError(f"the normalized law of {source.name!r} is not a {target.__name__}")
    return empirical


sample.register_route(_NormalizingThen("normalize", admits=_every_sample_shape, answer=_draws_of))
convert.register_route(
    _NormalizingThen("normalize", admits=_an_empirical_target, answer=_empirical_of)
)
