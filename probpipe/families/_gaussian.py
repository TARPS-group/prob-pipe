"""The Gaussian algebra: the factored Gaussian joint and the Gaussian random functions.

Three families form a closed algebra built on ``LinOp``: the parametric
``MultivariateNormal``, the random-function member ``GaussianRandomFunction``,
and the factored joint ``FactoredMultivariateGaussian``. An affine
pushforward of a member is again a member, and conditioning a Gaussian prior on
a linear-Gaussian observation is exact.

Provides:
  - ``FactoredMultivariateGaussian`` – the factored joint of jointly Gaussian
    factors, which ``&`` and ``joint`` derive through the refinement it
    registers at import.
  - ``GaussianRandomFunction`` – the random function whose finite-dimensional
    laws are Gaussian, closed under shifts, scalings, output-side linear maps,
    and sums of independent members.
  - ``GaussianProcess`` – the random function specified by a mean function and
    a covariance kernel.
  - ``LinearBasisFunction`` – the random function ``f(x) = basis(x)ᵀ w`` with
    Gaussian weights ``w``.

``MultivariateNormal`` is defined in :mod:`probpipe.families._multivariate`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from .._messages import unknown_names
from ..core._dispatch import Feasibility
from ..core._expression import Operator, constant
from ..core._repr import format_value
from ..core._specs import OutputSpec, _root_output_name
from ..core.provenance import Provenance
from ..core.record import Record
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions._capabilities import (
    SupportsExactConditioning,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution, _check_output, _class_label
from ..distributions._factored import (
    FactoredDistribution,
    FactoredNumericDistribution,
    _derived_product,
    _register_refinement,
)
from ..linalg import DenseLinOp, LinOp
from ..values._function_base import FunctionSpec
from ._continuous import Normal
from ._multivariate import MultivariateNormal
from ._random_functions import RandomFunction

__all__ = [
    "FactoredMultivariateGaussian",
    "GaussianProcess",
    "GaussianRandomFunction",
    "LinearBasisFunction",
]


def _is_gaussian(factor: Any) -> bool:
    """Whether *factor* is a Gaussian law: a normal or a multivariate normal family, or a
    packaged joint of Gaussian factors, which a joint keeps as one factor."""
    return isinstance(factor, (Normal, MultivariateNormal, FactoredMultivariateGaussian))


def _jointly_gaussian(factors: Sequence[Distribution | ConditionalDistribution]) -> bool:
    """Whether the flattened *factors* are jointly Gaussian: every one is a Gaussian law.

    The linear-Gaussian conditional is the algebra's conditional member, and it
    joins this test once it can be constructed.
    """
    return bool(factors) and all(_is_gaussian(factor) for factor in factors)


class FactoredMultivariateGaussian(FactoredNumericDistribution, SupportsExactConditioning):
    """The factored joint whose factors are jointly Gaussian.

    The class registers with the factored joints at import, so ``&``, a joint
    rebuilt by a transform, a packaged joint that a regrouping rename builds,
    and a conditional joint bound at its givens construct it as the most
    specific class whenever every factor is a ``Normal``, a
    ``MultivariateNormal``, or a packaged joint of such factors; it is derived
    rather than built by hand. Such factors are independent, so the joint's sampling, log-density,
    moments, quantiles, and marginals are the edge-free joint's, each in closed
    form, and its covariance is block diagonal over the flattened draw.
    Conditioning on some of its outputs is exact: the conditional law of
    the others is the joint of their factors, which the values conditioned on
    leave unchanged.

    Parameters
    ----------
    factors : Sequence[Distribution | ConditionalDistribution]
        The jointly Gaussian factors, in conditional-first order.
    label : str, optional
        The joint's label. By default the label joins the factors' labels with
        ``·``, as ``a·b``, and the notation reads factor by factor.
    _scope : Mapping[str, int], optional
        The sizes already bound in the joint's dimension scope, by dimension name, which a
        joint rebuilt from its factors passes on.
    _output_name : str, optional
        The one output under which a packaged joint declares its record.

    Raises
    ------
    TypeError
        If a factor is not a Gaussian law.
    ValueError
        If the factors break a merge rule, as for
        :class:`~probpipe.distributions.FactoredDistribution`.
    """

    def __init__(
        self,
        factors: Sequence[Distribution | ConditionalDistribution],
        *,
        label: str | None = None,
        _scope: Mapping[str, int] | None = None,
        _output_name: str | None = None,
    ) -> None:
        super().__init__(
            factors,
            _scope=_scope,
            _output_name=_output_name,
            label=label,
        )
        if not _jointly_gaussian(self.factors):
            kinds = sorted(
                {type(factor).__name__ for factor in self.factors if not _is_gaussian(factor)}
            )
            raise TypeError(
                f"FactoredMultivariateGaussian {self.label!r} accepts only Normal or "
                f"MultivariateNormal factors, or Gaussian joints of them, got {kinds}"
            )

    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution:
        """The joint of the factors whose outputs *given* leaves unconditioned.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some of the joint's outputs, keyed by output name.
        **options : Any
            Unused, since the conditional is in closed form.

        Returns
        -------
        FactoredMultivariateGaussian
            The joint of the remaining factors, under the same label.

        Raises
        ------
        KeyError
            If a key of *given* is not an output of the joint.
        ValueError
            If *given* covers every output, so no law remains.
        """
        top = given.children if hasattr(given, "children") else given
        conditioned = set(dict(top.items()))
        unknown = sorted(conditioned - set(self.event_spec.outputs))
        if unknown:
            raise KeyError(self._unknown_outputs(unknown))
        kept = [
            factor for factor in self.factors if not set(factor.event_spec.outputs) <= conditioned
        ]
        if not kept:
            raise ValueError(self._every_output())
        law = _derived_product(
            FactoredDistribution(
                kept,
                label=self.label,
            ),
            self,
        )
        return law.with_provenance(
            Provenance.create(
                "condition_on", parents=[self], metadata={"conditioned": sorted(conditioned)}
            )
        )

    def _condition_on_guard(self, paths: tuple[str, ...]) -> Feasibility:
        """Every path is an output of the joint, and some output stays unconditioned."""
        outputs = set(self.event_spec.outputs)
        outside = sorted(set(paths) - outputs)
        if outside:
            return Feasibility(False, self._unknown_outputs(outside), actionable=True)
        if outputs <= set(paths):
            return Feasibility(False, self._every_output(), actionable=True)
        return Feasibility(True)

    def _unknown_outputs(self, unknown: Sequence[str]) -> str:
        """The message for conditioning on the names *unknown*, which are not outputs."""
        outputs = list(self.event_spec.outputs)
        return f"cannot condition {self.label!r}: {unknown_names('output', unknown, outputs)}"

    def _every_output(self) -> str:
        """The message for conditioning on every output, which leaves no distribution."""
        return (
            f"cannot condition {self.label!r} on all of its outputs "
            f"{list(self.event_spec.outputs)}; leave at least one unconditioned"
        )


_register_refinement(FactoredMultivariateGaussian, _jointly_gaussian)


# ---------------------------------------------------------------------------
# The Gaussian random functions
# ---------------------------------------------------------------------------


def _declarations(
    output_name: str, output_spec: OutputSpec | None, event_spec: OutputSpec | None
) -> tuple[OutputSpec, OutputSpec]:
    """The drawn function's output spec and the event spec, each defaulting to *output_name*.

    The event is a function, so a declared event type is a ``FunctionSpec``
    whose output spec names the drawn function's one output. An event spec
    whose type is unspecified, or a ``FunctionSpec`` whose output spec is
    unspecified, is completed with the drawn function's output spec.

    Parameters
    ----------
    output_name : str
        The name of the event's one output, which also names the drawn
        function's output by default.
    output_spec : OutputSpec or None
        The drawn function's output spec, or None for one output named
        *output_name* whose type is unspecified.
    event_spec : OutputSpec or None
        The event spec of the function-valued event, naming *output_name*, or
        None for a ``FunctionSpec`` under *output_name*.

    Returns
    -------
    output : OutputSpec
        The drawn function's output spec, which names one output.
    event : OutputSpec
        The event spec, whose ``FunctionSpec`` has that output spec.

    Raises
    ------
    TypeError
        If *output_spec* is not an ``OutputSpec`` naming one output, or
        *event_spec* is not an ``OutputSpec`` or declares a type that is not a
        ``FunctionSpec``.
    ValueError
        If the event's ``FunctionSpec`` names an output other than the drawn
        function's.
    """
    output = OutputSpec(**{output_name: None}) if output_spec is None else output_spec
    if not isinstance(output, OutputSpec) or _root_output_name(output) is None:
        raise TypeError(
            f"output_spec of {output_name!r} must be an OutputSpec naming one output, got "
            f"{output_spec!r}"
        )
    if event_spec is None:
        return output, OutputSpec(**{output_name: FunctionSpec(output_spec=output)})
    if not isinstance(event_spec, OutputSpec):
        raise TypeError(f"event_spec of {output_name!r} must be an OutputSpec, got {event_spec!r}")
    declared = event_spec.term_spec
    if declared is None:
        return output, event_spec._with_term_spec(FunctionSpec(output_spec=output))
    if not isinstance(declared, FunctionSpec):
        raise TypeError(
            f"event_spec of {output_name!r} must declare a FunctionSpec, got {type(declared).__name__}"
        )
    if declared.output_spec is None:
        filled = FunctionSpec(input_spec=declared.input_spec, output_spec=output)
        return output, event_spec._with_term_spec(filled)
    named = _root_output_name(declared.output_spec)
    if named != _root_output_name(output):
        raise ValueError(
            f"event_spec of {output_name!r} names the function output {named!r}, but output_spec "
            f"names it {_root_output_name(output)!r}. Make them match"
        )
    return output, event_spec


def _output_of(law: GaussianRandomFunction) -> str:
    """The name of *law*'s one output, whose draws are functions; a map of *law* keeps it."""
    (output_name,) = law.event_spec.outputs
    return output_name


def _stacked(X: ArrayLike) -> Array:
    """*X* as an array whose leading axis stacks the input points."""
    X = jnp.asarray(X)
    if X.ndim == 0:
        raise ValueError(
            "X must have a leading axis of input points, got a 0-d array; pass shape (n, ...), "
            "such as X[None] for one point"
        )
    return X


class GaussianRandomFunction(RandomFunction, SupportsMean, SupportsVariance, ABC):
    """A random function whose finite-dimensional laws are Gaussian.

    A concrete member implements :meth:`predict_mean` and
    :meth:`predict_variance`, and :meth:`predict_covariance` when it evaluates
    jointly. Each takes stacked inputs ``X``, whose leading axis indexes ``n``
    input points, and the mean and variance have the shape
    ``(n, *output_shape)`` of the drawn function's value there. The class is
    not restricted to Gaussian processes: any model with Gaussian predictions
    is a member.

    Calling the random function at ``X`` returns the finite-dimensional law
    there: a ``MultivariateNormal`` over the flattened values when the member
    evaluates jointly and there is more than one value, and otherwise a
    ``Normal`` whose coordinates are the values, which are independent when
    the member does not evaluate jointly. The law's label is the random
    function's, and its one output is the drawn function's output.

    The function-valued event is a whole term under *output_name*, and the
    drawn function's output takes the same name by default; ``output_spec``
    names another, and ``event_spec`` declares the event's type. A type that
    ``event_spec`` declares is a ``FunctionSpec`` whose output spec names the
    drawn function's output; an event spec whose type is unspecified, or a
    ``FunctionSpec`` whose output spec is unspecified, is completed with the
    drawn function's output spec. The mean is the mean function and the
    variance the pointwise variance function, each a callable on stacked
    inputs.

    Shifts ``f + b``, scalings ``alpha * f`` by a scalar, output-side linear
    maps ``A @ f``, and sums ``f + g`` of independent members are again
    members, evaluated in closed form. A shift, a scaling, and a linear map of
    ``f`` keep ``f``'s label, and a sum is labeled by its expression, as
    ``f + g``.

    Parameters
    ----------
    output_name : str
        The name of the random function's one output, which names its draws.
    label : str, optional
        The random function's label, its class name by default.
    output_spec : OutputSpec, optional
        The drawn function's output spec, which names its one output.
    event_spec : OutputSpec, optional
        The event spec of the function-valued draw, naming *output_name*.

    Raises
    ------
    TypeError
        If *output_name* is not a string, *label* is not a non-empty string,
        *output_spec* is not an ``OutputSpec`` naming one output, or
        *event_spec* is not an ``OutputSpec`` or declares a type that is not a
        ``FunctionSpec``.
    ValueError
        If *event_spec* names an output other than *output_name*, or the
        ``FunctionSpec`` it declares names an output other than the drawn
        function's.
    """

    def __init__(
        self,
        output_name: str,
        *,
        label: str | None = None,
        output_spec: OutputSpec | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        _check_output(output_name, _class_label(self))
        output, event = _declarations(output_name, output_spec, event_spec)
        self._output_spec = output
        super().__init__(output_name, event, label=label)

    @abstractmethod
    def predict_mean(self, X: Array) -> Array:
        """The mean of the drawn function's value at the stacked inputs *X*."""

    @abstractmethod
    def predict_variance(self, X: Array) -> Array:
        """The marginal variance of each value at the stacked inputs *X*."""

    def predict_covariance(self, X: Array) -> LinOp:
        """The joint covariance of the flattened values at the stacked inputs *X*.

        The values are flattened in row-major order of ``(n, *output_shape)``,
        so the point index varies slowest.

        Parameters
        ----------
        X : Array
            The ``n`` input points, stacked along the leading axis.

        Returns
        -------
        LinOp
            A square operator whose size is ``n`` times the size of one output value.

        Raises
        ------
        NotImplementedError
            If the member does not evaluate jointly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not evaluate jointly")

    @property
    def _joint(self) -> bool:
        """Whether the member implements the joint covariance."""
        return type(self).predict_covariance is not GaussianRandomFunction.predict_covariance

    def __call__(self, X: Array) -> Normal | MultivariateNormal:
        """The finite-dimensional law of the drawn function's value at the stacked inputs *X*.

        Parameters
        ----------
        X : Array
            The input points, stacked along the leading axis.

        Returns
        -------
        Normal or MultivariateNormal
            A ``MultivariateNormal`` over the flattened values when the member evaluates
            jointly and there is more than one value, and a ``Normal`` otherwise.

        Raises
        ------
        ValueError
            If *X* is a scalar, which stacks no points.
        """
        X = _stacked(X)
        mean = jnp.asarray(self.predict_mean(X))
        output = _root_output_name(self._output_spec)
        if mean.size > 1 and self._joint:
            cov = self.predict_covariance(X)
            return MultivariateNormal(output, jnp.reshape(mean, (-1,)), cov=cov, label=self.label)
        scale = jnp.sqrt(jnp.asarray(self.predict_variance(X)))
        return Normal(output, mean, scale, label=self.label)

    def _mean(self) -> Callable[[Array], Array]:
        """The mean function on stacked inputs."""
        return self.predict_mean

    def _variance(self) -> Callable[[Array], Array]:
        """The pointwise variance function on stacked inputs."""
        return self.predict_variance

    # -- The closed-form algebra --------------------------------------------

    def __rmatmul__(self, other: ArrayLike) -> GaussianRandomFunction:
        """``A @ f``, the output-side linear map by the matrix *other*."""
        return _LinearMapGRF(self, jnp.asarray(other))

    def __add__(self, other: Any) -> GaussianRandomFunction:
        if isinstance(other, GaussianRandomFunction):
            return _IndependentSumGRF(self, other)
        return _ShiftedGRF(self, jnp.asarray(other))

    def __radd__(self, other: Any) -> GaussianRandomFunction:
        if isinstance(other, GaussianRandomFunction):
            return _IndependentSumGRF(other, self)
        return _ShiftedGRF(self, jnp.asarray(other))

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (Distribution, ConditionalDistribution)):
            return Distribution.__mul__(self, other)
        return _ScaledGRF(self, jnp.asarray(other))

    def __rmul__(self, other: Any) -> GaussianRandomFunction:
        return _ScaledGRF(self, jnp.asarray(other))

    def __neg__(self) -> GaussianRandomFunction:
        return _ScaledGRF(self, jnp.asarray(-1.0))

    def __sub__(self, other: Any) -> GaussianRandomFunction:
        if isinstance(other, GaussianRandomFunction):
            return self + (-other)
        return _ShiftedGRF(self, -jnp.asarray(other))

    def __rsub__(self, other: Any) -> GaussianRandomFunction:
        return (-self) + other


class GaussianProcess(GaussianRandomFunction):
    """The Gaussian random function specified by a mean function and a covariance kernel.

    Its finite-dimensional law at stacked inputs ``X`` has the mean
    ``mean_fn(X)`` and the covariance ``cov_kernel(X, X)``. The kernel takes
    two stacks of ``n`` and ``m`` points and returns the covariance between
    their flattened values, of shape ``(n, m)`` for a scalar output. A
    Gaussian process does not draw whole functions.

    Parameters
    ----------
    output_name : str
        The name of the random function's one output, which names its draws.
    mean_fn : Callable[[Array], Array]
        The mean function, evaluated at stacked input points.
    cov_kernel : Callable[[Array, Array], Array]
        The covariance kernel, evaluated at two stacks of input points.
    label : str, optional
        The process's label, ``GaussianProcess`` by default.
    output_spec : OutputSpec, optional
        The drawn function's output spec.
    event_spec : OutputSpec, optional
        The event spec of the function-valued draw.

    Raises
    ------
    TypeError
        If *mean_fn* or *cov_kernel* is not callable, or *output_spec* or
        *event_spec* is refused as :class:`GaussianRandomFunction` refuses it.
    ValueError
        As :class:`GaussianRandomFunction` raises.
    """

    def __init__(
        self,
        output_name: str,
        mean_fn: Callable[[Array], Array],
        cov_kernel: Callable[[Array, Array], Array],
        *,
        label: str | None = None,
        output_spec: OutputSpec | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        if not callable(mean_fn) or not callable(cov_kernel):
            raise TypeError(
                f"mean_fn and cov_kernel of GaussianProcess must be callable, got "
                f"{type(mean_fn).__name__} and {type(cov_kernel).__name__}"
            )
        self._mean_fn = mean_fn
        self._cov_kernel = cov_kernel
        super().__init__(output_name, label=label, output_spec=output_spec, event_spec=event_spec)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameters ``mean_fn`` and ``cov_kernel``."""
        return [
            ("mean_fn", format_value(self._mean_fn)),
            ("cov_kernel", format_value(self._cov_kernel)),
        ]

    def predict_mean(self, X: Array) -> Array:
        """The mean function at the stacked input points *X*."""
        return jnp.asarray(self._mean_fn(_stacked(X)))

    def predict_variance(self, X: Array) -> Array:
        """The kernel's diagonal at each stacked input point of *X*, shaped like the mean."""
        X = _stacked(X)

        def at(point: Array) -> Array:
            block = jnp.asarray(self._cov_kernel(point[None], point[None]))
            side = math.isqrt(block.size)
            return jnp.diagonal(jnp.reshape(block, (side, side)))

        shape = jax.eval_shape(self.predict_mean, X).shape
        return jnp.reshape(jax.vmap(at)(X), shape)

    def predict_covariance(self, X: Array) -> LinOp:
        """The kernel at the stacked input points *X*, over their flattened values."""
        X = _stacked(X)
        kernel = jnp.asarray(self._cov_kernel(X, X))
        side = math.isqrt(kernel.size)
        return DenseLinOp(jnp.reshape(kernel, (side, side)))


class LinearBasisFunction(GaussianRandomFunction, SupportsSampling):
    r"""The random function ``f(x) = basis(x)ᵀ w`` with Gaussian weights ``w``.

    For ``w ~ N(m, Σ)``, the value at ``x`` has the mean ``basis(x)ᵀ m``, and
    the covariance kernel is ``basis(x)ᵀ Σ basis(x′)``, so the member
    evaluates jointly. The basis maps stacked inputs ``X`` of ``n`` points to
    features of shape ``(n, d_w)`` for a scalar output, or
    ``(n, *output_shape, d_w)`` for an array output, where ``d_w`` is the
    weights' dimension. A draw is the function of one weight draw.

    Parameters
    ----------
    output_name : str
        The name of the random function's one output, which names its draws.
    basis : Callable[[Array], Array]
        The feature map, evaluated at stacked input points.
    weights : MultivariateNormal
        The law of the weight vector.
    label : str, optional
        The random function's label, ``LinearBasisFunction`` by default.
    output_spec : OutputSpec, optional
        The drawn function's output spec.
    event_spec : OutputSpec, optional
        The event spec of the function-valued draw.

    Raises
    ------
    TypeError
        If *weights* is not a ``MultivariateNormal``, *basis* is not callable,
        or *output_spec* or *event_spec* is refused as
        :class:`GaussianRandomFunction` refuses it.
    ValueError
        As :class:`GaussianRandomFunction` raises.
    """

    def __init__(
        self,
        output_name: str,
        basis: Callable[[Array], Array],
        weights: MultivariateNormal,
        *,
        label: str | None = None,
        output_spec: OutputSpec | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        if not isinstance(weights, MultivariateNormal):
            raise TypeError(f"weights must be a MultivariateNormal, got {type(weights).__name__}")
        if not callable(basis):
            raise TypeError(
                f"basis of LinearBasisFunction must be callable, got {type(basis).__name__}"
            )
        self._basis = basis
        self._weights = weights
        self._w_mean = weights.loc
        self._w_cov = weights.cov
        super().__init__(output_name, label=label, output_spec=output_spec, event_spec=event_spec)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameters ``basis`` and ``weights``."""
        return [("basis", format_value(self._basis)), ("weights", repr(self._weights))]

    def _features(self, X: Array) -> Array:
        """The basis at the stacked inputs *X*, checked against the weights' dimension."""
        phi = jnp.asarray(self._basis(_stacked(X)))
        if phi.ndim < 2 or phi.shape[-1] != self._w_mean.shape[0]:
            raise ValueError(
                f"basis of {self.label!r} must return features whose last axis has the weights' "
                f"size {self._w_mean.shape[0]}, got shape {phi.shape}"
            )
        return phi

    def predict_mean(self, X: Array) -> Array:
        """``basis(X) m`` at the stacked inputs *X*."""
        return jnp.einsum("...w,w->...", self._features(X), self._w_mean)

    def predict_variance(self, X: Array) -> Array:
        """The diagonal of ``basis(X) Σ basis(X)ᵀ``, shaped like the mean."""
        phi = self._features(X)
        return jnp.einsum("...w,wv,...v->...", phi, self._w_cov, phi)

    def predict_covariance(self, X: Array) -> LinOp:
        """``basis(X) Σ basis(X)ᵀ`` over the flattened values at the stacked inputs *X*."""
        phi = self._features(X)
        flat = jnp.reshape(phi, (-1, phi.shape[-1]))
        return DenseLinOp(flat @ self._w_cov @ flat.T)

    def _sample(
        self, key: PRNGKey, sample_shape: tuple[int, ...] = ()
    ) -> Callable[[ArrayLike], Array]:
        """The function of one weight draw, or of *sample_shape* draws evaluated together.

        For a nonempty *sample_shape* the callable returns the values of every
        draw, with *sample_shape* leading.
        """
        shape = tuple(sample_shape)
        weights = jnp.asarray(self._weights._sample(key, shape))
        stacked = jnp.reshape(weights, (-1, weights.shape[-1]))
        features = self._features

        def draw(X: ArrayLike) -> Array:
            phi = features(jnp.asarray(X))
            values = jnp.einsum("...w,sw->s...", phi, stacked)
            return jnp.reshape(values, (*shape, *phi.shape[:-1]))

        return draw


# ---------------------------------------------------------------------------
# The closed-form algebra, constructed by the operators of a member
# ---------------------------------------------------------------------------


class _OperatorGRF(GaussianRandomFunction):
    """A member that an operator builds from other members and constants, labeled by the operator's expression."""

    def _operator_expression(self) -> Operator:
        """The expression of the operator applied to the construction's operands."""
        raise NotImplementedError

    def _derived_label(self) -> str:
        """The label the construction gives: the operator's expression, as ``f + g``."""
        return self._operator_expression().render_label()


class _LinearMapGRF(_OperatorGRF):
    """``h(x) = A g(x)`` for a member *g* with a vector output; constructed by ``A @ g``."""

    def __init__(self, base: GaussianRandomFunction, A: Array) -> None:
        if not isinstance(base, GaussianRandomFunction):
            raise TypeError(f"base must be a GaussianRandomFunction, got {type(base).__name__}")
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D (d_out, d_in), got shape {A.shape}")
        self._base = base
        self._A = A
        super().__init__(
            _output_of(base),
            label=base.label,
            output_spec=base._output_spec,
            event_spec=base.event_spec,
        )
        self._store_expression(self._operator_expression())

    def _operator_expression(self) -> Operator:
        """The expression ``A @ f`` that the construction gives."""
        return Operator("@", (constant(self._A), self._base._embedded_expression()))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameters ``base`` and ``A``."""
        return [("base", format_value(self._base)), ("A", format_value(self._A))]

    @property
    def _joint(self) -> bool:
        return self._base._joint

    def _check(self, X: Array) -> None:
        """Raise unless the base's value at each point is a vector of A's input size."""
        shape = jax.eval_shape(self._base.predict_mean, X).shape
        if len(shape) != 2 or shape[1] != self._A.shape[1]:
            raise ValueError(
                f"A @ f needs the value of f at each point to be a vector of size "
                f"{self._A.shape[1]} to match A's columns, but {self._base.label!r} has shape "
                f"{tuple(shape[1:])}"
            )

    def predict_mean(self, X: Array) -> Array:
        X = _stacked(X)
        self._check(X)
        return jnp.einsum("ow,iw->io", self._A, self._base.predict_mean(X))

    def predict_variance(self, X: Array) -> Array:
        return jnp.diagonal(self._point_covariances(X), axis1=-2, axis2=-1)

    def predict_covariance(self, X: Array) -> LinOp:
        X = _stacked(X)
        self._check(X)
        n, (d_out, d_in) = X.shape[0], self._A.shape
        base = jnp.reshape(self._base.predict_covariance(X).to_dense(), (n, d_in, n, d_in))
        joint = jnp.einsum("ow,iwjv,pv->iojp", self._A, base, self._A)
        return DenseLinOp(jnp.reshape(joint, (n * d_out, n * d_out)))

    def _point_covariances(self, X: Array) -> Array:
        """``A Σ(x) Aᵀ`` at each stacked point, with ``Σ(x)`` the base's output covariance there.

        A base that does not evaluate jointly has independent outputs.
        """
        X = _stacked(X)
        self._check(X)
        if self._base._joint:
            blocks = jax.vmap(lambda point: self._base.predict_covariance(point[None]).to_dense())(
                X
            )
        else:
            blocks = jax.vmap(jnp.diag)(self._base.predict_variance(X))
        return jnp.einsum("ow,iwv,pv->iop", self._A, blocks, self._A)


class _ShiftedGRF(_OperatorGRF):
    """``h(x) = g(x) + b`` for a constant *b*; constructed by ``g + b``."""

    def __init__(self, base: GaussianRandomFunction, b: Array) -> None:
        self._base = base
        self._b = b
        super().__init__(
            _output_of(base),
            label=base.label,
            output_spec=base._output_spec,
            event_spec=base.event_spec,
        )
        self._store_expression(self._operator_expression())

    def _operator_expression(self) -> Operator:
        """The expression ``f + b`` that the construction gives."""
        return Operator("+", (self._base._embedded_expression(), constant(self._b)))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameters ``base`` and ``b``."""
        return [("base", format_value(self._base)), ("b", format_value(self._b))]

    @property
    def _joint(self) -> bool:
        return self._base._joint

    def predict_mean(self, X: Array) -> Array:
        return self._base.predict_mean(X) + self._b

    def predict_variance(self, X: Array) -> Array:
        return self._base.predict_variance(X)

    def predict_covariance(self, X: Array) -> LinOp:
        return self._base.predict_covariance(X)


class _ScaledGRF(_OperatorGRF):
    """``h(x) = alpha g(x)`` for a scalar *alpha*; constructed by ``alpha * g``."""

    def __init__(self, base: GaussianRandomFunction, alpha: Array) -> None:
        if alpha.ndim != 0:
            raise ValueError(
                f"alpha * f needs a scalar alpha, got shape {alpha.shape}; to apply a matrix to "
                f"the outputs, use A @ f"
            )
        self._base = base
        self._alpha = alpha
        super().__init__(
            _output_of(base),
            label=base.label,
            output_spec=base._output_spec,
            event_spec=base.event_spec,
        )
        self._store_expression(self._operator_expression())

    def _operator_expression(self) -> Operator:
        """The expression ``alpha * f`` that the construction gives."""
        return Operator("*", (constant(self._alpha), self._base._embedded_expression()))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameters ``base`` and ``alpha``."""
        return [("base", format_value(self._base)), ("alpha", format_value(self._alpha))]

    @property
    def _joint(self) -> bool:
        return self._base._joint

    def predict_mean(self, X: Array) -> Array:
        return self._alpha * self._base.predict_mean(X)

    def predict_variance(self, X: Array) -> Array:
        return self._alpha**2 * self._base.predict_variance(X)

    def predict_covariance(self, X: Array) -> LinOp:
        return self._base.predict_covariance(X) * (self._alpha**2)


def _summed(left: Array, right: Array) -> Array:
    """``left + right`` for the values of two members at the same points.

    Parameters
    ----------
    left : Array
        The first member's mean or variance at the stacked points.
    right : Array
        The second member's mean or variance at the same points.

    Returns
    -------
    Array
        The sum, in the shape the two values share.

    Raises
    ------
    ValueError
        If the two members' values have different shapes, which would otherwise
        broadcast.
    """
    left, right = jnp.asarray(left), jnp.asarray(right)
    if left.shape != right.shape:
        raise ValueError(
            f"f + g needs both functions to have the same output shape, got {left.shape[1:]} "
            f"and {right.shape[1:]}"
        )
    return left + right


class _IndependentSumGRF(_OperatorGRF):
    """``h(x) = g₁(x) + g₂(x)`` for independent members; constructed by ``g₁ + g₂``.

    Independence is the caller's statement, so the covariances add; a member
    added to itself raises, since it is not independent of itself.
    """

    def __init__(self, left: GaussianRandomFunction, right: GaussianRandomFunction) -> None:
        if left is right:
            raise ValueError(
                f"cannot add {left.label!r} to itself, since f + g needs independent functions; "
                f"use 2 * f instead"
            )
        self._left = left
        self._right = right
        expression = self._operator_expression()
        rendering = expression.label_rendering()
        super().__init__(
            _output_of(left),
            label=rendering[0],
            output_spec=left._output_spec,
            event_spec=left.event_spec,
        )
        self._store_expression(expression, rendering)

    def _operator_expression(self) -> Operator:
        """The expression ``f + g`` that the construction gives."""
        return Operator(
            "+", (self._left._embedded_expression(), self._right._embedded_expression())
        )

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameters ``left`` and ``right``."""
        return [("left", format_value(self._left)), ("right", format_value(self._right))]

    @property
    def _joint(self) -> bool:
        return self._left._joint and self._right._joint

    def predict_mean(self, X: Array) -> Array:
        return _summed(self._left.predict_mean(X), self._right.predict_mean(X))

    def predict_variance(self, X: Array) -> Array:
        return _summed(self._left.predict_variance(X), self._right.predict_variance(X))

    def predict_covariance(self, X: Array) -> LinOp:
        return self._left.predict_covariance(X) + self._right.predict_covariance(X)
