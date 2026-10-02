"""The Gaussian algebra: the factored Gaussian joint and the Gaussian random functions.

Three families form a closed algebra built on ``LinOp``: the parametric
``MultivariateNormal``, the random-function member ``GaussianRandomFunction``,
and the factored joint ``FactoredMultivariateGaussian``. An affine
pushforward of a member is again a member, and conditioning a Gaussian prior on
a linear-Gaussian observation is exact.

Provides:
  - ``FactoredMultivariateGaussian`` – the factored joint of jointly Gaussian
    factors, which ``*`` and ``joint`` derive through the refinement it
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

from ..core._dispatch import Feasibility
from ..core._repr import format_value
from ..core._specs import OutputSpec
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
from ..distributions._distribution import Distribution
from ..distributions._factored import (
    FactoredDistribution,
    FactoredNumericDistribution,
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

    The class registers with the factored joints at import, so ``*``, a joint
    rebuilt by a transform, a packaged joint that a regrouping rename builds,
    and a conditional joint bound at its givens construct it as the most
    specific class whenever every factor is a ``Normal``, a
    ``MultivariateNormal``, or a packaged joint of such factors; it is derived
    rather than built by hand. Such factors are independent, so the joint's sampling, log-density,
    moments, quantiles, and marginals are the edge-free joint's, each in closed
    form, and its covariance is block diagonal over the flattened draw.
    Conditioning on some of its components is exact: the conditional law of
    the others is the joint of their factors, which the values conditioned on
    leave unchanged.

    Parameters
    ----------
    label : str
        The joint's label.
    factors : Sequence[Distribution | ConditionalDistribution]
        The jointly Gaussian factors, in conditional-first order.
    _component : str, optional
        The one component under which a packaged joint declares its record.

    Raises
    ------
    TypeError
        If a factor is not a Gaussian law.
    ValueError
        If the factors break a composition rule, as for
        :class:`~probpipe.distributions.FactoredDistribution`.
    """

    def __init__(
        self,
        label: str,
        factors: Sequence[Distribution | ConditionalDistribution],
        *,
        _scope: Mapping[str, int] | None = None,
        _component: str | None = None,
    ) -> None:
        super().__init__(label, factors, _scope=_scope, _component=_component)
        if not _jointly_gaussian(self.factors):
            kinds = sorted(
                {type(factor).__name__ for factor in self.factors if not _is_gaussian(factor)}
            )
            raise TypeError(
                f"the factors of {label!r} are jointly Gaussian only when each is a Normal, a "
                f"MultivariateNormal, or a packaged joint of them, got {kinds}"
            )

    def _condition_on(self, given: Record | Mapping[str, Any], /, **options: Any) -> Distribution:
        """The joint of the factors whose components *given* leaves unconditioned.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some of the joint's components, keyed by component.
        **options : Any
            Unused, since the conditional is in closed form.

        Returns
        -------
        FactoredMultivariateGaussian
            The joint of the remaining factors, under the same name.

        Raises
        ------
        KeyError
            If a key of *given* is not a component of the joint.
        ValueError
            If *given* covers every component, so no law remains.
        """
        top = given.children if hasattr(given, "children") else given
        conditioned = set(dict(top.items()))
        unknown = sorted(conditioned - set(self.event_spec.components))
        if unknown:
            raise KeyError(f"{unknown} are not components of {self.label!r}")
        kept = [
            factor
            for factor in self.factors
            if not set(factor.event_spec.components) <= conditioned
        ]
        if not kept:
            raise ValueError(
                f"the given covers every component of {self.label!r}, so no law remains"
            )
        law = FactoredDistribution(self.label, kept)
        return law.with_provenance(
            Provenance.create(
                "condition_on", parents=[self], metadata={"conditioned": sorted(conditioned)}
            )
        )

    def _condition_on_guard(self, paths: tuple[str, ...]) -> Feasibility:
        """Every path is a component of the joint, and some component stays unconditioned."""
        components = set(self.event_spec.components)
        outside = sorted(set(paths) - components)
        if outside:
            return Feasibility(False, f"{outside} are not components of {self.label!r}")
        if components <= set(paths):
            return Feasibility(False, f"the paths cover every component of {self.label!r}")
        return Feasibility(True)


_register_refinement(FactoredMultivariateGaussian, _jointly_gaussian)


# ---------------------------------------------------------------------------
# The Gaussian random functions
# ---------------------------------------------------------------------------


def _declarations(
    name: str, output_spec: OutputSpec | None, event_spec: OutputSpec | None
) -> tuple[OutputSpec, OutputSpec]:
    """The drawn function's output declaration and the event's, each defaulting to *name*.

    The event is a function, so a declared event type is a ``FunctionSpec`` whose
    output side names the drawn function's output component. A type hole in the
    event, or in the function's output side, is filled with the output declaration.

    Raises
    ------
    TypeError
        If *output_spec* is not an ``OutputSpec`` naming one component, or
        *event_spec* is not an ``OutputSpec`` or declares a type that is not a
        ``FunctionSpec``.
    ValueError
        If the event's ``FunctionSpec`` names another output component.
    """
    output = OutputSpec(**{name: None}) if output_spec is None else output_spec
    if not isinstance(output, OutputSpec) or output._component_name is None:
        raise TypeError(
            f"output_spec of {name!r} must be an OutputSpec naming one component, got "
            f"{output_spec!r}"
        )
    if event_spec is None:
        return output, OutputSpec(**{name: FunctionSpec(output_spec=output)})
    if not isinstance(event_spec, OutputSpec):
        raise TypeError(f"event_spec of {name!r} must be an OutputSpec, got {event_spec!r}")
    declared = event_spec.spec
    if declared is None:
        return output, event_spec._with_spec(FunctionSpec(output_spec=output))
    if not isinstance(declared, FunctionSpec):
        raise TypeError(
            f"the event of the random function {name!r} is a function, so event_spec declares "
            f"a FunctionSpec; got {type(declared).__name__}"
        )
    if declared.output_spec is None:
        filled = FunctionSpec(input_spec=declared.input_spec, output_spec=output)
        return output, event_spec._with_spec(filled)
    named = declared.output_spec._component_name
    if named != output._component_name:
        raise ValueError(
            f"the event of {name!r} declares a function whose output is {named!r}, but the "
            f"drawn function names the output component {output._component_name!r}"
        )
    return output, event_spec


def _stacked(X: ArrayLike) -> Array:
    """*X* as an array whose leading axis stacks the input points."""
    X = jnp.asarray(X)
    if X.ndim == 0:
        raise ValueError("X stacks the input points along its leading axis, so it has an axis")
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
    function's, and its event is declared under the drawn function's output
    component.

    The drawn function's output component and the function-valued event's
    component both default to the label; ``output_spec`` names the former and
    ``event_spec`` the latter. A type that ``event_spec`` declares is a
    ``FunctionSpec`` naming the output component, and a type hole in it, or in
    its output side, is filled with the output declaration. The mean is the
    mean function and the variance the pointwise variance function, each a
    callable on stacked inputs.

    Shifts ``f + b``, scalings ``alpha * f`` by a scalar, output-side linear
    maps ``A @ f``, and sums ``f + g`` of independent members are again
    members, evaluated in closed form.

    Parameters
    ----------
    label : str
        The random function's label.
    output_spec : OutputSpec, optional
        The declaration of the drawn function's output, naming its component.
    event_spec : OutputSpec, optional
        The declaration of the function-valued draw, naming its component.

    Raises
    ------
    TypeError
        If *output_spec* is not an ``OutputSpec`` naming one component, or
        *event_spec* is not an ``OutputSpec`` or declares a type that is not a
        ``FunctionSpec``.
    ValueError
        If the ``FunctionSpec`` that *event_spec* declares names another output
        component.
    """

    def __init__(
        self,
        label: str,
        *,
        output_spec: OutputSpec | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        output, event = _declarations(label, output_spec, event_spec)
        self._output_spec = output
        super().__init__(label, event)

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

        Raises
        ------
        ValueError
            If *X* is a scalar, which stacks no points.
        """
        X = _stacked(X)
        mean = jnp.asarray(self.predict_mean(X))
        event_spec = OutputSpec(**{self._output_spec._component_name: None})
        if mean.size > 1 and self._joint:
            cov = self.predict_covariance(X)
            return MultivariateNormal(
                self.label, jnp.reshape(mean, (-1,)), cov=cov, event_spec=event_spec
            )
        scale = jnp.sqrt(jnp.asarray(self.predict_variance(X)))
        return Normal(self.label, mean, scale, event_spec=event_spec)

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
    label : str
        The process's label.
    mean_fn : Callable[[Array], Array]
        The mean function, evaluated at stacked input points.
    cov_kernel : Callable[[Array, Array], Array]
        The covariance kernel, evaluated at two stacks of input points.
    output_spec : OutputSpec, optional
        The declaration of the drawn function's output.
    event_spec : OutputSpec, optional
        The declaration of the function-valued draw.

    Raises
    ------
    TypeError
        If *mean_fn* or *cov_kernel* is not callable, or a declaration is
        refused as :class:`GaussianRandomFunction` refuses it.
    ValueError
        As :class:`GaussianRandomFunction` raises.
    """

    def __init__(
        self,
        label: str,
        mean_fn: Callable[[Array], Array],
        cov_kernel: Callable[[Array, Array], Array],
        *,
        output_spec: OutputSpec | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        if not callable(mean_fn) or not callable(cov_kernel):
            raise TypeError(f"the mean function and covariance kernel of {label!r} are callables")
        self._mean_fn = mean_fn
        self._cov_kernel = cov_kernel
        super().__init__(label, output_spec=output_spec, event_spec=event_spec)

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
    label : str
        The random function's label.
    basis : Callable[[Array], Array]
        The feature map, evaluated at stacked input points.
    weights : MultivariateNormal
        The law of the weight vector.
    output_spec : OutputSpec, optional
        The declaration of the drawn function's output.
    event_spec : OutputSpec, optional
        The declaration of the function-valued draw.

    Raises
    ------
    TypeError
        If *weights* is not a ``MultivariateNormal``, *basis* is not callable,
        or a declaration is refused as :class:`GaussianRandomFunction` refuses
        it.
    ValueError
        As :class:`GaussianRandomFunction` raises.
    """

    def __init__(
        self,
        label: str,
        basis: Callable[[Array], Array],
        weights: MultivariateNormal,
        *,
        output_spec: OutputSpec | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        if not isinstance(weights, MultivariateNormal):
            raise TypeError(f"weights must be a MultivariateNormal, got {type(weights).__name__}")
        if not callable(basis):
            raise TypeError(f"the basis of {label!r} is a callable")
        self._basis = basis
        self._weights = weights
        self._w_mean = weights.loc
        self._w_cov = weights.cov
        super().__init__(label, output_spec=output_spec, event_spec=event_spec)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The parameters ``basis`` and ``weights``."""
        return [("basis", format_value(self._basis)), ("weights", repr(self._weights))]

    def _features(self, X: Array) -> Array:
        """The basis at the stacked inputs *X*, checked against the weights' dimension."""
        phi = jnp.asarray(self._basis(_stacked(X)))
        if phi.ndim < 2 or phi.shape[-1] != self._w_mean.shape[0]:
            raise ValueError(
                f"the basis of {self.label!r} returns features of shape {phi.shape}, whose last "
                f"axis must be the weights' dimension {self._w_mean.shape[0]}"
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


class _LinearMapGRF(GaussianRandomFunction):
    """``h(x) = A g(x)`` for a member *g* with a vector output; constructed by ``A @ g``."""

    def __init__(self, base: GaussianRandomFunction, A: Array) -> None:
        if not isinstance(base, GaussianRandomFunction):
            raise TypeError(f"base must be a GaussianRandomFunction, got {type(base).__name__}")
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D (d_out, d_in), got shape {A.shape}")
        self._base = base
        self._A = A
        super().__init__(
            f"linear_map({base.label})", output_spec=base._output_spec, event_spec=base.event_spec
        )

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
                f"A @ f maps an output vector of size {self._A.shape[1]}, but the value of "
                f"{self._base.label!r} at each point has shape {tuple(shape[1:])}"
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


class _ShiftedGRF(GaussianRandomFunction):
    """``h(x) = g(x) + b`` for a constant *b*; constructed by ``g + b``."""

    def __init__(self, base: GaussianRandomFunction, b: Array) -> None:
        self._base = base
        self._b = b
        super().__init__(
            f"shift({base.label})", output_spec=base._output_spec, event_spec=base.event_spec
        )

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


class _ScaledGRF(GaussianRandomFunction):
    """``h(x) = alpha g(x)`` for a scalar *alpha*; constructed by ``alpha * g``."""

    def __init__(self, base: GaussianRandomFunction, alpha: Array) -> None:
        if alpha.ndim != 0:
            raise ValueError(
                f"alpha * f scales by a scalar, got shape {alpha.shape}; A @ f maps the outputs"
            )
        self._base = base
        self._alpha = alpha
        super().__init__(
            f"scale({base.label})", output_spec=base._output_spec, event_spec=base.event_spec
        )

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

    Raises
    ------
    ValueError
        If the two members' values have different shapes, which would otherwise
        broadcast.
    """
    left, right = jnp.asarray(left), jnp.asarray(right)
    if left.shape != right.shape:
        raise ValueError(
            f"f + g adds values of one shape, got {left.shape[1:]} and {right.shape[1:]}"
        )
    return left + right


class _IndependentSumGRF(GaussianRandomFunction):
    """``h(x) = g₁(x) + g₂(x)`` for independent members; constructed by ``g₁ + g₂``.

    Independence is the caller's statement, so the covariances add; a member
    added to itself raises, since it is not independent of itself.
    """

    def __init__(self, left: GaussianRandomFunction, right: GaussianRandomFunction) -> None:
        if left is right:
            raise ValueError(
                "Cannot add a GaussianRandomFunction to itself, which is not independent of "
                "itself; use 2 * f instead."
            )
        self._left = left
        self._right = right
        super().__init__(
            f"sum({left.label},{right.label})",
            output_spec=left._output_spec,
            event_spec=left.event_spec,
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
