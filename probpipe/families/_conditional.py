"""The conditional families: the linear-Gaussian kernel and the GLM assembly.

Provides:
  - ``LinearGaussianConditional`` – the kernel ``s ↦ N(A @ s + b, Σ)``, the
    conditional member of the Gaussian algebra.
  - ``GLMFamily`` – a mean-parameterized response family, with
    ``GaussianFamily``, ``BernoulliFamily``, and ``PoissonFamily``.
  - ``glm_likelihood`` – the kernel of a generalized linear model, assembled
    from a family, a link, and the linear predictor ``X @ beta``.

A GLM likelihood conditions on the slots ``X`` of shape ``("obs",
"features")``, ``beta`` of shape ``("features",)``, and ``dispersion`` when
its family takes one, and its event is the response vector of shape
``("obs",)``. Its law at a given value is ``family.build(name, link⁻¹(X @
beta), dispersion)``, so changing the family or the link changes the model
without a new class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..core._dispatch import ResolutionError
from ..core._spec_base import _unify_specs
from ..core._specs import InputSpec, NumericArraySpec, OutputSpec
from ..core.constraints import Constraint, boolean, non_negative_integer, positive, real

# The links wrap callables whose annotations a Function resolves at construction.
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions._capabilities import (
    SupportsConditionalCovariance,
    SupportsConditionalLogProb,
    SupportsConditionalMean,
    SupportsConditionalSampling,
    SupportsConditionalVariance,
)
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..linalg import LinOp
from ..values import Function
from ._backend import TFPDistribution
from ._continuous import Normal
from ._discrete import Bernoulli, Poisson

if TYPE_CHECKING:
    from ..core.record import Record
    from ..distributions._distribution import Distribution

__all__ = [
    "BernoulliFamily",
    "GLMFamily",
    "GaussianFamily",
    "LinearGaussianConditional",
    "PoissonFamily",
    "glm_likelihood",
]


# ---------------------------------------------------------------------------
# The linear-Gaussian kernel
# ---------------------------------------------------------------------------


class LinearGaussianConditional(ConditionalDistribution):
    """The linear-Gaussian kernel ``s ↦ N(A @ s + b, cov)``.

    It is the conditional member of the Gaussian algebra: composed with a
    Gaussian prior it yields a ``FactoredMultivariateGaussian``, and
    conditioning through it is exact. The given slot is ``A``'s input slot, and
    the event type is ``A``'s output type under the kernel's own component.

    Parameters
    ----------
    name : str
        The kernel's label, and the component of its event.
    A : LinOp
        The linear map of the mean, whose input slot is the given slot.
    b : Array
        The offset of the mean, shaped like ``A``'s output.
    cov : LinOp
        The covariance of the event, square over ``A``'s output.

    Raises
    ------
    NotImplementedError
        Always, until ``LinOp`` declares its input slot and output type.
    """

    def __init__(self, name: str, A: LinOp, b: Array, cov: LinOp) -> None:
        raise NotImplementedError("LinearGaussianConditional.__init__")

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The Gaussian law ``N(A @ s + b, cov)`` at the given value ``s``."""
        raise NotImplementedError("LinearGaussianConditional._condition_on")


# ---------------------------------------------------------------------------
# The links
# ---------------------------------------------------------------------------


class _Link(Function):
    """An elementwise invertible link ``g``, mapping a mean to the linear predictor.

    The forward map is the function's own evaluation, and ``_inverse`` is the
    inverse map ``g⁻¹``, which takes the linear predictor to the mean.
    """

    def __init__(
        self, name: str, forward: Callable[[Array], Array], inverse: Callable[[Array], Array]
    ) -> None:
        super().__init__(name, forward)
        object.__setattr__(self, "_inverse_map", inverse)

    def _inverse(self, y: Array) -> Array:
        """The mean at the linear predictor *y*, ``g⁻¹(y)``."""
        return self._inverse_map(y)

    def __repr__(self) -> str:
        return f"link({self.name!r})"


def _identity(mean: Array) -> Array:
    return mean


def _logit(mean: Array) -> Array:
    return jax.scipy.special.logit(mean)


def _log(mean: Array) -> Array:
    return jnp.log(mean)


_IDENTITY_LINK = _Link("identity", _identity, _identity)
_LOGIT_LINK = _Link("logit", _logit, jax.nn.sigmoid)
_LOG_LINK = _Link("log", _log, jnp.exp)


def _is_invertible(link: Any) -> bool:
    """Whether *link* is a ``Function`` that provides its inverse map, ``_inverse``."""
    return isinstance(link, Function) and callable(getattr(link, "_inverse", None))


def _require_invertible(link: Any, owner: str) -> None:
    """Raise unless *link* is an invertible ``Function``.

    Raises
    ------
    TypeError
        If *link* is not a ``Function``.
    ResolutionError
        If *link* does not provide its inverse map.
    """
    if not isinstance(link, Function):
        raise TypeError(f"{owner} takes a link Function, got {type(link).__name__}")
    if not _is_invertible(link):
        raise ResolutionError(
            f"the link {link.name!r} of {owner} is not invertible: it provides no inverse map"
        )


# ---------------------------------------------------------------------------
# The response families
# ---------------------------------------------------------------------------


class _LogRatePoisson(Poisson):
    """The Poisson family at its natural parameter, the log-rate.

    The backend scores from the log-rate directly, which keeps the log-density
    and its gradient finite where the rate underflows. The ``rate`` accessor is
    the rate the log-rate gives.
    """

    def __init__(self, name: str, log_rate: Array, *, event_spec: OutputSpec | None = None) -> None:
        self._rate = jnp.exp(log_rate)
        TFPDistribution.__init__(self, name, tfd.Poisson(log_rate=log_rate), event_spec=event_spec)


def _observation_vector(values: ArrayLike, owner: str, quantity: str) -> Array:
    """*values* as a floating vector with one *quantity* per observation.

    Raises
    ------
    ValueError
        If *values* is not one-dimensional.
    """
    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.result_type(float))
    if array.ndim != 1:
        raise ValueError(
            f"{owner} takes one {quantity} per observation, a vector, got shape {array.shape}"
        )
    return array


class GLMFamily(ABC):
    """A mean-parameterized response family of a generalized linear model.

    :meth:`build` returns the law of conditionally independent observations,
    one per entry of ``mean``. ``canonical_link`` is the family's canonical link
    ``g``, which maps the mean invertibly to the linear predictor, and
    ``has_dispersion`` declares whether :meth:`build` takes a dispersion, such
    as a Gaussian scale. A family needs only these three members. It may also
    set ``_support`` to the support of one observation, which the likelihood
    declares for its response, and it may override :meth:`_build_canonical` to
    build the law from the linear predictor of the canonical link.

    Raises
    ------
    ResolutionError
        At construction, if the canonical link is not invertible.
    """

    canonical_link: Function
    has_dispersion: bool
    _support: ClassVar[Constraint | None] = None

    def __init__(self) -> None:
        _require_invertible(self.canonical_link, type(self).__name__)

    @abstractmethod
    def build(
        self,
        name: str,
        mean: Array,
        dispersion: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ) -> Distribution:
        """The law of ``len(mean)`` conditionally independent observations with these means.

        Parameters
        ----------
        name : str
            The law's label, and the component of its event unless
            *event_spec* names another.
        mean : Array
            One mean per observation, a vector.
        dispersion : ArrayLike, optional
            The dispersion, required when the family has one and refused
            otherwise.
        event_spec : OutputSpec, optional
            The declaration of one draw, which names its component; the
            family fills its type with the response vector.

        Returns
        -------
        Distribution
            The law of the response vector, one entry per observation.

        Raises
        ------
        TypeError
            If a dispersion is missing for a family that has one, or given to a
            family that has none, or *event_spec* is not an ``OutputSpec``.
        ValueError
            If *mean* is not a vector, or *event_spec* declares a type the
            response vector does not conform to.
        """

    def _build_canonical(
        self,
        name: str,
        predictor: Array,
        dispersion: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ) -> Distribution:
        """The law of :meth:`build` at the means ``g⁻¹(predictor)``, for the canonical link ``g``.

        The likelihood builds its law here under the canonical link, where the
        linear predictor is the natural parameter. A family whose backend takes
        the natural parameter overrides this to build the law from *predictor*
        directly, which keeps the log-density and its gradient finite where the
        mean rounds to a boundary of its range. Parameters and errors are those
        of :meth:`build`, except that *predictor* takes the place of the mean,
        with one entry per observation.
        """
        mean = self.canonical_link._inverse(predictor)
        return self.build(name, mean, dispersion, event_spec=event_spec)

    def _dispersion(self, dispersion: ArrayLike | None, mean: Array) -> Array | None:
        """The dispersion checked against ``has_dispersion``, in the floating dtype of *mean*.

        Raises
        ------
        TypeError
            If it is missing for a family that has one or given to one that has none.
        """
        if self.has_dispersion and dispersion is None:
            raise TypeError(f"{type(self).__name__}.build requires a dispersion")
        if not self.has_dispersion and dispersion is not None:
            raise TypeError(f"{type(self).__name__}.build takes no dispersion")
        return None if dispersion is None else jnp.asarray(dispersion, dtype=mean.dtype)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class GaussianFamily(GLMFamily):
    """The Gaussian response family: independent normal observations.

    The canonical link is the identity, which makes the GLM a linear
    regression, and the dispersion is the observation scale, the standard
    deviation of each observation.
    """

    canonical_link = _IDENTITY_LINK
    has_dispersion = True
    _support = real

    def build(
        self,
        name: str,
        mean: Array,
        dispersion: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ) -> Distribution:
        """Independent normal observations with these means and the scale *dispersion*.

        Parameters and errors are those of :meth:`GLMFamily.build`.
        """
        mean = _observation_vector(mean, f"{type(self).__name__}.build", "mean")
        scale = self._dispersion(dispersion, mean)
        return Normal(name, mean, scale, event_spec=event_spec)


class BernoulliFamily(GLMFamily):
    """The Bernoulli response family: independent binary observations.

    The canonical link is the logit, which makes the GLM a logistic regression,
    and the family takes no dispersion. The mean of an observation is its
    probability of a one.
    """

    canonical_link = _LOGIT_LINK
    has_dispersion = False
    _support = boolean

    def build(
        self,
        name: str,
        mean: Array,
        dispersion: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ) -> Distribution:
        """Independent Bernoulli observations with the probabilities *mean*.

        Parameters and errors are those of :meth:`GLMFamily.build`.
        """
        mean = _observation_vector(mean, f"{type(self).__name__}.build", "mean")
        self._dispersion(dispersion, mean)
        return Bernoulli(name, probs=mean, event_spec=event_spec)

    def _build_canonical(
        self,
        name: str,
        predictor: Array,
        dispersion: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ) -> Distribution:
        """Independent Bernoulli observations with the log-odds *predictor*."""
        logits = _observation_vector(
            predictor, f"{type(self).__name__}._build_canonical", "linear predictor"
        )
        self._dispersion(dispersion, logits)
        return Bernoulli(name, logits=logits, event_spec=event_spec)


class PoissonFamily(GLMFamily):
    """The Poisson response family: independent count observations.

    The canonical link is the logarithm, which makes the GLM a Poisson
    regression, and the family takes no dispersion. The mean of an observation
    is its rate.
    """

    canonical_link = _LOG_LINK
    has_dispersion = False
    _support = non_negative_integer

    def build(
        self,
        name: str,
        mean: Array,
        dispersion: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ) -> Distribution:
        """Independent Poisson observations with the rates *mean*.

        Parameters and errors are those of :meth:`GLMFamily.build`.
        """
        mean = _observation_vector(mean, f"{type(self).__name__}.build", "mean")
        self._dispersion(dispersion, mean)
        return Poisson(name, mean, event_spec=event_spec)

    def _build_canonical(
        self,
        name: str,
        predictor: Array,
        dispersion: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ) -> Distribution:
        """Independent Poisson observations with the log-rates *predictor*."""
        log_rate = _observation_vector(
            predictor, f"{type(self).__name__}._build_canonical", "linear predictor"
        )
        self._dispersion(dispersion, log_rate)
        return _LogRatePoisson(name, log_rate, event_spec=event_spec)


# ---------------------------------------------------------------------------
# The GLM likelihood
# ---------------------------------------------------------------------------


def _declared_sizes(event_spec: OutputSpec, response: NumericArraySpec) -> dict[str, int]:
    """The sizes the declared array type of *event_spec* fixes for the dimensions of *response*.

    ``OutputSpec.with_spec`` checks a declared type against the response and
    then replaces it, so the sizes it fixes are read here first. A type hole
    fixes no size, and a declared type of another kind is left to
    ``with_spec``, which refuses it.

    Raises
    ------
    ValueError
        If the declared array type does not conform to *response*.
    """
    sizes: dict[str, int] = {}
    if isinstance(event_spec.spec, NumericArraySpec):
        (component,) = event_spec.components
        _unify_specs(event_spec.spec, response, sizes, f"Declared component {component!r}")
    return sizes


class _GLMLikelihood(
    ConditionalDistribution,
    SupportsConditionalSampling,
    SupportsConditionalLogProb,
    SupportsConditionalMean,
    SupportsConditionalVariance,
    SupportsConditionalCovariance,
):
    """The kernel of a GLM, which :func:`glm_likelihood` constructs.

    A slot with a fixed value is not a given slot, and its value binds the
    dimensions it declares. Binding some given slots curries to the kernel of
    the others. Each conditional capability is the matching capability of the
    law :meth:`_condition_on` returns for a value of every given slot. The law
    carries the kernel's event declaration, whose component is completed once,
    at construction. Under the family's canonical link the family builds the
    law from the linear predictor, and under another link from the mean.
    """

    def __init__(
        self,
        name: str,
        family: GLMFamily,
        link: Function,
        *,
        event_spec: OutputSpec | None,
        X: ArrayLike | None,
        dispersion: ArrayLike | None,
    ) -> None:
        if not isinstance(family, GLMFamily):
            raise TypeError(f"glm_likelihood takes a GLMFamily, got {type(family).__name__}")
        _require_invertible(link, "glm_likelihood")
        if event_spec is not None and not isinstance(event_spec, OutputSpec):
            raise TypeError(
                f"glm_likelihood takes an OutputSpec event_spec, got {type(event_spec).__name__}"
            )
        if dispersion is not None and not family.has_dispersion:
            raise TypeError(f"{type(family).__name__} takes no dispersion")
        slots = {
            "X": NumericArraySpec(("obs", "features")),
            "beta": NumericArraySpec(("features",)),
        }
        if family.has_dispersion:
            slots["dispersion"] = NumericArraySpec((), support=positive)
        response = NumericArraySpec(("obs",), support=family._support)
        if event_spec is None:
            declaration = OutputSpec.default(response, component=name)
        else:
            # The declared type's sizes bind on both sides before X is fixed against them.
            sizes = _declared_sizes(event_spec, response)
            slots = {slot: spec._substitute_dims(sizes) for slot, spec in slots.items()}
            declaration = event_spec.with_spec(response._substitute_dims(sizes))
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_link", link)
        object.__setattr__(self, "_canonical", link is family.canonical_link)
        object.__setattr__(self, "_fixed", {})
        super().__init__(name, InputSpec(slots), declaration)
        fixed: dict[str, Array] = {}
        if X is not None:
            fixed["X"] = jnp.asarray(X)
        if dispersion is not None:
            fixed["dispersion"] = jnp.asarray(dispersion)
        if fixed:
            self._fix(fixed)

    def _fix(self, values: Mapping[str, Array]) -> None:
        """Fix *values* of given slots, binding the dimensions they declare.

        Called during construction and on a fresh copy, so the kernel is never
        observed mid-change.

        Raises
        ------
        ValueError
            If a value does not conform to its slot's shape, or two values bind
            one dimension to different sizes.
        """
        bindings: dict[str, int] = {}
        for slot, value in values.items():
            self.given_spec[slot]._bind_dims_from_value(value, bindings, f"{self.name}/{slot}")
        remaining = InputSpec(
            {
                slot: spec._substitute_dims(bindings)
                for slot, spec in self.given_spec.items()
                if slot not in values
            }
        )
        object.__setattr__(
            self,
            "_spec",
            ConditionalDistributionSpec(remaining, self.event_spec.with_dim_sizes(**bindings)),
        )
        object.__setattr__(self, "_fixed", {**self._fixed, **values})

    # -- the primitive ----------------------------------------------------------

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The response law at a value of every given slot, or the curried kernel.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values of some or all given slots, by slot name.
        **kwargs : Any
            Further given values, by slot name.

        Returns
        -------
        Distribution or ConditionalDistribution
            The family's law of the response vector when every slot is bound, and
            otherwise the GLM kernel of the remaining slots.

        Raises
        ------
        KeyError
            If a bound name is not a given slot.
        ValueError
            If a value does not conform to its slot's shape.
        """
        values = self._given_values(given, kwargs)
        if set(values) == set(self.given_spec):
            return self._law(values)
        curried = self._shallow_copy()
        object.__setattr__(curried, "_provenance", None)
        curried._fix(values)
        return curried

    def _given_values(self, given: Any, kwargs: Mapping[str, Any]) -> dict[str, Array]:
        """The given values by slot name, each slot a known one.

        Raises
        ------
        KeyError
            If a name is not a given slot.
        """
        top = given.children if hasattr(given, "children") else given
        values = {**dict(top.items()), **kwargs}
        unknown = set(values) - set(self.given_spec)
        if unknown:
            raise KeyError(f"{sorted(unknown)} are not given slots of {self.name!r}")
        return {slot: jnp.asarray(value) for slot, value in values.items()}

    def _complete_values(self, given: Record | Mapping[str, Any]) -> dict[str, Array]:
        """The values of every given slot, for a conditional capability.

        Raises
        ------
        KeyError
            If a name is not a given slot, or a given slot has no value.
        """
        values = self._given_values(given, {})
        missing = set(self.given_spec) - set(values)
        if missing:
            raise KeyError(f"{self.name!r} needs a value for every given slot; {sorted(missing)}")
        return values

    def _law(self, values: Mapping[str, Array]) -> Distribution:
        """The response law at the values of every given slot.

        Raises
        ------
        ValueError
            If a value does not conform to its slot's shape.
        """
        self.given_spec.bind_dims_from_value(dict(values))
        every = {**self._fixed, **values}
        predictor = every["X"] @ every["beta"]
        dispersion = every.get("dispersion")
        if self._canonical:
            return self._family._build_canonical(
                self.name, predictor, dispersion, event_spec=self.event_spec
            )
        return self._family.build(
            self.name, self._link._inverse(predictor), dispersion, event_spec=self.event_spec
        )

    # -- the conditional capabilities ------------------------------------------

    def _conditional_sample(
        self,
        given: Record | Mapping[str, Any],
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draws of the response vector at a value of every given slot."""
        return self._law(self._complete_values(given))._sample(key, sample_shape)

    def _conditional_log_prob(self, given: Record | Mapping[str, Any], value: Any) -> Array:
        """The log-density of the response vector *value* at a value of every given slot."""
        return self._law(self._complete_values(given))._log_prob(value)

    def _conditional_mean(self, given: Record | Mapping[str, Any]) -> Array:
        """The mean response vector, ``link⁻¹(X @ beta)``, at a value of every given slot."""
        return self._law(self._complete_values(given))._mean()

    def _conditional_variance(self, given: Record | Mapping[str, Any]) -> Array:
        """The per-observation variance at a value of every given slot."""
        return self._law(self._complete_values(given))._variance()

    def _conditional_cov(self, given: Record | Mapping[str, Any]) -> LinOp:
        """The covariance of the response vector at a value of every given slot."""
        return self._law(self._complete_values(given))._cov()

    def __repr__(self) -> str:
        return (
            f"glm_likelihood(name={self.name!r}, family={type(self._family).__name__}, "
            f"link={self._link.name!r})"
        )


def glm_likelihood(
    name: str,
    family: GLMFamily,
    link: Function | None = None,
    *,
    event_spec: OutputSpec | None = None,
    X: Array | None = None,
    dispersion: ArrayLike | None = None,
) -> ConditionalDistribution:
    """The kernel of a generalized linear model, from a family, a link, and ``X @ beta``.

    The kernel's given slots are ``X`` of shape ``("obs", "features")``,
    ``beta`` of shape ``("features",)``, and ``dispersion`` when the family has
    one, and its event is the response vector of shape ``("obs",)``. Its law at
    a given value is ``family.build(name, link⁻¹(X @ beta), dispersion,
    event_spec=event_spec)``: ``GaussianFamily`` with the identity link is linear
    regression, ``BernoulliFamily`` with the logit is logistic regression, and
    ``PoissonFamily`` with the logarithm is Poisson regression. A value of ``X``
    or of the dispersion supplied here is fixed at construction, as
    ``condition_on`` would bind it, and ``X`` binds the dimensions ``obs`` and
    ``features``; the dimensions are symbolic otherwise.

    Parameters
    ----------
    name : str
        The kernel's label, and the component of the response unless
        *event_spec* names another.
    family : GLMFamily
        The response family.
    link : Function, optional
        The invertible link from the mean to the linear predictor. Defaults to
        the family's canonical link.
    event_spec : OutputSpec, optional
        The declaration of the response, passed through to the family.
    X : Array, optional
        The design matrix, of shape ``(obs, features)``, fixed at construction.
    dispersion : ArrayLike, optional
        The dispersion, fixed at construction; only for a family that has one.

    Returns
    -------
    ConditionalDistribution
        The GLM kernel, which claims conditional sampling, log-density, mean,
        variance, and covariance.

    Raises
    ------
    TypeError
        If *family* is not a ``GLMFamily``, *link* is not a ``Function``,
        *event_spec* is not an ``OutputSpec``, or a dispersion is given to a
        family that has none.
    ResolutionError
        If *link* is not invertible.
    ValueError
        If *X* is not a matrix, *event_spec* declares a type the response vector
        does not conform to or a number of observations that *X* contradicts, or
        the event's component is also a given slot's name.
    """
    return _GLMLikelihood(
        name,
        family,
        family.canonical_link if link is None and isinstance(family, GLMFamily) else link,
        event_spec=event_spec,
        X=X,
        dispersion=dispersion,
    )
