"""The multivariate parametric families.

``MultivariateNormal``, ``Dirichlet``, ``Multinomial``, ``Wishart``, and
``VonMisesFisher`` each derive their event term spec from their parameters and
take an ``event_spec`` declaration that names the event's component.

A family's parameters describe one law over an array; a batch of separate laws
is a ``DistributionBatch``. Each family claims the moments its backend
computes: all but the Wishart claim the covariance, all but the von
Mises-Fisher claim the variance, and the multivariate normal also claims the
quantile of each coordinate, from its normal marginals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _as_float_array, _promote_floats
from ..core._specs import OutputSpec
from ..core.constraints import (
    Constraint,
    non_negative_integer,
    positive_definite,
    real,
    simplex,
    sphere,
)
from ..custom_types import Array, ArrayLike
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsMean,
    SupportsQuantile,
    SupportsVariance,
)
from ..linalg import CholeskyLinOp, DenseLinOp, LinOp, TriangularLinOp
from ._backend import TFPDistribution

__all__ = [
    "Dirichlet",
    "Multinomial",
    "MultivariateNormal",
    "VonMisesFisher",
    "Wishart",
]


# ---------------------------------------------------------------------------
# MultivariateNormal
# ---------------------------------------------------------------------------


class MultivariateNormal(TFPDistribution):
    """
    Multivariate normal (Gaussian) distribution.

    Its covariance keeps the structure it was given: a ``LinOp`` as it is, a
    dense matrix as a ``DenseLinOp``, and a Cholesky factor as a
    ``CholeskyLinOp`` over it. The quantile of each coordinate is that of its
    normal marginal, ``loc + sqrt(cov_ii) Φ⁻¹(q)``.

    Parameters
    ----------
    name : str
        Distribution name.
    loc : array-like, shape ``(d,)``
        Mean vector.
    scale_tril : array-like, shape ``(d, d)``, optional
        Lower-triangular Cholesky factor of the covariance.  Exactly one of
        *scale_tril* or *cov* must be provided.
    cov : LinOp or array-like, shape ``(d, d)``, optional
        Covariance, as an operator or a matrix, Cholesky-decomposed for the
        backend.
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *scale_tril* and *cov* is given, *cov* does not
        match the length of *loc*, *loc* has more than one axis, or
        *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset(
        {SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile}
    )

    def __init__(
        self,
        name: str,
        loc: ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        cov: LinOp | ArrayLike | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if scale_tril is not None and cov is not None:
            raise ValueError("Provide exactly one of scale_tril or cov, not both.")

        operator = cov if isinstance(cov, LinOp) else None
        if operator is not None:
            cov = operator.to_dense()
        if scale_tril is not None:
            _, (loc, scale_tril) = _promote_floats(loc, scale_tril)
        elif cov is not None:
            _, (loc, cov) = _promote_floats(loc, cov)
        else:
            raise ValueError("One of scale_tril or cov must be provided.")

        if loc.ndim == 0:
            loc = loc.reshape(1)

        if cov is not None:
            if cov.shape != (loc.shape[0], loc.shape[0]):
                raise ValueError(f"cov shape {cov.shape} does not match loc length {loc.shape[0]}.")
            scale_tril = jnp.linalg.cholesky(cov)

        self._loc = loc
        self._scale_tril = scale_tril
        self._given_cov = operator if operator is not None else cov
        backend = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        super().__init__(name, backend, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale_tril(self) -> Array:
        return self._scale_tril

    @property
    def cov(self) -> Array:
        """Full covariance matrix (computed from Cholesky factor)."""
        return self._scale_tril @ self._scale_tril.T

    @property
    def dim(self) -> int:
        return self._loc.shape[0]

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return real

    # -- the covariance and the quantiles ------------------------------------

    def _cov(self) -> LinOp:
        """The covariance operator, with the structure it was given."""
        given = self._given_cov
        if isinstance(given, LinOp):
            return given
        if given is not None:
            return DenseLinOp(given)
        return CholeskyLinOp(TriangularLinOp(self._scale_tril, lower=True))

    def _quantile(self, q: ArrayLike) -> Array:
        """The quantile of each coordinate's normal marginal, of shape ``(*q.shape, d)``."""
        levels = jnp.asarray(q, dtype=self._loc.dtype)
        scale = jnp.sqrt(jnp.diagonal(self.cov))
        return self._loc + scale * jax.scipy.special.ndtri(levels)[..., None]


# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------


class Dirichlet(TFPDistribution):
    """
    Dirichlet distribution over the probability simplex.

    Parameters
    ----------
    name : str
        Distribution name.
    concentration : array-like, shape ``(k,)``
        Positive concentration (alpha) parameters.
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *concentration* is a scalar or has more than one axis, or
        *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})

    def __init__(
        self, name: str, concentration: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        concentration = _as_float_array(concentration)
        if concentration.ndim == 0:
            raise ValueError("concentration must be at least 1-D.")

        self._concentration = concentration
        backend = tfd.Dirichlet(concentration=concentration)
        super().__init__(name, backend, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def dim(self) -> int:
        return self._concentration.shape[-1]

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return simplex


# ---------------------------------------------------------------------------
# Multinomial
# ---------------------------------------------------------------------------


class Multinomial(TFPDistribution):
    """
    Multinomial distribution over count vectors.

    Exactly one of *probs* or *logits* must be provided.

    Parameters
    ----------
    name : str
        Distribution name.
    total_count : int or array-like
        Number of trials.
    probs : array-like, shape ``(k,)``, optional
        Event probabilities (need not be normalised).
    logits : array-like, shape ``(k,)``, optional
        Log-odds of each event.
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *probs* and *logits* is given, the parameters
        describe more than one law, or *event_spec* declares a type that one
        draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})

    def __init__(
        self,
        name: str,
        total_count: int | ArrayLike,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        *,
        event_spec: OutputSpec | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be provided.")

        if probs is not None:
            _, (total_count, probs) = _promote_floats(total_count, probs)
            self._probs = probs
            self._logits = None
            backend = tfd.Multinomial(total_count=total_count, probs=probs)
        else:
            _, (total_count, logits) = _promote_floats(total_count, logits)
            self._logits = logits
            self._probs = None
            backend = tfd.Multinomial(total_count=total_count, logits=logits)

        self._total_count = total_count
        super().__init__(name, backend, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def total_count(self) -> Array:
        return self._total_count

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return non_negative_integer


# ---------------------------------------------------------------------------
# Wishart
# ---------------------------------------------------------------------------


class Wishart(TFPDistribution):
    """
    Wishart distribution over positive-definite matrices.

    Exactly one of *scale_tril* or *scale* must be provided.

    Parameters
    ----------
    name : str
        Distribution name.
    df : float or array-like
        Degrees of freedom (must be >= dimension).
    scale_tril : array-like, shape ``(d, d)``, optional
        Lower-triangular Cholesky factor of the scale matrix.
    scale : array-like, shape ``(d, d)``, optional
        Full scale matrix (Cholesky-decomposed internally).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *scale_tril* and *scale* is given, the parameters
        describe more than one law, or *event_spec* declares a type that one
        draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance})

    def __init__(
        self,
        name: str,
        df: float | ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        scale: ArrayLike | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if scale_tril is not None and scale is not None:
            raise ValueError("Provide exactly one of scale_tril or scale, not both.")
        if scale_tril is None and scale is None:
            raise ValueError("One of scale_tril or scale must be provided.")

        if scale is not None:
            _, (df, scale) = _promote_floats(df, scale)
            scale_tril = jnp.linalg.cholesky(scale)
        else:
            _, (df, scale_tril) = _promote_floats(df, scale_tril)

        self._df = df
        self._scale_tril = scale_tril
        backend = tfd.WishartTriL(df=df, scale_tril=scale_tril)
        super().__init__(name, backend, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def df(self) -> Array:
        return self._df

    @property
    def scale_tril(self) -> Array:
        return self._scale_tril

    @property
    def scale(self) -> Array:
        """Full scale matrix (computed from Cholesky factor)."""
        return self._scale_tril @ self._scale_tril.T

    @property
    def dim(self) -> int:
        return self._scale_tril.shape[-1]

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return positive_definite


# ---------------------------------------------------------------------------
# VonMisesFisher
# ---------------------------------------------------------------------------


class VonMisesFisher(TFPDistribution):
    """
    Von Mises-Fisher distribution on the unit hypersphere.

    Parameters
    ----------
    name : str
        Distribution name.
    mean_direction : array-like, shape ``(d,)``
        Unit vector giving the mean direction.
    concentration : float or array-like
        Scalar concentration parameter (kappa >= 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If the parameters describe more than one law, or *event_spec* declares
        a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsCovariance})

    def __init__(
        self,
        name: str,
        mean_direction: ArrayLike,
        concentration: float | ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (mean_direction, concentration) = _promote_floats(mean_direction, concentration)

        self._mean_direction = mean_direction
        self._concentration = concentration
        backend = tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)
        super().__init__(name, backend, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def mean_direction(self) -> Array:
        return self._mean_direction

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def dim(self) -> int:
        return self._mean_direction.shape[-1]

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return sphere
