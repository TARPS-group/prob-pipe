"""The multivariate parametric families.

``MultivariateNormal``, ``Dirichlet``, ``Multinomial``, ``Wishart``, and
``VonMisesFisher`` each derive their event term spec from their parameters and
take an ``event_spec`` declaration that names the event's component.

Parameters with more axes than one law needs give one law over independent
rows, whose leading event axes are those axes: a ``MultivariateNormal`` whose
``loc`` has shape ``(n, d)`` draws an ``(n, d)`` array of n independent rows,
its density sums the rows' densities, and its covariance over the flattened
draw is block-diagonal. Separate laws form a ``DistributionBatch``. Each family
claims the moments it computes in
closed form: all but the Wishart claim the covariance, and every family claims
the variance, the von Mises-Fisher's as the diagonal of its backend's
covariance. The multivariate normal also claims the quantile of each
coordinate, from its normal marginals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _as_float_array, _promote_floats
from ..core._dispatch import MathematicalDomainError
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
from ._backend import TFPDistribution, _block_diagonal, _coordinates

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


def _rank_tolerant_factor(cov: Array) -> Array:
    """A lower-triangular ``L`` with ``L Lᵀ`` the positive semidefinite part of *cov*.

    The root ``R = V max(Λ, 0)^½`` of the symmetric eigendecomposition
    ``cov = V Λ Vᵀ`` is reduced to a triangular factor by the QR decomposition
    ``Rᵀ = Q U``, since ``R Rᵀ = Uᵀ U``, with signs that make the diagonal
    nonnegative.
    """
    eigenvalues, vectors = jnp.linalg.eigh(cov)
    root = vectors * jnp.sqrt(jnp.clip(eigenvalues, 0.0))[..., None, :]
    _, upper = jnp.linalg.qr(jnp.swapaxes(root, -1, -2))
    diagonal = jnp.diagonal(upper, axis1=-2, axis2=-1)
    signs = jnp.where(diagonal < 0, -1.0, 1.0).astype(upper.dtype)
    return jnp.swapaxes(upper * signs[..., :, None], -1, -2)


def _covariance_factor(cov: Array) -> tuple[Array, bool | Array]:
    """A lower-triangular factor of *cov* and whether *cov* is positive definite.

    A positive definite *cov* has its Cholesky factor, and a singular one,
    whose Cholesky factorization fails, the rank-tolerant factor. The answer is
    a ``bool`` for a concrete *cov*, and a traced boolean otherwise, on which
    the factor is chosen when the computation runs.
    """
    cholesky = jnp.linalg.cholesky(cov)
    positive_definite = jnp.all(jnp.isfinite(cholesky))
    if isinstance(positive_definite, jax.core.Tracer):
        factor = jax.lax.cond(
            positive_definite,
            lambda factor, _: factor,
            lambda _, matrix: _rank_tolerant_factor(matrix),
            cholesky,
            cov,
        )
        return factor, positive_definite
    if bool(positive_definite):
        return cholesky, True
    return _rank_tolerant_factor(cov), False


def _nonsingular_factor(scale_tril: Array) -> bool | Array:
    """Whether the triangular *scale_tril* is nonsingular, a traced boolean when it is traced."""
    nonsingular = jnp.all(jnp.diagonal(scale_tril, axis1=-2, axis2=-1) != 0)
    return nonsingular if isinstance(nonsingular, jax.core.Tracer) else bool(nonsingular)


class MultivariateNormal(TFPDistribution):
    """
    Multivariate normal (Gaussian) distribution.

    Its covariance keeps the structure it was given: a ``LinOp`` as it is, a
    dense matrix as a ``DenseLinOp``, and a Cholesky factor as a
    ``CholeskyLinOp`` over it. The variance is the covariance's diagonal, and
    the quantile of each coordinate is that of its normal marginal,
    ``loc + sqrt(cov_ii) Φ⁻¹(q)``.

    **A singular covariance.** A covariance that is positive semidefinite but
    singular has no Cholesky factor, so the family draws through a
    rank-tolerant one: the symmetric eigendecomposition with its eigenvalues
    clipped at zero, reduced to a lower-triangular factor. The law is then
    concentrated on an affine subspace and has no density with respect to
    Lebesgue measure, so its log-density raises ``MathematicalDomainError``. A
    covariance whose Cholesky factorization fails is treated as singular. A
    traced covariance chooses its factor when the computation runs, and its
    log-density is NaN where the covariance is singular.

    **Independent rows.** Parameters with axes beyond one law's, a *loc* of
    shape ``(n, d)`` or a covariance or factor of shape ``(n, d, d)``, give one
    law over an ``(n, d)`` array of n independent rows. Its density sums the
    rows' densities, its mean and variance have the shape ``(n, d)``, its
    covariance over the flattened draw is the block-diagonal matrix of the
    rows' covariances, and its quantiles are per coordinate.

    Parameters
    ----------
    component : str
        The component of the law's event.
    loc : array-like, shape ``(..., d)``
        Mean vector, or one per row.
    scale_tril : array-like, shape ``(..., d, d)``, optional
        Lower-triangular Cholesky factor of the covariance, or one per row.
        Exactly one of *scale_tril* or *cov* must be provided.
    cov : LinOp or array-like, shape ``(..., d, d)``, optional
        Covariance, as an operator or a matrix, or one matrix per row, factored
        for the backend.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *scale_tril* and *cov* is given, the trailing axes
        of *cov* are not ``(d, d)`` for the length ``d`` of *loc*, or
        *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset(
        {SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile}
    )

    def __init__(
        self,
        component: str,
        loc: ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        cov: LinOp | ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if scale_tril is not None and cov is not None:
            raise ValueError("scale_tril and cov cannot both be given; pass one of them")

        operator = cov if isinstance(cov, LinOp) else None
        if operator is not None:
            cov = operator.to_dense()
        if scale_tril is not None:
            _, (loc, scale_tril) = _promote_floats(loc, scale_tril)
        elif cov is not None:
            _, (loc, cov) = _promote_floats(loc, cov)
        else:
            raise ValueError("one of scale_tril or cov must be provided")

        if loc.ndim == 0:
            loc = loc.reshape(1)

        if cov is not None:
            d = loc.shape[-1]
            if cov.ndim < 2 or cov.shape[-2:] != (d, d):
                raise ValueError(
                    f"cov shape {cov.shape} does not match loc length {d}: its trailing axes "
                    f"must be ({d}, {d})"
                )
            scale_tril, positive_definite = _covariance_factor(cov)
        else:
            positive_definite = _nonsingular_factor(scale_tril)

        self._loc = loc
        self._scale_tril = scale_tril
        self._dense_cov = cov
        self._given_cov = operator if operator is not None else cov
        self._positive_definite = positive_definite
        backend = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        super().__init__(component, backend, label=label, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale_tril(self) -> Array:
        """A lower-triangular factor ``L`` of the covariance, ``L Lᵀ = cov``.

        It is the Cholesky factor when the covariance is positive definite, and
        the rank-tolerant factor when it is singular.
        """
        return self._scale_tril

    @property
    def cov(self) -> Array:
        """The covariance matrix of a row: the one given, or ``L Lᵀ`` for a given factor ``L``."""
        if self._dense_cov is not None:
            return self._dense_cov
        return self._scale_tril @ jnp.swapaxes(self._scale_tril, -1, -2)

    @property
    def dim(self) -> int:
        """The length ``d`` of a row."""
        return self._loc.shape[-1]

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return real

    # -- the density, the moments, and the quantiles -------------------------

    def _log_prob(self, value: ArrayLike) -> Array:
        """The normal log-density, keeping the leading axes of *value*.

        Parameters
        ----------
        value : ArrayLike
            A draw, or a batch of draws along leading axes.

        Returns
        -------
        Array
            The log-density, which is ``nan`` when a traced covariance is singular.

        Raises
        ------
        MathematicalDomainError
            If the covariance is singular, since the law then has no density
            with respect to Lebesgue measure.
        """
        positive_definite = self._positive_definite
        if positive_definite is False:
            raise MathematicalDomainError(
                f"the covariance of {self.label!r} is singular, so the law is concentrated on an "
                f"affine subspace and has no density with respect to Lebesgue measure"
            )
        log_density = super()._log_prob(value)
        if positive_definite is True:
            return log_density
        return jnp.where(positive_definite, log_density, jnp.nan)

    def _variance(self) -> Array:
        """The variance of each coordinate, the diagonal of its row's covariance."""
        return jnp.broadcast_to(jnp.diagonal(self.cov, axis1=-2, axis2=-1), self.event_shape)

    def _cov(self) -> LinOp:
        """The covariance operator over the flattened draw.

        One row's keeps the structure it was given, and independent rows have
        the block-diagonal matrix of the rows' covariances.
        """
        if len(self.event_shape) > 1:
            rows = jnp.broadcast_to(self.cov, (*self.event_shape, self.dim))
            return DenseLinOp(_block_diagonal(rows))
        given = self._given_cov
        if isinstance(given, LinOp):
            return given
        if given is not None:
            return DenseLinOp(given)
        return CholeskyLinOp(TriangularLinOp(self._scale_tril, lower=True))

    def _quantile(self, q: ArrayLike) -> Array:
        """The quantile of each coordinate's normal marginal, ``(*q.shape, *event_shape)``."""
        levels = jnp.asarray(q, dtype=self._loc.dtype)
        standard = jax.scipy.special.ndtri(levels)
        standard = jnp.reshape(standard, standard.shape + (1,) * len(self.event_shape))
        return self._tfp_dist.mean() + jnp.sqrt(self._variance()) * standard


# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------


class Dirichlet(TFPDistribution):
    """
    Dirichlet distribution over the probability simplex.

    Parameters
    ----------
    component : str
        The component of the law's event.
    concentration : array-like, shape ``(..., k)``
        Positive concentration (alpha) parameters, or one vector per row.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *concentration* is a scalar, or *event_spec* declares a type that one
        draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})

    def __init__(
        self,
        component: str,
        concentration: ArrayLike,
        *,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        concentration = _as_float_array(concentration)
        if concentration.ndim == 0:
            raise ValueError(f"concentration must be at least 1-D, got shape {concentration.shape}")

        self._concentration = concentration
        backend = tfd.Dirichlet(concentration=concentration)
        super().__init__(component, backend, label=label, event_spec=event_spec)

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
    component : str
        The component of the law's event.
    total_count : int or array-like
        Number of trials, or one per row.
    probs : array-like, shape ``(..., k)``, optional
        Event probabilities (need not be normalised), or one vector per row.
    logits : array-like, shape ``(..., k)``, optional
        Log-odds of each event, or one vector per row.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *probs* and *logits* is given, or *event_spec*
        declares a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})

    def __init__(
        self,
        component: str,
        total_count: int | ArrayLike,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        *,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("exactly one of probs or logits must be provided")

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
        super().__init__(component, backend, label=label, event_spec=event_spec)

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
    component : str
        The component of the law's event.
    df : float or array-like
        Degrees of freedom (must be >= dimension), or one per row.
    scale_tril : array-like, shape ``(..., d, d)``, optional
        Lower-triangular Cholesky factor of the scale matrix, or one per row.
    scale : array-like, shape ``(..., d, d)``, optional
        Full scale matrix (Cholesky-decomposed internally), or one per row.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *scale_tril* and *scale* is given, or *event_spec*
        declares a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance})

    def __init__(
        self,
        component: str,
        df: float | ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        scale: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if scale_tril is not None and scale is not None:
            raise ValueError("scale_tril and scale cannot both be given; pass one of them")
        if scale_tril is None and scale is None:
            raise ValueError("one of scale_tril or scale must be provided")

        if scale is not None:
            _, (df, scale) = _promote_floats(df, scale)
            scale_tril = jnp.linalg.cholesky(scale)
        else:
            _, (df, scale_tril) = _promote_floats(df, scale_tril)

        self._df = df
        self._scale_tril = scale_tril
        backend = tfd.WishartTriL(df=df, scale_tril=scale_tril)
        super().__init__(component, backend, label=label, event_spec=event_spec)

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
        return self._scale_tril @ jnp.swapaxes(self._scale_tril, -1, -2)

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

    Its variance is the diagonal of its covariance, which the backend computes.

    Parameters
    ----------
    component : str
        The component of the law's event.
    mean_direction : array-like, shape ``(..., d)``
        Unit vector giving the mean direction, or one per row.
    concentration : float or array-like
        Scalar concentration parameter (kappa >= 0), or one per row.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *component* is not a string, or *event_spec* is not an
        :class:`~probpipe.OutputSpec` or exposes a record.
    ValueError
        If *component* is not a valid component name, or *event_spec* names
        another component or declares a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})

    def __init__(
        self,
        component: str,
        mean_direction: ArrayLike,
        concentration: float | ArrayLike,
        *,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        _, (mean_direction, concentration) = _promote_floats(mean_direction, concentration)

        self._mean_direction = mean_direction
        self._concentration = concentration
        backend = tfd.VonMisesFisher(mean_direction=mean_direction, concentration=concentration)
        super().__init__(component, backend, label=label, event_spec=event_spec)

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

    def _variance(self) -> Array:
        """The variance of each coordinate, the diagonal of its row's covariance."""
        return jnp.diagonal(_coordinates(self._tfp_dist).covariance(), axis1=-2, axis2=-1)
