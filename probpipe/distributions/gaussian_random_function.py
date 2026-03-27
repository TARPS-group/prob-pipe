"""
Gaussian random function classes for ProbPipe.

Provides:
  - ``GaussianRandomFunction``    – Abstract base for random functions with
                                    Gaussian predictive distributions.
                                    Supports algebraic operations that preserve
                                    the Gaussian property: ``A @ grf``,
                                    ``grf + b``, ``alpha * grf``,
                                    ``grf1 + grf2``.
  - ``LinearBasisFunction``       – f(x) = a + Φ(x) @ w, w ~ N(m, C).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from ..core.distribution import _auto_key
from .._utils import prod
from .random_function import ArrayRandomFunction

# Delay import to avoid circular import at module level; these are
# imported from the *same* package, so we import lazily inside methods
# where needed.  For type annotations we use strings.


# ---------------------------------------------------------------------------
# GaussianRandomFunction
# ---------------------------------------------------------------------------


class GaussianRandomFunction(ArrayRandomFunction):
    """Abstract random function with Gaussian predictive distributions.

    Subclasses implement :meth:`predict_mean` and :meth:`predict_variance`
    at minimum.  If the model supports joint modes it must also implement
    :meth:`predict_covariance`.

    The base :meth:`predict` method assembles these into the appropriate
    :class:`~probpipe.Normal` or :class:`~probpipe.MultivariateNormal`
    distribution with the correct batch/event shape partition.

    This class is not restricted to GPs — any model that produces Gaussian
    (or Gaussian-approximated) predictions can inherit from it.
    """

    @abstractmethod
    def predict_mean(self, X: Array) -> Array:
        """Predictive mean at input points X.

        Parameters
        ----------
        X : Array
            Shape ``(*extra_batch, n, *input_shape)``.

        Returns
        -------
        Array
            Shape ``(*extra_batch, n, *output_shape)``.
        """
        ...

    @abstractmethod
    def predict_variance(self, X: Array) -> Array:
        """Marginal predictive variance at input points X.

        Parameters
        ----------
        X : Array
            Shape ``(*extra_batch, n, *input_shape)``.

        Returns
        -------
        Array
            Shape ``(*extra_batch, n, *output_shape)``.
            Each element is the marginal variance of the corresponding
            scalar prediction.
        """
        ...

    def predict_covariance(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Array:
        """Predictive covariance matrix.

        Required only if the model supports joint modes.  Returns the
        covariance over whichever axes are flagged as joint.

        Parameters
        ----------
        X : Array
            Shape ``(*extra_batch, n, *input_shape)``.
        joint_inputs : bool
            Include cross-input covariance.
        joint_outputs : bool
            Include cross-output covariance.

        Returns
        -------
        Array
            The shape depends on the joint flags:

            - ``joint_inputs=True, joint_outputs=True``:
              ``(*extra_batch, n * prod(output_shape), n * prod(output_shape))``

            - ``joint_inputs=True, joint_outputs=False``:
              ``(*extra_batch, *output_shape, n, n)``

            - ``joint_inputs=False, joint_outputs=True``:
              ``(*extra_batch, n, prod(output_shape), prod(output_shape))``

        Raises
        ------
        NotImplementedError
            If the subclass does not support the requested joint mode.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_covariance "
            f"for joint_inputs={joint_inputs}, joint_outputs={joint_outputs}."
        )

    # -- Algebraic operations preserving Gaussianity -------------------------

    def __rmatmul__(self, other):
        """``A @ grf`` — linear map of outputs.

        *other* must be a 2-D array of shape ``(d_out, d_w)`` and
        ``self.output_shape`` must be 1-D ``(d_w,)``.
        """
        return _LinearMapGRF(self, jnp.asarray(other, dtype=jnp.float32))

    def __add__(self, other):
        if isinstance(other, GaussianRandomFunction):
            return _IndependentSumGRF(self, other)
        return _ShiftedGRF(self, jnp.asarray(other, dtype=jnp.float32))

    def __radd__(self, other):
        if isinstance(other, GaussianRandomFunction):
            return _IndependentSumGRF(other, self)
        return _ShiftedGRF(self, jnp.asarray(other, dtype=jnp.float32))

    def __mul__(self, other):
        return _ScaledGRF(self, jnp.asarray(other, dtype=jnp.float32))

    def __rmul__(self, other):
        return _ScaledGRF(self, jnp.asarray(other, dtype=jnp.float32))

    def __neg__(self):
        return _ScaledGRF(self, jnp.float32(-1.0))

    def __sub__(self, other):
        if isinstance(other, GaussianRandomFunction):
            return self + (-other)
        return _ShiftedGRF(self, -jnp.asarray(other, dtype=jnp.float32))

    def __rsub__(self, other):
        return (-self) + other

    # -- Prediction assembly ---------------------------------------------------

    def predict(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ):
        """Assemble a Gaussian distribution from mean / variance / covariance.

        1. Compute the mean: shape ``(*extra_batch, n, *output_shape)``.
        2. If neither joint flag is set, return a batch of independent
           :class:`~probpipe.Normal` distributions from marginal variances.
        3. Otherwise, compute the appropriate covariance matrix via
           :meth:`predict_covariance` and return a
           :class:`~probpipe.MultivariateNormal`.

        Subclasses may override this if they need non-standard assembly
        (e.g. structured covariance representations).
        """
        from . import MultivariateNormal, Normal

        mean = self.predict_mean(X)  # (*eb, n, *out)

        # -- Fully marginal ---------------------------------------------------
        if not joint_inputs and not joint_outputs:
            variance = self.predict_variance(X)
            return Normal(loc=mean, scale=jnp.sqrt(variance))

        # -- At least one joint axis — need covariance ------------------------
        cov = self.predict_covariance(
            X, joint_inputs=joint_inputs, joint_outputs=joint_outputs
        )
        extra_batch, n = self._parse_X(X)
        d_out = prod(self._output_shape)

        scale_tril = jnp.linalg.cholesky(cov)

        if joint_inputs and joint_outputs:
            flat_dim = n * d_out if self._output_shape else n
            flat_mean = mean.reshape(*extra_batch, flat_dim)
            return MultivariateNormal(loc=flat_mean, scale_tril=scale_tril)

        if joint_inputs and not joint_outputs:
            # Joint over n, independent over outputs.
            # mean: (*eb, n, *out) → need (*eb, *out, n)
            if self._output_shape:
                ndim_eb = len(extra_batch)
                ndim_out = len(self._output_shape)
                source_axes = list(range(ndim_eb + 1, ndim_eb + 1 + ndim_out))
                dest_axes = list(range(ndim_eb, ndim_eb + ndim_out))
                mean_t = jnp.moveaxis(mean, source_axes, dest_axes)
            else:
                mean_t = mean  # (*eb, n) — nothing to rearrange
            return MultivariateNormal(loc=mean_t, scale_tril=scale_tril)

        # joint_outputs only (not joint_inputs)
        # mean: (*eb, n, *out) → flatten output dims: (*eb, n, d_out)
        flat_mean = mean.reshape(*extra_batch, n, d_out)
        return MultivariateNormal(loc=flat_mean, scale_tril=scale_tril)


# ---------------------------------------------------------------------------
# LinearBasisFunction
# ---------------------------------------------------------------------------


class LinearBasisFunction(GaussianRandomFunction):
    r"""Linear model with fixed Gaussian weights.

    Implements the model:

    .. math::

        f(x) = a + \Phi(x)\, w

    where :math:`w \sim \mathcal{N}(m, C)` is a fixed Gaussian distribution
    over weights, :math:`\Phi(x)` is a user-supplied feature map, and
    :math:`a` is an optional bias.

    The feature map maps each input to a vector (scalar output) or matrix
    (multi-output) of basis-function evaluations.  The weight distribution
    is supplied as a :class:`~probpipe.MultivariateNormal`.

    This model always supports ``joint_inputs=True`` since the cross-input
    covariance :math:`\Phi(x_i)\, C\, \Phi(x_j)^T` is available
    analytically.

    Parameters
    ----------
    feature_map : callable
        Maps input ``X`` of shape ``(*extra_batch, n, *input_shape)`` to
        features:

        - Scalar output: shape ``(*extra_batch, n, d_w)``
        - Multi-output: shape ``(*extra_batch, n, d_out, d_w)``

        where ``d_w`` matches the dimensionality of *weights*.
    weights : MultivariateNormal
        Fixed Gaussian distribution over weight vector, with
        ``event_shape = (d_w,)``.
    input_shape : tuple of int
        Shape of a single input point.
    output_shape : tuple of int, optional
        Shape of a single output.  Default ``()`` (scalar).
    bias : array-like, optional
        Additive bias of shape ``(*output_shape,)``.  Defaults to zero.
    """

    supports_joint_inputs = True

    def __init__(
        self,
        feature_map: Callable[[Array], Array],
        weights,  # MultivariateNormal — avoid top-level import
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...] = (),
        bias: ArrayLike | None = None,
    ) -> None:
        from . import MultivariateNormal

        if not isinstance(weights, MultivariateNormal):
            raise TypeError(
                f"weights must be a MultivariateNormal, "
                f"got {type(weights).__name__}"
            )

        self._feature_map = feature_map
        self._weights = weights
        self._w_mean = weights.loc          # (d_w,)
        self._w_cov = weights.cov           # (d_w, d_w)

        if bias is not None:
            self._bias = jnp.asarray(bias, dtype=jnp.float32)
        elif output_shape:
            self._bias = jnp.zeros(output_shape, dtype=jnp.float32)
        else:
            self._bias = jnp.float32(0.0)

        super().__init__(input_shape=input_shape, output_shape=output_shape)

        # Multi-output with shared weights implies coupled outputs.
        if self._output_shape:
            self.supports_joint_outputs = True

    # -- GaussianRandomFunction interface -----------------------------------

    def predict_mean(self, X: Array) -> Array:
        r"""Predictive mean: ``a + Phi(X) @ m``.

        Returns shape ``(*extra_batch, n, *output_shape)``.
        """
        phi = self._feature_map(X)  # (*eb, n, [*out,] d_w)
        return self._bias + jnp.einsum("...w,w->...", phi, self._w_mean)

    def predict_variance(self, X: Array) -> Array:
        """Marginal predictive variance.

        For scalar output: ``diag(Phi(X) C Phi(X)^T)`` element-wise.
        Returns shape ``(*extra_batch, n, *output_shape)``.
        """
        phi = self._feature_map(X)
        return jnp.einsum("...w,wv,...v->...", phi, self._w_cov, phi)

    def predict_covariance(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Array:
        if not joint_inputs and not joint_outputs:
            raise ValueError(
                "Use predict_variance for fully marginal predictions."
            )

        phi = self._feature_map(X)  # (*eb, n, [*out,] d_w)
        extra_batch, n = self._parse_X(X)

        if joint_inputs and not joint_outputs:
            if self._output_shape:
                d_out = prod(self._output_shape)
                phi_flat = phi.reshape(*extra_batch, n, d_out, -1)
                cov = jnp.einsum(
                    "...iow,wv,...jov->...oij",
                    phi_flat, self._w_cov, phi_flat
                )
                return cov.reshape(*extra_batch, *self._output_shape, n, n)
            return jnp.einsum(
                "...iw,wv,...jv->...ij", phi, self._w_cov, phi
            )

        if not joint_inputs and joint_outputs:
            d_out = prod(self._output_shape)
            phi_flat = phi.reshape(*extra_batch, n, d_out, -1)
            return jnp.einsum(
                "...ow,wv,...pv->...op",
                phi_flat, self._w_cov, phi_flat
            )

        # Full joint: (*eb, n*d_out, n*d_out)
        d_out = prod(self._output_shape)
        if self._output_shape:
            phi_flat = phi.reshape(*extra_batch, n * d_out, -1)
        else:
            phi_flat = phi
        return jnp.einsum(
            "...iw,wv,...jv->...ij", phi_flat, self._w_cov, phi_flat
        )

    # -- Function sampling (finite-dimensional) -----------------------------

    def _sample_one(self, key: PRNGKey) -> Callable[[ArrayLike], Array]:
        """Draw a single function realization via weight-space sampling.

        Returns a callable ``f(X) -> Array`` that evaluates the linear
        model at arbitrary inputs using a single weight draw.
        """
        w = self._weights._sample(key)  # (d_w,)
        bias = self._bias

        # Capture feature_map in closure for consistency across calls.
        feature_map = self._feature_map

        def f(X: ArrayLike) -> Array:
            X = jnp.asarray(X, dtype=jnp.float32)
            phi = feature_map(X)  # (*eb, n, [*out,] d_w)
            return bias + jnp.einsum("...w,w->...", phi, w)

        return f

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Callable[[ArrayLike], Array]:
        """Draw function realization(s) via weight-space sampling.

        When ``sample_shape`` is non-empty, returns a single callable
        that evaluates all draws at once, accepting ``X`` with shape
        ``(*extra_batch, n, *input_shape)`` and returning array of shape
        ``(*sample_shape, *extra_batch, n, *output_shape)``.
        """
        if sample_shape == ():
            return self._sample_one(key)
        n_samples = prod(sample_shape)
        w_samples = self._weights._sample(key, sample_shape=(n_samples,))  # (n, d_w)
        bias = self._bias
        feature_map = self._feature_map
        reshape_to = sample_shape

        def f(X: ArrayLike) -> Array:
            X = jnp.asarray(X, dtype=jnp.float32)
            phi = feature_map(X)  # (*eb, n_pts, [*out,] d_w)
            # w_samples: (n_samples, d_w), phi: (*eb, n_pts, [*out,] d_w)
            # result: (n_samples, *eb, n_pts, [*out])
            result = bias + jnp.einsum("tw,...w->t...", w_samples, phi)
            return result.reshape(*reshape_to, *phi.shape[:-1])

        return f


# ---------------------------------------------------------------------------
# Algebraic wrappers (private — constructed via operators on GRF)
# ---------------------------------------------------------------------------


class _LinearMapGRF(GaussianRandomFunction):
    r"""``h(x) = A @ g(x)`` where *g* is a :class:`GaussianRandomFunction`.

    Constructed via ``A @ grf``.  *g* must have 1-D ``output_shape``.
    """

    def __init__(self, base: GaussianRandomFunction, A: Array) -> None:
        if not isinstance(base, GaussianRandomFunction):
            raise TypeError(
                f"base must be a GaussianRandomFunction, "
                f"got {type(base).__name__}"
            )
        if len(base.output_shape) != 1:
            raise ValueError(
                f"A @ grf requires grf.output_shape to be 1-D (d,), "
                f"got {base.output_shape}"
            )
        if A.ndim != 2:
            raise ValueError(
                f"A must be 2-D (d_out, d_w), got shape {A.shape}"
            )
        d_out, d_w = A.shape
        if d_w != base.output_shape[0]:
            raise ValueError(
                f"A columns ({d_w}) must match "
                f"grf.output_shape[0] ({base.output_shape[0]})"
            )

        self._base = base
        self._A = A

        super().__init__(
            input_shape=base.input_shape,
            output_shape=(d_out,),
        )
        self.supports_joint_inputs = base.supports_joint_inputs
        self.supports_joint_outputs = True

    def predict_mean(self, X: Array) -> Array:
        g_mean = self._base.predict_mean(X)  # (*eb, n, d_w)
        return jnp.einsum("ow,...w->...o", self._A, g_mean)

    def predict_variance(self, X: Array) -> Array:
        return jnp.diagonal(
            self._output_covariance_per_point(X), axis1=-2, axis2=-1
        )

    def predict_covariance(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Array:
        if not joint_inputs and not joint_outputs:
            raise ValueError(
                "Use predict_variance for fully marginal predictions."
            )

        if not joint_inputs and joint_outputs:
            return self._output_covariance_per_point(X)

        extra_batch, n = self._parse_X(X)
        d_w = self._base.output_shape[0]
        d_out = self._A.shape[0]

        if joint_inputs and not joint_outputs:
            # Per-output cross-input covariance: (*eb, d_out, n, n).
            if self._base.supports_joint_outputs:
                # Use full cross-output covariance for correctness
                # when base outputs are correlated.
                g_full = self._base.predict_covariance(
                    X, joint_inputs=True, joint_outputs=True
                )  # (*eb, n*d_w, n*d_w)
                g_full = g_full.reshape(
                    *extra_batch, n, d_w, n, d_w
                )
                # A_{o,w} * g_{i,w,j,v} * A_{o,v} → result_{o,i,j}
                return jnp.einsum(
                    "ow,...iwjv,ov->...oij", self._A, g_full, self._A
                )
            else:
                # Base outputs independent — fast path.
                g_cov = self._base.predict_covariance(
                    X, joint_inputs=True, joint_outputs=False
                )  # (*eb, d_w, n, n)
                return jnp.einsum(
                    "ow,ov,...wij->...oij", self._A, self._A, g_cov
                )

        # joint_inputs=True, joint_outputs=True
        # Full joint: (*eb, n*d_out, n*d_out).
        if self._base.supports_joint_outputs:
            g_full = self._base.predict_covariance(
                X, joint_inputs=True, joint_outputs=True
            )  # (*eb, n*d_w, n*d_w)
            g_full = g_full.reshape(*extra_batch, n, d_w, n, d_w)
            # A_{o,w} * g_{i,w,j,v} * A_{p,v} → result_{i,o,j,p}
            result = jnp.einsum(
                "ow,...iwjv,pv->...iojp", self._A, g_full, self._A
            )
            return result.reshape(*extra_batch, n * d_out, n * d_out)
        else:
            g_cov = self._base.predict_covariance(
                X, joint_inputs=True, joint_outputs=False
            )  # (*eb, d_w, n, n)
            # A_{o,w} * A_{p,w} * g_{w,i,j} → result block (o,p) at (i,j)
            cov_block = jnp.einsum(
                "ow,pw,...wij->...opij", self._A, self._A, g_cov
            )
            ndim_eb = len(extra_batch)
            perm = [
                *range(ndim_eb),
                ndim_eb + 2, ndim_eb, ndim_eb + 3, ndim_eb + 1,
            ]
            cov_reordered = cov_block.transpose(perm)
            return cov_reordered.reshape(*extra_batch, n * d_out, n * d_out)

    def _output_covariance_per_point(self, X: Array) -> Array:
        """Per-point output covariance: ``A @ Cov_g @ A^T``.

        Returns shape ``(*extra_batch, n, d_out, d_out)``.
        """
        if self._base.supports_joint_outputs:
            g_cov = self._base.predict_covariance(
                X, joint_inputs=False, joint_outputs=True
            )  # (*eb, n, d_w, d_w)
            return jnp.einsum(
                "ow,...wv,pv->...op", self._A, g_cov, self._A
            )
        else:
            g_var = self._base.predict_variance(X)  # (*eb, n, d_w)
            return jnp.einsum(
                "ow,pw,...w->...op", self._A, self._A, g_var
            )


class _ShiftedGRF(GaussianRandomFunction):
    """``h(x) = g(x) + b`` — constant bias shift.

    Constructed via ``grf + b``.
    """

    def __init__(self, base: GaussianRandomFunction, b: Array) -> None:
        self._base = base
        self._b = b
        super().__init__(
            input_shape=base.input_shape,
            output_shape=base.output_shape,
        )
        self.supports_joint_inputs = base.supports_joint_inputs
        self.supports_joint_outputs = base.supports_joint_outputs

    def predict_mean(self, X: Array) -> Array:
        return self._base.predict_mean(X) + self._b

    def predict_variance(self, X: Array) -> Array:
        return self._base.predict_variance(X)

    def predict_covariance(
        self, X: Array, *, joint_inputs: bool = False, joint_outputs: bool = False,
    ) -> Array:
        return self._base.predict_covariance(
            X, joint_inputs=joint_inputs, joint_outputs=joint_outputs,
        )


class _ScaledGRF(GaussianRandomFunction):
    """``h(x) = alpha * g(x)`` — scalar scaling.

    Constructed via ``alpha * grf``.
    """

    def __init__(self, base: GaussianRandomFunction, alpha: Array) -> None:
        self._base = base
        self._alpha = alpha
        super().__init__(
            input_shape=base.input_shape,
            output_shape=base.output_shape,
        )
        self.supports_joint_inputs = base.supports_joint_inputs
        self.supports_joint_outputs = base.supports_joint_outputs

    def predict_mean(self, X: Array) -> Array:
        return self._alpha * self._base.predict_mean(X)

    def predict_variance(self, X: Array) -> Array:
        return self._alpha ** 2 * self._base.predict_variance(X)

    def predict_covariance(
        self, X: Array, *, joint_inputs: bool = False, joint_outputs: bool = False,
    ) -> Array:
        return self._alpha ** 2 * self._base.predict_covariance(
            X, joint_inputs=joint_inputs, joint_outputs=joint_outputs,
        )


class _IndependentSumGRF(GaussianRandomFunction):
    r"""``h(x) = g_1(x) + g_2(x)`` — sum of independent GRFs.

    Constructed via ``grf1 + grf2``.  The two GRFs are assumed to be
    **independent**; the resulting covariance is the sum of the
    individual covariances.

    A basic safety check rejects ``grf + grf`` (same object), but
    beyond that, independence is the caller's responsibility.
    """

    def __init__(
        self, left: GaussianRandomFunction, right: GaussianRandomFunction,
    ) -> None:
        if left is right:
            raise ValueError(
                "Cannot add a GaussianRandomFunction to itself — the "
                "result would not be independent.  Use 2 * grf instead."
            )
        if left.input_shape != right.input_shape:
            raise ValueError(
                f"input_shape mismatch: {left.input_shape} vs "
                f"{right.input_shape}"
            )
        if left.output_shape != right.output_shape:
            raise ValueError(
                f"output_shape mismatch: {left.output_shape} vs "
                f"{right.output_shape}"
            )
        self._left = left
        self._right = right
        super().__init__(
            input_shape=left.input_shape,
            output_shape=left.output_shape,
        )
        self.supports_joint_inputs = (
            left.supports_joint_inputs and right.supports_joint_inputs
        )
        self.supports_joint_outputs = (
            left.supports_joint_outputs and right.supports_joint_outputs
        )

    def predict_mean(self, X: Array) -> Array:
        return self._left.predict_mean(X) + self._right.predict_mean(X)

    def predict_variance(self, X: Array) -> Array:
        return (
            self._left.predict_variance(X) + self._right.predict_variance(X)
        )

    def predict_covariance(
        self, X: Array, *, joint_inputs: bool = False, joint_outputs: bool = False,
    ) -> Array:
        return (
            self._left.predict_covariance(
                X, joint_inputs=joint_inputs, joint_outputs=joint_outputs,
            )
            + self._right.predict_covariance(
                X, joint_inputs=joint_inputs, joint_outputs=joint_outputs,
            )
        )
