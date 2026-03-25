"""
Gaussian random function classes for ProbPipe.

Provides:
  - ``GaussianRandomFunction``    – Abstract base for random functions with
                                    Gaussian predictive distributions.
  - ``LinearBasisFunction``       – f(x) = a + Φ(x) @ w, w ~ N(m, C).
  - ``LinearOutputTransform``     – f(x) = a + Φ @ g(x), where g is a
                                    GaussianRandomFunction.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions.distribution import _auto_key
from .random_function import ArrayRandomFunction, _prod

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
        d_out = _prod(self._output_shape)

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
                d_out = _prod(self._output_shape)
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
            d_out = _prod(self._output_shape)
            phi_flat = phi.reshape(*extra_batch, n, d_out, -1)
            return jnp.einsum(
                "...ow,wv,...pv->...op",
                phi_flat, self._w_cov, phi_flat
            )

        # Full joint: (*eb, n*d_out, n*d_out)
        d_out = _prod(self._output_shape) if self._output_shape else 1
        if self._output_shape:
            phi_flat = phi.reshape(*extra_batch, n * d_out, -1)
        else:
            phi_flat = phi
        return jnp.einsum(
            "...iw,wv,...jv->...ij", phi_flat, self._w_cov, phi_flat
        )

    # -- Function sampling (finite-dimensional) -----------------------------

    def _sample(self, key: PRNGKey) -> Callable[[ArrayLike], Array]:
        """Draw a single function realization via weight-space sampling.

        Returns a callable ``f(X) -> Array`` that evaluates the linear
        model at arbitrary inputs using a single weight draw.
        """
        w = self._weights.sample(key)  # (d_w,)
        bias = self._bias

        # Capture feature_map in closure for consistency across calls.
        feature_map = self._feature_map

        def f(X: ArrayLike) -> Array:
            X = jnp.asarray(X, dtype=jnp.float32)
            phi = feature_map(X)  # (*eb, n, [*out,] d_w)
            return bias + jnp.einsum("...w,w->...", phi, w)

        return f

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Callable[[ArrayLike], Array]:
        """Draw function realization(s) via weight-space sampling.

        Parameters
        ----------
        key : PRNGKey, optional
            JAX PRNG key.
        sample_shape : tuple of int
            Number of independent function draws.

        Returns
        -------
        callable
            Accepts ``X`` with shape ``(*extra_batch, n, *input_shape)``
            and returns array of shape
            ``(*sample_shape, *extra_batch, n, *output_shape)``.
        """
        if key is None:
            key = _auto_key()
        if sample_shape == ():
            return self._sample(key)

        n_samples = math.prod(sample_shape)
        w_samples = self._weights.sample(key, sample_shape=(n_samples,))  # (n, d_w)
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
# LinearOutputTransform
# ---------------------------------------------------------------------------


class LinearOutputTransform(GaussianRandomFunction):
    r"""Gaussian random function formed by linearly transforming another.

    Implements the model:

    .. math::

        f(x) = a + \Phi\, g(x)

    where :math:`g(x)` is itself a :class:`GaussianRandomFunction` mapping
    inputs to a Gaussian distribution over weight vectors, :math:`\Phi` is
    a fixed matrix mapping from :math:`g`'s output space to this model's
    output space, and :math:`a` is an optional bias.

    Since linear transformations of Gaussians are Gaussian, this produces
    a new :class:`GaussianRandomFunction`.

    Parameters
    ----------
    base_function : GaussianRandomFunction
        The underlying random function.  Must have 1-D ``output_shape``,
        i.e. ``output_shape = (d_w,)``.
    phi : array-like, shape ``(d_out, d_w)``
        Linear map from base output space to this model's output space.
    bias : array-like, shape ``(d_out,)`` or broadcastable, optional
        Additive bias.  Defaults to zero.
    """

    def __init__(
        self,
        base_function: GaussianRandomFunction,
        phi: ArrayLike,
        bias: ArrayLike | None = None,
    ) -> None:
        if not isinstance(base_function, GaussianRandomFunction):
            raise TypeError(
                f"base_function must be a GaussianRandomFunction, "
                f"got {type(base_function).__name__}"
            )
        if len(base_function.output_shape) != 1:
            raise ValueError(
                f"base_function.output_shape must be 1-D (d_w,), "
                f"got {base_function.output_shape}"
            )

        phi = jnp.asarray(phi, dtype=jnp.float32)
        if phi.ndim != 2:
            raise ValueError(
                f"phi must be 2-D (d_out, d_w), got shape {phi.shape}"
            )

        d_out, d_w = phi.shape
        if d_w != base_function.output_shape[0]:
            raise ValueError(
                f"phi columns ({d_w}) must match "
                f"base_function.output_shape[0] "
                f"({base_function.output_shape[0]})"
            )

        self._base_function = base_function
        self._phi = phi
        self._bias = (
            jnp.asarray(bias, dtype=jnp.float32)
            if bias is not None
            else jnp.zeros(d_out, dtype=jnp.float32)
        )

        super().__init__(
            input_shape=base_function.input_shape,
            output_shape=(d_out,),
        )

        # Inherit joint_inputs capability; always support joint_outputs
        # since Phi couples the outputs.
        self.supports_joint_inputs = base_function.supports_joint_inputs
        self.supports_joint_outputs = True

    # -- GaussianRandomFunction interface -----------------------------------

    def predict_mean(self, X: Array) -> Array:
        r"""Predictive mean: ``a + Phi @ mean_g(x)``."""
        g_mean = self._base_function.predict_mean(X)  # (*eb, n, d_w)
        return self._bias + jnp.einsum("ow,...w->...o", self._phi, g_mean)

    def predict_variance(self, X: Array) -> Array:
        r"""Marginal predictive variance: ``diag(Phi @ Cov_g @ Phi^T)``."""
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

        if joint_inputs and not joint_outputs:
            # Per-output cross-input covariance: (*eb, d_out, n, n).
            g_cov = self._base_function.predict_covariance(
                X, joint_inputs=True, joint_outputs=False
            )
            return jnp.einsum(
                "ow,...wij->...oij",
                self._phi ** 2,
                g_cov,
            )

        # joint_inputs=True, joint_outputs=True
        # Full joint covariance: (*eb, n*d_out, n*d_out).
        extra_batch, n = self._parse_X(X)
        d_out = self._phi.shape[0]

        g_cov = self._base_function.predict_covariance(
            X, joint_inputs=True, joint_outputs=False
        )
        cov_block = jnp.einsum(
            "ow,pw,...wij->...opij", self._phi, self._phi, g_cov
        )
        ndim_eb = len(extra_batch)
        perm = [*range(ndim_eb), ndim_eb + 2, ndim_eb, ndim_eb + 3, ndim_eb + 1]
        cov_reordered = cov_block.transpose(perm)
        return cov_reordered.reshape(*extra_batch, n * d_out, n * d_out)

    def _output_covariance_per_point(self, X: Array) -> Array:
        """Compute per-point output covariance: ``Phi @ Cov_g @ Phi^T``.

        Returns shape ``(*extra_batch, n, d_out, d_out)``.
        """
        if self._base_function.supports_joint_outputs:
            g_cov = self._base_function.predict_covariance(
                X, joint_inputs=False, joint_outputs=True
            )
            return jnp.einsum(
                "ow,...wv,pv->...op", self._phi, g_cov, self._phi
            )
        else:
            g_var = self._base_function.predict_variance(X)  # (*eb, n, d_w)
            return jnp.einsum(
                "ow,pw,...w->...op", self._phi, self._phi, g_var
            )
