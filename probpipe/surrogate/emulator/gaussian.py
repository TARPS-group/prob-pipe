"""
Gaussian emulator classes for ProbPipe.

Provides:
  - ``GaussianEmulator``          - Abstract base for emulators with Gaussian
                                    predictive distributions.
  - ``LinCombGaussianWeights``    - Linear combination with Gaussian weight
                                    emulator: ``f(x) = a + Phi @ w(x)``.
  - ``LinearGaussianRegressor``   - Linear model with fixed Gaussian weights:
                                    ``f(x) = a + Phi(x) @ w``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import jax.numpy as jnp

from ...custom_types import Array, ArrayLike, PRNGKey
from ...distributions import MultivariateNormal, Normal
from .base import Emulator, _prod

# ---------------------------------------------------------------------------
# GaussianEmulator
# ---------------------------------------------------------------------------


class GaussianEmulator(Emulator):
    """Abstract emulator with Gaussian predictive distributions.

    Subclasses implement :meth:`predict_mean` and :meth:`predict_variance`
    at minimum.  If the emulator supports joint modes it must also implement
    :meth:`predict_covariance`.

    The base :meth:`predict` method assembles these into the appropriate
    :class:`~probpipe.Normal` or :class:`~probpipe.MultivariateNormal`
    distribution with the correct batch/event shape partition.

    This class is not restricted to GPs - any model that produces Gaussian
    (or Gaussian-approximated) predictions can use it.
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

        Required only if the emulator supports joint modes.  Returns the
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
              Full covariance over the flattened ``(n, *output_shape)`` event.

            - ``joint_inputs=True, joint_outputs=False``:
              ``(*extra_batch, *output_shape, n, n)``
              One ``n x n`` covariance matrix per output.

            - ``joint_inputs=False, joint_outputs=True``:
              ``(*extra_batch, n, prod(output_shape), prod(output_shape))``
              One ``d_out x d_out`` covariance matrix per input point.

            - ``joint_inputs=False, joint_outputs=False``:
              Not applicable - use :meth:`predict_variance` instead.

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
    ) -> Normal | MultivariateNormal:
        """Assemble a Gaussian distribution from mean / variance / covariance.

        The implementation follows this logic:

        1. Compute the mean: shape ``(*extra_batch, n, *output_shape)``.
        2. If neither joint flag is set, compute marginal variances and
           return a batch of independent :class:`~probpipe.Normal`
           distributions.
        3. Otherwise, compute the appropriate covariance matrix via
           :meth:`predict_covariance` and return a
           :class:`~probpipe.MultivariateNormal` (or block-structured
           distribution) with the correct event / batch partition.

        Subclasses may override this if they need non-standard assembly
        (e.g. structured covariance representations).
        """
        mean = self.predict_mean(X)  # (*eb, n, *out)

        # -- Fully marginal ---------------------------------------------------
        if not joint_inputs and not joint_outputs:
            variance = self.predict_variance(X)
            return Normal(loc=mean, scale=jnp.sqrt(variance))

        # -- At least one joint axis - need covariance ------------------------
        cov = self.predict_covariance(
            X, joint_inputs=joint_inputs, joint_outputs=joint_outputs
        )
        extra_batch, n = self._parse_X(X)
        d_out = _prod(self._output_shape)

        # Compute Cholesky factor for batched MVN construction.
        # probpipe's MultivariateNormal validates cov shape strictly,
        # so we pass scale_tril to support batched shapes.
        scale_tril = jnp.linalg.cholesky(cov)

        if joint_inputs and joint_outputs:
            # Full joint: flatten (n, *output_shape) into a single event dim.
            flat_dim = n * d_out if self._output_shape else n
            flat_mean = mean.reshape(*extra_batch, flat_dim)
            return MultivariateNormal(loc=flat_mean, scale_tril=scale_tril)

        if joint_inputs and not joint_outputs:
            # Joint over n, independent over outputs.
            # mean: (*eb, n, *out) -> need (*eb, *out, n)
            # cov:  (*eb, *out, n, n)
            if self._output_shape:
                ndim_eb = len(extra_batch)
                ndim_out = len(self._output_shape)
                source_axes = list(range(ndim_eb + 1, ndim_eb + 1 + ndim_out))
                dest_axes = list(range(ndim_eb, ndim_eb + ndim_out))
                mean_t = jnp.moveaxis(mean, source_axes, dest_axes)
            else:
                mean_t = mean  # (*eb, n) - nothing to rearrange
            return MultivariateNormal(loc=mean_t, scale_tril=scale_tril)

        # joint_outputs only (not joint_inputs)
        # mean: (*eb, n, *out) -> flatten output dims: (*eb, n, d_out)
        # cov:  (*eb, n, d_out, d_out)
        flat_mean = mean.reshape(*extra_batch, n, d_out)
        return MultivariateNormal(loc=flat_mean, scale_tril=scale_tril)


# ---------------------------------------------------------------------------
# LinCombGaussianWeights
# ---------------------------------------------------------------------------


class LinCombGaussianWeights(GaussianEmulator):
    r"""Emulator formed by linearly transforming a Gaussian weight emulator.

    Implements the model:

    .. math::

        f(x) = a + \Phi\, w(x)

    where :math:`w(x)` is itself a :class:`GaussianEmulator` mapping inputs
    to a Gaussian distribution over weight vectors, :math:`\Phi` is a fixed
    matrix mapping weight space to output space, and :math:`a` is an optional
    bias.

    Since linear transformations of Gaussians are Gaussian, this class
    transforms a :class:`GaussianEmulator` to produce a new
    :class:`GaussianEmulator`.

    Parameters
    ----------
    weight_emulator : GaussianEmulator
        Emulator mapping inputs to weight distributions.  Must have
        1-D ``output_shape``, i.e. ``output_shape = (d_w,)``.
    phi : array-like, shape ``(d_out, d_w)``
        Linear map from weight space to output space.
    bias : array-like, shape ``(d_out,)`` or broadcastable, optional
        Additive bias.  Defaults to zero.
    """

    def __init__(
        self,
        weight_emulator: GaussianEmulator,
        phi: ArrayLike,
        bias: ArrayLike | None = None,
    ) -> None:
        if not isinstance(weight_emulator, GaussianEmulator):
            raise TypeError(
                f"weight_emulator must be a GaussianEmulator, "
                f"got {type(weight_emulator).__name__}"
            )
        if len(weight_emulator.output_shape) != 1:
            raise ValueError(
                f"weight_emulator.output_shape must be 1-D (d_w,), "
                f"got {weight_emulator.output_shape}"
            )

        phi = jnp.asarray(phi, dtype=jnp.float32)
        if phi.ndim != 2:
            raise ValueError(
                f"phi must be 2-D (d_out, d_w), got shape {phi.shape}"
            )

        d_out, d_w = phi.shape
        if d_w != weight_emulator.output_shape[0]:
            raise ValueError(
                f"phi columns ({d_w}) must match "
                f"weight_emulator.output_shape[0] "
                f"({weight_emulator.output_shape[0]})"
            )

        self._weight_emulator = weight_emulator
        self._phi = phi
        self._bias = (
            jnp.asarray(bias, dtype=jnp.float32)
            if bias is not None
            else jnp.zeros(d_out, dtype=jnp.float32)
        )

        super().__init__(
            input_shape=weight_emulator.input_shape,
            output_shape=(d_out,),
        )

        # Inherit joint_inputs capability; always support joint_outputs
        # since Phi couples the outputs.
        self.supports_joint_inputs = weight_emulator.supports_joint_inputs
        self.supports_joint_outputs = True

    def predict_mean(self, X: Array) -> Array:
        """Predictive mean: ``a + Phi @ mean_w(x)``.

        Returns shape ``(*extra_batch, n, d_out)``.
        """
        w_mean = self._weight_emulator.predict_mean(X)  # (*eb, n, d_w)
        return self._bias + jnp.einsum("ow,...w->...o", self._phi, w_mean)

    def predict_variance(self, X: Array) -> Array:
        """Marginal predictive variance: ``diag(Phi @ Cov_w @ Phi^T)``.

        Returns shape ``(*extra_batch, n, d_out)``.
        """
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
            # Per-point cross-output covariance: (*eb, n, d_out, d_out)
            return self._output_covariance_per_point(X)

        if joint_inputs and not joint_outputs:
            # Per-output cross-input covariance: (*eb, d_out, n, n).
            # Weight cov is (*eb, d_w, n, n) — one nxn matrix per weight dim.
            # Output cov[o,i,j] = sum_w phi[o,w]^2 * w_cov[w,i,j].
            w_cov = self._weight_emulator.predict_covariance(
                X, joint_inputs=True, joint_outputs=False
            )
            return jnp.einsum(
                "ow,...wij->...oij",
                self._phi ** 2,
                w_cov,
            )

        # joint_inputs=True, joint_outputs=True
        # Full joint covariance: (*eb, n*d_out, n*d_out).
        extra_batch, n = self._parse_X(X)
        d_out = self._phi.shape[0]

        # Per-weight cross-input cov: (*eb, d_w, n, n).
        w_cov = self._weight_emulator.predict_covariance(
            X, joint_inputs=True, joint_outputs=False
        )
        # cov_block[o1,o2,i,j] = sum_w phi[o1,w]*phi[o2,w]*w_cov[w,i,j]
        cov_block = jnp.einsum(
            "ow,pw,...wij->...opij", self._phi, self._phi, w_cov
        )
        # Reorder from (*eb, d_out, d_out, n, n) to (*eb, n, d_out, n, d_out)
        # then flatten to (*eb, n*d_out, n*d_out).
        ndim_eb = len(extra_batch)
        perm = [*range(ndim_eb), ndim_eb + 2, ndim_eb, ndim_eb + 3, ndim_eb + 1]
        cov_reordered = cov_block.transpose(perm)
        return cov_reordered.reshape(*extra_batch, n * d_out, n * d_out)

    def _output_covariance_per_point(self, X: Array) -> Array:
        """Compute per-point output covariance: ``Phi @ Cov_w @ Phi^T``.

        Returns shape ``(*extra_batch, n, d_out, d_out)``.
        """
        if self._weight_emulator.supports_joint_outputs:
            # Full weight covariance per point: (*eb, n, d_w, d_w)
            w_cov = self._weight_emulator.predict_covariance(
                X, joint_inputs=False, joint_outputs=True
            )
            return jnp.einsum(
                "ow,...wv,pv->...op", self._phi, w_cov, self._phi
            )
        else:
            # Independent weights: use diagonal variance
            w_var = self._weight_emulator.predict_variance(X)  # (*eb, n, d_w)
            # Phi @ diag(var) @ Phi^T = sum_w phi[o,w]*phi[p,w]*var[w]
            return jnp.einsum(
                "ow,pw,...w->...op", self._phi, self._phi, w_var
            )


# ---------------------------------------------------------------------------
# LinearGaussianRegressor
# ---------------------------------------------------------------------------


class LinearGaussianRegressor(GaussianEmulator):
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

    This emulator always supports ``joint_inputs=True`` since the cross-input
    covariance :math:`\Phi(x_i)\, C\, \Phi(x_j)^T` is available analytically.

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
        weights: MultivariateNormal,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...] = (),
        bias: ArrayLike | None = None,
    ) -> None:
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

    def predict_mean(self, X: Array) -> Array:
        """Predictive mean: ``a + Phi(X) @ m``.

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
        # var = sum_{w,v} phi[..., w] * C[w, v] * phi[..., v]
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
                # Per-output cross-input cov: (*eb, *out, n, n)
                d_out = _prod(self._output_shape)
                phi_flat = phi.reshape(*extra_batch, n, d_out, -1)
                cov = jnp.einsum(
                    "...iow,wv,...jov->...oij",
                    phi_flat, self._w_cov, phi_flat
                )
                return cov.reshape(*extra_batch, *self._output_shape, n, n)
            # Scalar output: cross-input cov (*eb, n, n)
            return jnp.einsum(
                "...iw,wv,...jv->...ij", phi, self._w_cov, phi
            )

        if not joint_inputs and joint_outputs:
            # Per-point cross-output cov: (*eb, n, d_out, d_out)
            d_out = _prod(self._output_shape)
            phi_flat = phi.reshape(*extra_batch, n, d_out, -1)
            return jnp.einsum(
                "...ow,wv,...pv->...op",
                phi_flat, self._w_cov, phi_flat
            )

        # Full joint cov: (*eb, n*d_out, n*d_out)
        d_out = _prod(self._output_shape) if self._output_shape else 1
        if self._output_shape:
            phi_flat = phi.reshape(*extra_batch, n * d_out, -1)
        else:
            phi_flat = phi
        return jnp.einsum(
            "...iw,wv,...jv->...ij", phi_flat, self._w_cov, phi_flat
        )

    def sample_trajectory(
        self,
        key: PRNGKey,
        n_trajectories: int,
    ) -> Callable[[ArrayLike], Array]:
        """Draw function realizations via weight-space sampling.

        Draws ``n_trajectories`` weight vectors from the weight distribution
        and returns a callable that evaluates
        ``f_t(X) = bias + Phi(X) @ w_t`` for each trajectory.

        The returned callable is consistent across calls: evaluating it at
        different input locations uses the same weight draws, producing
        values from the same underlying function realization.

        Parameters
        ----------
        key : PRNGKey
            JAX PRNG key for sampling.
        n_trajectories : int
            Number of trajectories to draw.

        Returns
        -------
        g : callable
            Accepts ``X`` with shape ``(*extra_batch, n, *input_shape)``
            and returns array of shape
            ``(n_trajectories, *extra_batch, n, *output_shape)``.
        """
        # Draw weight samples: (n_trajectories, d_w)
        w_samples = self._weights.sample(
            key, sample_shape=(n_trajectories,)
        )

        def g(X: ArrayLike) -> Array:
            X = jnp.asarray(X, dtype=jnp.float32)
            phi = self._feature_map(X)  # (*eb, n, [*out,] d_w)
            # f = bias + phi @ w for each trajectory
            # w_samples: (t, d_w), phi: (*eb, n, [*out,] d_w)
            # result: (t, *eb, n, *out)
            return self._bias + jnp.einsum("tw,...w->t...", w_samples, phi)

        return g
