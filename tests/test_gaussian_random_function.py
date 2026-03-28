"""Tests for GaussianRandomFunction, LinearBasisFunction, and GRF algebra."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    Normal,
    MultivariateNormal,
    RandomFunction,
    ArrayRandomFunction,
    GaussianRandomFunction,
    LinearBasisFunction,
)
from probpipe import sample


# ---------------------------------------------------------------------------
# Dense ground-truth helpers
# ---------------------------------------------------------------------------
#
# For a LinearBasisFunction with feature map Φ(x) and weights w ~ N(m, C):
#   f(x) = bias + Φ(x) @ w
#
# Scalar output:
#   Φ(x): (n, d_w)
#   mean(x) = bias + Φ(x) @ m                            shape: (n,)
#   Cov(x, x') = Φ(x) @ C @ Φ(x')^T                     shape: (n, n)
#   Var(x) = diag(Cov(x, x))                              shape: (n,)
#
# Multi-output (d_out outputs):
#   Φ(x): (n, d_out, d_w)
#   mean(x)_o = bias_o + Φ(x)_{o,:} @ m                  shape: (n, d_out)
#   Full joint cov_{io,jp} = Φ(x_i)_{o,:} @ C @ Φ(x_j)_{p,:}^T
#     → shape: (n*d_out, n*d_out) when flattened
#
# After linear map h(x) = A @ g(x):
#   Φ_h(x)_{o,:} = Σ_w A_{o,w} * Φ_g(x)_{w,:}
#   This is a (n, d_out_h, d_w) feature map, and the same formulas apply.


def _dense_lbf_full_joint_cov(phi_X, w_cov):
    """Compute dense full-joint covariance for a LinearBasisFunction.

    Parameters
    ----------
    phi_X : ndarray, shape (n, d_out, d_w) or (n, d_w) for scalar output
        Feature map evaluations at n input points.
    w_cov : ndarray, shape (d_w, d_w)
        Weight covariance.

    Returns
    -------
    ndarray, shape (n*d_out, n*d_out) or (n, n) for scalar output
        Full joint covariance matrix.
    """
    if phi_X.ndim == 2:
        # Scalar output: (n, d_w)
        return phi_X @ w_cov @ phi_X.T
    # Multi-output: (n, d_out, d_w) → flatten to (n*d_out, d_w)
    n, d_out, d_w = phi_X.shape
    phi_flat = phi_X.reshape(n * d_out, d_w)
    return phi_flat @ w_cov @ phi_flat.T


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Mock GaussianRandomFunction subclasses
# ---------------------------------------------------------------------------


def _rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """Simple RBF kernel for mock GP."""
    sq_dist = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale ** 2)


class _ScalarGP(GaussianRandomFunction):
    """Mock scalar-output GP with RBF kernel."""

    supports_joint_inputs = True

    def __init__(self, lengthscale=1.0, variance=1.0, noise=0.01):
        super().__init__(input_shape=(2,), output_shape=())
        self._ls = lengthscale
        self._var = variance
        self._noise = noise

    def predict_mean(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.zeros((*extra_batch, n))

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.full((*extra_batch, n), self._var)

    def predict_covariance(self, X, *, joint_inputs=False, joint_outputs=False):
        if not joint_inputs:
            raise NotImplementedError
        extra_batch, n = self._parse_X(X)
        if extra_batch:
            flat_X = X.reshape(-1, n, *self.input_shape)
            covs = []
            for i in range(flat_X.shape[0]):
                K = _rbf_kernel(flat_X[i], flat_X[i], self._ls, self._var)
                K = K + self._noise * jnp.eye(n)
                covs.append(K)
            return jnp.stack(covs).reshape(*extra_batch, n, n)
        else:
            K = _rbf_kernel(X, X, self._ls, self._var)
            return K + self._noise * jnp.eye(n)


class _MultiOutputGRF(GaussianRandomFunction):
    """Mock 2-output GRF with joint inputs and outputs."""

    supports_joint_inputs = True
    supports_joint_outputs = True

    def __init__(self):
        super().__init__(input_shape=(2,), output_shape=(2,))

    def predict_mean(self, X):
        s = jnp.sum(X, axis=-1)  # (*eb, n)
        return jnp.stack([s, 2 * s], axis=-1)  # (*eb, n, 2)

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.ones((*extra_batch, n, 2))

    def predict_covariance(self, X, *, joint_inputs=False, joint_outputs=False):
        extra_batch, n = self._parse_X(X)

        if joint_inputs and joint_outputs:
            full = jnp.kron(jnp.eye(n), jnp.eye(2))
            return jnp.broadcast_to(full, (*extra_batch, 2 * n, 2 * n))

        if joint_inputs and not joint_outputs:
            K = jnp.eye(n)
            return jnp.broadcast_to(
                jnp.stack([K, K]),
                (*extra_batch, 2, n, n),
            )

        if not joint_inputs and joint_outputs:
            C = jnp.eye(2)
            return jnp.broadcast_to(C, (*extra_batch, n, 2, 2))

        raise ValueError("Use predict_variance for fully marginal.")


class _MarginalOnlyGRF(GaussianRandomFunction):
    """GRF that only supports marginal predictions."""

    def __init__(self):
        super().__init__(input_shape=(2,), output_shape=())

    def predict_mean(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.zeros((*extra_batch, n))

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.ones((*extra_batch, n))


# ---------------------------------------------------------------------------
# GaussianRandomFunction tests
# ---------------------------------------------------------------------------


class TestGaussianRandomFunction:
    """Tests for the GaussianRandomFunction base class."""

    def test_isinstance_hierarchy(self):
        grf = _ScalarGP()
        assert isinstance(grf, GaussianRandomFunction)
        assert isinstance(grf, ArrayRandomFunction)
        assert isinstance(grf, RandomFunction)
        assert isinstance(grf, Distribution)

    def test_marginal_returns_normal(self):
        grf = _ScalarGP()
        X = jnp.ones((5, 2))
        dist = grf(X)
        assert isinstance(dist, Normal)
        assert dist.batch_shape == (5,)
        assert dist.event_shape == ()

    def test_marginal_multi_output(self):
        grf = _MultiOutputGRF()
        X = jnp.ones((5, 2))
        dist = grf(X)
        assert isinstance(dist, Normal)
        assert dist.batch_shape == (5, 2)
        assert dist.event_shape == ()

    def test_joint_inputs_returns_mvn(self):
        grf = _ScalarGP()
        X = jnp.ones((5, 2))
        dist = grf(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (5,)

    def test_joint_outputs_returns_mvn(self):
        grf = _MultiOutputGRF()
        X = jnp.ones((5, 2))
        dist = grf(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (2,)
        assert dist.batch_shape == (5,)

    def test_full_joint_returns_mvn(self):
        grf = _MultiOutputGRF()
        X = jnp.ones((5, 2))
        dist = grf(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)  # 5 * 2

    def test_marginal_only_rejects_joint(self):
        grf = _MarginalOnlyGRF()
        X = jnp.ones((5, 2))
        with pytest.raises(ValueError, match="joint_inputs"):
            grf(X, joint_inputs=True)

    def test_extra_batch_marginal(self):
        grf = _ScalarGP()
        X = jnp.ones((3, 5, 2))  # extra_batch=(3,)
        dist = grf(X)
        assert dist.batch_shape == (3, 5)

    def test_predict_covariance_raises_by_default(self):
        grf = _MarginalOnlyGRF()
        X = jnp.ones((5, 2))
        with pytest.raises(NotImplementedError):
            grf.predict_covariance(X, joint_inputs=True)


# ---------------------------------------------------------------------------
# LinearBasisFunction tests
# ---------------------------------------------------------------------------


def _polynomial_feature_map(X):
    """Simple polynomial features: [1, x, x^2] for scalar input."""
    return jnp.stack([jnp.ones_like(X[..., 0]), X[..., 0], X[..., 0] ** 2], axis=-1)


def _multi_output_feature_map(X):
    """Feature map for 2-output model: (*eb, n, 2, 2)."""
    x = X[..., 0]  # (*eb, n)
    phi = jnp.stack([jnp.ones_like(x), x], axis=-1)  # (*eb, n, 2)
    return jnp.stack([phi, 0.5 * phi], axis=-2)  # (*eb, n, 2, 2)


@pytest.fixture
def scalar_lbf():
    """Scalar LinearBasisFunction with polynomial features."""
    weights = MultivariateNormal(
        loc=jnp.array([1.0, 0.5, 0.1]),
        cov=0.01 * jnp.eye(3),
    )
    return LinearBasisFunction(
        feature_map=_polynomial_feature_map,
        weights=weights,
        input_shape=(1,),
    )


@pytest.fixture
def multi_output_lbf():
    """Multi-output LinearBasisFunction."""
    weights = MultivariateNormal(
        loc=jnp.array([1.0, 0.5]),
        cov=0.01 * jnp.eye(2),
    )
    return LinearBasisFunction(
        feature_map=_multi_output_feature_map,
        weights=weights,
        input_shape=(1,),
        output_shape=(2,),
    )


class TestLinearBasisFunction:
    """Tests for the LinearBasisFunction class."""

    def test_isinstance_hierarchy(self, scalar_lbf):
        assert isinstance(scalar_lbf, LinearBasisFunction)
        assert isinstance(scalar_lbf, GaussianRandomFunction)
        assert isinstance(scalar_lbf, ArrayRandomFunction)
        assert isinstance(scalar_lbf, RandomFunction)

    def test_shapes(self, scalar_lbf):
        assert scalar_lbf.input_shape == (1,)
        assert scalar_lbf.output_shape == ()

    def test_supports_joint_inputs(self, scalar_lbf):
        assert scalar_lbf.supports_joint_inputs is True

    def test_multi_output_supports_joint_outputs(self, multi_output_lbf):
        assert multi_output_lbf.supports_joint_outputs is True

    # -- Marginal prediction ------------------------------------------------

    def test_marginal_scalar(self, scalar_lbf):
        X = jnp.linspace(-1, 1, 10).reshape(-1, 1)
        dist = scalar_lbf(X)
        assert isinstance(dist, Normal)
        assert dist.batch_shape == (10,)
        assert dist.event_shape == ()

    def test_marginal_multi_output(self, multi_output_lbf):
        X = jnp.linspace(-1, 1, 8).reshape(-1, 1)
        dist = multi_output_lbf(X)
        assert isinstance(dist, Normal)
        assert dist.batch_shape == (8, 2)

    def test_marginal_mean_value(self, scalar_lbf):
        """Check mean matches manual computation."""
        X = jnp.array([[0.0], [1.0]])
        dist = scalar_lbf(X)
        mean = dist.loc
        # At x=0: phi=[1,0,0], mean = [1,0.5,0.1]@[1,0,0] = 1.0
        # At x=1: phi=[1,1,1], mean = [1,0.5,0.1]@[1,1,1] = 1.6
        np.testing.assert_allclose(mean, [1.0, 1.6], atol=1e-5)

    # -- Joint predictions --------------------------------------------------

    def test_joint_inputs_scalar(self, scalar_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = scalar_lbf(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (5,)

    def test_joint_outputs_multi(self, multi_output_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = multi_output_lbf(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (2,)
        assert dist.batch_shape == (5,)

    def test_full_joint_multi(self, multi_output_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = multi_output_lbf(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)  # 5 * 2

    # -- Extra batch --------------------------------------------------------

    def test_extra_batch_marginal(self, scalar_lbf):
        X = jnp.ones((3, 5, 1))
        dist = scalar_lbf(X)
        assert dist.batch_shape == (3, 5)

    def test_extra_batch_joint_inputs(self, scalar_lbf):
        X = jnp.ones((3, 5, 1))
        dist = scalar_lbf(X, joint_inputs=True)
        assert dist.event_shape == (5,)
        assert dist.batch_shape == (3,)

    # -- Covariance shapes --------------------------------------------------

    def test_covariance_joint_inputs_scalar(self, scalar_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        cov = scalar_lbf.predict_covariance(X, joint_inputs=True)
        assert cov.shape == (5, 5)

    def test_covariance_joint_outputs_multi(self, multi_output_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        cov = multi_output_lbf.predict_covariance(X, joint_outputs=True)
        assert cov.shape == (5, 2, 2)

    def test_covariance_full_joint_multi(self, multi_output_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        cov = multi_output_lbf.predict_covariance(
            X, joint_inputs=True, joint_outputs=True
        )
        assert cov.shape == (10, 10)  # 5*2 x 5*2

    # -- Function sampling --------------------------------------------------

    def test_sample_single(self, key, scalar_lbf):
        f = sample(scalar_lbf, key=key)
        assert callable(f)
        X = jnp.linspace(-1, 1, 10).reshape(-1, 1)
        y = f(X)
        assert y.shape == (10,)

    def test_sample_batched(self, key, scalar_lbf):
        f = sample(scalar_lbf, key=key, sample_shape=(7,))
        assert callable(f)
        X = jnp.linspace(-1, 1, 10).reshape(-1, 1)
        y = f(X)
        assert y.shape == (7, 10)

    def test_sample_multi_dim_shape(self, key, scalar_lbf):
        f = sample(scalar_lbf, key=key, sample_shape=(3, 4))
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        y = f(X)
        assert y.shape == (3, 4, 5)

    def test_sample_consistency(self, key, scalar_lbf):
        """Same function realization evaluates consistently."""
        f = sample(scalar_lbf, key=key)
        X1 = jnp.array([[0.0], [1.0]])
        X2 = jnp.array([[0.0], [2.0]])
        y1 = f(X1)
        y2 = f(X2)
        # f(0) should be the same in both calls
        np.testing.assert_allclose(y1[0], y2[0], atol=1e-6)

    def test_sample_batched_consistency(self, key, scalar_lbf):
        """Batched realizations: each trajectory is consistent."""
        f = sample(scalar_lbf, key=key, sample_shape=(5,))
        X1 = jnp.array([[0.0]])
        X2 = jnp.array([[0.0]])
        y1 = f(X1)  # (5, 1)
        y2 = f(X2)  # (5, 1)
        np.testing.assert_allclose(y1, y2, atol=1e-6)

    def test_sample_multi_output(self, key, multi_output_lbf):
        f = sample(multi_output_lbf, key=key)
        X = jnp.linspace(-1, 1, 8).reshape(-1, 1)
        y = f(X)
        assert y.shape == (8, 2)

    def test_sample_batched_multi_output(self, key, multi_output_lbf):
        f = sample(multi_output_lbf, key=key, sample_shape=(5,))
        X = jnp.linspace(-1, 1, 8).reshape(-1, 1)
        y = f(X)
        assert y.shape == (5, 8, 2)

    # -- Validation ---------------------------------------------------------

    def test_invalid_weights_type(self):
        with pytest.raises(TypeError, match="MultivariateNormal"):
            LinearBasisFunction(
                feature_map=_polynomial_feature_map,
                weights="not_a_distribution",
                input_shape=(1,),
            )


# ---------------------------------------------------------------------------
# GRF algebra tests
# ---------------------------------------------------------------------------


def _weight_feature_map(X):
    """Feature map for 3-output model with 2 basis functions: (*eb, n, 3, 2)."""
    x = X[..., 0]  # (*eb, n)
    ones = jnp.ones_like(x)
    features = jnp.stack([ones, x], axis=-1)  # (*eb, n, 2)
    return jnp.stack([features, 0.5 * features, 2.0 * features], axis=-2)  # (*eb, n, 3, 2)


@pytest.fixture
def weight_grf():
    """A GaussianRandomFunction over a 3-D weight space (output_shape=(3,))."""
    weights = MultivariateNormal(
        loc=jnp.array([1.0, 0.5]),
        cov=0.01 * jnp.eye(2),
    )
    return LinearBasisFunction(
        feature_map=_weight_feature_map,
        weights=weights,
        input_shape=(1,),
        output_shape=(3,),
    )


class TestLinearMap:
    """Tests for ``A @ grf``."""

    def test_isinstance_hierarchy(self, weight_grf):
        h = jnp.eye(3, 3) @ weight_grf
        assert isinstance(h, GaussianRandomFunction)
        assert isinstance(h, ArrayRandomFunction)
        assert isinstance(h, RandomFunction)

    def test_shapes(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # (2, 3)
        h = A @ weight_grf
        assert h.input_shape == (1,)
        assert h.output_shape == (2,)

    def test_supports_joint_outputs(self, weight_grf):
        A = jnp.ones((2, 3))
        h = A @ weight_grf
        assert h.supports_joint_outputs is True

    def test_inherits_joint_inputs(self, weight_grf):
        A = jnp.ones((2, 3))
        h = A @ weight_grf
        assert h.supports_joint_inputs == weight_grf.supports_joint_inputs

    def test_marginal(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = h(X)
        assert isinstance(dist, Normal)
        assert dist.batch_shape == (5, 2)

    def test_joint_outputs(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = h(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (2,)
        assert dist.batch_shape == (5,)

    def test_joint_inputs(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = h(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.batch_shape == (2,)
        assert dist.event_shape == (5,)

    def test_full_joint(self, weight_grf):
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = h(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)  # 5 * 2

    def test_mean_value(self, weight_grf):
        """Transformed mean = A @ base_mean."""
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.array([[0.5]])
        base_mean = weight_grf.predict_mean(X)  # (1, 3)
        expected = jnp.einsum("ow,...w->...o", A, base_mean)
        np.testing.assert_allclose(h.predict_mean(X), expected, atol=1e-5)

    def test_covariance_with_correlated_outputs(self, weight_grf):
        """joint_inputs covariance should use full cross-output correlations."""
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        h = A @ weight_grf
        X = jnp.linspace(-1, 1, 3).reshape(-1, 1)

        # Get via per-output path
        cov_per_out = h.predict_covariance(X, joint_inputs=True)
        assert cov_per_out.shape == (2, 3, 3)

        # Get via full joint and extract
        cov_full = h.predict_covariance(
            X, joint_inputs=True, joint_outputs=True
        )
        assert cov_full.shape == (6, 6)

        # The per-output covariance for output 0 should match the
        # corresponding block of the full joint.
        # Full joint layout: (x0_o0, x0_o1, x1_o0, x1_o1, x2_o0, x2_o1)
        # Output 0 indices: [0, 2, 4]
        idx = jnp.array([0, 2, 4])
        cov_o0_from_full = cov_full[jnp.ix_(idx, idx)]
        np.testing.assert_allclose(cov_per_out[0], cov_o0_from_full, atol=1e-5)

    def test_rejects_scalar_output(self, scalar_lbf):
        """A @ grf requires 1-D output_shape."""
        with pytest.raises(ValueError, match="1-D"):
            jnp.eye(2) @ scalar_lbf

    def test_rejects_dim_mismatch(self, weight_grf):
        """A columns must match output dim."""
        with pytest.raises(ValueError, match="columns"):
            jnp.eye(2) @ weight_grf  # (2,2) but output is (3,)


class TestShift:
    """Tests for ``grf + b``."""

    def test_mean_shifted(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        b = jnp.float32(10.0)
        h = scalar_lbf + b
        np.testing.assert_allclose(
            h.predict_mean(X),
            scalar_lbf.predict_mean(X) + b,
            atol=1e-5,
        )

    def test_variance_unchanged(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = scalar_lbf + 10.0
        np.testing.assert_allclose(
            h.predict_variance(X),
            scalar_lbf.predict_variance(X),
            atol=1e-6,
        )

    def test_covariance_unchanged(self, scalar_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        h = scalar_lbf + 10.0
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True),
            scalar_lbf.predict_covariance(X, joint_inputs=True),
            atol=1e-6,
        )

    def test_shapes_preserved(self, scalar_lbf):
        h = scalar_lbf + 1.0
        assert h.input_shape == scalar_lbf.input_shape
        assert h.output_shape == scalar_lbf.output_shape

    def test_radd(self, scalar_lbf):
        """``b + grf`` works via __radd__."""
        X = jnp.array([[0.0]])
        h = 5.0 + scalar_lbf
        np.testing.assert_allclose(
            h.predict_mean(X),
            scalar_lbf.predict_mean(X) + 5.0,
            atol=1e-5,
        )

    def test_sub(self, scalar_lbf):
        """``grf - b`` is equivalent to ``grf + (-b)``."""
        X = jnp.array([[0.0], [1.0]])
        h = scalar_lbf - 3.0
        np.testing.assert_allclose(
            h.predict_mean(X),
            scalar_lbf.predict_mean(X) - 3.0,
            atol=1e-5,
        )


class TestScale:
    """Tests for ``alpha * grf``."""

    def test_mean_scaled(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = 3.0 * scalar_lbf
        np.testing.assert_allclose(
            h.predict_mean(X),
            3.0 * scalar_lbf.predict_mean(X),
            atol=1e-5,
        )

    def test_variance_scaled_squared(self, scalar_lbf):
        X = jnp.array([[0.0], [1.0]])
        h = 3.0 * scalar_lbf
        np.testing.assert_allclose(
            h.predict_variance(X),
            9.0 * scalar_lbf.predict_variance(X),
            atol=1e-5,
        )

    def test_covariance_scaled_squared(self, scalar_lbf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        h = 2.0 * scalar_lbf
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True),
            4.0 * scalar_lbf.predict_covariance(X, joint_inputs=True),
            atol=1e-5,
        )

    def test_rmul(self, scalar_lbf):
        """``grf * alpha`` works via __mul__."""
        X = jnp.array([[0.0]])
        h = scalar_lbf * 2.0
        np.testing.assert_allclose(
            h.predict_mean(X),
            2.0 * scalar_lbf.predict_mean(X),
            atol=1e-5,
        )

    def test_neg(self, scalar_lbf):
        """``-grf`` negates the mean, preserves variance."""
        X = jnp.array([[0.0], [1.0]])
        h = -scalar_lbf
        np.testing.assert_allclose(
            h.predict_mean(X),
            -scalar_lbf.predict_mean(X),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            h.predict_variance(X),
            scalar_lbf.predict_variance(X),
            atol=1e-6,
        )


class TestIndependentSum:
    """Tests for ``grf1 + grf2``."""

    def test_mean_is_sum(self):
        gp1 = _ScalarGP(lengthscale=1.0, variance=1.0)
        gp2 = _ScalarGP(lengthscale=0.5, variance=0.5)
        h = gp1 + gp2
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_mean(X),
            gp1.predict_mean(X) + gp2.predict_mean(X),
            atol=1e-6,
        )

    def test_variance_is_sum(self):
        gp1 = _ScalarGP(lengthscale=1.0, variance=1.0)
        gp2 = _ScalarGP(lengthscale=0.5, variance=0.5)
        h = gp1 + gp2
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_variance(X),
            gp1.predict_variance(X) + gp2.predict_variance(X),
            atol=1e-6,
        )

    def test_covariance_is_sum(self):
        gp1 = _ScalarGP(lengthscale=1.0, variance=1.0)
        gp2 = _ScalarGP(lengthscale=0.5, variance=0.5)
        h = gp1 + gp2
        X = jnp.stack([jnp.linspace(-1, 1, 5), jnp.zeros(5)], axis=-1)  # (5, 2)
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True),
            (gp1.predict_covariance(X, joint_inputs=True)
             + gp2.predict_covariance(X, joint_inputs=True)),
            atol=1e-5,
        )

    def test_joint_support_intersection(self):
        """Sum supports joint only if both operands do."""
        gp = _ScalarGP()  # supports_joint_inputs=True
        marginal = _MarginalOnlyGRF()  # supports_joint_inputs=False
        h = gp + marginal
        assert h.supports_joint_inputs is False

    def test_same_object_raises(self):
        gp = _ScalarGP()
        with pytest.raises(ValueError, match="itself"):
            gp + gp

    def test_shape_mismatch_raises(self):
        gp = _ScalarGP()  # output_shape=()
        multi = _MultiOutputGRF()  # output_shape=(2,)
        with pytest.raises(ValueError, match="output_shape"):
            gp + multi

    def test_sub_grfs(self):
        """``grf1 - grf2`` = grf1 + (-grf2)."""
        gp1 = _ScalarGP(lengthscale=1.0, variance=1.0)
        gp2 = _ScalarGP(lengthscale=0.5, variance=0.5)
        h = gp1 - gp2
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_mean(X),
            gp1.predict_mean(X) - gp2.predict_mean(X),
            atol=1e-6,
        )


class TestAlgebraComposition:
    """Tests for composing multiple algebraic operations."""

    def test_affine_transform(self, weight_grf):
        """``A @ grf + b`` composes linear map and shift."""
        A = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        b = jnp.array([0.1, -0.2])
        h = A @ weight_grf + b

        X = jnp.array([[0.5]])
        base_mean = weight_grf.predict_mean(X)
        expected = jnp.einsum("ow,...w->...o", A, base_mean) + b
        np.testing.assert_allclose(h.predict_mean(X), expected, atol=1e-5)

    def test_scale_then_shift(self, scalar_lbf):
        """``alpha * grf + b``."""
        h = 2.0 * scalar_lbf + 5.0
        X = jnp.array([[1.0]])
        np.testing.assert_allclose(
            h.predict_mean(X),
            2.0 * scalar_lbf.predict_mean(X) + 5.0,
            atol=1e-5,
        )

    def test_scale_sum(self):
        """``alpha * (grf1 + grf2)``."""
        gp1 = _ScalarGP(lengthscale=1.0, variance=1.0)
        gp2 = _ScalarGP(lengthscale=0.5, variance=0.5)
        h = 3.0 * (gp1 + gp2)
        X = jnp.ones((5, 2))
        np.testing.assert_allclose(
            h.predict_mean(X),
            3.0 * (gp1.predict_mean(X) + gp2.predict_mean(X)),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            h.predict_variance(X),
            9.0 * (gp1.predict_variance(X) + gp2.predict_variance(X)),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Mathematical correctness tests (ground-truth verification)
# ---------------------------------------------------------------------------
#
# These tests verify that the algebra operations produce the correct
# moments by comparing against dense numpy computations.  They also
# cross-check internal consistency (variance = diag of per-point cov,
# per-output cov blocks = submatrix of full joint, etc.).
#
# The strategy is:
#   1. Build a LinearBasisFunction with *known* feature map + weight dist.
#   2. Evaluate the feature map at test points → get Φ(X) as a dense array.
#   3. Compute ground-truth moments via Φ(X) @ C @ Φ(X)^T in numpy.
#   4. Compare against the GRF algebra wrapper's predict_* methods.
#   5. Monte Carlo sanity: sample many realizations and compare empirical
#      moments to predicted moments.
# ---------------------------------------------------------------------------


# ---------- shared fixtures for correctness tests ----------

@pytest.fixture
def correctness_X():
    """4 test input points for correctness checks."""
    return jnp.array([[-1.0], [0.0], [0.5], [1.0]])


@pytest.fixture
def correctness_w_mean():
    return jnp.array([2.0, -1.0])


@pytest.fixture
def correctness_w_cov():
    """Non-diagonal weight covariance to ensure cross-correlations."""
    return jnp.array([[1.0, 0.3], [0.3, 0.5]])


def _simple_multi_output_features(X):
    """Feature map: scalar input → (n, 3, 2) features.

    Output 0: [1, x]
    Output 1: [x, x^2]
    Output 2: [1, -x]

    The outputs are *correlated* through the shared weight vector.
    """
    x = X[..., 0]  # (n,)
    ones = jnp.ones_like(x)
    return jnp.stack([
        jnp.stack([ones, x], axis=-1),       # output 0: [1, x]
        jnp.stack([x, x ** 2], axis=-1),     # output 1: [x, x^2]
        jnp.stack([ones, -x], axis=-1),      # output 2: [1, -x]
    ], axis=-2)  # (n, 3, 2)


@pytest.fixture
def correlated_lbf(correctness_w_mean, correctness_w_cov):
    """Multi-output LBF with correlated outputs (via shared weights)."""
    weights = MultivariateNormal(
        loc=correctness_w_mean,
        cov=correctness_w_cov,
    )
    return LinearBasisFunction(
        feature_map=_simple_multi_output_features,
        weights=weights,
        input_shape=(1,),
        output_shape=(3,),
    )


def _simple_scalar_features(X):
    """Feature map: scalar input → [1, x, x^2]."""
    x = X[..., 0]
    return jnp.stack([jnp.ones_like(x), x, x ** 2], axis=-1)  # (n, 3)


@pytest.fixture
def scalar_correctness_lbf():
    """Scalar LBF with non-trivial weight covariance."""
    w_mean = jnp.array([1.0, -0.5, 0.2])
    w_cov = jnp.array([
        [1.0, 0.2, -0.1],
        [0.2, 0.8, 0.05],
        [-0.1, 0.05, 0.3],
    ])
    weights = MultivariateNormal(loc=w_mean, cov=w_cov)
    return LinearBasisFunction(
        feature_map=_simple_scalar_features,
        weights=weights,
        input_shape=(1,),
    )


class TestLinearBasisFunctionCorrectness:
    """Verify LinearBasisFunction moments against dense ground truth."""

    def test_mean_scalar(self, scalar_correctness_lbf, correctness_X):
        X = correctness_X
        phi = np.array(_simple_scalar_features(X))   # (4, 3)
        m = np.array([1.0, -0.5, 0.2])
        expected = phi @ m
        np.testing.assert_allclose(
            scalar_correctness_lbf.predict_mean(X), expected, atol=1e-5
        )

    def test_variance_equals_cov_diag_scalar(self, scalar_correctness_lbf, correctness_X):
        X = correctness_X
        var = scalar_correctness_lbf.predict_variance(X)
        cov = scalar_correctness_lbf.predict_covariance(X, joint_inputs=True)
        np.testing.assert_allclose(var, np.diag(np.array(cov)), atol=1e-5)

    def test_joint_cov_scalar(self, scalar_correctness_lbf, correctness_X):
        X = correctness_X
        phi = np.array(_simple_scalar_features(X))   # (4, 3)
        C = np.array([
            [1.0, 0.2, -0.1],
            [0.2, 0.8, 0.05],
            [-0.1, 0.05, 0.3],
        ])
        expected = phi @ C @ phi.T  # (4, 4)
        cov = scalar_correctness_lbf.predict_covariance(X, joint_inputs=True)
        np.testing.assert_allclose(cov, expected, atol=1e-5)

    def test_mean_multi_output(self, correlated_lbf, correctness_X, correctness_w_mean):
        X = correctness_X
        phi = np.array(_simple_multi_output_features(X))  # (4, 3, 2)
        m = np.array(correctness_w_mean)
        expected = np.einsum("now,w->no", phi, m)  # (4, 3)
        np.testing.assert_allclose(
            correlated_lbf.predict_mean(X), expected, atol=1e-5
        )

    def test_full_joint_cov_multi_output(
        self, correlated_lbf, correctness_X, correctness_w_cov
    ):
        X = correctness_X
        phi = np.array(_simple_multi_output_features(X))  # (4, 3, 2)
        C = np.array(correctness_w_cov)
        expected = _dense_lbf_full_joint_cov(phi, C)  # (12, 12)
        cov = correlated_lbf.predict_covariance(
            X, joint_inputs=True, joint_outputs=True
        )
        np.testing.assert_allclose(cov, expected, atol=1e-5)

    def test_variance_equals_diag_of_per_point_cov(
        self, correlated_lbf, correctness_X, correctness_w_cov
    ):
        """predict_variance should equal diag of joint_outputs cov at each point."""
        X = correctness_X
        var = correlated_lbf.predict_variance(X)      # (4, 3)
        per_pt = correlated_lbf.predict_covariance(
            X, joint_outputs=True
        )                                               # (4, 3, 3)
        for i in range(4):
            np.testing.assert_allclose(
                var[i], np.diag(np.array(per_pt[i])), atol=1e-5
            )

    def test_per_output_cov_matches_full_joint_blocks(
        self, correlated_lbf, correctness_X
    ):
        """joint_inputs per-output cov should match blocks of full joint."""
        X = correctness_X
        n = 4
        d_out = 3
        per_out = correlated_lbf.predict_covariance(
            X, joint_inputs=True
        )  # (3, 4, 4)
        full = correlated_lbf.predict_covariance(
            X, joint_inputs=True, joint_outputs=True
        )  # (12, 12)
        for o in range(d_out):
            idx = np.arange(n) * d_out + o
            block = np.array(full)[np.ix_(idx, idx)]
            np.testing.assert_allclose(per_out[o], block, atol=1e-5)


class TestLinearMapCorrectness:
    """Verify A @ grf moments against dense ground truth.

    h(x) = A @ g(x) where g is a multi-output LBF.
    Effective feature map: Φ_h(x) = (A @ Φ_g(x)^T)^T, then standard formulas.
    """

    def test_mean_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_mean
    ):
        A = jnp.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])  # (2, 3)
        h = A @ correlated_lbf
        X = correctness_X

        # Ground truth: A @ Φ_g(x) @ m
        phi_g = np.array(_simple_multi_output_features(X))  # (4, 3, 2)
        m = np.array(correctness_w_mean)
        g_mean = np.einsum("now,w->no", phi_g, m)  # (4, 3)
        expected = np.einsum("oh,...h->...o", np.array(A), g_mean)  # (4, 2)
        np.testing.assert_allclose(h.predict_mean(X), expected, atol=1e-5)

    def test_per_point_cov_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_cov
    ):
        """Per-point output covariance: A @ Φ(x) @ C @ Φ(x)^T @ A^T."""
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        A = jnp.array(A_np)
        h = A @ correlated_lbf
        X = correctness_X

        phi_g = np.array(_simple_multi_output_features(X))  # (4, 3, 2)
        C = np.array(correctness_w_cov)

        cov_jo = h.predict_covariance(X, joint_outputs=True)  # (4, 2, 2)
        for i in range(4):
            # Effective features for h at point i: A @ phi_g[i]
            phi_h_i = A_np @ phi_g[i]   # (2, 2)
            expected_i = phi_h_i @ C @ phi_h_i.T  # (2, 2)
            np.testing.assert_allclose(cov_jo[i], expected_i, atol=1e-5)

    def test_variance_is_diag_of_per_point_cov(
        self, correlated_lbf, correctness_X
    ):
        A = jnp.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        h = A @ correlated_lbf
        X = correctness_X

        var = h.predict_variance(X)  # (4, 2)
        cov_jo = h.predict_covariance(X, joint_outputs=True)  # (4, 2, 2)
        for i in range(4):
            np.testing.assert_allclose(
                var[i], np.diag(np.array(cov_jo[i])), atol=1e-6
            )

    def test_full_joint_cov_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_cov
    ):
        """Full joint covariance against dense computation."""
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        A = jnp.array(A_np)
        h = A @ correlated_lbf
        X = correctness_X

        phi_g = np.array(_simple_multi_output_features(X))  # (4, 3, 2)
        C = np.array(correctness_w_cov)

        # Effective features for h: phi_h[i, o, :] = sum_w A[o, w] * phi_g[i, w, :]
        phi_h = np.einsum("oh,nhw->now", A_np, phi_g)  # (4, 2, 2)
        expected = _dense_lbf_full_joint_cov(phi_h, C)  # (8, 8)

        cov = h.predict_covariance(
            X, joint_inputs=True, joint_outputs=True
        )  # (8, 8)
        np.testing.assert_allclose(cov, expected, atol=1e-5)

    def test_per_output_cov_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_cov
    ):
        """Per-output cross-input cov: key test for the covariance fix.

        This is the path that was buggy before (used phi**2 instead of
        full cross-output covariance).  With correlated base outputs,
        the correct formula requires the full covariance, not just the
        diagonal.
        """
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        A = jnp.array(A_np)
        h = A @ correlated_lbf
        X = correctness_X

        phi_g = np.array(_simple_multi_output_features(X))  # (4, 3, 2)
        C = np.array(correctness_w_cov)
        phi_h = np.einsum("oh,nhw->now", A_np, phi_g)  # (4, 2, 2)
        full = _dense_lbf_full_joint_cov(phi_h, C)       # (8, 8)

        n, d_out = 4, 2
        per_out = h.predict_covariance(X, joint_inputs=True)  # (2, 4, 4)
        for o in range(d_out):
            # Extract the n×n block for output o from the full joint.
            # Full joint layout: (x0_o0, x0_o1, x1_o0, x1_o1, ...)
            idx = np.arange(n) * d_out + o
            expected_block = full[np.ix_(idx, idx)]
            np.testing.assert_allclose(
                per_out[o], expected_block, atol=1e-5,
                err_msg=f"Per-output cov mismatch for output {o}"
            )

    def test_per_output_cov_cross_output_matters(
        self, correctness_w_cov
    ):
        """Demonstrate that ignoring cross-output correlations gives
        wrong answers, and our code gets it right.

        Build a base where outputs 0 and 2 are anti-correlated
        through shared weights, then use A = [[1, 0, 1], ...] so
        that the cross-correlation affects the result.
        """
        # Use the correlated LBF fixture values
        w_mean = jnp.array([2.0, -1.0])
        w_cov = jnp.array([[1.0, 0.3], [0.3, 0.5]])
        weights = MultivariateNormal(loc=w_mean, cov=w_cov)
        base = LinearBasisFunction(
            feature_map=_simple_multi_output_features,
            weights=weights,
            input_shape=(1,),
            output_shape=(3,),
        )

        # A that mixes outputs 0 and 2 (which are anti-correlated)
        A_np = np.array([[1.0, 0.0, 1.0]])  # (1, 3) → scalar output
        A = jnp.array(A_np)
        h = A @ base

        X = jnp.array([[-1.0], [1.0]])
        phi_g = np.array(_simple_multi_output_features(X))  # (2, 3, 2)
        C = np.array(w_cov)

        # Correct: full formula using effective features
        phi_h = np.einsum("oh,nhw->now", A_np, phi_g)  # (2, 1, 2)
        correct_cov = _dense_lbf_full_joint_cov(phi_h, C)  # (2, 2)

        # Wrong: the old phi**2 formula (ignores cross-output correlations)
        # Per-output base cov (joint_inputs, not joint_outputs):
        #   base_cov_w[i, j] for each output w independently
        base_per_out = np.array(
            base.predict_covariance(X, joint_inputs=True)
        )  # (3, 2, 2)
        wrong_cov = np.einsum(
            "ow,...wij->...oij", A_np ** 2, base_per_out
        )[0]  # squeeze the single output dim → (2, 2)

        # These should differ
        assert not np.allclose(correct_cov, wrong_cov, atol=1e-3), \
            "Test is degenerate: correct and wrong formulas agree"

        # Our implementation should match the correct formula
        h_cov = h.predict_covariance(X, joint_inputs=True)  # (1, 2, 2)
        np.testing.assert_allclose(h_cov[0], correct_cov, atol=1e-5)


class TestScaleCorrectness:
    """Verify scalar scaling moments against ground truth."""

    def test_variance_ground_truth(
        self, scalar_correctness_lbf, correctness_X
    ):
        alpha = 3.0
        h = alpha * scalar_correctness_lbf
        X = correctness_X

        phi = np.array(_simple_scalar_features(X))
        C = np.array([
            [1.0, 0.2, -0.1],
            [0.2, 0.8, 0.05],
            [-0.1, 0.05, 0.3],
        ])
        base_var = np.diag(phi @ C @ phi.T)
        np.testing.assert_allclose(
            h.predict_variance(X), alpha ** 2 * base_var, atol=1e-5
        )

    def test_full_joint_cov_ground_truth(
        self, scalar_correctness_lbf, correctness_X
    ):
        alpha = -2.5
        h = alpha * scalar_correctness_lbf
        X = correctness_X

        phi = np.array(_simple_scalar_features(X))
        C = np.array([
            [1.0, 0.2, -0.1],
            [0.2, 0.8, 0.05],
            [-0.1, 0.05, 0.3],
        ])
        expected = alpha ** 2 * (phi @ C @ phi.T)
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True), expected, atol=1e-5
        )


class TestShiftCorrectness:
    """Verify bias shift moments against ground truth."""

    def test_mean_ground_truth(self, scalar_correctness_lbf, correctness_X):
        b = 42.0
        h = scalar_correctness_lbf + b
        X = correctness_X

        phi = np.array(_simple_scalar_features(X))
        m = np.array([1.0, -0.5, 0.2])
        expected = phi @ m + b
        np.testing.assert_allclose(h.predict_mean(X), expected, atol=1e-5)

    def test_cov_unchanged_ground_truth(
        self, scalar_correctness_lbf, correctness_X
    ):
        h = scalar_correctness_lbf + 999.0
        X = correctness_X

        phi = np.array(_simple_scalar_features(X))
        C = np.array([
            [1.0, 0.2, -0.1],
            [0.2, 0.8, 0.05],
            [-0.1, 0.05, 0.3],
        ])
        expected = phi @ C @ phi.T
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True), expected, atol=1e-5
        )


class TestIndependentSumCorrectness:
    """Verify independent sum moments against ground truth."""

    def test_mean_with_nonzero_means(self, correctness_X):
        """Sum of LBFs with different non-zero means."""
        w1 = MultivariateNormal(
            loc=jnp.array([1.0, -0.5, 0.2]),
            cov=0.1 * jnp.eye(3),
        )
        w2 = MultivariateNormal(
            loc=jnp.array([-0.3, 0.8, 0.0]),
            cov=0.2 * jnp.eye(3),
        )
        lbf1 = LinearBasisFunction(
            feature_map=_simple_scalar_features,
            weights=w1,
            input_shape=(1,),
        )
        lbf2 = LinearBasisFunction(
            feature_map=_simple_scalar_features,
            weights=w2,
            input_shape=(1,),
        )
        h = lbf1 + lbf2
        X = correctness_X

        phi = np.array(_simple_scalar_features(X))
        m1 = np.array([1.0, -0.5, 0.2])
        m2 = np.array([-0.3, 0.8, 0.0])
        expected = phi @ m1 + phi @ m2
        np.testing.assert_allclose(h.predict_mean(X), expected, atol=1e-5)

    def test_variance_ground_truth(self, correctness_X):
        """Sum variance = sum of individual variances."""
        C1 = np.array([
            [1.0, 0.2, 0.0],
            [0.2, 0.5, 0.0],
            [0.0, 0.0, 0.3],
        ])
        C2 = np.array([
            [0.5, 0.0, 0.1],
            [0.0, 0.8, 0.0],
            [0.1, 0.0, 0.2],
        ])
        w1 = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.array(C1))
        w2 = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.array(C2))
        lbf1 = LinearBasisFunction(
            feature_map=_simple_scalar_features, weights=w1, input_shape=(1,),
        )
        lbf2 = LinearBasisFunction(
            feature_map=_simple_scalar_features, weights=w2, input_shape=(1,),
        )
        h = lbf1 + lbf2
        X = correctness_X

        phi = np.array(_simple_scalar_features(X))
        expected_var = np.diag(phi @ C1 @ phi.T) + np.diag(phi @ C2 @ phi.T)
        np.testing.assert_allclose(
            h.predict_variance(X), expected_var, atol=1e-5
        )

    def test_covariance_ground_truth(self, correctness_X):
        """Sum covariance = sum of individual covariances."""
        C1 = np.array([
            [1.0, 0.2, 0.0],
            [0.2, 0.5, 0.0],
            [0.0, 0.0, 0.3],
        ])
        C2 = np.array([
            [0.5, 0.0, 0.1],
            [0.0, 0.8, 0.0],
            [0.1, 0.0, 0.2],
        ])
        w1 = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.array(C1))
        w2 = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.array(C2))
        lbf1 = LinearBasisFunction(
            feature_map=_simple_scalar_features, weights=w1, input_shape=(1,),
        )
        lbf2 = LinearBasisFunction(
            feature_map=_simple_scalar_features, weights=w2, input_shape=(1,),
        )
        h = lbf1 + lbf2
        X = correctness_X

        phi = np.array(_simple_scalar_features(X))
        expected_cov = phi @ C1 @ phi.T + phi @ C2 @ phi.T
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True), expected_cov, atol=1e-5
        )


class TestCompositionCorrectness:
    """Verify composed operations: A @ grf + b, alpha * (grf1 + grf2), etc."""

    def test_affine_full_joint_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_mean, correctness_w_cov
    ):
        """A @ grf + b: full joint covariance (bias doesn't affect cov)."""
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        b = jnp.array([10.0, -5.0])
        h = jnp.array(A_np) @ correlated_lbf + b
        X = correctness_X

        phi_g = np.array(_simple_multi_output_features(X))
        C = np.array(correctness_w_cov)
        m = np.array(correctness_w_mean)

        # Mean
        g_mean = np.einsum("now,w->no", phi_g, m)
        expected_mean = np.einsum("oh,...h->...o", A_np, g_mean) + np.array(b)
        np.testing.assert_allclose(h.predict_mean(X), expected_mean, atol=1e-5)

        # Covariance (unaffected by bias)
        phi_h = np.einsum("oh,nhw->now", A_np, phi_g)
        expected_cov = _dense_lbf_full_joint_cov(phi_h, C)
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True, joint_outputs=True),
            expected_cov,
            atol=1e-5,
        )

    def test_scale_of_linear_map_ground_truth(
        self, correlated_lbf, correctness_X, correctness_w_cov
    ):
        """alpha * (A @ grf): cov = alpha^2 * A @ Cov_g @ A^T."""
        alpha = 2.5
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        h = alpha * (jnp.array(A_np) @ correlated_lbf)
        X = correctness_X

        phi_g = np.array(_simple_multi_output_features(X))
        C = np.array(correctness_w_cov)
        phi_h = np.einsum("oh,nhw->now", A_np, phi_g)
        expected_cov = alpha ** 2 * _dense_lbf_full_joint_cov(phi_h, C)
        np.testing.assert_allclose(
            h.predict_covariance(X, joint_inputs=True, joint_outputs=True),
            expected_cov,
            atol=1e-4,
        )


class TestMonteCarlo:
    """Monte Carlo validation: sample realizations and compare empirical
    moments to predicted moments.

    This catches math bugs that a formula-vs-formula test would miss
    (e.g. both the test and implementation use the same wrong formula).
    """

    N_SAMPLES = 20_000
    MC_ATOL_MEAN = 0.05
    MC_ATOL_VAR = 0.25     # variance/covariance estimation has higher MC noise

    def _sample_outputs(self, lbf, X, key, n_samples):
        """Draw n_samples function realizations and evaluate at X."""
        f = sample(lbf, key=key, sample_shape=(n_samples,))
        return np.array(f(X))  # (n_samples, n, [*output_shape])

    def test_linear_basis_function_moments(self, scalar_correctness_lbf, correctness_X):
        """LBF sample moments should match predicted moments."""
        X = correctness_X
        key = jax.random.PRNGKey(123)
        Y = self._sample_outputs(
            scalar_correctness_lbf, X, key, self.N_SAMPLES
        )  # (N, 4)

        pred_mean = np.array(scalar_correctness_lbf.predict_mean(X))
        pred_cov = np.array(
            scalar_correctness_lbf.predict_covariance(X, joint_inputs=True)
        )

        emp_mean = Y.mean(axis=0)
        emp_cov = np.cov(Y, rowvar=False)

        np.testing.assert_allclose(emp_mean, pred_mean, atol=self.MC_ATOL_MEAN)
        np.testing.assert_allclose(emp_cov, pred_cov, atol=self.MC_ATOL_VAR)

    def test_linear_map_moments(self, correlated_lbf, correctness_X):
        """A @ grf: sample the base, apply A in sample space, check moments."""
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        A = jnp.array(A_np)
        h = A @ correlated_lbf
        X = correctness_X
        key = jax.random.PRNGKey(456)

        # Sample base, apply A in sample space
        Y_base = self._sample_outputs(
            correlated_lbf, X, key, self.N_SAMPLES
        )  # (N, 4, 3)
        Y_h = np.einsum("oh,...h->...o", A_np, Y_base)  # (N, 4, 2)

        pred_mean = np.array(h.predict_mean(X))  # (4, 2)
        pred_var = np.array(h.predict_variance(X))  # (4, 2)

        emp_mean = Y_h.mean(axis=0)
        emp_var = Y_h.var(axis=0)

        np.testing.assert_allclose(emp_mean, pred_mean, atol=self.MC_ATOL_MEAN)
        np.testing.assert_allclose(emp_var, pred_var, atol=self.MC_ATOL_VAR)

    def test_linear_map_full_joint_cov(self, correlated_lbf, correctness_X):
        """A @ grf: full joint covariance via MC."""
        A_np = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        A = jnp.array(A_np)
        h = A @ correlated_lbf
        X = correctness_X
        key = jax.random.PRNGKey(789)

        Y_base = self._sample_outputs(
            correlated_lbf, X, key, self.N_SAMPLES
        )
        Y_h = np.einsum("oh,...h->...o", A_np, Y_base)  # (N, 4, 2)
        # Flatten to (N, 8) for covariance estimation
        Y_flat = Y_h.reshape(self.N_SAMPLES, -1)

        pred_cov = np.array(
            h.predict_covariance(X, joint_inputs=True, joint_outputs=True)
        )
        emp_cov = np.cov(Y_flat, rowvar=False)

        np.testing.assert_allclose(emp_cov, pred_cov, atol=self.MC_ATOL_VAR)

    def test_scale_moments(self, scalar_correctness_lbf, correctness_X):
        """alpha * grf: sample base, scale, check moments."""
        alpha = 3.0
        h = alpha * scalar_correctness_lbf
        X = correctness_X
        key = jax.random.PRNGKey(101)

        Y_base = self._sample_outputs(
            scalar_correctness_lbf, X, key, self.N_SAMPLES
        )
        Y_h = alpha * Y_base

        pred_mean = np.array(h.predict_mean(X))
        pred_var = np.array(h.predict_variance(X))

        np.testing.assert_allclose(
            Y_h.mean(axis=0), pred_mean, atol=self.MC_ATOL_MEAN
        )
        np.testing.assert_allclose(
            Y_h.var(axis=0), pred_var, atol=self.MC_ATOL_VAR
        )

    def test_shift_moments(self, scalar_correctness_lbf, correctness_X):
        """grf + b: sample base, shift, check moments."""
        b = 42.0
        h = scalar_correctness_lbf + b
        X = correctness_X
        key = jax.random.PRNGKey(202)

        Y_base = self._sample_outputs(
            scalar_correctness_lbf, X, key, self.N_SAMPLES
        )
        Y_h = Y_base + b

        pred_mean = np.array(h.predict_mean(X))
        pred_var = np.array(h.predict_variance(X))

        np.testing.assert_allclose(
            Y_h.mean(axis=0), pred_mean, atol=self.MC_ATOL_MEAN
        )
        np.testing.assert_allclose(
            Y_h.var(axis=0), pred_var, atol=self.MC_ATOL_VAR
        )

    def test_independent_sum_moments(self, correctness_X):
        """grf1 + grf2: sample both independently, add, check moments."""
        w1 = MultivariateNormal(
            loc=jnp.array([1.0, -0.5, 0.2]),
            cov=0.5 * jnp.eye(3),
        )
        w2 = MultivariateNormal(
            loc=jnp.array([-0.3, 0.8, 0.0]),
            cov=0.3 * jnp.eye(3),
        )
        lbf1 = LinearBasisFunction(
            feature_map=_simple_scalar_features, weights=w1, input_shape=(1,),
        )
        lbf2 = LinearBasisFunction(
            feature_map=_simple_scalar_features, weights=w2, input_shape=(1,),
        )
        h = lbf1 + lbf2
        X = correctness_X

        key1, key2 = jax.random.split(jax.random.PRNGKey(303))
        Y1 = self._sample_outputs(lbf1, X, key1, self.N_SAMPLES)
        Y2 = self._sample_outputs(lbf2, X, key2, self.N_SAMPLES)
        Y_h = Y1 + Y2

        pred_mean = np.array(h.predict_mean(X))
        pred_cov = np.array(h.predict_covariance(X, joint_inputs=True))
        emp_cov = np.cov(Y_h, rowvar=False)

        np.testing.assert_allclose(
            Y_h.mean(axis=0), pred_mean, atol=self.MC_ATOL_MEAN
        )
        np.testing.assert_allclose(emp_cov, pred_cov, atol=self.MC_ATOL_VAR)
