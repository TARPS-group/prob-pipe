"""Tests for GaussianRandomFunction, LinearBasisFunction, and GRF algebra."""

from __future__ import annotations

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
        f = scalar_lbf.sample(key)
        assert callable(f)
        X = jnp.linspace(-1, 1, 10).reshape(-1, 1)
        y = f(X)
        assert y.shape == (10,)

    def test_sample_batched(self, key, scalar_lbf):
        f = scalar_lbf.sample(key, sample_shape=(7,))
        assert callable(f)
        X = jnp.linspace(-1, 1, 10).reshape(-1, 1)
        y = f(X)
        assert y.shape == (7, 10)

    def test_sample_multi_dim_shape(self, key, scalar_lbf):
        f = scalar_lbf.sample(key, sample_shape=(3, 4))
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        y = f(X)
        assert y.shape == (3, 4, 5)

    def test_sample_consistency(self, key, scalar_lbf):
        """Same function realization evaluates consistently."""
        f = scalar_lbf.sample(key)
        X1 = jnp.array([[0.0], [1.0]])
        X2 = jnp.array([[0.0], [2.0]])
        y1 = f(X1)
        y2 = f(X2)
        # f(0) should be the same in both calls
        np.testing.assert_allclose(y1[0], y2[0], atol=1e-6)

    def test_sample_batched_consistency(self, key, scalar_lbf):
        """Batched realizations: each trajectory is consistent."""
        f = scalar_lbf.sample(key, sample_shape=(5,))
        X1 = jnp.array([[0.0]])
        X2 = jnp.array([[0.0]])
        y1 = f(X1)  # (5, 1)
        y2 = f(X2)  # (5, 1)
        np.testing.assert_allclose(y1, y2, atol=1e-6)

    def test_sample_multi_output(self, key, multi_output_lbf):
        f = multi_output_lbf.sample(key)
        X = jnp.linspace(-1, 1, 8).reshape(-1, 1)
        y = f(X)
        assert y.shape == (8, 2)

    def test_sample_batched_multi_output(self, key, multi_output_lbf):
        f = multi_output_lbf.sample(key, sample_shape=(5,))
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
