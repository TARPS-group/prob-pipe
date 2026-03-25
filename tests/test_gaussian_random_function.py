"""Tests for GaussianRandomFunction, LinearBasisFunction, and LinearOutputTransform."""

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
    LinearOutputTransform,
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
# LinearOutputTransform tests
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


@pytest.fixture
def transformed_grf(weight_grf):
    """LinearOutputTransform wrapping a 3→2 linear map."""
    phi = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # (2, 3)
    return LinearOutputTransform(
        base_function=weight_grf,
        phi=phi,
    )


class TestLinearOutputTransform:
    """Tests for the LinearOutputTransform class."""

    def test_isinstance_hierarchy(self, transformed_grf):
        assert isinstance(transformed_grf, LinearOutputTransform)
        assert isinstance(transformed_grf, GaussianRandomFunction)
        assert isinstance(transformed_grf, ArrayRandomFunction)
        assert isinstance(transformed_grf, RandomFunction)

    def test_shapes(self, transformed_grf):
        assert transformed_grf.input_shape == (1,)
        assert transformed_grf.output_shape == (2,)

    def test_supports_joint_outputs(self, transformed_grf):
        assert transformed_grf.supports_joint_outputs is True

    def test_inherits_joint_inputs(self, weight_grf, transformed_grf):
        assert transformed_grf.supports_joint_inputs == weight_grf.supports_joint_inputs

    def test_marginal(self, transformed_grf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = transformed_grf(X)
        assert isinstance(dist, Normal)
        assert dist.batch_shape == (5, 2)

    def test_joint_outputs(self, transformed_grf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = transformed_grf(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (2,)
        assert dist.batch_shape == (5,)

    def test_joint_inputs(self, transformed_grf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = transformed_grf(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        # Joint over n=5 input points, independent over 2 outputs
        assert dist.batch_shape == (2,)
        assert dist.event_shape == (5,)

    def test_full_joint(self, transformed_grf):
        X = jnp.linspace(-1, 1, 5).reshape(-1, 1)
        dist = transformed_grf(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert dist.event_shape == (10,)  # 5 * 2

    # -- Validation ---------------------------------------------------------

    def test_invalid_base_type(self):
        with pytest.raises(TypeError, match="GaussianRandomFunction"):
            LinearOutputTransform(
                base_function="not_a_grf",
                phi=jnp.eye(2),
            )

    def test_invalid_base_output_shape(self, scalar_lbf):
        """base_function must have 1-D output_shape."""
        with pytest.raises(ValueError, match="1-D"):
            LinearOutputTransform(
                base_function=scalar_lbf,
                phi=jnp.eye(2),
            )

    def test_phi_dim_mismatch(self, weight_grf):
        """phi columns must match base output dim."""
        with pytest.raises(ValueError, match="phi columns"):
            LinearOutputTransform(
                base_function=weight_grf,
                phi=jnp.eye(2),  # (2, 2) but weight_grf output is (3,)
            )

    def test_mean_value(self, weight_grf, transformed_grf):
        """Transformed mean = bias + Phi @ base_mean."""
        X = jnp.array([[0.5]])
        base_mean = weight_grf.predict_mean(X)  # (1, 3)
        transformed_mean = transformed_grf.predict_mean(X)  # (1, 2)

        phi = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        expected = jnp.einsum("ow,...w->...o", phi, base_mean)  # (1, 2)
        np.testing.assert_allclose(transformed_mean, expected, atol=1e-5)
