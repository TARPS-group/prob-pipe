"""Comprehensive tests for the emulator submodule."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal, Normal
from probpipe.surrogate.emulator import (
    Emulator,
    GaussianEmulator,
    LinCombGaussianWeights,
    LinearGaussianRegressor,
)

# ---------------------------------------------------------------------------
# Mock emulators for testing
# ---------------------------------------------------------------------------


def _rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """Simple RBF kernel for mock GP."""
    sq_dist = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale ** 2)


class _ScalarGP(GaussianEmulator):
    """Mock scalar-output GP with RBF kernel."""

    supports_joint_inputs = True

    def __init__(self, lengthscale=1.0, variance=1.0, noise=0.01):
        super().__init__(input_shape=(2,), output_shape=())
        self._ls = lengthscale
        self._var = variance
        self._noise = noise

    def predict_mean(self, X):
        # Zero prior mean for simplicity.
        extra_batch, n = self._parse_X(X)
        return jnp.zeros((*extra_batch, n))

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.full((*extra_batch, n), self._var)

    def predict_covariance(self, X, *, joint_inputs=False, joint_outputs=False):
        if not joint_inputs:
            raise NotImplementedError
        extra_batch, n = self._parse_X(X)
        # Compute kernel for each batch element.
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


class _MultiOutputEmulator(GaussianEmulator):
    """Mock 2-output emulator with independent outputs and joint inputs."""

    supports_joint_inputs = True
    supports_joint_outputs = True

    def __init__(self):
        super().__init__(input_shape=(2,), output_shape=(2,))

    def predict_mean(self, X):
        _, _n = self._parse_X(X)
        # Output 0 = sum(x), Output 1 = 2*sum(x)
        s = jnp.sum(X, axis=-1)  # (*eb, n)
        return jnp.stack([s, 2 * s], axis=-1)  # (*eb, n, 2)

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.ones((*extra_batch, n, 2))

    def predict_covariance(self, X, *, joint_inputs=False, joint_outputs=False):
        extra_batch, n = self._parse_X(X)

        if joint_inputs and joint_outputs:
            # Full: (*eb, 2n, 2n) - block diagonal (outputs independent)
            K = jnp.eye(n)
            full = jnp.kron(K, jnp.eye(2))  # (2n, 2n)
            return jnp.broadcast_to(full, (*extra_batch, 2 * n, 2 * n))

        if joint_inputs and not joint_outputs:
            # (*eb, 2, n, n) - one nxn identity per output
            K = jnp.eye(n)
            return jnp.broadcast_to(
                jnp.stack([K, K])[None] if extra_batch else jnp.stack([K, K]),
                (*extra_batch, 2, n, n),
            )

        if not joint_inputs and joint_outputs:
            # (*eb, n, 2, 2) - identity covariance per point
            C = jnp.eye(2)
            return jnp.broadcast_to(C, (*extra_batch, n, 2, 2))

        raise ValueError("Use predict_variance for fully marginal.")


class _MarginalOnlyEmulator(GaussianEmulator):
    """Mock emulator that only supports marginal predictions."""

    def __init__(self):
        super().__init__(input_shape=(3,), output_shape=())

    def predict_mean(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.zeros((*extra_batch, n))

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        return jnp.ones((*extra_batch, n))


class _NonGaussianEmulator(Emulator):
    """Mock non-Gaussian emulator that directly implements predict."""

    def __init__(self):
        super().__init__(input_shape=(2,), output_shape=())

    def predict(self, X, *, joint_inputs=False, joint_outputs=False):
        extra_batch, n = self._parse_X(X)
        return Normal(
            loc=jnp.zeros((*extra_batch, n)),
            scale=jnp.ones((*extra_batch, n)),
        )


# ---------------------------------------------------------------------------
# Tests: Emulator base validation
# ---------------------------------------------------------------------------


class TestEmulatorValidation:
    def test_bad_input_shape_too_few_dims(self):
        em = _MarginalOnlyEmulator()
        X = jnp.ones((3,))  # Need at least (n, 3)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            em(X)

    def test_bad_input_shape_wrong_trailing(self):
        em = _MarginalOnlyEmulator()  # input_shape=(3,)
        X = jnp.ones((5, 4))  # trailing dim is 4, not 3
        with pytest.raises(ValueError, match="do not match input_shape"):
            em(X)

    def test_unsupported_joint_inputs(self):
        em = _MarginalOnlyEmulator()
        X = jnp.ones((5, 3))
        with pytest.raises(ValueError, match="does not support joint_inputs"):
            em(X, joint_inputs=True)

    def test_unsupported_joint_outputs(self):
        em = _MarginalOnlyEmulator()
        X = jnp.ones((5, 3))
        with pytest.raises(ValueError, match="does not support joint_outputs"):
            em(X, joint_outputs=True)

    def test_valid_input_passes(self):
        em = _MarginalOnlyEmulator()
        X = jnp.ones((5, 3))
        dist = em(X)
        assert isinstance(dist, Normal)

    def test_parse_X_basic(self):
        em = _MarginalOnlyEmulator()  # input_shape=(3,)
        X = jnp.ones((10, 3))
        eb, n = em._parse_X(X)
        assert eb == ()
        assert n == 10

    def test_parse_X_extra_batch(self):
        em = _MarginalOnlyEmulator()  # input_shape=(3,)
        X = jnp.ones((4, 7, 10, 3))
        eb, n = em._parse_X(X)
        assert eb == (4, 7)
        assert n == 10

    def test_repr(self):
        em = _MarginalOnlyEmulator()
        r = repr(em)
        assert "_MarginalOnlyEmulator" in r
        assert "input_shape=(3,)" in r
        assert "output_shape=()" in r

    def test_properties(self):
        em = _MarginalOnlyEmulator()
        assert em.input_shape == (3,)
        assert em.output_shape == ()
        assert em.supports_joint_inputs is False
        assert em.supports_joint_outputs is False

    def test_non_gaussian_emulator(self):
        em = _NonGaussianEmulator()
        X = jnp.ones((5, 2))
        dist = em(X)
        assert isinstance(dist, Normal)


# ---------------------------------------------------------------------------
# Tests: GaussianEmulator - marginal mode
# ---------------------------------------------------------------------------


class TestGaussianEmulatorMarginal:
    def test_scalar_output_returns_normal(self):
        em = _ScalarGP()
        X = jnp.ones((10, 2))
        dist = em(X)
        assert isinstance(dist, Normal)

    def test_scalar_output_batch_shape(self):
        em = _ScalarGP()
        X = jnp.ones((10, 2))
        dist = em(X)
        assert tuple(dist.batch_shape) == (10,)
        assert tuple(dist.event_shape) == ()

    def test_multi_output_returns_normal(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((10, 2))
        dist = em(X)
        assert isinstance(dist, Normal)

    def test_multi_output_batch_shape(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((10, 2))
        dist = em(X)
        assert tuple(dist.batch_shape) == (10, 2)
        assert tuple(dist.event_shape) == ()

    def test_extra_batch_scalar(self):
        em = _ScalarGP()
        X = jnp.ones((3, 10, 2))
        dist = em(X)
        assert tuple(dist.batch_shape) == (3, 10)
        assert tuple(dist.event_shape) == ()

    def test_extra_batch_multi_output(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((3, 10, 2))
        dist = em(X)
        assert tuple(dist.batch_shape) == (3, 10, 2)
        assert tuple(dist.event_shape) == ()

    def test_mean_values_scalar(self):
        em = _ScalarGP()
        X = jnp.ones((5, 2))
        dist = em(X)
        m = dist.mean()
        np.testing.assert_allclose(m, jnp.zeros(5), atol=1e-6)

    def test_mean_values_multi_output(self):
        em = _MultiOutputEmulator()
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        dist = em(X)
        m = dist.mean()
        expected = jnp.array([[3.0, 6.0], [7.0, 14.0]])
        np.testing.assert_allclose(m, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: GaussianEmulator - joint modes
# ---------------------------------------------------------------------------


class TestGaussianEmulatorJoint:
    def test_joint_inputs_scalar(self):
        em = _ScalarGP()
        X = jnp.ones((10, 2))
        dist = em(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == ()
        assert tuple(dist.event_shape) == (10,)

    def test_joint_inputs_scalar_extra_batch(self):
        em = _ScalarGP()
        X = jnp.ones((3, 10, 2))
        dist = em(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (3,)
        assert tuple(dist.event_shape) == (10,)

    def test_joint_inputs_multi_output(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((10, 2))
        dist = em(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        # batch_shape = (*output_shape,) = (2,), event_shape = (n,) = (10,)
        assert tuple(dist.batch_shape) == (2,)
        assert tuple(dist.event_shape) == (10,)

    def test_joint_outputs(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((10, 2))
        dist = em(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (10,)
        assert tuple(dist.event_shape) == (2,)

    def test_joint_outputs_extra_batch(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((4, 10, 2))
        dist = em(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (4, 10)
        assert tuple(dist.event_shape) == (2,)

    def test_full_joint(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((10, 2))
        dist = em(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == ()
        assert tuple(dist.event_shape) == (20,)  # 10 * 2

    def test_full_joint_extra_batch(self):
        em = _MultiOutputEmulator()
        X = jnp.ones((3, 10, 2))
        dist = em(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (3,)
        assert tuple(dist.event_shape) == (20,)

    def test_joint_inputs_scalar_cov_shape(self, key):
        em = _ScalarGP()
        X = jnp.ones((5, 2))
        dist = em(X, joint_inputs=True)
        sample = dist.sample(key)
        assert sample.shape == (5,)

    def test_marginal_only_rejects_joint(self):
        em = _MarginalOnlyEmulator()
        X = jnp.ones((5, 3))
        with pytest.raises(ValueError, match="does not support joint_inputs"):
            em(X, joint_inputs=True)


# ---------------------------------------------------------------------------
# Tests: LinCombGaussianWeights
# ---------------------------------------------------------------------------


class _WeightEmulator(GaussianEmulator):
    """Mock weight emulator with d_w=3 outputs, supports joint_outputs."""

    supports_joint_inputs = True
    supports_joint_outputs = True

    def __init__(self):
        super().__init__(input_shape=(2,), output_shape=(3,))

    def predict_mean(self, X):
        extra_batch, n = self._parse_X(X)
        # Weights are [1, 2, 3] everywhere
        w = jnp.array([1.0, 2.0, 3.0])
        return jnp.broadcast_to(w, (*extra_batch, n, 3))

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        v = jnp.array([0.1, 0.2, 0.3])
        return jnp.broadcast_to(v, (*extra_batch, n, 3))

    def predict_covariance(self, X, *, joint_inputs=False, joint_outputs=False):
        extra_batch, n = self._parse_X(X)

        if not joint_inputs and joint_outputs:
            # Per-point weight covariance: (*eb, n, 3, 3)
            C = jnp.diag(jnp.array([0.1, 0.2, 0.3]))
            return jnp.broadcast_to(C, (*extra_batch, n, 3, 3))

        if joint_inputs and not joint_outputs:
            # Per-weight nxn: (*eb, 3, n, n)
            K = jnp.eye(n)
            return jnp.broadcast_to(
                K, (*extra_batch, 3, n, n)
            )

        if joint_inputs and joint_outputs:
            # Full: (*eb, 3n, 3n)
            K = jnp.kron(jnp.eye(n), jnp.diag(jnp.array([0.1, 0.2, 0.3])))
            return jnp.broadcast_to(K, (*extra_batch, 3 * n, 3 * n))

        raise ValueError("Use predict_variance.")


class _IndependentWeightEmulator(GaussianEmulator):
    """Weight emulator without joint_outputs support."""

    supports_joint_inputs = True
    supports_joint_outputs = False

    def __init__(self):
        super().__init__(input_shape=(2,), output_shape=(3,))

    def predict_mean(self, X):
        extra_batch, n = self._parse_X(X)
        w = jnp.array([1.0, 2.0, 3.0])
        return jnp.broadcast_to(w, (*extra_batch, n, 3))

    def predict_variance(self, X):
        extra_batch, n = self._parse_X(X)
        v = jnp.array([0.1, 0.2, 0.3])
        return jnp.broadcast_to(v, (*extra_batch, n, 3))

    def predict_covariance(self, X, *, joint_inputs=False, joint_outputs=False):
        extra_batch, n = self._parse_X(X)
        if joint_inputs and not joint_outputs:
            K = jnp.eye(n)
            return jnp.broadcast_to(K, (*extra_batch, 3, n, n))
        raise NotImplementedError


class TestLinCombGaussianWeights:
    @pytest.fixture
    def phi(self):
        # 2x3: maps 3 weights to 2 outputs
        return jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]])

    @pytest.fixture
    def bias(self):
        return jnp.array([10.0, 20.0])

    @pytest.fixture
    def lc(self, phi, bias):
        return LinCombGaussianWeights(_WeightEmulator(), phi, bias=bias)

    @pytest.fixture
    def lc_indep(self, phi, bias):
        return LinCombGaussianWeights(
            _IndependentWeightEmulator(), phi, bias=bias
        )

    def test_constructor_shapes(self, lc):
        assert lc.input_shape == (2,)
        assert lc.output_shape == (2,)

    def test_capabilities(self, lc):
        assert lc.supports_joint_inputs is True
        assert lc.supports_joint_outputs is True

    def test_inherits_joint_inputs(self):
        """LinComb inherits supports_joint_inputs from weight emulator."""
        # Use an emulator without joint_inputs
        we = _IndependentWeightEmulator()
        we.supports_joint_inputs = False
        phi = jnp.eye(3)
        lc = LinCombGaussianWeights(we, phi)
        assert lc.supports_joint_inputs is False

    def test_bad_weight_emulator_type(self):
        with pytest.raises(TypeError, match="must be a GaussianEmulator"):
            LinCombGaussianWeights("not an emulator", jnp.eye(3))

    def test_bad_weight_output_shape(self):
        em = _ScalarGP()  # output_shape = ()
        with pytest.raises(ValueError, match="must be 1-D"):
            LinCombGaussianWeights(em, jnp.eye(2))

    def test_bad_phi_ndim(self):
        we = _WeightEmulator()
        with pytest.raises(ValueError, match="must be 2-D"):
            LinCombGaussianWeights(we, jnp.ones((2, 3, 4)))

    def test_bad_phi_columns(self):
        we = _WeightEmulator()  # d_w=3
        with pytest.raises(ValueError, match="phi columns"):
            LinCombGaussianWeights(we, jnp.ones((2, 5)))

    def test_predict_mean(self, lc, phi, bias):
        X = jnp.ones((5, 2))
        mean = lc.predict_mean(X)
        # w_mean = [1, 2, 3], phi @ w = [1*1+0*2+1*3, 0*1+1*2+0.5*3]=[4, 3.5]
        # mean = bias + phi @ w = [14, 23.5]
        expected = bias + jnp.einsum("ow,w->o", phi, jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(mean[0], expected, atol=1e-5)
        assert mean.shape == (5, 2)

    def test_predict_variance(self, lc, phi):
        X = jnp.ones((5, 2))
        var = lc.predict_variance(X)
        # w_cov = diag([0.1, 0.2, 0.3])
        # output_cov = phi @ diag(w_var) @ phi^T
        w_var = jnp.array([0.1, 0.2, 0.3])
        output_cov = jnp.einsum("ow,w,pw->op", phi, w_var, phi)
        expected_var = jnp.diagonal(output_cov)
        np.testing.assert_allclose(var[0], expected_var, atol=1e-5)
        assert var.shape == (5, 2)

    def test_marginal_returns_normal(self, lc):
        X = jnp.ones((5, 2))
        dist = lc(X)
        assert isinstance(dist, Normal)
        assert tuple(dist.batch_shape) == (5, 2)
        assert tuple(dist.event_shape) == ()

    def test_joint_outputs(self, lc):
        X = jnp.ones((5, 2))
        dist = lc(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (5,)
        assert tuple(dist.event_shape) == (2,)

    def test_joint_inputs(self, lc):
        X = jnp.ones((5, 2))
        dist = lc(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (2,)
        assert tuple(dist.event_shape) == (5,)

    def test_full_joint(self, lc):
        X = jnp.ones((5, 2))
        dist = lc(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == ()
        assert tuple(dist.event_shape) == (10,)  # 5 * 2

    def test_independent_weights_variance(self, lc_indep, phi):
        """LinComb with independent weights uses diagonal approximation."""
        X = jnp.ones((5, 2))
        var = lc_indep.predict_variance(X)
        # Same result as with full covariance when weights are independent
        w_var = jnp.array([0.1, 0.2, 0.3])
        output_cov = jnp.einsum("ow,w,pw->op", phi, w_var, phi)
        expected_var = jnp.diagonal(output_cov)
        np.testing.assert_allclose(var[0], expected_var, atol=1e-5)

    def test_independent_weights_joint_outputs(self, lc_indep):
        X = jnp.ones((5, 2))
        dist = lc_indep(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (5,)
        assert tuple(dist.event_shape) == (2,)

    def test_extra_batch(self, lc):
        X = jnp.ones((3, 5, 2))
        dist = lc(X)
        assert tuple(dist.batch_shape) == (3, 5, 2)

    def test_extra_batch_joint(self, lc):
        X = jnp.ones((3, 5, 2))
        dist = lc(X, joint_inputs=True, joint_outputs=True)
        assert tuple(dist.batch_shape) == (3,)
        assert tuple(dist.event_shape) == (10,)


# ---------------------------------------------------------------------------
# Tests: LinearGaussianRegressor
# ---------------------------------------------------------------------------


class TestLinearGaussianRegressor:
    @pytest.fixture
    def weights(self):
        """Simple 3-D Gaussian weights."""
        m = jnp.array([1.0, 2.0, 3.0])
        C = jnp.array([
            [1.0, 0.1, 0.0],
            [0.1, 2.0, 0.2],
            [0.0, 0.2, 0.5],
        ])
        return MultivariateNormal(loc=m, cov=C)

    @pytest.fixture
    def scalar_regressor(self, weights):
        """Scalar-output linear regressor with polynomial features."""
        def feature_map(X):
            # X: (*eb, n, 1) -> features: (*eb, n, 3) = [1, x, x^2]
            x = X[..., 0]  # (*eb, n)
            return jnp.stack([jnp.ones_like(x), x, x ** 2], axis=-1)

        return LinearGaussianRegressor(
            feature_map=feature_map,
            weights=weights,
            input_shape=(1,),
            output_shape=(),
        )

    @pytest.fixture
    def multi_regressor(self, weights):
        """Multi-output linear regressor."""
        def feature_map(X):
            # X: (*eb, n, 1) -> (*eb, n, 2, 3)
            x = X[..., 0]  # (*eb, n)
            phi0 = jnp.stack([jnp.ones_like(x), x, x ** 2], axis=-1)
            phi1 = jnp.stack([x, x ** 2, x ** 3], axis=-1)
            return jnp.stack([phi0, phi1], axis=-2)  # (*eb, n, 2, 3)

        return LinearGaussianRegressor(
            feature_map=feature_map,
            weights=weights,
            input_shape=(1,),
            output_shape=(2,),
        )

    def test_constructor_scalar(self, scalar_regressor):
        assert scalar_regressor.input_shape == (1,)
        assert scalar_regressor.output_shape == ()
        assert scalar_regressor.supports_joint_inputs is True
        assert scalar_regressor.supports_joint_outputs is False

    def test_constructor_multi(self, multi_regressor):
        assert multi_regressor.input_shape == (1,)
        assert multi_regressor.output_shape == (2,)
        assert multi_regressor.supports_joint_inputs is True
        assert multi_regressor.supports_joint_outputs is True

    def test_bad_weights_type(self):
        with pytest.raises(TypeError, match="must be a MultivariateNormal"):
            LinearGaussianRegressor(
                feature_map=lambda x: x,
                weights="not a distribution",
                input_shape=(1,),
            )

    def test_scalar_mean(self, scalar_regressor, weights):
        X = jnp.array([[0.0], [1.0], [2.0]])
        mean = scalar_regressor.predict_mean(X)
        # phi(0) = [1,0,0], phi(1) = [1,1,1], phi(2) = [1,2,4]
        # mean = phi @ m where m = [1,2,3]
        expected = jnp.array([
            1 * 1 + 0 * 2 + 0 * 3,    # 1.0
            1 * 1 + 1 * 2 + 1 * 3,    # 6.0
            1 * 1 + 2 * 2 + 4 * 3,    # 17.0
        ])
        np.testing.assert_allclose(mean, expected, atol=1e-5)

    def test_scalar_variance(self, scalar_regressor, weights):
        X = jnp.array([[0.0], [1.0]])
        var = scalar_regressor.predict_variance(X)
        C = weights.cov
        # phi(0) = [1,0,0] -> var = phi C phi^T = C[0,0] = 1.0
        # phi(1) = [1,1,1] -> var = sum C = all elements sum
        phi0 = jnp.array([1.0, 0.0, 0.0])
        phi1 = jnp.array([1.0, 1.0, 1.0])
        expected = jnp.array([
            phi0 @ C @ phi0,
            phi1 @ C @ phi1,
        ])
        np.testing.assert_allclose(var, expected, atol=1e-5)

    def test_scalar_covariance_joint_inputs(self, scalar_regressor, weights):
        X = jnp.array([[0.0], [1.0]])
        cov = scalar_regressor.predict_covariance(X, joint_inputs=True)
        C = weights.cov
        phi0 = jnp.array([1.0, 0.0, 0.0])
        phi1 = jnp.array([1.0, 1.0, 1.0])
        Phi = jnp.stack([phi0, phi1])
        expected = Phi @ C @ Phi.T
        np.testing.assert_allclose(cov, expected, atol=1e-5)

    def test_scalar_marginal_returns_normal(self, scalar_regressor):
        X = jnp.ones((5, 1))
        dist = scalar_regressor(X)
        assert isinstance(dist, Normal)
        assert tuple(dist.batch_shape) == (5,)
        assert tuple(dist.event_shape) == ()

    def test_scalar_joint_inputs_returns_mvn(self, scalar_regressor):
        X = jnp.ones((5, 1))
        dist = scalar_regressor(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == ()
        assert tuple(dist.event_shape) == (5,)

    def test_multi_mean(self, multi_regressor, weights):
        X = jnp.array([[1.0]])
        mean = multi_regressor.predict_mean(X)
        # phi(1) output 0: [1,1,1], output 1: [1,1,1]
        # mean_0 = [1,1,1] @ [1,2,3] = 6.0
        # mean_1 = [1,1,1] @ [1,2,3] = 6.0
        m = weights.loc
        expected = jnp.array([[
            jnp.array([1.0, 1.0, 1.0]) @ m,
            jnp.array([1.0, 1.0, 1.0]) @ m,
        ]])
        np.testing.assert_allclose(mean, expected, atol=1e-5)

    def test_multi_marginal_shapes(self, multi_regressor):
        X = jnp.ones((5, 1))
        dist = multi_regressor(X)
        assert isinstance(dist, Normal)
        assert tuple(dist.batch_shape) == (5, 2)
        assert tuple(dist.event_shape) == ()

    def test_multi_joint_outputs(self, multi_regressor):
        X = jnp.ones((5, 1))
        dist = multi_regressor(X, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (5,)
        assert tuple(dist.event_shape) == (2,)

    def test_multi_joint_inputs(self, multi_regressor):
        X = jnp.ones((5, 1))
        dist = multi_regressor(X, joint_inputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == (2,)
        assert tuple(dist.event_shape) == (5,)

    def test_multi_full_joint(self, multi_regressor):
        X = jnp.ones((5, 1))
        dist = multi_regressor(X, joint_inputs=True, joint_outputs=True)
        assert isinstance(dist, MultivariateNormal)
        assert tuple(dist.batch_shape) == ()
        assert tuple(dist.event_shape) == (10,)

    def test_extra_batch_scalar(self, scalar_regressor):
        X = jnp.ones((3, 5, 1))
        dist = scalar_regressor(X)
        assert tuple(dist.batch_shape) == (3, 5)

    def test_extra_batch_joint(self, scalar_regressor):
        X = jnp.ones((3, 5, 1))
        dist = scalar_regressor(X, joint_inputs=True)
        assert tuple(dist.batch_shape) == (3,)
        assert tuple(dist.event_shape) == (5,)

    def test_with_bias(self, weights):
        def feature_map(X):
            x = X[..., 0]
            return jnp.stack([jnp.ones_like(x), x], axis=-1)

        # Use 2-D weights
        w2 = MultivariateNormal(loc=jnp.array([1.0, 2.0]), cov=jnp.eye(2))
        reg = LinearGaussianRegressor(
            feature_map=feature_map,
            weights=w2,
            input_shape=(1,),
            output_shape=(),
            bias=jnp.float32(5.0),
        )
        X = jnp.array([[0.0], [1.0]])
        mean = reg.predict_mean(X)
        # phi(0) = [1,0], phi(1) = [1,1]
        # mean = 5 + phi @ [1,2] = [5+1, 5+3] = [6, 8]
        np.testing.assert_allclose(mean, jnp.array([6.0, 8.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: sample_trajectory
# ---------------------------------------------------------------------------


class TestSampleTrajectory:
    def test_not_implemented_base(self):
        em = _NonGaussianEmulator()
        with pytest.raises(NotImplementedError):
            em.sample_trajectory(jax.random.PRNGKey(0), 5)

    def test_not_implemented_gaussian_base(self):
        em = _MarginalOnlyEmulator()
        with pytest.raises(NotImplementedError):
            em.sample_trajectory(jax.random.PRNGKey(0), 5)

    def test_linear_regressor_trajectory_shape_scalar(self, key):
        m = jnp.array([1.0, 2.0])
        w = MultivariateNormal(loc=m, cov=jnp.eye(2))

        def feature_map(X):
            x = X[..., 0]
            return jnp.stack([jnp.ones_like(x), x], axis=-1)

        reg = LinearGaussianRegressor(
            feature_map=feature_map, weights=w,
            input_shape=(1,), output_shape=(),
        )
        g = reg.sample_trajectory(key, n_trajectories=7)
        X = jnp.linspace(0, 1, 10).reshape(10, 1)
        result = g(X)
        assert result.shape == (7, 10)

    def test_linear_regressor_trajectory_shape_multi(self, key):
        m = jnp.array([1.0, 2.0, 3.0])
        w = MultivariateNormal(loc=m, cov=jnp.eye(3))

        def feature_map(X):
            x = X[..., 0]
            phi0 = jnp.stack([jnp.ones_like(x), x, x ** 2], axis=-1)
            phi1 = jnp.stack([x, x ** 2, x ** 3], axis=-1)
            return jnp.stack([phi0, phi1], axis=-2)

        reg = LinearGaussianRegressor(
            feature_map=feature_map, weights=w,
            input_shape=(1,), output_shape=(2,),
        )
        g = reg.sample_trajectory(key, n_trajectories=5)
        X = jnp.linspace(0, 1, 8).reshape(8, 1)
        result = g(X)
        assert result.shape == (5, 8, 2)

    def test_linear_regressor_trajectory_extra_batch(self, key):
        m = jnp.array([1.0, 2.0])
        w = MultivariateNormal(loc=m, cov=jnp.eye(2))

        def feature_map(X):
            x = X[..., 0]
            return jnp.stack([jnp.ones_like(x), x], axis=-1)

        reg = LinearGaussianRegressor(
            feature_map=feature_map, weights=w,
            input_shape=(1,), output_shape=(),
        )
        g = reg.sample_trajectory(key, n_trajectories=3)
        X = jnp.ones((4, 10, 1))
        result = g(X)
        assert result.shape == (3, 4, 10)

    def test_linear_regressor_trajectory_consistency(self, key):
        """Same trajectory callable at different X gives consistent results."""
        m = jnp.array([1.0, 2.0])
        w = MultivariateNormal(loc=m, cov=jnp.eye(2))

        def feature_map(X):
            x = X[..., 0]
            return jnp.stack([jnp.ones_like(x), x], axis=-1)

        reg = LinearGaussianRegressor(
            feature_map=feature_map, weights=w,
            input_shape=(1,), output_shape=(),
        )
        g = reg.sample_trajectory(key, n_trajectories=2)

        # Evaluate at x=0.5 as part of different arrays
        X1 = jnp.array([[0.0], [0.5], [1.0]])
        X2 = jnp.array([[0.5]])
        r1 = g(X1)  # (2, 3)
        r2 = g(X2)  # (2, 1)

        # The value at x=0.5 should be the same
        np.testing.assert_allclose(r1[:, 1], r2[:, 0], atol=1e-6)
