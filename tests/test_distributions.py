"""Tests for the refactored distribution classes (Phase 1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ArrayDistribution,
    ArrayEmpiricalDistribution,
    TFPDistribution,
    EmpiricalDistribution,
    Provenance,
    MultivariateNormal,
)
from probpipe import cov, from_distribution, log_prob, mean, prob, sample, variance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def dim():
    return 3


@pytest.fixture
def loc(dim):
    return jnp.arange(dim, dtype=jnp.float32)


@pytest.fixture
def cov_matrix(dim):
    A = jnp.eye(dim) * 2.0
    A = A.at[0, 1].set(0.3)
    A = A.at[1, 0].set(0.3)
    return A


@pytest.fixture
def gaussian(loc, cov_matrix):
    return MultivariateNormal(loc=loc, cov=cov_matrix, name="test_gaussian")


@pytest.fixture
def simple_samples():
    return jnp.array([[1.0], [2.0], [3.0]])


@pytest.fixture
def simple_weights():
    return jnp.array([0.2, 0.3, 0.5])


# ---------------------------------------------------------------------------
# MultivariateNormal
# ---------------------------------------------------------------------------


class TestMultivariateNormal:
    def test_construction_with_cov(self, loc, cov_matrix):
        g = MultivariateNormal(loc=loc, cov=cov_matrix, name="z")
        assert g.event_shape == (3,)
        assert g.batch_shape == ()
        assert g.dim == 3
        np.testing.assert_allclose(g.loc, loc, atol=1e-6)

    def test_construction_with_scale_tril(self, loc, cov_matrix):
        L = jnp.linalg.cholesky(cov_matrix)
        g = MultivariateNormal(loc=loc, scale_tril=L, name="z")
        np.testing.assert_allclose(g.cov, cov_matrix, atol=1e-5)

    def test_scalar_loc_promoted(self):
        g = MultivariateNormal(loc=1.0, scale_tril=jnp.eye(1), name="z")
        assert g.event_shape == (1,)
        assert g.dim == 1

    def test_rejects_both_cov_and_scale_tril(self, loc, cov_matrix):
        L = jnp.linalg.cholesky(cov_matrix)
        with pytest.raises(ValueError, match="exactly one"):
            MultivariateNormal(loc=loc, scale_tril=L, cov=cov_matrix, name="z")

    def test_rejects_neither_cov_nor_scale_tril(self, loc):
        with pytest.raises(ValueError, match="One of"):
            MultivariateNormal(loc=loc, name="z")

    def test_rejects_dim_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(3), name="z")

    def test_sample_shape(self, gaussian, key):
        s = sample(gaussian, key=key, sample_shape=(5,))
        assert s.shape == (5, 3)

    def test_sample_no_shape(self, gaussian, key):
        s = sample(gaussian, key=key)
        assert s.shape == (3,)

    def test_sample_statistics(self, gaussian, key):
        s = sample(gaussian, key=key, sample_shape=(5000,))
        np.testing.assert_allclose(s.mean(axis=0), gaussian.loc, atol=0.15)

    def test_log_prob_shape(self, gaussian, key):
        s = sample(gaussian, key=key, sample_shape=(7,))
        lp = log_prob(gaussian, s)
        assert lp.shape == (7,)

    def test_log_prob_single(self, gaussian, loc):
        lp = log_prob(gaussian, loc)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_log_prob_matches_scipy(self, gaussian, loc, cov_matrix, key):
        """log_prob and prob must match scipy.stats.multivariate_normal."""
        import scipy.stats
        scipy_mvn = scipy.stats.multivariate_normal(
            mean=np.asarray(loc), cov=np.asarray(cov_matrix)
        )
        s = sample(gaussian, key=key, sample_shape=(5,))
        s_np = np.asarray(s)
        np.testing.assert_allclose(
            log_prob(gaussian, s), scipy_mvn.logpdf(s_np), rtol=1e-5
        )
        np.testing.assert_allclose(
            prob(gaussian, s), scipy_mvn.pdf(s_np), rtol=1e-5
        )

    def test_mean_and_cov(self, gaussian, loc, cov_matrix, key):
        """Sample mean and cov from 50k draws must match analytical values."""
        import scipy.stats
        draws = np.asarray(sample(gaussian, key=key, sample_shape=(50_000,)))
        np.testing.assert_allclose(draws.mean(0), np.asarray(loc), atol=0.02)
        np.testing.assert_allclose(
            np.cov(draws, rowvar=False), np.asarray(cov_matrix), atol=0.05,
        )

    def test_marginal_ks(self, gaussian, loc, cov_matrix, key):
        """Each MVN marginal X_i ~ N(loc_i, cov_ii): KS test on 50k samples."""
        import scipy.stats
        draws = np.asarray(sample(gaussian, key=key, sample_shape=(50_000,)))
        for i in range(draws.shape[1]):
            marginal = scipy.stats.norm(
                loc=float(loc[i]), scale=float(jnp.sqrt(cov_matrix[i, i]))
            )
            _, p = scipy.stats.kstest(draws[:, i], marginal.cdf)
            assert p > 0.001, f"KS failed for marginal {i}: p={p:.4e}"

    def test_cov_property(self, gaussian, cov_matrix):
        np.testing.assert_allclose(gaussian.cov, cov_matrix, atol=1e-5)

    def test_name(self, gaussian):
        assert gaussian.name == "test_gaussian"

    def test_name_set(self, loc, cov_matrix):
        g = MultivariateNormal(loc=loc, cov=cov_matrix, name="z")
        assert g.name == "z"

    def test_repr(self, gaussian):
        r = repr(gaussian)
        assert "MultivariateNormal" in r
        assert "test_gaussian" in r
        assert "event_shape=(3,)" in r

    def test_dtype(self, gaussian):
        assert gaussian.dtype == jnp.float32

    def test_from_distribution_empirical(self, gaussian, key):
        ed = from_distribution(
            gaussian, ArrayEmpiricalDistribution, key=key, num_samples=2000
        )
        g2 = from_distribution(ed, MultivariateNormal, name="fitted")
        np.testing.assert_allclose(g2.loc, gaussian.loc, atol=0.2)
        assert g2.name == "fitted"
        assert g2.source is not None
        assert g2.source.operation == "from_distribution"

    def test_from_distribution_gaussian(self, gaussian, key):
        """Moment-match from another MultivariateNormal via sampling."""
        g2 = from_distribution(gaussian, MultivariateNormal, key=key, num_samples=5000)
        np.testing.assert_allclose(g2.loc, gaussian.loc, atol=0.15)


# ---------------------------------------------------------------------------
# EmpiricalDistribution
# ---------------------------------------------------------------------------


class TestEmpiricalDistribution:
    def test_uniform_weights(self, simple_samples):
        ed = ArrayEmpiricalDistribution(simple_samples)
        assert ed.n == 3
        assert ed.dim == 1
        np.testing.assert_allclose(ed.weights, jnp.ones(3) / 3)

    def test_custom_weights(self, simple_samples, simple_weights):
        ed = ArrayEmpiricalDistribution(simple_samples, simple_weights)
        np.testing.assert_allclose(ed.weights.sum(), 1.0)
        expected_mean = jnp.sum(simple_samples.ravel() * ed.weights)
        np.testing.assert_allclose(mean(ed).ravel(), expected_mean, atol=1e-6)

    def test_weights_normalized(self):
        """Unnormalized weights should be normalized internally."""
        samples = jnp.array([[1.0], [2.0]])
        ed = EmpiricalDistribution(samples, jnp.array([2.0, 8.0]))
        np.testing.assert_allclose(ed.weights, jnp.array([0.2, 0.8]))

    def test_invalid_weights_negative(self, simple_samples):
        with pytest.raises(ValueError, match="non-negative"):
            EmpiricalDistribution(simple_samples, jnp.array([-0.1, 0.5, 0.6]))

    def test_invalid_weights_zero_sum(self, simple_samples):
        with pytest.raises(ValueError, match="positive value"):
            EmpiricalDistribution(simple_samples, jnp.array([0.0, 0.0, 0.0]))

    def test_invalid_weights_wrong_length(self, simple_samples):
        with pytest.raises(ValueError, match="does not match"):
            EmpiricalDistribution(simple_samples, jnp.array([0.5, 0.5]))

    def test_1d_input_scalar_event(self):
        ed = ArrayEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]))
        assert ed.event_shape == ()
        assert ed.samples.shape == (3,)

    def test_multidim_event_shape(self):
        samples = jnp.ones((5, 3))
        ed = ArrayEmpiricalDistribution(samples)
        assert ed.event_shape == (3,)
        assert ed.dim == 3

    def test_event_shape(self, simple_samples):
        ed = ArrayEmpiricalDistribution(simple_samples)
        assert ed.event_shape == (1,)

    def test_sample_shape(self, simple_samples, key):
        ed = EmpiricalDistribution(simple_samples)
        s = sample(ed, key=key, sample_shape=(10,))
        assert s.shape == (10, 1)

    def test_sample_no_shape(self, simple_samples, key):
        ed = EmpiricalDistribution(simple_samples)
        s = sample(ed, key=key)
        assert s.shape == (1,)

    def test_sample_values_from_support(self, simple_samples, key):
        ed = EmpiricalDistribution(simple_samples)
        s = sample(ed, key=key, sample_shape=(100,))
        for val in s:
            assert jnp.any(jnp.all(jnp.isclose(simple_samples, val), axis=-1))

    def test_mean(self, simple_samples, simple_weights):
        ed = ArrayEmpiricalDistribution(simple_samples, simple_weights)
        expected = jnp.sum(simple_samples.ravel() * ed.weights)
        np.testing.assert_allclose(mean(ed).ravel(), expected, atol=1e-6)

    def test_variance(self, simple_samples, simple_weights):
        ed = ArrayEmpiricalDistribution(simple_samples, simple_weights)
        mu = mean(ed)
        expected = jnp.sum(ed.weights * (simple_samples.ravel() - mu.ravel()) ** 2)
        np.testing.assert_allclose(variance(ed).ravel(), expected, atol=1e-6)

    def test_cov_matrix(self):
        samples = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ed = ArrayEmpiricalDistribution(samples)
        C = cov(ed)
        assert C.shape == (2, 2)
        np.testing.assert_allclose(C, C.T, atol=1e-6)

    def test_cov_psd(self):
        """Covariance matrix should be positive semi-definite."""
        samples = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        ed = ArrayEmpiricalDistribution(samples)
        C = cov(ed)
        eigvals = jnp.linalg.eigvalsh(C)
        assert jnp.all(eigvals >= -1e-8)

    def test_name(self, simple_samples):
        ed = EmpiricalDistribution(simple_samples, name="emp")
        assert ed.name == "emp"

    def test_from_distribution(self, gaussian, key):
        ed = from_distribution(
            gaussian, ArrayEmpiricalDistribution, key=key, num_samples=50
        )
        assert ed.n == 50
        assert ed.event_shape == gaussian.event_shape
        assert ed.source is not None
        assert ed.source.operation == "from_distribution"
        assert ed.name == gaussian.name

    def test_from_distribution_custom_name(self, gaussian, key):
        ed = from_distribution(
            gaussian, ArrayEmpiricalDistribution, key=key, num_samples=10, name="custom"
        )
        assert ed.name == "custom"

    def test_from_distribution_default_key(self, gaussian):
        """from_distribution should work without explicit key."""
        ed = from_distribution(gaussian, ArrayEmpiricalDistribution, num_samples=10)
        assert ed.n == 10


class TestEmpiricalLogWeights:
    """Tests for log_weights parameterisation and uniform optimisation."""

    def test_log_weights_construction(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        lw = jnp.log(jnp.array([0.2, 0.3, 0.5]))
        ed = EmpiricalDistribution(samples, log_weights=lw)
        assert not ed.is_uniform
        np.testing.assert_allclose(ed.weights, jnp.array([0.2, 0.3, 0.5]), atol=1e-5)

    def test_log_weights_unnormalized(self):
        """Unnormalised log-weights should produce correct normalised weights."""
        samples = jnp.array([[1.0], [2.0]])
        # log(2) and log(8) → weights 0.2 and 0.8
        lw = jnp.array([jnp.log(2.0), jnp.log(8.0)])
        ed = EmpiricalDistribution(samples, log_weights=lw)
        np.testing.assert_allclose(ed.weights, jnp.array([0.2, 0.8]), atol=1e-5)

    def test_log_weights_property_normalized(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        lw = jnp.array([1.0, 2.0, 3.0])  # unnormalized
        ed = EmpiricalDistribution(samples, log_weights=lw)
        # Normalised log-weights should sum to 0 in exp-space
        np.testing.assert_allclose(jnp.exp(ed.log_weights).sum(), 1.0, atol=1e-5)

    def test_both_weights_and_log_weights_raises(self):
        samples = jnp.array([[1.0], [2.0]])
        with pytest.raises(ValueError, match="not both"):
            EmpiricalDistribution(
                samples,
                weights=jnp.array([0.5, 0.5]),
                log_weights=jnp.array([0.0, 0.0]),
            )

    def test_log_weights_wrong_length_raises(self):
        samples = jnp.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="does not match"):
            EmpiricalDistribution(samples, log_weights=jnp.array([0.0, 0.0]))

    def test_uniform_flag(self, simple_samples):
        ed = EmpiricalDistribution(simple_samples)
        assert ed.is_uniform
        np.testing.assert_allclose(ed.weights, jnp.ones(3) / 3)
        np.testing.assert_allclose(ed.log_weights, jnp.full(3, -jnp.log(3.0)), atol=1e-6)

    def test_weighted_not_uniform(self, simple_samples, simple_weights):
        ed = EmpiricalDistribution(simple_samples, simple_weights)
        assert not ed.is_uniform

    def test_uniform_mean_matches_numpy(self):
        samples = jnp.array([[1.0], [3.0], [5.0]])
        ed = ArrayEmpiricalDistribution(samples)
        np.testing.assert_allclose(mean(ed), jnp.array([3.0]), atol=1e-6)

    def test_uniform_variance_matches_numpy(self):
        samples = jnp.array([[1.0], [3.0], [5.0]])
        ed = ArrayEmpiricalDistribution(samples)
        expected_var = jnp.mean((samples - jnp.array([[3.0]])) ** 2, axis=0)
        np.testing.assert_allclose(variance(ed), expected_var, atol=1e-6)

    def test_uniform_sampling(self, key):
        samples = jnp.array([[10.0], [20.0], [30.0]])
        ed = EmpiricalDistribution(samples)
        draws = sample(ed, key=key, sample_shape=(1000,))
        # All draws should be from the support
        for val in [10.0, 20.0, 30.0]:
            assert jnp.any(jnp.isclose(draws, val))

    def test_log_weights_numerical_stability(self):
        """Very large log-weights should not overflow."""
        samples = jnp.array([[1.0], [2.0], [3.0]])
        lw = jnp.array([1000.0, 1001.0, 1000.5])
        ed = EmpiricalDistribution(samples, log_weights=lw)
        # Should produce valid weights that sum to 1
        assert jnp.all(jnp.isfinite(ed.weights))
        np.testing.assert_allclose(ed.weights.sum(), 1.0, atol=1e-5)

    def test_weights_backward_compatible(self, simple_samples, simple_weights):
        """Existing weights= API should work unchanged."""
        ed = EmpiricalDistribution(simple_samples, simple_weights)
        expected = simple_weights / simple_weights.sum()
        np.testing.assert_allclose(ed.weights, expected, atol=1e-5)

    def test_log_weights_mean(self):
        """Mean with log_weights should match weights-based mean."""
        samples = jnp.array([[1.0], [2.0], [3.0]])
        weights = jnp.array([0.2, 0.3, 0.5])
        ed_w = ArrayEmpiricalDistribution(samples, weights)
        ed_lw = ArrayEmpiricalDistribution(samples, log_weights=jnp.log(weights))
        np.testing.assert_allclose(mean(ed_w), mean(ed_lw), atol=1e-5)

    def test_uniform_cov(self):
        """Uniform cov should match standard formula."""
        samples = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ed = ArrayEmpiricalDistribution(samples)
        mu = jnp.mean(samples, axis=0)
        diff = samples - mu
        expected = diff.T @ diff / 3
        np.testing.assert_allclose(cov(ed), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_creation(self, gaussian):
        p = Provenance("test_op", parents=(gaussian,), metadata={"key": "val"})
        assert p.operation == "test_op"
        assert p.parents == (gaussian,)
        assert p.metadata == {"key": "val"}

    def test_default_parents_and_metadata(self):
        p = Provenance("op")
        assert p.parents == ()
        assert p.metadata == {}

    def test_repr(self, gaussian):
        p = Provenance("test_op", parents=(gaussian,))
        r = repr(p)
        assert "test_op" in r
        assert "test_gaussian" in r

    def test_repr_unnamed_parent(self, loc, cov_matrix):
        g = MultivariateNormal(loc=loc, cov=cov_matrix, name="z")
        p = Provenance("op", parents=(g,))
        r = repr(p)
        assert "z" in r

    def test_frozen(self, gaussian):
        p = Provenance("test_op")
        with pytest.raises(AttributeError):
            p.operation = "changed"


# ---------------------------------------------------------------------------
# ArrayDistribution ABC
# ---------------------------------------------------------------------------


class TestDistributionABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ArrayDistribution()

    def test_no_condition_on_method(self, gaussian):
        """condition_on() is only on JointDistribution, not ArrayDistribution ABC."""
        assert not hasattr(gaussian, "condition_on")

    def test_default_batch_shape(self, gaussian):
        assert gaussian.batch_shape == ()

    def test_default_dtype(self, gaussian):
        assert gaussian.dtype == jnp.float32

    def test_prob_delegates_to_log_prob(self, gaussian, key):
        x = sample(gaussian, key=key, sample_shape=(3,))
        np.testing.assert_allclose(
            prob(gaussian, x),
            jnp.exp(log_prob(gaussian, x)),
            rtol=1e-5,
        )

    def test_mean_requires_supports_mean(self):
        """mean op raises TypeError for distributions without SupportsMean."""
        from probpipe.core.protocols import SupportsSampling, SupportsExpectation
        from probpipe.core.distribution import _vmap_sample, _mc_expectation

        class MinimalDist(ArrayDistribution, SupportsSampling, SupportsExpectation):
            _sampling_cost = "low"
            _preferred_orchestration = None
            @property
            def event_shape(self):
                return (1,)
            def _sample_one(self, key):
                return jnp.zeros((1,))
            def _sample(self, key, sample_shape=()):
                return _vmap_sample(self, key, sample_shape)
            def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
                return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)
        d = MinimalDist(name="minimal")
        with pytest.raises(TypeError, match="does not support mean"):
            mean(d)

    def test_variance_requires_supports_variance(self):
        """variance op raises TypeError for distributions without SupportsVariance."""
        from probpipe.core.protocols import SupportsSampling, SupportsExpectation
        from probpipe.core.distribution import _vmap_sample, _mc_expectation

        class MinimalDist(ArrayDistribution, SupportsSampling, SupportsExpectation):
            _sampling_cost = "low"
            _preferred_orchestration = None
            @property
            def event_shape(self):
                return (1,)
            def _sample_one(self, key):
                return jnp.zeros((1,))
            def _sample(self, key, sample_shape=()):
                return _vmap_sample(self, key, sample_shape)
            def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
                return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)
        d = MinimalDist(name="minimal")
        with pytest.raises(TypeError, match="does not support variance"):
            variance(d)

    def test_from_distribution_raises_for_invalid_input(self):
        with pytest.raises(TypeError):
            from_distribution(None, ArrayDistribution)

    def test_source_default_none(self, gaussian):
        g = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")
        assert g.source is None

    def test_with_source(self, gaussian):
        p = Provenance("test")
        gaussian.with_source(p)
        assert gaussian.source is p

    def test_with_source_write_once(self, gaussian):
        p1 = Provenance("first")
        gaussian.with_source(p1)
        p2 = Provenance("second")
        with pytest.raises(RuntimeError, match="Source already set"):
            gaussian.with_source(p2)

    def test_name(self, loc, cov_matrix):
        g = MultivariateNormal(loc=loc, cov=cov_matrix, name="z")
        assert g.name == "z"


# ---------------------------------------------------------------------------
# TFPDistribution mixin
# ---------------------------------------------------------------------------


class TestTFPDistribution:
    def test_gaussian_is_tfp_distribution(self, gaussian):
        assert isinstance(gaussian, TFPDistribution)
        assert isinstance(gaussian, ArrayDistribution)

    def test_dtype(self, gaussian):
        assert gaussian.dtype == jnp.float32

    def test_mean_delegates(self, gaussian, loc):
        np.testing.assert_allclose(mean(gaussian), loc, atol=1e-6)

    def test_variance_delegates(self, gaussian, cov_matrix):
        np.testing.assert_allclose(
            variance(gaussian), jnp.diag(cov_matrix), atol=1e-5
        )


# ---------------------------------------------------------------------------
# Shape semantics
# ---------------------------------------------------------------------------


class TestShapeSemantics:
    def test_sample_shape_convention(self, gaussian, key):
        """sample(key, sample_shape) → sample_shape + batch_shape + event_shape"""
        s = sample(gaussian, key=key, sample_shape=(4, 2))
        assert s.shape == (4, 2, 3)

    def test_log_prob_batch_shape(self, gaussian, key):
        s = sample(gaussian, key=key, sample_shape=(4, 2))
        lp = log_prob(gaussian, s)
        assert lp.shape == (4, 2)

    def test_empirical_sample_shape(self, simple_samples, key):
        ed = EmpiricalDistribution(simple_samples)
        s = sample(ed, key=key, sample_shape=(5, 3))
        assert s.shape == (5, 3, 1)

    def test_empirical_2d_sample_shape(self, key):
        samples = jnp.ones((4, 3))
        ed = EmpiricalDistribution(samples)
        s = sample(ed, key=key, sample_shape=(10,))
        assert s.shape == (10, 3)


# ---------------------------------------------------------------------------
# ArrayDistribution / TFPDistribution coverage gaps
# ---------------------------------------------------------------------------


class TestDistributionCoverageGaps:
    """Cover otherwise-uncovered defaults and helpers in core.distribution."""

    def test_batch_shape_default(self):
        """ArrayDistribution.batch_shape defaults to ()."""
        class Scalar(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        assert Scalar(name="s").batch_shape == ()

    def test_dtype_none_without_template(self):
        """NumericRecordDistribution.dtype is None when no template is set."""
        class Scalar(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        assert Scalar(name="s").dtype is None

    def test_dtype_uniform_with_template(self):
        """NumericRecordDistribution.dtype is the common dtype when all fields match."""
        from probpipe import Normal
        n = Normal(loc=0.0, scale=1.0, name="x")
        assert n.dtype == n._tfp_dist.dtype

    def test_repr_with_batch_shape(self):
        """TFPDistribution repr includes batch_shape when non-trivial."""
        from probpipe import Normal

        d = Normal(loc=jnp.array([0.0, 1.0]), scale=jnp.array([1.0, 1.0]), name="x")
        assert "batch_shape" in repr(d)

    def test_array_empirical_dtype(self):
        """ArrayEmpiricalDistribution.dtype returns sample dtype."""
        samples = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        assert ArrayEmpiricalDistribution(samples).dtype == jnp.float32
