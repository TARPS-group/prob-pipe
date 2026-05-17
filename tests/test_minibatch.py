"""Tests for :class:`MinibatchedDistribution`.

A ``RandomMeasure[Record]`` whose draws are unbiased stochastic
surrogates of the full-data unnormalized log-posterior. Consumed by
stochastic-gradient MCMC kernels (Phase 3) and by tempered SMC
(Phase 5).
"""

from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    DistributionArray,
    GLMLikelihood,
    MultivariateNormal,
    Record,
    SimpleModel,
    log_prob,
    random_unnormalized_log_prob,
    sample,
)
from probpipe.core._random_functions import RandomFunction
from probpipe.core._random_measures import RandomMeasure
from probpipe.core.protocols import (
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
)
from probpipe.inference._minibatch import (
    MinibatchedDistribution,
    _FixedMinibatchDistribution,
    _MinibatchLogProbAtPoint,
    _RandomMinibatchLogProb,
)


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """200-observation logistic regression with 2D coefficients."""
    N, P = 200, 2
    X = jax.random.normal(jax.random.PRNGKey(0), (N, P))
    y = ((X @ jnp.array([1.0, -0.5])) > 0).astype(jnp.float32)
    return X, y


@pytest.fixture
def model(regression_data):
    X, _ = regression_data
    prior = MultivariateNormal(
        loc=jnp.zeros(2), cov=jnp.eye(2), name="theta",
    )
    likelihood = GLMLikelihood(tfp_glm.Bernoulli(), x=X)
    return SimpleModel(prior=prior, likelihood=likelihood)


@pytest.fixture
def data_record(regression_data):
    X, y = regression_data
    return Record(X=X, y=y)


@pytest.fixture
def measure(model, data_record):
    return MinibatchedDistribution(model, data_record, batch_size=40)


# -- Construction --------------------------------------------------------------


class TestConstruction:
    def test_construction_from_simple_model(self, model, data_record):
        m = MinibatchedDistribution(model, data_record, batch_size=32)
        assert isinstance(m, MinibatchedDistribution)
        assert m.dataset_size == 200
        assert m.batch_size == 32

    def test_construction_explicit_callable(self, model, data_record):
        """The bare ``SupportsLogProb`` + explicit callable path."""
        m = MinibatchedDistribution(
            model, data_record, batch_size=32,
            per_datum_log_likelihood=model.likelihood.per_datum_log_likelihood,
        )
        assert isinstance(m, MinibatchedDistribution)

    def test_construction_rejects_non_iid_likelihood(self, data_record):
        """A bare ``Likelihood`` (no ``per_datum``) is rejected."""
        class _BareLikelihood:
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta")
        sm = SimpleModel(prior=prior, likelihood=_BareLikelihood())
        with pytest.raises(TypeError, match="ConditionallyIndependentLikelihood"):
            MinibatchedDistribution(sm, data_record, batch_size=32)

    def test_construction_rejects_neither_path(self, data_record):
        """No SimpleModel and no explicit callable → TypeError."""
        class _BarePrior:
            pass

        with pytest.raises(TypeError, match="MinibatchedDistribution requires"):
            MinibatchedDistribution(_BarePrior(), data_record, batch_size=32)

    def test_construction_validates_batch_size_too_small(self, model, data_record):
        with pytest.raises(ValueError, match="batch_size must be in"):
            MinibatchedDistribution(model, data_record, batch_size=0)

    def test_construction_validates_batch_size_too_large(self, model, data_record):
        with pytest.raises(ValueError, match="batch_size must be in"):
            MinibatchedDistribution(model, data_record, batch_size=999)

    def test_construction_rejects_nested_records(self, model):
        """Nested Records fail at construction with an actionable error."""
        nested = Record(
            features=Record(X=jnp.zeros((200, 2)), extra=jnp.zeros((200,))),
            y=jnp.zeros((200,)),
        )
        with pytest.raises(ValueError, match="flat Record"):
            MinibatchedDistribution(model, nested, batch_size=32)


# -- Property accessors -------------------------------------------------------


class TestAccessors:
    """The convenience properties exposed for inspection / debugging."""

    def test_properties_match_constructor_args(self, model, data_record):
        m = MinibatchedDistribution(
            model, data_record, batch_size=25,
            with_replacement=True, rescale=False,
            name="custom_name",
        )
        assert m.dataset_size == 200
        assert m.batch_size == 25
        assert m.with_replacement is True
        assert m.rescale is False
        assert m.model is model
        assert m.data is data_record
        assert m.name == "custom_name"

    def test_default_name_includes_batch_size(self, measure):
        assert "batch_size=40" in measure.name


# -- Protocol membership -------------------------------------------------------


class TestProtocols:
    def test_isinstance_random_measure(self, measure):
        assert isinstance(measure, RandomMeasure)

    def test_isinstance_supports_sampling(self, measure):
        assert isinstance(measure, SupportsSampling)

    def test_isinstance_supports_random_unnormalized_log_prob(self, measure):
        assert isinstance(measure, SupportsRandomUnnormalizedLogProb)

    def test_not_iterable(self, measure):
        """STYLE_GUIDE §1.11 — Distribution subclasses are non-iterable."""
        assert not isinstance(measure, Iterable)


# -- Sampling ------------------------------------------------------------------


class TestSampling:
    def test_sample_returns_inner_distribution(self, measure):
        inner = sample(measure, key=jax.random.PRNGKey(0))
        # WorkflowFunction wraps the inner Distribution in a single-field
        # NumericRecord — the .fields layer is harmless; just unwrap.
        # Easier: call _sample directly to check type.
        inner_raw = measure._sample(jax.random.PRNGKey(0))
        assert isinstance(inner_raw, _FixedMinibatchDistribution)
        assert isinstance(inner_raw, SupportsUnnormalizedLogProb)

    def test_sample_batched_returns_distribution_array(self, measure):
        draws = measure._sample(jax.random.PRNGKey(0), sample_shape=(5,))
        assert isinstance(draws, DistributionArray)
        assert draws.batch_shape == (5,)

    def test_inner_log_prob_factorises(self, measure, regression_data):
        """For a fixed minibatch, log~D_B(theta) = log_prior(theta) + (N/b)*sum_batch."""
        X, y = regression_data
        inner = measure._sample(jax.random.PRNGKey(7))
        theta = jnp.array([0.1, -0.2])

        # Hand-compute the expected value from the captured batch.
        batch = inner.batch
        prior_lp = measure._log_prior_fn(theta)
        per_datum = jax.vmap(measure._per_datum_log_lkl_fn, in_axes=(None, 0))(theta, batch)
        expected = prior_lp + measure._rescale_factor * jnp.sum(per_datum)

        actual = inner._unnormalized_log_prob(theta)
        np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5)

    def test_with_replacement_flag(self, model, data_record):
        """``with_replacement=True`` allows repeated indices."""
        m_wr = MinibatchedDistribution(
            model, data_record, batch_size=5,
            with_replacement=True,
        )
        # Stress test: with batch_size=5 and replacement, over many draws
        # we should see at least one repeat (probability ~ 1 for many draws).
        repeats_seen = 0
        for k in jax.random.split(jax.random.PRNGKey(0), 50):
            inner = m_wr._sample(k)
            batch_X = inner.batch["X"]
            # Check for duplicate rows by comparing rounded hashes.
            unique = jnp.unique(batch_X, axis=0)
            if unique.shape[0] < batch_X.shape[0]:
                repeats_seen += 1
        assert repeats_seen > 0, (
            "Expected at least one minibatch with repeats under with_replacement=True"
        )

    def test_rescale_false_skips_n_over_b_factor(self, model, data_record):
        """``rescale=False`` produces the raw per-datum sum (no N/b factor)."""
        m_unrescaled = MinibatchedDistribution(
            model, data_record, batch_size=20, rescale=False,
        )
        inner = m_unrescaled._sample(jax.random.PRNGKey(3))
        assert inner.rescale_factor == 1.0

        # The unnormalized log-density is just prior + sum over batch.
        theta = jnp.array([0.1, -0.1])
        batch = inner.batch
        prior_lp = model.prior._log_prob(theta)
        per_datum = jax.vmap(
            model.likelihood.per_datum_log_likelihood, in_axes=(None, 0),
        )(theta, batch)
        expected = prior_lp + jnp.sum(per_datum)  # no N/b multiplier

        actual = inner._unnormalized_log_prob(theta)
        np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5)

    def test_batch_size_equals_dataset_size_matches_full(
        self, model, data_record, regression_data,
    ):
        """``batch_size == N`` is the degenerate full-batch case.

        Without replacement, this picks a permutation of all observations,
        and the rescale factor is 1.0, so the surrogate exactly equals
        the full-data unnormalized log-density (up to FP).
        """
        X, _ = regression_data
        N = X.shape[0]
        m_full = MinibatchedDistribution(
            model, data_record, batch_size=N, with_replacement=False,
        )
        inner = m_full._sample(jax.random.PRNGKey(11))
        assert inner.rescale_factor == 1.0

        theta = jnp.array([0.2, -0.3])
        # Full-data log-density
        per_datum = jax.vmap(
            model.likelihood.per_datum_log_likelihood, in_axes=(None, 0),
        )(theta, data_record)
        full = model.prior._log_prob(theta) + jnp.sum(per_datum)

        actual = inner._unnormalized_log_prob(theta)
        np.testing.assert_allclose(float(actual), float(full), rtol=1e-5)


# -- Mathematical correctness --------------------------------------------------


class TestMathematicalCorrectness:
    """Unbiasedness checks (parent plan §3.4)."""

    def test_unbiased_log_density(self, measure, model, data_record):
        """Average of random log-densities at fixed theta ≈ full-data log-density."""
        theta = jnp.array([0.3, -0.2])
        # Full reference:
        per_datum = jax.vmap(
            model.likelihood.per_datum_log_likelihood, in_axes=(None, 0),
        )(theta, data_record)
        full_lp = float(model.prior._log_prob(theta) + jnp.sum(per_datum))

        # MC estimate over 500 minibatches.
        rf = measure._random_unnormalized_log_prob()
        keys = jax.random.split(jax.random.PRNGKey(1), 500)
        vals = jnp.array([rf._sample(k)(theta) for k in keys])
        mc_mean = float(jnp.mean(vals))

        # With batch_size=40 from N=200, the MC SE is moderate. atol=1 is
        # generous but catches off-by-rescale-factor bugs (which would be ~N).
        np.testing.assert_allclose(mc_mean, full_lp, atol=1.0)

    def test_unbiased_gradient(self, measure, model, data_record):
        """Average gradient over minibatches ≈ full-data gradient."""
        theta = jnp.array([0.3, -0.2])

        def full_log_density(t):
            per_datum = jax.vmap(
                model.likelihood.per_datum_log_likelihood, in_axes=(None, 0),
            )(t, data_record)
            return model.prior._log_prob(t) + jnp.sum(per_datum)

        full_grad = np.asarray(jax.grad(full_log_density)(theta))

        rf = measure._random_unnormalized_log_prob()
        keys = jax.random.split(jax.random.PRNGKey(2), 500)
        grads = [
            np.asarray(jax.grad(lambda t: rf._sample(k)(t))(theta))
            for k in keys
        ]
        mc_grad = np.mean(grads, axis=0)

        # Coordinate-wise tolerance — 500 samples gives ~SE 0.5 here.
        np.testing.assert_allclose(mc_grad, full_grad, atol=2.0)


# -- random_unnormalized_log_prob op ------------------------------------------


class TestRandomLogProbOp:
    def test_zero_arg_form_returns_random_function(self, measure):
        rf = random_unnormalized_log_prob(measure)
        assert isinstance(rf, RandomFunction)
        assert isinstance(rf, _RandomMinibatchLogProb)

    def test_random_function_sample_returns_callable(self, measure):
        rf = measure._random_unnormalized_log_prob()
        callable_at_k = rf._sample(jax.random.PRNGKey(0))
        assert callable(callable_at_k)
        # Calling with theta gives a scalar log-density
        theta = jnp.zeros(2)
        result = callable_at_k(theta)
        assert jnp.asarray(result).shape == ()

    def test_random_function_batched_sample_not_supported(self, measure):
        """Batched ``_sample`` of a RandomFunction isn't supported (returns
        a structure of functions, awkward to type)."""
        rf = measure._random_unnormalized_log_prob()
        with pytest.raises(NotImplementedError, match="sample_shape"):
            rf._sample(jax.random.PRNGKey(0), sample_shape=(3,))

    def test_two_arg_form_returns_distribution_at_theta(self, measure):
        """``random_unnormalized_log_prob(measure, theta)`` returns a
        ``Distribution[Array]`` over log-density estimates at theta.
        """
        theta = jnp.array([0.0, 0.0])
        dist_at_theta = random_unnormalized_log_prob(measure, theta)
        assert isinstance(dist_at_theta, _MinibatchLogProbAtPoint)
        # The distribution is sampleable
        val = dist_at_theta._sample(jax.random.PRNGKey(0))
        assert jnp.asarray(val).shape == ()


# -- JIT traceability ----------------------------------------------------------


class TestJITTraceability:
    """SGMCMC kernels need to JIT-trace through the inner log-density
    callable. This regression test confirms the callable compiles
    under ``jax.jit``.
    """

    def test_log_density_jits(self, measure):
        rf = measure._random_unnormalized_log_prob()
        key = jax.random.PRNGKey(0)
        target_fn = rf._sample(key)

        @jax.jit
        def step(theta):
            return jax.grad(target_fn)(theta)

        theta = jnp.array([0.1, -0.1])
        grad = step(theta)
        assert grad.shape == (2,)
        # Re-call to confirm no retracing failure
        grad2 = step(theta + 0.01)
        assert grad2.shape == (2,)
