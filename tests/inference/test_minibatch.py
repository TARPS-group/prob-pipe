"""Tests for :class:`MinibatchedDistribution`.

A ``RandomMeasure`` whose draws are unbiased stochastic
surrogates of the full-data unnormalized log-posterior. Consumed by
stochastic-gradient MCMC kernels and by tempered SMC.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    MultivariateNormal,
    NumericArraySpec,
    random_unnormalized_log_prob,
)
from probpipe.core._specs import OpaqueSpec, OutputSpec
from probpipe.distributions._capabilities import (
    SupportsRandomUnnormalizedLogProb,
    SupportsUnnormalizedLogProb,
)
from probpipe.families import BernoulliFamily, RandomFunction, RandomMeasure, glm_likelihood
from probpipe.inference._minibatch import (
    MinibatchedDistribution,
    _FixedMinibatchDistribution,
    _RandomMinibatchLogProb,
)
from tests.inference.canonical import ObservationKernel

# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """200-observation logistic regression with 2 covariates, no intercept."""
    N, P = 200, 2
    X = jax.random.normal(jax.random.PRNGKey(0), (N, P))
    y = ((X @ jnp.array([1.0, -0.5])) > 0).astype(jnp.float32)
    return X, y


@pytest.fixture
def prior():
    return MultivariateNormal("beta", loc=jnp.zeros(2), cov=jnp.eye(2))


@pytest.fixture
def likelihood(regression_data):
    X, _ = regression_data
    # A no-intercept 2-slope logistic regression: both prior dims are slopes
    # paired with X's columns.
    return glm_likelihood("y", BernoulliFamily(), X=X)


@pytest.fixture
def response(regression_data):
    return regression_data[1]


@pytest.fixture
def measure(prior, likelihood, response):
    return MinibatchedDistribution(
        prior,
        likelihood,
        response,
        batch_size=40,
        label="measure",
    )


def _log_density(prior, X, y, theta, rows=None, rescale=1.0):
    """The prior's log-density plus *rescale* times the Bernoulli log-likelihood at *rows*."""
    if rows is not None:
        X, y = X[rows], y[rows]
    per_datum = tfd.Bernoulli(logits=X @ theta).log_prob(y)
    return prior._log_prob(theta) + rescale * jnp.sum(per_datum)


# -- Construction --------------------------------------------------------------


class TestConstruction:
    def test_construction_basic(self, prior, likelihood, response):
        m = MinibatchedDistribution(
            prior,
            likelihood,
            response,
            batch_size=32,
            label="m",
        )
        assert isinstance(m, MinibatchedDistribution)
        assert m.dataset_size == 200
        assert m.batch_size == 32

    def test_construction_rejects_non_log_prob_prior(self, likelihood, response):
        """Prior must satisfy SupportsLogProb."""

        class _BarePrior:
            pass

        with pytest.raises(TypeError, match="SupportsLogProb"):
            MinibatchedDistribution(
                _BarePrior(),
                likelihood,
                response,
                batch_size=32,
                label="measure",
            )

    def test_the_parameters_of_a_prior_that_is_no_distribution_are_opaque(
        self, likelihood, response
    ):
        """The measure and its draws declare them under one component."""

        class _LogDensity:
            def _log_prob(self, value):
                return jnp.asarray(0.0)

            def _unnormalized_log_prob(self, value):
                return jnp.asarray(0.0)

        m = MinibatchedDistribution(
            _LogDensity(),
            likelihood,
            response,
            batch_size=40,
            label="measure",
        )
        draw = m._draw_one(jax.random.PRNGKey(0))
        assert m.event_spec.spec.event_spec == draw.event_spec
        assert draw.event_spec == OutputSpec(parameters=OpaqueSpec())

    def test_construction_rejects_a_likelihood_that_scores_no_subset(self, prior, response):
        """A kernel that cannot score a subset of its observations is rejected."""
        whole = ObservationKernel(
            "y",
            {"beta": prior.event_spec.spec},
            NumericArraySpec((200,)),
            lambda beta: tfd.Independent(tfd.Normal(jnp.zeros(200) + beta[0], 1.0), 1),
        )
        with pytest.raises(TypeError, match="score a subset of its observations"):
            MinibatchedDistribution(
                prior,
                whole,
                response,
                batch_size=32,
                label="measure",
            )

    def test_construction_validates_batch_size_too_small(self, prior, likelihood, response):
        with pytest.raises(ValueError, match="batch_size must be in"):
            MinibatchedDistribution(
                prior,
                likelihood,
                response,
                batch_size=0,
                label="measure",
            )

    def test_construction_validates_batch_size_too_large(self, prior, likelihood, response):
        with pytest.raises(ValueError, match="batch_size must be in"):
            MinibatchedDistribution(
                prior,
                likelihood,
                response,
                batch_size=999,
                label="measure",
            )

    def test_construction_rejects_data_without_a_leading_axis(self, prior, likelihood):
        with pytest.raises(ValueError, match="leading axis"):
            MinibatchedDistribution(
                prior,
                likelihood,
                jnp.asarray(1.0),
                batch_size=1,
                label="measure",
            )


class TestTheDefaultLabel:
    def test_the_notation_shows_the_construction_over_its_operands(
        self, prior, likelihood, response
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = MinibatchedDistribution(prior, likelihood, response, batch_size=40)
        assert m.label == "minibatch"
        assert m.notation == "minibatch(MultivariateNormal(beta), ℙ(y | beta), batch_size=40)"

    def test_the_repr_leaves_the_derived_label_out(self, prior, likelihood, response):
        m = MinibatchedDistribution(prior, likelihood, response, batch_size=40)
        assert "label=" not in repr(m)
        named = MinibatchedDistribution(prior, likelihood, response, batch_size=40, label="mb")
        assert repr(named).endswith("    label='mb',\n)")

    def test_a_prior_without_a_label_requires_one(self, likelihood, response):
        class _LogDensity:
            def _log_prob(self, value):
                return jnp.asarray(0.0)

            def _unnormalized_log_prob(self, value):
                return jnp.asarray(0.0)

        with pytest.raises(
            TypeError,
            match="MinibatchedDistribution cannot take its label from a prior of type "
            "_LogDensity, which has no label; pass label=",
        ):
            MinibatchedDistribution(_LogDensity(), likelihood, response, batch_size=40)


# -- Property accessors -------------------------------------------------------


class TestAccessors:
    """The convenience properties exposed for inspection / debugging."""

    def test_properties_match_constructor_args(self, prior, likelihood, response):
        m = MinibatchedDistribution(
            prior,
            likelihood,
            response,
            batch_size=25,
            with_replacement=True,
            label="custom_name",
        )
        assert m.dataset_size == 200
        assert m.batch_size == 25
        assert m.with_replacement is True
        assert m.prior is prior
        assert m.likelihood is likelihood
        assert m.data is response
        assert m.label == "custom_name"


# -- Protocol membership -------------------------------------------------------


class TestProtocols:
    def test_isinstance_random_measure(self, measure):
        assert isinstance(measure, RandomMeasure)

    def test_not_supports_sampling(self, measure):
        """MinibatchedDistribution doesn't generically support sampling
        — its "samples" are themselves distributions, not values.
        Use ``_random_unnormalized_log_prob`` to get the stochastic
        log-density callable that SGMCMC kernels consume."""
        from probpipe.distributions._capabilities import SupportsSampling

        assert not isinstance(measure, SupportsSampling)

    def test_isinstance_supports_random_unnormalized_log_prob(self, measure):
        assert isinstance(measure, SupportsRandomUnnormalizedLogProb)

    def test_not_iterable(self, measure):
        """STYLE_GUIDE §1.11 — Distribution subclasses are non-iterable."""
        assert not isinstance(measure, Iterable)


# -- Inner draw ----------------------------------------------------------------


class TestInnerDraw:
    """``_draw_one`` returns one fixed-minibatch realisation; the inner
    draw's `_unnormalized_log_prob` is the stochastic surrogate at θ.
    """

    def test_draw_one_returns_fixed_minibatch_distribution(self, measure):
        inner = measure._draw_one(jax.random.PRNGKey(0))
        assert isinstance(inner, _FixedMinibatchDistribution)
        assert isinstance(inner, SupportsUnnormalizedLogProb)

    def test_batch_size_one(self, prior, likelihood, response):
        """A minibatch of one observation."""
        m = MinibatchedDistribution(
            prior,
            likelihood,
            response,
            batch_size=1,
            label="m",
        )
        inner = m._draw_one(jax.random.PRNGKey(0))
        assert inner.rows.shape == (1,)
        lp = inner._unnormalized_log_prob(jnp.zeros(2))
        assert jnp.asarray(lp).shape == ()
        assert jnp.isfinite(lp)

    def test_inner_log_prob_factorises(self, measure, prior, regression_data):
        """For a fixed minibatch, log~D_B(theta) = log_prior(theta) + (N/b)*sum_batch."""
        X, y = regression_data
        inner = measure._draw_one(jax.random.PRNGKey(7))
        theta = jnp.array([0.1, -0.2])
        expected = _log_density(prior, X, y, theta, inner.rows, measure._rescale_factor)
        actual = inner._unnormalized_log_prob(theta)
        np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5)

    def test_with_replacement_flag(self, prior, likelihood, response):
        """``with_replacement=True`` allows repeated indices."""
        m_wr = MinibatchedDistribution(
            prior,
            likelihood,
            response,
            batch_size=5,
            with_replacement=True,
            label="m_wr",
        )
        # Stress test: with batch_size=5 and replacement, over many draws
        # we should see at least one repeat (probability ~ 1 for many draws).
        repeats_seen = 0
        for k in jax.random.split(jax.random.PRNGKey(0), 50):
            rows = m_wr._draw_one(k).rows
            if jnp.unique(rows).shape[0] < rows.shape[0]:
                repeats_seen += 1
        assert repeats_seen > 0, (
            "Expected at least one minibatch with repeats under with_replacement=True"
        )

    def test_batch_size_equals_dataset_size_matches_full(
        self, prior, likelihood, response, regression_data
    ):
        """``batch_size == N`` is the degenerate full-batch case.

        Without replacement this picks a permutation of all observations,
        and the rescale factor is 1.0, so the surrogate exactly equals
        the full-data unnormalized log-density (up to FP).
        """
        X, y = regression_data
        N = X.shape[0]
        m_full = MinibatchedDistribution(
            prior,
            likelihood,
            response,
            batch_size=N,
            with_replacement=False,
            label="m_full",
        )
        inner = m_full._draw_one(jax.random.PRNGKey(11))
        assert inner.rescale_factor == 1.0

        theta = jnp.array([0.2, -0.3])
        actual = inner._unnormalized_log_prob(theta)
        np.testing.assert_allclose(
            float(actual), float(_log_density(prior, X, y, theta)), rtol=1e-5
        )


# -- Mathematical correctness --------------------------------------------------


class TestMathematicalCorrectness:
    """Unbiasedness of the minibatched stochastic-gradient estimator."""

    def test_unbiased_log_density(self, measure, prior, regression_data):
        """Average of random log-densities at fixed theta ≈ full-data log-density.

        2000 minibatches at the test parameters gives MC SE ~0.15-0.3;
        atol=0.5 is ~2 SE — tight enough to catch off-by-rescale-factor
        bugs (~N) and sign bugs (~|full_lp|), loose enough not to flake
        on the fixed PRNG seed.
        """
        X, y = regression_data
        theta = jnp.array([0.3, -0.2])
        full_lp = float(_log_density(prior, X, y, theta))

        # MC estimate over 2000 minibatches (vmapped for speed).
        rf = measure._random_unnormalized_log_prob()
        keys = jax.random.split(jax.random.PRNGKey(1), 2000)
        vals = jax.vmap(lambda k: rf._sample(k)(theta))(keys)
        mc_mean = float(jnp.mean(vals))

        np.testing.assert_allclose(mc_mean, full_lp, atol=0.5)

    def test_unbiased_gradient(self, measure, prior, regression_data):
        """Average gradient over minibatches ≈ full-data gradient.

        2000 vmapped minibatches → per-coord SE ~0.3-0.5. atol=0.75 is
        ~1.5-2.5 SE; catches sign-flip and rescale bugs, tolerates the
        moderate MC noise from a stochastic-gradient estimator.
        """
        X, y = regression_data
        theta = jnp.array([0.3, -0.2])
        full_grad = np.asarray(jax.grad(lambda t: _log_density(prior, X, y, t))(theta))

        rf = measure._random_unnormalized_log_prob()
        keys = jax.random.split(jax.random.PRNGKey(2), 2000)

        def one_grad(k):
            return jax.grad(lambda t: rf._sample(k)(t))(theta)

        grads = jax.vmap(one_grad)(keys)
        mc_grad = np.asarray(jnp.mean(grads, axis=0))

        np.testing.assert_allclose(mc_grad, full_grad, atol=0.75)


# -- random_unnormalized_log_prob op ------------------------------------------


class TestRandomLogProbOp:
    def test_zero_arg_form_returns_random_function(self, measure):
        rf = random_unnormalized_log_prob(measure)
        assert isinstance(rf, RandomFunction)
        assert isinstance(rf, _RandomMinibatchLogProb)

    def test_random_function_sample_returns_callable(self, measure):
        rf = measure._random_unnormalized_log_prob()
        callable_at_k = rf._sample(jax.random.PRNGKey(0))
        # The contract is "callable that returns a scalar log-density"; the
        # shape assertion below is the load-bearing part of that contract.
        theta = jnp.zeros(2)
        result = callable_at_k(theta)
        assert jnp.asarray(result).shape == ()

    def test_random_function_batched_sample_not_supported(self, measure):
        """Batched ``_sample`` of a RandomFunction isn't supported (returns
        a structure of functions, awkward to type)."""
        rf = measure._random_unnormalized_log_prob()
        with pytest.raises(NotImplementedError, match="sample_shape"):
            rf._sample(jax.random.PRNGKey(0), sample_shape=(3,))


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
