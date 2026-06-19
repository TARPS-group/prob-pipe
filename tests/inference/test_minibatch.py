"""Tests for :class:`MinibatchedDistribution`.

A ``RandomMeasure[Record]`` whose draws are unbiased stochastic
surrogates of the full-data unnormalized log-posterior. Consumed by
stochastic-gradient MCMC kernels and by tempered SMC.
"""

from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.glm as tfp_glm

from probpipe import (
    GLMLikelihood,
    MultivariateNormal,
    Record,
    random_unnormalized_log_prob,
)
from probpipe.core._random_functions import RandomFunction
from probpipe.core._random_measures import RandomMeasure
from probpipe.core.protocols import (
    SupportsRandomUnnormalizedLogProb,
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
    """200-observation logistic regression with 2 covariates, no intercept."""
    N, P = 200, 2
    X = jax.random.normal(jax.random.PRNGKey(0), (N, P))
    y = ((X @ jnp.array([1.0, -0.5])) > 0).astype(jnp.float32)
    return X, y


@pytest.fixture
def prior():
    return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta")


@pytest.fixture
def likelihood(regression_data):
    X, _ = regression_data
    # ``fit_intercept=False``: this is a no-intercept 2-slope logistic
    # regression — both prior dims are slopes paired with X's columns.
    return GLMLikelihood(tfp_glm.Bernoulli(), x=X, fit_intercept=False)


@pytest.fixture
def data_record(regression_data):
    X, y = regression_data
    return Record(X=X, y=y)


@pytest.fixture
def measure(prior, likelihood, data_record):
    return MinibatchedDistribution(prior, likelihood, data_record, batch_size=40)


# -- Construction --------------------------------------------------------------


class TestConstruction:
    def test_construction_basic(self, prior, likelihood, data_record):
        m = MinibatchedDistribution(prior, likelihood, data_record, batch_size=32)
        assert isinstance(m, MinibatchedDistribution)
        assert m.dataset_size == 200
        assert m.batch_size == 32

    def test_construction_rejects_non_log_prob_prior(self, likelihood, data_record):
        """Prior must satisfy SupportsLogProb."""
        class _BarePrior:
            pass
        with pytest.raises(TypeError, match="SupportsLogProb"):
            MinibatchedDistribution(_BarePrior(), likelihood, data_record, batch_size=32)

    def test_construction_rejects_non_cil_likelihood(self, prior, data_record):
        """A bare ``Likelihood`` (no ``per_datum_log_likelihood``) is rejected."""
        class _BareLikelihood:
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        with pytest.raises(TypeError, match="ConditionallyIndependentLikelihood"):
            MinibatchedDistribution(prior, _BareLikelihood(), data_record, batch_size=32)

    def test_construction_validates_batch_size_too_small(self, prior, likelihood, data_record):
        with pytest.raises(ValueError, match="batch_size must be in"):
            MinibatchedDistribution(prior, likelihood, data_record, batch_size=0)

    def test_construction_validates_batch_size_too_large(self, prior, likelihood, data_record):
        with pytest.raises(ValueError, match="batch_size must be in"):
            MinibatchedDistribution(prior, likelihood, data_record, batch_size=999)

    def test_construction_rejects_nested_records(self, prior, likelihood):
        """Nested Records fail at construction with an actionable error."""
        nested = Record(
            features=Record(X=jnp.zeros((200, 2)), extra=jnp.zeros((200,))),
            y=jnp.zeros((200,)),
        )
        with pytest.raises(ValueError, match="flat Record"):
            MinibatchedDistribution(prior, likelihood, nested, batch_size=32)


# -- Property accessors -------------------------------------------------------


class TestAccessors:
    """The convenience properties exposed for inspection / debugging."""

    def test_properties_match_constructor_args(
        self, prior, likelihood, data_record,
    ):
        m = MinibatchedDistribution(
            prior, likelihood, data_record, batch_size=25,
            with_replacement=True,
            name="custom_name",
        )
        assert m.dataset_size == 200
        assert m.batch_size == 25
        assert m.with_replacement is True
        assert m.prior is prior
        assert m.likelihood is likelihood
        assert m.data is data_record
        assert m.name == "custom_name"

    def test_default_name_includes_batch_size(self, measure):
        assert "batch_size=40" in measure.name


# -- Protocol membership -------------------------------------------------------


class TestProtocols:
    def test_isinstance_random_measure(self, measure):
        assert isinstance(measure, RandomMeasure)

    def test_not_supports_sampling(self, measure):
        """MinibatchedDistribution doesn't generically support sampling
        — its "samples" are themselves distributions, not values.
        Use ``_random_unnormalized_log_prob`` to get the stochastic
        log-density callable that SGMCMC kernels consume."""
        from probpipe.core.protocols import SupportsSampling
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

    def test_batch_size_one(self, prior, likelihood, data_record):
        """Exercise the vmap-over-1-element-axis edge case."""
        m = MinibatchedDistribution(prior, likelihood, data_record, batch_size=1)
        inner = m._draw_one(jax.random.PRNGKey(0))
        assert inner.batch["X"].shape == (1, 2)
        assert inner.batch["y"].shape == (1,)
        # log_prob still works (vmap over a length-1 axis).
        lp = inner._unnormalized_log_prob(jnp.zeros(2))
        assert jnp.asarray(lp).shape == ()
        assert jnp.isfinite(lp)

    def test_inner_log_prob_factorises(self, measure, prior, likelihood):
        """For a fixed minibatch, log~D_B(theta) = log_prior(theta) + (N/b)*sum_batch."""
        inner = measure._draw_one(jax.random.PRNGKey(7))
        theta = jnp.array([0.1, -0.2])

        # Hand-compute the expected value from the captured batch.
        batch = inner.batch
        prior_lp = prior._log_prob(theta)
        per_datum = jax.vmap(
            likelihood.per_datum_log_likelihood, in_axes=(None, 0),
        )(theta, batch)
        expected = prior_lp + measure._rescale_factor * jnp.sum(per_datum)

        actual = inner._unnormalized_log_prob(theta)
        np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5)

    def test_record_and_recordarray_inputs_equivalent(
        self, prior, likelihood, regression_data,
    ):
        """``Record`` and ``NumericRecordArray`` data inputs produce
        identical log-densities given the same minibatch indices.

        Locks the ``_index_along_leading`` RecordArray-via-Record-subclass
        path: indexing each leaf along the batch axis must give the same
        per-minibatch surrogate regardless of which container type the
        user passes.
        """
        from probpipe import NumericRecordArray, NumericEventTemplate
        X, y = regression_data
        n = X.shape[0]
        record_data = Record(X=X, y=y)
        recordarray_data = NumericRecordArray(
            {"X": jnp.asarray(X), "y": jnp.asarray(y)},
            batch_shape=(n,),
            template=NumericEventTemplate(X=(X.shape[1],), y=()),
        )

        m_rec = MinibatchedDistribution(prior, likelihood, record_data, batch_size=20)
        m_ra = MinibatchedDistribution(prior, likelihood, recordarray_data, batch_size=20)

        # Same key → same minibatch indices → same log-density at theta.
        key = jax.random.PRNGKey(13)
        theta = jnp.array([0.1, -0.1])
        lp_rec = float(m_rec._draw_one(key)._unnormalized_log_prob(theta))
        lp_ra = float(m_ra._draw_one(key)._unnormalized_log_prob(theta))
        np.testing.assert_allclose(lp_rec, lp_ra, rtol=1e-5)

    def test_with_replacement_flag(self, prior, likelihood, data_record):
        """``with_replacement=True`` allows repeated indices."""
        m_wr = MinibatchedDistribution(
            prior, likelihood, data_record, batch_size=5,
            with_replacement=True,
        )
        # Stress test: with batch_size=5 and replacement, over many draws
        # we should see at least one repeat (probability ~ 1 for many draws).
        repeats_seen = 0
        for k in jax.random.split(jax.random.PRNGKey(0), 50):
            inner = m_wr._draw_one(k)
            batch_X = inner.batch["X"]
            unique = jnp.unique(batch_X, axis=0)
            if unique.shape[0] < batch_X.shape[0]:
                repeats_seen += 1
        assert repeats_seen > 0, (
            "Expected at least one minibatch with repeats under with_replacement=True"
        )

    def test_batch_size_equals_dataset_size_matches_full(
        self, prior, likelihood, data_record, regression_data,
    ):
        """``batch_size == N`` is the degenerate full-batch case.

        Without replacement this picks a permutation of all observations,
        and the rescale factor is 1.0, so the surrogate exactly equals
        the full-data unnormalized log-density (up to FP).
        """
        X, _ = regression_data
        N = X.shape[0]
        m_full = MinibatchedDistribution(
            prior, likelihood, data_record, batch_size=N, with_replacement=False,
        )
        inner = m_full._draw_one(jax.random.PRNGKey(11))
        assert inner.rescale_factor == 1.0

        theta = jnp.array([0.2, -0.3])
        # Full-data log-density
        per_datum = jax.vmap(
            likelihood.per_datum_log_likelihood, in_axes=(None, 0),
        )(theta, data_record)
        full = prior._log_prob(theta) + jnp.sum(per_datum)

        actual = inner._unnormalized_log_prob(theta)
        np.testing.assert_allclose(float(actual), float(full), rtol=1e-5)


# -- Bare-array data path -----------------------------------------------------


class TestBareArrayData:
    """Bare ``jnp.ndarray`` data works when the CIL likelihood's
    ``per_datum_log_likelihood`` accepts a scalar datum.

    The canonical container is ``Record`` / ``RecordArray`` (so covariates
    have named-field provenance), but ``_data_size`` and
    ``_index_along_leading`` also work on a leading-axis-arrayed
    response-only dataset.
    """

    def test_bare_array_data_evaluates(self, prior, regression_data):
        _, y = regression_data
        N = y.shape[0]

        class _ResponseOnlyLikelihood:
            """CIL that handles bare-array datum (scalar y_i).

            Defines ``log_likelihood`` and ``per_datum_log_likelihood``
            so the protocol check (``isinstance(lik, CIL)``) succeeds.
            """
            def log_likelihood(self, params, data):
                return -0.5 * jnp.sum(jnp.asarray(data) ** 2)

            def per_datum_log_likelihood(self, params, datum):
                return -0.5 * jnp.asarray(datum) ** 2

        m = MinibatchedDistribution(
            prior, _ResponseOnlyLikelihood(), y, batch_size=20,
        )
        assert m.dataset_size == N

        theta = jnp.zeros(2)
        lp = m._draw_one(jax.random.PRNGKey(0))._unnormalized_log_prob(theta)
        assert jnp.isfinite(lp)


# -- Mathematical correctness --------------------------------------------------


class TestMathematicalCorrectness:
    """Unbiasedness of the minibatched stochastic-gradient estimator."""

    def test_unbiased_log_density(self, measure, prior, likelihood, data_record):
        """Average of random log-densities at fixed theta ≈ full-data log-density.

        2000 minibatches at the test parameters gives MC SE ~0.15-0.3;
        atol=0.5 is ~2 SE — tight enough to catch off-by-rescale-factor
        bugs (~N) and sign bugs (~|full_lp|), loose enough not to flake
        on the fixed PRNG seed.
        """
        theta = jnp.array([0.3, -0.2])
        # Full reference:
        per_datum = jax.vmap(
            likelihood.per_datum_log_likelihood, in_axes=(None, 0),
        )(theta, data_record)
        full_lp = float(prior._log_prob(theta) + jnp.sum(per_datum))

        # MC estimate over 2000 minibatches (vmapped for speed).
        rf = measure._random_unnormalized_log_prob()
        keys = jax.random.split(jax.random.PRNGKey(1), 2000)
        vals = jax.vmap(lambda k: rf._sample(k)(theta))(keys)
        mc_mean = float(jnp.mean(vals))

        np.testing.assert_allclose(mc_mean, full_lp, atol=0.5)

    def test_unbiased_gradient(self, measure, prior, likelihood, data_record):
        """Average gradient over minibatches ≈ full-data gradient.

        2000 vmapped minibatches → per-coord SE ~0.3-0.5. atol=0.75 is
        ~1.5-2.5 SE; catches sign-flip and rescale bugs, tolerates the
        moderate MC noise from a stochastic-gradient estimator.
        """
        theta = jnp.array([0.3, -0.2])

        def full_log_density(t):
            per_datum = jax.vmap(
                likelihood.per_datum_log_likelihood, in_axes=(None, 0),
            )(t, data_record)
            return prior._log_prob(t) + jnp.sum(per_datum)

        full_grad = np.asarray(jax.grad(full_log_density)(theta))

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
