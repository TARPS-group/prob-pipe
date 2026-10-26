"""Tests for :class:`MinibatchedDistribution`.

A ``RandomMeasure`` whose draws are unbiased stochastic
surrogates of the full-data unnormalized log-posterior. Consumed by
stochastic-gradient MCMC kernels and by tempered SMC.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import partial
from itertools import combinations, permutations

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
from probpipe.inference._blackjax_sgmcmc import _build_grad_estimator
from probpipe.inference._minibatch import (
    MinibatchedDistribution,
    _draw_indices,
    _FixedMinibatchDistribution,
    _RandomMinibatchLogProb,
    _sample_unique_indices,
)
from tests._compilation import count_compilations
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
        """The measure and its draws declare them under one output."""

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
        assert m.event_spec.term_spec.event_spec == draw.event_spec
        assert draw.event_spec == OutputSpec(parameters=OpaqueSpec())

    def test_construction_rejects_a_likelihood_that_scores_no_subset(self, prior, response):
        """A kernel that cannot score a subset of its observations is rejected."""
        whole = ObservationKernel(
            "y",
            {"beta": prior.event_spec.term_spec},
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

    def test_the_repr_shows_a_label_that_names_the_class(self, prior, likelihood, response):
        m = MinibatchedDistribution(
            prior, likelihood, response, batch_size=40, label="MinibatchedDistribution"
        )
        assert repr(m).endswith("    label='MinibatchedDistribution',\n)")

    def test_a_prior_without_a_label_requires_one(self, likelihood, response):
        class _LogDensity:
            def _log_prob(self, value):
                return jnp.asarray(0.0)

            def _unnormalized_log_prob(self, value):
                return jnp.asarray(0.0)

        with pytest.raises(
            TypeError,
            match="MinibatchedDistribution cannot take its label from a prior of type "
            r"_LogDensity, which has no label; pass label=\.\.\.$",
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


def _array_sizes(program):
    """Array sizes in a JAX program, including its nested loops and calls."""
    if hasattr(program, "jaxpr"):
        yield from _array_sizes(program.jaxpr)
    elif hasattr(program, "eqns"):
        for equation in program.eqns:
            for variable in equation.outvars:
                yield np.prod(getattr(variable.aval, "shape", ()), dtype=np.int64)
            for value in equation.params.values():
                yield from _array_sizes(value)
    elif isinstance(program, (tuple, list)):
        for value in program:
            yield from _array_sizes(value)


def _check_uniform_samples(rows, n, size):
    """Check cells and pooled goodness of fit against enumerated outcomes."""
    from scipy.stats import chi2

    assert np.all(np.diff(np.sort(rows, axis=-1), axis=-1) > 0)
    assert np.all((rows >= 0) & (rows < n))
    statistics = []
    for samples, outcomes in [
        (rows, list(permutations(range(n), size))),
        (np.sort(rows, axis=-1), list(combinations(range(n), size))),
    ]:
        expected_codes = np.ravel_multi_index(np.asarray(outcomes).T, (n,) * size)
        frequencies = []
        for sample in samples:
            codes = np.ravel_multi_index(sample.T, (n,) * size)
            observed = np.bincount(codes, minlength=n**size)[expected_codes]
            count = len(sample)
            probability = 1 / len(outcomes)
            standard_error = np.sqrt(count * probability * (1 - probability))
            # Seeds 0-3, 40,000 draws each: CPU/CUDA maximum cell error is 3.044
            # SE (including the reduced-candidate path). Seven exceeds twice
            # this measured discrepancy.
            np.testing.assert_allclose(
                observed, count * probability, rtol=0, atol=7 * standard_error
            )
            frequencies.append(observed)
        pooled = np.sum(frequencies, axis=0)
        if len(outcomes) == 1:
            assert pooled[0] == rows.shape[0] * rows.shape[1]
            continue
        expected = pooled.sum() / len(outcomes)
        statistic = np.sum((pooled - expected) ** 2 / expected)
        # Pooled CPU/CUDA statistics: ordinary ordered outcomes 5.656-108.643,
        # subsets 1.470-17.908; reduced candidates 135.242/22.373.
        # Critical values use each independent reference's degrees of freedom.
        threshold = chi2.isf(1e-6, df=len(outcomes) - 1)
        assert statistic < threshold, (statistic, threshold)
        statistics.append((len(outcomes) - 1, float(statistic), float(threshold)))
    return statistics


class TestIndexSampling:
    """Sampling contracts, distributions and work bounds."""

    @pytest.mark.parametrize(
        ("n", "size"),
        [
            (1, 1),
            (17, 1),
            (5, 2),
            (5, 3),
            (64, 32),
            (64, 33),
            (63, 8),
            (64, 8),
            (65, 8),
            (65536, 256),
            (66049, 257),
            (1000, 500),
            (1000, 999),
            (17, 16),
            (17, 17),
        ],
    )
    @pytest.mark.parametrize("typed", [False, True])
    def test_indices_are_distinct_in_range_and_reproducible(self, n, size, typed):
        key = jax.random.key(41) if typed else jax.random.PRNGKey(41)
        draw = partial(_draw_indices, n=n, batch_size=size, with_replacement=False)
        with jax.disable_jit():
            expected = draw(key)
        actual = jax.jit(draw)(key)
        assert actual.shape == (size,)
        assert jnp.issubdtype(actual.dtype, jnp.integer)
        assert len(np.unique(actual)) == size
        assert np.all((actual >= 0) & (actual < n))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(draw(key), expected)

    @pytest.mark.parametrize("n", [1, 17, 1024])
    def test_full_batch_is_a_random_permutation(self, n):
        key = jax.random.key(7)
        np.testing.assert_array_equal(
            _draw_indices(key, n, n, with_replacement=False),
            jax.random.permutation(key, n),
        )

    def test_sampling_with_replacement_keeps_the_existing_draws(self):
        key = jax.random.key(7)
        np.testing.assert_array_equal(
            _draw_indices(key, 1000, 32, with_replacement=True),
            jax.random.randint(key, (32,), 0, 1000),
        )

    @pytest.mark.parametrize("n", [10_000, 1_000_000])
    def test_small_batch_work_does_not_allocate_the_population(self, n):
        draw = partial(_draw_indices, n=n, batch_size=32, with_replacement=False)
        program = jax.make_jaxpr(draw)(jax.random.key(0))
        assert max(_array_sizes(program)) <= 8 * 32

    def test_supplement_keeps_previously_selected_indices(self, monkeypatch):
        key = jax.random.key(42)
        next_key, first_key = jax.random.split(key)
        _, second_key = jax.random.split(next_key)

        def candidates(key, shape, minval, maxval):
            positions = jnp.arange(shape[0])
            first = jnp.all(jax.random.key_data(key) == jax.random.key_data(first_key))
            second = jnp.all(jax.random.key_data(key) == jax.random.key_data(second_key))
            # The first batch supplies 0; the second supplies 1 and 2. A third
            # batch supplies 3, 4 and 5 so discarding prior values fails promptly.
            return jnp.where(first, 0, jnp.where(second, 1 + positions % 2, 3 + positions % 3))

        monkeypatch.setattr(jax.random, "randint", candidates)
        draw = jax.jit(partial(_sample_unique_indices, n=6, size=3))
        np.testing.assert_array_equal(draw(key), [0, 1, 2])

    @pytest.mark.parametrize("outer_jit", [False, True])
    @pytest.mark.parametrize(("n", "size"), [(1_000_000, 32), (64, 33)])
    def test_new_keys_reuse_the_compiled_sampler(self, outer_jit, n, size):
        keys = list(jax.random.split(jax.random.key(8), 4))
        jax.block_until_ready(keys)
        draw = partial(_draw_indices, n=n, batch_size=size, with_replacement=False)
        if outer_jit:
            draw = jax.jit(draw)
        draw(keys[0]).block_until_ready()
        with count_compilations() as compilations:
            for key in keys[1:]:
                draw(key).block_until_ready()
        assert compilations.count == 0

    @pytest.mark.parametrize("size", [32, 33])
    def test_mapping_keys_preserves_individual_draws(self, size):
        keys = jax.random.split(jax.random.key(19), 64)
        draw = partial(_draw_indices, n=64, batch_size=size, with_replacement=False)
        expected = jnp.stack([draw(key) for key in keys])
        np.testing.assert_array_equal(jax.jit(jax.vmap(draw))(keys), expected)

    @pytest.mark.parametrize(("n", "size"), [(5, 1), (5, 2), (5, 3), (5, 4), (4, 4)])
    def test_ordered_samples_and_subsets_have_uniform_frequencies(self, n, size):
        draw = partial(_draw_indices, n=n, batch_size=size, with_replacement=False)
        mapped = jax.jit(jax.vmap(draw))
        rows = np.stack(
            [
                np.asarray(mapped(jax.random.split(jax.random.key(seed), 40_000)))
                for seed in range(4)
            ]
        )
        _check_uniform_samples(rows, n, size)

    def test_supplement_uniformity_with_small_candidate_arrays(self, monkeypatch):
        import probpipe.inference._minibatch as minibatch

        monkeypatch.setattr(minibatch, "_candidate_count", lambda n, size: size)
        # A fresh wrapper around the undecorated helper isolates tracing from
        # the production sampler's cache. No patched trace can escape this test.
        draw = jax.jit(jax.vmap(partial(_sample_unique_indices, n=6, size=3)))
        try:
            rows = np.stack(
                [
                    np.asarray(draw(jax.random.split(jax.random.key(seed), 40_000)))
                    for seed in range(4)
                ]
            )
            _check_uniform_samples(rows, 6, 3)
        finally:
            draw.clear_cache()

    def test_supplement_mapping_preserves_individual_draws(self, monkeypatch):
        import probpipe.inference._minibatch as minibatch

        monkeypatch.setattr(minibatch, "_candidate_count", lambda n, size: size)
        individual = jax.jit(partial(_sample_unique_indices, n=64, size=32))
        mapped = jax.jit(jax.vmap(individual))
        try:
            keys = jax.random.split(jax.random.key(19), 64)
            expected = np.stack([np.asarray(individual(key)) for key in keys])
            np.testing.assert_array_equal(mapped(keys), expected)
        finally:
            mapped.clear_cache()
            individual.clear_cache()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("size", [8, 64, 120])
    def test_minibatch_gradient_mean_and_covariance_match_finite_population(self, seed, size):
        n, count = 128, 8192
        x = np.linspace(-2, 2, n, dtype=np.float32)
        design = np.column_stack([np.ones(n, dtype=np.float32), x])
        response = (np.arange(n) % 3 == 0).astype(np.float32)
        position = np.array([0.3, -0.2], dtype=np.float32)
        prior = MultivariateNormal("beta", loc=jnp.zeros(2), cov=jnp.eye(2))
        likelihood = glm_likelihood("y", BernoulliFamily(), X=jnp.asarray(design))
        measure = MinibatchedDistribution(prior, likelihood, jnp.asarray(response), size)
        estimator = _build_grad_estimator(measure)

        def gradient(key):
            return estimator(jnp.asarray(position), key)

        keys = jax.random.split(jax.random.key(seed), count)
        gradients = np.asarray(jax.jit(jax.vmap(gradient))(keys))
        # Analytic logistic scores and the finite-population correction are
        # independent of the minibatch likelihood and its autodiff implementation.
        probability = 1 / (1 + np.exp(-(design.astype(np.float64) @ position)))
        scores = design * (response - probability)[:, None]
        expected_mean = -position + scores.sum(axis=0)
        expected_covariance = n**2 * (1 - size / n) / size * np.cov(scores, rowvar=False)
        standard_error = np.sqrt(np.diag(expected_covariance) / count)
        # Across seeds 0-3 on CPU and CUDA, maximum mean errors were 0.20-2.04
        # standard errors and normalized covariance errors were 0.007-0.040.
        np.testing.assert_allclose(
            (gradients.mean(axis=0) - expected_mean) / standard_error, 0, rtol=0, atol=6
        )
        # Normalize covariance entries by their marginal scales, including the
        # off-diagonal entry, which need not be bounded away from zero.
        scales = np.sqrt(np.outer(np.diag(expected_covariance), np.diag(expected_covariance)))
        np.testing.assert_allclose(
            np.cov(gradients, rowvar=False) / scales,
            expected_covariance / scales,
            rtol=0,
            atol=0.08,
        )
