"""Sampling contracts and work bounds for minibatch indices."""

from functools import partial
from itertools import combinations, permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal
from probpipe.families import BernoulliFamily, glm_likelihood
from probpipe.inference._blackjax_sgmcmc import _build_grad_estimator
from probpipe.inference._minibatch import (
    MinibatchedDistribution,
    _draw_indices,
    _sample_unique_indices,
)
from tests._compilation import count_compilations


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
def test_indices_are_distinct_in_range_and_reproducible(n, size, typed):
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
def test_full_batch_is_a_random_permutation(n):
    key = jax.random.key(7)
    np.testing.assert_array_equal(
        _draw_indices(key, n, n, with_replacement=False),
        jax.random.permutation(key, n),
    )


def test_sampling_with_replacement_keeps_the_existing_draws():
    key = jax.random.key(7)
    np.testing.assert_array_equal(
        _draw_indices(key, 1000, 32, with_replacement=True),
        jax.random.randint(key, (32,), 0, 1000),
    )


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


@pytest.mark.parametrize("n", [10_000, 1_000_000])
def test_small_batch_work_does_not_allocate_the_population(n):
    draw = partial(_draw_indices, n=n, batch_size=32, with_replacement=False)
    program = jax.make_jaxpr(draw)(jax.random.key(0))
    assert max(_array_sizes(program)) <= 8 * 32


def test_supplement_keeps_previously_selected_indices(monkeypatch):
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
def test_new_keys_reuse_the_compiled_sampler(outer_jit):
    keys = list(jax.random.split(jax.random.key(8), 4))
    jax.block_until_ready(keys)
    draw = partial(_draw_indices, n=1_000_000, batch_size=32, with_replacement=False)
    if outer_jit:
        draw = jax.jit(draw)
    draw(keys[0]).block_until_ready()
    with count_compilations() as compilations:
        for key in keys[1:]:
            draw(key).block_until_ready()
    assert compilations.count == 0


@pytest.mark.parametrize("size", [32, 33])
def test_mapping_keys_preserves_individual_draws(size):
    keys = jax.random.split(jax.random.key(19), 64)
    draw = partial(_draw_indices, n=64, batch_size=size, with_replacement=False)
    expected = jnp.stack([draw(key) for key in keys])
    np.testing.assert_array_equal(jax.jit(jax.vmap(draw))(keys), expected)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(("n", "size"), [(5, 2), (5, 3), (4, 4)])
def test_ordered_samples_and_subsets_have_uniform_frequencies(seed, n, size):
    count = 40_000
    keys = jax.random.split(jax.random.key(seed), count)
    draw = partial(_draw_indices, n=n, batch_size=size, with_replacement=False)
    rows = np.asarray(jax.jit(jax.vmap(draw))(keys))
    assert np.all(np.diff(np.sort(rows, axis=1), axis=1) > 0)
    assert np.all((rows >= 0) & (rows < n))
    # Enumerate the independent combinatorial reference for both ordered
    # tuples and subsets, including the complement and full-batch cases.
    for samples, outcomes in [
        (rows, list(permutations(range(n), size))),
        (np.sort(rows, axis=1), list(combinations(range(n), size))),
    ]:
        codes = np.ravel_multi_index(samples.T, (n,) * size)
        frequencies = np.bincount(codes, minlength=n**size)
        expected_codes = np.ravel_multi_index(np.asarray(outcomes).T, (n,) * size)
        observed = frequencies[expected_codes]
        probability = 1 / len(outcomes)
        standard_error = np.sqrt(count * probability * (1 - probability))
        # Across seeds 0-3 on CPU and CUDA, the largest discrepancy was
        # 2.84 standard deviations. Seven allows more than twice that margin.
        np.testing.assert_allclose(observed, count * probability, rtol=0, atol=7 * standard_error)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("size", [8, 64, 120])
def test_minibatch_gradient_mean_and_covariance_match_finite_population(seed, size):
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
