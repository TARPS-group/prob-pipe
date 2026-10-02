"""Diagnostics export paths: leaf-keyed variables, nested and flat.

A posterior over a *nested* record exports one variable per leaf,
named by the leaf's full ``/``-path, with values drawn from the leaf's column
of the flat draw matrix (canonical leaf order). A *flat* posterior must keep
its plain, un-prefixed variable names — existing user code addresses ArviZ
variables by those names, so a silent rename is a break. Both directions are
pinned here with exact values against seeded chains.
"""

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import MultivariateNormal, RecordSpec
from probpipe.diagnostics._arviz_bridge import extract_draws
from probpipe.diagnostics._datatree_store import to_named_posterior_dataset
from probpipe.inference._approximate_distribution import make_posterior


def _posterior(template, vector_size, *, n_chains=2, n_draws=30, seed0=0):
    """A real posterior over *template*, plus the seeded chains behind it."""
    prior = MultivariateNormal(loc=jnp.zeros(vector_size), cov=jnp.eye(vector_size), name="z")
    chains = [
        jax.random.normal(jax.random.PRNGKey(seed0 + i), (n_draws, vector_size))
        for i in range(n_chains)
    ]
    post = make_posterior(chains, parents=(prior,), method="test", event_spec=template)
    return post, chains


def _nested_posterior(n_chains=2, n_draws=30):
    template = RecordSpec(params=RecordSpec(a=(), b=()), scale=())
    return _posterior(template, 3, n_chains=n_chains, n_draws=n_draws)


def _flat_posterior(n_chains=2, n_draws=30):
    template = RecordSpec(a=(), b=())
    return _posterior(template, 2, n_chains=n_chains, n_draws=n_draws, seed0=10)


def test_extract_draws_keys_nested_by_leaf_path():
    n_chains, n_draws = 2, 30
    post, chains = _nested_posterior(n_chains, n_draws)
    draws = extract_draws(post)
    assert set(draws) == {"params/a", "params/b", "scale"}
    # Exact leaf<->column mapping against the seeded chains: extract_draws
    # concatenates the chains, and each leaf is one column of the flat draw
    # matrix in canonical leaf order. A swapped or transposed mapping fails.
    stacked = np.concatenate([np.asarray(c) for c in chains], axis=0)  # (60, 3)
    for i, key in enumerate(["params/a", "params/b", "scale"]):
        assert draws[key].shape == (n_chains * n_draws,)
        np.testing.assert_array_equal(draws[key], stacked[:, i])


def test_extract_draws_flat_names_unprefixed():
    # Backward compatibility: a flat posterior keeps its plain field names.
    post, chains = _flat_posterior()
    draws = extract_draws(post)
    assert set(draws) == {"a", "b"}
    stacked = np.concatenate([np.asarray(c) for c in chains], axis=0)
    np.testing.assert_array_equal(draws["a"], stacked[:, 0])
    np.testing.assert_array_equal(draws["b"], stacked[:, 1])


def test_to_named_posterior_dataset_nested_by_leaf_path():
    post, chains = _nested_posterior()
    ds = to_named_posterior_dataset(post)
    assert set(ds.data_vars) == {"params/a", "params/b", "scale"}
    # One (chain, draw) variable per leaf, with the leaf's exact column values.
    per_chain = np.stack([np.asarray(c) for c in chains], axis=0)  # (2, 30, 3)
    for i, key in enumerate(["params/a", "params/b", "scale"]):
        assert ds[key].dims == ("chain", "draw")
        np.testing.assert_array_equal(ds[key].values, per_chain[:, :, i])


def test_to_named_posterior_dataset_flat_names_unprefixed():
    post, chains = _flat_posterior()
    ds = to_named_posterior_dataset(post)
    assert set(ds.data_vars) == {"a", "b"}
    per_chain = np.stack([np.asarray(c) for c in chains], axis=0)
    np.testing.assert_array_equal(ds["a"].values, per_chain[:, :, 0])
    np.testing.assert_array_equal(ds["b"].values, per_chain[:, :, 1])
