"""Diagnostics export paths: leaf-keyed variables, nested and flat.

A posterior over a *nested* event template exports one variable per leaf,
named by the leaf's full ``/``-path, with values drawn from the leaf's column
of the flat draw matrix (canonical leaf order). A *flat* posterior must keep
its plain, un-prefixed variable names — existing user code addresses ArviZ
variables by those names, so a silent rename is a break. Both directions are
pinned here with exact values against seeded chains.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import EventTemplate, MultivariateNormal, NumericEventTemplate, NumericRecordArray
from probpipe.diagnostics._arviz_bridge import extract_draws
from probpipe.diagnostics._datatree_store import to_named_posterior_dataset
from probpipe.diagnostics._loo import _add_log_likelihood
from probpipe.inference._approximate_distribution import make_posterior


def _posterior(template, vector_size, *, n_chains=2, n_draws=30, seed0=0):
    """A real posterior over *template*, plus the seeded chains behind it."""
    prior = MultivariateNormal(loc=jnp.zeros(vector_size), cov=jnp.eye(vector_size), name="z")
    chains = [
        jax.random.normal(jax.random.PRNGKey(seed0 + i), (n_draws, vector_size))
        for i in range(n_chains)
    ]
    post = make_posterior(chains, parents=(prior,), algorithm="test", event_template=template)
    return post, chains


def _nested_posterior(n_chains=2, n_draws=30):
    template = EventTemplate(params=EventTemplate(a=(), b=()), scale=())
    return _posterior(template, 3, n_chains=n_chains, n_draws=n_draws)


def _flat_posterior(n_chains=2, n_draws=30):
    template = EventTemplate(a=(), b=())
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


class _RecordDrawsPosterior:
    """Posterior whose per-chain draws are reconstructed from a template.

    Mirrors the surface ``_add_log_likelihood`` reads (``num_chains`` /
    ``num_draws`` / ``draws(chain=)``). With a nested template the draws
    expose full ``/``-path keys; the top-level ``.fields`` view would lose
    the leaves, so this pins the leaf-keyed reconstruction.
    """

    def __init__(self, template, n_chains=2, n_draws=8):
        self._annotations = None
        self._n_chains = n_chains
        self._n_draws = n_draws
        self._template = template
        self._chains = [
            jnp.asarray(np.random.default_rng(i).standard_normal((n_draws, template.vector_size)))
            for i in range(n_chains)
        ]

    @property
    def num_chains(self):
        return self._n_chains

    @property
    def num_draws(self):
        return self._n_draws

    def draws(self, *, chain):
        return NumericRecordArray.from_vector("nra", self._template, self._chains[chain])


class _LinearModel:
    """Likelihood reading intercept/slope leaves off the reconstructed params."""

    def __init__(self, intercept_key, slope_key, n_obs=6):
        self._x = np.random.default_rng(0).standard_normal((n_obs, 1))

        class _Likelihood:
            def __init__(self, x):
                self._x = x

            def per_datum_log_likelihood(self, params, datum):
                # Leaf-path access — succeeds only if _flat_to_record rebuilt
                # the record with the template's (possibly nested) keys.
                intercept = float(np.asarray(params[intercept_key]))
                slope = np.atleast_1d(np.asarray(params[slope_key]))
                x_i = np.atleast_1d(np.asarray(datum["X"]))
                y_i = float(np.asarray(datum["y"]))
                eta = intercept + x_i @ slope
                return float(-0.5 * (y_i - eta) ** 2)

        self._likelihood = _Likelihood(self._x)


@pytest.mark.parametrize(
    ("template", "intercept_key", "slope_key"),
    [
        (
            NumericEventTemplate(coeffs=NumericEventTemplate(intercept=(), slope=(1,))),
            "coeffs/intercept",
            "coeffs/slope",
        ),
        (NumericEventTemplate(intercept=(), slope=(1,)), "intercept", "slope"),
    ],
    ids=["nested", "flat"],
)
def test_add_log_likelihood_fallback_loop_values(template, intercept_key, slope_key):
    """The fallback loop reconstructs draws by leaf key and computes the right values.

    Forcing the Python fallback exercises ``_flat_to_record``. The result is
    checked against an independent NumPy baseline, so a swapped leaf<->column
    mapping (intercept <-> slope) fails — finiteness alone would not catch it.
    """
    n_obs = 6
    n_chains, n_draws = 2, 8
    post = _RecordDrawsPosterior(template, n_chains=n_chains, n_draws=n_draws)
    model = _LinearModel(intercept_key, slope_key, n_obs=n_obs)
    data = {"X": model._x, "y": np.random.default_rng(1).standard_normal(n_obs)}
    with patch("jax.vmap", side_effect=Exception("no vmap")):
        _add_log_likelihood(post, model, data)
    ll = post._annotations["arviz"]["log_likelihood"].to_dataset()["y"]
    assert ll.shape == (n_chains, n_draws, n_obs)

    # Independent baseline: column 0 of the flat draw is the intercept, the
    # rest the slope (the template's canonical leaf order).
    expected = np.empty((n_chains, n_draws, n_obs))
    x = model._x  # (n_obs, 1)
    for c in range(n_chains):
        flat = np.asarray(post._chains[c])  # (n_draws, 2)
        for d in range(n_draws):
            eta = flat[d, 0] + (x @ flat[d, 1:])  # (n_obs,)
            expected[c, d] = -0.5 * (np.asarray(data["y"]) - eta) ** 2
    # Draws are stored float32; observed max deviation vs the float64 baseline
    # is ~7e-7, so 1e-5 is comfortably tight.
    np.testing.assert_allclose(np.asarray(ll), expected, atol=1e-5)
