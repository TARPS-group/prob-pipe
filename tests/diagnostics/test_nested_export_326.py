"""Nested-posterior coverage for the diagnostics export paths (#326).

The leaf-keyed flip means a posterior over a *nested* event template must export
its leaves by full ``/``-path. The pre-#326 ``for f in posterior.fields`` idiom
broke on nested posteriors (a top-level name that is a subtree is not a leaf key);
these tests pin the nested behaviour. The flat case is covered elsewhere.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import EventTemplate, MultivariateNormal, NumericEventTemplate
from probpipe.diagnostics._arviz_bridge import extract_draws
from probpipe.diagnostics._datatree_store import to_named_posterior_dataset
from probpipe.diagnostics._loo import _add_log_likelihood
from probpipe.inference._approximate_distribution import make_posterior


def _nested_posterior(n_chains=2, n_draws=30):
    template = EventTemplate(params=EventTemplate(a=(), b=()), scale=())
    vector_size = 3  # a + b + scale
    prior = MultivariateNormal(loc=jnp.zeros(vector_size), cov=jnp.eye(vector_size), name="z")
    chains = [
        jax.random.normal(jax.random.PRNGKey(i), (n_draws, vector_size)) for i in range(n_chains)
    ]
    return make_posterior(chains, parents=(prior,), algorithm="test", event_template=template)


def test_extract_draws_keys_nested_by_leaf_path():
    post = _nested_posterior()
    draws = extract_draws(post)
    assert set(draws) == {"params/a", "params/b", "scale"}
    assert draws["params/a"].shape[-1] == 30 or draws["params/a"].ndim >= 1


def test_to_named_posterior_dataset_nested_by_leaf_path():
    post = _nested_posterior()
    ds = to_named_posterior_dataset(post)
    assert set(ds.data_vars) == {"params/a", "params/b", "scale"}
    # chain / draw axes are present on each leaf variable.
    assert ds["params/a"].dims[:2] == ("chain", "draw")


class _NestedDrawsPosterior:
    """Posterior whose per-chain draws are a *nested* ``NumericRecordArray``.

    Mirrors the surface ``_add_log_likelihood`` reads (``num_chains`` /
    ``num_draws`` / ``draws(chain=)``). The draws expose a nested
    ``event_template`` whose ``keys()`` are full ``/``-paths; the pre-#326
    top-level ``.fields`` would yield only ``("coeffs",)`` and lose the leaves.
    """

    def __init__(self, n_chains=2, n_draws=8):
        self._auxiliary = None
        self._n_chains = n_chains
        self._n_draws = n_draws
        self._template = NumericEventTemplate(coeffs=NumericEventTemplate(intercept=(), slope=(1,)))
        self._chains = [
            jnp.asarray(
                np.random.default_rng(i).standard_normal((n_draws, self._template.vector_size))
            )
            for i in range(n_chains)
        ]

    @property
    def num_chains(self):
        return self._n_chains

    @property
    def num_draws(self):
        return self._n_draws

    def draws(self, *, chain):
        return self._template.from_vector(self._chains[chain])


class _NestedModel:
    """Likelihood reading *nested* leaf paths off the reconstructed params."""

    def __init__(self, n_obs=6):
        self._x = np.random.default_rng(0).standard_normal((n_obs, 1))

        class _Likelihood:
            def __init__(self, x):
                self._x = x

            def per_datum_log_likelihood(self, params, datum):
                # Nested leaf-path access — only succeeds if _flat_to_record
                # rebuilt the `coeffs/...` nesting from the flat draw.
                intercept = float(np.asarray(params["coeffs/intercept"]))
                slope = np.atleast_1d(np.asarray(params["coeffs/slope"]))
                x_i = np.atleast_1d(np.asarray(datum["X"]))
                y_i = float(np.asarray(datum["y"]))
                eta = intercept + x_i @ slope
                return float(-0.5 * (y_i - eta) ** 2)

        self._likelihood = _Likelihood(self._x)


def test_add_log_likelihood_nested_posterior_fallback_loop():
    """The fallback loop reconstructs a nested Record from a nested posterior.

    Forcing the Python fallback exercises ``_flat_to_record``; the likelihood
    reads ``coeffs/intercept`` / ``coeffs/slope``, so a finite result proves the
    flat↔Record mapping kept the nesting (leaf-path keyed, not top-level).
    """
    n_obs = 6
    post = _NestedDrawsPosterior(n_chains=2, n_draws=8)
    model = _NestedModel(n_obs=n_obs)
    data = {"X": model._x, "y": np.random.default_rng(1).standard_normal(n_obs)}
    with patch("jax.vmap", side_effect=Exception("no vmap")):
        _add_log_likelihood(post, model, data)
    ll = post._auxiliary["arviz"]["log_likelihood"].to_dataset()["y"]
    assert ll.shape == (2, 8, n_obs)
    assert np.all(np.isfinite(np.asarray(ll)))
