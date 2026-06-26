"""Nested-posterior coverage for the diagnostics export paths (#326).

The leaf-keyed flip means a posterior over a *nested* event template must export
its leaves by full ``/``-path. The pre-#326 ``for f in posterior.fields`` idiom
broke on nested posteriors (a top-level name that is a subtree is not a leaf key);
these tests pin the nested behaviour. The flat case is covered elsewhere.
"""

import jax
import jax.numpy as jnp

from probpipe import EventTemplate, MultivariateNormal
from probpipe.diagnostics._arviz_bridge import extract_draws
from probpipe.diagnostics._datatree_store import to_named_posterior_dataset
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
