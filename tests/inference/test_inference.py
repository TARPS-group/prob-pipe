"""Tests for the probpipe.inference package.

Covers:
- make_posterior: an inference result's levels, annotations, warmup, and draws
- make_posterior with a record target: named draws
- FieldView: component views, select, broadcasting
- rwmh Function: basic sampling with SupportsLogProb
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    MultivariateNormal,
    Normal,
    NumericRecordBatch,
    OpaqueSpec,
    Record,
    RecordSpec,
    mean,
    sample,
    variance,
    workflow_run,
)
from probpipe.core._record_batch import RecordBatch
from probpipe.core._specs import NumericArraySpec, OutputSpec
from probpipe.distributions import FieldView
from probpipe.inference import rwmh
from probpipe.inference._approximate_distribution import make_posterior
from probpipe.inference._inference_utils import build_mcmc_datatree
from tests._posterior import (
    arviz_data,
    flat_chains,
    flat_draws,
    method_of,
    num_chains,
    num_draws,
    posterior_of,
    warmup_samples,
)

# ---------------------------------------------------------------------------
# make_posterior
# ---------------------------------------------------------------------------


class TestMakePosterior:
    """An inference result: the empirical law of a run's draws on the levels chain and draw."""

    @pytest.fixture
    def two_chain_dist(self):
        """Two chains, 50 draws each, 2D event, built via make_posterior."""
        chain1 = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        chain2 = jax.random.normal(jax.random.PRNGKey(1), (50, 2))
        warmup1 = jax.random.normal(jax.random.PRNGKey(2), (10, 2))
        warmup2 = jax.random.normal(jax.random.PRNGKey(3), (10, 2))
        chains = [chain1, chain2]
        annotations = build_mcmc_datatree(chains, warmup_chains=[warmup1, warmup2])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        return make_posterior(
            chains,
            parents=(prior,),
            method="test",
            annotations=annotations,
        )

    def test_empty_chains_raises(self):
        with pytest.raises(ValueError, match="at least one chain"):
            posterior_of([], label="x")

    def test_the_result_is_an_empirical_law(self, two_chain_dist):
        assert isinstance(two_chain_dist, EmpiricalDistribution)

    def test_atoms_lie_on_the_chain_and_draw_levels(self, two_chain_dist):
        assert two_chain_dist.atoms.level_names == ("chain", "draw")
        assert two_chain_dist.atoms.batch_shape == (2, 50)
        assert (num_chains(two_chain_dist), num_draws(two_chain_dist)) == (2, 50)

    def test_event_shape(self, two_chain_dist):
        assert two_chain_dist.event_shape == (2,)

    def test_num_atoms(self, two_chain_dist):
        assert two_chain_dist.num_atoms == 100  # 50 * 2 chains

    def test_annotations_record_the_method(self, two_chain_dist):
        assert two_chain_dist.annotations.attrs["method"] == "test"
        assert method_of(two_chain_dist) == "test"
        assert two_chain_dist.provenance.metadata["method"] == "test"

    def test_annotations_contains_arviz_data(self, two_chain_dist):
        assert two_chain_dist.annotations is not None
        # ArviZ-compatible data use xarray DataTree on ArviZ 1.x.
        assert hasattr(two_chain_dist.annotations, "groups") or hasattr(
            two_chain_dist.annotations, "children"
        )

    def test_arviz_data_accessor(self, two_chain_dist):
        aux = two_chain_dist.annotations
        assert aux is not None
        assert "arviz" in aux.children

        tree = arviz_data(two_chain_dist)
        assert tree is not None
        assert "posterior" in tree.children
        assert "warmup" in tree.children

    def test_no_arviz_groups_without_the_methods_annotations(self):
        dist = posterior_of(
            [jax.random.normal(jax.random.PRNGKey(0), (5, 2))],
            label="x",
        )
        assert "arviz" not in dist.annotations.children
        assert dist.annotations.attrs["method"] == "test"
        assert arviz_data(dist) is None

    def test_make_posterior_accepts_annotations_dataset_nodes(self):
        import xarray as xr

        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        posterior = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            annotations={"posterior": xr.Dataset()},
        )

        assert arviz_data(posterior) is not None
        assert "posterior" in arviz_data(posterior).children

    def test_make_posterior_skips_annotations_root_group(self):
        import xarray as xr

        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        posterior = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            annotations={
                "/": xr.Dataset(attrs={"ignored": True}),
                "posterior": xr.Dataset(),
            },
        )

        assert posterior.annotations is not None
        assert posterior.annotations.attrs == {"method": "test"}
        assert "arviz" in posterior.annotations.children
        assert "posterior" in arviz_data(posterior).children
        assert "" not in arviz_data(posterior).children

    def test_warmup_from_annotations(self, two_chain_dist):
        warmup = warmup_samples(two_chain_dist)
        assert warmup is not None
        assert len(warmup) == 2
        assert warmup[0].shape == (10, 2)

    def test_draws_single_chain(self, two_chain_dist):
        d = flat_draws(two_chain_dist, chain=0)
        assert d.shape == (50, 2)

    def test_draws_all_chains(self, two_chain_dist):
        d = flat_draws(two_chain_dist)
        assert d.shape == (100, 2)

    def test_draws_with_warmup(self, two_chain_dist):
        d = flat_draws(two_chain_dist, chain=0, include_warmup=True)
        assert d.shape == (60, 2)

    def test_draws_all_with_warmup(self, two_chain_dist):
        d = flat_draws(two_chain_dist, include_warmup=True)
        assert d.shape == (120, 2)

    def test_mean_and_variance(self, two_chain_dist):
        m = mean(two_chain_dist)
        v = variance(two_chain_dist)
        assert m.shape == (2,)
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(m))
        assert jnp.all(jnp.isfinite(v))

    def test_sample(self, two_chain_dist):
        s = sample(two_chain_dist)
        assert s.shape == (2,)

    def test_repr(self, two_chain_dist):
        r = repr(two_chain_dist)
        assert r.startswith("EmpiricalDistribution(")
        assert "levels={'chain': 2, 'draw': 50}" in r

    def test_make_posterior_forwards_weights(self):
        """make_posterior(weights=) flows through to the posterior so a
        weighted backend (e.g. SMC-ABC) keeps its importance weights and
        the weighted mean reflects them."""
        # Two particles at 0 and 10; weighting the second 0.8 pulls the
        # mean from the unweighted 5 to 0.2*0 + 0.8*10 = 8.
        chain = jnp.array([[0.0], [10.0]])
        prior = Normal(loc=0.0, scale=1.0, label="theta")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            weights=jnp.array([0.2, 0.8]),
        )
        assert post.weights is not None
        np.testing.assert_allclose(np.asarray(mean(post)).ravel(), [8.0], atol=1e-6)

    def test_make_posterior_weights_default_unweighted(self):
        """Without weights=, make_posterior yields an equal-weight posterior
        (the weighted-mean change is opt-in, not a default behaviour shift)."""
        chain = jnp.array([[0.0], [10.0]])
        prior = Normal(loc=0.0, scale=1.0, label="theta")
        post = make_posterior([chain], parents=(prior,), method="test")
        np.testing.assert_allclose(np.asarray(mean(post)).ravel(), [5.0], atol=1e-6)


class TestMakePosteriorRecordTarget:
    """A declared record target names the fields of the draws."""

    @pytest.fixture
    def template(self):
        return RecordSpec(r=(), K=(), phi=())

    @pytest.fixture
    def posterior_with_template(self, template):
        # 3 scalar params → flat draw vectors of size 3
        chain = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        return make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
        )

    def test_draws_returns_values(self, posterior_with_template):
        draws = flat_draws(posterior_with_template)
        assert isinstance(draws, NumericRecordBatch)
        # Insertion order from the template fixture: r, K, phi.
        assert tuple(draws.event_template.keys()) == ("r", "K", "phi")
        assert draws["r"].shape == (100,)

    def test_draws_has_correct_fields(self, posterior_with_template):
        draws = flat_draws(posterior_with_template)
        assert tuple(draws.event_template.keys()) == ("r", "K", "phi")

    def test_draws_field_shapes(self, posterior_with_template):
        draws = flat_draws(posterior_with_template)
        assert draws["r"].shape == (100,)
        assert draws["K"].shape == (100,)
        assert draws["phi"].shape == (100,)

    def test_draws_values_match_raw(self, posterior_with_template):
        """Named draws must contain the same data as raw flat draws."""
        raw = flat_draws(posterior_with_template)
        # Reconstruct flat from named (template insertion order).
        flat = jnp.stack([raw["r"], raw["K"], raw["phi"]], axis=-1)
        chain = flat_chains(posterior_with_template)[0]
        np.testing.assert_allclose(flat, chain, atol=1e-6)

    def test_draws_single_chain_returns_values(self, posterior_with_template):
        draws = flat_draws(posterior_with_template, chain=0)
        assert isinstance(draws, NumericRecordBatch)
        assert draws["r"].shape == (100,)

    def test_without_template_returns_array(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        post = make_posterior([chain], parents=(prior,), method="test")
        draws = flat_draws(post)
        assert isinstance(draws, jnp.ndarray)
        assert draws.shape == (50, 3)

    def test_the_target_names_the_fields(self, posterior_with_template, template):
        assert tuple(posterior_with_template.event_spec.components) == template.fields

    @pytest.mark.parametrize("shape", [(), (2,)])
    def test_a_one_field_target_declares_the_fields_shape(self, shape):
        """A posterior over a one-field record declares the field's shape, as its draws have it."""
        width = int(np.prod(shape))
        chains = [jnp.zeros((5, width)), jnp.ones((5, width))]
        post = make_posterior(chains, parents=(), method="test", event_spec=RecordSpec(theta=shape))
        assert post.event_spec.spec["theta"].shape == shape
        assert flat_draws(post)["theta"].shape == (10, *shape)
        assert jnp.shape(post._mean()["theta"]) == shape

    def test_field_order_reassembles_by_name(self):
        """field_order maps chain column-blocks to template fields by name.

        The chain's columns are laid out in a different order than the
        template (``b`` block, then scalar ``a``). Passing ``field_order``
        must reassemble each field from its own columns — not split the
        flat chain positionally in template order (which would scramble
        the draws).
        """
        template = RecordSpec(a=(), b=(2,))  # sizes: a=1, b=2
        # Columns laid out in field_order = (b, a): [b0, b1, a0].
        b_block = jnp.array([[10.0, 11.0], [12.0, 13.0]])  # (2, 2)
        a_block = jnp.array([[1.0], [2.0]])  # (2, 1)
        chain = jnp.concatenate([b_block, a_block], axis=-1)  # (2, 3)
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
            field_order=["b", "a"],
        )
        draws = flat_draws(post)
        # a is the trailing column; b is the leading 2-column block.
        np.testing.assert_allclose(np.asarray(draws["a"]), [1.0, 2.0])
        np.testing.assert_allclose(np.asarray(draws["b"]), b_block)

    def test_field_order_none_is_positional(self):
        """field_order=None keeps the historical positional layout."""
        template = RecordSpec(a=(), b=(2,))
        chain = jnp.array([[1.0, 10.0, 11.0], [2.0, 12.0, 13.0]])  # a, then b
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
        )
        draws = flat_draws(post)
        np.testing.assert_allclose(np.asarray(draws["a"]), [1.0, 2.0])
        np.testing.assert_allclose(np.asarray(draws["b"]), [[10.0, 11.0], [12.0, 13.0]])

    def test_field_order_must_be_permutation(self):
        """A field_order that isn't a permutation of template fields raises."""
        template = RecordSpec(a=(), b=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        with pytest.raises(ValueError, match="not a permutation"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                event_spec=template,
                field_order=["a", "c"],
            )

    def test_field_order_chain_too_wide_raises(self):
        """With field_order, a chain wider than the template's total flat
        size raises rather than silently dropping the extra columns in the
        permutation gather."""
        template = RecordSpec(a=(), b=())  # total flat size 2
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 3))  # 3 columns
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        with pytest.raises(ValueError, match="doesn't match"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                event_spec=template,
                field_order=["a", "b"],
            )

    def test_field_order_chain_too_narrow_raises(self):
        """With field_order, a chain narrower than the template's total
        flat size raises clearly rather than clamping the out-of-bounds
        gather indices."""
        template = RecordSpec(a=(), b=(), c=())  # total flat size 3
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 2))  # 2 columns
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        with pytest.raises(ValueError, match="doesn't match"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                event_spec=template,
                field_order=["a", "b", "c"],
            )

    def test_field_order_without_template_raises(self):
        """field_order without an event_spec is a caller error, not a
        silent no-op."""
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        with pytest.raises(ValueError, match="requires an event_spec"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                field_order=["a", "b"],
            )

    def test_field_order_single_field_invalid_permutation_raises(self):
        """field_order is validated even for a single-field template, so a
        wrong name is caught rather than silently ignored."""
        template = RecordSpec(a=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 1))
        prior = MultivariateNormal(loc=jnp.zeros(1), cov=jnp.eye(1), label="z")
        with pytest.raises(ValueError, match="not a permutation"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                event_spec=template,
                field_order=["b"],
            )

    def test_field_order_single_field_width_mismatch_raises(self):
        """With field_order, the chain width is validated for a
        single-field template too — not only for multi-field ones."""
        template = RecordSpec(a=(2,))  # flat size 2
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 3))  # 3 columns
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        with pytest.raises(ValueError, match="doesn't match"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                event_spec=template,
                field_order=["a"],
            )

    def test_field_order_opaque_template_raises_clear_error(self):
        """field_order cannot compute a permutation for opaque fields."""
        template = RecordSpec(a=OpaqueSpec(), b=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        with pytest.raises(ValueError, match="field 'a' has an opaque spec"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                event_spec=template,
                field_order=["a", "b"],
            )

    def test_multi_field_opaque_template_raises_clear_error(self):
        """Multi-field splitting rejects opaque fields before sizing."""
        template = RecordSpec(a=OpaqueSpec(), b=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        with pytest.raises(ValueError, match="field 'a' has an opaque spec"):
            make_posterior(
                [chain],
                parents=(prior,),
                method="test",
                event_spec=template,
            )

    def test_array_shaped_fields(self):
        """Template with non-scalar fields unflattens correctly."""
        template = RecordSpec(
            mean=(3,),
            cov=(2, 2),
        )
        vector_size = 3 + 4  # 3 + 2*2
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, vector_size))
        prior = MultivariateNormal(loc=jnp.zeros(vector_size), cov=jnp.eye(vector_size), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
        )
        draws = flat_draws(post)
        assert draws["mean"].shape == (20, 3)
        assert draws["cov"].shape == (20, 2, 2)

    def test_draws_with_warmup_and_template(self):
        """draws(include_warmup=True) returns Record when template is set."""
        template = RecordSpec(a=(), b=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        warmup = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
        annotations = build_mcmc_datatree([chain], warmup_chains=[warmup])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            annotations=annotations,
            event_spec=template,
        )
        draws = flat_draws(post, include_warmup=True)
        assert isinstance(draws, NumericRecordBatch)
        assert draws["a"].shape == (60,)  # 10 warmup + 50 draws
        assert draws["b"].shape == (60,)

    def test_nested_target_unflatten(self):
        """Nested Record template unflattens draws into nested structure."""
        template = RecordSpec(
            params=RecordSpec(a=(), b=()),
            scale=(),
        )
        vector_size = 3  # a + b + scale
        chain = jax.random.normal(jax.random.PRNGKey(0), (30, vector_size))
        prior = MultivariateNormal(loc=jnp.zeros(vector_size), cov=jnp.eye(vector_size), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
        )
        draws = flat_draws(post)
        assert isinstance(draws, NumericRecordBatch)
        assert isinstance(draws["params"], RecordBatch)
        assert draws["params/a"].shape == (30,)
        assert draws["params/b"].shape == (30,)
        assert draws["scale"].shape == (30,)

    def test_a_nested_posterior_keeps_its_nesting_for_views_and_kde(self):
        """A field view and a KDE of the posterior read its target record."""
        from probpipe import KDEDistribution, convert

        prior = (Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)).with_path_names(
            {"a": "params/a", "b": "params/b"}
        ) * Normal("s", 0.0, 1.0)
        chain = jax.random.normal(jax.random.PRNGKey(0), (40, 3))
        post = make_posterior([chain], parents=(prior,), method="test", event_spec=prior.event_spec)
        assert post["params/a"].event_spec.spec == prior.event_spec.spec.at_path(("params", "a"))
        assert convert(post, KDEDistribution).event_spec == prior.event_spec

    def test_a_nested_target_keeps_its_groups(self):
        """A posterior over a nested record declares, stores, and reports its groups nested."""
        template = RecordSpec(
            params=RecordSpec(a=(), b=()),
            scale=(),
        )
        vector_size = 3  # a + b + scale
        chain = jax.random.normal(jax.random.PRNGKey(0), (40, vector_size))
        prior = MultivariateNormal(
            loc=jnp.zeros(vector_size),
            cov=jnp.eye(vector_size),
            label="z",
        )
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
        )
        expected_fields = ("params", "scale")
        assert tuple(post.event_spec.components) == expected_fields
        assert post.event_spec == OutputSpec(template)
        # The atoms keep the nesting, in the chain's flat layout.
        np.testing.assert_allclose(post.atoms["params/b"][0], chain[:, 1])
        np.testing.assert_allclose(post.atoms["scale"][0], chain[:, 2])
        # A moment names each group in call form, as ``mean(params)``, and keeps its fields.
        from probpipe import mean as op_mean
        from probpipe import variance as op_variance

        m = op_mean(post)
        assert m.fields == ("mean(params)", "mean(scale)")
        assert jnp.shape(m["mean(params)/a"]) == ()
        assert jnp.shape(m["mean(scale)"]) == ()
        v = op_variance(post)
        assert v.fields == ("variance(params)", "variance(scale)")
        assert jnp.shape(v["variance(params)/b"]) == ()
        # ``draws()`` walks the full template, nesting included.
        draws = flat_draws(post)
        assert tuple(draws.event_template.children) == expected_fields
        assert draws["params/a"].shape == (40,)
        assert draws["params/b"].shape == (40,)
        assert draws["scale"].shape == (40,)

    def test_without_warmup(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = posterior_of([chain], label="x")
        assert warmup_samples(dist) is None
        assert num_chains(dist) == 1
        assert num_draws(dist) == 20

    def test_without_the_methods_groups_the_annotations_record_the_method(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = posterior_of([chain], label="x")
        assert dist.annotations.attrs["method"] == "test"
        assert arviz_data(dist) is None

    def test_annotations_has_posterior_group(self):
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        annotations = build_mcmc_datatree([chain])
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            annotations=annotations,
        )

        assert post.annotations is not None
        assert "arviz" in post.annotations.children

        idata = arviz_data(post)
        assert idata is not None
        assert "posterior" in idata.children

    def test_the_arviz_draws_are_named_by_leaf(self):
        """The ArviZ draw groups hold one variable per leaf, a nested path joined by dots."""
        template = RecordSpec(params=RecordSpec(a=(), b=()), scale=(3,))
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 5))
        warmup = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
        prior = MultivariateNormal(loc=jnp.zeros(5), cov=jnp.eye(5), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            annotations=build_mcmc_datatree([chain], warmup_chains=[warmup]),
            event_spec=template,
        )
        tree = arviz_data(post)
        for group, draws in (("posterior", chain), ("warmup", warmup)):
            variables = tree[group].data_vars
            assert list(variables) == ["params.a", "params.b", "scale"]
            np.testing.assert_allclose(variables["params.b"].values[0], draws[:, 1])
            np.testing.assert_allclose(variables["scale"].values[0], draws[:, 2:])
        assert tree["posterior"]["scale"].dims == ("chain", "draw", "scale_dim_0")

    def test_the_arviz_draws_of_a_whole_term_are_named_by_its_component(self):
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="theta")
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            annotations=build_mcmc_datatree([chain]),
            event_spec=prior.event_spec,
        )
        variable = arviz_data(post)["posterior"]["theta"]
        assert variable.dims == ("chain", "draw", "theta_dim_0")
        np.testing.assert_allclose(variable.values[0], chain)


# ---------------------------------------------------------------------------
# rwmh Function
# ---------------------------------------------------------------------------


class TestRWMH:
    """Test the standalone rwmh Function."""

    def test_basic_sampling(self):
        """RWMH samples from a simple Normal distribution."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        result = rwmh(
            dist=dist,
            num_results=100,
            num_warmup=50,
            step_size=0.5,
            random_seed=42,
        )
        assert isinstance(result, EmpiricalDistribution)
        assert num_draws(result) == 100
        assert num_chains(result) == 4
        assert result.event_shape == (2,)
        assert method_of(result) == "blackjax_rwmh"

    def test_inference_data_produced(self):
        """RWMH produces an annotations DataTree with posterior group."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert arviz_data(result) is not None
        assert "posterior" in arviz_data(result)
        # RWMH scalar stats (accept_rate, step_size) live in provenance,
        # not as per-draw arrays in sample_stats.
        assert result.provenance.metadata["accept_rate"] > 0
        assert result.provenance.metadata["step_size"] == 0.5

    def test_multi_chain(self):
        """RWMH with multiple chains."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            num_chains=3,
            step_size=0.5,
            random_seed=42,
        )
        assert num_chains(result) == 3
        assert num_draws(result) == 50
        assert result.num_atoms == 150  # 50 * 3

    def test_warmup_stored(self):
        """RWMH stores warmup samples."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert warmup_samples(result) is not None
        assert warmup_samples(result)[0].shape == (20, 2)

    def test_provenance(self):
        """RWMH attaches provenance."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=20,
            step_size=0.5,
            random_seed=42,
        )
        assert result.provenance is not None
        assert result.provenance.operation == "blackjax_rwmh"

    def test_with_log_prob_fn_normal_normal_conjugate(self):
        """RWMH posterior must recover the analytical Normal-Normal conjugate.

        Prior: params ~ N(0, sigma_p^2 I).
        Likelihood: y_i ~ N(params, sigma_y^2 I), i.i.d.
        Closed-form posterior:
            mean = (sigma_y^2 * 0 + n * sigma_p^2 * y_bar) / (sigma_y^2 + n * sigma_p^2)
            var  = (sigma_p^2 * sigma_y^2) / (sigma_y^2 + n * sigma_p^2)
        """
        sigma_p = np.sqrt(10.0)  # prior std
        sigma_y = 1.0  # likelihood std
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=sigma_p**2 * jnp.eye(2), label="params")
        data = jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])
        n = data.shape[0]

        def log_lik(params, data):
            return -0.5 / sigma_y**2 * jnp.sum((data - params) ** 2)

        result = rwmh(
            dist=prior,
            data=data,
            log_prob_fn=log_lik,
            num_results=8000,
            num_warmup=2000,
            step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, EmpiricalDistribution)

        # Analytical posterior.
        y_bar = np.asarray(jnp.mean(data, axis=0))
        denom = sigma_y**2 + n * sigma_p**2
        analytical_mean = (n * sigma_p**2 / denom) * y_bar
        analytical_var = (sigma_p**2 * sigma_y**2) / denom

        raw_draws = flat_draws(result)
        if hasattr(raw_draws, "fields"):
            raw_draws = jnp.concatenate([raw_draws[f] for f in raw_draws.event_template], axis=-1)
        draws = np.asarray(raw_draws).reshape(-1, 2)
        # MC standard error: posterior_sd / sqrt(effective_n).
        # ``adapt=True`` (the default) fits the proposal covariance from
        # warmup, so ``step_size=0.3`` is ignored for sampling and mixing
        # is better than a fixed-step chain would give. Assume n_eff ~ 300
        # conservatively — the tolerance is looser than the true effective
        # sample size warrants, which keeps the test robust.
        n_eff = 300
        mc_se_mean = 4.0 * np.sqrt(analytical_var / n_eff)
        np.testing.assert_allclose(draws.mean(0), analytical_mean, atol=mc_se_mean)
        mc_se_var = 4.0 * np.sqrt(2.0 * analytical_var**2 / (n_eff - 1))
        np.testing.assert_allclose(draws.var(0, ddof=1), [analytical_var] * 2, atol=mc_se_var)

    def test_requires_log_prob(self):
        """RWMH raises for distributions without SupportsLogProb and no conversion path."""
        from probpipe import NumericDistribution

        class NoLogProbNoSample(NumericDistribution):
            def __init__(self, label):
                super().__init__(label, NumericArraySpec((2,)))

        dist = NoLogProbNoSample(label="test")
        with pytest.raises(TypeError):
            rwmh(dist=dist, num_results=10, num_warmup=5)

    def test_custom_init(self):
        """RWMH actually starts the chain from the user-supplied ``init``.

        The target is ``N(0, I)``, so a chain that ignored ``init`` would
        sit within a few units of the origin from its very first draw. We
        start from a far-flung ``init=[20, 20]`` with ``num_warmup=0`` and
        ``adapt=False`` (so the proposal stays ``step_size * I`` and the
        chain has no warmup window to drift back toward the mode), then
        assert the first retained draw is still out near ``init`` rather
        than at the origin. A modest RWMH step from ``[20, 20]`` cannot
        reach the origin in one move, so this fails loudly if ``init`` is
        silently dropped.

        As a second, init-is-not-ignored guard we also confirm two
        *different* far-flung inits produce visibly different first draws.
        """
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        far_init = jnp.array([20.0, 20.0])
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=0,
            step_size=0.5,
            adapt=False,
            init=far_init,
            random_seed=42,
        )
        assert isinstance(result, EmpiricalDistribution)

        first = np.asarray(flat_chains(result)[0][0])
        # A chain seeded at the origin (init ignored) would land within a
        # few units of it; a step_size=0.5 RWMH move from [20, 20] stays
        # far out. Use a conservative band well clear of both regimes.
        assert np.linalg.norm(first - np.asarray(far_init)) < 5.0, (
            f"First draw {first} is not near init {far_init} — init may be ignored."
        )
        assert np.linalg.norm(first) > 10.0, (
            f"First draw {first} sits near the origin — init appears ignored."
        )

        # Two distinct inits must yield distinct early draws.
        other_init = jnp.array([-20.0, -20.0])
        result_other = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=0,
            step_size=0.5,
            adapt=False,
            init=other_init,
            random_seed=42,
        )
        first_other = np.asarray(flat_chains(result_other)[0][0])
        assert np.linalg.norm(first - first_other) > 1.0, (
            "Different inits produced near-identical first draws — init may be ignored."
        )

    def test_zero_warmup(self):
        """RWMH with num_warmup=0 stores no warmup samples."""
        dist = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        result = rwmh(
            dist=dist,
            num_results=50,
            num_warmup=0,
            step_size=0.5,
            random_seed=42,
        )
        assert warmup_samples(result) is None
        assert num_draws(result) == 50

    def test_non_supports_mean_init(self):
        """RWMH falls back to zeros init when dist has no SupportsMean."""
        from probpipe import NumericDistribution
        from probpipe.distributions._capabilities import SupportsLogProb

        class LogProbOnlyDist(NumericDistribution, SupportsLogProb):
            def __init__(self, label):
                super().__init__(label, NumericArraySpec((2,), "float32"))

            def _log_prob(self, value):
                return -0.5 * jnp.sum(value**2)

            def _prob(self, value):
                return jnp.exp(self._log_prob(value))

            def _unnormalized_log_prob(self, value):
                return self._log_prob(value)

            def _unnormalized_prob(self, value):
                return self._prob(value)

        dist = LogProbOnlyDist(label="test")
        result = rwmh(
            dist=dist,
            num_results=30,
            num_warmup=10,
            step_size=0.5,
            random_seed=42,
        )
        assert isinstance(result, EmpiricalDistribution)

    def test_mean_exception_fallback(self):
        """RWMH falls back to zeros init when _mean() raises."""
        from probpipe import NumericDistribution
        from probpipe.distributions._capabilities import SupportsLogProb, SupportsMean

        class BrokenMeanLogProbDist(NumericDistribution, SupportsLogProb, SupportsMean):
            def __init__(self, label):
                super().__init__(label, NumericArraySpec((2,), "float32"))

            def _log_prob(self, value):
                return -0.5 * jnp.sum(value**2)

            def _prob(self, value):
                return jnp.exp(self._log_prob(value))

            def _unnormalized_log_prob(self, value):
                return self._log_prob(value)

            def _unnormalized_prob(self, value):
                return self._prob(value)

            def _mean(self):
                raise RuntimeError("broken")

        dist = BrokenMeanLogProbDist(label="test")
        result = rwmh(
            dist=dist,
            num_results=30,
            num_warmup=10,
            step_size=0.5,
            random_seed=42,
        )
        assert isinstance(result, EmpiricalDistribution)


# ---------------------------------------------------------------------------
# FieldView + select
# ---------------------------------------------------------------------------


class TestPosteriorFieldView:
    """Field views of a posterior over a record."""

    @pytest.fixture
    def template(self):
        return RecordSpec(K=(), phi=(), r=())

    @pytest.fixture
    def posterior(self, template):
        chain = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        return make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
        )

    def test_getitem_returns_view(self, posterior):
        view = posterior["r"]
        assert isinstance(view, FieldView)
        assert view.parent is posterior

    def test_getitem_missing_field_raises(self, posterior):
        with pytest.raises(KeyError, match="nonexistent"):
            posterior["nonexistent"]

    def test_getitem_without_template_uses_single_field_autowrap(self):
        """Without a multi-field template, an inference result
        wraps the chain as a single-field Record keyed by its label.
        Indexing the field returns a view; accessing a different name
        raises ``KeyError``."""
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = posterior_of([chain], label="x")
        # The auto-wrap field is "x"; that should resolve to a view.
        view = dist["x"]
        assert view is not None
        # Other names raise.
        with pytest.raises(KeyError):
            dist["nonexistent"]

    def test_components(self, posterior):
        assert tuple(posterior.event_spec.components) == ("K", "phi", "r")

    def test_a_posterior_without_a_target_is_a_whole_term(self):
        """Without a target, each draw is one array under the result's label."""
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
        dist = posterior_of([chain], label="x")
        assert tuple(dist.event_spec.components) == ("x",)
        assert dist.event_shape == (3,)

    def test_view_event_shape_scalar(self, posterior):
        view = posterior["r"]
        assert view.event_shape == ()

    def test_view_event_shape_vector(self):
        template = RecordSpec(vec=(5,), scalar=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 6))
        prior = MultivariateNormal(loc=jnp.zeros(6), cov=jnp.eye(6), label="z")
        post = make_posterior([chain], parents=(prior,), method="test", event_spec=template)
        assert post["scalar"].event_shape == ()
        assert post["vec"].event_shape == (5,)

    def test_view_mean(self, posterior):
        view = posterior["K"]
        draws = flat_draws(posterior)
        np.testing.assert_allclose(float(view._mean()), float(jnp.mean(draws["K"])), atol=1e-5)

    def test_view_variance(self, posterior):
        view = posterior["K"]
        draws = flat_draws(posterior)
        np.testing.assert_allclose(float(view._variance()), float(jnp.var(draws["K"])), atol=1e-5)

    def test_view_sample(self, posterior):
        view = posterior["r"]
        s = view._sample(jax.random.PRNGKey(42), (10,))
        assert s.shape == (10,)

    def test_repr(self, posterior):
        view = posterior["r"]
        r = repr(view)
        assert "FieldView" in r
        assert repr(posterior.label) in r
        assert "'r'" in r

    def test_view_mean_fallback_without_supports_mean(self):
        """_mean() falls back to _field_draws() when parent lacks SupportsMean."""
        # An inference result IS SupportsMean, so we test the fallback
        # by checking the empirical mean matches the draws directly.
        template = RecordSpec(a=(), b=())
        chain = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        post = make_posterior([chain], parents=(prior,), method="test", event_spec=template)
        view = post["a"]
        # Mean of column 0 (field "a"): (1+3+5)/3 = 3.0
        np.testing.assert_allclose(float(view._mean()), 3.0, atol=1e-5)
        # Variance of column 0: var([1,3,5]) = 8/3
        np.testing.assert_allclose(float(view._variance()), jnp.var(chain[:, 0]), atol=1e-5)

    def test_a_view_keeps_the_posterior_label_and_exposes_its_field(self, posterior):
        """A view keeps its parent's label, and its event exposes the selected field."""
        for field in ("K", "phi", "r"):
            assert posterior[field].label == posterior.label
            assert list(posterior[field].event_spec.components) == [field]

    def test_a_view_of_a_factored_joint_keeps_the_joint_label(self):
        """The views of a factored joint keep its label, and their events expose the fields."""
        p = Normal(loc=0.0, scale=1.0, label="x") * Normal(loc=0.0, scale=1.0, label="y")
        assert p["x"].label == p["y"].label == p.label
        assert list(p["x"].event_spec.components) == ["x"]


class TestViewProtocolDuckTyping:
    """A field view claims the capabilities its parent's derive.

    A joint's view claims the density its marginal at the path reports, and a
    posterior, which has no density, gives its views none.
    """

    def test_a_joint_view_claims_the_density_of_its_factor(self):
        from probpipe import SupportsLogProb

        joint = Normal("x", 0, 1) * Normal("y", 3, 2)
        view = joint["x"]
        assert isinstance(view, SupportsLogProb)

    def test_view_from_posterior_not_isinstance_log_prob(self):
        """An inference result lacks SupportsLogProb → view doesn't have it."""
        from probpipe import SupportsLogProb

        template = RecordSpec(a=(), b=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), label="z")
        post = make_posterior([chain], parents=(prior,), method="test", event_spec=template)
        view = post["a"]
        assert not isinstance(view, SupportsLogProb)

    def test_view_always_isinstance_sampling(self):
        """A view samples when its parent does."""
        from probpipe import SupportsSampling

        joint = Normal("x", 0, 1) * Normal("y", 3, 2)
        assert isinstance(joint["x"], SupportsSampling)

        template = RecordSpec(a=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 1))
        prior = Normal("x", 0, 1)
        post = make_posterior([chain], parents=(prior,), method="test", event_spec=template)
        assert isinstance(post["a"], SupportsSampling)

    def test_view_always_isinstance_mean_variance(self):
        """A view of a parent with moments has them."""
        from probpipe import SupportsMean, SupportsVariance

        joint = Normal("x", 0, 1) * Normal("y", 3, 2)
        view = joint["x"]
        assert isinstance(view, SupportsMean)
        assert isinstance(view, SupportsVariance)

    def test_view_log_prob_delegates_to_component(self):
        """A joint view's density is its factor's, the joint's marginal at the path."""
        import scipy.stats

        joint = Normal(loc=2.0, scale=0.5, label="x") * Normal("y", 0, 1)
        view = joint["x"]
        lp = float(view._log_prob(jnp.array(2.0)))
        expected = scipy.stats.norm.logpdf(2.0, loc=2.0, scale=0.5)
        np.testing.assert_allclose(lp, expected, rtol=1e-5)

    def test_view_covariance_follows_its_parent(self):
        """A view claims a covariance exactly when its parent does."""
        from probpipe import SupportsCovariance

        joint = Normal("x", 0, 1) * Normal("y", 3, 2)
        assert isinstance(joint["x"], SupportsCovariance) == isinstance(joint, SupportsCovariance)

    def test_dynamic_protocol_depends_on_parent(self):
        """One FieldView class, different claims by parent."""
        from probpipe import SupportsLogProb

        joint = Normal("x", 0, 1) * Normal("y", 3, 2)
        view_with = joint["x"]
        assert isinstance(view_with, SupportsLogProb)

        template = RecordSpec(a=())
        chain = jax.random.normal(jax.random.PRNGKey(0), (20, 1))
        prior = Normal("x", 0, 1)
        post = make_posterior([chain], parents=(prior,), method="test", event_spec=template)
        view_without = post["a"]
        assert not isinstance(view_without, SupportsLogProb)

    def test_view_still_isinstance_base_class(self):
        """A view of any parent is a FieldView."""
        joint = Normal("x", 0, 1) * Normal("y", 3, 2)
        assert isinstance(joint["x"], FieldView)


class TestValuesSelect:
    """Record.select() for concrete data."""

    def test_positional(self):
        v = Record("r", r=1.0, K=70.0, phi=10.0)
        sel = v.select("r", "K")
        assert set(sel.keys()) == {"r", "K"}
        np.testing.assert_allclose(float(sel["r"]), 1.0)
        np.testing.assert_allclose(float(sel["K"]), 70.0)

    def test_keyword_remap(self):
        v = Record("r", r=1.0, K=70.0)
        sel = v.select(growth_rate="r")
        assert "growth_rate" in sel
        np.testing.assert_allclose(float(sel["growth_rate"]), 1.0)

    def test_mixed(self):
        v = Record("r", r=1.0, K=70.0, phi=10.0)
        sel = v.select("phi", growth_rate="r")
        assert set(sel.keys()) == {"phi", "growth_rate"}

    def test_missing_field_raises(self):
        v = Record("r", r=1.0)
        with pytest.raises(KeyError, match="nonexistent"):
            v.select("nonexistent")

    def test_missing_mapping_target_raises(self):
        v = Record("r", r=1.0)
        with pytest.raises(KeyError, match="z"):
            v.select(x="z")

    def test_empty_select(self):
        v = Record("r", r=1.0, K=70.0)
        sel = v.select()
        assert sel == {}


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


class TestEndToEndValuesPipeline:
    """Full pipeline: named prior → inference → named draws → views → broadcasting.

    Validates correctness at every step, not just types and shapes.
    """

    @pytest.fixture
    def posterior(self):
        """Run inference once for all end-to-end tests."""
        import tensorflow_probability.substrates.jax.distributions as tfd

        from probpipe import condition_on
        from tests.inference.canonical import ObservationKernel

        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10, label="params")
        likelihood = ObservationKernel(
            "y",
            {"params": prior.event_spec.spec},
            NumericArraySpec((2,)),
            lambda params: tfd.Independent(tfd.Normal(params, 1.0), 1),
        )
        return condition_on.with_options(
            method_options={
                "num_results": 500,
                "num_warmup": 200,
                "step_size": 0.3,
                "random_seed": 42,
            }
        )(likelihood * prior, {"y": jnp.array([1.0, 2.0])})

    def test_template_propagation(self, posterior):
        """The posterior is a law over the joint's unconditioned field, a record of params."""
        assert tuple(posterior.event_spec.components) == ("params",)
        assert posterior.event_spec.components["params"].shape == (2,)
        assert posterior.event_spec.components["params"].dtype == jnp.asarray(0.0).dtype

    def test_draws_are_named_values(self, posterior):
        """draws() returns Record with correct field names and shapes."""
        draws = flat_draws(posterior)
        assert isinstance(draws, NumericRecordBatch)
        assert tuple(draws.event_template.keys()) == ("params",)
        # Four chains of 500 draws.
        assert draws["params"].shape == (4 * 500, 2)

    def test_draws_values_correct(self, posterior):
        """Posterior mean and std match analytical conjugate values."""
        # Prior N(0, 10I) + likelihood N(data, I), data=[1,2], n=1
        # Posterior mean = sigma_prior^2 / (sigma_lik^2 + sigma_prior^2) * data
        #                = 10/11 * [1, 2] ≈ [0.909, 1.818]
        # Posterior var  = sigma_prior^2 * sigma_lik^2 / (sigma_lik^2 + sigma_prior^2)
        #                = 10/11 ≈ 0.909
        draws = flat_draws(posterior)
        post_mean = np.asarray(draws["params"].raw().mean(axis=0))
        post_std = np.asarray(draws["params"].raw().std(axis=0))
        analytical_mean = np.array([10 / 11, 20 / 11])
        analytical_std = np.sqrt(10 / 11)
        np.testing.assert_allclose(post_mean, analytical_mean, atol=0.15)
        np.testing.assert_allclose(post_std, analytical_std, atol=0.15)

    def test_view_values_match_draws(self, posterior):
        """The view of the field is a law whose mean matches the draws."""
        view = posterior["params"]
        assert isinstance(view, FieldView)
        assert view.event_shape == (2,)

        # Delegation check: view._mean() == draws().params.mean()
        draws = flat_draws(posterior)
        np.testing.assert_allclose(
            np.asarray(view._mean()),
            np.asarray(draws["params"].raw().mean(axis=0)),
            atol=1e-5,
        )
        # Analytical check: view._mean() near analytical posterior mean
        np.testing.assert_allclose(
            np.asarray(view._mean()),
            np.array([10 / 11, 20 / 11]),
            atol=0.15,
        )

    def test_workflow_broadcasting_values_correct(self, posterior):
        """Broadcast predict(params, x) computes correct function of posterior."""
        from probpipe.functions import function

        @function(n_broadcast_samples=100, dispatch="sequential")
        def predict(params, x):
            return params[0] + params[1] * x

        with workflow_run(seed=0):
            result = predict(params=posterior["params"], x=0.5)
        assert result.num_atoms == 100
        # predict([~0.91, ~1.82], 0.5) ≈ 0.91 + 1.82*0.5 ≈ 1.82
        analytical = 10 / 11 + 0.5 * 20 / 11
        np.testing.assert_allclose(float(mean(result)), analytical, atol=0.2)

    def test_workflow_broadcasting_preserves_correlation(self, posterior):
        """Two views from same posterior sample jointly (not independently).

        Both mean AND variance of a-b must be ~0.  An independent-sampling
        bug would produce mean≈0 (by symmetry) but var≈2*var(params),
        so checking var is the real correlation test.
        """
        from probpipe.functions import function

        @function(n_broadcast_samples=50, dispatch="sequential")
        def identity_pair(a, b):
            return a - b

        with workflow_run(seed=0):
            result = identity_pair(a=posterior["params"], b=posterior["params"])
        # Mean check: necessary but insufficient
        np.testing.assert_allclose(np.asarray(mean(result)), 0.0, atol=1e-5)
        # Variance check: this is what actually validates correlation
        np.testing.assert_allclose(np.asarray(variance(result)), 0.0, atol=1e-5)

    def test_multi_field_posterior(self):
        """Posterior with multiple named scalar fields."""
        template = RecordSpec(a=(), b=(), c=())
        # 3 scalar fields → flat draw vectors of size 3
        chain = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        prior = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), label="z")
        post = make_posterior(
            [chain],
            parents=(prior,),
            method="test",
            event_spec=template,
        )
        draws = flat_draws(post)
        assert isinstance(draws, NumericRecordBatch)
        assert tuple(draws.event_template.keys()) == ("a", "b", "c")
        assert draws["a"].shape == (200,)

        # Per-field views
        view_a = post["a"]
        assert isinstance(view_a, FieldView)
        np.testing.assert_allclose(float(view_a._mean()), float(draws["a"].raw().mean()), atol=1e-5)

    def test_workflow_mixed_posterior_and_independent(self, posterior):
        """Workflow with both posterior views and an independent distribution."""
        from probpipe.functions import function

        @function(n_broadcast_samples=posterior.num_atoms, dispatch="sequential")
        def noisy_predict(params, noise):
            return params[0] + params[1] * 0.5 + noise

        params = np.asarray(flat_draws(posterior)["params"])
        expected_values = params[:, 0] + params[:, 1] * 0.5
        expected_mean = float(np.mean(expected_values))
        expected_variance = float(np.var(expected_values) + 0.01**2)

        with workflow_run(seed=0):
            result = noisy_predict(
                params=posterior["params"],
                noise=Normal("noise", 0, 0.01),
            )
        assert result.num_atoms == posterior.num_atoms
        # Across workflow seeds 0-15, mean errors were 0.000003-0.000526 and
        # variance errors were 0.000010-0.002307 against the materialized posterior.
        np.testing.assert_allclose(float(mean(result)), expected_mean, rtol=0.0, atol=0.003)
        np.testing.assert_allclose(
            float(variance(result)),
            expected_variance,
            rtol=0.0,
            atol=0.01,
        )
