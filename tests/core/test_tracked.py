"""Contract tests for the ``Tracked`` / ``Annotated`` identity mixins.

Asserts the identity-and-metadata contract shared by every tracked term:
``name`` / ``name_is_auto`` semantics (user-given vs. auto-derived),
``with_name`` copy semantics, ``with_provenance`` write-once behaviour, and
the ``annotations`` store.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import pytest

import probpipe
from probpipe import (
    EmpiricalDistribution,
    Normal,
    NumericRecord,
    NumericRecordArray,
    ProductDistribution,
    Provenance,
    ProvenanceMode,
    Record,
    RecordArray,
)
from probpipe.core.event_template import EventTemplate
from probpipe.core.tracked import Annotated, Tracked, auto_name

# ===========================================================================
# 1. Mixin membership — every core object is a tracked term
# ===========================================================================


class TestMixinMembership:
    def test_distribution_is_tracked_and_annotated(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        assert isinstance(n, Tracked)
        assert isinstance(n, Annotated)

    def test_record_is_tracked_and_annotated(self):
        r = Record(a=1.0)
        assert isinstance(r, Tracked)
        assert isinstance(r, Annotated)

    def test_numeric_record_is_tracked_and_annotated(self):
        nr = NumericRecord(a=jnp.array(1.0))
        assert isinstance(nr, Tracked)
        assert isinstance(nr, Annotated)

    def test_record_array_is_tracked(self):
        ra = RecordArray(
            {"a": jnp.zeros((3,))},
            batch_shape=(3,),
            template=EventTemplate(a=()),
        )
        assert isinstance(ra, Tracked)

    def test_distribution_array_is_tracked(self):
        da = Normal.from_batched_params(loc=jnp.zeros(3), scale=1.0, name="batch")
        assert isinstance(da, Tracked)


# ===========================================================================
# 2. name_is_auto — user-given vs. auto-derived
# ===========================================================================


class TestNameEnforcement:
    def test_tracked_host_must_set_nonempty_name(self):
        """The construction-time name check lives on Tracked, not per host."""

        class Nameless(Tracked):
            def __init__(self):
                pass

        with pytest.raises(TypeError, match="non-empty name"):
            Nameless()

        class EmptyNamed(Tracked):
            def __init__(self):
                self._init_tracked("")

        with pytest.raises(TypeError, match="non-empty name"):
            EmptyNamed()


class TestAutoNameHelper:
    def test_supplied_name_is_user_given(self):
        assert auto_name("mine", "default") == ("mine", False)

    def test_missing_name_takes_default_as_auto(self):
        assert auto_name(None, "default") == ("default", True)


class TestNameIsAuto:
    def test_user_named_distribution_is_not_auto(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        assert n.name == "x"
        assert n.name_is_auto is False

    def test_user_named_record_is_not_auto(self):
        r = Record(a=1.0, name="mine")
        assert r.name == "mine"
        assert r.name_is_auto is False

    def test_unnamed_record_is_auto(self):
        r = Record(a=1.0, b=2.0)
        assert r.name == "record(a,b)"
        assert r.name_is_auto is True

    def test_unnamed_record_array_is_auto(self):
        ra = RecordArray(
            {"a": jnp.zeros((3,))},
            batch_shape=(3,),
            template=EventTemplate(a=()),
        )
        assert ra.name_is_auto is True
        named = RecordArray(
            {"a": jnp.zeros((3,))},
            batch_shape=(3,),
            template=EventTemplate(a=()),
            name="mine",
        )
        assert named.name_is_auto is False

    def test_unnamed_composite_distribution_is_auto(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=0.0, scale=1.0, name="sigma"),
        )
        assert joint.name == "product(mu,sigma)"
        assert joint.name_is_auto is True

    def test_named_composite_distribution_is_not_auto(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            name="my_joint",
        )
        assert joint.name_is_auto is False

    def test_unnamed_empirical_is_auto(self):
        # Opaque (object) samples take the base EmpiricalDistribution path,
        # which auto-derives the name "empirical" when none is given.
        emp = EmpiricalDistribution(["heads", "tails", "heads"])
        assert emp.name == "empirical"
        assert emp.name_is_auto is True

    def test_name_is_auto_survives_record_pickle(self):
        auto = Record(a=1.0)
        named = Record(a=1.0, name="mine")
        assert pickle.loads(pickle.dumps(auto)).name_is_auto is True
        assert pickle.loads(pickle.dumps(named)).name_is_auto is False


# ===========================================================================
# 3. with_name — rename-as-copy semantics
# ===========================================================================


class TestWithName:
    def test_with_name_returns_copy_original_unchanged(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        m = n.with_name("y")
        assert m is not n
        assert m.name == "y"
        assert n.name == "x"

    def test_with_name_clears_auto_flag(self):
        r = Record(a=1.0)  # auto-named
        assert r.name_is_auto is True
        r2 = r.with_name("mine")
        assert r2.name == "mine"
        assert r2.name_is_auto is False
        # the original keeps its auto flag
        assert r.name_is_auto is True

    def test_with_name_records_provenance(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        m = n.with_name("y")
        assert m.provenance is not None
        assert m.provenance.operation == "with_name"
        assert m.provenance.metadata == {"old_name": "x", "new_name": "y"}
        assert m.provenance.parents[0].name == "x"

    def test_with_name_on_immutable_record(self):
        r = Record(a=jnp.array(1.0), b=jnp.array(2.0), name="orig")
        r2 = r.with_name("new")
        assert r2.name == "new"
        # shallow copy: field data is shared, not copied
        assert r2["a"] is r["a"]
        assert r2.event_template is r.event_template
        assert r == r2 or r2["b"] is r["b"]

    def test_with_name_rejects_empty_or_non_string(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        with pytest.raises(TypeError, match="non-empty string"):
            n.with_name("")
        with pytest.raises(TypeError, match="non-empty string"):
            n.with_name(3)  # type: ignore[arg-type]

    def test_with_name_off_mode_attaches_no_provenance(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal(loc=0.0, scale=1.0, name="x")
        m = n.with_name("y")
        assert m.name == "y"
        assert m.provenance is None

    def test_with_name_decouples_annotations_container(self):
        # Post-rename annotation writes must not show through on the
        # original (the container is copied; entry values are shared).
        n = Normal(loc=0.0, scale=1.0, name="x")
        n._annotations = {"fit": "exact"}
        m = n.with_name("y")
        m.annotations["check"] = "added-on-copy"
        assert "check" not in n.annotations
        assert m.annotations["fit"] == "exact"

    def test_with_name_decouples_datatree_annotations(self):
        xr = pytest.importorskip("xarray")
        n = Normal(loc=0.0, scale=1.0, name="x")
        n._annotations = xr.DataTree.from_dict({"arviz": xr.Dataset()})
        m = n.with_name("y")
        m.annotations["diagnostics"] = xr.DataTree()
        assert "diagnostics" not in n.annotations.children
        assert "arviz" in m.annotations.children

    def test_with_name_after_provenance_starts_fresh_chain(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n.with_provenance(Provenance("first"))
        m = n.with_name("y")
        # the clone's provenance is the rename, not the original's chain
        assert m.provenance.operation == "with_name"
        # and the original's chain is reachable through the parent descriptor
        assert m.provenance.parents[0].provenance is n.provenance


# ===========================================================================
# 3b. with_name on hosts with argument-taking __new__ (views, routers)
# ===========================================================================


class TestWithNameOnCustomNewHosts:
    """with_name must work on every Tracked host, including classes whose
    __new__ takes required arguments (dynamic class selection / views)."""

    def test_transformed_distribution(self):
        import tensorflow_probability.substrates.jax.bijectors as tfb

        from probpipe import TransformedDistribution

        t = TransformedDistribution(Normal(loc=0.0, scale=1.0, name="x"), tfb.Exp())
        t2 = t.with_name("y")
        assert t2.name == "y"
        assert t2.name_is_auto is False
        key = jax.random.PRNGKey(0)
        assert jnp.allclose(jnp.asarray(t._sample(key, (5,))), jnp.asarray(t2._sample(key, (5,))))

    def test_flattened_distribution_view(self):
        from probpipe import MultivariateNormal

        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="theta")
        flat = mvn.as_flat_distribution()
        renamed = flat.with_name("theta_flat")
        assert renamed.name == "theta_flat"
        assert renamed.name_is_auto is False
        assert renamed.event_shape == flat.event_shape

    def test_record_distribution_view(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
        )
        view = joint["mu"]
        renamed = view.with_name("mu_view")
        assert renamed.name == "mu_view"
        assert renamed.name_is_auto is False

    def test_empirical_router(self):
        emp = EmpiricalDistribution(["a", "b", "c"])
        renamed = emp.with_name("labels")
        assert renamed.name == "labels"
        assert renamed.name_is_auto is False
        assert emp.name_is_auto is True


# ===========================================================================
# 3c. name_is_auto propagation through derived objects
# ===========================================================================


class TestFlagPropagation:
    """Operations that build a new object from a parent keep the identity
    flag consistent with where the name actually came from."""

    def test_minibatched_distribution_default_name_is_auto(self):
        import tensorflow_probability.substrates.jax.glm as tfp_glm

        from probpipe import MultivariateNormal
        from probpipe.inference._minibatch import MinibatchedDistribution
        from probpipe.modeling import GLMLikelihood

        X = jnp.eye(4)
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        prior = MultivariateNormal(loc=jnp.zeros(4), cov=jnp.eye(4), name="theta")
        lik = GLMLikelihood(tfp_glm.Bernoulli(), x=X)
        m = MinibatchedDistribution(prior, lik, Record(X=X, y=y), batch_size=2)
        assert m.name == "MinibatchedDistribution(batch_size=2)"
        assert m.name_is_auto is True
        named = MinibatchedDistribution(prior, lik, Record(X=X, y=y), batch_size=2, name="mine")
        assert named.name_is_auto is False

    def test_product_conditioning_mirrors_flag(self):
        auto_joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
        )
        cond = auto_joint._condition_on(mu=0.5)
        assert cond.name_is_auto is True
        named_joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
            name="my_joint",
        )
        cond_named = named_joint._condition_on(mu=0.5)
        assert cond_named.name_is_auto is False

    def test_distribution_array_slice_mirrors_flag(self):
        da = Normal.from_batched_params(loc=jnp.zeros(4), scale=1.0, name="batch")
        assert da.name_is_auto is False
        assert da[0:2].name_is_auto is False
        # An auto-named array (e.g. a sweep product) keeps auto on slices.
        object.__setattr__(da, "_name_is_auto", True)
        assert da[0:2].name_is_auto is True

    def test_from_batched_params_cells_are_auto(self):
        da = Normal.from_batched_params(loc=jnp.zeros(3), scale=1.0, name="x")
        cell = da[0]
        assert cell.name == "x_0"
        assert cell.name_is_auto is True

    def test_full_factorial_design_name_is_auto(self):
        from probpipe.record import FullFactorialDesign

        design = FullFactorialDesign(a=jnp.arange(2.0), b=jnp.arange(3.0))
        assert design.name.startswith("FullFactorialDesign")
        assert design.name_is_auto is True


# ===========================================================================
# 4. with_provenance — write-once
# ===========================================================================


class TestWithProvenance:
    def test_returns_self_for_chaining(self):
        r = Record(a=1.0)
        assert r.with_provenance(Provenance("op")) is r
        assert r.provenance.operation == "op"

    def test_none_is_noop(self):
        r = Record(a=1.0)
        assert r.with_provenance(None) is r
        assert r.provenance is None

    def test_write_once_raises(self):
        for obj in (Record(a=1.0), Normal(loc=0.0, scale=1.0, name="x")):
            obj.with_provenance(Provenance("first"))
            with pytest.raises(RuntimeError, match="write-once"):
                obj.with_provenance(Provenance("second"))


# ===========================================================================
# 5. Annotated — the annotations store
# ===========================================================================


class TestAnnotated:
    def test_annotations_default_none(self):
        assert Record(a=1.0).annotations is None
        assert Normal(loc=0.0, scale=1.0, name="x").annotations is None

    def test_annotations_accepts_plain_mapping(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        n._annotations = {"note": "fitted by hand"}
        assert n.annotations == {"note": "fitted by hand"}

    def test_annotations_accepts_datatree(self):
        xr = pytest.importorskip("xarray")
        n = Normal(loc=0.0, scale=1.0, name="x")
        n._annotations = xr.DataTree.from_dict({"diagnostics": xr.Dataset()})
        assert "diagnostics" in n.annotations.children

    def test_annotations_on_record_via_object_setattr(self):
        # Record is immutable; the annotations channel is written by
        # framework code via object.__setattr__.
        r = Record(a=1.0)
        object.__setattr__(r, "_annotations", {"k": 1})
        assert r.annotations == {"k": 1}


# ===========================================================================
# 6. Batch element types round-trip identity state
# ===========================================================================


class TestProductPickleRoundTrip:
    def test_auto_named_product_keeps_auto_flag(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
        )
        assert joint.name_is_auto is True
        back = pickle.loads(pickle.dumps(joint))
        assert back.name == joint.name
        assert back.name_is_auto is True

    def test_user_named_product_keeps_identity_and_provenance(self):
        joint = ProductDistribution(mu=Normal(loc=0.0, scale=1.0, name="mu"), name="my_joint")
        joint.with_provenance(Provenance("op"))
        back = pickle.loads(pickle.dumps(joint))
        assert back.name == "my_joint"
        assert back.name_is_auto is False
        assert back.provenance.operation == "op"


class TestBatchPickleRoundTrip:
    def test_numeric_record_array_pickle_preserves_identity(self):
        nra = NumericRecordArray(
            {"a": jnp.zeros((3,))},
            batch_shape=(3,),
            template=EventTemplate(a=()),
            name="mine",
        )
        nra.with_provenance(Provenance("op"))
        back = pickle.loads(pickle.dumps(nra))
        assert back.name == "mine"
        assert back.name_is_auto is False
        assert back.provenance.operation == "op"
