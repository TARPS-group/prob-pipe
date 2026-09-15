"""Contract tests for the ``TrackedTerm`` / ``Annotated`` identity mixins.

Asserts the identity-and-metadata contract shared by every tracked term:
construction-time names and preservation through transforms,
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
    NumericRecordBatch,
    ProductDistribution,
    Provenance,
    ProvenanceMode,
    Record,
    RecordBatch,
)
from probpipe.core.event_template import EventTemplate
from probpipe.core.tracked import Annotated, TrackedTerm, auto_name

# ===========================================================================
# 1. Mixin membership — every core object is a tracked term
# ===========================================================================


class TestMixinMembership:
    def test_distribution_is_tracked_and_annotated(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        assert isinstance(n, TrackedTerm)
        assert isinstance(n, Annotated)

    def test_record_is_tracked_and_annotated(self):
        r = Record("r", a=1.0)
        assert isinstance(r, TrackedTerm)
        assert isinstance(r, Annotated)

    def test_numeric_record_is_tracked_and_annotated(self):
        nr = NumericRecord("nr", a=jnp.array(1.0))
        assert isinstance(nr, TrackedTerm)
        assert isinstance(nr, Annotated)

    def test_record_batch_is_tracked(self):
        ra = RecordBatch(
            "batch",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=EventTemplate(a=()),
        )
        assert isinstance(ra, TrackedTerm)

    def test_distribution_array_is_tracked(self):
        da = Normal.from_batched_params(loc=jnp.zeros(3), scale=1.0, name="batch")
        assert isinstance(da, TrackedTerm)


# ===========================================================================
# 2. Construction-time names
# ===========================================================================


class TestNameEnforcement:
    def test_tracked_host_must_set_nonempty_name(self):
        """The construction-time name check lives on TrackedTerm, not per host."""

        class Nameless(TrackedTerm):
            def __init__(self):
                pass

        with pytest.raises(TypeError, match="non-empty name"):
            Nameless()

        class EmptyNamed(TrackedTerm):
            def __init__(self):
                self._init_tracked("")

        with pytest.raises(TypeError, match="non-empty name"):
            EmptyNamed()


class TestAutoNameHelper:
    def test_supplied_name_is_user_given(self):
        assert auto_name("mine", "default") == "mine"

    def test_missing_name_takes_default(self):
        assert auto_name(None, "default") == "default"


class TestNameLifecycle:
    def test_distribution_keeps_explicit_name(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        assert n.name == "x"

    def test_record_keeps_explicit_name(self):
        r = Record("mine", a=1.0)
        assert r.name == "mine"

    def test_constructor_requires_name(self):
        # The name guard fires in ``Record.__new__`` before promotion picks a
        # class, so a name-less call reports ``Record`` and its custom message
        # — not the promoted ``NumericRecord`` nor the bare Python "missing
        # positional argument" a reverted guard would leave.
        with pytest.raises(TypeError, match="Record requires its name"):
            Record(a=1.0)

    def test_record_keeps_operation_name(self):
        r = Record("sample", {"a": 1.0, "b": 2.0})
        assert r.name == "sample"

    def test_batch_keeps_its_construction_name(self):
        """A caller that derives a name says so; there is no unnamed batch."""
        ra = RecordBatch(
            "derived",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=EventTemplate(a=()),
        )
        assert ra.name == "derived"
        named = RecordBatch(
            "mine",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=EventTemplate(a=()),
        )
        assert named.name == "mine"

    def test_composite_distribution_derives_default_name(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=0.0, scale=1.0, name="sigma"),
        )
        assert joint.name == "product(mu,sigma)"

    def test_composite_distribution_keeps_explicit_name(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            name="my_joint",
        )
        assert joint.name == "my_joint"

    def test_empirical_derives_default_name(self):
        # Opaque (object) samples take the base EmpiricalDistribution path,
        # which auto-derives the name "empirical" when none is given.
        emp = EmpiricalDistribution(["heads", "tails", "heads"])
        assert emp.name == "empirical"

    @pytest.mark.parametrize("name", [None, "mine"], ids=["derived", "supplied"])
    @pytest.mark.parametrize(
        "transform, expected",
        [
            pytest.param(lambda r: r.without("b"), {"a": 1.0}, id="without"),
            pytest.param(lambda r: r.map(lambda x: x + 1), {"a": 2.0, "b": 3.0}, id="map"),
            pytest.param(lambda r: r.replace(a=3.0), {"a": 3.0, "b": 2.0}, id="replace"),
            pytest.param(
                lambda r: r.merge(Record("other", c=3.0)),
                {"a": 1.0, "b": 2.0, "c": 3.0},
                id="merge",
            ),
            pytest.param(lambda r: r.with_path_names(a="z"), {"z": 1.0, "b": 2.0}, id="rename"),
        ],
    )
    def test_structural_transforms_preserve_names_and_apply_the_edit(
        self, name, transform, expected
    ):
        record = Record.ensure({"a": jnp.array(1.0), "b": jnp.array(2.0)}, name=name)

        result = transform(record)

        assert result.name == record.name
        assert {key: float(value) for key, value in result.items()} == expected
        assert {key: float(value) for key, value in record.items()} == {"a": 1.0, "b": 2.0}

    def test_nested_auto_name_derives_from_top_level_keys(self):
        # The derived name uses top-level field keys (not full leaf paths),
        # so every transform agrees regardless of nesting depth.
        nested = Record(
            "record(a)",
            {"a": Record("a", {"b": jnp.array(1.0), "c": jnp.array(2.0)})},
        )
        assert nested.name == "record(a)"
        assert nested.with_path_names({"a/b": "z"}).name == "record(a)"
        assert nested.map(lambda x: x).name == "record(a)"

    def test_record_names_survive_pickle(self):
        auto = Record("record(a)", {"a": 1.0})
        named = Record("mine", a=1.0)
        assert pickle.loads(pickle.dumps(auto)).name == auto.name
        assert pickle.loads(pickle.dumps(named)).name == named.name


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

    def test_with_name_replaces_a_derived_name(self):
        r = Record("record(a)", {"a": 1.0})  # operation-derived (auto) name
        r2 = r.with_name("mine")
        assert r2.name == "mine"

    def test_with_name_records_provenance(self):
        n = Normal(loc=0.0, scale=1.0, name="x")
        m = n.with_name("y")
        assert m.provenance is not None
        assert m.provenance.operation == "with_name"
        assert m.provenance.metadata == {"old_name": "x", "new_name": "y"}
        assert m.provenance.parents[0].name == "x"

    def test_with_name_on_immutable_record(self):
        r = Record("orig", a=jnp.array(1.0), b=jnp.array(2.0))
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


class TestWithNameOnBatchTypes:
    """with_name on the batch types: a copy under the new user-given name,
    sharing field data, with the original unchanged."""

    def test_record_batch(self):
        ra = RecordBatch(
            "derived",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=EventTemplate(a=()),
        )
        ra2 = ra.with_name("mine")
        assert ra2 is not ra
        assert ra2.name == "mine"
        assert ra2["a"] is ra["a"]
        assert ra2.batch_shape == ra.batch_shape
        assert ra2.event_template is ra.event_template

    def test_numeric_record_batch(self):
        nrb = NumericRecordBatch(
            "orig",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=EventTemplate(a=()),
        )
        nra2 = nrb.with_name("new")
        assert nra2.name == "new"
        assert nra2["a"] is nrb["a"]
        assert nrb.name == "orig"

    def test_distribution_array(self):
        da = Normal.from_batched_params(loc=jnp.zeros(3), scale=1.0, name="batch")
        da2 = da.with_name("renamed_batch")
        assert da2.name == "renamed_batch"
        assert da2.batch_shape == da.batch_shape
        assert da.name == "batch"


# ===========================================================================
# 3b. with_name on hosts with argument-taking __new__ (views, routers)
# ===========================================================================


class TestWithNameOnCustomNewHosts:
    """with_name must work on every TrackedTerm host, including classes whose
    __new__ takes required arguments (dynamic class selection / views)."""

    def test_transformed_distribution(self):
        import tensorflow_probability.substrates.jax.bijectors as tfb

        from probpipe import TransformedDistribution

        t = TransformedDistribution(Normal(loc=0.0, scale=1.0, name="x"), tfb.Exp())
        t2 = t.with_name("y")
        assert t2.name == "y"
        key = jax.random.PRNGKey(0)
        assert jnp.allclose(jnp.asarray(t._sample(key, (5,))), jnp.asarray(t2._sample(key, (5,))))

    def test_flattened_distribution_view(self):
        from probpipe import MultivariateNormal

        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="theta")
        flat = mvn.as_flat_distribution()
        renamed = flat.with_name("theta_flat")
        assert renamed.name == "theta_flat"
        assert renamed.event_shape == flat.event_shape

    def test_record_distribution_view(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
        )
        view = joint["mu"]
        renamed = view.with_name("mu_view")
        assert renamed.name == "mu_view"

    def test_empirical_router(self):
        emp = EmpiricalDistribution(["a", "b", "c"])
        renamed = emp.with_name("labels")
        assert renamed.name == "labels"


# ===========================================================================
# 3c. Name preservation through derived objects
# ===========================================================================


class TestNamePreservation:
    """Transformations preserve the names assigned by their constructors."""

    def test_minibatched_distribution_default_and_explicit_names(self):
        import tensorflow_probability.substrates.jax.glm as tfp_glm

        from probpipe import MultivariateNormal
        from probpipe.inference._minibatch import MinibatchedDistribution
        from probpipe.modeling import GLMLikelihood

        X = jnp.eye(4)
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        prior = MultivariateNormal(loc=jnp.zeros(4), cov=jnp.eye(4), name="theta")
        lik = GLMLikelihood(tfp_glm.Bernoulli(), x=X)
        m = MinibatchedDistribution(prior, lik, Record("r", X=X, y=y), batch_size=2)
        assert m.name == "MinibatchedDistribution(batch_size=2)"
        named = MinibatchedDistribution(
            prior, lik, Record("r", X=X, y=y), batch_size=2, name="mine"
        )
        assert named.name == "mine"

    def test_product_conditioning_preserves_names(self):
        auto_joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
        )
        cond = auto_joint._condition_on(mu=0.5)
        assert cond.name == auto_joint.name
        named_joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
            name="my_joint",
        )
        cond_named = named_joint._condition_on(mu=0.5)
        assert cond_named.name == named_joint.name

    def test_distribution_array_slice_preserves_names(self):
        da = Normal.from_batched_params(loc=jnp.zeros(4), scale=1.0, name="batch")
        assert da[0:2].name == da.name
        renamed = da.with_name("renamed")
        assert renamed[0:2].name == "renamed"

    def test_from_batched_params_cells_derive_names(self):
        da = Normal.from_batched_params(loc=jnp.zeros(3), scale=1.0, name="x")
        cell = da[0]
        assert cell.name == "x_0"

    def test_full_factorial_design_derives_name(self):
        from probpipe.record import FullFactorialDesign

        design = FullFactorialDesign(a=jnp.arange(2.0), b=jnp.arange(3.0))
        assert design.name.startswith("FullFactorialDesign")


# ===========================================================================
# 4. with_provenance — write-once
# ===========================================================================


class TestWithProvenance:
    def test_returns_self_for_chaining(self):
        r = Record("r", a=1.0)
        assert r.with_provenance(Provenance("op")) is r
        assert r.provenance.operation == "op"

    def test_none_is_noop(self):
        r = Record("r", a=1.0)
        assert r.with_provenance(None) is r
        assert r.provenance is None

    def test_write_once_raises(self):
        for obj in (Record("r", a=1.0), Normal(loc=0.0, scale=1.0, name="x")):
            obj.with_provenance(Provenance("first"))
            with pytest.raises(RuntimeError, match="write-once"):
                obj.with_provenance(Provenance("second"))


# ===========================================================================
# 5. Annotated — the annotations store
# ===========================================================================


class TestAnnotated:
    def test_annotations_default_none(self):
        assert Record("r", a=1.0).annotations is None
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
        r = Record("r", a=1.0)
        object.__setattr__(r, "_annotations", {"k": 1})
        assert r.annotations == {"k": 1}


# ===========================================================================
# 6. Batch element types round-trip identity state
# ===========================================================================


class TestProductPickleRoundTrip:
    def test_product_keeps_derived_name_through_pickle(self):
        joint = ProductDistribution(
            mu=Normal(loc=0.0, scale=1.0, name="mu"),
            sigma=Normal(loc=1.0, scale=0.5, name="sigma"),
        )
        back = pickle.loads(pickle.dumps(joint))
        assert back.name == joint.name

    def test_user_named_product_keeps_identity_and_provenance(self):
        joint = ProductDistribution(mu=Normal(loc=0.0, scale=1.0, name="mu"), name="my_joint")
        joint.with_provenance(Provenance("op"))
        back = pickle.loads(pickle.dumps(joint))
        assert back.name == "my_joint"
        assert back.provenance.operation == "op"


class TestBatchPickleRoundTrip:
    def test_numeric_record_batch_pickle_preserves_identity(self):
        nrb = NumericRecordBatch(
            "mine",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=EventTemplate(a=()),
        )
        nrb.with_provenance(Provenance("op"))
        back = pickle.loads(pickle.dumps(nrb))
        assert back.name == "mine"
        assert back.provenance.operation == "op"
