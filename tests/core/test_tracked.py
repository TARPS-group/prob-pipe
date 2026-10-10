"""Contract tests for the ``TrackedTerm`` / ``Annotated`` identity mixins.

Asserts the identity-and-metadata contract shared by every tracked term:
construction-time labels and preservation through transforms,
``with_label`` copy semantics, ``with_provenance`` write-once behaviour, and
the ``annotations`` store.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import pytest

import probpipe
from probpipe import (
    DistributionBatch,
    EmpiricalDistribution,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    OpaqueBatch,
    Provenance,
    ProvenanceMode,
    Record,
    RecordBatch,
)
from probpipe.core._specs import RecordSpec
from probpipe.core.tracked import Annotated, TrackedTerm, auto_label
from probpipe.distributions import FactoredDistribution


def _normals(label: str, count: int) -> DistributionBatch:
    """A batch of *count* standard normal laws on a level named ``law``."""
    return DistributionBatch(label, [Normal("x", 0.0, 1.0) for _ in range(count)], "law")


# ===========================================================================
# 1. Mixin membership — every core object is a tracked term
# ===========================================================================


class TestMixinMembership:
    def test_distribution_is_tracked_and_annotated(self):
        n = Normal("x", loc=0.0, scale=1.0)
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
            element_spec=RecordSpec(a=()),
        )
        assert isinstance(ra, TrackedTerm)

    def test_distribution_batch_is_tracked(self):
        assert isinstance(_normals("batch", 3), TrackedTerm)


# ===========================================================================
# 2. Construction-time labels
# ===========================================================================


class TestLabelEnforcement:
    def test_tracked_host_must_set_nonempty_label(self):
        """The construction-time label check lives on TrackedTerm, not per host."""

        class Unlabeled(TrackedTerm):
            def __init__(self):
                pass

            def raw(self):
                return None

        with pytest.raises(TypeError, match="non-empty label"):
            Unlabeled()

        class EmptyLabeled(TrackedTerm):
            def __init__(self):
                self._init_tracked("")

            def raw(self):
                return None

        with pytest.raises(TypeError, match="non-empty label"):
            EmptyLabeled()

    def test_a_misplaced_label_names_the_value_the_constructor_got(self):
        with pytest.raises(
            TypeError, match=r"Record requires a non-empty label, got \{'x': 1.0\}, which is not"
        ):
            Record({"x": 1.0})
        with pytest.raises(TypeError, match=r"Record requires a non-empty label, got ''$"):
            Record("", x=1.0)


class TestAutoLabelHelper:
    def test_supplied_label_is_user_given(self):
        assert auto_label("mine", "default") == "mine"

    def test_missing_label_takes_default(self):
        assert auto_label(None, "default") == "default"


class TestLabelLifecycle:
    def test_distribution_keeps_explicit_label(self):
        n = Normal("mu", loc=0.0, scale=1.0, label="prior")
        assert n.label == "prior"

    def test_record_keeps_explicit_label(self):
        r = Record("mine", a=1.0)
        assert r.label == "mine"

    def test_constructor_requires_label(self):
        # The label guard fires in ``Record.__new__`` before promotion picks a
        # class, so a call without a label reports ``Record`` and its custom message
        # — not the promoted ``NumericRecord`` nor the bare Python "missing
        # positional argument" a reverted guard would leave.
        with pytest.raises(TypeError, match="Record requires its label"):
            Record(a=1.0)

    def test_record_keeps_operation_label(self):
        r = Record("sample", {"a": 1.0, "b": 2.0})
        assert r.label == "sample"

    def test_batch_keeps_its_construction_label(self):
        """A caller that derives a label says so; there is no unlabeled batch."""
        ra = RecordBatch(
            "derived",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        assert ra.label == "derived"
        named = RecordBatch(
            "mine",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        assert named.label == "mine"

    def test_composite_distribution_derives_default_label(self):
        joint = Normal("mu", loc=0.0, scale=1.0) * Normal("sigma", loc=0.0, scale=1.0)
        assert joint.label == "Normal·Normal"
        assert str(joint) == "Normal(mu)·Normal(sigma)"

    def test_composite_distribution_keeps_explicit_label(self):
        joint = FactoredDistribution("my_joint", [Normal("mu", loc=0.0, scale=1.0)])
        assert joint.label == "my_joint"

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
    def test_structural_transforms_preserve_labels_and_apply_the_edit(
        self, name, transform, expected
    ):
        record = Record.ensure({"a": jnp.array(1.0), "b": jnp.array(2.0)}, label=name)

        result = transform(record)

        assert result.label == record.label
        assert {key: float(value) for key, value in result.items()} == expected
        assert {key: float(value) for key, value in record.items()} == {"a": 1.0, "b": 2.0}

    def test_nested_auto_label_derives_from_top_level_keys(self):
        # The derived label uses top-level field keys (not full leaf paths),
        # so every transform agrees regardless of nesting depth.
        nested = Record(
            "record(a)",
            {"a": Record("a", {"b": jnp.array(1.0), "c": jnp.array(2.0)})},
        )
        assert nested.label == "record(a)"
        assert nested.with_path_names({"a/b": "z"}).label == "record(a)"
        assert nested.map(lambda x: x).label == "record(a)"

    def test_record_labels_survive_pickle(self):
        auto = Record("record(a)", {"a": 1.0})
        named = Record("mine", a=1.0)
        assert pickle.loads(pickle.dumps(auto)).label == auto.label
        assert pickle.loads(pickle.dumps(named)).label == named.label


# ===========================================================================
# 3. with_label — relabel-as-copy semantics
# ===========================================================================


class TestWithLabel:
    def test_with_label_returns_copy_original_unchanged(self):
        n = Normal("mu", loc=0.0, scale=1.0, label="x")
        m = n.with_label("y")
        assert m is not n
        assert m.label == "y"
        assert n.label == "x"

    def test_with_label_replaces_a_derived_label(self):
        r = Record("record(a)", {"a": 1.0})  # operation-derived (auto) label
        r2 = r.with_label("mine")
        assert r2.label == "mine"

    def test_with_label_records_provenance(self):
        n = Normal("mu", loc=0.0, scale=1.0, label="x")
        m = n.with_label("y")
        assert m.provenance is not None
        assert m.provenance.operation == "with_label"
        assert m.provenance.metadata == {"old_label": "x", "new_label": "y"}
        assert m.provenance.parents[0].label == "x"

    def test_with_label_on_immutable_record(self):
        r = Record("orig", a=jnp.array(1.0), b=jnp.array(2.0))
        r2 = r.with_label("new")
        assert r2.label == "new"
        # shallow copy: field data is shared, not copied
        assert r2.raw("a") is r.raw("a")
        assert r2.event_template is r.event_template
        assert r == r2 or r2["b"] is r["b"]

    def test_with_label_rejects_empty_or_non_string(self):
        n = Normal("x", loc=0.0, scale=1.0)
        with pytest.raises(TypeError, match="non-empty string"):
            n.with_label("")
        with pytest.raises(TypeError, match="non-empty string"):
            n.with_label(3)  # type: ignore[arg-type]

    def test_with_label_off_mode_attaches_no_provenance(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal("x", loc=0.0, scale=1.0)
        m = n.with_label("y")
        assert m.label == "y"
        assert m.provenance is None

    def test_with_label_decouples_annotations_container(self):
        # Post-relabel annotation writes must not show through on the
        # original (the container is copied; entry values are shared).
        n = Normal("x", loc=0.0, scale=1.0)
        object.__setattr__(n, "_annotations", {"fit": "exact"})
        m = n.with_label("y")
        m.annotations["check"] = "added-on-copy"
        assert "check" not in n.annotations
        assert m.annotations["fit"] == "exact"

    def test_with_label_decouples_datatree_annotations(self):
        xr = pytest.importorskip("xarray")
        n = Normal("x", loc=0.0, scale=1.0)
        object.__setattr__(n, "_annotations", xr.DataTree.from_dict({"arviz": xr.Dataset()}))
        m = n.with_label("y")
        m.annotations["diagnostics"] = xr.DataTree()
        assert "diagnostics" not in n.annotations.children
        assert "arviz" in m.annotations.children

    def test_with_label_after_provenance_starts_fresh_chain(self):
        n = Normal("x", loc=0.0, scale=1.0)
        n.with_provenance(Provenance("first"))
        m = n.with_label("y")
        # the clone's provenance is the relabeling, not the original's chain
        assert m.provenance.operation == "with_label"
        # and the original's chain is reachable through the parent descriptor
        assert m.provenance.parents[0].provenance is n.provenance


class TestWithLabelOnBatchTypes:
    """with_label on the batch types: a copy under the new user-given label,
    sharing field data, with the original unchanged."""

    def test_record_batch(self):
        ra = RecordBatch(
            "derived",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        ra2 = ra.with_label("mine")
        assert ra2 is not ra
        assert ra2.label == "mine"
        assert ra2["a"].raw() is ra["a"].raw()
        assert ra2.batch_shape == ra.batch_shape
        assert ra2.event_template is ra.event_template

    def test_numeric_record_batch(self):
        nrb = NumericRecordBatch(
            "orig",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        nra2 = nrb.with_label("new")
        assert nra2.label == "new"
        assert nra2["a"].raw() is nrb["a"].raw()
        assert nrb.label == "orig"

    def test_distribution_batch(self):
        batch = _normals("batch", 3)
        renamed = batch.with_label("renamed_batch")
        assert renamed.label == "renamed_batch"
        assert renamed.batch_shape == batch.batch_shape
        assert batch.label == "batch"


# ===========================================================================
# 3b. with_label on hosts with argument-taking __new__ (views, routers)
# ===========================================================================


class TestWithLabelOnCustomNewHosts:
    """with_label must work on every TrackedTerm host, including classes whose
    __new__ takes required arguments (dynamic class selection / views)."""

    def test_transformed_distribution(self):
        import tensorflow_probability.substrates.jax.bijectors as tfb

        from probpipe import BijectorTransformedDistribution

        t = BijectorTransformedDistribution("t", Normal("x", loc=0.0, scale=1.0), tfb.Exp())
        t2 = t.with_label("y")
        assert t2.label == "y"
        key = jax.random.PRNGKey(0)
        assert jnp.allclose(jnp.asarray(t._sample(key, (5,))), jnp.asarray(t2._sample(key, (5,))))

    def test_field_view(self):
        joint = Normal("mu", loc=0.0, scale=1.0) * Normal("sigma", loc=1.0, scale=0.5)
        view = joint["mu"]
        renamed = view.with_label("mu_view")
        assert renamed.label == "mu_view"

    def test_empirical_capability_subclass(self):
        emp = EmpiricalDistribution(OpaqueBatch("labels", ["a", "b", "c"], "emp"), component="emp")
        renamed = emp.with_label("labels")
        assert renamed.label == "labels"


# ===========================================================================
# 3c. Label preservation through derived objects
# ===========================================================================


class TestLabelPreservation:
    """Transformations preserve the labels assigned by their constructors."""

    def test_minibatched_distribution_keeps_its_label(self):
        from probpipe import MultivariateNormal
        from probpipe.families import BernoulliFamily, glm_likelihood
        from probpipe.inference._minibatch import MinibatchedDistribution

        X = jnp.eye(4)
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        prior = MultivariateNormal("beta", loc=jnp.zeros(4), cov=jnp.eye(4))
        lik = glm_likelihood("y", BernoulliFamily(), X=X)
        named = MinibatchedDistribution("mine", prior, lik, y, batch_size=2)
        assert named.label == "mine"

    def test_joint_conditioning_preserves_labels(self):
        auto_joint = Normal("mu", loc=0.0, scale=1.0) * Normal("sigma", loc=1.0, scale=0.5)
        cond = auto_joint._condition_on({"mu": 0.5})
        assert cond.label == auto_joint.label
        named_joint = auto_joint.with_label("my_joint")
        cond_named = named_joint._condition_on({"mu": 0.5})
        assert cond_named.label == named_joint.label

    def test_distribution_batch_slice_derives_its_label_from_the_batch(self):
        batch = _normals("batch", 4)
        assert batch[0:2].label == "batch[law=0:2]"
        renamed = batch.with_label("renamed")
        assert renamed[0:2].label == "renamed[law=0:2]"

    def test_distribution_batch_elements_derive_labels(self):
        assert _normals("x", 3)[0].label == "x[law=0]"

    def test_full_factorial_design_derives_label(self):
        from probpipe.record import FullFactorialDesign

        design = FullFactorialDesign(a=jnp.arange(2.0), b=jnp.arange(3.0))
        assert design.label.startswith("FullFactorialDesign")


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
        for obj in (Record("r", a=1.0), Normal("x", loc=0.0, scale=1.0)):
            obj.with_provenance(Provenance("first"))
            with pytest.raises(RuntimeError, match="set only once"):
                obj.with_provenance(Provenance("second"))


# ===========================================================================
# 5. Annotated — the annotations store
# ===========================================================================


class TestAnnotated:
    def test_annotations_default_none(self):
        assert Record("r", a=1.0).annotations is None
        assert Normal("x", loc=0.0, scale=1.0).annotations is None

    def test_annotations_accepts_plain_mapping(self):
        n = Normal("x", loc=0.0, scale=1.0)
        object.__setattr__(n, "_annotations", {"note": "fitted by hand"})
        assert n.annotations == {"note": "fitted by hand"}

    def test_annotations_accepts_datatree(self):
        xr = pytest.importorskip("xarray")
        n = Normal("x", loc=0.0, scale=1.0)
        object.__setattr__(n, "_annotations", xr.DataTree.from_dict({"diagnostics": xr.Dataset()}))
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


class TestJointPickleRoundTrip:
    def test_joint_keeps_derived_label_through_pickle(self):
        joint = Normal("mu", loc=0.0, scale=1.0) * Normal("sigma", loc=1.0, scale=0.5)
        back = pickle.loads(pickle.dumps(joint))
        assert back.label == joint.label

    def test_user_labeled_joint_keeps_identity_and_provenance(self):
        joint = FactoredDistribution("my_joint", [Normal("mu", loc=0.0, scale=1.0)])
        joint.with_provenance(Provenance("op"))
        back = pickle.loads(pickle.dumps(joint))
        assert back.label == "my_joint"
        assert back.provenance.operation == "op"


class TestBatchPickleRoundTrip:
    def test_numeric_record_batch_pickle_preserves_identity(self):
        nrb = NumericRecordBatch(
            "mine",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        nrb.with_provenance(Provenance("op"))
        back = pickle.loads(pickle.dumps(nrb))
        assert back.label == "mine"
        assert back.provenance.operation == "op"
