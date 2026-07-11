"""Contract tests for the ``NamedTree`` substrate.

Asserts the shared tree contract every family inherits: the public class,
``with_path_names`` renaming semantics, the mappings-are-never-leaves rule,
``is_multi_field``, and the declared-leaf-type validation hook.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import EventTemplate, NumericRecord, Record
from probpipe.core.event_template import ArraySpec, NumericEventTemplate, OpaqueSpec, ValueSpec
from probpipe.core.named_tree import NamedTree
from probpipe.core.record import _auto_record

# ===========================================================================
# 1. NamedTree is the public substrate
# ===========================================================================


class TestPublicSubstrate:
    def test_families_are_named_trees(self):
        assert isinstance(Record("r", a=1.0), NamedTree)
        assert isinstance(EventTemplate(a=()), NamedTree)
        assert isinstance(NumericRecord("nr", a=jnp.array(1.0)), NamedTree)

    def test_leaf_type_hooks(self):
        assert EventTemplate._leaf_type() is ValueSpec
        assert Record._leaf_type() is object

    def test_template_rejects_non_spec_leaf(self):
        with pytest.raises(TypeError):
            EventTemplate(a=object())


# ===========================================================================
# 2. is_multi_field (substrate-level, so Record has it too)
# ===========================================================================


class TestIsMultiField:
    def test_single_field_record(self):
        assert Record("r", a=1.0).is_multi_field is False

    def test_nested_single_field_record(self):
        assert Record("r", g=Record("r", a=1.0)).is_multi_field is False

    def test_multi_field_record(self):
        assert Record("r", a=1.0, b=2.0).is_multi_field is True

    def test_nested_multi_field_record(self):
        assert Record("r", g=Record("r", a=1.0, b=2.0)).is_multi_field is True

    def test_template(self):
        assert EventTemplate(a=()).is_multi_field is False
        assert EventTemplate(a=(), b=(2,)).is_multi_field is True


# ===========================================================================
# 3. with_path_names
# ===========================================================================


class TestWithPathNames:
    @pytest.fixture
    def record(self):
        return Record("r", x=1.0, g=Record("r", mu=2.0, sigma=3.0))

    def test_bare_name_unique(self, record):
        renamed = record.with_path_names(mu="loc")
        assert tuple(renamed.keys()) == ("x", "g/loc", "g/sigma")
        assert tuple(renamed.event_template.keys()) == ("x", "g/loc", "g/sigma")

    def test_full_path(self, record):
        renamed = record.with_path_names({"g/mu": "loc"})
        assert tuple(renamed.keys()) == ("x", "g/loc", "g/sigma")

    def test_interior_node_rename(self, record):
        renamed = record.with_path_names(g="group")
        assert tuple(renamed.keys()) == ("x", "group/mu", "group/sigma")
        assert tuple(renamed.event_template.keys()) == ("x", "group/mu", "group/sigma")

    def test_values_and_order_unchanged(self, record):
        renamed = record.with_path_names(mu="loc")
        assert renamed["g/loc"] == record["g/mu"]
        assert tuple(renamed.children) == tuple(record.children)

    def test_sibling_swap_is_simultaneous(self):
        r = Record("r", a=1.0, b=2.0)
        swapped = r.with_path_names(a="b", b="a")
        assert swapped["b"] == 1.0
        assert swapped["a"] == 2.0

    def test_ambiguous_bare_name_raises(self):
        r = Record("r", g=Record("r", x=1.0), h=Record("r", x=2.0))
        with pytest.raises(ValueError, match="ambiguous"):
            r.with_path_names(x="y")

    def test_missing_key_raises(self, record):
        with pytest.raises(KeyError):
            record.with_path_names(nope="x")

    def test_sibling_collision_raises(self, record):
        with pytest.raises(ValueError, match="collide"):
            record.with_path_names(mu="sigma")

    def test_malformed_new_name_raises(self, record):
        with pytest.raises(ValueError):
            record.with_path_names(mu="")
        with pytest.raises(ValueError):
            record.with_path_names(mu="a/b")

    def test_no_renames_raises(self, record):
        with pytest.raises(ValueError):
            record.with_path_names()

    def test_duplicate_rename_of_one_node_raises(self, record):
        with pytest.raises(ValueError, match="more than once"):
            record.with_path_names({"g/mu": "a"}, mu="b")

    def test_template_family_preserved(self):
        t = EventTemplate(a=(), b=(2,))
        renamed = t.with_path_names(a="alpha")
        assert isinstance(renamed, NumericEventTemplate)
        assert tuple(renamed.keys()) == ("alpha", "b")

    def test_numeric_record_family_preserved(self):
        nr = NumericRecord("nr", a=jnp.array(1.0))
        renamed = nr.with_path_names(a="alpha")
        assert isinstance(renamed, NumericRecord)
        assert tuple(renamed.keys()) == ("alpha",)

    def test_auto_name_rederives_user_name_preserved(self):
        auto = _auto_record({"a": 1.0, "b": 2.0})  # derived name "record(a,b)"
        renamed = auto.with_path_names(a="alpha")
        assert renamed.name == "record(alpha,b)"
        assert renamed.name_is_auto is True
        named = Record("mine", a=1.0, b=2.0)
        renamed_named = named.with_path_names(a="alpha")
        assert renamed_named.name == "mine"
        assert renamed_named.name_is_auto is False

    def test_explicit_template_metadata_survives(self):
        spec = ArraySpec((), dtype=jnp.float32)
        r = Record("r", a=jnp.array(1.0, dtype=jnp.float32), event_template=EventTemplate(a=spec))
        renamed = r.with_path_names(a="alpha")
        assert renamed.event_template["alpha"] == spec

    def test_record_array_defers(self):
        from probpipe import RecordArray

        ra = RecordArray({"a": jnp.zeros((3,))}, batch_shape=(3,), template=EventTemplate(a=()))
        with pytest.raises(NotImplementedError):
            ra.with_path_names(a="b")


# ===========================================================================
# 4. Mappings are never leaves
# ===========================================================================


class TestMappingsAreNeverLeaves:
    def test_constructor_rejects_mapping_leaf(self):
        with pytest.raises(TypeError, match="never leaves"):
            Record("r", meta={"seed": 0})

    def test_nested_constructor_rejects_mapping_leaf(self):
        with pytest.raises(TypeError, match="never leaves"):
            Record("r", g=Record("r", meta={"seed": 0}))

    def test_replace_rejects_mapping_leaf(self):
        r = Record("r", a=1.0)
        with pytest.raises(TypeError, match="never leaves"):
            r.replace(a={"seed": 0})

    def test_from_nested_dict_reads_mappings_as_structure(self):
        r = Record.from_nested_dict("r", {"a": {"b": 1.0}, "c": 2.0})
        assert tuple(r.keys()) == ("a/b", "c")

    def test_serialization_round_trips(self):
        r = Record("r", g=Record("r", a=1.0, b=2.0), c=3.0)
        back = Record.from_nested_dict("r", r.to_nested_dict())
        assert tuple(back.keys()) == tuple(r.keys())
        assert back == r

    def test_opaque_spec_agrees(self):
        # The spec layer and the record layer enforce the same rule.
        assert not OpaqueSpec().is_valid({"a": 1})


# ===========================================================================
# 5. Pytree children/aux split (template + identity ride the aux)
# ===========================================================================


class TestPytreeAuxSplit:
    def test_template_and_identity_survive_roundtrip(self):
        import jax

        spec = ArraySpec((), dtype=jnp.float32)
        r = Record(
            "mine",
            a=jnp.array(1.0, dtype=jnp.float32),
            b="label",
            event_template=EventTemplate(a=spec, b=None),
        )
        leaves, treedef = jax.tree_util.tree_flatten(r)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        assert back.event_template["a"] == spec  # explicit template threaded, not re-inferred
        assert back.name == "mine"
        assert back.name_is_auto is False

    def test_auto_flag_survives_roundtrip(self):
        import jax

        r = _auto_record({"a": jnp.array(1.0)})  # operation-derived (auto)
        back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(r)))
        assert back.name_is_auto is True

    def test_provenance_and_annotations_do_not_cross(self):
        import jax

        from probpipe import Provenance

        r = Record("r", a=jnp.array(1.0)).with_provenance(Provenance("op"))
        object.__setattr__(r, "_annotations", {"k": 1})
        back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(r)))
        assert back.provenance is None
        assert back.annotations is None

    def test_numeric_record_jit_and_vmap(self):
        import jax

        nr = NumericRecord("nr", x=jnp.arange(3.0), g=NumericRecord("nr", y=jnp.array(2.0)))
        assert float(jax.jit(lambda rec: rec["x"].sum())(nr)) == 3.0
        batched = NumericRecord("nr", x=jnp.ones((4, 3)), g=NumericRecord("nr", y=jnp.ones(4)))
        out = jax.vmap(lambda rec: rec["x"].sum() + rec["g/y"])(batched)
        assert out.shape == (4,)

    def test_treedefs_equal_iff_templates_equal(self):
        import jax

        r1 = Record("r", a=jnp.array(1.0, dtype=jnp.float32))
        r2 = Record("r", a=jnp.array(9.0, dtype=jnp.float32))
        assert jax.tree_util.tree_structure(r1) == jax.tree_util.tree_structure(r2)
        richer = Record(
            "r",
            a=jnp.array(1.0, dtype=jnp.float32),
            event_template=EventTemplate(a=ArraySpec((), dtype=jnp.float32)),
        )
        # Treedef equality is stricter than record equality: a richer explicit
        # template distinguishes the treedefs even when the data is equal.
        assert jax.tree_util.tree_structure(richer) != jax.tree_util.tree_structure(r1)


# ===========================================================================
# 6. Record -> NumericRecord auto-promotion (the numeric axis re-derives)
# ===========================================================================


class TestRecordAutoPromotion:
    def test_all_numeric_construction_promotes(self):
        assert type(Record("r", a=1.0, b=jnp.arange(3.0))) is NumericRecord

    def test_mixed_construction_stays_plain(self):
        assert type(Record("r", a=1.0, label="tag")) is Record

    def test_nested_path_keyed_children_promote(self):
        r = Record("r", {"g/a": 1.0, "g/h/b": 2.0, "c": "tag"})
        assert type(r) is Record
        assert type(r.at_path("g")) is NumericRecord
        assert type(r.at_path("g/h")) is NumericRecord

    def test_explicit_non_numeric_template_wins(self):
        r = Record("r", a=1.0, event_template=EventTemplate(a=OpaqueSpec()))
        assert type(r) is Record

    def test_backend_leaves_stay_verbatim(self):
        xr = pytest.importorskip("xarray")
        import numpy as np

        da = xr.DataArray(np.arange(3.0), dims=["t"])
        r = Record("r", a=da)
        # Promotion never silently coerces a backend-typed leaf; the caller
        # opts in via to_numeric() / NumericRecord(...).
        assert type(r) is Record
        assert r["a"] is da
        assert type(r.to_numeric().to_native()["a"]) is xr.DataArray

    def test_edits_rederive_promotion_and_demotion(self):
        mixed = Record("r", a=1.0, label="tag")
        assert type(mixed.without("label")) is NumericRecord
        numeric = Record("r", a=1.0, b=2.0)
        assert type(numeric.replace(b="tag")) is Record
        assert type(numeric.merge(Record("r", c="tag"))) is Record

    def test_pytree_roundtrip_is_class_stable(self):
        import jax

        xr = pytest.importorskip("xarray")
        import numpy as np

        # A verbatim backend leaf has a numeric template but flattened as a
        # plain Record; unflatten must reproduce the treedef, not re-promote.
        r = Record("r", a=xr.DataArray(np.arange(3.0), dims=["t"]))
        leaves, treedef = jax.tree_util.tree_flatten(r)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(back) is Record
        assert jax.tree_util.tree_structure(back) == treedef

    def test_batch_subclasses_unaffected(self):
        from probpipe import NumericRecordArray, RecordArray

        ra = RecordArray({"a": jnp.zeros((3,))}, batch_shape=(3,), template=EventTemplate(a=()))
        assert type(ra) is RecordArray
        nra = NumericRecordArray(
            {"a": jnp.zeros((3,))}, batch_shape=(3,), template=EventTemplate(a=())
        )
        assert type(nra) is NumericRecordArray


# ===========================================================================
# 7. Value-level (de)serialization entry points
# ===========================================================================


class TestValueLevelEntryPoints:
    def test_from_field_values_round_trip_with_name(self):
        r = Record("r", a=jnp.array(1.0), b="tag", name="mine")
        rebuilt = Record.from_field_values("mine", r.event_template, r.values())
        assert rebuilt == r
        assert rebuilt.name == "mine"
        assert rebuilt.name_is_auto is False

    def test_from_field_values_numeric_template_promotes(self):
        tpl = EventTemplate(a=(), b=(2,))
        rebuilt = Record.from_field_values("v", tpl, [jnp.array(1.0), jnp.zeros(2)])
        assert type(rebuilt) is NumericRecord
        assert rebuilt.event_template is tpl

    def test_from_field_values_count_mismatch(self):
        with pytest.raises(ValueError, match="expected"):
            Record.from_field_values("v", EventTemplate(a=(), b=()), [1.0])

    def test_numeric_record_from_vector_round_trip(self):
        nr = NumericRecord("nr", x=jnp.arange(3.0), g=NumericRecord("nr", y=jnp.array(2.0)))
        back = NumericRecord.from_vector("mine", nr.event_template, nr.to_vector())
        assert back == nr
        assert back.name == "mine"
        assert back.name_is_auto is False

    def test_numeric_record_from_vector_rejects_batched(self):
        nr = NumericRecord("nr", x=jnp.arange(3.0))
        with pytest.raises(TypeError, match="1-D"):
            NumericRecord.from_vector("v", nr.event_template, jnp.ones((4, 3)))
