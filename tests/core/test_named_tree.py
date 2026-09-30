"""Contract tests for the ``NamedTree`` substrate.

Asserts the shared tree contract every family inherits: the public class,
``with_path_names`` renaming semantics, the mappings-are-never-leaves rule,
``is_multi_field``, and the declared-leaf-type validation hook.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import NumericRecord, Record, RecordSpec
from probpipe.core._opaque import OpaqueSpec
from probpipe.core._specs import NumericArraySpec, NumericRecordSpec, TermSpec
from probpipe.core.named_tree import NamedTree

# ===========================================================================
# 1. NamedTree is the public substrate
# ===========================================================================


class TestPublicSubstrate:
    def test_families_are_named_trees(self):
        assert isinstance(Record("r", a=1.0), NamedTree)
        assert isinstance(RecordSpec(a=()), NamedTree)
        assert isinstance(NumericRecord("nr", a=jnp.array(1.0)), NamedTree)

    def test_leaf_type_hooks(self):
        assert RecordSpec._leaf_type() is TermSpec
        assert Record._leaf_type() is object

    def test_template_rejects_non_spec_leaf(self):
        with pytest.raises(TypeError):
            RecordSpec(a=object())

    def test_substrate_is_not_directly_instantiable(self):
        # ``NamedTree`` is the abstract substrate; only concrete families
        # (Record / RecordSpec / batch types) own a ``_tree`` store.
        with pytest.raises(TypeError, match="abstract substrate"):
            NamedTree()


class TestErrorPaths:
    def test_at_path_rejects_non_string_segment(self):
        r = Record("r", a=jnp.array(1.0))
        with pytest.raises(TypeError):
            r.at_path(123)

    def test_replace_rejects_positional_and_keyword(self):
        r = Record("r", a=jnp.array(1.0), b=jnp.array(2.0))
        with pytest.raises(ValueError, match="both"):
            r.replace({"a": jnp.array(3.0)}, b=jnp.array(4.0))

    def test_replace_no_op_returns_self(self):
        r = Record("r", a=jnp.array(1.0))
        assert r.replace() is r

    def test_with_path_names_field_vs_prefix_collision(self):
        # Renaming a leaf onto a name already used as a path prefix (here the
        # subtree ``n``) is the field-versus-prefix collision.
        r = Record("r", m=jnp.array(1.0), n=Record("n", b=jnp.array(2.0)))
        with pytest.raises(ValueError, match="field and as a path prefix"):
            r.with_path_names(m="n")


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
        assert RecordSpec(a=()).is_multi_field is False
        assert RecordSpec(a=(), b=(2,)).is_multi_field is True


# ===========================================================================
# 3. with_path_names
# ===========================================================================


class TestWithPathNames:
    @pytest.fixture
    def record(self):
        return Record("r", x=1.0, g=Record("r", mu=2.0, sigma=3.0))

    def test_a_single_name_addresses_a_top_level_node(self, record):
        renamed = record.with_path_names(x="y")
        assert tuple(renamed.keys()) == ("y", "g/mu", "g/sigma")
        assert tuple(renamed.event_template.keys()) == ("y", "g/mu", "g/sigma")

    def test_a_nested_node_takes_its_full_path(self, record):
        with pytest.raises(KeyError):
            record.with_path_names(mu="loc")

    def test_full_path(self, record):
        # A rename within a group spells the full new path.
        renamed = record.with_path_names({"g/mu": "g/loc"})
        assert tuple(renamed.keys()) == ("x", "g/loc", "g/sigma")

    def test_interior_node_rename(self, record):
        renamed = record.with_path_names(g="group")
        assert tuple(renamed.keys()) == ("x", "group/mu", "group/sigma")
        assert tuple(renamed.event_template.keys()) == ("x", "group/mu", "group/sigma")

    def test_values_and_order_unchanged(self, record):
        renamed = record.with_path_names({"g/mu": "g/loc"})
        assert renamed["g/loc"] == record["g/mu"]
        assert tuple(renamed.children) == tuple(record.children)

    def test_sibling_swap_is_simultaneous(self):
        r = Record("r", a=1.0, b=2.0)
        swapped = r.with_path_names(a="b", b="a")
        assert swapped["b"] == 1.0
        assert swapped["a"] == 2.0

    def test_a_name_shared_across_levels_addresses_each_node_by_its_path(self):
        r = Record("r", beta=Record("beta", beta=1.0), s=2.0)
        assert tuple(r.with_path_names(beta="b").keys()) == ("b/beta", "s")
        assert tuple(r.with_path_names({"beta/beta": "beta/b"}).keys()) == ("beta/b", "s")

    def test_missing_key_raises(self, record):
        with pytest.raises(KeyError):
            record.with_path_names(nope="x")

    def test_sibling_collision_raises(self, record):
        with pytest.raises(ValueError, match="collide"):
            record.with_path_names({"g/mu": "g/sigma"})

    def test_malformed_new_name_raises(self, record):
        with pytest.raises(ValueError, match="non-empty"):
            record.with_path_names(x="")
        # A target is a path, so only an empty segment malforms it.
        for target in ("a//b", "/a", "a/"):
            with pytest.raises(ValueError, match="empty segment"):
                record.with_path_names(x=target)

    def test_no_renames_raises(self, record):
        with pytest.raises(ValueError):
            record.with_path_names()

    def test_duplicate_rename_of_one_node_raises(self, record):
        with pytest.raises(ValueError, match="more than once"):
            record.with_path_names({"x": "a"}, x="b")

    def test_template_family_preserved(self):
        t = RecordSpec(a=(), b=(2,))
        renamed = t.with_path_names(a="alpha")
        assert isinstance(renamed, NumericRecordSpec)
        assert tuple(renamed.keys()) == ("alpha", "b")

    def test_numeric_record_family_preserved(self):
        nr = NumericRecord("nr", a=jnp.array(1.0))
        renamed = nr.with_path_names(a="alpha")
        assert isinstance(renamed, NumericRecord)
        assert tuple(renamed.keys()) == ("alpha",)

    def test_field_renaming_preserves_both_default_and_explicit_names(self):
        auto = Record("record(a,b)", {"a": 1.0, "b": 2.0})  # operation-derived (auto)
        renamed = auto.with_path_names(a="alpha")
        assert renamed.name == auto.name
        named = Record("mine", a=1.0, b=2.0)
        renamed_named = named.with_path_names(a="alpha")
        assert renamed_named.name == "mine"

    def test_explicit_template_metadata_survives(self):
        spec = NumericArraySpec((), dtype=jnp.float32)
        r = Record("r", a=jnp.array(1.0, dtype=jnp.float32), event_template=RecordSpec(a=spec))
        renamed = r.with_path_names(a="alpha")
        assert renamed.event_template["alpha"] == spec

    def test_record_batch_defers(self):
        from probpipe import RecordBatch

        ra = RecordBatch(
            "batch",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        renamed = ra.with_path_names(a="b")
        assert list(renamed.event_template) == ["b"]


class TestPathMoves:
    """A target is the node's new exact path, so a rename may move a node (design II.6)."""

    @pytest.fixture
    def record(self):
        return Record("r", x=1.0, g=Record("g", mu=2.0, sigma=3.0))

    def test_a_bare_target_moves_a_nested_field_to_the_top_level(self, record):
        moved = record.with_path_names({"g/mu": "mu"})
        assert tuple(moved.keys()) == ("x", "g/sigma", "mu")
        assert moved["mu"] == 2.0
        assert tuple(moved.event_template.keys()) == tuple(moved.keys())

    def test_a_path_target_moves_a_top_level_field_into_a_group(self, record):
        moved = record.with_path_names({"x": "g/x"})
        assert tuple(moved.keys()) == ("g/mu", "g/sigma", "g/x")
        assert moved["g/x"] == 1.0

    def test_a_move_into_a_missing_group_creates_it(self, record):
        moved = record.with_path_names({"x": "h/k/x"})
        assert tuple(moved.keys()) == ("g/mu", "g/sigma", "h/k/x")
        assert tuple(moved.event_template.keys()) == tuple(moved.keys())

    def test_a_group_a_move_empties_is_removed(self):
        record = Record("r", x=1.0, g=Record("g", mu=2.0))
        moved = record.with_path_names({"g/mu": "mu"})
        assert tuple(moved.children) == ("x", "mu")
        assert tuple(moved.event_template.children) == ("x", "mu")

    def test_a_field_can_replace_the_group_it_empties(self):
        record = Record("r", g=Record("g", mu=2.0), x=1.0)
        moved = record.with_path_names({"g/mu": "g"})
        assert tuple(moved.keys()) == ("x", "g")
        assert moved["g"] == 2.0

    def test_a_group_a_move_refills_keeps_its_position(self):
        record = Record("r", g=Record("g", mu=2.0), x=1.0)
        moved = record.with_path_names({"g/mu": "mu", "x": "g/x"})
        assert tuple(moved.keys()) == ("g/x", "mu")

    def test_moved_nodes_append_in_the_order_the_renames_are_given(self, record):
        forward = record.with_path_names({"x": "h/x", "g/sigma": "h/sigma"})
        backward = record.with_path_names({"g/sigma": "h/sigma", "x": "h/x"})
        assert tuple(forward.keys()) == ("g/mu", "h/x", "h/sigma")
        assert tuple(backward.keys()) == ("g/mu", "h/sigma", "h/x")

    def test_a_descendant_with_its_own_target_leaves_its_moved_ancestor(self, record):
        moved = record.with_path_names({"g": "h", "g/mu": "mu"})
        assert tuple(moved.keys()) == ("x", "h/sigma", "mu")

    def test_moves_apply_simultaneously(self, record):
        moved = record.with_path_names({"x": "g/x", "g/mu": "x"})
        assert tuple(moved.keys()) == ("g/sigma", "g/x", "x")
        assert (moved["g/x"], moved["x"]) == (1.0, 2.0)

    def test_a_schema_moves_as_its_record_does(self, record):
        moved = record.event_template.with_path_names({"g/mu": "mu"})
        assert moved == record.with_path_names({"g/mu": "mu"}).event_template
        assert isinstance(moved, NumericRecordSpec)

    def test_a_moved_numeric_record_keeps_its_family_and_its_leaves(self):
        nr = NumericRecord("nr", a=jnp.array(1.0), g=NumericRecord("g", b=jnp.array([2.0, 3.0])))
        moved = nr.with_path_names({"a": "g/a"})
        assert isinstance(moved, NumericRecord)
        assert tuple(moved.keys()) == ("g/b", "g/a")
        assert jnp.array_equal(moved.to_vector(), jnp.array([2.0, 3.0, 1.0]))

    @pytest.mark.parametrize(
        ("renames", "match"),
        [
            pytest.param({"g/mu": "g/sigma"}, "collides", id="onto-a-sibling"),
            pytest.param({"x": "g/mu"}, "collides", id="onto-a-node-that-stays"),
            pytest.param({"g/mu": "x/mu"}, "field and as a path prefix", id="through-a-field"),
            pytest.param({"g": "g/h"}, "own subtree", id="into-its-own-subtree"),
            pytest.param({"x": "h", "g/mu": "h"}, "collide", id="two-onto-one-path"),
            pytest.param({"x": "h", "g/mu": "h/mu"}, "overlap", id="one-target-inside-another"),
            pytest.param({"x": "a//b"}, "empty segment", id="empty-segment"),
        ],
    )
    def test_a_move_the_tree_cannot_take_raises(self, record, renames, match):
        with pytest.raises(ValueError, match=match):
            record.with_path_names(renames)
        with pytest.raises(ValueError, match=match):
            record.event_template.with_path_names(renames)

    def test_renaming_a_node_twice_raises_before_its_targets_are_read(self, record):
        with pytest.raises(ValueError, match="more than once"):
            record.with_path_names({"g/mu": "mu"}, **{"g/mu": "g/m"})


# ===========================================================================
# 4. Mappings are never leaves
# ===========================================================================


class TestMappingsAreNeverLeaves:
    def test_all_numeric_mapping_value_materializes_and_promotes(self):
        # A dict field value is nested structure, not a leaf. When every leaf
        # beneath it is numeric the result promotes to NumericRecord and the
        # leaves coerce to jax arrays (same as any all-numeric construction).
        r = Record("r", cfg={"a": 1.0, "b": 2.0}, x=3.0)
        assert type(r) is NumericRecord
        assert tuple(r.keys()) == ("cfg/a", "cfg/b", "x")
        assert isinstance(r["cfg/a"], jnp.ndarray)
        assert isinstance(r.at_path("cfg"), NumericRecord)

    def test_mapping_value_with_opaque_leaf_stays_plain(self):
        # A non-numeric leaf beneath the materialised subtree keeps the record a
        # plain Record and stores that leaf verbatim (no coercion).
        r = Record("r", cfg={"label": "horseshoe", "scale": 1.0})
        assert type(r) is Record
        assert tuple(r.keys()) == ("cfg/label", "cfg/scale")
        assert r["cfg/label"] == "horseshoe"  # opaque leaf, stored as-is
        assert isinstance(r.at_path("cfg"), Record)

    def test_multi_level_nested_mapping_materializes(self):
        # Nesting recurses to arbitrary depth.
        r = Record("r", a={"b": {"c": 1.0}}, x=2.0)
        assert tuple(r.keys()) == ("a/b/c", "x")

    def test_replace_materializes_mapping_value(self):
        r = Record("r", a=1.0)
        r2 = r.replace(a={"seed": 0})
        assert tuple(r2.keys()) == ("a/seed",)

    def test_constructor_reads_positional_nested_dict(self):
        r = Record("r", {"a": {"b": 1.0}, "c": 2.0})
        assert tuple(r.keys()) == ("a/b", "c")

    def test_serialization_round_trips(self):
        r = Record("r", g=Record("r", a=1.0, b=2.0), c=3.0)
        back = Record("r", r.to_nested_dict())
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

        spec = NumericArraySpec((), dtype=jnp.float32)
        r = Record(
            "mine",
            a=jnp.array(1.0, dtype=jnp.float32),
            b="label",
            event_template=RecordSpec(a=spec, b=None),
        )
        leaves, treedef = jax.tree_util.tree_flatten(r)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        assert back.event_template["a"] == spec  # explicit template threaded, not re-inferred
        assert back.name == "mine"

    def test_derived_name_survives_roundtrip(self):
        import jax

        r = Record("record(a)", {"a": jnp.array(1.0)})  # operation-derived (auto)
        back = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(r)))
        assert back.name == r.name

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
        # Each x row sums to 3.0 and g/y is 1.0, so every element is 4.0; the
        # value check would catch an x/g-y transpose that preserves shape.
        assert [float(v) for v in out] == [4.0, 4.0, 4.0, 4.0]

    def test_treedefs_equal_iff_templates_equal(self):
        import jax

        r1 = Record("r", a=jnp.array(1.0, dtype=jnp.float32))
        r2 = Record("r", a=jnp.array(9.0, dtype=jnp.float32))
        assert jax.tree_util.tree_structure(r1) == jax.tree_util.tree_structure(r2)
        richer = Record(
            "r",
            a=jnp.array(1.0, dtype=jnp.float32),
            event_template=RecordSpec(a=NumericArraySpec((), dtype=jnp.float32)),
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
        r = Record("r", a=1.0, event_template=RecordSpec(a=OpaqueSpec()))
        assert type(r) is Record

    def test_backend_leaves_stay_verbatim(self):
        xr = pytest.importorskip("xarray")
        import numpy as np

        da = xr.DataArray(np.arange(3.0), dims=["t"])
        r = Record("r", a=da)
        # A native backend leaf is first-class numeric: the record promotes
        # and the leaf is stored verbatim — navigation returns it directly.
        from probpipe import NumericRecord

        assert type(r) is NumericRecord
        assert r["a"] is da
        assert type(r["a"]) is xr.DataArray

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

        # A native backend leaf promotes; flatten converts it at the
        # boundary, and unflatten reproduces the same treedef and class.
        from probpipe import NumericRecord

        r = Record("r", a=xr.DataArray(np.arange(3.0), dims=["t"]))
        assert type(r) is NumericRecord
        leaves, treedef = jax.tree_util.tree_flatten(r)
        assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(back) is NumericRecord
        assert jax.tree_util.tree_structure(back) == treedef

    def test_batch_subclasses_unaffected(self):
        from probpipe import NumericRecordBatch, RecordBatch

        ra = RecordBatch(
            "batch",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        assert type(ra) is RecordBatch
        nrb = NumericRecordBatch(
            "batch",
            {"a": jnp.zeros((3,))},
            level_names="draw",
            axes_per_level=(1,),
            element_spec=RecordSpec(a=()),
        )
        assert type(nrb) is NumericRecordBatch


# ===========================================================================
# 7. Value-level (de)serialization entry points
# ===========================================================================


class TestValueLevelEntryPoints:
    def test_from_field_values_round_trip_with_name(self):
        r = Record("mine", a=jnp.array(1.0), b="tag")
        assert list(r.keys()) == ["a", "b"]  # name is positional-only, not a field
        rebuilt = Record.from_field_values(r.name, r.event_template, r.values())
        assert rebuilt == r
        assert rebuilt.name == "mine"

    def test_from_field_values_numeric_template_promotes(self):
        tpl = RecordSpec(a=(), b=(2,))
        rebuilt = Record.from_field_values("v", tpl, [jnp.array(1.0), jnp.zeros(2)])
        assert type(rebuilt) is NumericRecord
        assert rebuilt.event_template is tpl

    def test_from_field_values_count_mismatch(self):
        with pytest.raises(ValueError, match="expected"):
            Record.from_field_values("v", RecordSpec(a=(), b=()), [1.0])

    def test_numeric_record_from_vector_round_trip(self):
        nr = NumericRecord("nr", x=jnp.arange(3.0), g=NumericRecord("nr", y=jnp.array(2.0)))
        back = NumericRecord.from_vector("mine", nr.event_template, nr.to_vector())
        assert back == nr
        assert back.name == "mine"

    def test_numeric_record_from_vector_rejects_batched(self):
        nr = NumericRecord("nr", x=jnp.arange(3.0))
        with pytest.raises(TypeError, match="1-D"):
            NumericRecord.from_vector("v", nr.event_template, jnp.ones((4, 3)))
