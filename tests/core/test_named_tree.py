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

# ===========================================================================
# 1. NamedTree is the public substrate
# ===========================================================================


class TestPublicSubstrate:
    def test_families_are_named_trees(self):
        assert isinstance(Record(a=1.0), NamedTree)
        assert isinstance(EventTemplate(a=()), NamedTree)
        assert isinstance(NumericRecord(a=jnp.array(1.0)), NamedTree)

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
        assert Record(a=1.0).is_multi_field is False

    def test_nested_single_field_record(self):
        assert Record(g=Record(a=1.0)).is_multi_field is False

    def test_multi_field_record(self):
        assert Record(a=1.0, b=2.0).is_multi_field is True

    def test_nested_multi_field_record(self):
        assert Record(g=Record(a=1.0, b=2.0)).is_multi_field is True

    def test_template(self):
        assert EventTemplate(a=()).is_multi_field is False
        assert EventTemplate(a=(), b=(2,)).is_multi_field is True


# ===========================================================================
# 3. with_path_names
# ===========================================================================


class TestWithPathNames:
    @pytest.fixture
    def record(self):
        return Record(x=1.0, g=Record(mu=2.0, sigma=3.0))

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
        r = Record(a=1.0, b=2.0)
        swapped = r.with_path_names(a="b", b="a")
        assert swapped["b"] == 1.0
        assert swapped["a"] == 2.0

    def test_ambiguous_bare_name_raises(self):
        r = Record(g=Record(x=1.0), h=Record(x=2.0))
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
        nr = NumericRecord(a=jnp.array(1.0))
        renamed = nr.with_path_names(a="alpha")
        assert isinstance(renamed, NumericRecord)
        assert tuple(renamed.keys()) == ("alpha",)

    def test_auto_name_rederives_user_name_preserved(self):
        auto = Record(a=1.0, b=2.0)  # auto-named "record(a,b)"
        renamed = auto.with_path_names(a="alpha")
        assert renamed.name == "record(alpha,b)"
        assert renamed.name_is_auto is True
        named = Record(a=1.0, b=2.0, name="mine")
        renamed_named = named.with_path_names(a="alpha")
        assert renamed_named.name == "mine"
        assert renamed_named.name_is_auto is False

    def test_explicit_template_metadata_survives(self):
        spec = ArraySpec((), dtype=jnp.float32)
        r = Record(a=jnp.array(1.0, dtype=jnp.float32), event_template=EventTemplate(a=spec))
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
            Record(meta={"seed": 0})

    def test_nested_constructor_rejects_mapping_leaf(self):
        with pytest.raises(TypeError, match="never leaves"):
            Record(g=Record(meta={"seed": 0}))

    def test_replace_rejects_mapping_leaf(self):
        r = Record(a=1.0)
        with pytest.raises(TypeError, match="never leaves"):
            r.replace(a={"seed": 0})

    def test_from_nested_dict_reads_mappings_as_structure(self):
        r = Record.from_nested_dict({"a": {"b": 1.0}, "c": 2.0})
        assert tuple(r.keys()) == ("a/b", "c")

    def test_serialization_round_trips(self):
        r = Record(g=Record(a=1.0, b=2.0), c=3.0)
        back = Record.from_nested_dict(r.to_nested_dict())
        assert tuple(back.keys()) == tuple(r.keys())
        assert back == r

    def test_opaque_spec_agrees(self):
        # The spec layer and the record layer enforce the same rule.
        assert not OpaqueSpec().is_valid({"a": 1})
