"""Contract guards for the named-collection redesign (Record / EventTemplate).

These assert the *contracts* introduced by the leaf-keyed collection model —
construction rules and their error cases, canonical ordering, the conditional
round-trip and its lossiness, the subtree-template consistency invariant, the
nested-dict / field-value (de)constructors, template-threaded edits, and the
batch field-navigation surface — not just happy-path values.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import EventTemplate, NumericRecord, Record
from probpipe.core.event_template import ArraySpec, NumericEventTemplate, OpaqueSpec

# ---------------------------------------------------------------------------
# Construction: path-keyed unflattening and its error cases
# ---------------------------------------------------------------------------


class TestPathKeyedConstruction:
    def test_path_keys_equal_nested_keyword(self):
        a = Record({"physics/force": 1.0, "physics/mass": 2.0, "observation": 3.0})
        b = Record(physics=Record(force=1.0, mass=2.0), observation=3.0)
        assert a == b
        # EventTemplate mirrors the same path convention.
        ta = EventTemplate({"physics/force": (), "physics/mass": (), "observation": ()})
        tb = EventTemplate(physics=EventTemplate(force=(), mass=()), observation=())
        assert ta == tb

    def test_canonical_first_appearance_order(self):
        # 'observation' follows the whole 'physics' subtree because the prefix
        # 'physics' first appears before 'observation'.
        r = Record({"physics/force": 1.0, "observation": 2.0, "physics/mass": 3.0})
        assert tuple(r.keys()) == ("physics/force", "physics/mass", "observation")

    @pytest.mark.parametrize(
        "bad",
        [
            {"a/b": 1.0, "a": 5.0},  # complete-then-prefix, leaf value
            {"a/b": 1.0, "a": {"b": 2.0}},  # complete-then-prefix, dict value
            {"a/b": 1.0, "a": Record(b=2.0)},  # complete-then-prefix, Record value
        ],
    )
    def test_field_versus_prefix_collision(self, bad):
        with pytest.raises(ValueError, match="both as a field and as a path prefix"):
            Record(bad)

    @pytest.mark.parametrize("bad_key", ["", "a/", "/a", "a//b"])
    def test_malformed_keys_raise_valueerror(self, bad_key):
        with pytest.raises(ValueError):
            Record({bad_key: 1.0})

    def test_non_string_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            Record({42: 1.0})

    def test_plain_dict_value_is_opaque_leaf(self):
        # A dict VALUE is opaque data, not structure: it is one leaf, not exploded.
        r = Record(meta={"seed": 0}, x=1.0)
        assert tuple(r.keys()) == ("meta", "x")
        assert r["meta"] == {"seed": 0}
        assert isinstance(r.event_template["meta"], OpaqueSpec)


# ---------------------------------------------------------------------------
# Conditional round-trip via dict(r) — and the negative (lossiness pinned)
# ---------------------------------------------------------------------------


class TestConditionalRoundTrip:
    def test_faithful_with_schema_and_class(self):
        for r in [
            Record(a=1.0, b="tag"),  # flat mixed
            Record(physics=Record(force=jnp.zeros(3), mass=2.0), obs="y"),  # nested mixed
            NumericRecord(a=jnp.zeros(2), b=NumericRecord(c=1.0)),  # nested numeric
        ]:
            assert type(r)(dict(r), event_template=r.event_template) == r

    def test_value_only_dict_is_lossy_for_dtype(self):
        # A template carrying dtype is not recoverable from a value-only dict.
        tpl = EventTemplate(x=ArraySpec((), dtype=jnp.dtype("float32")))
        r = Record({"x": jnp.float32(1.0)}, event_template=tpl)
        assert Record(dict(r)) != r  # re-inferred template drops the dtype

    def test_numeric_record_value_only_is_lossy(self):
        nr = NumericRecord(a=1.0, b=2.0)
        # Re-inference builds a base Record, not a NumericRecord.
        assert Record(dict(nr)) != nr


# ---------------------------------------------------------------------------
# Subtree-template consistency invariant (the headline #326 guarantee)
# ---------------------------------------------------------------------------


class TestSubtreeTemplateInvariant:
    def _check(self, r):
        for p in ("physics",):
            assert r.at_path(p).event_template == r.event_template.at_path(p)

    def test_inferred_template(self):
        self._check(Record(physics=Record(force=1.0, mass=2.0), obs=3.0))

    def test_supplied_template_via_path_keys(self):
        tpl = EventTemplate(
            physics=EventTemplate(force=ArraySpec((), dtype=jnp.dtype("float32")), mass=()),
            obs=(),
        )
        r = Record(
            {"physics/force": jnp.float32(1.0), "physics/mass": 2.0, "obs": 3.0},
            event_template=tpl,
        )
        self._check(r)
        assert r.event_template.at_path("physics/force").dtype == jnp.dtype("float32")

    def test_supplied_template_via_prebuilt_child(self):
        # Regression: a pre-built child Record must be re-templated with the slice.
        tpl = EventTemplate(
            physics=EventTemplate(force=ArraySpec((), dtype=jnp.dtype("float64")), mass=()),
            obs=(),
        )
        child = Record({"force": 1.0, "mass": 2.0})  # bare-shape inferred template
        r = Record(physics=child, obs=3.0, event_template=tpl)
        self._check(r)
        # The authoritative dtype must be carried onto the pre-built child. Compare
        # via str(): `dtype == jnp.dtype("float64")` is True even for a None dtype
        # (numpy treats None as the default float), which would mask a lost dtype.
        force_dtype = r.at_path("physics").event_template.at_path("force").dtype
        assert force_dtype is not None and str(force_dtype) == "float64"


# ---------------------------------------------------------------------------
# Nested-dict and field-value (de)constructors
# ---------------------------------------------------------------------------


class TestConvenienceConstructors:
    def test_from_nested_dict_round_trips_structure(self):
        r = Record(physics=Record(force=1.0, mass=2.0), obs=3.0)
        assert Record.from_nested_dict(r.to_nested_dict()) == r

    def test_from_nested_dict_reads_every_dict_as_structure(self):
        r = Record.from_nested_dict({"physics": {"force": 1.0}, "obs": 2.0})
        assert tuple(r.keys()) == ("physics/force", "obs")

    def test_from_nested_dict_template_keeps_dict_leaf_opaque(self):
        # With a template marking 'meta' a leaf, a dict there stays opaque data.
        tpl = EventTemplate(meta=OpaqueSpec(), x=())
        r = Record.from_nested_dict({"meta": {"seed": 0}, "x": 1.0}, event_template=tpl)
        assert tuple(r.keys()) == ("meta", "x")
        assert r["meta"] == {"seed": 0}

    def test_to_nested_dict_distinct_from_flat_dict(self):
        r = Record(physics=Record(force=1.0, mass=2.0), obs=3.0)
        assert r.to_nested_dict() == {"physics": {"force": 1.0, "mass": 2.0}, "obs": 3.0}
        assert dict(r) == {"physics/force": 1.0, "physics/mass": 2.0, "obs": 3.0}

    def test_from_field_values_round_trip(self):
        # The reconstructed class follows the template's numericness, so the round
        # trip holds when the source class matches: NumericRecord for a numeric
        # template, base Record for a mixed one.
        for r in [
            NumericRecord(physics=NumericRecord(force=jnp.zeros(2), mass=1.0), obs=3.0),
            # mixed top, but the numeric subtree is a NumericRecord (canonical class
            # layout, which from_field_values reconstructs).
            Record(physics=NumericRecord(force=jnp.zeros(2), mass=1.0), obs="tag"),
        ]:
            rebuilt = r.event_template.from_field_values(list(r.values()))
            assert rebuilt == r

    def test_from_field_values_count_mismatch_raises(self):
        tpl = EventTemplate(a=(), b=())
        with pytest.raises(ValueError):
            tpl.from_field_values([1.0])


# ---------------------------------------------------------------------------
# Template-threaded edits: untouched fields keep their authoritative specs
# ---------------------------------------------------------------------------


class TestEditTemplateThreading:
    def _rich(self):
        tpl = EventTemplate(
            physics=EventTemplate(force=ArraySpec((), dtype=jnp.dtype("float32")), mass=()),
            obs=ArraySpec((), dtype=jnp.dtype("float32")),
        )
        return Record(
            {"physics/force": jnp.float32(1.0), "physics/mass": 2.0, "obs": jnp.float32(3.0)},
            event_template=tpl,
        )

    def test_without_threads_specs_no_reinference(self):
        r = self._rich()
        w = r.without("physics/mass")
        assert tuple(w.keys()) == ("physics/force", "obs")
        # dtype of a surviving, untouched field is preserved (threaded, not re-inferred).
        assert w.event_template.at_path("physics/force").dtype == jnp.dtype("float32")

    def test_merge_threads_both_specs(self):
        left = self._rich().without("obs")  # physics/force(f32), physics/mass
        right = Record(
            {"obs": jnp.float32(9.0)},
            event_template=EventTemplate(obs=ArraySpec((), dtype=jnp.dtype("float32"))),
        )
        m = left.merge(right)
        assert m.event_template.at_path("physics/force").dtype == jnp.dtype("float32")
        assert m.event_template["obs"].dtype == jnp.dtype("float32")

    def test_replace_keeps_untouched_specs(self):
        r = self._rich()
        r2 = r.replace({"obs": 5.0})
        # physics/force untouched -> dtype preserved; obs re-inferred.
        assert r2.event_template.at_path("physics/force").dtype == jnp.dtype("float32")

    def test_replace_preserves_field_order(self):
        # A replaced subtree (or leaf) must stay in its position, not jump to the
        # end — canonical order is part of the template's identity.
        t = EventTemplate(p=EventTemplate(x=(), y=()), q=())
        assert tuple(t.replace({"p": EventTemplate(z=())}).keys()) == ("p/z", "q")
        r = Record(p=Record(x=1.0, y=2.0), q=3.0)
        assert tuple(r.replace({"p": Record(z=9.0)}).keys()) == ("p/z", "q")
        assert tuple(r.replace({"q": 7.0}).keys()) == ("p/x", "p/y", "q")

    def test_merge_field_versus_prefix_clash_raises(self):
        with pytest.raises(ValueError):
            Record({"a/b": 1.0}).merge(Record(a=2.0))

    def test_deep_merge_combines_subtree(self):
        m = Record({"g/x": 1.0}).merge(Record({"g/y": 2.0}))
        assert tuple(m.keys()) == ("g/x", "g/y")


# ---------------------------------------------------------------------------
# EventTemplate edits / map operate on specs directly
# ---------------------------------------------------------------------------


class TestEventTemplateOps:
    def test_without_merge_on_template(self):
        tpl = EventTemplate(physics=EventTemplate(force=(), mass=()), obs=())
        assert tuple(tpl.without("physics/mass").keys()) == ("physics/force", "obs")
        merged = EventTemplate(a=()).merge(EventTemplate(b=(3,)))
        assert tuple(merged.keys()) == ("a", "b")

    def test_map_over_specs_requires_coercible_output(self):
        tpl = EventTemplate(a=(2,), b=())
        # identity over specs
        assert tpl.map(lambda s: s) == tpl
        # non-spec-coercible output raises TypeError
        with pytest.raises(TypeError):
            tpl.map(lambda s: object())

    def test_map_to_numeric_promotes(self):
        tpl = EventTemplate(a=None, b=())  # mixed -> base EventTemplate
        assert not isinstance(tpl, NumericEventTemplate)
        mapped = tpl.map(lambda s: ArraySpec((2,)))  # every spec numeric now
        assert isinstance(mapped, NumericEventTemplate)


# ---------------------------------------------------------------------------
# Boundary rules on a single Record (the shared §3.0 contract)
# ---------------------------------------------------------------------------


class TestBoundaryRules:
    @pytest.fixture
    def r(self):
        return Record(physics=Record(force=1.0, mass=2.0), obs=3.0)

    def test_len_iter_keys_over_leaves(self, r):
        assert len(r) == 3
        assert list(r) == ["physics/force", "physics/mass", "obs"]

    def test_membership_and_indexing_coincide_leaf_only(self, r):
        assert "physics/force" in r and "obs" in r
        assert "physics" not in r  # subtree is not a member
        with pytest.raises(KeyError):
            r["physics"]  # subtree is not indexable

    def test_at_path_and_children(self, r):
        assert isinstance(r.at_path("physics"), Record)
        assert r.at_path("physics/force") == r["physics/force"]
        assert tuple(r.children) == ("physics", "obs")
        assert isinstance(r.children["physics"], Record)  # children includes subtrees

    def test_is_field_equals_membership(self, r):
        assert r.is_field("physics/force") is True
        assert r.is_field("physics") is False
        assert r.is_field("physics/force") == ("physics/force" in r)


# ---------------------------------------------------------------------------
# Batch field-navigation surface (RecordArray): string [] is leaf-only
# ---------------------------------------------------------------------------


class TestBatchFieldNav:
    def _nested_array(self):
        tpl = EventTemplate(outer=EventTemplate(a=(), b=()), m=())
        return tpl.from_vector(jnp.arange(15.0).reshape(5, 3))

    def test_string_index_is_leaf_only(self):
        nra = self._nested_array()
        # a partial-path string raises; at_path reaches the sub-batch
        with pytest.raises(KeyError):
            nra["outer"]
        sub = nra.at_path("outer")
        np.testing.assert_allclose(nra["outer/a"], sub["a"])

    def test_edits_unsupported_on_batch(self):
        nra = self._nested_array()
        for op in (lambda: nra.replace(m=jnp.zeros(5)), nra.merge, nra.without):
            with pytest.raises(NotImplementedError):
                op() if op is not nra.merge and op is not nra.without else op(nra)

    def test_stack_nested_records_raises_clearly(self):
        # Stacking nested records into a batch needs nested-batch construction
        # (deferred); it fails with a clear TypeError, not a cryptic KeyError.
        from probpipe.core._record_array import NumericRecordArray

        recs = [
            NumericRecord(physics=NumericRecord(force=jnp.zeros(()), mass=jnp.zeros(())), obs=1.0)
            for _ in range(3)
        ]
        with pytest.raises(TypeError, match="nested"):
            NumericRecordArray.stack(recs)
