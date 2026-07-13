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

from probpipe import EventTemplate, NumericRecord, NumericRecordArray, Record
from probpipe.core.event_template import ArraySpec, NumericEventTemplate, OpaqueSpec

# ---------------------------------------------------------------------------
# Construction: path-keyed unflattening and its error cases
# ---------------------------------------------------------------------------


class TestPathKeyedConstruction:
    def test_path_keys_equal_nested_keyword(self):
        a = Record("r", {"physics/force": 1.0, "physics/mass": 2.0, "observation": 3.0})
        b = Record("r", physics=Record("r", force=1.0, mass=2.0), observation=3.0)
        assert a == b
        # EventTemplate mirrors the same path convention.
        ta = EventTemplate({"physics/force": (), "physics/mass": (), "observation": ()})
        tb = EventTemplate(physics=EventTemplate(force=(), mass=()), observation=())
        assert ta == tb

    def test_canonical_first_appearance_order(self):
        # 'observation' follows the whole 'physics' subtree because the prefix
        # 'physics' first appears before 'observation'.
        r = Record("r", {"physics/force": 1.0, "observation": 2.0, "physics/mass": 3.0})
        assert tuple(r.keys()) == ("physics/force", "physics/mass", "observation")

    @pytest.mark.parametrize(
        "bad",
        [
            {"a/b": 1.0, "a": 5.0},  # complete-then-prefix, leaf value
            {"a/b": 1.0, "a": {"b": 2.0}},  # complete-then-prefix, dict value
            {"a/b": 1.0, "a": Record("r", b=2.0)},  # complete-then-prefix, Record value
            # prefix-then-complete takes a different _unflatten_paths branch
            {"a": 5.0, "a/b": 1.0},
            {"a": Record("r", b=2.0), "a/b": 1.0},
        ],
    )
    def test_field_versus_prefix_collision(self, bad):
        with pytest.raises(ValueError, match="both as a field and as a path prefix"):
            Record("r", bad)

    @pytest.mark.parametrize("bad_key", ["", "a/", "/a", "a//b"])
    def test_malformed_keys_raise_valueerror(self, bad_key):
        with pytest.raises(ValueError):
            Record("r", {bad_key: 1.0})

    def test_non_string_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            Record("r", {42: 1.0})

    def test_mapping_value_materializes_as_subtree(self):
        # Mappings are never leaves: a mapping value denotes tree structure,
        # so the constructor materialises it into a nested subtree rather than
        # storing an opaque leaf.
        r = Record("r", meta={"seed": 0}, x=1.0)
        assert tuple(r.keys()) == ("meta/seed", "x")


# ---------------------------------------------------------------------------
# Conditional round-trip via dict(r) — and the negative (lossiness pinned)
# ---------------------------------------------------------------------------


class TestConditionalRoundTrip:
    def test_faithful_with_schema_and_class(self):
        for r in [
            Record("r", a=1.0, b="tag"),  # flat mixed
            Record("r", physics=Record("r", force=jnp.zeros(3), mass=2.0), obs="y"),  # nested mixed
            NumericRecord("nr", a=jnp.zeros(2), b=NumericRecord("nr", c=1.0)),  # nested numeric
        ]:
            assert type(r)(r.name, dict(r), event_template=r.event_template) == r

    def test_value_only_dict_is_lossy_for_dtype(self):
        # A template carrying dtype is not recoverable from a value-only dict.
        tpl = EventTemplate(x=ArraySpec((), dtype=jnp.dtype("float32")))
        r = Record("r", {"x": jnp.float32(1.0)}, event_template=tpl)
        assert Record("r", dict(r)) != r  # re-inferred template drops the dtype

    def test_numeric_record_value_only_round_trips(self):
        nr = NumericRecord("nr", a=1.0, b=2.0)
        # Auto-promotion re-derives the numeric class from the values, so
        # the value-only rebuild is no longer lossy.
        assert Record("r", dict(nr)) == nr


# ---------------------------------------------------------------------------
# Subtree-template consistency invariant
# ---------------------------------------------------------------------------


class TestSubtreeTemplateInvariant:
    def _check(self, r):
        for p in ("physics",):
            assert r.at_path(p).event_template == r.event_template.at_path(p)

    def test_inferred_template(self):
        self._check(Record("r", physics=Record("r", force=1.0, mass=2.0), obs=3.0))

    def test_supplied_template_via_path_keys(self):
        tpl = EventTemplate(
            physics=EventTemplate(force=ArraySpec((), dtype=jnp.dtype("float32")), mass=()),
            obs=(),
        )
        r = Record(
            "r",
            {"physics/force": jnp.float32(1.0), "physics/mass": 2.0, "obs": 3.0},
            event_template=tpl,
        )
        self._check(r)
        assert r.event_template.at_path("physics/force").dtype == jnp.dtype("float32")

    def test_supplied_template_via_prebuilt_child(self):
        # A pre-built child Record whose own template differs from the supplied
        # slice must adopt the slice's specs.
        tpl = EventTemplate(
            physics=EventTemplate(force=ArraySpec((), dtype=jnp.dtype("float64")), mass=()),
            obs=(),
        )
        child = Record("r", {"force": 1.0, "mass": 2.0})  # bare-shape inferred template
        r = Record("r", physics=child, obs=3.0, event_template=tpl)
        self._check(r)
        # The authoritative dtype must be carried onto the pre-built child. Compare
        # via str(): `dtype == jnp.dtype("float64")` is True even for a None dtype
        # (numpy treats None as the default float), which would mask a lost dtype.
        force_dtype = r.at_path("physics").event_template.at_path("force").dtype
        assert force_dtype is not None and str(force_dtype) == "float64"

    def test_prebuilt_child_with_identical_template_is_reused(self):
        # When the supplied slice IS the child's own template object and the
        # child is already named by its field key, the child is stored
        # verbatim — preserving its identity and metadata (backend aux)
        # instead of being rebuilt.
        child = NumericRecord("physics", force=1.0, mass=2.0)
        tpl = EventTemplate(physics=child.event_template, obs=())
        r = Record("r", physics=child, obs=3.0, event_template=tpl)
        assert r.at_path("physics") is child
        assert r.at_path("physics").name == "physics"


# ---------------------------------------------------------------------------
# Nested-dict and field-value (de)constructors
# ---------------------------------------------------------------------------


class TestConvenienceConstructors:
    def test_from_nested_dict_round_trips_structure(self):
        r = Record("r", physics=Record("r", force=1.0, mass=2.0), obs=3.0)
        assert Record.from_nested_dict("r", r.to_nested_dict()) == r

    def test_from_nested_dict_reads_every_dict_as_structure(self):
        r = Record.from_nested_dict("r", {"physics": {"force": 1.0}, "obs": 2.0})
        assert tuple(r.keys()) == ("physics/force", "obs")

    def test_from_nested_dict_reads_every_mapping_as_structure(self):
        # Mappings are never leaves: every mapping level becomes a subtree,
        # even where a template proposes a leaf there — the mismatch raises.
        tpl = EventTemplate(meta=OpaqueSpec(), x=())
        with pytest.raises(ValueError):
            Record.from_nested_dict("r", {"meta": {"seed": 0}, "x": 1.0}, event_template=tpl)
        r = Record.from_nested_dict("r", {"meta": {"seed": 0}, "x": 1.0})
        assert tuple(r.keys()) == ("meta/seed", "x")

    def test_to_nested_dict_distinct_from_flat_dict(self):
        r = Record("r", physics=Record("r", force=1.0, mass=2.0), obs=3.0)
        assert r.to_nested_dict() == {"physics": {"force": 1.0, "mass": 2.0}, "obs": 3.0}
        assert dict(r) == {"physics/force": 1.0, "physics/mass": 2.0, "obs": 3.0}

    def test_from_field_values_round_trip(self):
        # The reconstructed class follows the template's numericness, so the round
        # trip holds when the source class matches: NumericRecord for a numeric
        # template, base Record for a mixed one.
        for r in [
            NumericRecord("nr", physics=NumericRecord("nr", force=jnp.zeros(2), mass=1.0), obs=3.0),
            # mixed top, but the numeric subtree is a NumericRecord (canonical class
            # layout, which from_field_values reconstructs).
            Record("r", physics=NumericRecord("nr", force=jnp.zeros(2), mass=1.0), obs="tag"),
        ]:
            rebuilt = Record.from_field_values(r.name, r.event_template, r.values())
            assert rebuilt == r

    def test_from_field_values_count_mismatch_raises(self):
        tpl = EventTemplate(a=(), b=())
        with pytest.raises(ValueError):
            Record.from_field_values("r", tpl, [1.0])


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
            "r",
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
            "r",
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
        r = Record("r", p=Record("r", x=1.0, y=2.0), q=3.0)
        assert tuple(r.replace({"p": Record("r", z=9.0)}).keys()) == ("p/z", "q")
        assert tuple(r.replace({"q": 7.0}).keys()) == ("p/x", "p/y", "q")

    def test_merge_field_versus_prefix_clash_raises(self):
        with pytest.raises(ValueError):
            Record("r", {"a/b": 1.0}).merge(Record("r", a=2.0))

    def test_deep_merge_combines_subtree(self):
        m = Record("r", {"g/x": 1.0}).merge(Record("r", {"g/y": 2.0}))
        assert tuple(m.keys()) == ("g/x", "g/y")

    def test_deep_merge_regroups_into_earlier_subtree(self):
        # A later key sharing an earlier subtree's name regroups INTO that
        # subtree (first-appearance order), ahead of later top-level names.
        m = Record("r", {"g/x": 1.0, "h/y": 2.0}).merge(Record("r", {"g/z": 3.0}))
        assert tuple(m.keys()) == ("g/x", "g/z", "h/y")

    def test_replace_to_opaque_demotes_numeric_template(self):
        # An all-numeric Record's template auto-promotes; replacing a field
        # with a non-numeric value must re-decide the promotion, not raise.
        r = Record("r", x=1.0, y=2.0)
        assert isinstance(r.event_template, NumericEventTemplate)
        r2 = r.replace(x="hello")
        assert r2["x"] == "hello"
        assert not isinstance(r2.event_template, NumericEventTemplate)
        # ... and merging a mixed record into a numeric one likewise demotes.
        m = r.merge(Record("r", label="fox"))
        assert not isinstance(m.event_template, NumericEventTemplate)
        # The template's own edits re-decide in both directions.
        t = EventTemplate(x=(), y=(3,))
        t2 = t.replace(x=OpaqueSpec())
        assert type(t2) is EventTemplate
        assert isinstance(t2.replace(x=ArraySpec(())), NumericEventTemplate)

    def test_edits_reuse_untouched_children_verbatim(self):
        # An untouched nested child already named by its field key survives
        # an edit as the SAME object — class, name, and metadata preserved
        # (never demoted to the outer record's class).
        child = NumericRecord("phys", x=1.0, y=2.0)
        r = Record("r", phys=child, obs="tag")
        for edited in (r.without("obs"), r.replace(obs="new"), r.merge(Record("r", extra=5.0))):
            assert edited.at_path("phys") is child
            assert isinstance(edited.at_path("phys"), NumericRecord)
        # A nested edit rebuilds only the touched child, recursively — and the
        # result keeps the child's class and stays in template lock-step.
        r2 = r.replace({"phys/x": 9.0})
        assert isinstance(r2.at_path("phys"), NumericRecord)
        assert float(r2["phys/x"]) == 9.0
        assert r2.at_path("phys").event_template == r2.event_template.at_path("phys")

    def test_replace_overlapping_paths_raise(self):
        r = Record("r", {"physics/force": 1.0, "physics/mass": 2.0, "obs": 3.0})
        for updates in (
            {"physics": 9.0, "physics/mass": 5.0},  # ancestor listed first
            {"physics/mass": 5.0, "physics": 9.0},  # descendant listed first
        ):
            with pytest.raises(ValueError, match="overlap"):
                r.replace(updates)
        with pytest.raises(ValueError, match="overlap"):
            r.event_template.replace({"physics": ArraySpec((2,)), "physics/mass": ArraySpec((5,))})


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
# Boundary rules on a single Record (the shared collection contract)
# ---------------------------------------------------------------------------


class TestBoundaryRules:
    @pytest.fixture
    def r(self):
        return Record("r", physics=Record("r", force=1.0, mass=2.0), obs=3.0)

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

    def test_map_preserves_class_and_reinfers_template(self):
        # map on a NumericRecord returns a NumericRecord, nested children
        # included; the result's template reflects the mapped leaf shapes.
        nr = NumericRecord("nr", physics=NumericRecord("nr", force=jnp.ones(3), mass=1.0), obs=2.0)
        doubled = nr.map(lambda x: 2 * x)
        assert isinstance(doubled, NumericRecord)
        assert isinstance(doubled.at_path("physics"), NumericRecord)
        # Non-zero inputs so the map is observable (2*0 == 0 would pass a no-op).
        np.testing.assert_allclose(doubled["physics/force"], 2.0 * jnp.ones(3))
        assert float(doubled["physics/mass"]) == 2.0
        assert float(doubled["obs"]) == 4.0
        # Shape-changing map re-infers the template to the new shapes.
        summed = nr.map(jnp.sum)
        assert summed.event_template.at_path("physics/force").shape == ()

    def test_nested_pickle_round_trip(self, r):
        import pickle

        nr = NumericRecord("nr", physics=NumericRecord("nr", force=jnp.zeros(2), mass=1.0), obs=3.0)
        assert pickle.loads(pickle.dumps(nr)) == nr
        assert pickle.loads(pickle.dumps(r)) == r

    def test_path_keyed_numeric_construction_preserves_backend_aux(self):
        # Backend metadata on a nested leaf survives path-keyed construction:
        # aux is captured by the nested record that owns the leaf, so
        # to_native() restores the original type at its /-path.
        xr = pytest.importorskip("xarray")
        da = xr.DataArray(jnp.array([1.0, 2.0]), dims=["t"], coords={"t": [10, 20]})
        nr = NumericRecord("nr", {"a/b": da, "c": 3.0})
        restored = nr.to_native().at_path("a/b")
        assert isinstance(restored, xr.DataArray)
        assert restored.dims == ("t",)


# ---------------------------------------------------------------------------
# Batch field-navigation surface (RecordArray): string [] is leaf-only
# ---------------------------------------------------------------------------


class TestBatchFieldNav:
    def _nested_array(self):
        tpl = EventTemplate(outer=EventTemplate(a=(), b=()), m=())
        return NumericRecordArray.from_vector("nra", tpl, jnp.arange(15.0).reshape(5, 3))

    def test_string_index_is_leaf_only(self):
        nra = self._nested_array()
        # a partial-path string raises; at_path reaches the sub-batch
        with pytest.raises(KeyError):
            nra["outer"]
        sub = nra.at_path("outer")
        np.testing.assert_allclose(nra["outer/a"], sub["a"])

    def test_batch_children_and_is_field(self):
        nra = self._nested_array()
        assert tuple(nra.children) == ("outer", "m")
        assert nra.is_field("outer/a") is True
        assert nra.is_field("outer") is False

    def test_batch_mapping_surface_is_top_level_transitionally(self):
        # Pins the documented transitional split (STYLE_GUIDE 1.10): keys /
        # len / in on a batch are TOP-LEVEL while string [] is leaf-only, so
        # membership and indexing deliberately disagree on a nested batch
        # until the batch-axis rework.
        nra = self._nested_array()
        assert list(nra.keys()) == ["outer", "m"]
        assert len(nra) == 2  # top-level field count, not the 3 leaves
        assert "outer" in nra  # top-level membership...
        with pytest.raises(KeyError):
            nra["outer"]  # ...even though [] rejects the interior node
        assert "outer/a" not in nra  # leaf path is not a member...
        np.testing.assert_allclose(nra["outer/a"], nra.at_path("outer")["a"])  # ...but indexes

    def test_view_and_select_all_on_nested_batch(self):
        # A top-level field that is an interior node can still be viewed /
        # splatted (view resolves it via template.children, not leaf-only []).
        nra = self._nested_array()
        v = nra.view("outer")
        assert v.field == "outer"
        selected = nra.select_all()
        assert set(selected) == {"outer", "m"}

    def test_nested_batch_round_trips_compare_equal(self):
        import pickle

        nra = self._nested_array()
        tpl = nra.template
        assert NumericRecordArray.from_vector("nra", tpl, nra.to_vector()) == nra
        assert pickle.loads(pickle.dumps(nra)) == nra

    def test_edits_unsupported_on_batch(self):
        nra = self._nested_array()
        with pytest.raises(NotImplementedError):
            nra.replace(m=jnp.zeros(5))
        with pytest.raises(NotImplementedError):
            nra.merge(nra)
        with pytest.raises(NotImplementedError):
            nra.without("m")
        with pytest.raises(NotImplementedError):
            nra.map(lambda x: x)
        with pytest.raises(NotImplementedError):
            nra.map_with_keys(lambda k, x: x)
        with pytest.raises(NotImplementedError):
            # The batch override keeps its own (data-only) signature and
            # rejects the operation outright.
            type(nra).from_nested_dict({"a": jnp.zeros(3)})

    def test_stack_nested_records_raises_clearly(self):
        # Stacking nested records into a batch needs nested-batch construction
        # (deferred); it fails with a clear TypeError, not a cryptic KeyError.
        from probpipe.core._record_array import NumericRecordArray

        recs = [
            NumericRecord(
                "nr", physics=NumericRecord("nr", force=jnp.zeros(()), mass=jnp.zeros(())), obs=1.0
            )
            for _ in range(3)
        ]
        with pytest.raises(TypeError, match="nested"):
            NumericRecordArray.stack(recs)
