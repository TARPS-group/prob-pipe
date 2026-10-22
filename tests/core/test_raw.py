"""Every kind defines ``raw``, its representation detached from the workflow (design II.4)."""

from __future__ import annotations

import copy
import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    FunctionBatch,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    Opaque,
    OpaqueBatch,
    Record,
    RecordBatch,
    RecordSpec,
    ResolutionError,
)
from probpipe.core._fingerprint import fingerprint
from probpipe.core._immutable import transient_memo
from probpipe.core._record_spec import NumericRecordSpec
from probpipe.core.provenance import Provenance
from probpipe.core.tracked import TrackedTerm, _as_raw_form, _is_raw_form
from probpipe.distributions._batches import _element_source
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._distribution import Distribution

from ..operations._laws import Gaussian, Kernel, Pair


class TestTheContract:
    def test_raw_is_abstract_on_tracked_term(self):
        assert "raw" in TrackedTerm.__abstractmethods__

    def test_raw_is_a_property_on_tracked_term(self):
        assert isinstance(TrackedTerm.__dict__["raw"], property)

    def test_a_kind_that_defines_raw_as_a_method_cannot_be_defined(self):
        with pytest.raises(TypeError, match="raw is a function, but it must be a property"):

            class Stale(TrackedTerm):
                def raw(self):
                    return None

    @pytest.mark.parametrize("wrap", [classmethod, staticmethod, functools.cache])
    def test_a_kind_whose_raw_is_a_wrapped_function_cannot_be_defined(self, wrap):
        with pytest.raises(TypeError, match="but it must be a property"):
            type("Stale", (TrackedTerm,), {"raw": wrap(lambda *args: None)})

    def test_a_kind_that_inherits_raw_as_a_method_from_a_mixin_cannot_be_defined(self):
        class Mixin:
            def raw(self):
                return None

        with pytest.raises(TypeError, match="raw is a function, but it must be a property"):

            class Stale(Mixin, TrackedTerm):
                pass

    def test_a_kind_that_does_not_define_raw_cannot_be_constructed(self):
        class Bare(TrackedTerm):
            def __init__(self):
                self._init_tracked("bare")

        with pytest.raises(TypeError, match="raw"):
            Bare()


class TestValues:
    def test_an_array_is_its_stored_array(self):
        stored = jnp.arange(3.0)
        assert (
            NumericArray(
                stored,
                label="x",
            ).raw
            is stored
        )

    def test_an_opaque_value_is_the_wrapped_value(self):
        wrapped = ("Rubin (1981)", "SAT points")
        assert (
            Opaque(
                wrapped,
                label="source",
            ).raw
            is wrapped
        )

    def test_a_record_is_the_nested_mapping_of_its_raw_leaves(self):
        effect = jnp.asarray(28.0)
        school = Record(
            {"data": {"effect": effect, "se": 15.0}, "label": "A"},
            label="school",
        )
        raw = school.raw
        assert type(raw) is dict and list(raw) == ["data", "label"]
        assert raw["data"]["effect"] is effect and raw["label"] == "A"

    def test_a_subtree_of_the_mapping_is_the_nested_mapping_beneath_it(self):
        effect = jnp.asarray(28.0)
        school = Record(
            {"data": {"effect": effect, "se": 15.0}, "label": "A"},
            label="school",
        )
        assert list(school.raw["data"]) == ["effect", "se"]
        assert school.raw["data"]["effect"] is effect

    def test_an_array_field_view_gives_the_stored_array(self):
        effect = jnp.asarray(28.0)
        school = Record(
            {"data": {"effect": effect, "se": 15.0}, "label": "A"},
            label="school",
        )
        assert school["data/effect"].raw is effect

    def test_a_field_is_read_from_the_mapping_inside_a_trace(self):
        record = NumericRecord(
            {"x": jnp.ones(2), "g": {"y": jnp.arange(3.0)}},
            label="r",
        )
        total = jax.jit(lambda r: r.raw["x"].sum() + r.raw["g"]["y"].sum())(record)
        assert float(total) == 5.0

    def test_a_family_held_as_a_leaf_is_its_backend_distribution(self):
        record = Record(
            {"law": Normal("theta", 0.0, 1.0), "x": 1.0},
            label="r",
        )
        assert isinstance(record.raw["law"], tfd.Normal)

    def test_a_law_held_as_a_leaf_is_detached(self):
        law = EmpiricalDistribution(jnp.arange(3.0), output_name="e").with_provenance(
            Provenance.create("made", parents=[])
        )
        record = Record(
            {"law": law, "x": 1.0},
            label="r",
        )
        detached = record.raw["law"]
        assert isinstance(detached, EmpiricalDistribution) and detached is not law
        assert detached.provenance is None

    def test_a_field_view_presents_a_stored_python_scalar_as_an_array(self):
        record = Record(
            {"x": 42.0, "label": "tag"},
            label="r",
        )
        assert isinstance(record.raw["x"], float)
        assert isinstance(record["x"].raw, jnp.ndarray)


class TestTheRawFormAtAPath:
    """The internal reader of one node's raw form, which builds no view."""

    def test_a_record_field_is_its_stored_leaf(self):
        effect = jnp.asarray(28.0)
        school = Record(
            {"data": {"effect": effect, "se": 15.0}, "label": "A"},
            label="school",
        )
        assert school._raw_at("data/effect") is effect
        assert school._raw_at("data") == {"effect": effect, "se": 15.0}

    def test_an_unknown_record_path_raises_key_error(self):
        with pytest.raises(KeyError):
            Record(
                {"x": 1.0},
                label="r",
            )._raw_at("y")

    def test_a_batch_field_is_its_column_and_an_interior_node_the_columns_beneath_it(self):
        batch = _numeric_record_batch()
        np.testing.assert_array_equal(np.asarray(batch._raw_at("g/a")), np.arange(3.0))
        node = batch._raw_at("g")
        assert set(node) == {"a", "b"}
        np.testing.assert_array_equal(np.asarray(node["b"]), np.ones(3))

    def test_an_unknown_batch_path_names_the_batch_paths(self):
        with pytest.raises(KeyError, match="g/a"):
            _numeric_record_batch()._raw_at("g/z")


def _numeric_record_batch():
    return NumericRecordBatch(
        {"g/a": jnp.arange(3.0), "g/b": jnp.ones(3), "c": jnp.zeros((3, 2))},
        "draw",
        element=NumericRecordSpec(g=NumericRecordSpec(a=(), b=()), c=(2,)),
        label="draws",
    )


class TestBatches:
    def test_a_record_batch_field_view_gives_its_column(self):
        np.testing.assert_array_equal(
            np.asarray(_numeric_record_batch()["g/a"].raw), np.arange(3.0)
        )

    def test_a_record_batch_interior_view_gives_the_columns_beneath_it(self):
        node = _numeric_record_batch()["g"].raw
        assert set(node) == {"a", "b"}
        np.testing.assert_array_equal(np.asarray(node["b"]), np.ones(3))

    def test_an_array_batch_is_its_stored_array(self):
        stored = jnp.zeros((4, 2))
        batch = NumericArrayBatch(
            stored,
            "draw",
            element=NumericArraySpec((2,)),
            label="x",
        )
        assert batch.raw is stored

    def test_a_record_batch_is_the_nested_mapping_of_its_columns(self):
        effects = jnp.arange(3.0)
        batch = RecordBatch(
            {"data/effect": effects, "label": np.array(["A", "B", "C"], dtype=object)},
            "school",
            element=RecordSpec(
                {
                    "data/effect": (),
                    "label": Opaque(
                        "A",
                        label="l",
                    ).spec,
                }
            ),
            label="schools",
        )
        raw = batch.raw
        assert list(raw) == ["data", "label"]
        assert raw["data"]["effect"] is effects
        assert raw["label"].dtype == object and list(raw["label"]) == ["A", "B", "C"]

    def test_an_object_batch_is_its_frozen_object_array(self):
        batch = OpaqueBatch(
            ["north", "south"],
            "site",
            label="labels",
        )
        raw = batch.raw
        assert raw.dtype == object and list(raw) == ["north", "south"]
        assert not raw.flags.writeable

    def test_a_function_batch_is_the_object_array_of_its_callables(self):
        def double(x):
            return 2 * x

        assert (
            FunctionBatch(
                [double],
                "variant",
                label="f",
            ).raw[0]
            is double
        )

    def test_a_batch_of_laws_is_the_object_array_of_the_stored_laws(self):
        law = Gaussian("g", 1.0)
        raw = DistributionBatch(
            [law, Gaussian("g", 2.0)],
            "law",
            label="laws",
        ).raw
        assert raw.dtype == object and raw[0] is law

    def test_a_sub_batch_is_a_view_of_the_same_store(self):
        stored = jnp.arange(6.0)
        batch = NumericArrayBatch(
            stored,
            "draw",
            element=NumericArraySpec(()),
            label="x",
        )
        np.testing.assert_array_equal(batch[2:4].raw, stored[2:4])


class TestFunctions:
    def test_a_function_is_its_wrapped_callable(self):
        def add(x, y):
            return x + y

        assert (
            Function(
                add,
                label="add",
            ).raw
            is add
        )


class TestDistributions:
    def test_a_law_is_itself_detached(self):
        law = Gaussian("g", 1.0).with_provenance(Provenance.create("made", parents=[]))
        object.__setattr__(law, "_annotations", {"diagnostics": {"n_eff": 42}})
        detached = law.raw
        assert isinstance(detached, Distribution) and detached is not law
        assert (detached.label, detached.spec) == (law.label, law.spec)
        assert detached.provenance is None and detached.annotations is None
        assert detached.loc == 1.0

    def test_a_batch_element_drops_its_container(self):
        element = DistributionBatch(
            [Gaussian("g"), Gaussian("g", 2.0)],
            "law",
            label="laws",
        )[1]
        assert _element_source(element) is not None
        assert _element_source(element.raw) is None

    def test_an_empirical_law_keeps_its_atoms(self):
        law = EmpiricalDistribution(jnp.arange(4.0), output_name="e")
        detached = law.raw
        assert isinstance(detached, EmpiricalDistribution)
        np.testing.assert_array_equal(detached.atoms.raw, law.atoms.raw)

    def test_the_backend_adapter_is_its_backend_distribution(self):
        assert isinstance(Normal("x", 0.0, 1.0).raw, tfd.Normal)

    def test_a_kernel_is_itself_detached(self):
        kernel = Kernel("k")
        detached = kernel.raw
        assert isinstance(detached, ConditionalDistribution) and detached is not kernel
        assert (detached.label, detached.spec) == (kernel.label, kernel.spec)
        assert detached.provenance is None


class TestTheRawFormIsShared:
    """Every read of ``raw`` returns the same object, and a record a new mapping of the same leaves."""

    def test_a_law_returns_one_detached_copy(self):
        law = Gaussian("g", 1.0)
        assert law.raw is law.raw

    def test_a_kernel_returns_one_detached_copy(self):
        kernel = Kernel("k")
        assert kernel.raw is kernel.raw

    def test_a_field_view_resolves_its_marginal_once(self):
        view = (Normal("a", 0.0, 1.0) * Normal("b", 1.0, 1.0))["a"]
        assert view.raw is view.raw

    def test_a_failed_resolution_is_not_kept(self):
        view = Pair("p")["a"]
        for _ in range(2):
            with pytest.raises(ResolutionError):
                _ = view.raw
        assert "raw" not in transient_memo(view)

    def test_a_record_reads_the_same_leaves_into_a_new_mapping(self):
        law = EmpiricalDistribution(jnp.arange(3.0), output_name="e")
        record = Record(
            {"g": {"law": law, "x": jnp.ones(2)}, "label": "A"},
            label="r",
        )
        first, second = record.raw, record.raw
        assert type(first) is dict and type(first["g"]) is dict
        assert first is not second and first["g"] is not second["g"]
        assert first["g"]["law"] is second["g"]["law"]
        first["g"]["x"] = None
        first["label"] = None
        assert record.raw["g"]["x"] is not None and record.raw["label"] == "A"

    def test_a_record_batch_reads_the_same_columns_into_a_new_mapping(self):
        batch = _numeric_record_batch()
        first = batch.raw
        first["g"]["a"] = None
        first["c"] = None
        second = batch.raw
        assert second["g"]["a"] is batch._raw_column("g/a")
        assert second["c"] is batch._raw_column("c")

    def test_a_stored_array_stays_read_only_on_a_second_read(self):
        record = Record(
            {"x": np.arange(3.0)},
            label="r",
        )
        _ = record.raw
        assert not record.raw["x"].flags.writeable

    def test_a_copy_and_an_unpickled_law_rebuild_their_raw_form(self):
        law = Gaussian("g", 1.0)
        detached = law.raw
        assert copy.copy(law).raw is not detached
        restored = pickle.loads(pickle.dumps(law))
        assert restored.raw is not detached and restored.raw.loc == detached.loc

    def test_a_copy_and_an_unpickled_record_rebuild_their_raw_form(self):
        record = Record(
            {"x": jnp.ones(2)},
            label="r",
        )
        _ = record.raw
        assert getattr(copy.copy(record), "_raw_cache", None) is None
        assert getattr(pickle.loads(pickle.dumps(record)), "_raw_cache", None) is None

    def test_a_relabeled_law_has_a_raw_form_of_its_own(self):
        law = Gaussian("g", 1.0)
        assert law.with_label("h").raw is not law.raw

    def test_a_read_changes_no_hash_equality_or_fingerprint(self):
        law = Gaussian("g", 1.0)
        record = Record(
            {"law": law, "x": jnp.ones(2)},
            label="r",
        )
        before = (hash(record), fingerprint(law), fingerprint(record))
        _ = law.raw
        _ = record.raw
        assert (hash(record), fingerprint(law), fingerprint(record)) == before
        assert record == Record({"law": law, "x": jnp.ones(2)}, label="r")

    def test_a_read_inside_a_trace_keeps_nothing(self):
        traced = []

        def total(r):
            traced.append(r)
            return r.raw["x"].sum()

        record = NumericRecord(
            {"x": jnp.ones(2)},
            label="r",
        )
        assert float(jax.jit(total)(record)) == 2.0
        assert getattr(traced[0], "_raw_cache", None) is None
        assert getattr(record, "_raw_cache", None) is None


class TestARawFormRefusesProvenance:
    def test_a_raw_form_reports_no_provenance_and_refuses_one(self):
        detached = Gaussian("g").raw
        assert detached.provenance is None
        with pytest.raises(RuntimeError, match="it is the raw form of a term, which carries none"):
            detached.with_provenance(Provenance.create("made", parents=[]))

    def test_the_raw_form_of_a_raw_form_is_itself(self):
        detached = Gaussian("g").raw
        assert detached.raw is detached

    def test_a_relabeled_copy_is_not_a_raw_form(self):
        assert not _is_raw_form(Gaussian("g").raw.with_label("h"))

    def test_a_slotted_raw_form_clears_its_mark_on_a_copy(self):
        class Slotted(TrackedTerm):
            __slots__ = ("_expression", "_label", "_label_collapse", "_provenance")

            def __init__(self):
                self._init_tracked("slotted")

            @property
            def raw(self):
                return self

        term = _as_raw_form(Slotted())
        assert repr(term._provenance) == "<raw form>"
        assert _is_raw_form(term) and not _is_raw_form(copy.copy(term))

    def test_an_unpickled_copy_can_carry_provenance(self):
        restored = pickle.loads(pickle.dumps(Gaussian("g").raw))
        provenance = Provenance.create("made", parents=[])
        assert restored.with_provenance(provenance).provenance is provenance


class TestTheRemovedAccessors:
    def test_an_array_has_no_value(self):
        assert not hasattr(NumericArray(jnp.ones(2), label="x"), "value")

    def test_an_opaque_value_has_no_value(self):
        assert not hasattr(Opaque("north", label="site"), "value")

    def test_an_array_batch_has_no_values(self):
        batch = NumericArrayBatch(
            jnp.zeros((4, 2)),
            "draw",
            element=NumericArraySpec((2,)),
            label="x",
        )
        assert not hasattr(batch, "values")
