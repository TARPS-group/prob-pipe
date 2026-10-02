"""Field access returns tracked views of the field's kind (II.4, III.5, III.6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    Opaque,
    OpaqueBatch,
    OpaqueSpec,
    Record,
    RecordBatch,
    RecordSpec,
    mean,
)
from probpipe.core._fingerprint import fingerprint
from probpipe.core.config import ProvenanceMode


def _square(z):
    return z * z


def _mixed() -> Record:
    return Record("school", {"effect": jnp.arange(3.0), "label": "A", "model": _square})


class TestARecordFieldIsAViewOfItsKind:
    def test_an_array_field_is_a_numeric_array_under_its_key(self):
        record = _mixed()

        view = record["effect"]

        assert isinstance(view, NumericArray) and view.label == "effect"
        assert view.spec == record.spec["effect"]
        assert view.raw() is record.raw("effect")

    def test_an_opaque_field_is_an_opaque_under_its_key(self):
        view = _mixed()["label"]

        assert isinstance(view, Opaque) and view.label == "label"
        assert view.raw() == "A"

    def test_a_callable_field_is_a_function_under_its_key(self):
        view = _mixed()["model"]

        assert isinstance(view, Function) and view.label == "model"
        assert view.raw() is _square

    def test_a_stored_law_is_a_copy_under_its_key(self):
        law = Normal("prior", 0.0, 1.0)
        record = Record("r", theta=law)

        view = record["theta"]

        assert type(view) is type(law) and view.label == "theta"
        assert law.label == "prior"

    def test_a_nested_field_is_named_by_its_key(self):
        record = Record("r", {"g/x": jnp.zeros(2), "y": 1.0})

        assert record["g/x"].label == "g/x"
        assert record.at_path("g", "x").label == "g/x"
        assert isinstance(record.at_path("g"), Record)

    def test_values_and_items_give_the_views_indexing_gives(self):
        record = _mixed()

        for (key, view), value in zip(record.items(), record.values(), strict=True):
            assert type(view) is type(record[key]) is type(value)
            assert view.label == key == value.label


class TestAViewRecordsItsContainer:
    def test_the_provenance_names_the_record_and_the_path(self):
        view = _mixed()["effect"]

        assert view.provenance.operation == "__getitem__"
        assert [parent.label for parent in view.provenance.parents] == ["school"]
        assert view.provenance.metadata == {"path": "effect"}

    def test_a_stored_term_is_the_second_parent(self):
        record = Record("r", theta=Normal("prior", 0.0, 1.0))

        parents = record["theta"].provenance.parents

        assert [parent.label for parent in parents] == ["r", "prior"]

    def test_the_parents_are_identity_descriptors(self, full_provenance_mode):
        record = _mixed()

        (parent,) = record["effect"].provenance.parents

        assert parent.fingerprint_is_weak and parent.parent is record

    def test_no_provenance_is_recorded_when_tracking_is_off(self):
        import probpipe

        probpipe.provenance_config.mode = ProvenanceMode.OFF

        assert _mixed()["effect"].provenance is None


class TestAFieldInsideATraceIsItsLeaf:
    def test_a_traced_field_is_the_traced_array(self):
        seen = []

        @jax.jit
        def double(record):
            seen.append(record["x"])
            return record["x"] * 2

        out = double(Record("r", x=jnp.arange(3.0)))

        assert isinstance(seen[0], jax.core.Tracer)
        np.testing.assert_array_equal(out, [0.0, 2.0, 4.0])


class TestARecordOfViewsIsTheRecord:
    def test_a_rebuilt_record_is_equal_hashes_alike_and_fingerprints_alike(self):
        record = _mixed()

        rebuilt = Record(record.label, dict(record))

        assert rebuilt == record
        assert hash(rebuilt) == hash(record)
        assert fingerprint(rebuilt) == fingerprint(record)


class TestABatchColumnIsTheBatchOfItsKind:
    @pytest.fixture
    def batch(self) -> RecordBatch:
        return RecordBatch(
            "draws",
            {"x": jnp.arange(6.0).reshape(3, 2), "tag": np.array(["a", "b", "c"], dtype=object)},
            "draw",
            element_spec=RecordSpec(x=NumericArraySpec((2,)), tag=OpaqueSpec(type=str)),
        )

    def test_an_array_column_is_a_numeric_array_batch_on_the_batch_levels(self, batch):
        column = batch["x"]

        assert isinstance(column, NumericArrayBatch)
        assert column.label == "x" and column.level_names == ("draw",)
        assert column.element_spec == NumericArraySpec((2,))
        assert column.raw() is batch._raw_column("x")

    def test_an_opaque_column_is_an_opaque_batch(self, batch):
        column = batch["tag"]

        assert isinstance(column, OpaqueBatch)
        assert [element.raw() for element in column] == ["a", "b", "c"]

    def test_a_traced_column_is_the_traced_array(self):
        batch = NumericRecordBatch(
            "draws", {"x": jnp.arange(3.0)}, "draw", element_spec=RecordSpec(x=())
        )

        out = jax.jit(lambda b: b["x"] * 2)(batch)

        np.testing.assert_array_equal(out, [0.0, 2.0, 4.0])


class TestAMomentReadsTheSameWay:
    def test_a_record_mean_gives_views_of_its_fields(self):
        moment = mean(Normal("a", 0.0, 1.0) * Normal("b", 2.0, 1.0))

        assert isinstance(moment["b"], NumericArray) and moment["b"].label == "b"
        assert float(moment["b"]) == pytest.approx(2.0)
