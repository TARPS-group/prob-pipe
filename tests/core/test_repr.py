"""The repr convention of design II.4, across the kinds."""

from __future__ import annotations

import copy
import pickle

import jax.numpy as jnp
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    InputSpec,
    Normal,
    NumericArray,
    NumericArraySpec,
    Opaque,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordBatch,
    RecordSpec,
    positive,
)
from probpipe.core._dispatch import MethodInfo
from probpipe.core._repr import WIDTH
from probpipe.distributions._batches import DistributionBatch
from probpipe.linalg import DenseLinOp, DiagonalLinOp


def _schools() -> RecordBatch:
    return RecordBatch.stack(
        [
            Record("school", {"data": {"effect": float(y), "se": 1.0}, "label": label})
            for y, label in zip(range(8), "ABCDEFGH", strict=True)
        ],
        level_name="school",
        name="schools",
    )


class TestValuesAndBatches:
    def test_an_array_reads_by_its_label_shape_and_dtype(self):
        assert repr(NumericArray("x", jnp.zeros(3))) == (
            "NumericArray('x', shape=(3,), dtype=float32)"
        )

    def test_a_declared_support_is_shown(self):
        spec = NumericArraySpec((), jnp.float32, positive)
        assert repr(NumericArray("tau", jnp.asarray(1.0), spec=spec)) == (
            "NumericArray('tau', shape=(), dtype=float32, support=positive)"
        )

    def test_an_opaque_value_reads_by_its_type(self):
        assert repr(Opaque("note", "Rubin")) == "Opaque('note', type=str)"

    def test_a_record_reads_by_its_field_paths(self):
        school = Record("school", {"data": {"effect": 28.0, "se": 15.0}, "label": "A"})
        assert repr(school) == "Record('school', fields=('data/effect', 'data/se', 'label'))"

    def test_a_batch_of_records_reads_by_its_levels_and_field_paths(self):
        assert repr(_schools()) == (
            "RecordBatch('schools', levels={'school': 8}, fields=('data/effect', 'data/se', "
            "'label'))"
        )

    def test_a_batch_of_laws_reads_by_its_element_spec(self):
        laws = DistributionBatch("laws", [Normal("a", 0.0, 1.0), Normal("a", 1.0, 1.0)], "law")
        assert repr(laws) == (
            "DistributionBatch(\n"
            "    'laws',\n"
            "    levels={'law': 2},\n"
            "    element_spec=DistributionSpec(\n"
            "        event_spec=OutputSpec(a=NumericArraySpec(shape=(), dtype=float32, "
            "support=real)),\n"
            "    ),\n"
            ")"
        )


class TestSpecs:
    def test_a_spec_shows_the_attributes_it_sets(self):
        assert repr(NumericArraySpec((3,))) == "NumericArraySpec(shape=(3,))"
        assert repr(OpaqueSpec(meta="units")) == "OpaqueSpec(meta='units')"

    def test_a_declaration_reads_as_its_constructor_call(self):
        assert repr(OutputSpec(beta=None)) == "OutputSpec(beta=None)"
        assert repr(OutputSpec(RecordSpec(a=()))) == "OutputSpec(NumericRecordSpec(a=()))"
        assert repr(InputSpec(x=NumericArraySpec((2,)))) == (
            "InputSpec(x=NumericArraySpec(shape=(2,)))"
        )

    def test_a_batch_spec_names_its_levels(self):
        assert repr(_schools().spec).endswith("    levels={'school': 8},\n)")


class TestDistributions:
    def test_a_family_reads_by_the_arguments_it_was_built_with(self):
        assert repr(Normal("x", 0.0, 1.0)) == "Normal('x', loc=0.0, scale=1.0)"

    def test_a_declared_component_shows_the_event_declaration(self):
        law = Normal("prior", 0.0, 1.0, event_spec=OutputSpec(beta=None))
        assert "event_spec=OutputSpec(beta=NumericArraySpec(" in repr(law)

    def test_a_field_view_reads_as_a_field_view_at_its_path(self):
        joint = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        assert repr(joint["a"]) == "FieldView('a·b', path='a')"

    def test_a_regrouped_rename_reads_as_a_factored_joint(self):
        joint = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        renamed = repr(joint.with_path_names({"a": "g/a"}))
        assert renamed.startswith("FactoredDistribution(\n    'a·b',\n    factors=(")
        assert "_Renamed" not in renamed

    def test_an_empirical_law_reads_by_its_atoms(self):
        law = EmpiricalDistribution("e", jnp.arange(5.0))
        assert repr(law).startswith(
            "EmpiricalDistribution(\n    'e',\n    atoms=NumericArrayBatch("
        )


class TestFunctionsAndOperators:
    def test_a_function_reads_by_its_label_and_parameters(self):
        def predict(theta, x):
            return x * theta

        assert repr(Function("predict", predict)) == (
            "Function('predict', parameters=('theta', 'x'))"
        )

    def test_a_composite_operator_shows_its_operands(self):
        product = DenseLinOp(jnp.eye(2)) @ DiagonalLinOp(jnp.array([1.0, 2.0]))
        assert repr(product) == (
            "ProductLinOp(\n"
            "    shape=(2, 2),\n"
            "    dtype=float32,\n"
            "    operands=(DenseLinOp(shape=(2, 2), dtype=float32), DiagonalLinOp(shape=(2, 2), "
            "dtype=float32)),\n"
            ")"
        )

    def test_a_report_shows_the_fields_it_sets(self):
        info = MethodInfo(True, method_name="closed_form", exact=True)
        assert repr(info) == "MethodInfo(True, method_name='closed_form', exact=True)"


class TestLayout:
    @pytest.mark.parametrize(
        "term",
        [
            _schools(),
            Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0),
            EmpiricalDistribution("e", jnp.arange(5.0)),
        ],
        ids=["batch", "joint", "empirical"],
    )
    def test_no_line_passes_the_width(self, term):
        assert all(len(line) <= WIDTH for line in repr(term).splitlines())

    def test_a_repr_copies_and_pickles_as_a_string(self):
        text = repr(Normal("x", 0.0, 1.0))
        assert type(copy.deepcopy(text)) is str and pickle.loads(pickle.dumps(text)) == text
