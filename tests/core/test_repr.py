"""The repr convention of design II.4 across the kinds, the grouping of labels, and the notation."""

from __future__ import annotations

import copy
import pickle

import jax.numpy as jnp
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    Gamma,
    InputSpec,
    KDEDistribution,
    Normal,
    NumericArray,
    NumericArraySpec,
    NumericRecordBatch,
    Opaque,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordBatch,
    RecordSpec,
    positive,
)
from probpipe.core._dispatch import MethodInfo
from probpipe.core._repr import (
    WIDTH,
    format_notation,
    format_signature,
    grouped_label,
    is_expression,
    is_product,
)
from probpipe.distributions._batches import DistributionBatch
from probpipe.linalg import DenseLinOp, DiagonalLinOp


def _schools() -> RecordBatch:
    return RecordBatch.stack(
        [
            Record("school", {"data": {"effect": float(y), "se": 1.0}, "label": label})
            for y, label in zip(range(8), "ABCDEFGH", strict=True)
        ],
        level_name="school",
        label="schools",
    )


def _two_fields() -> NumericRecordBatch:
    """Two record atoms over the fields ``a`` and ``b``."""
    columns = {"a": jnp.array([0.0, 1.0]), "b": jnp.array([1.0, 3.0])}
    return NumericRecordBatch("rows", columns, "row")


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

    def test_names_that_are_no_identifiers_read_as_one_mapping(self):
        """A derived component such as mean(theta) cannot be a keyword, so the call maps it."""
        output = OutputSpec(**{"mean(theta)": NumericArraySpec(())})
        assert repr(output) == "OutputSpec(**{'mean(theta)': NumericArraySpec(shape=())})"
        record = RecordSpec({"mean(mu)": (), "mean(tau)": ()})
        assert repr(record) == "NumericRecordSpec(**{'mean(mu)': (), 'mean(tau)': ()})"
        assert eval(repr(output)) == output

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
        assert renamed.startswith("FactoredMultivariateGaussian(\n    'a·b',\n    factors=(")
        assert "_Renamed" not in renamed

    def test_an_empirical_law_reads_by_its_atoms(self):
        law = EmpiricalDistribution("e", jnp.arange(5.0))
        assert repr(law).startswith(
            "EmpiricalDistribution(\n    'e',\n    atoms=NumericArrayBatch("
        )

    def test_a_renamed_empirical_law_reads_by_its_renamed_atoms(self):
        renamed = EmpiricalDistribution("e", _two_fields()).with_path_names({"a": "g/a"})
        assert repr(renamed) == (
            "EmpiricalDistribution('e', atoms=NumericRecordBatch('rows', levels={'row': 2}, "
            "fields=('b', 'g/a')))"
        )

    def test_a_rename_that_holds_its_law_reads_as_that_law_renamed(self):
        """A kernel density estimate does not rebuild itself, so its rename holds it."""
        kde = KDEDistribution("kde", _two_fields())
        renamed = kde.with_path_names({"a": "g/a"})
        assert repr(renamed) == repr(kde) + ".with_path_names({'a': 'g/a'})"
        assert repr(renamed.with_label("other")) == repr(renamed) + ".with_label('other')"

    def test_a_reordered_marginal_reads_as_the_joint_under_its_declaration(self):
        """No rename states a reorder, so the repr shows the reordered declaration."""
        joint = Normal("a", 0.0, 1.0) * Gamma("b", 2.0, 1.0)
        reordered = repr(joint._marginal(("b", "a")))
        assert reordered.startswith("FactoredDistribution(\n    'a·b',\n    factors=(")
        assert reordered.index("b=NumericArraySpec") < reordered.index("a=NumericArraySpec")


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


class TestGrouping:
    """A label is grouped where a derived label is built from it (II.4)."""

    @pytest.mark.parametrize(
        "label",
        [
            "effect + 1.0",
            "model | y",
            "-effect",
            "mu ~ prior",
            "(y, mu) ~ lik·prior",
            "lik·prior",
            "lik·(model | y)",
            "log prior(mu)",
        ],
    )
    def test_an_expression_is_parenthesized(self, label):
        assert is_expression(label)
        assert grouped_label(label) == f"({label})"

    @pytest.mark.parametrize("label", ["prior", "prior(mu)", "x[sample=0]", "E[mu ~ prior]"])
    def test_a_label_of_one_word_is_used_as_it_is(self, label):
        """A call or a selection holds its spaces and symbols inside its brackets."""
        assert not is_expression(label)
        assert grouped_label(label) == label

    @pytest.mark.parametrize("label", ["my prior", "logit prior"])
    def test_any_other_label_of_several_words_is_bracketed(self, label):
        """Only the word ``log`` opens a score, so ``logit prior`` is a label of two words."""
        assert not is_expression(label)
        assert grouped_label(label) == f"[{label}]"

    @pytest.mark.parametrize("label", ["f(lik·prior)", "E[lik·prior]", "f(mu ~ prior)"])
    def test_a_symbol_inside_parentheses_or_brackets_is_not_at_the_top_level(self, label):
        assert not is_expression(label)
        assert grouped_label(label) == label

    @pytest.mark.parametrize("label", ["lik·prior", "a·b·c", "lik·(model | y)", "lik·[my prior]"])
    def test_a_product_is_one_word_that_joins_labels_with_a_middle_dot(self, label):
        assert is_product(label)

    @pytest.mark.parametrize(
        "label", ["lik", "(lik·prior) | y", "(y, mu) ~ lik·prior", "log lik·prior", "-x·y"]
    )
    def test_any_other_label_is_not_a_product(self, label):
        assert not is_product(label)


class TestSignatureAndNotation:
    """The signature states what a term is over, and the notation is ``label(signature)``."""

    def test_a_signature_joins_the_components(self):
        assert format_signature(["y", "mu"]) == "y, mu"

    def test_given_slots_follow_a_bar(self):
        assert format_signature(["y"], ["beta", "sigma"]) == "y | beta, sigma"

    def test_fixed_paths_follow_a_semicolon(self):
        assert format_signature(["mu"], fixed=["y"]) == "mu; y"
        assert format_signature(["y"], ["sigma"], ["beta"]) == "y | sigma; beta"

    def test_a_signature_of_no_components_is_empty(self):
        assert format_signature([]) == ""

    @pytest.mark.parametrize(
        ("label", "notation"),
        [
            ("prior", "prior(mu)"),
            ("lik·prior", "(lik·prior)(mu)"),
            ("my prior", "[my prior](mu)"),
            ("model | y", "(model | y)(mu)"),
        ],
    )
    def test_the_notation_groups_the_label(self, label, notation):
        assert format_notation(label, "mu") == notation
