"""The repr convention of design II.4 across the kinds, the grouping of labels, and the notation."""

from __future__ import annotations

import copy
import pickle

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import probpipe
from probpipe import (
    BatchSpec,
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
    TFPDistribution,
    condition_on,
    conditional_distribution,
    distribution,
    positive,
    workflow_run,
)
from probpipe.core._dispatch import MethodInfo
from probpipe.core._repr import (
    WIDTH,
    format_default,
    format_notation,
    format_signature,
    grouped_label,
    is_compound,
    is_product,
)
from probpipe.distributions._batches import DistributionBatch
from probpipe.families import BernoulliFamily, MixtureDistribution, glm_likelihood
from probpipe.linalg import DenseLinOp, DiagonalLinOp


def _schools() -> RecordBatch:
    return RecordBatch.stack(
        [
            Record(
                {"data": {"effect": float(y), "se": 1.0}, "label": label},
                label="school",
            )
            for y, label in zip(range(8), "ABCDEFGH", strict=True)
        ],
        level_name="school",
        label="schools",
    )


def _two_fields() -> NumericRecordBatch:
    """Two record atoms over the fields ``a`` and ``b``."""
    columns = {"a": jnp.array([0.0, 1.0]), "b": jnp.array([1.0, 3.0])}
    return NumericRecordBatch(
        columns,
        "row",
        label="rows",
    )


class TestValuesAndBatches:
    def test_an_array_reads_by_its_label_shape_and_dtype(self):
        assert repr(
            NumericArray(
                jnp.zeros(3),
                label="x",
            )
        ) == ("NumericArray('x', shape=(3,), dtype=float32)")

    def test_a_declared_support_is_shown(self):
        spec = NumericArraySpec((), jnp.float32, positive)
        assert repr(
            NumericArray(
                jnp.asarray(1.0),
                spec=spec,
                label="tau",
            )
        ) == ("NumericArray('tau', shape=(), dtype=float32, support=positive)")

    def test_an_opaque_value_reads_by_its_type(self):
        assert (
            repr(
                Opaque(
                    "Rubin",
                    label="note",
                )
            )
            == "Opaque('note', type=str)"
        )

    def test_a_record_reads_by_its_field_paths(self):
        school = Record(
            {"data": {"effect": 28.0, "se": 15.0}, "label": "A"},
            label="school",
        )
        assert repr(school) == "Record('school', fields=('data/effect', 'data/se', 'label'))"

    def test_a_batch_of_records_reads_by_its_levels_and_field_paths(self):
        assert repr(_schools()) == (
            "RecordBatch('schools', levels={'school': 8}, fields=('data/effect', 'data/se', "
            "'label'))"
        )

    def test_a_batch_of_laws_reads_by_its_element_spec(self):
        laws = DistributionBatch(
            [Normal("a", 0.0, 1.0), Normal("a", 1.0, 1.0)],
            "law",
            label="laws",
        )
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
        assert repr(_schools().spec).endswith("    school=8,\n)")

    @pytest.mark.parametrize(
        "spec",
        [
            BatchSpec(NumericArraySpec(()), chain=4, draw="S"),
            BatchSpec(NumericArraySpec(("n",)), grid=(3, "n")),
            BatchSpec(NumericArraySpec(()), {"my level": 2, "draw": 3}),
        ],
        ids=["keywords", "multi-axis", "mapping"],
    )
    def test_a_batch_spec_reads_as_the_call_that_builds_it(self, spec):
        """A level of one axis shows its size alone; a name no keyword spells uses ``**``."""
        assert eval(repr(spec)) == spec

    def test_a_batch_spec_level_no_keyword_spells_is_written_in_a_mapping(self):
        spec = BatchSpec(NumericArraySpec(()), {"my level": 2})

        assert repr(spec) == "BatchSpec(NumericArraySpec(shape=()), **{'my level': 2})"


class TestDistributions:
    def test_a_family_reads_as_the_call_that_built_it(self):
        assert repr(Normal("mu", 0.0, 1.0, label="prior")) == (
            "Normal('mu', loc=0.0, scale=1.0, label='prior')"
        )

    def test_a_family_under_its_default_label_leaves_the_label_out(self):
        assert repr(Normal("x", 0.0, 1.0)) == "Normal('x', loc=0.0, scale=1.0)"

    def test_a_label_equal_to_the_default_is_left_out_however_it_was_given(self):
        relabeled = Normal("x", 0.0, 1.0, label="prior").with_label("Normal")
        assert repr(relabeled) == "Normal('x', loc=0.0, scale=1.0)"
        assert repr(Normal("x", 0.0, 1.0, label="Normal")) == repr(relabeled)

    def test_a_law_labeled_by_an_operation_shows_the_label(self):
        kernel = conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), label="lik", given_spec={"mu": NumericArraySpec(())}
        )
        assert repr(condition_on(kernel, {"mu": 0.5})).startswith(
            "Normal('y', loc=0.5, scale=1.0, label='lik'"
        )

    def test_the_adapter_under_its_backend_name_leaves_the_label_out(self):
        law = TFPDistribution("x", tfd.Normal(0.0, 1.0))
        assert repr(law).startswith("TFPDistribution(\n    'x',\n    backend_dist=")
        assert "label=" not in repr(law)
        labeled = TFPDistribution("x", tfd.Normal(0.0, 1.0), label="q")
        assert repr(labeled).endswith(",\n    label='q',\n)")

    def test_a_law_under_the_label_p_leaves_the_label_out(self):
        law = EmpiricalDistribution(jnp.arange(3.0), component="theta")
        assert repr(law).startswith("EmpiricalDistribution(\n    atoms=")
        assert repr(law).endswith("    component='theta',\n)")
        drawn = distribution(
            sample=lambda key: jax.random.normal(key),
            event_spec=NumericArraySpec(()),
            component="z",
        )
        assert repr(drawn).startswith("Distribution(\n    'z',\n    sample=")

    def test_a_kernel_under_the_label_p_leaves_the_label_out(self):
        kernel = conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), given_spec={"mu": NumericArraySpec(())}
        )
        assert repr(kernel) == "ConditionalDistribution('y', given=('mu',))"
        assert repr(glm_likelihood("damage", BernoulliFamily())).startswith(
            "ConditionalDistribution(\n    'damage',\n    family=BernoulliFamily(),"
        )

    def test_a_conditioned_law_shows_its_fixed_paths_after_the_call(self):
        kernel = conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), label="lik", given_spec={"mu": NumericArraySpec(())}
        )
        assert repr(condition_on(kernel, {"mu": 0.5})) == (
            "Normal('y', loc=0.5, scale=1.0, label='lik', fixed=('mu',))"
        )

    def test_a_curried_kernel_shows_its_fixed_slots(self):
        def two_slot(beta, sigma):
            return Normal("y", beta, sigma)

        kernel = conditional_distribution(
            two_slot,
            label="glm",
            given_spec={"beta": NumericArraySpec(()), "sigma": NumericArraySpec(())},
        )
        assert repr(condition_on(kernel, {"beta": 0.5})) == (
            "ConditionalDistribution('y', given=('sigma',), label='glm', fixed=('beta',))"
        )

    def test_a_posterior_reads_apart_from_its_atoms_law(self):
        """The fixed paths follow the call, so a posterior differs from its atoms' law."""
        likelihood = conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), label="lik", given_spec={"mu": NumericArraySpec(())}
        )
        prior = EmpiricalDistribution(jnp.linspace(-1.0, 1.0, 5), component="mu", label="prior")
        model = (likelihood * prior).with_label("model")
        with workflow_run(seed=0):
            posterior = condition_on(model, {"y": 0.5})
        text = repr(posterior)
        assert text.startswith("EmpiricalDistribution(\n    atoms=")
        assert text.endswith("    label='model',\n    fixed=('y',),\n)")
        assert "fixed=" not in repr(EmpiricalDistribution(posterior.atoms, label="model"))

    def test_a_law_that_holds_nothing_fixed_shows_no_fixed_item(self):
        assert "fixed=" not in repr(Normal("x", 0.0, 1.0, label="prior"))

    def test_a_kernel_named_by_its_function_shows_the_label(self):
        def y_given_mu(mu):
            return Normal("y", mu, 1.0)

        kernel = conditional_distribution(y_given_mu, given_spec={"mu": NumericArraySpec(())})
        assert repr(kernel) == ("ConditionalDistribution('y', given=('mu',), label='y_given_mu')")

    def test_a_whole_term_event_shows_its_component_and_no_declaration(self):
        law = Normal("beta", 0.0, 1.0, label="prior")
        assert repr(law).startswith("Normal('beta', ")
        assert "event_spec=" not in repr(law)

    def test_a_field_view_reads_as_a_call_of_its_constructor(self):
        joint = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        nested = repr(joint).replace("\n", "\n    ")
        assert repr(joint["a"]) == f"FieldView(\n    {nested},\n    path='a',\n)"
        assert repr(joint["a"].with_label("x")) == repr(joint["a"]) + ".with_label('x')"

    def test_a_regrouped_rename_reads_as_a_factored_joint(self):
        joint = Normal("a", 0.0, 1.0, label="a") * Normal("b", 0.0, 1.0, label="b")
        renamed = repr(joint.with_path_names({"a": "g/a"}))
        assert renamed.startswith("FactoredMultivariateGaussian(\n    factors=(")
        assert "label='a·b'" in renamed
        assert "_Renamed" not in renamed

    def test_an_empirical_law_reads_by_its_atoms(self):
        law = EmpiricalDistribution(jnp.arange(5.0), component="e", label="draws")
        assert repr(law).startswith("EmpiricalDistribution(\n    atoms=NumericArrayBatch(")
        assert repr(law).endswith("    component='e',\n    label='draws',\n)")

    def test_a_renamed_empirical_law_reads_by_its_renamed_atoms(self):
        renamed = EmpiricalDistribution(_two_fields(), label="e").with_path_names({"a": "g/a"})
        assert repr(renamed) == (
            "EmpiricalDistribution(\n"
            "    atoms=NumericRecordBatch('rows', levels={'row': 2}, fields=('b', 'g/a')),\n"
            "    label='e',\n"
            ")"
        )

    def test_a_rename_that_holds_its_law_reads_as_that_law_renamed(self):
        """A kernel density estimate does not rebuild itself, so its rename holds it."""
        kde = KDEDistribution(_two_fields(), label="kde")
        renamed = kde.with_path_names({"a": "g/a"})
        assert repr(renamed) == repr(kde) + ".with_path_names({'a': 'g/a'})"
        assert repr(renamed.with_label("other")) == repr(renamed) + ".with_label('other')"

    def test_a_reordered_marginal_reads_as_the_joint_under_its_declaration(self):
        """No rename states a reorder, so the repr shows the reordered declaration.

        The kernel ``y`` conditions on ``a``, so no product lists ``a`` first.
        """
        likelihood = conditional_distribution(
            lambda a: Normal("y", a, 1.0), given_spec={"a": NumericArraySpec(())}, label="y"
        )
        joint = likelihood * Normal("a", 0.0, 1.0, label="a")
        reordered = repr(joint._marginal(("a", "y")))
        assert reordered.startswith("FactoredDistribution(\n    factors=(")
        assert "label='y·a'" in reordered
        assert reordered.index("a=NumericArraySpec") < reordered.index("y=NumericArraySpec")

    def test_a_marginal_in_another_product_order_reads_as_that_product(self):
        joint = Normal("a", 0.0, 1.0) * Gamma("b", 2.0, 1.0)
        assert repr(joint._marginal(("b", "a"))) == (
            "FactoredDistribution(\n"
            "    factors=(Gamma('b', concentration=2.0, rate=1.0), Normal('a', loc=0.0, scale=1.0)),\n"
            "    label='Gamma·Normal',\n"
            ")"
        )


#: One law of each catalog family with a constructor-call repr, labeled so the label reads too.
_EVALUABLE = {
    "Normal": lambda: Normal("x", 0.0, 1.0, label="prior"),
    "Beta": lambda: probpipe.Beta("x", 2.0, 3.0, label="prior"),
    "Gamma": lambda: Gamma("x", 2.0, 1.0, label="prior"),
    "InverseGamma": lambda: probpipe.InverseGamma("x", 2.0, 1.0, label="prior"),
    "Exponential": lambda: probpipe.Exponential("x", 1.0, label="prior"),
    "LogNormal": lambda: probpipe.LogNormal("x", 0.0, 1.0, label="prior"),
    "StudentT": lambda: probpipe.StudentT("x", 3.0, 0.0, 1.0, label="prior"),
    "Uniform": lambda: probpipe.Uniform("x", -1.0, 2.0, label="prior"),
    "Cauchy": lambda: probpipe.Cauchy("x", 0.0, 1.0, label="prior"),
    "Laplace": lambda: probpipe.Laplace("x", 0.0, 1.0, label="prior"),
    "HalfNormal": lambda: probpipe.HalfNormal("x", 1.0, label="prior"),
    "HalfCauchy": lambda: probpipe.HalfCauchy("x", 0.5, 1.0, label="prior"),
    "Pareto": lambda: probpipe.Pareto("x", 2.0, 1.5, label="prior"),
    "TruncatedNormal": lambda: probpipe.TruncatedNormal("x", 0.0, 1.0, -1.0, 1.0, label="prior"),
    "Bernoulli": lambda: probpipe.Bernoulli("x", probs=0.3, label="prior"),
    "Binomial": lambda: probpipe.Binomial("x", 5, probs=0.3, label="prior"),
    "Poisson": lambda: probpipe.Poisson("x", 2.0, label="prior"),
    "Categorical": lambda: probpipe.Categorical("x", probs=[0.2, 0.3, 0.5], label="prior"),
    "NegativeBinomial": lambda: probpipe.NegativeBinomial("x", 5.0, probs=0.3, label="prior"),
    "MultivariateNormal": lambda: probpipe.MultivariateNormal(
        "x", jnp.zeros(2), cov=jnp.array([[2.0, 0.5], [0.5, 1.0]]), label="prior"
    ),
    "Dirichlet": lambda: probpipe.Dirichlet("x", jnp.ones(3), label="prior"),
    "Multinomial": lambda: probpipe.Multinomial(
        "x", 4.0, probs=jnp.array([0.2, 0.3, 0.5]), label="prior"
    ),
    "Wishart": lambda: probpipe.Wishart("x", 4.0, scale_tril=jnp.eye(2), label="prior"),
    "VonMisesFisher": lambda: probpipe.VonMisesFisher(
        "x", jnp.array([0.0, 1.0]), 2.0, label="prior"
    ),
    "MixtureDistribution": lambda: MixtureDistribution(
        [Normal("x", 0.0, 1.0), Normal("x", 1.0, 2.0)], jnp.array([0.25, 0.75]), label="mix"
    ),
}


@pytest.mark.parametrize("make", list(_EVALUABLE.values()), ids=list(_EVALUABLE))
def test_the_repr_of_a_law_that_holds_nothing_fixed_rebuilds_it(make):
    """``eval(repr(d))`` builds a law of the same class, label, and parameters (II.4)."""
    law = make()
    namespace = {name: getattr(probpipe, name) for name in probpipe.__all__}
    rebuilt = eval(repr(law), {**namespace, "MixtureDistribution": MixtureDistribution})
    assert type(rebuilt) is type(law)
    assert (rebuilt.label, rebuilt.event_spec) == (law.label, law.event_spec)
    assert repr(rebuilt) == repr(law)


def test_the_repr_of_a_law_that_holds_paths_fixed_follows_the_call_with_them():
    kernel = conditional_distribution(
        lambda mu: Normal("y", mu, 1.0), label="lik", given_spec={"mu": NumericArraySpec(())}
    )
    law = condition_on(kernel, {"mu": 0.5})
    call = "Normal('y', loc=0.5, scale=1.0, label='lik')"
    assert repr(law) == call[:-1] + ", fixed=('mu',))"
    assert repr(eval(call, {"Normal": Normal})) == call


class TestFunctionsAndOperators:
    def test_a_function_reads_by_its_label_and_parameters(self):
        def predict(theta, x):
            return x * theta

        assert repr(
            Function(
                predict,
                label="predict",
            )
        ) == ("Function('predict', parameters=('theta', 'x'))")

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
            EmpiricalDistribution(jnp.arange(5.0), component="e"),
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
        assert is_compound(label)
        assert grouped_label(label) == f"({label})"

    @pytest.mark.parametrize("label", ["prior", "prior(mu)", "x[sample=0]", "𝔼[mu ~ prior]"])
    def test_a_label_of_one_word_is_used_as_it_is(self, label):
        """A call or a selection holds its spaces and symbols inside its brackets."""
        assert not is_compound(label)
        assert grouped_label(label) == label

    @pytest.mark.parametrize("label", ["my prior", "logit prior"])
    def test_any_other_label_of_several_words_is_bracketed(self, label):
        """Only the word ``log`` opens a score, so ``logit prior`` is a label of two words."""
        assert not is_compound(label)
        assert grouped_label(label) == f"[{label}]"

    @pytest.mark.parametrize("label", ["f(lik·prior)", "𝔼[lik·prior]", "f(mu ~ prior)"])
    def test_a_symbol_inside_parentheses_or_brackets_is_not_at_the_top_level(self, label):
        assert not is_compound(label)
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

    def test_a_name_with_a_default_reads_name_equals_value(self):
        assert format_signature(["y"], ["K", "n0"], defaults={"n0": "50.0"}) == "y | K, n0=50.0"
        assert format_signature(["x", "scale"], defaults={"scale": "1.0"}) == "x, scale=1.0"

    @pytest.mark.parametrize(
        ("value", "text"),
        [(50.0, "50.0"), (3, "3"), ("exact", "'exact'"), (None, "None"), (jnp.asarray(2.0), "2.0")],
    )
    def test_a_scalar_default_reads_as_its_value(self, value, text):
        assert format_default(value) == text

    def test_any_other_default_reads_as_an_ellipsis(self):
        assert format_default(jnp.zeros(3)) == "…"
        assert format_default({"a": 1}) == "…"

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
