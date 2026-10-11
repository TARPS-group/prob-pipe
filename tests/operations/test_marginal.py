"""Contract tests of marginal and factor: the detached parts of structured and factored laws."""

from __future__ import annotations

import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    DistributionBatch,
    Function,
    Normal,
    RecordSpec,
    condition_on,
    workflow_run,
)
from probpipe.core._dispatch import ResolutionError
from probpipe.core._specs import OutputSpec
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._distribution import Distribution, DistributionSpec, _fixed_paths
from probpipe.functions import _descendants
from probpipe.operations._marginal import factor, marginal

from ._laws import REAL, Gaussian, Kernel, Marginalizing, Pair


class _Nested(Distribution):
    """A law with two groups that each hold a field named ``a``."""

    def __init__(self, label: str) -> None:
        super().__init__(
            RecordSpec(x=RecordSpec(a=REAL), y=RecordSpec(a=REAL)),
            label=label,
        )


class TestMarginal:
    def test_the_capability_returns_the_detached_marginal_at_the_path(self):
        law = Marginalizing("law")
        result = marginal(law, "a")
        assert isinstance(result, Distribution)
        assert result is not law and result.loc == 5.0
        assert law.paths == ["a"]
        assert marginal.check(law, "a").route == "exact"

    def test_the_declaration_is_the_node_under_the_path_s_final_segment(self):
        result = marginal.check(Marginalizing("law"), "a").result
        assert result == OutputSpec(DistributionSpec(OutputSpec(a=REAL)))
        assert dict(result.components) == {"a": REAL}

    def test_several_paths_declare_an_exposed_record_of_the_nodes(self):
        result = marginal.check(Marginalizing("law"), ("a", "b")).result
        assert result == OutputSpec(DistributionSpec(OutputSpec(RecordSpec(a=REAL, b=REAL))))

    def test_two_paths_ending_in_the_same_segment_raise(self):
        with pytest.raises(ApplicabilityError, match="more than one path ends in 'a'"):
            marginal.check(_Nested("n"), ("x/a", "y/a"))

    def test_a_path_the_law_lacks_raises_applicability_error(self):
        with pytest.raises(
            ApplicabilityError, match="'c' is not an event path of the law; its fields"
        ):
            marginal(Marginalizing("law"), "c")

    def test_a_path_that_is_not_a_string_raises_applicability_error(self):
        with pytest.raises(ApplicabilityError, match="field must be a path string"):
            marginal(Marginalizing("law"), 3)

    def test_a_rejecting_guard_and_no_sampling_raise_resolution_error(self):
        with pytest.raises(ResolutionError, match="the marginal is exact at the field a"):
            marginal(Marginalizing("law"), "b")

    def test_check_and_the_call_agree_while_the_fallback_is_not_implemented(self):
        report = marginal.check(Pair("p"), "a")
        assert report.feasible is False
        assert "not implemented" in report.description
        with pytest.raises(ResolutionError, match="not implemented"):
            marginal(Pair("p"), "a")

    @pytest.mark.pending(
        reason="an empirical marginal declares the node's event", raises=ResolutionError
    )
    def test_the_fallback_projects_draws_onto_the_field(self):
        assert marginal(Pair("p"), "a").event_spec == OutputSpec(a=REAL)

    def test_a_factored_joint_marginalizes_onto_its_prior(self):
        joint = Kernel("y", ("beta",)) * Gaussian("beta")
        assert marginal(joint, "beta").event_spec == Gaussian("beta").event_spec

    def test_a_marginal_that_is_one_factor_takes_its_label(self):
        joint = (Kernel("y", ("beta",)) * Gaussian("beta").with_label("prior")).with_label("model")
        assert marginal(joint, "beta").label == "prior"

    def test_a_marginal_that_is_several_factors_joins_their_labels(self):
        joint = Gaussian("a").with_label("first") * Gaussian("b").with_label("second")
        joint = (joint * Gaussian("c")).with_label("model")
        assert marginal._derived_label({"d": joint, "field": ("a", "b")}) == "first·second"

    @pytest.mark.parametrize(
        "field",
        [pytest.param("y", id="a-consumer"), pytest.param(("y",), id="a-tuple-of-one")],
    )
    def test_a_marginal_that_integrates_a_factor_out_keeps_the_joint_label(self, field):
        """The prior predictive integrates the prior out, so it describes the whole model."""
        joint = (Kernel("y", ("beta",)) * Gaussian("beta")).with_label("model")
        assert marginal._derived_label({"d": joint, "field": field}) == "model"

    def test_the_marginal_of_a_law_without_factors_keeps_its_label(self):
        assert marginal(Marginalizing("law"), "a").label == "law"


class TestTheMarginalsNotation:
    """A marginal reads by its label and its own components, and keeps the law's fixed paths."""

    def test_the_marginal_keeps_the_paths_the_law_holds_fixed(self):
        law = Marginalizing("law")
        law._store_expression(law._expression.with_fixed(("y",)))
        result = marginal(law, "a")
        assert _fixed_paths(result) == ("y",)
        assert result.notation == "law(a; y)"

    def test_the_marginal_of_a_law_with_no_fixed_paths_holds_none(self):
        assert _fixed_paths(marginal(Marginalizing("law"), "a")) == ()

    def test_a_marginal_over_several_factors_of_an_unlabeled_joint_reads_factor_by_factor(self):
        joint = Gaussian("a").with_label("first") * Gaussian("b").with_label("second")
        result = marginal(joint * Gaussian("c"), ("a", "b"))
        assert result.label == "first·second"
        assert result.notation == "first(a)·second(b)"

    def test_a_marginal_over_several_factors_of_a_labeled_joint_reads_factor_by_factor(self):
        joint = (Gaussian("a") * Gaussian("b") * Gaussian("c")).with_label("model")
        assert marginal(joint, ("a", "b")).notation == "a(a)·b(b)"

    def test_factors_selected_out_of_factor_order_read_in_the_selection_order(self):
        joint = (Gaussian("a") * Gaussian("b") * Gaussian("c")).with_label("model")
        result = marginal(joint, ("b", "a"))
        assert (result.label, result.notation) == ("b·a", "b(b)·a(a)")
        assert list(result.event_spec.components) == ["b", "a"]

    def test_a_dependent_pair_selected_producer_first_reads_by_the_joined_label(self):
        """No product lists ``beta`` before the kernel that conditions on it."""
        joint = (Kernel("y", ("beta",)) * Gaussian("beta")).with_label("model")
        result = marginal(joint, ("beta", "y"))
        assert list(result.event_spec.components) == ["beta", "y"]
        assert result.notation == "(y·beta)(beta, y)"

    def test_a_marginal_of_a_conditioned_joint_keeps_its_fixed_paths(self):
        joint = (Gaussian("a") * Gaussian("b") * Gaussian("c")).with_label("model")
        conditioned = condition_on(joint, {"c": 0.5})
        assert marginal(conditioned, "a").notation == "a(a; c)"
        assert marginal(conditioned, ("a", "b")).notation == "(a·b)(a, b; c)"


class TestOptionalSlots:
    """A factor whose optional slots no factor produces is closed, at its defaults."""

    def test_the_marginal_of_such_a_factor_is_its_law_at_the_defaults(self):
        from probpipe import Normal, conditional_distribution

        kernel = conditional_distribution(lambda scale=2.0: Normal("y", 0.0, scale), label="lik")
        law = marginal(kernel * Normal("mu", 0.0, 1.0), "y")
        assert isinstance(law, Distribution)
        assert law.label == "lik"
        assert float(law._variance()) == pytest.approx(4.0)

    def test_a_renamed_joint_keeps_the_marginal_at_the_defaults(self):
        from probpipe import Normal, conditional_distribution

        kernel = conditional_distribution(lambda scale=2.0: Normal("y", 0.0, scale), label="lik")
        joint = (kernel * Normal("mu", 0.0, 1.0)).with_path_names({"y": "obs"})
        law = marginal(joint, "obs")
        assert isinstance(law, Distribution)
        assert list(law.event_spec.components) == ["obs"]

    def test_a_factor_whose_optional_slot_is_produced_is_not_closed(self):
        from probpipe import HalfNormal, Normal, conditional_distribution

        kernel = conditional_distribution(lambda scale=2.0: Normal("y", 0.0, scale), label="lik")
        with pytest.raises(ResolutionError, match="integrates out the fields \\['scale'\\]"):
            marginal(kernel * HalfNormal("scale", 1.0), "y")


def _difference(fn):
    return Function(
        fn,
        dispatch="sequential",
        n_broadcast_samples=8,
        label="difference",
        output_spec=OutputSpec(difference=None),
    )


class TestDetachment:
    """A lift draws a marginal or a factor independently of its joint, and a view together with it."""

    def test_the_marginal_of_a_renamed_joint_draws_independently_of_its_factor(self):
        """The marginal is the factor renamed, and the factor is the law the user composed."""
        prior = Normal("a", 0.0, 1.0)
        detached = marginal((prior * Normal("b", 2.0, 1.0)).with_path_names(a="c"), "c")

        assert _descendants.capture_stochastic_consumer(detached).root is detached
        with workflow_run(seed=41):
            result = _difference(lambda x, y: x - y)(prior, detached)
        assert np.any(np.asarray(result.atoms) != 0.0)

    def test_the_marginal_at_a_factor_that_is_a_batch_element_is_its_own_root(self):
        batch = DistributionBatch(
            [Normal("a", 0.0, 1.0), Normal("a", 1.0, 1.0)],
            "law",
            label="laws",
        )
        detached = marginal(batch[0] * Normal("b", 2.0, 1.0), "a")

        assert _descendants.capture_stochastic_consumer(detached).root is detached

    def test_a_view_of_a_renamed_joint_draws_with_the_joint(self):
        renamed = (Normal("a", 0.0, 1.0) * Normal("b", 2.0, 1.0)).with_path_names(a="c")

        with workflow_run(seed=43):
            result = _difference(lambda x, y: x["c"] - y)(renamed, renamed["c"])

        np.testing.assert_array_equal(np.asarray(result.atoms), np.zeros(8))

    def test_the_factor_of_a_renamed_joint_draws_independently_of_the_original(self):
        """The factor is the original renamed, and the original is the law the user composed."""
        prior = Normal("a", 0.0, 1.0)
        detached = factor((prior * Normal("b", 2.0, 1.0)).with_path_names(a="c"), "c")

        assert _descendants.capture_stochastic_consumer(detached).root is detached
        with workflow_run(seed=45):
            result = _difference(lambda x, y: x - y)(prior, detached)
        assert np.any(np.asarray(result.atoms) != 0.0)

    def test_the_factor_of_a_joint_is_its_own_root(self):
        detached = factor(Normal("a", 0.0, 1.0) * Normal("b", 2.0, 1.0), "a")

        assert _descendants.capture_stochastic_consumer(detached).root is detached

    def test_the_factor_that_is_a_batch_element_is_its_own_root(self):
        batch = DistributionBatch(
            [Normal("a", 0.0, 1.0), Normal("a", 1.0, 1.0)],
            "law",
            label="laws",
        )
        detached = factor(batch[0] * Normal("b", 2.0, 1.0), "a")

        assert _descendants.capture_stochastic_consumer(detached).root is detached


class TestFactor:
    def test_the_factor_carries_its_own_declaration(self):
        joint = Kernel("y", ("beta",)) * Gaussian("beta", 2.0)
        assert factor.check(joint, "beta").result is None
        assert factor(joint, "beta").event_spec == Gaussian("beta").event_spec

    def test_factor_returns_the_factor_producing_the_component(self):
        joint = Kernel("y", ("beta",)) * Gaussian("beta", 2.0)
        prior = factor(joint, "beta")
        assert isinstance(prior, Gaussian) and prior.loc == 2.0
        assert isinstance(factor(joint, "y"), ConditionalDistribution)

    def test_the_factor_keeps_its_own_label(self):
        joint = (Kernel("likelihood", ("beta",), component="y") * Gaussian("beta")).with_label("m")
        assert factor(joint, "y").label == "likelihood"
        assert factor(joint, "beta").label == "beta"

    def test_a_conditional_joint_exposes_its_factors_too(self):
        joint = Kernel("y", ("beta",)) * Kernel("beta", ("alpha",))
        assert isinstance(factor(joint, "beta"), ConditionalDistribution)

    def test_a_name_that_is_not_a_component_raises_applicability_error(self):
        joint = Kernel("y", ("beta",)) * Gaussian("beta")
        with pytest.raises(ApplicabilityError, match="output component"):
            factor(joint, "gamma")

    def test_a_law_without_factors_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="does not implement SupportsFactors"):
            factor(Gaussian("g"), "g")
