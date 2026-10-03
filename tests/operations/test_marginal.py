"""Contract tests of marginal and factor: the detached parts of structured and factored laws."""

from __future__ import annotations

import pytest

from probpipe import ApplicabilityError, RecordSpec
from probpipe.core._dispatch import ResolutionError
from probpipe.core._specs import OutputSpec
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._distribution import Distribution, DistributionSpec
from probpipe.operations._marginal import factor, marginal

from ._laws import REAL, Gaussian, Kernel, Marginalizing, Pair


class _Nested(Distribution):
    """A law with two groups that each hold a field named ``a``."""

    def __init__(self, label: str) -> None:
        super().__init__(label, RecordSpec(x=RecordSpec(a=REAL), y=RecordSpec(a=REAL)))


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
        with pytest.raises(ApplicabilityError, match="same segment"):
            marginal.check(_Nested("n"), ("x/a", "y/a"))

    def test_a_path_the_law_lacks_raises_applicability_error(self):
        with pytest.raises(ApplicabilityError, match="not an event path"):
            marginal(Marginalizing("law"), "c")

    def test_a_rejecting_guard_and_no_sampling_raise_resolution_error(self):
        with pytest.raises(ResolutionError, match="The marginal is exact at the field a"):
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
        with pytest.raises(ResolutionError, match="does not claim SupportsFactors"):
            factor(Gaussian("g"), "g")
