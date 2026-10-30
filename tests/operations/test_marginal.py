"""Contract tests of marginal and factor: the detached parts of structured and factored laws."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    OpaqueSpec,
    RecordSpec,
    condition_on,
    conditional_distribution,
    replay_run,
    sample,
    workflow_run,
)
from probpipe.core._batch import BatchSpec
from probpipe.core._dispatch import Feasibility, ResolutionError
from probpipe.core._specs import OutputSpec
from probpipe.distributions._capabilities import SupportsSampling
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


class _JointSampler(Distribution, SupportsSampling):
    """A scalar, twice that scalar in a group, and an opaque field, with recorded draws."""

    def __init__(self, label="joint"):
        super().__init__(RecordSpec(a=REAL, g=RecordSpec(a=REAL), tag=OpaqueSpec(str)), label=label)
        self.sample_shapes = []
        self.draws = []

    def _sample(self, key, sample_shape=()):
        self.sample_shapes.append(tuple(sample_shape))
        values = jax.random.normal(key, sample_shape, dtype=jnp.float32)
        draws = {
            "a": values,
            "g": {"a": 2 * values},
            "tag": np.full(sample_shape, "tag", dtype=object),
        }
        self.draws.append(draws)
        return draws


class _BatchSampler(Distribution, SupportsSampling):
    """A record holding a batch, returned as one batch per draw."""

    def __init__(self, levels=("time",)):
        super().__init__(
            RecordSpec(trajectory=BatchSpec(REAL, dict.fromkeys(levels, 2))), label="trajectories"
        )
        self.levels = levels

    def _sample(self, key, sample_shape=()):
        values = jax.random.normal(
            key, (*sample_shape, *((2,) * len(self.levels))), dtype=jnp.float32
        )
        batches = np.empty(sample_shape, dtype=object)
        for index in np.ndindex(sample_shape):
            batches[index] = NumericArrayBatch(values[index], self.levels, label="trajectory")
        return {"trajectory": batches}


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
        assert dict(result.term_spec.event_spec.outputs) == {"a": REAL}
        # A law as a value has no outputs of its own.
        assert dict(result.outputs) == {}

    def test_several_paths_declare_an_exposed_record_of_the_nodes(self):
        result = marginal.check(Marginalizing("law"), ("a", "b")).result
        assert result == OutputSpec(DistributionSpec(OutputSpec(RecordSpec(a=REAL, b=REAL))))

    def test_two_paths_ending_in_the_same_name_keep_their_paths(self):
        result = marginal.check(_Nested("n"), ["y/a", "x/a"]).result
        assert result == OutputSpec(
            DistributionSpec(OutputSpec(RecordSpec(x=RecordSpec(a=REAL), y=RecordSpec(a=REAL))))
        )

    def test_a_list_and_a_tuple_of_paths_declare_alike(self):
        law = Marginalizing("law")
        assert marginal.check(law, ["b", "a"]).result == marginal.check(law, ("a", "b")).result

    def test_a_path_the_law_lacks_raises_applicability_error(self):
        with pytest.raises(
            ApplicabilityError, match="unknown path 'c' in 'law'; available paths: \\['a', 'b'\\]"
        ):
            marginal(Marginalizing("law"), "c")

    def test_a_path_that_is_not_a_string_raises_type_error(self):
        with pytest.raises(TypeError, match="it must be a path or a list of paths, got int"):
            marginal(Marginalizing("law"), 3)

    @pytest.mark.parametrize(
        ("paths", "match"),
        [
            pytest.param([], "since it names no paths", id="empty"),
            pytest.param(("a", "a"), "cannot select 'a' twice", id="repeated"),
            pytest.param(("x", "x/a"), "'x/a' is inside 'x'. Keep one of them", id="nested"),
            pytest.param("x//a", "has an empty name", id="empty-name"),
        ],
    )
    def test_malformed_paths_raise_value_error(self, paths, match):
        with pytest.raises(ValueError, match=match):
            marginal.check(_Nested("n"), paths)

    def test_a_rejecting_guard_and_no_sampling_raise_resolution_error(self):
        with pytest.raises(ResolutionError, match="the marginal is exact at the field a"):
            marginal(Marginalizing("law"), "b")

    def test_check_and_the_call_select_the_fallback_when_no_exact_marginal_exists(self):
        report = marginal.check(Pair("p"), "a")
        assert report.feasible is True
        assert (report.route, report.exact) == ("monte_carlo", False)
        with workflow_run(seed=0):
            result = marginal(Pair("p"), "a")
        assert result.provenance.metadata["route"] == report.route
        assert result.provenance.metadata["exact"] is False

    def test_the_fallback_projects_draws_onto_the_field(self):
        assert marginal(Pair("p"), "a").event_spec == OutputSpec(a=REAL)

    def test_a_factored_joint_marginalizes_onto_its_prior(self):
        joint = Kernel("y", ("beta",)) & Gaussian("beta")
        assert marginal(joint, "beta").event_spec == Gaussian("beta").event_spec

    def test_a_marginal_that_is_one_factor_takes_its_label(self):
        joint = (Kernel("y", ("beta",)) & Gaussian("beta").with_label("prior")).with_label("model")
        assert marginal(joint, "beta").label == "prior"

    def test_a_marginal_that_is_several_factors_joins_their_labels(self):
        joint = Gaussian("a").with_label("first") & Gaussian("b").with_label("second")
        joint = (joint & Gaussian("c")).with_label("model")
        assert marginal._derived_label({"d": joint, "paths": ("a", "b")}) == "first·second"

    @pytest.mark.parametrize(
        "paths",
        [pytest.param("y", id="a-consumer"), pytest.param(("y",), id="a-tuple-of-one")],
    )
    def test_a_marginal_that_integrates_a_factor_out_keeps_the_joint_label(self, paths):
        """The prior predictive integrates the prior out, so it describes the whole model."""
        joint = (Kernel("y", ("beta",)) & Gaussian("beta")).with_label("model")
        assert marginal._derived_label({"d": joint, "paths": paths}) == "model"

    def test_the_marginal_of_a_law_without_factors_keeps_its_label(self):
        assert marginal(Marginalizing("law"), "a").label == "law"


class TestMonteCarloMarginal:
    @pytest.mark.parametrize("count", [1, 13])
    @pytest.mark.parametrize("paths", ["a", "g/a", "g", ["g"], ["g/a", "a"], "tag", ["tag"]])
    def test_the_atoms_have_the_planned_event_and_keep_the_projected_values(self, paths, count):
        law = _JointSampler()
        configured = marginal.using(num_samples=count)
        planned = configured.check(law, paths).result.term_spec.event_spec
        assert law.sample_shapes == []
        with workflow_run(seed=5):
            result = configured(law, paths)
        assert isinstance(result, EmpiricalDistribution)
        assert result.event_spec == planned
        assert result.num_atoms == count
        assert law.sample_shapes == [(count,)]
        np.testing.assert_allclose(result.weights, np.full(count, 1 / count), rtol=1e-7, atol=0)
        if paths in ("a", "g/a"):
            expected = law.draws[-1]["a"] if paths == "a" else law.draws[-1]["g"]["a"]
            np.testing.assert_array_equal(result.atoms.raw, expected)
        elif paths == "g":
            np.testing.assert_array_equal(result.atoms.raw["a"], law.draws[-1]["g"]["a"])
        elif paths == ["g"]:
            np.testing.assert_array_equal(result.atoms.raw["g"]["a"], law.draws[-1]["g"]["a"])
        elif paths == ["g/a", "a"]:
            assert list(result.event_spec.outputs) == ["a", "g"]
            np.testing.assert_array_equal(result.atoms.raw["a"], law.draws[-1]["a"])
            np.testing.assert_array_equal(result.atoms.raw["g"]["a"], 2 * result.atoms.raw["a"])
        else:
            tags = result.atoms.raw if paths == "tag" else result.atoms.raw["tag"]
            np.testing.assert_array_equal(tags, np.full(count, "tag", dtype=object))

    def test_sampling_the_result_reuses_its_atoms_without_sampling_the_source(self):
        law = _JointSampler("a law with spaces")
        with workflow_run(seed=8):
            result = marginal.using(num_samples=13)(law, ["a", "g/a"])
            resampled = sample(result, sample_shape=100)
        assert law.sample_shapes == [(13,)]
        assert _descendants.capture_stochastic_consumer(result).root is result
        assert result.label == law.label
        assert set(np.asarray(resampled.raw["a"])) <= set(np.asarray(result.atoms.raw["a"]))
        np.testing.assert_array_equal(resampled.raw["g"]["a"], 2 * resampled.raw["a"])
        assert result.provenance.metadata["num_samples"] == 13

    def test_seeded_workflows_reproduce_the_atoms(self):
        law = _JointSampler()
        with workflow_run(seed=3):
            first = marginal.using(num_samples=13)(law, "a")
        with workflow_run(seed=3):
            second = marginal.using(num_samples=13)(law, "a")
        np.testing.assert_array_equal(first.atoms.raw, second.atoms.raw)

    def test_replay_reproduces_the_atoms(self):
        with workflow_run(seed=3):
            original = marginal.using(num_samples=13)(_JointSampler(), "a")
        with replay_run(original.provenance):
            replayed = marginal.using(num_samples=13)(_JointSampler(), "a")
        np.testing.assert_array_equal(original.atoms.raw, replayed.atoms.raw)

    @pytest.mark.parametrize(
        "guard",
        [Feasibility(False, "sampling guard"), Feasibility(None, pending=("sampling guard",))],
    )
    def test_check_respects_the_sampling_guard_without_sampling(self, monkeypatch, guard):
        monkeypatch.setattr(
            _JointSampler,
            "_sample_guard",
            lambda self: guard,
            raising=False,
        )
        law = _JointSampler()
        report = marginal.check(law, "a")
        assert report.feasible is guard.feasible
        assert law.sample_shapes == []
        if guard.feasible is False:
            with pytest.raises(ResolutionError, match="sampling guard"):
                marginal(law, "a")

    def test_an_exact_marginal_is_preferred_and_the_fallback_can_be_selected(self):
        class ExactMarginalSampler(_JointSampler):
            def _marginal(self, paths):
                return Gaussian("a")

        law = ExactMarginalSampler()
        assert marginal.check(law, "a").route == "exact"
        assert not isinstance(marginal(law, "a"), EmpiricalDistribution)
        assert law.sample_shapes == []
        with workflow_run(seed=2):
            result = marginal.using(method="monte_carlo", num_samples=7)(law, "a")
        assert isinstance(result, EmpiricalDistribution)
        assert law.sample_shapes == [(7,)]

    def test_exact_only_rejects_the_fallback_without_sampling(self):
        law = _JointSampler()
        with pytest.raises(ResolutionError, match="exact_only"):
            marginal.using(exact_only=True)(law, "a")
        assert law.sample_shapes == []

    @pytest.mark.parametrize(
        ("levels", "atom_level"),
        [(("time",), "sample"), (("sample",), "sample_"), (("sample", "sample_"), "sample__")],
    )
    def test_a_batch_valued_field_keeps_the_event_levels_separate_from_the_atom_level(
        self, levels, atom_level
    ):
        law = _BatchSampler(levels)
        with workflow_run(seed=9):
            result = marginal.using(num_samples=7)(law, "trajectory")
        assert result.event_spec == OutputSpec(trajectory=BatchSpec(REAL, dict.fromkeys(levels, 2)))
        assert result.num_atoms == 7
        assert result.atoms.level_names == (atom_level, *levels)
        assert result.atoms.batch_shape == (7, *((2,) * len(levels)))
        with workflow_run(seed=0):
            draw = sample(result)
        assert draw.level_names == levels
        assert draw.batch_shape == (2,) * len(levels)

    def test_given_uses_the_conditional_sampling_capability(self):
        kernel = conditional_distribution(
            lambda mu: Normal("a", mu, 1.0) & Normal("b", mu, 1.0),
            given_spec={"mu": NumericArraySpec((), jnp.float32)},
            label="kernel",
        )
        with workflow_run(seed=0):
            result = marginal.using(num_samples=7)(kernel, "a", given={"mu": 2.0})
            conditioned = marginal(condition_on(kernel, {"mu": 2.0}), "a")
            exact = marginal.using(exact_only=True)(kernel, "a", given={"mu": 2.0})
        assert isinstance(result, EmpiricalDistribution)
        assert result.num_atoms == 7
        assert result.event_spec == kernel.event_spec["a"]
        assert _fixed_paths(result) == ("mu",)
        assert result.provenance.metadata["route"] == "monte_carlo"
        assert result.provenance.metadata["exact"] is False
        for law in (conditioned, exact):
            assert isinstance(law, Normal)
            assert law.loc == 2.0
            assert law.scale == 1.0
            assert _fixed_paths(law) == ("mu",)

    def test_a_batch_of_laws_sweeps_the_fallback(self):
        laws = DistributionBatch(
            [_JointSampler("first"), _JointSampler("second")], "model", label="models"
        )
        with workflow_run(seed=0):
            result = marginal.using(num_samples=7)(laws, "a")
        assert isinstance(result, DistributionBatch)
        assert result.level_names == ("model",)
        assert all(law.num_atoms == 7 for law in result)


class TestTheMarginalsNotation:
    """A marginal reads by its label and its own outputs, and keeps the law's fixed paths."""

    def test_the_marginal_keeps_the_paths_the_law_holds_fixed(self):
        law = Marginalizing("law")
        law._store_expression(law._expression.with_fixed(("y",)))
        result = marginal(law, "a")
        assert _fixed_paths(result) == ("y",)
        assert result.notation == "law(a; y)"

    def test_the_marginal_of_a_law_with_no_fixed_paths_holds_none(self):
        assert _fixed_paths(marginal(Marginalizing("law"), "a")) == ()

    def test_a_marginal_over_several_factors_of_an_unlabeled_joint_reads_factor_by_factor(self):
        joint = Gaussian("a").with_label("first") & Gaussian("b").with_label("second")
        result = marginal((joint & Gaussian("c")), ("a", "b"))
        assert result.label == "first·second"
        assert result.notation == "first(a)·second(b)"

    def test_a_marginal_over_several_factors_of_a_labeled_joint_reads_factor_by_factor(self):
        joint = ((Gaussian("a") & Gaussian("b")) & Gaussian("c")).with_label("model")
        assert marginal(joint, ("a", "b")).notation == "a(a)·b(b)"

    def test_factors_named_out_of_factor_order_read_in_the_joint_order(self):
        joint = ((Gaussian("a") & Gaussian("b")) & Gaussian("c")).with_label("model")
        result = marginal(joint, ["b", "a"])
        assert (result.label, result.notation) == ("a·b", "a(a)·b(b)")
        assert list(result.event_spec.outputs) == ["a", "b"]

    def test_a_dependent_pair_named_producer_first_reads_in_the_joint_order(self):
        joint = (Kernel("y", ("beta",)) & Gaussian("beta")).with_label("model")
        result = marginal(joint, ("beta", "y"))
        assert list(result.event_spec.outputs) == ["y", "beta"]
        assert result.notation == "y(y | beta)·beta(beta)"

    def test_a_marginal_of_a_conditioned_joint_keeps_its_fixed_paths(self):
        joint = ((Gaussian("a") & Gaussian("b")) & Gaussian("c")).with_label("model")
        conditioned = condition_on(joint, {"c": 0.5})
        assert marginal(conditioned, "a").notation == "a(a; c)"
        assert marginal(conditioned, ("a", "b")).notation == "(a·b)(a, b; c)"


class TestOptionalInputs:
    """A factor whose optional inputs no factor produces is closed, at its defaults."""

    def test_the_marginal_of_such_a_factor_is_its_law_at_the_defaults(self):
        from probpipe import Normal, conditional_distribution

        kernel = conditional_distribution(lambda scale=2.0: Normal("y", 0.0, scale), label="lik")
        law = marginal((kernel & Normal("mu", 0.0, 1.0)), "y")
        assert isinstance(law, Distribution)
        assert law.label == "lik"
        assert float(law._variance()) == pytest.approx(4.0)

    def test_a_renamed_joint_keeps_the_marginal_at_the_defaults(self):
        from probpipe import Normal, conditional_distribution

        kernel = conditional_distribution(lambda scale=2.0: Normal("y", 0.0, scale), label="lik")
        joint = (kernel & Normal("mu", 0.0, 1.0)).rename({"y": "obs"})
        law = marginal(joint, "obs")
        assert isinstance(law, Distribution)
        assert list(law.event_spec.outputs) == ["obs"]

    def test_a_factor_whose_optional_input_is_produced_is_not_closed(self):
        from probpipe import HalfNormal, Normal, conditional_distribution

        kernel = conditional_distribution(lambda scale=2.0: Normal("y", 0.0, scale), label="lik")
        with pytest.raises(ResolutionError, match="integrates out the fields \\['scale'\\]"):
            marginal.using(exact_only=True)(kernel & HalfNormal("scale", 1.0), "y")
        with workflow_run(seed=0):
            result = marginal.using(num_samples=13)(kernel & HalfNormal("scale", 1.0), "y")
        assert isinstance(result, EmpiricalDistribution)
        assert result.provenance.metadata["route"] == "monte_carlo"


def _difference(fn):
    return Function(
        fn,
        executor="sequential",
        max_evaluations=8,
        label="difference",
        output_spec=OutputSpec(difference=None),
    )


class TestDetachment:
    """A lift draws a marginal or a factor independently of its joint, and a view together with it."""

    def test_the_marginal_of_a_renamed_joint_draws_independently_of_its_factor(self):
        """The marginal is the factor renamed, and the factor is the law the user composed."""
        prior = Normal("a", 0.0, 1.0)
        detached = marginal((prior & Normal("b", 2.0, 1.0)).rename(a="c"), "c")

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
        detached = marginal((batch[0] & Normal("b", 2.0, 1.0)), "a")

        assert _descendants.capture_stochastic_consumer(detached).root is detached

    def test_a_view_of_a_renamed_joint_draws_with_the_joint(self):
        renamed = (Normal("a", 0.0, 1.0) & Normal("b", 2.0, 1.0)).rename(a="c")

        with workflow_run(seed=43):
            result = _difference(lambda x, y: x["c"] - y)(renamed, renamed["c"])

        np.testing.assert_array_equal(np.asarray(result.atoms), np.zeros(8))

    def test_the_factor_of_a_renamed_joint_draws_independently_of_the_original(self):
        """The factor is the original renamed, and the original is the law the user composed."""
        prior = Normal("a", 0.0, 1.0)
        detached = factor((prior & Normal("b", 2.0, 1.0)).rename(a="c"), "c")

        assert _descendants.capture_stochastic_consumer(detached).root is detached
        with workflow_run(seed=45):
            result = _difference(lambda x, y: x - y)(prior, detached)
        assert np.any(np.asarray(result.atoms) != 0.0)

    def test_the_factor_of_a_joint_is_its_own_root(self):
        detached = factor((Normal("a", 0.0, 1.0) & Normal("b", 2.0, 1.0)), "a")

        assert _descendants.capture_stochastic_consumer(detached).root is detached

    def test_the_factor_that_is_a_batch_element_is_its_own_root(self):
        batch = DistributionBatch(
            [Normal("a", 0.0, 1.0), Normal("a", 1.0, 1.0)],
            "law",
            label="laws",
        )
        detached = factor((batch[0] & Normal("b", 2.0, 1.0)), "a")

        assert _descendants.capture_stochastic_consumer(detached).root is detached


class TestFactor:
    def test_the_factor_carries_its_own_declaration(self):
        joint = Kernel("y", ("beta",)) & Gaussian("beta", 2.0)
        assert factor.check(joint, "beta").result is None
        assert factor(joint, "beta").event_spec == Gaussian("beta").event_spec

    def test_factor_returns_the_factor_producing_the_output(self):
        joint = Kernel("y", ("beta",)) & Gaussian("beta", 2.0)
        prior = factor(joint, "beta")
        assert isinstance(prior, Gaussian) and prior.loc == 2.0
        assert isinstance(factor(joint, "y"), ConditionalDistribution)

    def test_the_factor_keeps_its_own_label(self):
        joint = (Kernel("likelihood", ("beta",), output_name="y") & Gaussian("beta")).with_label(
            "m"
        )
        assert factor(joint, "y").label == "likelihood"
        assert factor(joint, "beta").label == "beta"

    def test_a_conditional_joint_exposes_its_factors_too(self):
        joint = Kernel("y", ("beta",)) & Kernel("beta", ("alpha",))
        assert isinstance(factor(joint, "beta"), ConditionalDistribution)

    def test_a_name_that_is_not_an_output_raises_applicability_error(self):
        joint = Kernel("y", ("beta",)) & Gaussian("beta")
        with pytest.raises(ApplicabilityError, match="output"):
            factor(joint, "gamma")

    def test_a_law_without_factors_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="does not implement SupportsFactors"):
            factor(Gaussian("g"), "g")
