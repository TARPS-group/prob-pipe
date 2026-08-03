"""Tests for Function broadcast planning."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    DistributionArray,
    EmpiricalDistribution,
    Normal,
    NumericRecord,
    NumericRecordArray,
    ProductDistribution,
)
from probpipe.core import _workflow_call
from probpipe.core._workflow_distribution_normalization import (
    normalize_distribution_values,
)
from probpipe.core._workflow_plan import (
    ArrayBroadcastGroup,
    LogicalUnit,
    PlannedRandomEvent,
    StochasticConsumerPlan,
    StochasticPlan,
    StochasticSourceGroup,
    build_broadcast_plan,
    build_stochastic_plan,
)
from probpipe.core.distribution import Distribution
from probpipe.core.protocols import SupportsSampling


def _numeric_record_array(field: str, values: range) -> NumericRecordArray:
    return NumericRecordArray.stack(
        [NumericRecord("nr", **{field: float(value)}) for value in values]
    )


def _ref(name: str) -> _workflow_call.WorkflowInputRef:
    return _workflow_call.WorkflowInputRef(name)


def _plan(values, hints=None):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _workflow_call.make_signature_info_from_signature(
        signature,
        hints=hints,
    )
    return build_broadcast_plan(values=values, signature_info=signature_info)


def _stochastic_plan(values, n_broadcast_samples=16):
    return build_stochastic_plan(
        values,
        _plan(values),
        n_broadcast_samples,
    )


class TestBroadcastRegime:
    def test_plain_values_do_not_broadcast(self):
        plan = _plan({"x": 1.0})

        assert plan.regime == "none"
        assert plan.dist_args == ()
        assert plan.array_args == ()
        assert plan.array_groups == ()
        assert plan.sweep_batch_shape == ()
        assert plan.n_sweep == 1

    def test_distribution_value_selects_distribution_regime(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        plan = _plan({"x": dist})

        assert plan.regime == "distribution"
        assert plan.dist_args == (_ref("x"),)
        assert plan.array_args == ()

    def test_record_array_value_selects_sweep_regime(self):
        values = {"p": _numeric_record_array("x", range(4))}

        plan = _plan(values)

        assert plan.regime == "sweep"
        assert plan.dist_args == ()
        assert plan.array_args == (_ref("p"),)
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_refs=(_ref("p"),), batch_shape=(4,), size=4),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_array_and_distribution_values_select_nested_regime(self):
        values = {
            "p": _numeric_record_array("x", range(4)),
            "noise": Normal(loc=0.0, scale=1.0, name="noise"),
        }

        plan = _plan(values)

        assert plan.regime == "nested"
        assert plan.dist_args == (_ref("noise"),)
        assert plan.array_args == (_ref("p"),)


class TestHintClassification:
    def test_distribution_hints_skip_scalar_distribution_broadcast(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        concrete = _plan({"x": dist}, {"x": Distribution})
        protocol = _plan({"x": dist}, {"x": SupportsSampling})

        assert concrete.regime == "none"
        assert protocol.regime == "none"

    def test_array_hints_skip_array_sweep(self):
        ra = _numeric_record_array("x", range(4))
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2,),
            loc=jnp.asarray([0.0, 1.0]),
            scale=1.0,
            name="d",
        )

        record_plan = _plan({"p": ra}, {"p": NumericRecordArray})
        dist_plan = _plan({"d": da}, {"d": DistributionArray})
        any_plan = _plan({"p": ra}, {"p": Any})

        assert record_plan.regime == "none"
        assert dist_plan.regime == "none"
        assert any_plan.regime == "none"


class TestArrayGrouping:
    def test_sibling_views_zip_into_one_group(self):
        ra = NumericRecordArray.stack(
            [NumericRecord("nr", x=float(i), y=float(2 * i)) for i in range(4)]
        )

        plan = _plan({"x": ra.view("x"), "y": ra.view("y")})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("x"), _ref("y"))
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_refs=(_ref("x"), _ref("y")), batch_shape=(4,), size=4),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_views_from_different_parents_use_product_shape(self):
        ra_a = _numeric_record_array("a", range(3))
        ra_b = _numeric_record_array("b", range(2))

        plan = _plan({"a": ra_a.view("a"), "b": ra_b.view("b")})

        assert plan.regime == "sweep"
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_refs=(_ref("a"),), batch_shape=(3,), size=3),
            ArrayBroadcastGroup(arg_refs=(_ref("b"),), batch_shape=(2,), size=2),
        )
        assert plan.sweep_batch_shape == (3, 2)
        assert plan.n_sweep == 6

    def test_distribution_array_uses_sweep_group(self):
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2, 3),
            loc=jnp.arange(6.0).reshape(2, 3),
            scale=1.0,
            name="d",
        )

        plan = _plan({"d": da})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("d"),)
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_refs=(_ref("d"),), batch_shape=(2, 3), size=6),
        )
        assert plan.sweep_batch_shape == (2, 3)
        assert plan.n_sweep == 6


class TestPlanPurity:
    def test_planner_does_not_convert_or_mutate_external_distributions(self):
        external = tfd.Normal(loc=0.0, scale=1.0)
        values = {"x": external}

        signature = inspect.Signature(
            [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
        signature_info = _workflow_call.make_signature_info_from_signature(signature)
        raw_plan = build_broadcast_plan(
            values=values,
            signature_info=signature_info,
        )
        normalized = normalize_distribution_values(
            values=values,
            signature_info=signature_info,
        )
        normalized_plan = build_broadcast_plan(
            values=normalized,
            signature_info=signature_info,
        )

        assert values["x"] is external
        assert raw_plan.regime == "none"
        assert normalized_plan.regime == "distribution"


class TestStochasticPlanStructure:
    def test_none_and_pure_sweep_have_no_stochastic_plan(self):
        assert _stochastic_plan({"x": 1.0}) is None
        assert _stochastic_plan({"rows": _numeric_record_array("x", range(2))}) is None

    def test_sampled_singleton_plan_is_frozen_and_tuple_only(self):
        plan = _stochastic_plan(
            {
                "a": Normal(loc=0.0, scale=1.0, name="a"),
                "b": Normal(loc=1.0, scale=1.0, name="b"),
            },
            n_broadcast_samples=12,
        )

        assert isinstance(plan, StochasticPlan)
        assert plan.evaluation_mode == "sampled"
        assert plan.arg_refs == (_ref("a"), _ref("b"))
        assert plan.source_groups == (
            StochasticSourceGroup(
                0,
                (StochasticConsumerPlan(_ref("a"), (), None),),
                "sampled",
                None,
            ),
            StochasticSourceGroup(
                1,
                (StochasticConsumerPlan(_ref("b"), (), None),),
                "sampled",
                None,
            ),
        )
        assert plan.logical_units == (LogicalUnit("singleton", 0, ()),)
        assert plan.n_broadcast_samples == 12
        assert plan.sample_shape == (12,)
        assert plan.exact_group_order == ()
        assert plan.exact_combination_order == ((),)
        assert plan.repetitions_per_combination == 12
        assert plan.n_evaluations == 12
        assert plan.random_events == (
            PlannedRandomEvent(("source-group", 0), ("singleton",)),
            PlannedRandomEvent(("source-group", 1), ("singleton",)),
        )
        assert isinstance(plan.arg_refs, tuple)
        assert isinstance(plan.source_groups, tuple)
        assert isinstance(plan.runtime_bindings, tuple)
        assert isinstance(plan.logical_units, tuple)
        assert isinstance(plan.exact_group_order, tuple)
        assert isinstance(plan.exact_combination_order, tuple)
        assert all(isinstance(combo, tuple) for combo in plan.exact_combination_order)
        assert all(isinstance(group.arg_refs, tuple) for group in plan.source_groups)
        assert all(isinstance(group.consumers, tuple) for group in plan.source_groups)
        assert all(
            isinstance(binding.consumer_evaluators, tuple) for binding in plan.runtime_bindings
        )
        assert all(isinstance(unit.coordinates, tuple) for unit in plan.logical_units)
        assert isinstance(plan.random_events, tuple)

        with pytest.raises(FrozenInstanceError):
            plan.n_evaluations = 99
        with pytest.raises(FrozenInstanceError):
            plan.source_groups[0].index = 99
        with pytest.raises(FrozenInstanceError):
            plan.logical_units[0].flat_index = 99

    def test_nested_plan_uses_row_major_canonical_sweep_units(self):
        values = {
            "left": _numeric_record_array("x", range(2)),
            "right": _numeric_record_array("y", range(3)),
            "noise": Normal(loc=0.0, scale=1.0, name="noise"),
        }

        plan = _stochastic_plan(values, n_broadcast_samples=7)

        assert plan.logical_units == tuple(
            LogicalUnit("canonical_sweep", flat_index, coordinates)
            for flat_index, coordinates in enumerate(
                ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
            )
        )
        assert tuple(unit.logical_unit_id for unit in plan.logical_units) == (
            ("cell", 0, 0),
            ("cell", 0, 1),
            ("cell", 0, 2),
            ("cell", 1, 0),
            ("cell", 1, 1),
            ("cell", 1, 2),
        )
        assert len(plan.random_events) == 6


class TestStochasticSourceGrouping:
    def test_record_views_share_their_known_parent_group(self):
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=1.0, scale=1.0, name="y"),
        )

        plan = _stochastic_plan(
            {
                "x": joint["x"],
                "independent": Normal(loc=2.0, scale=1.0, name="independent"),
                "y": joint["y"],
            }
        )

        assert plan.source_groups == (
            StochasticSourceGroup(
                0,
                (
                    StochasticConsumerPlan(_ref("x"), ("x",), None),
                    StochasticConsumerPlan(_ref("y"), ("y",), None),
                ),
                "sampled",
                None,
            ),
            StochasticSourceGroup(
                1,
                (StochasticConsumerPlan(_ref("independent"), (), None),),
                "sampled",
                None,
            ),
        )

    def test_direct_aliases_share_one_root_group(self):
        shared = Normal(loc=0.0, scale=1.0, name="shared")

        plan = _stochastic_plan({"first": shared, "second": shared})

        assert tuple(group.arg_refs for group in plan.source_groups) == (
            (_ref("first"), _ref("second")),
        )
        assert tuple(group.stochastic_source_id for group in plan.source_groups) == (
            ("source-group", 0),
        )

    def test_equal_but_distinct_direct_objects_remain_independent(self):
        first = Normal(loc=0.0, scale=1.0, name="same")
        second = Normal(loc=0.0, scale=1.0, name="same")

        plan = _stochastic_plan({"first": first, "second": second})

        assert tuple(group.arg_refs for group in plan.source_groups) == (
            (_ref("first"),),
            (_ref("second"),),
        )
        assert plan.runtime_bindings[0].root is first
        assert plan.runtime_bindings[1].root is second

    def test_root_sibling_and_transitive_views_share_canonical_paths(self):
        joint = ProductDistribution(
            nested={
                "left": Normal(loc=0.0, scale=1.0, name="left"),
                "right": Normal(loc=1.0, scale=1.0, name="right"),
            },
            top=Normal(loc=2.0, scale=1.0, name="top"),
        )

        plan = _stochastic_plan(
            {
                "left": joint["nested"]["left"],
                "root": joint,
                "right": joint["nested"]["right"],
                "top": joint["top"],
            }
        )

        assert len(plan.source_groups) == 1
        assert plan.source_groups[0].consumers == (
            StochasticConsumerPlan(_ref("left"), ("nested", "left"), None),
            StochasticConsumerPlan(_ref("root"), (), None),
            StochasticConsumerPlan(_ref("right"), ("nested", "right"), None),
            StochasticConsumerPlan(_ref("top"), ("top",), None),
        )
        assert plan.runtime_bindings[0].root is joint

    def test_runtime_bindings_do_not_participate_in_canonical_plan_equality(self):
        first = _stochastic_plan({"x": Normal(loc=0.0, scale=1.0, name="first")})
        second = _stochastic_plan({"x": Normal(loc=9.0, scale=3.0, name="second")})

        assert first == second
        assert first.runtime_bindings[0].root is not second.runtime_bindings[0].root


class TestStochasticEvaluationPlanning:
    def test_wholly_exact_empiricals_record_cartesian_order_and_no_events(self):
        values = {
            "a": EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="a"),
            "b": EmpiricalDistribution(jnp.asarray([10.0, 20.0, 30.0]), name="b"),
        }

        plan = _stochastic_plan(values, n_broadcast_samples=10)

        assert plan.evaluation_mode == "exact"
        assert tuple(group.execution_mode for group in plan.source_groups) == (
            "exact",
            "exact",
        )
        assert tuple(group.exact_size for group in plan.source_groups) == (2, 3)
        assert plan.exact_group_order == (0, 1)
        assert plan.exact_combination_order == (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        )
        assert plan.repetitions_per_combination == 1
        assert plan.n_evaluations == 6
        assert plan.sample_shape is None
        assert plan.random_events == ()

    def test_mixed_plan_preserves_stable_greedy_order_and_actual_sample_shape(self):
        values = {
            "large_first": EmpiricalDistribution(jnp.arange(5.0), name="large_first"),
            "small_second": EmpiricalDistribution(jnp.arange(2.0), name="small_second"),
            "sampled": Normal(loc=0.0, scale=1.0, name="sampled"),
        }

        plan = _stochastic_plan(values, n_broadcast_samples=23)

        assert plan.evaluation_mode == "mixed_exact_sampled"
        assert plan.exact_group_order == (1, 0)
        assert plan.exact_combination_order[:6] == (
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
        )
        assert plan.repetitions_per_combination == 2
        assert plan.n_evaluations == 20
        assert plan.sample_shape == (20,)
        assert plan.random_events == (PlannedRandomEvent(("source-group", 2), ("singleton",)),)

    def test_over_budget_empirical_is_sampled(self):
        plan = _stochastic_plan(
            {"x": EmpiricalDistribution(jnp.arange(20.0), name="x")},
            n_broadcast_samples=5,
        )

        assert plan.evaluation_mode == "sampled"
        assert plan.source_groups[0].execution_mode == "sampled"
        assert plan.source_groups[0].exact_size is None
        assert plan.sample_shape == (5,)
        assert len(plan.random_events) == 1

    def test_equal_size_empiricals_keep_first_consumer_order_at_greedy_cutoff(self):
        plan = _stochastic_plan(
            {
                "first": EmpiricalDistribution(jnp.arange(3.0), name="first"),
                "second": EmpiricalDistribution(jnp.arange(3.0), name="second"),
                "third": EmpiricalDistribution(jnp.arange(3.0), name="third"),
            },
            n_broadcast_samples=10,
        )

        assert plan.exact_group_order == (0, 1)
        assert tuple(group.execution_mode for group in plan.source_groups) == (
            "exact",
            "exact",
            "sampled",
        )
        assert plan.n_evaluations == 9
        assert plan.sample_shape == (9,)
        assert plan.random_events == (PlannedRandomEvent(("source-group", 2), ("singleton",)),)

    def test_event_count_is_sampled_sources_times_units_not_draw_count(self):
        values = {
            "rows": _numeric_record_array("x", range(3)),
            "a": Normal(loc=0.0, scale=1.0, name="a"),
            "b": Normal(loc=1.0, scale=1.0, name="b"),
        }

        small = _stochastic_plan(values, n_broadcast_samples=5)
        large = _stochastic_plan(values, n_broadcast_samples=500)

        assert len(small.random_events) == 2 * 3
        assert small.random_events == large.random_events
        assert small.sample_shape == (5,)
        assert large.sample_shape == (500,)

    @pytest.mark.parametrize(
        ("n_broadcast_samples", "error", "message"),
        [
            (True, TypeError, "n_broadcast_samples must be an integer"),
            (False, TypeError, "n_broadcast_samples must be an integer"),
            (2.5, TypeError, "n_broadcast_samples must be an integer"),
            (0, ValueError, "n_broadcast_samples must be a positive integer"),
            (-1, ValueError, "n_broadcast_samples must be a positive integer"),
        ],
    )
    def test_invalid_sample_counts_fail_during_planning(
        self,
        n_broadcast_samples,
        error,
        message,
    ):
        values = {"x": Normal(loc=0.0, scale=1.0, name="x")}

        with pytest.raises(error, match=message):
            _stochastic_plan(values, n_broadcast_samples=n_broadcast_samples)


class TestStochasticPlanPurity:
    def test_planner_does_not_read_entropy_claim_keys_or_mutate_inputs(self):
        source = Normal(loc=0.0, scale=1.0, name="x")
        values = {"x": source}

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation"
            ) as commit_invocation,
        ):
            plan = _stochastic_plan(values)

        assert plan.random_events == (PlannedRandomEvent(("source-group", 0), ("singleton",)),)
        assert values == {"x": source}
        assert values["x"] is source
        urandom.assert_not_called()
        commit_invocation.assert_not_called()
