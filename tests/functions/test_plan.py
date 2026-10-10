"""Tests for Function broadcast planning."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    ApplicabilityError,
    Distribution,
    DistributionBatch,
    EmpiricalDistribution,
    Normal,
    NumericRecord,
    NumericRecordBatch,
)
from probpipe.distributions._capabilities import SupportsSampling
from probpipe.functions._normalization import (
    normalize_distribution_values,
)
from probpipe.functions._plan import (
    ArrayBroadcastGroup,
    LogicalUnit,
    PlannedRandomEvent,
    StochasticConsumerPlan,
    StochasticPlan,
    StochasticSourceGroup,
    build_broadcast_plan,
    build_stochastic_plan,
)
from probpipe.values import _binding


def _numeric_record_batch(
    field: str, values: range, *, level_name: str = "draw"
) -> NumericRecordBatch:
    return NumericRecordBatch.stack(
        [
            NumericRecord(
                {**{field: float(value)}},
                label="nr",
            )
            for value in values
        ],
        level_name=level_name,
    )


def _ref(name: str) -> _binding.FunctionInputRef:
    return _binding.FunctionInputRef(name)


def _plan(values, hints=None):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _binding.make_signature_info_from_signature(
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
        dist = Normal("x", loc=0.0, scale=1.0)

        plan = _plan({"x": dist})

        assert plan.regime == "distribution"
        assert plan.dist_args == (_ref("x"),)
        assert plan.array_args == ()

    def test_record_batch_value_selects_sweep_regime(self):
        values = {"p": _numeric_record_batch("x", range(4))}

        plan = _plan(values)

        assert plan.regime == "sweep"
        assert plan.dist_args == ()
        assert plan.array_args == (_ref("p"),)
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("p"),),
                batch_shape=(4,),
                size=4,
                level_names=("draw",),
                axis_groups=((4,),),
            ),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_array_and_distribution_values_select_nested_regime(self):
        values = {
            "p": _numeric_record_batch("x", range(4)),
            "noise": Normal("noise", loc=0.0, scale=1.0),
        }

        plan = _plan(values)

        assert plan.regime == "nested"
        assert plan.dist_args == (_ref("noise"),)
        assert plan.array_args == (_ref("p"),)


class TestHintClassification:
    def test_distribution_hints_skip_scalar_distribution_broadcast(self):
        dist = Normal("x", loc=0.0, scale=1.0)

        concrete = _plan({"x": dist}, {"x": Distribution})
        protocol = _plan({"x": dist}, {"x": SupportsSampling})

        assert concrete.regime == "none"
        assert protocol.regime == "none"

    def test_array_hints_skip_array_sweep(self):
        ra = _numeric_record_batch("x", range(4))
        laws = DistributionBatch(
            [Normal("x", 0.0, 1.0), Normal("x", 1.0, 1.0)],
            "law",
            label="d",
        )

        record_plan = _plan({"p": ra}, {"p": NumericRecordBatch})
        dist_plan = _plan({"d": laws}, {"d": DistributionBatch})
        any_plan = _plan({"p": ra}, {"p": Any})

        assert record_plan.regime == "none"
        assert dist_plan.regime == "none"
        assert any_plan.regime == "none"


class TestArrayGrouping:
    def test_sibling_views_zip_into_one_group(self):
        ra = NumericRecordBatch.stack(
            [
                NumericRecord(
                    {"x": float(i), "y": float(2 * i)},
                    label="nr",
                )
                for i in range(4)
            ],
            level_name="draw",
        )

        views = ra.select_all()
        plan = _plan({"x": views["x"], "y": views["y"]})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("x"), _ref("y"))
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("x"), _ref("y")),
                batch_shape=(4,),
                size=4,
                level_names=("draw",),
                axis_groups=((4,),),
            ),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_batches_with_no_level_in_common_form_a_product(self):
        """Levels align by name, so batches sharing none are independent: each is
        its own group and the sweep ranges over the grid."""
        ra_a = _numeric_record_batch("a", range(3), level_name="outer")
        ra_b = _numeric_record_batch("b", range(2), level_name="inner")

        plan = _plan({"a": ra_a.select("a")["a"], "b": ra_b.select("b")["b"]})

        assert plan.regime == "sweep"
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("a"),),
                batch_shape=(3,),
                size=3,
                level_names=("outer",),
                axis_groups=((3,),),
            ),
            ArrayBroadcastGroup(
                arg_refs=(_ref("b"),),
                batch_shape=(2,),
                size=2,
                level_names=("inner",),
                axis_groups=((2,),),
            ),
        )
        assert plan.sweep_batch_shape == (3, 2)
        assert plan.n_sweep == 6

    def test_one_level_name_at_two_sizes_is_refused(self):
        """Two batches naming the same level claim to range over the same thing,
        so disagreeing about its size is a mistake rather than a product."""
        ra_a = _numeric_record_batch("a", range(3))
        ra_b = _numeric_record_batch("b", range(2))

        with pytest.raises(ApplicabilityError, match="different shapes per level"):
            _plan({"a": ra_a.select("a")["a"], "b": ra_b.select("b")["b"]})

    def test_a_distribution_batch_sweeps_on_its_own_levels(self):
        store = np.empty(6, dtype=object)
        for position in range(6):
            store[position] = Normal("x", float(position), 1.0)
        laws = DistributionBatch(
            store.reshape(2, 3),
            ("row", "col"),
            label="d",
        )

        plan = _plan({"d": laws})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("d"),)
        assert plan.array_groups == (
            ArrayBroadcastGroup(
                arg_refs=(_ref("d"),),
                batch_shape=(2, 3),
                size=6,
                level_names=("row", "col"),
                axis_groups=((2,), (3,)),
            ),
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
        signature_info = _binding.make_signature_info_from_signature(signature)
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
        assert _stochastic_plan({"rows": _numeric_record_batch("x", range(2))}) is None

    def test_sampled_singleton_plan_is_frozen_and_tuple_only(self):
        plan = _stochastic_plan(
            {
                "a": Normal("a", loc=0.0, scale=1.0),
                "b": Normal("b", loc=1.0, scale=1.0),
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
            "left": _numeric_record_batch("x", range(2), level_name="left"),
            "right": _numeric_record_batch("y", range(3), level_name="right"),
            "noise": Normal("noise", loc=0.0, scale=1.0),
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
    def test_batch_alignment_does_not_merge_stochastic_source_identity(self):
        batch = NumericRecordBatch.stack(
            [
                NumericRecord(
                    {"x": float(i), "y": float(2 * i)},
                    label="row",
                )
                for i in range(3)
            ],
            level_name="draw",
        )
        views = batch.select_all()
        first = Normal("same", loc=0.0, scale=1.0)
        second = Normal("same", loc=0.0, scale=1.0)
        values = {
            "x": views["x"],
            "y": views["y"],
            "first": first,
            "second": second,
        }

        broadcast_plan = _plan(values)
        stochastic_plan = build_stochastic_plan(values, broadcast_plan, 5)

        assert len(broadcast_plan.array_groups) == 1
        assert broadcast_plan.array_groups[0].arg_refs == (_ref("x"), _ref("y"))
        assert tuple(group.arg_refs for group in stochastic_plan.source_groups) == (
            (_ref("first"),),
            (_ref("second"),),
        )
        assert len(stochastic_plan.logical_units) == 3
        assert len(stochastic_plan.random_events) == 6

    def test_record_views_share_their_known_parent_group(self):
        joint = Normal("x", loc=0.0, scale=1.0) * Normal("y", loc=1.0, scale=1.0)

        plan = _stochastic_plan(
            {
                "x": joint["x"],
                "independent": Normal("independent", loc=2.0, scale=1.0),
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
        shared = Normal("shared", loc=0.0, scale=1.0)

        plan = _stochastic_plan({"first": shared, "second": shared})

        assert tuple(group.arg_refs for group in plan.source_groups) == (
            (_ref("first"), _ref("second")),
        )
        assert tuple(group.stochastic_source_id for group in plan.source_groups) == (
            ("source-group", 0),
        )

    def test_equal_but_distinct_direct_objects_remain_independent(self):
        first = Normal("same", loc=0.0, scale=1.0)
        second = Normal("same", loc=0.0, scale=1.0)

        plan = _stochastic_plan({"first": first, "second": second})

        assert tuple(group.arg_refs for group in plan.source_groups) == (
            (_ref("first"),),
            (_ref("second"),),
        )
        assert plan.runtime_bindings[0].root is first
        assert plan.runtime_bindings[1].root is second

    def test_root_sibling_and_transitive_views_share_canonical_paths(self):
        joint = (
            Normal("left", loc=0.0, scale=1.0) * Normal("right", loc=1.0, scale=1.0)
        ).with_path_names({"left": "nested/left", "right": "nested/right"}) * Normal(
            "top", loc=2.0, scale=1.0
        )

        plan = _stochastic_plan(
            {
                "left": joint["nested/left"],
                "root": joint,
                "right": joint["nested/right"],
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
        first = _stochastic_plan({"x": Normal("first", loc=0.0, scale=1.0)})
        second = _stochastic_plan({"x": Normal("second", loc=9.0, scale=3.0)})

        assert first == second
        assert first.runtime_bindings[0].root is not second.runtime_bindings[0].root


class TestStochasticEvaluationPlanning:
    def test_wholly_exact_empiricals_record_cartesian_order_and_no_events(self):
        values = {
            "a": EmpiricalDistribution(jnp.asarray([1.0, 2.0]), component="a"),
            "b": EmpiricalDistribution(jnp.asarray([10.0, 20.0, 30.0]), component="b"),
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
            "large_first": EmpiricalDistribution(jnp.arange(5.0), component="large_first"),
            "small_second": EmpiricalDistribution(jnp.arange(2.0), component="small_second"),
            "sampled": Normal("sampled", loc=0.0, scale=1.0),
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
            {"x": EmpiricalDistribution(jnp.arange(20.0), component="x")},
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
                "first": EmpiricalDistribution(jnp.arange(3.0), component="first"),
                "second": EmpiricalDistribution(jnp.arange(3.0), component="second"),
                "third": EmpiricalDistribution(jnp.arange(3.0), component="third"),
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
            "rows": _numeric_record_batch("x", range(3)),
            "a": Normal("a", loc=0.0, scale=1.0),
            "b": Normal("b", loc=1.0, scale=1.0),
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
        values = {"x": Normal("x", loc=0.0, scale=1.0)}

        with pytest.raises(error, match=message):
            _stochastic_plan(values, n_broadcast_samples=n_broadcast_samples)


class TestStochasticPlanPurity:
    def test_planner_does_not_read_entropy_claim_keys_or_mutate_inputs(self):
        source = Normal("x", loc=0.0, scale=1.0)
        values = {"x": source}

        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit_invocation,
        ):
            plan = _stochastic_plan(values)

        assert plan.random_events == (PlannedRandomEvent(("source-group", 0), ("singleton",)),)
        assert values == {"x": source}
        assert values["x"] is source
        urandom.assert_not_called()
        commit_invocation.assert_not_called()


def _batch(level: str = "draw", n: int = 3, **fields) -> Any:
    from probpipe.core._numeric_record_batch import NumericRecordBatch
    from probpipe.core._specs import RecordSpec

    fields = fields or {"x": jnp.arange(float(n))}
    return NumericRecordBatch(
        dict(fields),
        (level,),
        element_spec=RecordSpec({name: value.shape[1:] for name, value in fields.items()}),
        label="batch",
    )


class TestBatchGrouping:
    """A batch has no parent pointer, so grouping follows its level names."""

    def test_sibling_select_all_views_zip_into_one_group(self):
        batch = _batch(x=jnp.arange(3.0), y=jnp.arange(3.0) * 10)
        views = batch.select_all()

        plan = _plan({"x": views["x"], "y": views["y"]})

        assert plan.regime == "sweep"
        assert len(plan.array_groups) == 1
        assert plan.array_groups[0].arg_refs == (_ref("x"), _ref("y"))
        assert plan.sweep_batch_shape == (3,)
        assert plan.n_sweep == 3

    def test_a_batch_zips_with_its_own_view(self):
        batch = _batch(x=jnp.arange(3.0), y=jnp.arange(3.0) * 10)

        plan = _plan({"whole": batch, "x": batch.select("x")["x"]})

        assert len(plan.array_groups) == 1
        assert plan.n_sweep == 3

    def test_batches_with_no_level_in_common_form_a_product(self):
        plan = _plan({"a": _batch("outer", 3), "b": _batch("inner", 2)})

        assert len(plan.array_groups) == 2
        assert plan.sweep_batch_shape == (3, 2)
        assert plan.n_sweep == 6

    def test_one_level_name_at_two_sizes_is_refused(self):
        """Two batches naming the same level claim to range over the same thing,
        so disagreeing about its size is a mistake rather than a product."""
        import pytest

        with pytest.raises(ApplicabilityError, match="different shapes per level"):
            _plan({"a": _batch("draw", 3), "b": _batch("draw", 2)})


class TestAnnotationDispatch:
    """The hint says what the body accepts whole, so the value answers it."""

    def test_the_exact_annotation_skips_the_sweep(self):
        from probpipe.core._numeric_record_batch import NumericRecordBatch

        batch = _batch()
        plan = _plan({"p": batch}, hints={"p": NumericRecordBatch})

        assert plan.regime == "none"

    def test_another_container_class_does_not_skip_it(self):
        """A batch passed where a DistributionBatch was declared is not what the
        body said it takes whole, so it sweeps."""
        batch = _batch()
        plan = _plan({"p": batch}, hints={"p": DistributionBatch})

        assert plan.regime == "sweep"
        assert plan.array_args == (_ref("p"),)


class TestPartialLevelOverlap:
    def test_a_shared_level_across_different_level_sets_is_refused(self):
        """Aligning one shared level across differently-leveled operands is not
        built; a product would read the shared name as two unrelated axes and
        mint the same level twice."""
        from probpipe.core._specs import RecordSpec

        two = NumericRecordBatch(
            {"x": jnp.arange(6.0).reshape(2, 3)},
            ("chain", "draw"),
            element_spec=RecordSpec(x=()),
            axes_per_level=(1, 1),
            label="batch",
        )
        one = _batch("draw", 3)

        with pytest.raises(
            ApplicabilityError, match="Rename 'draw' on one of them with with_level_names"
        ):
            _plan({"a": two, "b": one})

    def test_the_same_levels_at_different_geometries_are_refused(self):
        """The flat shape can agree while the partition does not; zipping would
        hand the output whichever partition arrived first."""
        from probpipe.core._specs import RecordSpec

        ga = NumericRecordBatch(
            {"x": jnp.zeros((2, 3, 4))},
            ("a", "b"),
            element_spec=RecordSpec(x=()),
            axes_per_level=(1, 2),
            label="batch",
        )
        gb = NumericRecordBatch(
            {"y": jnp.zeros((2, 3, 4))},
            ("a", "b"),
            element_spec=RecordSpec(y=()),
            axes_per_level=(2, 1),
            label="batch",
        )

        with pytest.raises(
            ApplicabilityError, match=r"same levels \('a', 'b'\) but different shapes per level"
        ):
            _plan({"a": ga, "b": gb})


class TestBatchAnnotationsSuppressTheSweep:
    def test_the_abstract_batch_annotation_takes_the_value_whole(self):
        from probpipe.core._batch import Batch

        plan = _plan({"p": _batch()}, hints={"p": Batch})

        assert plan.regime == "none"

    def test_a_generic_alias_answers_by_its_origin(self):
        from probpipe.core._batch import Batch

        plan = _plan({"p": _batch()}, hints={"p": Batch[dict]})

        assert plan.regime == "none"


class TestOptionalBatchAnnotations:
    def test_an_optional_container_annotation_still_takes_the_value_whole(self):
        """``Batch | None`` names the container as surely as ``Batch``: the value
        answers whichever arm it satisfies, and ``None`` answers none."""
        from typing import Optional

        from probpipe.core._batch import Batch
        from probpipe.core._record_batch import RecordBatch

        batch = _batch()
        hints = (Batch | None, RecordBatch | None, Optional[Batch], Batch[dict] | None)  # noqa: UP045
        for hint in hints:
            plan = _plan({"p": batch}, hints={"p": hint})

            assert plan.regime == "none", hint

    def test_a_non_container_union_still_sweeps(self):
        from probpipe import Record

        plan = _plan({"p": _batch()}, hints={"p": Record | None})

        assert plan.regime == "sweep"
