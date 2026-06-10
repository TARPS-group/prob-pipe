"""Tests for WorkflowFunction broadcast planning."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import DistributionArray, Normal, NumericRecord, NumericRecordArray
from probpipe.core._workflow_distribution_normalization import (
    normalize_distribution_values,
)
from probpipe.core._workflow_plan import ArrayBroadcastGroup, build_broadcast_plan
from probpipe.core.distribution import Distribution
from probpipe.core.protocols import SupportsSampling


def _numeric_record_array(field: str, values: range) -> NumericRecordArray:
    return NumericRecordArray.stack(
        [NumericRecord(**{field: float(value)}) for value in values]
    )


class TestBroadcastRegime:
    def test_plain_values_do_not_broadcast(self):
        plan = build_broadcast_plan(values={"x": 1.0}, hints={})

        assert plan.regime == "none"
        assert plan.dist_args == ()
        assert plan.array_args == ()
        assert plan.array_groups == ()
        assert plan.sweep_batch_shape == ()
        assert plan.n_sweep == 1

    def test_distribution_value_selects_distribution_regime(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        plan = build_broadcast_plan(values={"x": dist}, hints={})

        assert plan.regime == "distribution"
        assert plan.dist_args == ("x",)
        assert plan.array_args == ()

    def test_record_array_value_selects_sweep_regime(self):
        values = {"p": _numeric_record_array("x", range(4))}

        plan = build_broadcast_plan(values=values, hints={})

        assert plan.regime == "sweep"
        assert plan.dist_args == ()
        assert plan.array_args == ("p",)
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_names=("p",), batch_shape=(4,), size=4),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_array_and_distribution_values_select_nested_regime(self):
        values = {
            "p": _numeric_record_array("x", range(4)),
            "noise": Normal(loc=0.0, scale=1.0, name="noise"),
        }

        plan = build_broadcast_plan(values=values, hints={})

        assert plan.regime == "nested"
        assert plan.dist_args == ("noise",)
        assert plan.array_args == ("p",)


class TestHintClassification:
    def test_distribution_hints_skip_scalar_distribution_broadcast(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        concrete = build_broadcast_plan(
            values={"x": dist},
            hints={"x": Distribution},
        )
        protocol = build_broadcast_plan(
            values={"x": dist},
            hints={"x": SupportsSampling},
        )

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

        record_plan = build_broadcast_plan(
            values={"p": ra},
            hints={"p": NumericRecordArray},
        )
        dist_plan = build_broadcast_plan(
            values={"d": da},
            hints={"d": DistributionArray},
        )
        any_plan = build_broadcast_plan(
            values={"p": ra},
            hints={"p": Any},
        )

        assert record_plan.regime == "none"
        assert dist_plan.regime == "none"
        assert any_plan.regime == "none"


class TestArrayGrouping:
    def test_sibling_views_zip_into_one_group(self):
        ra = NumericRecordArray.stack(
            [
                NumericRecord(x=float(i), y=float(2 * i))
                for i in range(4)
            ]
        )

        plan = build_broadcast_plan(
            values={"x": ra.view("x"), "y": ra.view("y")},
            hints={},
        )

        assert plan.regime == "sweep"
        assert plan.array_args == ("x", "y")
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_names=("x", "y"), batch_shape=(4,), size=4),
        )
        assert plan.sweep_batch_shape == (4,)
        assert plan.n_sweep == 4

    def test_views_from_different_parents_use_product_shape(self):
        ra_a = _numeric_record_array("a", range(3))
        ra_b = _numeric_record_array("b", range(2))

        plan = build_broadcast_plan(
            values={"a": ra_a.view("a"), "b": ra_b.view("b")},
            hints={},
        )

        assert plan.regime == "sweep"
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_names=("a",), batch_shape=(3,), size=3),
            ArrayBroadcastGroup(arg_names=("b",), batch_shape=(2,), size=2),
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

        plan = build_broadcast_plan(values={"d": da}, hints={})

        assert plan.regime == "sweep"
        assert plan.array_args == ("d",)
        assert plan.array_groups == (
            ArrayBroadcastGroup(arg_names=("d",), batch_shape=(2, 3), size=6),
        )
        assert plan.sweep_batch_shape == (2, 3)
        assert plan.n_sweep == 6


class TestPlanPurity:
    def test_planner_does_not_convert_or_mutate_external_distributions(self):
        external = tfd.Normal(loc=0.0, scale=1.0)
        values = {"x": external}

        raw_plan = build_broadcast_plan(values=values, hints={})
        normalized = normalize_distribution_values(values=values, hints={})
        normalized_plan = build_broadcast_plan(values=normalized, hints={})

        assert values["x"] is external
        assert raw_plan.regime == "none"
        assert normalized_plan.regime == "distribution"
