"""Tests for Function broadcast planning."""

from __future__ import annotations

import inspect
from typing import Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import DistributionArray, Normal, NumericRecord, NumericRecordArray
from probpipe.core import _workflow_call
from probpipe.core._workflow_distribution_normalization import (
    normalize_distribution_values,
)
from probpipe.core._workflow_plan import ArrayBroadcastGroup, build_broadcast_plan
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
