"""Successful workflow RNG provenance recipe tests."""

from __future__ import annotations

import json
from unittest.mock import patch

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import bijectors as tfb

import probpipe
from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericRecord,
    NumericRecordArray,
    Provenance,
    ProvenanceMode,
    TransformedDistribution,
    sample,
    workflow_run,
)


def _identity(value):
    return value


def _difference(left, right):
    return right - left


def _add(row, noise):
    return row["x"] + noise


def _draw_value(value):
    return sample(Normal(loc=value, scale=1.0, name="draw"))["sample"]


_INNER_DRAW = Function(func=_draw_value, dispatch="sequential")


def _nested_automatic(value):
    return _INNER_DRAW(value=value)["_draw_value"]


def _recipe(result):
    assert result.provenance is not None
    return result.provenance.controls["workflow_rng"]


def _record_array():
    return NumericRecordArray.stack([NumericRecord("row", x=float(value)) for value in range(2)])


class TestWorkflowRecipeRecording:
    def test_seeded_lifting_records_root_plan_and_one_batched_event(self):
        workflow = Function(
            func=_identity,
            dispatch="sequential",
            n_broadcast_samples=11,
        )
        with workflow_run(seed=7):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        recipe = _recipe(result)
        assert recipe["abi"] == "probpipe.workflow_rng_recipe/v1"
        assert recipe["root_words"] == [0, 7]
        assert recipe["occurrence_path"] == [["invocation", 0]]
        assert recipe["expected_event_count"] == 1
        assert len(recipe["events"]) == 1
        assert recipe["events"][0]["effect"]["sample_shape"] == [11]
        assert recipe["stochastic_plan"]["abi"] == "probpipe.stochastic_plan/v1"
        assert recipe["stochastic_plan"]["n_evaluations"] == 11

    def test_anonymous_and_ephemeral_roots_are_recorded_only_after_success(self):
        workflow = Function(func=_identity, n_broadcast_samples=5)
        entropy = bytes.fromhex("0123456789abcdef")

        with patch(
            "probpipe.core._workflow_context._os_urandom",
            return_value=entropy,
        ) as urandom:
            with workflow_run():
                anonymous = workflow(value=Normal(loc=0.0, scale=1.0, name="x"))
            ephemeral = workflow(value=Normal(loc=0.0, scale=1.0, name="x"))

        assert _recipe(anonymous)["root_words"] == [0x01234567, 0x89ABCDEF]
        assert _recipe(ephemeral)["root_words"] == [0x01234567, 0x89ABCDEF]
        assert anonymous.provenance.diagnostics["workflow_rng"]["rng_origin"] == ("anonymous")
        assert ephemeral.provenance.diagnostics["workflow_rng"]["rng_origin"] == ("ephemeral")
        assert urandom.call_count == 2

    def test_deterministic_exact_and_caller_keyed_calls_have_no_recipe(self):
        deterministic = Function(func=_identity)(value=3.0)
        exact_workflow = Function(func=_identity, n_broadcast_samples=8)
        exact = exact_workflow(value=EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="exact"))
        caller_keyed = sample(
            Normal(loc=0.0, scale=1.0, name="x"),
            key=jax.random.key(4),
        )

        assert deterministic.provenance.controls == {}
        assert exact.provenance.controls == {}
        assert caller_keyed.provenance.controls == {}

    def test_direct_automatic_sample_recipe_remains_standalone(self):
        with workflow_run(seed=4):
            result = sample(Normal(loc=0.0, scale=1.0, name="x"))

        recipe = _recipe(result)
        assert recipe["stochastic_plan"]["kind"] == "direct_operation"
        assert recipe["events"][0]["effect"]["operation_kind"] == "sample"
        assert recipe["standalone_eligible"] is True

    def test_mixed_plan_records_only_the_sampled_root_event(self):
        workflow = Function(func=_difference, n_broadcast_samples=5)
        with workflow_run(seed=9):
            result = workflow(
                left=EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="left"),
                right=Normal(loc=0.0, scale=1.0, name="right"),
            )

        recipe = _recipe(result)
        assert recipe["expected_event_count"] == 1
        assert [
            group["execution_mode"] for group in recipe["stochastic_plan"]["source_groups"]
        ] == ["exact", "sampled"]

    def test_alias_and_supported_descendant_share_one_recipe_source(self):
        root = Normal(loc=0.0, scale=1.0, name="root")
        descendant = TransformedDistribution(root, tfb.Exp())
        workflow = Function(func=_difference, n_broadcast_samples=6)

        with workflow_run(seed=3):
            result = workflow(left=root, right=descendant)

        plan = _recipe(result)["stochastic_plan"]
        assert len(plan["source_groups"]) == 1
        assert len(plan["source_groups"][0]["consumers"]) == 2
        descriptor = plan["source_groups"][0]["consumers"][1]["descendant_descriptor"]
        assert descriptor[0] == "stochastic-descendant"
        assert "transformed-descendant" in json.dumps(descriptor)

    def test_nested_sweep_recipe_contains_every_canonical_unit(self):
        workflow = Function(func=_add, n_broadcast_samples=5, dispatch="sequential")
        with workflow_run(seed=12):
            result = workflow(
                row=_record_array(),
                noise=Normal(loc=0.0, scale=1.0, name="noise"),
            )

        recipe = _recipe(result)
        assert recipe["expected_event_count"] == 2
        assert [event["logical_unit_id"] for event in recipe["events"]] == [
            ["cell", 0],
            ["cell", 1],
        ]

    def test_nested_automatic_function_is_marked_non_standalone(self):
        workflow = Function(func=_nested_automatic, dispatch="thread")
        with workflow_run(seed=21):
            result = workflow(value=1.0)

        recipe = _recipe(result)
        assert recipe["standalone_eligible"] is False
        assert recipe["standalone_restriction"] == "nested_automatic_function"

    def test_recipe_roundtrip_contains_no_operational_ownership_state(self):
        workflow = Function(func=_identity, dispatch="thread", n_broadcast_samples=5)
        with workflow_run(seed=5):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        payload = result.provenance.to_dict()
        restored = Provenance.from_dict(json.loads(json.dumps(payload)))

        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        assert restored.controls == result.provenance.controls
        persisted_keys = keys(restored.controls)
        for forbidden in ("owner", "token", "attempt", "worker", "ledger"):
            assert forbidden not in persisted_keys

    def test_provenance_off_persists_no_recipe(self):
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        workflow = Function(func=_identity, n_broadcast_samples=5)

        with workflow_run(seed=7):
            result = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        assert result.provenance is None
