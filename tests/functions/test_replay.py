"""Standalone workflow RNG replay scope and preflight tests."""

from __future__ import annotations

import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    Normal,
    NumericArraySpec,
    OpaqueSpec,
    OutputSpec,
    Provenance,
    RecordSpec,
    ReplayCompatibilityError,
    ReplayUnsupportedCallableError,
    UnmanagedConcurrentWorkflowEntryError,
    replay_run,
    sample,
    workflow_run,
)
from probpipe.functions import _replay
from probpipe.functions._managed import (
    ManagedAttemptState,
    ManagedWorkItemToken,
)
from tests.functions import _replay_fixtures
from tests.functions._replay_fixtures import (
    replayable_affine,
    replayable_identity,
    replayable_optional_nested,
)


def _draw(seed: int = 7):
    with workflow_run(seed=seed):
        return sample(Normal(loc=0.0, scale=1.0, label="value"))


def _lifted_draw(seed: int = 7):
    workflow = Function(
        label="replayable_identity",
        fn=replayable_identity,
        dispatch="sequential",
        n_broadcast_samples=5,
    )
    with workflow_run(seed=seed):
        return workflow(value=Normal(loc=0.0, scale=1.0, label="value"))


def _sample_value(result):
    return np.asarray(result)


def _marginal_values(result):
    return np.asarray(result.atoms)


def _mutate_provenance(provenance, mutate):
    payload = provenance.to_dict()
    mutate(payload["controls"])
    return Provenance.from_dict(payload)


_DELETE = object()


def _edit_controls_path(controls, path, value):
    target = controls
    for segment in path[:-1]:
        target = target[segment]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = copy.deepcopy(value)


def _plan_for_effect(effect):
    return SimpleNamespace(
        operation_kind=effect.operation_kind,
        execution_mode=effect.execution_mode,
        event=SimpleNamespace(
            stochastic_source_id=effect.stochastic_source_id,
            logical_unit_id=effect.logical_unit_id,
        ),
        sample_shape=effect.sample_shape,
        sampling_abi=effect.sampling_abi,
        provider_abi=effect.provider_abi,
        record_path=effect.record_path,
        descendant_descriptor=effect.descendant_descriptor,
    )


class TestReplayScope:
    def test_seeded_serialized_and_replay_of_replay_roundtrip(self):
        original = _draw(seed=17)
        restored = Provenance.from_dict(json.loads(json.dumps(original.provenance.to_dict())))
        anchor = restored.controls["replay"]["callable"]
        assert anchor["definition_abi"] == "probpipe.callable_definition/v1"
        assert "signature_and_declarations" in anchor
        assert "signature_and_templates" not in anchor

        with replay_run(restored):
            first = sample(Normal(loc=0.0, scale=1.0, label="value"))
        with replay_run(first.provenance):
            second = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(first), _sample_value(original))
        np.testing.assert_array_equal(_sample_value(second), _sample_value(original))
        assert first.provenance.controls["randomness"] == original.provenance.controls["randomness"]
        assert (
            second.provenance.controls["randomness"] == original.provenance.controls["randomness"]
        )
        assert first.provenance.diagnostics["rng_origin"] == {
            "context_kind": "replay_run",
            "root_source": "replay_recipe",
            "supplied_seed": None,
        }

    @pytest.mark.parametrize("explicit_scope", [True, False])
    def test_anonymous_and_ephemeral_replay_do_not_read_entropy(self, explicit_scope):
        entropy = bytes.fromhex("0123456789abcdef")
        with patch(
            "probpipe.functions._context._os_urandom",
            return_value=entropy,
        ) as urandom:
            if explicit_scope:
                with workflow_run():
                    original = sample(Normal(loc=0.0, scale=1.0, label="value"))
            else:
                original = sample(Normal(loc=0.0, scale=1.0, label="value"))
        assert urandom.call_count == 1

        with (
            patch(
                "probpipe.functions._context._os_urandom",
                side_effect=AssertionError("replay must use recorded root words"),
            ),
            replay_run(original.provenance),
        ):
            replayed = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))

    def test_nested_managed_result_has_a_standalone_replay_recipe(self):
        captured = []

        def outer(value):
            result = sample(Normal(loc=value, scale=1.0, label="value"))
            captured.append(result)
            return result

        workflow = Function(label="outer", fn=outer, dispatch="sequential")
        with workflow_run(seed=17):
            workflow(value=0.0)
        nested = captured[0]

        with replay_run(nested.provenance):
            replayed = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(nested))

    def test_empty_second_root_and_apply_are_rejected(self):
        original = _draw()

        with (
            pytest.raises(ReplayCompatibilityError, match="without a Function call"),
            replay_run(original.provenance),
        ):
            pass

        with replay_run(original.provenance):
            sample(Normal(loc=0.0, scale=1.0, label="value"))
            with pytest.raises(ReplayCompatibilityError, match="second one"):
                sample(Normal(loc=0.0, scale=1.0, label="value"))

        with replay_run(original.provenance):
            sample(Normal(loc=0.0, scale=1.0, label="value"))
            with pytest.raises(ReplayCompatibilityError, match="apply"):
                sample.apply(Normal(loc=0.0, scale=1.0, label="value"))

    def test_workflow_run_cannot_replace_the_replay_root(self):
        original = _draw()

        with replay_run(original.provenance):
            with (
                pytest.raises(ReplayCompatibilityError, match="cannot be nested"),
                workflow_run(seed=999),
            ):
                pass
            sample(Normal(loc=0.0, scale=1.0, label="value"))

    def test_replay_scope_rejects_reentry_nesting_and_active_workflow(self):
        original = _draw()
        scope = replay_run(original.provenance)

        with scope:
            with pytest.raises(RuntimeError, match="already active"):
                scope.__enter__()
            with (
                pytest.raises(ReplayCompatibilityError, match="cannot be nested"),
                replay_run(original.provenance),
            ):
                pass
            sample(Normal(loc=0.0, scale=1.0, label="value"))

        with (
            workflow_run(seed=9),
            pytest.raises(ReplayCompatibilityError, match="outside an active workflow_run"),
            replay_run(original.provenance),
        ):
            pass

    def test_caught_failed_root_is_rejected_when_scope_exits(self):
        original = _draw()
        changed = Function(label="replayable_affine", fn=replayable_affine, n_broadcast_samples=5)

        with (
            pytest.raises(ReplayCompatibilityError, match="did not complete"),
            replay_run(original.provenance),
            pytest.raises(ReplayCompatibilityError, match="expected a call to sample"),
        ):
            changed(value=Normal(loc=0.0, scale=1.0, label="value"))


class TestReplayOwnership:
    def test_rejected_copied_thread_context_does_not_consume_owner_root(self):
        original = _draw()

        with replay_run(original.provenance):
            copied = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    copied.run,
                    sample,
                    Normal(loc=0.0, scale=1.0, label="value"),
                )
                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    future.result()
            replayed = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))

    def test_rejected_copied_task_context_does_not_consume_owner_root(self):
        original = _draw()

        async def run_replay():
            with replay_run(original.provenance):

                async def call_in_child():
                    return sample(Normal(loc=0.0, scale=1.0, label="value"))

                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    await asyncio.create_task(call_in_child())
                return sample(Normal(loc=0.0, scale=1.0, label="value"))

        replayed = asyncio.run(run_replay())

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))


class TestReplayAdmission:
    def test_unknown_callable_abi_is_rejected_before_fields_are_read(self):
        payload = _draw().provenance.to_dict()
        anchor = payload["controls"]["replay"]["callable"]
        anchor["definition_abi"] = "probpipe.callable_definition/v99"
        signature = anchor.pop("signature_and_declarations")
        anchor["signature_and_templates"] = signature
        signature["input_template"] = signature.pop("input_spec")
        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ),
            pytest.raises(
                ReplayCompatibilityError,
                match=r"replay\.callable\.definition_abi '.*/v99'.*needs '.*/v1'",
            ),
            replay_run(Provenance.from_dict(payload)),
        ):
            raise AssertionError("An unknown callable ABI was admitted")

    def test_an_anchor_that_names_its_declarations_signature_and_templates_is_refused(self):
        payload = _draw().provenance.to_dict()
        anchor = payload["controls"]["replay"]["callable"]
        anchor["signature_and_templates"] = anchor.pop("signature_and_declarations")
        with (
            pytest.raises(
                ReplayCompatibilityError,
                match=r"replay\.callable lacks fields \['signature_and_declarations'\]",
            ),
            replay_run(Provenance.from_dict(payload)),
        ):
            raise AssertionError("A former anchor was admitted")

    def test_legacy_unknown_and_malformed_recipes_fail_at_entry(self):
        with (
            pytest.raises(ReplayCompatibilityError, match="no recorded random draws"),
            replay_run(Provenance("legacy")),
        ):
            pass

        payload = _draw().provenance.to_dict()
        payload["controls"]["randomness"]["rng_abi"] = "unknown-rng/v99"
        with (
            pytest.raises(ReplayCompatibilityError, match=r"randomness\.rng_abi 'unknown-rng/v99'"),
            replay_run(Provenance.from_dict(payload)),
        ):
            pass

    def test_a_result_passed_in_place_of_its_provenance_is_named(self):
        original = _draw()

        with (
            pytest.raises(
                ReplayCompatibilityError,
                match=rf"expects a Provenance, got {type(original).__name__}\. Pass "
                r"result\.provenance",
            ),
            replay_run(original),
        ):
            pass

    def test_a_python_version_drift_names_both_versions(self):
        changed = _mutate_provenance(
            _draw().provenance,
            lambda controls: controls["replay"]["callable"].update(python_replay_abi="cpython-2.7"),
        )

        with (
            pytest.raises(
                ReplayCompatibilityError,
                match=r"recorded under cpython-2\.7, but this interpreter is",
            ),
            replay_run(changed),
        ):
            sample(Normal(loc=0.0, scale=1.0, label="value"))

    def test_a_sample_shape_drift_names_the_draw(self):
        original = _draw()

        with (
            pytest.raises(
                ReplayCompatibilityError,
                match=r"\(sample with sample_shape=\(3,\)\) that matches no draw",
            ),
            replay_run(original.provenance),
        ):
            sample(Normal(loc=0.0, scale=1.0, label="value"), sample_shape=(3,))

    @pytest.mark.parametrize(
        "mapping_path",
        [
            pytest.param((), id="controls"),
            pytest.param(("randomness",), id="randomness"),
            pytest.param(("replay",), id="replay"),
            pytest.param(("replay", "standalone"), id="standalone"),
            pytest.param(("replay", "callable"), id="callable"),
            pytest.param(
                ("replay", "callable", "signature_and_declarations"),
                id="callable-signature",
            ),
            pytest.param(
                ("replay", "callable", "signature_and_declarations", "parameters", 0),
                id="callable-parameter",
            ),
            pytest.param(("replay", "plan"), id="plan"),
            pytest.param(
                ("replay", "plan", "canonical_fields"),
                id="canonical-plan",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "arg_refs", 0),
                id="plan-arg-ref",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "source_groups", 0),
                id="plan-source-group",
            ),
            pytest.param(
                (
                    "replay",
                    "plan",
                    "canonical_fields",
                    "source_groups",
                    0,
                    "consumers",
                    0,
                ),
                id="plan-consumer",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "logical_units", 0),
                id="plan-logical-unit",
            ),
            pytest.param(("randomness", "events", 0), id="random-event"),
            pytest.param(("replay", "compatibility"), id="compatibility"),
            pytest.param(
                ("replay", "plan", "expected_effects", 0),
                id="expected-effect",
            ),
        ],
    )
    def test_unknown_version_one_fields_fail_at_replay_entry(self, mapping_path):
        payload = _lifted_draw().provenance.to_dict()
        target = payload["controls"]
        for segment in mapping_path:
            target = target[segment]
        target["unknown_field_v2"] = 1
        changed = Provenance.from_dict(payload)

        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(
                ReplayCompatibilityError, match=r"unexpected fields \['unknown_field_v2'\]"
            ),
            replay_run(changed),
        ):
            raise AssertionError("replay admission accepted unknown structure")

        derive_key.assert_not_called()

    @pytest.mark.parametrize(
        ("record_name", "schema", "message"),
        [
            pytest.param(
                "randomness",
                "probpipe.rng_recipe/v2",
                "has randomness.schema 'probpipe.rng_recipe/v2'",
                id="randomness",
            ),
            pytest.param(
                "replay",
                "probpipe.replay_anchor/v2",
                "has replay.schema 'probpipe.replay_anchor/v2'",
                id="replay",
            ),
        ],
    )
    def test_unknown_schema_precedes_version_one_field_validation(
        self,
        record_name,
        schema,
        message,
    ):
        payload = _lifted_draw().provenance.to_dict()
        record = payload["controls"][record_name]
        record["schema"] = schema
        record["future_field"] = 1
        changed = Provenance.from_dict(payload)

        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(ReplayCompatibilityError, match=message),
            replay_run(changed),
        ):
            raise AssertionError("replay admission accepted an unknown schema")

        derive_key.assert_not_called()

    @pytest.mark.parametrize(
        ("mapping_path", "field_name"),
        [
            pytest.param((), "randomness", id="controls"),
            pytest.param(("randomness",), "schema", id="randomness"),
            pytest.param(("replay",), "schema", id="replay"),
            pytest.param(("replay", "standalone"), "restriction", id="standalone"),
            pytest.param(("replay", "callable"), "sha256", id="callable"),
            pytest.param(
                ("replay", "callable", "signature_and_declarations"),
                "output_spec",
                id="callable-signature",
            ),
            pytest.param(
                ("replay", "callable", "signature_and_declarations", "parameters", 0),
                "annotation",
                id="callable-parameter",
            ),
            pytest.param(("replay", "plan"), "schema", id="plan"),
            pytest.param(
                ("replay", "plan", "canonical_fields"),
                "exact_group_order",
                id="canonical-plan",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "arg_refs", 0),
                "label",
                id="plan-arg-ref",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "source_groups", 0),
                "exact_size",
                id="plan-source-group",
            ),
            pytest.param(
                (
                    "replay",
                    "plan",
                    "canonical_fields",
                    "source_groups",
                    0,
                    "consumers",
                    0,
                ),
                "record_path",
                id="plan-consumer",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "logical_units", 0),
                "flat_index",
                id="plan-logical-unit",
            ),
            pytest.param(("randomness", "events", 0), "unit", id="random-event"),
            pytest.param(
                ("replay", "compatibility"),
                "provider_abi",
                id="compatibility",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0),
                "provider_abi",
                id="expected-effect",
            ),
        ],
    )
    def test_missing_version_one_fields_fail_at_replay_entry(
        self,
        mapping_path,
        field_name,
    ):
        payload = _lifted_draw().provenance.to_dict()
        target = payload["controls"]
        for segment in mapping_path:
            target = target[segment]
        del target[field_name]
        changed = Provenance.from_dict(payload)
        if mapping_path == () and field_name == "randomness":
            expected_error = "no recorded random draws"
        elif field_name == "schema":
            expected_error = r"has no .*schema|lacks fields \['schema'\]"
        else:
            expected_error = rf"lacks fields \['{field_name}'\]"

        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(ReplayCompatibilityError, match=expected_error),
            replay_run(changed),
        ):
            raise AssertionError("replay admission accepted missing structure")

        derive_key.assert_not_called()

    @pytest.mark.parametrize(
        ("eligibility", "restriction"),
        [
            pytest.param("supported", "nested_automatic_function", id="supported"),
            pytest.param("nested_workflow_rng_execution", None, id="nested"),
            pytest.param(
                "nested_workflow_rng_execution",
                "unknown_restriction_v2",
                id="unknown",
            ),
        ],
    )
    def test_standalone_restriction_must_match_eligibility(
        self,
        eligibility,
        restriction,
    ):
        payload = _draw().provenance.to_dict()
        payload["controls"]["replay"]["standalone"] = {
            "eligibility": eligibility,
            "restriction": restriction,
        }

        with (
            pytest.raises(
                ReplayCompatibilityError, match=r"restriction .* does not match eligibility"
            ),
            replay_run(Provenance.from_dict(payload)),
        ):
            raise AssertionError("replay admission accepted a mismatched restriction")

    @pytest.mark.parametrize(
        ("path", "value", "match"),
        [
            pytest.param(
                ("randomness", "schema"),
                "unknown-recipe/v99",
                "randomness.schema",
                id="rng-recipe-schema",
            ),
            pytest.param(
                ("replay", "schema"),
                "unknown-replay/v99",
                "replay.schema",
                id="replay-schema",
            ),
            pytest.param(
                ("replay", "standalone", "eligibility"),
                "unknown",
                "eligibility",
                id="standalone-eligibility",
            ),
            pytest.param(
                ("replay", "callable", "definition_abi"),
                "unknown-callable/v99",
                "replay.callable.definition_abi",
                id="callable-definition-abi",
            ),
            pytest.param(
                ("replay", "callable", "probpipe_replay_abi"),
                "unknown-probpipe/v99",
                "replay.callable.probpipe_replay_abi",
                id="probpipe-replay-abi",
            ),
            pytest.param(
                ("replay", "callable", "module"),
                None,
                "replay.callable.module must be a string",
                id="callable-module",
            ),
            pytest.param(
                ("replay", "callable", "signature_and_declarations"),
                [],
                "signature_and_declarations",
                id="callable-signature",
            ),
            pytest.param(
                ("replay", "plan", "schema"),
                "unknown-plan/v99",
                "replay.plan.schema",
                id="plan-schema",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "managed_child_policy"),
                "unknown-managed-child/v99",
                "managed_child_policy",
                id="managed-child-policy",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "key_ownership"),
                "caller",
                "key_ownership must be 'automatic', got 'caller'",
                id="plan-key-ownership",
            ),
            pytest.param(
                ("randomness", "expected_event_count"),
                True,
                "expected_event_count must be",
                id="event-count-bool",
            ),
            pytest.param(
                ("replay", "compatibility", "provider_abi"),
                _DELETE,
                r"replay\.compatibility lacks fields \[.provider_abi.\]",
                id="compatibility-fields",
            ),
            pytest.param(
                ("replay", "compatibility", "execution_contract"),
                "unknown-execution/v99",
                "replay.compatibility.execution_contract",
                id="execution-contract",
            ),
            pytest.param(
                ("replay", "compatibility", "descendant_adapter_abi"),
                ["unknown-descendant/v99"],
                r"descendant_adapter_abi is \[.unknown-descendant/v99.\]",
                id="descendant-adapter-abi",
            ),
            pytest.param(
                ("replay", "compatibility", "sampling_abi"),
                [""],
                "sampling_abi must contain only non-empty strings",
                id="empty-sampling-abi",
            ),
            pytest.param(
                ("replay", "compatibility", "provider_abi"),
                ["probpipe.distribution/v1", "probpipe.distribution/v1"],
                "duplicate entries",
                id="duplicate-provider-abi",
            ),
            pytest.param(
                ("randomness", "events", 0, "occurrence_path", 0, 1),
                1,
                "occurrence_path does not start with randomness.occurrence_path",
                id="event-outside-anchor",
            ),
            pytest.param(
                ("randomness", "events", 0, "occurrence_kind"),
                "child",
                r"occurrence_kind must be .* got .child.",
                id="event-occurrence-kind",
            ),
            pytest.param(
                ("randomness", "events", 0, "key_ownership"),
                "caller",
                "key_ownership must be 'automatic', got 'caller'",
                id="event-key-ownership",
            ),
            pytest.param(
                ("randomness", "events", 0, "source"),
                {},
                "source must be a list, got dict",
                id="event-source-sequence",
            ),
            pytest.param(
                ("randomness", "events", 0, "source"),
                ["source-group", True],
                "contains invalid entry True",
                id="event-source-value",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "provider_abi"),
                _DELETE,
                r"expected_effects\[0\] lacks fields",
                id="effect-fields",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "operation_kind"),
                "",
                "operation_kind must be a non-empty string",
                id="effect-operation-kind",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "sample_shape"),
                [-1],
                r"sample_shape must be null or a list of nonnegative integers, got \[-1\]",
                id="effect-sample-shape",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "record_path"),
                [1],
                "record_path must be a list of strings",
                id="effect-record-path",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "descendant_descriptor"),
                {},
                "descendant_descriptor must be null or a nested list",
                id="effect-descriptor-sequence",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "descendant_descriptor"),
                [{}],
                "descendant_descriptor must be null or a nested list",
                id="effect-descriptor-value",
            ),
        ],
    )
    def test_incompatible_recipe_fields_fail_at_entry_before_key_derivation(
        self,
        path,
        value,
        match,
    ):
        payload = _draw().provenance.to_dict()
        _edit_controls_path(payload["controls"], path, value)
        changed = Provenance.from_dict(payload)

        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(ReplayCompatibilityError, match=match),
            replay_run(changed),
        ):
            pass

        derive_key.assert_not_called()

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_mutated_nonfinite_descriptor_fails_at_entry_before_key_derivation(
        self,
        value,
    ):
        changed = Provenance.from_dict(_draw().provenance.to_dict())
        changed.controls["replay"]["plan"]["expected_effects"][0]["descendant_descriptor"] = [value]

        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(
                ReplayCompatibilityError,
                match="descendant_descriptor must be null or a nested list",
            ),
            replay_run(changed),
        ):
            pass

        derive_key.assert_not_called()

    def test_mutated_nonfinite_plan_fails_at_entry_before_key_derivation(self):
        changed = Provenance.from_dict(_draw().provenance.to_dict())
        changed.controls["replay"]["plan"]["canonical_fields"]["n_evaluations"] = float("nan")

        with (
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(ReplayCompatibilityError, match="finite JSON values"),
            replay_run(changed),
        ):
            pass

        derive_key.assert_not_called()

    def test_missing_optional_diagnostics_remain_replayable(self):
        original = _draw()
        payload = original.provenance.to_dict()
        payload["diagnostics"].pop("callable_source")
        payload["diagnostics"].pop("execution")
        restored = Provenance.from_dict(payload)

        with replay_run(restored):
            replayed = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))
        assert replayed.provenance.diagnostics["replay"]["source_artifact_drift"] is True
        assert replayed.provenance.diagnostics["replay"]["execution_drift"] is True

        payload = _draw().provenance.to_dict()
        payload["controls"]["randomness"]["root_words"] = [True, 1]
        with (
            pytest.raises(ReplayCompatibilityError, match="root_words"),
            replay_run(Provenance.from_dict(payload)),
        ):
            pass

    def test_extended_diagnostics_remain_replayable(self):
        original = _draw()
        payload = original.provenance.to_dict()
        payload["diagnostics"]["future_observation"] = {"version": 2}
        payload["diagnostics"]["callable_source"]["future_observation"] = True
        payload["diagnostics"]["execution"][0]["future_observation"] = "worker"

        with replay_run(Provenance.from_dict(payload)):
            replayed = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))

    @pytest.mark.parametrize(
        "occurrence_path",
        [
            [],
            [[]],
            [["unknown-segment", 0]],
            [["invocation", True]],
            [["operation", 0]],
            [["child", 0]],
            [["scope", 0], ["child", 0]],
            [["invocation", 0], ["invocation", 1]],
            [
                ["invocation", 0],
                ["managed-unit", "unknown-managed/v99", "point", 0],
                ["child", 0],
            ],
            [
                ["invocation", 0],
                ["managed-unit", "probpipe.managed_work_item/v1", "point", 1],
                ["child", 0],
            ],
            [
                ["invocation", 0],
                ["managed-unit", "probpipe.managed_work_item/v1", "sweep-cell"],
                ["child", 0],
            ],
            [
                ["invocation", 0],
                [
                    "managed-unit",
                    "probpipe.managed_work_item/v1",
                    "lifted-evaluation",
                    ["unknown-unit"],
                    0,
                ],
                ["child", 0],
            ],
            [
                ["invocation", 0],
                ["managed-unit", "probpipe.managed_work_item/v1", "unknown-layout", 0],
                ["child", 0],
            ],
        ],
    )
    def test_malformed_function_occurrence_paths_fail_at_entry(self, occurrence_path):
        payload = _draw().provenance.to_dict()
        randomness = payload["controls"]["randomness"]
        original_path = randomness["occurrence_path"]
        event_suffix = randomness["events"][0]["occurrence_path"][len(original_path) :]
        randomness["occurrence_path"] = occurrence_path
        randomness["events"][0]["occurrence_path"] = [*occurrence_path, *event_suffix]

        with (
            pytest.raises(ReplayCompatibilityError, match="occurrence_path"),
            replay_run(Provenance.from_dict(payload)),
        ):
            pass

    def test_unsupported_callable_and_nested_automatic_parent_fail_at_entry(self):
        unsupported = Function(label="function", fn=lambda value: value, n_broadcast_samples=5)
        with workflow_run(seed=3):
            unsupported_result = unsupported(value=Normal(loc=0.0, scale=1.0, label="value"))

        with (
            pytest.raises(ReplayUnsupportedCallableError, match="lambda"),
            replay_run(unsupported_result.provenance),
        ):
            pass

        inner = Function(
            label="function",
            fn=lambda value: sample(Normal(loc=value, scale=1.0, label="inner")),
            dispatch="sequential",
        )

        def nested(value):
            return inner(value=value)

        outer = Function(label="nested", fn=nested, dispatch="sequential")
        with workflow_run(seed=8):
            nested_result = outer(value=1.0)

        with (
            pytest.raises(ReplayCompatibilityError, match="nested Function call"),
            replay_run(nested_result.provenance),
        ):
            pass


class TestReplayPreflight:
    @pytest.mark.parametrize(
        ("path", "replacement"),
        [
            (("n_broadcast_samples",), 5.0),
            (("logical_units", 0, "flat_index"), False),
        ],
        ids=["float", "bool"],
    )
    def test_plan_scalar_types_are_exact_before_sampling(self, path, replacement):
        workflow = Function(
            label="replayable_identity", fn=replayable_identity, n_broadcast_samples=5
        )
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, label="value"))
        payload = original.provenance.to_dict()
        canonical_plan = payload["controls"]["replay"]["plan"]["canonical_fields"]
        _edit_controls_path(canonical_plan, path, replacement)
        changed = Provenance.from_dict(payload)
        candidate = Normal(loc=0.0, scale=1.0, label="value")

        with (
            patch.object(type(candidate), "_sample", side_effect=AssertionError("sampled")),
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(ReplayCompatibilityError, match="set up differently"),
            replay_run(changed),
        ):
            workflow(value=candidate)

        derive_key.assert_not_called()

    def test_callable_drift_fails_before_sampling(self):
        workflow = Function(
            label="replayable_identity", fn=replayable_identity, n_broadcast_samples=5
        )
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, label="value"))
        changed = Function(label="replayable_affine", fn=replayable_affine, n_broadcast_samples=5)

        with (
            pytest.raises(
                ReplayCompatibilityError,
                match=r"expected a call to .*replayable_identity, but .*replayable_affine",
            ),
            replay_run(original.provenance),
        ):
            changed(value=Normal(loc=0.0, scale=1.0, label="value"))

    def test_unsupported_current_callable_fails_before_sampling(self):
        workflow = Function(
            label="replayable_identity", fn=replayable_identity, n_broadcast_samples=5
        )
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, label="value"))
        changed = Function(label="function", fn=lambda value: value, n_broadcast_samples=5)
        candidate = Normal(loc=0.0, scale=1.0, label="value")

        with (
            patch.object(type(candidate), "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayUnsupportedCallableError, match="lambda"),
            replay_run(original.provenance),
        ):
            changed(value=candidate)

    def test_same_import_anchor_definition_drift_fails_before_sampling(self, monkeypatch):
        workflow = Function(
            label="replayable_identity", fn=replayable_identity, n_broadcast_samples=5
        )
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, label="value"))

        def changed_identity(value):
            return value + 1

        changed_identity.__module__ = _replay_fixtures.__name__
        changed_identity.__qualname__ = "replayable_identity"
        monkeypatch.setattr(
            _replay_fixtures,
            "replayable_identity",
            changed_identity,
        )
        changed = Function(label="changed_identity", fn=changed_identity, n_broadcast_samples=5)
        candidate = Normal(loc=0.0, scale=1.0, label="value")

        with (
            patch.object(type(candidate), "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(
                ReplayCompatibilityError, match="has changed since the call was recorded"
            ),
            replay_run(original.provenance),
        ):
            changed(value=candidate)

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            ({"output_label": "a"}, {"output_label": "b"}),
            ({"output_spec": OutputSpec(a=None)}, {"output_spec": OutputSpec(b=None)}),
            (
                {"output_label": "a", "output_spec": NumericArraySpec(())},
                {"output_label": "b", "output_spec": NumericArraySpec(())},
            ),
        ],
        ids=["output-label", "declared-component", "default-component"],
    )
    def test_a_renamed_output_replays_to_the_same_draws(self, before, after):
        """A rename changes no value, so the replay reproduces the draws under the new names."""

        def lifted(**options):
            return Function(
                label="replayable_identity",
                fn=replayable_identity,
                n_broadcast_samples=5,
                **options,
            )

        law = Normal(loc=0.0, scale=1.0, label="value")
        with workflow_run(seed=4):
            original = lifted(**before)(value=law)
        with replay_run(original.provenance):
            replayed = lifted(**after)(value=law)

        np.testing.assert_array_equal(_marginal_values(replayed), _marginal_values(original))
        assert list(replayed.event_spec.components) == ["b"]

    @pytest.mark.parametrize("change", ["shape", "kind", "packaging", "declaration"])
    def test_output_contract_drift_fails_before_sampling(self, change):
        record = RecordSpec(left=(), right=())
        declaration = OutputSpec(bundle=record)
        baseline = Function(
            "identity",
            replayable_identity,
            output_label="result",
            output_spec=declaration,
            dispatch="sequential",
            n_broadcast_samples=8,
        )
        law = Normal("left", 0.0, 1.0) * Normal("right", 0.0, 1.0)
        with workflow_run(seed=4):
            original = baseline(value=law)
        declarations = {
            "shape": OutputSpec(bundle=RecordSpec(left=(2,), right=())),
            "kind": OutputSpec(bundle=OpaqueSpec()),
            "packaging": OutputSpec(record),
            "declaration": None,
        }
        changed = Function(
            "identity",
            replayable_identity,
            output_label="result",
            output_spec=declarations[change],
            dispatch="sequential",
            n_broadcast_samples=8,
        )
        with (
            patch.object(type(law), "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(
                ReplayCompatibilityError, match="has changed since the call was recorded"
            ),
            replay_run(original.provenance),
        ):
            changed(value=law)

    def test_plan_drift_fails_before_distribution_sampling(self):
        workflow = Function(
            label="replayable_identity", fn=replayable_identity, n_broadcast_samples=5
        )
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, label="value"))
        changed = Function(
            label="replayable_identity", fn=replayable_identity, n_broadcast_samples=6
        )
        candidate = Normal(loc=0.0, scale=1.0, label="value")

        with (
            patch.object(type(candidate), "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayCompatibilityError, match="n_broadcast_samples is 6, recorded 5"),
            replay_run(original.provenance),
        ):
            changed(value=candidate)

    def test_direct_record_projection_drift_fails_before_key_derivation(self):
        original_root = Normal(loc=0.0, scale=1.0, label="x") * Normal(
            loc=2.0, scale=1.0, label="y"
        )
        with workflow_run(seed=4):
            original = sample(original_root["x"])
        assert original.provenance.controls["replay"]["plan"]["expected_effects"][0][
            "record_path"
        ] == ["x"]

        candidate_root = Normal(loc=0.0, scale=1.0, label="x") * Normal(
            loc=2.0, scale=1.0, label="y"
        )
        with (
            patch.object(type(candidate_root), "_sample", side_effect=AssertionError("sampled")),
            patch(
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ),
            pytest.raises(
                ReplayCompatibilityError,
                match=r"draw \(sample of 'y' with sample_shape=\(\)\) that matches no draw",
            ),
            replay_run(original.provenance),
        ):
            sample(candidate_root["y"])

    def test_route_drift_is_diagnostic_and_preserves_values(self):
        original_workflow = Function(
            label="replayable_identity",
            fn=replayable_identity,
            n_broadcast_samples=9,
            dispatch="sequential",
        )
        replay_workflow = Function(
            label="replayable_identity",
            fn=replayable_identity,
            n_broadcast_samples=9,
            dispatch="thread",
            max_workers=3,
        )
        with workflow_run(seed=31):
            original = original_workflow(value=Normal(loc=0.0, scale=1.0, label="value"))

        with replay_run(original.provenance):
            replayed = replay_workflow(value=Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_marginal_values(replayed), _marginal_values(original))
        assert replayed.provenance.diagnostics["replay"]["execution_drift"] is True

    def test_source_artifact_drift_is_diagnostic_only(self):
        original = _draw()

        def mutate(payload):
            payload["diagnostics"]["callable_source"]["source_artifact_digest"] = "0" * 64

        payload = original.provenance.to_dict()
        mutate(payload)
        changed = Provenance.from_dict(payload)

        with replay_run(changed):
            replayed = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))
        diagnostics = replayed.provenance.diagnostics["replay"]
        assert diagnostics["source_artifact_drift"] is True
        assert diagnostics["source_location_drift"] is False

    def test_source_location_drift_is_separate_from_artifact_drift(self):
        original = _draw()
        payload = original.provenance.to_dict()
        payload["diagnostics"]["callable_source"]["source_location"] = (
            "/relocated/probpipe/source.py"
        )
        changed = Provenance.from_dict(payload)

        with replay_run(changed):
            replayed = sample(Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))
        diagnostics = replayed.provenance.diagnostics["replay"]
        assert diagnostics["source_artifact_drift"] is False
        assert diagnostics["source_location_drift"] is True

    @pytest.mark.parametrize(
        "invalid_signature",
        ["not-a-signature", 1],
        ids=["value-error", "type-error"],
    )
    def test_invalid_custom_signature_fails_before_sampling(
        self,
        monkeypatch,
        invalid_signature,
    ):
        workflow = Function(
            label="replayable_identity", fn=replayable_identity, n_broadcast_samples=5
        )
        with workflow_run(seed=8):
            original = workflow(value=Normal(loc=0.0, scale=1.0, label="value"))

        monkeypatch.setattr(
            replayable_identity,
            "__signature__",
            invalid_signature,
            raising=False,
        )
        candidate = Normal(loc=0.0, scale=1.0, label="value")
        with (
            patch.object(type(candidate), "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(
                ReplayUnsupportedCallableError,
                match=r"signature, defaults, .* replay cannot record",
            ),
            replay_run(original.provenance),
        ):
            workflow(value=candidate)

    def test_recorded_sampling_abi_drift_fails_before_sampling(self):
        original = _draw()

        def mutate(controls):
            controls["replay"]["compatibility"]["sampling_abi"] = ["unknown-sampling/v99"]

        changed = _mutate_provenance(original.provenance, mutate)
        candidate = Normal(loc=0.0, scale=1.0, label="value")
        with (
            patch.object(type(candidate), "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(
                ReplayCompatibilityError, match=r"replay\.compatibility\.sampling_abi is"
            ),
            replay_run(changed),
        ):
            sample(candidate)

    def test_unknown_key_adapter_abi_fails_at_replay_entry(self):
        original = _draw()

        def mutate(controls):
            controls["replay"]["compatibility"]["key_adapter_abi"] = "unknown-key-adapter/v99"

        changed = _mutate_provenance(original.provenance, mutate)
        with (
            pytest.raises(
                ReplayCompatibilityError,
                match=r"replay\.compatibility\.key_adapter_abi 'unknown-key-adapter/v99'",
            ),
            replay_run(changed),
        ):
            pass

    def test_jax_to_rowwise_route_drift_preserves_values(self):
        original_workflow = Function(
            label="replayable_identity",
            fn=replayable_identity,
            n_broadcast_samples=7,
            dispatch="jax",
        )
        replay_workflow = Function(
            label="replayable_identity",
            fn=replayable_identity,
            n_broadcast_samples=7,
            dispatch="sequential",
        )
        with workflow_run(seed=41):
            original = original_workflow(value=Normal(loc=0.0, scale=1.0, label="value"))

        with replay_run(original.provenance):
            replayed = replay_workflow(value=Normal(loc=0.0, scale=1.0, label="value"))

        np.testing.assert_array_equal(_marginal_values(replayed), _marginal_values(original))
        assert replayed.provenance.diagnostics["replay"]["execution_drift"] is True


class TestReplayEventRegistry:
    @pytest.mark.parametrize("drift", ["identity", "effect"])
    def test_unexpected_identity_and_effect_drift_fail_before_sampling(self, drift):
        original = _draw()

        def mutate(controls):
            if drift == "identity":
                controls["randomness"]["events"][0]["source"] = [
                    "source-group",
                    99,
                ]
            else:
                controls["replay"]["plan"]["expected_effects"][0]["provider_abi"] = (
                    "unknown-provider/v99"
                )

        changed = _mutate_provenance(original.provenance, mutate)
        candidate = Normal(loc=0.0, scale=1.0, label="value")
        with (
            patch.object(type(candidate), "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(
                ReplayCompatibilityError,
                match=r"matches no draw|recorded call did not make|provider_abi is",
            ),
            replay_run(changed),
        ):
            sample(candidate)

    def test_duplicate_recorded_event_is_rejected_at_entry(self):
        original = _draw()

        def mutate(controls):
            controls["randomness"]["events"].append(
                copy.deepcopy(controls["randomness"]["events"][0])
            )
            controls["replay"]["plan"]["expected_effects"].append(
                copy.deepcopy(controls["replay"]["plan"]["expected_effects"][0])
            )
            controls["randomness"]["expected_event_count"] = 2

        changed = _mutate_provenance(original.provenance, mutate)
        with (
            pytest.raises(ReplayCompatibilityError, match="duplicate event"),
            replay_run(changed),
        ):
            pass

    def test_same_token_retry_is_idempotent_but_other_claims_fail(self):
        state = _replay._validate_provenance(_draw().provenance)
        effect = state.expected_events[0].managed_effect()
        token = ManagedWorkItemToken.create()
        first = ManagedAttemptState.create(token)
        retry = ManagedAttemptState.create(token)
        unclaimed = ManagedAttemptState.create(token)

        state.claim_effect(effect, attempt=first)
        state.claim_effect(effect, attempt=retry)
        with pytest.raises(
            ReplayCompatibilityError, match="1 random draw that this call did not make"
        ):
            state.assert_all_events_claimed()
        with pytest.raises(ReplayCompatibilityError, match="successful attempt"):
            state.mark_successful_effects((effect,), attempt=unclaimed)
        state.mark_successful_effects((effect,), attempt=retry)
        state.assert_all_events_claimed()
        claim = state.claims[state.expected_events[0].encoded_identity]
        assert claim.successful_attempt_token == retry.attempt_token

        with pytest.raises(ReplayCompatibilityError, match="already successful"):
            state.mark_successful_effects((effect,), attempt=first)

        with pytest.raises(ReplayCompatibilityError, match="duplicated"):
            state.claim_effect(effect, attempt=retry)
        with pytest.raises(ReplayCompatibilityError, match="different managed"):
            state.claim_effect(
                effect,
                attempt=ManagedAttemptState.create(ManagedWorkItemToken.create()),
            )

        direct_state = _replay._validate_provenance(_draw().provenance)
        direct_state.claim_effect(effect, attempt=None)
        with pytest.raises(ReplayCompatibilityError, match="directly claimed"):
            direct_state.mark_successful_effects((effect,), attempt=first)
        direct_state.mark_successful_effects((effect,), attempt=None)
        direct_state.assert_all_events_claimed()
        with pytest.raises(ReplayCompatibilityError, match="already successful"):
            direct_state.mark_successful_effects((effect,), attempt=None)
        with pytest.raises(ReplayCompatibilityError, match="duplicated"):
            direct_state.claim_effect(effect, attempt=None)

    def test_replay_claim_batch_is_atomic(self):
        state = _replay._validate_provenance(_draw().provenance)
        effect = state.expected_events[0].managed_effect()
        unexpected = copy.deepcopy(effect)
        object.__setattr__(unexpected, "stochastic_source_id", ("source-group", 99))
        attempt = ManagedAttemptState.create(ManagedWorkItemToken.create())

        with pytest.raises(ReplayCompatibilityError, match="that the recorded call did not make"):
            state._commit_effect_batch(
                (effect, unexpected),
                successful_effects=(),
                attempt=attempt,
            )

        claim = state.claims[state.expected_events[0].encoded_identity]
        assert claim.work_item_token is None
        assert claim.attempt_tokens == set()
        assert claim.successful_attempt_token is None

    def test_remote_replay_scope_requires_its_complete_namespace(self):
        state = _replay._validate_provenance(_draw().provenance)
        effect = state.expected_events[0].managed_effect()
        attempt = ManagedAttemptState.create(ManagedWorkItemToken.create())

        with (
            pytest.raises(
                ReplayCompatibilityError, match="1 random draw that this call did not make"
            ),
            _replay._remote_replay_claim_scope((effect,), attempt),
        ):
            pass

        with _replay._remote_replay_claim_scope((effect,), attempt):
            _replay._claim_effect_before_derivation(
                effect,
                attempt=attempt,
            )

    def test_remote_replay_scope_does_not_mask_worker_errors(self):
        state = _replay._validate_provenance(_draw().provenance)
        effect = state.expected_events[0].managed_effect()
        attempt = ManagedAttemptState.create(ManagedWorkItemToken.create())

        with (
            pytest.raises(ValueError, match="worker failed"),
            _replay._remote_replay_claim_scope((effect,), attempt),
        ):
            raise ValueError("worker failed")

    def test_plan_validation_uses_the_admission_index(self):
        state = _replay._validate_provenance(_draw().provenance)
        effect = state.expected_events[0].managed_effect()

        class IterationTrap(tuple):
            def __iter__(self):
                raise AssertionError("validate_plan rescanned expected events")

        state.expected_events = IterationTrap(state.expected_events)
        for _ in range(5):
            state.validate_effect_plan(_plan_for_effect(effect))

    def test_managed_namespace_index_handles_nested_prefixes(self):
        state = _replay._validate_provenance(_draw().provenance)
        original = state.expected_events[0]
        outer_path = original.occurrence_path
        outer_unit = (
            "managed-unit",
            "probpipe.managed_work_item/v1",
            "point",
            0,
        )
        nested_parent = (*outer_path, outer_unit, ("child", 0))
        nested_unit = (
            "managed-unit",
            "probpipe.managed_work_item/v1",
            "sweep-cell",
            2,
        )
        occurrence_path = (*nested_parent, nested_unit, ("child", 0))
        effect = replace(original.managed_effect(), occurrence_path=occurrence_path)
        expected = replace(
            original,
            occurrence_path=occurrence_path,
            encoded_identity=_replay._encoded_effect_identity(effect),
        )
        indexed = replace(state, expected_events=(expected,))

        class IterationTrap(tuple):
            def __iter__(self):
                raise AssertionError("managed lookup rescanned expected events")

        indexed.expected_events = IterationTrap(indexed.expected_events)
        assert indexed.expected_effects_for_unit(outer_path, outer_unit) == (effect,)
        assert indexed.expected_effects_for_unit(nested_parent, nested_unit) == (effect,)

    def test_remote_plan_validation_uses_its_namespace_index(self):
        state = _replay._validate_provenance(_draw().provenance)
        effect = state.expected_events[0].managed_effect()
        attempt = ManagedAttemptState.create(ManagedWorkItemToken.create())
        encoded = _replay._encoded_effect_identity(effect)
        registry = _replay._RemoteReplayClaims(
            expected_by_identity={encoded: effect},
            attempt=attempt,
        )

        class ValuesTrap(dict):
            def values(self):
                raise AssertionError("remote validation rescanned its namespace")

        registry.expected_by_identity = ValuesTrap(registry.expected_by_identity)
        for _ in range(5):
            registry.validate_plan(_plan_for_effect(effect))

    def test_nested_automatic_drift_in_thread_is_unexpected_before_sampling(
        self,
        monkeypatch,
    ):
        workflow = Function(
            label="replayable_optional_nested",
            fn=replayable_optional_nested,
            n_broadcast_samples=5,
            dispatch="thread",
            max_workers=2,
        )
        with workflow_run(seed=71):
            original = workflow(value=Normal(loc=0.0, scale=1.0, label="value"))
        monkeypatch.setattr(
            _replay_fixtures,
            "ENABLE_EXTRA_AUTOMATIC",
            True,
        )

        with (
            pytest.raises(ReplayCompatibilityError, match="matches no draw of the recorded call"),
            replay_run(original.provenance),
        ):
            workflow(value=Normal(loc=0.0, scale=1.0, label="value"))


def test_replay_provenance_inputs_are_not_mutated():
    original = _draw()
    before = copy.deepcopy(original.provenance.to_dict())

    with replay_run(original.provenance):
        sample(Normal(loc=jnp.asarray(0.0), scale=1.0, label="value"))

    assert original.provenance.to_dict() == before
