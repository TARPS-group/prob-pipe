"""Standalone workflow RNG replay scope and preflight tests."""

from __future__ import annotations

import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import bijectors as tfb

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    ProductDistribution,
    Provenance,
    ReplayCompatibilityError,
    ReplayUnsupportedCallableError,
    TransformedDistribution,
    UnmanagedConcurrentWorkflowEntryError,
    replay_run,
    sample,
    workflow_run,
)
from probpipe.core import _workflow_replay
from probpipe.core._workflow_managed import (
    ManagedAttemptState,
    ManagedWorkItemToken,
)
from tests.core import _workflow_replay_fixtures
from tests.core._workflow_replay_fixtures import (
    replayable_affine,
    replayable_difference,
    replayable_identity,
    replayable_optional_nested,
)


def _draw(seed: int = 7):
    with workflow_run(seed=seed):
        return sample(Normal(loc=0.0, scale=1.0, name="value"))


def _sample_value(result):
    return np.asarray(result["sample"])


def _marginal_values(result):
    return np.asarray(result.samples["marginal"])


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


class TestReplayScope:
    def test_seeded_serialized_and_replay_of_replay_roundtrip(self):
        original = _draw(seed=17)
        restored = Provenance.from_dict(json.loads(json.dumps(original.provenance.to_dict())))

        with replay_run(restored):
            first = sample(Normal(loc=0.0, scale=1.0, name="value"))
        with replay_run(first.provenance):
            second = sample(Normal(loc=0.0, scale=1.0, name="value"))

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
            "probpipe.core._workflow_context._os_urandom",
            return_value=entropy,
        ) as urandom:
            if explicit_scope:
                with workflow_run():
                    original = sample(Normal(loc=0.0, scale=1.0, name="value"))
            else:
                original = sample(Normal(loc=0.0, scale=1.0, name="value"))
        assert urandom.call_count == 1

        with (
            patch(
                "probpipe.core._workflow_context._os_urandom",
                side_effect=AssertionError("replay must use recorded root words"),
            ),
            replay_run(original.provenance),
        ):
            replayed = sample(Normal(loc=0.0, scale=1.0, name="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))

    def test_nested_managed_result_has_a_standalone_replay_recipe(self):
        captured = []

        def outer(value):
            result = sample(Normal(loc=value, scale=1.0, name="value"))
            captured.append(result)
            return result["sample"]

        workflow = Function(func=outer, dispatch="sequential")
        with workflow_run(seed=17):
            workflow(value=0.0)
        nested = captured[0]

        with replay_run(nested.provenance):
            replayed = sample(Normal(loc=0.0, scale=1.0, name="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(nested))

    def test_empty_second_root_and_apply_are_rejected(self):
        original = _draw()

        with (
            pytest.raises(ReplayCompatibilityError, match="exactly one"),
            replay_run(original.provenance),
        ):
            pass

        with replay_run(original.provenance):
            sample(Normal(loc=0.0, scale=1.0, name="value"))
            with pytest.raises(ReplayCompatibilityError, match="one top-level"):
                sample(Normal(loc=0.0, scale=1.0, name="value"))

        with replay_run(original.provenance):
            sample(Normal(loc=0.0, scale=1.0, name="value"))
            with pytest.raises(ReplayCompatibilityError, match="apply"):
                sample.apply(Normal(loc=0.0, scale=1.0, name="value"))

    def test_workflow_run_cannot_replace_the_replay_root(self):
        original = _draw()

        with replay_run(original.provenance):
            with (
                pytest.raises(ReplayCompatibilityError, match="cannot be nested"),
                workflow_run(seed=999),
            ):
                pass
            sample(Normal(loc=0.0, scale=1.0, name="value"))

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
            sample(Normal(loc=0.0, scale=1.0, name="value"))

        with (
            workflow_run(seed=9),
            pytest.raises(ReplayCompatibilityError, match="outside an active workflow_run"),
            replay_run(original.provenance),
        ):
            pass

    def test_caught_failed_root_is_rejected_when_scope_exits(self):
        original = _draw()
        changed = Function(func=replayable_affine, n_broadcast_samples=5)

        with (
            pytest.raises(ReplayCompatibilityError, match="did not complete"),
            replay_run(original.provenance),
            pytest.raises(ReplayCompatibilityError, match="callable"),
        ):
            changed(value=Normal(loc=0.0, scale=1.0, name="value"))


class TestReplayOwnership:
    def test_rejected_copied_thread_context_does_not_consume_owner_root(self):
        original = _draw()

        with replay_run(original.provenance):
            copied = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    copied.run,
                    sample,
                    Normal(loc=0.0, scale=1.0, name="value"),
                )
                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    future.result()
            replayed = sample(Normal(loc=0.0, scale=1.0, name="value"))

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))

    def test_rejected_copied_task_context_does_not_consume_owner_root(self):
        original = _draw()

        async def run_replay():
            with replay_run(original.provenance):

                async def call_in_child():
                    return sample(Normal(loc=0.0, scale=1.0, name="value"))

                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    await asyncio.create_task(call_in_child())
                return sample(Normal(loc=0.0, scale=1.0, name="value"))

        replayed = asyncio.run(run_replay())

        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))


class TestReplayAdmission:
    def test_legacy_unknown_and_malformed_recipes_fail_at_entry(self):
        with (
            pytest.raises(ReplayCompatibilityError, match="RNG recipe"),
            replay_run(Provenance("legacy")),
        ):
            pass

        payload = _draw().provenance.to_dict()
        payload["controls"]["randomness"]["rng_abi"] = "unknown-rng/v99"
        with (
            pytest.raises(ReplayCompatibilityError, match="RNG ABI"),
            replay_run(Provenance.from_dict(payload)),
        ):
            pass

    @pytest.mark.parametrize(
        ("path", "value", "match"),
        [
            pytest.param(
                ("randomness", "schema"),
                "unknown-recipe/v99",
                "RNG recipe schema",
                id="rng-recipe-schema",
            ),
            pytest.param(
                ("replay", "schema"),
                "unknown-replay/v99",
                "replay anchor schema",
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
                "callable definition ABI",
                id="callable-definition-abi",
            ),
            pytest.param(
                ("replay", "callable", "probpipe_replay_abi"),
                "unknown-probpipe/v99",
                "ProbPipe replay ABI",
                id="probpipe-replay-abi",
            ),
            pytest.param(
                ("replay", "callable", "module"),
                None,
                "invalid module",
                id="callable-module",
            ),
            pytest.param(
                ("replay", "callable", "signature_and_templates"),
                [],
                "signature_and_templates",
                id="callable-signature",
            ),
            pytest.param(
                ("replay", "plan", "schema"),
                "unknown-plan/v99",
                "stochastic plan ABI",
                id="plan-schema",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "managed_child_policy"),
                "unknown-managed-child/v99",
                "managed-child",
                id="managed-child-policy",
            ),
            pytest.param(
                ("replay", "plan", "canonical_fields", "key_ownership"),
                "caller",
                "not workflow-key-owned",
                id="plan-key-ownership",
            ),
            pytest.param(
                ("randomness", "expected_event_count"),
                True,
                "event count",
                id="event-count-bool",
            ),
            pytest.param(
                ("replay", "compatibility", "provider_abi"),
                _DELETE,
                "compatibility fields",
                id="compatibility-fields",
            ),
            pytest.param(
                ("replay", "compatibility", "execution_contract"),
                "unknown-execution/v99",
                "execution contract",
                id="execution-contract",
            ),
            pytest.param(
                ("replay", "compatibility", "descendant_adapter_abi"),
                ["unknown-descendant/v99"],
                "descendant-adapter ABI",
                id="descendant-adapter-abi",
            ),
            pytest.param(
                ("replay", "compatibility", "sampling_abi"),
                [""],
                "sampling ABI",
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
                "outside its anchored occurrence_path",
                id="event-outside-anchor",
            ),
            pytest.param(
                ("randomness", "events", 0, "occurrence_kind"),
                "child",
                "occurrence kind",
                id="event-occurrence-kind",
            ),
            pytest.param(
                ("randomness", "events", 0, "key_ownership"),
                "caller",
                "not workflow-key-owned",
                id="event-key-ownership",
            ),
            pytest.param(
                ("randomness", "events", 0, "source"),
                {},
                "must be a JSON sequence",
                id="event-source-sequence",
            ),
            pytest.param(
                ("randomness", "events", 0, "source"),
                ["source-group", True],
                "invalid structural value",
                id="event-source-value",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "provider_abi"),
                _DELETE,
                "incompatible fields",
                id="effect-fields",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "operation_kind"),
                "",
                "invalid operation_kind",
                id="effect-operation-kind",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "sample_shape"),
                [-1],
                "invalid sample_shape",
                id="effect-sample-shape",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "record_path"),
                [1],
                "invalid record_path",
                id="effect-record-path",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "descendant_descriptor"),
                {},
                "invalid descendant_descriptor",
                id="effect-descriptor-sequence",
            ),
            pytest.param(
                ("replay", "plan", "expected_effects", 0, "descendant_descriptor"),
                [{}],
                "invalid descendant_descriptor",
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
                "probpipe.core._workflow_context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ) as derive_key,
            pytest.raises(ReplayCompatibilityError, match=match),
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
            replayed = sample(Normal(loc=0.0, scale=1.0, name="value"))

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
        unsupported = Function(func=lambda value: value, n_broadcast_samples=5)
        with workflow_run(seed=3):
            unsupported_result = unsupported(value=Normal(loc=0.0, scale=1.0, name="value"))

        with (
            pytest.raises(ReplayUnsupportedCallableError, match="lambda"),
            replay_run(unsupported_result.provenance),
        ):
            pass

        inner = Function(
            func=lambda value: sample(Normal(loc=value, scale=1.0, name="inner"))["sample"],
            dispatch="sequential",
        )

        def nested(value):
            return inner(value=value)["<lambda>"]

        outer = Function(func=nested, dispatch="sequential")
        with workflow_run(seed=8):
            nested_result = outer(value=1.0)

        with (
            pytest.raises(ReplayCompatibilityError, match="nested automatic"),
            replay_run(nested_result.provenance),
        ):
            pass


class TestReplayPreflight:
    def test_callable_drift_fails_before_sampling(self):
        workflow = Function(func=replayable_identity, n_broadcast_samples=5)
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))
        changed = Function(func=replayable_affine, n_broadcast_samples=5)

        with (
            pytest.raises(ReplayCompatibilityError, match="callable"),
            replay_run(original.provenance),
        ):
            changed(value=Normal(loc=0.0, scale=1.0, name="value"))

    def test_unsupported_current_callable_fails_before_sampling(self):
        workflow = Function(func=replayable_identity, n_broadcast_samples=5)
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))
        changed = Function(func=lambda value: value, n_broadcast_samples=5)
        candidate = Normal(loc=0.0, scale=1.0, name="value")

        with (
            patch.object(candidate, "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayUnsupportedCallableError, match="lambda"),
            replay_run(original.provenance),
        ):
            changed(value=candidate)

    def test_same_import_anchor_definition_drift_fails_before_sampling(self, monkeypatch):
        workflow = Function(func=replayable_identity, n_broadcast_samples=5)
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        def changed_identity(value):
            return value + 1

        changed_identity.__module__ = _workflow_replay_fixtures.__name__
        changed_identity.__qualname__ = "replayable_identity"
        monkeypatch.setattr(
            _workflow_replay_fixtures,
            "replayable_identity",
            changed_identity,
        )
        changed = Function(func=changed_identity, n_broadcast_samples=5)
        candidate = Normal(loc=0.0, scale=1.0, name="value")

        with (
            patch.object(candidate, "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayCompatibilityError, match="definition changed"),
            replay_run(original.provenance),
        ):
            changed(value=candidate)

    def test_plan_drift_fails_before_distribution_sampling(self):
        workflow = Function(func=replayable_identity, n_broadcast_samples=5)
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))
        changed = Function(func=replayable_identity, n_broadcast_samples=6)
        candidate = Normal(loc=0.0, scale=1.0, name="value")

        with (
            patch.object(candidate, "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayCompatibilityError, match="stochastic plan"),
            replay_run(original.provenance),
        ):
            changed(value=candidate)

    def test_direct_descendant_drift_fails_before_key_derivation(self):
        original_dist = TransformedDistribution(
            Normal(loc=0.0, scale=1.0, name="root"),
            tfb.Exp(),
        )
        with workflow_run(seed=4):
            original = sample(original_dist)

        with replay_run(original.provenance):
            replayed = sample(
                TransformedDistribution(
                    Normal(loc=0.0, scale=1.0, name="root"),
                    tfb.Exp(),
                )
            )
        np.testing.assert_array_equal(_sample_value(replayed), _sample_value(original))

        candidate_root = Normal(loc=0.0, scale=1.0, name="root")
        candidate = TransformedDistribution(candidate_root, tfb.Square())
        with (
            patch.object(candidate_root, "_sample", side_effect=AssertionError("sampled")),
            patch(
                "probpipe.core._workflow_context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ),
            pytest.raises(ReplayCompatibilityError, match="stochastic effect plan"),
            replay_run(original.provenance),
        ):
            sample(candidate)

    def test_direct_record_projection_drift_fails_before_key_derivation(self):
        original_root = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=2.0, scale=1.0, name="y"),
        )
        with workflow_run(seed=4):
            original = sample(original_root["x"])
        assert original.provenance.controls["replay"]["plan"]["expected_effects"][0][
            "record_path"
        ] == ["x"]

        candidate_root = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=2.0, scale=1.0, name="y"),
        )
        with (
            patch.object(candidate_root, "_sample", side_effect=AssertionError("sampled")),
            patch(
                "probpipe.core._workflow_context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key"),
            ),
            pytest.raises(ReplayCompatibilityError, match="stochastic effect plan"),
            replay_run(original.provenance),
        ):
            sample(candidate_root["y"])

    def test_route_drift_is_diagnostic_and_preserves_values(self):
        original_workflow = Function(
            func=replayable_identity,
            n_broadcast_samples=9,
            dispatch="sequential",
        )
        replay_workflow = Function(
            func=replayable_identity,
            n_broadcast_samples=9,
            dispatch="thread",
            max_workers=3,
        )
        with workflow_run(seed=31):
            original = original_workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        with replay_run(original.provenance):
            replayed = replay_workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

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
            replayed = sample(Normal(loc=0.0, scale=1.0, name="value"))

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
            replayed = sample(Normal(loc=0.0, scale=1.0, name="value"))

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
        workflow = Function(func=replayable_identity, n_broadcast_samples=5)
        with workflow_run(seed=8):
            original = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        monkeypatch.setattr(
            replayable_identity,
            "__signature__",
            invalid_signature,
            raising=False,
        )
        candidate = Normal(loc=0.0, scale=1.0, name="value")
        with (
            patch.object(candidate, "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayUnsupportedCallableError, match="module-level"),
            replay_run(original.provenance),
        ):
            workflow(value=candidate)

    def test_recorded_sampling_abi_drift_fails_before_sampling(self):
        original = _draw()

        def mutate(controls):
            controls["replay"]["compatibility"]["sampling_abi"] = ["unknown-sampling/v99"]

        changed = _mutate_provenance(original.provenance, mutate)
        candidate = Normal(loc=0.0, scale=1.0, name="value")
        with (
            patch.object(candidate, "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayCompatibilityError, match="sampling ABI"),
            replay_run(changed),
        ):
            sample(candidate)

    def test_unknown_key_adapter_abi_fails_at_replay_entry(self):
        original = _draw()

        def mutate(controls):
            controls["replay"]["compatibility"]["key_adapter_abi"] = "unknown-key-adapter/v99"

        changed = _mutate_provenance(original.provenance, mutate)
        with (
            pytest.raises(ReplayCompatibilityError, match="key-adapter ABI"),
            replay_run(changed),
        ):
            pass

    def test_jax_to_rowwise_route_drift_preserves_values(self):
        original_workflow = Function(
            func=replayable_identity,
            n_broadcast_samples=7,
            dispatch="jax",
        )
        replay_workflow = Function(
            func=replayable_identity,
            n_broadcast_samples=7,
            dispatch="sequential",
        )
        with workflow_run(seed=41):
            original = original_workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        with replay_run(original.provenance):
            replayed = replay_workflow(value=Normal(loc=0.0, scale=1.0, name="value"))

        np.testing.assert_array_equal(_marginal_values(replayed), _marginal_values(original))
        assert replayed.provenance.diagnostics["replay"]["execution_drift"] is True


class TestReplayEventRegistry:
    def test_missing_event_is_reported_at_root_completion(self):
        original = _draw()

        with (
            pytest.raises(ReplayCompatibilityError, match="missing expected"),
            replay_run(original.provenance),
        ):
            sample(
                Normal(loc=0.0, scale=1.0, name="value"),
                key=jax.random.key(9),
            )

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
        candidate = Normal(loc=0.0, scale=1.0, name="value")
        with (
            patch.object(candidate, "_sample", side_effect=AssertionError("sampled")),
            pytest.raises(ReplayCompatibilityError, match=r"unexpected|provider ABI"),
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
        state = _workflow_replay._validate_provenance(_draw().provenance)
        effect = state.expected_events[0].managed_effect()
        token = ManagedWorkItemToken.create()
        first = ManagedAttemptState.create(token)
        retry = ManagedAttemptState.create(token)

        state.claim_effect(effect, attempt=first)
        state.claim_effect(effect, attempt=retry)
        with pytest.raises(ReplayCompatibilityError, match="missing expected"):
            state.assert_all_events_claimed()
        state.mark_successful_effects((effect,))
        state.assert_all_events_claimed()

        with pytest.raises(ReplayCompatibilityError, match="duplicated"):
            state.claim_effect(effect, attempt=retry)
        with pytest.raises(ReplayCompatibilityError, match="different managed"):
            state.claim_effect(
                effect,
                attempt=ManagedAttemptState.create(ManagedWorkItemToken.create()),
            )

        direct_state = _workflow_replay._validate_provenance(_draw().provenance)
        direct_state.claim_effect(effect, attempt=None)
        with pytest.raises(ReplayCompatibilityError, match="duplicated"):
            direct_state.claim_effect(effect, attempt=None)

    def test_nested_automatic_drift_in_thread_is_unexpected_before_sampling(
        self,
        monkeypatch,
    ):
        workflow = Function(
            func=replayable_optional_nested,
            n_broadcast_samples=5,
            dispatch="thread",
            max_workers=2,
        )
        with workflow_run(seed=71):
            original = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))
        monkeypatch.setattr(
            _workflow_replay_fixtures,
            "ENABLE_EXTRA_AUTOMATIC",
            True,
        )

        with (
            pytest.raises(ReplayCompatibilityError, match="unexpected replay event"),
            replay_run(original.provenance),
        ):
            workflow(value=Normal(loc=0.0, scale=1.0, name="value"))


class TestReplayCoSamplingPlans:
    @pytest.mark.parametrize("kind", ["alias", "record_view", "transform", "mixed"])
    def test_supported_joint_plans_roundtrip(self, kind):
        workflow = Function(
            func=replayable_difference,
            n_broadcast_samples=8,
            dispatch="sequential",
        )

        def values():
            if kind == "alias":
                root = Normal(loc=0.0, scale=1.0, name="root")
                return {"left": root, "right": root}
            if kind == "record_view":
                root = ProductDistribution(
                    x=Normal(loc=0.0, scale=1.0, name="x"),
                    y=Normal(loc=2.0, scale=1.0, name="y"),
                )
                return {"left": root["x"], "right": root["y"]}
            if kind == "transform":
                root = Normal(loc=0.0, scale=1.0, name="root")
                return {
                    "left": root,
                    "right": TransformedDistribution(root, tfb.Exp()),
                }
            return {
                "left": EmpiricalDistribution(
                    jnp.asarray([1.0, 3.0]),
                    weights=jnp.asarray([0.25, 0.75]),
                    name="left",
                ),
                "right": Normal(loc=0.0, scale=1.0, name="right"),
            }

        with workflow_run(seed=53):
            original = workflow(**values())
        with replay_run(original.provenance):
            replayed = workflow(**values())

        np.testing.assert_array_equal(_marginal_values(replayed), _marginal_values(original))


def test_replay_provenance_inputs_are_not_mutated():
    original = _draw()
    before = copy.deepcopy(original.provenance.to_dict())

    with replay_run(original.provenance):
        sample(Normal(loc=jnp.asarray(0.0), scale=1.0, name="value"))

    assert original.provenance.to_dict() == before
