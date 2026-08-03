"""Standalone workflow RNG replay scope and preflight tests."""

from __future__ import annotations

import copy
import json
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    Normal,
    Provenance,
    ReplayCompatibilityError,
    ReplayUnsupportedCallableError,
    replay_run,
    sample,
    workflow_run,
)
from tests.core._workflow_replay_fixtures import (
    replayable_affine,
    replayable_identity,
)


def _draw(seed: int = 7):
    with workflow_run(seed=seed):
        return sample(Normal(loc=0.0, scale=1.0, name="value"))


def _sample_value(result):
    return np.asarray(result["sample"])


def _marginal_values(result):
    return np.asarray(result.samples["marginal"])


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

        payload = _draw().provenance.to_dict()
        payload["controls"]["randomness"]["root_words"] = [True, 1]
        with (
            pytest.raises(ReplayCompatibilityError, match="root_words"),
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
            replay_run(original.provenance),
            pytest.raises(ReplayCompatibilityError, match="callable"),
        ):
            changed(value=Normal(loc=0.0, scale=1.0, name="value"))

    def test_plan_drift_fails_before_distribution_sampling(self):
        workflow = Function(func=replayable_identity, n_broadcast_samples=5)
        with workflow_run(seed=4):
            original = workflow(value=Normal(loc=0.0, scale=1.0, name="value"))
        changed = Function(func=replayable_identity, n_broadcast_samples=6)
        candidate = Normal(loc=0.0, scale=1.0, name="value")

        with (
            patch.object(candidate, "_sample", side_effect=AssertionError("sampled")),
            replay_run(original.provenance),
            pytest.raises(ReplayCompatibilityError, match="stochastic plan"),
        ):
            changed(value=candidate)

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


def test_replay_provenance_inputs_are_not_mutated():
    original = _draw()
    before = copy.deepcopy(original.provenance.to_dict())

    with replay_run(original.provenance):
        sample(Normal(loc=jnp.asarray(0.0), scale=1.0, name="value"))

    assert original.provenance.to_dict() == before
