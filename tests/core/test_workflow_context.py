"""Behavior tests for workflow-owned random execution contexts."""

from __future__ import annotations

from unittest.mock import patch

import jax
import pytest

import probpipe
from probpipe import workflow_run
from probpipe.core._workflow_context import (
    _commit_stochastic_invocation,
    _ephemeral_workflow_run,
)


def _claim_key_words() -> tuple[int, int]:
    invocation = _commit_stochastic_invocation()
    key = invocation.key_for(
        stochastic_source_id=("source", 0),
        logical_unit_id=("singleton",),
    )
    return tuple(int(word) for word in jax.random.key_data(key))


class TestWorkflowRunBoundary:
    def test_workflow_run_is_public_and_validates_seed_on_entry(self):
        assert probpipe.workflow_run is workflow_run
        assert "workflow_run" in probpipe.__all__

        for seed in (0, 2**64 - 1, None):
            with workflow_run(seed):
                pass

        for seed in (True, False, -1, 2**64, 1.5, "7"):
            with pytest.raises((TypeError, ValueError)), workflow_run(seed):
                pass

    def test_empty_anonymous_run_does_not_read_os_entropy(self):
        with patch("probpipe.core._workflow_context.os.urandom") as urandom, workflow_run():
            pass

        urandom.assert_not_called()

    def test_anonymous_root_reads_exactly_eight_entropy_bytes_on_first_claim(self):
        with (
            patch(
                "probpipe.core._workflow_context.os.urandom",
                return_value=bytes.fromhex("0123456789abcdef"),
            ) as urandom,
            workflow_run(),
        ):
            _claim_key_words()
            _claim_key_words()

        urandom.assert_called_once_with(8)

    def test_independent_ephemeral_runs_receive_independent_roots(self):
        with patch(
            "probpipe.core._workflow_context.os.urandom",
            side_effect=[bytes(8), bytes.fromhex("0000000000000001")],
        ) as urandom:
            with _ephemeral_workflow_run():
                first = _claim_key_words()
            with _ephemeral_workflow_run():
                second = _claim_key_words()

        assert first != second
        assert urandom.call_count == 2


class TestWorkflowOccurrences:
    def test_seeded_run_reproduces_a_sequence_of_distinct_occurrences(self):
        def run() -> tuple[tuple[int, int], tuple[int, int]]:
            with workflow_run(seed=7):
                return _claim_key_words(), _claim_key_words()

        first = run()
        second = run()

        assert first == second
        assert first[0] != first[1]

    def test_empty_nested_scope_does_not_shift_outer_occurrences(self):
        with workflow_run(seed=7):
            baseline = (_claim_key_words(), _claim_key_words())

        with workflow_run(seed=7):
            first = _claim_key_words()
            with workflow_run():
                pass
            second = _claim_key_words()

        assert (first, second) == baseline

    def test_unseeded_nested_scope_is_reproducible_and_isolates_outer_siblings(self):
        def run(num_inner_claims: int):
            with workflow_run(seed=7):
                outer_first = _claim_key_words()
                with workflow_run():
                    inner = tuple(_claim_key_words() for _ in range(num_inner_claims))
                outer_second = _claim_key_words()
            return outer_first, inner, outer_second

        one_child = run(1)
        two_children = run(2)

        assert one_child == run(1)
        assert one_child[0] == two_children[0]
        assert one_child[2] == two_children[2]
        assert one_child[1][0] == two_children[1][0]
        assert one_child[1][0] not in (one_child[0], one_child[2])

    def test_same_seed_sibling_scopes_have_distinct_occurrence_paths(self):
        with workflow_run(seed=7):
            with workflow_run(seed=42):
                first = _claim_key_words()
            with workflow_run(seed=42):
                second = _claim_key_words()

        assert first != second

    def test_explicit_nested_seed_replaces_outer_root_but_keeps_scope_path(self):
        def inner_key(outer_seed: int) -> tuple[int, int]:
            with workflow_run(seed=outer_seed), workflow_run(seed=42):
                return _claim_key_words()

        assert inner_key(1) == inner_key(2)
