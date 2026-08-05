"""Behavior tests for workflow-owned random execution contexts."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from unittest.mock import patch

import jax
import pytest

import probpipe
import probpipe.core._workflow_rng as workflow_rng
from probpipe import (
    Function,
    Normal,
    UnmanagedConcurrentWorkflowEntryError,
    sample,
    workflow_run,
)
from probpipe.core import _workflow_context
from probpipe.core._workflow_context import (
    _commit_stochastic_invocation,
    _ephemeral_workflow_run,
    _StochasticProbeSignal,
    _workflow_probe,
)


def _claim_key_words() -> tuple[int, int]:
    invocation = _commit_stochastic_invocation()
    key = invocation.key_for(
        stochastic_source_id=("source", 0),
        logical_unit_id=("singleton",),
    )
    return tuple(int(word) for word in jax.random.key_data(key))


def _identity(value):
    return value


def _nested_draw(value):
    return sample(Normal(loc=value, scale=1.0, name="draw"))


def _nested_seeded_draw(value):
    with workflow_run(seed=42):
        return sample(Normal(loc=value, scale=1.0, name="draw"))["sample"]


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
        with patch("probpipe.core._workflow_context._os_urandom") as urandom, workflow_run():
            pass

        urandom.assert_not_called()

    def test_anonymous_root_reads_exactly_eight_entropy_bytes_on_first_claim(self):
        with (
            patch(
                "probpipe.core._workflow_context._os_urandom",
                return_value=bytes.fromhex("0123456789abcdef"),
            ) as urandom,
            workflow_run(),
        ):
            _claim_key_words()
            _claim_key_words()

        urandom.assert_called_once_with(8)

    def test_independent_ephemeral_runs_receive_independent_roots(self):
        with patch(
            "probpipe.core._workflow_context._os_urandom",
            side_effect=[bytes(8), bytes.fromhex("0000000000000001")],
        ) as urandom:
            with _ephemeral_workflow_run():
                first = _claim_key_words()
            with _ephemeral_workflow_run():
                second = _claim_key_words()

        assert first != second
        assert urandom.call_count == 2

    def test_cache_miss_encodes_an_event_identity_once(self):
        original_encode = workflow_rng.encode_random_event
        with (
            patch.object(
                _workflow_context,
                "encode_random_event",
                wraps=original_encode,
            ) as callsite_encode,
            patch.object(
                workflow_rng,
                "encode_random_event",
                wraps=original_encode,
            ) as derivation_encode,
            workflow_run(seed=7),
        ):
            _claim_key_words()

        callsite_encode.assert_called_once()
        derivation_encode.assert_not_called()


class TestWorkflowAdmission:
    def test_owner_records_process_and_object_identities(self):
        owner = _workflow_context._current_workflow_owner()

        assert owner.process_id == os.getpid()
        assert owner.thread_ref() is threading.current_thread()
        assert owner.task_ref is None

        async def assert_task_identity():
            task_owner = _workflow_context._current_workflow_owner()
            assert task_owner.task_ref is not None
            assert task_owner.task_ref() is asyncio.current_task()

        asyncio.run(assert_task_identity())

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
    def test_forked_child_cannot_enter_an_inherited_workflow_frame(self):
        script = textwrap.dedent(
            """
            import os
            import signal
            import threading

            from probpipe import UnmanagedConcurrentWorkflowEntryError, workflow_run
            from probpipe.core import _workflow_context

            with workflow_run(seed=7):
                frame = _workflow_context._capture_active_workflow_frame()
                lock_held = threading.Event()
                release_lock = threading.Event()

                def hold_frame_lock():
                    with frame.state.lock:
                        lock_held.set()
                        release_lock.wait(timeout=10)

                holder = threading.Thread(target=hold_frame_lock)
                holder.start()
                if not lock_held.wait(timeout=10):
                    raise RuntimeError("failed to hold the workflow frame lock")

                child_pid = os.fork()
                if child_pid == 0:
                    signal.alarm(5)
                    try:
                        _workflow_context._assert_workflow_admission()
                    except UnmanagedConcurrentWorkflowEntryError:
                        os._exit(0)
                    except BaseException:
                        os._exit(2)
                    else:
                        os._exit(1)

                _, status = os.waitpid(child_pid, 0)
                release_lock.set()
                holder.join(timeout=10)
                child_status = os.waitstatus_to_exitcode(status)
                if child_status != 0:
                    raise SystemExit(child_status)
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert completed.returncode == 0, completed.stderr

    def test_unmanaged_copied_thread_context_fails_before_execution_or_entropy(self):
        called = False

        def track_call(value):
            nonlocal called
            called = True
            return value

        workflow = Function(func=track_call)
        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            workflow_run(),
        ):
            copied = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(copied.run, workflow, 1)
                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    future.result()

        assert called is False
        urandom.assert_not_called()

    def test_unmanaged_copied_async_task_context_is_rejected(self):
        async def run_child():
            workflow = Function(func=_identity)
            with workflow_run(seed=7):

                async def call_workflow():
                    return workflow(1)

                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    await asyncio.create_task(call_workflow())

        asyncio.run(run_child())

    def test_copied_context_cannot_reenter_a_closed_workflow_frame(self):
        with (
            patch(
                "probpipe.core._workflow_context._os_urandom",
                return_value=bytes(8),
            ) as urandom,
            workflow_run(),
        ):
            copied = copy_context()

        with pytest.raises(
            UnmanagedConcurrentWorkflowEntryError,
            match="already closed",
        ):
            copied.run(_claim_key_words)

        urandom.assert_not_called()

    def test_unpropagated_thread_receives_an_independent_ephemeral_run(self):
        entropy = bytes.fromhex("0123456789abcdef")
        with (
            patch(
                "probpipe.core._workflow_context._os_urandom",
                return_value=entropy,
            ) as urandom,
            workflow_run(seed=7),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):

            def claim_in_fresh_context():
                with _ephemeral_workflow_run():
                    return _claim_key_words()

            words = pool.submit(claim_in_fresh_context).result()

        assert words
        urandom.assert_called_once_with(8)

    def test_managed_thread_work_item_can_enter_the_parent_run(self):
        def run_once():
            workflow = Function(func=_nested_draw, dispatch="thread")
            with workflow_run(seed=7):
                return float(workflow(1.0)["sample"])

        assert run_once() == run_once()

    def test_concurrent_entry_error_is_public(self):
        assert probpipe.UnmanagedConcurrentWorkflowEntryError is (
            UnmanagedConcurrentWorkflowEntryError
        )
        assert "UnmanagedConcurrentWorkflowEntryError" in probpipe.__all__

    def test_transported_root_is_not_reported_as_a_user_seed(self):
        with _workflow_context._transported_workflow_frame((1, 2)):
            frame = _workflow_context._capture_active_workflow_frame()
            assert frame is not None
            origin = _workflow_context._describe_rng_origin(frame)

        assert origin == {
            "context_kind": "transported_run",
            "root_source": "transported_authority",
            "supplied_seed": None,
        }


class TestWorkflowOccurrences:
    def test_seeded_run_reproduces_a_sequence_of_distinct_occurrences(self):
        def run() -> tuple[tuple[int, int], tuple[int, int]]:
            with workflow_run(seed=7):
                return _claim_key_words(), _claim_key_words()

        first = run()
        second = run()

        assert first == second
        assert first[0] != first[1]

    def test_changing_seed_changes_each_structural_occurrence(self):
        def run(seed: int) -> tuple[tuple[int, int], ...]:
            with workflow_run(seed=seed):
                return tuple(_claim_key_words() for _ in range(3))

        first = run(7)
        second = run(8)

        assert all(left != right for left, right in zip(first, second, strict=True))

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

    @pytest.mark.parametrize("dispatch", ["sequential", "thread"])
    def test_explicit_nested_seed_inside_managed_function_replaces_outer_root(
        self,
        dispatch,
    ):
        workflow = Function(func=_nested_seeded_draw, dispatch=dispatch)
        occurrence_paths = []
        original_key_for = _workflow_context._WorkflowInvocation.key_for

        def recording_key_for(
            invocation,
            *,
            stochastic_source_id,
            logical_unit_id,
        ):
            occurrence_paths.append(invocation.occurrence_path)
            return original_key_for(
                invocation,
                stochastic_source_id=stochastic_source_id,
                logical_unit_id=logical_unit_id,
            )

        values = []
        with patch.object(
            _workflow_context._WorkflowInvocation,
            "key_for",
            new=recording_key_for,
        ):
            for outer_seed in (1, 2):
                with workflow_run(seed=outer_seed):
                    values.append(float(workflow(0.0)["_nested_seeded_draw"]))

        assert values[0] == values[1]
        assert len(occurrence_paths) == 2
        assert occurrence_paths[0] == occurrence_paths[1]
        assert any(
            isinstance(segment, tuple) and segment[:1] == ("scope",)
            for segment in occurrence_paths[0]
        )

    def test_probe_attempt_does_not_commit_or_shift_an_occurrence(self):
        with workflow_run(seed=7):
            baseline = (_claim_key_words(), _claim_key_words())

        with workflow_run(seed=7):
            first = _claim_key_words()
            with pytest.raises(_StochasticProbeSignal), _workflow_probe():
                _claim_key_words()
            second = _claim_key_words()

        assert (first, second) == baseline
