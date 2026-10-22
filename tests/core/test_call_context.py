"""Tests for the call context: how a call, a ``config`` block, and a worker derive, install, and restore it."""

import contextvars
import pickle
import threading
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from probpipe import Function, NumericArrayBatch, config, function
from probpipe.core._call_context import (
    _collecting_call_accumulators,
    _note_num_samples_used,
    _NumSamplesUsed,
    _under_call_context,
    current_call_context,
)
from probpipe.core.config import _num_samples_in_effect
from probpipe.functions import _function as function_engine


def _held_open_block():
    """A generator that holds a ``config.sampling`` block open across its first yield."""
    with config.sampling.using(num_samples=7):
        yield


class TestDerivingAContext:
    def test_outside_every_call_and_block_the_context_is_the_default(self):
        call_context = current_call_context()
        assert call_context.is_default()
        assert call_context.call_depth == 0
        assert call_context.num_samples_used is None
        assert dict(call_context.resolved_controls) == {}

    def test_a_block_derives_a_context_that_merges_over_the_enclosing_blocks(self):
        with config.sampling.using(num_samples=7, max_evaluations=3):
            with config.using(sampling={"num_samples": 9}, notation={"max_depth": 4}):
                blocks = current_call_context().config_blocks
                assert dict(blocks["sampling"]) == {"num_samples": 9, "max_evaluations": 3}
                assert dict(blocks["notation"]) == {"max_depth": 4}
                assert isinstance(blocks["sampling"], MappingProxyType)
            assert dict(current_call_context().config_blocks["sampling"]) == {
                "num_samples": 7,
                "max_evaluations": 3,
            }
            assert not current_call_context().is_default()
        assert current_call_context().is_default()

    def test_a_child_keeps_a_passed_exact_only_and_drops_the_passed_counts(self):
        parent = current_call_context().with_passed_controls(
            {"exact_only": True, "max_evaluations": 5, "num_samples": 6}
        )
        child = parent.child_for_call(
            entry_method_name="__call__", function_label="f", resolved_controls={"num_samples": 6}
        )
        assert dict(child.passed_controls) == {
            "exact_only": True,
            "max_evaluations": None,
            "num_samples": None,
        }
        assert child.call_depth == 1
        assert child.is_root_call()
        assert child.function_label == "f"
        grandchild = child.child_for_call(
            entry_method_name="apply", function_label="g", resolved_controls={}
        )
        assert grandchild.call_depth == 2
        assert not grandchild.is_root_call()

    def test_a_child_without_a_passed_exact_only_takes_no_passed_controls(self):
        parent = current_call_context().with_passed_controls(
            {"exact_only": False, "max_evaluations": None, "num_samples": 6}
        )
        child = parent.child_for_call(
            entry_method_name="__call__", function_label="f", resolved_controls={}
        )
        assert child.passed_controls is None

    @pytest.mark.parametrize(
        ("entry_method_name", "keeps_parents"),
        [("__call__", True), ("apply", True), ("check", False)],
    )
    def test_a_child_shares_its_parents_num_samples_used_and_a_check_takes_none(
        self, entry_method_name, keeps_parents
    ):
        parent = current_call_context().with_fresh_call_accumulators()
        child = parent.child_for_call(
            entry_method_name=entry_method_name, function_label="f", resolved_controls={}
        )
        assert (child.num_samples_used is parent.num_samples_used) is keeps_parents
        assert (child.num_samples_used is None) is not keeps_parents


class TestCallsOpenAContext:
    def test_a_call_runs_in_a_child_context_holding_its_resolved_controls(self):
        seen = []

        @function
        def body(x):
            call_context = current_call_context()
            seen.append(
                (
                    call_context.call_depth,
                    call_context.function_label,
                    call_context.resolved_controls["num_samples"],
                )
            )
            return x

        with config.sampling.using(num_samples=33):
            body(1.0)
            body.using(num_samples=5)(1.0)
        assert seen == [(1, "body", 33), (1, "body", 5)]
        assert current_call_context().is_default()

    def test_apply_and_check_read_their_own_resolved_num_samples(self, monkeypatch):
        """An apply or a check inside a call reads its own num_samples, not the enclosing call's."""
        inner = Function(lambda x: _num_samples_in_effect(), label="inner")
        read_by_check = []

        def check_reading_num_samples(checked_function, args, kwargs):
            read_by_check.append(_num_samples_in_effect())

        monkeypatch.setattr(function_engine, "_check_call", check_reading_num_samples)

        @function
        def outer(x):
            inner.using(num_samples=5).check(x)
            inner.check(x)
            return (inner.using(num_samples=5).apply(x), inner.apply(x))

        session_num_samples = config.sampling.num_samples
        assert outer.using(num_samples=7, raw=True)(1.0) == (5, session_num_samples)
        assert read_by_check == [5, session_num_samples]

    def test_the_context_is_restored_after_a_call_raises(self):
        @function
        def failing(x):
            raise RuntimeError("the body failed")

        with config.sampling.using(num_samples=7):
            enclosing_context = current_call_context()
            with pytest.raises(RuntimeError, match="the body failed"):
                failing(1.0)
            assert current_call_context() is enclosing_context
        assert current_call_context().is_default()

    def test_the_context_is_restored_after_a_block_raises(self):
        with (
            pytest.raises(ValueError, match="inside the block"),
            config.sampling.using(num_samples=7),
        ):
            raise ValueError("inside the block")
        assert current_call_context().is_default()


class TestScopesExitInOrder:
    """Every exit restores the context of the innermost scope still open, whatever the order."""

    @pytest.fixture
    def copied_context(self):
        """A copy of the context, so a failing test cannot leave a scope installed for the others."""
        return contextvars.copy_context()

    def test_a_block_held_open_by_a_generator_and_closed_inside_a_call_raises(self, copied_context):
        copied_context.run(self._close_a_held_open_block_inside_a_call)

    @staticmethod
    def _close_a_held_open_block_inside_a_call():
        session_num_samples = config.sampling.num_samples
        held_open = _held_open_block()
        next(held_open)
        closed_inside = []

        @function
        def closes_the_block(x):
            with pytest.raises(RuntimeError) as error:
                held_open.close()
            closed_inside.append(str(error.value))
            # The call keeps its context, the closed block's values included.
            assert current_call_context().function_label == "closes_the_block"
            assert config.sampling.num_samples == 7
            return x

        closes_the_block(1.0)
        assert closed_inside[0].startswith(
            "cannot close a block of config.sampling while the call of 'closes_the_block', "
            "opened inside it, is still running"
        )
        # Once the call ends, nothing of the closed block applies.
        assert current_call_context().is_default()
        assert config.sampling.num_samples == session_num_samples

    def test_a_block_held_open_past_the_end_of_its_call_is_closed_with_it(self, copied_context):
        copied_context.run(self._leave_a_block_open_past_its_call)

    @staticmethod
    def _leave_a_block_open_past_its_call():
        session_num_samples = config.sampling.num_samples
        holders = []

        @function
        def leaves_the_block_open(x):
            held_open = _held_open_block()
            next(held_open)
            holders.append(held_open)
            return x

        with pytest.raises(RuntimeError) as error:
            leaves_the_block_open(1.0)
        assert str(error.value).startswith(
            "the call of 'leaves_the_block_open' ended while a block of config.sampling, "
            "opened inside it, was still open, so that scope was closed with it"
        )
        # The call's context and the block's are gone, and a later call is a root call.
        assert current_call_context().is_default()
        assert current_call_context().call_depth == 0
        assert config.sampling.num_samples == session_num_samples
        holders[0].close()
        assert current_call_context().is_default()
        root_seen = []

        @function
        def reports_its_depth(x):
            root_seen.append(
                (current_call_context().call_depth, current_call_context().is_root_call())
            )
            return x

        reports_its_depth(1.0)
        assert root_seen == [(1, True)]

    def test_an_outer_block_that_exits_before_an_inner_one_closes_both(self, copied_context):
        copied_context.run(self._exit_an_outer_block_first)

    @staticmethod
    def _exit_an_outer_block_first():
        with pytest.raises(RuntimeError) as error, config.notation.using(max_depth=3):
            held_open = _held_open_block()
            next(held_open)
        assert str(error.value).startswith(
            "a block of config.notation ended while a block of config.sampling, opened inside "
            "it, was still open"
        )
        assert current_call_context().is_default()
        # The inner block was closed with the outer one, so its own exit changes nothing.
        held_open.close()
        assert current_call_context().is_default()

    def test_a_block_closed_in_a_copied_context_stops_applying_in_the_original(
        self, copied_context
    ):
        copied_context.run(self._close_in_a_copy_a_block_opened_outside_it)

    @staticmethod
    def _close_in_a_copy_a_block_opened_outside_it():
        session_num_samples = config.sampling.num_samples
        held_open = _held_open_block()
        next(held_open)

        @function
        def closes_the_block(x):
            with pytest.raises(RuntimeError, match="is still running"):
                held_open.close()
            return x

        contextvars.copy_context().run(closes_the_block, 1.0)
        assert current_call_context().is_default()
        assert config.sampling.num_samples == session_num_samples

    def test_a_block_that_repeats_its_enclosing_values_is_named_as_a_block(self, copied_context):
        def leave_an_equal_block_open():
            with (
                config.sampling.using(num_samples=7),
                pytest.raises(RuntimeError) as error,
                config.notation.using(max_depth=3),
            ):
                held_open = _held_open_block()
                next(held_open)
            return str(error.value)

        message = copied_context.run(leave_an_equal_block_open)
        assert message.startswith(
            "a block of config.notation ended while a block of config.sampling, opened inside it"
        )

    def test_scopes_that_exit_in_order_raise_nothing(self, copied_context):
        def nested_scopes():
            with config.notation.using(max_depth=3), config.sampling.using(num_samples=7):
                assert current_call_context().config_blocks["sampling"]["num_samples"] == 7
            return current_call_context()

        assert copied_context.run(nested_scopes).is_default()


class TestNumSamplesUsed:
    def test_provenance_reports_one_number_or_the_sorted_numbers(self):
        num_samples_used = _NumSamplesUsed()
        assert num_samples_used.provenance_num_samples is None
        num_samples_used.note(64)
        assert num_samples_used.provenance_num_samples == 64
        num_samples_used.note(16)
        num_samples_used.note(64)
        assert num_samples_used.provenance_num_samples == (16, 64)
        assert num_samples_used.num_samples_values == {16, 64}

    def test_a_scope_merges_into_a_collecting_parent_only_on_success(self):
        with _under_call_context(current_call_context().with_fresh_call_accumulators()):
            parent_num_samples_used = current_call_context().num_samples_used
            parent_num_samples_used.collects_calls = True
            with _collecting_call_accumulators("when_parent_collects"):
                _note_num_samples_used(8)
            with pytest.raises(ValueError), _collecting_call_accumulators("always"):
                _note_num_samples_used(9)
                raise ValueError("the scope failed")
        assert parent_num_samples_used.num_samples_values == {8}

    def test_a_call_scope_does_not_merge_into_a_parent_that_does_not_collect(self):
        with _under_call_context(current_call_context().with_fresh_call_accumulators()):
            parent_num_samples_used = current_call_context().num_samples_used
            with _collecting_call_accumulators("when_parent_collects"):
                _note_num_samples_used(8)
            with _collecting_call_accumulators("always") as scope_context:
                assert scope_context.num_samples_used.collects_calls is False
                _note_num_samples_used(9)
        assert parent_num_samples_used.num_samples_values == {9}

    def test_a_pickled_num_samples_used_arrives_empty_and_keeps_collecting_calls(self):
        num_samples_used = _NumSamplesUsed()
        num_samples_used.note(5)
        num_samples_used.collects_calls = True
        restored = pickle.loads(pickle.dumps(num_samples_used))
        assert restored.num_samples_values == frozenset()
        assert restored.collects_calls is True

    def test_threads_noting_at_once_lose_no_number(self):
        num_samples_used = _NumSamplesUsed()
        threads = [
            threading.Thread(
                target=lambda first=first: [
                    num_samples_used.note(n) for n in range(first, first + 200)
                ]
            )
            for first in range(0, 800, 200)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert num_samples_used.num_samples_values == set(range(800))


class TestCarryingTheContextIntoWorkers:
    def test_an_evaluation_context_round_trips_through_pickle(self):
        carried = []

        @function
        def observed(x):
            _note_num_samples_used(5)
            carried.append(current_call_context().evaluation_context())
            return x

        with config.notation.using(max_depth=3):
            observed.using(method_options={"budget": {"steps": 3}}, exact_only=True)(1.0)
        original_context = carried[0]
        restored_context = pickle.loads(pickle.dumps(original_context))
        assert restored_context.config_blocks == original_context.config_blocks
        assert isinstance(restored_context.config_blocks["notation"], MappingProxyType)
        assert restored_context.resolved_controls == original_context.resolved_controls
        assert restored_context.resolved_controls["exact_only"] is True
        assert restored_context.call_depth == 1
        assert restored_context.function_label == "observed"
        assert restored_context.num_samples_used.num_samples_values == frozenset()
        with _under_call_context(restored_context):
            assert _resolved_method_options() == {"budget": {"steps": 3}}

    def test_a_block_opened_inside_one_thread_point_applies_to_that_point_alone(self):
        seen = []

        @function(traceable=False, executor="thread", executor_options={"max_workers": 2})
        def body(x):
            if float(x) == 0.0:
                with config.sampling.using(num_samples=7):
                    seen.append((float(x), config.sampling.num_samples))
            else:
                seen.append((float(x), config.sampling.num_samples))
            return x

        with config.sampling.using(num_samples=512):
            body(NumericArrayBatch(jnp.arange(4.0), "x", label="xs"))
            assert config.sampling.num_samples == 512
        assert sorted(seen) == [(0.0, 7), (1.0, 512), (2.0, 512), (3.0, 512)]


def _resolved_method_options():
    """The running call's ``method_options``, read by a work function in a worker."""
    return {
        name: dict(options) if isinstance(options, MappingProxyType) else options
        for name, options in current_call_context().resolved_controls["method_options"].items()
    }


class TestTheSessionSnapshot:
    """A root call snapshots the session's values, which its evaluation points read."""

    def test_a_root_call_snapshots_every_section_but_prefect(self):
        seen = []

        @function
        def inner(x):
            seen.append(current_call_context().session_config_values)
            return x

        @function
        def outer(x):
            seen.append(current_call_context().session_config_values)
            return inner(x)

        config.sampling.num_samples = 333
        outer(1.0)
        root_snapshot, nested_snapshot = seen
        assert set(root_snapshot) == {"sampling", "notation", "provenance"}
        assert root_snapshot["sampling"]["num_samples"] == 333
        assert root_snapshot["provenance"]["mode"] is config.provenance.mode
        assert isinstance(root_snapshot["sampling"], MappingProxyType)
        assert nested_snapshot is root_snapshot

    def test_a_child_keeps_its_parents_snapshot_and_a_root_takes_the_given_one(self):
        root = current_call_context().child_for_call(
            entry_method_name="__call__",
            function_label="f",
            resolved_controls={},
            session_config_values={"sampling": {"num_samples": 5}},
        )
        child = root.child_for_call(
            entry_method_name="apply", function_label="g", resolved_controls={}
        )
        assert dict(root.session_config_values["sampling"]) == {"num_samples": 5}
        assert child.session_config_values is root.session_config_values

    def test_the_evaluation_context_is_marked_and_its_children_keep_the_mark(self):
        call_context = current_call_context().child_for_call(
            entry_method_name="__call__", function_label="f", resolved_controls={}
        )
        evaluation_context = call_context.evaluation_context()
        assert not call_context.in_evaluation_point
        assert evaluation_context.in_evaluation_point
        child = evaluation_context.child_for_call(
            entry_method_name="__call__", function_label="g", resolved_controls={}
        )
        assert child.in_evaluation_point
        assert child.with_config_block({"sampling": {"num_samples": 3}}).in_evaluation_point

    def test_a_pickled_evaluation_context_keeps_the_snapshot(self):
        carried = []

        @function
        def observed(x):
            carried.append(current_call_context().evaluation_context())
            return x

        config.notation.max_depth = 5
        observed(1.0)
        restored_context = pickle.loads(pickle.dumps(carried[0]))
        assert restored_context.in_evaluation_point
        assert restored_context.session_config_values == carried[0].session_config_values
        assert restored_context.session_config_values["notation"]["max_depth"] == 5
        assert isinstance(restored_context.session_config_values["notation"], MappingProxyType)

    @pytest.mark.parametrize("executor", ["sequential", "thread"])
    def test_a_session_assignment_during_a_call_reaches_none_of_its_points(self, executor):
        seen = []

        @function(
            traceable=False,
            executor=executor,
            executor_options={"max_workers": 1} if executor == "thread" else None,
        )
        def body(x):
            if float(x) == 0.0:
                config.sampling.num_samples = 7
            seen.append((float(x), config.sampling.num_samples))
            return x

        body(NumericArrayBatch(jnp.arange(3.0), "x", label="xs"))
        assert seen == [(0.0, 1024), (1.0, 1024), (2.0, 1024)]
        assert config.sampling.num_samples == 7

    def test_a_call_inside_a_point_resolves_its_controls_from_the_snapshot(self):
        seen = []

        @function
        def inner(x):
            seen.append(_num_samples_in_effect())
            return x

        @function(traceable=False, executor="thread", executor_options={"max_workers": 1})
        def body(x):
            config.sampling.num_samples = 7
            return inner(x)

        body(NumericArrayBatch(jnp.arange(2.0), "x", label="xs"))
        assert seen == [1024, 1024]

    def test_a_block_wins_over_the_snapshot_inside_a_point(self):
        seen = []

        @function(traceable=False, executor="thread", executor_options={"max_workers": 1})
        def body(x):
            seen.append(config.sampling.num_samples)
            return x

        with config.sampling.using(num_samples=55):
            body(NumericArrayBatch(jnp.arange(2.0), "x", label="xs"))
        assert seen == [55, 55]

    def test_a_point_reads_the_live_prefect_settings(self):
        task_runner_sentinel = object()
        seen = []

        @function(traceable=False, executor="thread", executor_options={"max_workers": 1})
        def body(x):
            config.prefect.task_runner = task_runner_sentinel
            seen.append(config.prefect.task_runner)
            return x

        body(NumericArrayBatch(jnp.arange(2.0), "x", label="xs"))
        assert seen == [task_runner_sentinel, task_runner_sentinel]

    def test_a_body_assigning_the_session_still_reads_the_snapshot(self):
        seen = []

        @function
        def body(x):
            config.sampling.num_samples = 11
            seen.append(config.sampling.num_samples)
            return x

        body(1.0)
        assert seen == [1024]
        assert config.sampling.num_samples == 11
