"""Tests for the call context: how a call, a ``config`` block, and a worker derive, install, and restore it."""

import contextvars
import pickle
import threading
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from probpipe import Function, NumericArrayBatch, config, function
from probpipe.core._call_context import (
    _carrying_call_context,
    _collecting_num_samples_used,
    _ContextBoundFunction,
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
        parent = current_call_context().with_fresh_num_samples_used()
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

        assert outer.using(num_samples=7, raw=True)(1.0) == (5, 1024)
        assert read_by_check == [5, 1024]

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
        assert config.sampling.num_samples == 1024


class TestScopesExitInOrder:
    def test_a_block_held_open_by_a_generator_and_closed_inside_a_call_raises(self):
        # The block that fails to close stays installed, so the test runs in a
        # copied context that keeps it from the other tests.
        contextvars.copy_context().run(self._close_a_held_open_block_inside_a_call)

    @staticmethod
    def _close_a_held_open_block_inside_a_call():
        held_open = _held_open_block()
        next(held_open)
        closed_inside = []

        @function
        def closes_the_block(x):
            with pytest.raises(
                RuntimeError, match=r"cannot close a block of config\.sampling"
            ) as error:
                held_open.close()
            closed_inside.append(str(error.value))
            assert current_call_context().function_label == "closes_the_block"
            return x

        closes_the_block(1.0)
        assert "the call of 'closes_the_block'" in closed_inside[0]
        assert "is still open" in closed_inside[0]
        # The block that failed to close restored nothing, so it still applies.
        assert config.sampling.num_samples == 7

    def test_a_scope_that_exits_before_one_it_encloses_discards_nothing(self):
        contextvars.copy_context().run(self._exit_an_enclosing_scope_first)

    @staticmethod
    def _exit_an_enclosing_scope_first():
        outer_context = current_call_context().with_config_block({"notation": {"max_depth": 3}})
        outer_scope = _under_call_context(outer_context)
        outer_scope.__enter__()
        inner_context = outer_context.with_fresh_num_samples_used()
        inner_scope = _under_call_context(inner_context)
        inner_scope.__enter__()
        with pytest.raises(RuntimeError, match=r"cannot close a block of config\.notation"):
            outer_scope.__exit__(None, None, None)
        assert current_call_context() is inner_context
        inner_scope.__exit__(None, None, None)
        assert current_call_context() is outer_context


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
        with _under_call_context(current_call_context().with_fresh_num_samples_used()):
            parent_num_samples_used = current_call_context().num_samples_used
            parent_num_samples_used.collects_calls = True
            with _collecting_num_samples_used("when_parent_collects"):
                _note_num_samples_used(8)
            with pytest.raises(ValueError), _collecting_num_samples_used("always"):
                _note_num_samples_used(9)
                raise ValueError("the scope failed")
        assert parent_num_samples_used.num_samples_values == {8}

    def test_a_call_scope_does_not_merge_into_a_parent_that_does_not_collect(self):
        with _under_call_context(current_call_context().with_fresh_num_samples_used()):
            parent_num_samples_used = current_call_context().num_samples_used
            with _collecting_num_samples_used("when_parent_collects"):
                _note_num_samples_used(8)
            with _collecting_num_samples_used("always") as scope_num_samples_used:
                assert scope_num_samples_used.collects_calls is False
                _note_num_samples_used(9)
        assert parent_num_samples_used.num_samples_values == {9}

    def test_a_pickled_num_samples_used_arrives_new_and_empty(self):
        num_samples_used = _NumSamplesUsed()
        num_samples_used.note(5)
        num_samples_used.collects_calls = True
        restored = pickle.loads(pickle.dumps(num_samples_used))
        assert restored.num_samples_values == frozenset()
        assert restored.collects_calls is False

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
    def test_the_default_context_carries_nothing(self):
        def work():
            return 1

        assert _carrying_call_context(work) is work

    def test_a_carried_context_round_trips_through_pickle(self):
        carried = []

        @function
        def dispatching(x):
            _note_num_samples_used(5)
            carried.append(_carrying_call_context(_resolved_method_options))
            return x

        with config.notation.using(max_depth=3):
            dispatching.using(method_options={"budget": {"steps": 3}}, exact_only=True)(1.0)
        restored = pickle.loads(pickle.dumps(carried[0]))
        assert isinstance(restored, _ContextBoundFunction)
        restored_context = restored.call_context
        original_context = carried[0].call_context
        assert restored_context.config_blocks == original_context.config_blocks
        assert isinstance(restored_context.config_blocks["notation"], MappingProxyType)
        assert restored_context.resolved_controls == original_context.resolved_controls
        assert restored_context.resolved_controls["exact_only"] is True
        assert restored_context.call_depth == 1
        assert restored_context.function_label == "dispatching"
        assert restored_context.num_samples_used.num_samples_values == frozenset()
        assert restored() == {"budget": {"steps": 3}}

    def test_a_worker_notes_its_draws_into_the_call_and_a_failed_one_notes_none(self):
        def noting(num_samples):
            _note_num_samples_used(num_samples)
            if num_samples < 0:
                raise ValueError("the work failed")

        with _under_call_context(current_call_context().with_fresh_num_samples_used()):
            call_num_samples_used = current_call_context().num_samples_used
            carried = _carrying_call_context(noting)
            worker = threading.Thread(target=carried, args=(12,))
            worker.start()
            worker.join()
            with pytest.raises(ValueError):
                carried(-1)
        assert call_num_samples_used.num_samples_values == {12}

    def test_a_block_opened_inside_one_thread_point_applies_to_that_point_alone(self):
        seen = []

        @function(dispatch="thread", max_workers=2)
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
