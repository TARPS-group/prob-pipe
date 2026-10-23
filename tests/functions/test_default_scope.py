"""Continuing default workflow streams and their call boundaries."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import (
    Function,
    Normal,
    OutputSpec,
    ProvenanceMode,
    UnmanagedConcurrentWorkflowEntryError,
    config,
    replay_run,
    sample,
    workflow_run,
)
from probpipe.functions import _context
from tests.functions._replay_fixtures import replayable_identity


@pytest.fixture(autouse=True)
def _clear_default():
    yield
    workflow_run.clear_default()


def _draw():
    return sample(Normal("x", loc=0.0, scale=1.0), sample_shape=(4,))


def _words():
    with _context._ephemeral_workflow_run():
        key = _context._commit_stochastic_invocation().key_for(
            stochastic_source_id=("source", 0), logical_unit_id=("singleton",)
        )
        return tuple(int(word) for word in jax.random.key_data(key))


def _nested_draw(x):
    with workflow_run(seed=13):
        return sample(Normal("x", loc=x, scale=1.0))


class TestDefaultScope:
    def test_public_entry_and_sequence_matches_explicit_scope(self):
        from probpipe.functions import workflow_run as exported

        assert exported is probpipe.workflow_run
        with workflow_run(114514):
            expected = [_words(), _words(), _words()]
        workflow_run.set_default(seed=114514)
        assert [_words(), _words(), _words()] == expected
        assert len(set(expected)) == 3
        workflow_run.set_default(seed=114514)
        assert _words() == expected[0]

    def test_explicit_override_and_replay_do_not_advance_default(self):
        with workflow_run(7):
            expected = [_draw(), _draw()]
        with workflow_run(9):
            separate = _draw()
        workflow_run.set_default(seed=7)
        first = _draw()
        with workflow_run(9):
            np.testing.assert_array_equal(_draw(), separate)
        with replay_run(first.provenance):
            np.testing.assert_array_equal(_draw(), first)
            with pytest.raises(RuntimeError, match="default"):
                workflow_run.clear_default()
        np.testing.assert_array_equal(first, expected[0])
        np.testing.assert_array_equal(_draw(), expected[1])

    def test_anonymous_root_is_lazy_and_clear_restores_fresh_roots(self):
        with patch.object(
            _context, "_os_urandom", side_effect=[bytes(8), bytes([1]) * 8, bytes([2]) * 8]
        ) as entropy:
            workflow_run.set_default()
            Function(replayable_identity)(1.0)
            entropy.assert_not_called()
            first, second = _words(), _words()
            entropy.assert_called_once_with(8)
            assert first != second
            workflow_run.clear_default()
            workflow_run.clear_default()
            assert _words() != _words()
            assert entropy.call_count == 3

    @pytest.mark.parametrize(
        ("seed", "error"),
        [
            (True, TypeError),
            (False, TypeError),
            (-1, ValueError),
            (2**64, ValueError),
            (1.5, TypeError),
            ("7", TypeError),
        ],
    )
    def test_invalid_seed_preserves_stream(self, seed, error):
        with workflow_run(3):
            expected = [_words(), _words()]
        workflow_run.set_default(seed=3)
        assert _words() == expected[0]
        with pytest.raises(error):
            workflow_run.set_default(seed=seed)
        assert _words() == expected[1]

    @pytest.mark.parametrize("seed", [0, 2**64 - 1])
    def test_seed_endpoints(self, seed):
        with workflow_run(seed):
            expected = _words()
        workflow_run.set_default(seed=seed)
        assert _words() == expected

    def test_explicit_anonymous_scope_still_uses_entropy(self):
        workflow_run.set_default(seed=7)
        with patch.object(_context, "_os_urandom", return_value=bytes(8)) as entropy:
            with workflow_run():
                _words()
            entropy.assert_called_once_with(8)

    def test_mode_changes_preserve_stream_and_config_reset_keeps_default(self):
        with workflow_run(7):
            expected = [_draw(), _draw(), _draw()]
        workflow_run.set_default(seed=7)
        for mode, reference in zip(ProvenanceMode, expected, strict=True):
            with config.provenance.using(mode=mode):
                result = _draw()
            np.testing.assert_array_equal(result, reference)
            assert (result.provenance is None) == (mode is ProvenanceMode.OFF)
            config.reset()

    def test_mode_is_fixed_inside_each_default_call(self):
        workflow_run.set_default(seed=7)
        with _context._ephemeral_workflow_run(), pytest.raises(RuntimeError, match="provenance"):
            config.provenance.mode = ProvenanceMode.OFF
        config.provenance.mode = ProvenanceMode.OFF
        assert _draw().provenance is None

    @pytest.mark.parametrize("method", ["set_default", "clear_default"])
    @pytest.mark.parametrize(
        "entry", ["scope", "probe", "apply", "call", "replay", "jit", "vmap", "grad"]
    )
    def test_refused_mutation_preserves_default_stream(self, method, entry):
        with workflow_run(7):
            expected = [_words(), _words()]
        with workflow_run(9):
            recorded = _draw()
        workflow_run.set_default(seed=7)
        assert _words() == expected[0]
        action = getattr(workflow_run, method)

        def transformed(x):
            action()
            return x

        with pytest.raises(RuntimeError, match="default"):
            if entry == "scope":
                with workflow_run(1):
                    action()
            elif entry == "probe":
                with _context._workflow_probe():
                    action()
            elif entry == "apply":
                Function(action).apply()
            elif entry == "call":
                Function(action)()
            elif entry == "replay":
                with replay_run(recorded.provenance):
                    _draw()
                    action()
            else:
                argument = jnp.ones(2) if entry == "vmap" else 1.0
                getattr(jax, entry)(transformed)(argument)
        assert _words() == expected[1]

    def test_failed_call_keeps_consumed_position_and_closes_its_frame(self):
        with workflow_run(7):
            expected = [_words(), _words()]
        workflow_run.set_default(seed=7)
        with pytest.raises(ValueError, match="failed"), _context._ephemeral_workflow_run():
            assert _words() == expected[0]
            saved = copy_context()
            raise ValueError("failed")
        with pytest.raises(UnmanagedConcurrentWorkflowEntryError, match="closed"):
            saved.run(_words)
        assert _words() == expected[1]

    def test_public_failed_call_keeps_consumed_draws(self):
        seen = []

        def consume_and_fail():
            seen.append(np.asarray(_draw()))
            raise ValueError("failed after drawing")

        workflow = Function(consume_and_fail, traceable=False)
        with workflow_run(7):
            with pytest.raises(ValueError, match="failed after drawing"):
                workflow()
            expected_next = _draw()
        workflow_run.set_default(seed=7)
        with pytest.raises(ValueError, match="failed after drawing"):
            workflow()
        assert len(seen) == 2
        np.testing.assert_array_equal(seen[0], seen[1])
        np.testing.assert_array_equal(_draw(), expected_next)

    def test_probe_and_explicit_key_do_not_consume_positions(self):
        with workflow_run(7):
            expected = _words()
        workflow_run.set_default(seed=7)
        with pytest.raises(_context._StochasticProbeSignal), _context._workflow_probe():
            _words()
        explicit_draw = Function(lambda rng_key: jax.random.normal(rng_key, (4,)))
        key = jax.random.PRNGKey(11)
        np.testing.assert_array_equal(explicit_draw(key), jax.random.normal(key, (4,)))
        assert _words() == expected

    @pytest.mark.parametrize("executor", ["sequential", "thread", "process"])
    def test_lifted_and_nested_draws_match_explicit_scope(self, executor):
        workflow = Function(_nested_draw, executor=executor, max_evaluations=5, traceable=False)
        law = Normal("x", loc=0.0, scale=1.0)
        with workflow_run(8):
            expected = [workflow(law), workflow(law)]
        workflow_run.set_default(seed=8)
        for reference in expected:
            np.testing.assert_array_equal(workflow(law).atoms, reference.atoms)

    def test_lifted_replay_uses_recorded_nonzero_position(self):
        workflow = Function(
            replayable_identity,
            executor="sequential",
            max_evaluations=5,
            output_spec=OutputSpec(replayable_identity=None),
        )
        law = Normal("x", loc=0.0, scale=1.0)
        workflow_run.set_default(seed=7)
        workflow(law)
        recorded = workflow(law)
        with replay_run(recorded.provenance):
            np.testing.assert_array_equal(workflow(law).atoms, recorded.atoms)

    def test_copied_thread_rejects_default_but_can_replace_it(self):
        workflow_run.set_default(seed=7)
        copied = copy_context()
        expected = _words()

        def worker():
            with pytest.raises(UnmanagedConcurrentWorkflowEntryError, match="default"):
                _words()
            workflow_run.set_default(seed=7)
            return _words()

        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(copied.run, worker).result() == expected

    def test_new_thread_has_no_default(self):
        workflow_run.set_default(seed=7)
        with patch.object(_context, "_os_urandom", return_value=bytes(8)) as entropy:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_words).result()
            entropy.assert_called_once_with(8)

    def test_copied_async_task_must_set_or_clear_default(self):
        async def child():
            with pytest.raises(UnmanagedConcurrentWorkflowEntryError, match="default"):
                _words()
            workflow_run.clear_default()
            with patch.object(_context, "_os_urandom", return_value=bytes(8)) as entropy:
                _words()
                entropy.assert_called_once_with(8)

        async def parent():
            workflow_run.set_default(seed=7)
            await asyncio.create_task(child())
            with patch.object(_context, "_os_urandom") as entropy:
                _words()
                entropy.assert_not_called()

        asyncio.run(parent())

    def test_script_default_is_not_shared_by_asyncio_run_or_to_thread(self):
        with workflow_run(7):
            expected = [_draw(), _draw()]
        workflow_run.set_default(seed=7)
        np.testing.assert_array_equal(_draw(), expected[0])

        def independent_worker():
            with workflow_run(7):
                return _draw()

        async def main():
            with pytest.raises(UnmanagedConcurrentWorkflowEntryError, match="default"):
                _draw()
            with pytest.raises(UnmanagedConcurrentWorkflowEntryError, match="default"):
                await asyncio.to_thread(_draw)
            np.testing.assert_array_equal(await asyncio.to_thread(independent_worker), expected[0])
            workflow_run.set_default(seed=7)
            np.testing.assert_array_equal(_draw(), expected[0])
            workflow_run.clear_default()

        asyncio.run(main())
        np.testing.assert_array_equal(_draw(), expected[1])
