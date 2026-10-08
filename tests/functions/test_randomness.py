"""Workflow scopes and structural keys (design V.8).

A workflow scope owns every ProbPipe-caused draw inside it, and each draw's key
is a pure function of the scope's root seed and the event's structural
identity. Hence a seeded scope reproduces its draws while an unseeded or absent
scope is fresh, and perturbing an input reuses the same keys. ``replay_run``
re-executes one recorded call on its recorded draws.
"""

from __future__ import annotations

from contextvars import copy_context
from threading import Thread

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    ReplayCompatibilityError,
    UnmanagedConcurrentWorkflowEntryError,
    converter_registry,
    replay_run,
    workflow_run,
)

from ._design_helpers import atom_leaves, standard_normal


def _shift(z, offset=0.0):
    return z + offset


def _lift() -> Function:
    """A lifted call of a module-level def, which replay can anchor."""
    return Function("shift", _shift, n_broadcast_samples=8, dispatch="sequential")


def _draws(result) -> np.ndarray:
    (leaf,) = atom_leaves(result)
    return leaf


def _two_empirical_draws(offset):
    """Two sets of four standard normal draws by the converter registry, shifted by *offset*."""
    law = standard_normal()
    first = converter_registry.convert(law, EmpiricalDistribution, num_samples=4)
    second = converter_registry.convert(law, EmpiricalDistribution, num_samples=4)
    return jnp.stack([first.atoms.values, second.atoms.values]) + offset


def _drawing() -> Function:
    """A Function whose body draws twice, each draw a workflow-owned event."""
    return Function("two_draws", _two_empirical_draws)


class TestScopes:
    def test_a_seeded_scope_reproduces_its_draws(self):
        law = standard_normal()
        with workflow_run(seed=42):
            first = _lift()(law)
        with workflow_run(seed=42):
            second = _lift()(law)

        np.testing.assert_array_equal(_draws(first), _draws(second))

    def test_another_seed_gives_other_draws(self):
        law = standard_normal()
        with workflow_run(seed=42):
            first = _lift()(law)
        with workflow_run(seed=43):
            second = _lift()(law)

        assert not np.array_equal(_draws(first), _draws(second))

    def test_an_anonymous_scope_is_fresh(self):
        law = standard_normal()
        with workflow_run():
            first = _lift()(law)
        with workflow_run():
            second = _lift()(law)

        assert not np.array_equal(_draws(first), _draws(second))

    def test_a_call_outside_any_scope_is_fresh(self):
        law = standard_normal()

        assert not np.array_equal(_draws(_lift()(law)), _draws(_lift()(law)))

    def test_repeated_invocations_in_one_scope_draw_distinct_keys(self):
        law = standard_normal()
        with workflow_run(seed=42):
            first = _lift()(law)
            second = _lift()(law)

        assert not np.array_equal(_draws(first), _draws(second))

    def test_an_empty_nested_scope_does_not_shift_the_enclosing_draws(self):
        law = standard_normal()
        with workflow_run(seed=42):
            _lift()(law)
            baseline = _lift()(law)
        with workflow_run(seed=42):
            _lift()(law)
            with workflow_run():
                pass
            nested = _lift()(law)

        np.testing.assert_array_equal(_draws(baseline), _draws(nested))

    def test_an_unmanaged_thread_cannot_enter_a_copied_scope(self):
        errors = []
        law = standard_normal()
        with workflow_run(seed=42):
            copied = copy_context()

            def run():
                try:
                    copied.run(_lift(), law)
                except UnmanagedConcurrentWorkflowEntryError as error:
                    errors.append(error)

            thread = Thread(target=run)
            thread.start()
            thread.join()

        assert len(errors) == 1


class TestStructuralKeys:
    def test_a_perturbed_input_reuses_the_same_keys(self):
        law = standard_normal()
        with workflow_run(seed=42):
            base = _lift()(law, 0.0)
        with workflow_run(seed=42):
            shifted = _lift()(law, 1.0)

        np.testing.assert_allclose(_draws(shifted), _draws(base) + 1.0, rtol=1e-6)

    def test_the_derivation_version_is_recorded_with_the_execution(self):
        with workflow_run(seed=42):
            result = _lift()(standard_normal())

        assert result.provenance.controls["randomness"]["rng_abi"] == "ProbPipe-RNG-v1"

    def test_there_is_no_framework_key_on_a_call(self):
        with pytest.raises(TypeError, match="unknown Function option"):
            _lift().with_options(key=jnp.zeros(2, dtype=jnp.uint32))


class TestApply:
    """Each draw in the body of an ``apply`` evaluation is its own workflow-owned event."""

    def test_a_seeded_scope_reproduces_distinct_draws(self):
        with workflow_run(seed=42):
            first = _drawing().apply(0.0)
        with workflow_run(seed=42):
            second = _drawing().apply(0.0)

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first[0], first[1])

    def test_another_seed_gives_other_draws(self):
        with workflow_run(seed=42):
            first = _drawing().apply(0.0)
        with workflow_run(seed=43):
            second = _drawing().apply(0.0)

        assert not np.array_equal(first[0], second[0])
        assert not np.array_equal(first[1], second[1])

    def test_repeated_evaluations_in_one_scope_draw_distinct_keys(self):
        with workflow_run(seed=42):
            first = _drawing().apply(0.0)
            second = _drawing().apply(0.0)

        assert not np.array_equal(first, second)

    def test_an_evaluation_outside_any_scope_is_fresh(self):
        assert not np.array_equal(_drawing().apply(0.0), _drawing().apply(0.0))

    def test_an_evaluation_draws_as_a_call_does(self):
        with workflow_run(seed=42):
            called = _drawing()(0.0)
        with workflow_run(seed=42):
            applied = _drawing().apply(0.0)

        np.testing.assert_array_equal(np.asarray(called), applied)

    def test_an_evaluation_that_draws_nothing_does_not_shift_later_draws(self):
        law = standard_normal()
        with workflow_run(seed=42):
            baseline = _lift()(law)
        with workflow_run(seed=42):
            Function("shift", _shift).apply(1.0)
            after = _lift()(law)

        np.testing.assert_array_equal(_draws(baseline), _draws(after))


class TestReplay:
    def test_replay_reexecutes_one_recorded_call_on_its_recorded_draws(self):
        law = standard_normal()
        with workflow_run(seed=42):
            recorded = _lift()(law)
        with replay_run(recorded.provenance):
            replayed = _lift()(law)

        np.testing.assert_array_equal(_draws(recorded), _draws(replayed))

    def test_replay_refuses_a_call_whose_plan_differs(self):
        law = standard_normal()
        with workflow_run(seed=42):
            recorded = _lift()(law)

        with pytest.raises(ReplayCompatibilityError), replay_run(recorded.provenance):
            _lift().with_options(n_broadcast_samples=9)(law)
