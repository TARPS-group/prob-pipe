"""Tests for ``probpipe.testing.check_executor``, the conformance check of an executor."""

import pytest

from probpipe import EvaluationExecutor, EvaluationOutcomes, executor_registry
from probpipe.core._dispatch import Feasibility
from probpipe.testing import check_executor


class _ReversingExecutor(EvaluationExecutor):
    """An executor that breaks the contract: it returns its outputs in reverse order."""

    name = "reversing"

    def check(self, evaluation_points, /):
        return Feasibility(feasible=True)

    def execute(self, evaluation_points, /):
        outcomes = evaluation_points.chunks()[0].run(evaluation_points.shared_payload())
        return EvaluationOutcomes(
            start=0, outputs=tuple(reversed(outcomes.outputs)), failures=outcomes.failures
        )


class _RefusingExecutor(EvaluationExecutor):
    """An executor whose check refuses every set of evaluation points."""

    name = "refusing"

    def check(self, evaluation_points, /):
        return Feasibility(feasible=False, description="never")

    def execute(self, evaluation_points, /):
        raise AssertionError("an executor that refuses a set must not run it")


class _ChunkedPoolExecutor(EvaluationExecutor):
    """A user's executor that follows the contract: it runs chunks of two in reverse order."""

    name = "chunked_pool"

    def check(self, evaluation_points, /):
        return Feasibility(feasible=True)

    def execute(self, evaluation_points, /):
        shared_payload = evaluation_points.shared_payload()
        chunks = evaluation_points.chunks(chunk_size=2)
        return EvaluationOutcomes.merge([chunk.run(shared_payload) for chunk in reversed(chunks)])


@pytest.mark.parametrize("executor_name", ["jax", "sequential", "thread", "process"])
def test_every_built_in_executor_passes(executor_name):
    check_executor(executor_registry.get_method(executor_name))


def test_a_users_executor_that_follows_the_contract_passes():
    check_executor(_ChunkedPoolExecutor())


def test_an_executor_that_reorders_its_outputs_fails_naming_the_set():
    with pytest.raises(
        AssertionError, match="executor 'reversing', on a simulator with draws nested"
    ):
        check_executor(_ReversingExecutor())


def test_an_executor_that_runs_no_set_fails():
    with pytest.raises(AssertionError, match="cannot run any of check_executor's sets"):
        check_executor(_RefusingExecutor())


def test_the_check_restores_the_parents_jax_settings():
    import jax

    settings_before = (jax.config.jax_enable_x64, jax.config.jax_threefry_partitionable)
    check_executor(executor_registry.get_method("sequential"))
    assert (jax.config.jax_enable_x64, jax.config.jax_threefry_partitionable) == settings_before


class _RebuildingExecutor(EvaluationExecutor):
    """An executor that breaks the contract: it rebuilds each chunk's outcomes and drops what was noted."""

    name = "rebuilding"

    def check(self, evaluation_points, /):
        return Feasibility(feasible=True)

    def execute(self, evaluation_points, /):
        outcomes = evaluation_points.chunks()[0].run(evaluation_points.shared_payload())
        return EvaluationOutcomes(
            start=outcomes.start, outputs=outcomes.outputs, failures=outcomes.failures
        )


class _BrokenCheckExecutor(EvaluationExecutor):
    """An executor whose own check raises."""

    name = "broken_check"

    def check(self, evaluation_points, /):
        raise AttributeError("the executor's check has a bug")

    def execute(self, evaluation_points, /):
        raise AssertionError("never run")


def test_an_executor_that_drops_the_noted_draw_counts_fails():
    with pytest.raises(AssertionError, match="brought back the draw counts"):
        check_executor(_RebuildingExecutor())


def test_an_error_in_the_executors_check_reaches_the_caller():
    with pytest.raises(AttributeError, match="the executor's check has a bug"):
        check_executor(_BrokenCheckExecutor())
