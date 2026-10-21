"""The conformance check of an executor (V.9, the executor contract).

:func:`check_executor` runs fixed sets of evaluation points on an executor and
compares each outcome with the outcome the ``sequential`` executor gives, so a
user can check an executor of their own before they register it, and CI checks
every built-in executor.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..core._evaluation_points import EvaluationOutcomes, EvaluationPoints
from ..core._execution_claims import NO_CLAIMS, ExecutionClaims
from ..core._executors import EvaluationExecutor, _run_on_executor, executor_registry
from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec

__all__ = ["check_executor"]

#: The number of evaluation points of each conformance set.
_NUM_EVALUATION_POINTS = 8

#: The seed of the workflow each conformance set runs in.
_CONFORMANCE_SEED = 20261021


def check_executor(executor: EvaluationExecutor) -> None:
    """Check that *executor* follows the executor contract, raising on the first rule it breaks.

    The check runs fixed sets of evaluation points on *executor* and on the
    ``sequential`` executor, and compares their outcomes. The sets include a
    simulator that draws from the workflow inside each evaluation point,
    evaluation points that raise, an output that does not match its output
    spec, a callable that takes a whole chunk at once, a callable that reads
    the call's settings, and a callable JAX can trace. They run with 64-bit
    floats on and ``jax_threefry_partitionable`` flipped, so an executor that
    runs work in another process must carry the parent's JAX settings; both
    settings are restored when the check ends. A set the executor's ``check``
    reports it cannot run is skipped, and the executor must run at least one.

    An executor passes when, for each set it runs, it returns one outcome per
    evaluation point in order, each output equal to the ``sequential`` one,
    bit for bit, or within a relative tolerance of ``1e-6`` for a set JAX
    compiles, and each failure at the same evaluation point with the same
    kind.

    Parameters
    ----------
    executor : EvaluationExecutor
        The executor, registered or not; it runs with its default options.

    Raises
    ------
    AssertionError
        If the executor breaks a rule of the contract, naming the set and the
        rule.

    Examples
    --------
    >>> from probpipe import executor_registry
    >>> from probpipe.testing import check_executor
    >>> check_executor(executor_registry.get_method("thread"))  # doctest: +SKIP
    """
    reference_executor = executor_registry.get_method("sequential")
    sets_run = []
    with _flipped_parent_jax_settings():
        for conformance_set in _CONFORMANCE_SETS:
            executor_outcomes = conformance_set.outcomes_on(executor)
            if executor_outcomes is None:
                continue
            reference_outcomes = conformance_set.outcomes_on(reference_executor)
            assert reference_outcomes is not None
            conformance_set.check_agreement(executor.name, executor_outcomes, reference_outcomes)
            sets_run.append(conformance_set.name)
    assert sets_run, (
        f"executor {executor.name!r} cannot run any of check_executor's sets of evaluation "
        "points. Its check must accept at least one of them"
    )


# ---------------------------------------------------------------------------
# The conformance sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ConformanceSet:
    """One fixed set of evaluation points that :func:`check_executor` runs.

    Attributes
    ----------
    name : str
        What the set tests, for messages.
    callable_under_test : callable
        The callable each evaluation point calls.
    claims : ExecutionClaims
        The callable's claims.
    output_spec : OutputSpec or None
        The callable's output spec.
    compiled : bool
        Whether ``jax`` compiles the set, so outputs agree within a tolerance
        rather than bit for bit.
    """

    name: str
    callable_under_test: Callable[..., Any]
    claims: ExecutionClaims = NO_CLAIMS
    output_spec: OutputSpec | None = None
    compiled: bool = False

    def evaluation_points(self) -> EvaluationPoints:
        """The set's evaluation points, built under the running call."""
        return EvaluationPoints(
            self.callable_under_test,
            varying={
                "theta": jnp.arange(2.0 * _NUM_EVALUATION_POINTS).reshape(
                    _NUM_EVALUATION_POINTS, 2
                ),
                "point_index": np.arange(_NUM_EVALUATION_POINTS),
            },
            shared={"scale": jnp.asarray(1.5)},
            origin=f"check_executor: {self.name}",
            output_spec=self.output_spec,
            claims=self.claims,
            traceable=self.claims.traceable,
        )

    def outcomes_on(self, executor: EvaluationExecutor) -> EvaluationOutcomes | None:
        """The set's outcomes on *executor*, or ``None`` when its check refuses the set."""
        from ..core.config import config
        from ..functions._context import workflow_run
        from ..functions._execution import _requester_workflow_scope
        from ..functions._function import function

        def ask_for_evaluations() -> EvaluationOutcomes | None:
            with _requester_workflow_scope():
                evaluation_points = self.evaluation_points()
                if executor.check(evaluation_points).feasible is not True:
                    evaluation_points._point_scopes.close()
                    return None
                return _run_on_executor(evaluation_points, executor, {})

        requester = function(ask_for_evaluations, label=f"check_executor: {self.name}")
        with (
            workflow_run(seed=_CONFORMANCE_SEED),
            config.sampling.using(num_samples=_BLOCK_NUM_SAMPLES),
        ):
            return requester.using(raw=True)()

    def check_agreement(
        self,
        executor_name: str,
        executor_outcomes: EvaluationOutcomes,
        reference_outcomes: EvaluationOutcomes,
    ) -> None:
        """Raise if *executor_outcomes* differ from the ``sequential`` executor's."""
        where = f"executor {executor_name!r}, on {self.name}"
        assert executor_outcomes.start == 0 and len(executor_outcomes.outputs) == len(
            reference_outcomes.outputs
        ), (
            f"{where}, returned {len(executor_outcomes.outputs)} outcomes starting at "
            f"{executor_outcomes.start} for {len(reference_outcomes.outputs)} evaluation points"
        )
        executor_failures = {
            point_index: failure.kind for point_index, failure in executor_outcomes.failures.items()
        }
        reference_failures = {
            point_index: failure.kind
            for point_index, failure in reference_outcomes.failures.items()
        }
        assert executor_failures == reference_failures, (
            f"{where}, reported the failures {executor_failures} where sequential reports "
            f"{reference_failures}"
        )
        for point_index, (executor_output, reference_output) in enumerate(
            zip(executor_outcomes.outputs, reference_outcomes.outputs, strict=True)
        ):
            if point_index in reference_failures:
                continue
            executor_array = np.asarray(executor_output)
            reference_array = np.asarray(reference_output)
            assert executor_array.dtype == reference_array.dtype, (
                f"{where}, returned dtype {executor_array.dtype} at evaluation point "
                f"{point_index} where sequential returns {reference_array.dtype}; a worker must "
                "run under the parent's JAX settings"
            )
            if self.compiled:
                agrees = np.allclose(executor_array, reference_array, rtol=1e-6, atol=0.0)
            else:
                agrees = np.array_equal(executor_array, reference_array)
            assert agrees, (
                f"{where}, returned {executor_array!r} at evaluation point {point_index} where "
                f"sequential returns {reference_array!r}"
            )


#: The value of ``num_samples`` that the block around each set sets, which the
#: set that reads the call's settings returns.
_BLOCK_NUM_SAMPLES = 37


def _simulate_with_a_nested_draw(theta: Any, point_index: Any, scale: Any) -> np.ndarray:
    """A simulator that draws from the workflow inside each evaluation point, finishing in reverse order."""
    from ..distributions import Normal
    from ..operations import sample

    time.sleep(0.002 * (_NUM_EVALUATION_POINTS - int(point_index)))
    noise = np.asarray(sample(Normal("noise", 0.0, 1.0)))
    return np.asarray(scale) * (np.asarray(theta) + noise)


def _fail_at_two_evaluation_points(theta: Any, point_index: Any, scale: Any) -> np.ndarray:
    """A simulator whose evaluation points 3 and 6 raise."""
    if int(point_index) in (3, 6):
        raise ValueError(f"the simulator failed at evaluation point {int(point_index)}")
    return np.asarray(scale) * np.asarray(theta)


def _return_a_wrong_shape_once(theta: Any, point_index: Any, scale: Any) -> np.ndarray:
    """A simulator whose evaluation point 1 returns a shape its output spec does not declare."""
    if int(point_index) == 1:
        return np.zeros(3)
    return np.asarray(scale) * np.asarray(theta)


def _simulate_a_whole_chunk(theta: Any, point_index: Any, scale: Any) -> np.ndarray:
    """A simulator that takes a whole chunk of evaluation points at once, every argument stacked."""
    return np.asarray(scale)[:, np.newaxis] * np.asarray(theta) * 2.0


def _read_the_call_settings(theta: Any, point_index: Any, scale: Any) -> np.ndarray:
    """A callable that returns the ``num_samples`` that the call's settings give it."""
    from ..core.config import config

    return np.asarray(float(config.sampling.num_samples)) + np.asarray(theta)


def _traceable_growth(theta: Any, point_index: Any, scale: Any) -> Any:
    """A callable JAX can trace."""
    return scale * jnp.exp(jnp.sin(theta))


_CONFORMANCE_SETS = (
    _ConformanceSet(
        "a simulator with draws nested in its evaluation points", _simulate_with_a_nested_draw
    ),
    _ConformanceSet("evaluation points that raise", _fail_at_two_evaluation_points),
    _ConformanceSet(
        "an output that does not match its output spec",
        _return_a_wrong_shape_once,
        output_spec=OutputSpec(simulated=NumericArraySpec((2,), "float64")),
    ),
    _ConformanceSet(
        "a callable that takes a whole chunk",
        _simulate_a_whole_chunk,
        claims=ExecutionClaims(traceable=False, vectorized=True),
    ),
    _ConformanceSet("a callable that reads the call's settings", _read_the_call_settings),
    _ConformanceSet(
        "a callable JAX can trace",
        _traceable_growth,
        claims=ExecutionClaims(traceable=True),
        compiled=True,
    ),
)


@contextmanager
def _flipped_parent_jax_settings() -> Iterator[None]:
    """64-bit floats on and ``jax_threefry_partitionable`` flipped, restored on exit."""
    x64_before = jax.config.jax_enable_x64
    partitionable_before = jax.config.jax_threefry_partitionable
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_threefry_partitionable", not partitionable_before)
    try:
        yield
    finally:
        jax.config.update("jax_threefry_partitionable", partitionable_before)
        jax.config.update("jax_enable_x64", x64_before)
