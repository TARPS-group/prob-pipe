"""Tests that lifted calls, sweeps, and plain calls run their evaluations on executors (V.9).

Each requester builds one set of evaluation points and runs it on the executor
its call chooses. The executor decides only how the evaluations run, so every
executor gives the same draws, and ``jax`` agrees with them up to rounding.
"""

import threading
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    EvaluationError,
    EvaluationExecutor,
    EvaluationOutcomes,
    EvaluationPoints,
    Function,
    Normal,
    NumericArrayBatch,
    ResolutionError,
    WorkflowKind,
    executor_registry,
    function,
    sample,
    workflow_run,
)
from probpipe.core import _executors

#: The executors that run a function one evaluation point at a time, each of
#: which gives the same draws as ``sequential``.
_PYTHON_EXECUTORS = ("sequential", "thread", "process", "reversed_chunks")


class _ReversedChunksExecutor(EvaluationExecutor):
    """A user's executor that runs its chunks in reverse order, one evaluation point each."""

    @property
    def name(self):
        return "reversed_chunks"

    @property
    def description(self):
        return "Run each evaluation point as a chunk of its own, last first."

    def check(self, evaluation_points, /):
        return evaluation_points.shipping_check()

    def execute(self, evaluation_points, /):
        shared_payload = evaluation_points.shared_payload()
        chunks = evaluation_points.chunks(chunk_size=1)
        return EvaluationOutcomes.merge([chunk.run(shared_payload) for chunk in reversed(chunks)])


@pytest.fixture(autouse=True)
def _a_users_executor_registered(monkeypatch):
    """The registry holds the built-in executors and the user's ``reversed_chunks``, for one test."""
    registry = _executors._ExecutorRegistry(name="executors", description="executors")
    for executor_name in executor_registry.list_methods():
        registry.register(executor_registry.get_method(executor_name))
    registry.register(_ReversedChunksExecutor())
    monkeypatch.setattr(_executors, "executor_registry", registry)


def _shift(x):
    return np.asarray(x) + 1.0


def _shift_with_a_nested_draw(x):
    return np.asarray(x) + np.asarray(sample(Normal("noise", 0.0, 1.0)))


def _scaled(x, scale):
    return np.asarray(x) * scale


def _lifted_atoms(function_of_x, law, *, seed=3, **controls):
    with workflow_run(seed=seed):
        return np.asarray(function_of_x.using(**controls)(x=law).atoms)


class TestAgreementAcrossExecutors:
    @pytest.mark.parametrize("executor", _PYTHON_EXECUTORS)
    def test_a_sampling_lift_draws_the_same_under_every_executor(self, executor):
        opaque = Function(_shift, label="shift", traceable=False, max_evaluations=12)
        law = Normal("x", 0.0, 1.0)
        np.testing.assert_array_equal(
            _lifted_atoms(opaque, law, executor=executor),
            _lifted_atoms(opaque, law, executor="sequential"),
        )

    @pytest.mark.parametrize("executor", _PYTHON_EXECUTORS)
    def test_a_nested_draw_inside_a_lift_takes_the_same_key_under_every_executor(self, executor):
        opaque = Function(
            _shift_with_a_nested_draw, label="noisy", traceable=False, max_evaluations=6
        )
        law = Normal("x", 0.0, 1.0)
        reference = _lifted_atoms(opaque, law, executor="sequential")
        np.testing.assert_array_equal(_lifted_atoms(opaque, law, executor=executor), reference)
        assert len(np.unique(reference)) == len(reference)

    def test_jax_agrees_with_the_executors_that_run_one_point_at_a_time(self):
        traceable = Function(lambda x: jnp.exp(x), label="growth", max_evaluations=64)
        opaque = Function(
            lambda x: np.exp(np.asarray(x)), label="growth", traceable=False, max_evaluations=64
        )
        law = Normal("x", 0.0, 0.5)
        with workflow_run(seed=5):
            mapped = traceable(x=law)
        with workflow_run(seed=5):
            one_at_a_time = opaque.using(executor="thread")(x=law)
        assert mapped.provenance.metadata["evaluations"][0]["executor"] == "jax"
        assert one_at_a_time.provenance.metadata["evaluations"][0]["executor"] == "thread"
        np.testing.assert_allclose(
            np.asarray(mapped.atoms), np.asarray(one_at_a_time.atoms), rtol=1e-6
        )

    @pytest.mark.parametrize("executor", _PYTHON_EXECUTORS)
    def test_an_enumerated_lift_gives_the_same_atoms_and_weights_under_every_executor(
        self, executor
    ):
        opaque = Function(_shift, label="shift", traceable=False)
        law = EmpiricalDistribution(jnp.asarray([1.0, 2.0, 4.0]), output_name="x")
        with workflow_run(seed=0):
            reference = opaque.using(executor="sequential")(x=law)
        with workflow_run(seed=0):
            result = opaque.using(executor=executor)(x=law)
        np.testing.assert_array_equal(np.asarray(result.atoms), [2.0, 3.0, 5.0])
        np.testing.assert_array_equal(np.asarray(result.atoms), np.asarray(reference.atoms))
        np.testing.assert_array_equal(np.asarray(result.weights), np.asarray(reference.weights))

    @pytest.mark.parametrize("executor", _PYTHON_EXECUTORS)
    def test_a_sweep_gives_the_same_cells_under_every_executor(self, executor):
        opaque = Function(_scaled, label="scaled", traceable=False)
        grid = NumericArrayBatch(jnp.arange(5.0), "theta", label="grid")
        np.testing.assert_array_equal(
            np.asarray(opaque.using(executor=executor)(x=grid, scale=3.0)),
            [0.0, 3.0, 6.0, 9.0, 12.0],
        )

    @pytest.mark.parametrize("executor", _PYTHON_EXECUTORS)
    def test_a_sweep_of_lifts_draws_the_same_under_every_executor(self, executor):
        opaque = Function(
            lambda x, scale: np.asarray(x) * scale,
            label="scaled",
            traceable=False,
            max_evaluations=4,
        )
        scales = NumericArrayBatch(jnp.asarray([1.0, 10.0]), "scale", label="scales")
        law = Normal("x", 0.0, 1.0)

        def cells(chosen_executor):
            with workflow_run(seed=8):
                laws = opaque.using(executor=chosen_executor)(x=law, scale=scales)
            return [np.asarray(laws[index].atoms) for index in range(2)]

        for cell, reference_cell in zip(cells(executor), cells("sequential"), strict=True):
            np.testing.assert_array_equal(cell, reference_cell)

    @pytest.mark.parametrize("executor", _PYTHON_EXECUTORS)
    def test_a_plain_call_returns_the_same_under_every_executor(self, executor):
        opaque = Function(_shift_with_a_nested_draw, label="noisy", traceable=False)
        with workflow_run(seed=2):
            reference = np.asarray(opaque.using(executor="sequential")(x=1.0))
        with workflow_run(seed=2):
            np.testing.assert_array_equal(
                np.asarray(opaque.using(executor=executor)(x=1.0)), reference
            )

    @pytest.mark.parametrize("executor", _PYTHON_EXECUTORS)
    def test_tuple_and_list_returns_are_held_whole_under_every_executor(self, executor):
        pairs = Function(
            lambda x: (float(x), 2.0 * float(x)), label="pair", traceable=False, max_evaluations=4
        )
        listed = Function(
            lambda x: [float(x), 2.0 * float(x)], label="listed", traceable=False, max_evaluations=4
        )
        law = Normal("x", 0.0, 1.0)
        for returning in (pairs, listed):
            with workflow_run(seed=6):
                reference = returning.using(executor="sequential")(x=law)
            with workflow_run(seed=6):
                result = returning.using(executor=executor)(x=law)
            assert result.num_atoms == 4
            raw_atoms = [list(atom.raw) for atom in result.atoms]
            assert raw_atoms == [list(atom.raw) for atom in reference.atoms]
            assert all(second == 2.0 * first for first, second in raw_atoms)

    @pytest.mark.parametrize(
        "executor",
        [
            "sequential",
            "thread",
            pytest.param(
                "process",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "a chunk sent to a worker process commits its call's place in the "
                        "workflow, so a call that draws nothing moves the keys of later draws"
                    ),
                ),
            ),
        ],
    )
    def test_a_call_that_draws_nothing_leaves_later_draws_unchanged(self, executor):
        sweep = Function(_scaled, label="scaled", traceable=False)
        grid = NumericArrayBatch(jnp.arange(3.0), "theta", label="grid")
        with workflow_run(seed=1):
            reference = np.asarray(sample(Normal("z", 0.0, 1.0)))
        with workflow_run(seed=1):
            sweep.using(executor=executor)(x=grid, scale=2.0)
            after_the_sweep = np.asarray(sample(Normal("z", 0.0, 1.0)))
        np.testing.assert_array_equal(after_the_sweep, reference)


class TestFailures:
    def test_a_failing_draw_of_a_lift_raises_evaluation_error_naming_it(self):
        def fail_above_one(x):
            if float(x) > 1.0:
                raise ValueError("the solver diverged")
            return np.asarray(x)

        failing = Function(fail_above_one, label="solve", traceable=False, max_evaluations=40)
        with workflow_run(seed=1), pytest.raises(EvaluationError) as raised:
            failing(x=Normal("x", 0.0, 1.0))
        error = raised.value
        assert isinstance(error.__cause__, ValueError)
        assert str(error).startswith(
            f"evaluation of 'solve' failed at {len(error.failures)} of 40 "
        )
        assert "draw " in str(error) and "ValueError: the solver diverged" in str(error)
        assert all(failure.kind == "error" for failure in error.failures)
        succeeded = [output for output in error.outputs if output is not None]
        assert len(succeeded) == 40 - len(error.failures)

    def test_a_failing_cell_of_a_sweep_names_its_grid_point(self):
        def fail_at_two(x):
            if float(x) == 2.0:
                raise ValueError("bad cell")
            return np.asarray(x)

        failing = Function(fail_at_two, label="cell", traceable=False)
        with pytest.raises(EvaluationError, match="grid point 2: ValueError: bad cell"):
            failing(x=NumericArrayBatch(jnp.arange(4.0), "x", label="xs"))

    def test_a_plain_call_raises_the_exception_its_body_raised(self):
        def refuse(x):
            raise KeyError("missing input")

        with pytest.raises(KeyError, match="missing input"):
            Function(refuse, label="refuse", executor="thread")(x=1.0)

    def test_a_plain_call_that_names_jax_runs_its_body_in_the_calling_thread(self):
        threads_that_ran = []

        def shifted(x):
            threads_that_ran.append(threading.get_ident())
            return x + 1.0

        traceable = Function(shifted, label="shifted", executor="jax")
        assert float(traceable(x=1.0)) == 2.0
        assert threads_that_ran == [threading.get_ident()]

    def test_naming_jax_for_a_function_that_claims_traceable_false_raises(self):
        opaque = Function(_shift, label="shift", traceable=False, executor="jax")
        with (
            workflow_run(seed=0),
            pytest.raises(
                ResolutionError,
                match=(
                    r"executor 'jax' cannot run the evaluation points of 'shift': JAX cannot "
                    r"trace 'shift': it claims traceable=False\. Name another executor"
                ),
            ),
        ):
            opaque(x=Normal("x", 0.0, 1.0))

    def test_a_worker_count_that_is_not_positive_raises_naming_the_executor_option(self):
        opaque = Function(
            _shift,
            label="shift",
            traceable=False,
            executor="thread",
            executor_options={"max_workers": 0},
        )
        with (
            workflow_run(seed=0),
            pytest.raises(
                ValueError,
                match=r"^executor_options\['max_workers'\] must be a positive integer, got 0$",
            ),
        ):
            opaque(x=Normal("x", 0.0, 1.0))


class TestTheChoiceOfExecutor:
    def test_a_traceable_function_is_mapped_by_jax_whatever_executor_is_named(self):
        traceable = Function(lambda x: x + 1.0, label="shifted", max_evaluations=8)
        with workflow_run(seed=0):
            result = traceable.using(executor="thread", executor_options={"max_workers": 2})(
                x=Normal("x", 0.0, 1.0)
            )
        assert result.provenance.metadata["evaluations"][0]["executor"] == "jax"

    @pytest.mark.parametrize("traceable", [True, False])
    def test_a_plain_call_runs_its_body_on_the_named_executor(self, traceable):
        threads_that_ran = []

        def shifted(x):
            threads_that_ran.append(threading.get_ident())
            return x + 1.0

        named = Function(shifted, label="shifted", traceable=traceable, executor="thread")
        assert float(named(x=1.0)) == 2.0
        assert len(threads_that_ran) == 1
        assert threads_that_ran[0] != threading.get_ident()

    @pytest.mark.parametrize("executor", ["thread", "process"])
    @pytest.mark.parametrize("traceable", [True, None])
    def test_a_traceable_plain_call_inside_a_jax_transformation_ignores_the_executor(
        self, executor, traceable
    ):
        doubled = Function(lambda x: 2.0 * x, label="doubled", traceable=traceable)
        named = doubled.using(executor=executor)
        assert float(jax.grad(lambda x: named(x=x))(3.0)) == 2.0
        np.testing.assert_allclose(
            np.asarray(jax.vmap(lambda x: named(x=x))(jnp.arange(3.0))), [0.0, 2.0, 4.0]
        )
        assert float(jax.jit(lambda x: named(x=x))(1.5)) == 3.0

    def test_the_removed_controls_raise_naming_the_unknown_control(self):
        with pytest.raises(TypeError, match="unknown Function option 'dispatch'"):
            Function(_shift, label="shift", dispatch="thread")
        with pytest.raises(TypeError, match="unknown Function option 'max_workers'"):
            Function(_shift, label="shift").using(max_workers=4)

    def test_an_executor_named_under_prefect_warns_once_that_it_has_no_effect(self, monkeypatch):
        from probpipe.functions import _execution

        recorded_names = []

        def fake_task(*, name, **_):
            recorded_names.append(name)

            def decorate(run):
                class _Mapped:
                    def map(self, *, chunk):
                        return [_Done(run(one_chunk)) for one_chunk in chunk]

                    def __call__(self, *args):
                        return run(*args)

                return _Mapped()

            return decorate

        def fake_flow(*, name, **_):
            recorded_names.append(name)
            return lambda run: run

        class _Done:
            def __init__(self, outcomes):
                self._outcomes = outcomes

            def result(self):
                return self._outcomes

        monkeypatch.setattr(_execution, "task", fake_task)
        monkeypatch.setattr(_execution, "flow", fake_flow)
        opaque = Function(
            _shift,
            label="shift",
            traceable=False,
            workflow_kind=WorkflowKind.TASK,
            executor="thread",
            max_evaluations=3,
        )
        with (
            workflow_run(seed=0),
            pytest.warns(UserWarning, match="executor 'thread' on 'shift' has no effect") as caught,
        ):
            result = opaque(x=Normal("x", 0.0, 1.0))
        assert len([warning for warning in caught if "has no effect" in str(warning.message)]) == 1
        assert set(recorded_names) == {"shift", "shift_map"}
        assert result.provenance.metadata["evaluations"][0]["executor"] == "prefect"


class TestWhereASetCanBeBuilt:
    def test_a_set_built_outside_every_call_is_refused(self):
        with pytest.raises(
            RuntimeError,
            match="cannot build evaluation points of 'total' where no ProbPipe call gives them",
        ):
            EvaluationPoints(lambda x: x, varying={"x": np.arange(3.0)}, origin="total")

    def test_a_set_built_inside_a_function_body_is_refused(self):
        @function(traceable=False)
        def builds_a_set(x):
            return EvaluationPoints(lambda y: y, varying={"y": np.arange(3.0)}, origin="inner")

        with pytest.raises(RuntimeError, match="such as directly in a function's body"):
            builds_a_set(1.0)

    def test_a_traceable_function_making_a_threaded_call_runs_it_inline(self):
        inner = Function(lambda x: x * 2.0, label="inner", executor="thread")
        outer = Function(lambda x: inner(x) + 1.0, label="outer", max_evaluations=8)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with workflow_run(seed=4):
                with_executor = outer(x=Normal("x", 0.0, 1.0))
        plain_inner = Function(lambda x: x * 2.0, label="inner")
        plain_outer = Function(lambda x: plain_inner(x) + 1.0, label="outer", max_evaluations=8)
        with workflow_run(seed=4):
            without_executor = plain_outer(x=Normal("x", 0.0, 1.0))
        np.testing.assert_array_equal(
            np.asarray(with_executor.atoms), np.asarray(without_executor.atoms)
        )
