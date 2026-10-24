"""Tests of what ProbPipe reports about how a call runs its evaluation points (V.9).

``check`` reports, per callable, the number of evaluation points, the executor
that will run them and the executors it rejected, and whether the callable
traces. A call that names an executor that runs nothing warns. A call's
provenance records each set of evaluation points it ran, read from its
evaluation log.
"""

import pickle
import re
import threading
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EvaluationError,
    EvaluationReport,
    ExecutionReport,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    ResolutionError,
    conditional_distribution,
    function,
    mean,
    sample,
    workflow_run,
)
from probpipe.core import _executors
from probpipe.core._dispatch import Feasibility, MethodInfo
from probpipe.core._evaluation_points import EvaluationOutcomes
from probpipe.core._executors import EvaluationExecutor, executor_registry

#: The number of draws each lifted call in these tests takes.
_NUM_DRAWS = 64


@function(traceable=False, output_spec=NumericArraySpec((), "float32"))
def forward(theta):
    """An opaque simulator: it converts its input to NumPy."""
    return np.float32(np.asarray(theta) ** 2)


@function
def growth(theta):
    """A function JAX can trace, which claims nothing."""
    return jnp.sin(theta)


def _prior():
    return Normal("theta", 0.0, 1.0)


def _cells():
    return NumericArrayBatch(jnp.arange(3.0), level_names=("cell",), label="cells")


class _RefusingExecutor(EvaluationExecutor):
    """A user's executor that refuses every set of evaluation points."""

    name = "refuses_everything"
    description = "Refuse every set of evaluation points."

    def check(self, evaluation_points, /):
        return Feasibility(False, description="it refuses every set")

    def execute(self, evaluation_points, /):
        shared_payload = evaluation_points.shared_payload()
        return EvaluationOutcomes.merge(
            chunk.run(shared_payload) for chunk in evaluation_points.chunks()
        )


@pytest.fixture
def _refusing_executor_registered(monkeypatch):
    """The registry holds the built-in executors and one that refuses every set, for one test."""
    registry = _executors._ExecutorRegistry()
    for executor_name in ("jax", "sequential", "thread", "process"):
        registry.register(executor_registry.get_method(executor_name))
    registry.register(_RefusingExecutor())
    monkeypatch.setattr(_executors, "executor_registry", registry)


def _model_with_an_opaque_kernel():
    """A joint whose kernel calls ``forward``, which runs through an embedded call."""
    return _model_whose_kernel_calls(forward)


def _model_whose_kernel_calls(location):
    """A joint whose kernel draws ``y`` around ``location(theta)``."""

    @conditional_distribution(given_spec={"theta": NumericArraySpec((), "float32")})
    def likelihood(theta):
        return Normal("y", location(theta), 1.0)

    return likelihood & _prior()


def _ran_nothing_warnings(caught):
    return [str(warning.message) for warning in caught if "ran nothing" in str(warning.message)]


class TestCheckReportsEachCallable:
    def test_an_opaque_lift_runs_on_sequential_after_jax_refuses_it(self):
        report = forward.using(max_evaluations=_NUM_DRAWS).check(_prior())

        (evaluation,) = report.execution.evaluations
        assert evaluation == EvaluationReport(
            callable_name="forward",
            num_evaluation_points=_NUM_DRAWS,
            traceable=False,
            traceability_source="claimed",
            executor=MethodInfo(True, method_name="sequential", exact=True),
            rejected=(
                MethodInfo(
                    False,
                    "JAX cannot trace 'forward': it claims traceable=False",
                    method_name="jax",
                    exact=True,
                ),
            ),
            failure_rule="raise",
        )

    def test_a_traceable_lift_is_probed_and_compiled_by_jax(self):
        (evaluation,) = (
            growth.using(max_evaluations=_NUM_DRAWS).check(_prior()).execution.evaluations
        )

        assert evaluation.traceable is True
        assert evaluation.traceability_source == "probed"
        assert evaluation.executor == MethodInfo(True, method_name="jax", exact=True)
        assert evaluation.rejected == ()

    def test_a_named_executor_runs_an_opaque_lift(self):
        report = forward.using(executor="thread", executor_options={"max_workers": 2}).check(
            _prior()
        )

        (evaluation,) = report.execution.evaluations
        assert evaluation.executor.method_name == "thread"
        assert [info.method_name for info in evaluation.rejected] == ["jax"]

    def test_an_executor_named_for_the_callable_that_cannot_run_it_is_reported_infeasible(self):
        unpicklable_lock = threading.Lock()

        @function(traceable=False, output_spec=NumericArraySpec((), "float32"))
        def locked_forward(theta):
            with unpicklable_lock:
                return np.float32(np.asarray(theta))

        checked = locked_forward.using(executor="process").check(_prior())

        (evaluation,) = checked.execution.evaluations
        assert evaluation.executor.feasible is False
        assert evaluation.executor.method_name == "process"
        assert "cannot be pickled" in evaluation.executor.description
        with workflow_run(seed=0), pytest.raises(ResolutionError, match="cannot be pickled"):
            locked_forward.using(executor="process")(_prior())

    def test_a_sweep_reports_one_evaluation_point_per_grid_point(self):
        (evaluation,) = forward.check(_cells()).execution.evaluations

        assert evaluation.num_evaluation_points == 3
        assert evaluation.executor.method_name == "sequential"

    def test_a_sweep_of_lifts_reports_every_draw_of_every_cell(self):
        @function(traceable=False, output_spec=NumericArraySpec((), "float32"))
        def shifted(theta, offset):
            return np.float32(np.asarray(theta) + np.asarray(offset))

        (evaluation,) = (
            shifted.using(max_evaluations=_NUM_DRAWS)
            .check(_prior(), _cells())
            .execution.evaluations
        )

        assert evaluation.num_evaluation_points == 3 * _NUM_DRAWS

    def test_a_sweep_over_no_cells_reports_no_set(self):
        no_cells = NumericArrayBatch(jnp.zeros((0,)), level_names=("cell",), label="no_cells")

        assert forward.using(executor="thread").check(no_cells).execution == ExecutionReport()

    def test_a_plain_call_reports_no_set_of_evaluation_points(self):
        assert growth.check(2.0).execution == ExecutionReport()
        assert forward.using(executor="thread").check(2.0).execution == ExecutionReport()


class TestTheReprShowsExecutionOnlyBeyondCompiledCode:
    def test_an_opaque_callable_shows_the_execution_report(self):
        assert "execution=ExecutionReport(" in repr(forward.check(_prior()))

    def test_a_lift_compiled_by_jax_shows_none(self):
        assert "execution" not in repr(growth.check(_prior()))

    def test_an_operation_over_an_all_jax_model_shows_none(self):
        assert "execution" not in repr(mean.check(_prior()))

    def test_the_report_reads_as_a_constructor_call(self):
        text = repr(forward.using(max_evaluations=_NUM_DRAWS).check(_prior()).execution)

        assert text.startswith("ExecutionReport(")
        assert "EvaluationReport(" in text
        assert f"num_evaluation_points={_NUM_DRAWS}" in text
        assert "failure_rule='raise'" in text


class TestAnExecutorThatRanNothingWarns:
    def test_a_traceable_lift_naming_an_executor_warns_once(self):
        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            growth.using(executor="thread", max_evaluations=_NUM_DRAWS)(_prior())

        assert _ran_nothing_warnings(caught) == [
            "growth named executor='thread', but every callable in the call is traceable "
            "and was compiled by jax, so 'thread' ran nothing"
        ]

    def test_the_warning_points_at_the_caller(self):
        with workflow_run(seed=0), pytest.warns(UserWarning, match="ran nothing") as caught:
            growth.using(executor="thread", max_evaluations=_NUM_DRAWS)(_prior())

        (warning,) = [record for record in caught if "ran nothing" in str(record.message)]
        assert warning.filename == __file__

    def test_a_decorator_executor_on_a_traceable_function_warns(self):
        @function(executor="sequential")
        def compiled_anyway(theta):
            return jnp.cos(theta)

        with workflow_run(seed=0), pytest.warns(UserWarning, match=re.escape("'sequential' ran")):
            compiled_anyway.using(max_evaluations=_NUM_DRAWS)(_prior())

    def test_an_executor_that_ran_a_set_does_not_warn(self):
        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forward.using(executor="thread", max_evaluations=_NUM_DRAWS)(_prior())
            forward.using(executor="thread")(2.0)
            growth.using(executor="jax")(2.0)

        assert _ran_nothing_warnings(caught) == []

    def test_a_call_naming_no_executor_does_not_warn(self):
        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            growth.using(max_evaluations=_NUM_DRAWS)(_prior())
            growth.using(executor_options={})(_prior())

        assert _ran_nothing_warnings(caught) == []

    def test_check_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            growth.using(executor="thread").check(_prior())

        assert _ran_nothing_warnings(caught) == []

    def test_a_call_that_raises_does_not_warn(self):
        @function(traceable=True)
        def failing(theta):
            raise ValueError("r must be positive")

        with (
            workflow_run(seed=0),
            warnings.catch_warnings(record=True) as caught,
            pytest.raises(EvaluationError),
        ):
            warnings.simplefilter("always")
            failing.using(executor="thread", max_evaluations=_NUM_DRAWS)(_prior())

        assert _ran_nothing_warnings(caught) == []

    def test_an_operation_that_evaluates_no_user_callable_warns(self):
        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mean.using(executor="thread")(_prior())

        assert _ran_nothing_warnings(caught) == [
            "mean named executor='thread', but the call evaluated no user callable, "
            "so 'thread' ran nothing"
        ]

    def test_an_operation_whose_embedded_call_ran_on_its_executor_does_not_warn(self):
        model = _model_with_an_opaque_kernel()

        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample.using(executor="thread")(model, 3)

        assert _ran_nothing_warnings(caught) == []

    @pytest.mark.usefixtures("_refusing_executor_registered")
    def test_a_fallback_warns_once_and_no_more(self):
        model = _model_with_an_opaque_kernel()

        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample.using(executor="refuses_everything")(model, 3)

        messages = [str(warning.message) for warning in caught]
        assert any("they run on 'sequential' instead" in message for message in messages)
        assert _ran_nothing_warnings(caught) == []

    @pytest.mark.parametrize("location", [jnp.sin, growth], ids=["jnp", "traceable_function"])
    def test_an_operation_over_an_all_jax_model_warns_that_jax_compiled_it(self, location):
        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample.using(executor="thread")(_model_whose_kernel_calls(location), 3)

        assert _ran_nothing_warnings(caught) == [
            "sample named executor='thread', but every callable in the call is traceable "
            "and was compiled by jax, so 'thread' ran nothing"
        ]

    @pytest.mark.parametrize("own_executor", ["thread", "sequential"])
    def test_a_callable_that_names_its_own_executor_leaves_the_operation_silent(self, own_executor):
        model = _model_whose_kernel_calls(forward.using(executor=own_executor))

        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample.using(executor="thread")(model, 3)

        assert _ran_nothing_warnings(caught) == []

    def test_a_call_opened_while_a_kernel_is_traced_does_not_warn(self):
        @conditional_distribution(given_spec={"theta": NumericArraySpec((), "float32")})
        def likelihood(theta):
            return Normal("y", growth.using(executor="thread")(theta), 1.0)

        with workflow_run(seed=0), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sample(likelihood & _prior(), 3)

        assert _ran_nothing_warnings(caught) == []

    def test_a_plain_call_inside_a_jax_transformation_warns_that_it_ran_there(self):
        def doubled(x):
            return 2.0 * x

        on_thread = function(doubled).using(executor="thread")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jax.grad(lambda x: on_thread(x=x))(3.0)

        assert _ran_nothing_warnings(caught) == [
            "doubled named executor='thread', but the call ran inside a JAX transformation, "
            "which runs its body in the calling thread, so 'thread' ran nothing"
        ]


class TestProvenanceRecordsEachSetOfEvaluationPoints:
    def test_a_lift_records_its_executor_options_and_count(self):
        with workflow_run(seed=0):
            draws = forward.using(
                executor="thread", executor_options={"max_workers": 2}, max_evaluations=_NUM_DRAWS
            )(_prior())

        assert draws.provenance.metadata["evaluations"] == [
            {
                "callable": "forward",
                "executor": "thread",
                "executor_options": {"max_workers": 2},
                "num_evaluation_points": _NUM_DRAWS,
                "failed_point_indices": [],
            }
        ]
        assert "dispatch" not in draws.provenance.metadata

    def test_a_lift_compiled_by_jax_records_jax(self):
        with workflow_run(seed=0):
            draws = growth.using(max_evaluations=_NUM_DRAWS)(_prior())

        (evaluations,) = draws.provenance.metadata["evaluations"]
        assert evaluations["executor"] == "jax"
        assert evaluations["executor_options"] == {}

    def test_a_sweep_records_its_set_of_grid_points(self):
        with workflow_run(seed=0):
            swept = forward(_cells())

        assert swept.provenance.metadata["evaluations"] == [
            {
                "callable": "forward",
                "executor": "sequential",
                "executor_options": {},
                "num_evaluation_points": 3,
                "failed_point_indices": [],
            }
        ]

    def test_a_sweep_of_lifts_records_one_set_per_cell(self):
        @function(traceable=False, output_spec=NumericArraySpec((), "float32"))
        def shifted(theta, offset):
            return np.float32(np.asarray(theta) + np.asarray(offset))

        with workflow_run(seed=0):
            swept = shifted.using(max_evaluations=_NUM_DRAWS)(_prior(), _cells())

        evaluations = swept.provenance.metadata["evaluations"]
        assert [entry["num_evaluation_points"] for entry in evaluations] == [_NUM_DRAWS] * 3


class TestAFailedMapStatesItsErrorOnce:
    def test_one_error_that_fails_every_mapped_draw_is_stated_once(self):
        @function(traceable=True)
        def failing(theta):
            raise ValueError("r must be positive")

        with workflow_run(seed=0), pytest.raises(EvaluationError) as raised:
            failing.using(max_evaluations=_NUM_DRAWS)(_prior())

        assert str(raised.value) == (
            f"evaluation of 'failing' failed at all {_NUM_DRAWS} draws, which jax.vmap "
            "evaluates together: ValueError: r must be positive"
        )
        assert len(raised.value.failures) == _NUM_DRAWS
        assert isinstance(raised.value.__cause__, ValueError)

    def test_separate_failures_on_a_python_executor_are_listed(self):
        @function(traceable=False, output_spec=NumericArraySpec((), "float32"))
        def failing(theta):
            raise ValueError("r must be positive")

        with workflow_run(seed=0), pytest.raises(EvaluationError) as raised:
            failing.using(max_evaluations=_NUM_DRAWS)(_prior())

        assert str(raised.value).startswith(
            f"evaluation of 'failing' failed at {_NUM_DRAWS} of {_NUM_DRAWS} evaluation points: "
            "draw 0: ValueError: r must be positive; draw 1:"
        )


class TestTheRecordOfANamedExecutor:
    def test_a_pickled_choice_arrives_without_its_record(self):
        from probpipe.core._call_context import _ExecutionChoice, _NamedExecutorUse

        named_choice = _ExecutionChoice(
            executor_name="thread", source="own", named_executor_use=_NamedExecutorUse()
        )
        unpickled_choice = pickle.loads(pickle.dumps(named_choice))
        assert unpickled_choice == named_choice
        assert unpickled_choice.named_executor_use is None

    def test_an_inner_choice_tells_the_enclosing_record(self):
        from probpipe.core._call_context import _NamedExecutorUse

        enclosing_use = _NamedExecutorUse()
        inner_use = _NamedExecutorUse(enclosing=enclosing_use)
        inner_use.note_ran_a_set()

        assert inner_use.ran_a_set
        assert enclosing_use.ran_elsewhere
        assert not enclosing_use.ran_a_set
