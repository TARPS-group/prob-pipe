"""Tests for the executor registry, its built-in executors, and the choice of an executor (V.9).

Each test runs a set of evaluation points from where a lifted call runs one:
inside a call of a ``Function``, with the call's context open and a key broker
of its own, through the one dispatcher, ``_run_on_chosen_executor``.
"""

import threading
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EvaluationError,
    EvaluationPoints,
    Normal,
    NumericArraySpec,
    OutputSpec,
    ResolutionError,
    UnshippableSettingError,
    WorkflowKind,
    condition_on,
    config,
    executor_registry,
    function,
    registry_catalog,
    sample,
    workflow_run,
)
from probpipe.core._call_context import (
    _note_num_samples_used,
    _open_call_context,
    _under_call_context,
    current_call_context,
)
from probpipe.core._execution_claims import ExecutionClaims
from probpipe.core._executors import _PROCESS_POOLS, _run_on_chosen_executor
from probpipe.functions._execution import _requester_workflow_scope
from probpipe.values._function_base import _child_context_for_call


@contextmanager
def requester_call(function_view):
    """The engine's place in a call of *function_view*: the call's context open, with a key broker of its own."""
    child_context = _child_context_for_call(
        function_view, current_call_context(), entry_method_name="__call__"
    )
    with (
        _open_call_context(child_context, entry_method_name="__call__"),
        _requester_workflow_scope(),
    ):
        yield current_call_context()


def _requester(**controls):
    """A Function to stand as the requester, with *controls* as its own."""
    requester = function(lambda: None, label="requester")
    return requester.using(**controls) if controls else requester


def _evaluation_points(fn, num_points=6, **options):
    return EvaluationPoints(
        fn,
        varying={"x": np.arange(2.0 * num_points).reshape(num_points, 2)},
        shared={"scale": 2.0},
        origin=options.pop("origin", "simulate"),
        **options,
    )


def _opaque_sum(x, scale):
    return scale * np.sum(np.asarray(x))


def _traceable_sum(x, scale):
    return scale * jnp.sum(x)


def _simulate_with_a_nested_draw(x, scale):
    noise = np.asarray(sample(Normal("noise", 0.0, 1.0)))
    return scale * np.asarray(x) + noise


def _run_set(fn, *, requester_controls=None, seed=0, **point_options):
    """The outputs of a set of evaluation points of *fn*, and the call's record of it."""
    with (
        workflow_run(seed=seed),
        requester_call(_requester(**(requester_controls or {}))) as call_context,
    ):
        outputs = _run_on_chosen_executor(_evaluation_points(fn, **point_options))
        return outputs, call_context.evaluation_log.set_records[-1]


class TestTheRegistry:
    def test_it_holds_the_four_built_in_executors_in_selection_order(self):
        assert executor_registry.list_methods() == ["jax", "sequential", "thread", "process"]

    def test_the_registry_catalog_lists_it(self):
        assert registry_catalog["executors"] is executor_registry

    def test_jax_and_sequential_are_selected_automatically_and_thread_and_process_only_by_name(
        self,
    ):
        priorities = {
            name: executor_registry.get_method(name).priority
            for name in executor_registry.list_methods()
        }
        assert priorities == {"jax": 20, "sequential": 0, "thread": None, "process": None}


class TestAutomaticSelection:
    def test_an_opaque_callable_runs_on_sequential(self):
        _, set_record = _run_set(_opaque_sum)
        assert set_record.executor_name == "sequential"

    def test_a_traceable_callable_is_compiled_by_jax(self):
        outputs, set_record = _run_set(_traceable_sum, claims=ExecutionClaims(traceable=True))
        assert set_record.executor_name == "jax"
        np.testing.assert_allclose(np.asarray(outputs), [2.0, 10.0, 18.0, 26.0, 34.0, 42.0])

    def test_a_callable_whose_traceability_is_unknown_runs_as_opaque(self):
        _, set_record = _run_set(_traceable_sum)
        assert set_record.executor_name == "sequential"

    def test_jax_never_runs_a_callable_that_returns_a_distribution(self):
        law_spec = Normal("y", 0.0, 1.0).spec
        _, set_record = _run_set(
            lambda x, scale: Normal("y", 0.0, 1.0),
            claims=ExecutionClaims(traceable=True),
            output_spec=OutputSpec(law_spec),
        )
        assert set_record.executor_name == "sequential"


class TestNamingAnExecutor:
    def test_a_named_executor_runs_the_set_with_its_options(self):
        _, set_record = _run_set(
            _opaque_sum,
            requester_controls={"executor": "thread", "executor_options": {"max_workers": 2}},
        )
        assert (set_record.executor_name, set_record.executor_options) == (
            "thread",
            {"max_workers": 2},
        )

    def test_an_unknown_executor_raises_at_the_call_and_using_accepts_it(self):
        named_unknown = _requester(executor="threads")
        with pytest.raises(ResolutionError) as raised, requester_call(named_unknown):
            pass
        assert str(raised.value) == (
            "unknown executor 'threads'; available executors: "
            "['jax', 'process', 'sequential', 'thread']"
        )

    def test_an_option_the_executor_does_not_read_raises_at_the_call(self):
        with (
            pytest.raises(ValueError) as raised,
            requester_call(_requester(executor="thread", executor_options={"n_jobs": 50})),
        ):
            pass
        assert str(raised.value) == (
            "executor 'thread' does not read the option 'n_jobs'; its options are chunk_size, max_workers"
        )

    def test_jax_refuses_a_batch_size_until_compiled_lifting(self):
        with (
            pytest.raises(ValueError, match="its options are none"),
            requester_call(_requester(executor="jax", executor_options={"batch_size": 8})),
        ):
            pass

    def test_options_without_an_executor_go_to_the_automatically_selected_one(self):
        _, set_record = _run_set(
            _opaque_sum, requester_controls={"executor_options": {"chunk_size": 2}}
        )
        assert (set_record.executor_name, set_record.executor_options) == (
            "sequential",
            {"chunk_size": 2},
        )

    def test_an_operation_given_options_without_an_executor_raises_at_the_call(self):
        with (
            pytest.raises(ValueError) as raised,
            requester_call(condition_on.using(executor_options={"max_workers": 2})),
        ):
            pass
        assert str(raised.value) == (
            "condition_on was given executor_options without an executor. Name the executor "
            "the options belong to, as executor='thread'"
        )

    def test_a_refusing_executor_named_for_the_callable_raises(self):
        lock = threading.Lock()

        def holds_a_lock(x, scale):
            with lock:
                return scale * x

        with pytest.raises(ResolutionError) as raised:
            _run_set(holds_a_lock, requester_controls={"executor": "process"})
        assert str(raised.value).startswith(
            "executor 'process' cannot run the evaluation points of 'simulate': its callable "
            "cannot be pickled"
        )
        assert str(raised.value).endswith(". Name another executor")

    def test_a_traceable_callable_is_compiled_by_jax_and_receives_no_named_options(self):
        _, set_record = _run_set(
            _traceable_sum,
            requester_controls={"executor": "thread", "executor_options": {"max_workers": 2}},
            claims=ExecutionClaims(traceable=True),
        )
        assert (set_record.executor_name, set_record.executor_options) == ("jax", {})


class TestInheritingAnExecutor:
    def test_a_call_inherits_the_executor_an_enclosing_call_names(self):
        with requester_call(_requester(executor="thread")), requester_call(_requester()) as inner:
            assert inner.execution_choice.executor_name == "thread"
            assert inner.execution_choice.source == "inherited"

    def test_an_executor_set_on_the_function_wins_and_takes_only_its_own_options(self):
        outer = _requester(executor="thread", executor_options={"max_workers": 2})
        inner = _requester(executor_options={"chunk_size": 3})
        with requester_call(outer), requester_call(inner) as inner_context:
            choice = inner_context.execution_choice
            assert (choice.executor_name, dict(choice.executor_options), choice.source) == (
                None,
                {"chunk_size": 3},
                "own",
            )

    def test_using_executor_none_restores_inheritance(self):
        restored = _requester(executor="sequential").using(executor=None)
        with requester_call(_requester(executor="thread")), requester_call(restored) as inner:
            assert inner.execution_choice.executor_name == "thread"

    def test_naming_sequential_opts_out_of_an_inherited_executor(self):
        opted_out = _requester(executor="sequential")
        with requester_call(_requester(executor="thread")), requester_call(opted_out) as inner:
            assert (inner.execution_choice.executor_name, inner.execution_choice.source) == (
                "sequential",
                "own",
            )

    def test_an_inherited_executor_that_cannot_run_a_set_falls_back_and_warns_once_per_call(self):
        lock = threading.Lock()

        def holds_a_lock(x, scale):
            with lock:
                return scale * x

        with (
            workflow_run(seed=0),
            requester_call(
                condition_on.using(executor="process", executor_options={"max_workers": 2})
            ) as call_context,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            _run_on_chosen_executor(_evaluation_points(holds_a_lock))
            _run_on_chosen_executor(_evaluation_points(holds_a_lock))
        fallback_warnings = [
            warning for warning in caught if "inherited from" in str(warning.message)
        ]
        assert len(fallback_warnings) == 1
        assert str(fallback_warnings[0].message).startswith(
            "executor 'process', inherited from condition_on, cannot run the evaluation points of "
            "'simulate' (its callable cannot be pickled"
        )
        assert [
            (set_record.executor_name, set_record.executor_options)
            for set_record in call_context.evaluation_log.set_records
        ] == [("sequential", {}), ("sequential", {})]

    def test_a_call_inside_an_evaluation_point_chooses_its_executor_afresh(self):
        def read_the_nested_choice(x, scale):
            with requester_call(_requester()) as nested_context:
                return nested_context.execution_choice.source

        outputs, _ = _run_set(read_the_nested_choice, requester_controls={"executor": "thread"})
        assert set(outputs) == {"automatic"}


class TestTheTwoControls:
    def test_setting_the_executor_resets_the_options(self):
        view = _requester(executor="thread", executor_options={"max_workers": 2}).using(
            executor="sequential"
        )
        assert (view.options["executor"], dict(view.options["executor_options"])) == (
            "sequential",
            {},
        )

    def test_setting_the_options_resets_the_executor(self):
        view = _requester(executor="thread").using(executor_options={"chunk_size": 2})
        assert (view.options["executor"], dict(view.options["executor_options"])) == (
            None,
            {"chunk_size": 2},
        )

    @pytest.mark.parametrize(
        ("controls", "message"),
        [
            ({"executor": ""}, "executor must be a non-empty string or None"),
            ({"executor_options": [1]}, "executor_options must map option names"),
        ],
    )
    def test_using_checks_only_the_form_of_their_values(self, controls, message):
        with pytest.raises(TypeError, match=message):
            _requester(**controls)

    def test_the_resolved_controls_of_a_call_leave_them_to_its_executor_choice(self):
        with requester_call(_requester(executor="thread")) as call_context:
            assert not {"executor", "executor_options"} & set(call_context.resolved_controls)
            assert call_context.execution_choice.executor_name == "thread"


class TestAgreement:
    @pytest.mark.parametrize("executor_name", ["sequential", "thread", "process"])
    def test_every_executor_draws_the_same_nested_values_at_one_seed(self, executor_name):
        reference_outputs, _ = _run_set(_simulate_with_a_nested_draw, seed=11)
        outputs, set_record = _run_set(
            _simulate_with_a_nested_draw,
            seed=11,
            requester_controls={"executor": executor_name, "executor_options": {"max_workers": 2}}
            if executor_name != "sequential"
            else {"executor": "sequential"},
        )
        assert set_record.executor_name == executor_name
        for output, reference_output in zip(outputs, reference_outputs, strict=True):
            np.testing.assert_array_equal(output, reference_output)

    def test_jax_agrees_with_sequential_within_tolerance(self):
        jax_outputs, _ = _run_set(_traceable_sum, claims=ExecutionClaims(traceable=True))
        sequential_outputs, _ = _run_set(
            _traceable_sum,
            claims=ExecutionClaims(traceable=True),
            requester_controls={"executor": "sequential"},
        )
        np.testing.assert_allclose(
            np.asarray(jax_outputs), np.asarray(sequential_outputs), rtol=1e-6
        )

    def test_a_worker_process_computes_under_the_parents_64_bit_floats(self):
        x64_before = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            outputs, _ = _run_set(
                lambda x, scale: jnp.asarray(x) * scale,
                requester_controls={"executor": "process", "executor_options": {"max_workers": 2}},
            )
        finally:
            jax.config.update("jax_enable_x64", x64_before)
        assert {np.asarray(output).dtype for output in outputs} == {np.dtype("float64")}


class TestTheCallContextInWorkers:
    def _read_in_workers(self, read_setting, executor_name):
        outputs, _ = _run_set(
            read_setting,
            requester_controls={"executor": executor_name, "executor_options": {"max_workers": 2}},
        )
        return set(outputs)

    @pytest.mark.parametrize("executor_name", ["thread", "process"])
    def test_a_session_value_set_before_the_call_reaches_every_worker(self, executor_name):
        session_num_samples = config.sampling.num_samples
        config.sampling.num_samples = 4096
        try:
            read = self._read_in_workers(
                lambda x, scale: config.sampling.num_samples, executor_name
            )
        finally:
            config.sampling.num_samples = session_num_samples
        assert read == {4096}

    def test_a_worker_process_runs_without_the_prefect_section(self):
        def read_workflow_kind(x, scale):
            return config.prefect.workflow_kind

        with config.prefect.using(workflow_kind=WorkflowKind.TASK):
            in_threads = self._read_in_workers(read_workflow_kind, "thread")
            in_processes = self._read_in_workers(read_workflow_kind, "process")
        assert in_threads == {WorkflowKind.TASK}
        assert WorkflowKind.TASK not in in_processes

    def test_a_setting_that_does_not_pickle_raises_only_when_a_worker_reads_it(self):
        lock = threading.Lock()

        def read_max_depth(x, scale):
            return config.notation.max_depth

        with (
            workflow_run(seed=0),
            requester_call(_requester(executor="process", executor_options={"max_workers": 2})),
        ):
            unshippable_context = current_call_context().with_config_block(
                {"notation": {"max_depth": lock}}
            )
            with _under_call_context(unshippable_context):
                with pytest.raises(EvaluationError) as raised:
                    _run_on_chosen_executor(_evaluation_points(read_max_depth, num_points=2))
                assert _run_on_chosen_executor(_evaluation_points(_opaque_sum, num_points=2))
        failure = raised.value.failures[0]
        assert failure.error_type == "UnshippableSettingError"
        assert isinstance(failure.error, UnshippableSettingError)
        assert failure.message.startswith(
            "config.notation.max_depth holds a lock that could not be sent to this worker process"
        )

    def test_the_draws_counted_in_a_worker_process_reach_the_call(self):
        def note_draws(x, scale):
            _note_num_samples_used(7)
            return scale

        with (
            workflow_run(seed=0),
            requester_call(
                _requester(executor="process", executor_options={"max_workers": 2})
            ) as call_context,
        ):
            _run_on_chosen_executor(_evaluation_points(note_draws))
            assert call_context.num_samples_used.num_samples_values == frozenset({7})

    def test_a_process_pool_is_reused_by_calls_under_equal_jax_settings(self):
        controls = {"executor": "process", "executor_options": {"max_workers": 2}}
        _run_set(_opaque_sum, requester_controls=controls)
        pools_after_one_call = dict(_PROCESS_POOLS)
        _run_set(_opaque_sum, requester_controls=controls)
        assert pools_after_one_call == _PROCESS_POOLS


class TestFailures:
    def test_a_failure_raises_evaluation_error_chained_to_the_users_exception(self):
        def fail_at_two(x, scale):
            if x[0] == 4.0:
                raise ValueError("the solver diverged")
            return scale * np.sum(x)

        with pytest.raises(EvaluationError) as raised:
            _run_set(fail_at_two)
        error = raised.value
        assert str(error) == (
            "evaluation of 'simulate' failed at 1 of 6 evaluation points: "
            "evaluation point 2: ValueError: the solver diverged"
        )
        assert isinstance(error.__cause__, ValueError)
        assert error.outputs[3] == 26.0
        assert "fail_at_two" in [
            frame.name for frame in traceback.extract_tb(error.__cause__.__traceback__)
        ]

    def test_a_failure_in_a_worker_process_keeps_the_users_exception(self):
        def fail_everywhere(x, scale):
            raise KeyError("missing input file")

        with pytest.raises(EvaluationError) as raised:
            _run_set(
                fail_everywhere,
                num_points=2,
                requester_controls={"executor": "process", "executor_options": {"max_workers": 2}},
            )
        assert isinstance(raised.value.__cause__, KeyError)
        assert [failure.kind for failure in raised.value.failures] == ["error", "error"]

    def test_an_output_of_the_wrong_shape_fails_its_evaluation_point_only(self):
        def wrong_at_one(x, scale):
            return np.zeros(3) if x[0] == 2.0 else scale * x

        with pytest.raises(EvaluationError) as raised:
            _run_set(wrong_at_one, output_spec=OutputSpec(y=NumericArraySpec((2,), "float64")))
        assert [(failure.point_index, failure.kind) for failure in raised.value.failures] == [
            (1, "invalid")
        ]

    def test_a_cast_that_loses_precision_warns_once_per_call(self):
        def float64_output(x, scale):
            return np.asarray(scale * x, dtype=np.float64)

        output_spec = OutputSpec(y=NumericArraySpec((2,), "float32"))
        with (
            workflow_run(seed=0),
            requester_call(_requester()),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            for _ in range(2):
                _run_on_chosen_executor(_evaluation_points(float64_output, output_spec=output_spec))
        cast_warnings = [warning for warning in caught if "lost precision" in str(warning.message)]
        assert len(cast_warnings) == 1
        assert str(cast_warnings[0].message) == (
            "'simulate' returned float64 outputs where its output spec declares float32, so they "
            "were cast and lost precision. Return float32, or declare the dtype the callable returns"
        )


class TestTheEvaluationLog:
    def test_it_records_each_set_with_its_executor_its_size_and_its_failures(self):
        def fail_at_zero(x, scale):
            if x[0] == 0.0:
                raise ValueError("bad")
            return scale

        with workflow_run(seed=0), requester_call(_requester()) as call_context:
            _run_on_chosen_executor(_evaluation_points(_opaque_sum, num_points=3))
            with pytest.raises(EvaluationError):
                _run_on_chosen_executor(_evaluation_points(fail_at_zero, num_points=4))
            _run_on_chosen_executor(_evaluation_points(_opaque_sum, num_points=2, origin="other"))
        assert [
            (
                set_record.origin,
                set_record.set_ordinal,
                set_record.num_evaluation_points,
                set_record.num_runs,
                set_record.failed_point_indices,
            )
            for set_record in call_context.evaluation_log.set_records
        ] == [("simulate", 0, 3, 3, ()), ("simulate", 1, 4, 4, (0,)), ("other", 0, 2, 2, ())]

    def test_a_failed_evaluation_point_names_its_set_in_its_identity(self):
        def fail_at_zero(x, scale):
            raise ValueError("bad")

        with workflow_run(seed=0), requester_call(_requester()):
            _run_on_chosen_executor(_evaluation_points(_opaque_sum, num_points=1))
            with pytest.raises(EvaluationError) as raised:
                _run_on_chosen_executor(_evaluation_points(fail_at_zero, num_points=1))
        assert raised.value.failures[0].point_identity == ("simulate", 1, 0)

    def test_an_evaluation_context_holds_automatic_selection(self):
        with requester_call(_requester(executor="thread")) as call_context:
            assert call_context.evaluation_context().execution_choice.source == "automatic"
            assert (
                replace(call_context, in_evaluation_point=True) != call_context.evaluation_context()
            )
