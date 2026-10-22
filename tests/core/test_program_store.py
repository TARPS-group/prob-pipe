"""Tests for compiled JAX programs: the fallback rule, the program store, its static settings, and notes made while tracing."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Gamma,
    Normal,
    NumericArraySpec,
    conditional_distribution,
    config,
    sample,
    workflow_run,
)
from probpipe.core._call_context import (
    _AUTOMATIC_SELECTION,
    _collecting_call_accumulators,
    _EvaluationSetRecord,
    _ExecutionChoice,
    _note_num_samples_used,
    _under_call_context,
    current_call_context,
)
from probpipe.core._compilation import (
    _eager_decision_keys,
    _program_store_of,
    _ProgramStore,
    _static_settings_in_effect,
    _StaticSettings,
    _StoreEntry,
    _trace_cache_key,
    _TraceNotes,
    compiled_or_eager,
)
from probpipe.core.config import _num_samples_in_effect
from tests._compilation import RecordingProgram, count_compilations

SCALAR = NumericArraySpec(())


def _counted_program(body: Callable[..., Any], python_runs: list) -> RecordingProgram:
    """A ``jax.jit`` object of *body* that adds to *python_runs* each time the body's Python runs, which only a trace does."""

    def counted_body(*args: Any) -> Any:
        python_runs.append(args)
        return body(*args)

    return RecordingProgram(jax.jit(counted_body), runs=[])


def _compiled_or_eager(
    decision_record: dict, decision_key: str, program: Any, eager: Any, *arguments: Any
) -> Any:
    """``compiled_or_eager`` keyed by the trace cache key of all of *arguments*."""
    return compiled_or_eager(
        decision_record,
        decision_key,
        program,
        eager,
        *arguments,
        trace_cache_key=_trace_cache_key(arguments),
    )


def _failing_on_the_host(x: Any) -> Any:
    """*x* passed through a host callback that raises ``ValueError("solver failed")`` when it runs."""

    def solver(value: Any) -> Any:
        raise ValueError("solver failed")

    return jax.pure_callback(
        solver, jax.ShapeDtypeStruct(jnp.shape(x), jnp.float32), x, vmap_method="sequential"
    )


def _joint(kernel: Any) -> Any:
    """``kernel`` over ``mu ~ Normal(0, 1e-3)``, whose kernel the joint maps over the draws of ``mu``."""
    return kernel & Normal("mu", 0.0, 1e-3)


def _rounded_draws(draws: Any) -> list[float]:
    """The draws of ``y``, rounded to integers, which the kernels below shift by a setting."""
    return np.round(np.asarray(draws["y"])).tolist()


class TestTheFallbackRule:
    def test_a_program_that_does_not_trace_runs_eagerly_and_is_not_traced_again(self):
        python_runs: list = []
        program = _counted_program(float, python_runs)
        eager_runs: list = []
        decision_record: dict = {}

        def eager(x):
            eager_runs.append(x)
            return float(x)

        for _ in range(2):
            assert (
                _compiled_or_eager(decision_record, "to float", program, eager, jnp.ones(())) == 1.0
            )
        assert len(python_runs) == 1
        assert len(eager_runs) == 2
        assert _eager_decision_keys(decision_record) == ("to float",)

    def test_an_eager_error_propagates_and_records_nothing(self):
        program = _counted_program(float, [])
        decision_record: dict = {}

        def eager(x):
            raise ValueError("invalid argument")

        with pytest.raises(ValueError, match="invalid argument"):
            _compiled_or_eager(decision_record, "to float", program, eager, jnp.ones(()))
        assert _eager_decision_keys(decision_record) == ()

    def test_a_program_that_traces_and_then_fails_while_running_raises_and_runs_no_eager_retry(
        self,
    ):
        python_runs: list = []
        program = _counted_program(_failing_on_the_host, python_runs)
        eager_runs: list = []

        def eager(x):
            eager_runs.append(x)
            return x

        with pytest.raises(Exception, match="solver failed"):
            _compiled_or_eager({}, "callback", program, eager, jnp.ones(3))
        assert eager_runs == []
        assert len(python_runs) == 1

    def test_a_cold_call_traces_once_and_compiles_once(self):
        python_runs: list = []
        program = _counted_program(lambda x: jnp.sin(x) * 2.0, python_runs)
        x = jnp.ones(3)
        with count_compilations() as compiled:
            _compiled_or_eager({}, "sine", program, jnp.sin, x)
        assert compiled.count == 1
        assert len(python_runs) == 1

    def test_a_new_trace_cache_key_is_traced_before_it_runs(self):
        python_runs: list = []
        traces: list = []
        program = _counted_program(lambda x: x + 1.0, python_runs)
        program.traces = traces
        decision_record: dict = {}
        for x in (jnp.ones(3), jnp.zeros(3), np.ones(3), jnp.ones(4)):
            _compiled_or_eager(decision_record, "add", program, None, x)
        assert len(python_runs) == 2
        assert len(traces) == 2

    def test_a_failed_trace_runs_eagerly_only_at_its_own_shapes(self):
        python_runs: list = []

        def to_number_at_one_entry(x):
            return float(x[0]) + 1.0 if x.shape == (1,) else x + 1.0

        program = _counted_program(to_number_at_one_entry, python_runs)
        decision_record: dict = {}
        for x in (jnp.ones(1), jnp.ones(3), jnp.ones(1), jnp.ones(3)):
            _compiled_or_eager(decision_record, "add", program, to_number_at_one_entry, x)
        # One failed trace at shape (1,), one trace at (3,), and no trace again at either.
        assert [args[0].shape for args in python_runs] == [(1,), (3,)]
        assert _eager_decision_keys(decision_record) == ("add",)

    def test_a_failed_trace_under_one_num_samples_leaves_another_compiled(self):
        traced_counts: list = []

        def body(static_options, x):
            traced_counts.append(config.sampling.num_samples)
            if config.sampling.num_samples == 7:
                return float(x)
            return x + config.sampling.num_samples

        def eager(x):
            if config.sampling.num_samples == 7:
                return float(x)
            return x + config.sampling.num_samples

        program_store = _ProgramStore()
        results = {}
        for count in (7, 9, 7, 9):
            with config.sampling.using(num_samples=count):
                results[count] = program_store.run("body", body, eager, jnp.ones(()))
        assert traced_counts == [7, 9]
        assert results[7] == 1.0
        assert float(results[9]) == 10.0

    def test_a_failed_trace_under_one_set_of_static_options_leaves_another_compiled(self):
        traced_options: list = []

        def body(static_options, x):
            traced_options.append(static_options)
            return float(x) if static_options == "to number" else x + 1.0

        def eager(x):
            return x + 1.0

        program_store = _ProgramStore()
        for static_options in ("to number", "add", "to number", "add"):
            program_store.run("body", body, eager, jnp.ones(()), static_options=static_options)
        assert traced_options == ["to number", "add"]

    def test_without_a_trace_cache_key_each_run_traces_and_nothing_is_recorded(self):
        python_runs: list = []
        program = _counted_program(float, python_runs)
        decision_record: dict = {}
        for _ in range(2):
            compiled_or_eager(
                decision_record, "to float", program, float, jnp.ones(()), trace_cache_key=None
            )
        assert len(python_runs) == 2
        assert decision_record == {}


class TestTheTraceCacheKey:
    def test_values_jax_traces_alike_give_one_key(self):
        assert _trace_cache_key((1.0,)) == _trace_cache_key((jnp.asarray(2.0),))
        assert _trace_cache_key((np.ones(3),)) == _trace_cache_key((jnp.zeros(3),))
        assert _trace_cache_key((2,)) == _trace_cache_key((3,))

    def test_values_jax_traces_apart_give_two_keys(self):
        assert _trace_cache_key((1.0,)) != _trace_cache_key((jnp.float32(1.0),))
        assert _trace_cache_key((jnp.ones(3),)) != _trace_cache_key((jnp.ones(4),))
        assert _trace_cache_key(((3,),), ints_by_value=True) != _trace_cache_key(
            ((4,),), ints_by_value=True
        )

    def test_a_tracer_gives_the_key_of_the_value_it_stands_for(self):
        concrete_key = _trace_cache_key((jnp.ones(3),))
        traced_keys = []
        jax.jit(lambda x: traced_keys.append(_trace_cache_key((x,))) or x)(jnp.ones(3))
        assert traced_keys == [concrete_key]

    def test_a_leaf_neither_an_array_nor_hashable_gives_no_key(self):
        assert _trace_cache_key(([object()], "tag"))[1][-1] == (str, "tag")
        assert _trace_cache_key((jnp.ones(2), {"a": [1]})) is not None
        assert _trace_cache_key((jnp.ones(2), {"a": np.array(["x"])})) is None


def _counting_kernel(read_setting: Callable[[], float], python_runs: list) -> Any:
    """``y | mu ~ Normal(mu + read_setting(), 1e-3)``, counting the runs of its function."""

    @conditional_distribution(given_spec={"mu": SCALAR}, label="lik")
    def lik(mu):
        python_runs.append(1)
        return Normal("y", mu + read_setting(), 1e-3)

    return lik


class TestStaticSettings:
    def test_two_counts_on_the_call_use_two_store_entries_and_their_own_counts(self):
        joint = _joint(_counting_kernel(_num_samples_in_effect, []))
        with workflow_run(seed=0):
            draws_500 = sample.using(num_samples=500)(joint, 4)
            draws_2000 = sample.using(num_samples=2000)(joint, 4)
            with count_compilations() as compiled:
                warm_500 = sample.using(num_samples=500)(joint, 4)
        assert _rounded_draws(draws_500) == [500.0] * 4
        assert _rounded_draws(draws_2000) == [2000.0] * 4
        assert _rounded_draws(warm_500) == [500.0] * 4
        assert compiled.count == 0
        kernel = joint.factors[0]
        assert len(_program_store_of(kernel)._entries) == 2

    def test_a_static_config_value_from_a_block_or_the_session_is_part_of_the_cache_key(self):
        joint = _joint(_counting_kernel(lambda: config.sampling.max_evaluations, []))
        key = jax.random.PRNGKey(0)
        with config.sampling.using(max_evaluations=3):
            in_block = joint._sample(key, (4,))
        session_value = config.sampling.max_evaluations
        try:
            config.sampling.max_evaluations = 5
            from_session = joint._sample(key, (4,))
        finally:
            config.sampling.max_evaluations = session_value
        assert _rounded_draws(in_block) == [3.0] * 4
        assert _rounded_draws(from_session) == [5.0] * 4

    def test_an_inert_setting_is_not_part_of_the_cache_key(self):
        python_runs: list = []
        joint = _joint(_counting_kernel(lambda: 0.0, python_runs))
        key = jax.random.PRNGKey(0)
        joint._sample(key, (4,))
        runs_before = len(python_runs)
        with config.notation.using(max_depth=12):
            joint._sample(key, (4,))
        assert len(python_runs) == runs_before

    def test_a_family_sampler_compiles_once_across_a_sweep_over_num_samples(self):
        law = Gamma("x", 2.0, 3.0)
        with workflow_run(seed=0):
            sample.using(num_samples=100)(law, 8)
            with count_compilations() as compiled:
                sample.using(num_samples=200)(law, 8)
        assert compiled.count == 0

    def test_a_block_the_traced_code_opens_applies_over_the_static_settings(self):
        def read_inside_a_block() -> float:
            with config.sampling.using(max_evaluations=11):
                return config.sampling.max_evaluations

        joint = _joint(_counting_kernel(read_inside_a_block, []))
        with config.sampling.using(max_evaluations=3):
            draws = joint._sample(jax.random.PRNGKey(0), (4,))
        assert _rounded_draws(draws) == [11.0] * 4

    def test_a_trace_jax_starts_on_its_own_sees_the_same_static_settings(self):
        python_runs: list = []
        joint = _joint(_counting_kernel(_num_samples_in_effect, python_runs))
        with workflow_run(seed=0):
            sample.using(num_samples=500)(joint, 4)
            runs_before = len(python_runs)
            jax.clear_caches()
            # The store has traced this entry, so only JAX traces it again, inside the run.
            retraced = sample.using(num_samples=500)(joint, 4)
        assert len(python_runs) > runs_before
        assert _rounded_draws(retraced) == [500.0] * 4

    def test_the_static_settings_hold_the_calls_counts_the_passed_controls_and_the_config(self):
        passed_controls = {"exact_only": True, "max_evaluations": None, "num_samples": 5}
        with config.sampling.using(num_samples=7):
            call_context = current_call_context().child_for_call(
                entry_method_name="__call__",
                function_label="f",
                resolved_controls={"max_evaluations": 3, "num_samples": 9, "method": None},
            )
        with _under_call_context(call_context.with_passed_controls(passed_controls)):
            static_settings = _static_settings_in_effect()
        assert static_settings == _StaticSettings(
            resolved_controls=(("max_evaluations", 3), ("num_samples", 9)),
            passed_controls=tuple(sorted(passed_controls.items())),
            config_values=(
                (
                    "sampling",
                    (("max_evaluations", config.sampling.max_evaluations), ("num_samples", 7)),
                ),
            ),
        )
        hash(static_settings)


class TestTheTracedContext:
    def test_the_static_settings_apply_as_the_innermost_block(self):
        static_settings = _StaticSettings(
            resolved_controls=(("max_evaluations", 9), ("num_samples", 99)),
            passed_controls=(("exact_only", True), ("max_evaluations", None), ("num_samples", 5)),
            config_values=(("sampling", (("max_evaluations", 2), ("num_samples", 3))),),
        )
        running_context = current_call_context().with_config_block(
            {"sampling": {"num_samples": 1}, "notation": {"max_depth": 4}}
        )
        traced_context = running_context.with_static_settings(static_settings)
        assert dict(traced_context.config_blocks["sampling"]) == {
            "max_evaluations": 2,
            "num_samples": 3,
        }
        assert dict(traced_context.config_blocks["notation"]) == {"max_depth": 4}
        assert dict(traced_context.passed_controls) == dict(static_settings.passed_controls)
        assert dict(traced_context.resolved_controls) == {"max_evaluations": 9, "num_samples": 99}

    def test_the_traced_context_chooses_automatically_and_notes_into_its_own_accumulators(self):
        with _collecting_call_accumulators("always") as running_context:
            choosing_context = replace(
                running_context, execution_choice=_ExecutionChoice("thread", {}, "own")
            )
            traced_context = choosing_context.with_static_settings(_static_settings_in_effect())
        assert traced_context.execution_choice == _AUTOMATIC_SELECTION
        assert traced_context.num_samples_used is not running_context.num_samples_used
        assert traced_context.evaluation_log is not running_context.evaluation_log
        assert traced_context.num_samples_used.num_samples_values == frozenset()


def _noting_kernel(noted_count: Callable[[], int], python_runs: list) -> Any:
    """``y | mu ~ Normal(mu, 1)``, whose function notes ``noted_count()`` draws used while it runs."""

    @conditional_distribution(given_spec={"mu": SCALAR}, label="noting")
    def noting(mu):
        python_runs.append(1)
        _note_num_samples_used(noted_count())
        return Normal("y", mu, 1.0)

    return noting


class TestNotesMadeWhileTracing:
    def test_a_cold_and_a_warm_call_report_the_same_num_samples(self):
        python_runs: list = []
        joint = _joint(_noting_kernel(lambda: 37, python_runs))
        with workflow_run(seed=0):
            cold = sample(joint, 4)
            runs_after_cold = len(python_runs)
            warm = sample(joint, 4)
        assert len(python_runs) == runs_after_cold
        assert cold.provenance.metadata["num_samples"] == 37
        assert warm.provenance.metadata["num_samples"] == 37

    def test_the_notes_apply_once_per_run_and_never_while_tracing(self):
        joint = _joint(_noting_kernel(lambda: 37, []))
        key = jax.random.PRNGKey(0)
        for _ in range(2):
            with _collecting_call_accumulators("always") as scope_context:
                joint._sample(key, (4,))
            assert scope_context.num_samples_used.num_samples_values == frozenset({37})

    def test_a_trace_jax_starts_on_its_own_adds_no_note(self):
        joint = _joint(_noting_kernel(lambda: 37, []))
        key = jax.random.PRNGKey(0)
        joint._sample(key, (4,))
        jax.clear_caches()
        with _collecting_call_accumulators("always") as scope_context:
            joint._sample(key, (4,))
        assert scope_context.num_samples_used.num_samples_values == frozenset({37})

    def test_a_note_that_varies_between_traces_of_one_entry_raises(self):
        noted = {"count": 37}
        joint = _joint(_noting_kernel(lambda: noted["count"], []))
        key = jax.random.PRNGKey(0)
        joint._sample(key, (4,))
        noted["count"] = 38
        jax.clear_caches()
        with pytest.raises(RuntimeError, match=r"Monte Carlo draws \[38\] on one trace but \[37\]"):
            joint._sample(key, (4,))

    def test_a_run_that_raises_applies_no_notes(self):
        @conditional_distribution(given_spec={"mu": SCALAR}, label="failing")
        def failing(mu):
            _note_num_samples_used(37)
            return Normal("y", _failing_on_the_host(mu), 1.0)

        joint = _joint(failing)
        with (
            _collecting_call_accumulators("always") as scope_context,
            pytest.raises(Exception, match="solver failed"),
        ):
            joint._sample(jax.random.PRNGKey(0), (4,))
        assert scope_context.num_samples_used.num_samples_values == frozenset()


def _noting_store_run(program_store: Any, *arguments: Any) -> Any:
    """``x + n`` through *program_store*, whose body notes 37 draws used while it is traced."""

    def add_noting(static_options: Any, x: Any, n: Any) -> Any:
        _note_num_samples_used(37)
        return x + n

    return program_store.run("add", add_noting, None, *arguments)


def _noted_by(run: Callable[..., Any], *arguments: Any) -> frozenset[int]:
    """The numbers of draws that ``run(*arguments)`` notes on the scope it runs in."""
    with _collecting_call_accumulators("always") as scope_context:
        run(*arguments)
    return scope_context.num_samples_used.num_samples_values


class TestNotesReachEveryCallThatReusesATrace:
    def test_an_integer_argument_with_a_new_value_reuses_the_trace_and_its_notes(self):
        program_store = _ProgramStore()
        for n in (2, 3, 3):
            assert _noted_by(_noting_store_run, program_store, jnp.ones(3), n) == {37}
        assert len(program_store._entries) == 1

    def test_a_new_weak_type_traces_again_and_both_entries_note(self):
        program_store = _ProgramStore()
        for x in (1.0, jnp.float32(1.0), 2.0):
            assert _noted_by(_noting_store_run, program_store, x, 1) == {37}
        assert len(program_store._entries) == 2

    @pytest.mark.parametrize("concrete_first", [True, False], ids=["concrete-first", "jit-first"])
    def test_a_run_under_an_outer_jit_and_a_concrete_run_share_notes(self, concrete_first):
        program_store = _ProgramStore()

        def concrete_run():
            return _noting_store_run(program_store, jnp.ones(3), 2)

        def jitted_run():
            return jax.jit(lambda x: _noting_store_run(program_store, x, 2))(jnp.zeros(3))

        for run in (concrete_run, jitted_run) if concrete_first else (jitted_run, concrete_run):
            assert _noted_by(run) == {37}

    def test_a_jitted_density_of_a_joint_notes_after_a_concrete_density(self):
        joint = _joint(_noting_kernel(lambda: 37, []))
        value = joint._sample(jax.random.PRNGKey(0), (4,))
        assert _noted_by(lambda: joint._log_prob(value)) == {37}
        assert _noted_by(lambda: jax.jit(joint._log_prob)(value)) == {37}

    def test_runs_on_several_threads_each_apply_the_notes_once(self):
        program_store = _ProgramStore()
        noted_by_thread: list = []

        def run_in_thread() -> None:
            noted_by_thread.append(
                _noted_by(lambda: _noting_store_run(program_store, jnp.ones(3), 2))
            )

        threads = [threading.Thread(target=run_in_thread) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert noted_by_thread == [frozenset({37})] * 8


@dataclass(frozen=True)
class _Retry:
    """An executor option that compares by value."""

    attempts: int


class TestComparingTheNotesOfTwoTraces:
    def _set_record(self, executor_options: dict) -> _EvaluationSetRecord:
        return _EvaluationSetRecord("simulate", 0, "thread", executor_options, 4, ())

    def test_records_whose_arrays_differ_beyond_numpys_display_raise(self):
        weights = np.zeros(2000)
        changed_weights = weights.copy()
        changed_weights[1000] = 1.0
        store_entry = _StoreEntry()
        store_entry.add_trace_notes(
            _TraceNotes(frozenset(), (self._set_record({"weights": weights}),))
        )
        assert repr(weights) == repr(changed_weights)
        with pytest.raises(RuntimeError, match="sets of evaluation points"):
            store_entry.add_trace_notes(
                _TraceNotes(frozenset(), (self._set_record({"weights": changed_weights}),))
            )

    @pytest.mark.parametrize(
        ("first_options", "second_options"),
        [
            ({"chunk": MappingProxyType({"size": 8})}, {"chunk": MappingProxyType({"size": 8})}),
            ({"retry": _Retry(3)}, {"retry": _Retry(3)}),
            ({"timeout": 1}, {"timeout": 1.0}),
            ({"weights": np.zeros(3, np.float32)}, {"weights": jnp.zeros(3)}),
        ],
        ids=["read-only-mapping", "frozen-dataclass", "int-and-float", "numpy-and-jax-array"],
    )
    def test_records_of_equal_but_distinct_options_are_accepted(
        self, first_options, second_options
    ):
        store_entry = _StoreEntry()
        store_entry.add_trace_notes(_TraceNotes(frozenset(), (self._set_record(first_options),)))
        store_entry.add_trace_notes(_TraceNotes(frozenset(), (self._set_record(second_options),)))

    def test_records_of_equal_content_in_another_order_are_accepted(self):
        pool = object()
        first = self._set_record({"weights": np.zeros(2000), "pool": pool})
        second = replace(first, origin="summarize")
        store_entry = _StoreEntry()
        store_entry.add_trace_notes(_TraceNotes(frozenset({5}), (first, second)))
        store_entry.add_trace_notes(
            _TraceNotes(
                frozenset({5}),
                (
                    replace(second, executor_options={"weights": np.zeros(2000), "pool": pool}),
                    replace(first, executor_options={"weights": np.zeros(2000), "pool": pool}),
                ),
            )
        )


class TestTheJointsMappedFactor:
    def test_a_numeric_given_with_a_new_value_or_type_keeps_the_notes(self):
        @conditional_distribution(given_spec={"mu": SCALAR, "offset": SCALAR}, label="lik")
        def likelihood(mu, offset):
            _note_num_samples_used(17)
            return Normal("y", mu + offset, 1.0)

        joint = likelihood & Normal("mu", 0.0, 1.0)
        for offset in (1, 2, 1, 1.0, jnp.asarray(2.0)):
            noted = _noted_by(
                joint._conditional_sample, {"offset": offset}, jax.random.key(0), (3,)
            )
            assert noted == {17}, offset

    def test_a_kernel_that_fails_while_its_program_runs_raises_and_is_not_rerun_eagerly(self):
        python_runs: list = []

        @conditional_distribution(given_spec={"mu": SCALAR}, label="failing")
        def failing(mu):
            python_runs.append(1)
            return Normal("y", _failing_on_the_host(mu), 1.0)

        joint = _joint(failing)
        runs_after_construction = len(python_runs)
        with pytest.raises(Exception, match="solver failed"):
            joint._sample(jax.random.PRNGKey(0), (4,))
        # One trace of the kernel's function, and no eager run under jax.vmap.
        assert len(python_runs) == runs_after_construction + 1
        assert not _program_store_of(failing).runs_eagerly(("mapped", "_conditional_sample"))

    def test_a_kernel_read_outside_every_call_reads_the_session(self):
        joint = _joint(_counting_kernel(_num_samples_in_effect, []))
        with _under_call_context(current_call_context()):
            draws = joint._sample(jax.random.PRNGKey(0), (4,))
        assert _rounded_draws(draws) == [float(config.sampling.num_samples)] * 4
