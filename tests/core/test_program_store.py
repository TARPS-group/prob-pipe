"""Tests for compiled JAX programs: the fallback rule, the program store, its static settings, and notes made while tracing."""

from collections.abc import Callable
from dataclasses import replace
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
    _ExecutionChoice,
    _note_num_samples_used,
    _under_call_context,
    current_call_context,
)
from probpipe.core._compilation import (
    _program_store_of,
    _static_settings_in_effect,
    _StaticSettings,
    compiled_or_eager,
)
from probpipe.core.config import _num_samples_in_effect
from tests._compilation import count_compilations

SCALAR = NumericArraySpec(())


class _CountingProgram:
    """A ``jax.jit`` object of *body*, counting the runs of the body's Python, which only a trace makes."""

    def __init__(self, body: Callable[..., Any], **jit_options: Any) -> None:
        self.python_runs = 0

        def counted_body(*args: Any, **kwargs: Any) -> Any:
            self.python_runs += 1
            return body(*args, **kwargs)

        self.jitted = jax.jit(counted_body, **jit_options)

    def trace(self, *args: Any, **kwargs: Any) -> Any:
        return self.jitted.trace(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.jitted(*args, **kwargs)


def _failing_on_the_host(x: Any) -> Any:
    """*x* passed through a host callback that raises ``ValueError("solver failed")`` when it runs."""

    def solver(value: Any) -> Any:
        raise ValueError("solver failed")

    return jax.pure_callback(
        solver, jax.ShapeDtypeStruct(jnp.shape(x), jnp.float32), x, vmap_method="sequential"
    )


def _joint(kernel: Any) -> Any:
    """``kernel`` over ``mu ~ Normal(0, 1e-3)``, whose kernel the joint maps over the draws of ``mu``."""
    return kernel * Normal("mu", 0.0, 1e-3)


def _rounded_draws(draws: Any) -> list[float]:
    """The draws of ``y``, rounded to integers, which the kernels below shift by a setting."""
    return np.round(np.asarray(draws["y"])).tolist()


class TestTheFallbackRule:
    def test_a_program_that_does_not_trace_runs_eagerly_and_is_not_traced_again(self):
        program = _CountingProgram(float)
        eager_runs: list = []
        decision_record: dict = {}

        def eager(x):
            eager_runs.append(x)
            return float(x)

        assert compiled_or_eager(decision_record, "to float", program, eager, jnp.ones(())) == 1.0
        assert compiled_or_eager(decision_record, "to float", program, eager, jnp.ones(())) == 1.0
        assert program.python_runs == 1
        assert len(eager_runs) == 2

    def test_an_eager_error_propagates_and_records_nothing(self):
        program = _CountingProgram(float)
        decision_record: dict = {}

        def eager(x):
            raise ValueError("invalid argument")

        with pytest.raises(ValueError, match="invalid argument"):
            compiled_or_eager(decision_record, "to float", program, eager, jnp.ones(()))
        assert not any(recorded_key[0] == "runs eagerly" for recorded_key in decision_record)

    def test_a_program_that_traces_and_then_fails_while_running_raises_and_runs_no_eager_retry(
        self,
    ):
        program = _CountingProgram(_failing_on_the_host)
        eager_runs: list = []

        def eager(x):
            eager_runs.append(x)
            return x

        with pytest.raises(Exception, match="solver failed"):
            compiled_or_eager({}, "callback", program, eager, jnp.ones(3))
        assert eager_runs == []
        assert program.python_runs == 1

    def test_a_cold_call_traces_once_and_compiles_once(self):
        program = _CountingProgram(lambda x: jnp.sin(x) * 2.0)
        x = jnp.ones(3)
        with count_compilations() as compiled:
            compiled_or_eager({}, "sine", program, jnp.sin, x)
        assert compiled.count == 1
        assert program.python_runs == 1

    def test_a_new_input_structure_is_traced_before_it_runs(self):
        program = _CountingProgram(lambda x: x + 1.0)
        decision_record: dict = {}
        compiled_or_eager(decision_record, "add", program, None, jnp.ones(3))
        compiled_or_eager(decision_record, "add", program, None, jnp.zeros(3))
        compiled_or_eager(decision_record, "add", program, None, jnp.ones(4))
        assert program.python_runs == 2
        traced_keys = [key for key in decision_record if key[0] == "traced"]
        assert len(traced_keys) == 2


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

    def test_the_static_settings_hold_the_calls_counts_and_the_passed_controls(self):
        with config.sampling.using(num_samples=7):
            static_settings = _static_settings_in_effect()
        assert static_settings == _StaticSettings(
            resolved_controls=(),
            passed_controls=None,
            config_values=(("sampling", (("max_evaluations", 256), ("num_samples", 7))),),
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
        with pytest.raises(RuntimeError, match=r"noted num_samples used \[38\].*\[37\]"):
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


class TestTheJointsMappedFactor:
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
