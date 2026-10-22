"""Opaque Functions cross a JAX boundary without running while it is staged."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    InputSpec,
    Normal,
    NumericArraySpec,
    conditional_distribution,
    config,
    log_prob,
    sample,
    workflow_run,
)
from probpipe.core._call_context import (
    _ACTIVE_CALLS,
    _CallContext,
    _jax_tracing_active,
    _jax_transformation_active,
    _open_call_context,
    current_call_context,
)
from probpipe.core._compilation import _ProgramStore
from probpipe.core._dispatch import Feasibility, ResolutionError
from probpipe.core._evaluation_points import EvaluationError, EvaluationOutcomes
from probpipe.core._executors import EvaluationExecutor, executor_registry

SCALAR = NumericArraySpec((), dtype="float32")


class _Recorder(EvaluationExecutor):
    name = "embedded_boundary_recorder"
    description = "Record batches sent through an embedded boundary."
    priority = None

    def __init__(self):
        self.sizes = []

    def check(self, evaluation_points, /):
        return Feasibility(True)

    def execute(self, evaluation_points, /):
        self.sizes.append(len(evaluation_points))
        payload = evaluation_points.shared_payload()
        return EvaluationOutcomes.merge(chunk.run(payload) for chunk in evaluation_points.chunks())


class _Refusing(_Recorder):
    name = "embedded_boundary_refusing"

    def check(self, evaluation_points, /):
        return Feasibility(False, description="this executor refuses the embedded points")


RECORDER = _Recorder()


@pytest.fixture(autouse=True)
def isolated_executor_registry(monkeypatch):
    from probpipe.core import _executors

    registry = _executors._ExecutorRegistry()
    for name in ("jax", "sequential", "thread", "process"):
        registry.register(executor_registry.get_method(name))
    registry.register(RECORDER)
    registry.register(_Refusing())
    monkeypatch.setattr(_executors, "executor_registry", registry)


def _stored_call(body, *, store=None, argument=None):
    if store is None:
        store = _ProgramStore()
    if argument is None:
        argument = jnp.float32(2)
    with _open_call_context(
        replace(current_call_context(), call_depth=1), entry_method_name="__call__"
    ):
        return store.run("body", lambda options, x: body(x), body, argument)


def _opaque(runs):
    def body(x):
        runs.append(np.asarray(x).copy())
        return np.asarray(x, dtype="float32") + np.float32(1)

    return Function(body, input_spec=InputSpec(x=SCALAR), output_spec=SCALAR, traceable=False)


def test_a_user_jit_stages_without_running_and_then_evaluates_on_the_host():
    runs = []
    opaque = _opaque(runs)
    compiled = jax.jit(lambda x: opaque(x))
    compiled.trace(jnp.float32(2))
    assert not runs
    assert float(compiled(jnp.float32(2))) == 3
    assert len(runs) == 1


def test_a_kernel_builds_without_running_its_opaque_function():
    runs = []
    opaque = _opaque(runs)

    @conditional_distribution(given_spec={"mu": SCALAR})
    def kernel(mu):
        return Normal("y", opaque(mu), 1.0)

    assert not runs
    assert list(kernel.event_spec.outputs) == ["y"]
    from probpipe.core._execution_claims import Traceability

    assert kernel._traceability() is Traceability.EMBEDDABLE


def test_a_stored_program_carries_a_live_token_and_releases_it():
    runs = []
    opaque = _opaque(runs)
    store = _ProgramStore()
    context = _CallContext(function_label="outer", call_depth=1)
    with _open_call_context(context, entry_method_name="__call__"):
        result = store.run(
            "body", lambda options, x: opaque(x), lambda x: opaque.apply(x), jnp.float32(2)
        )
        assert float(result) == 3
        assert len(_ACTIVE_CALLS._contexts) == 1
    assert not _ACTIVE_CALLS._contexts
    assert len(runs) == 1


@pytest.mark.parametrize("transformation", ["jit", "shape", "vmap", "grad", "scan", "map"])
def test_staging_detection_distinguishes_eager_transformations(transformation):
    seen = []

    def body(x):
        seen.append((_jax_tracing_active(), _jax_transformation_active()))
        return x + 1

    if transformation == "jit":
        jax.jit(body)(jnp.float32(1))
    elif transformation == "shape":
        jax.eval_shape(body, jax.ShapeDtypeStruct((), "float32"))
    elif transformation == "vmap":
        jax.vmap(body)(jnp.ones(2))
    elif transformation == "grad":
        jax.grad(body)(jnp.float32(1))
    elif transformation == "scan":
        jax.lax.scan(lambda carry, x: (carry, body(x)), None, jnp.ones(2))
    else:
        jax.lax.map(body, jnp.ones(2))
    assert seen == [(transformation not in {"vmap", "grad"}, True)]


@pytest.mark.parametrize("nested", [False, True])
def test_callback_lanes_form_one_executor_request(nested):
    RECORDER.sizes.clear()
    opaque = _opaque([]).using(executor=RECORDER.name)
    body = jax.vmap(jax.vmap(opaque)) if nested else jax.vmap(opaque)
    arguments = jnp.ones((3, 4), dtype="float32") if nested else jnp.ones(4, dtype="float32")
    np.testing.assert_array_equal(_stored_call(body, argument=arguments), arguments + 1)
    assert RECORDER.sizes == [12 if nested else 4]


def test_a_refused_own_executor_keeps_its_exception_type_and_releases_the_token():
    opaque = _opaque([]).using(executor=_Refusing.name)
    with pytest.raises(ResolutionError, match="cannot run"):
        _stored_call(opaque)
    assert not _ACTIVE_CALLS._contexts


def test_a_wrong_host_output_becomes_an_evaluation_error():
    function = Function(lambda x: np.ones(2, dtype="float32"), output_spec=SCALAR, traceable=False)
    with pytest.raises(EvaluationError, match="shape"):
        _stored_call(function)
    assert not _ACTIVE_CALLS._contexts


@pytest.mark.parametrize("executor", ["sequential", "thread"])
def test_an_embedded_evaluation_cannot_draw_workflow_randomness(executor):
    function = Function(
        lambda x: np.asarray(sample(Normal("noise", 0.0, 1.0))),
        output_spec=SCALAR,
        traceable=False,
        executor=executor,
    )
    with pytest.raises(EvaluationError, match="cannot draw random numbers"):
        _stored_call(function)
    assert not _ACTIVE_CALLS._contexts


def test_gradient_refusal_names_the_opaque_function():
    opaque = _opaque([]).with_label("solver")
    with pytest.raises(TypeError, match="gradients cannot pass through 'solver'"):
        jax.grad(lambda x: opaque(x))(jnp.float32(1))


def test_a_cached_inner_jit_cannot_capture_a_token():
    opaque = _opaque([])

    @jax.jit
    def cached_helper(x):
        return opaque(x)

    cached_helper.trace(jnp.float32(1))
    with pytest.raises(TypeError, match="captured an embedded call's token"):
        _stored_call(cached_helper)
    assert not _ACTIVE_CALLS._contexts


@pytest.mark.parametrize("method", ["apply", "__call__"])
def test_float64_output_declarations_require_enabled_64_bit_floats(method):
    with jax.enable_x64(False):
        function = Function(lambda x: x, output_spec=NumericArraySpec((), dtype="float64"))
        with pytest.raises(ValueError, match="Set jax_enable_x64=True"):
            getattr(function, method)(jnp.float32(1))


@pytest.mark.parametrize("executor", ["sequential", "thread", RECORDER.name])
def test_a_joint_with_an_opaque_kernel_samples_and_scores(executor):
    opaque = _opaque([])
    RECORDER.sizes.clear()

    @conditional_distribution(given_spec={"mu": SCALAR})
    def kernel(mu):
        return Normal("y", opaque(mu), 1.0)

    model = kernel & Normal("mu", 0.0, 1.0)
    with workflow_run(seed=41):
        draws = sample.using(executor=executor)(model, 3)
    assert draws.batch_shape == (3,)
    if executor == RECORDER.name:
        assert RECORDER.sizes == [3]
    values = {"y": jnp.float32(2), "mu": jnp.float32(1)}
    score = float(log_prob.using(executor=executor)(model, values))
    expected = -np.log(2 * np.pi) - 0.5
    np.testing.assert_allclose(score, expected, rtol=1e-6)


def test_a_compiled_failure_stops_later_iterations():
    runs = []

    def fail(x):
        runs.append(float(x))
        raise ValueError("solver failed")

    opaque = Function(fail, output_spec=SCALAR, traceable=False)

    def body(x):
        return jax.lax.fori_loop(0, 5, lambda index, value: opaque(value), x)

    with pytest.raises(EvaluationError, match="solver failed"):
        _stored_call(body)
    assert runs == [2.0]
    assert not _ACTIVE_CALLS._contexts


def test_a_user_program_raises_instead_of_returning_a_placeholder():
    def fail(x):
        raise ValueError("user solver failed")

    opaque = Function(fail, output_spec=SCALAR, traceable=False)
    with pytest.raises(Exception, match="user solver failed"):
        jax.jit(lambda x: opaque(x))(jnp.float32(1)).block_until_ready()


def test_cold_and_warm_store_runs_replay_trace_warnings_once():
    store = _ProgramStore()

    def body(options, x):
        import warnings

        warnings.warn("traced solver note", UserWarning, stacklevel=2)
        return x + 1

    for value in (1, 2):
        with pytest.warns(UserWarning, match="traced solver note") as captured:
            result = store.run("warning", body, lambda x: x + 1, jnp.float32(value))
        assert len(captured) == 1
        assert float(result) == value + 1


@pytest.mark.parametrize("primitive", ["checkpoint", "cond", "scan", "map", "while"])
def test_callback_detection_searches_nested_jaxprs(primitive):
    opaque = _opaque([])

    def body(x):
        if primitive == "checkpoint":
            return jax.checkpoint(opaque)(x)
        if primitive == "cond":
            return jax.lax.cond(x > 0, opaque, lambda y: y - 1, x)
        if primitive == "scan":
            return jax.lax.scan(lambda c, y: (c, opaque(y)), None, x[None])[1][0]
        if primitive == "map":
            return jax.lax.map(opaque, x[None])[0]
        return jax.lax.while_loop(
            lambda state: state[0] < 1, lambda state: (state[0] + 1, opaque(state[1])), (0, x)
        )[1]

    assert float(_stored_call(body)) == 3
    assert not _ACTIVE_CALLS._contexts


def test_host_code_reads_the_calls_config_block_on_cold_and_warm_runs():
    function = Function(
        lambda x: np.float32(config.sampling.num_samples), output_spec=SCALAR, traceable=False
    )
    store = _ProgramStore()
    for count in (17, 29, 17):
        with config.sampling.using(num_samples=count):
            assert float(_stored_call(function, store=store)) == count


@pytest.mark.parametrize("output_spec", [None, NumericArraySpec(())])
def test_a_host_only_declaration_is_refused_while_staging(output_spec):
    function = Function(lambda x: np.asarray(x), traceable=False, output_spec=output_spec)
    with pytest.raises(TypeError, match="host-only Function"):
        jax.jit(lambda x: function(x)).trace(jnp.float32(1))


def test_a_kernel_with_unwrapped_numpy_names_the_fix():
    with pytest.raises(TypeError, match="Wrap the code that is not JAX"):
        conditional_distribution(
            lambda mu: Normal("y", np.asarray(mu), 1.0), given_spec={"mu": SCALAR}
        )


def test_unknown_traceability_uses_one_stored_probe():
    probes = []

    def body(x):
        if isinstance(x, jax.core.Tracer):
            probes.append(x.shape)
        return np.asarray(x, dtype="float32") + np.float32(1)

    function = Function(body, output_spec=SCALAR)
    for _ in range(2):
        assert float(_stored_call(function)) == 3
    assert probes == [()]


def test_prefect_requested_inside_a_staging_trace_is_inline():
    from probpipe import WorkflowKind

    opaque = _opaque([])
    inner = Function(
        lambda x: opaque(x), traceable=True, output_spec=SCALAR, workflow_kind=WorkflowKind.TASK
    )
    assert float(_stored_call(inner)) == 3


def test_unclassified_config_is_refused_only_in_a_probpipe_trace():
    from probpipe.core.config import _Section

    class Section(_Section):
        _name = "unclassified_boundary_test"
        _settings: ClassVar[dict] = {}
        _value = 3

    section = Section()
    assert section._value_in_effect("value") == 3
    with pytest.raises(RuntimeError, match="has no declared kind"):
        _stored_call(lambda x: x + section._value_in_effect("value"))


def test_pmap_callbacks_receive_the_context_and_refuse_workflow_draws():
    code = """
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from probpipe import Function, Normal, NumericArraySpec, config, sample
from probpipe.core._call_context import _ACTIVE_CALLS, _open_call_context, current_call_context
from probpipe.core._compilation import _ProgramStore
from probpipe.core._evaluation_points import EvaluationError
scalar = NumericArraySpec((), dtype="float32")
def run(function):
    with _open_call_context(replace(current_call_context(), call_depth=1), entry_method_name="__call__"):
        return _ProgramStore().run("pmap", lambda options, x: jax.pmap(function)(x), function.apply, jnp.ones(4))
assert jax.local_device_count() == 4
read = Function(lambda x: np.float32(config.sampling.num_samples), output_spec=scalar, traceable=False)
with config.sampling.using(num_samples=17):
    np.testing.assert_array_equal(run(read), np.full(4, 17, dtype="float32"))
draw = Function(lambda x: np.asarray(sample(Normal("noise", 0.0, 1.0))), output_spec=scalar, traceable=False)
try:
    run(draw)
except EvaluationError as error:
    assert "cannot draw random numbers" in str(error)
else:
    raise AssertionError("a callback drew workflow randomness")
assert not _ACTIVE_CALLS._contexts
"""
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_python_evaluation_points_refuse_disabled_float64_output():
    from probpipe.core._evaluation_points import EvaluationPoints
    from probpipe.core._specs import OutputSpec

    with jax.enable_x64(False), pytest.raises(ValueError, match="jax_enable_x64"):
        EvaluationPoints(
            lambda x: x,
            varying={"x": np.ones(2)},
            origin="solver",
            output_spec=OutputSpec.default(
                NumericArraySpec((), dtype="float64"), output_name="value"
            ),
        )


def test_a_cached_nested_trace_may_emit_a_warning_only_once():
    import warnings

    store = _ProgramStore()

    @jax.jit
    def inner(x):
        warnings.warn("nested solver note", UserWarning, stacklevel=2)
        return x + 1

    def body(options, x):
        return inner(x)

    with pytest.warns(UserWarning, match="nested solver note"):
        assert float(store.run("body", body, inner, jnp.float32(1))) == 2
    with jax.disable_jit(), pytest.warns(UserWarning, match="nested solver note"):
        assert float(store.run("body", body, inner, jnp.float32(2))) == 3


def test_one_program_uses_each_calls_executor_without_retracing():
    from probpipe.core._call_context import _ExecutionChoice

    store = _ProgramStore()
    opaque = _opaque([])
    traces = []

    def body(options, x):
        traces.append(None)
        return opaque(x)

    RECORDER.sizes.clear()
    for executor in ("sequential", RECORDER.name):
        context = replace(
            current_call_context(),
            call_depth=1,
            execution_choice=_ExecutionChoice(executor_name=executor, source="own"),
        )
        with _open_call_context(context, entry_method_name="__call__"):
            assert float(store.run("body", body, opaque.apply, jnp.float32(2))) == 3
    assert len(traces) == 1
    assert RECORDER.sizes == [1]
    assert not _ACTIVE_CALLS._contexts


@pytest.mark.parametrize("has_argument", [False, True])
def test_an_opaque_constant_call_is_not_run_while_staging(has_argument):
    runs = []

    def body(*args):
        runs.append(None)
        return np.float32(3)

    opaque = Function(body, output_spec=SCALAR, traceable=False)
    compiled = jax.jit(lambda x: opaque(np.float32(2)) if has_argument else opaque())
    compiled.trace(jnp.float32(1))
    assert not runs
    assert float(compiled(jnp.float32(1))) == 3
    assert len(runs) == 1


def test_an_embedded_function_can_use_a_spawned_process_executor():
    opaque = _opaque([]).using(executor="process", executor_options={"max_workers": 1})
    assert float(_stored_call(opaque)) == 3
    assert not _ACTIVE_CALLS._contexts


def test_concurrent_calls_of_one_program_read_their_own_contexts():
    from concurrent.futures import ThreadPoolExecutor

    seen = []
    traces = []

    def host_body(x):
        seen.append(config.notation.max_depth)
        return np.asarray(x) + np.float32(1)

    opaque = Function(host_body, output_spec=SCALAR, traceable=False)
    store = _ProgramStore()

    def body(options, x):
        traces.append(None)
        return opaque(x)

    def run(depth):
        with config.notation.using(max_depth=depth):
            return float(store.run("body", body, opaque.apply, jnp.float32(1)))

    # An inert setting changes no cache key or computed value, but each host
    # evaluation must still read the context belonging to its runtime token.
    assert run(7) == 2
    seen.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, [17, 31])) == [2, 2]
    assert sorted(seen) == [17, 31]
    assert len(traces) == 1
    assert not _ACTIVE_CALLS._contexts


def test_an_embedded_parameter_does_not_promote_a_host_only_law(monkeypatch):
    from probpipe.core._execution_claims import Traceability

    monkeypatch.setattr(Normal, "_traceability", lambda self: Traceability.HOST_ONLY)
    opaque = _opaque([])

    @conditional_distribution(given_spec={"mu": SCALAR})
    def kernel(mu):
        return Normal("y", opaque(mu), 1.0)

    assert kernel._traceability() is Traceability.HOST_ONLY


def test_an_operation_route_keeps_the_calling_thread_and_its_single_worker_is_available():
    import threading

    from probpipe.core._specs import OutputSpec
    from probpipe.operations._operation import operation

    caller = threading.get_ident()
    seen = []
    opaque = Function(
        lambda x: np.asarray(x, dtype="float32") + np.float32(1),
        traceable=False,
        output_spec=SCALAR,
    )

    @operation(result=lambda x: OutputSpec.default(SCALAR, output_name="coordinate"))
    def coordinate(x):
        seen.append(threading.get_ident())
        return _stored_call(opaque, argument=x)

    assert (
        float(
            coordinate.using(executor="thread", executor_options={"max_workers": 1})(np.float32(2))
        )
        == 3
    )
    assert seen == [caller]


def test_a_refused_operation_executor_falls_back_for_an_embedded_function():
    opaque = _opaque([])

    @conditional_distribution(given_spec={"mu": SCALAR})
    def kernel(mu):
        return Normal("y", opaque(mu), 1.0)

    model = kernel & Normal("mu", 0.0, 1.0)
    with workflow_run(seed=41), pytest.warns(UserWarning, match="cannot run|refuses|fall"):
        draws = sample.using(executor=_Refusing.name)(model, 3)
    assert draws.batch_shape == (3,)
