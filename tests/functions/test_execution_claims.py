"""Execution claims and the stored probe result of a ``Function`` (design V.13).

A ``Function`` states ``traceable`` and ``vectorized`` when it is built. A
lifted call of a ``Function`` that claims nothing probes it once per input
structure, stores the outcome, and shares it with ``check`` and with every
``using`` copy; a ``Function`` that claims ``traceable=False`` is never probed,
and a false ``traceable=True`` claim raises an error naming the claim.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Function,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    conditional_distribution,
    function,
    workflow_run,
)
from probpipe.core._execution_claims import ExecutionClaims


def _prior():
    return Normal("r", 0.0, 1.0)


class TestTheClaims:
    def test_a_function_states_its_claims_when_it_is_built(self):
        @function(traceable=False, vectorized=True)
        def forward(theta):
            return theta

        assert forward._execution_claims == ExecutionClaims(traceable=False, vectorized=True)

    def test_a_function_that_states_nothing_has_no_claims(self):
        assert Function(lambda x: x, label="f")._execution_claims == ExecutionClaims()

    def test_a_using_copy_keeps_the_claims_and_the_probe_results(self):
        @function(traceable=True)
        def growth(r):
            return jnp.exp(r)

        copy = growth.using(max_evaluations=16)
        assert copy._execution_claims is growth._execution_claims
        assert copy._probe_results is growth._probe_results

    @pytest.mark.parametrize(
        ("keyword", "value", "message"),
        [
            ("traceable", "yes", "traceable of 'f' must be True, False, or None, got 'yes'"),
            ("vectorized", None, "vectorized of 'f' must be True or False, got None"),
        ],
    )
    def test_a_claim_that_is_not_a_bool_raises(self, keyword, value, message):
        with pytest.raises(TypeError, match=message):
            Function(lambda x: x, label="f", **{keyword: value})


class TestTheProbe:
    def test_a_function_that_states_nothing_is_probed_once_per_input_structure(
        self, probe_attempts
    ):
        @function
        def growth(r, n0):
            return n0 * jnp.exp(r)

        with workflow_run(seed=0):
            growth.using(max_evaluations=8)(_prior(), n0=10.0)
            growth.using(max_evaluations=8)(_prior(), n0=11.0)
            growth.using(max_evaluations=16)(_prior(), n0=12.0)
        assert len(probe_attempts) == 1
        assert len(growth._probe_results) == 1

    def test_a_new_input_shape_probes_again(self, probe_attempts):
        @function
        def total(x):
            return jnp.sum(x)

        with workflow_run(seed=0):
            total(NumericArrayBatch(jnp.ones((3, 2)), level_names=("row",), label="a"))
            total(NumericArrayBatch(jnp.ones((3, 2)), level_names=("row",), label="b"))
            total(NumericArrayBatch(jnp.ones((3, 4)), level_names=("row",), label="c"))
        assert len(probe_attempts) == 2

    def test_check_probes_and_the_call_reads_the_stored_result(self, probe_attempts):
        body_runs = []

        @function
        def growth(r):
            body_runs.append(r)
            return jnp.exp(r)

        growth.check(_prior())
        assert len(probe_attempts) == 1
        assert len(body_runs) == 1
        with workflow_run(seed=0):
            growth.using(max_evaluations=8)(_prior())
        assert len(probe_attempts) == 1

    def test_check_of_a_call_that_lifts_nothing_probes_nothing(self, probe_attempts):
        @function
        def growth(r):
            return jnp.exp(r)

        growth.check(1.0)
        assert probe_attempts == []

    def test_a_python_float_is_data_so_it_adds_no_probe_result(self, probe_attempts):
        @function
        def scaled(r, scale):
            return scale * r

        with workflow_run(seed=0):
            for scale in (1.0, 2.0, 3.0):
                scaled.using(max_evaluations=8)(_prior(), scale=scale)
        assert len(probe_attempts) == 1
        assert len(scaled._probe_results) == 1

    def test_an_int_used_as_a_loop_count_is_retried_static_in_the_same_probe(self, probe_attempts):
        @function
        def stepped(r, steps):
            value = r
            for _ in range(steps):
                value = value + 1.0
            return value

        with workflow_run(seed=0):
            stepped.using(max_evaluations=8)(_prior(), steps=3)
        # One attempt with the int as data, which fails, and one with it static.
        assert len(probe_attempts) == 2
        static_result = stepped._probe_results.stored_result(
            next(reversed(stepped._probe_results._results))
        )
        assert static_result.traces and static_result.ints_static
        with workflow_run(seed=0):
            stepped.using(max_evaluations=8)(_prior(), steps=3)
        assert len(probe_attempts) == 2

    def test_a_new_int_value_probes_only_the_static_form(self, probe_attempts):
        @function
        def stepped(r, steps):
            return r + jnp.zeros(steps).sum()

        with workflow_run(seed=0):
            stepped.using(max_evaluations=8)(_prior(), steps=3)
            stepped.using(max_evaluations=8)(_prior(), steps=4)
        assert len(probe_attempts) == 3

    def test_a_function_that_draws_from_numpys_global_generator_is_opaque_and_warns(
        self, probe_attempts
    ):
        body_runs = []

        @function
        def simulate(r):
            body_runs.append(r)
            return r + np.random.normal()

        with workflow_run(seed=0), pytest.warns(UserWarning, match="'simulate' draws from NumPy"):
            simulate.using(max_evaluations=5)(_prior())
        # One probe, then each of the five points runs on its concrete value.
        assert len(probe_attempts) == 1
        assert sum(isinstance(r, jax.core.Tracer) for r in body_runs) == 1
        assert sum(not isinstance(r, jax.core.Tracer) for r in body_runs) == 5
        with workflow_run(seed=0), warnings.catch_warnings():
            warnings.simplefilter("error")
            simulate.using(max_evaluations=5)(_prior())
        assert len(probe_attempts) == 1


class TestTraceableFalse:
    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "thread"])
    def test_it_runs_once_per_point_with_no_probe(self, probe_attempts, dispatch):
        traced_runs = []
        concrete_runs = []

        @function(traceable=False)
        def forward(theta):
            (traced_runs if isinstance(theta, jax.core.Tracer) else concrete_runs).append(theta)
            return np.asarray(theta) * 2.0

        with workflow_run(seed=0):
            forward.using(max_evaluations=6, dispatch=dispatch)(_prior())
        assert probe_attempts == []
        assert traced_runs == []
        assert len(concrete_runs) == 6

    def test_check_does_not_probe_it(self, probe_attempts):
        @function(traceable=False)
        def forward(theta):
            return np.asarray(theta)

        forward.check(_prior())
        assert probe_attempts == []

    def test_jax_dispatch_refuses_it(self):
        @function(traceable=False)
        def forward(theta):
            return np.asarray(theta)

        with (
            workflow_run(seed=0),
            pytest.raises(
                ValueError,
                match=r"dispatch='jax' cannot run 'forward', which claims traceable=False",
            ),
        ):
            forward.using(max_evaluations=4, dispatch="jax")(_prior())


class TestTraceableTrue:
    def test_it_is_mapped_with_no_probe(self, probe_attempts):
        @function(traceable=True)
        def growth(r):
            return jnp.exp(r)

        with workflow_run(seed=0):
            growth.using(max_evaluations=8)(_prior())
        growth.check(_prior())
        assert probe_attempts == []

    def test_a_false_claim_raises_naming_the_claim_when_called_alone(self):
        @function(traceable=True)
        def forward(theta):
            return jnp.asarray(np.asarray(theta) * 2.0)

        with workflow_run(seed=0), pytest.raises(TypeError) as raised:
            forward.using(max_evaluations=4)(_prior())
        assert str(raised.value) == (
            "'forward' claims traceable=True, but tracing it failed: it converts a traced "
            "value to NumPy (TracerArrayConversionError). Remove the claim, or claim "
            "traceable=False"
        )
        assert isinstance(raised.value.__cause__, jax.errors.TracerArrayConversionError)

    def test_a_false_claim_raises_inside_a_larger_trace(self):
        @function(traceable=True)
        def forward(theta):
            if theta > 0:
                return theta
            return -theta

        with pytest.raises(TypeError, match="it branches in Python on a traced value") as raised:
            jax.jit(lambda theta: forward.apply(theta))(1.0)
        assert isinstance(raised.value.__cause__, jax.errors.TracerBoolConversionError)

    def test_a_false_claim_raises_inside_a_kernel(self):
        @function(traceable=True)
        def forward(theta):
            return jnp.asarray(np.asarray(theta))

        with pytest.raises(TypeError, match="'forward' claims traceable=True"):
            conditional_distribution(
                lambda theta: Normal("y", forward(theta), 1.0),
                given_spec={"theta": NumericArraySpec(())},
                label="likelihood",
            )

    def test_a_true_claim_maps_the_same_values_as_sequential_dispatch(self):
        @function(traceable=True)
        def growth(r):
            return jnp.exp(r)

        assert float(growth.apply(0.0)) == 1.0
        with workflow_run(seed=0):
            mapped = growth.using(max_evaluations=8)(_prior())
        with workflow_run(seed=0):
            sequential = growth.using(max_evaluations=8, dispatch="sequential")(_prior())
        np.testing.assert_allclose(
            np.asarray(mapped.atoms), np.asarray(sequential.atoms), rtol=1e-6
        )

    def test_a_false_claim_inside_a_probed_function_is_not_swallowed(self):
        @function(traceable=True)
        def inner(theta):
            return jnp.asarray(np.asarray(theta))

        @function
        def outer(theta):
            return inner(theta) + 1.0

        with workflow_run(seed=0), pytest.raises(TypeError, match="'inner' claims traceable=True"):
            outer.using(max_evaluations=8)(_prior())


class TestCheckSharesItsProbe:
    def test_a_sweep_reads_the_probe_check_stored(self, probe_attempts):
        @function
        def total(x):
            return jnp.sum(x)

        rows = NumericArrayBatch(jnp.ones((3, 2)), level_names=("row",), label="rows")
        total.check(rows)
        assert len(probe_attempts) == 1
        with workflow_run(seed=0):
            total(rows)
        assert len(probe_attempts) == 1

    def test_a_nested_sweep_reads_the_probe_check_stored(self, probe_attempts):
        @function
        def shifted(shift, noise):
            return shift + noise

        rows = NumericArrayBatch(jnp.arange(3.0), "row", label="shift")
        shifted.using(max_evaluations=8).check(rows, _prior())
        assert len(probe_attempts) == 1
        with workflow_run(seed=0):
            shifted.using(max_evaluations=8)(rows, _prior())
        assert len(probe_attempts) == 1

    def test_jax_dispatch_refuses_a_function_that_draws_from_numpys_generator(self):
        @function(dispatch="jax")
        def simulate(r):
            return r + np.random.normal()

        with (
            workflow_run(seed=0),
            pytest.warns(UserWarning, match="global random generator"),
            pytest.raises(ValueError, match="dispatch='jax' cannot trace 'simulate'"),
        ):
            simulate.using(max_evaluations=5)(_prior())
