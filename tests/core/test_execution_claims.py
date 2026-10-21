"""The probe, its stored results, and the traceability lattice (design V.13)."""

from __future__ import annotations

import pickle
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.core._execution_claims import (
    FalseTraceableClaimError,
    ProbeResult,
    ProbeResults,
    Traceability,
    probe,
    probe_closure,
    spec_stand_in,
)
from probpipe.core._record_spec import RecordSpec
from probpipe.core._spec_base import NumericArraySpec


class TestTraceability:
    def test_the_least_permissive_comes_last_in_the_order_traceable_embeddable_host_only(self):
        assert Traceability.least_permissive() is Traceability.TRACEABLE
        assert (
            Traceability.least_permissive(Traceability.TRACEABLE, Traceability.EMBEDDABLE)
            is Traceability.EMBEDDABLE
        )
        assert (
            Traceability.least_permissive(
                Traceability.EMBEDDABLE, Traceability.HOST_ONLY, Traceability.TRACEABLE
            )
            is Traceability.HOST_ONLY
        )


class TestProbeResults:
    def test_the_store_drops_the_least_recently_used_result_past_its_bound(self):
        probe_results = ProbeResults(max_entries=2)
        probe_results.store_result("a", ProbeResult(traces=True))
        probe_results.store_result("b", ProbeResult(traces=True))
        probe_results.stored_result("a")
        probe_results.store_result("c", ProbeResult(traces=False))
        assert probe_results.stored_result("b") is None
        assert probe_results.stored_result("a") is not None
        assert len(probe_results) == 2

    def test_a_pickled_store_arrives_empty_with_its_bound(self):
        probe_results = ProbeResults(max_entries=7)
        probe_results.store_result("a", ProbeResult(traces=True))
        restored = pickle.loads(pickle.dumps(probe_results))
        assert len(restored) == 0
        assert restored._max_entries == 7


class TestProbe:
    def test_a_result_is_stored_by_input_structure(self):
        probe_results = ProbeResults()
        traced = []

        def double(x):
            traced.append(x)
            return 2 * x

        first = probe(probe_results, double, jnp.ones(3), label="double")
        probe(probe_results, double, jnp.zeros(3), label="double")
        probe(probe_results, double, jnp.ones(4), label="double")
        assert first.traces
        assert first.abstract_output == jax.ShapeDtypeStruct((3,), jnp.float32)
        assert len(traced) == 2

    def test_a_failure_keeps_its_error_without_a_traceback(self):
        result = probe(None, lambda x: np.asarray(x), jnp.ones(2), label="host")
        assert not result.traces
        assert isinstance(result.error, jax.errors.TracerArrayConversionError)
        assert result.error.__traceback__ is None

    def test_a_string_argument_enters_the_structure_by_its_value(self):
        probe_results = ProbeResults()

        def scaled(arguments):
            x, unit = arguments
            return x * (100.0 if unit == "cm" else 1.0)

        probe(probe_results, scaled, (jnp.ones(2), "cm"), label="scaled")
        probe(probe_results, scaled, (jnp.ones(2), "m"), label="scaled")
        probe(probe_results, scaled, (jnp.ones(2), "cm"), label="scaled")
        assert len(probe_results) == 2

    def test_an_unhashable_argument_stores_nothing(self):
        probe_results = ProbeResults()
        result = probe(
            probe_results, lambda arguments: arguments[0], (jnp.ones(2), {"a"}), label="f"
        )
        assert result.traces
        assert len(probe_results) == 0

    def test_the_global_random_states_are_restored(self):
        np.random.seed(3)
        random.seed(3)
        numpy_state = np.random.get_state()[1].copy()
        python_state = random.getstate()

        def noisy(x):
            random.random()
            return x + np.random.normal()

        with pytest.warns(UserWarning, match="'noisy' draws from NumPy's or Python's global"):
            result = probe(None, noisy, jnp.ones(2), label="noisy")
        assert not result.traces and result.draws_global_random
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
        assert random.getstate() == python_state


def _call_closure(closure, arguments):
    return closure(arguments)


class TestProbeClosure:
    @staticmethod
    def _scaled_by(captured):
        return lambda y: y * captured

    def test_closures_of_one_code_share_a_result_by_the_shapes_they_capture(self, probe_attempts):
        probe_results = ProbeResults()
        for captured in (jnp.ones(2), jnp.zeros(2), 3.0 * jnp.ones(2)):
            result = probe_closure(
                probe_results,
                self._scaled_by(captured),
                _call_closure,
                jnp.ones(2),
                label="f",
                role="f",
            )
            assert result.traces
        probe_closure(
            probe_results,
            self._scaled_by(jnp.ones(3)),
            _call_closure,
            jnp.ones(3),
            label="f",
            role="f",
        )
        assert probe_attempts == [True, True]

    def test_a_closure_that_reads_a_captured_value_shares_only_with_equal_captures(
        self, probe_attempts
    ):
        def sized_by(size):
            count = np.array(size)
            return lambda key: jax.random.normal(key, (int(count),))

        probe_results = ProbeResults()
        three = probe_closure(
            probe_results, sized_by(3), _call_closure, jax.random.key(0), label="f", role="sample"
        )
        five = probe_closure(
            probe_results, sized_by(5), _call_closure, jax.random.key(0), label="f", role="sample"
        )
        again = probe_closure(
            probe_results, sized_by(3), _call_closure, jax.random.key(0), label="f", role="sample"
        )
        assert three.abstract_output.shape == (3,)
        assert five.abstract_output.shape == (5,)
        assert again.abstract_output.shape == (3,)
        # With the count as data each fails; as a constant each traces, keyed by its value.
        assert probe_attempts == [False, True, True]

    def test_a_closure_capturing_a_plain_callable_is_probed_for_itself(self, probe_attempts):
        def scale(y):
            return 2 * y

        def make(captured):
            return lambda y: scale(y) * captured(y)

        probe_results = ProbeResults()
        probe_closure(probe_results, make(jnp.sin), _call_closure, jnp.ones(2), label="f", role="f")
        probe_closure(probe_results, make(jnp.sin), _call_closure, jnp.ones(2), label="f", role="f")
        assert probe_attempts == [True, True]
        assert len(probe_results) == 0


class TestStandIns:
    def test_a_record_of_fixed_arrays_stands_in_field_by_field(self):
        spec = RecordSpec(a=NumericArraySpec((2,), jnp.float32), b=NumericArraySpec(()))
        assert spec_stand_in(spec) == {
            "a": jax.ShapeDtypeStruct((2,), jnp.float32),
            "b": jax.ShapeDtypeStruct((), jnp.result_type(float)),
        }

    def test_an_unbound_dimension_has_no_stand_in(self):
        assert spec_stand_in(NumericArraySpec(("n",))) is None


class TestFalseClaim:
    @pytest.mark.parametrize(
        ("traced_function", "failure"),
        [
            (lambda x: np.asarray(x), "it converts a traced value to NumPy"),
            (lambda x: x if x > 0 else -x, "it branches in Python on a traced value"),
            (lambda x: list(range(x)), "it converts a traced value to a Python int"),
        ],
    )
    def test_the_message_says_what_the_callable_did(self, traced_function, failure):
        with pytest.raises(jax.errors.JAXTypeError) as raised:
            jax.make_jaxpr(traced_function)(1)
        error = FalseTraceableClaimError.for_tracing_error("forward", raised.value)
        assert isinstance(error, TypeError)
        assert str(error) == (
            f"'forward' claims traceable=True, but tracing it failed: {failure} "
            f"({type(raised.value).__name__}). Remove the claim, or claim traceable=False"
        )

    def test_a_probe_lets_a_false_claim_propagate(self):
        def claims_falsely(x):
            raise FalseTraceableClaimError("'inner' claims traceable=True, but tracing it failed")

        with pytest.raises(FalseTraceableClaimError):
            probe(None, claims_falsely, jnp.ones(2), label="outer")
