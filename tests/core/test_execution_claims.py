"""The probe, its stored results, and the traceability lattice (design V.13)."""

from __future__ import annotations

import pickle
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe.core._execution_claims import (
    ProbeResult,
    ProbeResults,
    Traceability,
    closure_probe_key,
    probe,
    raise_for_false_traceable_claim,
    spec_stand_in,
)
from probpipe.core._record_spec import RecordSpec
from probpipe.core._spec_base import NumericArraySpec


class TestTraceability:
    def test_the_meet_is_the_least_permissive(self):
        assert Traceability.meet() is Traceability.TRACEABLE
        assert (
            Traceability.meet(Traceability.TRACEABLE, Traceability.EMBEDDABLE)
            is Traceability.EMBEDDABLE
        )
        assert (
            Traceability.meet(
                Traceability.EMBEDDABLE, Traceability.HOST_ONLY, Traceability.TRACEABLE
            )
            is Traceability.HOST_ONLY
        )


class TestProbeResults:
    def test_the_store_drops_the_least_recently_used_result_past_its_bound(self):
        store = ProbeResults(max_entries=2)
        store.put("a", ProbeResult(traces=True))
        store.put("b", ProbeResult(traces=True))
        store.get("a")
        store.put("c", ProbeResult(traces=False))
        assert store.get("b") is None
        assert store.get("a") is not None
        assert len(store) == 2

    def test_a_pickled_store_arrives_empty_with_its_bound(self):
        store = ProbeResults(max_entries=7)
        store.put("a", ProbeResult(traces=True))
        restored = pickle.loads(pickle.dumps(store))
        assert len(restored) == 0
        assert restored._max_entries == 7


class TestProbe:
    def test_a_result_is_stored_by_input_structure(self):
        store = ProbeResults()
        traced = []

        def double(x):
            traced.append(x)
            return 2 * x

        first = probe(store, double, jnp.ones(3), label="double")
        probe(store, double, jnp.zeros(3), label="double")
        probe(store, double, jnp.ones(4), label="double")
        assert first.traces
        assert first.output == jax.ShapeDtypeStruct((3,), jnp.float32)
        assert len(traced) == 2

    def test_a_failure_keeps_its_error_without_a_traceback(self):
        result = probe(None, lambda x: np.asarray(x), jnp.ones(2), label="host")
        assert not result.traces
        assert isinstance(result.error, jax.errors.TracerArrayConversionError)
        assert result.error.__traceback__ is None

    def test_a_string_argument_enters_the_structure_by_its_value(self):
        store = ProbeResults()

        def scaled(arguments):
            x, unit = arguments
            return x * (100.0 if unit == "cm" else 1.0)

        probe(store, scaled, (jnp.ones(2), "cm"), label="scaled")
        probe(store, scaled, (jnp.ones(2), "m"), label="scaled")
        probe(store, scaled, (jnp.ones(2), "cm"), label="scaled")
        assert len(store) == 2

    def test_an_unhashable_argument_stores_nothing(self):
        store = ProbeResults()
        result = probe(store, lambda arguments: arguments[0], (jnp.ones(2), {"a"}), label="f")
        assert result.traces
        assert len(store) == 0

    def test_the_global_random_states_are_restored(self):
        np.random.seed(3)
        random.seed(3)
        numpy_state = np.random.get_state()[1].copy()
        python_state = random.getstate()

        def noisy(x):
            random.random()
            return x + np.random.normal()

        with pytest.warns(UserWarning, match="'noisy' draws from NumPy's global random"):
            result = probe(None, noisy, jnp.ones(2), label="noisy")
        assert not result.traces and result.draws_global_random
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
        assert random.getstate() == python_state


class TestClosureProbeKey:
    @staticmethod
    def _closure(captured):
        return lambda y: y * captured

    def test_closures_of_one_code_share_a_key_by_the_structure_they_capture(self):
        assert closure_probe_key(self._closure(jnp.ones(2))) == closure_probe_key(
            self._closure(jnp.zeros(2))
        )
        assert closure_probe_key(self._closure(jnp.ones(2))) != closure_probe_key(
            self._closure(jnp.ones(3))
        )
        assert closure_probe_key(self._closure(2.0)) != closure_probe_key(self._closure(3.0))

    def test_a_closure_capturing_an_unhashable_value_shares_only_with_itself(self):
        first = self._closure([1.0])
        assert closure_probe_key(first) == closure_probe_key(first)
        assert closure_probe_key(first) != closure_probe_key(self._closure([1.0]))


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
        ("trace", "failure"),
        [
            (lambda x: np.asarray(x), "it converts a traced value to NumPy"),
            (lambda x: x if x > 0 else -x, "it branches in Python on a traced value"),
            (lambda x: list(range(x)), "it converts a traced value to a Python int"),
        ],
    )
    def test_the_message_says_what_the_callable_did(self, trace, failure):
        try:
            jax.make_jaxpr(trace)(1)
        except Exception as error:
            tracing_error = error
        with pytest.raises(TypeError, match=failure) as raised:
            raise_for_false_traceable_claim("forward", tracing_error)
        assert raised.value.__cause__ is tracing_error
