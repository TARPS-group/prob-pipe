"""Tests for the working memory of a JAX program, which sizes the chunks of the ``jax`` executor's map (V.9)."""

import random

import jax
import jax.numpy as jnp
import numpy as np

from probpipe.core._compilation import _working_memory_bytes

# One float32 array of this many entries occupies 4,000 bytes.
_ENTRIES = 1000
_ARRAY_BYTES = 4 * _ENTRIES
_SCALAR_BYTES = 4


def _scalar():
    return jax.ShapeDtypeStruct((), jnp.float32)


class TestTheWorkingMemoryOfAJaxProgram:
    def test_an_array_the_callable_captures_is_not_counted(self):
        grid = jnp.linspace(0.0, 1.0, _ENTRIES)
        # x, then x * grid and its sine together.
        assert (
            _working_memory_bytes(lambda x: jnp.sum(jnp.sin(x * grid)), _scalar())
            == _SCALAR_BYTES + 2 * _ARRAY_BYTES
        )

    def test_a_value_is_released_after_the_last_operation_that_reads_it(self):
        def chain_of_operations(x):
            y = jnp.sin(x)
            y = jnp.cos(y)
            y = jnp.exp(y)
            return jnp.tanh(y)

        # Each step reads only the step before it, so at most two arrays are
        # held beside the input, never all four.
        assert (
            _working_memory_bytes(
                chain_of_operations, jax.ShapeDtypeStruct((_ENTRIES,), jnp.float32)
            )
            == 3 * _ARRAY_BYTES
        )

    def test_a_value_read_again_later_is_held_until_then(self):
        def reads_its_first_array_last(x):
            first = jnp.sin(x)
            second = jnp.cos(first)
            third = jnp.exp(second)
            return third + first

        # first stays alive until the sum, beside second and then third.
        assert (
            _working_memory_bytes(
                reads_its_first_array_last, jax.ShapeDtypeStruct((_ENTRIES,), jnp.float32)
            )
            == 4 * _ARRAY_BYTES
        )

    def test_the_body_of_a_scan_is_counted_once_not_once_per_step(self):
        grid = jnp.linspace(0.0, 1.0, _ENTRIES)

        def scanned(x):
            def step(total, _):
                return total + jnp.sum(jnp.sin(x * grid)), None

            return jax.lax.scan(step, 0.0, None, length=50)[0]

        working_memory_bytes = _working_memory_bytes(scanned, _scalar())
        assert 2 * _ARRAY_BYTES <= working_memory_bytes < 4 * _ARRAY_BYTES

    def test_an_inner_jit_holds_its_own_working_memory_while_it_runs(self):
        grid = jnp.linspace(0.0, 1.0, _ENTRIES)
        inner = jax.jit(lambda x: jnp.sum(jnp.sin(x * grid)))
        # x and the jit's scalar result, with the jit's two arrays while it runs.
        assert _working_memory_bytes(inner, _scalar()) == 2 * _SCALAR_BYTES + 2 * _ARRAY_BYTES

    def test_a_callable_that_does_not_trace_has_none(self):
        assert _working_memory_bytes(lambda x: float(x) + 1.0, _scalar()) is None

    def test_the_trace_leaves_the_global_random_generators_as_they_were(self):
        np.random.seed(7)
        random.seed(7)
        numpy_state, python_state = np.random.get_state(), random.getstate()

        def draws_from_the_global_generators(x):
            return x + np.random.rand() + random.random()

        _working_memory_bytes(draws_from_the_global_generators, _scalar())
        assert random.getstate() == python_state
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
