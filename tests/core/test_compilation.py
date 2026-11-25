"""Tests for the trace of one evaluation point, which decides how the ``jax`` executor chunks its map (V.9)."""

import jax
import jax.numpy as jnp
import pytest

from probpipe.core._compilation import _traced_point

# One float32 array of this many entries occupies 4,000 bytes.
_ENTRIES = 1000
_ARRAY_BYTES = 4 * _ENTRIES
_SCALAR_BYTES = 4


def _scalar():
    return jax.ShapeDtypeStruct((), jnp.float32)


def _key(impl="threefry2x32"):
    return jax.ShapeDtypeStruct((), jax.random.key(0, impl=impl).dtype)


def _working_memory_bytes(fn, *abstract_arguments):
    return _traced_point(fn, *abstract_arguments).working_memory_bytes


class TestTheWorkingMemoryOfOnePoint:
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

    @pytest.mark.parametrize(
        ("impl", "key_data_bytes"), [("threefry2x32", 8), ("rbg", 16), ("unsafe_rbg", 16)]
    )
    def test_a_typed_key_occupies_the_bytes_of_its_key_data(self, impl, key_data_bytes):
        def split_into_many_keys(key):
            return jax.random.key_data(jax.random.split(key, _ENTRIES)[-1])

        # The input key, then the split keys, then the last key's data.
        assert _working_memory_bytes(split_into_many_keys, _key(impl)) == (
            key_data_bytes * (_ENTRIES + 1) + key_data_bytes
        )

    def test_a_callable_that_does_not_trace_has_no_trace(self):
        assert _traced_point(lambda x: float(x) + 1.0, _scalar()) is None


class TestWhetherTheDrawsDependOnTheChunks:
    @pytest.mark.parametrize(
        "evaluate_one_point",
        [
            lambda key: jax.random.normal(key),
            lambda key: jax.random.normal(jax.random.split(key)[0]),
        ],
        ids=["a draw", "a draw from a split key"],
    )
    def test_draws_with_a_threefry_key_are_independent_of_the_chunks(self, evaluate_one_point):
        assert _traced_point(evaluate_one_point, _key()).draws_independent_of_chunking

    def test_draws_from_a_seed_or_a_raw_key_under_the_default_generator_are_independent(self):
        from_a_seed = _traced_point(
            lambda seed: jax.random.normal(jax.random.key(seed)),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        from_a_raw_key = _traced_point(
            lambda raw_key: jax.random.normal(raw_key), jax.ShapeDtypeStruct((2,), jnp.uint32)
        )
        assert from_a_seed.draws_independent_of_chunking
        assert from_a_raw_key.draws_independent_of_chunking

    @pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg"])
    def test_draws_with_an_rbg_key_depend_on_the_chunks(self, impl):
        assert not _traced_point(
            lambda key: jax.random.normal(key), _key(impl)
        ).draws_independent_of_chunking

    def test_a_draw_inside_an_inner_jit_is_found(self):
        draw = jax.jit(lambda key: jax.random.normal(key))
        assert not _traced_point(draw, _key("rbg")).draws_independent_of_chunking

    def test_a_direct_call_of_rng_bit_generator_depends_on_the_chunks(self):
        assert not _traced_point(
            lambda state: jax.lax.rng_bit_generator(state, (3,))[1],
            jax.ShapeDtypeStruct((4,), jnp.uint32),
        ).draws_independent_of_chunking

    def test_a_raw_key_under_an_rbg_default_generator_depends_on_the_chunks(self):
        default_generator = jax.config.jax_default_prng_impl
        jax.config.update("jax_default_prng_impl", "rbg")
        try:
            point_trace = _traced_point(
                lambda raw_key: jax.random.normal(raw_key), jax.ShapeDtypeStruct((4,), jnp.uint32)
            )
        finally:
            jax.config.update("jax_default_prng_impl", default_generator)
        assert not point_trace.draws_independent_of_chunking
