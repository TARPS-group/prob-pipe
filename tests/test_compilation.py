"""The compilation counter that the tests of warm calls use."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tests._compilation import count_compilations


def test_the_counter_counts_the_compilation_of_a_new_program():
    x = jnp.ones(3)
    with count_compilations() as compiled:
        jax.jit(lambda x: x + 1.0)(x)
    assert compiled.count >= 1


def test_the_counter_counts_nothing_for_a_cached_program():
    program = jax.jit(lambda x: x * 2.0)
    program(jnp.ones(3))
    x = jnp.zeros(3)
    with count_compilations() as compiled:
        program(x)
    assert compiled.count == 0
