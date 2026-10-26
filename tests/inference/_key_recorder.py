"""The keys a gradient MCMC method's run consumes, recorded by the phase that consumes them.

:func:`record_consumed_keys` patches JAX's three key-consuming primitives,
``random_bits``, ``random_split``, and ``random_fold_in``, so that every key
any JAX random function consumes is recorded as its raw words. The recording
is labeled by phase: ``"init"`` while the method resolves its initial states
and ``"chains"`` while its runner warms up and samples the chains. Both phases
split the run's key, which is recorded under ``"run"``. A test then compares
the sets, with no derivation of its own.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import jax
import numpy as np
import pytest
from jax._src import prng

_CONSUMERS = ("random_bits", "random_split", "random_fold_in")


@contextlib.contextmanager
def record_consumed_keys(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, runner: str
) -> Iterator[dict[str, set[tuple[int, int]]]]:
    """Record the keys consumed while *module* resolves ``init`` and while its *runner* runs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The test's patcher, which undoes every patch at the test's end.
    module : ModuleType
        The method's module, which calls ``chain_initial_states`` and *runner*.
    runner : str
        The name of the module's function that warms up and samples the chains.

    Yields
    ------
    dict of str to set of tuple of int
        The raw words of each consumed key under ``"init"`` and ``"chains"``,
        and those of the run's key under ``"run"``.
    """
    keys: dict[str, set[tuple[int, int]]] = {"init": set(), "chains": set(), "run": set()}
    phase: list[str | None] = [None]

    def record(words: np.ndarray) -> None:
        if phase[0] is not None:
            keys[phase[0]].update(map(tuple, np.asarray(words).reshape(-1, 2).tolist()))

    def recording(consume: Callable[..., Any]) -> Callable[..., Any]:
        def consume_and_record(key: Any, *args: Any, **kwargs: Any) -> Any:
            jax.debug.callback(record, jax.random.key_data(key))
            return consume(key, *args, **kwargs)

        return consume_and_record

    def record_run_key(key: Any) -> None:
        words = (
            jax.random.key_data(key)
            if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
            else key
        )
        keys["run"].add(tuple(np.asarray(words).tolist()))

    def in_phase(name: str, call: Callable[..., Any]) -> Callable[..., Any]:
        def call_in_phase(*args: Any, **kwargs: Any) -> Any:
            if name == "init":
                record_run_key(kwargs["key"])
            jax.effects_barrier()
            phase[0] = name
            try:
                result = call(*args, **kwargs)
                jax.block_until_ready(result)
                jax.effects_barrier()
            finally:
                phase[0] = None
            return result

        return call_in_phase

    for name in _CONSUMERS:
        monkeypatch.setattr(prng, name, recording(getattr(prng, name)))
    monkeypatch.setattr(
        module, "chain_initial_states", in_phase("init", module.chain_initial_states)
    )
    monkeypatch.setattr(module, runner, in_phase("chains", getattr(module, runner)))
    # Compiled programs traced before the patch would consume keys unrecorded.
    jax.clear_caches()
    try:
        yield keys
    finally:
        jax.effects_barrier()
        monkeypatch.undo()
        # Compiled programs traced under the patch would keep recording.
        jax.clear_caches()
