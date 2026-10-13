"""A counter of the XLA compilations a block of code triggers.

The tests that a warm call compiles nothing count compilations rather than
time them, because a count does not depend on the load of the machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from jax import monitoring

__all__ = ["Compilations", "count_compilations"]

_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


@dataclass
class Compilations:
    """The number of XLA compilations counted so far."""

    count: int = 0


_open: list[Compilations] = []


def _listener(event: str, duration: float, **kwargs: object) -> None:
    """Add one to each open counter when XLA compiles a program."""
    if event == _COMPILE_EVENT:
        for counter in _open:
            counter.count += 1


monitoring.register_event_duration_secs_listener(_listener)


@contextmanager
def count_compilations() -> Iterator[Compilations]:
    """Count the XLA compilations within the block.

    Yields
    ------
    Compilations
        A counter whose ``count`` is the number of programs XLA compiled in
        the block so far. A program that JAX reuses from its in-memory cache
        or from the persistent compilation cache is not counted.
    """
    counter = Compilations()
    _open.append(counter)
    try:
        yield counter
    finally:
        _open.remove(counter)
