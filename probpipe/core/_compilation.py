"""A compiled call that falls back to the eager call when its program does not trace.

ProbPipe compiles the computations it calls repeatedly, such as a family's
sampler or a joint's mapped conditional factor, so that a later call with
inputs of the same structure reuses the compiled program. Some computations do
not trace, such as one that converts an input to a Python number, and they run
eagerly. :func:`compiled_or_eager` makes that choice once for each key its
caller gives, and records it in the caller's cache.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, MutableMapping
from typing import Any

__all__ = ["compiled_or_eager"]

logger = logging.getLogger(__name__)


def compiled_or_eager(
    cache: MutableMapping[Hashable, Any],
    key: Hashable,
    program: Callable[..., Any],
    eager: Callable[..., Any],
    *arguments: Any,
) -> Any:
    """*program* at *arguments*, or *eager* at them when *program* does not run.

    The first call under *key* runs *program*. When it raises, the call runs
    *eager* instead. If *eager* succeeds, *cache* records *key*, and later calls
    under *key* run *eager* without attempting *program*. If *eager* raises
    too, its error propagates and nothing is recorded, so a call that fails
    eagerly, such as one with an invalid argument, does not disable the program.

    Parameters
    ----------
    cache : MutableMapping
        The caller's record of the keys whose program does not run, such as
        the transient memo of the object that owns the program.
    key : Hashable
        What decides whether *program* runs: the method, and the structure of
        the arguments when it can change the outcome.
    program : Callable
        The compiled computation.
    eager : Callable
        The same computation, uncompiled.
    *arguments : Any
        The arguments of both.

    Returns
    -------
    Any
        The result of *program*, or of *eager* when *program* raised.

    Raises
    ------
    Exception
        Whatever *eager* raises after *program* has raised.
    """
    if key not in cache:
        try:
            return program(*arguments)
        except Exception as error:
            failure = error
        result = eager(*arguments)
        cache[key] = True
        logger.debug("%r runs uncompiled: its compiled call raised %r", key, failure)
        return result
    return eager(*arguments)
