"""Tools for testing code that extends ProbPipe.

Provides:
  - :func:`check_executor` – check that an executor follows the executor
    contract, before registering it.
"""

from ._executors import check_executor

__all__ = ["check_executor"]
