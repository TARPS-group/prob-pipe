"""Regression tests for PEP 695 generics in @workflow_function.

PEP 695 ``def f[T](...)`` syntax stores type parameters on
``func.__type_params__`` rather than injecting them into the function's
globals. With ``from __future__ import annotations`` enabled, all
annotations are lazy strings, so ``get_type_hints`` would raise
``NameError: name 'T' is not defined`` on Python 3.12 unless the type
parameters are passed through ``localns``.

Python 3.13 fixed ``get_type_hints`` to consult ``__type_params__``
automatically; these tests guard the 3.12 path.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from probpipe.core.node import WorkflowFunction, workflow_function


_T = TypeVar("_T")


class _Box(Generic[_T]):
    def __init__(self, value: _T):
        self.value = value


def test_pep695_generic_function_can_be_wrapped():
    """A PEP 695 generic function should wrap without NameError."""

    @workflow_function
    def identity[T](x: _Box[T]) -> _Box[T]:
        return x

    assert isinstance(identity, WorkflowFunction)
    assert identity.__name__ == "identity"
    # The hint for ``x`` should resolve to a parameterized _Box, not raise.
    assert "x" in identity._hints
    assert "return" in identity._hints


def test_pep695_multiple_type_params():
    """Multiple PEP 695 type parameters should all resolve."""

    @workflow_function
    def pair[T, S](a: _Box[T], b: _Box[S]) -> tuple[_Box[T], _Box[S]]:
        return (a, b)

    assert "a" in pair._hints
    assert "b" in pair._hints
    assert "return" in pair._hints


def test_iterate_imports():
    """``probpipe.iterate`` (declared with PEP 695 syntax) must import."""

    from probpipe import iterate

    assert iterate.__name__ == "iterate"
