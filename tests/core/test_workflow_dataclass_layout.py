"""Object-layout contracts for private workflow dataclasses."""

from __future__ import annotations

import inspect
from dataclasses import is_dataclass
from types import ModuleType

from probpipe.core import (
    _workflow_broker,
    _workflow_call,
    _workflow_callable,
    _workflow_context,
    _workflow_descendants,
    _workflow_execution,
    _workflow_execution_contract,
    _workflow_managed,
    _workflow_plan,
    _workflow_replay,
    _workflow_rng,
)

_WORKFLOW_MODULES = (
    _workflow_broker,
    _workflow_call,
    _workflow_callable,
    _workflow_context,
    _workflow_descendants,
    _workflow_execution,
    _workflow_execution_contract,
    _workflow_managed,
    _workflow_plan,
    _workflow_replay,
    _workflow_rng,
)


def _module_dataclasses(module: ModuleType) -> tuple[type, ...]:
    return tuple(
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate.__module__ == module.__name__ and is_dataclass(candidate)
    )


def _workflow_dataclasses(*, frozen: bool) -> tuple[type, ...]:
    return tuple(
        candidate
        for module in _WORKFLOW_MODULES
        for candidate in _module_dataclasses(module)
        if candidate.__dataclass_params__.frozen is frozen
    )


def test_immutable_workflow_dataclasses_use_slots() -> None:
    classes = _workflow_dataclasses(frozen=True)

    assert classes
    assert [
        f"{candidate.__module__}.{candidate.__qualname__}"
        for candidate in classes
        if "__slots__" not in candidate.__dict__ or "__dict__" in candidate.__dict__
    ] == []


def test_mutable_workflow_dataclasses_use_slots() -> None:
    classes = _workflow_dataclasses(frozen=False)

    assert classes
    assert [
        f"{candidate.__module__}.{candidate.__qualname__}"
        for candidate in classes
        if "__slots__" not in candidate.__dict__ or "__dict__" in candidate.__dict__
    ] == []
