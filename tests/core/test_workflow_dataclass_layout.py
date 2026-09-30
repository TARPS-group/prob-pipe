"""Storage and immutability contracts for private workflow dataclasses."""

from __future__ import annotations

import copy
import inspect
import pickle
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from types import ModuleType

import pytest

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


@pytest.fixture(
    params=[
        _workflow_call.WorkflowInputRef("x"),
        _workflow_callable.CallableAnchor(False, "local_function", None, None),
        _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise", transport="local_inline", stochastic_plan=None
        ),
        _workflow_rng.RandomEventIdentity(("invocation", 0), ("source", 0), ("singleton",)),
        _workflow_managed.ManagedWorkItemToken(bytes(range(16))),
    ],
    ids=["input-ref", "callable-anchor", "execution-contract", "random-event", "managed-token"],
)
def frozen_workflow_value(request):
    return request.param


class TestFrozenWorkflowValues:
    @pytest.mark.parametrize("operation", ["assign", "delete"])
    def test_fields_and_unknown_attributes_raise_frozen_instance_error(
        self, frozen_workflow_value, operation
    ):
        value = frozen_workflow_value
        originals = {field.name: getattr(value, field.name) for field in fields(value)}
        for name in (*originals, "unknown_attribute"):
            with pytest.raises(FrozenInstanceError):
                if operation == "assign":
                    setattr(value, name, object())
                else:
                    delattr(value, name)
        for name, original in originals.items():
            assert getattr(value, name) is original
        assert not hasattr(value, "unknown_attribute")

    def test_copy_and_serialization_preserve_fields(self, frozen_workflow_value):
        value = frozen_workflow_value
        for restored in (
            copy.copy(value),
            copy.deepcopy(value),
            pickle.loads(pickle.dumps(value)),
            replace(value),
        ):
            assert type(restored) is type(value)
            assert restored == value
            assert hash(restored) == hash(value)
            for field in fields(value):
                assert getattr(restored, field.name) == getattr(value, field.name)


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


def test_immutable_workflow_dataclasses_do_not_request_slots() -> None:
    classes = _workflow_dataclasses(frozen=True)

    assert classes
    assert [
        f"{candidate.__module__}.{candidate.__qualname__}"
        for candidate in classes
        if "__slots__" in candidate.__dict__
    ] == []


def test_mutable_workflow_dataclasses_use_slots() -> None:
    classes = _workflow_dataclasses(frozen=False)

    assert classes
    assert [
        f"{candidate.__module__}.{candidate.__qualname__}"
        for candidate in classes
        if "__slots__" not in candidate.__dict__ or "__dict__" in candidate.__dict__
    ] == []
