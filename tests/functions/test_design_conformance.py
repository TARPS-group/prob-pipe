"""The function engine agrees with the code blocks of the design reference.

Every class, member, field, and name that Part V and the Function base of
III.3 declare is checked against the modules the package structure places it
in: the class exists and derives from the declared bases, and each member takes
the declared parameters, in order, with the declared kinds and defaults.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import probpipe.functions as functions
import probpipe.testing as testing
import probpipe.values as values
from probpipe.core import _evaluation_points, _execution_claims, _executors
from probpipe.core import config as _config
from probpipe.functions import (
    _call,
    _context,
    _function,
    _reparameterization,
    _replay,
    _result,
    _rules,
)
from probpipe.values import _function_base

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "scripts" / "design" / "design_blocks.py"

pytestmark = pytest.mark.skipif(
    not (_ROOT / "design").is_dir() or not _TOOL.exists(),
    reason="the design reference is not checked out",
)

#: The sections the engine and its value-layer base realize, in the order the reference gives them.
_SECTIONS = (
    "III.3",
    "V.1",
    "V.2",
    "V.3",
    "V.4",
    "V.5",
    "V.6",
    "V.7",
    "V.8",
    "V.9",
    "V.10",
    "V.11",
    "V.12",
    "V.13",
)

#: The modules a declared name is looked up in, first match winning. The
#: session's defaults of the controls are settings of the core configuration.
_MODULES = (
    _function_base,
    _function,
    _call,
    _result,
    _rules,
    _context,
    _replay,
    _reparameterization,
    _config,
    _execution_claims,
    _evaluation_points,
    _executors,
    testing,
)

#: Names the code blocks define as worked examples rather than as declarations.
_EXAMPLES = frozenset(
    {
        "growth",
        "predict",
        "predict_impl",
        "rate",
        "ricker",
        "forward",
        "forward_batch",
        "forward_with_adjoint",
    }
)

#: The problem the design check reports for a member that does not exist yet.
_NOT_IMPLEMENTED = "is not implemented"

#: The members the implementation does not match yet, by the name a finding
#: gives them: the problem the check reports for each, and the behavior it
#: awaits. Each has a pending case of its own, and any other difference fails,
#: including another problem with a listed member.
_PENDING: dict[str, tuple[str, str]] = {
    "Function.__getitem__": (
        _NOT_IMPLEMENTED,
        "the function of the outputs at paths, f[paths], is not implemented",
    ),
    "Function.output": (
        _NOT_IMPLEMENTED,
        "the function of one output, f.output(name), is not implemented",
    ),
    "Function.without": (
        _NOT_IMPLEMENTED,
        "a function without some outputs, f.without(*paths), is not implemented",
    ),
    "Function.rename": (
        _NOT_IMPLEMENTED,
        "renaming a function's inputs and outputs, f.rename(...), is not implemented",
    ),
    "Function.__and__": (_NOT_IMPLEMENTED, "merging two functions with & is not implemented"),
}

#: Public names of the engine that the prose declares rather than a code block.
_PROSE_DECLARED = frozenset({"function", "workflow_run", "replay_run", "evaluation_rule_registry"})


def _tool():
    if "design_blocks" in sys.modules:
        return sys.modules["design_blocks"]
    spec = importlib.util.spec_from_file_location("design_blocks", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["design_blocks"] = module
    spec.loader.exec_module(module)
    return module


def _resolve(name: str):
    for module in _MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _declared_name(declaration) -> str:
    member = getattr(declaration, "member", None)
    return member.name if member is not None else declaration.name


def _cases() -> list:
    if not _TOOL.exists():
        return []
    tool = _tool()
    cases = []
    for declaration in tool.declarations(_SECTIONS):
        name = _declared_name(declaration)
        if name in _EXAMPLES:
            continue
        cases.append(pytest.param(declaration, id=f"{declaration.section}-{name}"))
    return cases


def _pending_member_cases() -> list:
    return [
        pytest.param(
            member_name,
            id=member_name,
            marks=pytest.mark.pending(reason=reason, raises=AssertionError),
        )
        for member_name, (_, reason) in _PENDING.items()
    ]


class TestDeclarationsAreImplemented:
    @pytest.mark.parametrize("declaration", _cases())
    def test_the_declaration_matches_the_implementation(self, declaration):
        findings = _tool().check([declaration], _resolve)
        unexpected = [
            f"{f.name} {f.problem}"
            for f in findings
            if f.name not in _PENDING or f.problem != _PENDING[f.name][0]
        ]
        assert not unexpected, "; ".join(unexpected)

    @pytest.mark.parametrize("member_name", _pending_member_cases())
    def test_a_pending_member_matches_its_declaration(self, member_name):
        tool = _tool()
        declarations = [
            d for d in tool.declarations(_SECTIONS) if _declared_name(d) not in _EXAMPLES
        ]
        findings = [f for f in tool.check(declarations, _resolve) if f.name == member_name]
        assert not findings, "; ".join(f"{f.name} {f.problem}" for f in findings)

    def test_the_examples_are_the_only_declarations_skipped(self):
        tool = _tool()
        declared = {_declared_name(d) for d in tool.declarations(_SECTIONS)}
        assert declared >= _EXAMPLES


class TestExportsAreDeclared:
    def test_every_exported_engine_name_is_declared(self):
        """A public name the reference does not declare is drift."""
        tool = _tool()
        declared = {_declared_name(d) for d in tool.declarations(_SECTIONS)} - _EXAMPLES
        exported = set(functions.__all__) | set(values.__all__)
        undeclared = exported - declared - _PROSE_DECLARED
        assert not undeclared, sorted(undeclared)

    def test_the_errors_of_the_stack_are_exported_where_the_package_structure_places_them(self):
        assert functions.ApplicabilityError is _call.ApplicabilityError
        assert functions.ResultKindError is _result.ResultKindError
        assert functions.ResultSchemaError is _result.ResultSchemaError

    def test_the_function_capabilities_are_defined_with_the_function_base(self):
        for name in ("SupportsInverse", "SupportsLogDetJacobian", "SupportsDifferentiation"):
            assert getattr(values, name) is getattr(_function_base, name)
        assert values.is_invertible is _function_base.is_invertible
        assert values.is_differentiable is _function_base.is_differentiable

    def test_the_bijector_factory_is_defined_with_the_engine(self):
        assert functions.bijector_for is _reparameterization.bijector_for
        assert functions.register_bijector is _reparameterization.register_bijector
