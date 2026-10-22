"""The function engine agrees with the code blocks of the design reference.

Every class, member, field, and name that Part V and the Function base of
III.3 declare is checked against the modules the package structure places it
in: the class exists and derives from the declared bases, and each member takes
the declared parameters, in order, with the declared kinds and defaults.
"""

from __future__ import annotations

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
from tests._design_conformance import (
    DESIGN_IS_CHECKED_OUT,
    NOT_IMPLEMENTED,
    PendingMembers,
    declared_name,
    design_blocks,
    findings_of_member,
    module_resolver,
    pending_member_cases,
    unexpected_findings,
)

pytestmark = pytest.mark.skipif(
    not DESIGN_IS_CHECKED_OUT, reason="the design reference is not checked out"
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

_AWAITS_REPORTING = "check's report of how each callable's evaluation points will run (V.9)"

#: The members that differ from the design today, by finding name, each with
#: the problem the check reports and the behavior it awaits. The rule is stated
#: in ``tests/_design_conformance.py``.
_PENDING: PendingMembers = {
    "Function.__getitem__": (
        NOT_IMPLEMENTED,
        "the function of the outputs at paths, f[paths], is not implemented",
    ),
    "Function.output": (
        NOT_IMPLEMENTED,
        "the function of one output, f.output(name), is not implemented",
    ),
    "Function.without": (
        NOT_IMPLEMENTED,
        "a function without some outputs, f.without(*paths), is not implemented",
    ),
    "Function.rename": (
        NOT_IMPLEMENTED,
        "renaming a function's inputs and outputs, f.rename(...), is not implemented",
    ),
    "Function.__and__": (NOT_IMPLEMENTED, "merging two functions with & is not implemented"),
    "CallReport.execution": ("is not declared", _AWAITS_REPORTING),
    "EvaluationReport": (NOT_IMPLEMENTED, _AWAITS_REPORTING),
    "ExecutionReport": (NOT_IMPLEMENTED, _AWAITS_REPORTING),
}

#: Public names of the engine that the prose declares rather than a code block.
_PROSE_DECLARED = frozenset({"function", "workflow_run", "replay_run", "evaluation_rule_registry"})


_resolve = module_resolver(_MODULES)


def _declarations() -> list:
    """The declarations of the sections that this package realizes."""
    if not DESIGN_IS_CHECKED_OUT:
        return []
    return [
        declaration
        for declaration in design_blocks().declarations(_SECTIONS)
        if declared_name(declaration) not in _EXAMPLES
    ]


def _findings(declaration) -> list:
    """The tool's findings for *declaration*."""
    return design_blocks().check([declaration], _resolve)


def _cases() -> list:
    return [
        pytest.param(declaration, id=f"{declaration.section}-{declared_name(declaration)}")
        for declaration in _declarations()
    ]


class TestDeclarationsAreImplemented:
    @pytest.mark.parametrize("declaration", _cases())
    def test_the_declaration_matches_the_implementation(self, declaration):
        unexpected = unexpected_findings(_findings(declaration), _PENDING)
        assert not unexpected, "; ".join(unexpected)

    @pytest.mark.parametrize("member_name", pending_member_cases(_PENDING))
    def test_a_pending_member_matches_its_declaration(self, member_name):
        findings = findings_of_member(member_name, _declarations(), _findings)
        assert not findings, "; ".join(findings)

    def test_the_examples_are_the_only_declarations_skipped(self):
        design_blocks_tool = design_blocks()
        declared = {
            declared_name(declaration) for declaration in design_blocks_tool.declarations(_SECTIONS)
        }
        assert declared >= _EXAMPLES


class TestExportsAreDeclared:
    def test_every_exported_engine_name_is_declared(self):
        """A public name the reference does not declare is drift."""
        design_blocks_tool = design_blocks()
        declared = {
            declared_name(declaration) for declaration in design_blocks_tool.declarations(_SECTIONS)
        } - _EXAMPLES
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
