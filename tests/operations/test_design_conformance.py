"""The operations agree with the declarations of Part VI of the design reference.

VI.0's code blocks declare the operation model, and each later section states
its operations' signatures in prose. Every declaration is checked against the
modules the package structure places it in: a class exists and derives from
its declared bases, and each function or member takes the declared parameters,
in order, with the declared kinds and defaults.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

import probpipe.operations as operations
from probpipe.operations import (
    RouteSource,
    _condition,
    _convert,
    _density,
    _evaluate,
    _inverse,
    _joint,
    _marginal,
    _mixture,
    _moments,
    _operation,
    _sample,
    operation_registry,
)

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "scripts" / "design" / "design_blocks.py"

pytestmark = pytest.mark.skipif(
    not (_ROOT / "design").is_dir() or not _TOOL.exists(),
    reason="the design reference is not checked out",
)

#: The sections whose code blocks the operation layer realizes.
_BLOCK_SECTIONS = ("VI.0",)

#: The modules a declared name is looked up in, first match winning.
_MODULES = (
    _operation,
    _evaluate,
    _inverse,
    _sample,
    _density,
    _moments,
    _condition,
    _joint,
    _marginal,
    _mixture,
    _convert,
)

#: The operations each later section declares in its prose, as the signatures it states.
_PROSE = {
    "VI.1": ("def evaluate(f, v, fixed_args): ...",),
    "VI.2": ("def inverse(f): ...", "def log_det_jacobian(f, x): ..."),
    "VI.3": ("def sample(d, sample_shape=()): ...",),
    "VI.4": (
        "def log_prob(d, value): ...",
        "def unnormalized_log_prob(d, value): ...",
        "def prob(d, value): ...",
        "def unnormalized_prob(d, value): ...",
        "def random_log_prob(M): ...",
        "def random_unnormalized_log_prob(M): ...",
    ),
    "VI.5": (
        "def mean(d): ...",
        "def variance(d): ...",
        "def cov(d): ...",
        "def quantile(d, q): ...",
        "def expectation(d, f, fixed_args): ...",
    ),
    "VI.6": ("def condition_on(d, given): ...",),
    "VI.7": ("def joint(A, B, **align): ...",),
    "VI.8": ("def marginal(d, field): ...", "def factor(d, component_name): ..."),
    "VI.9": ("def mixture(K, mixing): ...",),
    "VI.10": ("def convert(d, target): ...",),
}

#: Declarations the implementation does not match yet, with the change each awaits.
_PENDING = {
    "OperationSummary": "the registry catalog's EntrySummary is defined in core",
    "OperationRegistry": "the registry catalog's SupportsRegistryCataloging is defined in core",
}

#: Names the package exports that Part VI does not declare, with where each is declared.
_DECLARED_ELSEWHERE = {
    "operation": "VI.0's prose: @operation registers an operation",
    "inference_method_registry": "VI.6's prose: the registry of inference methods",
}

#: The route helper each construction call names, with the source of the route it builds.
_HELPERS = {
    "structural_route": RouteSource.STRUCTURAL,
    "capability_route": RouteSource.CAPABILITY,
    "registry_route": RouteSource.REGISTRY,
    "fallback_route": RouteSource.FALLBACK,
}


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


def _declarations() -> list:
    if not _TOOL.exists():
        return []
    tool = _tool()
    found = list(tool.declarations(_BLOCK_SECTIONS))
    for section, sources in _PROSE.items():
        for source in sources:
            found.extend(tool.parse_block(section, source))
    return found


def _cases() -> list:
    cases = []
    for declaration in _declarations():
        name = _declared_name(declaration)
        marks = ()
        if name in _PENDING:
            marks = pytest.mark.pending(reason=_PENDING[name], raises=AssertionError)
        cases.append(pytest.param(declaration, id=f"{declaration.section}-{name}", marks=marks))
    return cases


def _route_constructions() -> list:
    """Each ``op.<helper>(...)`` call of VI.0's code blocks, as (operation, helper, call)."""
    if not _TOOL.exists():
        return []
    constructions = []
    for block in _tool().load_sections()["VI.0"].blocks:
        for node in ast.parse(block).body:
            call = node.value if isinstance(node, ast.Expr) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.attr in _HELPERS
            ):
                param = pytest.param(
                    call.func.value.id,
                    call.func.attr,
                    call,
                    id=f"{call.func.value.id}.{call.func.attr}",
                )
                constructions.append(param)
    return constructions


class TestDeclarationsAreImplemented:
    @pytest.mark.parametrize("declaration", _cases())
    def test_the_declaration_matches_the_implementation(self, declaration):
        findings = _tool().check([declaration], _resolve)
        assert not findings, "; ".join(f"{f.name} {f.problem}" for f in findings)

    @pytest.mark.parametrize(("name", "helper", "call"), _route_constructions())
    def test_a_route_construction_of_vi0_binds_to_its_helper(self, name, helper, call):
        op = operation_registry[name]
        placeholders = [object()] * len(call.args)
        keywords = {keyword.arg: object() for keyword in call.keywords}
        inspect.signature(getattr(op, helper)).bind(*placeholders, **keywords)
        route_name = ast.literal_eval(call.args[0])
        routes = {route.name: route for route in op.routes}
        assert route_name in routes, f"{name} has no route {route_name!r}"
        assert routes[route_name].source is _HELPERS[helper]


class TestTheVocabulary:
    def test_the_registry_holds_exactly_the_declared_operations(self):
        declared = {
            _declared_name(d)
            for section in _PROSE
            for d in _tool().parse_block(section, "\n".join(_PROSE[section]))
        }
        registered = {summary.name for summary in operation_registry.list()}
        assert registered == declared

    def test_every_operation_carries_a_route(self):
        for summary in operation_registry.list():
            assert summary.routes, f"{summary.name} has no route"

    def test_every_exported_name_is_declared(self):
        declared = {_declared_name(d) for d in _declarations()}
        undeclared = set(operations.__all__) - declared - set(_DECLARED_ELSEWHERE)
        assert not undeclared, sorted(undeclared)
