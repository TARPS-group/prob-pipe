"""The distribution layer agrees with the code blocks of the design reference.

Every class, member, field, and name that the distribution sections declare is
checked against the modules the package structure places it in: the class
exists and derives from the declared bases, and each member takes the declared
parameters, in order, with the declared kinds and defaults.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import probpipe.distributions as distributions
from probpipe.distributions import (
    _batches,
    _capabilities,
    _conditional,
    _conversion,
    _distribution,
    _empirical,
    _factored,
    _views,
)

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "scripts" / "design" / "design_blocks.py"

pytestmark = pytest.mark.skipif(
    not (_ROOT / "design").is_dir() or not _TOOL.exists(),
    reason="the design reference is not checked out",
)

#: The sections the distribution layer realizes, in the order the reference gives them.
_SECTIONS = ("III.7", "III.8", "III.9", "III.10", "IV.1", "IV.2", "IV.3", "VII.2")

#: The modules a declared name is looked up in, first match winning.
_MODULES = (
    _distribution,
    _views,
    _capabilities,
    _conditional,
    _batches,
    _factored,
    _conversion,
    _empirical,
)

#: Declarations of those sections that another package owns.
_OTHER_PACKAGES = frozenset(
    {
        "BootstrapReplicateDistribution",
        "BootstrapDistribution",
        "SmoothingKernel",
        "GaussianKernel",
        "EpanechnikovKernel",
        "KDEDistribution",
    }
)

#: Declarations the implementation does not match yet, with the change each awaits.
_PENDING: dict[str, str] = {
    "Converter": "check takes no exact_only, which the registry enforces",
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
    if name == "__mul__":
        return _distribution.Distribution.__mul__
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
        if name in _OTHER_PACKAGES:
            continue
        marks = ()
        if name in _PENDING:
            marks = pytest.mark.pending(reason=_PENDING[name], raises=AssertionError)
        cases.append(pytest.param(declaration, id=f"{declaration.section}-{name}", marks=marks))
    return cases


class TestDeclarationsAreImplemented:
    @pytest.mark.parametrize("declaration", _cases())
    def test_the_declaration_matches_the_implementation(self, declaration):
        findings = _tool().check([declaration], _resolve)
        assert not findings, "; ".join(f"{f.name} {f.problem}" for f in findings)

    def test_a_kernel_composes_by_the_declared_signature(self):
        tool = _tool()
        (declaration,) = [d for d in tool.declarations(["IV.2"]) if _declared_name(d) == "__mul__"]
        findings = tool.check(
            [declaration],
            lambda name: (
                _conditional.ConditionalDistribution.__mul__ if name == "__mul__" else None
            ),
        )
        assert not findings, findings


class TestExportsAreDeclared:
    def test_every_exported_distribution_kind_is_declared_or_derived_by_rule(self):
        """A public name the reference does not declare, or derive by its closure rule, is drift."""
        tool = _tool()
        declared = {_declared_name(d) for d in tool.declarations(_SECTIONS)}
        # III.9 names three conditional twins and derives the rest from its closure rule.
        by_rule = {twin.__name__ for twin in _capabilities._CONDITIONAL_TWINS.values()}
        new_modules = (_views, _capabilities, _conditional, _batches, _factored)
        exported = {
            name
            for name in distributions.__all__
            if any(
                getattr(module, name, None) is getattr(distributions, name)
                for module in new_modules
            )
        }
        undeclared = exported - declared - by_rule
        assert not undeclared, sorted(undeclared)

    def test_the_conditional_mirror_covers_every_unconditional_capability(self):
        tool = _tool()
        protocols = {
            _declared_name(d)
            for d in tool.declarations(["III.8"])
            if _declared_name(d).startswith("Supports") and "Conditioning" not in _declared_name(d)
        }
        mirrored = {protocol.__name__ for protocol in _capabilities._CONDITIONAL_TWINS}
        assert mirrored == protocols
