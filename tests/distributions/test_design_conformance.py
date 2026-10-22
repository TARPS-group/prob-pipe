"""The distribution layer agrees with the code blocks of the design reference.

Every class, member, field, and name that the distribution sections declare is
checked against the modules the package structure places it in: the class
exists and derives from the declared bases, and each member takes the declared
parameters, in order, with the declared kinds and defaults.
"""

from __future__ import annotations

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
    _from_functions,
    _views,
)
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

#: The sections the distribution layer realizes, in the order the reference gives them.
_SECTIONS = ("III.7", "III.8", "III.9", "III.10", "IV.1", "IV.2", "IV.3", "IV.4", "VII.2")

#: The modules a declared name is looked up in, first match winning.
_MODULES = (
    _distribution,
    _views,
    _capabilities,
    _conditional,
    _from_functions,
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

#: The members that differ from the design today, by finding name, each with
#: the problem the check reports and the behavior it awaits. The rule is stated
#: in ``tests/_design_conformance.py``.
_PENDING: PendingMembers = {
    "Distribution.output": (
        NOT_IMPLEMENTED,
        "the term at one output of a law, d.output(name), is not implemented",
    ),
    "Distribution.without": (
        NOT_IMPLEMENTED,
        "a law over its event without some paths, d.without(*paths), is not implemented",
    ),
    "ConditionalDistribution.__getitem__": (
        NOT_IMPLEMENTED,
        "the marginal kernel at paths, k[paths], is not implemented",
    ),
    "ConditionalDistribution.output": (
        NOT_IMPLEMENTED,
        "the term at one output of a kernel, k.output(name), is not implemented",
    ),
    "ConditionalDistribution.without": (
        NOT_IMPLEMENTED,
        "a kernel over its event without some paths, k.without(*paths), is not implemented",
    ),
}


_resolve_in_modules = module_resolver(_MODULES)


def _resolve(name: str):
    """A declared name: the merge, product, and truth operators of a law on Distribution itself."""
    if name in {"__and__", "__mul__", "__bool__"}:
        return getattr(_distribution.Distribution, name)
    return _resolve_in_modules(name)


def _declarations() -> list:
    """The declarations of the sections that this package realizes."""
    if not DESIGN_IS_CHECKED_OUT:
        return []
    return [
        declaration
        for declaration in design_blocks().declarations(_SECTIONS)
        if declared_name(declaration) not in _OTHER_PACKAGES
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

    def test_a_kernel_merges_by_the_declared_signature(self):
        design_blocks_tool = design_blocks()
        (merge_declaration,) = [
            declaration
            for declaration in design_blocks_tool.declarations(["IV.2"])
            if declared_name(declaration) == "__and__"
        ]
        findings = design_blocks_tool.check(
            [merge_declaration],
            lambda name: (
                _conditional.ConditionalDistribution.__and__ if name == "__and__" else None
            ),
        )
        assert not findings, findings


class TestExportsAreDeclared:
    def test_every_exported_distribution_kind_is_declared_or_derived_by_rule(self):
        """A public name the reference does not declare, or derive by its closure rule, is drift."""
        design_blocks_tool = design_blocks()
        declared = {
            declared_name(declaration) for declaration in design_blocks_tool.declarations(_SECTIONS)
        }
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
        design_blocks_tool = design_blocks()
        protocols = {
            declared_name(declaration)
            for declaration in design_blocks_tool.declarations(["III.8"])
            if declared_name(declaration).startswith("Supports")
            and "Conditioning" not in declared_name(declaration)
        }
        mirrored = {protocol.__name__ for protocol in _capabilities._CONDITIONAL_TWINS}
        assert mirrored == protocols
