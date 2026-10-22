"""The shared abstractions and the value kinds agree with the code blocks of the design reference.

Every class, member, field, and function that Part II and sections III.1,
III.2, III.4, III.5, and III.6 declare is checked against the modules that
define it today: the class exists and derives from the declared bases, and each
member takes the declared parameters, in order, with the declared kinds and
defaults. ``design/package-structure.md`` lists where each module moves.
"""

from __future__ import annotations

import pytest

from probpipe.core import (
    _batch,
    _catalog,
    _dispatch,
    _kinds,
    _numeric,
    _numeric_array,
    _numeric_record,
    _numeric_record_batch,
    _record_batch,
    _record_spec,
    _spec_base,
    _specs,
    config,
    constraints,
    named_tree,
    provenance,
    record,
    tracked,
)
from probpipe.linalg import linear_operator
from tests._design_conformance import (
    DESIGN_IS_CHECKED_OUT,
    NOT_IMPLEMENTED,
    declared_name,
    design_blocks,
    findings_of_member,
    pending_member_cases,
    unexpected_findings,
)

pytestmark = pytest.mark.skipif(
    not DESIGN_IS_CHECKED_OUT, reason="the design reference is not checked out"
)

#: The sections whose code blocks the shared layer and the value kinds realize.
_SECTIONS = (
    "II.1",
    "II.2",
    "II.3",
    "II.4",
    "II.5",
    "II.6",
    "II.7",
    "III.1",
    "III.2",
    "III.4",
    "III.5",
    "III.6",
)

#: The modules a declared name is looked up in, first match winning.
_MODULES = (
    _spec_base,
    _kinds,
    _specs,
    _numeric,
    constraints,
    tracked,
    provenance,
    config,
    _batch,
    named_tree,
    _dispatch,
    _catalog,
    _numeric_array,
    _record_spec,
    record,
    _numeric_record,
    _record_batch,
    _numeric_record_batch,
    linear_operator,
)

#: The members the implementation does not match yet, by the name a finding
#: gives them: the problem the check reports for each, and the behavior it
#: awaits. Each has a pending case of its own, and any other difference fails,
#: including another problem with a listed member.
_PENDING: dict[str, tuple[str | tuple[str, ...], str]] = {
    "InputSpec.rename": (
        NOT_IMPLEMENTED,
        "renaming the inputs of an input spec, input_spec.rename(...), is not implemented",
    ),
    "OutputSpec.__init__": (
        (
            "declares parameters ['term_spec'], has ['args', 'term']",
            "declares parameters ['named_output'], has ['args', 'term']",
        ),
        "the constructor takes the positional and the keyword form in one signature, "
        "(*args, **term), where the design declares one overload for each",
    ),
    "Numeric.__array__": (
        "declares parameters [], has ['dtype', 'copy']",
        "the NumPy conversion takes NumPy's dtype and copy arguments, which the "
        "declaration leaves out",
    ),
    "Constraint": (
        "does not derive from ABC",
        "a constraint class is not an abstract base class",
    ),
    "TrackedTerm": (
        "does not derive from ABC",
        "the tracked base is not an abstract base class",
    ),
    "TrackedTerm.with_provenance": (
        "declares parameters ['p'], has ['provenance']",
        "the parameter of with_provenance is named provenance, where the design names it p",
    ),
    "TrackedTerm.spec": (
        "is not declared",
        "the tracked base sets spec on each instance, with no annotation on the class",
    ),
    "TrackedTerm.annotations": (
        "is not declared",
        "the tracked base sets annotations on each instance, with no annotation on the class",
    ),
    "ParentInfo.name": (
        "is not declared",
        "a provenance parent's record calls the parent's name label, where the design calls it name",
    ),
    "provenance_ancestors": (
        "declares parameters ['term'], has ['node']",
        "the walk of a term's ancestors names its argument node, where the design names it term",
    ),
    "provenance_dag": (
        "declares parameters ['term'], has ['node']",
        "the graph of a term's ancestors names its argument node, where the design names it term",
    ),
    "Batch.__getitem__": (
        "declares parameters ['key'], has ['paths_or_positions']",
        "a batch's [] names its argument paths_or_positions, where the design names it key",
    ),
    "Batch.output": (
        NOT_IMPLEMENTED,
        "the batch at one output of a batch, b.output(name), is not implemented",
    ),
    "Batch.without": (
        NOT_IMPLEMENTED,
        "a batch without some paths, b.without(*paths), is not implemented",
    ),
    "Batch.rename": (
        NOT_IMPLEMENTED,
        "renaming the paths of a batch of any kind, b.rename(...), is not implemented",
    ),
    "Batch.__and__": (
        NOT_IMPLEMENTED,
        "merging two batches of any kind with & is not implemented",
    ),
    "NamedTree.replace": (
        "declares parameters ['path', 'leaf'], has ['_updates', 'updates']",
        "replace takes a mapping of updates, where the design declares one path and its leaf",
    ),
    "NamedTree.__and__": (
        "declares other as POSITIONAL_ONLY, has POSITIONAL_OR_KEYWORD",
        "& on a tree takes its other operand by keyword too, where the design makes it "
        "positional only",
    ),
    "LinOp": (
        "does not derive from Function",
        "a linear operator is not a Function",
    ),
    "LinOp.solve": (
        "declares parameters ['b'], has ['b', 'kwargs']",
        "solve takes keyword options the declaration leaves out",
    ),
    "LinOp.cholesky": (
        "declares parameters [], has ['lower', 'kwargs']",
        "cholesky takes lower and keyword options the declaration leaves out",
    ),
    "LinOp.with_flag": (
        NOT_IMPLEMENTED,
        "a linear operator with a structural flag set, op.with_flag(...), is not implemented",
    ),
    "RecordSpec.__init__": (
        "declares parameters ['field_specs', 'fields'], has ['_field_specs', 'field_specs']",
        "the constructor names its parameters _field_specs and field_specs, where the "
        "design names them field_specs and fields",
    ),
    "Record.output": (
        NOT_IMPLEMENTED,
        "the term at one output of a record, r.output(name), is not implemented",
    ),
    "Record.select": (
        "declares parameters ['paths', 'remap'], has ['paths', 'mapping']",
        "select names its renaming argument mapping, where the design names it remap",
    ),
    "RecordBatch.__getitem__": (
        "declares parameters ['key'], has ['paths_or_positions']",
        "a record batch's [] names its argument paths_or_positions, where the design names it key",
    ),
}


def _resolve(name: str):
    for module in _MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _cases() -> list:
    if not DESIGN_IS_CHECKED_OUT:
        return []
    return [
        pytest.param(declaration, id=f"{declaration.section}-{declared_name(declaration)}")
        for declaration in design_blocks().declarations(_SECTIONS)
    ]


class TestDeclarationsAreImplemented:
    @pytest.mark.parametrize("declaration", _cases())
    def test_the_declaration_matches_the_implementation(self, declaration):
        unexpected = unexpected_findings(design_blocks().check([declaration], _resolve), _PENDING)
        assert not unexpected, "; ".join(unexpected)

    @pytest.mark.parametrize("member_name", pending_member_cases(_PENDING))
    def test_a_pending_member_matches_its_declaration(self, member_name):
        findings = findings_of_member(
            member_name,
            design_blocks().declarations(_SECTIONS),
            lambda declaration: _resolve,
        )
        assert not findings, "; ".join(findings)

    def test_every_declared_name_is_defined(self):
        """A declared class or function no module defines is drift, not a pending member."""
        undefined = [
            declared_name(declaration)
            for declaration in design_blocks().declarations(_SECTIONS)
            if getattr(declaration, "member", None) is None
            and _resolve(declared_name(declaration)) is None
        ]
        assert not undefined, undefined
