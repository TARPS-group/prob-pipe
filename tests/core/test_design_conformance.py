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

#: The members that differ from the design today, by finding name, each with
#: the problem the check reports and the behavior it awaits; the rule is in
#: ``tests/_design_conformance.py``.
_PENDING: PendingMembers = {
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
        "Constraint is a plain class whose check raises NotImplementedError, not an "
        "abstract base class",
    ),
    "TrackedTerm.with_provenance": (
        "declares parameters ['p'], has ['provenance']",
        "the parameter of with_provenance is named provenance, where the design names it p",
    ),
    "TrackedTerm.spec": (
        "is not declared",
        "each kind defines spec as its own property, and the tracked base declares none",
    ),
    "TrackedTerm.annotations": (
        "is not declared",
        "annotations is a property of the Annotated mixin that each kind adds, where the "
        "design puts it on the tracked base",
    ),
    "ParentInfo.name": (
        "is not declared",
        "ParentInfo stores the parent's label in a field named label, where the design "
        "names the field name",
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
        "the batch at one output of a batch, b.output(name), is not implemented on any batch",
    ),
    "Batch.without": (
        NOT_IMPLEMENTED,
        "the base Batch declares no without, and only RecordBatch defines one",
    ),
    "Batch.rename": (
        NOT_IMPLEMENTED,
        "the base Batch declares no rename, and only RecordBatch defines one",
    ),
    "Batch.__and__": (
        NOT_IMPLEMENTED,
        "the base Batch declares no &, and only RecordBatch defines one",
    ),
    "NamedTree.replace": (
        "declares parameters ['path', 'leaf'], has ['_updates', 'updates']",
        "replace takes its updates as a mapping or as keywords, (_updates=None, /, "
        "**updates), where the design declares one path and its leaf",
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
        "solve takes **kwargs, which the base ignores and TriangularLinOp reads as "
        "unit_diagonal, where the design declares b alone",
    ),
    "LinOp.cholesky": (
        "declares parameters [], has ['lower', 'kwargs']",
        "cholesky takes lower, which chooses the upper factor, and **kwargs, which it "
        "ignores, where the design declares no parameter",
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


_resolve = module_resolver(_MODULES)


def _declarations() -> list:
    """The declarations of the sections."""
    if not DESIGN_IS_CHECKED_OUT:
        return []
    return list(design_blocks().declarations(_SECTIONS))


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
