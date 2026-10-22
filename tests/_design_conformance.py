"""What the design conformance tests share: the design-block tool and the members they expect to differ.

Each package's ``test_design_conformance.py`` checks the declarations of the
design sections it realizes with ``scripts/design/design_blocks.py``. A member
the implementation does not match yet is listed in that file's pending table,
under the name a finding gives it, with the problem the tool reports for it
and the behavior it awaits. Then:

1. the test of a declaration fails on any other finding, including another
   problem with a listed member;
2. each listed member has a strict pending case of its own, so the design
   ledger counts it, and the case fails once the member matches.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

__all__ = [
    "DESIGN_IS_CHECKED_OUT",
    "NOT_IMPLEMENTED",
    "PendingMembers",
    "declared_name",
    "design_blocks",
    "findings_of_member",
    "pending_member_cases",
    "unexpected_findings",
]

_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "scripts" / "design" / "design_blocks.py"

#: Whether the design reference and its tool are checked out, which every
#: conformance test needs.
DESIGN_IS_CHECKED_OUT = (_ROOT / "design").is_dir() and _TOOL.exists()

#: The problem the tool reports for a declared name the code does not define.
NOT_IMPLEMENTED = "is not implemented"

#: A pending table, by member name: the problem the tool reports for the member,
#: or a tuple of them when it reports several, and the behavior awaited.
PendingMembers = Mapping[str, tuple[str | tuple[str, ...], str]]


def design_blocks() -> ModuleType:
    """The ``design_blocks`` tool, loaded once."""
    if "design_blocks" in sys.modules:
        return sys.modules["design_blocks"]
    spec = importlib.util.spec_from_file_location("design_blocks", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["design_blocks"] = module
    spec.loader.exec_module(module)
    return module


def declared_name(declaration: Any) -> str:
    """The name a declaration declares: its member's for a member, else its own."""
    member = getattr(declaration, "member", None)
    return member.name if member is not None else declaration.name


def unexpected_findings(findings: Iterable[Any], pending: PendingMembers) -> list[str]:
    """Each finding the pending table does not list with its problem, as ``"<name> <problem>"``."""
    return [
        f"{finding.name} {finding.problem}"
        for finding in findings
        if finding.name not in pending
        or finding.problem not in _expected_problems(pending, finding)
    ]


def pending_member_cases(pending: PendingMembers) -> list:
    """One strict pending case per listed member, each named by the member."""
    return [
        pytest.param(
            member_name,
            id=member_name,
            marks=pytest.mark.pending(reason=reason, raises=AssertionError),
        )
        for member_name, (_, reason) in pending.items()
    ]


def findings_of_member(
    member_name: str, declarations: Iterable[Any], resolve: Callable[[Any], Callable[[str], Any]]
) -> list[str]:
    """Each finding about *member_name*, as ``"<name> <problem>"``.

    Parameters
    ----------
    member_name : str
        The member, as a finding names it, such as ``"Distribution.output"``.
    declarations : iterable
        The declarations to check.
    resolve : callable
        Gives, for a declaration, the function that looks a declared name up.
    """
    tool = design_blocks()
    return [
        f"{finding.name} {finding.problem}"
        for declaration in declarations
        for finding in tool.check([declaration], resolve(declaration))
        if finding.name == member_name
    ]


def _expected_problems(pending: PendingMembers, finding: Any) -> tuple[str, ...]:
    """The problems the pending table lists for the member *finding* names."""
    expected = pending[finding.name][0]
    return (expected,) if isinstance(expected, str) else expected
