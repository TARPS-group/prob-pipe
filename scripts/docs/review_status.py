"""Print each documentation page's review status, so maintainers can choose what to review next.

A page of the site and README.md carries one status label, and a reviewed page
may also mark the parts changed since its review, as CONTRIBUTING.md
§ Documentation states:

1. ``> **AI-generated.** ...`` opens a page that no maintainer has reviewed.
2. ``> **Human-validated** by <reviewer> on <YYYY-MM-DD>.`` closes a page that a
   maintainer has reviewed. A change an AI assistant makes to at most two of its
   sections puts ``> **AI-generated section.** ...`` under each changed
   section's heading.
3. ``> **AI-revised.** ...`` opens a reviewed page that an AI assistant changed
   in more than two sections. It names the earlier review, and comments mark
   each changed part: ``<!-- unreviewed -->`` and ``<!-- /unreviewed -->`` in
   markdown, and ``# unreviewed`` and ``# /unreviewed`` in a notebook's code
   cell, each opening marker with an optional note after a colon.

The pages are README.md and the pages the ``nav`` of ``mkdocs.yml`` lists. A
page's lines are a markdown page's lines, or a notebook's cells in order, and
the first and last lines are read from the markdown.

Usage::

    python scripts/docs/review_status.py

It is stdlib-only, like the CI helpers.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The opening label of a page that no maintainer has reviewed.
AI_GENERATED = (
    "> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it "
    "yet. Please report errors on the issue tracker."
)

#: The opening label of a reviewed page that an AI assistant changed in more than two sections.
AI_REVISED = re.compile(
    r"^> \*\*AI-revised\.\*\* An AI assistant changed this page after "
    r"(?P<reviewer>\S[^\n]*?) reviewed it on (?P<date>\d{4}-\d{2}-\d{2}), and no maintainer "
    r"has reviewed the changes yet\. Please report errors on the issue tracker\.$"
)

#: The closing label of a page a maintainer has reviewed, naming the reviewer and the date.
HUMAN_VALIDATED = re.compile(
    r"^> \*\*Human-validated\*\* by (?P<reviewer>\S[^\n]*?) on (?P<date>\d{4}-\d{2}-\d{2})\.$"
)

#: The label under the heading of a section changed since the page's review.
AI_SECTION = (
    "> **AI-generated section.** An AI assistant changed this section after the page was "
    "reviewed, and no maintainer has reviewed the change yet."
)

#: The most sections a reviewed page marks as changed; a larger change makes the page AI-revised.
MAX_CHANGED_SECTIONS = 2

#: The markers of a changed part of an AI-revised page, in markdown and in a notebook's code.
_OPEN = {
    "markdown": re.compile(r"^<!-- unreviewed(?:: .+)? -->$"),
    "cell": re.compile(r"^# unreviewed(?:: .+)?$"),
}
_CLOSE = {"markdown": "<!-- /unreviewed -->", "cell": "# /unreviewed"}

#: A nav entry of mkdocs.yml that names a page, with or without a title.
_NAV_PAGE = re.compile(r"^\s*-\s*(?:[^:\n]+:\s*)?(?P<path>[\w./-]+\.(?:md|ipynb))\s*$")

_HEADING = re.compile(r"^#{1,6} \S")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Label:
    """A page's status label, the review it names, and its pending parts.

    *kind* is ``"AI-generated"``, ``"AI-revised"``, or ``"human-validated"``.
    *pending* counts the changed sections of a validated page, or the marked
    parts of an AI-revised one.
    """

    kind: str
    reviewer: str | None = None
    date: str | None = None
    pending: int = 0


def pages(root: Path = ROOT) -> list[Path]:
    """README.md and every page the nav of ``mkdocs.yml`` lists, in nav order."""
    found = [root / "README.md"]
    in_nav = False
    for line in (root / "mkdocs.yml").read_text().splitlines():
        if line.startswith("nav:"):
            in_nav = True
            continue
        if in_nav and line and not line[0].isspace():
            break
        match = _NAV_PAGE.match(line) if in_nav else None
        if match:
            found.append(root / "docs" / match["path"])
    return found


def lines(path: Path) -> list[tuple[str, str]]:
    """The lines of *path* in reading order, each with its kind.

    A line is ``"markdown"``, ``"fence"`` inside a fenced code block of a
    markdown page or cell, or ``"cell"`` in a notebook's code cell.
    """
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        found: list[tuple[str, str]] = []
        for cell in cells:
            text = "".join(cell["source"]).splitlines()
            if cell["cell_type"] == "markdown":
                found.extend(_markdown_lines(text))
            elif cell["cell_type"] == "code":
                found.extend(("cell", line) for line in text)
        return found
    return _markdown_lines(path.read_text().splitlines())


def _markdown_lines(text: list[str]) -> list[tuple[str, str]]:
    found, fenced = [], False
    for line in text:
        if _FENCE.match(line):
            found.append(("fence", line))
            fenced = not fenced
        else:
            found.append(("fence" if fenced else "markdown", line))
    return found


def _status(line: str) -> Label | None:
    """The status label *line* is, or None."""
    if line == AI_GENERATED:
        return Label("AI-generated")
    for kind, pattern in (("AI-revised", AI_REVISED), ("human-validated", HUMAN_VALIDATED)):
        match = pattern.match(line)
        if match:
            return Label(kind, match["reviewer"], match["date"])
    return None


def _changed_sections(page: list[tuple[str, str]]) -> list[int]:
    """The indices of the section labels of *page*."""
    return [
        index
        for index, (kind, line) in enumerate(page)
        if kind == "markdown" and line == AI_SECTION
    ]


def _marked_parts(page: list[tuple[str, str]]) -> tuple[int, list[str]]:
    """The count of changed parts *page* marks, and what is wrong with its markers."""
    problems, depth, parts = [], 0, 0
    for kind, line in page:
        if kind not in _OPEN:
            continue
        if _OPEN[kind].match(line):
            if depth:
                problems.append("an unreviewed marker opens inside another")
            depth, parts = depth + 1, parts + 1
        elif line == _CLOSE[kind]:
            if not depth:
                problems.append("an unreviewed marker closes with none open")
            depth = max(depth - 1, 0)
    if depth:
        problems.append("an unreviewed marker is not closed")
    return parts, problems


def label(path: Path) -> Label | None:
    """The status of *path*, read from its first and last markdown lines, or None without one."""
    markdown = [line for kind, line in lines(path) if kind == "markdown" and line.strip()]
    if not markdown:
        return None
    page = lines(path)
    first, last = _status(markdown[0]), _status(markdown[-1])
    if first is not None and first.kind != "human-validated":
        found = first
    elif last is not None and last.kind == "human-validated":
        found = last
    else:
        return None
    if found.kind == "AI-revised":
        return Label(found.kind, found.reviewer, found.date, _marked_parts(page)[0])
    if found.kind == "human-validated":
        return Label(found.kind, found.reviewer, found.date, len(_changed_sections(page)))
    return found


def problems(path: Path) -> list[str]:
    """What breaks the review-label rules on *path*, empty for a page that keeps them."""
    page = lines(path)
    found = label(path)
    if found is None:
        return [
            "carries no status label: AI-generated or AI-revised as its first line, or "
            "Human-validated as its last"
        ]
    issues = []
    statuses = sum(_status(line) is not None for kind, line in page if kind == "markdown")
    if statuses != 1:
        issues.append(f"carries {statuses} status labels, and a page carries one")
    sections = _changed_sections(page)
    if sections and found.kind != "human-validated":
        issues.append("marks a changed section, which only a Human-validated page does")
    if len(sections) > MAX_CHANGED_SECTIONS:
        issues.append(
            f"marks {len(sections)} changed sections; past {MAX_CHANGED_SECTIONS}, the page "
            f"opens with the AI-revised label and marks each changed part"
        )
    for index in sections:
        previous = next((line for kind, line in reversed(page[:index]) if line.strip()), "")
        if not _HEADING.match(previous):
            issues.append("a changed-section label does not follow its section's heading")
    parts, marker_issues = _marked_parts(page)
    issues.extend(marker_issues)
    if parts and found.kind != "AI-revised":
        issues.append("marks an unreviewed part, which only an AI-revised page does")
    if found.kind == "AI-revised" and not parts:
        issues.append("is AI-revised but marks no unreviewed part")
    return issues


#: The order in which the report lists statuses, the most pressing first.
_ORDER = ("unlabeled", "AI-generated", "AI-revised", "human-validated")


def main() -> int:
    rows, failures = [], []
    for path in pages():
        name = str(path.relative_to(ROOT))
        found = label(path)
        failures.extend(f"{name}: {issue}" for issue in problems(path))
        if found is None:
            rows.append(("unlabeled", name, "", "", ""))
            continue
        pending = ""
        if found.kind == "AI-revised":
            pending = f"{found.pending} parts to review"
        elif found.pending:
            pending = f"{found.pending} sections to review"
        rows.append((found.kind, name, found.reviewer or "", found.date or "", pending))
    rows.sort(key=lambda row: (_ORDER.index(row[0]), row[4] == ""))
    width = max(len(row[1]) for row in rows)
    for kind, name, reviewer, date, pending in rows:
        print(f"{name:<{width}}  {kind:<15}  {reviewer}  {date}  {pending}".rstrip())
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
