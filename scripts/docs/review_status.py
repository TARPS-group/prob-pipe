"""Print each documentation page's review label, so maintainers can choose what to review next.

A page of the site and README.md open with one of two labels, which
CONTRIBUTING.md § Documentation states:

1. ``> **AI-generated.** ...``, for a page an AI assistant drafted or changed
   that no maintainer has reviewed since;
2. ``> **Human-validated** by <reviewer> on <YYYY-MM-DD>.``, for a page a
   maintainer has read as it renders, whose code the maintainer has run.

The pages are README.md and the pages the ``nav`` of ``mkdocs.yml`` lists. A
markdown page's label is its first line, and a notebook's is the first line of
its first markdown cell.

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

#: The label of a page no maintainer has reviewed since an AI assistant drafted or changed it.
AI_GENERATED = (
    "> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it "
    "yet. Please report errors on the issue tracker."
)

#: The label of a page a maintainer has reviewed, naming the reviewer and the date.
HUMAN_VALIDATED = re.compile(
    r"^> \*\*Human-validated\*\* by (?P<reviewer>\S[^\n]*?) on (?P<date>\d{4}-\d{2}-\d{2})\.$"
)

#: A nav entry of mkdocs.yml that names a page, with or without a title.
_NAV_PAGE = re.compile(r"^\s*-\s*(?:[^:\n]+:\s*)?(?P<path>[\w./-]+\.(?:md|ipynb))\s*$")


@dataclass(frozen=True)
class Label:
    """A page's review label: ``"AI-generated"`` or ``"human-validated"``, with its reviewer and date."""

    kind: str
    reviewer: str | None = None
    date: str | None = None


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


def _lines(path: Path) -> list[str]:
    """The lines a page's label is read from: a markdown page's, or a notebook's first markdown cell's."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        markdown = next((cell for cell in cells if cell["cell_type"] == "markdown"), None)
        return [] if markdown is None else "".join(markdown["source"]).splitlines()
    return path.read_text().splitlines()


def label(path: Path) -> Label | None:
    """The review label on the first line of *path*, or None when it opens with neither form."""
    lines = _lines(path)
    first = lines[0] if lines else ""
    if first == AI_GENERATED:
        return Label("AI-generated")
    match = HUMAN_VALIDATED.match(first)
    if match:
        return Label("human-validated", match["reviewer"], match["date"])
    return None


def label_count(path: Path) -> int:
    """How many lines of *path* are a review label of either form."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        text = [
            line
            for cell in cells
            if cell["cell_type"] == "markdown"
            for line in "".join(cell["source"]).splitlines()
        ]
    else:
        text = path.read_text().splitlines()
    return sum(line == AI_GENERATED or bool(HUMAN_VALIDATED.match(line)) for line in text)


def main() -> int:
    rows = []
    for path in pages():
        found = label(path)
        name = str(path.relative_to(ROOT))
        if found is None:
            rows.append(("unlabeled", name, "", ""))
        else:
            rows.append((found.kind, name, found.reviewer or "", found.date or ""))
    rows.sort(key=lambda row: ("unlabeled", "AI-generated", "human-validated").index(row[0]))
    width = max(len(row[1]) for row in rows)
    for kind, name, reviewer, date in rows:
        print(f"{name:<{width}}  {kind:<15}  {reviewer}  {date}".rstrip())
    return 1 if any(row[0] == "unlabeled" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
