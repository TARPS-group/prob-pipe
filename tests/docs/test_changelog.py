"""Each release of the CHANGELOG has one heading per change type.

A release is a ``## `` heading of ``CHANGELOG.md``, and its change types are its
``### `` headings, such as ``### Added`` or ``### Changed (breaking)``. A PR adds
its entry under the heading of its change type, which CONTRIBUTING.md states, so
a second heading of one type in a release splits that type's entries.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"


def _change_types() -> dict[str, list[str]]:
    """The ``### `` headings of each release, in order, outside fenced code blocks."""
    releases: dict[str, list[str]] = {}
    headings: list[str] | None = None
    in_fence = False
    for line in CHANGELOG.read_text().splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif in_fence:
            continue
        elif line.startswith("## "):
            headings = releases.setdefault(line.removeprefix("## ").strip(), [])
        elif line.startswith("### ") and headings is not None:
            headings.append(line.removeprefix("### ").strip())
    return releases


def test_each_release_has_one_heading_per_change_type():
    repeated = {
        release: sorted(kind for kind, count in Counter(kinds).items() if count > 1)
        for release, kinds in _change_types().items()
    }
    assert {release: kinds for release, kinds in repeated.items() if kinds} == {}


def test_the_first_release_is_unreleased():
    """New entries go under ``## [Unreleased]``, so it heads the list of releases."""
    assert next(iter(_change_types())) == "[Unreleased]"
