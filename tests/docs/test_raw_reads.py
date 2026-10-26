"""No page calls ``raw`` with no arguments, since ``raw`` is a property on every term.

A stale ``x.raw()`` fails or misleads at run time: it calls a wrapped function
with no arguments, or raises because a backend object is not callable. The
check reads README.md, the site's pages and notebooks, and the design.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: A call of ``raw`` with no arguments, in code or in prose.
_CALL = re.compile(r"\braw\(\s*\)")

PAGES = sorted(
    [
        ROOT / "README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "docs").rglob("*.ipynb"),
        *(ROOT / "design").rglob("*.md"),
    ]
)


def test_the_check_finds_a_call_and_leaves_reads_and_calls_with_arguments():
    assert _CALL.search("mean(d).raw()") and _CALL.search("its `raw()` is")
    assert not _CALL.search("mean(d).raw") and not _CALL.search("f.raw(x)")
    assert not _CALL.search("record._raw()")


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(ROOT)))
def test_a_page_reads_raw_as_a_property(page):
    calls = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(page.read_text().splitlines(), start=1)
        if _CALL.search(line)
    ]
    assert not calls, f"{page.relative_to(ROOT)} calls raw(); read it as .raw:\n" + "\n".join(calls)
