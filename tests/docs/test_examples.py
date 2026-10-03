"""The code of README.md, of the site's markdown pages, and of the docstrings' examples runs.

A page's Python blocks run in order as one script, in a fresh interpreter at the
repository root, so a later block uses the names an earlier one defines and a
page leaves no state behind, such as an inference method it registers. A block
that needs a service, such as a Ray cluster, follows a line ``<!-- docs-test: skip ... -->``,
which renders as nothing. A docstring's ``>>>`` examples run through doctest.
"""

from __future__ import annotations

import doctest
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "docs"))

import review_status

#: A fenced Python block, with the line before its fence.
_BLOCK = re.compile(r"(?P<before>[^\n]*)\n```python\n(?P<code>.*?)\n```", re.S)

#: The marker of a block the test skips, as an HTML comment on the line before the fence.
_SKIP = "<!-- docs-test: skip"

MARKDOWN_PAGES = [page for page in review_status.pages() if page.suffix == ".md"]

EXAMPLE_MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    for path in (ROOT / "probpipe").rglob("*.py")
    if ">>>" in path.read_text()
)


def blocks(page: Path) -> list[str]:
    """The Python blocks of *page* that run, in order."""
    return [
        match["code"]
        for match in _BLOCK.finditer("\n" + page.read_text())
        if not match["before"].lstrip().startswith(_SKIP)
    ]


def test_a_skip_marker_drops_its_block(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        "<!-- docs-test: skip, a cluster -->\n```python\nraise\n```\n\n```python\nx = 1\n```\n"
    )
    assert blocks(page) == ["x = 1"]


@pytest.mark.parametrize("page", MARKDOWN_PAGES, ids=lambda page: str(page.relative_to(ROOT)))
def test_the_python_blocks_of_a_page_run(page, tmp_path):
    code = blocks(page)
    if not code:
        return
    script = tmp_path / f"{page.stem}.py"
    script.write_text("\n\n".join(code) + "\n")
    path = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    run = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, f"{page.name} failed:\n{run.stderr[-3000:]}"


@pytest.mark.parametrize("module", EXAMPLE_MODULES)
def test_the_docstring_examples_of_a_module_run(module):
    result = doctest.testmod(
        importlib.import_module(module),
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        report=False,
    )
    assert result.failed == 0, f"{result.failed} of {result.attempted} examples failed in {module}"
