"""The docs build links each design citation of a page to its section on GitHub.

``mkdocs_hooks.py`` rewrites the rendered HTML of each page. A docstring cites a
section of the design reference by its id, such as ``II.4``, and the hook links
the id to the section's heading on GitHub. These tests call the hook's functions
on HTML fragments, with no docs build.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]

URL = "https://example.org/design/02.md#ii4--identity"
SECTIONS = {"II.4": (URL, "II.4 — Identity")}
LINK = f'<a href="{URL}" title="Design II.4 — Identity">II.4</a>'


def _load_hooks() -> ModuleType:
    """``mkdocs_hooks.py``, which is a file at the repository root rather than a package module."""
    spec = importlib.util.spec_from_file_location("mkdocs_hooks", ROOT / "mkdocs_hooks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hooks = _load_hooks()


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p>See design II.4.</p>", f"<p>See design {LINK}.</p>"),
        ("<td>the label (II.4), twice: II.4</td>", f"<td>the label ({LINK}), twice: {LINK}</td>"),
    ],
)
def test_a_cited_section_links_to_its_heading(html, expected):
    assert hooks.link_design_citations(html, SECTIONS) == expected


@pytest.mark.parametrize(
    "html",
    [
        "<p>the kind of a return (V.0)</p>",
        "<p><code>II.4</code></p>",
        '<p><a href="#x">II.4</a></p>',
        "<pre><code>see II.4</code></pre>",
        "<p>II.4.1 and II.45 and XII.4 and probpipe.II.4</p>",
    ],
)
def test_other_text_stays_as_it_is(html):
    """An id with no section, a citation in code or in a link, and a longer number stay text."""
    assert hooks.link_design_citations(html, SECTIONS) == html


def test_the_anchor_follows_githubs_rule():
    heading = "II.4 — Identity, type & metadata: `TrackedTerm`, `Provenance`"
    assert hooks.heading_anchor(heading) == "ii4--identity-type--metadata-trackedterm-provenance"


def test_each_section_heading_of_the_design_is_a_target(tmp_path):
    (tmp_path / "02-shared.md").write_text(
        "# Part II\n\n## II.6 — `NamedTree`\n\n### II.6 — not a section\n"
    )
    assert hooks.design_sections(tmp_path) == {
        "II.6": (f"{hooks.DESIGN_URL}/02-shared.md#ii6--namedtree", "II.6 — NamedTree")
    }


def test_the_page_hook_strips_sphinx_roles_and_links_citations(monkeypatch):
    monkeypatch.setattr(hooks, "_sections", lambda: SECTIONS)
    html = "<p>:class:<code>~probpipe.Weights</code> (II.4)</p>"
    assert hooks.on_page_content(html) == f"<p><code>Weights</code> ({LINK})</p>"
