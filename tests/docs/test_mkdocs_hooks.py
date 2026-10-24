"""The docs build links each design citation of a page to its section on GitHub.

``mkdocs_hooks.py`` rewrites the rendered HTML of each page. A docstring cites a
section of the design reference by its id, such as ``II.4``, and the hook links
the id to the section's heading on GitHub. These tests call the hook's functions
on HTML fragments, with no docs build.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def _page(src_uri: str, url: str) -> SimpleNamespace:
    """A stand-in for an MkDocs page, with the attributes the hook reads."""
    return SimpleNamespace(file=SimpleNamespace(src_uri=src_uri), url=url)


def _files(urls: dict[str, str]) -> SimpleNamespace:
    """A stand-in for MkDocs' files, whose documentation pages are *urls*."""
    pages = [SimpleNamespace(src_uri=src, url=url) for src, url in urls.items()]
    return SimpleNamespace(documentation_pages=lambda: pages)


def test_the_page_hook_strips_sphinx_roles_and_links_citations(monkeypatch):
    monkeypatch.setattr(hooks, "_sections", lambda: SECTIONS)
    html = "<p>:class:<code>~probpipe.Weights</code> (II.4)</p>"
    page = _page("api/values.md", "api/values/")
    assert hooks.on_page_content(html, page=page, files=_files({})) == (
        f"<p><code>Weights</code> ({LINK})</p>"
    )


URLS = {
    "index.md": "",
    "tutorials/01_first_analysis.ipynb": "tutorials/01_first_analysis/",
    "tutorials/02_forecasting.ipynb": "tutorials/02_forecasting/",
    "get_started/installation.md": "get_started/installation/",
}


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("02_forecasting.ipynb", "../02_forecasting/"),
        ("../get_started/installation.md#colab", "../../get_started/installation/#colab"),
        ("../index.md", "../.."),
    ],
)
def test_a_notebook_links_to_a_page_by_its_file(link, expected):
    html = f'<p><a href="{link}">next</a></p>'
    rewritten = hooks.link_notebook_pages(
        html, "tutorials/01_first_analysis.ipynb", "tutorials/01_first_analysis/", URLS
    )
    assert rewritten == f'<p><a href="{expected}">next</a></p>'


@pytest.mark.parametrize(
    "link", ["https://example.org/page.md", "#section", "/docs/page.md", "data/moose.csv"]
)
def test_a_link_to_no_page_file_stays(link):
    html = f'<a href="{link}">x</a>'
    assert hooks.link_notebook_pages(html, "tutorials/01_first_analysis.ipynb", "x/", URLS) == html


def test_a_link_to_a_file_the_site_does_not_build_warns(caplog):
    html = '<a href="03_updating.ipynb">next</a>'
    rewritten = hooks.link_notebook_pages(
        html, "tutorials/01_first_analysis.ipynb", "tutorials/01_first_analysis/", URLS
    )
    assert rewritten == html
    assert "03_updating.ipynb" in caplog.text


def test_the_page_hook_rewrites_a_notebook_s_page_links_only():
    html = '<a href="02_forecasting.ipynb">next</a>'
    notebook = _page("tutorials/01_first_analysis.ipynb", "tutorials/01_first_analysis/")
    markdown = _page("tutorials/index.md", "tutorials/")
    assert hooks.on_page_content(html, page=notebook, files=_files(URLS)) == (
        '<a href="../02_forecasting/">next</a>'
    )
    assert hooks.on_page_content(html, page=markdown, files=_files(URLS)) == html
