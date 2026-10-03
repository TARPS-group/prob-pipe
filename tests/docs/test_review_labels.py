"""Every documentation page keeps the review-label rules of CONTRIBUTING.md § Documentation.

The pages are README.md and those the nav of ``mkdocs.yml`` lists, and
``scripts/docs/review_status.py`` reads their labels and checks the rules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "docs"))

import review_status

PAGES = review_status.pages()

VALIDATED = "> **Human-validated** by Jonathan Huggins on 2026-10-20."
REVISED = (
    "> **AI-revised.** An AI assistant changed this page after Jonathan Huggins reviewed it on "
    "2026-10-20, and no maintainer has reviewed the changes yet. Please report errors on the "
    "issue tracker."
)
SECTION = review_status.AI_SECTION


def _page(tmp_path: Path, *lines: str) -> Path:
    page = tmp_path / "page.md"
    page.write_text("\n".join(lines) + "\n")
    return page


def _notebook(tmp_path: Path, *cells: tuple[str, str]) -> Path:
    notebook = tmp_path / "page.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": kind, "source": text.splitlines(keepends=True), "metadata": {}}
                    for kind, text in cells
                ]
            }
        )
    )
    return notebook


def test_the_pages_are_the_readme_and_the_nav():
    names = {str(page.relative_to(review_status.ROOT)) for page in PAGES}
    assert {"README.md", "docs/index.md"} <= names
    assert all(page.exists() for page in PAGES)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(review_status.ROOT)))
def test_each_page_keeps_the_label_rules(page):
    assert review_status.problems(page) == []


class TestTheStatus:
    def test_an_unreviewed_page_opens_with_its_label(self, tmp_path):
        page = _page(tmp_path, review_status.AI_GENERATED, "", "# Title", "", "Text.")
        assert review_status.label(page) == review_status.Label("AI-generated")
        assert review_status.problems(page) == []

    def test_a_reviewed_page_closes_with_its_label(self, tmp_path):
        page = _page(tmp_path, "# Title", "", "Text.", "", VALIDATED)
        assert review_status.label(page) == review_status.Label(
            "human-validated", "Jonathan Huggins", "2026-10-20"
        )
        assert review_status.problems(page) == []

    def test_a_validated_label_at_the_top_is_no_status(self, tmp_path):
        page = _page(tmp_path, VALIDATED, "", "# Title", "", "Text.")
        assert review_status.label(page) is None
        assert review_status.problems(page)

    def test_an_unreviewed_label_at_the_end_is_no_status(self, tmp_path):
        page = _page(tmp_path, "# Title", "", "Text.", "", review_status.AI_GENERATED)
        assert review_status.label(page) is None

    def test_a_validated_label_names_the_reviewer_and_the_date(self, tmp_path):
        page = _page(tmp_path, "# Title", "", "> **Human-validated** by Jonathan Huggins.")
        assert review_status.label(page) is None

    def test_a_page_carries_one_status_label(self, tmp_path):
        page = _page(tmp_path, review_status.AI_GENERATED, "", "# Title", "", VALIDATED)
        assert any("2 status labels" in issue for issue in review_status.problems(page))

    def test_an_unlabeled_page_is_a_problem(self, tmp_path):
        page = _page(tmp_path, "# Title", "", "Text.")
        assert review_status.label(page) is None
        assert review_status.problems(page)


class TestChangedSections:
    def test_a_reviewed_page_marks_up_to_two_changed_sections(self, tmp_path):
        page = _page(
            tmp_path, "# Title", "", "## One", "", SECTION, "", "Text.", "", "## Two", "",
            SECTION, "", "Text.", "", VALIDATED,
        )  # fmt: skip
        assert review_status.label(page).pending == 2
        assert review_status.problems(page) == []

    def test_a_third_changed_section_asks_for_the_revised_label(self, tmp_path):
        sections = [line for name in ("One", "Two", "Three") for line in (f"## {name}", SECTION)]
        page = _page(tmp_path, "# Title", *sections, VALIDATED)
        assert any("AI-revised" in issue for issue in review_status.problems(page))

    def test_a_section_label_follows_its_heading(self, tmp_path):
        page = _page(tmp_path, "# Title", "", "Text.", "", SECTION, "", VALIDATED)
        assert any("heading" in issue for issue in review_status.problems(page))

    def test_only_a_validated_page_marks_a_changed_section(self, tmp_path):
        page = _page(tmp_path, review_status.AI_GENERATED, "# Title", "", SECTION)
        assert any("Human-validated" in issue for issue in review_status.problems(page))

    def test_a_heading_in_a_code_block_is_no_section(self, tmp_path):
        page = _page(tmp_path, "# Title", "", "```python", "# comment", "```", SECTION, VALIDATED)
        assert any("heading" in issue for issue in review_status.problems(page))


class TestRevisedPages:
    def test_a_revised_page_names_the_review_and_marks_its_changed_parts(self, tmp_path):
        page = _page(
            tmp_path, REVISED, "", "# Title", "", "<!-- unreviewed: the new example -->",
            "Text.", "<!-- /unreviewed -->", "", "<!-- unreviewed -->", "More.",
            "<!-- /unreviewed -->",
        )  # fmt: skip
        assert review_status.label(page) == review_status.Label(
            "AI-revised", "Jonathan Huggins", "2026-10-20", pending=2
        )
        assert review_status.problems(page) == []

    def test_a_revised_page_marks_a_part(self, tmp_path):
        page = _page(tmp_path, REVISED, "", "# Title", "", "Text.")
        assert any("marks no unreviewed part" in issue for issue in review_status.problems(page))

    @pytest.mark.parametrize(
        ("markers", "issue"),
        [
            (["<!-- unreviewed -->", "Text."], "not closed"),
            (["Text.", "<!-- /unreviewed -->"], "none open"),
            (
                ["<!-- unreviewed -->", "<!-- unreviewed -->", "<!-- /unreviewed -->"],
                "inside another",
            ),
        ],
    )
    def test_the_markers_pair(self, tmp_path, markers, issue):
        page = _page(tmp_path, REVISED, "# Title", *markers)
        assert any(issue in found for found in review_status.problems(page))

    def test_only_a_revised_page_marks_an_unreviewed_part(self, tmp_path):
        page = _page(
            tmp_path, "# Title", "<!-- unreviewed -->", "Text.", "<!-- /unreviewed -->", VALIDATED
        )
        assert any("AI-revised page" in issue for issue in review_status.problems(page))

    def test_a_marker_in_a_code_block_is_code(self, tmp_path):
        page = _page(tmp_path, REVISED, "# Title", "```html", "<!-- unreviewed -->", "```")
        assert any("marks no unreviewed part" in issue for issue in review_status.problems(page))


class TestNotebooks:
    def test_an_unreviewed_notebook_opens_its_first_markdown_cell_with_the_label(self, tmp_path):
        notebook = _notebook(
            tmp_path, ("code", "x = 1"), ("markdown", f"{review_status.AI_GENERATED}\n\n# T")
        )
        assert review_status.label(notebook) == review_status.Label("AI-generated")
        assert review_status.problems(notebook) == []

    def test_a_reviewed_notebook_closes_its_last_markdown_cell_with_the_label(self, tmp_path):
        notebook = _notebook(
            tmp_path, ("markdown", "# T"), ("markdown", VALIDATED), ("code", "x = 1")
        )
        assert review_status.label(notebook).kind == "human-validated"
        assert review_status.problems(notebook) == []

    def test_a_revised_notebook_marks_a_code_cell_with_comments(self, tmp_path):
        notebook = _notebook(
            tmp_path,
            ("markdown", f"{REVISED}\n\n# T"),
            ("code", "# unreviewed: the new budget\nx = 1\n# /unreviewed"),
        )
        assert review_status.label(notebook).pending == 1
        assert review_status.problems(notebook) == []

    def test_a_code_comment_marks_nothing_in_a_markdown_cell(self, tmp_path):
        notebook = _notebook(
            tmp_path, ("markdown", f"{REVISED}\n\n# T"), ("markdown", "# unreviewed")
        )
        assert any(
            "marks no unreviewed part" in issue for issue in review_status.problems(notebook)
        )
