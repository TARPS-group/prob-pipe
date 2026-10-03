"""Every documentation page opens with one review label, as CONTRIBUTING.md § Documentation states.

The pages are README.md and those the nav of ``mkdocs.yml`` lists, and
``scripts/docs/review_status.py`` reads their labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "docs"))

import review_status

PAGES = review_status.pages()


def test_the_pages_are_the_readme_and_the_nav():
    names = {str(page.relative_to(review_status.ROOT)) for page in PAGES}
    assert {"README.md", "docs/index.md"} <= names
    assert all(page.exists() for page in PAGES)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(review_status.ROOT)))
def test_each_page_opens_with_one_review_label(page):
    assert review_status.label(page) is not None, f"{page.name} does not open with a review label"
    assert review_status.label_count(page) == 1


def test_a_validated_label_names_the_reviewer_and_the_date(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("> **Human-validated** by Jonathan Huggins on 2026-10-20.\n\n# Title\n")
    assert review_status.label(page) == review_status.Label(
        "human-validated", "Jonathan Huggins", "2026-10-20"
    )
    page.write_text("> **Human-validated** by Jonathan Huggins.\n\n# Title\n")
    assert review_status.label(page) is None


def test_a_notebook_carries_its_label_in_its_first_markdown_cell(tmp_path):
    notebook = tmp_path / "page.ipynb"
    cells = [
        {"cell_type": "code", "source": ["x = 1\n"], "metadata": {}, "outputs": []},
        {"cell_type": "markdown", "source": [review_status.AI_GENERATED + "\n", "\n", "# T\n"]},
    ]
    notebook.write_text(json.dumps({"cells": cells}))
    assert review_status.label(notebook) == review_status.Label("AI-generated")


def test_a_second_label_is_counted(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(f"{review_status.AI_GENERATED}\n\n# Title\n\n{review_status.AI_GENERATED}\n")
    assert review_status.label_count(page) == 2
