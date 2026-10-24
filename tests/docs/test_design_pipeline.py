"""The design-pipeline assistant's capability table, symbol check, and corpus builder.

The assistant is in ``.claude/skills/design-pipeline/``. Its ``CAPABILITIES.md``
names ProbPipe entry points and example sections, ``check_symbols.py`` checks a
pipeline's ProbPipe names and call arguments, and ``build_corpus.py`` flattens
the documentation into part files and an index.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSISTANT = ROOT / ".claude" / "skills" / "design-pipeline"
sys.path.insert(0, str(ASSISTANT))

import build_corpus
import check_symbols


def _capability_rows() -> list[list[str]]:
    """The cells of each body row of the tables of ``CAPABILITIES.md``."""
    lines = (ASSISTANT / "CAPABILITIES.md").read_text().splitlines()
    rows = []
    for line, following in zip(lines, [*lines[1:], ""], strict=True):
        is_header = following.startswith("|---")
        if line.startswith("| ") and not is_header:
            rows.append([cell.strip() for cell in line.strip("|").split(" | ")])
    return rows


ROWS = _capability_rows()
ENTRY_POINTS = sorted({span for row in ROWS for span in re.findall(r"`([^`]+)`", row[1])})
EXAMPLES = sorted(
    {
        (match.group(1), match.group(2))
        for row in ROWS
        if (match := re.fullmatch(r'`([^`]+)`(?:, section "(.+)")?', row[2]))
    },
    key=str,
)


def _resolves(dotted: str) -> bool:
    """Whether *dotted* names a module, or an attribute chain of one, that imports."""
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:end]))
        except ImportError:
            continue
        for attribute in parts[end:]:
            if not hasattr(target, attribute):
                return False
            target = getattr(target, attribute)
        return True
    return False


class TestCapabilities:
    def test_every_row_names_entry_points_and_an_example(self):
        assert ROWS
        assert all(len(row) == 5 for row in ROWS)
        assert len(EXAMPLES) >= len({row[2] for row in ROWS})

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_each_entry_point_imports(self, name):
        dotted = name if name.startswith("probpipe.") else f"probpipe.{name}"
        assert _resolves(dotted), f"{name} does not resolve"

    @pytest.mark.parametrize(("page", "section"), EXAMPLES)
    def test_each_example_names_a_section_of_its_page(self, page, section):
        path = ROOT / page
        assert path.is_file()
        if section:
            text = build_corpus.page_text(path)
            assert section in [heading for _, _, heading in build_corpus.headings(text, 6)]


GOOD = """
import os
import probpipe as pp
from probpipe import HalfNormal, Normal, sample, workflow_run
from probpipe.validation import interval_coverage

SMOKE = os.environ.get("SMOKE") == "1"
prior = Normal("mu", 0.0, 1.0, label="prior") & HalfNormal("sigma", scale=1.0)
with workflow_run(seed=0):
    draws = sample(prior, sample_shape=10 if SMOKE else 1000)
fit = pp.condition_on.using(method="blackjax_nuts")
"""

BAD = """
import probpipe as pp
from probpipe import Normal, sampel

prior = Normal("mu", 0.0, 1.0, lable="prior")
fit = pp.condition_on.using(method="nuts")
coverage = pp.validation.coverage
"""


class TestCheckSymbols:
    def test_a_correct_script_passes(self, tmp_path, capsys):
        script = tmp_path / "pipeline.py"
        script.write_text(GOOD)
        assert check_symbols.main([str(script)]) == 0
        assert "no problems" in capsys.readouterr().out

    def test_each_kind_of_mistake_is_reported(self, tmp_path, capsys):
        script = tmp_path / "pipeline.py"
        script.write_text(BAD)
        assert check_symbols.main([str(script)]) == 1
        out = capsys.readouterr().out
        assert "pipeline.py:3: `from probpipe import sampel` fails" in out
        assert "pipeline.py:5:" in out and "unexpected keyword argument 'lable'" in out
        assert "no inference method `nuts`" in out
        assert "`probpipe.validation.coverage` is missing" in out

    def test_a_notebook_is_checked_by_cell(self, tmp_path, capsys):
        cells = [
            {"cell_type": "code", "source": ["%pip install probpipe\n", "import probpipe as pp"]},
            {"cell_type": "markdown", "source": ["# Fit"]},
            {"cell_type": "code", "source": ["pp.sample(pp.Normal('x', 0.0, 1.0), shape=3)"]},
        ]
        notebook = tmp_path / "pipeline.ipynb"
        notebook.write_text(json.dumps({"cells": cells}))
        assert check_symbols.main([str(notebook)]) == 1
        assert "pipeline.ipynb[cell 3]:1:" in capsys.readouterr().out

    def test_a_magic_that_is_the_only_statement_of_a_block_parses(self, tmp_path, capsys):
        source = "IN_COLAB = False\nif IN_COLAB:\n    %pip install --quiet \\\n        probpipe\n"
        notebook = tmp_path / "pipeline.ipynb"
        notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source}]}))
        assert check_symbols.main([str(notebook)]) == 0
        assert "no problems" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "notebook", sorted(ROOT.glob("docs/**/*.ipynb")), ids=lambda path: path.name
    )
    def test_each_documentation_notebook_parses(self, notebook):
        ast.parse(check_symbols.read_source(notebook).code)

    def test_a_reassigned_name_is_not_checked(self, tmp_path):
        script = tmp_path / "pipeline.py"
        script.write_text("from probpipe import Normal\nNormal = dict\nNormal(anything=1)\n")
        assert check_symbols.main([str(script)]) == 0

    def test_an_unreadable_file_exits_with_status_2(self, tmp_path):
        script = tmp_path / "pipeline.py"
        script.write_text("def broken(:\n")
        assert check_symbols.main([str(script)]) == 2


class TestBuildCorpus:
    def test_the_checkout_is_found_from_a_subdirectory(self, monkeypatch):
        monkeypatch.delenv("PROBPIPE_REPO", raising=False)
        monkeypatch.chdir(ROOT / "docs")
        assert build_corpus.find_repo(None) == ROOT

    def test_a_path_that_is_no_checkout_stops_the_build(self, tmp_path):
        with pytest.raises(SystemExit, match="not a ProbPipe checkout"):
            build_corpus.find_repo(tmp_path)

    def test_a_long_document_is_cut_into_parts_within_the_limits(self):
        lines = []
        for section in range(40):
            lines += [f"## Section {section}", "", "```python"]
            lines += [f"x_{section}_{n} = {n}" for n in range(60)]
            lines += ["```", ""]
        document = "# FILE: long.md\n\n" + "\n".join(lines) + "\n"
        tier = build_corpus.Tier("9-test", "Test", "A test tier.", ())
        parts = build_corpus.parts(tier, [("long.md", document)])
        assert len(parts) > 1
        for part in parts:
            assert len(part.splitlines()) <= build_corpus.PART_LINES
            assert len(part) <= build_corpus.PART_CHARS
        assert all(part.count("# FILE: long.md (continued)") == 1 for part in parts[1:])
        kept = [line for part in parts for line in part.splitlines()]
        assert [line for line in kept if line.startswith("x_")] == [
            line for line in lines if line.startswith("x_")
        ]

    def test_the_inference_table_lists_the_default_method(self):
        methods = {name: priority for name, priority, _, _ in build_corpus.inference_methods(ROOT)}
        assert methods["blackjax_nuts"] == 85
        assert methods["blackjax_hmc"] is None

    def test_the_index_lists_every_top_level_export(self):
        facts = {"version": "0", "branch": "b", "commit": "c" * 40, "built": "today"}
        index = build_corpus.build_index(ROOT, facts, build_corpus.nav_pages(ROOT), {})
        probpipe = importlib.import_module("probpipe")
        missing = [
            name
            for name in probpipe.__all__
            if not re.search(rf"^- `(class )?{name}\b", index, re.M)
        ]
        assert missing == []
