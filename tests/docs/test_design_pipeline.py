"""The design-pipeline assistant's capability table, symbol check, and corpus builder.

The assistant is in ``.claude/skills/design-pipeline/``. Its ``CAPABILITIES.md``
names ProbPipe entry points and example sections, ``check_symbols.py`` checks a
pipeline's ProbPipe names and call arguments, and ``build_corpus.py`` flattens
the documentation into part files and an index. The worked example of a
pipeline in ``templates/`` passes the symbol check and runs its smoke stage end
to end, whole and with its optional blocks deleted.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSISTANT = ROOT / ".claude" / "skills" / "design-pipeline"
TEMPLATES = ASSISTANT / "templates"
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

    def test_a_method_named_by_a_top_level_string_is_checked(self, tmp_path, capsys):
        script = tmp_path / "pipeline.py"
        script.write_text(
            'import probpipe as pp\nMETHOD = "nuts"\npp.condition_on.using(method=METHOD)\n'
        )
        assert check_symbols.main([str(script)]) == 1
        assert "no inference method `nuts`" in capsys.readouterr().out

    def test_a_call_of_an_unimplemented_export_is_reported(self, tmp_path, capsys, monkeypatch):
        probpipe = importlib.import_module("probpipe")
        monkeypatch.setattr(
            check_symbols, "_unimplemented", lambda: {id(probpipe.Normal): "Normal"}
        )
        script = tmp_path / "pipeline.py"
        script.write_text("from probpipe import Normal\nNormal('x', 0.0, 1.0)\n")
        assert check_symbols.main([str(script)]) == 1
        assert "`Normal` is not implemented" in capsys.readouterr().out

    def test_a_module_beside_the_file_that_it_imports_is_checked(self, tmp_path, capsys):
        (tmp_path / "report.py").write_text("from probpipe import sampel\n")
        (tmp_path / "unused.py").write_text("from probpipe import sampel\n")
        cells = [{"cell_type": "code", "source": "import report\nfrom report import show\n"}]
        notebook = tmp_path / "pipeline.ipynb"
        notebook.write_text(json.dumps({"cells": cells}))
        assert check_symbols.main([str(notebook)]) == 1
        out = capsys.readouterr().out
        assert out.count("`from probpipe import sampel` fails") == 1
        assert "report.py:1:" in out

    def test_a_product_of_laws_is_reported_once_per_chain(self, tmp_path, capsys):
        script = tmp_path / "pipeline.py"
        script.write_text(
            "from probpipe import Beta, Normal, conditional_distribution\n"
            "prior = (Normal('a', 0.0, 1.0) * Beta('b', 2.0, 2.0) * Normal('c', 0.0, 1.0)).with_label('p')\n"
            "likelihood = conditional_distribution(lambda a: Normal('y', a, 1.0))\n"
            "model = likelihood * prior\n"
            "merged = likelihood & prior\n"
            "scaled = 2.0 * prior\n"
        )
        assert check_symbols.main([str(script)]) == 1
        out = capsys.readouterr().out
        assert out.count("multiplies laws") == 2
        assert "pipeline.py:2:" in out and "pipeline.py:4: `likelihood * prior`" in out

    def test_an_attribute_that_an_output_spec_lacks_is_reported(self, tmp_path, capsys):
        script = tmp_path / "pipeline.py"
        script.write_text(
            "from probpipe import Normal\n"
            "prior = Normal('a', 0.0, 1.0)\n"
            "print(prior.event_spec.outputs, prior.event_spec.components)\n"
        )
        assert check_symbols.main([str(script)]) == 1
        out = capsys.readouterr().out
        assert "an `OutputSpec` has no attribute `components`" in out
        assert "`outputs`" not in out

    def test_a_reassigned_name_is_not_checked(self, tmp_path):
        script = tmp_path / "pipeline.py"
        script.write_text("from probpipe import Normal\nNormal = dict\nNormal(anything=1)\n")
        assert check_symbols.main([str(script)]) == 0

    def test_an_unreadable_file_exits_with_status_2(self, tmp_path):
        script = tmp_path / "pipeline.py"
        script.write_text("def broken(:\n")
        assert check_symbols.main([str(script)]) == 2


class TestPipelineSkeleton:
    def test_the_skeleton_passes_the_symbol_check(self, capsys):
        assert check_symbols.main([str(TEMPLATES / "pipeline_skeleton.py")]) == 0
        assert "no problems" in capsys.readouterr().out

    @staticmethod
    def _pare(workspace: Path) -> None:
        """Delete each block tagged ``OPTIONAL:``, by phases/4-proposal.md "Write the code"."""
        for path in workspace.glob("*.py"):
            kept, in_block = [], False
            for line in path.read_text().splitlines():
                if line.startswith(("# ---", "if __name__")):
                    in_block = line.startswith("# --- OPTIONAL:")
                if not in_block and not re.search(r"# OPTIONAL: .+$", line):
                    kept.append(line)
            path.write_text("\n".join(kept) + "\n")

    @classmethod
    def _run(
        cls, workspace: Path, stage: str, pared: bool = False, **settings: str
    ) -> subprocess.CompletedProcess:
        """Run the example copied to *workspace* in *stage*, from another directory."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("pandas")
        for path in TEMPLATES.iterdir():
            if path.is_file():
                shutil.copy(path, workspace)
        if pared:
            cls._pare(workspace)
        pythonpath = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
        env = os.environ | settings | {"STAGE": stage, "PYTHONPATH": pythonpath}
        return subprocess.run(
            [sys.executable, str(workspace / "pipeline_skeleton.py")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def test_the_smoke_stage_runs_every_block(self, tmp_path):
        result = self._run(tmp_path, "smoke")
        assert result.returncode == 0, result.stderr
        label = "smoke run, not for inference"
        assert f"{label}: normal: change in yield per degree Celsius:" in result.stdout
        out = tmp_path / "checks" / "smoke"
        for name in (
            "normal_answers.json",
            "student_t_prior_predictive.png",
            "model_comparison.csv",
            "normal_algorithms.csv",
            "heldout_scores.json",
            "student_t_by_year.csv",
        ):
            assert (out / name).exists(), name
        assert json.loads((out / "provenance.json").read_text())["seed"] == 20261025

    def test_the_pared_example_passes_the_symbol_check_and_runs(self, tmp_path, capsys):
        result = self._run(tmp_path, "smoke", pared=True)
        assert result.returncode == 0, result.stderr
        assert "OPTIONAL:" not in (tmp_path / "pipeline_skeleton.py").read_text().split('"""')[2]
        assert check_symbols.main([str(tmp_path / "pipeline_skeleton.py")]) == 0
        assert not (tmp_path / "checks" / "smoke" / "model_comparison.csv").exists()

    def test_a_hold_leaves_the_smoke_stage_without_a_fit(self, tmp_path):
        result = self._run(tmp_path, "smoke", pared=True, HOLD_SIMULATED_FITS="2026-10-25")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "checks" / "smoke" / "provenance.json").exists()
        assert not (tmp_path / "checks" / "smoke" / "normal_answers.json").exists()

    def test_the_fit_stage_stops_at_the_gate_without_a_validation(self, tmp_path):
        result = self._run(tmp_path, "fit", pared=True)
        assert result.returncode != 0
        assert "Gate: no model may be fit" in result.stderr
        assert not (tmp_path / "results" / "normal_answers.json").exists()

    def test_a_fit_without_a_validation_is_labeled_provisional(self, tmp_path):
        reason = "the user declined the validation"
        result = self._run(tmp_path, "fit", pared=True, ALLOW_UNVALIDATED=reason)
        assert result.returncode == 0, result.stderr
        label = "PROVISIONAL: not validated"
        assert f"{label}: student_t: change in yield per degree Celsius:" in result.stdout
        assert (tmp_path / "results" / "PROVISIONAL_normal_answers.json").exists()

    @staticmethod
    def _code_hash(workspace: Path) -> str:
        """The hash of what the validation of the example in *workspace* depends on."""
        pythonpath = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
        code = (
            "import pipeline_skeleton as p, pipeline_support as s; print(s.code_hash(p.VALIDATED))"
        )
        env = os.environ | {"PYTHONPATH": pythonpath}
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=workspace, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip().splitlines()[-1]

    def test_the_hash_covers_the_code_that_the_validation_depends_on(self, tmp_path):
        pytest.importorskip("matplotlib")
        for path in TEMPLATES.iterdir():
            if path.is_file():
                shutil.copy(path, tmp_path)

        def code_hash() -> str:
            return self._code_hash(tmp_path)

        script = tmp_path / "pipeline_skeleton.py"
        source = script.read_text()
        first = code_hash()
        step_after_fit = "def estimands(posterior, X_dry_hot: jnp.ndarray, scales: NumericRecord, settings) -> dict:\n"
        script.write_text(source.replace(step_after_fit, step_after_fit + "    # edited\n"))
        (tmp_path / "report.py").write_text((tmp_path / "report.py").read_text() + "# edited\n")
        assert code_hash() == first
        prior = "prior_scale * jnp.array([1.5, 1.0, 1.0])"
        script.write_text(source.replace(prior, "prior_scale * jnp.array([1.5, 1.0, 2.0])"))
        assert code_hash() != first
        script.write_text(source)
        data_file = tmp_path / "example_data.csv"
        rows = data_file.read_text().splitlines()
        data_file.write_text("\n".join([*rows, rows[-1]]) + "\n")
        assert code_hash() != first

    def test_the_hash_covers_the_split_and_the_settings_of_the_fit(self, tmp_path):
        pytest.importorskip("matplotlib")
        for path in TEMPLATES.iterdir():
            if path.is_file():
                shutil.copy(path, tmp_path)
        script = tmp_path / "pipeline_skeleton.py"
        source = script.read_text()
        first = self._code_hash(tmp_path)
        script.write_text(source.replace("HOLDOUT_YEAR = 2024", "HOLDOUT_YEAR = 2023"))
        assert self._code_hash(tmp_path) != first
        split = "def hold_out("
        start = source.index(split)
        body = source.index("\n", source.index('"""', source.index('"""', start) + 3)) + 1
        script.write_text(source[:body] + "    # edited\n" + source[body:])
        assert self._code_hash(tmp_path) != first
        remedy = '"remedy": {"blackjax_nuts": FULL | REMEDY}'
        head, _, tail = source.rpartition(remedy)  # the remedy of the fit stage
        script.write_text(
            head + '"remedy": {"blackjax_nuts": FULL | {"max_num_doublings": 12}}' + tail
        )
        assert self._code_hash(tmp_path) != first

    def test_comparison_fits_and_combined_outputs_carry_each_models_status(self, tmp_path):
        """A fit at the smoke budgets after a validation that passed one model with the remedy.

        The model ``normal`` passed with the remedy, and ``student_t`` was not
        validated, so it is fitted provisionally.
        """
        for path in TEMPLATES.iterdir():
            if path.is_file():
                shutil.copy(path, tmp_path)
        script = tmp_path / "pipeline_skeleton.py"
        source = script.read_text().replace(
            "\n\ndef main(", '\nBUDGETS["fit"] = BUDGETS["smoke"]\n\n\ndef main(', 1
        )
        script.write_text(source)
        validation = {
            "stage": "validate",
            "design_version": 1,
            "code_hash": self._code_hash(tmp_path),
            "models": {"normal": {"grade": "passed", "remedy": True}},
        }
        (tmp_path / "checks").mkdir()
        (tmp_path / "checks" / "validation.json").write_text(json.dumps(validation))
        # Each call of fit() is recorded with the model of the run_model() call that made it.
        (tmp_path / "driver.py").write_text(
            "import json, sys\n"
            "import pipeline_skeleton as p\n"
            "calls, fit = [], p.fit\n"
            "def recording_fit(joint, observed, settings, method, remedy=False):\n"
            "    frame = sys._getframe(1)\n"
            "    while frame is not None and frame.f_code.co_name != 'run_model':\n"
            "        frame = frame.f_back\n"
            "    model = frame.f_locals['name'] if frame is not None else None\n"
            "    calls.append([model, method, remedy])\n"
            "    return fit(joint, observed, settings, method, remedy)\n"
            "p.fit = recording_fit\n"
            "p.main()\n"
            "print('FIT CALLS', json.dumps(calls))\n"
        )
        pythonpath = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
        env = os.environ | {
            "STAGE": "fit",
            "ALLOW_UNVALIDATED": "the user declined the validation of student_t",
            "PYTHONPATH": pythonpath,
        }
        result = subprocess.run(
            [sys.executable, str(tmp_path / "driver.py")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr
        out = tmp_path / "results"
        comparison = (out / "PROVISIONAL_model_comparison.csv").read_text()
        assert "status" in comparison.splitlines()[0]
        assert "PROVISIONAL: not validated" in comparison
        scores = json.loads((out / "PROVISIONAL_heldout_scores.json").read_text())
        assert scores["student_t"]["status"] == "PROVISIONAL: not validated"
        assert scores["normal"]["status"] == "validated"
        assert "status" in (out / "normal_calibration.csv").read_text().splitlines()[0]
        for file, label in (
            ("normal_prior_sensitivity.csv", "Sensitivity fit, priors 3× wider"),
            ("normal_algorithms.csv", "Comparison fit, tfp_nuts in place of blackjax_nuts"),
            ("normal_by_year.csv", "Per-year fit, 2023 alone"),
        ):
            label += ", not checked on simulated data"
            assert label in (out / file).read_text(), file
            assert label in result.stdout, label
        sensitivity = (out / "PROVISIONAL_student_t_prior_sensitivity.csv").read_text()
        assert "PROVISIONAL: not validated" in sensitivity
        calls = json.loads(result.stdout.split("FIT CALLS", 1)[1])
        normal_nuts = [
            remedy
            for model, method, remedy in calls
            if (model, method) == ("normal", "blackjax_nuts")
        ]
        assert len(normal_nuts) > 1 and all(normal_nuts), calls
        assert not any(remedy for model, _, remedy in calls if model == "student_t"), calls


class TestValidationGrade:
    """``grade_validation()`` and ``fits_to_rerun()`` follow RESOURCES.md stage D step 6."""

    @pytest.fixture(scope="class")
    def support(self):
        sys.path.insert(0, str(TEMPLATES))
        try:
            return importlib.import_module("pipeline_support")
        finally:
            sys.path.remove(str(TEMPLATES))

    @staticmethod
    def _grade(support, z, coverage, all_fits_pass=True, n_rerun=0, quick_start=True):
        names = [f"c{j}" for j in range(z.shape[1])]
        contraction = np.full(z.shape[1], 0.5)
        return support.grade_validation(
            z, coverage, contraction, names, all_fits_pass, n_rerun, quick_start
        )

    def test_quick_start_reports_coverage_without_a_grade(self, support):
        z = np.zeros((20, 2))
        low_coverage = np.array([0.5, 0.5])
        assert self._grade(support, z, low_coverage)["grade"] == "passed"
        assert self._grade(support, z, low_coverage, quick_start=False)["grade"] == "failed"

    def test_a_rerun_of_a_few_fits_is_a_note(self, support):
        result = self._grade(support, np.zeros((20, 2)), np.full(2, 0.9), n_rerun=1)
        assert result["grade"] == "passed with notes"
        assert "1 fits passed only after a rerun" in result["notes"][0]

    def test_a_bias_beyond_the_failing_bound_fails(self, support):
        assert self._grade(support, np.full((20, 2), 1.5), np.full(2, 0.9))["grade"] == "failed"

    def test_nonfinite_statistics_fail(self, support):
        names = ["c0", "c1"]
        z, coverage, contraction = np.zeros((20, 2)), np.full(2, 0.9), np.full(2, 0.5)
        z_nan = z.copy()
        z_nan[4, 1] = np.nan
        result = support.grade_validation(z_nan, coverage, contraction, names, True, 0, True)
        assert result["grade"] == "failed"
        assert any("finite" in note for note in result["notes"])
        contraction_inf = np.array([0.5, np.inf])
        result = support.grade_validation(z, coverage, contraction_inf, names, True, 0, True)
        assert result["grade"] == "failed"

    def test_only_a_few_fits_just_past_the_criteria_are_rerun(self, support):
        diagnostics = {
            "divergences": np.zeros(20, int),
            "max_r_hat": np.full(20, 1.001),
            "min_ess_bulk": np.full(20, 1000.0),
        }
        diagnostics["max_r_hat"][3] = 1.02
        assert support.fits_to_rerun(diagnostics, 4).tolist() == [3]
        diagnostics["max_r_hat"][5] = 1.2
        assert support.fits_to_rerun(diagnostics, 4).tolist() == []


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
        assert "could not resolve" not in index.split("## Subpackage names")[0]

    def test_an_export_that_only_raises_not_implemented_error_is_found(self, tmp_path):
        package = tmp_path / "probpipe"
        package.mkdir()
        (package / "__init__.py").write_text(
            "from ._ops import branched, called, done, raises, routed\n"
            "__all__ = ['branched', 'called', 'done', 'raises', 'routed']\n"
        )
        (package / "_ops.py").write_text(
            "def _stub():\n    raise NotImplementedError('later')\n\n"
            'def raises():\n    """Doc."""\n    raise NotImplementedError\n\n'
            "def called(x):\n    return _stub()\n\n"
            "def branched(x):\n    if x:\n        return _stub()\n    return x\n\n"
            "def done(x):\n    return x\n\n"
            "def routed(x):\n    pass\n\n"
            "routed.register_route(_stub)\n"
        )
        assert build_corpus.unimplemented_exports(tmp_path) == ["called", "raises", "routed"]

    def test_the_index_marks_each_unimplemented_export(self):
        facts = {"version": "0", "branch": "b", "commit": "c" * 40, "built": "today"}
        index = build_corpus.build_index(ROOT, facts, build_corpus.nav_pages(ROOT), {})
        for name in build_corpus.unimplemented_exports(ROOT):
            assert re.search(rf"^- `{name}\b[^\n]*`: \*\*not implemented\*\*", index, re.M)

    def test_a_lazily_exported_class_resolves_to_its_definition(self):
        signature, _ = build_corpus.summarize_symbol(ROOT, "probpipe.PyMCModel")
        assert signature.startswith("class PyMCModel(")
