"""The stages, the gate, and the grade of the validation, beside the pipeline.

This module is part of the design-pipeline assistant's skeleton, beside
`pipeline_skeleton.py`. It holds the code that every pipeline shares, so the
pipeline itself shows the data, the models with their priors, and the named
steps. Its code serves every design, so a design changes the budgets and the
models in the pipeline. The rules are those of phases/4-proposal.md "Write the
code":

1. `resolve_stage()` resolves the stage once into a `Settings` object that
   holds the budgets, the steps to run, the models to fit with the label of
   each, and the output folder. Every step then runs the same way whatever
   the stage, so the code that a smoke run tests is the code that the fit
   runs.
2. A hold on the fits to simulated data leaves the smoke stage the data step
   and the checks of the code, and it stops the validate stage at once.
3. The gate of the fit stage reads `runs/validation.json`, which grades each
   model. A validation of the current code matches its stage, its design
   version, and the hash of the pipeline, of the code cells of
   `pipeline.ipynb`, and of each module beside them other than `report.py`.
   A model without a passed validation of the current code is fitted only
   when the user gave a reason for provisional results, and its outputs carry
   the label of its status.
4. `fits_to_rerun()` and `grade_validation()` implement RESOURCES.md stage D
   step 6, which states the rule.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.stats

import probpipe

HERE = Path(__file__).resolve().parent
OUT_DIRS = {"smoke": "runs/smoke", "validate": "runs/validate", "fit": "outputs"}
VALIDATION_FILE = HERE / "runs" / "validation.json"
SMOKE_LABEL = "smoke run, not for inference"


@dataclass(frozen=True)
class Settings:
    """What the stage decides: the budgets, the steps to run, and where results go."""

    stage: str
    seed: int
    budget: dict  # the budgets of the stage, such as the settings of the sampler
    checks_code: bool  # the checks of the pipeline's own code, in the smoke stage
    checks_priors: bool  # the prior predictive check
    validates: bool  # the validation on realistic simulated data
    fit: dict[str, str]  # the models whose fit runs, each with the label of its outputs
    remedied: frozenset[str]  # the models whose validation needed the remedy of a rerun
    real_response: bool  # True only in the fit stage, after the gate passed
    label: str  # the label of the outputs of no one model
    out_dir: Path
    record: dict  # the provenance written beside every output


def resolve_stage(
    stage: str,
    budgets: dict[str, dict],
    models: tuple[str, ...],
    design_version: int,
    seed: int,
    allow_unvalidated: str,
    hold: str,
) -> Settings:
    """The settings of `stage`, after the gate of the fit stage has passed.

    `allow_unvalidated` holds the user's reason for provisional results, and
    `hold` the date of the user's hold on fits to simulated data. Each is empty
    when it does not apply.
    """
    if stage not in budgets:
        raise SystemExit(f"STAGE must be smoke, validate, or fit, not {stage!r}.")
    if stage == "validate" and hold:
        raise SystemExit(f"The fits to simulated data are on hold since {hold}.")
    out_dir = HERE / OUT_DIRS[stage]
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "design_version": design_version,
        "code_hash": code_hash(),
        "seed": seed,
        "probpipe_version": probpipe.__version__,
        "probpipe_commit": probpipe_commit(),
    }
    if stage == "validate" and VALIDATION_FILE.exists():
        VALIDATION_FILE.unlink()  # a validate run that stops early leaves no stale grade
    fit, remedied = {}, frozenset()
    if stage == "smoke" and not hold:
        fit = dict.fromkeys(models, SMOKE_LABEL)
    if stage == "fit":
        fit, remedied = gate(record, models, allow_unvalidated)
    return Settings(
        stage=stage,
        seed=seed,
        budget=budgets[stage],
        checks_code=stage == "smoke",
        checks_priors=stage == "validate" or (stage == "smoke" and not hold),
        validates=stage == "validate",
        fit=fit,
        remedied=remedied,
        real_response=stage == "fit",
        label=SMOKE_LABEL if stage == "smoke" else "",
        out_dir=out_dir,
        record=record | {"models": fit},
    )


def gate(
    record: dict, models: tuple[str, ...], allow_unvalidated: str
) -> tuple[dict[str, str], frozenset[str]]:
    """The models that may be fit with the label of each, and those that need the remedy."""
    grades = validation_grades(record)
    fit = {}
    for name in models:
        grade = grades.get(name, {}).get("grade", "not validated")
        if grade in ("passed", "passed with notes"):
            fit[name] = ""
        elif allow_unvalidated:
            status = "validation failed" if grade == "failed" else "not validated"
            fit[name] = f"PROVISIONAL: {status}"
        else:
            print(f"Gate: the model {name} is {grade}, so it is not fit to the real data.")
    if not fit:
        raise SystemExit(
            "Gate: no model may be fit. Run STAGE=validate, or set ALLOW_UNVALIDATED "
            "to the reason for provisional results."
        )
    return fit, frozenset(name for name in fit if grades.get(name, {}).get("remedy"))


def validation_grades(record: dict) -> dict:
    """The grade of each model in the validation of the current code."""
    if not VALIDATION_FILE.exists():
        return {}
    validation = json.loads(VALIDATION_FILE.read_text())
    current = {"stage": "validate"} | {k: record[k] for k in ("design_version", "code_hash")}
    if any(validation.get(key) != value for key, value in current.items()):
        return {}
    return validation["models"]


def code_hash() -> str:
    """A hash of the pipeline's code, without `report.py` and the lines that read the environment."""
    modules = {
        Path(m.__file__).resolve()
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
    }
    sources = [
        p.read_text()
        for p in sorted(modules)
        if p.parent == HERE and p.suffix == ".py" and p.name != "report.py"
    ]
    notebook = HERE / "pipeline.ipynb"
    if notebook.exists():  # the code cells of the notebook, without their outputs
        cells = json.loads(notebook.read_text())["cells"]
        sources += ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    digest = hashlib.sha256()
    for source in sources:
        lines = source.splitlines()
        digest.update("\n".join(line for line in lines if "os.environ" not in line).encode())
    return digest.hexdigest()


def probpipe_commit() -> str:
    """The commit of the ProbPipe checkout, or "unknown" for an installed package."""
    checkout = Path(probpipe.__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


# --- The grade of the validation, by RESOURCES.md stage D step 6 --------------


def fits_pass(diagnostics: dict, num_chains: int) -> np.ndarray:
    """Whether each fit passes the criteria of RESOURCES.md stage E."""
    return (
        (diagnostics["divergences"] == 0)
        & (diagnostics["max_r_hat"] < 1.01)
        & (diagnostics["min_ess_bulk"] > 100 * num_chains)
    )


def fits_to_rerun(diagnostics: dict, num_chains: int) -> np.ndarray:
    """The fits that are rerun with a remedy: at least one, at most 5%, each just past the criteria."""
    failing = ~fits_pass(diagnostics, num_chains)
    just_past = (diagnostics["max_r_hat"] <= 1.05) & (
        diagnostics["min_ess_bulk"] >= 50 * num_chains
    )
    if failing.sum() <= 0.05 * failing.size and just_past[failing].all():
        return np.flatnonzero(failing)
    return np.array([], dtype=int)


SENTENCES = {
    "passed": "The method recovered known answers from data like yours, so its intervals can be trusted.",
    "failed": "The method did not reliably recover known answers from data like yours, so its estimates cannot be trusted yet.",
}


def grade_validation(
    z: np.ndarray,
    coverage: np.ndarray,
    contraction: np.ndarray,
    names: list[str],
    fits_ok: bool,
    rerun: int,
    quick_start: bool,
) -> dict:
    """The grade of one group of datasets drawn from the prior, with its sentence and notes.

    `z` holds the z-scores of shape (dataset, coordinate), `coverage` the 90%
    coverage of each coordinate, `contraction` its mean contraction, and
    `names` its name, which each note uses.
    `fits_ok` says whether every fit passes the criteria of stage E, after the
    rerun of the `rerun` fits that `fits_to_rerun()` chose.
    """
    n = z.shape[0]
    results = {"bias": z.mean(axis=0) * np.sqrt(n), "single misses": z}
    if not quick_start:  # quick start reports the coverage without a grade
        results["coverage"] = (coverage - 0.9) / np.sqrt(0.9 * 0.1 / n)
    # The coverage fails only below its level, and the other results on either side.
    worst = {k: -v.min() if k == "coverage" else np.abs(v).max() for k, v in results.items()}
    if not fits_ok or any(worst[k] > bound(0.0027, v.size) for k, v in results.items()):
        return {"grade": "failed", "sentence": SENTENCES["failed"], "notes": []}
    notes = [f"{rerun} fits passed only after a rerun with the remedy"] if rerun else []
    notes += [
        f"the {k} lie between their two bounds"
        for k, v in results.items()
        if np.abs(v).max() > bound(0.0455, v.size)
    ]
    notes += [f"{names[j]} has a contraction below 0.1" for j in np.flatnonzero(contraction < 0.1)]
    if not notes:
        return {"grade": "passed", "sentence": SENTENCES["passed"], "notes": []}
    sentence = SENTENCES["passed"] + " Notes: " + "; ".join(notes) + "."
    return {"grade": "passed with notes", "sentence": sentence, "notes": notes}


def coordinate_names(model, response: str) -> list[str]:
    """The name of each coordinate of the parameters, in the order of the posterior summaries."""
    names = []
    for name, spec in model.event_spec.outputs.items():
        size = int(np.prod(spec.shape))
        if name != response:
            names += [name] if size == 1 else [f"{name}[{i}]" for i in range(size)]
    return names


def bound(tail: float, results: int) -> float:
    """The bound whose two-sided tail over `results` results totals `tail`."""
    return float(scipy.stats.norm.isf(tail / (2 * results)))
