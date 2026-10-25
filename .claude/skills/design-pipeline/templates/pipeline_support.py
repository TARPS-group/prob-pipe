"""The stages, the gate, the records of a run, and the grade of the validation.

This module is part of the design-pipeline assistant's skeleton, beside
`pipeline_skeleton.py`. It holds the code that every pipeline shares, so the
pipeline itself shows the data, the models with their priors, and the named
steps. Its code serves every design, so a design changes the budgets and the
models in the pipeline. The rules are those of phases/4-proposal.md "Write the
code":

1. `resolve_stage()` reads the stage, the hold, and the reason for provisional
   results from the environment, and it resolves them once into a `Settings`
   object that holds the budgets, the steps to run, the models to fit with
   the label of each, and the output folder. Every step then runs the same
   way whatever the stage, so the code that a smoke run tests is the code
   that the fit runs.
2. A hold on the fits to simulated data leaves the smoke stage the data step
   and the checks of the code, and it stops the validate stage at once.
3. The gate of the fit stage reads `checks/validation.json`, which
   `record_validation()` writes. A validation of the current code matches
   its stage, its design version, and the hash of what the validation
   depends on: the data file, the data step, the model definitions, the
   simulation of datasets, the fit function with its methods and budgets, and
   the grading code of stage D. The pipeline declares these in `VALIDATED`, and
   `code_hash()` adds `GRADING`. A step after the fit, such as an answer, a
   check, or a comparison, and `report.py` leave the hash unchanged. A model
   without a passed validation of the current code is fitted only when the
   user gave a reason for provisional results, and its outputs carry the
   label of its status.
4. `record_run()` writes the provenance and the Python environment into the
   output folder, so each result can be traced to the code that computed it.
5. `fits_to_rerun()` and `grade_validation()` implement RESOURCES.md stage D
   step 6, which states the rule.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.stats

import probpipe
from probpipe.diagnostics import NotComputed

HERE = Path(__file__).resolve().parent
OUT_DIRS = {"smoke": "checks/smoke", "validate": "checks/validate", "fit": "results"}
VALIDATION_FILE = HERE / "checks" / "validation.json"
SMOKE_LABEL = "smoke run, not for inference"
# The quantiles of every reported interval, "median [2.5%, 97.5%]" by rule 4 of
# the assistant's Messages, and the names of their columns in each table.
QUANTILES = np.array([0.025, 0.5, 0.975])
QUANTILE_COLUMNS = [f"q{100 * q:g}" for q in QUANTILES]
# The level of the intervals whose coverage the validation grades.
COVERAGE_LEVEL = 0.9
# The two-sided tails of the bounds b2(M) of "passed" and b3(M) of "failed".
PASS_TAIL, FAIL_TAIL = 0.0455, 0.0027


@dataclass(frozen=True)
class Settings:
    """What the stage decides: the budgets, the steps to run, and where results go."""

    stage: str
    budget: dict  # the budgets of the stage, such as the settings of the sampler
    checks_code: bool  # the checks of the pipeline's own code, in the smoke stage
    checks_priors: bool  # the prior predictive check
    validates: bool  # the validation on realistic simulated data
    models_to_fit: dict[str, str]  # each model whose fit runs, with the label of its outputs
    models_with_remedy: frozenset[str]  # the models whose validation needed a rerun's remedy
    uses_real_response: bool  # True only in the fit stage, after the gate passed
    stage_label: str  # the label of the outputs of no one model
    out_dir: Path
    record: dict  # the provenance written beside every output


def resolve_stage(
    budgets: dict[str, dict], *, models: list[str], design_version: int, seed: int, validated: tuple
) -> Settings:
    """The settings of the stage, after the gate of the fit stage has passed.

    The environment variable STAGE names the stage, smoke by default.
    HOLD_SIMULATED_FITS holds the date of the user's hold on fits to simulated
    data, and ALLOW_UNVALIDATED the user's reason for provisional results.
    Each is empty when it does not apply. `validated` holds the functions and
    settings that the validation depends on, whose hash `code_hash()` computes.
    """
    stage = os.environ.get("STAGE", "smoke")
    hold = os.environ.get("HOLD_SIMULATED_FITS", "")
    allow_unvalidated = os.environ.get("ALLOW_UNVALIDATED", "")
    if stage not in budgets:
        raise SystemExit(f"STAGE must be smoke, validate, or fit, not {stage!r}.")
    if stage == "validate" and hold:
        raise SystemExit(f"The fits to simulated data are on hold since {hold}.")
    out_dir = HERE / OUT_DIRS[stage]
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "design_version": design_version,
        "code_hash": code_hash(validated),
        "seed": seed,
        "probpipe_version": probpipe.__version__,
        "probpipe_commit": probpipe_commit(),
    }
    if stage == "validate" and VALIDATION_FILE.exists():
        VALIDATION_FILE.unlink()  # a validate run that stops early leaves no stale grade
    models_to_fit, models_with_remedy = {}, frozenset()
    if stage == "smoke" and not hold:
        models_to_fit = dict.fromkeys(models, SMOKE_LABEL)
    if stage == "fit":
        models_to_fit, models_with_remedy = gate(record, models, allow_unvalidated)
    return Settings(
        stage=stage,
        budget=budgets[stage],
        checks_code=stage == "smoke",
        checks_priors=stage == "validate" or (stage == "smoke" and not hold),
        validates=stage == "validate",
        models_to_fit=models_to_fit,
        models_with_remedy=models_with_remedy,
        uses_real_response=stage == "fit",
        stage_label=SMOKE_LABEL if stage == "smoke" else "",
        out_dir=out_dir,
        record=record | {"models": models_to_fit},
    )


def gate(
    record: dict, models: list[str], allow_unvalidated: str
) -> tuple[dict[str, str], frozenset[str]]:
    """The models that may be fit with the label of each, and those that need the remedy."""
    grades = validation_grades(record)
    models_to_fit = {}
    for name in models:
        grade = grades.get(name, {}).get("grade", "not validated")
        if grade in ("passed", "passed with notes"):
            models_to_fit[name] = ""
        elif allow_unvalidated:
            status = "validation failed" if grade == "failed" else "not validated"
            models_to_fit[name] = f"PROVISIONAL: {status}"
        else:
            print(f"Gate: the model {name} is {grade}, so it is not fit to the real data.")
    if not models_to_fit:
        raise SystemExit(
            "Gate: no model may be fit. Run STAGE=validate, or set ALLOW_UNVALIDATED "
            "to the reason for provisional results."
        )
    remedied = frozenset(name for name in models_to_fit if grades.get(name, {}).get("remedy"))
    return models_to_fit, remedied


def validation_grades(record: dict) -> dict:
    """The grade of each model in the validation of the current code."""
    if not VALIDATION_FILE.exists():
        return {}
    validation = json.loads(VALIDATION_FILE.read_text())
    current = {"stage": "validate"} | {k: record[k] for k in ("design_version", "code_hash")}
    if any(validation.get(key) != value for key, value in current.items()):
        return {}
    return validation["models"]


def code_hash(validated: tuple) -> str:
    """A hash of the functions and settings that the validation depends on, with GRADING.

    A data file contributes its bytes, a function its source, and a setting
    its value, so a change to any of them makes a passed validation stale.
    """
    digest = hashlib.sha256()
    for item in (*validated, *GRADING):
        if isinstance(item, Path):  # the validation simulates at the file's covariates and size
            digest.update(item.read_bytes())
        elif callable(item):  # a ProbPipe function's source is that of the callable it wraps
            function = item.raw if isinstance(item, probpipe.Function) else item
            digest.update(inspect.getsource(function).encode())
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def probpipe_commit() -> str:
    """The commit of the ProbPipe checkout, or "unknown" for an installed package."""
    checkout = Path(probpipe.__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


# --- The records of a run ---------------------------------------------------------


def record_run(settings: Settings) -> None:
    """Write the provenance and the Python environment of the run into its output folder."""
    write_json(settings.out_dir / "provenance.json", settings.record)
    packages = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
    )
    environment = [f"python {platform.python_version()}", *packages]
    (settings.out_dir / "environment.txt").write_text("\n".join(environment) + "\n")


def record_validation(name: str, validation: dict, settings: Settings) -> None:
    """Add the model's grade to the file that the gate of the fit stage reads."""
    saved = json.loads(VALIDATION_FILE.read_text()) if VALIDATION_FILE.exists() else {}
    models = saved.get("models", {}) | {name: validation}
    write_json(VALIDATION_FILE, settings.record | {"models": models})


def divergences_of(posterior) -> int | str:
    """The divergent transitions of an MCMC fit, or a note for a method that has none.

    A method without divergences, such as random-walk Metropolis, records no
    count, and ProbPipe then reports a `NotComputed` value.
    """
    count = posterior.diagnostics.n_divergences
    return "not applicable for this method" if isinstance(count, NotComputed) else int(count)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=float))


# --- The grade of the validation, by RESOURCES.md stage D step 6 --------------


def meets_fit_criteria(diagnostics: dict, num_chains: int) -> np.ndarray:
    """Whether each fit meets the criteria of RESOURCES.md stage E."""
    return (
        (diagnostics["divergences"] == 0)
        & (diagnostics["max_r_hat"] < 1.01)
        & (diagnostics["min_ess_bulk"] > 100 * num_chains)
    )


def fits_to_rerun(diagnostics: dict, num_chains: int) -> np.ndarray:
    """The fits that are rerun with a remedy: at least one, at most 5%, each just past the criteria."""
    failing = ~meets_fit_criteria(diagnostics, num_chains)
    just_past = (diagnostics["max_r_hat"] <= 1.05) & (
        diagnostics["min_ess_bulk"] >= 50 * num_chains
    )
    if failing.sum() <= 0.05 * failing.size and just_past[failing].all():
        return np.flatnonzero(failing)
    return np.array([], dtype=int)


GRADE_SENTENCES = {
    "passed": "The method recovered known answers from data like yours, so its intervals can be trusted.",
    "failed": "The method did not reliably recover known answers from data like yours, so its estimates cannot be trusted yet.",
}


def grade_validation(
    z: np.ndarray,
    coverage: np.ndarray,
    contraction: np.ndarray,
    names: list[str],
    all_fits_pass: bool,
    n_rerun: int,
    quick_start: bool,
) -> dict:
    """The grade of one group of datasets drawn from the prior, with its sentence and notes.

    `z` holds the z-scores of shape (dataset, coordinate), `coverage` the
    coverage of each coordinate at COVERAGE_LEVEL, `contraction` its mean
    contraction, and `names` its name, which each note uses. `all_fits_pass`
    says whether every fit meets the criteria of stage E, after the rerun of
    the `n_rerun` fits that `fits_to_rerun()` chose. Beside the grade, each
    z-score beyond 3 is listed with its dataset, with the count that chance
    gives an exact method, so that no mean hides a large miss.
    """
    n = z.shape[0]
    results = {"bias": z.mean(axis=0) * np.sqrt(n), "single misses": z}
    if not quick_start:  # quick start reports the coverage without a grade
        standard_error = np.sqrt(COVERAGE_LEVEL * (1 - COVERAGE_LEVEL) / n)
        results["coverage"] = (coverage - COVERAGE_LEVEL) / standard_error
    misses = {
        "z_beyond_3": [(names[j], int(i)) for i, j in np.argwhere(np.abs(z) > 3)],
        "z_beyond_3_by_chance": FAIL_TAIL * z.size,
    }
    # The coverage fails only below its level, and the other results on either side.
    worst = {k: -v.min() if k == "coverage" else np.abs(v).max() for k, v in results.items()}
    if not all_fits_pass or any(
        worst[k] > tail_bound(FAIL_TAIL, v.size) for k, v in results.items()
    ):
        return {"grade": "failed", "sentence": GRADE_SENTENCES["failed"], "notes": []} | misses
    notes = [f"{n_rerun} fits passed only after a rerun with the remedy"] if n_rerun else []
    notes += [
        f"the {k} lie between their two bounds"
        for k, v in results.items()
        if np.abs(v).max() > tail_bound(PASS_TAIL, v.size)
    ]
    notes += [f"{names[j]} has a contraction below 0.1" for j in np.flatnonzero(contraction < 0.1)]
    if not notes:
        return {"grade": "passed", "sentence": GRADE_SENTENCES["passed"], "notes": []} | misses
    sentence = GRADE_SENTENCES["passed"] + " Notes: " + "; ".join(notes) + "."
    return {"grade": "passed with notes", "sentence": sentence, "notes": notes} | misses


def coordinate_names(model, response: str) -> list[str]:
    """The name of each coordinate of the parameters, in the order of the posterior summaries."""
    names = []
    for name, spec in model.event_spec.outputs.items():
        size = int(np.prod(spec.shape))
        if name != response:
            names += [name] if size == 1 else [f"{name}[{i}]" for i in range(size)]
    return names


def tail_bound(tail: float, n_results: int) -> float:
    """The bound b(M) whose two-sided tail over M = `n_results` results totals `tail`."""
    return float(scipy.stats.norm.isf(tail / (2 * n_results)))


# The grading code of stage D, which the hash of every validation covers.
GRADING = (
    meets_fit_criteria,
    fits_to_rerun,
    grade_validation,
    coordinate_names,
    tail_bound,
    COVERAGE_LEVEL,
    PASS_TAIL,
    FAIL_TAIL,
)
