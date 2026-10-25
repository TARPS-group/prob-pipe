"""The stages, the gate, and the grade of the validation, beside the pipeline.

This module is part of the design-pipeline assistant's skeleton, beside
`pipeline_skeleton.py`. It holds the code that every pipeline shares, so the
pipeline itself shows the data, the model with its priors, and the named
steps. Copy it unchanged, and change the budgets in the pipeline. The rules
are those of phases/4-proposal.md "Write the code":

1. `resolve_stage()` resolves the stage once into a `Settings` object that
   holds the budgets, the steps to run, the output folder, and the result of
   the gate. Every step then runs the same way whatever the stage, so the code
   that a smoke run tests is the code that the fit runs.
2. The smoke stage runs the data step, the checks of the code, the prior
   predictive draws, and one fit to one simulated dataset with the steps after
   it. The validate stage runs the prior predictive check and the validation.
   The fit stage runs the fit to the real data and the steps after it.
3. A hold on the fits to simulated data leaves the smoke stage the data step
   and the checks of the code, and it stops the validate stage at once.
4. The gate of the fit stage reads `runs/validation.json`. A validation of the
   current code matches its stage, its design version, and the hash of the
   pipeline and of each module beside it other than `report.py`. A change to
   the reporting therefore keeps a passed validation, and a change to the
   model forces a new one. A model without a passed validation of the current
   code is fitted only when the user gave a reason for provisional results,
   and its outputs carry the label of its status.
5. `grade_validation()` grades by RESOURCES.md stage D step 6.
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


@dataclass(frozen=True)
class Settings:
    """What the stage decides: the budgets, the steps to run, and where results go."""

    stage: str
    seed: int
    budget: dict  # the budgets of the stage, such as the settings of the sampler
    checks_code: bool  # the checks of the pipeline's own code, in the smoke stage
    checks_priors: bool  # the prior predictive check
    validates: bool  # the validation on realistic simulated data
    fits: bool  # the fit and the steps after it
    real_response: bool  # True only in the fit stage, after the gate passed
    label: str  # the label of every output, such as "smoke run, not for inference"
    out_dir: Path
    validation_file: Path
    record: dict  # the provenance written beside every output


def resolve_stage(
    stage: str,
    budgets: dict[str, dict],
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
        VALIDATION_FILE.unlink()  # a validate run that stops early leaves no stale file
    label = {"smoke": "smoke run, not for inference", "validate": ""}.get(stage)
    if stage == "fit":
        label = gate(record, allow_unvalidated)
    simulated_fits = stage == "smoke" and not hold
    return Settings(
        stage=stage,
        seed=seed,
        budget=budgets[stage],
        checks_code=stage == "smoke",
        checks_priors=stage == "validate" or simulated_fits,
        validates=stage == "validate",
        fits=stage == "fit" or simulated_fits,
        real_response=stage == "fit",
        label=label,
        out_dir=out_dir,
        validation_file=VALIDATION_FILE,
        record=record | {"label": label},
    )


def gate(record: dict, allow_unvalidated: str) -> str:
    """Stop unless the fit may run, and return the label of its outputs."""
    status = validation_status(record)
    if status in ("passed", "passed with notes"):
        return ""
    if not allow_unvalidated:
        raise SystemExit(
            f"Gate: the model is {status}, so it is not fit to the real data. "
            "Run STAGE=validate, or set ALLOW_UNVALIDATED to the reason for provisional results."
        )
    return f"PROVISIONAL: {'validation failed' if status == 'failed' else 'not validated'}"


def validation_status(record: dict) -> str:
    """The grade of the validation of the current code, or "not validated"."""
    if not VALIDATION_FILE.exists():
        return "not validated"
    validation = json.loads(VALIDATION_FILE.read_text())
    current = {"stage": "validate"} | {k: record[k] for k in ("design_version", "code_hash")}
    if any(validation.get(key) != value for key, value in current.items()):
        return "not validated"
    return validation["grade"]


def code_hash() -> str:
    """A hash of the pipeline and the modules beside it, without `report.py` and the stage lines."""
    paths = sorted(
        {
            Path(module.__file__).resolve()
            for module in list(sys.modules.values())
            if getattr(module, "__file__", None)
        }
    )
    digest = hashlib.sha256()
    for path in paths:
        if path.parent == HERE and path.suffix == ".py" and path.name != "report.py":
            lines = path.read_text().splitlines()
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


SENTENCES = {
    "passed": "The method recovered known answers from data like yours, so its intervals can be trusted.",
    "failed": "The method did not reliably recover known answers from data like yours, so its estimates cannot be trusted yet.",
}


def grade_validation(z: np.ndarray, coverage: np.ndarray, fits_ok: bool, n: int) -> tuple[str, str]:
    """The grade and its sentence, with the prior draws as one group of `n` datasets."""
    shortfall = (0.9 - coverage) / np.sqrt(0.9 * 0.1 / n)  # in Monte Carlo standard errors
    bias = z.mean(axis=0) * np.sqrt(n)  # in standard errors of 1/sqrt(n)
    criteria = [shortfall, np.abs(bias), np.abs(z)]
    if not fits_ok or any(c.max() > bound(0.0027, c.size) for c in criteria):
        return "failed", SENTENCES["failed"]
    if all(np.abs(c).max() <= bound(0.0455, c.size) for c in criteria):
        return "passed", SENTENCES["passed"]
    note = " Some results lie between their two bounds, as the table shows."
    return "passed with notes", SENTENCES["passed"] + note


def bound(tail: float, results: int) -> float:
    """The bound whose two-sided tail over `results` results totals `tail`."""
    return float(scipy.stats.norm.isf(tail / (2 * results)))
