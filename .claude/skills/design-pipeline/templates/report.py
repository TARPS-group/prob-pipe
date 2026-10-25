"""The printing, tables, and figures of the pipeline, and the files that hold them.

This module is part of the design-pipeline assistant's skeleton, beside
`pipeline_skeleton.py`. The pipeline's steps return values, and each function
here shows and saves the values of one step of one model, so a change to the
presentation leaves the model, data, and inference code unchanged. The hash of
the validation leaves this file out for that reason, and a change here keeps a
passed validation. Replace the example's functions with one per step of the
design, and keep these rules of phases/4-proposal.md "Write the code":

1. Each function takes the model's name and prefixes the model's label, such
   as "smoke run, not for inference", and its name to what it prints and to
   each figure's title. `path()` names each file after the model, prefixed
   "PROVISIONAL_" for a provisional fit.
2. Each interval has the form "median [2.5%, 97.5%] units" of `interval()`.
3. Each figure is saved with a CSV file of the numbers it plots, and each
   answer with the draws behind it, so every number in the design document is
   read from a file in the output folder.
4. `start()` records the provenance and the Python environment in each output
   folder, and `validation()` adds each model's grade to the file that the
   gate of the fit stage reads.

The sections tagged `OPTIONAL:` show the results of the pipeline's blocks of
the same tag, and they are deleted with those blocks.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, with no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline_support import VALIDATION_FILE


def start(settings) -> None:
    """Print the stage, and record the provenance and the Python environment."""
    print(f"stage {settings.stage}, design version {settings.record['design_version']}:")
    print(f"results in {settings.out_dir}")
    write_json(settings.out_dir / "provenance.json", settings.record)
    packages = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
    )
    environment = [f"python {platform.python_version()}", *packages]
    (settings.out_dir / "environment.txt").write_text("\n".join(environment) + "\n")


def prior_predictive(name: str, y_t_per_ha: np.ndarray, settings) -> None:
    """The mean yield of each prior predictive dataset, as a histogram and a summary."""
    mean_yield = y_t_per_ha.mean(axis=1)
    say(settings, name, "prior predictive mean yield:", interval(mean_yield, "t/ha"))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(mean_yield, bins=30)
    ax.set(
        xlabel="mean yield of a simulated dataset (t/ha)",
        ylabel="datasets",
        title=titled(settings, name, "Prior predictive"),
    )
    # How to read it: the simulated means should cover plausible yields and rule out impossible ones.
    save_figure(
        fig,
        pd.DataFrame({"mean_yield_t_per_ha": mean_yield}),
        path(settings, name, "prior_predictive"),
    )


def validation(name: str, result: dict, settings) -> None:
    """Add the model's grade to the file that the gate reads, and print its verdict."""
    saved = json.loads(VALIDATION_FILE.read_text()) if VALIDATION_FILE.exists() else {}
    models = saved.get("models", {}) | {name: result}
    write_json(VALIDATION_FILE, settings.record | {"models": models})
    say(settings, name, f"Validation {result['grade']}: {result['sentence']}")


def diagnostics(name: str, posterior, settings) -> None:
    """The diagnostics table of the fit, with its divergences, by RESOURCES.md stage E."""
    table = str(posterior.diagnostics.summary_table())
    divergences = int(posterior.diagnostics.n_divergences)
    path(settings, name, "diagnostics.txt").write_text(f"{table}\ndivergences: {divergences}\n")
    say(settings, name, f"diagnostics, with {divergences} divergences:\n{table}")


def model_check(name: str, result: dict, settings) -> None:
    """The posterior predictive p-value of each test statistic."""
    p_values = {
        k.removesuffix("/p_value"): float(v) for k, v in result.items() if k.endswith("/p_value")
    }
    write_json(path(settings, name, "predictive_check.json"), p_values)
    for statistic, p_value in p_values.items():
        say(settings, name, f"posterior predictive p-value of {statistic}: {p_value:.2f}")
    # How to read it: a p-value near 0 or 1 means the model rarely reproduces that feature of the data.


def answers(name: str, result: dict, settings) -> None:
    """Each estimand as 'median [2.5%, 97.5%] units', and the draws behind it."""
    change = np.asarray(result["change_per_c"].atoms.raw)
    p_low = float(np.asarray(result["p_low_yield"]))
    summary = {"change_per_c": interval(change, "t/ha per °C"), "p_low_yield_dry_hot": p_low}
    write_json(path(settings, name, "answers.json"), summary | settings.record)
    pd.DataFrame({"yield_change_t_per_ha_per_c": change}).to_csv(
        path(settings, name, "answer_draws.csv"), index=False
    )
    say(settings, name, "change in yield per degree Celsius:", summary["change_per_c"])
    say(settings, name, f"chance of a low mean yield in a dry, hot year: {p_low:.2f}")


# --- OPTIONAL: plug-in forecast --------------------------------------------------


def plug_in(name: str, plug_in_t_per_ha: float, result: dict, settings) -> None:
    """The plug-in mean yield of a dry, hot year beside the propagated answer."""
    dry_hot = np.asarray(result["dry_hot"].atoms.raw)
    say(settings, name, f"dry, hot year at the posterior mean: {plug_in_t_per_ha:.2f} t/ha")
    say(settings, name, "dry, hot year, propagated:", interval(dry_hot, "t/ha"))


# --- OPTIONAL: scenarios ---------------------------------------------------------


def scenarios(name: str, result: dict, settings) -> None:
    """The mean yield and the chance of a low mean yield in each scenario."""
    table = pd.DataFrame(
        result["quantiles"].T, index=result["names"], columns=["q2.5", "q50", "q97.5"]
    )
    table["p_low_yield"] = result["p_low_yield"]
    table.to_csv(path(settings, name, "scenarios.csv"))
    say(settings, name, f"scenarios, mean yield in t/ha:\n{table.round(2)}")


# --- OPTIONAL: prior sensitivity -------------------------------------------------


def sensitivity(name: str, result: dict, settings) -> None:
    """The change per degree Celsius under each prior scale."""
    table = pd.DataFrame(
        result["change_per_c"], index=result["prior_scale"], columns=["q2.5", "q50", "q97.5"]
    )
    table.index.name = "prior_scale"
    table.to_csv(path(settings, name, "prior_sensitivity.csv"))
    say(settings, name, f"change per degree Celsius by prior scale:\n{table.round(3)}")


# --- OPTIONAL: external result ---------------------------------------------------


def external(name: str, table: pd.DataFrame, settings) -> None:
    """The quantiles of the change per degree Celsius beside the external result's."""
    table.to_csv(path(settings, name, "external_comparison.csv"))
    say(settings, name, f"change per degree Celsius, ProbPipe and external:\n{table.round(3)}")


# --- OPTIONAL: provenance --------------------------------------------------------


def routes(name: str, result: dict, settings) -> None:
    """The route of each result, written beside the provenance of the run."""
    write_json(path(settings, name, "routes.json"), result)


# --- OPTIONAL: compare models ----------------------------------------------------


def compare_models(results: dict, settings) -> None:
    """The p-values of the predictive check and the answers of each fitted model."""
    rows = {}
    for name, result in results.items():  # one row per fitted model
        if result:
            checks = {k: float(v) for k, v in result["check"].items() if k.endswith("p_value")}
            rows[name] = checks | {
                "p_low_yield": float(np.asarray(result["answers"]["p_low_yield"]))
            }
    table = pd.DataFrame(rows).T
    table.to_csv(settings.out_dir / "model_comparison.csv")
    print(f"the models side by side:\n{table.round(3)}")


# --- OPTIONAL: held-out subset ---------------------------------------------------


def heldout(scores: dict, settings) -> None:
    """The mean CRPS of each model on the held-out subset."""
    crps = {name: float(score["crps"].mean()) for name, score in scores.items()}
    write_json(settings.out_dir / "heldout_scores.json", crps)
    for name, value in crps.items():
        say(settings, name, f"mean CRPS on the held-out year: {value:.3f} t/ha (lower is better)")


# --- OPTIONAL: calibration -------------------------------------------------------


def calibration(tables: dict, settings) -> None:
    """The reliability table and the Brier score of each model."""
    columns = ["from", "to", "fields", "expected", "range_low", "range_high", "observed"]
    for name, result in tables.items():
        table = pd.DataFrame(result["rows"], columns=columns)
        table.to_csv(path(settings, name, "calibration.csv"), index=False)
        say(settings, name, f"reliability of P(yield < threshold):\n{table.round(2)}")
        say(settings, name, f"Brier score: {result['brier']:.3f}")
    # How to read it: observed counts inside their ranges are consistent with calibration,
    # which a few held-out fields cannot establish.


# --- OPTIONAL: compare algorithms ------------------------------------------------


def algorithms(result: dict, settings) -> None:
    """The method that condition_on selects, and each method's means and largest R-hat."""
    if not result:
        return
    table = pd.DataFrame(result["table"]).T
    table.columns = [*(f"mean {i}" for i in range(table.shape[1] - 1)), "largest R-hat"]
    table.to_csv(settings.out_dir / "algorithms.csv")
    print(f"condition_on selects {result['selected']}; each method on the baseline:")
    print(table.round(3))


# --- OPTIONAL: several datasets --------------------------------------------------


def by_year(result: dict, settings) -> None:
    """The change per degree Celsius in each year, labeled not validated."""
    if not result:
        return
    table = pd.DataFrame(
        result["change_per_c"], index=result["years"], columns=["q2.5", "q50", "q97.5"]
    )
    table.to_csv(settings.out_dir / "PROVISIONAL_by_year.csv")
    print(f"PROVISIONAL: not validated: change per degree Celsius by year:\n{table.round(3)}")


# --- Formatting and files --------------------------------------------------------


def interval(draws: np.ndarray, units: str, digits: int = 2) -> str:
    """Draws as 'median [2.5%, 97.5%] units', by rule 4 of the assistant's Messages."""
    low, median, high = np.percentile(np.asarray(draws), [2.5, 50, 97.5])
    return f"{median:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}] {units}"


def label(settings, name: str) -> str:
    """The label of the model's outputs: its fit's label, or else the stage's."""
    return settings.fit.get(name, settings.label)


def path(settings, name: str, file: str) -> Path:
    """The path of the model's output `file`, prefixed "PROVISIONAL_" for a provisional fit."""
    prefix = "PROVISIONAL_" if label(settings, name).startswith("PROVISIONAL") else ""
    return settings.out_dir / f"{prefix}{name}_{file}"


def say(settings, name: str, *parts: str) -> None:
    prefix = label(settings, name)
    print(*([f"{prefix}:"] if prefix else []), f"{name}:", *parts)


def titled(settings, name: str, title: str) -> str:
    return ": ".join(filter(None, [label(settings, name), name, title]))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=float))


def save_figure(fig, data: pd.DataFrame, stem: Path) -> None:
    """Save the figure as `stem`.png and the numbers it plots as `stem`.csv."""
    fig.savefig(stem.with_suffix(".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(stem.with_suffix(".csv"), index=False)
