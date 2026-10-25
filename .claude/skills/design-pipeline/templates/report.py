"""The printing, tables, and figures of the pipeline, and the files that hold them.

This module is part of the design-pipeline assistant's skeleton, beside
`pipeline_skeleton.py`. Each `show_*` function prints the result of one step
of one model and saves it under the output folder. It computes nothing that a
later step uses, so a change to the presentation leaves the model, data, and
inference code unchanged. The hash of the validation leaves this file out for
that reason, and a change here keeps a passed validation. Replace the
example's functions with one per step of the design, and keep these rules of
phases/4-proposal.md "Write the code":

1. Each function takes the model's name and prefixes the model's label, such
   as "smoke run, not for inference" or "PROVISIONAL: not validated", and its
   name to what it prints and to each figure, which `save_figure()` stamps
   with them. `output_path()` names each file after the model, prefixed
   "PROVISIONAL_" for a provisional result. A table or file of several models
   has a status column with each model's label, and it is prefixed
   "PROVISIONAL_" when any of them is provisional. A fit that changes the
   validated model's prior, method, or data carries the label of
   `refit_label()` of `pipeline_support.py` in the status column of its row.
2. Each interval has the form "median [2.5%, 97.5%] units" of `interval()`,
   at the `QUANTILES` of `pipeline_support.py`.
3. Each figure is saved with a CSV file of the numbers it plots, and each
   answer with the draws behind it, so every number in the design document is
   read from a file in the output folder.

The sections tagged `OPTIONAL:` show the results of the pipeline's blocks of
the same tag, and they are deleted with those blocks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, with no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline_support import QUANTILES, divergences_of, refit_label, write_json

from probpipe import quantile


def show_stage(settings) -> None:
    """Print the stage, the design version, and the output folder."""
    print(f"stage {settings.stage}, design version {settings.record['design_version']}:")
    print(f"results in {settings.out_dir}")


def show_prior_predictive(name: str, y_prior_t_per_ha: np.ndarray, shares: dict, settings) -> None:
    """Print the shares outside the plausible ranges, and plot the mean yield of each dataset."""
    verdict = "passes" if shares["passes"] else "fails, so tighten the priors"
    print_labeled(
        settings,
        name,
        f"prior predictive check {verdict}: {shares['yields']:.1%} of the simulated yields and "
        f"{shares['datasets']:.1%} of the simulated datasets lie outside their plausible ranges",
    )
    mean_yield_per_dataset = y_prior_t_per_ha.mean(axis=1)  # computed for this figure alone
    summary = interval(np.quantile(mean_yield_per_dataset, QUANTILES), "t/ha")
    print_labeled(settings, name, "prior predictive mean yield:", summary)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(mean_yield_per_dataset, bins=30)
    ax.set(xlabel="mean yield of a simulated dataset (t/ha)", ylabel="datasets")
    # How to read it: the simulated means should cover plausible yields and rule out impossible ones.
    data = pd.DataFrame({"mean_yield_t_per_ha": mean_yield_per_dataset})
    save_figure(fig, data, settings, name, "prior_predictive")


def show_validation(name: str, validation: dict, settings) -> None:
    """Print the model's validation verdict in one line of plain words."""
    print_labeled(settings, name, f"Validation {validation['grade']}: {validation['sentence']}")


def show_diagnostics(name: str, posterior, settings) -> None:
    """Print and save the diagnostics of the fit and its divergences, by RESOURCES.md stage E."""
    table = str(posterior.diagnostics.summary_table())
    divergences = divergences_of(posterior)
    output_path(settings, name, "diagnostics.txt").write_text(
        f"{table}\ndivergences: {divergences}\n"
    )
    print_labeled(settings, name, f"diagnostics, divergences {divergences}:\n{table}")


def show_predictive_check(name: str, p_values: dict, settings) -> None:
    """Print and save the posterior predictive p-value of each test statistic."""
    write_json(output_path(settings, name, "predictive_check.json"), p_values)
    for statistic, p_value in p_values.items():
        print_labeled(settings, name, f"posterior predictive p-value of {statistic}: {p_value:.2f}")
    # How to read it: a p-value near 0 or 1 means the model rarely reproduces that feature of the data.


def show_answers(name: str, answers: dict, settings) -> None:
    """Print each estimand as 'median [2.5%, 97.5%] units', and save it with the draws behind it."""
    change_per_c = interval(quantile(answers["change_per_c"], QUANTILES), "t/ha per °C")
    p_low = float(np.asarray(answers["p_low_yield_dry_hot"]))
    summary = {"change_per_c": change_per_c, "p_low_yield_dry_hot": p_low}
    write_json(output_path(settings, name, "answers.json"), summary | settings.record)
    draws = np.asarray(answers["change_per_c"].atoms.raw)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(draws, bins=30)
    ax.set(xlabel="change in yield per degree Celsius (t/ha per °C)", ylabel="posterior draws")
    data = pd.DataFrame({"yield_change_t_per_ha_per_c": draws})
    save_figure(fig, data, settings, name, "answer_draws")
    print_labeled(settings, name, "change in yield per degree Celsius:", change_per_c)
    print_labeled(settings, name, f"chance of a low mean yield in a dry, hot year: {p_low:.2f}")


# --- OPTIONAL: plug-in forecast --------------------------------------------------


def show_plug_in(name: str, plug_in_t_per_ha: float, answers: dict, settings) -> None:
    """Print the plug-in mean yield of a dry, hot year beside the propagated answer."""
    propagated = interval(quantile(answers["yield_dry_hot"], QUANTILES), "t/ha")
    print_labeled(
        settings, name, f"dry, hot year at the posterior mean: {plug_in_t_per_ha:.2f} t/ha"
    )
    print_labeled(settings, name, "dry, hot year, propagated:", propagated)


# --- OPTIONAL: scenarios ---------------------------------------------------------


def show_scenarios(name: str, table: pd.DataFrame, settings) -> None:
    """Print and save the mean yield and the chance of a low mean yield in each scenario."""
    print_and_save_table(settings, name, table, "scenarios.csv", "scenarios, mean yield in t/ha")


# --- OPTIONAL: prior sensitivity -------------------------------------------------


def show_prior_sensitivity(name: str, table: pd.DataFrame, settings) -> None:
    """Print and save the change per degree Celsius under each prior scale, with each fit's status."""
    differences = [f"priors {scale:g}× wider" if scale != 1 else "" for scale in table.index]
    table = with_status(settings, name, table, "Sensitivity", differences)
    heading = "change per degree Celsius by prior scale"
    print_and_save_table(settings, name, table, "prior_sensitivity.csv", heading)


# --- OPTIONAL: external result ---------------------------------------------------


def show_external_comparison(name: str, table: pd.DataFrame, settings) -> None:
    """Print and save the quantiles of the change per degree Celsius beside the external ones."""
    heading = "change per degree Celsius, ProbPipe and external"
    print_and_save_table(settings, name, table, "external_comparison.csv", heading)


# --- OPTIONAL: provenance --------------------------------------------------------


def show_routes(name: str, routes: dict, settings) -> None:
    """Print and save the route of each result, beside the provenance of the run."""
    write_json(output_path(settings, name, "routes.json"), routes)
    for result, route in routes.items():
        line = f"{result}: route {route['route']}, method {route['method']}, exact {route['exact']}"
        print_labeled(settings, name, line)


# --- OPTIONAL: compare algorithms ------------------------------------------------


def show_algorithms(name: str, comparison: dict, settings) -> None:
    """Print and save the method that condition_on selects, and each method's results."""
    print_labeled(settings, name, f"condition_on selects {comparison['selected_method']}")
    method = settings.methods[name]
    table = comparison["table"]
    differences = [f"{m} in place of {method}" if m != method else "" for m in table.index]
    table = with_status(settings, name, table, "Comparison", differences)
    heading = "each method's posterior means, largest R-hat, and divergences"
    print_and_save_table(settings, name, table, "algorithms.csv", heading)


# --- OPTIONAL: several datasets --------------------------------------------------


def show_by_year(name: str, table: pd.DataFrame, settings) -> None:
    """Print and save the change per degree Celsius in each year, with each fit's status."""
    table = with_status(
        settings, name, table, "Per-year", [f"{year} alone" for year in table.index]
    )
    heading = "change per degree Celsius by year"
    print_and_save_table(settings, name, table, "by_year.csv", heading)


# --- OPTIONAL: compare models ----------------------------------------------------


def show_model_comparison(fits: dict, settings) -> None:
    """Print and save the p-values of the predictive check and the answers of each fitted model."""
    rows = {}
    for name, result in fits.items():
        p_low = float(np.asarray(result["answers"]["p_low_yield_dry_hot"]))
        rows[name] = result["p_values"] | {"p_low_yield_dry_hot": p_low}
    table = pd.DataFrame(rows).T
    table["status"] = [status(output_label(settings, name)) for name in table.index]
    label = combined_label(settings, fits)
    table.to_csv(settings.out_dir / f"{file_prefix(label)}model_comparison.csv")
    print_labeled(settings, None, f"the models side by side:\n{table.round(3)}", label=label)


# --- OPTIONAL: held-out subset ---------------------------------------------------


def show_heldout_scores(scores: dict, settings) -> None:
    """Print and save the mean CRPS of each model on the held-out subset."""
    mean_crps = {name: score["mean_crps"] for name, score in scores.items()}
    saved = {
        name: {"mean_crps": value, "status": status(output_label(settings, name))}
        for name, value in mean_crps.items()
    }
    label = combined_label(settings, scores)
    write_json(settings.out_dir / f"{file_prefix(label)}heldout_scores.json", saved)
    for name, value in mean_crps.items():
        print_labeled(
            settings, name, f"mean CRPS on the held-out year: {value:.3f} t/ha (lower is better)"
        )


# --- OPTIONAL: calibration -------------------------------------------------------


def show_reliability(tables: dict, settings) -> None:
    """Print and save the reliability table and the Brier score of each model."""
    for name, result in tables.items():
        heading = "reliability of P(yield < threshold)"
        print_and_save_table(settings, name, result["table"], "calibration.csv", heading, 2)
        print_labeled(settings, name, f"Brier score: {result['brier']:.3f}")
    # How to read it: observed counts inside their ranges are consistent with calibration,
    # which a few held-out fields cannot establish.


# --- Formatting and files --------------------------------------------------------


def interval(quantiles, units: str, digits: int = 2) -> str:
    """The quantiles at QUANTILES as 'median [2.5%, 97.5%] units', by rule 4 of Messages."""
    low, median, high = np.asarray(quantiles)
    return f"{median:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}] {units}"


def output_label(settings, name: str | None) -> str:
    """The label of a model's outputs: its fit's label, or else the stage's."""
    return settings.models_to_fit.get(name, settings.stage_label)


def status(label: str) -> str:
    """The status that a saved table records for outputs with `label`: the label, or "validated"."""
    return label or "validated"


def combined_label(settings, names) -> str:
    """The label of an output of several models: a provisional label of any of them, else the stage's."""
    labels = [output_label(settings, name) for name in names]
    return next((label for label in labels if file_prefix(label)), settings.stage_label)


def file_prefix(label: str) -> str:
    return "PROVISIONAL_" if label.startswith("PROVISIONAL") else ""


def with_status(settings, name: str, table: pd.DataFrame, kind: str, differences) -> pd.DataFrame:
    """`table` with the status of each row's fit, of `kind` and its difference from the model."""
    labels = [refit_label(settings, name, kind, difference) for difference in differences]
    return table.assign(status=[status(label) for label in labels])


def output_path(settings, name: str, file: str) -> Path:
    """The path of the model's output `file`, prefixed "PROVISIONAL_" for a provisional fit."""
    return settings.out_dir / f"{file_prefix(output_label(settings, name))}{name}_{file}"


def print_labeled(settings, name: str | None, *parts: str, label: str | None = None) -> None:
    """Print `parts` after the label of the model's outputs, or `label`, and its name, if any."""
    label = output_label(settings, name) if label is None else label
    print(*([f"{label}:"] if label else []), *([f"{name}:"] if name else []), *parts)


def print_and_save_table(
    settings, name: str, table: pd.DataFrame, file: str, heading: str, digits: int = 3
) -> None:
    """Print `table` under `heading` after the model's label, and save it with a status column.

    A table of fits that differ in status, which `with_status()` made, prints its status column.
    """
    print_labeled(settings, name, f"{heading}:\n{table.round(digits).to_string()}")
    if "status" not in table:
        table = table.assign(status=status(output_label(settings, name)))
    table.to_csv(output_path(settings, name, file))


def save_figure(fig, data: pd.DataFrame, settings, name: str, stem: str) -> None:
    """Save the figure, titled with the model's label and name, and the numbers it plots.

    The label is on the image itself, so a figure shared without its file name
    still says that it is a smoke run or a provisional result.
    """
    fig.suptitle(": ".join(filter(None, [output_label(settings, name), name])))
    path = output_path(settings, name, stem)
    fig.savefig(path.with_suffix(".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(path.with_suffix(".csv"), index=False)
