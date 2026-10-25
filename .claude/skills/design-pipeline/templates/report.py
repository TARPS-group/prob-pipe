"""The printing, tables, and figures of the pipeline, and the files that hold them.

This module is part of the design-pipeline assistant's skeleton, beside
`pipeline_skeleton.py`. The pipeline's steps return values, and each function
here shows and saves the values of one step, so a change to the presentation
leaves the model, data, and inference code unchanged. The hash of the
validation leaves this file out for that reason, and a change here keeps a
passed validation. Replace the example's functions with one per step of the
design, and keep these rules of phases/4-proposal.md "Write the code":

1. Each function prefixes the stage's label, such as "smoke run, not for
   inference", to what it prints and to each figure's title.
2. Each interval has the form "median [2.5%, 97.5%] units" of `interval()`.
3. Each figure is saved with a CSV file of the numbers it plots, and each
   answer with the draws behind it, so every number in the design document is
   read from a file in the output folder.
4. `start()` records the provenance and the Python environment in each output
   folder, and `validation()` writes the file that the gate of the fit stage
   reads.
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


def start(settings) -> None:
    """Print the stage, and record the provenance and the Python environment."""
    print(
        f"stage {settings.stage}, design version {settings.record['design_version']}: results in {settings.out_dir}"
    )
    write_json(settings.out_dir / "provenance.json", settings.record)
    packages = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
    )
    environment = [f"python {platform.python_version()}", *packages]
    (settings.out_dir / "environment.txt").write_text("\n".join(environment) + "\n")


def prior_predictive(y_t_per_ha: np.ndarray, settings) -> None:
    """The mean yield of each prior predictive dataset, as a histogram and a summary."""
    mean_yield = y_t_per_ha.mean(axis=1)
    say(settings, "prior predictive mean yield:", interval(mean_yield, "t/ha"))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(mean_yield, bins=30)
    ax.set(
        xlabel="mean yield of a simulated dataset (t/ha)",
        ylabel="datasets",
        title=titled(settings, "Prior predictive"),
    )
    # How to read it: the simulated means should cover plausible yields and rule out impossible ones.
    save_figure(
        fig,
        pd.DataFrame({"mean_yield_t_per_ha": mean_yield}),
        settings.out_dir / "prior_predictive",
    )


def validation(result: dict, settings) -> None:
    """The validation's file, which the gate reads, and its verdict in one line of plain words."""
    write_json(settings.validation_file, {**settings.record, **result})
    say(settings, f"Validation {result['grade']}: {result['sentence']}")


def diagnostics(result: dict, settings) -> None:
    """The R-hat, the ESS, and the divergences of the fit."""
    write_json(settings.out_dir / "diagnostics.json", {k: v.tolist() for k, v in result.items()})
    say(
        settings,
        f"largest R-hat {result['max_r_hat'].max():.3f}, smallest bulk ESS {result['min_ess_bulk'].min():.0f},",
        f"divergences {int(result['divergences'].sum())}",
    )


def model_check(result: dict, settings) -> None:
    """The posterior predictive p-value of each test statistic."""
    p_values = {
        k.removesuffix("/p_value"): float(v) for k, v in result.items() if k.endswith("/p_value")
    }
    write_json(settings.out_dir / "predictive_check.json", p_values)
    for statistic, p_value in p_values.items():
        say(settings, f"posterior predictive p-value of {statistic}: {p_value:.2f}")
    # How to read it: a p-value near 0 or 1 means the model rarely reproduces that feature of the data.


def answers(result: dict, settings) -> None:
    """Each estimand as 'median [2.5%, 97.5%] units', and the draws behind it."""
    change = result["yield_change_t_per_ha_per_c"]
    write_json(
        settings.out_dir / "answers.json",
        {"yield_change_per_c": interval(change, "t/ha per °C"), **settings.record},
    )
    pd.DataFrame({"yield_change_t_per_ha_per_c": change}).to_csv(
        settings.out_dir / "answer_draws.csv", index=False
    )
    say(settings, "change in yield per degree Celsius:", interval(change, "t/ha per °C"))


def interval(draws: np.ndarray, units: str, digits: int = 2) -> str:
    """Draws as 'median [2.5%, 97.5%] units', by rule 4 of the assistant's Messages."""
    low, median, high = np.percentile(np.asarray(draws), [2.5, 50, 97.5])
    return f"{median:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}] {units}"


def say(settings, *parts: str) -> None:
    print(*([f"{settings.label}:"] if settings.label else []), *parts)


def titled(settings, title: str) -> str:
    return f"{settings.label}: {title}" if settings.label else title


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=float))


def save_figure(fig, data: pd.DataFrame, stem: Path) -> None:
    """Save the figure as `stem`.png and the numbers it plots as `stem`.csv."""
    fig.savefig(stem.with_suffix(".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(stem.with_suffix(".csv"), index=False)
