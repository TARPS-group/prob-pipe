"""Crop yield and growing-season weather: the skeleton of a pipeline script.

This file is a template for the design-pipeline assistant. Copy it to
`<workspace>/pipeline.py`, copy `pipeline_support.py`, `report.py`, and
`sim_data.py` beside it, and replace the example model, a normal regression of
crop yield on rainfall and temperature, with the model of the design document. The comments explain the
reason for each part of the structure, and the rules they cite are those of
the assistant's files. In the user's copy, replace this docstring with the
guide to where each choice is changed, by phases/4-proposal.md
"Write the code", and drop the citations of the assistant's files.

The structure follows the rules of phases/4-proposal.md "Write the code" for
these reasons:

1. `main()` comes right after the settings and reads as the pipeline, one or
   two lines per step. A reader learns what the pipeline does from one screen,
   and each step's details are in a function whose name says what it does.
2. The stage, the gate of the fit, and the grade of the validation are in
   `pipeline_support.py`, so this file shows the data, the model with its
   priors, and the steps. `main()` resolves the stage with one call.
3. Step functions are pure. They take their inputs as arguments and return
   values, and they print, plot, and write nothing. The reads and writes of
   files happen in `main()` and in `pipeline_support.py`, and the printing,
   tables, and figures are in `report.py`. A step can therefore be rerun,
   tested, or called from a notebook cell with no side effects.
4. Simulated data come from the model itself, through `sim_data.py`. The
   smoke run and the validation then exercise the model that the real fit
   uses, at the real covariates.
5. The code stays close to the mathematics, with the design document's
   notation as variable names, and each repeated computation is batched: the
   validation draws all its datasets in one call and fits them in one sweep.
6. Every draw happens inside `workflow_run(seed=...)`, with a seed of its own
   derived from the run's seed, so a rerun reproduces each result, and the
   seed is recorded with the outputs.
7. Units are carried in variable names, such as `rain_mm`, and each unit
   conversion is one small named function, which `check_code()` tests.
8. The data are checked right after they are read, so a wrong file or a
   changed column stops the run at once with a message that names the column.
9. The smoke stage runs `check_code()`, a few asserts on the pipeline's own
   code, because a run that raises no error can still compute the wrong
   numbers.
10. `report.py` saves each figure with a CSV file of the numbers it plots, and
    each output folder holds the provenance of its results, so every number
    that the design document reports is read from a file.
"""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import report
import scipy.stats
import sim_data
from pipeline_support import fits_pass, grade_validation, resolve_stage

from probpipe import (
    DistributionBatch,
    HalfNormal,
    Normal,
    condition_on,
    log_prob,
    predictive_check,
    quantile,
    sample,
    summarize,
    workflow_run,
)
from probpipe.families import GaussianFamily, glm_likelihood

# The settings that the user edits. The stage, the reason for provisional
# results, and the date of a hold on fits to simulated data come from the
# environment, and a notebook user edits their defaults. The hash of the
# validation leaves out the lines that read the environment, so a change of
# stage keeps a passed validation. The data file is read by a path relative to
# this file, so the script runs from any directory.
STAGE = os.environ.get("STAGE", "smoke")
ALLOW_UNVALIDATED = os.environ.get("ALLOW_UNVALIDATED", "")
HOLD_SIMULATED_FITS = os.environ.get("HOLD_SIMULATED_FITS", "")
DATA_FILE = Path(__file__).resolve().parent / "example_data.csv"
DESIGN_VERSION = 1
SEED = 20261025
RESPONSE = "yield_t_per_ha"
PARAMETERS = ("beta", "dispersion")
# The budgets of each stage. The smoke budgets only show that the code runs,
# and the validate and fit budgets are the planned settings of the design
# document (RESOURCES.md stage E for the sampler, stage D for the datasets).
BUDGETS = {
    "smoke": {
        "nuts": {"num_chains": 2, "num_warmup": 100, "num_results": 100},
        "n_sims": 0,
        "n_prior_draws": 50,
    },
    "validate": {
        "nuts": {"num_chains": 4, "num_warmup": 500, "num_results": 1000},
        "n_sims": 20,
        "n_prior_draws": 1000,
    },
    "fit": {
        "nuts": {"num_chains": 4, "num_warmup": 500, "num_results": 1000},
        "n_sims": 0,
        "n_prior_draws": 0,
    },
}


def main():
    """Run the pipeline of design version 1 in the stage that `STAGE` names."""
    settings = resolve_stage(
        STAGE, BUDGETS, DESIGN_VERSION, SEED, ALLOW_UNVALIDATED, HOLD_SIMULATED_FITS
    )
    report.start(settings)
    data = prepare_data(pd.read_csv(DATA_FILE))
    likelihood, model = build_model(data["X"])
    if settings.checks_code:
        check_code(data, likelihood, model, settings)
    if settings.checks_priors:
        report.prior_predictive(prior_predictive(model, settings), settings)
    if settings.validates:
        report.validation(validate(model, settings), settings)
    if settings.fits:
        y_t_per_ha = observed_response(data, model, settings)
        posterior = fit(model, {RESPONSE: y_t_per_ha}, settings)
        report.diagnostics(diagnose(posterior), settings)
        report.model_check(check_model(likelihood, posterior, y_t_per_ha, settings), settings)
        report.answers(answer(posterior, data, settings), settings)


# --- Step 1: load and check the data ----------------------------------------
# The data are checked right after the read, so a wrong file or a changed
# column fails here with a message that names the problem, and not later as a
# sampler error. Replace the columns, types, ranges, and missing-value codes
# with those of the user's data, from the data notes of Phase 2.

MISSING_CODE = -99


def prepare_data(table: pd.DataFrame) -> dict:
    """The design matrix and the response, from the checked columns of `table`."""
    check_columns(table)
    rain_mm = inches_to_mm(table["rain_in"].to_numpy())
    temp_c = fahrenheit_to_celsius(table["temp_f"].to_numpy())
    X = np.column_stack([np.ones(len(table)), standardize(rain_mm), standardize(temp_c)])
    return {
        "X": jnp.asarray(X, dtype=jnp.float32),
        RESPONSE: jnp.asarray(table[RESPONSE].to_numpy(), dtype=jnp.float32),
        "temp_c_sd": float(temp_c.std()),
    }


def check_columns(table: pd.DataFrame) -> None:
    """Raise a ValueError that names the first column that is absent, not numeric, or out of range."""
    expected = {"rain_in": (0, 200), "temp_f": (-40, 130), RESPONSE: (0, 30)}
    absent = set(expected) - set(table.columns)
    if absent:
        raise ValueError(f"The data file lacks the columns {sorted(absent)}.")
    for column, (low, high) in expected.items():
        values = table[column]
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"The column {column} is not numeric.")
        if values.isna().any() or (values == MISSING_CODE).any():
            raise ValueError(
                f"The column {column} has missing values, which this model does not handle."
            )
        if not values.between(low, high).all():
            raise ValueError(f"The column {column} has values outside [{low}, {high}].")


# Each unit conversion is one named function, which check_code() tests.
def inches_to_mm(x_in: np.ndarray) -> np.ndarray:
    return x_in * 25.4


def fahrenheit_to_celsius(x_f: np.ndarray) -> np.ndarray:
    return (x_f - 32.0) * 5.0 / 9.0


def standardize(x: np.ndarray) -> np.ndarray:
    """`x` in standard deviations from its mean, so each slope is per standard deviation."""
    return (x - x.mean()) / x.std()


# --- Step 2: the model -------------------------------------------------------
# The model is written as its generative equations, with the design document's
# notation as names. The route by which the model enters ProbPipe is chosen by
# step 3 of phases/4-proposal.md, and the ProbPipe calls are those of the
# CAPABILITIES.md rows for stages A and B. A model written for the project
# that glm_likelihood expresses uses it, as here.


def build_model(X: jnp.ndarray):
    """The likelihood and the joint model of yield given rainfall and temperature.

    yield_i ~ Normal(X_i beta, dispersion), with X_i = [1, rain_i, temp_i] in
    standard deviations. beta_0 ~ Normal(5, 2.5) puts the mean yield between
    0 and 10 t/ha, each slope ~ Normal(0, 1) allows a change of up to about
    2 t/ha per standard deviation of the weather, and dispersion ~ HalfNormal(1).
    """
    likelihood = glm_likelihood(RESPONSE, GaussianFamily(), X=X)
    beta = Normal("beta", jnp.array([5.0, 0.0, 0.0]), jnp.array([2.5, 1.0, 1.0]))
    dispersion = HalfNormal("dispersion", 1.0)
    return likelihood, likelihood & beta & dispersion


# --- The checks of the pipeline's own code, in the smoke stage ---------------
# A handful of asserts, with no test framework: each conversion matches a
# known value, the log density matches a hand computation at one point, and a
# check of random draws holds within a statistical tolerance, never by
# equality. For a model re-implemented from the user's code, add the agreement
# check of phases/4-existing-pipeline.md here.


def check_code(data: dict, likelihood, model, settings) -> None:
    """Raise an AssertionError when a part of the pipeline's own code is wrong."""
    assert np.isclose(inches_to_mm(1.0), 25.4), "inches_to_mm is wrong"
    assert np.isclose(fahrenheit_to_celsius(212.0), 100.0), "fahrenheit_to_celsius is wrong"
    beta, dispersion = jnp.array([5.0, 0.5, -0.3]), 0.8
    y_t_per_ha = data[RESPONSE]
    by_hand = scipy.stats.norm.logpdf(y_t_per_ha, data["X"] @ beta, dispersion).sum()
    law = condition_on(likelihood, {"beta": beta, "dispersion": jnp.asarray(dispersion)})
    assert np.isclose(float(log_prob(law, y_t_per_ha)), by_hand, rtol=1e-4), (
        "the log density differs from the hand computation"
    )
    _, truth = sim_data.simulate(model, 400, PARAMETERS, RESPONSE, seed=settings.seed)
    standard_error = 2.5 / np.sqrt(400)  # of the mean of 400 draws of beta_0 ~ Normal(5, 2.5)
    assert abs(truth[:, 0].mean() - 5.0) < 4 * standard_error, "the draws of beta_0 are off"


# --- Step 3: the prior predictive check --------------------------------------
# RESOURCES.md stage C says what to look for, and report.prior_predictive()
# shows it. The step returns the simulated data, and the report summarizes it.


def prior_predictive(model, settings) -> np.ndarray:
    """Datasets simulated from the prior, of shape (draw, field), in t/ha."""
    with workflow_run(seed=settings.seed + 1):
        draws = sample(model, sample_shape=(settings.budget["n_prior_draws"],))
    return np.asarray(draws[RESPONSE].raw)


# --- Step 4: validate the computation on simulated data ----------------------
# The batched recipe of the CAPABILITIES.md row "Validate the computation on
# realistic simulated data", graded by RESOURCES.md stage D step 6. This
# skeleton validates on prior draws alone. A design with values that the user
# judges realistic adds the recovery at those values from the same recipe.


def validate(model, settings) -> dict:
    """Fit datasets simulated with known parameters, and grade the recovery."""
    datasets, truth = sim_data.simulate(
        model, settings.budget["n_sims"], PARAMETERS, RESPONSE, settings.seed + 2
    )
    posteriors = fit(model, datasets, settings)  # one sweep: one posterior per dataset
    summary = summarize(posteriors)
    z = (np.asarray(summary["mean"].to_vector()) - truth) / np.asarray(summary["sd"].to_vector())
    # The 90% interval of each coordinate in each fit, of shape (sim, 2, coordinate).
    bounds = np.asarray(quantile(posteriors, jnp.array([0.05, 0.95])).to_vector())
    coverage = ((bounds[:, 0] <= truth) & (truth <= bounds[:, 1])).mean(axis=0)
    diagnostics = diagnose(posteriors)
    fits_ok = bool(fits_pass(diagnostics, settings.budget["nuts"]["num_chains"]).all())
    grade, sentence = grade_validation(z, coverage, fits_ok, settings.budget["n_sims"])
    table = {"coverage_90": coverage, "mean_z": z.mean(axis=0), "max_abs_z": np.abs(z).max()}
    table |= diagnostics
    return {
        "grade": grade,
        "sentence": sentence,
        "table": {k: np.asarray(v).tolist() for k, v in table.items()},
    }


# --- Step 5: fit -------------------------------------------------------------
# One function fits one dataset or a batch of them, so the validation and the
# real fit use the same method and settings. The method is the one that the
# design document names, by the CAPABILITIES.md row "Condition on the data".


def fit(model, observed, settings):
    """The posterior of `model` given `observed`, one per dataset for a batch."""
    with workflow_run(seed=settings.seed + 3):
        return condition_on.using(method="blackjax_nuts", method_options=settings.budget["nuts"])(
            model, observed
        )


def observed_response(data: dict, model, settings) -> jnp.ndarray:
    """The real response after the gate passed, and otherwise one simulated dataset."""
    if settings.real_response:
        return data[RESPONSE]
    datasets, _ = sim_data.simulate(model, 1, PARAMETERS, RESPONSE, settings.seed + 4)
    return jnp.asarray(datasets[RESPONSE].raw[0])


def diagnose(posteriors) -> dict:
    """The largest R-hat, the smallest bulk ESS, and the divergences of each fit."""
    fits = list(posteriors) if isinstance(posteriors, DistributionBatch) else [posteriors]
    summary = summarize(posteriors)
    return {
        "max_r_hat": np.atleast_1d(np.asarray(summary["r_hat"].to_vector()).max(axis=-1)),
        "min_ess_bulk": np.atleast_1d(np.asarray(summary["ess_bulk"].to_vector()).min(axis=-1)),
        "divergences": np.array(
            [int(p.annotations["arviz"]["sample_stats"]["diverging"].sum()) for p in fits]
        ),
    }


# --- Step 6: check the model against the data --------------------------------
# Test statistics chosen for the problem, by RESOURCES.md stage F.


def lowest_yield(y_t_per_ha):
    return jnp.min(y_t_per_ha)


def yield_spread(y_t_per_ha):
    return jnp.std(y_t_per_ha)


def check_model(likelihood, posterior, y_t_per_ha: jnp.ndarray, settings) -> dict:
    """The posterior predictive p-value of each test statistic."""
    with workflow_run(seed=settings.seed + 5):
        return predictive_check(likelihood, posterior, [lowest_yield, yield_spread], y_t_per_ha)


# --- Step 7: answer the question ---------------------------------------------
# Each estimand of the design document, as posterior draws in the user's units.


def answer(posterior, data: dict, settings) -> dict:
    """Posterior draws of the change in yield per degree Celsius, in t/ha."""
    with workflow_run(seed=settings.seed + 6):
        beta = np.asarray(sample(posterior, sample_shape=(4000,))["beta"].raw)
    return {"yield_change_t_per_ha_per_c": per_degree_c(beta[:, 2], data["temp_c_sd"])}


def per_degree_c(slope_per_sd: np.ndarray, temp_c_sd: float) -> np.ndarray:
    """A slope per standard deviation of temperature, converted to a slope per degree Celsius."""
    return slope_per_sd / temp_c_sd


if __name__ == "__main__":
    main()
