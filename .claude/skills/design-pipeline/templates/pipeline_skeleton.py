"""Crop yield and growing-season weather: a worked example of a pipeline script.

This file is a template for the design-pipeline assistant. It is a worked
example of the common tasks of a pipeline, on yields of ten fields over four
years. Copy it to `<workspace>/pipeline.py` with `pipeline_support.py`,
`report.py`, and `sim_data.py` beside it, and pare it down by
phases/4-proposal.md "Write the code": keep each block tagged `OPTIONAL:` that
the design document calls for, delete the others, and replace the example's
data, models, and estimands with those of the design. In the user's copy,
replace this docstring with the guide to where each choice is changed, and
drop the citations of the assistant's files.

The core of the example, which every pipeline keeps, is these steps:

1. the data step, a ProbPipe `@function` from the raw table to the model's
   inputs, so the provenance of every result names the file it came from;
2. the models, each a prior and a likelihood written with
   `conditional_distribution` in the notation of the design document: a
   baseline with normal errors, and an elaboration that changes one
   component, to Student-t errors, so the steps downstream stay the same;
3. the checks of the code, the prior predictive check, and the validation on
   simulated data, run in the stages that `pipeline_support.py` resolves;
4. the fit with its diagnostics, a predictive check with named test
   statistics, and the answers as functions lifted to the posterior, with an
   `expectation` for the probability of an event;
5. one `workflow_run(seed=SEED)` around every random step, so a rerun
   reproduces each result.

The structure follows the rules of phases/4-proposal.md "Write the code".
`main()` reads as the pipeline, and `check_before_fit()` and `run_model()` as
the steps of one model, one or two lines per step. Each step function returns
plain values, such as arrays, laws, and tables. `pipeline_support.py` writes
the records that later stages read, and `report.py` prints and saves each
result. The code stays close to the mathematics, units are carried in names,
such as `rain_mm`, and each repeated computation is batched, such as the
validation's datasets, the years, and the scenarios, which each run as one
call.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import report
import scipy.stats
import sim_data
from pipeline_support import (
    COVERAGE_LEVEL,
    QUANTILE_COLUMNS,
    QUANTILES,
    coordinate_names,
    divergences_of,
    fits_to_rerun,
    grade_validation,
    meets_fit_criteria,
    record_run,
    record_validation,
    resolve_stage,
)

import probpipe  # OPTIONAL: plug-in forecast
from probpipe import (
    Bernoulli,
    DistributionBatch,
    Gamma,
    HalfNormal,
    Normal,
    NumericArraySpec,
    RecordBatch,
    StudentT,
    condition_on,
    conditional_distribution,
    expectation,
    function,
    log_prob,
    predictive_check,
    quantile,
    sample,
    summarize,
    workflow_run,
)
from probpipe.diagnostics import add_mcmc_diagnostics

# --- Settings -----------------------------------------------------------------
# The settings that the user edits. resolve_stage() reads the stage, the reason
# for provisional results, and the date of a hold on fits to simulated data
# from the environment variables STAGE, ALLOW_UNVALIDATED, and
# HOLD_SIMULATED_FITS. Each file is read by a path relative to this file, so
# the script runs from any directory.
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "example_data.csv"
DESIGN_VERSION = 1
SEED = 20261025  # the seed of the run, from which ProbPipe derives the key of each random step
QUICK_START = True  # the mode, which decides whether the validation grades coverage
RESPONSE = "yield_t_per_ha"
# The models by name, each with the inference method that fits it. The baseline
# is first, and the elaboration changes one component of it. Changing a method
# here changes it everywhere, the validation included.
MODELS = {"normal": "blackjax_nuts", "student_t": "blackjax_nuts"}
BASELINE = next(iter(MODELS))
# The event of the answer: a mean yield below LOW_YIELD_T_PER_HA in a dry, hot
# year. The threshold is an arbitrary example, which the user sets.
LOW_YIELD_T_PER_HA = 4.0
DRY_HOT = {"rain_in": 15.0, "temp_f": 76.0}
# The plausible range of one field's yield, and of the spread of the yields of
# one dataset, which the user names. The prior predictive check counts the
# simulated values outside them, by RESOURCES.md stage B, and the spread catches
# nearly constant datasets as well as wild ones.
PLAUSIBLE_YIELD_T_PER_HA = (0.0, 20.0)
PLAUSIBLE_SPREAD_T_PER_HA = (0.2, 5.0)
HOLDOUT_YEAR = 2024  # OPTIONAL: held-out subset
EXTERNAL_FILE = HERE / "external_quantiles.csv"  # OPTIONAL: external result
COMPARED_METHODS = ("blackjax_nuts", "tfp_nuts", "blackjax_rwmh")  # OPTIONAL: compare algorithms
# The budgets of each stage. The smoke budgets only show that the code runs,
# and the validate and fit budgets are the planned settings of the design
# document (RESOURCES.md stage E for the sampler, stage D for the datasets).
# "options" holds the settings of each method, and "remedy" the settings with
# which a few fits that miss the criteria of stage E are rerun. The fit to the
# real data uses the remedy when the validation needed it. "n_sims" counts the
# datasets of the validation, "n_prior_draws" the prior draws of the prior
# predictive check and of the contraction, and "n_draws" the draws of each
# answer and of each held-out prediction.
SMOKE = {"num_chains": 2, "num_warmup": 100, "num_results": 100}
FULL = {"num_chains": 4, "num_warmup": 500, "num_results": 1000}
SLOW = {"num_chains": 4, "num_warmup": 2000, "num_results": 4000}  # for random-walk Metropolis
REMEDY = {"target_accept_prob": 0.95}  # the second remedy of RESOURCES.md stage E
BUDGETS = {
    "smoke": {
        "options": {"blackjax_nuts": SMOKE, "tfp_nuts": SMOKE, "blackjax_rwmh": SMOKE},
        "remedy": {"blackjax_nuts": SMOKE | REMEDY},
        "n_sims": 0,
        "n_prior_draws": 50,
        "n_draws": 200,
    },
    "validate": {
        "options": {"blackjax_nuts": FULL},
        "remedy": {"blackjax_nuts": FULL | REMEDY},
        "n_sims": 20,
        "n_prior_draws": 1000,
        "n_draws": 0,
    },
    "fit": {
        "options": {"blackjax_nuts": FULL, "tfp_nuts": FULL, "blackjax_rwmh": SLOW},
        "remedy": {"blackjax_nuts": FULL | REMEDY},
        "n_sims": 0,
        "n_prior_draws": 0,
        "n_draws": 4000,
    },
}


def main():
    """Run the pipeline of design version DESIGN_VERSION in the stage that STAGE names."""
    settings = resolve_stage(
        BUDGETS, models=list(MODELS), design_version=DESIGN_VERSION, seed=SEED, validated=VALIDATED
    )
    record_run(settings)
    # Each report.show_* call prints the result of a step and saves it under the
    # output folder. It computes nothing that a later step uses.
    report.show_stage(settings)
    data = prepare_inputs(pd.read_csv(DATA_FILE))
    data, heldout = hold_out(data, HOLDOUT_YEAR)  # OPTIONAL: held-out subset
    models = {name: build_model(name, data) for name in MODELS}
    # ProbPipe derives the key of each random step inside this block from SEED,
    # in the order that the steps run. A rerun of a stage therefore reproduces
    # each result, and each draw and each fit of a batch has its own key.
    with workflow_run(seed=SEED):
        if settings.checks_code:
            check_code(data, models[BASELINE])
        for name, model in models.items():  # the models differ in their parameters
            check_before_fit(name, model, settings)
        if not settings.models_to_fit:  # the validate stage and a hold fit no model
            return
        fits = {}  # the results of each fitted model, which the comparisons across models read
        for name in settings.models_to_fit:  # the models that the stage and the gate let fit
            fits[name] = run_model(name, models[name], data, settings)
        report.show_model_comparison(fits, settings)  # OPTIONAL: compare models
        scores = score_heldout(fits, heldout, settings)  # OPTIONAL: held-out subset
        report.show_heldout_scores(scores, settings)  # OPTIONAL: held-out subset
        reliability = reliability_tables(scores, settings)  # OPTIONAL: calibration
        report.show_reliability(reliability, settings)  # OPTIONAL: calibration


def check_before_fit(name: str, model: dict, settings) -> None:
    """The prior predictive check and the validation of one model, in the stages that run them."""
    if settings.checks_priors:
        y_prior_t_per_ha = prior_predictive_yields(model["joint"], settings)
        prior_check = prior_predictive_shares(y_prior_t_per_ha)
        report.show_prior_predictive(name, y_prior_t_per_ha, prior_check, settings)
    if settings.validates:
        validation = validate(model["joint"], MODELS[name], settings)
        record_validation(name, validation, settings)  # the grade that the gate of the fit reads
        report.show_validation(name, validation, settings)


def run_model(name: str, model: dict, data: dict, settings) -> dict:
    """Fit one model to the response of the stage, and run the steps after the fit."""
    likelihood, joint = model["likelihood"], model["joint"]
    y_t_per_ha = response_for_stage(data, joint, settings)
    remedy = name in settings.models_with_remedy  # the validation needed it, so the fit uses it
    posterior = fit(joint, {RESPONSE: y_t_per_ha}, settings, MODELS[name], remedy)
    report.show_diagnostics(name, posterior, settings)
    p_values = posterior_predictive_p_values(likelihood, posterior, y_t_per_ha)
    report.show_predictive_check(name, p_values, settings)
    answers = estimands(posterior, data, settings)
    report.show_answers(name, answers, settings)
    plug_in_t_per_ha = yield_at_posterior_mean(posterior, data)  # OPTIONAL: plug-in forecast
    report.show_plug_in(name, plug_in_t_per_ha, answers, settings)  # OPTIONAL: plug-in forecast
    scenarios = answers_by_scenario(posterior, data, settings)  # OPTIONAL: scenarios
    report.show_scenarios(name, scenarios, settings)  # OPTIONAL: scenarios
    sensitivity = prior_sensitivity(name, data, y_t_per_ha, settings)  # OPTIONAL: prior sensitivity
    report.show_prior_sensitivity(name, sensitivity, settings)  # OPTIONAL: prior sensitivity
    external = compare_external(answers, pd.read_csv(EXTERNAL_FILE))  # OPTIONAL: external result
    report.show_external_comparison(name, external, settings)  # OPTIONAL: external result
    routes = computation_routes(posterior, answers)  # OPTIONAL: provenance
    report.show_routes(name, routes, settings)  # OPTIONAL: provenance
    methods = compare_algorithms(joint, y_t_per_ha, settings)  # OPTIONAL: compare algorithms
    report.show_algorithms(name, methods, settings)  # OPTIONAL: compare algorithms
    by_year = fit_by_year(name, data, y_t_per_ha, settings)  # OPTIONAL: several datasets
    report.show_by_year(name, by_year, settings)  # OPTIONAL: several datasets
    return {"posterior": posterior, "p_values": p_values, "answers": answers}


# --- Step 1: load and check the data ----------------------------------------
# The data step is a ProbPipe function, so the provenance of each result names
# the step and the file it came from. The data are checked right after the read,
# so a wrong file or a changed column fails here, before any fit, with a
# message that names the column. Replace the columns, types, ranges, and
# missing-value codes with those of the user's data, from the data notes.

MISSING_CODE = -99


@function
def prepare_inputs(table: pd.DataFrame) -> dict:
    """The model's inputs in standard deviations, the response, and the scales of the weather."""
    check_columns(table)
    rain_mm = inches_to_mm(table["rain_in"].to_numpy())
    temp_c = fahrenheit_to_celsius(table["temp_f"].to_numpy())
    return {
        "rain": standardize(rain_mm),
        "temp": standardize(temp_c),
        "year": table["year"].to_numpy(),
        RESPONSE: table[RESPONSE].to_numpy(),
        "rain_mean_mm": rain_mm.mean(),
        "rain_sd_mm": rain_mm.std(),
        "temp_mean_c": temp_c.mean(),
        "temp_sd_c": temp_c.std(),
    }


def check_columns(table: pd.DataFrame) -> None:
    """Raise a ValueError that names the first column that is absent, not numeric, or out of range."""
    expected = {"year": (1900, 2100), "rain_in": (0, 200), "temp_f": (-40, 130), RESPONSE: (0, 30)}
    absent = set(expected) - set(table.columns)
    if absent:
        raise ValueError(f"The data file lacks the columns {sorted(absent)}.")
    for column, (low, high) in expected.items():  # one message per column, which names it
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
def inches_to_mm(length_in):
    return length_in * 25.4


def fahrenheit_to_celsius(temp_f):
    return (temp_f - 32.0) * 5.0 / 9.0


def standardize(values: np.ndarray) -> np.ndarray:
    """`values` in standard deviations from their mean, so each slope is per standard deviation."""
    return (values - values.mean()) / values.std()


def design_matrix(data) -> jnp.ndarray:
    """The design matrix X, whose row i is [1, rain_i, temp_i] in standard deviations."""
    rain, temp = jnp.asarray(data["rain"]), jnp.asarray(data["temp"])
    return jnp.stack([jnp.ones_like(rain), rain, temp], axis=-1)


def design_rows(data, rain_in, temp_f) -> jnp.ndarray:
    """The rows of X for years with rainfall `rain_in` and temperature `temp_f`, scalars or arrays.

    The weather is standardized by the mean and standard deviation of the data,
    so the rows match the design matrix that the model was fitted to.
    """
    rain_mm = inches_to_mm(jnp.asarray(rain_in))
    temp_c = fahrenheit_to_celsius(jnp.asarray(temp_f))
    rain = (rain_mm - float(data["rain_mean_mm"])) / float(data["rain_sd_mm"])
    temp = (temp_c - float(data["temp_mean_c"])) / float(data["temp_sd_c"])
    return design_matrix({"rain": rain, "temp": temp})


def select_rows(data, rows: np.ndarray) -> dict:
    """The inputs of the rows that `rows` selects, with the same scales of the weather."""
    return {k: np.asarray(v)[rows] if np.ndim(v) else np.asarray(v) for k, v in data.items()}


# --- OPTIONAL: held-out subset -------------------------------------------------
# Keep this block when the data allow a held-out subset, by the rules for
# revisions of phases/5-feedback.md: a period, a region, or a random split,
# chosen before any elaboration and scored once, after the elaboration ends.


def hold_out(data, year: int) -> tuple[dict, dict]:
    """The rows of the years other than `year`, and the rows of `year`, held out."""
    years = np.asarray(data["year"])
    return select_rows(data, years != year), select_rows(data, years == year)


# --- Step 2: the models --------------------------------------------------------
# Each model is a prior and a likelihood, written as the generative equations
# with the design document's notation as names. A likelihood that only
# glm_likelihood expresses could use it, by step 3 of phases/4-proposal.md, but
# the elaboration's Student-t errors need a likelihood written by hand, so both
# models are written this way and share every step downstream. The ProbPipe
# calls are those of the CAPABILITIES.md rows for stages A and B. main() builds
# each model once, and the steps receive it as a dict of two laws: the
# likelihood, which the predictive checks draw from, and the joint model, which
# the fits condition.


def build_model(name: str, data, prior_scale: float = 1.0) -> dict:
    """The likelihood and the joint model of yield given rainfall and temperature.

    yield_i ~ Normal(X_i beta, sigma) for "normal", and
    yield_i ~ StudentT(nu, X_i beta, sigma) for "student_t", with
    X_i = [1, rain_i, temp_i] in standard deviations. beta_0 ~ Normal(6, 1.5)
    puts the mean yield between about 3 and 9 t/ha, each slope ~ Normal(0, 1) allows
    a change of up to about 2 t/ha per standard deviation of the weather,
    sigma ~ HalfNormal(1), and nu ~ Gamma(2, 0.1) has its mean at 20, where
    the errors are close to normal. `prior_scale` widens every prior scale.
    """
    X = design_matrix(data)
    prior = Normal("beta", jnp.array([6.0, 0.0, 0.0]), prior_scale * jnp.array([1.5, 1.0, 1.0]))
    prior = prior & HalfNormal("sigma", prior_scale * 1.0)
    if name == "normal":

        @conditional_distribution(label="yield")
        def likelihood(beta: NumericArraySpec((3,)), sigma: NumericArraySpec(())) -> Normal:
            return Normal(RESPONSE, X @ beta, sigma)

    else:
        prior = prior & Gamma("nu", 2.0, 0.1 / prior_scale)

        @conditional_distribution(label="yield")
        def likelihood(
            beta: NumericArraySpec((3,)), sigma: NumericArraySpec(()), nu: NumericArraySpec(())
        ) -> StudentT:
            return StudentT(RESPONSE, nu, X @ beta, sigma)

    return {"likelihood": likelihood, "joint": likelihood & prior}


# --- OPTIONAL: bring in a model from another package (comments only) ---------
# A model written in another package enters by its route of CAPABILITIES.md
# "Bring any model", and the steps that need a separate likelihood, such as the
# predictive check, change as that row says. The baseline in PyMC, for example:
#
#     def build_pymc(y_t_per_ha=None):
#         with pymc.Model() as m:
#             beta = pymc.Normal("beta", [6.0, 0.0, 0.0], [1.5, 1.0, 1.0])
#             sigma = pymc.HalfNormal("sigma", 1.0)
#             pymc.Normal(RESPONSE, X @ beta, sigma, observed=y_t_per_ha)
#         return m
#
#     joint = PyMCModel(build_pymc)
#
# A Stan program enters as StanModel(stan_file, data=...), and existing code
# that only scores or draws the data enters as distribution(log_prob=...,
# sample=...), or as a simulator for approximate Bayesian computation.


# --- The checks of the pipeline's own code, in the smoke stage ---------------
# A handful of asserts, with no test framework: each conversion matches a
# known value, the log density matches a hand computation at one point, and a
# check of random draws holds within a statistical tolerance, never by
# equality. The expected values are written out by hand, so each check
# compares the code with an independent statement of its result. For a model
# re-implemented from the user's code, add the agreement check of
# phases/4-existing-pipeline.md here.


def check_code(data, model: dict) -> None:
    """Raise an AssertionError when a part of the pipeline's own code is wrong."""
    assert np.isclose(inches_to_mm(1.0), 25.4), "inches_to_mm is wrong"
    assert np.isclose(fahrenheit_to_celsius(212.0), 100.0), "fahrenheit_to_celsius is wrong"
    beta, sigma = jnp.array([5.0, 0.5, -0.3]), 0.8
    y_t_per_ha = np.asarray(data[RESPONSE])
    by_hand = scipy.stats.norm.logpdf(y_t_per_ha, design_matrix(data) @ beta, sigma).sum()
    law = condition_on(model["likelihood"], {"beta": beta, "sigma": jnp.asarray(sigma)})
    assert np.isclose(float(log_prob(law, y_t_per_ha)), by_hand, rtol=1e-4), (
        "the log density differs from the hand computation"
    )
    n_draws = 400
    _, theta = sim_data.simulate(model["joint"], n_draws, RESPONSE)
    standard_error = 1.5 / np.sqrt(n_draws)  # of the mean of draws of beta_0 ~ Normal(6, 1.5)
    assert abs(theta[:, 0].mean() - 6.0) < 4 * standard_error, "the draws of beta_0 are off"


# --- Step 3: the prior predictive check --------------------------------------
# RESOURCES.md stage C says what to look for, and report.show_prior_predictive()
# shows it.


def prior_predictive_yields(joint, settings) -> np.ndarray:
    """Yields simulated from the prior predictive distribution, of shape (draw, field), in t/ha."""
    draws = sample(joint, sample_shape=(settings.budget["n_prior_draws"],))
    return np.asarray(draws[RESPONSE].raw)


def prior_predictive_shares(y_prior_t_per_ha: np.ndarray) -> dict:
    """The shares of simulated yields and of simulated datasets outside their plausible ranges.

    The priors pass RESOURCES.md stage B when at most 5% of the yields and at
    most 10% of the datasets lie outside.
    """
    low, high = PLAUSIBLE_YIELD_T_PER_HA
    yields_outside = (y_prior_t_per_ha < low) | (y_prior_t_per_ha > high)
    spread_t_per_ha = y_prior_t_per_ha.std(axis=1)
    low_spread, high_spread = PLAUSIBLE_SPREAD_T_PER_HA
    datasets_outside = (spread_t_per_ha < low_spread) | (spread_t_per_ha > high_spread)
    shares = {"yields": float(yields_outside.mean()), "datasets": float(datasets_outside.mean())}
    return shares | {"passes": shares["yields"] <= 0.05 and shares["datasets"] <= 0.10}


# --- Step 4: validate the computation on simulated data ----------------------
# The batched recipe of the CAPABILITIES.md row "Validate the computation on
# realistic simulated data", graded by RESOURCES.md stage D step 6. This
# example validates on prior draws alone. A design with values that the user
# judges realistic adds the recovery at those values from the same recipe.


def validate(joint, method: str, settings) -> dict:
    """Fit datasets simulated with known parameters theta, and grade the recovery of theta.

    z = (posterior mean - theta) / posterior sd for each fit and coordinate,
    and contraction = 1 - posterior variance / prior variance, by stage D.
    """
    datasets, theta = sim_data.simulate(joint, settings.budget["n_sims"], RESPONSE)
    fits = posterior_summaries(fit(joint, datasets, settings, method))  # one sweep
    num_chains = settings.budget["options"][method]["num_chains"]
    rerun = fits_to_rerun(fits, num_chains)
    if rerun.size:  # a few fits just past the criteria are rerun with the remedy, in one sweep
        again = RecordBatch({RESPONSE: datasets[RESPONSE].raw[rerun]}, "sim")
        refits = posterior_summaries(fit(joint, again, settings, method, remedy=True))
        for key, values in refits.items():  # each summary is its own array
            fits[key][rerun] = values
    z = (fits["mean"] - theta) / fits["sd"]
    coverage = ((fits["low"] <= theta) & (theta <= fits["high"])).mean(axis=0)
    draws = sample(joint, sample_shape=(settings.budget["n_prior_draws"],))
    theta_prior = np.asarray(draws.without(RESPONSE).to_vector())  # (draw, coordinate)
    contraction = 1 - (fits["sd"] ** 2).mean(axis=0) / theta_prior.var(axis=0)
    names = coordinate_names(joint, RESPONSE)
    all_pass = bool(meets_fit_criteria(fits, num_chains).all())
    grade = grade_validation(z, coverage, contraction, names, all_pass, rerun.size, QUICK_START)
    diagnostics = {k: fits[k] for k in ("max_r_hat", "min_ess_bulk", "divergences")}
    table = {"coordinates": names, "coverage": coverage, "mean_z": z.mean(axis=0)}
    table |= {"contraction": contraction} | diagnostics
    return grade | {
        "remedy": bool(rerun.size),
        "table": {k: np.asarray(v).tolist() for k, v in table.items()},
    }


def posterior_summaries(posteriors) -> dict:
    """The mean, sd, and central interval of each coordinate of each fit, with its diagnostics.

    The interval has the probability COVERAGE_LEVEL, whose coverage the
    validation grades. The arrays are copies, so the rerun of a few fits can
    replace their rows.
    """
    summary = summarize(posteriors)
    ends = np.array([1 - COVERAGE_LEVEL, 1 + COVERAGE_LEVEL]) / 2
    interval = np.array(quantile(posteriors, ends).to_vector())  # (fit, end, coordinate)
    # summarize() reports no divergences, so each posterior's sampler statistics
    # are read on their own. A batched accessor is a ProbPipe gap. A method
    # without divergences, such as random-walk Metropolis, records none.
    stats = [p.annotations["arviz"]["sample_stats"] for p in posteriors]
    divergences = [int(np.sum(s.get("diverging", 0))) for s in stats]
    return {
        "mean": np.array(summary["mean"].to_vector()),
        "sd": np.array(summary["sd"].to_vector()),
        "low": interval[:, 0],
        "high": interval[:, 1],
        "max_r_hat": np.asarray(summary["r_hat"].to_vector()).max(axis=-1),
        "min_ess_bulk": np.asarray(summary["ess_bulk"].to_vector()).min(axis=-1),
        "divergences": np.array(divergences),
    }


# --- Step 5: fit ---------------------------------------------------------------
# One function fits one dataset or a batch of them, so the validation and the
# real fit use the same method and settings. The method of each model is set in
# MODELS, by the CAPABILITIES.md row "Condition on the data". The diagnostics
# of RESOURCES.md stage E are computed on every fit to the real data.


def fit(joint, observed, settings, method: str, remedy: bool = False):
    """The posterior of `joint` given `observed`, one per dataset for a batch."""
    options = settings.budget["remedy" if remedy else "options"][method]
    posterior = condition_on.using(method=method, method_options=options)(joint, observed)
    if not isinstance(posterior, DistributionBatch):  # a batch is diagnosed by summarize()
        add_mcmc_diagnostics(posterior)
    return posterior


def response_for_stage(data, joint, settings) -> jnp.ndarray:
    """The observed response after the gate passed, and else one dataset simulated from `joint`."""
    if settings.uses_real_response:
        return jnp.asarray(data[RESPONSE])
    datasets, _ = sim_data.simulate(joint, 1, RESPONSE)
    return jnp.asarray(datasets[RESPONSE].raw[0])


# --- What the validation depends on ----------------------------------------------
# The gate accepts a validation whose hash matches the hash of these functions
# and settings, with the grading code of pipeline_support.py. A change to any of
# them, the data file included, makes a passed validation stale, and a change
# to a step after the fit, such as an answer, a check, or a comparison, keeps
# it. Add each function that the data step, a model, the simulation, or the fit
# of a design calls.

VALIDATED = (
    # the data file, whose covariates and size the simulated datasets share, and the data step
    DATA_FILE,
    prepare_inputs,
    check_columns,
    inches_to_mm,
    fahrenheit_to_celsius,
    standardize,
    MISSING_CODE,
    # the models
    design_matrix,
    build_model,
    RESPONSE,
    # the simulation of datasets, and the fit with its methods and budgets
    sim_data.simulate,
    fit,
    MODELS,
    BUDGETS["validate"],
    {method: BUDGETS["fit"]["options"][method] for method in MODELS.values()},
    # the validation itself
    validate,
    posterior_summaries,
    QUICK_START,
)


# --- Step 6: check the model against the data --------------------------------
# Test statistics named for the problem, by RESOURCES.md stage F.


def lowest_yield(y_t_per_ha):
    return jnp.min(y_t_per_ha)


def yield_spread(y_t_per_ha):
    return jnp.std(y_t_per_ha)


TEST_STATISTICS = [lowest_yield, yield_spread]


def posterior_predictive_p_values(likelihood, posterior, y_t_per_ha: jnp.ndarray) -> dict:
    """The posterior predictive p-value of each test statistic, by the statistic's name."""
    check = predictive_check(likelihood, posterior, TEST_STATISTICS, y_t_per_ha)
    return {t.__name__: float(check[f"{t.__name__}/p_value"]) for t in TEST_STATISTICS}


# --- Step 7: answer the question ---------------------------------------------
# Each estimand is a function written for one value of the parameters, which
# ProbPipe lifts to the posterior, so the answer carries the posterior's
# uncertainty. The probability of an event is an expectation, the posterior
# mean, by rule 4 of the assistant's Messages.


@function
def per_degree_c(beta: jax.Array, temp_sd_c: jax.Array) -> jax.Array:
    """The change in yield per degree Celsius, beta_temp / sd(temp_c), in t/ha per °C.

    It also converts quantiles of beta, since quantiles commute with the
    division by the positive `temp_sd_c`.
    """
    return beta[..., 2] / temp_sd_c


@function
def expected_yield(beta: jax.Array, X: jax.Array) -> jax.Array:
    """The mean yield X beta in t/ha of each year whose design-matrix row is a row of `X`."""
    return X @ beta


def is_low_yield(y_t_per_ha):
    """Whether each yield is below LOW_YIELD_T_PER_HA, the event of the answer."""
    return y_t_per_ha < LOW_YIELD_T_PER_HA


def estimands(posterior, data, settings) -> dict:
    """The change in yield per °C, and the chance of a low mean yield in a dry, hot year."""
    n_draws = settings.budget["n_draws"]
    change_per_c = per_degree_c.using(max_evaluations=n_draws)(posterior["beta"], data["temp_sd_c"])
    X_dry_hot = design_rows(data, **DRY_HOT)
    yield_dry_hot = expected_yield.using(max_evaluations=n_draws)(posterior["beta"], X_dry_hot)
    return {
        "change_per_c": change_per_c,
        "yield_dry_hot": yield_dry_hot,
        "p_low_yield_dry_hot": expectation(yield_dry_hot, is_low_yield),
    }


# --- OPTIONAL: plug-in forecast --------------------------------------------------
# The answer at the posterior mean beside the propagated one. A plug-in answer
# hides the uncertainty, and the difference is often the most persuasive result
# for a scientist. Keep it when the answer is a probability or a decision.


def yield_at_posterior_mean(posterior, data) -> float:
    """The mean yield in t/ha of a dry, hot year at the posterior mean of beta."""
    beta_mean = np.asarray(probpipe.mean(posterior["beta"]))
    return float(np.asarray(expected_yield(beta_mean, design_rows(data, **DRY_HOT))))


# --- OPTIONAL: scenarios ---------------------------------------------------------
# The answer under several scenarios in one call: the lifted function takes a
# matrix with one row per scenario. Keep it when the user compares conditions.

# One row per scenario, with its weather.
SCENARIOS = pd.DataFrame({"dry_hot": DRY_HOT, "wet_cool": {"rain_in": 30.0, "temp_f": 66.0}}).T


def answers_by_scenario(posterior, data, settings) -> pd.DataFrame:
    """The quantiles of the mean yield, and the chance of a low mean yield, in each scenario."""
    X = design_rows(data, SCENARIOS["rain_in"].to_numpy(), SCENARIOS["temp_f"].to_numpy())
    yields = expected_yield.using(max_evaluations=settings.budget["n_draws"])(posterior["beta"], X)
    quantiles = np.asarray(quantile(yields, QUANTILES)).T  # (scenario, quantile)
    table = pd.DataFrame(quantiles, index=SCENARIOS.index, columns=QUANTILE_COLUMNS)
    table["p_low_yield"] = np.asarray(expectation(yields, is_low_yield))
    return table


# --- OPTIONAL: prior sensitivity -------------------------------------------------
# A refit under priors three times as wide, in one sweep with the default
# priors, by RESOURCES.md stage H. Keep it when the user asks how much the
# priors matter.

PRIOR_SCALES = [1.0, 3.0]


def prior_sensitivity(name: str, data, y_t_per_ha, settings) -> pd.DataFrame:
    """The quantiles of the change in yield per °C under each prior scale of PRIOR_SCALES."""
    # One joint model per scale, since each holds its own priors.
    joints = [build_model(name, data, scale)["joint"] for scale in PRIOR_SCALES]
    observed = {RESPONSE: jnp.tile(y_t_per_ha, (len(PRIOR_SCALES), 1))}
    posteriors = fit(
        DistributionBatch(joints, "prior_scale"),
        RecordBatch(observed, "prior_scale"),
        settings,
        MODELS[name],
    )
    change_per_c = per_degree_c(quantile(posteriors, QUANTILES)["beta"], data["temp_sd_c"])
    index = pd.Index(PRIOR_SCALES, name="prior_scale")
    return pd.DataFrame(np.asarray(change_per_c), index=index, columns=QUANTILE_COLUMNS)


# --- OPTIONAL: external result ---------------------------------------------------
# The quantiles of another tool or of a saved earlier run beside ProbPipe's,
# such as a colleague's INLA or lme4 fit. Keep it when the user must reconcile
# the two.


def compare_external(answers: dict, external: pd.DataFrame) -> pd.DataFrame:
    """The quantiles of the change per degree Celsius beside those of the external result."""
    ours = np.asarray(quantile(answers["change_per_c"], QUANTILES))
    theirs = external.set_index("estimand").loc["change_per_c", QUANTILE_COLUMNS].to_numpy(float)
    return pd.DataFrame([ours, theirs], index=["probpipe", "external"], columns=QUANTILE_COLUMNS)


# --- OPTIONAL: provenance --------------------------------------------------------
# Each result records how it was computed, and whether exactly or approximately.
# Keep it when the user reports the computation, as in a methods section.


def computation_routes(posterior, answers: dict) -> dict:
    """The route and method of the posterior and of each answer, and whether each is exact."""
    results = {"posterior": posterior, **answers}
    return {
        name: {k: result.provenance.metadata.get(k) for k in ("route", "method", "exact")}
        for name, result in results.items()
    }


# --- OPTIONAL: compare algorithms ------------------------------------------------
# The model fitted by each method of COMPARED_METHODS, with the means, the
# largest R-hat, and the divergences side by side, by RESOURCES.md stage E. condition_on.check names
# the method that a call would select, before any computation. Keep it when the
# choice of method is in question.


def compare_algorithms(joint, y_t_per_ha, settings) -> dict:
    """The method that condition_on selects, and each method's means, R-hat, and divergences."""
    observed = {RESPONSE: y_t_per_ha}
    rows = {}
    for method in COMPARED_METHODS:  # each method runs its own algorithm
        posterior = fit(joint, observed, settings, method)
        summary = summarize(posterior)
        r_hat = float(np.max(summary["r_hat"].to_vector()))
        means = np.asarray(summary["mean"].to_vector(), dtype=float)
        rows[method] = [*means, r_hat, divergences_of(posterior)]
    columns = [*coordinate_names(joint, RESPONSE), "largest R-hat", "divergences"]
    return {
        "selected_method": condition_on.check(joint, observed).selected.method_name,
        "table": pd.DataFrame.from_dict(rows, orient="index", columns=columns),
    }


# --- OPTIONAL: several datasets ----------------------------------------------------
# The model fitted to each year on its own, as one batched sweep over a batch
# of models, one per year at its own weather. The validation covered the pooled
# design, so report.show_by_year() labels these fits not validated. Keep it
# when the design reports an answer per site, group, or year.


def fit_by_year(name: str, data, y_t_per_ha, settings) -> pd.DataFrame:
    """The quantiles of the change in yield per degree Celsius in each year, from one sweep."""
    years, rows_per_year = np.unique(np.asarray(data["year"]), return_counts=True)
    if np.any(rows_per_year != rows_per_year[0]):  # a batch stacks the years
        counts = dict(zip(years.tolist(), rows_per_year.tolist(), strict=True))
        raise ValueError(f"fit_by_year needs the same number of rows in each year, not {counts}.")
    rows = np.argsort(np.asarray(data["year"]), kind="stable").reshape(len(years), -1)
    # One joint model per year, since each holds the weather of its year.
    joints = [build_model(name, select_rows(data, year_rows))["joint"] for year_rows in rows]
    observed = {RESPONSE: jnp.asarray(y_t_per_ha)[rows]}  # (year, field)
    posteriors = fit(
        DistributionBatch(joints, "year"), RecordBatch(observed, "year"), settings, MODELS[name]
    )
    change_per_c = per_degree_c(quantile(posteriors, QUANTILES)["beta"], data["temp_sd_c"])
    index = pd.Index(years, name="year")
    return pd.DataFrame(np.asarray(change_per_c), index=index, columns=QUANTILE_COLUMNS)


# --- OPTIONAL: held-out subset (its score) ---------------------------------------
# Each model scored once on the held-out subset, after the elaboration ends,
# by the continuous ranked probability score of its predictive draws (lower is
# better). In the smoke stage the held-out yields are simulated.


def score_heldout(fits: dict, heldout, settings) -> dict:
    """The predictive draws of the held-out yields, the yields, and the mean CRPS of each model."""
    scores = {}
    for name, result in fits.items():  # the models differ in their parameters
        # The held-out year has its own weather, so its likelihood is built from its rows.
        model = build_model(name, heldout)
        predictive = model["likelihood"] & result["posterior"]
        draws = sample(predictive, sample_shape=(settings.budget["n_draws"],))
        y_draws = np.asarray(draws[RESPONSE].raw)  # (draw, field)
        y_t_per_ha = np.asarray(response_for_stage(heldout, model["joint"], settings))
        scores[name] = {
            "y_draws": y_draws,
            "y_t_per_ha": y_t_per_ha,
            "mean_crps": mean_crps(y_draws, y_t_per_ha),
        }
    return scores


def mean_crps(y_draws: np.ndarray, y: np.ndarray) -> float:
    """The CRPS E|Y - y| - E|Y - Y'| / 2 of each field, averaged over the fields.

    Y and Y' are independent predictive draws, here the even and the odd draws.
    """
    spread = np.abs(y_draws[::2] - y_draws[1::2]).mean(axis=0)
    return float((np.abs(y_draws - y).mean(axis=0) - spread / 2).mean())


# --- OPTIONAL: calibration ---------------------------------------------------------
# Needs the block "held-out subset". A reliability table of the probability of a low
# yield on each held-out field: the observed count of low yields in each bin of
# predicted probability beside its expected count, with the range that
# calibrated probabilities give, and the Brier score, by RESOURCES.md stage F.

PROBABILITY_BINS = np.array([0.0, 0.2, 0.5, 0.8, 1.0])


def reliability_tables(scores: dict, settings) -> dict:
    """The reliability table and the Brier score of each model's probability of a low yield."""
    n_bins, n_draws = len(PROBABILITY_BINS) - 1, settings.budget["n_draws"]
    tables = {}
    for name, score in scores.items():  # each model puts different fields in each bin
        p_low = is_low_yield(score["y_draws"]).mean(axis=0)  # (field,)
        observed_low = is_low_yield(score["y_t_per_ha"])
        bin_of_field = np.clip(np.digitize(p_low, PROBABILITY_BINS) - 1, 0, n_bins - 1)
        in_bin = (bin_of_field[:, None] == np.arange(n_bins)).astype(float)  # (field, bin)
        # Outcomes drawn as if p_low were calibrated give the range of each bin's count.
        calibrated = sample(Bernoulli("low", probs=p_low), sample_shape=(n_draws,))
        counts = np.asarray(calibrated.raw) @ in_bin  # (draw, bin)
        range_low, _, range_high = np.quantile(counts, QUANTILES, axis=0)
        table = pd.DataFrame(
            {
                "from": PROBABILITY_BINS[:-1],
                "to": PROBABILITY_BINS[1:],
                "fields": in_bin.sum(axis=0),
                "expected": p_low @ in_bin,
                "range_low": range_low,
                "range_high": range_high,
                "observed": observed_low @ in_bin,
            }
        )
        tables[name] = {"table": table, "brier": float(np.mean((p_low - observed_low) ** 2))}
    return tables


# --- OPTIONAL: large data (comments only) ----------------------------------------
# For many observations, stochastic-gradient MCMC estimates each gradient from a
# minibatch: condition_on.using(method="blackjax_sgld", method_options={...})
# with a MinibatchedDistribution, by RESOURCES.md stage E "Many observations".
# Many independent fits, such as one per site or per bootstrap replicate, run
# in parallel through an executor of executor_registry, or as Prefect tasks on
# Ray, by the CAPABILITIES.md row "Run a pipeline at scale".


if __name__ == "__main__":
    main()
