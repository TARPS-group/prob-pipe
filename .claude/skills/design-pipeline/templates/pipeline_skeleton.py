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
5. one `workflow_run(seed=...)` around each random step, with its own seed
   derived from `SEED`, so a rerun reproduces each result.

The structure follows the rules of phases/4-proposal.md "Write the code":
`main()` reads as the pipeline and `run_model()` as the steps of one model,
one or two lines per step. Step functions are pure, and the printing, tables,
figures, and files are in `report.py`. The code stays close to the
mathematics, units are carried in names, such as `rain_mm`, and each repeated
computation is batched, such as the validation's datasets, the years, and the
scenarios, which each run as one call.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import report
import scipy.stats
import sim_data
from pipeline_support import (
    coordinate_names,
    fits_pass,
    fits_to_rerun,
    grade_validation,
    resolve_stage,
)

import probpipe  # OPTIONAL: plug-in forecast
from probpipe import (
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
# The settings that the user edits. The stage, the reason for provisional
# results, and the date of a hold on fits to simulated data come from the
# environment, and a notebook user edits their defaults. The hash of the
# validation leaves out the lines that read the environment, so a change of
# stage keeps a passed validation. Each file is read by a path relative to this
# file, so the script runs from any directory.
STAGE = os.environ.get("STAGE", "smoke")
ALLOW_UNVALIDATED = os.environ.get("ALLOW_UNVALIDATED", "")
HOLD_SIMULATED_FITS = os.environ.get("HOLD_SIMULATED_FITS", "")
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "example_data.csv"
DESIGN_VERSION = 1
SEED = 20261025
QUICK_START = True  # the mode, which decides whether the validation grades coverage
RESPONSE = "yield_t_per_ha"
# The models by name, the baseline first, each with the inference method that
# fits it. Changing a method here changes it everywhere, the validation included.
MODELS = {"normal": "blackjax_nuts", "student_t": "blackjax_nuts"}
# The event of the answer: a mean yield below LOW_YIELD_T_PER_HA in a dry, hot
# year. The threshold is an arbitrary example, which the user sets.
LOW_YIELD_T_PER_HA = 4.0
DRY_HOT = {"rain_in": 15.0, "temp_f": 76.0}
HOLDOUT_YEAR = 2024  # OPTIONAL: held-out subset
EXTERNAL_FILE = HERE / "external_quantiles.csv"  # OPTIONAL: external result
COMPARED_METHODS = ("blackjax_nuts", "tfp_nuts", "blackjax_rwmh")  # OPTIONAL: compare algorithms
# The budgets of each stage. The smoke budgets only show that the code runs,
# and the validate and fit budgets are the planned settings of the design
# document (RESOURCES.md stage E for the sampler, stage D for the datasets).
# "options" holds the settings of each method, and "remedy" the settings with
# which a few fits that miss the criteria of stage E are rerun, here the
# higher target acceptance rate of its second remedy. The fit to the real data
# uses the remedy when the validation needed it.
SMOKE = {"num_chains": 2, "num_warmup": 100, "num_results": 100}
FULL = {"num_chains": 4, "num_warmup": 500, "num_results": 1000}
SLOW = {"num_chains": 4, "num_warmup": 2000, "num_results": 4000}  # for random-walk Metropolis
BUDGETS = {
    "smoke": {
        "options": {"blackjax_nuts": SMOKE, "tfp_nuts": SMOKE, "blackjax_rwmh": SMOKE},
        "remedy": {"blackjax_nuts": SMOKE | {"target_accept_prob": 0.95}},
        "n_sims": 0,
        "n_prior_draws": 50,
        "n_draws": 200,
    },
    "validate": {
        "options": {"blackjax_nuts": FULL},
        "remedy": {"blackjax_nuts": FULL | {"target_accept_prob": 0.95}},
        "n_sims": 20,
        "n_prior_draws": 1000,
        "n_draws": 0,
    },
    "fit": {
        "options": {"blackjax_nuts": FULL, "tfp_nuts": FULL, "blackjax_rwmh": SLOW},
        "remedy": {"blackjax_nuts": FULL | {"target_accept_prob": 0.95}},
        "n_sims": 0,
        "n_prior_draws": 0,
        "n_draws": 4000,
    },
}


def main():
    """Run the pipeline of design version 1 in the stage that `STAGE` names."""
    settings = resolve_stage(
        STAGE, BUDGETS, tuple(MODELS), DESIGN_VERSION, SEED, ALLOW_UNVALIDATED, HOLD_SIMULATED_FITS
    )
    report.start(settings)
    data = prepare(pd.read_csv(DATA_FILE))
    data, heldout = hold_out(data, HOLDOUT_YEAR)  # OPTIONAL: held-out subset
    if settings.checks_code:
        check_code(data, settings)
    results = {}
    for name, method in MODELS.items():  # each model in turn, by its own steps
        results[name] = run_model(name, method, data, settings)
    if not settings.fit:
        return
    report.compare_models(results, settings)  # OPTIONAL: compare models
    scores = score_heldout(results, heldout, settings)  # OPTIONAL: held-out subset
    report.heldout(scores, settings)  # OPTIONAL: held-out subset
    report.calibration(calibrate(scores, settings), settings)  # OPTIONAL: calibration
    algorithms = compare_algorithms(data, results, settings)  # OPTIONAL: compare algorithms
    report.algorithms(algorithms, settings)  # OPTIONAL: compare algorithms
    report.by_year(fit_by_year(data, results, settings), settings)  # OPTIONAL: several datasets


def run_model(name: str, method: str, data: dict, settings) -> dict:
    """The steps of one model in this stage, and its fit and answers when it was fit."""
    likelihood, model = build_model(name, data)
    if settings.checks_priors:
        report.prior_predictive(name, prior_predictive(model, settings), settings)
    if settings.validates:
        report.validation(name, validate(model, method, settings), settings)
    if name not in settings.fit:
        return {}
    y_t_per_ha = observed_response(data, model, settings)
    posterior = fit(model, {RESPONSE: y_t_per_ha}, settings, method, name in settings.remedied)
    report.diagnostics(name, posterior, settings)
    check = check_model(likelihood, posterior, y_t_per_ha, settings)
    report.model_check(name, check, settings)
    answers = answer(posterior, data, settings)
    report.answers(name, answers, settings)
    report.plug_in(name, plug_in(posterior, data), answers, settings)  # OPTIONAL: plug-in forecast
    report.scenarios(name, scenarios(posterior, data, settings), settings)  # OPTIONAL: scenarios
    sensitivity = prior_sensitivity(name, data, y_t_per_ha, settings)  # OPTIONAL: prior sensitivity
    report.sensitivity(name, sensitivity, settings)  # OPTIONAL: prior sensitivity
    external = compare_external(answers, pd.read_csv(EXTERNAL_FILE))  # OPTIONAL: external result
    report.external(name, external, settings)  # OPTIONAL: external result
    report.routes(name, routes(posterior, answers), settings)  # OPTIONAL: provenance
    return {"posterior": posterior, "y": y_t_per_ha, "check": check, "answers": answers}


# --- Step 1: load and check the data ----------------------------------------
# The data step is a ProbPipe function, so the provenance of each result names
# the step and the file it came from. The data are checked right after the read,
# so a wrong file or a changed column fails here with a message that names the
# column, and not later as a sampler error. Replace the columns, types, ranges,
# and missing-value codes with those of the user's data, from the data notes.

MISSING_CODE = -99


@function
def prepare(table: pd.DataFrame) -> dict:
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


def subset(data, rows: np.ndarray) -> dict:
    """The inputs of the rows that `rows` selects, with the same scales of the weather."""
    return {k: np.asarray(data[k])[rows] if np.ndim(data[k]) else np.asarray(data[k]) for k in data}


def weather_row(data, rain_in: float, temp_f: float) -> jnp.ndarray:
    """The row [1, rain, temp] of the design matrix for a year with this weather."""
    rain = (inches_to_mm(rain_in) - float(data["rain_mean_mm"])) / float(data["rain_sd_mm"])
    temp = (fahrenheit_to_celsius(temp_f) - float(data["temp_mean_c"])) / float(data["temp_sd_c"])
    return jnp.array([1.0, rain, temp])


# --- OPTIONAL: held-out subset -------------------------------------------------
# Keep this block when the data allow a held-out subset, by the rules for
# revisions of phases/5-feedback.md: a period, a region, or a random split,
# chosen before any elaboration and scored once, after the elaboration ends.


def hold_out(data, year: int) -> tuple[dict, dict]:
    """The rows of the years before `year`, and the rows of `year`, held out."""
    years = np.asarray(data["year"])
    return subset(data, years != year), subset(data, years == year)


# --- Step 2: the models --------------------------------------------------------
# Each model is a prior and a likelihood, written as the generative equations
# with the design document's notation as names. A likelihood that only
# glm_likelihood expresses could use it, by step 3 of phases/4-proposal.md, but
# the elaboration's Student-t errors need a likelihood written by hand, so both
# models are written this way and share every step downstream. The ProbPipe
# calls are those of the CAPABILITIES.md rows for stages A and B.


def build_model(name: str, data, prior_scale: float = 1.0):
    """The likelihood and the joint model of yield given rainfall and temperature.

    yield_i ~ Normal(X_i beta, sigma) for "normal", and
    yield_i ~ StudentT(nu, X_i beta, sigma) for "student_t", with
    X_i = [1, rain_i, temp_i] in standard deviations. beta_0 ~ Normal(5, 2.5)
    puts the mean yield between 0 and 10 t/ha, each slope ~ Normal(0, 1) allows
    a change of up to about 2 t/ha per standard deviation of the weather,
    sigma ~ HalfNormal(1), and nu ~ Gamma(2, 0.1) has its mean at 20, where
    the errors are close to normal. `prior_scale` widens every prior scale.
    """
    X = jnp.column_stack([jnp.ones_like(data["rain"]), data["rain"], data["temp"]])
    s = prior_scale
    prior = Normal("beta", jnp.array([5.0, 0.0, 0.0]), s * jnp.array([2.5, 1.0, 1.0]))
    prior = prior & HalfNormal("sigma", s * 1.0)
    if name == "normal":

        @conditional_distribution(label="yield")
        def likelihood(beta: NumericArraySpec((3,)), sigma: NumericArraySpec(())) -> Normal:
            return Normal(RESPONSE, X @ beta, sigma)

    else:
        prior = prior & Gamma("nu", 2.0, 0.1 / s)

        @conditional_distribution(label="yield")
        def likelihood(
            beta: NumericArraySpec((3,)), sigma: NumericArraySpec(()), nu: NumericArraySpec(())
        ) -> StudentT:
            return StudentT(RESPONSE, nu, X @ beta, sigma)

    return likelihood, likelihood & prior


# --- OPTIONAL: bring in a model from another package (comments only) ---------
# A model written in another package enters by its route of CAPABILITIES.md
# "Bring any model", and the steps that need a separate likelihood, such as the
# predictive check, change as that row says. The baseline in PyMC, for example:
#
#     def build_pymc(y_t_per_ha=None):
#         with pymc.Model() as m:
#             beta = pymc.Normal("beta", [5.0, 0.0, 0.0], [2.5, 1.0, 1.0])
#             sigma = pymc.HalfNormal("sigma", 1.0)
#             pymc.Normal(RESPONSE, X @ beta, sigma, observed=y_t_per_ha)
#         return m
#
#     model = PyMCModel(build_pymc)
#
# A Stan program enters as StanModel(stan_file, data=...), and existing code
# that only scores or draws the data enters as distribution(log_prob=...,
# sample=...), or as a simulator for approximate Bayesian computation.


# --- The checks of the pipeline's own code, in the smoke stage ---------------
# A handful of asserts, with no test framework: each conversion matches a
# known value, the log density matches a hand computation at one point, and a
# check of random draws holds within a statistical tolerance, never by
# equality. For a model re-implemented from the user's code, add the agreement
# check of phases/4-existing-pipeline.md here.


def check_code(data, settings) -> None:
    """Raise an AssertionError when a part of the pipeline's own code is wrong."""
    assert np.isclose(inches_to_mm(1.0), 25.4), "inches_to_mm is wrong"
    assert np.isclose(fahrenheit_to_celsius(212.0), 100.0), "fahrenheit_to_celsius is wrong"
    likelihood, model = build_model("normal", data)
    beta, sigma = jnp.array([5.0, 0.5, -0.3]), 0.8
    X = np.column_stack([np.ones_like(data["rain"]), data["rain"], data["temp"]])
    y_t_per_ha = np.asarray(data[RESPONSE])
    by_hand = scipy.stats.norm.logpdf(y_t_per_ha, X @ beta, sigma).sum()
    law = condition_on(likelihood, {"beta": beta, "sigma": jnp.asarray(sigma)})
    assert np.isclose(float(log_prob(law, y_t_per_ha)), by_hand, rtol=1e-4), (
        "the log density differs from the hand computation"
    )
    _, truth = sim_data.simulate(model, 400, RESPONSE, seed=settings.seed)
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
# example validates on prior draws alone. A design with values that the user
# judges realistic adds the recovery at those values from the same recipe.


def validate(model, method: str, settings) -> dict:
    """Fit datasets simulated with known parameters, and grade the recovery."""
    datasets, truth = sim_data.simulate(
        model, settings.budget["n_sims"], RESPONSE, settings.seed + 2
    )
    fits = recovery(fit(model, datasets, settings, method))  # one sweep over the datasets
    num_chains = settings.budget["options"][method]["num_chains"]
    rerun = fits_to_rerun(fits, num_chains)
    if rerun.size:  # a few fits just past the criteria are rerun with the remedy, in one sweep
        again = RecordBatch({RESPONSE: datasets[RESPONSE].raw[rerun]}, "sim")
        for key, values in recovery(fit(model, again, settings, method, remedy=True)).items():
            fits[key][rerun] = values
    z = (fits["mean"] - truth) / fits["sd"]
    coverage = ((fits["low"] <= truth) & (truth <= fits["high"])).mean(axis=0)
    with workflow_run(seed=settings.seed + 7):
        prior = np.asarray(sample(model, sample_shape=(4000,)).without(RESPONSE).to_vector())
    contraction = 1 - (fits["sd"] ** 2).mean(axis=0) / prior.var(axis=0)
    fits_ok = bool(fits_pass(fits, num_chains).all())
    names = coordinate_names(model, RESPONSE)
    grade = grade_validation(z, coverage, contraction, names, fits_ok, rerun.size, QUICK_START)
    table = {"coordinates": names, "coverage_90": coverage, "mean_z": z.mean(axis=0)}
    table |= {"contraction": contraction, "z_beyond_3_by_chance": 0.0027 * z.size}
    table["z_beyond_3"] = [(names[j], int(i)) for i, j in np.argwhere(np.abs(z) > 3)]
    table |= {k: fits[k] for k in ("max_r_hat", "min_ess_bulk", "divergences")}
    return grade | {
        "remedy": bool(rerun.size),
        "table": {k: np.asarray(v).tolist() for k, v in table.items()},
    }


def recovery(posteriors) -> dict:
    """The mean, sd, and 90% interval of each coordinate of each fit, with its diagnostics.

    The arrays are copies, so the rerun of a few fits can replace their rows.
    """
    summary = summarize(posteriors)
    bounds = np.array(quantile(posteriors, jnp.array([0.05, 0.95])).to_vector())
    return {
        "mean": np.array(summary["mean"].to_vector()),
        "sd": np.array(summary["sd"].to_vector()),
        "low": bounds[:, 0],
        "high": bounds[:, 1],
    } | diagnose(posteriors)


def diagnose(posteriors) -> dict:
    """The largest R-hat, the smallest bulk ESS, and the divergences of each fit of a batch."""
    summary = summarize(posteriors)
    return {
        "max_r_hat": np.asarray(summary["r_hat"].to_vector()).max(axis=-1),
        "min_ess_bulk": np.asarray(summary["ess_bulk"].to_vector()).min(axis=-1),
        "divergences": np.array(
            [int(p.annotations["arviz"]["sample_stats"]["diverging"].sum()) for p in posteriors]
        ),
    }


# --- Step 5: fit ---------------------------------------------------------------
# One function fits one dataset or a batch of them, so the validation and the
# real fit use the same method and settings. The method of each model is set in
# MODELS, by the CAPABILITIES.md row "Condition on the data". The diagnostics
# of RESOURCES.md stage E are computed on every fit to the real data.


def fit(model, observed, settings, method: str, remedy: bool = False):
    """The posterior of `model` given `observed`, one per dataset for a batch."""
    options = settings.budget["remedy" if remedy else "options"][method]
    with workflow_run(seed=settings.seed + 3):
        posterior = condition_on.using(method=method, method_options=options)(model, observed)
    if not isinstance(posterior, DistributionBatch):
        add_mcmc_diagnostics(posterior)
    return posterior


def observed_response(data, model, settings) -> jnp.ndarray:
    """The real response after the gate passed, and otherwise one simulated dataset."""
    if settings.real_response:
        return jnp.asarray(data[RESPONSE])
    datasets, _ = sim_data.simulate(model, 1, RESPONSE, settings.seed + 4)
    return jnp.asarray(datasets[RESPONSE].raw[0])


# --- Step 6: check the model against the data --------------------------------
# Test statistics named for the problem, by RESOURCES.md stage F.


def lowest_yield(y_t_per_ha):
    return jnp.min(y_t_per_ha)


def yield_spread(y_t_per_ha):
    return jnp.std(y_t_per_ha)


def check_model(likelihood, posterior, y_t_per_ha: jnp.ndarray, settings) -> dict:
    """The posterior predictive p-value of each test statistic."""
    with workflow_run(seed=settings.seed + 5):
        return predictive_check(likelihood, posterior, [lowest_yield, yield_spread], y_t_per_ha)


# --- Step 7: answer the question ---------------------------------------------
# Each estimand is a function written for one value of the parameters, which
# ProbPipe lifts to the posterior, so the answer carries the posterior's
# uncertainty. The probability of an event is an expectation, the posterior
# mean, by rule 4 of the assistant's Messages.


@function
def per_degree_c(beta: jax.Array, temp_sd_c: jax.Array) -> jax.Array:
    """The change in yield per degree Celsius, from the slope per standard deviation."""
    return beta[..., 2] / temp_sd_c


@function
def expected_yield(beta: jax.Array, x: jax.Array) -> jax.Array:
    """The mean yield in t/ha of a year whose design-matrix row, or rows, are `x`."""
    return x @ beta


def answer(posterior, data, settings) -> dict:
    """The change in yield per degree Celsius, and the chance of a low mean yield when dry and hot."""
    n = settings.budget["n_draws"]
    with workflow_run(seed=settings.seed + 6):
        change = per_degree_c.using(max_evaluations=n)(posterior["beta"], data["temp_sd_c"])
        dry_hot = expected_yield.using(max_evaluations=n)(
            posterior["beta"], weather_row(data, **DRY_HOT)
        )
        p_low = expectation(dry_hot, lambda v: v < LOW_YIELD_T_PER_HA)
    return {"change_per_c": change, "dry_hot": dry_hot, "p_low_yield": p_low}


# --- OPTIONAL: plug-in forecast --------------------------------------------------
# The answer at the posterior mean beside the propagated one. A plug-in answer
# hides the uncertainty, and the difference is often the most persuasive result
# for a scientist. Keep it when the answer is a probability or a decision.


def plug_in(posterior, data) -> float:
    """The mean yield in a dry, hot year at the posterior mean of the parameters."""
    beta = np.asarray(probpipe.mean(posterior["beta"]))
    return float(np.asarray(expected_yield(beta, weather_row(data, **DRY_HOT))))


# --- OPTIONAL: scenarios ---------------------------------------------------------
# The answer under several scenarios in one call: the lifted function takes a
# matrix with one row per scenario. Keep it when the user compares conditions.

SCENARIOS = {"dry_hot": DRY_HOT, "wet_cool": {"rain_in": 30.0, "temp_f": 66.0}}


def scenarios(posterior, data, settings) -> dict:
    """The mean yield and the chance of a low mean yield in each scenario of SCENARIOS."""
    x = jnp.stack([weather_row(data, **weather) for weather in SCENARIOS.values()])
    with workflow_run(seed=settings.seed + 8):
        yields = expected_yield.using(max_evaluations=settings.budget["n_draws"])(
            posterior["beta"], x
        )
        p_low = expectation(yields, lambda v: v < LOW_YIELD_T_PER_HA)
        bounds = quantile(yields, jnp.array([0.025, 0.5, 0.975]))
    return {
        "names": list(SCENARIOS),
        "quantiles": np.asarray(bounds),
        "p_low_yield": np.asarray(p_low),
    }


# --- OPTIONAL: prior sensitivity -------------------------------------------------
# A refit under priors three times as wide, in one sweep with the default
# priors, by RESOURCES.md stage H. Keep it when the user asks how much the
# priors matter.


def prior_sensitivity(name: str, data, y_t_per_ha, settings) -> dict:
    """The change in yield per degree Celsius under the default and the wider priors."""
    models = [build_model(name, data, scale)[1] for scale in (1.0, 3.0)]
    observed = RecordBatch({RESPONSE: jnp.stack([y_t_per_ha, y_t_per_ha])}, "prior")
    posteriors = fit(DistributionBatch(models, "prior"), observed, settings, MODELS[name])
    beta_temp = np.asarray(quantile(posteriors, jnp.array([0.025, 0.5, 0.975])).to_vector())
    return {
        "prior_scale": [1.0, 3.0],
        "change_per_c": beta_temp[:, :, 2] / float(data["temp_sd_c"]),
    }


# --- OPTIONAL: external result ---------------------------------------------------
# The quantiles of another tool or of a saved earlier run beside ProbPipe's,
# such as a colleague's INLA or lme4 fit. Keep it when the user must reconcile
# the two.


def compare_external(answers: dict, external: pd.DataFrame) -> pd.DataFrame:
    """The quantiles of the change per degree Celsius beside those of the external result."""
    ours = np.asarray(quantile(answers["change_per_c"], jnp.array([0.025, 0.5, 0.975])))
    row = external.set_index("estimand").loc["change_per_c"]
    return pd.DataFrame(
        {"probpipe": ours, "external": row[["q2.5", "q50", "q97.5"]].to_numpy(float)},
        index=["2.5%", "50%", "97.5%"],
    )


# --- OPTIONAL: provenance --------------------------------------------------------
# Each result records how it was computed, and whether exactly or approximately.
# Keep it when the user reports the computation, as in a methods section.


def routes(posterior, answers: dict) -> dict:
    """The route of the posterior and of each answer, and whether each is exact."""
    results = {"posterior": posterior, **answers}
    return {
        name: {k: result.provenance.metadata.get(k) for k in ("route", "method", "exact")}
        for name, result in results.items()
    }


# --- OPTIONAL: held-out subset (its score) ---------------------------------------
# Each model scored once on the held-out subset, after the elaboration ends,
# by the continuous ranked probability score of its predictive draws (lower is
# better). In the smoke stage the held-out yields are simulated.


def score_heldout(results: dict, heldout, settings) -> dict:
    """The predictive draws, the observed yields, and the CRPS of each fitted model."""
    scores = {}
    for name, result in results.items():  # one entry per fitted model
        if not result:
            continue
        likelihood, model = build_model(name, heldout)
        with workflow_run(seed=settings.seed + 9):
            draws = np.asarray(
                sample(likelihood & result["posterior"], sample_shape=(1000,))[RESPONSE].raw
            )
        y = np.asarray(observed_response(heldout, model, settings))
        spread = np.abs(draws[::2] - draws[1::2]).mean(axis=0)
        crps = np.abs(draws - y).mean(axis=0) - 0.5 * spread
        scores[name] = {"draws": draws, "y": y, "crps": crps}
    return scores


# --- OPTIONAL: calibration ---------------------------------------------------------
# Needs the block "held-out subset". A reliability table of the probability of a low
# yield on each held-out field: the observed count of low yields in each bin of
# predicted probability beside its expected count, with the range that
# calibrated probabilities give, and the Brier score, by RESOURCES.md stage F.


def calibrate(scores: dict, settings) -> dict:
    """The reliability table and the Brier score of each model."""
    bins = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    rng = np.random.default_rng(settings.seed + 10)
    tables = {}
    for name, score in scores.items():
        p = (score["draws"] < LOW_YIELD_T_PER_HA).mean(axis=0)
        event = score["y"] < LOW_YIELD_T_PER_HA
        which = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
        calibrated = rng.random((4000, p.size)) < p  # outcomes drawn as if p were calibrated
        rows = []
        for b in range(len(bins) - 1):
            members = which == b
            counts = calibrated[:, members].sum(axis=1)
            low, high = np.percentile(counts, [2.5, 97.5])
            rows.append(
                [
                    bins[b],
                    bins[b + 1],
                    members.sum(),
                    p[members].sum(),
                    low,
                    high,
                    event[members].sum(),
                ]
            )
        tables[name] = {"rows": np.array(rows), "brier": float(((p - event) ** 2).mean())}
    return tables


# --- OPTIONAL: compare algorithms ------------------------------------------------
# The baseline fitted by each method of COMPARED_METHODS, with the means and the
# largest R-hat side by side, by RESOURCES.md stage E. condition_on.check names
# the method that a call would select, before any computation. Keep it when the
# choice of method is in question.


def compare_algorithms(data, results: dict, settings) -> dict:
    """The method that condition_on selects, and each method's means and largest R-hat."""
    name = next(iter(MODELS))
    if not results[name]:
        return {}
    _, model = build_model(name, data)
    observed = {RESPONSE: results[name]["y"]}
    table = {}
    for method in COMPARED_METHODS:  # one fit per method
        summary = summarize(fit(model, observed, settings, method))
        table[method] = np.r_[summary["mean"].to_vector(), summary["r_hat"].to_vector().max()]
    selected = condition_on.check(model, observed).selected.method_name
    return {"selected": selected, "table": table}


# --- OPTIONAL: several datasets ----------------------------------------------------
# The baseline fitted to each year on its own, as one batched sweep over a batch
# of models, one per year at its own weather. The validation covered the pooled
# design, so these fits are labeled not validated. Keep it when the design
# reports an answer per site, group, or year.


def fit_by_year(data, results: dict, settings) -> dict:
    """The change in yield per degree Celsius in each year, from one sweep of fits."""
    name = next(iter(MODELS))
    if not results[name]:
        return {}
    years = np.unique(np.asarray(data["year"]))
    rows = [np.asarray(data["year"]) == year for year in years]
    models = DistributionBatch([build_model(name, subset(data, r))[1] for r in rows], "year")
    y = np.asarray(results[name]["y"])
    observed = RecordBatch({RESPONSE: jnp.stack([y[r] for r in rows])}, "year")
    posteriors = fit(models, observed, settings, MODELS[name])
    beta = np.asarray(quantile(posteriors, jnp.array([0.025, 0.5, 0.975])).to_vector())
    return {"years": years, "change_per_c": beta[:, :, 2] / float(data["temp_sd_c"])}


# --- OPTIONAL: large data (comments only) ----------------------------------------
# For many observations, stochastic-gradient MCMC estimates each gradient from a
# minibatch: condition_on.using(method="blackjax_sgld", method_options={...})
# with a MinibatchedDistribution, by RESOURCES.md stage E "Many observations".
# Many independent fits, such as one per site or per bootstrap replicate, run
# in parallel through an executor of executor_registry, or as Prefect tasks on
# Ray, by the CAPABILITIES.md row "Run a pipeline at scale".


if __name__ == "__main__":
    main()
