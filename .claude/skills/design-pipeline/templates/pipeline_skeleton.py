"""Crop yield and growing-season weather: the yields of ten fields over four years.

The pipeline answers two questions: how much the yield changes per degree
Celsius, and how likely a low mean yield is in a dry, hot year. Each choice is
made in one place:

1. the data file, its checks, and its units are in `prepare_inputs()` and
   `check_columns()`;
2. the models, their priors, and their likelihoods are in `build_model()`;
3. the inference method of each model and the budgets of each stage are in
   `INFERENCE_METHODS` and `BUDGETS`;
4. the test statistics of the predictive check are in `TEST_STATISTICS`;
5. the answers are in `estimands()`, and the event of a low yield is set by
   `LOW_YIELD_T_PER_HA` and `DRY_HOT`.

Run it with `STAGE=smoke`, then `STAGE=validate`, then `STAGE=fit`. Each
stage prints its results and saves them in its output folder, and only the
fit stage writes to `results/`.
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
    QUANTILES,
    fit_diagnostics,
    fits_to_rerun,
    grade_recovery,
    record_run,
    record_validation,
    resolve_stage,
)

from probpipe import (
    Bernoulli,
    DistributionBatch,
    Gamma,
    HalfNormal,
    Normal,
    NumericArraySpec,
    NumericRecord,
    RecordBatch,
    StudentT,
    condition_on,
    conditional_distribution,
    expectation,
    factor,
    function,
    log_prob,
    mean,
    predictive_check,
    quantile,
    sample,
    summarize,
    workflow_run,
)
from probpipe.diagnostics import add_mcmc_diagnostics

# --- Settings -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "example_data.csv"
DESIGN_VERSION = 1
SEED = 20261025  # the seed of the run, from which ProbPipe derives the key of each random step
QUICK_START = True  # the mode, which decides whether the validation grades coverage
RESPONSE = "yield_t_per_ha"
# Each model by name with the inference method that fits it, the baseline first.
INFERENCE_METHODS = {"normal": "blackjax_nuts", "student_t": "blackjax_nuts"}
BASELINE = next(iter(INFERENCE_METHODS))
# The event of the answer: a mean yield below LOW_YIELD_T_PER_HA in a dry, hot year.
LOW_YIELD_T_PER_HA = 4.0
DRY_HOT = {"rain_in": 15.0, "temp_f": 76.0}
# The plausible ranges of one field's yield and of the spread of one dataset's yields.
PLAUSIBLE_YIELD_T_PER_HA = (0.0, 20.0)
PLAUSIBLE_SPREAD_T_PER_HA = (0.2, 5.0)
HOLDOUT_YEAR = 2024  # OPTIONAL: held-out subset
EXTERNAL_FILE = HERE / "external_quantiles.csv"  # OPTIONAL: external result
COMPARED_METHODS = ("blackjax_nuts", "tfp_nuts", "blackjax_rwmh")  # OPTIONAL: compare algorithms
# The budgets of each stage. "options" holds the settings of each method, and
# "remedy" the settings of a rerun of a fit that misses the criteria of the
# diagnostics. "n_sims" counts the simulated datasets of the validation,
# "n_prior_draws" the prior draws, and "n_draws" the draws of each answer.
SMOKE = {"num_chains": 2, "num_warmup": 100, "num_results": 100}
FULL = {"num_chains": 4, "num_warmup": 500, "num_results": 1000}
SLOW = {"num_chains": 4, "num_warmup": 2000, "num_results": 4000}  # for random-walk Metropolis
REMEDY = {"target_accept_prob": 0.95}
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
        BUDGETS,
        models=INFERENCE_METHODS,
        design_version=DESIGN_VERSION,
        seed=SEED,
        validated=VALIDATED,
    )
    record_run(settings)
    report.show_stage(settings)
    data = prepare_inputs(pd.read_csv(DATA_FILE))
    scales = weather_scales(data)
    data, heldout = hold_out(data, HOLDOUT_YEAR)  # OPTIONAL: held-out subset
    models = {name: build_model(name, data, scales) for name in INFERENCE_METHODS}
    # Each random step takes its key from SEED in the order of the steps, so a rerun repeats it.
    with workflow_run(seed=SEED):
        if settings.checks_code:
            check_code(data, scales, models[BASELINE])
        for name, model in models.items():  # the models differ in their parameters
            check_before_fit(name, model, settings)
        if not settings.models_to_fit:  # the validate stage and a hold fit no model
            return
        fits = {
            name: run_model(name, models[name], data, scales, settings)
            for name in settings.models_to_fit
        }
        report.show_model_comparison(fits, settings)  # OPTIONAL: compare models
        scores = score_heldout(fits, heldout, scales, settings)  # OPTIONAL: held-out subset
        report.show_heldout_scores(scores, settings)  # OPTIONAL: held-out subset
        reliability = reliability_tables(scores, settings)  # OPTIONAL: calibration
        report.show_reliability(reliability, settings)  # OPTIONAL: calibration


def check_before_fit(name: str, model, settings) -> None:
    """The prior predictive check and the validation of one model, in the stages that run them."""
    if settings.checks_priors:
        y_prior_t_per_ha = prior_predictive_yields(model, settings)
        prior_check = prior_predictive_shares(y_prior_t_per_ha)
        report.show_prior_predictive(name, y_prior_t_per_ha, prior_check, settings)
    if settings.validates:
        validation = validate(model, INFERENCE_METHODS[name], settings)
        record_validation(name, validation, settings)  # the grade that the gate of the fit reads
        report.show_validation(name, validation, settings)


def run_model(name: str, model, data: NumericRecord, scales: NumericRecord, settings) -> dict:
    """Fit one model to the response of the stage, and run the steps after the fit."""
    data = with_stage_response(data, model, settings)
    remedy = name in settings.models_with_remedy  # the validation needed it, so every fit uses it
    posterior = fit(model, {RESPONSE: data[RESPONSE]}, settings, INFERENCE_METHODS[name], remedy)
    report.show_diagnostics(name, posterior, settings)
    check = predictive_check(factor(model, RESPONSE), posterior, TEST_STATISTICS, data[RESPONSE])
    report.show_predictive_check(name, check, settings)
    X_dry_hot = design_matrix(weather(**DRY_HOT), scales)
    answers = estimands(posterior, X_dry_hot, scales, settings)
    report.show_answers(name, answers, settings)
    plug_in = linear_predictor(mean(posterior["beta"]), X_dry_hot)  # OPTIONAL: plug-in forecast
    report.show_plug_in(name, plug_in, answers, settings)  # OPTIONAL: plug-in forecast
    scenarios = answers_by_scenario(posterior, scales, settings)  # OPTIONAL: scenarios
    report.show_scenarios(name, scenarios, settings)  # OPTIONAL: scenarios
    sensitivity = prior_sensitivity(name, data, scales, settings)  # OPTIONAL: prior sensitivity
    report.show_prior_sensitivity(name, sensitivity, settings)  # OPTIONAL: prior sensitivity
    external = pd.read_csv(EXTERNAL_FILE)  # OPTIONAL: external result
    report.show_external_comparison(name, answers, external, settings)  # OPTIONAL: external result
    report.show_routes(name, {"posterior": posterior, **answers}, settings)  # OPTIONAL: provenance
    methods = compare_algorithms(name, model, data, settings)  # OPTIONAL: compare algorithms
    report.show_algorithms(name, methods, settings)  # OPTIONAL: compare algorithms
    by_year = fit_by_year(name, data, scales, settings)  # OPTIONAL: several datasets
    report.show_by_year(name, by_year, settings)  # OPTIONAL: several datasets
    return {"posterior": posterior, "check": check, "answers": answers}


# --- Step 1: load and check the data ----------------------------------------

MISSING_CODE = -99


@function
def prepare_inputs(table: pd.DataFrame) -> NumericRecord:
    """The year, the rainfall in mm, the temperature in °C, and the yield of each field."""
    check_columns(table)
    return NumericRecord(
        {
            "year": table["year"].to_numpy(),
            "rain_mm": inches_to_mm(table["rain_in"].to_numpy()),
            "temp_c": fahrenheit_to_celsius(table["temp_f"].to_numpy()),
            RESPONSE: table[RESPONSE].to_numpy(),
        }
    )


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


def inches_to_mm(length_in):
    return length_in * 25.4


def fahrenheit_to_celsius(temp_f):
    return (temp_f - 32.0) * 5.0 / 9.0


def weather(rain_in, temp_f) -> dict:
    """The rainfall `rain_in` and the temperature `temp_f` in the model's units, mm and °C."""
    rain_mm = inches_to_mm(jnp.asarray(rain_in))
    return {"rain_mm": rain_mm, "temp_c": fahrenheit_to_celsius(jnp.asarray(temp_f))}


@function
def weather_scales(data: NumericRecord) -> NumericRecord:
    """The mean and the standard deviation of the rainfall and of the temperature."""
    return NumericRecord(
        {x: {"mean": jnp.mean(data[x]), "sd": jnp.std(data[x])} for x in ("rain_mm", "temp_c")}
    )


@function
def select_rows(data: NumericRecord, rows) -> NumericRecord:
    """The observations of the rows that `rows` selects, as a mask or as indices."""
    return NumericRecord({field: values[rows] for field, values in data.items()})


TEMP = 2  # the column of temperature in X, and so the coordinate of its slope in beta


def design_matrix(weather, scales: NumericRecord) -> jnp.ndarray:
    """The design matrix X, whose row i is [1, rain_i, temp_i] in standard deviations."""
    rain = (jnp.asarray(weather["rain_mm"]) - scales["rain_mm/mean"]) / scales["rain_mm/sd"]
    temp = (jnp.asarray(weather["temp_c"]) - scales["temp_c/mean"]) / scales["temp_c/sd"]
    return jnp.stack([jnp.ones_like(rain), rain, temp], axis=-1)


# --- OPTIONAL: held-out subset -------------------------------------------------
# One year set aside before any elaboration, and scored once after it ends.


def hold_out(data: NumericRecord, year: int) -> tuple[NumericRecord, NumericRecord]:
    """The rows of the years other than `year`, and the rows of `year`, held out."""
    years = np.asarray(data["year"])
    return select_rows(data, years != year), select_rows(data, years == year)


# --- Step 2: the models --------------------------------------------------------


def build_model(name: str, data: NumericRecord, scales: NumericRecord, prior_scale: float = 1.0):
    """The joint model `likelihood & prior` of yield given rainfall and temperature.

    yield_i ~ Normal(X_i beta, sigma) for "normal", and
    yield_i ~ StudentT(nu, X_i beta, sigma) for "student_t", with
    X_i = [1, rain_i, temp_i] in standard deviations. beta_0 ~ Normal(6, 1.5)
    puts the mean yield between about 3 and 9 t/ha, each slope ~ Normal(0, 1) allows
    a change of up to about 2 t/ha per standard deviation of the weather,
    sigma ~ HalfNormal(1), and nu ~ Gamma(2, 0.1) has its mean at 20, where
    the errors are close to normal. `prior_scale` widens every prior scale.
    """
    X = design_matrix(data, scales)
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

    return likelihood & prior


# --- OPTIONAL: bring in a model from another package (comments only) ---------
# The baseline written in PyMC, which enters by the row "Bring any model" of CAPABILITIES.md:
#
#     def build_pymc(y_t_per_ha=None):
#         with pymc.Model() as m:
#             beta = pymc.Normal("beta", [6.0, 0.0, 0.0], [1.5, 1.0, 1.0])
#             sigma = pymc.HalfNormal("sigma", 1.0)
#             pymc.Normal(RESPONSE, X @ beta, sigma, observed=y_t_per_ha)
#         return m
#
#     model = PyMCModel(build_pymc)


# --- The checks of the pipeline's own code, in the smoke stage ---------------


def check_code(data: NumericRecord, scales: NumericRecord, model) -> None:
    """Raise an AssertionError when a part of the pipeline's own code is wrong."""
    assert np.isclose(inches_to_mm(1.0), 25.4), "inches_to_mm is wrong"
    assert np.isclose(fahrenheit_to_celsius(212.0), 100.0), "fahrenheit_to_celsius is wrong"
    beta, sigma = jnp.array([5.0, 0.5, -0.3]), 0.8
    y_t_per_ha = np.asarray(data[RESPONSE])
    by_hand = scipy.stats.norm.logpdf(y_t_per_ha, design_matrix(data, scales) @ beta, sigma).sum()
    law = condition_on(factor(model, RESPONSE), {"beta": beta, "sigma": jnp.asarray(sigma)})
    assert np.isclose(float(log_prob(law, y_t_per_ha)), by_hand, rtol=1e-4), (
        "the log density differs from the hand computation"
    )
    n_draws = 400
    beta_0 = np.asarray(sample(model, sample_shape=(n_draws,))["beta"])[:, 0]
    standard_error = 1.5 / np.sqrt(n_draws)  # of the mean of draws of beta_0 ~ Normal(6, 1.5)
    # A check of random draws holds within four standard errors.
    assert abs(beta_0.mean() - 6.0) < 4 * standard_error, "the draws of beta_0 are off"


# --- Step 3: the prior predictive check --------------------------------------


def prior_predictive_yields(model, settings) -> np.ndarray:
    """Yields simulated from the prior predictive distribution, of shape (draw, field), in t/ha."""
    draws = sample(model, sample_shape=(settings.budget["n_prior_draws"],))
    return np.asarray(draws[RESPONSE].raw)


def prior_predictive_shares(y_prior_t_per_ha: np.ndarray) -> dict:
    """The shares of simulated yields and of simulated datasets outside their plausible ranges.

    The priors pass when at most 5% of the yields and at most 10% of the
    datasets lie outside.
    """
    low, high = PLAUSIBLE_YIELD_T_PER_HA
    yields_outside = (y_prior_t_per_ha < low) | (y_prior_t_per_ha > high)
    spread_t_per_ha = y_prior_t_per_ha.std(axis=1)
    low_spread, high_spread = PLAUSIBLE_SPREAD_T_PER_HA
    datasets_outside = (spread_t_per_ha < low_spread) | (spread_t_per_ha > high_spread)
    shares = {"yields": float(yields_outside.mean()), "datasets": float(datasets_outside.mean())}
    return shares | {"passes": shares["yields"] <= 0.05 and shares["datasets"] <= 0.10}


# --- Step 4: validate the computation on simulated data ----------------------


def validate(model, method: str, settings) -> dict:
    """Fit datasets simulated with known parameters theta, and grade the recovery of theta."""
    datasets, theta = sim_data.simulate(model, settings.budget["n_sims"], RESPONSE)
    posteriors = fit(model, datasets, settings, method)  # one sweep
    summaries = summarize(posteriors, interval=COVERAGE_LEVEL)
    num_chains = settings.budget["options"][method]["num_chains"]
    rerun = fits_to_rerun(fit_diagnostics(summaries, posteriors), num_chains)
    if rerun.size:  # a few fits just past the criteria are rerun with the remedy, in one sweep
        again = RecordBatch({RESPONSE: datasets[RESPONSE].raw[rerun]}, "sim")
        refits = dict(zip(rerun.tolist(), fit(model, again, settings, method, remedy=True)))
        posteriors = DistributionBatch([refits.get(i, p) for i, p in enumerate(posteriors)], "sim")
        summaries = summarize(posteriors, interval=COVERAGE_LEVEL)
    theta_prior = sample(model, sample_shape=(settings.budget["n_prior_draws"],)).without(RESPONSE)
    return grade_recovery(
        summaries, posteriors, theta, theta_prior, num_chains, rerun.size, QUICK_START
    )


# --- Step 5: fit ---------------------------------------------------------------


def fit(model, observed, settings, method: str, remedy: bool = False):
    """The posterior of `model` given `observed`, one per dataset for a batch."""
    options = settings.budget["remedy" if remedy else "options"][method]
    posterior = condition_on.using(method=method, method_options=options)(model, observed)
    if not isinstance(posterior, DistributionBatch):  # a batch is diagnosed by summarize()
        add_mcmc_diagnostics(posterior)
    return posterior


def with_stage_response(data: NumericRecord, model, settings) -> NumericRecord:
    """`data` after the gate passed, and else `data` with a response simulated from `model`."""
    if settings.uses_real_response:
        return data
    datasets, _ = sim_data.simulate(model, 1, RESPONSE)
    return data.replace({RESPONSE: datasets[RESPONSE].raw[0]})


# --- What the validation depends on ----------------------------------------------
# A change to any of these makes a passed validation stale.

VALIDATED = (
    # the data file, whose covariates and size the simulated datasets share, and the data step
    DATA_FILE,
    prepare_inputs,
    check_columns,
    inches_to_mm,
    fahrenheit_to_celsius,
    weather_scales,
    MISSING_CODE,
    # the split, which decides the rows of the fitted data
    select_rows,
    hold_out,  # OPTIONAL: held-out subset
    HOLDOUT_YEAR,  # OPTIONAL: held-out subset
    # the models
    design_matrix,
    build_model,
    RESPONSE,
    # the simulation of datasets, and the fit with its methods, its budgets, and the remedies
    # of the validation and of the fit to the real data
    sim_data.simulate,
    fit,
    INFERENCE_METHODS,
    BUDGETS["validate"],
    {
        kind: {method: BUDGETS["fit"][kind].get(method) for method in INFERENCE_METHODS.values()}
        for kind in ("options", "remedy")
    },
    # the validation itself
    validate,
    QUICK_START,
)


# --- Step 6: check the model against the data --------------------------------


def lowest_yield(y_t_per_ha):
    return jnp.min(y_t_per_ha)


def yield_spread(y_t_per_ha):
    return jnp.std(y_t_per_ha)


TEST_STATISTICS = [lowest_yield, yield_spread]


# --- Step 7: answer the question ---------------------------------------------


@function
def unstandardize_slope(beta: jax.Array, sd: jax.Array, coordinate: int) -> jax.Array:
    """The slope per unit of a covariate, beta[coordinate] / sd, with sd the covariate's.

    It also converts quantiles of beta, since quantiles commute with the
    division by the positive `sd`.
    """
    return beta[..., coordinate] / sd


@function
def linear_predictor(beta: jax.Array, X: jax.Array) -> jax.Array:
    """The mean yield X beta in t/ha of each row of `X`."""
    return X @ beta


def is_low_yield(y_t_per_ha):
    """Whether each yield is below LOW_YIELD_T_PER_HA, the event of the answer."""
    return y_t_per_ha < LOW_YIELD_T_PER_HA


def estimands(posterior, X_dry_hot: jnp.ndarray, scales: NumericRecord, settings) -> dict:
    """The change in yield per °C, and the chance of a low mean yield in a dry, hot year."""
    n_draws = settings.budget["n_draws"]
    beta = posterior["beta"]
    change_per_c = unstandardize_slope.using(max_evaluations=n_draws)(
        beta, scales["temp_c/sd"], TEMP
    )
    yield_dry_hot = linear_predictor.using(max_evaluations=n_draws)(beta, X_dry_hot)
    return {
        "change_per_c": change_per_c,
        "yield_dry_hot": yield_dry_hot,
        "p_low_yield_dry_hot": expectation(yield_dry_hot, is_low_yield),
    }


# --- OPTIONAL: plug-in forecast --------------------------------------------------
# The answer at the posterior mean, which hides the uncertainty, beside the propagated one.


# --- OPTIONAL: scenarios ---------------------------------------------------------
# The answer under several scenarios in one call, with one row of X per scenario.

SCENARIOS = pd.DataFrame({"dry_hot": DRY_HOT, "wet_cool": {"rain_in": 30.0, "temp_f": 66.0}}).T


def answers_by_scenario(posterior, scales: NumericRecord, settings) -> dict:
    """The mean yield, and the chance of a low mean yield, in each scenario of SCENARIOS."""
    rain_in, temp_f = SCENARIOS["rain_in"].to_numpy(), SCENARIOS["temp_f"].to_numpy()
    X = design_matrix(weather(rain_in, temp_f), scales)
    n_draws = settings.budget["n_draws"]
    yields = linear_predictor.using(max_evaluations=n_draws)(posterior["beta"], X)
    return {
        "scenario": SCENARIOS.index,
        "yield": yields,
        "p_low": expectation(yields, is_low_yield),
    }


# --- OPTIONAL: prior sensitivity -------------------------------------------------
# A refit under priors three times as wide, in one sweep with the default priors.

PRIOR_SCALES = [1.0, 3.0]


def prior_sensitivity(name: str, data: NumericRecord, scales: NumericRecord, settings) -> dict:
    """The quantiles of the change in yield per °C under each prior scale of PRIOR_SCALES."""
    # One model per scale, since each holds its own priors.
    models = [build_model(name, data, scales, scale) for scale in PRIOR_SCALES]
    observed = {RESPONSE: jnp.tile(jnp.asarray(data[RESPONSE]), (len(PRIOR_SCALES), 1))}
    batch = DistributionBatch(models, "prior_scale"), RecordBatch(observed, "prior_scale")
    remedy = name in settings.models_with_remedy
    posteriors = fit(*batch, settings, INFERENCE_METHODS[name], remedy)
    beta = quantile(posteriors, QUANTILES)["beta"]
    change_per_c = unstandardize_slope(beta, scales["temp_c/sd"], TEMP)
    return {"prior_scale": PRIOR_SCALES, "change_per_c": change_per_c}


# --- OPTIONAL: external result ---------------------------------------------------
# The quantiles of another tool or of an earlier run, read from EXTERNAL_FILE.


# --- OPTIONAL: provenance --------------------------------------------------------
# The route and the method of each result, and whether it is exact.


# --- OPTIONAL: compare algorithms ------------------------------------------------
# The model fitted by each method of COMPARED_METHODS.


def compare_algorithms(name: str, model, data: NumericRecord, settings) -> dict:
    """The method that condition_on selects, and the posterior of each method."""
    observed = {RESPONSE: data[RESPONSE]}
    remedied = name in settings.models_with_remedy
    posteriors = {}
    for method in COMPARED_METHODS:  # each method runs its own algorithm
        # The budgets define a remedy for the methods of the validation alone.
        remedy = remedied and method in settings.budget["remedy"]
        posteriors[method] = fit(model, observed, settings, method, remedy)
    selected = condition_on.check(model, observed).selected.method_name
    return {"selected_method": selected, "posteriors": posteriors}


# --- OPTIONAL: several datasets ----------------------------------------------------
# The model fitted to each year on its own, in one sweep over a batch of models.


def fit_by_year(name: str, data: NumericRecord, scales: NumericRecord, settings) -> dict:
    """The quantiles of the change in yield per degree Celsius in each year, from one sweep."""
    years, rows_per_year = np.unique(np.asarray(data["year"]), return_counts=True)
    # A batch stacks the years, so each year needs the same number of rows.
    if np.any(rows_per_year != rows_per_year[0]):
        counts = dict(zip(years.tolist(), rows_per_year.tolist(), strict=True))
        raise ValueError(f"fit_by_year needs the same number of rows in each year, not {counts}.")
    rows = np.argsort(np.asarray(data["year"]), kind="stable").reshape(len(years), -1)
    # One model per year, since each holds the weather of its year.
    models = [build_model(name, select_rows(data, year_rows), scales) for year_rows in rows]
    observed = {RESPONSE: select_rows(data, rows)[RESPONSE]}  # (year, field)
    batch = DistributionBatch(models, "year"), RecordBatch(observed, "year")
    remedy = name in settings.models_with_remedy
    posteriors = fit(*batch, settings, INFERENCE_METHODS[name], remedy)
    beta = quantile(posteriors, QUANTILES)["beta"]
    change_per_c = unstandardize_slope(beta, scales["temp_c/sd"], TEMP)
    return {"year": years, "change_per_c": change_per_c}


# --- OPTIONAL: held-out subset (its score) ---------------------------------------
# Each model's predictive draws of the held-out yields, scored once.


def score_heldout(fits: dict, heldout: NumericRecord, scales: NumericRecord, settings) -> dict:
    """The predictive draws of the held-out yields, the yields, and the mean CRPS of each model."""
    scores = {}
    for name, result in fits.items():  # the models differ in their parameters
        # The held-out year has its own weather, so its model is built from its rows.
        model = build_model(name, heldout, scales)
        predictive = factor(model, RESPONSE) & result["posterior"]
        draws = sample(predictive, sample_shape=(settings.budget["n_draws"],))
        y_draws = jnp.asarray(draws[RESPONSE].raw)  # (draw, field)
        y_t_per_ha = jnp.asarray(with_stage_response(heldout, model, settings)[RESPONSE])
        scores[name] = {
            "y_draws": y_draws,
            "y_t_per_ha": y_t_per_ha,
            "mean_crps": mean_crps(y_draws, y_t_per_ha),
        }
    return scores


def mean_crps(y_draws: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """The CRPS E|Y - y| - E|Y - Y'| / 2 of each field, averaged over the fields.

    Y and Y' are independent predictive draws, here the even and the odd draws.
    """
    spread = jnp.abs(y_draws[::2] - y_draws[1::2]).mean(axis=0)
    return (jnp.abs(y_draws - y).mean(axis=0) - spread / 2).mean()


# --- OPTIONAL: calibration ---------------------------------------------------------
# The reliability of each model's probability of a low yield on the held-out fields.

PROBABILITY_BINS = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])


def reliability_tables(scores: dict, settings) -> dict:
    """The reliability table and the Brier score of each model's probability of a low yield.

    The table counts the observed and the expected low yields in each bin of
    predicted probability, with the range of the count under calibrated
    probabilities.
    """
    n_bins, n_draws = len(PROBABILITY_BINS) - 1, settings.budget["n_draws"]
    tables = {}
    for name, score in scores.items():  # each model puts different fields in each bin
        p_low = is_low_yield(score["y_draws"]).mean(axis=0)  # (field,)
        observed_low = is_low_yield(score["y_t_per_ha"])
        bin_of_field = jnp.clip(jnp.digitize(p_low, PROBABILITY_BINS) - 1, 0, n_bins - 1)
        in_bin = (bin_of_field[:, None] == jnp.arange(n_bins)).astype(float)  # (field, bin)
        calibrated = sample(Bernoulli("low", probs=p_low), sample_shape=(n_draws,))
        counts = jnp.asarray(calibrated.raw) @ in_bin  # (draw, bin)
        range_low, _, range_high = jnp.quantile(counts, QUANTILES, axis=0)
        tables[name] = {
            "from": PROBABILITY_BINS[:-1],
            "to": PROBABILITY_BINS[1:],
            "fields": in_bin.sum(axis=0),
            "expected": p_low @ in_bin,
            "range_low": range_low,
            "range_high": range_high,
            "observed": observed_low @ in_bin,
            "brier": jnp.mean((p_low - observed_low) ** 2),
        }
    return tables


# --- OPTIONAL: large data (comments only) ----------------------------------------
# Stochastic-gradient MCMC for many observations, on a MinibatchedDistribution:
# condition_on.using(method="blackjax_sgld", method_options={...}).


if __name__ == "__main__":
    main()
