"""summarize: a law's standard summaries in one record that displays as a table (VI.5)."""

from __future__ import annotations

import arviz_stats as azs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from scipy import stats

from probpipe import (
    EmpiricalDistribution,
    Gamma,
    Normal,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    OutputSpec,
    Poisson,
    RecordSpec,
    conditional_distribution,
    mean,
    quantile,
    standard_deviation,
    summarize,
    variance,
    workflow_run,
)
from probpipe.distributions._batches import DistributionBatch
from probpipe.inference._approximate_distribution import make_posterior

# Each summary of these laws is a float32 closed form, which matches SciPy's
# float64 value to within a few units in the last place.
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}

# The posterior stores its draws in float32, so ArviZ's diagnostics of it differ
# from those of the float64 draws by at most 3e-6 relative, as measured here.
ARVIZ_TOLERANCE = {"rtol": 1e-5}

# sd / sqrt(n) of a float32 standard deviation, computed in float32 by summarize
# and in float64 here, agree to float32 rounding.
FLOAT32_TOLERANCE = {"rtol": 1e-6}


@pytest.fixture
def model():
    return (Normal("mu", 0.0, 2.0) * Gamma("tau", 2.0, 1.0)).with_label("model")


def _kernel_model():
    """A law whose summaries of ``y`` take Monte Carlo draws and of ``mu`` a closed form."""

    def y_given_mu(mu: jax.Array) -> Normal:
        return Normal("y", jnp.full(2, mu), 1.0)

    prior = Normal("mu", 0.0, 2.0)
    kernel = conditional_distribution(y_given_mu, given_spec=prior.event_spec.components)
    return (kernel * prior).with_label("model")


def _posterior(
    draws: np.ndarray, *, vector: bool = False, nested: bool = False
) -> EmpiricalDistribution:
    """An MCMC posterior of *draws*, whose shape is ``(chains, draws, 2)``.

    The posterior is over ``mu`` and ``sigma``, over a vector ``beta`` when
    *vector*, or over ``population/mu`` and ``population/sigma`` when *nested*.
    """
    scalar = NumericArraySpec(())
    if vector:
        event = OutputSpec(beta=NumericArraySpec((2,)))
    elif nested:
        event = OutputSpec(RecordSpec(population=RecordSpec(mu=scalar, sigma=scalar)))
    else:
        event = OutputSpec(RecordSpec(mu=scalar, sigma=scalar))
    return make_posterior(list(draws), (), "test", event_spec=event)


@pytest.fixture
def chain_draws():
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 500, 2)).astype(np.float32)


def _expected_defaults(interval: float = 0.8) -> dict[str, dict[str, float]]:
    """The default columns of ``model``, from SciPy."""
    low, high = (1 - interval) / 2, (1 + interval) / 2
    normal, gamma = stats.norm(0.0, 2.0), stats.gamma(2.0, scale=1.0)
    return {
        "mean": {"mu": 0.0, "tau": 2.0},
        "sd": {"mu": 2.0, "tau": np.sqrt(2.0)},
        "median": {"mu": 0.0, "tau": gamma.ppf(0.5)},
        "low": {"mu": normal.ppf(low), "tau": gamma.ppf(low)},
        "high": {"mu": normal.ppf(high), "tau": gamma.ppf(high)},
    }


class TestTheDefaultColumns:
    def test_each_column_holds_its_summary_at_each_path(self, model):
        summary = summarize(model)
        expected = _expected_defaults()

        assert tuple(summary.children) == ("mean", "sd", "median", "q10", "q90")
        for name, key in [("mean", "mean"), ("sd", "sd"), ("median", "median")]:
            for path in ("mu", "tau"):
                np.testing.assert_allclose(
                    float(summary[f"{name}/{path}"]), expected[key][path], **TOLERANCE
                )
        for name, key in [("q10", "low"), ("q90", "high")]:
            for path in ("mu", "tau"):
                np.testing.assert_allclose(
                    float(summary[f"{name}/{path}"]), expected[key][path], **TOLERANCE
                )

    def test_each_column_is_the_term_its_operation_returns(self, model):
        summary = summarize(model)

        for name, operation in [("mean", "mean"), ("sd", "standard_deviation")]:
            assert summary.at_path(name).provenance.metadata["func"] == operation
        assert summary.at_path("q90").provenance.metadata["func"] == "quantile"
        assert summary.at_path("mean") == mean(model).with_label("mean")

    def test_the_summary_is_labeled_by_the_laws_notation(self, model):
        assert summarize(model).label == "summary of model(mu, tau)"

    def test_a_law_of_one_whole_term_holds_each_column_under_its_component(self):
        summary = summarize(Normal("theta", 1.0, 3.0, label="prior"))

        assert summary.label == "summary of prior(theta)"
        assert summary.keys() == (
            "mean/theta",
            "sd/theta",
            "median/theta",
            "q10/theta",
            "q90/theta",
        )
        np.testing.assert_allclose(
            float(summary["q90/theta"]), stats.norm(1.0, 3.0).ppf(0.9), **TOLERANCE
        )

    def test_an_array_component_takes_a_row_per_element(self):
        law = Normal("y", jnp.zeros(3), jnp.array([1.0, 2.0, 3.0]), label="noise")

        summary = summarize(law)

        np.testing.assert_allclose(np.asarray(summary["sd/y"]), [1.0, 2.0, 3.0], **TOLERANCE)
        rows = [line.strip().split("  ")[0] for line in repr(summary).splitlines()[2:-1]]
        assert rows == ["y[0]", "y[1]", "y[2]"]

    def test_a_nested_law_keeps_its_paths(self):
        hyper = (Normal("mu", 0.0, 2.0) * Gamma("tau", 2.0, 1.0)).with_path_names(
            {"mu": "population/mu", "tau": "population/tau"}
        )
        law = (hyper * Normal("x", 1.0, 1.0)).with_label("hierarchical")

        summary = summarize(law)

        assert summary.at_path("mean").keys() == ("population/mu", "population/tau", "x")
        np.testing.assert_allclose(float(summary["sd/population/tau"]), np.sqrt(2.0), **TOLERANCE)
        rows = [line.strip().split("  ")[0] for line in repr(summary).splitlines()[2:-1]]
        assert rows == ["population/mu", "population/tau", "x"]


class TestTheInterval:
    @pytest.mark.parametrize(
        ("interval", "low", "high"),
        [(0.9, "q5", "q95"), (0.95, "q2.5", "q97.5"), (0.5, "q25", "q75")],
    )
    def test_the_interval_sets_the_quantile_columns(self, model, interval, low, high):
        summary = summarize(model, interval=interval)
        expected = _expected_defaults(interval)

        assert tuple(summary.children)[3:] == (low, high)
        np.testing.assert_allclose(float(summary[f"{low}/mu"]), expected["low"]["mu"], **TOLERANCE)
        np.testing.assert_allclose(
            float(summary[f"{high}/tau"]), expected["high"]["tau"], **TOLERANCE
        )

    @pytest.mark.parametrize("interval", [0.0, 1.0, 1.5, -0.2])
    def test_an_interval_outside_zero_and_one_raises(self, model, interval):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            summarize(model, interval=interval)

    @pytest.mark.parametrize("interval", ["0.8", True, None])
    def test_an_interval_that_is_not_a_number_raises(self, model, interval):
        with pytest.raises(TypeError, match="interval must be a number"):
            summarize(model, interval=interval)


class TestGivenStatistics:
    def test_operations_are_named_by_their_labels(self, model):
        summary = summarize(model, mean, variance)

        assert tuple(summary.children) == ("mean", "variance")
        np.testing.assert_allclose(float(summary["variance/mu"]), 4.0, **TOLERANCE)
        np.testing.assert_allclose(float(summary["variance/tau"]), 2.0, **TOLERANCE)

    def test_a_function_is_named_by_its_name(self, model):
        def spread(d):
            return standard_deviation(d)

        assert tuple(summarize(model, spread).children) == ("spread",)

    def test_a_mapping_names_its_statistics(self, model):
        summary = summarize(model, {"spread": standard_deviation}, mean)

        assert tuple(summary.children) == ("spread", "mean")
        np.testing.assert_allclose(float(summary["spread/mu"]), 2.0, **TOLERANCE)

    def test_the_interval_applies_to_the_default_columns_alone(self, model):
        assert tuple(summarize(model, mean, interval=0.9).children) == ("mean",)

    def test_a_callable_without_a_name_raises(self, model):
        with pytest.raises(ValueError, match="Pass it in a mapping"):
            summarize(model, lambda d: mean(d))

    def test_two_statistics_of_one_name_raise(self, model):
        with pytest.raises(ValueError, match=r"got \['mean'\] more than once"):
            summarize(model, mean, {"mean": standard_deviation})

    def test_a_name_with_a_slash_raises(self, model):
        with pytest.raises(ValueError, match="without '/'"):
            summarize(model, {"a/b": mean})

    def test_a_statistic_that_is_not_callable_raises(self, model):
        with pytest.raises(TypeError, match="must be a callable or a mapping"):
            summarize(model, 3)
        with pytest.raises(TypeError, match="'x' must be callable"):
            summarize(model, {"x": 3})

    def test_a_statistic_that_returns_no_numeric_value_raises(self, model):
        with pytest.raises(TypeError, match="'law' must return a numeric value"):
            summarize(model, {"law": lambda d: d})

    def test_a_law_is_required(self):
        with pytest.raises(TypeError, match="d must be a Distribution"):
            summarize(np.zeros(3))


class TestAnMCMCPosterior:
    def test_it_gets_r_hat_and_the_effective_sample_sizes_by_path(self, chain_draws):
        summary = summarize(_posterior(chain_draws))

        assert tuple(summary.children) == (
            *("mean", "sd", "median", "q10", "q90"),
            *("r_hat", "ess_bulk", "ess_tail"),
        )
        for index, path in enumerate(("mu", "sigma")):
            draws = chain_draws[..., index].astype(np.float64)
            tail = azs.ess(xr.Dataset({"x": (("chain", "draw"), draws)}), method="tail")["x"]
            np.testing.assert_allclose(
                float(summary[f"r_hat/{path}"]), azs.rhat(draws), **ARVIZ_TOLERANCE
            )
            np.testing.assert_allclose(
                float(summary[f"ess_bulk/{path}"]), azs.ess(draws, method="bulk"), **ARVIZ_TOLERANCE
            )
            np.testing.assert_allclose(
                float(summary[f"ess_tail/{path}"]), float(tail), **ARVIZ_TOLERANCE
            )

    def test_its_statistics_are_those_of_its_draws(self, chain_draws):
        summary = summarize(_posterior(chain_draws))
        draws = chain_draws[..., 0].reshape(-1)

        np.testing.assert_allclose(float(summary["mean/mu"]), draws.mean(), rtol=1e-5)
        np.testing.assert_allclose(float(summary["sd/mu"]), draws.std(), rtol=1e-4)

    def test_a_nested_parameter_has_its_diagnostics_at_its_path(self, chain_draws):
        summary = summarize(_posterior(chain_draws, nested=True))

        assert summary.at_path("r_hat").keys() == ("population/mu", "population/sigma")
        np.testing.assert_allclose(
            float(summary["r_hat/population/sigma"]),
            azs.rhat(chain_draws[..., 1].astype(np.float64)),
            **ARVIZ_TOLERANCE,
        )
        rows = [line.split()[0] for line in repr(summary).splitlines()[2:-1]]
        assert rows == ["population/mu", "population/sigma"]

    def test_a_vector_parameter_has_a_diagnostic_per_element(self, chain_draws):
        summary = summarize(_posterior(chain_draws, vector=True))

        assert np.shape(summary["r_hat/beta"]) == (2,)
        np.testing.assert_allclose(
            np.asarray(summary["r_hat/beta"]),
            [azs.rhat(chain_draws[..., i].astype(np.float64)) for i in range(2)],
            **ARVIZ_TOLERANCE,
        )

    def test_one_chain_has_no_r_hat(self, chain_draws):
        summary = summarize(_posterior(chain_draws[:1]))

        assert np.isnan(float(summary["r_hat/mu"]))

    def test_the_posteriors_annotations_are_left_as_they_are(self, chain_draws):
        from probpipe.diagnostics._datatree_store import _mcmc_has_field

        posterior = _posterior(chain_draws)
        summarize(posterior, mcse=True)

        assert not _mcmc_has_field(posterior, "rhat")

    def test_mcse_gives_each_statistic_its_arviz_standard_error(self, chain_draws):
        summary = summarize(_posterior(chain_draws), mcse=True)
        draws = chain_draws[..., 1].astype(np.float64)

        assert "mcse_q90" in summary.children
        np.testing.assert_allclose(
            float(summary["mcse_mean/sigma"]), azs.mcse(draws, method="mean"), **ARVIZ_TOLERANCE
        )
        np.testing.assert_allclose(
            float(summary["mcse_sd/sigma"]), azs.mcse(draws, method="sd"), **ARVIZ_TOLERANCE
        )
        np.testing.assert_allclose(
            float(summary["mcse_median/sigma"]), azs.mcse(draws, method="median"), **ARVIZ_TOLERANCE
        )
        np.testing.assert_allclose(
            float(summary["mcse_q10/sigma"]),
            azs.mcse(draws, method="quantile", prob=0.1),
            **ARVIZ_TOLERANCE,
        )

    def test_a_statistic_arviz_has_no_standard_error_for_has_nan(self, chain_draws):
        summary = summarize(_posterior(chain_draws), variance, mcse=True)

        assert np.isnan(float(summary["mcse_variance/mu"]))


class TestMonteCarloStandardErrors:
    def test_an_exact_summary_has_errors_of_zero(self, model):
        summary = summarize(model, mcse=True)

        assert tuple(summary.children)[5:] == (
            "mcse_mean",
            "mcse_sd",
            "mcse_median",
            "mcse_q10",
            "mcse_q90",
        )
        for name in tuple(summary.children)[5:]:
            assert float(summary[f"{name}/mu"]) == float(summary[f"{name}/tau"]) == 0.0

    def test_a_monte_carlo_mean_has_sd_over_root_n_and_an_exact_path_zero(self):
        draws = mean.options["num_samples"]

        with workflow_run(seed=0):
            summary = summarize(_kernel_model(), mcse=True)

        np.testing.assert_allclose(
            np.asarray(summary["mcse_mean/y"]),
            np.asarray(summary["sd/y"]) / np.sqrt(draws),
            **FLOAT32_TOLERANCE,
        )
        assert float(summary["mcse_mean/mu"]) == 0.0

    def test_another_monte_carlo_statistic_has_nan_where_it_drew(self):
        with workflow_run(seed=0):
            summary = summarize(_kernel_model(), mcse=True)

        assert np.isnan(np.asarray(summary["mcse_q10/y"])).all()
        assert float(summary["mcse_q10/mu"]) == 0.0
        # The standard deviation's draws are not recorded, so no path is known exact.
        assert np.isnan(float(summary["mcse_sd/mu"]))


class TestTheFooter:
    def test_an_exact_summary_says_so(self, model):
        assert repr(summarize(model)).splitlines()[-1] == "mean, sd, median, q10, q90: exact."

    def test_monte_carlo_columns_state_their_draws(self):
        law = (Normal("mu", 0.0, 2.0) * Poisson("count", 4.0)).with_label("pair")
        draws = quantile.options["num_samples"]

        with workflow_run(seed=0):
            summary = summarize(law)

        lines = repr(summary).splitlines()
        assert lines[0] == "summary of pair(mu, count)"
        assert lines[-1] == f"mean, sd: exact. median, q10, q90: Monte Carlo, {draws:,} draws."

    def test_an_mcmc_posterior_states_its_chains_and_draws(self, chain_draws):
        summary = summarize(_posterior(chain_draws), mean, mcse=True)

        assert repr(summary).splitlines()[-1] == "mean: MCMC, 4 chains of 500 draws."

    def test_the_summarys_provenance_records_the_line(self, model):
        provenance = summarize(model, mean, {"sd": standard_deviation}).provenance

        assert provenance.metadata == {
            "func": "summarize",
            "computations": {"mean": "exact", "sd": "exact"},
        }


class TestABatchOfLaws:
    def test_it_gives_a_batch_of_summary_records(self):
        laws = DistributionBatch(
            [
                Normal("mu", 0.0, 1.0) * Gamma("tau", 2.0, 1.0),
                Normal("mu", 1.0, 2.0) * Gamma("tau", 3.0, 1.0),
            ],
            "law",
            label="laws",
        )

        summary = summarize(laws)

        assert isinstance(summary, NumericRecordBatch)
        assert summary.label == "summary of laws(mu, tau) over law"
        assert dict(zip(summary.level_names, summary.batch_shape, strict=True)) == {"law": 2}
        for index in range(2):
            element = summarize(laws[index])
            for path in element:
                np.testing.assert_allclose(
                    np.asarray(summary[index][path].raw),
                    np.asarray(element[path].raw),
                    **TOLERANCE,
                )
        assert repr(summary).splitlines()[1] == "  law=0"

    def test_mcse_holds_for_laws_computed_differently(self):
        # The median of the Normal count is exact, and of the Poisson count Monte Carlo.
        laws = DistributionBatch(
            [
                Normal("mu", 0.0, 1.0) * Normal("count", 4.0, 1.0),
                Normal("mu", 0.0, 1.0) * Poisson("count", 4.0),
            ],
            "law",
        )

        with workflow_run(seed=0):
            summary = summarize(laws, mcse=True)

        assert float(summary[0]["mcse_median/count"].raw) == 0.0
        assert np.isnan(float(summary[1]["mcse_median/count"].raw))

    def test_a_column_only_some_laws_give_is_nan_for_the_others(self, chain_draws):
        laws = DistributionBatch(
            [_posterior(chain_draws), Normal("mu", 0.0, 1.0) * Normal("sigma", 1.0, 1.0)], "law"
        )

        summary = summarize(laws, mean)

        assert np.isfinite(float(summary[0]["r_hat/mu"].raw))
        assert np.isnan(float(summary[1]["r_hat/mu"].raw))

    def test_a_statistic_that_differs_in_shape_across_the_laws_raises(self):
        def shifting(d):
            return mean(d) if float(mean(d)) == 0.0 else mean(Normal("x", jnp.zeros(2), 1.0))

        laws = DistributionBatch([Normal("x", 0.0, 1.0), Normal("x", 1.0, 1.0)], "law")

        with pytest.raises(ValueError, match="must give the column 'shifting' over the same"):
            summarize(laws, shifting)


class TestATableForPandas:
    def test_a_data_frame_of_the_raw_summary_is_the_table(self, model):
        pd = pytest.importorskip("pandas")
        summary = summarize(model)

        frame = pd.DataFrame(summary.raw).astype(float)

        assert list(frame.columns) == ["mean", "sd", "median", "q10", "q90"]
        assert list(frame.index) == ["mu", "tau"]
        np.testing.assert_allclose(frame.loc["tau", "sd"], np.sqrt(2.0), **TOLERANCE)


def test_the_summary_is_a_numeric_record(model):
    assert isinstance(summarize(model), NumericRecord)
