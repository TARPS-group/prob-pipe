"""Tests for the predictive check of probpipe.validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from probpipe import (
    EmpiricalDistribution,
    Normal,
    NumericArraySpec,
    OutputSpec,
    Record,
    RecordSpec,
    conditional_distribution,
    predictive_check,
    sample,
    workflow_run,
)
from probpipe.core._numeric_record import NumericRecord
from probpipe.validation import predictive_check as pc_direct
from tests._posterior import posterior_of

N = 20  # observations per dataset


def _location_kernel(prior, n=N, scale=1.0):
    """The kernel ``y ~ Normal(mu, scale)``, iid over *n* observations, given *prior*'s ``mu``."""
    return conditional_distribution(
        lambda mu: Normal("y", mu * jnp.ones(n), scale),
        given_spec=prior.event_spec.components,
        label="y_given_mu",
    )


def sample_mean(data):
    return jnp.mean(data)


def sample_variance(data):
    return jnp.var(data, ddof=1)


def sample_max(data):
    return jnp.max(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prior():
    return Normal("mu", 0.0, 1.0)


@pytest.fixture
def likelihood(prior):
    return _location_kernel(prior)


@pytest.fixture
def observed_data():
    return jnp.asarray(np.random.default_rng(0).normal(0.5, 1.0, size=N), jnp.float32)


# ---------------------------------------------------------------------------
# The result and its annotations
# ---------------------------------------------------------------------------


class TestPredictiveCheck:
    """The result of one check and the record it appends to the law's annotations."""

    def test_prior_check_returns_replicated_statistics(self, prior, likelihood):
        with workflow_run(seed=0):
            result = predictive_check(likelihood, prior, sample_mean, num_replications=50)
        assert result["replicated_statistics"].num_atoms == 50
        # The law is labeled by the statistic, and the record's field view by its key.
        assert str(result.raw()["replicated_statistics"]) == "sample_mean(replicated_statistics)"
        assert result["test_fn_name"].value == "sample_mean"
        assert "observed_statistic" not in result
        assert "p_value" not in result

    def test_posterior_check_returns_p_value(self, prior, likelihood, observed_data):
        with workflow_run(seed=1):
            result = predictive_check(
                likelihood, prior, sample_mean, observed_data, num_replications=100
            )
        assert 0.0 <= float(result["p_value"]) <= 1.0
        assert float(result["observed_statistic"]) == pytest.approx(float(observed_data.mean()))
        assert result["replicated_statistics"].num_atoms == 100

    def test_several_statistics_give_one_result_each(self, prior, likelihood, observed_data):
        with workflow_run(seed=2):
            result = predictive_check(
                likelihood, prior, [sample_mean, sample_max], observed_data, num_replications=30
            )
        assert result["sample_mean/test_fn_name"].value == "sample_mean"
        assert result["sample_max/test_fn_name"].value == "sample_max"
        assert result.at_path("sample_max")["replicated_statistics"].num_atoms == 30
        assert float(result["sample_max/observed_statistic"]) == pytest.approx(
            float(observed_data.max())
        )

    def test_the_statistics_read_the_same_replications(self, prior, likelihood):
        def sample_sum(data):
            return jnp.sum(data)

        with workflow_run(seed=3):
            result = predictive_check(
                likelihood, prior, [sample_mean, sample_sum], num_replications=40
            )
        means = np.asarray(result["sample_mean/replicated_statistics"].atoms.values)
        sums = np.asarray(result["sample_sum/replicated_statistics"].atoms.values)
        np.testing.assert_allclose(sums, N * means, rtol=1e-5)

    def test_a_statistic_jax_cannot_trace_is_computed_in_a_loop(self, prior, likelihood):
        def host_mean(data):
            return float(np.mean(np.asarray(data)))

        with workflow_run(seed=4):
            result = predictive_check(
                likelihood, prior, [sample_mean, host_mean], num_replications=25
            )
        np.testing.assert_allclose(
            np.asarray(result["host_mean/replicated_statistics"].atoms.values),
            np.asarray(result["sample_mean/replicated_statistics"].atoms.values),
            rtol=1e-6,
        )

    def test_is_function(self):
        from probpipe import Function

        assert isinstance(predictive_check, Function)

    def test_importable_from_top_level(self):
        from probpipe import predictive_check as pc

        assert callable(pc)

    def test_importable_from_subpackage(self):
        assert callable(pc_direct)

    def test_results_attached_to_law_annotations(self, prior, likelihood):
        """predictive_check appends its result to the law's annotations."""
        assert prior.annotations is None or "predictive_check" not in prior.annotations

        with workflow_run(seed=10):
            predictive_check(likelihood, prior, sample_mean, num_replications=10)
        group = prior.annotations["predictive_check"]
        assert len(list(group.children)) == 1
        check_ds = group["check_0"].dataset
        assert check_ds["replicated_statistics"].dims == ("replication",)
        assert check_ds.attrs["test_fn_name"] == "sample_mean"

    def test_checks_accumulate_one_child_per_statistic(self, prior, likelihood, observed_data):
        with workflow_run(seed=20):
            predictive_check(likelihood, prior, sample_mean, observed_data, num_replications=10)
            predictive_check(
                likelihood,
                prior,
                [sample_variance, sample_max],
                observed_data,
                num_replications=10,
            )
        group = prior.annotations["predictive_check"]
        names = [group[f"check_{i}"].dataset.attrs["test_fn_name"] for i in range(3)]
        assert names == ["sample_mean", "sample_variance", "sample_max"]
        assert "p_value" in group["check_2"].dataset.attrs

    def test_a_failed_statistic_appends_no_check(self, prior, likelihood, observed_data):
        def failing(data):
            raise RuntimeError("statistic failed")

        with workflow_run(seed=22), pytest.raises(RuntimeError, match="statistic failed"):
            predictive_check(
                likelihood, prior, [sample_mean, failing], observed_data, num_replications=5
            )
        assert prior.annotations is None or "predictive_check" not in prior.annotations

    def test_frozen_distribution_skips_attachment_silently(self):
        """Distributions that disallow post-construction attribute
        writes (e.g., a custom subclass with ``__slots__`` that
        excludes ``_annotations``) cause the in-place attachment to
        fail silently. The caller still gets the result from
        the public return so the validation itself isn't lost.

        Exercises the ``except AttributeError`` arm of the
        ``except (AttributeError, TypeError)`` branch in
        ``_record_check_in_annotations``.
        """
        from probpipe.validation._predictive_check import (
            _record_check_in_annotations,
        )

        class _FrozenDist:
            """Slotted dummy: ``object.__setattr__`` for ``_annotations``
            raises ``AttributeError``."""

            __slots__ = ()

        frozen = _FrozenDist()
        stats_array = jnp.zeros(5)
        result = {"test_fn_name": "stub", "replicated_statistics": stats_array}
        _record_check_in_annotations(frozen, stats_array, result)
        assert not hasattr(frozen, "_annotations")

    def test_typeerror_during_attachment_skips_silently(self):
        """Companion to the slotted-dummy test above — exercises the
        ``except TypeError`` arm explicitly. A distribution that
        carries an ``_annotations`` property whose setter raises
        ``TypeError`` (e.g., an immutable wrapper that explicitly
        rejects writes with ``TypeError`` rather than the more usual
        ``AttributeError``) bypasses attachment silently.

        ``object.__setattr__`` honors data-descriptor protocol, so
        the property setter on the class runs even though the call
        site uses ``object.__setattr__`` to bypass any custom
        ``__setattr__``.
        """
        from probpipe.validation._predictive_check import (
            _record_check_in_annotations,
        )

        class _AuxTypeErrorDist:
            """``_annotations`` is a property whose setter raises
            ``TypeError`` — represents an immutable wrapper rejecting
            attachment with a typed error rather than ``AttributeError``."""

            @property
            def _annotations(self):
                return None

            @_annotations.setter
            def _annotations(self, value):
                raise TypeError("immutable: cannot set _annotations")

        dist = _AuxTypeErrorDist()
        stats_array = jnp.zeros(5)
        result = {"test_fn_name": "stub", "replicated_statistics": stats_array}
        # No raise — the ``except TypeError`` clause swallows it.
        _record_check_in_annotations(dist, stats_array, result)

    def test_xarray_importerror_skips_attachment_silently(self, monkeypatch):
        """If xarray is unavailable, auxiliary attachment is skipped."""
        import builtins

        from probpipe.validation._predictive_check import (
            _record_check_in_annotations,
        )

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "xarray":
                raise ImportError("xarray unavailable")
            return real_import(name, *args, **kwargs)

        class _Dist:
            pass

        dist = _Dist()
        stats_array = jnp.zeros(5)
        result = {"test_fn_name": "stub", "replicated_statistics": stats_array}
        monkeypatch.setattr(builtins, "__import__", fake_import)

        _record_check_in_annotations(dist, stats_array, result)

        assert not hasattr(dist, "_annotations")


# ---------------------------------------------------------------------------
# Calibration against the closed-form predictive p-values
# ---------------------------------------------------------------------------


def _exact_p_values(y, loc, scale):
    """The predictive p-value of each statistic when ``mu ~ Normal(loc, scale)``.

    Given ``mu``, the data are iid ``Normal(mu, 1)``. The sample mean is then
    ``Normal(loc, sqrt(scale**2 + 1/n))``, the sample variance satisfies
    ``(n - 1) S**2 ~ chi2(n - 1)`` whatever ``mu``, and the maximum's CDF at
    ``t`` is ``E[Phi(t - mu)**n]``, integrated over ``mu`` by quadrature.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    max_cdf, _ = integrate.quad(
        lambda mu: stats.norm.cdf(y.max() - mu) ** n * stats.norm.pdf(mu, loc, scale),
        loc - 12.0 * scale,
        loc + 12.0 * scale,
    )
    return {
        "sample_mean": stats.norm.sf(y.mean(), loc, np.sqrt(scale**2 + 1.0 / n)),
        "sample_variance": stats.chi2.sf((n - 1) * y.var(ddof=1), n - 1),
        "sample_max": 1.0 - max_cdf,
    }


def _datasets():
    """Three datasets of one shape: typical, overdispersed, and underdispersed."""
    z = np.random.default_rng(2026).standard_normal(N)
    return {
        "typical": 0.4 + z,
        "overdispersed": 0.4 + 2.0 * z,
        "underdispersed": 0.4 + 0.5 * z,
    }


class TestCalibration:
    """Monte Carlo p-values match the closed-form ones of a normal model with a normal prior.

    The model is ``mu ~ Normal(0, 1)`` and ``y_i | mu ~ Normal(mu, 1)``, so the
    posterior given ``y`` is ``Normal(sum(y) / (n + 1), 1 / sqrt(n + 1))``.
    """

    NUM_REPLICATIONS = 4000

    @pytest.mark.parametrize("dataset", ["typical", "overdispersed", "underdispersed"])
    @pytest.mark.parametrize("source", ["posterior", "empirical posterior", "prior"])
    def test_p_values_match_the_closed_form(self, dataset, source):
        y = _datasets()[dataset]
        if source == "prior":
            loc, scale = 0.0, 1.0
            law = Normal("mu", loc, scale)
        else:
            loc, scale = y.sum() / (N + 1), 1.0 / np.sqrt(N + 1)
            law = Normal("mu", loc, scale)
            if source == "empirical posterior":
                draws = np.random.default_rng(7).normal(loc, scale, size=20_000)
                law = EmpiricalDistribution(jnp.asarray(draws, jnp.float32), component="mu")
        with workflow_run(seed=11):
            result = predictive_check(
                _location_kernel(law),
                law,
                [sample_mean, sample_variance, sample_max],
                jnp.asarray(y, jnp.float32),
                num_replications=self.NUM_REPLICATIONS,
            )
        for name, exact in _exact_p_values(y, loc, scale).items():
            # Four Monte Carlo standard errors, plus the float32 rounding of the statistics.
            # Observed across eight seeds: the largest error is 0.62 of this tolerance.
            tolerance = 4.0 * np.sqrt(exact * (1.0 - exact) / self.NUM_REPLICATIONS) + 2e-3
            assert float(result[f"{name}/p_value"]) == pytest.approx(exact, abs=tolerance), name

    def test_the_closed_form_p_values_cover_the_unit_interval(self):
        """The datasets of the calibration test give p-values near 0, near 1, and between."""
        p_values = [
            p
            for y in _datasets().values()
            for p in _exact_p_values(y, y.sum() / (N + 1), 1.0 / np.sqrt(N + 1)).values()
        ]
        assert min(p_values) < 0.01
        assert max(p_values) > 0.99
        assert any(0.2 < p < 0.8 for p in p_values)

    def test_a_record_event_is_checked_as_its_fields(self, prior):
        """A kernel over the record ``(a, b)`` hands each replication on as its fields.

        With ``a ~ Normal(mu, 1)`` and ``b ~ Normal(mu, 2)`` iid over three
        entries, the statistic ``a - mean(b)`` is ``Normal(0, sqrt(1 + 4/3))``
        whatever ``mu``.
        """
        kernel = conditional_distribution(
            lambda mu: Normal("a", mu, 1.0) * Normal("b", mu * jnp.ones(3), 2.0),
            given_spec=prior.event_spec.components,
            label="ab",
        )

        def contrast(data):
            return data["a"] - jnp.mean(data["b"])

        observed = {"a": jnp.float32(1.5), "b": jnp.zeros(3, jnp.float32)}
        with workflow_run(seed=5):
            result = predictive_check(kernel, prior, contrast, observed, num_replications=4000)
        exact = stats.norm.sf(1.5, 0.0, np.sqrt(1.0 + 4.0 / 3.0))
        assert float(result["p_value"]) == pytest.approx(exact, abs=0.03)
        with workflow_run(seed=5):
            as_record = predictive_check(
                kernel, prior, contrast, Record("observed", **observed), num_replications=4000
            )
        assert float(as_record["p_value"]) == float(result["p_value"])


# ---------------------------------------------------------------------------
# The forms of the law and the observed data
# ---------------------------------------------------------------------------


def _posteriors_of(draws, spec):
    """A posterior over the record of one field holding *draws*, and the same as a whole term."""
    chains = [draws[: len(draws) // 2], draws[len(draws) // 2 :]]
    record = posterior_of(chains, event_spec=RecordSpec(mu=spec))
    whole = posterior_of(chains, event_spec=OutputSpec(mu=spec))
    return record, whole


class TestForms:
    """The law may be a record, a whole term, or stacked draws, and the data a mapping."""

    def test_a_record_posterior_and_a_whole_term_posterior_agree(self, observed_data):
        draws = jnp.asarray(np.random.default_rng(0).normal(size=200), jnp.float32)
        results = []
        for law in _posteriors_of(draws, NumericArraySpec((), jnp.float32)):
            with workflow_run(seed=0):
                results.append(
                    predictive_check(
                        _location_kernel(law), law, sample_mean, observed_data, num_replications=40
                    )
                )
        assert float(results[0]["p_value"]) == float(results[1]["p_value"])
        np.testing.assert_array_equal(
            results[0]["replicated_statistics"].atoms.values,
            results[1]["replicated_statistics"].atoms.values,
        )

    def test_a_numeric_record_of_draws_is_checked_as_its_rows(self, observed_data):
        draws = jnp.asarray(np.random.default_rng(0).normal(size=200), jnp.float32)
        record, _ = _posteriors_of(draws, NumericArraySpec((), jnp.float32))
        kernel = _location_kernel(record)
        rows = NumericRecord("posterior", mu=np.asarray(draws))
        with workflow_run(seed=0):
            result = predictive_check(kernel, rows, sample_mean, observed_data, num_replications=40)
        with workflow_run(seed=0):
            expected = predictive_check(
                kernel, record, sample_mean, observed_data, num_replications=40
            )
        assert float(result["p_value"]) == float(expected["p_value"])

    def test_an_empirical_law_draws_its_atoms(self):
        """Each replication's parameter is an atom of the empirical law."""
        atoms = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
        law = EmpiricalDistribution(jnp.asarray(atoms), component="mu")
        with workflow_run(seed=3):
            result = predictive_check(
                _location_kernel(law, n=10, scale=1e-4), law, sample_mean, num_replications=20
            )
        means = np.asarray(result["replicated_statistics"].atoms.values)
        nearest = np.abs(means[:, None] - atoms).argmin(axis=1)
        # The kernel's scale of 1e-4 puts each replication's mean within 1e-4 of its atom.
        np.testing.assert_allclose(means, atoms[nearest], rtol=0, atol=2e-4)
        assert np.unique(nearest).size > 1

    def test_the_data_may_map_the_kernel_components(self, prior, likelihood, observed_data):
        with workflow_run(seed=9):
            as_value = predictive_check(
                likelihood, prior, sample_max, observed_data, num_replications=30
            )
        with workflow_run(seed=9):
            as_mapping = predictive_check(
                likelihood, prior, sample_max, {"y": observed_data}, num_replications=30
            )
        assert float(as_mapping["observed_statistic"]) == float(as_value["observed_statistic"])
        assert float(as_mapping["p_value"]) == float(as_value["p_value"])

    @pytest.mark.parametrize("host", ["pandas", "xarray", "xarray in a mapping"])
    def test_data_in_a_pandas_or_xarray_host_is_checked_as_its_array(
        self, prior, likelihood, observed_data, host
    ):
        pd = pytest.importorskip("pandas")
        xr = pytest.importorskip("xarray")
        values = np.asarray(observed_data)
        hosted = {
            "pandas": pd.Series(values, name="y"),
            "xarray": xr.DataArray(values, dims="observation"),
            "xarray in a mapping": {"y": xr.DataArray(values, dims="observation")},
        }[host]
        with workflow_run(seed=9):
            as_array = predictive_check(
                likelihood, prior, sample_max, observed_data, num_replications=30
            )
        with workflow_run(seed=9):
            as_host = predictive_check(likelihood, prior, sample_max, hosted, num_replications=30)
        assert float(as_host["observed_statistic"]) == float(as_array["observed_statistic"])
        assert float(as_host["p_value"]) == float(as_array["p_value"])


# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------


def _draw_after(run_first) -> float:
    """The draw that follows *run_first* in a workflow scope seeded 7."""
    with workflow_run(seed=7):
        run_first()
        return float(sample.with_options(raw=True)(Normal("z", 0.0, 1.0)))


class TestRandomness:
    """The replications are one workflow-owned random event of the enclosing workflow scope."""

    @staticmethod
    def _replications(likelihood, prior):
        check = predictive_check(likelihood, prior, sample_mean, num_replications=20)
        return np.asarray(check["replicated_statistics"].atoms.values)

    def test_a_seed_reproduces_the_replications_and_another_seed_changes_them(
        self, prior, likelihood
    ):
        def replications(seed):
            with workflow_run(seed=seed):
                return self._replications(likelihood, prior)

        first = replications(3)
        np.testing.assert_array_equal(replications(3), first)
        assert not np.array_equal(replications(4), first)

    def test_calls_outside_every_scope_draw_fresh_replications(self, prior, likelihood):
        first = self._replications(likelihood, prior)
        assert not np.array_equal(self._replications(likelihood, prior), first)

    def test_a_call_claims_one_event_of_the_enclosing_scope(self, prior, likelihood):
        after_check = _draw_after(lambda: self._replications(likelihood, prior))
        assert after_check == _draw_after(lambda: sample(Normal("w", 0.0, 1.0)))
        assert after_check != _draw_after(lambda: None)

    def test_a_key_keyword_raises_type_error(self, prior, likelihood):
        with pytest.raises(TypeError, match="unexpected keyword argument 'key'"):
            predictive_check(likelihood, prior, sample_mean, key=jax.random.key(0))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestOptionalSlots:
    """The replications take an optional slot's default unless the law produces the slot."""

    @staticmethod
    def _scaled():
        return conditional_distribution(
            lambda mu, scale=2.0: Normal("y", mu * jnp.ones(200), scale),
            given_spec={"mu": NumericArraySpec(())},
            label="y_given_mu",
        )

    def test_the_replications_take_the_default(self):
        with workflow_run(seed=0):
            check = predictive_check(self._scaled(), Normal("mu", 0.0, 1e-3), sample_variance)
        replicated = np.asarray(check["replicated_statistics"].atoms)
        np.testing.assert_allclose(replicated.mean(), 4.0, rtol=0.05)

    def test_the_replications_take_the_slot_the_law_produces(self):
        law = Normal("mu", 0.0, 1e-3) * Normal("scale", 3.0, 1e-3)
        with workflow_run(seed=0):
            check = predictive_check(self._scaled(), law, sample_variance)
        replicated = np.asarray(check["replicated_statistics"].atoms)
        np.testing.assert_allclose(replicated.mean(), 9.0, rtol=0.05)

    def test_the_error_names_only_the_required_slots(self, prior, observed_data):
        kernel = conditional_distribution(
            lambda mu, sigma, scale=1.0: Normal("y", mu * jnp.ones(N), sigma * scale),
            given_spec={"mu": prior.event_spec.components["mu"], "sigma": NumericArraySpec(())},
            label="y_given_mu_sigma",
        )
        with pytest.raises(ValueError, match=r"does not produce \['sigma'\], which kernel"):
            predictive_check(kernel, prior, sample_mean, observed_data)


class TestErrors:
    def test_a_law_that_misses_a_given_slot_raises_naming_it(self, prior, observed_data):
        kernel = conditional_distribution(
            lambda mu, sigma: Normal("y", mu * jnp.ones(N), sigma),
            given_spec={"mu": prior.event_spec.components["mu"], "sigma": NumericArraySpec(())},
            label="y_given_mu_sigma",
        )
        with pytest.raises(ValueError, match=r"does not produce \['sigma'\]"):
            predictive_check(kernel, prior, sample_mean, observed_data)

    def test_a_simulator_in_place_of_the_kernel_raises(self, prior, observed_data):
        class _Simulator:
            def generate_data(self, params, num_observations, *, key=None):
                return jnp.zeros(num_observations)

        with pytest.raises(TypeError, match="ConditionalDistribution"):
            predictive_check(_Simulator(), prior, sample_mean, observed_data)

    def test_a_law_that_produces_the_observations_raises(self, prior, likelihood):
        with pytest.raises(ValueError, match="'y' is produced by both"):
            predictive_check(likelihood, likelihood * prior, sample_mean)

    @pytest.mark.parametrize(
        ("test_fns", "error", "match"),
        [
            ([], ValueError, "at least one"),
            ([sample_mean, sample_mean], ValueError, "unique names"),
            ([sample_mean, 3], TypeError, r"test_fns\[1\] must be callable"),
        ],
    )
    def test_the_statistics_are_checked(self, prior, likelihood, test_fns, error, match):
        with pytest.raises(error, match=match):
            predictive_check(likelihood, prior, test_fns)

    def test_num_replications_must_be_positive(self, prior, likelihood):
        with pytest.raises(ValueError, match="positive integer"):
            predictive_check(likelihood, prior, sample_mean, num_replications=0)
