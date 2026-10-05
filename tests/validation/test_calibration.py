"""Simulation-based calibration and interval coverage.

Tolerances are measured (independent baselines / known calibration properties)
per STYLE_GUIDE §8.6.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from probpipe import (
    EmpiricalDistribution,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    OutputSpec,
    RecordSpec,
    conditional_distribution,
)
from probpipe.core.record import Record
from probpipe.distributions import ConditionalDistribution
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
)
from probpipe.families import GaussianFamily, glm_likelihood
from probpipe.validation import SBCResult, interval_coverage, simulation_based_calibration
from probpipe.validation._calibration import (
    _component_names,
    _flatten,
    _kolmogorov_sf,
    _ks_uniform,
    _ranks,
)

tfd = tfp.distributions

#: The number of observations of the conjugate normal model, and the precision
#: of its posterior under the prior N(0, 2²) and unit observation noise.
_N = 5
_PRECISION = 1 / 4 + _N


class _BiasedMeanKernel(
    ConditionalDistribution, SupportsConditionalSampling, SupportsConditionalLogProb
):
    """A deliberately miscalibrated Gaussian-mean likelihood of ``n`` observations.

    Its density is ``N(y; μ, 1)``, but it samples from ``N(μ + bias, 1)``. Data
    drawn from the joint therefore carry an unmodeled shift, which makes the
    posterior systematically miss ``θ★``, so SBC ranks are non-uniform. Used to
    test that SBC *detects* miscalibration.
    """

    def __init__(self, bias: float, n: int = 10):
        super().__init__("y", {"mu": NumericArraySpec(())}, OutputSpec(y=NumericArraySpec((n,))))
        object.__setattr__(self, "_bias", float(bias))
        object.__setattr__(self, "_n", n)

    @staticmethod
    def _mu(given):
        return jnp.asarray(dict(given.children if isinstance(given, Record) else given)["mu"])

    def _condition_on(self, given, /, **options):
        # The law the kernel draws from carries the shift; its density does not.
        return Normal("y", jnp.full(self._n, self._mu(given) + self._bias), 1.0)

    def _conditional_log_prob(self, given, value):
        return jnp.sum(tfd.Normal(self._mu(given), 1.0).log_prob(jnp.asarray(value)))

    def _conditional_sample(self, given, key, sample_shape=()):
        shift = self._mu(given) + self._bias
        return shift + jax.random.normal(key, (*sample_shape, self._n))


class _GridPosterior(ConditionalDistribution, SupportsConditionalSampling):
    """The conjugate model's posterior as atoms on a grid, weighted by the posterior density.

    The atoms span the prior's range at a spacing of 0.005, so their weighted law
    is the posterior up to the grid's resolution. With ``weighted=False`` the
    atoms are equally weighted, which gives the uniform law on the grid.
    """

    def __init__(self, *, weighted: bool = True):
        super().__init__(
            "posterior", {"y": NumericArraySpec((_N,))}, OutputSpec(mu=NumericArraySpec(()))
        )
        object.__setattr__(self, "_weighted", weighted)

    def _law(self, given):
        y = jnp.asarray(dict(given.children if isinstance(given, Record) else given)["y"])
        grid = jnp.linspace(-10.0, 10.0, 4001)
        if not self._weighted:
            return EmpiricalDistribution("mu", grid)
        log_density = -0.5 * _PRECISION * (grid - jnp.sum(y) / _PRECISION) ** 2
        return EmpiricalDistribution("mu", grid, jnp.exp(log_density - jnp.max(log_density)))

    def _condition_on(self, given, /, **options):
        return self._law(given)

    def _conditional_sample(self, given, key, sample_shape=()):
        return self._law(given)._sample(key, sample_shape)


def _gaussian_glm(p: int = 2, n: int = 12, seed: int = 7):
    """A well-specified Gaussian linear model with an intercept: the joint of y and beta."""
    X = jax.random.normal(jax.random.PRNGKey(seed), (n, p - 1))
    design = jnp.concatenate([jnp.ones((n, 1)), X], axis=1)
    prior = MultivariateNormal(loc=jnp.zeros(p), cov=jnp.eye(p), label="beta")
    return glm_likelihood("y", GaussianFamily(), X=design, dispersion=1.0) * prior


def _conjugate_model():
    """The joint of ``mu ~ N(0, 2²)`` and five observations ``y | mu ~ N(mu, 1)``."""
    prior = Normal("mu", 0.0, 2.0)
    likelihood = conditional_distribution(
        "y_given_mu",
        lambda mu: Normal("y", mu * jnp.ones(_N), 1.0),
        given_spec=prior.event_spec.components,
    )
    return likelihood * prior


def _posterior_at(y, scale_factor: float = 1.0):
    """The conjugate model's posterior at *y*, ``N(Σy / τ, τ^(-1/2))``, its scale times *scale_factor*."""
    return Normal("mu", jnp.sum(y) / _PRECISION, scale_factor / jnp.sqrt(_PRECISION))


def _exact_posterior(scale_factor: float = 1.0):
    """The kernel of :func:`_posterior_at`, whose one given slot is ``y``."""
    return conditional_distribution(
        "posterior",
        lambda y: _posterior_at(y, scale_factor),
        given_spec={"y": NumericArraySpec((_N,))},
    )


def _observation_posterior():
    """The kernel of :func:`_posterior_at`, whose one given slot is ``observation``."""
    return conditional_distribution(
        "posterior",
        lambda observation: _posterior_at(observation),
        given_spec={"observation": NumericArraySpec((_N,))},
    )


class TestIntervalCoverage:
    def test_contains_center_excludes_far(self):
        draws = jax.random.normal(jax.random.PRNGKey(0), (5000, 2))  # ~ N(0, I)
        cov = interval_coverage(draws, jnp.zeros(2), levels=(0.5, 0.9))
        for level in (0.5, 0.9):
            assert bool(jnp.all(cov[level]))  # truth = 0 is central → covered
        far = interval_coverage(draws, jnp.array([5.0, -5.0]), levels=(0.9,))
        assert not bool(jnp.any(far[0.9]))  # truth deep in the tails → not covered

    def test_frequentist_rate_matches_nominal(self):
        # truth and draws from the same N(0, 1): a fresh truth lands in the
        # central-`level` interval with probability ≈ level.
        draws = jax.random.normal(jax.random.PRNGKey(1), (4000, 1))
        truths = jax.random.normal(jax.random.PRNGKey(2), (400,))
        for level in (0.5, 0.9):
            hits = np.mean(
                [
                    bool(interval_coverage(draws, jnp.array([t]), levels=(level,))[level][0])
                    for t in truths
                ]
            )
            # 400 Bernoulli(level) draws → SE ≈ 0.015–0.025; abs=0.05 is ~2–3 SE.
            assert hits == pytest.approx(level, abs=0.05)

    def test_returns_per_level_per_param(self):
        draws = jax.random.normal(jax.random.PRNGKey(3), (1000, 3))
        cov = interval_coverage(draws, jnp.zeros(3), levels=(0.8, 0.95))
        assert set(cov) == {0.8, 0.95}
        assert cov[0.8].shape == (3,)

    def test_accepts_distribution_input(self):
        # An empirical law scores identically to its atoms' flat coordinates.
        draws = jax.random.normal(jax.random.PRNGKey(4), (2000, 2))
        emp = EmpiricalDistribution("z", draws)
        from_dist = interval_coverage(emp, jnp.array([0.3, -0.4]), levels=(0.9,))
        from_array = interval_coverage(draws, jnp.array([0.3, -0.4]), levels=(0.9,))
        assert bool(jnp.all(from_dist[0.9] == from_array[0.9]))

    def test_default_levels(self):
        draws = jax.random.normal(jax.random.PRNGKey(5), (1000, 2))
        assert set(interval_coverage(draws, jnp.zeros(2))) == {0.5, 0.8, 0.9, 0.95}

    def test_rejects_dimension_mismatch(self):
        draws = jax.random.normal(jax.random.PRNGKey(6), (500, 2))
        with pytest.raises(ValueError, match="dimension"):
            interval_coverage(draws, jnp.zeros(3))


class TestKSFunctions:
    def test_kolmogorov_sf_matches_scipy(self):
        # _kolmogorov_sf sums the Kolmogorov Q_KS series at the Stephens-corrected
        # λ; scipy.special.kolmogorov sums the same series — they must agree.
        sp = pytest.importorskip("scipy.special")
        for d, n in [(0.05, 50), (0.1, 100), (0.15, 200), (0.3, 30)]:
            lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
            assert _kolmogorov_sf(d, n) == pytest.approx(float(sp.kolmogorov(lam)), abs=1e-10)

    def test_kolmogorov_sf_boundaries(self):
        assert _kolmogorov_sf(0.0, 50) == 1.0  # zero distance → no evidence against H0
        assert _kolmogorov_sf(-1.0, 50) == 1.0  # guarded
        assert _kolmogorov_sf(5.0, 50) == pytest.approx(0.0, abs=1e-9)  # huge stat → ~0
        vals = [_kolmogorov_sf(d, 100) for d in (0.05, 0.1, 0.2, 0.3)]
        assert vals == sorted(vals, reverse=True)  # monotone decreasing in d

    def test_ks_statistic_matches_scipy(self):
        # The KS distance equals scipy's one-sample statistic against Uniform[0, 1].
        st = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        for big_l, s in [(100, 50), (200, 200), (50, 120)]:
            ranks = rng.integers(0, big_l + 1, size=(s, 1))
            u = (ranks[:, 0] + 0.5) / (big_l + 1)
            d = float(_ks_uniform(ranks, big_l)[0][0])
            assert d == pytest.approx(float(st.kstest(u, "uniform").statistic), abs=1e-10)

    def test_per_column_statistic_and_pvalue(self):
        # Each column is scored independently: D_j is its KS distance and p_j is
        # _kolmogorov_sf(D_j, num_simulations).
        st = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(1)
        ranks = rng.integers(0, 101, size=(80, 3))
        d, p = _ks_uniform(ranks, 100)
        assert d.shape == (3,) and p.shape == (3,)
        for j in range(3):
            u = (ranks[:, j] + 0.5) / 101
            assert float(d[j]) == pytest.approx(float(st.kstest(u, "uniform").statistic), abs=1e-10)
            assert float(p[j]) == pytest.approx(_kolmogorov_sf(float(d[j]), 80), abs=1e-12)

    def test_flags_uniform_vs_skewed(self):
        # End-to-end sanity: spread ranks are not rejected; bottom-decile ranks are.
        big_l, s = 100, 200
        uniform = np.linspace(0, big_l, s).astype(int)[:, None]
        assert float(_ks_uniform(uniform, big_l)[1][0]) > 0.05
        skewed = (np.arange(s) % (big_l // 10))[:, None]
        assert float(_ks_uniform(skewed, big_l)[1][0]) < 0.05


class TestRanks:
    def test_counts_draws_strictly_below(self):
        draws = jnp.array([[0.0], [1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(np.asarray(_ranks(draws, jnp.array([1.5]))), [2])
        # Strict < : a tie at the point is not counted (3 below 3.0, not 4).
        np.testing.assert_array_equal(np.asarray(_ranks(draws, jnp.array([3.0]))), [3])
        np.testing.assert_array_equal(np.asarray(_ranks(draws, jnp.array([-1.0]))), [0])
        np.testing.assert_array_equal(np.asarray(_ranks(draws, jnp.array([9.0]))), [4])

    def test_per_component(self):
        draws = jnp.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        # col 0: 2 draws below 1.5; col 1: 0 draws below 5.0.
        np.testing.assert_array_equal(np.asarray(_ranks(draws, jnp.array([1.5, 5.0]))), [2, 0])


class TestFlattening:
    """The multi-field θ★ flattening + component-naming that ranks rely on."""

    def test_flatten_honors_field_order(self):
        point = Record("r", a=jnp.array([1.0, 2.0]), b=jnp.array([3.0]))
        np.testing.assert_array_equal(
            np.asarray(_flatten(point, OutputSpec(RecordSpec(a=(2,), b=(1,))))),
            [1.0, 2.0, 3.0],
        )
        # The posterior's field order is authoritative (b before a).
        np.testing.assert_array_equal(
            np.asarray(_flatten(point, OutputSpec(RecordSpec(b=(1,), a=(2,))))),
            [3.0, 1.0, 2.0],
        )

    def test_flatten_ravels_an_array_draw(self):
        point = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        flat = _flatten(point, OutputSpec(theta=NumericArraySpec((2, 2))))
        np.testing.assert_array_equal(np.asarray(flat), [1.0, 2.0, 3.0, 4.0])

    def test_a_batch_of_draws_flattens_row_by_row(self):
        # A batch led by the draw axis gives one row per draw, each the flattening of its draw.
        spec = OutputSpec(RecordSpec(a=(2,), b=()))
        draws = {"a": jnp.arange(6.0).reshape(3, 2), "b": jnp.array([10.0, 11.0, 12.0])}
        rows = _flatten(draws, spec, batch_ndim=1)
        assert rows.shape == (3, 3)
        for i in range(3):
            row = _flatten({"a": draws["a"][i], "b": draws["b"][i]}, spec)
            np.testing.assert_array_equal(np.asarray(rows[i]), np.asarray(row))

    def test_component_names_expand_per_field(self):
        # A length-k field becomes field[0..k-1]; a scalar field keeps its name —
        # in posterior field order, matching the flat coordinates of its atoms.
        atoms = NumericRecordBatch(
            "r",
            {"a": jnp.zeros((10, 2)), "b": jnp.zeros((10,))},
            "atom",
            element_spec=NumericRecordSpec(a=(2,), b=()),
        )
        emp = EmpiricalDistribution("m", atoms)
        assert _component_names(emp) == ("a[0]", "a[1]", "b")

    def test_a_whole_term_posterior_names_its_component(self):
        emp = EmpiricalDistribution("m", jnp.zeros((10, 2)))
        assert _component_names(emp) == ("m[0]", "m[1]")


class TestCoverage:
    """``SBCResult.coverage``: the share of replications that cover ``θ★``, from the ranks."""

    @staticmethod
    def _calibrated(num_simulations: int = 400, num_draws: int = 99):
        """Truths and draws of the same N(0, I) in two coordinates, with their ``SBCResult``."""
        truth_key, draws_key = jax.random.split(jax.random.key(0))
        truths = jax.random.normal(truth_key, (num_simulations, 2))
        draws = jax.random.normal(draws_key, (num_simulations, num_draws, 2))
        ranks = np.stack([np.asarray(_ranks(draws[s], truths[s])) for s in range(num_simulations)])
        statistic, pvalue = _ks_uniform(ranks, num_draws)
        return truths, draws, SBCResult(ranks, num_draws, ("a", "b"), statistic, pvalue)

    def test_agrees_with_interval_coverage_on_the_same_draws(self):
        """The shares are the mean of ``interval_coverage`` over the replications' draws.

        The two differ only for a replication whose rank lies next to an end of
        the interval: there ``jnp.quantile`` puts the end between the two draws on
        either side of ``θ★``, at the rank ``⌊(L − 1) q⌋ + 1`` for the end's level ``q``.
        """
        truths, draws, result = self._calibrated()
        num_draws = result.num_posterior_draws
        shares = result.coverage((0.5, 0.9))
        for level in (0.5, 0.9):
            covered = np.stack(
                [
                    np.asarray(interval_coverage(draws[s], truths[s], levels=(level,))[level])
                    for s in range(truths.shape[0])
                ]
            )
            ends = [int((num_draws - 1) * q) + 1 for q in ((1 - level) / 2, (1 + level) / 2)]
            next_to_an_end = np.isin(result.ranks, ends).mean(axis=0)
            assert np.all(np.abs(shares[level] - covered.mean(axis=0)) <= next_to_an_end)

    def test_a_share_per_level_and_parameter(self):
        _, _, result = self._calibrated(num_simulations=50)
        shares = result.coverage((0.8, 0.95))
        assert set(shares) == {0.8, 0.95}
        for level, share in shares.items():
            assert isinstance(level, float)
            assert share.shape == (2,)
            assert np.all((share >= 0.0) & (share <= 1.0))
        assert set(result.coverage()) == {0.5, 0.8, 0.9, 0.95}

    def test_extreme_ranks_lie_outside_every_interval(self):
        # Rank 0 and rank L have normalized ranks 0.5 / (L + 1) and 1 − 0.5 / (L + 1).
        result = SBCResult(np.array([[0], [99], [50]]), 99, ("a",), np.zeros(1), np.ones(1))
        shares = result.coverage((0.5, 0.95))
        np.testing.assert_allclose(shares[0.5], [1 / 3])
        np.testing.assert_allclose(shares[0.95], [1 / 3])

    @pytest.mark.parametrize("levels", [(0.0,), (1.0,), (1.5,), (-0.1,), (float("nan"),)])
    def test_rejects_a_level_outside_the_unit_interval(self, levels):
        _, _, result = self._calibrated(num_simulations=10)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            result.coverage(levels)

    @pytest.mark.parametrize("levels", [0.9, "0.9", (True,), (None,)])
    def test_rejects_levels_that_are_not_a_sequence_of_numbers(self, levels):
        _, _, result = self._calibrated(num_simulations=10)
        with pytest.raises(TypeError, match="level"):
            result.coverage(levels)


class TestSBCPosteriorKernel:
    """Calibration of a posterior kernel, which each replication evaluates at its observed values."""

    def test_the_exact_posterior_gives_uniform_ranks_and_nominal_coverage(self):
        result = simulation_based_calibration(
            _conjugate_model(),
            observed="y",
            posterior=_exact_posterior(),
            num_simulations=200,
            num_posterior_draws=99,
            key=jax.random.key(0),
        )
        assert isinstance(result, SBCResult)
        assert result.ranks.shape == (200, 1)
        assert result.param_names == ("mu",)
        assert result.num_posterior_draws == 99
        assert result.ranks.min() >= 0 and result.ranks.max() <= 99
        # Measured at keys 0-3 (S=200, L=99): KS p-values 0.15-0.57, and coverage
        # 0.47-0.55 at level 0.5 and 0.905-0.92 at level 0.9.
        assert float(result.ks_pvalue[0]) > 0.01
        coverage = result.coverage((0.5, 0.9))
        np.testing.assert_allclose(coverage[0.5], [0.5], atol=0.1)
        np.testing.assert_allclose(coverage[0.9], [0.9], atol=0.06)

    def test_a_posterior_of_half_the_scale_fails_both(self):
        """Draws at half the posterior's scale reject uniformity and cover ``θ★`` too rarely.

        A central interval of half the width covers a draw of the posterior with
        probability ``P(|Z| ≤ z / 2)``: 0.26 at level 0.5 and 0.59 at level 0.9.
        """
        result = simulation_based_calibration(
            _conjugate_model(),
            observed="y",
            posterior=_exact_posterior(scale_factor=0.5),
            num_simulations=200,
            num_posterior_draws=99,
            key=jax.random.key(0),
        )
        # Measured at keys 0-3: KS p-values below 2e-5, and coverage 0.25-0.30 at
        # level 0.5 and 0.58-0.64 at level 0.9.
        assert float(result.ks_pvalue[0]) < 1e-3
        coverage = result.coverage((0.5, 0.9))
        assert float(coverage[0.5][0]) < 0.4
        assert float(coverage[0.9][0]) < 0.75

    def test_a_weighted_posterior_is_resampled_by_its_weights(self):
        """The ranks of a weighted empirical posterior are uniform, and those of its atoms unweighted are not.

        Equally weighted, the atoms are the uniform law on the grid, among whose
        draws ``θ★ ~ N(0, 2²)`` ranks near the middle.
        """
        common = {
            "observed": "y",
            "num_simulations": 200,
            "num_posterior_draws": 99,
            "key": jax.random.key(0),
        }
        weighted = simulation_based_calibration(
            _conjugate_model(), posterior=_GridPosterior(), **common
        )
        unweighted = simulation_based_calibration(
            _conjugate_model(), posterior=_GridPosterior(weighted=False), **common
        )
        # Measured at keys 0-3: KS p-values 0.20-0.90 weighted and below 1e-14 unweighted.
        assert float(weighted.ks_pvalue[0]) > 0.01
        assert float(unweighted.ks_pvalue[0]) < 1e-3

    def test_one_given_slot_takes_the_one_observed_field_whatever_its_name(self):
        common = {
            "observed": "y",
            "num_simulations": 8,
            "num_posterior_draws": 20,
            "key": jax.random.key(3),
        }
        by_name = simulation_based_calibration(
            _conjugate_model(), posterior=_exact_posterior(), **common
        )
        by_position = simulation_based_calibration(
            _conjugate_model(), posterior=_observation_posterior(), **common
        )
        np.testing.assert_array_equal(by_position.ranks, by_name.ranks)

    @pytest.mark.bayesflow
    def test_an_amortized_posterior_takes_the_observed_field_in_its_observation_slot(self):
        """A briefly trained amortized posterior calibrates, its ``observation`` slot taking ``y``.

        The training is too short to calibrate the network, so the test checks
        the ranks' shape and range, not their uniformity.
        """
        os.environ.setdefault("KERAS_BACKEND", "jax")
        pytest.importorskip("bayesflow")
        from probpipe import learn_amortized_posterior

        prior = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        simulator = conditional_distribution(
            "y_given_ab",
            lambda a, b: Normal("y", jnp.stack([a + b, a - b]), 0.1),
            given_spec=prior.event_spec.components,
        )
        amortized = learn_amortized_posterior(
            prior,
            simulator,
            method="npe",
            num_simulations=500,
            epochs=2,
            batch_size=128,
            random_seed=0,
            verbose=0,
        )
        assert list(amortized.given_spec) == ["observation"]
        result = simulation_based_calibration(
            simulator * prior,
            observed="y",
            posterior=amortized,
            num_simulations=6,
            num_posterior_draws=20,
            key=jax.random.key(0),
        )
        assert result.ranks.shape == (6, 2)
        assert result.param_names == ("a", "b")
        assert result.ranks.min() >= 0 and result.ranks.max() <= 20

    @pytest.mark.parametrize(
        "options", [{"method": "blackjax_nuts"}, {"method_options": {"num_warmup": 10}}]
    )
    def test_rejects_a_method_beside_a_posterior(self, options):
        with pytest.raises(ValueError, match="method"):
            simulation_based_calibration(
                _conjugate_model(),
                observed="y",
                posterior=_exact_posterior(),
                num_simulations=2,
                num_posterior_draws=10,
                **options,
            )

    def test_rejects_a_posterior_that_is_not_a_kernel(self):
        with pytest.raises(TypeError, match="ConditionalDistribution"):
            simulation_based_calibration(
                _conjugate_model(),
                observed="y",
                posterior=Normal("mu", 0.0, 1.0),
                num_simulations=2,
                num_posterior_draws=10,
            )

    def test_rejects_given_slots_that_do_not_take_the_observed_fields(self):
        two_slots = conditional_distribution(
            "posterior",
            lambda y, z: Normal("mu", jnp.sum(y) + z, 1.0),
            given_spec={"y": NumericArraySpec((_N,)), "z": NumericArraySpec(())},
        )
        with pytest.raises(ValueError, match=r"given slots \['y', 'z'\]"):
            simulation_based_calibration(
                _conjugate_model(),
                observed="y",
                posterior=two_slots,
                num_simulations=2,
                num_posterior_draws=10,
            )

    def test_one_given_slot_rejects_several_observed_fields(self):
        model = _conjugate_model() * Normal("w", 0.0, 1.0)
        with pytest.raises(ValueError, match=r"observed fields \['y', 'w'\]"):
            simulation_based_calibration(
                model,
                observed=("y", "w"),
                posterior=_observation_posterior(),
                num_simulations=2,
                num_posterior_draws=10,
            )

    def test_rejects_a_posterior_whose_draw_is_not_the_parameters(self):
        other_fields = conditional_distribution(
            "posterior",
            lambda y: Normal("a", jnp.sum(y), 1.0) * Normal("b", 0.0, 1.0),
            given_spec={"y": NumericArraySpec((_N,))},
        )
        with pytest.raises(ValueError, match="not the parameters"):
            simulation_based_calibration(
                _conjugate_model(),
                observed="y",
                posterior=other_fields,
                num_simulations=2,
                num_posterior_draws=10,
            )


class TestSBCFit:
    """Calibration of an inference method, which each replication runs at its observed values."""

    def test_well_specified_ranks_uniform(self):
        model = _gaussian_glm()
        res = simulation_based_calibration(
            model,
            observed="y",
            num_simulations=32,
            num_posterior_draws=100,
            method_options={"num_warmup": 100, "num_results": 100},
            key=jax.random.PRNGKey(0),
        )
        assert isinstance(res, SBCResult)
        assert res.ranks.shape == (32, 2)
        assert res.ranks.min() >= 0 and res.ranks.max() <= res.num_posterior_draws
        # param_names are per flattened component, aligned with the rank columns.
        assert res.param_names == ("beta[0]", "beta[1]")
        assert len(res.param_names) == res.ranks.shape[1]
        # Well-specified model + NUTS → ranks ~ uniform. Measured across seeds 0–3
        # (S=32, L=100 drawn from 4 chains of 100): mean normalized rank ∈
        # [0.42, 0.58], median ks_pvalue ∈ [0.29, 0.69]. Assert stable statistics,
        # not a tail bound on the min.
        u = (res.ranks + 0.5) / (res.num_posterior_draws + 1)
        assert np.all((u.mean(axis=0) > 0.35) & (u.mean(axis=0) < 0.65))
        assert float(np.median(res.ks_pvalue)) > 0.1
        # Rank histogram: (num_params, num_bins), each row sums to num_simulations.
        hist = res.rank_histogram(num_bins=10)
        assert hist.shape == (2, 10)
        assert np.all(hist.sum(axis=1) == 32)

    def test_detects_subtle_miscalibration(self):
        # A *sub-posterior-SD* mean shift is still caught. The kernel's draws carry an
        # unmodeled +0.25 shift while the posterior SD is ≈ 0.31 (precision
        # 1/4 + 10 = 10.25), so the per-fit bias is only ≈ 0.8 posterior SD — yet
        # SBC rejects because the shift is *systematic* across simulations and
        # accumulates. Measured ks_pvalue.max ≤ 0.002 and mean rank ≤ 0.36 over
        # seeds 0–3 at S=48.
        model = _BiasedMeanKernel(0.25) * Normal(loc=0.0, scale=2.0, label="mu")
        res = simulation_based_calibration(
            model,
            observed="y",
            num_simulations=48,
            num_posterior_draws=100,
            method_options={"num_warmup": 100, "num_results": 100},
            key=jax.random.PRNGKey(0),
        )
        # Uniformity is rejected for every parameter — a clear miscalibration signal.
        assert float(res.ks_pvalue.max()) < 0.05
        # The upward bias pushes θ★ into the lower tail → mean rank below 0.5.
        assert ((res.ranks + 0.5) / (res.num_posterior_draws + 1)).mean() < 0.45

    def test_a_pyabc_fit_takes_its_budget_in_method_options(self):
        pytest.importorskip("pyabc")
        result = simulation_based_calibration(
            _conjugate_model(),
            observed="y",
            method="pyabc_smcabc",
            method_options={"n_particles": 50, "max_populations": 2},
            num_simulations=3,
            num_posterior_draws=20,
            key=jax.random.key(1),
        )
        assert result.ranks.shape == (3, 1)
        assert result.ranks.min() >= 0 and result.ranks.max() <= 20

    def test_rejects_bad_num_simulations(self):
        model = _gaussian_glm()
        with pytest.raises(ValueError, match="num_simulations"):
            simulation_based_calibration(
                model, observed="y", num_simulations=0, num_posterior_draws=50
            )

    def test_a_method_budget_is_no_keyword_of_its_own(self):
        with pytest.raises(TypeError, match="num_warmup"):
            simulation_based_calibration(
                _gaussian_glm(),
                observed="y",
                num_simulations=2,
                num_posterior_draws=50,
                num_warmup=100,
            )

    def test_rejects_method_options_that_are_not_a_mapping(self):
        with pytest.raises(TypeError, match="method_options"):
            simulation_based_calibration(
                _gaussian_glm(),
                observed="y",
                num_simulations=2,
                num_posterior_draws=50,
                method_options=[("num_warmup", 100)],
            )

    def test_rejects_an_observed_name_that_is_no_field(self):
        with pytest.raises(ValueError, match="are not fields"):
            simulation_based_calibration(
                _gaussian_glm(), observed="z", num_simulations=2, num_posterior_draws=50
            )
