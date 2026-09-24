"""Tests specific to the BlackJAX-backed RWMH implementation.

Covers behavior beyond the generic ``TestRWMH`` suite in
``test_inference.py``:

* the adaptive warmup (RGG-scaled proposal with Welford covariance refit),
* the refit guard that keeps the proposal from collapsing,
* the eager-fallback path for non-JAX-traceable log-densities,
* fast-vs-eager equivalence: both paths recover the same analytic target.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Distribution, MultivariateNormal, NumericRecordDistribution
from probpipe.core.protocols import SupportsLogProb
from probpipe.inference import (
    inference_method_registry,
    rwmh,
)
from probpipe.inference._blackjax_rwmh import (
    BlackJAXRWMHMethod,
    _initial_sigma,
    _production_sigma,
    _refit_proposal,
    _rgg_scale,
    _window_sizes,
)

# Suppress an unrelated TFP/JAX deprecation that fires during random-key
# construction inside the test fixtures.
pytestmark = pytest.mark.filterwarnings(
    "ignore:shape requires ndarray or scalar arguments:DeprecationWarning",
)


# ---------------------------------------------------------------------------
# Shared targets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iso_gaussian():
    """A 2-D isotropic standard normal ``N(0, I)`` — analytic stds [1, 1]."""
    return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="z")


@pytest.fixture(scope="module")
def aniso_gaussian():
    """A 2-D anisotropic ``N(0, diag(1, 4))`` — analytic stds [1, 2]."""
    return MultivariateNormal(
        loc=jnp.zeros(2),
        cov=jnp.diag(jnp.array([1.0, 4.0])),
        name="z",
    )


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Registry surface for ``blackjax_rwmh``."""

    def test_blackjax_rwmh_registered(self):
        names = inference_method_registry.list_methods()
        assert "blackjax_rwmh" in names
        assert inference_method_registry.get_method("blackjax_rwmh").priority == 55

    def test_tfp_rwmh_removed(self):
        """The hand-rolled-RWMH alias ``tfp_rwmh`` no longer exists."""
        names = inference_method_registry.list_methods()
        assert "tfp_rwmh" not in names


# ---------------------------------------------------------------------------
# RGG scaling — the unit-level math behind the adaptive proposal
# ---------------------------------------------------------------------------


class TestProductionSigma:
    """``_production_sigma`` produces ``chol(Σ̂) · 2.38 / √d`` (the
    Roberts-Gelman-Gilks optimal-scaling proposal Cholesky)."""

    def test_matches_chol_times_rgg_scale(self):
        # A non-trivial PD covariance with off-diagonal structure.
        cov = jnp.array([[4.0, 1.0], [1.0, 2.0]])
        d = 2
        sigma = _production_sigma(cov, d)
        # Reference: jitter is negligible relative to the matrix scale,
        # so chol(cov) · 2.38/√d to floating tolerance.
        expected = np.linalg.cholesky(np.asarray(cov)) * (2.38 / np.sqrt(d))
        np.testing.assert_allclose(np.asarray(sigma), expected, atol=1e-5)

    def test_proposal_covariance_is_rgg_scaled(self):
        """The *covariance* L Lᵀ of the proposal equals (2.38²/d)·Σ̂ —
        the RGG asymptotic optimum, since ``sigma`` is the Cholesky
        factor (not the covariance)."""
        cov = jnp.array([[4.0, 1.0], [1.0, 2.0]])
        d = 2
        sigma = np.asarray(_production_sigma(cov, d))
        proposal_cov = sigma @ sigma.T
        expected = (2.38**2 / d) * np.asarray(cov)
        np.testing.assert_allclose(proposal_cov, expected, atol=1e-4)

    def test_rgg_scale_value(self):
        assert _rgg_scale(1) == pytest.approx(2.38)
        assert _rgg_scale(4) == pytest.approx(2.38 / 2.0)

    def test_non_positive_definite_input_gives_non_finite_factor(self):
        """A singular covariance yields non-finite entries rather than an
        exception; ``_refit_proposal`` relies on that to detect a failed
        factorization."""
        sigma = np.asarray(_production_sigma(jnp.zeros((2, 2)), 2))
        assert not np.isfinite(sigma).all()


# ---------------------------------------------------------------------------
# Refit guard — the proposal never collapses
# ---------------------------------------------------------------------------


class TestRefitProposal:
    """``_refit_proposal`` refits to
    ``(n * welford_cov + 5 * proposal_cov) / (n + 5)``, which shrinks the
    Welford covariance toward the proposal in use, so the refit proposal is
    positive definite even when the estimate is singular."""

    def test_matches_shrinkage_formula(self):
        welford_cov = jnp.array([[4.0, 1.0], [1.0, 2.0]])
        proposal_cov = jnp.array([[1.0, 0.5], [0.5, 3.0]])
        sigma = _production_sigma(proposal_cov, 2)
        cov, new_sigma = _refit_proposal(welford_cov, 45, proposal_cov, sigma)
        expected = (45 * np.asarray(welford_cov) + 5 * np.asarray(proposal_cov)) / 50
        np.testing.assert_allclose(np.asarray(cov), expected, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(new_sigma),
            np.linalg.cholesky(expected) * _rgg_scale(2),
            rtol=1e-5,
        )

    def test_zero_spread_shrinks_proposal_in_use(self):
        """A window that rejects every proposal leaves ``welford_cov == 0``;
        the refit is ``5 / (n + 5)`` times the proposal in use."""
        d, n = 2, 33
        cov, sigma = _refit_proposal(jnp.zeros((d, d)), n, jnp.eye(d), _initial_sigma(d))
        np.testing.assert_allclose(np.asarray(cov), 5 / (n + 5) * np.eye(d), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(sigma),
            np.sqrt(5 / (n + 5)) * _rgg_scale(d) * np.eye(d),
            rtol=1e-6,
        )

    def test_rank_deficient_estimate_gives_positive_definite_proposal(self):
        """Eight accepted moves in a 33-step window leave the positions in an
        eight-dimensional affine subspace of the 20-dimensional target; every
        singular value of the refit factor is still at least
        ``sqrt(5 / (n + 5))`` times the RGG scale."""
        d, n = 20, 33
        rng = np.random.default_rng(0)
        steps = np.zeros((n, d), dtype=np.float32)
        accepted = rng.choice(n, size=8, replace=False)
        steps[accepted] = 0.5 * rng.normal(size=(8, d))
        positions = np.cumsum(steps, axis=0)
        welford_cov = jnp.asarray(np.cov(positions, rowvar=False), dtype=jnp.float32)
        assert np.linalg.matrix_rank(np.asarray(welford_cov)) < d

        _, sigma = _refit_proposal(welford_cov, n, jnp.eye(d), _initial_sigma(d))
        singular_values = np.linalg.svd(np.asarray(sigma), compute_uv=False)
        assert np.isfinite(singular_values).all()
        assert singular_values.min() >= np.sqrt(5 / (n + 5)) * _rgg_scale(d) * (1 - 1e-4)

    def test_single_position_estimate_is_ignored(self):
        """With one position Welford reports ``0 / 0``; the refit treats the
        estimate as zero and shrinks the proposal in use."""
        d = 2
        nan_cov = jnp.full((d, d), jnp.nan)
        cov, sigma = _refit_proposal(nan_cov, 1, jnp.eye(d), _initial_sigma(d))
        np.testing.assert_allclose(np.asarray(cov), 5 / 6 * np.eye(d), rtol=1e-6)
        assert np.isfinite(np.asarray(sigma)).all()

    def test_non_finite_factor_keeps_proposal_in_use(self):
        """An estimate with an infinite variance gives a non-finite refit
        factor, so the proposal in use is returned unchanged."""
        bad_cov = jnp.array([[jnp.inf, 0.0], [0.0, 1.0]])
        proposal_cov = jnp.array([[2.0, 0.3], [0.3, 1.0]])
        sigma = _production_sigma(proposal_cov, 2)
        cov, new_sigma = _refit_proposal(bad_cov, 50, proposal_cov, sigma)
        np.testing.assert_array_equal(np.asarray(cov), np.asarray(proposal_cov))
        np.testing.assert_array_equal(np.asarray(new_sigma), np.asarray(sigma))

    def test_refit_is_scale_equivariant(self):
        """The shrinkage imposes no absolute scale: scaling both covariances
        by ``c`` scales the refit covariance by ``c`` and its factor by
        ``sqrt(c)``."""
        welford_cov = jnp.array([[4.0, 1.0], [1.0, 2.0]])
        base_cov, base_sigma = _refit_proposal(welford_cov, 40, jnp.eye(2), _initial_sigma(2))
        c = 1e-6
        cov, sigma = _refit_proposal(
            c * welford_cov, 40, c * jnp.eye(2), np.sqrt(c) * _initial_sigma(2)
        )
        np.testing.assert_allclose(np.asarray(cov), c * np.asarray(base_cov), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(sigma), np.sqrt(c) * np.asarray(base_sigma), rtol=1e-5
        )

    def test_jit_matches_eager(self):
        """The fast path calls the refit under tracing, with a traced
        ``count``; the jitted result equals the eager one."""
        welford_cov = jnp.array([[4.0, 1.0], [1.0, 2.0]])
        args = (welford_cov, jnp.asarray(33), jnp.eye(2), _initial_sigma(2))
        eager = _refit_proposal(*args)
        jitted = jax.jit(_refit_proposal)(*args)
        for e, j in zip(eager, jitted, strict=True):
            np.testing.assert_allclose(np.asarray(j), np.asarray(e), rtol=1e-6)


# ---------------------------------------------------------------------------
# Adaptive warmup
# ---------------------------------------------------------------------------


class TestAdaptiveWarmup:
    """The default ``adapt=True`` warmup must recover near-RGG acceptance.

    Production proposal is ``chol(Sigma) * 2.38 / sqrt(d)``, where
    ``Sigma`` is the regularized Welford refit on the warmup positions.
    Acceptance shouldn't be perfectly
    on target (we don't dual-average), but should sit comfortably in
    the operating range — small enough that the chain isn't trivially
    rejecting, large enough that we aren't stuck.
    """

    def test_anisotropic_target_recovers_per_dim_variance(self, aniso_gaussian):
        # N(0, diag(1, 4)) — adapt should fit the elongation and produce
        # sample stds close to [1, 2].
        result = rwmh(
            dist=aniso_gaussian,
            num_results=4000,
            num_warmup=1500,
            num_chains=2,
            random_seed=7,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains],
            axis=0,
        )
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.15)
        np.testing.assert_allclose(
            draws.std(0, ddof=1),
            [1.0, 2.0],
            rtol=0.15,
        )

    def test_accept_rate_in_operating_range(self):
        # 5-D isotropic Gaussian. RGG asymptotic optimum is 0.234;
        # finite-d adaptation (no dual-averaging) lands in roughly
        # [0.10, 0.65] — small enough to mix, large enough not to stick.
        dist = MultivariateNormal(
            loc=jnp.zeros(5),
            cov=jnp.eye(5),
            name="z",
        )
        result = rwmh(
            dist=dist,
            num_results=2000,
            num_warmup=1000,
            num_chains=1,
            random_seed=3,
        )
        accept_rate = result.provenance.metadata["accept_rate"]
        assert 0.10 < accept_rate < 0.65, f"unexpected accept_rate {accept_rate}"

    def test_adapt_false_falls_back_to_fixed_step(self, iso_gaussian):
        """``adapt=False`` runs with ``sigma = step_size * I`` throughout.

        Verified behaviorally, not just via metadata: a tiny fixed
        ``step_size`` makes every local proposal almost surely accepted
        on a smooth target, so the accept rate must be very high. If
        adaptation had instead kicked in, the RGG-scaled proposal would
        land the accept rate down in the operating band (~0.2-0.5), not
        near one.
        """
        result = rwmh(
            dist=iso_gaussian,
            num_results=400,
            num_warmup=100,
            step_size=0.01,
            adapt=False,
            random_seed=0,
        )
        assert result.provenance.metadata["step_size"] == 0.01
        assert result.provenance.metadata["adapt"] is False
        # sigma = 0.01 * I → near-degenerate proposal → almost all accepted.
        assert result.provenance.metadata["accept_rate"] > 0.9

    def test_explicit_proposal_cov_overrides_adaptation(self, iso_gaussian):
        """``proposal_cov=`` takes precedence over both adaptation and step_size.

        Verified behaviorally: a deliberately tiny proposal Cholesky
        (near-degenerate) yields an almost-always-accepted chain. Were
        the supplied cov ignored in favour of the adaptive RGG fit, the
        accept rate would sit in the operating band rather than near one.
        """
        tiny_chol = jnp.eye(2) * 1e-2
        result = rwmh(
            dist=iso_gaussian,
            num_results=400,
            num_warmup=50,
            proposal_cov=tiny_chol,
            random_seed=0,
        )
        assert result.num_draws == 400
        assert result.provenance.metadata["accept_rate"] > 0.9

    def test_explicit_proposal_cov_huge_kills_acceptance(self, iso_gaussian):
        """The mirror case: a huge proposal cov drives acceptance toward zero,
        a second proof the supplied cov is actually used."""
        huge_chol = jnp.eye(2) * 50.0
        result = rwmh(
            dist=iso_gaussian,
            num_results=400,
            num_warmup=50,
            proposal_cov=huge_chol,
            random_seed=0,
        )
        assert result.provenance.metadata["accept_rate"] < 0.1


class TestNumWarmupZeroWarning:
    """``adapt=True`` with ``num_warmup=0`` cannot fit a proposal cov."""

    def test_warns_when_adapt_true_and_no_warmup(self, iso_gaussian):
        with pytest.warns(UserWarning, match="num_warmup=0"):
            rwmh(
                dist=iso_gaussian,
                num_results=50,
                num_warmup=0,
                adapt=True,
                random_seed=0,
            )

    def test_no_warning_when_adapt_false(self, iso_gaussian):
        """``adapt=False`` with ``num_warmup=0`` is the explicit fixed-step
        path — no fallback, so no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rwmh(
                dist=iso_gaussian,
                num_results=50,
                num_warmup=0,
                adapt=False,
                random_seed=0,
            )

    def test_no_warning_when_proposal_cov_given(self, iso_gaussian):
        """An explicit ``proposal_cov`` supplies the proposal directly, so
        ``num_warmup=0`` adapts nothing and must not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rwmh(
                dist=iso_gaussian,
                num_results=50,
                num_warmup=0,
                adapt=True,
                proposal_cov=jnp.eye(2) * 0.5,
                random_seed=0,
            )

    def test_step_size_sets_proposal_without_warmup(self, iso_gaussian):
        """With no warmup positions to adapt on, ``adapt=True`` samples with
        ``sigma = step_size * I``, so a tiny ``step_size`` is almost always
        accepted, whereas an adapted proposal accepts about 0.3 to 0.5."""
        # Observed across seeds 0-3: accept 0.99-1.0.
        with pytest.warns(UserWarning, match="num_warmup=0"):
            result = rwmh(
                dist=iso_gaussian,
                num_results=400,
                num_warmup=0,
                step_size=0.01,
                adapt=True,
                random_seed=0,
            )
        assert result.provenance.metadata["accept_rate"] > 0.9

    def test_target_without_density_raises_typeerror(self):
        """A target that does not implement ``SupportsUnnormalizedLogProb``
        is rejected before any sampling."""

        class NoDensityDist(Distribution):
            event_shape = (2,)

            def __init__(self):
                super().__init__(name="no_density")

        with pytest.raises(TypeError, match="SupportsUnnormalizedLogProb"):
            rwmh(dist=NoDensityDist(), num_results=10, num_warmup=10, random_seed=0)

    def test_bad_proposal_cov_shape_raises_valueerror(self, iso_gaussian):
        """A wrong-shape ``proposal_cov`` (here ``(3, 3)`` for a 2-D target)
        is validated up front: ``rwmh`` raises a ``ValueError`` naming the
        expected vs actual shape, rather than letting the mismatch surface
        as an opaque error from the BlackJAX kernel's matmul downstream.
        """
        with pytest.raises(ValueError, match=r"proposal_cov must be a square"):
            rwmh(
                dist=iso_gaussian,
                num_results=50,
                num_warmup=20,
                proposal_cov=jnp.eye(3),
                random_seed=0,
            )


class TestWindowSizing:
    """``_window_sizes`` splits a warmup into geometric windows of at least
    25 steps, using fewer windows than requested when the warmup is short."""

    def test_zero_warmup_returns_empty(self):
        assert _window_sizes(0, n_windows=4) == []

    @pytest.mark.parametrize("n_windows", [1, 2, 3, 4, 6])
    def test_every_window_holds_min_steps(self, n_windows):
        """For every warmup length below 1500 steps, the window sizes:

        - sum to ``num_warmup``;
        - never decrease;
        - number at most ``n_windows``;
        - hold at least 25 steps each whenever there are two or more.
        """
        for num_warmup in range(1, 1500):
            sizes = _window_sizes(num_warmup, n_windows=n_windows)
            assert sum(sizes) == num_warmup
            assert 1 <= len(sizes) <= n_windows
            assert sizes == sorted(sizes)
            if len(sizes) > 1:
                assert min(sizes) >= 25, (num_warmup, sizes)

    def test_short_warmup_is_one_window(self):
        """A warmup too short for two windows of 25 steps runs as one window,
        even when that window is shorter than 25 steps."""
        assert _window_sizes(10, n_windows=4) == [10]
        assert _window_sizes(50, n_windows=4) == [50]
        assert _window_sizes(73, n_windows=4) == [73]

    @pytest.mark.parametrize("n_windows", [1, 0, -3])
    def test_n_windows_at_most_one_gives_single_window(self, n_windows):
        assert _window_sizes(500, n_windows=n_windows) == [500]

    def test_window_count_boundaries(self):
        """Each extra window starts at the first warmup length whose
        geometric split keeps every window at 25 steps or more."""
        assert _window_sizes(74, n_windows=4) == [25, 49]
        assert _window_sizes(171, n_windows=4) == [57, 114]
        assert _window_sizes(172, n_windows=4) == [25, 49, 98]
        assert _window_sizes(367, n_windows=4) == [52, 105, 210]
        assert _window_sizes(368, n_windows=4) == [25, 49, 98, 196]

    def test_window_count_reduced_below_request(self):
        """A 100-step warmup splits into four windows as ``[7, 13, 27, 53]``
        and into three as ``[14, 29, 57]``; both break the minimum, so it
        uses two windows."""
        assert _window_sizes(100, n_windows=4) == [33, 67]

    def test_long_warmup_uses_all_windows(self):
        assert _window_sizes(1000, n_windows=4) == [67, 133, 267, 533]


class TestWindowedWarmup:
    """The (default) windowed warmup recovers a strongly-anisotropic target.

    The original framing of this suite claimed the windowed schedule
    beats ``n_windows=1`` on cov recovery. A direct per-seed Frobenius
    comparison was found to be too flaky to assert robustly: the
    single-window warmup occasionally lands a lucky covariance fit and
    wins, and even averaging the error over a panel of seeds flips on
    some panels. The *recovery* claim — that the default schedule fits
    a 30x-stretched dimension — is rock-solid across every seed tried,
    so the suite asserts only that. (See the maintainer notes / PR for
    the seed sweep behind this rescoping.)
    """

    def test_windowed_recovers_anisotropic_cov(self):
        # 5-D with a 30x stretch in the last dim.
        true_stds = jnp.array([1.0, 1.0, 1.0, 1.0, 30.0])
        dist = MultivariateNormal(
            loc=jnp.zeros(5),
            cov=jnp.diag(true_stds**2),
            name="z",
        )
        result = rwmh(
            dist=dist,
            num_results=4000,
            num_warmup=3000,
            num_chains=1,
            random_seed=11,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains],
            axis=0,
        )
        np.testing.assert_allclose(
            draws.std(0, ddof=1),
            np.asarray(true_stds),
            rtol=0.15,
        )

    def test_n_windows_one_collapses_and_recovers(self, aniso_gaussian):
        """``n_windows=1`` collapses to a single-phase warmup (fixed
        RGG-scaled identity proposal + one-shot Welford fit). It should
        still run end-to-end and recover the moderate ``N(0, diag(1, 4))``
        target's per-dim stds."""
        result = rwmh(
            dist=aniso_gaussian,
            num_results=4000,
            num_warmup=1500,
            num_chains=2,
            n_windows=1,
            random_seed=7,
        )
        assert result.num_draws == 4000
        assert result.provenance.metadata["n_windows"] == 1
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains],
            axis=0,
        )
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.2)
        np.testing.assert_allclose(
            draws.std(0, ddof=1),
            [1.0, 2.0],
            rtol=0.2,
        )


# ---------------------------------------------------------------------------
# End to end: every chain moves after adaptive warmup
# ---------------------------------------------------------------------------


def _assert_every_chain_moves(result, min_std):
    """Each chain accepts some but not all proposals and spreads in every
    coordinate.

    A collapsed proposal leaves its chain at a single position: a zero-scale
    proposal is always accepted and a NaN one never is, and either leaves
    the coordinates' standard deviations at zero.
    """
    is_accepted = np.asarray(result.inference_data["sample_stats"]["is_accepted"])
    for accept_rate, chain in zip(is_accepted.mean(axis=1), result.chains, strict=True):
        assert 0.1 < accept_rate < 0.9, f"accept rate {accept_rate}"
        stds = np.asarray(chain).std(0, ddof=1)
        assert stds.min() > min_std, f"per-coordinate std {stds}"


class TestProposalNeverCollapses:
    """Adaptive warmup leaves a proposal that moves every chain, however
    short the warmup or high the target dimension."""

    def test_short_warmup_moves_every_chain(self, iso_gaussian):
        # Eight chains run eight independent warmups on the vmap path.
        # Observed across seeds 0-5: per-chain accept 0.23-0.54, per-chain
        # min std 0.75.
        result = rwmh(
            dist=iso_gaussian,
            num_results=200,
            num_warmup=100,
            num_chains=8,
            random_seed=1,
        )
        _assert_every_chain_moves(result, min_std=0.35)

    @pytest.mark.parametrize("num_warmup", [1, 10])
    def test_warmup_below_min_window_moves_every_chain(self, iso_gaussian, num_warmup):
        """A single window shorter than 25 steps still refits to a moving
        proposal. That includes a one-step warmup, whose Welford covariance
        is ``0 / 0``."""
        # Observed across seeds 0-5: per-chain accept 0.27-0.59, per-chain
        # min std 0.73.
        result = rwmh(
            dist=iso_gaussian,
            num_results=200,
            num_warmup=num_warmup,
            num_chains=4,
            random_seed=1,
        )
        _assert_every_chain_moves(result, min_std=0.35)

    def test_rank_deficient_first_window_in_high_dimension(self):
        """At ``d = 20`` the first 33-step window accepts far fewer than 20
        proposals, so its covariance estimate is singular; the default warmup
        still moves every chain."""
        dist = MultivariateNormal(loc=jnp.zeros(20), cov=jnp.eye(20), name="z")
        # Observed across seeds 0-7: per-chain accept 0.42-0.48, per-chain
        # min std 0.42.
        result = rwmh(
            dist=dist,
            num_results=1000,
            num_warmup=500,
            num_chains=2,
            random_seed=0,
        )
        _assert_every_chain_moves(result, min_std=0.2)


# ---------------------------------------------------------------------------
# Eager fallback (non-traceable log-density)
# ---------------------------------------------------------------------------


class _NumpyLogProbDist(NumericRecordDistribution, SupportsLogProb):
    """A 2-D Gaussian whose log-density is *not* JAX-traceable.

    Uses numpy + Python control flow — the same shape as a likelihood
    that calls into BridgeStan / scipy / an external simulator. The
    density is ``-0.5 * sum(precision * v**2)`` with ``precision`` a
    class attribute, so a subclass picks any diagonal Gaussian while
    sharing the traceability-killing Python branch. The default is the
    standard normal (``precision = (1, 1)``).
    """

    # 1 / variance per coordinate. Standard normal by default.
    precision = (1.0, 1.0)

    @property
    def event_shape(self):
        return (2,)

    @property
    def dtypes(self):
        return self._per_field_dict(jnp.float32)

    def _log_prob(self, value):
        v = np.asarray(value)
        # Python control flow on a value branch — kills JAX traceability.
        if np.any(np.abs(v) > 50):
            return jnp.asarray(-np.inf)
        prec = np.asarray(self.precision)
        return jnp.asarray(-0.5 * float(np.sum(prec * v**2)))

    def _prob(self, value):
        return jnp.exp(self._log_prob(value))

    def _unnormalized_log_prob(self, value):
        return self._log_prob(value)

    def _unnormalized_prob(self, value):
        return self._prob(value)


class _NumpyAnisoLogProbDist(_NumpyLogProbDist):
    """Non-traceable ``N(0, diag(1, 4))`` — analytic stds [1, 2].

    The numpy mirror of the traceable ``aniso_gaussian`` fixture, used
    to run the *same* analytic target through both execution paths.
    """

    precision = (1.0, 0.25)  # 1 / var = (1/1, 1/4)


class _NumpyStdNormal10(_NumpyLogProbDist):
    """Non-traceable 10-D standard normal, which runs a target of more than
    two dimensions on the eager path."""

    precision = (1.0,) * 10

    @property
    def event_shape(self):
        return (10,)


class TestEagerFallback:
    """The eager Python-loop path supports non-JAX-traceable targets."""

    def test_short_warmup_moves_chain_in_ten_dimensions(self):
        """The eager warmup uses the same refit, so a 100-step warmup in ten
        dimensions leaves a proposal that moves the chain."""
        dist = _NumpyStdNormal10(name="np10")
        # Observed across seeds 0-3: accept 0.33-0.43, min std 0.58.
        result = rwmh(dist=dist, num_results=300, num_warmup=100, random_seed=0)
        assert result.event_shape == (10,)
        _assert_every_chain_moves(result, min_std=0.25)

    def test_runs_end_to_end(self):
        dist = _NumpyLogProbDist(name="np_dist")
        result = rwmh(
            dist=dist,
            num_results=400,
            num_warmup=200,
            num_chains=2,
            random_seed=42,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains],
            axis=0,
        )
        # Standard normal target — sample mean ~ 0, sample sd ~ 1.
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.3)
        np.testing.assert_allclose(
            draws.std(0, ddof=1),
            [1.0, 1.0],
            rtol=0.3,
        )

    def test_accept_rate_positive(self):
        dist = _NumpyLogProbDist(name="np_dist")
        result = rwmh(
            dist=dist,
            num_results=400,
            num_warmup=200,
            random_seed=42,
        )
        assert result.provenance.metadata["accept_rate"] > 0.10


# ---------------------------------------------------------------------------
# Fast-vs-eager equivalence + per-path determinism
# ---------------------------------------------------------------------------


# Analytic baseline for N(0, diag(1, 4)): per-coordinate stds.
_ANISO_STD = np.array([1.0, 2.0])


class TestFastEagerEquivalence:
    """Both execution paths must recover the *same* analytic target.

    The fast path (``lax.scan`` + ``vmap``, JAX-traceable target) and
    the eager Python-loop fallback (non-traceable target) split their
    RNG differently, so the two traces are *not* bit-identical. What
    must hold is that each path independently recovers the target it is
    sampling, and that each is deterministic for a fixed seed. We run
    the same ``N(0, diag(1, 4))`` math through a traceable
    ``MultivariateNormal`` (fast) and a numpy wrapper (eager) and check
    both against the analytic baseline ``stds = [1, 2]``.
    """

    def test_fast_path_recovers_aniso(self, aniso_gaussian):
        # Confirm we are exercising the fast path: the traceable target.
        from probpipe.inference._inference_utils import is_jax_traceable

        assert is_jax_traceable(
            aniso_gaussian._unnormalized_log_prob,
            jnp.zeros(2),
        )
        result = rwmh(
            dist=aniso_gaussian,
            num_results=4000,
            num_warmup=1500,
            num_chains=2,
            random_seed=7,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains],
            axis=0,
        )
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.2)
        np.testing.assert_allclose(draws.std(0, ddof=1), _ANISO_STD, rtol=0.15)

    def test_eager_path_recovers_aniso(self):
        # Confirm we are exercising the eager path: the non-traceable target.
        from probpipe.inference._inference_utils import is_jax_traceable

        dist = _NumpyAnisoLogProbDist(name="np_aniso")
        assert not is_jax_traceable(dist._unnormalized_log_prob, jnp.zeros(2))
        # Lighter counts than the fast path: the Python loop is ~100x
        # slower per step. Empirically (seed sweep 1/2/7) the worst-case
        # std error here is ~10% and the worst-case mean offset ~0.25,
        # so the bands below carry comfortable MC margin.
        result = rwmh(
            dist=dist,
            num_results=600,
            num_warmup=300,
            num_chains=2,
            random_seed=7,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains],
            axis=0,
        )
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.4)
        np.testing.assert_allclose(draws.std(0, ddof=1), _ANISO_STD, rtol=0.2)

    def test_fast_path_deterministic(self, aniso_gaussian):
        """Fast path: identical seed → bit-identical draws on a rerun."""
        kw = dict(num_results=500, num_warmup=200, num_chains=2, random_seed=11)
        a = rwmh(dist=aniso_gaussian, **kw)
        b = rwmh(dist=aniso_gaussian, **kw)
        da = np.concatenate([np.asarray(c) for c in a.chains], axis=0)
        db = np.concatenate([np.asarray(c) for c in b.chains], axis=0)
        np.testing.assert_array_equal(da, db)

    def test_eager_path_deterministic(self):
        """Eager path: identical seed → bit-identical draws on a rerun."""
        dist = _NumpyAnisoLogProbDist(name="np_aniso")
        kw = dict(num_results=150, num_warmup=80, num_chains=1, random_seed=5)
        a = rwmh(dist=dist, **kw)
        b = rwmh(dist=dist, **kw)
        da = np.concatenate([np.asarray(c) for c in a.chains], axis=0)
        db = np.concatenate([np.asarray(c) for c in b.chains], axis=0)
        np.testing.assert_array_equal(da, db)


# ---------------------------------------------------------------------------
# Class-level smoke
# ---------------------------------------------------------------------------


class TestClassesExpose:
    def test_blackjax_rwmh_method_has_expected_check(self, iso_gaussian):
        m = BlackJAXRWMHMethod()
        info = m.check(iso_gaussian, None)
        assert info.feasible
