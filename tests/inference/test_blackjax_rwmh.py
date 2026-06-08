"""Tests specific to the BlackJAX-backed RWMH implementation.

Covers behavior beyond the generic ``TestRWMH`` suite in
``test_inference.py``:

* the adaptive warmup (RGG-scaled proposal with Welford covariance refit),
* the eager-fallback path for non-JAX-traceable log-densities,
* fast-vs-eager equivalence: both paths recover the same analytic target.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal, NumericRecordDistribution
from probpipe.core.protocols import SupportsLogProb
from probpipe.inference import (
    inference_method_registry,
    rwmh,
)
from probpipe.inference._blackjax_rwmh import (
    BlackJAXRWMHMethod,
    _production_sigma,
    _rgg_scale,
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
        loc=jnp.zeros(2), cov=jnp.diag(jnp.array([1.0, 4.0])), name="z",
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
        expected = (2.38 ** 2 / d) * np.asarray(cov)
        np.testing.assert_allclose(proposal_cov, expected, atol=1e-4)

    def test_rgg_scale_value(self):
        assert _rgg_scale(1) == pytest.approx(2.38)
        assert _rgg_scale(4) == pytest.approx(2.38 / 2.0)

    def test_jitter_is_scale_relative(self):
        """A large-variance target still gets a Cholesky close to the
        un-jittered factor — the scale-relative jitter doesn't dominate
        (an absolute 1e-8 jitter would be negligible here too, but a
        tiny-variance target is where scale-relative matters)."""
        # Tiny-variance covariance: absolute 1e-8 jitter would be ~comparable
        # to the signal; scale-relative jitter stays proportionate.
        cov = jnp.eye(2) * 1e-6
        sigma = np.asarray(_production_sigma(cov, 2))
        expected = np.linalg.cholesky(np.asarray(cov)) * (2.38 / np.sqrt(2))
        # Within 0.1% — jitter is 1e-8 * scale, far below the signal.
        np.testing.assert_allclose(sigma, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Adaptive warmup
# ---------------------------------------------------------------------------


class TestAdaptiveWarmup:
    """The default ``adapt=True`` warmup must recover near-RGG acceptance.

    Production proposal is ``chol(Sigma_hat) * 2.38 / sqrt(d)`` after
    Welford on the warmup positions. Acceptance shouldn't be perfectly
    on target (we don't dual-average), but should sit comfortably in
    the operating range — small enough that the chain isn't trivially
    rejecting, large enough that we aren't stuck.
    """

    def test_anisotropic_target_recovers_per_dim_variance(self, aniso_gaussian):
        # N(0, diag(1, 4)) — adapt should fit the elongation and produce
        # sample stds close to [1, 2].
        result = rwmh(
            dist=aniso_gaussian, num_results=4000, num_warmup=1500,
            num_chains=2, random_seed=7,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains], axis=0,
        )
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.15)
        np.testing.assert_allclose(
            draws.std(0, ddof=1), [1.0, 2.0], rtol=0.15,
        )

    def test_accept_rate_in_operating_range(self):
        # 5-D isotropic Gaussian. RGG asymptotic optimum is 0.234;
        # finite-d adaptation (no dual-averaging) lands in roughly
        # [0.10, 0.65] — small enough to mix, large enough not to stick.
        dist = MultivariateNormal(
            loc=jnp.zeros(5), cov=jnp.eye(5), name="z",
        )
        result = rwmh(
            dist=dist, num_results=2000, num_warmup=1000,
            num_chains=1, random_seed=3,
        )
        accept_rate = result.source.metadata["accept_rate"]
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
            dist=iso_gaussian, num_results=400, num_warmup=100,
            step_size=0.01, adapt=False, random_seed=0,
        )
        assert result.source.metadata["step_size"] == 0.01
        assert result.source.metadata["adapt"] is False
        # sigma = 0.01 * I → near-degenerate proposal → almost all accepted.
        assert result.source.metadata["accept_rate"] > 0.9

    def test_explicit_proposal_cov_overrides_adaptation(self, iso_gaussian):
        """``proposal_cov=`` takes precedence over both adaptation and step_size.

        Verified behaviorally: a deliberately tiny proposal Cholesky
        (near-degenerate) yields an almost-always-accepted chain. Were
        the supplied cov ignored in favour of the adaptive RGG fit, the
        accept rate would sit in the operating band rather than near one.
        """
        tiny_chol = jnp.eye(2) * 1e-2
        result = rwmh(
            dist=iso_gaussian, num_results=400, num_warmup=50,
            proposal_cov=tiny_chol, random_seed=0,
        )
        assert result.num_draws == 400
        assert result.source.metadata["accept_rate"] > 0.9

    def test_explicit_proposal_cov_huge_kills_acceptance(self, iso_gaussian):
        """The mirror case: a huge proposal cov drives acceptance toward zero,
        a second proof the supplied cov is actually used."""
        huge_chol = jnp.eye(2) * 50.0
        result = rwmh(
            dist=iso_gaussian, num_results=400, num_warmup=50,
            proposal_cov=huge_chol, random_seed=0,
        )
        assert result.source.metadata["accept_rate"] < 0.1


class TestNumWarmupZeroWarning:
    """``adapt=True`` with ``num_warmup=0`` cannot fit a proposal cov."""

    def test_warns_when_adapt_true_and_no_warmup(self, iso_gaussian):
        with pytest.warns(UserWarning, match="num_warmup=0"):
            rwmh(
                dist=iso_gaussian, num_results=50, num_warmup=0,
                adapt=True, random_seed=0,
            )

    def test_no_warning_when_adapt_false(self, iso_gaussian):
        """``adapt=False`` with ``num_warmup=0`` is the explicit fixed-step
        path — no fallback, so no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rwmh(
                dist=iso_gaussian, num_results=50, num_warmup=0,
                adapt=False, random_seed=0,
            )

    def test_no_warning_when_proposal_cov_given(self, iso_gaussian):
        """An explicit ``proposal_cov`` supplies the proposal directly, so
        ``num_warmup=0`` adapts nothing and must not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rwmh(
                dist=iso_gaussian, num_results=50, num_warmup=0,
                adapt=True, proposal_cov=jnp.eye(2) * 0.5, random_seed=0,
            )

    def test_bad_proposal_cov_shape_raises_valueerror(self, iso_gaussian):
        """A wrong-shape ``proposal_cov`` (here ``(3, 3)`` for a 2-D target)
        is validated up front: ``rwmh`` raises a ``ValueError`` naming the
        expected vs actual shape, rather than letting the mismatch surface
        as an opaque error from the BlackJAX kernel's matmul downstream.
        """
        with pytest.raises(ValueError, match=r"proposal_cov must be a square"):
            rwmh(
                dist=iso_gaussian, num_results=50, num_warmup=20,
                proposal_cov=jnp.eye(3), random_seed=0,
            )


class TestWindowSizing:
    """``_window_sizes`` clamps short warmups to a single phase."""

    def test_zero_warmup_returns_empty(self):
        from probpipe.inference._blackjax_rwmh import _window_sizes
        assert _window_sizes(0, n_windows=4) == []

    def test_short_warmup_collapses_to_single_window(self):
        """Under 50 warmup steps → single window (Stan's threshold).

        Avoids degenerate cov fits from too-few-sample windows.
        """
        from probpipe.inference._blackjax_rwmh import _window_sizes
        assert _window_sizes(20, n_windows=4) == [20]
        assert _window_sizes(40, n_windows=4) == [40]

    def test_single_window_boundary(self):
        """The 50-step threshold (= 2 * _MIN_STEPS_PER_WINDOW): 49 still
        collapses to one window, 50 is the first to split."""
        from probpipe.inference._blackjax_rwmh import _window_sizes
        # 49 // 25 == 1 → single window.
        assert _window_sizes(49, n_windows=4) == [49]
        # 50 // 25 == 2 → at least two windows.
        assert len(_window_sizes(50, n_windows=4)) >= 2

    def test_moderate_warmup_uses_three_windows(self):
        from probpipe.inference._blackjax_rwmh import _window_sizes
        # 80 steps / 25 = 3 max windows, clamped from 4 requested.
        sizes = _window_sizes(80, n_windows=4)
        assert len(sizes) == 3
        assert sum(sizes) == 80
        # Geometric — sizes should be non-decreasing.
        assert sizes == sorted(sizes)

    def test_long_warmup_uses_all_windows(self):
        from probpipe.inference._blackjax_rwmh import _window_sizes
        sizes = _window_sizes(1000, n_windows=4)
        assert len(sizes) == 4
        assert sum(sizes) == 1000
        # Last window is the largest under geometric (ratio=2) growth.
        assert sizes[-1] > sizes[0]


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
            loc=jnp.zeros(5), cov=jnp.diag(true_stds ** 2), name="z",
        )
        result = rwmh(
            dist=dist, num_results=4000, num_warmup=3000,
            num_chains=1, random_seed=11,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains], axis=0,
        )
        np.testing.assert_allclose(
            draws.std(0, ddof=1), np.asarray(true_stds),
            rtol=0.15,
        )

    def test_n_windows_one_collapses_and_recovers(self, aniso_gaussian):
        """``n_windows=1`` collapses to a single-phase warmup (fixed
        RGG-scaled identity proposal + one-shot Welford fit). It should
        still run end-to-end and recover the moderate ``N(0, diag(1, 4))``
        target's per-dim stds."""
        result = rwmh(
            dist=aniso_gaussian, num_results=4000, num_warmup=1500,
            num_chains=2, n_windows=1, random_seed=7,
        )
        assert result.num_draws == 4000
        assert result.source.metadata["n_windows"] == 1
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains], axis=0,
        )
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.2)
        np.testing.assert_allclose(
            draws.std(0, ddof=1), [1.0, 2.0], rtol=0.2,
        )


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
        return jnp.asarray(-0.5 * float(np.sum(prec * v ** 2)))

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


class TestEagerFallback:
    """The eager Python-loop path supports non-JAX-traceable targets."""

    def test_runs_end_to_end(self):
        dist = _NumpyLogProbDist(name="np_dist")
        result = rwmh(
            dist=dist, num_results=400, num_warmup=200,
            num_chains=2, random_seed=42,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains], axis=0,
        )
        # Standard normal target — sample mean ~ 0, sample sd ~ 1.
        np.testing.assert_allclose(draws.mean(0), [0.0, 0.0], atol=0.3)
        np.testing.assert_allclose(
            draws.std(0, ddof=1), [1.0, 1.0], rtol=0.3,
        )

    def test_accept_rate_positive(self):
        dist = _NumpyLogProbDist(name="np_dist")
        result = rwmh(
            dist=dist, num_results=400, num_warmup=200, random_seed=42,
        )
        assert result.source.metadata["accept_rate"] > 0.10


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
            aniso_gaussian._unnormalized_log_prob, jnp.zeros(2),
        )
        result = rwmh(
            dist=aniso_gaussian, num_results=4000, num_warmup=1500,
            num_chains=2, random_seed=7,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains], axis=0,
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
            dist=dist, num_results=600, num_warmup=300,
            num_chains=2, random_seed=7,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in result.chains], axis=0,
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
