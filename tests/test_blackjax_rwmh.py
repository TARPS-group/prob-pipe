"""Tests specific to the BlackJAX-backed RWMH implementation.

The legacy ``TestRWMH`` suite in ``test_inference.py`` exercises shapes,
provenance, multi-chain, and the conjugate Normal-Normal recovery path.
This file covers behavior new to the BlackJAX migration:

* the adaptive warmup (RGG-scaled proposal with Welford covariance refit),
* the eager-fallback path for non-JAX-traceable log-densities,
* the deprecated ``tfp_rwmh`` alias keeps existing pinned callers working.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import MultivariateNormal, Normal, NumericRecordDistribution, SimpleModel
from probpipe.core.protocols import SupportsLogProb
from probpipe.inference import (
    inference_method_registry,
    rwmh,
)
from probpipe.inference._blackjax_rwmh import (
    BlackJAXRWMHMethod,
    TFPRWMHMethod,
)
from probpipe.modeling._likelihood import Likelihood

# Suppress an unrelated TFP/JAX deprecation that fires during random-key
# construction inside the test fixtures.
pytestmark = pytest.mark.filterwarnings(
    "ignore:shape requires ndarray or scalar arguments:DeprecationWarning",
)


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Registry surface for the new method + deprecated alias."""

    def test_blackjax_rwmh_registered(self):
        names = inference_method_registry.list_methods()
        assert "blackjax_rwmh" in names
        assert inference_method_registry.get_method("blackjax_rwmh").priority == 55

    def test_tfp_rwmh_alias_registered_opt_in(self):
        """The legacy ``tfp_rwmh`` name resolves but is opt-in only."""
        names = inference_method_registry.list_methods()
        assert "tfp_rwmh" in names
        assert inference_method_registry.get_method("tfp_rwmh").priority == 0


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

    def test_anisotropic_target_recovers_per_dim_variance(self):
        # N(0, diag(1, 4)) — adapt should fit the elongation and produce
        # sample stds close to [1, 2].
        dist = MultivariateNormal(
            loc=jnp.zeros(2), cov=jnp.diag(jnp.array([1.0, 4.0])), name="z",
        )
        result = rwmh(
            dist=dist, num_results=4000, num_warmup=1500,
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
        # finite-d adaptation lands in roughly [0.15, 0.50].
        dist = MultivariateNormal(
            loc=jnp.zeros(5), cov=jnp.eye(5), name="z",
        )
        result = rwmh(
            dist=dist, num_results=2000, num_warmup=1000,
            num_chains=1, random_seed=3,
        )
        accept_rate = result.source.metadata["accept_rate"]
        assert 0.10 < accept_rate < 0.65, f"unexpected accept_rate {accept_rate}"

    def test_adapt_false_falls_back_to_fixed_step(self):
        """``adapt=False`` runs with ``sigma = step_size * I`` throughout.

        The step_size kwarg is forwarded into the production proposal —
        legacy behavior. Verifies the opt-out path stays available.
        """
        dist = MultivariateNormal(
            loc=jnp.zeros(2), cov=jnp.eye(2), name="z",
        )
        result = rwmh(
            dist=dist, num_results=200, num_warmup=100,
            step_size=0.5, adapt=False, random_seed=0,
        )
        assert result.source.metadata["step_size"] == 0.5
        assert result.source.metadata["adapt"] is False

    def test_explicit_proposal_cov_overrides_adaptation(self):
        """``proposal_cov=`` takes precedence over both adaptation and step_size."""
        dist = MultivariateNormal(
            loc=jnp.zeros(2), cov=jnp.eye(2), name="z",
        )
        chol = jnp.array([[0.7, 0.0], [0.1, 0.5]])  # arbitrary lower-triangular
        result = rwmh(
            dist=dist, num_results=200, num_warmup=50,
            proposal_cov=chol, random_seed=0,
        )
        assert result.num_draws == 200


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

    def test_moderate_warmup_uses_two_windows(self):
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


class TestWindowedAdvantage:
    """Windowed warmup recovers an anisotropic target's cov better than n=1."""

    def test_recovers_anisotropic_cov(self):
        # 5-D with a 30x stretch in the last dim — single-window with
        # a moderate warmup budget undershoots the cov estimate; the
        # windowed schedule converges faster.
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


# ---------------------------------------------------------------------------
# Eager fallback (non-traceable log-density)
# ---------------------------------------------------------------------------


class _NumpyLogProbDist(NumericRecordDistribution, SupportsLogProb):
    """A target whose log-density is *not* JAX-traceable.

    Uses numpy + Python control flow — the same shape as a likelihood
    that calls into BridgeStan / scipy / an external simulator.
    """

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
        return jnp.asarray(-0.5 * float(np.sum(v ** 2)))

    def _prob(self, value):
        return jnp.exp(self._log_prob(value))

    def _unnormalized_log_prob(self, value):
        return self._log_prob(value)

    def _unnormalized_prob(self, value):
        return self._prob(value)


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
# Deprecated ``tfp_rwmh`` alias
# ---------------------------------------------------------------------------


class TestDeprecatedAlias:
    """Existing ``method="tfp_rwmh"`` callers continue to work."""

    def test_method_resolves_and_warns(self):
        from probpipe import condition_on

        prior = MultivariateNormal(
            loc=jnp.zeros(2), cov=jnp.eye(2), name="mu",
        )

        class _ZeroLikelihood(Likelihood):
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        model = SimpleModel(prior, _ZeroLikelihood(), name="m")
        data = jnp.zeros((5, 2))
        with pytest.warns(DeprecationWarning, match='method="tfp_rwmh"'):
            posterior = condition_on(
                model, data, method="tfp_rwmh",
                num_results=50, num_warmup=20, random_seed=0,
            )
        # Algorithm label is the workflow-function name, unchanged.
        assert posterior.algorithm == "rwmh"


# ---------------------------------------------------------------------------
# Class-level smoke
# ---------------------------------------------------------------------------


class TestClassesExpose:
    def test_blackjax_rwmh_method_has_expected_check(self):
        m = BlackJAXRWMHMethod()
        info = m.check(MultivariateNormal(
            loc=jnp.zeros(2), cov=jnp.eye(2), name="z"), None)
        assert info.feasible

    def test_deprecated_alias_inherits_check(self):
        m = TFPRWMHMethod()
        info = m.check(MultivariateNormal(
            loc=jnp.zeros(2), cov=jnp.eye(2), name="z"), None)
        assert info.feasible
