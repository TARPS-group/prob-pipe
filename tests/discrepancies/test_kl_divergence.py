"""Tests for :func:`probpipe.kl_divergence` and the ``"kl"`` registry.

Covers method selection by priority, named overrides, Monte Carlo
accuracy against the closed form, elementwise broadcasting over a
``DistributionArray``, and catalog registration.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import probpipe as pp
from probpipe import Gamma, Normal, StudentT, kl_divergence
from probpipe.core._distribution_array import _make_distribution_array
from probpipe.discrepancies._kl_registry import kl_registry


def _selected(p, q) -> str:
    """Name of the method auto-dispatch would pick for ``(p, q)``."""
    return kl_registry.check(p, q).method_name


# ---------------------------------------------------------------------------
# Method selection by priority
# ---------------------------------------------------------------------------


class TestMethodSelection:
    def test_normal_pair_uses_closed_form(self):
        p = Normal(loc=0.0, scale=1.0, name="p")
        q = Normal(loc=1.0, scale=1.0, name="q")
        assert _selected(p, q) == "kl_normal_normal"
        # KL(N(0,1) || N(1,1)) = 0.5.
        np.testing.assert_allclose(float(kl_divergence(p, q)), 0.5, rtol=1e-5)

    def test_tfp_registered_pair_uses_tfp(self):
        # TFP has a closed-form KL for Gamma/Gamma but ProbPipe's
        # GaussianKL does not apply, so TFPKL (priority 70) wins.
        p = Gamma(concentration=2.0, rate=1.0, name="p")
        q = Gamma(concentration=3.0, rate=2.0, name="q")
        assert _selected(p, q) == "kl_tfp"
        expected = float(tfd.kl_divergence(p._tfp_dist, q._tfp_dist))
        np.testing.assert_allclose(float(kl_divergence(p, q)), expected, rtol=1e-5)

    def test_unregistered_pair_falls_through_to_mc(self):
        # TFP has no registered KL for StudentT/StudentT, so both exact
        # methods report infeasible and MCKL (priority 5) is selected.
        p = StudentT(df=3.0, loc=0.0, scale=1.0, name="p")
        q = StudentT(df=5.0, loc=0.0, scale=1.0, name="q")
        assert _selected(p, q) == "kl_mc"
        # A finite, non-negative estimate is produced.
        value = float(kl_divergence(p, q, random_seed=0, n_samples=20_000))
        assert np.isfinite(value)
        assert value >= -1e-3


# ---------------------------------------------------------------------------
# Named override (method=...)
# ---------------------------------------------------------------------------


class TestNamedOverride:
    def test_force_mc_over_available_closed_form(self):
        p = Normal(loc=0.0, scale=1.0, name="p")
        q = Normal(loc=1.0, scale=1.0, name="q")
        mc = float(kl_divergence(p, q, method="kl_mc", random_seed=0, n_samples=200_000))
        np.testing.assert_allclose(mc, 0.5, atol=2e-2)

    def test_force_tfp_on_unregistered_pair_raises(self):
        p = StudentT(df=3.0, loc=0.0, scale=1.0, name="p")
        q = StudentT(df=5.0, loc=0.0, scale=1.0, name="q")
        with pytest.raises(TypeError):
            kl_divergence(p, q, method="kl_tfp")

    def test_unknown_method_raises_keyerror(self):
        p = Normal(loc=0.0, scale=1.0, name="p")
        q = Normal(loc=0.0, scale=1.0, name="q")
        with pytest.raises(KeyError):
            kl_divergence(p, q, method="does_not_exist")


# ---------------------------------------------------------------------------
# Monte Carlo estimator behaviour
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_mc_agrees_with_closed_form(self):
        p = Normal(loc=0.5, scale=1.0, name="p")
        q = Normal(loc=-0.5, scale=2.0, name="q")
        exact = float(kl_divergence(p, q))  # GaussianKL
        mc = float(kl_divergence(p, q, method="kl_mc", random_seed=1, n_samples=200_000))
        np.testing.assert_allclose(mc, exact, atol=2e-2)

    def test_mc_is_deterministic_with_seed(self):
        p = Normal(loc=0.0, scale=1.0, name="p")
        q = Normal(loc=1.0, scale=1.0, name="q")
        a = float(kl_divergence(p, q, method="kl_mc", random_seed=7, n_samples=1_000))
        b = float(kl_divergence(p, q, method="kl_mc", random_seed=7, n_samples=1_000))
        assert a == b


# ---------------------------------------------------------------------------
# Broadcasting over a DistributionArray
# ---------------------------------------------------------------------------


class TestBroadcasting:
    def test_distribution_array_broadcasts_elementwise(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"p{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        q = Normal(loc=0.0, scale=1.0, name="q")
        result = jnp.asarray(kl_divergence(da, q))
        assert result.shape == (3,)
        # KL(N(i,1) || N(0,1)) = 0.5 * i^2.
        np.testing.assert_allclose(np.asarray(result), [0.0, 0.5, 2.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_kl_registry_is_in_catalog(self):
        assert "kl" in pp.registry_catalog
        assert pp.registry_catalog["kl"] is kl_registry

    def test_describe_lists_all_methods(self):
        text = pp.registry_catalog.describe("kl")
        assert "kl_normal_normal" in text
        assert "kl_tfp" in text
        assert "kl_mc" in text
