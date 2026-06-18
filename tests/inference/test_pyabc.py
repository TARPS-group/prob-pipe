"""Tests for the pyabc SMC-ABC inference backend.

Covers:
- registration + feasibility ``check``;
- ``condition_on`` auto-dispatch to ``pyabc_smcabc`` for a
  ``SimpleGenerativeModel``;
- posterior recovery near a known truth (1-D and 2-D — 1-D in particular
  exercises the path the old sbijax SMC-ABC backend mishandled);
- weighted-particle preservation and name-keyed ``draws()`` / ``mean()``;
- the TFP-marginal -> ``pyabc.RV`` adapter, including its error paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pyabc")  # requires the [pyabc] extra; skipped otherwise

from probpipe import (
    Beta,
    Exponential,
    Gamma,
    Normal,
    ProductDistribution,
    Uniform,
    condition_on,
    mean,
)
from probpipe.inference import inference_method_registry
from probpipe.inference._pyabc import _build_pyabc_prior, _marginal_to_pyabc_rv
from probpipe.modeling import GenerativeLikelihood, Likelihood
from probpipe.modeling._simple_generative import SimpleGenerativeModel

# Small observation noise so the conjugate posterior concentrates near the truth.
_SIGMA = 0.2


class _ConjugateGaussianLikelihood(Likelihood, GenerativeLikelihood):
    """Conjugate model (any dimension): ``theta ~ N(0, tau^2 I)``,
    ``y = theta + sigma * noise``, so a (near-)noiseless observation pins the
    posterior mean close to the truth."""

    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        t = jnp.asarray(params).flatten()
        return t[None, :] + _SIGMA * jax.random.normal(key, (num_observations, t.shape[-1]))


def _model(*names: str) -> SimpleGenerativeModel:
    prior = ProductDistribution(*[Normal(loc=0.0, scale=3.0, name=n) for n in names])
    return SimpleGenerativeModel(prior, _ConjugateGaussianLikelihood())


def _means(post) -> dict[str, float]:
    """Posterior means keyed by parameter name (honours particle weights)."""
    m = mean(post)
    return {f: float(np.asarray(m[f]).reshape(-1)[0]) for f in post.record_template.fields}


class TestPyABCRegistration:
    def test_registered(self):
        assert "pyabc_smcabc" in inference_method_registry.list_methods()

    def test_check_rejects_non_generative_model(self):
        from probpipe.inference._pyabc import PyABCSMCMethod

        info = PyABCSMCMethod().check(Normal(loc=0.0, scale=1.0, name="x"), jnp.array([0.0]))
        assert not info.feasible


class TestPyABCRecovery:
    def test_recovery_1d(self):
        """1-D recovery — the case the old sbijax backend's chol-factor
        monkeypatch existed to work around."""
        post = condition_on(_model("theta"), jnp.array([2.0]), method="pyabc_smcabc",
                            n_particles=200, max_populations=6, random_seed=0)
        assert _means(post)["theta"] == pytest.approx(2.0, abs=0.5)

    def test_recovery_2d(self):
        truth = jnp.array([1.5, -1.0])
        post = condition_on(_model("a", "b"), truth, method="pyabc_smcabc",
                            n_particles=200, max_populations=6, random_seed=0)
        means = _means(post)
        assert means["a"] == pytest.approx(1.5, abs=0.6)
        assert means["b"] == pytest.approx(-1.0, abs=0.6)

    def test_auto_dispatch(self):
        """With no method given, condition_on auto-selects pyabc_smcabc for a
        SimpleGenerativeModel."""
        post = condition_on(_model("theta"), jnp.array([2.0]),
                            n_particles=200, max_populations=6, random_seed=0)
        assert post.algorithm == "pyabc_smcabc"
        assert _means(post)["theta"] == pytest.approx(2.0, abs=0.5)

    def test_draws_are_name_keyed(self):
        """draws() returns a Record keyed by the prior's parameter names, so
        callers can index by name (``draws["theta"]``)."""
        post = condition_on(_model("theta"), jnp.array([2.0]), method="pyabc_smcabc",
                            n_particles=80, max_populations=3, random_seed=0)
        draws = post.draws()
        assert "theta" in draws.fields
        assert np.asarray(draws["theta"]).shape == (post.num_atoms,)

    def test_posterior_is_weighted(self):
        post = condition_on(_model("theta"), jnp.array([2.0]), method="pyabc_smcabc",
                            n_particles=100, max_populations=3, random_seed=0)
        assert post.weights is not None
        assert post.num_atoms == 100

    def test_summary_fn(self):
        """A user summary_fn is applied to both simulated and observed data."""
        def summary_fn(y):
            return jnp.mean(jnp.atleast_2d(y), axis=-1, keepdims=True)

        post = condition_on(_model("a", "b"), jnp.array([2.0, -1.0]),
                            method="pyabc_smcabc", summary_fn=summary_fn,
                            n_particles=80, max_populations=3, random_seed=0)
        assert set(post.record_template.fields) == {"a", "b"}


class TestPyABCPriorAdapter:
    def test_supported_families(self):
        prior = ProductDistribution(
            Uniform(low=0.0, high=1.0, name="u"),
            Normal(loc=0.0, scale=1.0, name="n"),
            Beta(2.0, 3.0, name="b"),
            Gamma(concentration=2.0, rate=1.5, name="g"),
        )
        pyabc_prior, names = _build_pyabc_prior(prior)
        assert names == ["u", "n", "b", "g"]
        assert set(pyabc_prior.get_parameter_names()) == {"u", "n", "b", "g"}

    def test_unsupported_family_raises(self):
        with pytest.raises(NotImplementedError, match="No pyabc RV mapping"):
            _marginal_to_pyabc_rv(Exponential(rate=1.0, name="e"))

    def test_non_tfp_marginal_raises(self):
        with pytest.raises(TypeError, match="TFP-backed"):
            _marginal_to_pyabc_rv(object())

    def test_build_prior_requires_product(self):
        with pytest.raises(TypeError, match="ProductDistribution"):
            _build_pyabc_prior(Normal(loc=0.0, scale=1.0, name="x"))
