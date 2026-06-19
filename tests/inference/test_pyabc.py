"""Tests for the pyabc SMC-ABC inference backend.

Covers registration + feasibility ``check`` (including the priors that must be
rejected rather than crash auto-dispatch), ``condition_on`` auto-dispatch,
posterior recovery (mean *and* spread, across seeds) against a known analytic
conjugate posterior, importance-weight preservation, reproducibility, the
``SingleCoreSampler`` default, and the TFP/ProbPipe -> ``pyabc.RV`` mapping.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pyabc")  # requires the [pyabc] extra; skipped otherwise

import probpipe.distributions.continuous as C
from probpipe import (
    Beta,
    Normal,
    ProductDistribution,
    Uniform,
    condition_on,
    mean,
)
from probpipe.inference import inference_method_registry
from probpipe.inference._pyabc import (
    PyABCSMCMethod,
    _build_pyabc_prior,
    _ensure_pyabc,
    _marginal_to_pyabc_rv,
)
from probpipe.modeling import GenerativeLikelihood, Likelihood
from probpipe.modeling._simple_generative import SimpleGenerativeModel

# Observation noise: small enough that the conjugate posterior concentrates.
_SIGMA = 0.2


class _ConjugateGaussianLikelihood(Likelihood, GenerativeLikelihood):
    """Conjugate model (any dim): ``theta ~ N(0, tau^2 I)`` with ``tau = 3``,
    ``y = theta + sigma * noise`` (``sigma = 0.2``). For a single observation the
    posterior is analytic: ``N(tau^2/(tau^2+sigma^2) * y, tau^2 sigma^2/(tau^2+sigma^2))``
    — i.e. mean ~= y and std ~= 0.20 for the 1-D fixture below."""

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
    m = mean(post)
    return {f: float(np.asarray(m[f]).reshape(-1)[0]) for f in post.record_template.fields}


class TestPyABCCheck:
    def test_registered(self):
        assert "pyabc_smcabc" in inference_method_registry.list_methods()

    def test_rejects_non_generative_model(self):
        info = PyABCSMCMethod().check(Normal(loc=0.0, scale=1.0, name="x"), jnp.array([0.0]))
        assert not info.feasible

    def test_rejects_bare_marginal_prior(self):
        """A SimpleGenerativeModel whose prior is a bare marginal (no
        ``.components``) must report infeasible — not pass check() and then crash
        inside execute(), since pyabc is the sole auto-dispatch pick."""
        model = SimpleGenerativeModel(
            Normal(loc=0.0, scale=1.0, name="theta"), _ConjugateGaussianLikelihood()
        )
        assert not PyABCSMCMethod().check(model, jnp.array([0.0])).feasible

    def test_accepts_any_sampleable_marginal(self):
        """No fixed family list: a marginal backed directly by a ProbPipe
        distribution — e.g. StudentT, which has no scipy-converter mapping — is
        feasible. It only needs sampling and a density."""
        model = SimpleGenerativeModel(
            ProductDistribution(C.StudentT(df=5.0, loc=0.0, scale=3.0, name="theta")),
            _ConjugateGaussianLikelihood(),
        )
        assert PyABCSMCMethod().check(model, jnp.array([2.0])).feasible


class TestPyABCRecovery:
    # Measured across seeds 0-3 (n_particles=300, max_populations=6): weighted
    # mean in [1.98, 2.02] and per-draw std in [0.15, 0.17], vs the analytic
    # conjugate posterior (mean 1.99, std 0.20). The bands below are loose around
    # those, and well clear of the 3.0 prior std / a degenerate point mass.
    @pytest.mark.parametrize("seed", [0, 1])
    def test_recovery_1d_mean_and_spread(self, seed):
        post = condition_on(_model("theta"), jnp.array([2.0]), method="pyabc_smcabc",
                            n_particles=300, max_populations=6, random_seed=seed)
        assert _means(post)["theta"] == pytest.approx(2.0, abs=0.15)
        std = float(np.asarray(post.draws()["theta"]).std())
        assert 0.08 < std < 0.30

    def test_recovery_2d(self):
        truth = jnp.array([1.5, -1.0])
        post = condition_on(_model("a", "b"), truth, method="pyabc_smcabc",
                            n_particles=300, max_populations=6, random_seed=0)
        means = _means(post)
        assert means["a"] == pytest.approx(1.5, abs=0.5)
        assert means["b"] == pytest.approx(-1.0, abs=0.5)

    def test_auto_dispatch(self):
        post = condition_on(_model("theta"), jnp.array([2.0]),
                            n_particles=200, max_populations=4, random_seed=0)
        assert post.algorithm == "pyabc_smcabc"
        assert _means(post)["theta"] == pytest.approx(2.0, abs=0.2)


class TestPyABCWeightsAndDraws:
    def test_posterior_weights_are_non_uniform(self):
        """SMC-ABC's importance weights are kept, not resampled to a uniform
        chain — so the weighted mean actually means something."""
        post = condition_on(_model("theta"), jnp.array([2.0]), method="pyabc_smcabc",
                            n_particles=200, max_populations=4, random_seed=0)
        assert post.weights is not None
        w = np.asarray(post.weights)
        assert not np.allclose(w, w.mean())  # genuine importance weights, not equal
        assert post.num_atoms == 200

    def test_reproducible_across_calls(self):
        kw = dict(method="pyabc_smcabc", n_particles=100, max_populations=3, random_seed=0)
        a = condition_on(_model("theta"), jnp.array([2.0]), **kw)
        b = condition_on(_model("theta"), jnp.array([2.0]), **kw)
        np.testing.assert_array_equal(
            np.asarray(a.draws()["theta"]), np.asarray(b.draws()["theta"])
        )

    def test_draws_are_name_keyed(self):
        post = condition_on(_model("theta"), jnp.array([2.0]), method="pyabc_smcabc",
                            n_particles=80, max_populations=3, random_seed=0)
        draws = post.draws()
        assert "theta" in draws.fields
        assert np.asarray(draws["theta"]).shape == (post.num_atoms,)

    def test_summary_fn_applied(self):
        def summary_fn(y):
            return jnp.mean(jnp.atleast_2d(y), axis=-1, keepdims=True)

        post = condition_on(_model("a", "b"), jnp.array([2.0, -1.0]),
                            method="pyabc_smcabc", summary_fn=summary_fn,
                            n_particles=80, max_populations=3, random_seed=0)
        assert set(post.record_template.fields) == {"a", "b"}


class TestPyABCDefaults:
    def test_default_sampler_is_single_core(self, monkeypatch):
        """Regression guard: the default must be SingleCoreSampler — a forking
        sampler can deadlock against JAX threads and *hang* CI, not just fail."""
        import pyabc
        from pyabc.sampler import SingleCoreSampler

        captured = {}

        class _Stop(Exception):
            pass

        def spy(*args, **kwargs):
            captured["sampler"] = kwargs.get("sampler")
            raise _Stop

        monkeypatch.setattr(pyabc, "ABCSMC", spy)
        with pytest.raises(_Stop):
            condition_on(_model("theta"), jnp.array([2.0]), method="pyabc_smcabc",
                         n_particles=10, max_populations=1, random_seed=0)
        assert isinstance(captured["sampler"], SingleCoreSampler)


class TestPyABCPriorAdapter:
    def test_marginal_rv_is_backed_by_the_distribution(self):
        """The pyabc RV wraps the ProbPipe marginal directly (no scipy): its
        density is the marginal's ``prob`` and its samples have the marginal's
        moments."""
        from probpipe import prob

        pyabc = _ensure_pyabc()
        marginal = Normal(loc=1.0, scale=2.0, name="n")
        rv = _marginal_to_pyabc_rv(marginal, pyabc)
        assert rv.pdf(0.5) == pytest.approx(float(np.asarray(prob(marginal, 0.5))), rel=1e-5)
        np.random.seed(0)
        draws = np.array([rv.rvs() for _ in range(4000)])
        assert draws.mean() == pytest.approx(1.0, abs=0.12)
        assert draws.std() == pytest.approx(2.0, abs=0.15)

    def test_marginal_rv_interface(self):
        """pmf falls back to pdf; cdf is unsupported (unused by SMC-ABC); copy
        returns an equivalent RV."""
        rv = _marginal_to_pyabc_rv(Normal(loc=0.0, scale=1.0, name="n"), _ensure_pyabc())
        assert rv.pmf(0.0) == rv.pdf(0.0)
        with pytest.raises(NotImplementedError):
            rv.cdf(0.0)
        assert isinstance(rv.copy(), type(rv))

    def test_build_prior_orders_by_component(self):
        prior = ProductDistribution(
            Uniform(low=0.0, high=1.0, name="u"),
            Normal(loc=0.0, scale=1.0, name="n"),
            Beta(2.0, 3.0, name="b"),
        )
        _, names = _build_pyabc_prior(prior, _ensure_pyabc())
        assert names == ["u", "n", "b"]

    def test_build_prior_rejects_non_product(self):
        with pytest.raises(TypeError, match="ProductDistribution"):
            _build_pyabc_prior(Normal(loc=0.0, scale=1.0, name="x"), _ensure_pyabc())
