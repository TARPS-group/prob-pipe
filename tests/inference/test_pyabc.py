"""Tests for the pyabc SMC-ABC inference backend.

Covers registration + feasibility ``check``, ``condition_on`` auto-dispatch,
posterior recovery (mean *and* spread, across seeds) against a known analytic
conjugate posterior — including a correlated/multivariate prior, which the
flattened-joint design supports — plus importance-weight preservation,
reproducibility, the ``SingleCoreSampler`` default, and the
``PyABCDistribution`` backing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pyabc")  # requires the [pyabc] extra; skipped otherwise

import probpipe.distributions.continuous as C
from probpipe import (
    MultivariateNormal,
    Normal,
    ProductDistribution,
    condition_on,
    log_prob,
    mean,
)
from probpipe.inference import inference_method_registry
from probpipe.inference._pyabc import PyABCDistribution, PyABCSMCMethod
from probpipe.modeling import GenerativeLikelihood, Likelihood
from probpipe.modeling._simple_generative import SimpleGenerativeModel

# Observation noise: small enough that the conjugate posterior concentrates.
_SIGMA = 0.2


class _ConjugateGaussianLikelihood(Likelihood, GenerativeLikelihood):
    """Conjugate model (any dim): ``theta ~ N(0, tau^2 I)`` with ``tau = 3``,
    ``y = theta + sigma * noise`` (``sigma = 0.2``). For a single observation the
    posterior mean ~= y and std ~= 0.20."""

    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        t = jnp.asarray(params).flatten()
        return t[None, :] + _SIGMA * jax.random.normal(key, (num_observations, t.shape[-1]))


def _model(prior) -> SimpleGenerativeModel:
    return SimpleGenerativeModel(prior, _ConjugateGaussianLikelihood())


def _product(*names: str):
    return ProductDistribution(*[Normal(loc=0.0, scale=3.0, name=n) for n in names])


def _means(post) -> dict[str, np.ndarray]:
    m = mean(post)
    return {f: np.asarray(m[f]).reshape(-1) for f in post.event_template.fields}


class TestPyABCCheck:
    def test_registered(self):
        assert "pyabc_smcabc" in inference_method_registry.list_methods()

    def test_rejects_non_generative_model(self):
        info = PyABCSMCMethod().check(Normal(loc=0.0, scale=1.0, name="x"), jnp.array([0.0]))
        assert not info.feasible

    def test_accepts_bare_marginal(self):
        """A bare (non-product) marginal flattens to a length-1 vector, so it's
        feasible — check() and execute() agree (no feasible-then-crash)."""
        model = _model(Normal(loc=0.0, scale=3.0, name="theta"))
        assert PyABCSMCMethod().check(model, jnp.array([2.0])).feasible

    def test_accepts_multivariate_prior(self):
        """A correlated/multivariate prior is feasible — the joint design isn't
        restricted to products of independent scalar marginals."""
        model = _model(MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 9.0, name="m"))
        assert PyABCSMCMethod().check(model, jnp.array([2.0, -1.0])).feasible

    def test_rejects_prior_without_usable_density(self, monkeypatch):
        """check() scores one in-support draw, so a prior that samples/flattens
        but has no usable joint density is infeasible — not a feasible check
        followed by a crash in pyabc's weight computation."""
        from probpipe.inference import _pyabc

        monkeypatch.setattr(_pyabc.PyABCDistribution, "pdf", lambda self, x: float("nan"))
        model = _model(_product("theta"))
        assert not PyABCSMCMethod().check(model, jnp.array([2.0])).feasible


class TestPyABCRecovery:
    # Measured across seeds (n_particles=300, max_populations=6): weighted mean
    # within ~0.05 of truth, per-draw std ~0.15, vs the analytic conjugate
    # posterior (mean ~= y, std ~= 0.20). Bands below are loose around those.
    @pytest.mark.parametrize("seed", [0, 1])
    def test_recovery_1d_mean_and_spread(self, seed):
        post = condition_on(
            _model(_product("theta")),
            jnp.array([2.0]),
            method="pyabc_smcabc",
            n_particles=300,
            max_populations=6,
            random_seed=seed,
        )
        assert _means(post)["theta"][0] == pytest.approx(2.0, abs=0.15)
        std = float(np.asarray(post.draws()["theta"]).std())
        assert 0.08 < std < 0.30

    def test_recovery_2d(self):
        post = condition_on(
            _model(_product("a", "b")),
            jnp.array([1.5, -1.0]),
            method="pyabc_smcabc",
            n_particles=300,
            max_populations=6,
            random_seed=0,
        )
        means = _means(post)
        assert means["a"][0] == pytest.approx(1.5, abs=0.5)
        assert means["b"][0] == pytest.approx(-1.0, abs=0.5)

    def test_recovery_multivariate(self):
        """Recovery with a multivariate prior — draws come back as the named
        vector-valued component."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 9.0, name="m")
        post = condition_on(
            _model(prior),
            jnp.array([1.5, -1.0]),
            method="pyabc_smcabc",
            n_particles=300,
            max_populations=6,
            random_seed=0,
        )
        m = _means(post)["m"]
        assert np.asarray(post.draws()["m"]).shape == (post.num_atoms, 2)
        np.testing.assert_allclose(m, [1.5, -1.0], atol=0.6)

    def test_auto_dispatch(self):
        post = condition_on(
            _model(_product("theta")),
            jnp.array([2.0]),
            n_particles=200,
            max_populations=4,
            random_seed=0,
        )
        assert post.algorithm == "pyabc_smcabc"
        assert _means(post)["theta"][0] == pytest.approx(2.0, abs=0.2)


class TestPyABCWeightsAndDraws:
    def test_posterior_weights_are_non_uniform(self):
        """SMC-ABC's importance weights are kept, not resampled to a uniform
        chain — so the weighted mean actually means something."""
        post = condition_on(
            _model(_product("theta")),
            jnp.array([2.0]),
            method="pyabc_smcabc",
            n_particles=200,
            max_populations=4,
            random_seed=0,
        )
        w = np.asarray(post.weights)
        assert not np.allclose(w, w.mean())
        assert post.num_atoms == 200

    def test_weighted_mean_differs_from_unweighted(self):
        """The kept weights actually change the estimate: the weighted
        posterior mean is not the equal-weight mean of the raw particles."""
        post = condition_on(
            _model(_product("theta")),
            jnp.array([2.0]),
            method="pyabc_smcabc",
            n_particles=200,
            max_populations=4,
            random_seed=0,
        )
        draws = np.asarray(post.draws()["theta"]).reshape(-1)
        weighted = float(np.asarray(mean(post)["theta"]).reshape(-1)[0])
        assert weighted != pytest.approx(float(draws.mean()), abs=1e-6)

    def test_reproducible_across_calls(self):
        kw = dict(method="pyabc_smcabc", n_particles=100, max_populations=3, random_seed=0)
        a = condition_on(_model(_product("theta")), jnp.array([2.0]), **kw)
        b = condition_on(_model(_product("theta")), jnp.array([2.0]), **kw)
        np.testing.assert_array_equal(
            np.asarray(a.draws()["theta"]), np.asarray(b.draws()["theta"])
        )

    def test_draws_are_name_keyed(self):
        post = condition_on(
            _model(_product("theta")),
            jnp.array([2.0]),
            method="pyabc_smcabc",
            n_particles=80,
            max_populations=3,
            random_seed=0,
        )
        draws = post.draws()
        assert "theta" in draws.fields
        assert np.asarray(draws["theta"]).shape == (post.num_atoms,)

    def test_summary_fn_applied(self):
        def summary_fn(y):
            return jnp.mean(jnp.atleast_2d(y), axis=-1, keepdims=True)

        post = condition_on(
            _model(_product("a", "b")),
            jnp.array([2.0, -1.0]),
            method="pyabc_smcabc",
            summary_fn=summary_fn,
            n_particles=80,
            max_populations=3,
            random_seed=0,
        )
        assert set(post.event_template.fields) == {"a", "b"}

    def test_custom_distance_fn_is_used(self):
        """A user-supplied distance_fn over the {"y": vector} sumstats replaces
        the Euclidean default — it is actually called and still recovers."""
        calls = {"n": 0}

        def distance_fn(x, x0):
            calls["n"] += 1
            return float(np.linalg.norm(np.asarray(x["y"]) - np.asarray(x0["y"])))

        post = condition_on(
            _model(_product("theta")),
            jnp.array([2.0]),
            method="pyabc_smcabc",
            distance_fn=distance_fn,
            n_particles=80,
            max_populations=3,
            random_seed=0,
        )
        assert calls["n"] > 0
        assert _means(post)["theta"][0] == pytest.approx(2.0, abs=0.3)


class TestPyABCDiagnostics:
    def test_history_exposed_as_auxiliary(self):
        """The SMC-ABC convergence trajectory is attached as auxiliary
        diagnostics: one row per generation, a non-increasing epsilon schedule,
        acceptance rates in (0, 1], and the total simulation count."""
        post = condition_on(
            _model(_product("theta")),
            jnp.array([2.0]),
            method="pyabc_smcabc",
            n_particles=100,
            max_populations=4,
            random_seed=0,
        )
        diag = post.arviz_data["smc_diagnostics"]
        eps = np.asarray(diag["epsilon"].values)
        rate = np.asarray(diag["acceptance_rate"].values)
        assert 1 <= eps.shape[0] <= 4
        assert np.all(np.diff(eps) <= 1e-8)  # epsilon schedule is non-increasing
        assert np.all((rate > 0) & (rate <= 1.0))
        assert diag.dataset.attrs["total_nr_simulations"] > 0


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
            condition_on(
                _model(_product("theta")),
                jnp.array([2.0]),
                method="pyabc_smcabc",
                n_particles=10,
                max_populations=1,
                random_seed=0,
            )
        assert isinstance(captured["sampler"], SingleCoreSampler)

    def test_eps_and_transitions_are_forwarded(self, monkeypatch):
        """A custom epsilon strategy and transition kernel override the defaults
        via ABCSMC; omitted, eps falls back to ``QuantileEpsilon``."""
        import pyabc

        captured = {}

        class _Stop(Exception):
            pass

        def spy(*args, **kwargs):
            captured.update(eps=kwargs.get("eps"), transitions=kwargs.get("transitions"))
            raise _Stop

        monkeypatch.setattr(pyabc, "ABCSMC", spy)
        my_eps = pyabc.MedianEpsilon()
        my_transitions = pyabc.MultivariateNormalTransition()
        with pytest.raises(_Stop):
            condition_on(
                _model(_product("theta")),
                jnp.array([2.0]),
                method="pyabc_smcabc",
                eps=my_eps,
                transitions=my_transitions,
                n_particles=10,
                max_populations=1,
                random_seed=0,
            )
        assert captured["eps"] is my_eps
        assert captured["transitions"] is my_transitions

    def test_default_eps_is_quantile(self, monkeypatch):
        """Without an explicit ``eps``, the default schedule is QuantileEpsilon."""
        import pyabc

        captured = {}

        class _Stop(Exception):
            pass

        def spy(*args, **kwargs):
            captured["eps"] = kwargs.get("eps")
            raise _Stop

        monkeypatch.setattr(pyabc, "ABCSMC", spy)
        with pytest.raises(_Stop):
            condition_on(
                _model(_product("theta")),
                jnp.array([2.0]),
                method="pyabc_smcabc",
                n_particles=10,
                max_populations=1,
                random_seed=0,
            )
        assert isinstance(captured["eps"], pyabc.QuantileEpsilon)

    def test_run_stopping_criteria_are_forwarded(self, monkeypatch):
        """Caller-supplied stopping criteria reach ``ABCSMC.run`` alongside
        ``max_nr_populations``; omitted ones are not forwarded."""
        import pyabc

        captured = {}

        class _Stop(Exception):
            pass

        def spy_run(self, *args, **kwargs):
            captured.update(kwargs)
            raise _Stop

        monkeypatch.setattr(pyabc.ABCSMC, "run", spy_run)
        with pytest.raises(_Stop):
            condition_on(
                _model(_product("theta")),
                jnp.array([2.0]),
                method="pyabc_smcabc",
                n_particles=10,
                max_populations=2,
                minimum_epsilon=0.5,
                max_total_nr_simulations=1000,
                random_seed=0,
            )
        assert captured["max_nr_populations"] == 2
        assert captured["minimum_epsilon"] == 0.5
        assert captured["max_total_nr_simulations"] == 1000
        assert "min_acceptance_rate" not in captured  # omitted -> pyabc default


class TestPyABCDistributionBacking:
    def test_pdf_is_the_joint_prior_density(self):
        prior = _product("a", "b")
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        assert len(pd.get_parameter_names()) == 2
        flat = jnp.array([0.5, -0.3])
        expected = float(np.exp(np.asarray(log_prob(prior.as_flat_distribution(), flat))))
        assert pd.pdf({"p0": 0.5, "p1": -0.3}) == pytest.approx(expected, rel=1e-5)

    def test_pdf_uses_the_correlated_joint_density(self):
        """With off-diagonal covariance the density is genuinely joint — a
        product of marginals would give a different number."""
        cov = jnp.array([[2.0, 1.2], [1.2, 1.5]])
        prior = MultivariateNormal(loc=jnp.array([0.5, -0.5]), cov=cov, name="m")
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        flat = jnp.array([0.3, -0.2])
        expected = float(np.exp(np.asarray(log_prob(prior.as_flat_distribution(), flat))))
        assert pd.pdf({"p0": 0.3, "p1": -0.2}) == pytest.approx(expected, rel=1e-5)

    def test_rvs_samples_from_the_prior(self):
        prior = _product("a", "b")  # both N(0, 3)
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        draws = np.array([[pd.rvs()[f"p{i}"] for i in range(2)] for _ in range(3000)])
        assert draws.shape == (3000, 2)
        np.testing.assert_allclose(draws.mean(axis=0), [0.0, 0.0], atol=0.2)
        np.testing.assert_allclose(draws.std(axis=0), [3.0, 3.0], atol=0.3)

    def test_supports_non_converter_family(self):
        """Any sampleable marginal with a density works (no fixed family list):
        StudentT, which has no scipy-converter mapping, is feasible."""
        model = _model(ProductDistribution(C.StudentT(df=5.0, loc=0.0, scale=3.0, name="t")))
        assert PyABCSMCMethod().check(model, jnp.array([2.0])).feasible
