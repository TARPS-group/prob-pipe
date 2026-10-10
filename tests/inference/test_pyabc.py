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
import tensorflow_probability.substrates.jax.bijectors as tfb

pytest.importorskip("pyabc")  # requires the [pyabc] extra; skipped otherwise

import probpipe.families._continuous as C
from probpipe import (
    Beta,
    Dirichlet,
    Gamma,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    OutputSpec,
    Record,
    bijector_for,
    condition_on,
    mean,
    workflow_run,
)
from probpipe.distributions import (
    ConditionalDistribution,
    FactoredDistribution,
    SupportsConditionalSampling,
)
from probpipe.families import BijectorTransformedDistribution
from probpipe.inference import inference_method_registry
from probpipe.inference._inference_utils import observed_target
from probpipe.inference._pyabc import PyABCDistribution, PyABCSMCMethod
from tests._posterior import arviz_data, flat_draws, method_of
from tests.inference._harness import validate_method

# Observation noise: small enough that the conjugate posterior concentrates.
_SIGMA = 0.2


class _Simulator(ConditionalDistribution, SupportsConditionalSampling):
    """``y = theta + sigma * noise`` over the prior's fields, concatenated, which only samples."""

    def __init__(self, prior):
        width = prior.event_spec.spec.vector_size
        super().__init__(
            "y", dict(prior.event_spec.components), OutputSpec(y=NumericArraySpec((width,)))
        )

    def _condition_on(self, given, /, **options):
        values = dict(given.children if isinstance(given, Record) else given)
        theta = jnp.concatenate([jnp.ravel(values[slot]) for slot in self.given_spec])
        return Normal("y", theta, _SIGMA)

    def _conditional_sample(self, given, key, sample_shape=()):
        return self._condition_on(given)._sample(key, sample_shape)


def _model(prior):
    """Conjugate model (any dim): ``theta ~ N(0, tau^2 I)`` with ``tau = 3``,
    ``y = theta + sigma * noise`` (``sigma = 0.2``), the simulator a kernel that
    samples but has no density, composed with *prior*. For a single observation
    the posterior mean ~= y and std ~= 0.20."""
    return _Simulator(prior) * prior


def _observed(*values: float) -> dict:
    return {"y": jnp.array(values)}


def _product(*names: str):
    return FactoredDistribution("prior", [Normal(n, loc=0.0, scale=3.0) for n in names])


def _means(post) -> dict[str, np.ndarray]:
    """The mean of each of the posterior's components, keyed by the component."""
    m = mean(post)
    return {f: np.asarray(m[f"mean({f})"]).reshape(-1) for f in post.event_spec.components}


class TestPyABCCheck:
    def test_registered(self):
        assert "pyabc_smcabc" in inference_method_registry.list_methods()

    def test_rejects_non_generative_model(self):
        info = PyABCSMCMethod().check(
            observed_target(Normal("x", loc=0.0, scale=1.0), jnp.array([0.0]))
        )
        assert not info.feasible

    def test_accepts_bare_marginal(self):
        """A bare (non-product) marginal flattens to a length-1 vector, so it's
        feasible — check() and execute() agree (no feasible-then-crash)."""
        model = _model(Normal("theta", loc=0.0, scale=3.0))
        assert PyABCSMCMethod().check(observed_target(model, _observed(2.0))).feasible

    def test_accepts_multivariate_prior(self):
        """A correlated/multivariate prior is feasible — the joint design isn't
        restricted to products of independent scalar marginals."""
        model = _model(MultivariateNormal("m", loc=jnp.zeros(2), cov=jnp.eye(2) * 9.0))
        assert PyABCSMCMethod().check(observed_target(model, _observed(2.0, -1.0))).feasible

    def test_rejects_prior_without_usable_density(self, monkeypatch):
        """check() scores one in-support draw, so a prior that samples/flattens
        but has no usable joint density is infeasible — not a feasible check
        followed by a crash in pyabc's weight computation."""
        from probpipe.inference import _pyabc

        monkeypatch.setattr(_pyabc.PyABCDistribution, "pdf", lambda self, x: float("nan"))
        model = _model(_product("theta"))
        assert not PyABCSMCMethod().check(observed_target(model, _observed(2.0))).feasible

    def test_accepts_a_transformed_prior(self):
        """A bijector-transformed prior flattens, samples, and scores, so it is feasible."""
        prior = BijectorTransformedDistribution("theta", Normal("z", 0.0, 1.0), tfb.Exp())
        model = _model(prior)
        assert PyABCSMCMethod().check(observed_target(model, _observed(2.0))).feasible


class TestPyABCRecovery:
    # Measured across four workflow seeds for each parametrized seed
    # (n_particles=300, max_populations=6): weighted mean within 0.03 of truth,
    # per-draw std 0.14-0.18, vs the analytic conjugate posterior (mean ~= y,
    # std ~= 0.20). Bands below are loose around those.
    @pytest.mark.parametrize("seed", [0, 1])
    def test_recovery_1d_mean_and_spread(self, seed):
        with workflow_run(seed=seed):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 300, "max_populations": 6},
            )(_model(_product("theta")), _observed(2.0))
        assert _means(post)["theta"][0] == pytest.approx(2.0, abs=0.15)
        std = float(np.asarray(flat_draws(post)["theta"]).std())
        assert 0.08 < std < 0.30

    def test_recovery_2d(self):
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 300, "max_populations": 6},
            )(_model(_product("a", "b")), _observed(1.5, -1.0))
        means = _means(post)
        # Observed across four workflow seeds: |mean error| up to 0.06.
        assert means["a"][0] == pytest.approx(1.5, abs=0.5)
        assert means["b"][0] == pytest.approx(-1.0, abs=0.5)

    def test_recovery_multivariate(self):
        """Recovery with a multivariate prior — draws come back as the named
        vector-valued component."""
        prior = MultivariateNormal("m", loc=jnp.zeros(2), cov=jnp.eye(2) * 9.0)
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 300, "max_populations": 6},
            )(_model(prior), _observed(1.5, -1.0))
        m = _means(post)["m"]
        assert np.asarray(flat_draws(post)["m"]).shape == (post.num_atoms, 2)
        # Observed across four workflow seeds: max |mean error| 0.02-0.05.
        np.testing.assert_allclose(m, [1.5, -1.0], atol=0.6)

    def test_auto_dispatch(self):
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method_options={"n_particles": 200, "max_populations": 4}
            )(_model(_product("theta")), _observed(2.0))
        assert method_of(post) == "pyabc_smcabc"
        # Observed across four workflow seeds: |mean error| 0.01-0.03.
        assert _means(post)["theta"][0] == pytest.approx(2.0, abs=0.2)


class TestPyABCWeightsAndDraws:
    def test_posterior_weights_are_non_uniform(self):
        """SMC-ABC's importance weights are kept, not resampled to a uniform
        chain — so the weighted mean actually means something."""
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 200, "max_populations": 4},
            )(_model(_product("theta")), _observed(2.0))
        w = np.asarray(post.weights)
        assert not np.allclose(w, w.mean())
        assert post.num_atoms == 200

    def test_weighted_mean_differs_from_unweighted(self):
        """The kept weights actually change the estimate: the weighted
        posterior mean is not the equal-weight mean of the raw particles."""
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 200, "max_populations": 4},
            )(_model(_product("theta")), _observed(2.0))
        draws = np.asarray(flat_draws(post)["theta"]).reshape(-1)
        weighted = float(np.asarray(mean(post)["mean(theta)"]).reshape(-1)[0])
        assert weighted != pytest.approx(float(draws.mean()), abs=1e-6)

    def test_reproducible_across_calls(self):
        smc = condition_on.with_options(
            method="pyabc_smcabc",
            method_options={"n_particles": 100, "max_populations": 3},
        )
        with workflow_run(seed=0):
            a = smc(_model(_product("theta")), _observed(2.0))
        with workflow_run(seed=0):
            b = smc(_model(_product("theta")), _observed(2.0))
        np.testing.assert_array_equal(
            np.asarray(flat_draws(a)["theta"]), np.asarray(flat_draws(b)["theta"])
        )

    def test_draws_are_name_keyed(self):
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 80, "max_populations": 3},
            )(_model(_product("theta")), _observed(2.0))
        draws = flat_draws(post)
        assert "theta" in draws.event_template.fields
        assert np.asarray(draws["theta"]).shape == (post.num_atoms,)

    def test_summary_fn_applied(self):
        def summary_fn(y):
            return jnp.mean(jnp.atleast_2d(y), axis=-1, keepdims=True)

        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={
                    "summary_fn": summary_fn,
                    "n_particles": 80,
                    "max_populations": 3,
                },
            )(_model(_product("a", "b")), _observed(2.0, -1.0))
        assert set(post.event_spec.components) == {"a", "b"}

    def test_custom_distance_fn_is_used(self):
        """A user-supplied distance_fn over the {"y": vector} sumstats replaces
        the Euclidean default — it is actually called and still recovers."""
        calls = {"n": 0}

        def distance_fn(x, x0):
            calls["n"] += 1
            return float(np.linalg.norm(np.asarray(x["y"]) - np.asarray(x0["y"])))

        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={
                    "distance_fn": distance_fn,
                    "n_particles": 80,
                    "max_populations": 3,
                },
            )(_model(_product("theta")), _observed(2.0))
        assert calls["n"] > 0
        # Observed across four workflow seeds: |mean error| 0.02-0.06.
        assert _means(post)["theta"][0] == pytest.approx(2.0, abs=0.3)


class TestPyABCDiagnostics:
    def test_history_exposed_as_annotations(self):
        """The SMC-ABC convergence trajectory is attached as annotations
        diagnostics: one row per generation, a non-increasing epsilon schedule,
        acceptance rates in (0, 1], and the total simulation count."""
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 100, "max_populations": 4},
            )(_model(_product("theta")), _observed(2.0))
        diag = arviz_data(post)["smc_diagnostics"]
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
            condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 10, "max_populations": 1},
            )(_model(_product("theta")), _observed(2.0))
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
            condition_on.with_options(
                method="pyabc_smcabc",
                method_options={
                    "eps": my_eps,
                    "transitions": my_transitions,
                    "n_particles": 10,
                    "max_populations": 1,
                },
            )(_model(_product("theta")), _observed(2.0))
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
            condition_on.with_options(
                method="pyabc_smcabc",
                method_options={"n_particles": 10, "max_populations": 1},
            )(_model(_product("theta")), _observed(2.0))
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
            condition_on.with_options(
                method="pyabc_smcabc",
                method_options={
                    "n_particles": 10,
                    "max_populations": 2,
                    "minimum_epsilon": 0.5,
                    "max_total_nr_simulations": 1000,
                },
            )(_model(_product("theta")), _observed(2.0))
        assert captured["max_nr_populations"] == 2
        assert captured["minimum_epsilon"] == 0.5
        assert captured["max_total_nr_simulations"] == 1000
        assert "min_acceptance_rate" not in captured  # omitted -> pyabc default


class TestPyABCDistributionBacking:
    def test_pdf_is_the_joint_prior_density(self):
        prior = _product("a", "b")
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        assert len(pd.get_parameter_names()) == 2
        expected = float(np.exp(np.asarray(prior._log_prob({"a": 0.5, "b": -0.3}))))
        assert pd.pdf({"p0": 0.5, "p1": -0.3}) == pytest.approx(expected, rel=1e-5)

    def test_pdf_uses_the_correlated_joint_density(self):
        """With off-diagonal covariance the density is genuinely joint — a
        product of marginals would give a different number."""
        cov = jnp.array([[2.0, 1.2], [1.2, 1.5]])
        prior = MultivariateNormal("m", loc=jnp.array([0.5, -0.5]), cov=cov)
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        expected = float(np.exp(np.asarray(prior._log_prob(jnp.array([0.3, -0.2])))))
        assert pd.pdf({"p0": 0.3, "p1": -0.2}) == pytest.approx(expected, rel=1e-5)

    def test_rvs_samples_from_the_prior(self):
        prior = _product("a", "b")  # both N(0, 3)
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        draws = np.array([[pd.rvs()[f"p{i}"] for i in range(2)] for _ in range(3000)])
        assert draws.shape == (3000, 2)
        np.testing.assert_allclose(draws.mean(axis=0), [0.0, 0.0], atol=0.2)
        np.testing.assert_allclose(draws.std(axis=0), [3.0, 3.0], atol=0.3)

    def test_a_simplex_prior_has_one_coordinate_fewer_than_its_event(self):
        pd = PyABCDistribution(Dirichlet("w", jnp.ones(3)), jax.random.PRNGKey(0))
        assert pd.get_parameter_names() == ["p0", "p1"]

    def test_pdf_is_the_density_in_unconstrained_coordinates(self):
        """A positive prior's coordinate is a log, so its density gains the Jacobian ``exp(z)``."""
        prior = Gamma("g", 2.0, 1.0)
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        expected = float(np.exp(np.asarray(prior._log_prob(jnp.exp(0.3))))) * np.exp(0.3)
        assert pd.pdf({"p0": 0.3}) == pytest.approx(expected, rel=1e-5)

    def test_rvs_gives_the_coordinates_of_a_prior_draw(self):
        prior = Beta("p", 2.0, 3.0)  # mean 0.4
        pd = PyABCDistribution(prior, jax.random.PRNGKey(0))
        onto_support = bijector_for(prior.event_spec.spec.support).raw()
        coordinates = jnp.array([pd.rvs()["p0"] for _ in range(2000)])
        draws = np.asarray(jax.vmap(onto_support)(coordinates))
        assert draws.mean() == pytest.approx(0.4, abs=0.03)

    def test_supports_non_converter_family(self):
        """Any sampleable marginal with a density works (no fixed family list):
        StudentT, which has no scipy-converter mapping, is feasible."""
        model = _model(FactoredDistribution("prior", [C.StudentT("t", df=5.0, loc=0.0, scale=3.0)]))
        assert PyABCSMCMethod().check(observed_target(model, _observed(2.0))).feasible


#: The SMC-ABC budget of the support tests.
_SUPPORT_OPTIONS = {"n_particles": 100, "max_populations": 3}


class TestPyABCSupport:
    """pyabc perturbs the prior's unconstrained coordinates, so every particle lies in the support."""

    def test_a_posterior_near_a_bound_stays_in_the_unit_interval(self):
        """A uniform prior's backend density is finite past the bounds, so only the
        coordinates keep a perturbed particle inside them."""
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc", method_options=_SUPPORT_OPTIONS
            )(_model(Beta("p", 1.0, 1.0)), _observed(0.95))
        draws = np.asarray(flat_draws(post)["p"]).ravel()
        assert np.all((draws > 0) & (draws < 1))

    def test_a_simplex_posterior_stays_on_the_simplex(self):
        with workflow_run(seed=0):
            post = condition_on.with_options(
                method="pyabc_smcabc", method_options=_SUPPORT_OPTIONS
            )(_model(Dirichlet("w", jnp.ones(3))), _observed(0.7, 0.2, 0.1))
        draws = np.asarray(flat_draws(post)["w"])
        assert draws.shape == (post.num_atoms, 3)
        assert np.all(draws > 0)
        np.testing.assert_allclose(draws.sum(axis=-1), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# The canonical cases of the cross-method validation harness
# ---------------------------------------------------------------------------

test_pyabc_smcabc_canonical = validate_method("pyabc_smcabc")


class TestProcessKeys:
    """A key stream differs in each process and is unchanged in the one that made it."""

    def test_the_owning_process_draws_the_split_keys(self):
        from probpipe.inference._pyabc import _ProcessKeys

        keys = _ProcessKeys(jax.random.PRNGKey(3))
        expected, first = jax.random.split(jax.random.PRNGKey(3))
        _, second = jax.random.split(expected)
        np.testing.assert_array_equal(keys.next(), first)
        np.testing.assert_array_equal(keys.next(), second)

    def test_another_process_draws_keys_of_its_own(self, monkeypatch):
        from probpipe.inference import _pyabc

        parent = _pyabc._ProcessKeys(jax.random.PRNGKey(3))
        workers = []
        for pid, seed in ((10_001, 1), (10_002, 2)):
            worker = _pyabc._ProcessKeys(jax.random.PRNGKey(3))
            monkeypatch.setattr(_pyabc.os, "getpid", lambda pid=pid: pid)
            np.random.seed(seed)
            workers.append(np.asarray(worker.next()))
        monkeypatch.undo()
        own = np.asarray(parent.next())

        assert not np.array_equal(workers[0], workers[1])
        assert not np.array_equal(workers[0], own)
