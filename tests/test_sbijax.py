"""Tests for sbijax integration (amortized NPE + SMCABC).

These tests require sbijax to be installed: pip install probpipe[sbi]
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sbijax = pytest.importorskip("sbijax")

from probpipe import (
    Normal,
    SimpleGenerativeModel,
    SimpleModel,
    condition_on,
    sbi_learn_conditional,
    sbi_learn_likelihood,
)
from probpipe.core.distribution import Distribution
from probpipe.core.protocols import SupportsConditioning, SupportsSampling
from probpipe.distributions.multivariate import MultivariateNormal
from probpipe.inference import (
    ApproximateDistribution,
    DirectSamplerSBIModel,
    inference_method_registry,
)
from probpipe.inference._sbijax import (
    PARAM_KEY,
    SbiSMCABCMethod,
    _adapt_prior as adapt_prior,
    _adapt_simulator as adapt_simulator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class GaussianSimulator2D:
    """2D Gaussian simulator: y = theta + 0.1 * noise."""

    def generate_data(self, params, n_samples, *, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=(n_samples, 2))
        return params + 0.1 * noise


class Gaussian2To5Simulator:
    """2D params, 5D observations: y = A @ theta + 0.1 * noise.

    Used by tests where ``data_dim != theta_dim`` is required to detect
    any code path that conflates the two dimensions (e.g. passing the
    parameter dimensionality to a network factory that should receive
    the observation dimensionality, or vice-versa).
    """

    _A = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [0.5, 0.5],
        ]
    )

    def generate_data(self, params, n_samples, *, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        mean = self._A @ jnp.asarray(params)  # shape (5,)
        noise = jax.random.normal(key, shape=(n_samples, 5))
        return mean + 0.1 * noise


@pytest.fixture
def prior_2d():
    return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta")


@pytest.fixture
def simulator_2d():
    return GaussianSimulator2D()


@pytest.fixture
def simulator_2to5():
    return Gaussian2To5Simulator()


@pytest.fixture
def generative_model_2d(prior_2d, simulator_2d):
    return SimpleGenerativeModel(prior_2d, simulator_2d)


@pytest.fixture
def observed_2d():
    return jnp.array([0.5, -0.3])


@pytest.fixture(scope="module")
def _trained_npe_smoke():
    """Tiny, fast NPE training — for smoke tests only (no correctness checks)."""
    prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta")
    simulator = GaussianSimulator2D()
    return sbi_learn_conditional._func(
        prior, simulator,
        method="npe",
        n_simulations=200,
        n_iter=5,
        batch_size=32,
        n_samples=100,
        random_seed=42,
    )


@pytest.fixture(scope="module")
def _trained_nle_correct():
    """Properly trained NLE — used to assert likelihood/posterior correctness."""
    prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta")
    simulator = GaussianSimulator2D()
    return sbi_learn_likelihood._func(
        prior, simulator,
        method="nle",
        n_simulations=2000,
        n_iter=300,
        batch_size=128,
        random_seed=0,
    )


@pytest.fixture(scope="module")
def _trained_npe_correct():
    """Properly trained NPE — slower, used to assert posterior correctness."""
    prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta")
    simulator = GaussianSimulator2D()
    return sbi_learn_conditional._func(
        prior, simulator,
        method="npe",
        n_simulations=2000,
        n_iter=300,
        batch_size=128,
        n_samples=2000,
        random_seed=0,
    )


# ---------------------------------------------------------------------------
# Amortized NPE via sbi_learn_conditional
# ---------------------------------------------------------------------------


@pytest.mark.sbi
class TestTrainSBI:
    """sbi_learn_* workflow functions: construction, protocols, correctness."""

    def test_returns_trained_model_and_is_distribution(self, _trained_npe_smoke):
        trained = _trained_npe_smoke
        assert isinstance(trained, DirectSamplerSBIModel)
        assert isinstance(trained, Distribution)
        assert isinstance(trained, SupportsConditioning)
        assert trained.name == "DirectSamplerSBIModel(sbijax_npe)"
        # repr should include the type and the algorithm tag.
        r = repr(trained)
        assert "DirectSamplerSBIModel" in r
        assert "sbijax_npe" in r

    def test_condition_on_n_samples_override(self, _trained_npe_smoke, observed_2d):
        """Smoke-level dispatch: output type, algorithm tag, and n_samples override."""
        posterior = condition_on._func(
            _trained_npe_smoke, observed_2d, n_samples=321,
        )
        assert isinstance(posterior, ApproximateDistribution)
        assert posterior.algorithm == "sbijax_npe"
        draws = np.asarray(posterior.draws())
        assert draws.shape[0] == 321
        # Draws should lie in the parameter space (2D vectors).
        assert draws.shape[-1] == 2

    def test_posterior_correctness(self, _trained_npe_correct):
        """Well-trained NPE must recover posteriors near each observation."""
        from probpipe import mean

        obs = jnp.array([1.2, -0.8])
        posterior = condition_on._func(_trained_npe_correct, obs)
        post_mean = np.asarray(mean(posterior))
        # Likelihood is tight (noise=0.1), prior is standard; posterior mean
        # should be very close to observation.
        np.testing.assert_allclose(post_mean, obs, atol=0.3)

    def test_amortization_different_observations(self, _trained_npe_correct):
        """Same trained model conditioned on different obs → different posteriors."""
        from probpipe import mean

        obs_a = jnp.array([1.5, 0.2])
        obs_b = jnp.array([-0.9, 1.1])
        post_a = condition_on._func(_trained_npe_correct, obs_a)
        post_b = condition_on._func(_trained_npe_correct, obs_b)

        mean_a = np.asarray(mean(post_a))
        mean_b = np.asarray(mean(post_b))
        # Each near its own observation, far from the other.
        np.testing.assert_allclose(mean_a, obs_a, atol=0.3)
        np.testing.assert_allclose(mean_b, obs_b, atol=0.3)
        assert np.linalg.norm(mean_a - mean_b) > 1.0

    def test_nle_returns_simple_model(self, _trained_nle_correct):
        """NLE returns a SimpleModel with a working neural likelihood."""
        trained = _trained_nle_correct
        assert isinstance(trained, SimpleModel)

    def test_nle_likelihood_peaks_at_truth(self, _trained_nle_correct):
        """Neural likelihood log p(y|theta) should be maximized near theta=y.

        For ``y = theta + 0.1*noise`` the true MLE is ``theta=y``; a well-
        trained NLE network must reflect this, otherwise the posterior
        track is broken.
        """
        trained = _trained_nle_correct
        likelihood = trained._likelihood
        obs = jnp.array([1.0, -0.5])
        # log p(y|theta) at the truth vs at far-away points.
        ll_truth = likelihood.log_likelihood(obs, obs)
        ll_far_a = likelihood.log_likelihood(jnp.array([3.0, 3.0]), obs)
        ll_far_b = likelihood.log_likelihood(jnp.array([-2.0, 2.0]), obs)
        assert ll_truth > ll_far_a
        assert ll_truth > ll_far_b

    def test_nle_posterior_recovery_via_registry(
        self, _trained_nle_correct, prior_2d
    ):
        """NLE → SimpleModel → registry MCMC must recover the truth.

        End-to-end check of the MCMC-required track: ``condition_on``
        dispatches through the standard inference registry (NUTS here)
        on the SimpleModel returned by ``sbi_learn_likelihood``, and the resulting
        posterior mean should be close to the observation since the
        likelihood is tight (noise=0.1) and the prior is standard normal.
        """
        from probpipe import mean

        obs = jnp.array([1.0, -0.5])
        posterior = condition_on._func(
            _trained_nle_correct, obs, method="tfp_nuts",
            n_samples=500, n_warmup=500, n_chains=2, random_seed=0,
        )
        assert isinstance(posterior, ApproximateDistribution)
        post_mean = np.asarray(mean(posterior))
        np.testing.assert_allclose(post_mean, np.asarray(obs), atol=0.3)

    def test_nle_return_likelihood_only(self, prior_2d, simulator_2d):
        """``return_likelihood_only=True`` returns just the trained Likelihood.

        The returned object must satisfy the Likelihood protocol and be
        usable as the likelihood of a fresh SimpleModel built with a
        different prior — proving the prior is not baked into the
        learned likelihood.
        """
        from probpipe.modeling._likelihood import Likelihood

        likelihood = sbi_learn_likelihood._func(
            prior_2d, simulator_2d,
            method="nle",
            n_simulations=200,
            n_iter=5,
            batch_size=32,
            return_likelihood_only=True,
        )
        assert isinstance(likelihood, Likelihood)
        # Not a SimpleModel — just the bare likelihood.
        assert not isinstance(likelihood, SimpleModel)

        # Compose with a different prior and verify joint log_prob is finite
        # and depends on the parameters.
        alt_prior = MultivariateNormal(
            loc=jnp.ones(2), cov=2.0 * jnp.eye(2), name="theta"
        )
        model = SimpleModel(alt_prior, likelihood)
        obs = jnp.array([0.5, -0.3])
        lp_a = model._log_prob((jnp.zeros(2), obs))
        lp_b = model._log_prob((jnp.array([1.5, -1.5]), obs))
        assert jnp.isfinite(lp_a) and jnp.isfinite(lp_b)
        assert not jnp.allclose(lp_a, lp_b)

    def test_invalid_method(self, prior_2d, simulator_2d):
        with pytest.raises(ValueError, match="Unknown conditional SBI method"):
            sbi_learn_conditional._func(prior_2d, simulator_2d, method="invalid")
        with pytest.raises(ValueError, match="Unknown likelihood SBI method"):
            sbi_learn_likelihood._func(prior_2d, simulator_2d, method="invalid")
        # Cross-track methods should also be rejected.
        with pytest.raises(ValueError, match="Unknown conditional SBI method"):
            sbi_learn_conditional._func(prior_2d, simulator_2d, method="nle")
        with pytest.raises(ValueError, match="Unknown likelihood SBI method"):
            sbi_learn_likelihood._func(prior_2d, simulator_2d, method="npe")

    def test_scalar_prior_is_wrapped(self):
        """Scalar priors (event_shape=()) must be wrapped via tfd.Sample.

        Exercises the ``adapt_prior`` branch that wraps scalar TFP
        distributions, and verifies NPE training + conditioning works
        on a 1D problem end-to-end.
        """
        from probpipe import mean
        scalar_prior = Normal(loc=0.0, scale=1.0, name="theta")

        class Scalar1DSimulator:
            def generate_data(self, params, n_samples, *, key=None):
                if key is None:
                    key = jax.random.PRNGKey(0)
                noise = jax.random.normal(
                    key, shape=(n_samples,) + jnp.atleast_1d(params).shape,
                )
                return jnp.atleast_1d(params) + 0.1 * noise

        trained = sbi_learn_conditional._func(
            scalar_prior, Scalar1DSimulator(),
            method="npe",
            n_simulations=2000,
            n_iter=300,
            batch_size=128,
            n_samples=1000,
            random_seed=1,
        )
        obs = jnp.array([1.3])
        posterior = condition_on._func(trained, obs)
        # Tight likelihood → posterior mean near observation.
        post_mean = float(np.asarray(mean(posterior)).item())
        np.testing.assert_allclose(post_mean, float(obs[0]), atol=0.3)

    def test_conditional_network_factory_receives_theta_dim(
        self, prior_2d, simulator_2to5
    ):
        """NPE factory must receive the parameter dim, not the data dim.

        Uses a simulator with ``data_dim=5 != theta_dim=2`` so that any
        confusion between the two dimensionalities would show up as the
        wrong captured value.  NPE models ``p(theta|y)``, so the flow's
        ``n_dimension`` is the parameter dimension (2).
        """
        from sbijax.nn import make_maf
        captured = {}

        def factory(ndim):
            captured["ndim"] = ndim
            return make_maf(ndim)

        sbi_learn_conditional._func(
            prior_2d, simulator_2to5,
            method="npe",
            network_factory=factory,
            n_simulations=100,
            n_iter=2,
            batch_size=32,
        )
        assert captured["ndim"] == 2  # theta dim, not data dim (5)

    def test_nle_network_factory_receives_data_dim(
        self, prior_2d, simulator_2to5
    ):
        """NLE factory must receive the DATA dim, not the parameter dim.

        Regression test: before the fix, ``_train`` always passed the
        prior dim to the network factory, so NLE built a flow with the
        wrong ``n_dimension`` and crashed at the first ``model.init``.
        NLE models ``p(y|theta)``, so the flow's ``n_dimension`` is the
        observation dimension (5 here, not the prior dim 2).
        """
        from sbijax.nn import make_maf
        captured = {}

        def factory(ndim):
            captured["ndim"] = ndim
            return make_maf(ndim)

        sbi_learn_likelihood._func(
            prior_2d, simulator_2to5,
            method="nle",
            network_factory=factory,
            n_simulations=100,
            n_iter=2,
            batch_size=32,
        )
        assert captured["ndim"] == 5  # data dim, not theta dim (2)

    def test_nle_end_to_end_with_mismatched_dims(
        self, prior_2d, simulator_2to5
    ):
        """NLE train + likelihood eval works when data_dim != theta_dim.

        End-to-end regression test: before the fix, this call crashed at
        ``model.init`` with a dot-product shape mismatch because the
        MAF was built with the wrong ``n_dimension``.  Also verifies
        that the trained ``_NLELikelihood`` evaluates on 5D observations
        with 2D parameters and returns a finite log-probability.
        """
        trained = sbi_learn_likelihood._func(
            prior_2d, simulator_2to5,
            method="nle",
            n_simulations=200,
            n_iter=5,
            batch_size=32,
        )
        assert isinstance(trained, SimpleModel)

        theta = jnp.array([0.5, -0.3])
        obs = simulator_2to5.generate_data(
            theta, 1, key=jax.random.PRNGKey(42)
        )[0]
        assert obs.shape == (5,)
        ll = trained._likelihood.log_likelihood(theta, obs)
        assert jnp.isfinite(ll)

    def test_nle_fails_if_factory_builds_wrong_ndim(
        self, prior_2d, simulator_2to5
    ):
        """NLE with a network sized to theta_dim instead of data_dim must fail."""
        from sbijax.nn import make_maf

        def buggy_factory(ndim):
            return make_maf(2)  # theta_dim, not the correct data_dim (5)

        with pytest.raises(Exception):  # noqa: B017 — jax shape error
            sbi_learn_likelihood._func(
                prior_2d, simulator_2to5,
                method="nle",
                network_factory=buggy_factory,
                n_simulations=200,
                n_iter=5,
                batch_size=32,
            )

    def test_nre_network_factory_called_with_no_args(
        self, prior_2d, simulator_2d
    ):
        """NRE factory must be called with no positional arguments.

        Regression test: NRE uses an MLP classifier whose default
        factory (``make_mlp``) takes zero arguments.  Before the fix,
        ``_train`` unconditionally passed an int dim to the factory,
        which would have broken any attempt to use a signature-correct
        MLP factory here.
        """
        from sbijax.nn import make_mlp
        captured = {"called": False, "args": None, "kwargs": None}

        def factory(*args, **kwargs):
            captured["called"] = True
            captured["args"] = args
            captured["kwargs"] = kwargs
            return make_mlp()

        sbi_learn_likelihood._func(
            prior_2d, simulator_2d,
            method="nre",
            network_factory=factory,
            n_simulations=100,
            n_iter=2,
            batch_size=32,
        )
        assert captured["called"]
        assert captured["args"] == ()
        assert captured["kwargs"] == {}

    def test_nre_trains_with_default_network(self, prior_2d, simulator_2d):
        """NRE training succeeds with its default ``make_mlp`` factory.

        Regression test: the builder table previously paired NRE with
        ``make_maf`` (the NLE flow), which is the wrong network type
        and crashed at init.  A bare ``sbi_learn_likelihood`` call with
        ``method='nre'`` must succeed end-to-end with no overrides.
        """
        trained = sbi_learn_likelihood._func(
            prior_2d, simulator_2d,
            method="nre",
            n_simulations=200,
            n_iter=5,
            batch_size=32,
        )
        assert isinstance(trained, SimpleModel)
        ll = trained._likelihood.log_likelihood(
            jnp.array([0.5, -0.3]), jnp.array([0.5, -0.3])
        )
        assert jnp.isfinite(ll)

    def test_rejects_non_tfp_prior(self, simulator_2d):
        class NonTFPPrior:
            # Satisfies SupportsSampling structurally but has no _tfp_dist.
            _sampling_cost = "low"
            _preferred_orchestration = None

            def _sample(self, key, sample_shape=()):
                return jnp.zeros(sample_shape + (2,))

        with pytest.raises(TypeError, match="TFP-backed"):
            sbi_learn_conditional._func(
                NonTFPPrior(), simulator_2d, n_simulations=10, n_iter=1
            )
        with pytest.raises(TypeError, match="TFP-backed"):
            sbi_learn_likelihood._func(
                NonTFPPrior(), simulator_2d, n_simulations=10, n_iter=1
            )


# ---------------------------------------------------------------------------
# SMCABC via registry
# ---------------------------------------------------------------------------


@pytest.mark.sbi
class TestSMCABC:
    """Non-amortized SMCABC via the inference method registry."""

    def test_smcabc_via_method_override(self, generative_model_2d, observed_2d):
        """Registry-level explicit method override: type, tag, and draw shape."""
        result = inference_method_registry.execute(
            generative_model_2d, observed_2d,
            method="sbijax_smcabc",
            n_rounds=2,
            n_particles=100,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.algorithm == "sbijax_smcabc"
        draws = np.asarray(result.draws())
        # Each particle is a 2D parameter vector; we asked for 100 particles.
        assert draws.shape[-1] == 2
        assert draws.shape[0] >= 100

    def test_smcabc_via_condition_on_dispatch(
        self, generative_model_2d, observed_2d
    ):
        """condition_on auto-dispatches to SMCABC for a generative model."""
        from probpipe import mean

        result = condition_on._func(
            generative_model_2d, observed_2d,
            method="sbijax_smcabc",
            n_rounds=3,
            n_particles=200,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.algorithm == "sbijax_smcabc"
        # Posterior mean should be in the vicinity of the observation —
        # regression guard against the 1D _chol_factor workaround breaking.
        post_mean = np.asarray(mean(result))
        assert np.linalg.norm(post_mean - np.asarray(observed_2d)) < 1.0

    def test_smcabc_check_feasibility(self, generative_model_2d, observed_2d):
        method = SbiSMCABCMethod()
        info = method.check(generative_model_2d, observed_2d)
        assert info.feasible
        assert info.method_name == "sbijax_smcabc"

    def test_smcabc_method_registered(self, generative_model_2d, observed_2d):
        """SbiSMCABCMethod must be actually registered in the global registry."""
        assert "sbijax_smcabc" in inference_method_registry.list_methods()
        # Auto-dispatch on a SimpleGenerativeModel should resolve to SMCABC.
        info = inference_method_registry.check(generative_model_2d, observed_2d)
        assert info.feasible
        assert info.method_name == "sbijax_smcabc"

    def test_smcabc_rejects_non_generative_model(self):
        class DummyLik:
            def log_likelihood(self, params, data):
                return 0.0

        model = SimpleModel(Normal(0.0, 1.0, name="theta"), DummyLik())
        method = SbiSMCABCMethod()
        info = method.check(model, None)
        assert not info.feasible

    def test_smcabc_rejects_non_tfp_prior(self, simulator_2d):
        class NonTFPPrior:
            _sampling_cost = "low"
            _preferred_orchestration = None

            def _sample(self, key, sample_shape=()):
                return jnp.zeros(sample_shape + (2,))

        model = SimpleGenerativeModel(NonTFPPrior(), simulator_2d)
        method = SbiSMCABCMethod()
        info = method.check(model, jnp.array([0.0, 0.0]))
        assert not info.feasible
        assert "TFP" in info.description


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


@pytest.mark.sbi
class TestAdapters:
    """Unit tests for sbijax adapter helpers."""

    def test_adapt_prior_rejects_non_tfp(self):
        class NotTFP:
            pass

        with pytest.raises(TypeError, match="TFP-backed"):
            adapt_prior(NotTFP())

    def test_adapt_simulator_matches_simulator_semantics(
        self, prior_2d, simulator_2d
    ):
        """Adapted simulator should implement y = params + 0.1 * noise per particle.

        Verifies (a) distinct per-particle output, (b) output shape matches
        the number of particles × event size, and (c) each particle's output
        is within a few noise standard deviations of its own parameter.
        """
        sim_fn = adapt_simulator(simulator_2d)
        key = jax.random.PRNGKey(0)
        params = jnp.stack([jnp.zeros(2), jnp.ones(2), 2 * jnp.ones(2)])
        out = np.asarray(sim_fn(key, {PARAM_KEY: params}))

        # Shape: (n_particles, 2)
        assert out.shape == (3, 2)
        # Independence: distinct per-particle draws.
        assert not np.allclose(out[0], out[1])
        assert not np.allclose(out[1], out[2])
        # Semantics: y_i ≈ params_i with noise std 0.1 — each output within
        # ~6σ of its own parameter with huge margin (deterministic given key).
        for i in range(3):
            assert np.linalg.norm(out[i] - np.asarray(params[i])) < 1.0
        # Particle i's output should be closest to params_i, not other params.
        for i in range(3):
            dists = [np.linalg.norm(out[i] - np.asarray(params[j])) for j in range(3)]
            assert np.argmin(dists) == i


# ---------------------------------------------------------------------------
# Side-effect / robustness tests
# ---------------------------------------------------------------------------


@pytest.mark.sbi
def test_sbijax_import_preserves_rcparams():
    """Importing the sbijax adapters must not flip matplotlib rcParams."""
    import matplotlib as mpl
    import probpipe.inference._sbijax  # noqa: F401
    # sbijax's stylesheet touches these; all should be at matplotlib defaults.
    assert mpl.rcParams["text.usetex"] is False
    assert "cmr10" not in str(mpl.rcParams.get("font.family", ""))
