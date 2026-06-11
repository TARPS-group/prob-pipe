"""Tests for the BayesFlow amortized-SBI backend (NPE / FMPE / CMPE).

Requires the ``[bayesflow]`` extra (Python 3.12-3.13); skipped otherwise. Uses a
small, fast-training toy fixture so the suite stays tractable.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
pytest.importorskip("bayesflow")

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import (
    ApproximateDistribution,
    Normal,
    ProductDistribution,
    condition_on,
    learn_amortized_posterior,
)
from probpipe.modeling import GenerativeLikelihood, Likelihood


class _ToyLikelihood(Likelihood, GenerativeLikelihood):
    """Identifiable 2-parameter model: ``y = [a + b, a - b] + small noise``."""

    # ``log_likelihood`` is unused by the amortized path (only ``generate_data``
    # is called); stubbed here just to satisfy the ``Likelihood`` protocol.
    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        t = params.flatten()        # structured record (training) or raw array (direct)
        a, b = t[0], t[1]
        key = key if key is not None else jax.random.PRNGKey(0)
        mean = jnp.stack([a + b, a - b])
        return mean[None, :] + 0.1 * jax.random.normal(key, (num_observations, 2))


def _prior():
    return ProductDistribution(
        Normal(loc=0.0, scale=1.0, name="a"),
        Normal(loc=0.0, scale=1.0, name="b"),
    )


def _observe(a, b, seed):
    return _ToyLikelihood().generate_data(jnp.array([a, b]), 1, key=jax.random.PRNGKey(seed))[0]


class _VecLikelihood(Likelihood, GenerativeLikelihood):
    """Vector + scalar params [m0, m1, s]; y = [m0 + s, m1 - s] + small noise."""

    # ``log_likelihood`` is unused by the amortized path (see ``_ToyLikelihood``).
    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        t = params.flatten()
        m0, m1, s = t[0], t[1], t[2]
        key = key if key is not None else jax.random.PRNGKey(0)
        mean = jnp.stack([m0 + s, m1 - s])
        return mean[None, :] + 0.1 * jax.random.normal(key, (num_observations, 2))


def _vec_prior():
    import probpipe as pp

    return ProductDistribution(
        pp.MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="m"),
        Normal(loc=0.0, scale=1.0, name="s"),
    )


class _SingleFieldLikelihood(Likelihood, GenerativeLikelihood):
    """Single (vector) parameter field: ``y = theta + small noise``."""

    # ``log_likelihood`` is unused by the amortized path (see ``_ToyLikelihood``).
    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        t = params.flatten()
        key = key if key is not None else jax.random.PRNGKey(0)
        return t[None, :] + 0.1 * jax.random.normal(key, (num_observations, 2))


class _NonJaxLikelihood(Likelihood, GenerativeLikelihood):
    """A deliberately non-vmappable simulator: it concretizes the parameters
    (``float(...)``) and draws noise with numpy, so it runs only on the eager
    path (``sim_backend="sequential"``) -- ``jax.vmap`` would raise a tracer error."""

    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        t = np.asarray(params.flatten())
        a, b = float(t[0]), float(t[1])
        seed = 0 if key is None else int(jax.random.randint(key, (), 0, 2**16))
        noise = np.random.default_rng(seed).standard_normal((num_observations, 2))
        return np.array([a + b, a - b]) + 0.1 * noise


class _ScalarLikelihood(Likelihood, GenerativeLikelihood):
    """One-parameter model: ``y = a + small noise`` (a single scalar param)."""

    # ``log_likelihood`` is unused by the amortized path (see ``_ToyLikelihood``).
    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        a = params.flatten()[0]
        key = key if key is not None else jax.random.PRNGKey(0)
        return a + 0.1 * jax.random.normal(key, (num_observations, 1))


class _MultiFieldLikelihood(Likelihood, GenerativeLikelihood):
    """Three-field params (flat ``[a, b0, b1, c]``) -> an 8-d observation built
    from several param combinations, to exercise multiple mixed-shape parameter
    fields and higher-dimensional data."""

    # ``log_likelihood`` is unused by the amortized path (see ``_ToyLikelihood``).
    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        t = params.flatten()
        a, b0, b1, c = t[0], t[1], t[2], t[3]
        key = key if key is not None else jax.random.PRNGKey(0)
        mean = jnp.stack([a + b0, a - b0, b1 + c, b1 - c, a + c, b0 * b1, a, c])
        return mean[None, :] + 0.1 * jax.random.normal(key, (num_observations, mean.shape[0]))


def _multi_field_prior():
    import probpipe as pp

    return ProductDistribution(
        Normal(loc=0.0, scale=1.0, name="a"),
        pp.MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="b"),
        Normal(loc=0.0, scale=1.0, name="c"),
    )


# Conjugate Gaussian with an analytic posterior, used to check calibration.
_CONJ_SIGMA = 0.5


class _ConjugateGaussianLikelihood(Likelihood, GenerativeLikelihood):
    """Conjugate model (any dimension): prior ``theta ~ N(0, I)``, ``y = theta +
    sigma * noise``. The posterior is analytic -- ``N(y / (1 + sigma^2), sigma^2 /
    (1 + sigma^2) I)`` -- so the amortized posterior's mean *and* spread can be
    checked against it. The observation dimension follows the parameter dimension,
    so one likelihood serves both the 2-D and 1-D calibration tests."""

    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        t = params.flatten()
        return t[None, :] + _CONJ_SIGMA * jax.random.normal(key, (num_observations, t.shape[-1]))


class _NamedFieldLikelihood(Likelihood, GenerativeLikelihood):
    """Accesses params strictly by field name (``params["a"]``), never positionally
    -- locks the ``GenerativeLikelihood`` contract that training passes the prior's
    structured per-draw record (a flattened-vector regression raises here)."""

    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        a, b = params["a"], params["b"]
        key = key if key is not None else jax.random.PRNGKey(0)
        mean = jnp.stack([a + b, a - b])
        return mean[None, :] + 0.1 * jax.random.normal(key, (num_observations, 2))


class _PositiveLikelihood(Likelihood, GenerativeLikelihood):
    """Simulator for a constrained prior (positive ``r``, real ``m``):
    ``y = [r + m, r - m] + small noise``."""

    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        t = params.flatten()
        r, m = t[0], t[1]
        key = key if key is not None else jax.random.PRNGKey(0)
        mean = jnp.stack([r + m, r - m])
        return mean[None, :] + 0.1 * jax.random.normal(key, (num_observations, 2))


@pytest.fixture(scope="module")
def npe_model():
    """A briefly-trained NPE estimator, shared across the NPE tests."""
    return learn_amortized_posterior(
        _prior(), _ToyLikelihood(), method="npe",
        num_simulations=3000, epochs=6, batch_size=256,
        num_results=500, random_seed=0, verbose=0,
    )


class TestBayesFlowNPE:
    def test_recovery(self, npe_model):
        """The amortized posterior concentrates near the truth -- in *both*
        parameters. Both truths sit >0.5 from the prior mean (0), so neither
        assertion passes without the model actually learning the parameter."""
        post = condition_on(npe_model, _observe(0.6, -0.6, seed=7))
        draws = post.draws()
        a = float(np.mean(np.asarray(draws["a"])))
        b = float(np.mean(np.asarray(draws["b"])))
        # Loose, calibration-style tolerance (brief training, stochastic).
        assert abs(a - 0.6) < 0.5
        assert abs(b - (-0.6)) < 0.5

    def test_amortization(self, npe_model):
        """The same trained model conditions on distinct observations, and the
        posterior means land on the correct side of zero (the defining property
        of amortized inference)."""
        mean_a_hi = float(np.mean(np.asarray(condition_on(npe_model, _observe(1.0, 0.0, 2)).draws()["a"])))
        mean_a_lo = float(np.mean(np.asarray(condition_on(npe_model, _observe(-1.0, 0.0, 3)).draws()["a"])))
        assert mean_a_hi > 0 > mean_a_lo

    def test_contract(self, npe_model):
        """``condition_on`` honours ``num_results`` (and its model default) and
        ignores the MCMC-only kwargs; the result is a named
        ``ApproximateDistribution``."""
        post = condition_on(npe_model, _observe(0.0, 0.0, 1),
                            num_results=300, num_warmup=99, num_chains=4)
        assert isinstance(post, ApproximateDistribution)
        assert post.algorithm == "bayesflow_npe"
        draws = post.draws()
        # Named fields (record_template lifting), 300 draws, no warmup/chains effect.
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 300
        assert np.isfinite(np.asarray(draws["b"])).all()
        # Omitting num_results falls back to the model default (500 here).
        default_post = condition_on(npe_model, _observe(0.0, 0.0, 1))
        assert np.asarray(default_post.draws()["a"]).reshape(-1).shape[0] == 500

    def test_observation_dim_mismatch(self, npe_model):
        """Conditioning on a wrong-size observation raises a clear error rather
        than an opaque keras shape failure."""
        with pytest.raises(ValueError, match="trained on observations of size"):
            condition_on(npe_model, np.zeros(5, dtype="float32"))

    @pytest.mark.parametrize("bad", [0, -5])
    def test_condition_rejects_nonpositive_num_results(self, npe_model, bad):
        """Sample-time num_results is validated like the train-time one: zero or
        negative fails fast at the ProbPipe boundary."""
        with pytest.raises(ValueError, match="positive integer"):
            condition_on(npe_model, _observe(0.0, 0.0, 2), num_results=bad)

    def test_forward_sampling_from_joint(self, npe_model):
        """The model represents the joint p(theta, y): sampling draws
        (params, data) via prior + simulator -- params carry the prior's named
        fields, data has the simulator's output shape. Batched draws are
        rejected like SimpleGenerativeModel's."""
        from probpipe import SupportsSampling
        assert isinstance(npe_model, SupportsSampling)
        params, data = npe_model._sample(jax.random.PRNGKey(0))
        assert np.isfinite(float(params["a"]))
        assert np.isfinite(float(params["b"]))
        assert np.asarray(data).shape == (2,)   # _ToyLikelihood's per-dataset shape
        with pytest.raises(NotImplementedError, match="sample_shape"):
            npe_model._sample(jax.random.PRNGKey(0), (3,))

    def test_repr(self, npe_model):
        """``repr`` surfaces the method and the default draw count."""
        r = repr(npe_model)
        assert "method='npe'" in r
        assert "num_results=500" in r


class TestBayesFlowMethods:
    @pytest.mark.parametrize("method", ["npe", "fmpe", "cmpe"])
    def test_methods_smoke(self, method):
        """Each amortized method trains and conditions, returning named draws."""
        model = learn_amortized_posterior(
            _prior(), _ToyLikelihood(), method=method,
            num_simulations=1500, epochs=3, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        post = condition_on(model, _observe(0.5, 0.0, 0))
        assert post.algorithm == f"bayesflow_{method}"
        draws = post.draws()
        assert np.isfinite(np.asarray(draws["a"])).all()
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200

    def test_vector_valued_field(self):
        """A vector-valued parameter field round-trips through the per-field
        reshape/concatenate and returns named draws of the right shape."""
        model = learn_amortized_posterior(
            _vec_prior(), _VecLikelihood(), method="npe",
            num_simulations=2000, epochs=4, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        obs = _VecLikelihood().generate_data(jnp.array([0.5, -0.5, 0.2]), 1,
                                             key=jax.random.PRNGKey(5))[0]
        draws = condition_on(model, obs).draws()
        m = np.asarray(draws["m"]).reshape(200, -1)
        s = np.asarray(draws["s"]).reshape(200, -1)
        assert m.shape == (200, 2)  # the (2,)-vector field is preserved
        assert s.shape == (200, 1)
        assert np.isfinite(m).all() and np.isfinite(s).all()

    def test_custom_inference_network(self):
        """A caller-supplied ``inference_network`` overrides the method default
        and is the network actually wired into the trained approximator."""
        import bayesflow as bf

        net = bf.networks.CouplingFlow()
        model = learn_amortized_posterior(
            _prior(), _ToyLikelihood(), method="npe", inference_network=net,
            num_simulations=1500, epochs=3, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        # The exact instance passed in is the one used (not a method default).
        assert model._approximator.inference_network is net
        post = condition_on(model, _observe(0.5, 0.0, 0))
        assert np.asarray(post.draws()["a"]).reshape(-1).shape[0] == 200

    def test_single_field_prior(self):
        """A single-field prior (not a ProductDistribution) is supported: its
        draws are not field-indexable, but the canonical flat layout drives the
        per-field split, so it round-trips end-to-end to named draws."""
        import probpipe as pp

        prior = pp.MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta")
        model = learn_amortized_posterior(
            prior, _SingleFieldLikelihood(), method="npe",
            num_simulations=1500, epochs=3, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        obs = _SingleFieldLikelihood().generate_data(jnp.array([0.5, -0.5]), 1,
                                                     key=jax.random.PRNGKey(4))[0]
        draws = condition_on(model, obs).draws()
        assert np.asarray(draws["theta"]).reshape(200, -1).shape == (200, 2)
        assert np.isfinite(np.asarray(draws["theta"])).all()

    def test_non_jax_simulator(self):
        """A non-vmappable (non-JAX) simulator trains via the eager path
        (``sim_backend="sequential"``) and conditions to named draws -- ``vmap``
        would fail on it, so success proves the eager loop ran."""
        model = learn_amortized_posterior(
            _prior(), _NonJaxLikelihood(), method="npe", sim_backend="sequential",
            num_simulations=800, epochs=2, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        post = condition_on(model, _observe(0.5, 0.0, 0))
        assert post.algorithm == "bayesflow_npe"
        draws = post.draws()
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200
        assert np.isfinite(np.asarray(draws["a"])).all()

    def test_scalar_prior_fmpe(self):
        """A one-parameter (scalar) prior round-trips with FMPE: the single-field
        split handles ``event_shape=()``, and flow matching (unlike NPE's
        coupling flow) has no >= 2-parameter requirement."""
        model = learn_amortized_posterior(
            Normal(loc=0.0, scale=1.0, name="a"), _ScalarLikelihood(), method="fmpe",
            num_simulations=1500, epochs=3, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        obs = _ScalarLikelihood().generate_data(jnp.array([0.7]), 1,
                                                key=jax.random.PRNGKey(4))[0]
        draws = condition_on(model, obs).draws()
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200
        assert np.isfinite(np.asarray(draws["a"])).all()

    def test_npe_one_param_fallback_calibration(self):
        """The NPE one-parameter fallback is held to the same calibration standard
        as the multi-parameter path. A coupling flow can't split a 1-D vector, so
        NPE at d=1 falls back to a flow-matching network; here, against a 1-D
        conjugate Gaussian (analytic posterior), the amortized posterior mean and
        spread are checked against the analytic mean and std. The bounds are looser
        than ``test_calibration_against_conjugate_gaussian`` because flow matching
        at d=1 is a measurably noisier estimator -- its ODE sampling is less exact
        than a coupling flow's density (across training seeds the mean error spans
        ~0.04-0.19 posterior-std and the std ratio ~0.92-1.18). Training is seeded,
        so a given run is reproducible; the band absorbs that estimator imprecision
        plus cross-platform / library-version drift."""
        prior = Normal(loc=0.0, scale=1.0, name="a")   # event_size 1 -> FlowMatching
        model = learn_amortized_posterior(
            prior, _ConjugateGaussianLikelihood(), method="npe",
            num_simulations=5000, epochs=40, batch_size=256,
            num_results=2000, random_seed=0, verbose=0,
        )
        s2 = _CONJ_SIGMA**2
        post_std = (s2 / (1 + s2)) ** 0.5          # analytic posterior std
        sim = _ConjugateGaussianLikelihood()
        mean_errs, std_ratios = [], []
        for i in range(6):
            theta = jax.random.normal(jax.random.PRNGKey(100 + i), (1,))
            obs = sim.generate_data(theta, 1, key=jax.random.PRNGKey(900 + i))[0]
            x = np.asarray(condition_on(model, obs).draws()["a"]).reshape(-1)
            mean_errs.append(abs(float(x.mean()) - float(obs[0]) / (1 + s2)))
            std_ratios.append(float(x.std()) / post_std)
        # Estimate: mean posterior-mean error under 0.5 posterior-std.
        assert np.mean(mean_errs) < 0.5 * post_std
        # Uncertainty: mean std ratio in [0.7, 1.4] (flow matching at d=1 tends to
        # slightly under-disperse; band bounds the measured cross-seed spread).
        assert 0.7 < np.mean(std_ratios) < 1.4

    def test_multi_field_prior_and_data(self):
        """A richer scenario: three parameter fields (scalar + 2-vector + scalar)
        and a higher-dimensional (8-d) observation. Exercises the per-field
        split, adapter routing, and posterior assembly with multiple mixed-shape
        fields and bigger data, and checks the posterior responds to the data."""
        model = learn_amortized_posterior(
            _multi_field_prior(), _MultiFieldLikelihood(), method="npe",
            num_simulations=2500, epochs=5, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        sim = _MultiFieldLikelihood()
        obs = sim.generate_data(jnp.array([0.5, -0.5, 0.3, -0.2]), 1,
                                key=jax.random.PRNGKey(6))[0]
        assert obs.shape == (8,)  # higher-dimensional observation
        draws = condition_on(model, obs).draws()
        assert np.asarray(draws["a"]).reshape(200, -1).shape == (200, 1)
        assert np.asarray(draws["b"]).reshape(200, -1).shape == (200, 2)  # 2-vector field
        assert np.asarray(draws["c"]).reshape(200, -1).shape == (200, 1)
        assert all(np.isfinite(np.asarray(draws[f])).all() for f in ("a", "b", "c"))
        # Amortized response: the posterior mean of `a` tracks the observation.
        obs_hi = sim.generate_data(jnp.array([1.0, 0.0, 0.0, 0.0]), 1, key=jax.random.PRNGKey(1))[0]
        obs_lo = sim.generate_data(jnp.array([-1.0, 0.0, 0.0, 0.0]), 1, key=jax.random.PRNGKey(2))[0]
        mean_a_hi = float(np.mean(np.asarray(condition_on(model, obs_hi).draws()["a"])))
        mean_a_lo = float(np.mean(np.asarray(condition_on(model, obs_lo).draws()["a"])))
        assert mean_a_hi > mean_a_lo

    def test_calibration_against_conjugate_gaussian(self):
        """Estimate *and* uncertainty are roughly correct: against a conjugate
        Gaussian whose posterior is analytic, the amortized posterior mean tracks
        the analytic mean and its std matches the analytic std (averaged over
        several observations). Trains a bit longer than the smoke tests so the
        estimator is near-converged."""
        prior = ProductDistribution(Normal(loc=0.0, scale=1.0, name="a"),
                                    Normal(loc=0.0, scale=1.0, name="b"))
        model = learn_amortized_posterior(
            prior, _ConjugateGaussianLikelihood(), method="npe",
            num_simulations=5000, epochs=40, batch_size=256,
            num_results=4000, random_seed=0, verbose=0,
        )
        s2 = _CONJ_SIGMA**2
        post_std = (s2 / (1 + s2)) ** 0.5          # analytic posterior std
        sim = _ConjugateGaussianLikelihood()
        mean_errs, std_ratios = [], []
        for i in range(6):
            theta = jax.random.normal(jax.random.PRNGKey(100 + i), (2,))
            obs = sim.generate_data(theta, 1, key=jax.random.PRNGKey(900 + i))[0]
            draws = condition_on(model, obs).draws()
            for j, f in enumerate(("a", "b")):
                x = np.asarray(draws[f]).reshape(-1)
                analytic_mean = float(obs[j]) / (1 + s2)
                mean_errs.append(abs(float(x.mean()) - analytic_mean))
                std_ratios.append(float(x.std()) / post_std)
        # Training is seeded (reproducible); the margins absorb cross-platform /
        # library-version numerical drift. Estimate: mean posterior-mean error under
        # 0.3 posterior-std (observed ~0.03-0.11 across training seeds).
        assert np.mean(mean_errs) < 0.3 * post_std
        # Uncertainty: mean std ratio in [0.8, 1.25] (observed ~1.01-1.03 across seeds).
        assert 0.8 < np.mean(std_ratios) < 1.25

    def test_simulator_receives_named_record(self):
        """generate_data receives the prior's structured per-draw sample (named
        fields), per the GenerativeLikelihood contract -- the simulator uses
        params["a"]/["b"] exclusively, so training succeeds only when the
        structured record is passed."""
        model = learn_amortized_posterior(
            _prior(), _NamedFieldLikelihood(), method="npe",
            num_simulations=800, epochs=2, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        draws = condition_on(model, jnp.array([0.8, 0.2])).draws()
        assert np.asarray(draws["a"]).reshape(-1).shape[0] == 200
        assert np.isfinite(np.asarray(draws["a"])).all()

    def test_constrained_prior_draws_respect_support(self):
        """A constrained (positive) prior field is trained in unconstrained space and
        its draws are mapped back through the forward bijector, so they land in the
        support -- here all positive. The accompanying real-valued field is unaffected."""
        import probpipe as pp
        prior = ProductDistribution(pp.Gamma(3.0, 1.0, name="r"),
                                    Normal(loc=0.0, scale=1.0, name="m"))
        model = learn_amortized_posterior(
            prior, _PositiveLikelihood(), method="npe",
            num_simulations=1500, epochs=5, batch_size=256,
            num_results=400, random_seed=0, verbose=0,
        )
        r = np.asarray(condition_on(model, jnp.array([3.0, 1.0])).draws()["r"]).reshape(-1)
        assert np.isfinite(r).all()
        assert (r > 0).all()        # forward bijector (Exp) keeps every draw in support

    def test_interval_prior_draws_respect_support(self):
        """A bounded-interval prior field (Beta, unit-interval support) rounds
        through the Sigmoid bijector: trained unconstrained, every posterior
        draw lands strictly inside (0, 1)."""
        import probpipe as pp
        prior = ProductDistribution(pp.Beta(2.0, 2.0, name="q"),
                                    Normal(loc=0.0, scale=1.0, name="m"))
        model = learn_amortized_posterior(
            prior, _ConjugateGaussianLikelihood(), method="npe",
            num_simulations=800, epochs=2, batch_size=256,
            num_results=300, random_seed=0, verbose=0,
        )
        q = np.asarray(condition_on(model, jnp.array([0.5, 0.0])).draws()["q"]).reshape(-1)
        assert np.isfinite(q).all()
        assert ((q > 0) & (q < 1)).all()   # Sigmoid forward keeps draws in (0, 1)

    def test_wishart_matrix_prior_round_trip(self):
        """A matrix-valued constrained field (Wishart, positive-definite support)
        round-trips: the inverse bijector runs at the field's native (n, n) event
        shape at train time -- a flattened input would crash the
        CholeskyOuterProduct chain -- and the forward map at sample time returns
        draws that are symmetric positive definite."""
        import probpipe as pp
        prior = ProductDistribution(pp.Wishart(df=4.0, scale=jnp.eye(2), name="cov"),
                                    Normal(loc=0.0, scale=1.0, name="m"))
        model = learn_amortized_posterior(
            prior, _ConjugateGaussianLikelihood(), method="fmpe",
            num_simulations=600, epochs=2, batch_size=256,
            num_results=200, random_seed=0, verbose=0,
        )
        obs = jnp.array([1.5, 0.3, 0.3, 1.2, 0.5])    # flattened (cov, m) observation
        cov = np.asarray(condition_on(model, obs).draws()["cov"]).reshape(-1, 2, 2)
        assert np.isfinite(cov).all()
        assert np.allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-5)   # symmetric
        assert np.linalg.eigvalsh(cov).min() > 0                       # positive definite

    def test_dirichlet_simplex_prior_npe(self):
        """A single 2-simplex (Dirichlet) prior under method='npe': the coupling-flow
        guard counts *unconstrained* dimensions (one here, not the constrained
        event_size of two), so NPE falls back to flow matching instead of building a
        units=0 coupling flow; draws land on the simplex."""
        import probpipe as pp
        model = learn_amortized_posterior(
            pp.Dirichlet(jnp.ones(2), name="p"), _ConjugateGaussianLikelihood(),
            method="npe", num_simulations=600, epochs=2, batch_size=256,
            num_results=300, random_seed=0, verbose=0,
        )
        p = np.asarray(condition_on(model, jnp.array([0.7, 0.3])).draws()["p"]).reshape(-1, 2)
        assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-5)
        assert ((p > 0) & (p < 1)).all()

    def test_global_rng_state_restored(self):
        """Training seeds keras via the global RNG but snapshots and restores the
        caller's NumPy / Python random state, so a call does not silently make the
        caller's unrelated random streams deterministic."""
        import random as pyrandom
        np.random.seed(123)
        pyrandom.seed(7)
        expected_np = np.random.random()
        expected_py = pyrandom.random()
        np.random.seed(123)
        pyrandom.seed(7)
        learn_amortized_posterior(
            _prior(), _ToyLikelihood(), method="fmpe",
            num_simulations=256, epochs=1, batch_size=256,
            num_results=50, random_seed=0, verbose=0,
        )
        assert np.random.random() == expected_np
        assert pyrandom.random() == expected_py


class TestBayesFlowValidation:
    """Train-time input validation -- each raises before any simulation runs."""

    def test_rejects_reserved_field_name(self):
        """A prior field named like the reserved observation key is rejected."""
        bad_prior = ProductDistribution(
            Normal(loc=0.0, scale=1.0, name="observation"),
            Normal(loc=0.0, scale=1.0, name="b"),
        )
        with pytest.raises(ValueError, match="reserved"):
            learn_amortized_posterior(bad_prior, _ToyLikelihood(), num_simulations=8, epochs=1)

    def test_rejects_non_generative_simulator(self):
        """A simulator without ``generate_data`` is rejected with a clear TypeError."""

        class _NoGenerate:
            pass

        with pytest.raises(TypeError, match="generate_data"):
            learn_amortized_posterior(_prior(), _NoGenerate(), num_simulations=8, epochs=1)

    def test_rejects_non_record_prior(self):
        """A prior that is not a RecordDistribution (here a raw array, with no
        ``record_template``) is rejected with a clear TypeError."""
        with pytest.raises(TypeError, match="RecordDistribution"):
            learn_amortized_posterior(
                jnp.zeros(2), _ToyLikelihood(), num_simulations=8, epochs=1,
            )

    def test_rejects_unknown_method(self):
        """An unsupported amortized method is rejected up front."""
        with pytest.raises(ValueError, match="Unknown amortized SBI method"):
            learn_amortized_posterior(
                _prior(), _ToyLikelihood(), method="bogus",
                num_simulations=8, epochs=1,
            )

    def test_rejects_unknown_sim_backend(self):
        """An unsupported simulation backend is rejected up front."""
        with pytest.raises(ValueError, match="Unknown sim_backend"):
            learn_amortized_posterior(
                _prior(), _ToyLikelihood(), sim_backend="bogus",
                num_simulations=8, epochs=1,
            )

    @pytest.mark.parametrize(
        "override",
        [{"num_simulations": 0}, {"batch_size": 0}, {"epochs": 0}, {"num_results": 0}],
    )
    def test_rejects_nonpositive_counts(self, override):
        """num_simulations / batch_size / epochs / num_results must be positive."""
        kwargs = {"num_simulations": 8, "epochs": 1, **override}
        with pytest.raises(ValueError, match="positive integer"):
            learn_amortized_posterior(_prior(), _ToyLikelihood(), **kwargs)

    def test_rejects_discrete_prior(self):
        """A discrete prior field has no smooth bijector to R^d and is rejected up
        front with a clear error (here a Poisson count parameter)."""
        import probpipe as pp
        bad_prior = ProductDistribution(pp.Poisson(3.0, name="k"),
                                        Normal(loc=0.0, scale=1.0, name="m"))
        with pytest.raises(ValueError, match="discrete"):
            learn_amortized_posterior(bad_prior, _ToyLikelihood(), num_simulations=8, epochs=1)
