"""Tests for the BlackJAX-backed elliptical slice sampling method.

Covers:

* The Gaussian-prior detection helper (``_gaussian_prior_params``)
  against the recognised shapes (``Normal``, ``MultivariateNormal``,
  ``ProductDistribution`` over both) and the rejected shapes
  (non-Gaussian families, bare ``Distribution`` subclasses).
* ``check()`` infeasibility messages for the three failure modes:
  bare ``SupportsLogProb`` (no SimpleModel decomposition), non-Gaussian
  prior, and missing observed data.
* End-to-end posterior recovery on the conjugate Normal-Normal target
  (1-D) and a multivariate-Normal-prior + Gaussian-likelihood target
  (5-D anisotropic) against the closed-form posterior.
* Auxiliary DataTree contents and provenance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Beta,
    Gamma,
    MultivariateNormal,
    Normal,
    ProductDistribution,
    SimpleModel,
)
from probpipe.inference import (
    elliptical_slice,
    inference_method_registry,
)
from probpipe.inference._blackjax_ess import (
    BlackJAXESSMethod,
    _gaussian_prior_params,
)
from probpipe.modeling._likelihood import Likelihood

pytestmark = pytest.mark.filterwarnings(
    "ignore:shape requires ndarray or scalar arguments:DeprecationWarning",
)


# ---------------------------------------------------------------------------
# Gaussian-prior detection
# ---------------------------------------------------------------------------


class TestGaussianPriorDetection:
    """``_gaussian_prior_params`` recognises the documented Gaussian shapes."""

    def test_normal_scalar(self):
        params = _gaussian_prior_params(Normal(loc=1.5, scale=0.5, name="x"))
        assert params is not None
        mean, cov = params
        np.testing.assert_allclose(np.asarray(mean), [1.5])
        np.testing.assert_allclose(np.asarray(cov), [[0.25]])

    def test_multivariate_normal_diag(self):
        prior = MultivariateNormal(
            loc=jnp.array([0.0, 1.0]),
            cov=jnp.diag(jnp.array([1.0, 4.0])),
            name="m",
        )
        params = _gaussian_prior_params(prior)
        assert params is not None
        mean, cov = params
        np.testing.assert_allclose(np.asarray(mean), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(cov), [[1.0, 0.0], [0.0, 4.0]])

    def test_multivariate_normal_dense(self):
        cov_in = jnp.array([[1.0, 0.3], [0.3, 2.0]])
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=cov_in, name="m")
        params = _gaussian_prior_params(prior)
        assert params is not None
        _, cov = params
        np.testing.assert_allclose(np.asarray(cov), np.asarray(cov_in))

    def test_product_of_normals_block_diagonal(self):
        prior = ProductDistribution(
            a=Normal(loc=1.0, scale=0.5, name="a"),
            b=Normal(loc=-2.0, scale=0.7, name="b"),
        )
        params = _gaussian_prior_params(prior)
        assert params is not None
        mean, cov = params
        np.testing.assert_allclose(np.asarray(mean), [1.0, -2.0])
        np.testing.assert_allclose(
            np.asarray(cov), [[0.25, 0.0], [0.0, 0.49]], rtol=1e-5,
        )

    def test_product_mixed_normal_and_mvn(self):
        prior = ProductDistribution(
            theta=Normal(loc=0.0, scale=1.0, name="theta"),
            beta=MultivariateNormal(
                loc=jnp.zeros(2),
                cov=jnp.diag(jnp.array([0.5, 2.0])),
                name="beta",
            ),
        )
        params = _gaussian_prior_params(prior)
        assert params is not None
        mean, cov = params
        assert mean.shape == (3,)
        assert cov.shape == (3, 3)
        # Theta comes first (field order), then the two MVN components.
        np.testing.assert_allclose(np.asarray(mean), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            np.asarray(cov),
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 2.0]],
            atol=1e-6,
        )

    def test_gamma_returns_none(self):
        assert _gaussian_prior_params(Gamma(concentration=2.0, rate=1.0, name="g")) is None

    def test_beta_returns_none(self):
        assert _gaussian_prior_params(Beta(alpha=2.0, beta=2.0, name="b")) is None

    def test_product_with_non_gaussian_component_returns_none(self):
        prior = ProductDistribution(
            a=Normal(loc=0.0, scale=1.0, name="a"),
            b=Gamma(concentration=2.0, rate=1.0, name="b"),
        )
        assert _gaussian_prior_params(prior) is None


# ---------------------------------------------------------------------------
# Registration + feasibility check
# ---------------------------------------------------------------------------


class TestRegistration:

    def test_method_registered_at_75(self):
        names = inference_method_registry.list_methods()
        assert "blackjax_elliptical_slice" in names
        assert (
            inference_method_registry
            .get_method("blackjax_elliptical_slice").priority
            == 75
        )


class TestFeasibilityCheck:
    """``check()`` infeasibility messages cover the three failure modes."""

    def test_rejects_bare_distribution(self):
        m = BlackJAXESSMethod()
        info = m.check(Normal(loc=0.0, scale=1.0, name="x"), jnp.zeros(5))
        assert not info.feasible
        assert "SimpleModel" in info.description

    def test_rejects_non_gaussian_prior(self):
        prior = Gamma(concentration=2.0, rate=1.0, name="g")

        class _Lik(Likelihood):
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        model = SimpleModel(prior, _Lik(), name="m")
        info = BlackJAXESSMethod().check(model, jnp.zeros(5))
        assert not info.feasible
        assert "Gaussian" in info.description

    def test_rejects_missing_data(self):
        prior = Normal(loc=0.0, scale=1.0, name="mu")

        class _Lik(Likelihood):
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        model = SimpleModel(prior, _Lik(), name="m")
        info = BlackJAXESSMethod().check(model, None)
        assert not info.feasible
        assert "observed data" in info.description

    def test_rejects_dict_observed(self):
        prior = Normal(loc=0.0, scale=1.0, name="mu")

        class _Lik(Likelihood):
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        model = SimpleModel(prior, _Lik(), name="m")
        info = BlackJAXESSMethod().check(model, {"y": jnp.zeros(5)})
        assert not info.feasible
        assert "dict" in info.description

    def test_accepts_gaussian_simple_model(self):
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="m")

        class _Lik(Likelihood):
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        model = SimpleModel(prior, _Lik(), name="m")
        info = BlackJAXESSMethod().check(model, jnp.zeros((5, 2)))
        assert info.feasible


class _NonTraceableGaussianLik(Likelihood):
    """Gaussian-shaped likelihood whose body is *not* JAX-traceable.

    The ``np.asarray(...)`` + Python ``float`` coercion forces a
    concrete value, so ``jax.make_jaxpr`` can't trace it — standing in
    for a BridgeStan / scipy / external-simulator likelihood.
    """

    def log_likelihood(self, params, data):
        mu = params["mu"] if hasattr(params, "fields") else params
        resid = np.asarray(data) - np.asarray(mu)
        return float(-0.5 * np.sum(resid ** 2))


class TestDeclinesToRWMH:
    """When ESS declines, auto-dispatch must fall through to RWMH.

    ESS (priority 75) outranks RWMH (55), but ESS requires a
    JAX-traceable likelihood. With a Gaussian prior + non-traceable
    likelihood, the gradient methods (NUTS/HMC) also decline, so the
    highest-priority *feasible* method is ``blackjax_rwmh`` (which has
    an eager Python-loop fallback for exactly this case).
    """

    def _model(self):
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="mu")
        return SimpleModel(prior, _NonTraceableGaussianLik(), name="m")

    def test_ess_check_infeasible_on_non_traceable_likelihood(self):
        model = self._model()
        info = BlackJAXESSMethod().check(model, np.zeros((5, 2)))
        assert not info.feasible
        assert "traceable" in info.description.lower()

    def test_auto_dispatch_lands_on_rwmh(self):
        from probpipe import condition_on

        model = self._model()
        # No method= → registry auto-selects. ESS (75) declines
        # (non-traceable), NUTS/HMC (gradient) decline, so RWMH (55) wins.
        posterior = condition_on(
            model, np.zeros((5, 2)),
            num_results=50, num_warmup=20, random_seed=0,
        )
        assert posterior.algorithm == "blackjax_rwmh"


# ---------------------------------------------------------------------------
# Posterior recovery on conjugate targets
# ---------------------------------------------------------------------------


class _GaussianMeanLik(Likelihood):
    """``log p(y | mu) = sum_i log N(y_i; mu, 1)`` — Gaussian likelihood."""

    def log_likelihood(self, params, data):
        mu = params["mu"] if hasattr(params, "fields") else params
        return -0.5 * jnp.sum((data - mu) ** 2)


class TestPosteriorRecovery:
    """ESS samples must match the closed-form Gaussian conjugate posterior."""

    def test_one_dim_normal_normal(self):
        """N(0, 1) prior + N(mu, 1) likelihood — posterior is N(n*y_bar/(n+1), 1/(n+1))."""
        prior = Normal(loc=0.0, scale=1.0, name="mu")
        data = jax.random.normal(jax.random.PRNGKey(11), shape=(50,)) + 0.7
        model = SimpleModel(prior, _GaussianMeanLik(), name="m")

        post = elliptical_slice(
            model, data, num_results=3000, num_warmup=500,
            num_chains=2, random_seed=42,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in post.chains], axis=0,
        )
        n = data.shape[0]
        y_bar = float(np.asarray(data).mean())
        analytic_mean = n * y_bar / (n + 1)
        analytic_sd = float(np.sqrt(1.0 / (n + 1)))
        np.testing.assert_allclose(float(draws.mean()), analytic_mean, atol=0.05)
        np.testing.assert_allclose(
            float(draws.std(ddof=1)), analytic_sd, atol=0.04,
        )

    def test_multivariate_anisotropic_prior(self):
        """5-D MVN prior with non-trivial covariance.

        Closed-form posterior precision is
        ``Lambda_post = Lambda_prior + n * Lambda_lik``.
        Here ``Lambda_lik = I`` (unit-variance Gaussian likelihood) and
        ``Lambda_prior = inv(Sigma_prior)``.
        """
        d = 5
        rng = np.random.default_rng(0)
        # Random PSD prior covariance.
        A = rng.standard_normal((d, d))
        sigma_prior = np.asarray(0.5 * (A @ A.T + 2 * np.eye(d)))
        prior_mean_arr = np.zeros(d)
        prior = MultivariateNormal(
            loc=jnp.asarray(prior_mean_arr),
            cov=jnp.asarray(sigma_prior),
            name="theta",
        )

        n = 80
        truth = np.array([0.5, -0.2, 0.0, 0.7, -0.5])
        data = jnp.asarray(
            rng.standard_normal((n, d)) + truth,
        )

        class _MVNLik(Likelihood):
            def log_likelihood(self, params, data):
                theta = params["theta"] if hasattr(params, "fields") else params
                return -0.5 * jnp.sum((data - theta) ** 2)

        model = SimpleModel(prior, _MVNLik(), name="m")

        post = elliptical_slice(
            model, data, num_results=2000, num_warmup=500,
            num_chains=2, random_seed=42,
        )
        draws = np.concatenate(
            [np.asarray(c) for c in post.chains], axis=0,
        )

        # Closed-form posterior.
        lam_prior = np.linalg.inv(sigma_prior)
        lam_post = lam_prior + n * np.eye(d)
        sigma_post = np.linalg.inv(lam_post)
        y_bar = np.asarray(data).mean(axis=0)
        post_mean = sigma_post @ (lam_prior @ prior_mean_arr + n * y_bar)

        np.testing.assert_allclose(draws.mean(0), post_mean, atol=0.1)
        sample_cov = np.cov(draws, rowvar=False)
        frob = np.linalg.norm(sample_cov - sigma_post, ord="fro")
        np.testing.assert_array_less(
            frob, 0.15 * np.linalg.norm(sigma_post, ord="fro"),
        )


# ---------------------------------------------------------------------------
# Provenance and auxiliary
# ---------------------------------------------------------------------------


class TestProvenanceAndAuxiliary:

    def test_provenance(self):
        prior = Normal(loc=0.0, scale=1.0, name="mu")
        model = SimpleModel(prior, _GaussianMeanLik(), name="m")
        data = jnp.zeros(10)
        post = elliptical_slice(
            model, data, num_results=50, num_warmup=20, random_seed=0,
        )
        assert post.algorithm == "elliptical_slice"
        assert post.source.operation == "elliptical_slice"

    def test_auxiliary_datatree_has_subiter_stats(self):
        prior = Normal(loc=0.0, scale=1.0, name="mu")
        model = SimpleModel(prior, _GaussianMeanLik(), name="m")
        data = jnp.zeros(10)
        post = elliptical_slice(
            model, data, num_results=50, num_warmup=20, random_seed=0,
        )
        assert post.inference_data is not None
        assert "posterior" in post.inference_data
        assert "sample_stats" in post.inference_data
        # The ESS-specific stat is ``subiter`` (number of bracket shrinkages).
        ss = post.inference_data["sample_stats"]
        assert "subiter" in ss.variables

    def test_warmup_stored(self):
        prior = Normal(loc=0.0, scale=1.0, name="mu")
        model = SimpleModel(prior, _GaussianMeanLik(), name="m")
        data = jnp.zeros(10)
        post = elliptical_slice(
            model, data, num_results=20, num_warmup=15,
            num_chains=2, random_seed=0,
        )
        assert post.warmup_samples is not None
        assert post.warmup_samples[0].shape == (15, 1)

    def test_multi_chain_shape(self):
        prior = Normal(loc=0.0, scale=1.0, name="mu")
        model = SimpleModel(prior, _GaussianMeanLik(), name="m")
        data = jnp.zeros(10)
        post = elliptical_slice(
            model, data, num_results=30, num_warmup=10,
            num_chains=3, random_seed=0,
        )
        assert post.num_chains == 3
        assert post.num_draws == 30


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:

    def test_raises_on_bare_distribution(self):
        with pytest.raises(TypeError, match="SimpleModel"):
            elliptical_slice(
                Normal(loc=0.0, scale=1.0, name="x"),
                jnp.zeros(5), num_results=10, num_warmup=5,
            )

    def test_raises_on_non_gaussian_prior(self):
        prior = Gamma(concentration=2.0, rate=1.0, name="g")

        class _Lik(Likelihood):
            def log_likelihood(self, params, data):
                return jnp.asarray(0.0)

        model = SimpleModel(prior, _Lik(), name="m")
        with pytest.raises(TypeError, match="Gaussian"):
            elliptical_slice(
                model, jnp.zeros(5), num_results=10, num_warmup=5,
            )
