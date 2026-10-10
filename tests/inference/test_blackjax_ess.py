"""Tests for the BlackJAX-backed elliptical slice sampling method.

Covers:

* The Gaussian-prior detection helper (``_gaussian_prior_params``)
  against the recognised shapes (``Normal``, ``MultivariateNormal``, and a
  factored joint over them — with field-order-sensitive mean and covariance
  assertions) and
  the rejected shapes (non-Gaussian families, a ``DistributionBatch`` of
  separate ``Normal`` laws).
* ``check()`` infeasibility messages for the failure modes: a target that
  is no factored joint at observed fields, a non-Gaussian prior, and a
  non-traceable likelihood.
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
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Beta,
    EmpiricalDistribution,
    Gamma,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    workflow_run,
)
from probpipe.inference import (
    elliptical_slice,
    inference_method_registry,
)
from probpipe.inference._blackjax_ess import (
    BlackJAXESSMethod,
    _gaussian_prior_params,
)
from probpipe.inference._inference_utils import observed_target
from tests._posterior import (
    arviz_data,
    flat_chains,
    method_of,
    num_chains,
    num_draws,
    warmup_samples,
)
from tests.inference._harness import validate_method
from tests.inference.canonical import ObservationKernel

pytestmark = pytest.mark.filterwarnings(
    "ignore:shape requires ndarray or scalar arguments:DeprecationWarning",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _observations(prior, shape, obs_var=1.0, *, traceable=True):
    """``y ~ N(theta, obs_var)`` entrywise over *shape*, composed with *prior*.

    ``theta`` is the concatenation of the prior's fields, in the order of its
    components, broadcast against the rows of ``y``, so the likelihood is
    conjugate for any Gaussian prior. ``obs_var > 1`` weakens the likelihood so
    the prior covariance (its cross-field structure in particular) materially
    shapes the closed-form posterior. A likelihood that is not traceable reads
    its parameters through NumPy, as a BridgeStan, SciPy, or external
    simulator likelihood would.
    """
    slots = dict(prior.event_spec.components)

    def build(**values):
        parts = [jnp.atleast_1d(values[slot]) for slot in slots]
        if not traceable:
            parts = [jnp.asarray(np.asarray(part)) for part in parts]
        theta = jnp.concatenate(parts)
        return tfd.Independent(
            tfd.Normal(jnp.broadcast_to(theta, shape), jnp.sqrt(obs_var)), len(shape)
        )

    return ObservationKernel("y", slots, NumericArraySpec(shape), build) * prior


@pytest.fixture(scope="module")
def gaussian_model():
    """A 1-D ``N(0, 1)`` prior and a Gaussian-mean likelihood over 10 observations."""
    return _observations(Normal("mu", loc=0.0, scale=1.0), (10,))


@pytest.fixture(scope="module")
def data():
    """Observed data for :func:`gaussian_model` (10 zeros)."""
    return {"y": jnp.zeros(10)}


# ---------------------------------------------------------------------------
# Gaussian-prior detection
# ---------------------------------------------------------------------------


class TestGaussianPriorDetection:
    """``_gaussian_prior_params`` recognises the documented Gaussian shapes."""

    def test_normal_scalar(self):
        params = _gaussian_prior_params(Normal("x", loc=1.5, scale=0.5))
        assert params is not None
        mean, cov = params
        np.testing.assert_allclose(np.asarray(mean), [1.5])
        np.testing.assert_allclose(np.asarray(cov), [[0.25]])

    def test_multivariate_normal_diag(self):
        prior = MultivariateNormal(
            "m",
            loc=jnp.array([0.0, 1.0]),
            cov=jnp.diag(jnp.array([1.0, 4.0])),
        )
        params = _gaussian_prior_params(prior)
        assert params is not None
        mean, cov = params
        np.testing.assert_allclose(np.asarray(mean), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(cov), [[1.0, 0.0], [0.0, 4.0]])

    def test_multivariate_normal_dense(self):
        cov_in = jnp.array([[1.0, 0.3], [0.3, 2.0]])
        loc_in = jnp.array([1.0, -2.0])
        prior = MultivariateNormal("m", loc=loc_in, cov=cov_in)
        params = _gaussian_prior_params(prior)
        assert params is not None
        mean, cov = params
        np.testing.assert_allclose(np.asarray(mean), np.asarray(loc_in))
        np.testing.assert_allclose(np.asarray(cov), np.asarray(cov_in))

    def test_normal_scalar_promoted_to_length_one(self):
        """A scalar ``Normal`` is promoted to a length-1 vector.

        This is the only path the ``jnp.atleast_1d(scale)`` branch of
        ``_gaussian_prior_params`` ever serves: a genuinely vector-valued
        ``Normal(loc=[...], scale=[...])`` cannot be constructed — the
        ``Normal`` constructor rejects a non-scalar ``loc``/``scale`` with
        a ``ValueError``.
        """
        mean, cov = _gaussian_prior_params(Normal("x", loc=2.0, scale=3.0))
        assert mean.shape == (1,)
        assert cov.shape == (1, 1)
        np.testing.assert_allclose(np.asarray(mean), [2.0])
        np.testing.assert_allclose(np.asarray(cov), [[9.0]])

    def test_a_batch_of_separate_normal_laws_returns_none(self):
        """A ``DistributionBatch`` of ``Normal`` laws is *not* recognised.

        A batch of separate laws is not a ``Normal``, so it fails the
        ``isinstance(prior, Normal)`` check and ``_gaussian_prior_params``
        returns ``None``. ESS therefore declines such priors (cf. the
        finding noted on ``test_normal_scalar_promoted_to_length_one``).
        """
        from probpipe import DistributionBatch

        batch = DistributionBatch(
            "x",
            [Normal("x", loc=0.0, scale=0.5), Normal("x", loc=1.0, scale=2.0)],
            "law",
        )
        assert not isinstance(batch, Normal)
        assert _gaussian_prior_params(batch) is None

    def test_product_of_normals_block_diagonal(self):
        prior = Normal("a", loc=1.0, scale=0.5) * Normal("b", loc=-2.0, scale=0.7)
        params = _gaussian_prior_params(prior)
        assert params is not None
        mean, cov = params
        np.testing.assert_allclose(np.asarray(mean), [1.0, -2.0])
        np.testing.assert_allclose(
            np.asarray(cov),
            [[0.25, 0.0], [0.0, 0.49]],
            rtol=1e-5,
        )

    def test_product_mixed_normal_and_mvn(self):
        """Block assembly must respect field order for *both* mean and cov.

        Distinct non-zero component means and distinct diagonal variances
        pin the block order exactly: a field-order permutation in
        ``_gaussian_prior_params`` would scramble the mean *and* the cov
        diagonal away from the field-order concatenation
        ``[theta, beta_0, beta_1]``.
        """
        prior = Normal("theta", loc=3.0, scale=1.0) * MultivariateNormal(
            "beta",
            loc=jnp.array([5.0, -1.0]),
            cov=jnp.diag(jnp.array([0.5, 2.0])),
        )
        params = _gaussian_prior_params(prior)
        assert params is not None
        mean, cov = params
        assert mean.shape == (3,)
        assert cov.shape == (3, 3)
        # Field order is [theta, beta...]; both blocks carry distinct,
        # non-zero values so the assertion detects any permutation.
        np.testing.assert_allclose(np.asarray(mean), [3.0, 5.0, -1.0])
        np.testing.assert_allclose(
            np.asarray(cov),
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 2.0]],
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "prior",
        [
            pytest.param(Gamma("g", concentration=2.0, rate=1.0), id="gamma"),
            pytest.param(Beta("b", alpha=2.0, beta=2.0), id="beta"),
        ],
    )
    def test_non_gaussian_returns_none(self, prior):
        assert _gaussian_prior_params(prior) is None

    def test_product_with_non_gaussian_component_returns_none(self):
        prior = Normal("a", loc=0.0, scale=1.0) * Gamma("b", concentration=2.0, rate=1.0)
        assert _gaussian_prior_params(prior) is None


# ---------------------------------------------------------------------------
# Registration + feasibility check
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_method_registered_at_75(self):
        names = inference_method_registry.list_methods()
        assert "blackjax_elliptical_slice" in names
        assert inference_method_registry.get_method("blackjax_elliptical_slice").priority == 75


class TestFeasibilityCheck:
    """``check()`` infeasibility messages cover the failure modes."""

    def test_rejects_bare_distribution(self):
        m = BlackJAXESSMethod()
        info = m.check(observed_target(Normal("x", loc=0.0, scale=1.0, label="x"), jnp.zeros(5)))
        assert not info.feasible
        assert "Normal 'x' conditioned on data not keyed by its fields" in info.description

    def test_rejects_non_gaussian_prior(self):
        model = _observations(Gamma("g", concentration=2.0, rate=1.0), (5,))
        info = BlackJAXESSMethod().check(observed_target(model, {"y": jnp.ones(5)}))
        assert not info.feasible
        assert "Gaussian" in info.description

    def test_rejects_missing_data(self):
        model = _observations(Normal("mu", loc=0.0, scale=1.0), (5,))
        info = BlackJAXESSMethod().check(model)
        assert not info.feasible
        assert "with no observed fields" in info.description

    def test_accepts_a_joint_with_a_gaussian_prior(self):
        model = _observations(MultivariateNormal("m", loc=jnp.zeros(2), cov=jnp.eye(2)), (5, 2))
        info = BlackJAXESSMethod().check(observed_target(model, {"y": jnp.zeros((5, 2))}))
        assert info.feasible


class TestDeclinesToRWMH:
    """When ESS declines, auto-dispatch must fall through to RWMH.

    ESS (priority 75) outranks RWMH (55), but ESS requires a
    JAX-traceable likelihood. With a Gaussian prior + non-traceable
    likelihood, the gradient methods (NUTS/HMC) also decline, so the
    highest-priority *feasible* method is ``blackjax_rwmh`` (which has
    an eager Python-loop fallback for exactly this case).
    """

    def _model(self):
        prior = MultivariateNormal("mu", loc=jnp.zeros(2), cov=jnp.eye(2))
        return _observations(prior, (5, 2), traceable=False)

    def test_ess_check_infeasible_on_non_traceable_likelihood(self):
        model = self._model()
        info = BlackJAXESSMethod().check(observed_target(model, {"y": np.zeros((5, 2))}))
        assert not info.feasible
        assert "traceable" in info.description.lower()

    def test_auto_dispatch_lands_on_rwmh(self):
        from probpipe import condition_on

        model = self._model()
        # No method= → registry auto-selects. ESS (75) declines
        # (non-traceable), NUTS/HMC (gradient) decline, so RWMH (55) wins.
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method_options={"num_results": 50, "num_warmup": 20}
            )(model, {"y": np.zeros((5, 2))})
        assert method_of(posterior) == "blackjax_rwmh"


# ---------------------------------------------------------------------------
# Posterior recovery on conjugate targets
# ---------------------------------------------------------------------------


class TestPosteriorRecovery:
    """ESS samples must match the closed-form Gaussian conjugate posterior.

    Exercises every supported prior shape for end-to-end correctness, not
    just detection: scalar ``Normal``, dense ``MultivariateNormal``, an
    independent factored joint, and a strongly correlated
    ``MultivariateNormal``.
    """

    def test_one_dim_normal_normal(self):
        """N(0, 1) prior + N(mu, 1) likelihood — posterior is N(n*y_bar/(n+1), 1/(n+1))."""
        prior = Normal("mu", loc=0.0, scale=1.0)
        data = jax.random.normal(jax.random.PRNGKey(11), shape=(50,)) + 0.7
        model = _observations(prior, data.shape)

        with workflow_run(seed=42):
            post = elliptical_slice(
                model,
                {"y": data},
                num_results=3000,
                num_warmup=500,
                num_chains=2,
            )
        draws = np.concatenate(
            [np.asarray(c) for c in flat_chains(post)],
            axis=0,
        )
        n = data.shape[0]
        y_bar = float(np.asarray(data).mean())
        analytic_mean = n * y_bar / (n + 1)
        analytic_sd = float(np.sqrt(1.0 / (n + 1)))
        # Observed across four workflow seeds: |mean error| 0.0004-0.0036,
        # |sd error| 0.0018-0.0032.
        np.testing.assert_allclose(float(draws.mean()), analytic_mean, atol=0.05)
        np.testing.assert_allclose(
            float(draws.std(ddof=1)),
            analytic_sd,
            atol=0.04,
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
            "theta",
            loc=jnp.asarray(prior_mean_arr),
            cov=jnp.asarray(sigma_prior),
        )

        n = 80
        truth = np.array([0.5, -0.2, 0.0, 0.7, -0.5])
        data = jnp.asarray(
            rng.standard_normal((n, d)) + truth,
        )

        model = _observations(prior, data.shape)

        with workflow_run(seed=42):
            post = elliptical_slice(
                model,
                {"y": data},
                num_results=2000,
                num_warmup=500,
                num_chains=2,
            )
        draws = np.concatenate(
            [np.asarray(c) for c in flat_chains(post)],
            axis=0,
        )

        # Closed-form posterior.
        lam_prior = np.linalg.inv(sigma_prior)
        lam_post = lam_prior + n * np.eye(d)
        sigma_post = np.linalg.inv(lam_post)
        y_bar = np.asarray(data).mean(axis=0)
        post_mean = sigma_post @ (lam_prior @ prior_mean_arr + n * y_bar)

        # Observed across workflow seeds 0-15: max |mean error| 0.003-0.017,
        # and a covariance error of 0.06-0.15 of the posterior covariance's norm.
        np.testing.assert_allclose(draws.mean(0), post_mean, atol=0.1)
        sample_cov = np.cov(draws, rowvar=False)
        frob = np.linalg.norm(sample_cov - sigma_post, ord="fro")
        np.testing.assert_array_less(
            frob,
            0.3 * np.linalg.norm(sigma_post, ord="fro"),
        )

    def test_factored_joint_prior(self):
        """Independent ``N(0,1) * N(0,4)`` prior.

        Block-diagonal prior + per-coordinate Gaussian observations:
        the posterior stays diagonal, so the recovered draws must have
        the closed-form per-coordinate spread and *no* cross-correlation.
        """
        sigma_prior = np.diag([1.0, 4.0])
        prior = Normal("a", loc=0.0, scale=1.0) * Normal("b", loc=0.0, scale=2.0)
        n, obs_var = 15, 3.0
        rng = np.random.default_rng(1)
        truth = np.array([0.8, -1.2])
        data = jnp.asarray(np.sqrt(obs_var) * rng.standard_normal((n, 2)) + truth)
        model = _observations(prior, data.shape, obs_var)

        with workflow_run(seed=7):
            post = elliptical_slice(
                model,
                {"y": data},
                num_results=3000,
                num_warmup=500,
                num_chains=2,
            )
        draws = np.concatenate([np.asarray(c) for c in flat_chains(post)], axis=0)

        lam_prior = np.linalg.inv(sigma_prior)
        lam_post = lam_prior + (n / obs_var) * np.eye(2)
        sigma_post = np.linalg.inv(lam_post)
        y_bar = np.asarray(data).mean(0)
        post_mean = sigma_post @ ((n / obs_var) * y_bar)  # prior mean is 0

        # Observed across four workflow seeds: max |mean error| 0.009-0.033,
        # sd error 2-3%, and |sample covariance| 0.0003-0.0037.
        np.testing.assert_allclose(draws.mean(0), post_mean, atol=0.06)
        np.testing.assert_allclose(
            draws.std(0, ddof=1),
            np.sqrt(np.diag(sigma_post)),
            rtol=0.12,
        )
        # Independent prior + diagonal likelihood -> posterior is diagonal.
        sample_cov = np.cov(draws, rowvar=False)
        assert abs(sample_cov[0, 1]) < 0.04

    def test_cross_covariance_prior(self):
        """A ``MultivariateNormal`` prior with off-diagonal covariance.

        The prior correlation is strong (0.8) and the likelihood weak
        (``obs_var = 10``, ``n = 10``) so the prior's cross-covariance
        dominates the posterior. The closed-form posterior covariance is
        then strongly non-diagonal: a block-diagonalised prior would yield
        a *diagonal* ``sigma_post`` differing from the true one by a
        Frobenius distance (~0.36) far exceeding the 15%-of-norm tolerance
        (~0.10) below, so this test genuinely checks the cross-field
        covariance is *sampled*, not merely detected.
        """
        sigma_prior = np.array([[1.0, 0.8], [0.8, 1.0]])
        prior = MultivariateNormal("theta", loc=jnp.zeros(2), cov=jnp.asarray(sigma_prior))
        n, obs_var = 10, 10.0
        rng = np.random.default_rng(2)
        truth = np.array([0.5, -0.7])
        data = jnp.asarray(np.sqrt(obs_var) * rng.standard_normal((n, 2)) + truth)
        model = _observations(prior, data.shape, obs_var)

        with workflow_run(seed=13):
            post = elliptical_slice(
                model,
                {"y": data},
                num_results=4000,
                num_warmup=800,
                num_chains=2,
            )
        draws = np.concatenate([np.asarray(c) for c in flat_chains(post)], axis=0)

        lam_prior = np.linalg.inv(sigma_prior)
        lam_post = lam_prior + (n / obs_var) * np.eye(2)
        sigma_post = np.linalg.inv(lam_post)
        y_bar = np.asarray(data).mean(0)
        post_mean = sigma_post @ ((n / obs_var) * y_bar)  # prior mean is 0

        # The closed-form posterior must retain meaningful cross-covariance
        # (else this test would not discriminate a diagonalised prior).
        assert abs(sigma_post[0, 1]) > 0.1 * np.sqrt(sigma_post[0, 0] * sigma_post[1, 1])

        # Observed across four workflow seeds: max |mean error| 0.005-0.015, and
        # a covariance error of 0.006-0.018 of the posterior covariance's norm.
        np.testing.assert_allclose(draws.mean(0), post_mean, atol=0.1)
        sample_cov = np.cov(draws, rowvar=False)
        frob = np.linalg.norm(sample_cov - sigma_post, ord="fro")
        np.testing.assert_array_less(
            frob,
            0.15 * np.linalg.norm(sigma_post, ord="fro"),
        )


# ---------------------------------------------------------------------------
# Provenance and annotations
# ---------------------------------------------------------------------------


class TestProvenanceAndAnnotations:
    def test_provenance(self, gaussian_model, data):
        with workflow_run(seed=0):
            post = elliptical_slice(
                gaussian_model,
                data,
                num_results=50,
                num_warmup=20,
            )
        assert method_of(post) == "elliptical_slice"
        assert post.provenance.operation == "elliptical_slice"

    def test_annotations_datatree_has_subiter_stats(self, gaussian_model, data):
        num_chains, num_results = 2, 50
        with workflow_run(seed=0):
            post = elliptical_slice(
                gaussian_model,
                data,
                num_results=num_results,
                num_warmup=20,
                num_chains=num_chains,
            )
        assert arviz_data(post) is not None
        assert "posterior" in arviz_data(post)
        assert "sample_stats" in arviz_data(post)
        # The ESS-specific stat is ``subiter`` (number of bracket shrinkages).
        ss = arviz_data(post)["sample_stats"]
        assert "subiter" in ss.variables
        subiter = np.asarray(ss["subiter"].values)
        # One count per (chain, draw); shrinkage counts are positive integers.
        assert subiter.shape == (num_chains, num_results)
        assert np.issubdtype(subiter.dtype, np.integer)
        assert np.all(subiter >= 0)
        np.testing.assert_array_equal(subiter, np.round(subiter))

    def test_warmup_stored(self, gaussian_model):
        with workflow_run(seed=0):
            post = elliptical_slice(
                gaussian_model,
                {"y": jnp.zeros(10)},
                num_results=20,
                num_warmup=15,
                num_chains=2,
            )
        assert warmup_samples(post) is not None
        assert warmup_samples(post)[0].shape == (15, 1)

    def test_no_warmup_path(self, gaussian_model, data):
        """``num_warmup=0`` runs and stores no warmup chains."""
        with workflow_run(seed=0):
            post = elliptical_slice(
                gaussian_model,
                data,
                num_results=30,
                num_warmup=0,
            )
        assert isinstance(post, EmpiricalDistribution)
        assert warmup_samples(post) is None
        assert num_draws(post) == 30

    def test_explicit_init_smoke(self, gaussian_model, data):
        """An explicit ``init=`` (matching the 1-D param dim) runs cleanly."""
        with workflow_run(seed=0):
            post = elliptical_slice(
                gaussian_model,
                data,
                num_results=30,
                num_warmup=5,
                init=jnp.array([2.5]),
            )
        assert isinstance(post, EmpiricalDistribution)
        # 1-D Normal prior → single-parameter chains.
        assert np.asarray(flat_chains(post)[0]).shape == (30, 1)

    def test_multi_chain_shape(self, gaussian_model, data):
        with workflow_run(seed=0):
            post = elliptical_slice(
                gaussian_model,
                data,
                num_results=30,
                num_warmup=10,
                num_chains=3,
            )
        assert num_chains(post) == 3
        assert num_draws(post) == 30


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_raises_on_bare_distribution(self):
        with pytest.raises(TypeError, match="Normal 'x' conditioned on data not keyed"):
            elliptical_slice(
                Normal("x", loc=0.0, scale=1.0, label="x"),
                jnp.zeros(5),
                num_results=10,
                num_warmup=5,
            )

    def test_raises_on_non_gaussian_prior(self):
        model = _observations(Gamma("g", concentration=2.0, rate=1.0), (5,))
        with pytest.raises(TypeError, match="Gaussian"):
            elliptical_slice(
                model,
                {"y": jnp.ones(5)},
                num_results=10,
                num_warmup=5,
            )

    def test_raises_on_none_data(self, gaussian_model):
        with pytest.raises(TypeError, match="with no observed fields"):
            elliptical_slice(
                gaussian_model,
                data=None,
                num_results=10,
                num_warmup=5,
            )


# ---------------------------------------------------------------------------
# The canonical cases of the cross-method validation harness
# ---------------------------------------------------------------------------

test_blackjax_elliptical_slice_canonical = validate_method("blackjax_elliptical_slice")
