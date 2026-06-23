"""Tests for the jax-native NLE / NRE surrogates (BayesFlow backend).

Requires the ``[bayesflow]`` extra (Python 3.12-3.13); skipped otherwise. The
learned components are exercised end to end through ``SimpleModel`` +
``condition_on`` (BlackJAX NUTS), judged against analytic conjugate posteriors
and, for the constrained-prior case, against NUTS run with the exact
likelihood on the same model.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
pytest.importorskip("bayesflow")

import jax
import jax.numpy as jnp
import numpy as np

import probpipe as pp
from probpipe import (
    BayesFlowLikelihood,
    ConditionallyIndependentLikelihood,
    Normal,
    ProductDistribution,
    SimpleModel,
    condition_on,
    learn_amortized_likelihood,
    learn_amortized_ratio,
)
from probpipe.inference._bayesflow_common import _adapter_field_keys
from probpipe.modeling import GenerativeLikelihood, Likelihood


def _theta_vec(params):
    """Coerce a per-draw params object to its 1-D vector (mirrors ``_theta_row``).

    Structured records serialize via ``to_vector``; raw array-likes ravel.
    """
    return params.to_vector() if hasattr(params, "to_vector") else jnp.ravel(jnp.asarray(params))


# Conjugate model: theta ~ N(0, I_2), y_i = theta + sigma * eps. With n rows the
# posterior is N(sum(y) / (n + sigma^2), sigma^2 / (n + sigma^2) I).
_SIGMA = 0.5


class _ConjugateSim(Likelihood, GenerativeLikelihood):
    # ``log_likelihood`` is unused by the amortized path (only ``generate_data``
    # is called); stubbed here just to satisfy the ``Likelihood`` protocol.
    def log_likelihood(self, params, data):
        return jnp.array(0.0)

    def generate_data(self, params, num_observations, *, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        t = _theta_vec(params)
        return t[None, :] + _SIGMA * jax.random.normal(key, (num_observations, t.shape[-1]))


_SIM = _ConjugateSim()


def _prior():
    return ProductDistribution(
        Normal(loc=0.0, scale=1.0, name="a"),
        Normal(loc=0.0, scale=1.0, name="b"),
    )


def _nested_prior():
    """Nested conjugate prior (issue #262): a sub-record ``outer={a, b}`` plus a
    top-level ``m`` -- leaves ``outer/a``, ``outer/b``, ``m``, all ``N(0, 1)`` so
    ``_analytic_posterior`` applies per leaf (``flatten`` order ``[a, b, m]``)."""
    return ProductDistribution(
        name="joint",
        outer={
            "a": Normal(loc=0.0, scale=1.0, name="a"),
            "b": Normal(loc=0.0, scale=1.0, name="b"),
        },
        m=Normal(loc=0.0, scale=1.0, name="m"),
    )


def _analytic_posterior(y_rows: np.ndarray) -> tuple[np.ndarray, float]:
    n = y_rows.shape[0]
    s2 = _SIGMA**2
    mean = y_rows.sum(axis=0) / (n + s2)
    std = float(np.sqrt(s2 / (n + s2)))
    return mean, std


@pytest.fixture(scope="module")
def nle():
    """A briefly-trained NLE, shared across the NLE tests."""
    return learn_amortized_likelihood(
        _prior(),
        _SIM,
        num_simulations=4000,
        epochs=25,
        batch_size=256,
        random_seed=0,
        verbose=0,
    )


@pytest.fixture(scope="module")
def nre():
    """A briefly-trained NRE-C ratio, shared across the NRE tests."""
    return learn_amortized_ratio(
        _prior(),
        _SIM,
        num_simulations=8000,
        epochs=40,
        batch_size=256,
        random_seed=0,
        verbose=0,
    )


class TestSurrogateContract:
    def test_nle_faithful_to_public_log_prob(self, nle):
        """The traceable score path matches the public host-bound
        ``approximator.log_prob`` (same standardization + log-det-jacobian)."""
        theta = jnp.array([0.4, -0.3])
        y_row = np.array([[0.6, -0.1]], dtype="float32")
        ours = float(nle.log_likelihood(theta, y_row))
        keys = _adapter_field_keys(("a", "b"))
        data = {
            keys[0]: np.array([[0.4]], "float32"),
            keys[1]: np.array([[-0.3]], "float32"),
            "observation": y_row,
        }
        public = float(np.asarray(nle.approximator.log_prob(data=data)).reshape(-1)[0])
        # The bypass is the same op sequence on the same float32 buffers --
        # observed diff is exactly 0.0; the bound is float32 headroom.
        np.testing.assert_allclose(ours, public, atol=1e-5)

    def test_nre_faithful_to_public_log_ratio(self, nre):
        """The traceable logits path matches the public ``log_ratio``."""
        theta = jnp.array([0.4, -0.3])
        y_row = np.array([[0.6, -0.1]], dtype="float32")
        ours = float(nre.log_likelihood(theta, y_row))
        keys = _adapter_field_keys(("a", "b"))
        data = {
            keys[0]: np.array([[0.4]], "float32"),
            keys[1]: np.array([[-0.3]], "float32"),
            "observation": y_row,
        }
        public = float(np.asarray(nre.approximator.log_ratio(data=data)).reshape(-1)[0])
        # Same op sequence as the public path; observed diff exactly 0.0.
        np.testing.assert_allclose(ours, public, atol=1e-5)

    @pytest.mark.parametrize("which", ["nle", "nre"])
    def test_grad_transparent(self, which, request):
        """jax.grad of the learned score w.r.t. theta is finite, nonzero,
        matches central finite differences, and survives jit."""
        lik = request.getfixturevalue(which)
        y_row = jnp.array([0.6, -0.1])

        def f(th):
            return lik.log_likelihood(th, y_row)

        th0 = jnp.array([0.3, -0.2])
        g = jax.grad(f)(th0)
        assert jnp.isfinite(g).all() and (jnp.abs(g) > 1e-8).any()
        eps = 1e-3
        fd = np.array(
            [
                (float(f(th0.at[i].add(eps))) - float(f(th0.at[i].add(-eps)))) / (2 * eps)
                for i in range(2)
            ]
        )
        rel = np.abs(np.asarray(g) - fd) / (np.abs(fd) + 1e-6)
        # Observed max relative error across training seeds and probe points:
        # NLE <= 9.2e-4, NRE <= 6.4e-3 (the classifier is only piecewise-smooth).
        assert rel.max() < 2e-2
        v, gj = jax.jit(jax.value_and_grad(f))(th0)
        assert jnp.isfinite(v) and jnp.isfinite(gj).all()

    @pytest.mark.parametrize("which", ["nle", "nre"])
    def test_cil_membership_and_row_sum(self, which, request):
        """Both wrappers are ConditionallyIndependentLikelihoods, and the dataset
        log-likelihood is exactly the sum of per-datum scores."""
        lik = request.getfixturevalue(which)
        assert isinstance(lik, ConditionallyIndependentLikelihood)
        theta = jnp.array([0.2, 0.1])
        rows = jnp.array([[0.5, 0.0], [-0.2, 0.3], [0.1, 0.1]])
        total = float(lik.log_likelihood(theta, rows))
        per = sum(float(lik.per_datum_log_likelihood(theta, rows[i])) for i in range(3))
        np.testing.assert_allclose(total, per, rtol=1e-5)

    def test_params_coercion_record_and_flat(self, nle):
        """Structured per-draw records and flat vectors give identical scores
        (the MCMC helper passes flat vectors; predictive paths pass records)."""
        flat = jnp.array([0.4, -0.3])
        record = _prior().event_template.from_vector(flat)
        y_row = jnp.array([0.6, -0.1])
        np.testing.assert_allclose(
            float(nle.log_likelihood(flat, y_row)),
            float(nle.log_likelihood(record, y_row)),
            rtol=1e-6,
        )

    def test_generate_data_passthrough(self, nle):
        """The wrapper delegates generate_data to the training simulator."""
        params = jnp.array([0.3, 0.2])
        key = jax.random.PRNGKey(9)
        np.testing.assert_allclose(
            np.asarray(nle.generate_data(params, 3, key=key)),
            np.asarray(_SIM.generate_data(params, 3, key=key)),
        )

    def test_repr(self, nle, nre):
        assert repr(nle) == "BayesFlowLikelihood(theta_dim=2, data_dim=2)"
        assert repr(nre) == "BayesFlowRatio(theta_dim=2, data_dim=2)"

    def test_data_width_guard(self, nle):
        """Wrong-width data fails fast with an actionable message."""
        with pytest.raises(ValueError, match="trained on observations of size"):
            nle.log_likelihood(jnp.array([0.0, 0.0]), jnp.zeros(5))

    def test_params_width_guard(self, nle):
        with pytest.raises(ValueError, match="trained on"):
            nle.log_likelihood(jnp.zeros(3), jnp.zeros(2))

    def test_scalar_observations_accept_one_dimensional_dataset(self):
        """d_y == 1: a 1-D array is n scalar observations, not one n-wide row
        (the atleast_2d reading would reject every multi-row scalar dataset).
        Tiny untuned training -- this checks shape semantics, not calibration."""

        class _ScalarSim(Likelihood, GenerativeLikelihood):
            def log_likelihood(self, params, data):
                return jnp.array(0.0)

            def generate_data(self, params, num_observations, *, key=None):
                key = key if key is not None else jax.random.PRNGKey(0)
                a = _theta_vec(params)[0]
                return a + 0.1 * jax.random.normal(key, (num_observations, 1))

        lik = learn_amortized_ratio(
            _prior(),
            _ScalarSim(),
            num_simulations=256,
            epochs=2,
            batch_size=64,
            random_seed=0,
            verbose=0,
        )
        theta = jnp.array([0.3, -0.2])
        y3 = jnp.array([0.1, 0.4, -0.3])
        total = float(lik.log_likelihood(theta, y3))
        per = sum(
            float(lik.per_datum_log_likelihood(theta, jnp.array([v]))) for v in [0.1, 0.4, -0.3]
        )
        np.testing.assert_allclose(total, per, rtol=1e-5)
        # A (n, 1) column is the same dataset.
        np.testing.assert_allclose(total, float(lik.log_likelihood(theta, y3[:, None])), rtol=1e-6)


class TestConditioning:
    """End-to-end: SimpleModel(prior, learned) + condition_on -> NUTS, against
    the analytic conjugate posterior (mean AND spread).

    Bounds are measured: each test's config was run across 3-4 training seeds
    (the per-assertion comments give the observed ranges) and the bound covers
    the observed spread with ~2-3x margin for cross-platform / library-version
    drift (training is seeded, so a given environment is reproducible).
    """

    def _check_posterior(self, model, y_rows, mean_tol, ratio_band):
        post = condition_on(
            model, jnp.asarray(y_rows), num_results=1500, num_warmup=500, random_seed=0
        )
        draws = np.stack([np.asarray(post.draws()[f]).reshape(-1) for f in ("a", "b")], axis=-1)
        an_mean, an_std = _analytic_posterior(np.asarray(y_rows))
        mean_err = np.abs(draws.mean(0) - an_mean).max() / an_std
        ratio = draws.std(0) / an_std
        assert mean_err < mean_tol, (mean_err, draws.mean(0), an_mean)
        assert (ratio_band[0] < ratio).all() and (ratio < ratio_band[1]).all(), ratio

    def test_nle_single_observation(self, nle):
        # Observed across seeds: mean err 0.05-0.10 post-std, ratios 0.99-1.11.
        y = np.array([[0.8, -0.4]], dtype="float32")
        self._check_posterior(
            SimpleModel(prior=_prior(), likelihood=nle), y, mean_tol=0.3, ratio_band=(0.85, 1.25)
        )

    def test_nle_multi_observation_sharpens(self, nle):
        """n=8 i.i.d. rows: the posterior matches the analytic n-observation
        posterior -- the capability NPE's single-observation conditioning lacks.
        The analytic n=8 std (~0.17) is ~2.6x tighter than n=1 (~0.45), so the
        ratio band transitively enforces the sharpening."""
        theta_true = jnp.array([0.6, -0.6])
        y = np.asarray(_SIM.generate_data(theta_true, 8, key=jax.random.PRNGKey(3)))
        # Observed across seeds: mean err 0.03-0.34 post-std, ratios 0.99-1.10
        # (the per-row score errors accumulate over n rows, hence the wider
        # mean bound than n=1).
        self._check_posterior(
            SimpleModel(prior=_prior(), likelihood=nle), y, mean_tol=0.6, ratio_band=(0.85, 1.25)
        )

    def test_nre_single_observation(self, nre):
        # Observed across seeds: mean err 0.02-0.09 post-std, ratios 0.95-1.08.
        y = np.array([[0.8, -0.4]], dtype="float32")
        self._check_posterior(
            SimpleModel(prior=_prior(), likelihood=nre), y, mean_tol=0.3, ratio_band=(0.8, 1.25)
        )

    def _check_nested_posterior(self, model, y, mean_tol, ratio_band):
        """Like ``_check_posterior`` but over the three *nested* leaves
        (``outer/a``, ``outer/b``, ``m``) -- in ``flatten`` order, so leaf j of
        the analytic posterior lines up with observation column j."""
        post = condition_on(model, jnp.asarray(y), num_results=1500, num_warmup=500, random_seed=0)
        draws = np.stack(
            [np.asarray(post.draws()[f]).reshape(-1) for f in ("outer/a", "outer/b", "m")], axis=-1
        )
        an_mean, an_std = _analytic_posterior(np.asarray(y))
        mean_err = np.abs(draws.mean(0) - an_mean).max() / an_std
        ratio = draws.std(0) / an_std
        assert mean_err < mean_tol, (mean_err, draws.mean(0), an_mean)
        assert (ratio_band[0] < ratio).all() and (ratio < ratio_band[1]).all(), ratio

    def test_nle_nested_prior_end_to_end(self):
        """NLE lifts a nested prior (issue #262): SimpleModel(nested prior, learned
        likelihood) + condition_on -> NUTS recovers the analytic conjugate
        posterior, per nested leaf. NLE feeds raw theta to the network, so the
        nesting is purely the leaf-keyed adapter routing (no bijectors)."""
        prior = _nested_prior()
        nle = learn_amortized_likelihood(
            prior,
            _SIM,
            num_simulations=4000,
            epochs=25,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        y = np.array([[0.8, -0.4, 0.3]], dtype="float32")
        self._check_nested_posterior(
            SimpleModel(prior=prior, likelihood=nle), y, mean_tol=0.3, ratio_band=(0.85, 1.25)
        )

    def test_nre_nested_prior_end_to_end(self):
        """NRE lifts a nested prior (issue #262): the same nested conjugate
        recovery as NLE, via the leaf-keyed classifier routing."""
        prior = _nested_prior()
        nre = learn_amortized_ratio(
            prior,
            _SIM,
            num_simulations=8000,
            epochs=40,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        y = np.array([[0.8, -0.4, 0.3]], dtype="float32")
        self._check_nested_posterior(
            SimpleModel(prior=prior, likelihood=nre), y, mean_tol=0.3, ratio_band=(0.8, 1.25)
        )

    def test_nle_constrained_prior_matches_true_likelihood(self):
        """A constrained (Gamma) prior end to end, judged against NUTS run with
        the TRUE (analytic Gaussian) likelihood on the same model -- isolating
        the learned component's error from MCMC and prior effects. NUTS walks
        the natural space; the learned likelihood conditions on raw positive
        theta."""

        class _TrueGaussianLik(Likelihood):
            def log_likelihood(self, params, data):
                t = _theta_vec(params)
                t = jnp.ravel(jnp.asarray(t))
                rows = jnp.atleast_2d(jnp.asarray(data))
                resid = (rows - t[None, :]) / _SIGMA
                return -0.5 * jnp.sum(resid**2) - rows.size * jnp.log(_SIGMA * np.sqrt(2 * np.pi))

        def _gamma_prior():
            return ProductDistribution(
                pp.Gamma(5.0, 1.0, name="lam"), Normal(loc=0.0, scale=1.0, name="m")
            )

        y = np.asarray(_SIM.generate_data(jnp.array([5.0, 0.5]), 4, key=jax.random.PRNGKey(5)))

        def _lam_draws(likelihood):
            post = condition_on(
                SimpleModel(prior=_gamma_prior(), likelihood=likelihood),
                jnp.asarray(y),
                num_results=1500,
                num_warmup=500,
                random_seed=0,
            )
            return np.asarray(post.draws()["lam"]).reshape(-1)

        ref = _lam_draws(_TrueGaussianLik())
        lik = learn_amortized_likelihood(
            _gamma_prior(),
            _SIM,
            num_simulations=3000,
            epochs=20,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        lam = _lam_draws(lik)
        assert (lam > 0).all()
        # Observed across seeds: |mean diff| 0.00-0.26 reference-std units,
        # std ratio 1.02-1.10.
        assert abs(lam.mean() - ref.mean()) / ref.std() < 0.6
        assert 0.8 < lam.std() / ref.std() < 1.3

    def test_nle_dequantize_discrete_observations(self):
        """dequantize=True on integer count data, judged against the analytic
        Gamma-Poisson posterior: lam ~ Gamma(2, 2), y row = 2 iid Poisson(lam)
        counts, y_obs = (2, 1) -> Gamma(5, 4). This atom-heavy regime is where
        the raw (non-dequantized) flow measurably miscalibrates (std ratios
        1.13-1.27 across seeds); the cell-midpoint scoring must also match a
        non-dequantized wrapper of the same approximator at y + 1/2 exactly."""

        class _PoissonPairSim(Likelihood, GenerativeLikelihood):
            def log_likelihood(self, params, data):
                return jnp.array(0.0)

            def generate_data(self, params, num_observations, *, key=None):
                key = key if key is not None else jax.random.PRNGKey(0)
                lam = _theta_vec(params)[0]
                counts = jax.random.poisson(key, lam, (num_observations, 2))
                return counts.astype(jnp.float32)

        y_obs = jnp.array([[2.0, 1.0]])
        an_mean, an_std = 5.0 / 4.0, np.sqrt(5.0) / 4.0
        lik = learn_amortized_likelihood(
            pp.Gamma(2.0, 2.0, name="lam"),
            _PoissonPairSim(),
            num_simulations=4000,
            epochs=25,
            batch_size=256,
            random_seed=0,
            dequantize=True,
            verbose=0,
        )
        twin = BayesFlowLikelihood(
            lik.approximator, lik.prior, lik.simulator, data_dim=2, dequantized=False
        )
        np.testing.assert_allclose(
            float(lik.log_likelihood(jnp.array([1.3]), y_obs)),
            float(twin.log_likelihood(jnp.array([1.3]), y_obs + 0.5)),
            rtol=1e-6,
        )
        post = condition_on(
            SimpleModel(prior=pp.Gamma(2.0, 2.0, name="lam"), likelihood=lik),
            y_obs,
            num_results=1500,
            num_warmup=500,
            random_seed=0,
        )
        lam = np.asarray(post.draws()["lam"]).reshape(-1)
        assert (lam > 0).all()
        # Observed across seeds 0-2: mean err 0.03-0.17 posterior-std units,
        # std ratio 0.96-1.02.
        assert abs(lam.mean() - an_mean) / an_std < 0.5
        assert 0.8 < lam.std() / an_std < 1.2


class TestValidation:
    """Train-time validation -- each raises before any training runs."""

    def test_rejects_unknown_sim_backend(self):
        with pytest.raises(ValueError, match="Unknown sim_backend"):
            learn_amortized_likelihood(
                _prior(), _SIM, sim_backend="bogus", num_simulations=8, epochs=1
            )

    @pytest.mark.parametrize("override", [{"epochs": 0}, {"num_simulations": 0}])
    def test_rejects_nonpositive_counts(self, override):
        kwargs = {"num_simulations": 8, "epochs": 1, **override}
        with pytest.raises(ValueError, match="positive integer"):
            learn_amortized_ratio(_prior(), _SIM, **kwargs)

    def test_rejects_non_integer_counts(self):
        with pytest.raises(TypeError, match="must be an integer"):
            learn_amortized_likelihood(_prior(), _SIM, num_simulations=100.5, epochs=1)

    def test_rejects_non_generative_simulator(self):
        class _NoGenerate:
            pass

        with pytest.raises(TypeError, match="generate_data"):
            learn_amortized_likelihood(_prior(), _NoGenerate(), num_simulations=8, epochs=1)

    def test_rejects_non_record_prior(self):
        with pytest.raises(TypeError, match="RecordDistribution"):
            learn_amortized_ratio(jnp.zeros(2), _SIM, num_simulations=8, epochs=1)

    def test_dequantize_rejects_counts_at_float32_cell_limit(self):
        """dequantize=True enforces the documented 2**23 bound on the simulated
        observations (float32 spacing reaches 1.0 there, so the unit-cell
        arithmetic would silently round away)."""

        class _HugeCounts(Likelihood, GenerativeLikelihood):
            def log_likelihood(self, params, data):
                return jnp.array(0.0)

            def generate_data(self, params, num_observations, *, key=None):
                return jnp.full((num_observations, 2), 2.0**23)

        with pytest.raises(ValueError, match=r"2\*\*23"):
            learn_amortized_likelihood(
                _prior(), _HugeCounts(), num_simulations=8, epochs=1, dequantize=True
            )

    def test_nle_rejects_one_dimensional_observations(self):
        """The default coupling flow cannot model 1-D densities; the error points
        at learn_amortized_ratio (whose classifier has no minimum dimension)."""

        class _Scalar(Likelihood, GenerativeLikelihood):
            def log_likelihood(self, params, data):
                return jnp.array(0.0)

            def generate_data(self, params, num_observations, *, key=None):
                key = key if key is not None else jax.random.PRNGKey(0)
                a = _theta_vec(params)[0]
                return a + 0.1 * jax.random.normal(key, (num_observations, 1))

        with pytest.raises(ValueError, match="learn_amortized_ratio"):
            learn_amortized_likelihood(_prior(), _Scalar(), num_simulations=8, epochs=1)


class TestDeterminism:
    def test_training_deterministic_for_seed(self):
        """Two same-seed NLE trainings produce identical learned scores."""

        def _fit():
            return learn_amortized_likelihood(
                _prior(),
                _SIM,
                num_simulations=400,
                epochs=1,
                batch_size=256,
                random_seed=0,
                verbose=0,
            )

        theta, y = jnp.array([0.3, -0.1]), jnp.array([0.5, 0.0])
        v1 = float(_fit().log_likelihood(theta, y))
        v2 = float(_fit().log_likelihood(theta, y))
        assert v1 == v2
