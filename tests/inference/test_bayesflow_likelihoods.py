"""Tests for the jax-native NLE / NRE kernels (BayesFlow backend).

Requires the ``[bayesflow]`` extra (Python 3.12-3.13); skipped otherwise. The
learned kernels are exercised end to end through ``condition_on(learned * prior,
{"observation": y})`` (BlackJAX NUTS), judged against analytic conjugate
posteriors and, for the constrained-prior case, against NUTS run with the exact
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
import tensorflow_probability.substrates.jax.distributions as tfd

import probpipe as pp
from probpipe import (
    BayesFlowLikelihood,
    BayesFlowRatio,
    Normal,
    NumericArraySpec,
    NumericRecord,
    condition_on,
    learn_amortized_likelihood,
    learn_amortized_ratio,
)
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalUnnormalizedLogProb,
)
from probpipe.inference._bayesflow_common import _adapter_field_keys
from probpipe.operations._condition import condition_on as condition_on_operation

from ._bayesflow_helpers import SimulatorKernel, theta_vec
from .canonical import ObservationKernel

# Conjugate model: theta ~ N(0, I_2), y_i = theta + sigma * eps. With n rows the
# posterior is N(sum(y) / (n + sigma^2), sigma^2 / (n + sigma^2) I).
_SIGMA = 0.5


def _rows(params, num_observations, key):
    """*num_observations* i.i.d. rows ``y_i = theta + sigma * eps`` at the parameters *params*."""
    t = theta_vec(params)
    return t[None, :] + _SIGMA * jax.random.normal(key, (num_observations, t.shape[-1]))


def _sim(prior):
    """The conjugate simulator over *prior*'s fields: one row per draw."""
    return SimulatorKernel(
        prior, (prior.event_spec.spec.vector_size,), lambda params, key: _rows(params, 1, key)[0]
    )


def _prior():
    return Normal(loc=0.0, scale=1.0, name="a") * Normal(loc=0.0, scale=1.0, name="b")


_SIM = _sim(_prior())


def _nested_prior():
    """Nested conjugate prior: a sub-record ``outer={a, b}`` plus a
    top-level ``m`` -- leaves ``outer/a``, ``outer/b``, ``m``, all ``N(0, 1)`` so
    ``_analytic_posterior`` applies per leaf (``flatten`` order ``[a, b, m]``)."""
    outer = (
        Normal(loc=0.0, scale=1.0, name="a") * Normal(loc=0.0, scale=1.0, name="b")
    ).with_path_names({"a": "outer/a", "b": "outer/b"})
    return (outer * Normal(loc=0.0, scale=1.0, name="m")).with_label("joint")


def _analytic_posterior(y_rows: np.ndarray) -> tuple[np.ndarray, float]:
    n = y_rows.shape[0]
    s2 = _SIGMA**2
    mean = y_rows.sum(axis=0) / (n + s2)
    std = float(np.sqrt(s2 / (n + s2)))
    return mean, std


def _score(lik, theta, rows):
    """The learned kernel's score of the dataset *rows* at the flat parameters *theta*."""
    declaration = lik.prior.event_spec
    theta = jnp.asarray(theta)
    if declaration.exposes_record:
        given = NumericRecord.from_vector("theta", declaration.spec, theta)
    else:
        (component,) = declaration.components
        given = {component: jnp.reshape(theta, declaration.spec.shape)}
    return lik._conditional_unnormalized_log_prob(given, rows)


def _posterior(lik, prior, y):
    """The posterior of ``lik * prior`` at the observation rows *y*, by the registry's method."""
    view = condition_on_operation.with_options(
        method_options={"num_results": 1500, "num_warmup": 500, "random_seed": 0}
    )
    return view(lik * prior, {"observation": jnp.asarray(y)})


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
        ours = float(_score(nle, theta, y_row))
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
        ours = float(_score(nre, theta, y_row))
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
            return _score(lik, th, y_row)

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

    def test_the_kernels_run_from_the_priors_components_to_the_rows(self, nle, nre):
        """A learned kernel's given slots are the prior's components and its event
        the observation rows; NLE has the network's density and NRE one known up to
        a constant."""
        for lik in (nle, nre):
            assert list(lik.given_spec) == ["a", "b"]
            assert tuple(lik.event_spec.components) == ("observation",)
        assert isinstance(nle, SupportsConditionalLogProb)
        assert isinstance(nre, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(nre, SupportsConditionalLogProb)
        assert isinstance(nre, BayesFlowRatio)

    @pytest.mark.parametrize("which", ["nle", "nre"])
    def test_a_dataset_scores_as_the_sum_of_its_rows(self, which, request):
        """A dataset's score is the sum of its per-row scores."""
        lik = request.getfixturevalue(which)
        theta = jnp.array([0.2, 0.1])
        rows = jnp.array([[0.5, 0.0], [-0.2, 0.3], [0.1, 0.1]])
        total = float(_score(lik, theta, rows))
        per = sum(float(_score(lik, theta, rows[i])) for i in range(3))
        np.testing.assert_allclose(total, per, rtol=1e-5)

    def test_a_record_and_a_mapping_given_score_alike(self, nle):
        """A record of the parameters and a mapping of them give identical scores."""
        record = NumericRecord.from_vector("nr", _prior().event_spec.spec, jnp.array([0.4, -0.3]))
        y_row = jnp.array([0.6, -0.1])
        np.testing.assert_allclose(
            float(nle._conditional_log_prob(record, y_row)),
            float(nle._conditional_log_prob({"a": 0.4, "b": -0.3}, y_row)),
            rtol=1e-6,
        )

    def test_the_law_at_the_parameters_has_the_density(self, nle):
        law = condition_on_operation(nle, {"a": 0.4, "b": -0.3})
        y_row = jnp.array([0.6, -0.1])
        np.testing.assert_allclose(
            float(law._log_prob(y_row)),
            float(_score(nle, jnp.array([0.4, -0.3]), y_row)),
            rtol=1e-6,
        )

    def test_repr(self, nle, nre):
        for kernel, cls in ((nle, "BayesFlowLikelihood"), (nre, "BayesFlowRatio")):
            text = repr(kernel)
            assert text.startswith(f"{cls}(\n    '{kernel.label}',\n")
            assert "theta_dim=2," in text and "data_dim=2," in text

    def test_data_width_guard(self, nle):
        """Wrong-width data fails fast with an actionable message."""
        with pytest.raises(ValueError, match="trained on observations of size"):
            _score(nle, jnp.array([0.0, 0.0]), jnp.zeros(5))

    def test_params_width_guard(self, nle):
        with pytest.raises(ValueError, match="trained on"):
            nle._conditional_log_prob({"a": jnp.zeros(2), "b": 0.0}, jnp.zeros(2))

    def test_scalar_observations_accept_one_dimensional_dataset(self):
        """d_y == 1: a 1-D array is n scalar observations, not one n-wide row
        (the atleast_2d reading would reject every multi-row scalar dataset).
        Tiny untuned training -- this checks shape semantics, not calibration."""

        def _scalar(params, key):
            return theta_vec(params)[:1] + 0.1 * jax.random.normal(key, (1,))

        lik = learn_amortized_ratio(
            _prior(),
            SimulatorKernel(_prior(), (1,), _scalar),
            num_simulations=256,
            epochs=2,
            batch_size=64,
            random_seed=0,
            verbose=0,
        )
        theta = jnp.array([0.3, -0.2])
        y3 = jnp.array([0.1, 0.4, -0.3])
        total = float(_score(lik, theta, y3))
        per = sum(float(_score(lik, theta, jnp.array([v]))) for v in [0.1, 0.4, -0.3])
        np.testing.assert_allclose(total, per, rtol=1e-5)
        # A (n, 1) column is the same dataset.
        np.testing.assert_allclose(total, float(_score(lik, theta, y3[:, None])), rtol=1e-6)


class TestConditioning:
    """End-to-end: condition_on(learned * prior, observation) -> NUTS, against
    the analytic conjugate posterior (mean AND spread).

    Bounds are measured: each test's config was run across 3-4 training seeds
    (the per-assertion comments give the observed ranges) and the bound covers
    the observed spread with ~2-3x margin for cross-platform / library-version
    drift (training is seeded, so a given environment is reproducible).
    """

    def _check_posterior(self, lik, prior, y_rows, mean_tol, ratio_band):
        post = _posterior(lik, prior, y_rows)
        draws = np.stack([np.asarray(post.draws()[f]).reshape(-1) for f in ("a", "b")], axis=-1)
        an_mean, an_std = _analytic_posterior(np.asarray(y_rows))
        mean_err = np.abs(draws.mean(0) - an_mean).max() / an_std
        ratio = draws.std(0) / an_std
        assert mean_err < mean_tol, (mean_err, draws.mean(0), an_mean)
        assert (ratio_band[0] < ratio).all() and (ratio < ratio_band[1]).all(), ratio

    def test_nle_single_observation(self, nle):
        # Observed across seeds: mean err 0.05-0.10 post-std, ratios 0.99-1.11.
        y = np.array([[0.8, -0.4]], dtype="float32")
        self._check_posterior(nle, _prior(), y, mean_tol=0.3, ratio_band=(0.85, 1.25))

    def test_a_learned_likelihood_times_a_prior_runs_a_registered_method(self, nle):
        report = condition_on_operation.check(nle * _prior(), {"observation": jnp.zeros((1, 2))})
        assert (report.route, report.method, report.exact) == ("bayes", "blackjax_nuts", False)

    def test_nle_multi_observation_sharpens(self, nle):
        """n=8 i.i.d. rows: the posterior matches the analytic n-observation
        posterior -- the capability NPE's single-observation conditioning lacks.
        The analytic n=8 std (~0.17) is ~2.6x tighter than n=1 (~0.45), so the
        ratio band transitively enforces the sharpening."""
        theta_true = jnp.array([0.6, -0.6])
        y = np.asarray(_rows(theta_true, 8, jax.random.PRNGKey(3)))
        # Observed across seeds: mean err 0.03-0.34 post-std, ratios 0.99-1.10
        # (the per-row score errors accumulate over n rows, hence the wider
        # mean bound than n=1).
        self._check_posterior(nle, _prior(), y, mean_tol=0.6, ratio_band=(0.85, 1.25))

    def test_nre_single_observation(self, nre):
        # Observed across seeds: mean err 0.02-0.09 post-std, ratios 0.95-1.08.
        y = np.array([[0.8, -0.4]], dtype="float32")
        self._check_posterior(nre, _prior(), y, mean_tol=0.3, ratio_band=(0.8, 1.25))

    def _check_nested_posterior(self, lik, prior, y, mean_tol, ratio_band):
        """Like ``_check_posterior`` but over the three *nested* leaves
        (``outer/a``, ``outer/b``, ``m``) -- in ``flatten`` order, so leaf j of
        the analytic posterior lines up with observation column j."""
        post = _posterior(lik, prior, y)
        draws = np.stack(
            [np.asarray(post.draws()[f]).reshape(-1) for f in ("outer/a", "outer/b", "m")], axis=-1
        )
        an_mean, an_std = _analytic_posterior(np.asarray(y))
        mean_err = np.abs(draws.mean(0) - an_mean).max() / an_std
        ratio = draws.std(0) / an_std
        assert mean_err < mean_tol, (mean_err, draws.mean(0), an_mean)
        assert (ratio_band[0] < ratio).all() and (ratio < ratio_band[1]).all(), ratio

    def test_nle_nested_prior_end_to_end(self):
        """NLE lifts a nested prior: the learned likelihood times the nested
        prior, conditioned with condition_on -> NUTS recovers the analytic conjugate
        posterior, per nested leaf. NLE feeds raw theta to the network, so the
        nesting is purely the leaf-keyed adapter routing (no bijectors)."""
        prior = _nested_prior()
        nle = learn_amortized_likelihood(
            prior,
            _sim(prior),
            num_simulations=4000,
            epochs=25,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        y = np.array([[0.8, -0.4, 0.3]], dtype="float32")
        self._check_nested_posterior(nle, prior, y, mean_tol=0.3, ratio_band=(0.85, 1.25))

    def test_nre_nested_prior_end_to_end(self):
        """NRE lifts a nested prior: the same nested conjugate
        recovery as NLE, via the leaf-keyed classifier routing."""
        prior = _nested_prior()
        nre = learn_amortized_ratio(
            prior,
            _sim(prior),
            num_simulations=8000,
            epochs=40,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        y = np.array([[0.8, -0.4, 0.3]], dtype="float32")
        self._check_nested_posterior(nre, prior, y, mean_tol=0.3, ratio_band=(0.8, 1.25))

    def test_nle_constrained_prior_matches_true_likelihood(self):
        """A constrained (Gamma) prior end to end, judged against NUTS run with
        the TRUE (analytic Gaussian) likelihood on the same model -- isolating
        the learned component's error from MCMC and prior effects. NUTS walks
        the natural space; the learned likelihood conditions on raw positive
        theta."""

        def _gamma_prior():
            return pp.Gamma("lam", 5.0, 1.0) * Normal(loc=0.0, scale=1.0, name="m")

        y = np.asarray(_rows(jnp.array([5.0, 0.5]), 4, jax.random.PRNGKey(5)))

        # The analytic Gaussian likelihood of the four rows, as a kernel of (lam, m).
        prior = _gamma_prior()
        true_likelihood = ObservationKernel(
            "observation",
            dict(prior.event_spec.components),
            NumericArraySpec(y.shape),
            lambda lam, m: tfd.Independent(
                tfd.Normal(jnp.broadcast_to(jnp.stack([lam, m]), y.shape), _SIGMA), 2
            ),
        )
        ref_post = condition_on.with_options(
            method_options={"num_results": 1500, "num_warmup": 500, "random_seed": 0}
        )(true_likelihood * prior, {"observation": jnp.asarray(y)})
        ref = np.asarray(ref_post.draws()["lam"]).reshape(-1)
        lik = learn_amortized_likelihood(
            _gamma_prior(),
            _sim(_gamma_prior()),
            num_simulations=3000,
            epochs=20,
            batch_size=256,
            random_seed=0,
            verbose=0,
        )
        lam = np.asarray(_posterior(lik, _gamma_prior(), y).draws()["lam"]).reshape(-1)
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

        def _poisson_pair(params, key):
            lam = theta_vec(params)[0]
            return jax.random.poisson(key, lam, (2,)).astype(jnp.float32)

        y_obs = jnp.array([[2.0, 1.0]])
        an_mean, an_std = 5.0 / 4.0, np.sqrt(5.0) / 4.0
        lik = learn_amortized_likelihood(
            pp.Gamma("lam", 2.0, 2.0),
            SimulatorKernel(pp.Gamma("lam", 2.0, 2.0), (2,), _poisson_pair),
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
            float(_score(lik, jnp.array([1.3]), y_obs)),
            float(_score(twin, jnp.array([1.3]), y_obs + 0.5)),
            rtol=1e-6,
        )
        post = _posterior(lik, pp.Gamma("lam", 2.0, 2.0), y_obs)
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

        with pytest.raises(TypeError, match="ConditionalDistribution that samples"):
            learn_amortized_likelihood(_prior(), _NoGenerate(), num_simulations=8, epochs=1)

    def test_rejects_a_prior_that_is_not_numeric(self):
        with pytest.raises(TypeError, match="requires a numeric prior"):
            learn_amortized_ratio(jnp.zeros(2), _SIM, num_simulations=8, epochs=1)

    def test_dequantize_rejects_counts_at_float32_cell_limit(self):
        """dequantize=True enforces the documented 2**23 bound on the simulated
        observations (float32 spacing reaches 1.0 there, so the unit-cell
        arithmetic would silently round away)."""

        huge_counts = SimulatorKernel(_prior(), (2,), lambda params, key: jnp.full((2,), 2.0**23))
        with pytest.raises(ValueError, match=r"2\*\*23"):
            learn_amortized_likelihood(
                _prior(), huge_counts, num_simulations=8, epochs=1, dequantize=True
            )

    def test_nle_rejects_one_dimensional_observations(self):
        """The default coupling flow cannot model 1-D densities; the error points
        at learn_amortized_ratio (whose classifier has no minimum dimension)."""

        def _scalar(params, key):
            return theta_vec(params)[:1] + 0.1 * jax.random.normal(key, (1,))

        scalar = SimulatorKernel(_prior(), (1,), _scalar)
        with pytest.raises(ValueError, match="learn_amortized_ratio"):
            learn_amortized_likelihood(_prior(), scalar, num_simulations=8, epochs=1)


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
        v1 = float(_score(_fit(), theta, y))
        v2 = float(_score(_fit(), theta, y))
        assert v1 == v2
