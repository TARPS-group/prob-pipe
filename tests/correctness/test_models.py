"""Contract and correctness tests of the model families, through the operation model's operations.

Each family is exercised as a user writes it, and its answers are compared
with references computed independently: SciPy densities, PyMC's own
``logp``, closed-form posteriors and prior predictives, and Monte Carlo
moments bounded by four standard errors.

- GLM likelihoods composed with priors: the joint density, the prior
  predictive, and the Gaussian case's posterior against its closed form,
  including a design left as a given slot;
- ``PyMCModel``: the prior predictive, covariates as given slots, the
  normalized density against PyMC's ``logp``, and the unnormalized density
  and the posterior of a model with a potential;
- a law of an unnormalized density from ``distribution``: sampling, conversion,
  and conditioning through a method of the inference-method registry;
- factored joints: ancestral sampling moments, summed log-densities, and the
  moments and marginals the factor graph admits;
- ``StanModel`` and the learned kernels, where the local toolchain allows.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Gamma,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    OutputSpec,
    RecordSpec,
    distribution,
    workflow_run,
)
from probpipe.core._dispatch import ResolutionError
from probpipe.distributions import (
    ConditionalDistribution,
    Distribution,
    FactoredConditionalDistribution,
)
from probpipe.distributions._capabilities import (
    SupportsLogProb,
    SupportsMean,
    SupportsUnnormalizedLogProb,
)
from probpipe.distributions._factored import _raw_record
from probpipe.families import (
    BernoulliFamily,
    GaussianFamily,
    PoissonFamily,
    PyMCModel,
    glm_likelihood,
)
from tests._ops import (
    EmpiricalDistribution,
    condition_on,
    convert,
    inference_method_registry,
    log_prob,
    marginal,
    mean,
    sample,
    unnormalized_log_prob,
    variance,
)
from tests.correctness._laws import REAL, exact_reference
from tests.inference import canonical
from tests.inference._bayesflow_helpers import SimulatorKernel, theta_vec
from tests.inference._harness import PROFILES, Z, assert_matches
from tests.inference.canonical import LeafReference, ObservationKernel, PosteriorReference

#: The number of independent draws behind every Monte Carlo moment these tests read.
DRAWS = 4000

#: The controls of a gradient-based fit in these tests.
FIT = PROFILES["blackjax_nuts"].method_options


def _within_mcse(estimate, reference, sd, draws=DRAWS):
    """Assert that iid Monte Carlo estimates lie within four standard errors of the reference."""
    error = np.abs(np.asarray(estimate, np.float64) - np.asarray(reference, np.float64))
    band = Z * np.asarray(sd, np.float64) / np.sqrt(draws)
    assert np.all(error <= band), f"{estimate} vs {reference}: error {error} above {band}"


def _variance_band(values, reference_variance, draws=DRAWS):
    """Assert that the sample variance of iid *values* lies within four standard errors."""
    values = np.asarray(values, np.float64)
    centred = (values - values.mean(axis=0)) ** 2
    se = centred.std(axis=0) / np.sqrt(draws)
    error = np.abs(centred.mean(axis=0) - reference_variance)
    assert np.all(error <= Z * se), f"variance {centred.mean(0)} vs {reference_variance}"


# ---------------------------------------------------------------------------
# GLM likelihoods composed with priors
# ---------------------------------------------------------------------------

_X = jnp.array([[1.0, 0.5], [1.0, -0.3], [1.0, 1.2], [1.0, -1.1], [1.0, 0.1]], jnp.float32)
_BETA = np.array([0.2, -0.4])


class TestGLM:
    @pytest.mark.parametrize(
        ("family", "response", "density"),
        [
            (
                GaussianFamily(),
                np.array([0.3, -0.2, 1.1, 0.4, -0.6]),
                lambda eta, y: scipy.stats.norm.logpdf(y, eta, 0.7).sum(),
            ),
            (
                BernoulliFamily(),
                np.array([1.0, 0.0, 1.0, 1.0, 0.0]),
                lambda eta, y: scipy.stats.bernoulli.logpmf(y, 1 / (1 + np.exp(-eta))).sum(),
            ),
            (
                PoissonFamily(),
                np.array([2.0, 0.0, 1.0, 3.0, 1.0]),
                lambda eta, y: scipy.stats.poisson.logpmf(y, np.exp(eta)).sum(),
            ),
        ],
        ids=["gaussian", "bernoulli", "poisson"],
    )
    def test_the_joint_density_is_the_sum_of_the_factor_densities(self, family, response, density):
        """The joint's log-density is the GLM's under its canonical link plus the prior's."""
        dispersion = 0.7 if family.has_dispersion else None
        likelihood = glm_likelihood("y", family, X=_X, dispersion=dispersion)
        joint = likelihood * MultivariateNormal("beta", jnp.zeros(2), cov=2.0 * jnp.eye(2))
        value = {"y": jnp.asarray(response, jnp.float32), "beta": jnp.asarray(_BETA, jnp.float32)}
        expected = density(np.asarray(_X, np.float64) @ _BETA, response) + (
            scipy.stats.multivariate_normal.logpdf(_BETA, np.zeros(2), 2.0 * np.eye(2))
        )
        assert float(log_prob.with_options(raw=True)(joint, value)) == pytest.approx(
            expected, rel=1e-5
        )

    def test_the_prior_predictive_has_the_gaussian_moments(self):
        """``y = X beta + e`` has mean zero and variance ``v ||x_i||² + s²`` per observation."""
        joint = glm_likelihood("y", GaussianFamily(), X=_X, dispersion=0.7) * MultivariateNormal(
            "beta", jnp.zeros(2), cov=2.0 * jnp.eye(2)
        )
        with workflow_run(seed=11):
            draws = _raw_record(sample(joint, sample_shape=(DRAWS,)))
        X = np.asarray(_X, np.float64)
        expected_variance = 2.0 * (X**2).sum(axis=1) + 0.7**2
        _within_mcse(np.mean(draws["y"], axis=0), np.zeros(5), np.sqrt(expected_variance))
        _variance_band(draws["y"], expected_variance)

    def test_the_gaussian_posterior_through_the_default_route_is_the_closed_form(self):
        """``condition_on`` with no method normalizes the Gaussian GLM joint to its closed form.

        The default selection runs blackjax_nuts, which takes a few seconds.
        """
        case = canonical.case("gaussian_linear")
        report = condition_on.check(case.model, case.data)
        assert (report.route, report.method) == ("inference_methods", "blackjax_nuts")
        with workflow_run(seed=0):
            posterior = condition_on.with_options(method_options=FIT)(case.model, case.data)
        assert_matches(posterior, case.reference, label="the default route on gaussian_linear")

    def test_a_design_left_unbound_is_a_given_slot_of_the_joint(self):
        likelihood = glm_likelihood("y", GaussianFamily(), dispersion=1.0)
        joint = likelihood * MultivariateNormal("beta", jnp.zeros(2), cov=jnp.eye(2))
        assert isinstance(joint, FactoredConditionalDistribution)
        assert list(joint.given_spec) == ["X"]
        bound = condition_on(joint, {"X": _X})
        assert bound.event_spec.spec["y"].shape == (5,)

    def test_binding_the_design_and_the_response_gives_the_closed_form_posterior(self):
        """One call binds the design as a given slot and conditions on the response.

        The design is curried first and the response conditioned by Bayes'
        rule, so the result is the Gaussian closed form; the fit takes a few
        seconds.
        """
        case = canonical.case("gaussian_linear")
        X = jnp.asarray(case.stan_data["X"], jnp.float32)
        prior = MultivariateNormal("beta", jnp.zeros(X.shape[1]), cov=4.0 * jnp.eye(X.shape[1]))
        joint = glm_likelihood("y", GaussianFamily(), dispersion=1.0) * prior
        with workflow_run(seed=0):
            posterior = condition_on.with_options(method="blackjax_nuts", method_options=FIT)(
                joint, {"X": X, **case.data}
            )
        assert_matches(posterior, case.reference, label="binding X and y together")


# ---------------------------------------------------------------------------
# PyMCModel
# ---------------------------------------------------------------------------


def _normal_model(y=None):
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=y)
    return model


def _regression(x=None, y=None):
    x = np.zeros(4) if x is None else np.asarray(x)
    import pymc as pm

    with pm.Model() as model:
        beta = pm.Normal("beta", 0, 1)
        pm.Normal("y", beta * x, 1.0, observed=y, shape=x.shape[0])
    return model


#: The observations of the model with a potential.
_POTENTIAL_DATA = np.array([0.5, -0.2, 1.0, 0.3, 0.8])


def _with_potential(y=None):
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Potential("penalty", -(mu**2))
        pm.Normal("y", mu, 1.0, observed=y, shape=_POTENTIAL_DATA.shape[0])
    return model


class TestPyMCModel:
    @pytest.fixture(autouse=True)
    def _pymc(self):
        pytest.importorskip("pymc")

    def test_the_prior_predictive_has_the_models_moments(self):
        """``mu ~ N(0, 10)``, ``sigma ~ HalfNormal(1)``, and ``y ~ N(mu, sigma)`` give known moments.

        The mean and variance of ``sigma`` are ``sqrt(2/pi)`` and ``1 - 2/pi``,
        and ``y`` has mean zero and variance ``100 + E[sigma²] = 101``.
        """
        model = PyMCModel(_normal_model, label="normal")
        with workflow_run(seed=12):
            draws = _raw_record(sample(model, sample_shape=(DRAWS,)))
        _within_mcse(np.mean(draws["mu"]), 0.0, 10.0)
        _within_mcse(np.mean(draws["sigma"]), np.sqrt(2 / np.pi), np.sqrt(1 - 2 / np.pi))
        _within_mcse(np.mean(draws["y"]), 0.0, np.sqrt(101.0))
        _variance_band(draws["mu"], 100.0)
        _variance_band(draws["y"], 101.0)

    def test_the_density_is_pymcs_logp(self):
        """The normalized joint density is PyMC's ``logp`` without the transform's Jacobian."""
        model = PyMCModel(_normal_model, label="normal")
        expected = _normal_model().compile_logp(jacobian=False)(
            {"mu": 0.3, "sigma_log__": np.log(1.2), "y": 0.5}
        )
        value = {"mu": 0.3, "sigma": 1.2, "y": 0.5}
        assert float(log_prob.with_options(raw=True)(model, value)) == pytest.approx(
            float(expected), rel=1e-5
        )

    def test_a_covariate_is_a_given_slot_and_binding_it_returns_the_joint(self):
        kernel = PyMCModel(_regression, label="regression")
        assert isinstance(kernel, ConditionalDistribution)
        assert list(kernel.given_spec) == ["x"]
        law = condition_on(kernel, {"x": np.linspace(-1, 1, 6)})
        assert isinstance(law, PyMCModel)
        assert law.event_spec.spec["y"].shape == (6,)

    def test_binding_the_covariate_and_the_response_gives_the_closed_form_posterior(self):
        """``beta ~ N(0, 1)`` and ``y ~ N(beta x, 1)`` give ``beta | y ~ N(xᵀy / (1 + xᵀx), 1 / (1 + xᵀx))``.

        The covariate is curried and the response conditioned in one call; the
        fit runs nutpie, or PyMC's NUTS without it, for a few seconds.
        """
        x = np.linspace(-1.0, 1.0, 6)
        y = 0.7 * x + np.array([0.1, -0.3, 0.2, 0.0, -0.1, 0.4])
        method = (
            "nutpie_nuts"
            if "nutpie_nuts" in inference_method_registry.list_methods()
            else "pymc_nuts"
        )
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method=method, method_options=PROFILES[method].method_options
            )(PyMCModel(_regression, label="regression"), {"x": x, "y": y})
        precision = 1.0 + x @ x
        reference = exact_reference(np.array([x @ y / precision]), np.array([1.0 / precision]))
        assert_matches(posterior, reference, label=f"{method} on the PyMC regression")

    def test_a_model_with_a_potential_claims_only_the_unnormalized_density(self):
        model = PyMCModel(_with_potential, label="penalized")
        assert isinstance(model, SupportsUnnormalizedLogProb)
        assert not isinstance(model, SupportsLogProb)
        with pytest.raises(ResolutionError):
            log_prob(model, {"mu": 0.4, "y": _POTENTIAL_DATA})

    def test_the_unnormalized_density_of_a_model_with_a_potential_is_pymcs_logp(self):
        model = PyMCModel(_with_potential, label="penalized")
        expected = _with_potential().compile_logp(jacobian=False)({"mu": 0.4, "y": _POTENTIAL_DATA})
        value = {"mu": 0.4, "y": _POTENTIAL_DATA}
        assert float(unnormalized_log_prob.with_options(raw=True)(model, value)) == pytest.approx(
            float(expected), rel=1e-5
        )

    def test_conditioning_a_model_with_a_potential_gives_the_closed_form_posterior(self):
        """``mu ~ N(0, 1)``, the potential ``-mu²``, and ``y ~ N(mu, 1)`` give ``mu | y ~ N(Σy / (3 + n), 1 / (3 + n))``.

        The potential's factor ``exp(-mu²)`` is a Gaussian of precision 2, so
        the posterior precision is ``1 + 2 + n``; the fit takes a few seconds.
        """
        method = (
            "nutpie_nuts"
            if "nutpie_nuts" in inference_method_registry.list_methods()
            else "pymc_nuts"
        )
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method=method, method_options=PROFILES[method].method_options
            )(PyMCModel(_with_potential, label="penalized"), {"y": _POTENTIAL_DATA})
        precision = 3.0 + _POTENTIAL_DATA.shape[0]
        reference = exact_reference(
            np.array([_POTENTIAL_DATA.sum() / precision]), np.array([1.0 / precision]), path="mu"
        )
        assert_matches(posterior, reference, label=f"{method} on the model with a potential")

    @pytest.mark.parametrize("case_name", list(canonical.CASES))
    def test_the_pymc_representation_has_the_composed_joints_density(self, case_name):
        """At a draw of the composed joint, its density and its PyMC representation's agree.

        Both are normalized densities of the same variables, so they agree up
        to single-precision rounding, which keeps the representations of a
        canonical case from drifting apart. Each value is cast to the dtype of
        its PyMC variable first, since PyMC scores counts as integers.
        """
        case = canonical.case(case_name)
        with workflow_run(seed=13):
            draw = sample.with_options(raw=True)(case.model)
        pymc = case.pymc_model()
        dtypes = {rv.name: rv.dtype for rv in pymc._pymc_model().free_RVs}
        cast = {name: np.asarray(value).astype(dtypes[name]) for name, value in draw.items()}
        expected = float(log_prob.with_options(raw=True)(case.model, draw))
        assert float(log_prob.with_options(raw=True)(pymc, cast)) == pytest.approx(
            expected, rel=1e-4, abs=1e-3
        )

    def test_a_count_drawn_as_a_float_is_scored(self):
        """The Bernoulli count of ``beta_bernoulli`` is declared an integer, and scores as float32."""
        case = canonical.case("beta_bernoulli")
        pymc = case.pymc_model()
        value = {"theta": np.float32(0.3), "y": np.asarray(case.data["y"], np.float32)}
        assert pymc.event_spec.spec["y"].dtype == np.dtype(jnp.result_type(int))
        expected = (
            scipy.stats.beta.logpdf(0.3, 1, 1)
            + scipy.stats.bernoulli.logpmf(np.asarray(value["y"]), 0.3).sum()
        )
        assert float(log_prob.with_options(raw=True)(pymc, value)) == pytest.approx(
            expected, rel=1e-5
        )

    def test_a_count_at_a_value_that_is_not_an_integer_has_density_zero(self):
        case = canonical.case("beta_bernoulli")
        pymc = case.pymc_model()
        y = np.array(case.data["y"], np.float32)
        y[0] = 0.5
        value = {"theta": np.float32(0.3), "y": y}
        assert float(log_prob.with_options(raw=True)(pymc, value)) == -np.inf


# ---------------------------------------------------------------------------
# A law of an unnormalized density
# ---------------------------------------------------------------------------

#: The precision and the location of the Gaussian whose density the law knows up to a constant.
_PRECISION = np.array([[2.0, 0.5], [0.5, 1.0]])
_LOCATION = np.array([1.0, -1.0])


def _gaussian_density(theta):
    centred = jnp.asarray(theta) - jnp.asarray(_LOCATION, jnp.float32)
    return -0.5 * centred @ jnp.asarray(_PRECISION, jnp.float32) @ centred + 3.0


def _unnormalized() -> Distribution:
    return distribution(
        unnormalized_log_prob=_gaussian_density,
        event_spec=OutputSpec(theta=NumericArraySpec((2,), jnp.float32)),
        label="theta",
    )


def _gaussian_reference():
    cov = np.linalg.inv(_PRECISION)
    return exact_reference(_LOCATION, np.diag(cov), path="theta")


class TestUnnormalizedDensity:
    def test_sampling_selects_a_method_that_normalizes_the_law(self):
        report = sample.check(_unnormalized(), sample_shape=(10,))
        assert (report.feasible, report.route, report.method) == (
            True,
            "normalize",
            "blackjax_nuts",
        )

    def test_sampling_through_a_method_draws_atoms_of_the_normalized_law(self):
        """``sample`` normalizes with the method and resamples its result, so every draw is an atom.

        The same controls in a scope of the same seed run the same chain, so the
        draws of ``sample`` are draws of the atoms ``convert`` returns.
        """
        law = _unnormalized()
        with workflow_run(seed=14):
            draws = sample.with_options(method="blackjax_nuts", method_options=FIT)(
                law, sample_shape=(50,)
            )
        assert draws.batch_shape == (50,)
        assert draws.element_spec == law.event_spec.spec
        with workflow_run(seed=14):
            normalized = convert.with_options(method="blackjax_nuts", method_options=FIT)(
                law, EmpiricalDistribution
            )
        # One row per atom, across the levels chain and draw.
        atoms = np.reshape(np.asarray(normalized.atoms.values), (normalized.num_atoms, -1))
        for draw in np.asarray(draws.values):
            assert np.any(np.all(np.isclose(atoms, draw), axis=1))

    def test_conversion_through_a_method_keeps_the_declaration(self):
        law = _unnormalized()
        with workflow_run(seed=0):
            normalized = convert.with_options(method="blackjax_nuts", method_options=FIT)(
                law, EmpiricalDistribution
            )
        assert isinstance(normalized, EmpiricalDistribution)
        assert normalized.event_spec == law.event_spec

    def test_the_converted_law_has_the_moments_of_the_normalized_density(self):
        """The empirical law of the chains has the Gaussian's moments within the harness's band."""
        with workflow_run(seed=0):
            normalized = convert.with_options(method="blackjax_nuts", method_options=FIT)(
                _unnormalized(), EmpiricalDistribution
            )
        assert_matches(normalized, _gaussian_reference(), label="the converted law")

    def test_conditioning_an_unnormalized_joint_gives_the_closed_form(self):
        """A joint known up to a constant, ``N(mu; 0, 1) N(y; mu, 1)``, conditions to ``N(y / 2, 1 / 2)``."""

        def density(value):
            mu, y = jnp.asarray(value["mu"]), jnp.asarray(value["y"])
            return -0.5 * mu**2 - 0.5 * (y - mu) ** 2 + 7.0

        joint = distribution(
            unnormalized_log_prob=density,
            event_spec=OutputSpec(RecordSpec(mu=REAL, y=REAL)),
            label="joint",
        )
        with workflow_run(seed=0):
            posterior = condition_on.with_options(method="blackjax_nuts", method_options=FIT)(
                joint, {"y": 1.4}
            )
        reference = exact_reference(np.array(0.7), np.array(0.5), path="mu")
        assert_matches(posterior, reference, label="the unnormalized joint conditioned on y")


# ---------------------------------------------------------------------------
# Factored joints
# ---------------------------------------------------------------------------


def _chain():
    """``x ~ N(1, 2)`` and ``y | x ~ N(2x + 1, 0.5)``: ``E[y] = 3``, ``Var y = 16.25``, ``Cov = 8``."""
    x = Normal("x", 1.0, 2.0)
    y = ObservationKernel("y", {"x": x.event_spec.spec}, REAL, lambda x: tfd.Normal(2 * x + 1, 0.5))
    return y * x


class TestFactoredJoints:
    def test_ancestral_draws_have_the_joint_moments(self):
        with workflow_run(seed=15):
            draws = _raw_record(sample(_chain(), sample_shape=(DRAWS,)))
        x, y = np.asarray(draws["x"], np.float64), np.asarray(draws["y"], np.float64)
        _within_mcse([x.mean(), y.mean()], [1.0, 3.0], [2.0, np.sqrt(16.25)])
        _variance_band(np.stack([x, y], axis=1), np.array([4.0, 16.25]))
        product = (x - 1.0) * (y - 3.0)
        _within_mcse(product.mean(), 8.0, product.std())

    def test_the_log_density_is_the_sum_of_the_factor_log_densities(self):
        expected = scipy.stats.norm.logpdf(0.3, 1.0, 2.0) + scipy.stats.norm.logpdf(1.2, 1.6, 0.5)
        value = {"x": 0.3, "y": 1.2}
        assert float(log_prob.with_options(raw=True)(_chain(), value)) == pytest.approx(
            expected, rel=1e-6
        )

    def test_the_edge_free_joint_has_each_factors_moments(self):
        joint = Normal("a", 1.0, 2.0) * Gamma("b", 3.0, 2.0)
        means = mean.with_options(raw=True)(joint)
        variances = variance.with_options(raw=True)(joint)
        assert float(means["mean(a)"]) == pytest.approx(1.0)
        assert float(means["mean(b)"]) == pytest.approx(1.5)
        assert float(variances["variance(a)"]) == pytest.approx(4.0)
        assert float(variances["variance(b)"]) == pytest.approx(0.75)

    def test_a_dependent_joint_claims_no_moment_and_its_mean_is_estimated(self):
        """The mean of ``p(y | x) p(x)`` falls back to Monte Carlo, within four standard errors."""
        joint = _chain()
        assert not isinstance(joint, SupportsMean)
        with workflow_run(seed=16):
            means = mean.with_options(method="monte_carlo", n_broadcast_samples=DRAWS, raw=True)(
                joint
            )
        _within_mcse([means["mean(x)"], means["mean(y)"]], [1.0, 3.0], [2.0, np.sqrt(16.25)])

    def test_the_mean_of_a_dependent_joint_is_exact_at_its_root_factor(self):
        """``by_component`` gives ``E[x] = 1`` and ``Var x = 4`` exactly, and estimates ``y``."""
        joint = _chain()
        assert mean.check(joint).route == "by_component"
        with workflow_run(seed=17):
            means = mean.with_options(n_broadcast_samples=DRAWS, raw=True)(joint)
            variances = variance.with_options(n_broadcast_samples=DRAWS, raw=True)(joint)
        assert float(means["mean(x)"]) == pytest.approx(1.0, abs=1e-6)
        assert float(variances["variance(x)"]) == pytest.approx(4.0, abs=1e-5)
        _within_mcse(means["mean(y)"], 3.0, np.sqrt(16.25))

    def test_the_marginal_of_the_root_factor_is_the_factor(self):
        marginal_x = marginal(_chain(), "x")
        assert float(mean.with_options(raw=True)(marginal_x)) == pytest.approx(1.0)
        assert float(variance.with_options(raw=True)(marginal_x)) == pytest.approx(4.0)

    @pytest.mark.pending(
        reason="marginal's Monte Carlo route returns the empirical marginal",
        raises=ResolutionError,
    )
    def test_the_marginal_of_a_dependent_factor_is_estimated_from_draws(self):
        with workflow_run(seed=17):
            marginal_y = marginal(_chain(), "y")
        assert float(mean.with_options(raw=True)(marginal_y)) == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------------------
# StanModel and the learned kernels
# ---------------------------------------------------------------------------


class TestStanModel:
    @pytest.mark.usefixtures("_stan_toolchain")
    def test_conditioning_a_program_on_its_data_gives_the_closed_form_posterior(self, tmp_path):
        """The Beta-Bernoulli program's posterior is ``Beta(3, 12)``, through the method condition_on selects."""
        case = canonical.case("beta_bernoulli")
        kernel = case.stan_model(tmp_path)
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method_options=PROFILES["cmdstan_nuts"].method_options
            )(kernel, dict(case.stan_data))
        assert_matches(posterior, case.reference, label="the Stan program of beta_bernoulli")


def _bayesflow():
    os.environ.setdefault("KERAS_BACKEND", "jax")
    pytest.importorskip("bayesflow")


def _shift(params, key):
    """The simulator ``y = theta + N(0, 0.5²)`` of two coordinates, whose posterior is Gaussian."""
    theta = theta_vec(params)
    return theta + 0.5 * jax.random.normal(key, theta.shape)


def _shift_simulator(prior):
    return SimulatorKernel(prior, (2,), _shift)


def _shift_prior():
    return MultivariateNormal("theta", jnp.zeros(2), cov=jnp.eye(2))


def _shift_reference(observation):
    """``theta ~ N(0, I)`` and ``y ~ N(theta, 0.25 I)`` give ``theta | y ~ N(0.8 y, 0.2 I)``."""
    return exact_reference(0.8 * np.asarray(observation), np.full(2, 0.2), path="theta")


@pytest.mark.bayesflow
class TestLearnedKernels:
    def test_an_amortized_posterior_locates_the_conjugate_posterior(self):
        """The amortized posterior's law at an observation has the posterior's means.

        The prior is the factored joint of two standard normals, so the law
        exposes the record ``{a, b}``. The law is a biased stand-in, held to the
        harness's biased contract: means within a quarter of a posterior
        deviation beyond the harness's MCSE band. Training the network takes tens
        of seconds.
        """
        _bayesflow()
        from probpipe.inference import learn_amortized_posterior

        prior = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        with workflow_run(seed=0):
            kernel = learn_amortized_posterior(
                prior, _shift_simulator(prior), num_simulations=3000, epochs=8
            )
        observation = jnp.array([0.6, -0.4])
        law = condition_on(kernel, {"observation": observation})
        reference = _shift_reference(observation)
        leaf = reference.leaves["theta"]
        split = PosteriorReference(
            {
                name: LeafReference(
                    leaf.mean[i], leaf.variance[i], {q: v[i] for q, v in leaf.quantiles.items()}
                )
                for i, name in enumerate(("a", "b"))
            },
            "closed form",
        )
        assert_matches(law, split, consistent=False)

    def test_an_amortized_posterior_of_a_whole_term_prior_conditions(self):
        """Conditioning the amortized posterior of a whole-term prior returns a law over that term."""
        _bayesflow()
        from probpipe.inference import learn_amortized_posterior

        with workflow_run(seed=0):
            kernel = learn_amortized_posterior(
                _shift_prior(),
                _shift_simulator(_shift_prior()),
                num_simulations=500,
                epochs=1,
            )
        law = condition_on(kernel, {"observation": jnp.array([0.6, -0.4])})
        assert law.event_spec == _shift_prior().event_spec

    def test_a_learned_likelihood_times_the_prior_is_normalized_by_inference(self):
        """A learned likelihood composed with its prior conditions through an inference method.

        The posterior is a biased stand-in, held to the biased contract;
        training and the fit take tens of seconds.
        """
        _bayesflow()
        from probpipe.inference import learn_amortized_likelihood

        with workflow_run(seed=0):
            likelihood = learn_amortized_likelihood(
                _shift_prior(),
                _shift_simulator(_shift_prior()),
                num_simulations=3000,
                epochs=15,
            )
        observation = jnp.array([[0.6, -0.4]])
        joint = likelihood * _shift_prior()
        with workflow_run(seed=0):
            posterior = condition_on.with_options(method="blackjax_nuts", method_options=FIT)(
                joint, {"observation": observation}
            )
        assert_matches(posterior, _shift_reference(observation[0]), consistent=False)
