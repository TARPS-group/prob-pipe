"""Where models, inference, and complex records meet: posteriors over record-valued parameters.

- A hierarchical model whose parameters form a nested record, the
  eight-schools model as ``population/mu``, ``population/tau``, and
  ``groups/theta_tilde``, is conditioned on its data through a method and
  checked at the nested paths against the quadrature reference.
- An inference result keeps its target's event declaration, packaging and
  nested paths included, and its views, marginals, and moments read it at the
  target's paths.
- Each model family conditions to a posterior over record-valued parameters:
  a GLM with its dispersion, a ``PyMCModel`` with a mean and a scale, a
  ``StanModel`` with coefficients and a scale, a law of an unnormalized density
  over a nested record, and the factored joint above.

The references of the families with a scale are scale mixtures: given the
scale the coefficients are Gaussian, so a quadrature over the scale gives every
moment and quantile (:class:`tests.inference.canonical.ScaleMixture`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from probpipe import (
    HalfNormal,
    MultivariateNormal,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    OutputSpec,
    Record,
    RecordSpec,
    distribution,
    workflow_run,
)
from probpipe.distributions._factored import _raw_record
from probpipe.families import GaussianFamily, PyMCModel, glm_likelihood
from tests._ops import (
    EmpiricalDistribution,
    FieldView,
    condition_on,
    convert,
    inference_method_registry,
    marginal,
    mean,
    sample,
    variance,
)
from tests._posterior import flat_draws, num_chains
from tests.correctness._laws import (
    GROUPS,
    POPULATION,
    exact_reference,
    nested_schools,
    nested_schools_reference,
)
from tests.correctness._records import leaf_paths
from tests.inference import canonical
from tests.inference._bayesflow_helpers import SimulatorKernel
from tests.inference._harness import PROFILES, assert_matches
from tests.inference.canonical import LeafReference, PosteriorReference, ScaleMixture

#: The controls of a gradient-based fit.
FIT = PROFILES["blackjax_nuts"].method_options


def _from_draws(posterior, target) -> EmpiricalDistribution:
    """The posterior's draws as an EmpiricalDistribution on the levels ``chain`` and ``draw``.

    The draws are read in the target's layout, so the law declares the
    target's record whatever declaration the result itself carries.
    """
    raw = _raw_record(flat_draws(posterior))
    chains = num_chains(posterior)

    def split(leaf):
        values = jnp.asarray(leaf)
        return jnp.reshape(values, (chains, values.shape[0] // chains, *values.shape[1:]))

    batch = NumericRecordBatch(
        "draws",
        jax.tree.map(split, raw),
        ("chain", "draw"),
        element_spec=target.event_spec.spec,
        axes_per_level=(1, 1),
    )
    return EmpiricalDistribution(batch, label=posterior.label)


# ---------------------------------------------------------------------------
# The hierarchical model over a nested parameter record
# ---------------------------------------------------------------------------

#: An initial state inside the support, in the target's flat order
#: ``(groups/theta_tilde, population/mu, population/tau)``; the library's
#: default initial state can fall outside it, which the harness's pending
#: tests record.
_SCHOOLS_INIT = jnp.concatenate(
    [jnp.zeros(canonical.SCHOOL_EFFECTS.shape[0]), jnp.array([0.0, 2.0])]
)


@pytest.fixture(scope="module")
def schools():
    """The nested eight-schools joint, its target, and its blackjax_nuts posterior.

    The fit takes a few seconds and is shared across the module's tests.
    """
    joint = nested_schools()
    given = {"y": jnp.asarray(canonical.SCHOOL_EFFECTS, jnp.float32)}
    target = condition_on.with_options(method="unnormalized")(joint, given)
    with workflow_run(seed=0):
        posterior = condition_on.with_options(
            method="blackjax_nuts", method_options={"init": _SCHOOLS_INIT, **FIT}
        )(joint, given)
    return joint, target, posterior


class TestNestedHierarchicalModel:
    def test_the_target_is_the_nested_record_of_the_unconditioned_fields(self, schools):
        _, target, _ = schools
        assert target.event_spec == OutputSpec(RecordSpec(groups=GROUPS, population=POPULATION))

    def test_the_draws_match_the_reference_at_the_nested_paths(self, schools):
        """The chains, read in the target's layout, recover the eight-schools posterior.

        They are compared as the EmpiricalDistribution of the draws on the
        levels ``chain`` and ``draw``, through the operation model's mean,
        variance, and quantile at ``population/mu``, ``population/tau``, and
        ``groups/theta_tilde``.
        """
        _, target, posterior = schools
        assert_matches(_from_draws(posterior, target), nested_schools_reference())

    def test_the_posterior_matches_the_reference_at_the_nested_paths(self, schools):
        _, _, posterior = schools
        assert_matches(posterior, nested_schools_reference())

    def test_the_posterior_declares_the_targets_nested_record(self, schools):
        _, target, posterior = schools
        assert posterior.event_spec == target.event_spec

    def test_a_view_at_a_nested_path_reads_the_posterior(self, schools):
        _, _, posterior = schools
        view = FieldView(posterior, "population/tau")
        draws = _raw_record(flat_draws(posterior))
        assert float(mean.with_options(raw=True)(view)) == pytest.approx(
            float(np.mean(draws["population"]["tau"])), rel=1e-5
        )

    def test_the_tracked_mean_names_each_component_of_the_targets_schema(self, schools):
        _, target, posterior = schools
        result = mean(posterior)
        assert isinstance(result, NumericRecord)
        children = target.event_spec.spec.children
        assert result.spec == RecordSpec({f"mean({name})": spec for name, spec in children.items()})

    def test_draws_of_the_posterior_keep_the_nested_paths(self, schools):
        _, _, posterior = schools
        with workflow_run(seed=21):
            raw = sample.with_options(raw=True)(posterior, sample_shape=(3,))
        assert isinstance(raw["population"], dict) and set(raw["population"]) == {"mu", "tau"}
        assert np.shape(raw["groups"]["theta_tilde"]) == (3, canonical.SCHOOL_EFFECTS.shape[0])

    def test_the_marginal_of_a_nested_group_is_the_groups_empirical_law(self, schools):
        _, _, posterior = schools
        group = marginal(posterior, "population")
        assert group.event_spec == OutputSpec(population=POPULATION)


# ---------------------------------------------------------------------------
# The declaration an inference result keeps
# ---------------------------------------------------------------------------


def _glm_with_dispersion():
    """The Gaussian GLM with its dispersion a parameter: ``beta ~ N(0, 4 I)``, ``s ~ HalfNormal(2)``."""
    case = canonical.case("gaussian_linear")
    X = jnp.asarray(case.stan_data["X"], jnp.float32)
    joint = (
        glm_likelihood("y", GaussianFamily(), X=X)
        * MultivariateNormal("beta", jnp.zeros(X.shape[1]), cov=4.0 * jnp.eye(X.shape[1]))
        * HalfNormal("dispersion", 2.0)
    )
    return joint, case.data


def _glm_init():
    """An initial state inside the support, in the flat order ``(beta, dispersion)``."""
    return jnp.array([0.0, 0.0, 0.0, 1.0])


def _declaration_cases():
    return [
        pytest.param("gaussian_linear", {}, id="a-vector"),
        pytest.param("beta_bernoulli", {}, id="a-scalar"),
        pytest.param("eight_schools", {}, id="a-positive-scale"),
        pytest.param("glm_with_dispersion", {"init": _glm_init()}, id="two-fields"),
    ]


class TestDeclarations:
    @pytest.mark.parametrize(("name", "controls"), _declaration_cases())
    def test_the_posterior_declares_its_targets_event(self, name, controls):
        """The result of normalizing a target declares the target's event, supports included."""
        if name == "glm_with_dispersion":
            model, data = _glm_with_dispersion()
        else:
            case = canonical.case(name)
            model, data = case.model, case.data
        target = condition_on.with_options(method="unnormalized")(model, data)
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts",
                method_options={**FIT, "num_results": 100, "num_warmup": 50, **controls},
            )(model, data)
        assert posterior.event_spec == target.event_spec

    def test_a_pymc_posterior_declares_its_targets_event(self):
        pytest.importorskip("pymc")
        method = _pymc_method()
        model = PyMCModel(_normal_model, label="normal")
        data = {"y": _NORMAL_DATA}
        target = condition_on.with_options(method="unnormalized")(model, data)
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method=method,
                method_options={
                    **PROFILES[method].method_options,
                    "num_results": 100,
                    "num_warmup": 100,
                },
            )(model, data)
        assert posterior.event_spec == target.event_spec

    def test_an_mcmc_posterior_is_an_empirical_law(self):
        case = canonical.case("gaussian_linear")
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts", method_options={**FIT, "num_results": 100, "num_warmup": 50}
            )(case.model, case.data)
        assert isinstance(posterior, EmpiricalDistribution)


# ---------------------------------------------------------------------------
# Each family with record-valued parameters
# ---------------------------------------------------------------------------


def _glm_reference() -> PosteriorReference:
    """The posterior of ``beta`` and the dispersion ``s``, a scale mixture over ``s``."""
    case = canonical.case("gaussian_linear")
    X = np.asarray(case.stan_data["X"], np.float64)
    y = np.asarray(case.data["y"], np.float64)
    mixture = ScaleMixture.tabulate(
        y,
        log_prior=lambda s: scipy.stats.halfnorm.logpdf(s, scale=2.0),
        design=lambda s: X,
        noise_variance=lambda s: np.full(y.shape[0], s**2),
        prior_variance=np.full(X.shape[1], 4.0),
        log_scale=np.linspace(np.log(1e-3), np.log(20.0), 3000),
    )
    return PosteriorReference(
        {
            "beta": _leaf(*mixture.coefficients(slice(None))),
            "dispersion": _leaf(*mixture.scale()),
        },
        "quadrature",
    )


def _leaf(mean_value, variance_value, ppf) -> LeafReference:
    return LeafReference(
        np.asarray(mean_value),
        np.asarray(variance_value),
        {q: np.asarray(ppf(q)) for q in canonical.INTERVAL_LEVELS},
    )


#: The observations of the normal model with an unknown mean and scale.
_NORMAL_DATA = np.array([1.2, 0.4, 2.3, 1.9, 0.8, 1.5, 2.8, 0.1, 1.1, 1.7])


def _normal_model(y=None):
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=y, shape=_NORMAL_DATA.shape[0])
    return model


def _normal_reference() -> PosteriorReference:
    """``mu ~ N(0, 10)``, ``sigma ~ HalfNormal(1)``, and ``y ~ N(mu, sigma)``, mixed over ``sigma``."""
    n = _NORMAL_DATA.shape[0]
    mixture = ScaleMixture.tabulate(
        _NORMAL_DATA,
        log_prior=lambda s: scipy.stats.halfnorm.logpdf(s, scale=1.0),
        design=lambda s: np.ones((n, 1)),
        noise_variance=lambda s: np.full(n, s**2),
        prior_variance=np.array([100.0]),
        log_scale=np.linspace(np.log(1e-3), np.log(20.0), 3000),
    )
    return PosteriorReference(
        {"mu": _leaf(*mixture.coefficient(0)), "sigma": _leaf(*mixture.scale())}, "quadrature"
    )


def _pymc_method() -> str:
    return (
        "nutpie_nuts" if "nutpie_nuts" in inference_method_registry.list_methods() else "pymc_nuts"
    )


_REGRESSION_PROGRAM = """
data {
  int<lower=0> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] beta;
  real<lower=0> dispersion;
}
model {
  beta ~ normal(0, 2);
  dispersion ~ normal(0, 2);
  y ~ normal(X * beta, dispersion);
}
"""

#: The nested record of the unnormalized Gaussian law, and its means and variances.
_NESTED = RecordSpec(
    theta=RecordSpec(a=NumericArraySpec((2,), jnp.float32), b=NumericArraySpec((), jnp.float32))
)
_NESTED_MEAN = {"theta/a": np.array([1.0, -1.0]), "theta/b": np.array(2.0)}
_NESTED_VARIANCE = {"theta/a": np.array([1.0, 1.0]), "theta/b": np.array(0.25)}


def _nested_density(value):
    """The Gaussian log-density over ``theta/a`` and ``theta/b``, up to a constant."""
    a, b = jnp.asarray(value["theta/a"]), jnp.asarray(value["theta/b"])
    return -0.5 * jnp.sum((a - jnp.array([1.0, -1.0])) ** 2) - 0.5 * (b - 2.0) ** 2 / 0.25 + 4.0


def _nested_reference() -> PosteriorReference:
    leaves = {}
    for path in ("theta/a", "theta/b"):
        reference = exact_reference(_NESTED_MEAN[path], _NESTED_VARIANCE[path], path=path)
        leaves.update(reference.leaves)
    return PosteriorReference(leaves, "closed form")


class TestFamiliesWithRecordParameters:
    def test_a_glm_and_its_dispersion_condition_to_the_scale_mixture(self):
        """The Gaussian GLM with an unknown dispersion recovers the posterior of both fields.

        The fit starts inside the support and takes a few seconds.
        """
        joint, data = _glm_with_dispersion()
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method="blackjax_nuts", method_options={"init": _glm_init(), **FIT}
            )(joint, data)
        assert_matches(posterior, _glm_reference(), label="the GLM with its dispersion")

    def test_a_pymc_model_with_a_mean_and_a_scale_conditions_to_the_scale_mixture(self):
        """A PyMC normal model with an unknown mean and scale recovers both, in a few seconds."""
        pytest.importorskip("pymc")
        method = _pymc_method()
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method=method, method_options=PROFILES[method].method_options
            )(PyMCModel(_normal_model, label="normal"), {"y": _NORMAL_DATA})
        assert_matches(posterior, _normal_reference(), label=f"{method} on the normal model")

    @pytest.mark.usefixtures("_stan_toolchain")
    def test_a_stan_program_with_coefficients_and_a_scale_conditions_to_the_scale_mixture(
        self, tmp_path
    ):
        """A Stan regression with an unknown scale recovers both fields through the selected method."""
        from probpipe.families import StanModel

        case = canonical.case("gaussian_linear")
        path = tmp_path / "regression.stan"
        path.write_text(_REGRESSION_PROGRAM)
        data = {**case.stan_data, "y": np.asarray(case.data["y"], np.float64)}
        with workflow_run(seed=0):
            posterior = condition_on.with_options(
                method_options=PROFILES["cmdstan_nuts"].method_options
            )(StanModel(str(path), label="regression"), data)
        assert_matches(posterior, _glm_reference(), label="the Stan regression")

    def test_an_unnormalized_law_over_a_nested_record_converts_with_its_declaration(self):
        """Conversion through a method returns an empirical law that keeps the nested record."""
        law = distribution(
            unnormalized_log_prob=_nested_density, event_spec=OutputSpec(_NESTED), label="theta"
        )
        with workflow_run(seed=0):
            normalized = convert.with_options(method="blackjax_nuts", method_options=FIT)(
                law, EmpiricalDistribution
            )
        assert normalized.event_spec == law.event_spec
        assert set(leaf_paths(normalized.event_spec.spec)) == {"theta/a", "theta/b"}

    def test_the_converted_nested_law_has_the_gaussian_moments_at_its_paths(self):
        law = distribution(
            unnormalized_log_prob=_nested_density, event_spec=OutputSpec(_NESTED), label="theta"
        )
        with workflow_run(seed=0):
            normalized = convert.with_options(method="blackjax_nuts", method_options=FIT)(
                law, EmpiricalDistribution
            )
        assert_matches(normalized, _nested_reference(), label="the converted nested law")

    def test_the_views_and_marginals_of_the_converted_nested_law_agree(self):
        """A view and a marginal at a nested path read the converted law's draws at that path."""
        law = distribution(
            unnormalized_log_prob=_nested_density, event_spec=OutputSpec(_NESTED), label="theta"
        )
        with workflow_run(seed=0):
            normalized = convert.with_options(method="blackjax_nuts", method_options=FIT)(
                law, EmpiricalDistribution
            )
        parent_mean = mean.with_options(raw=True)(normalized)
        view = FieldView(normalized, "theta/b")
        group = marginal(normalized, "theta")
        np.testing.assert_allclose(
            np.asarray(mean.with_options(raw=True)(view)),
            np.asarray(parent_mean["mean(theta)"]["b"]),
        )
        assert group.event_spec == OutputSpec(theta=_NESTED.children["theta"])
        np.testing.assert_allclose(
            np.asarray(variance.with_options(raw=True)(group)["a"]),
            np.asarray(variance.with_options(raw=True)(normalized)["variance(theta)"]["a"]),
        )

    def test_a_density_over_a_nested_record_reads_an_interior_path(self):
        """A user's density reads a nested field by its key or through the interior node's view."""
        value = Record("value", {"theta": {"a": jnp.ones(2), "b": jnp.array(2.0)}})
        np.testing.assert_allclose(np.asarray(value["theta/a"]), np.ones(2))
        np.testing.assert_allclose(np.asarray(value.at_path("theta")["a"]), np.ones(2))
        with pytest.raises(KeyError):
            value["theta"]


def _schools(params, key):
    """The eight-schools simulator ``y = mu + tau theta_tilde + sigma e``, reading the nested record."""
    mu, tau = params["population/mu"], params["population/tau"]
    theta_tilde = jnp.asarray(params["groups/theta_tilde"])
    sigma = jnp.asarray(canonical.SCHOOL_ERRORS, jnp.float32)
    return mu + tau * theta_tilde + sigma * jax.random.normal(key, theta_tilde.shape)


@pytest.mark.bayesflow
class TestLearnedKernelOverANestedRecord:
    def test_the_amortized_posterior_of_a_nested_prior_draws_the_nested_record(self):
        """An amortized posterior trained on the nested eight-schools prior yields laws over that record.

        Training is brief, since only the declaration and the draws' paths are
        checked; it takes about ten seconds.
        """
        import os

        os.environ.setdefault("KERAS_BACKEND", "jax")
        pytest.importorskip("bayesflow")
        from probpipe.inference import learn_amortized_posterior
        from tests.correctness._laws import Groups, Population

        prior = Groups() * Population()
        with workflow_run(seed=0):
            kernel = learn_amortized_posterior(
                prior,
                SimulatorKernel(prior, canonical.SCHOOL_EFFECTS.shape, _schools),
                num_simulations=500,
                epochs=1,
            )
        assert kernel.event_spec == prior.event_spec
        law = condition_on(
            kernel, {"observation": jnp.asarray(canonical.SCHOOL_EFFECTS, jnp.float32)}
        )
        assert law.event_spec == prior.event_spec
        with workflow_run(seed=22):
            raw = sample.with_options(raw=True)(law, sample_shape=(4,))
        assert np.shape(raw["groups"]["theta_tilde"]) == (4, canonical.SCHOOL_EFFECTS.shape[0])
        assert np.shape(raw["population"]["tau"]) == (4,)
