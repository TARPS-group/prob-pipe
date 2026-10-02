"""Contracts of the program-defined families (VII.9): StanModel, PyMCModel, and UnnormalizedDistribution."""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as st

import probpipe
from probpipe import NumericArraySpec, OutputSpec, RecordSpec
from probpipe.core.constraints import (
    greater_than,
    interval,
    positive,
    positive_definite,
    real,
    simplex,
    unit_interval,
)
from probpipe.distributions import ConditionalDistribution, Distribution
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalSampling,
    SupportsConditionalUnnormalizedLogProb,
    SupportsLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    _is_normalized,
    _kernel_is_normalized,
)
from probpipe.families import PyMCModel, StanModel, UnnormalizedDistribution

_REGRESSION = """
// A linear regression, whose coefficient count is a data entry.
data {
  int<lower=0> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;  /* the responses */
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0, 1);
  y ~ normal(X * beta, sigma);
}
"""


@pytest.fixture
def regression_file(tmp_path):
    path = tmp_path / "regression.stan"
    path.write_text(_REGRESSION)
    return str(path)


def _data():
    return {"N": 3, "K": 2, "X": np.ones((3, 2)), "y": np.zeros(3)}


#: The dtypes the array backend gives a Stan real and a Stan int.
_FLOAT, _INT = jnp.result_type(float), jnp.result_type(int)

_CONSTRAINED = """
parameters {
  real<lower=-1, upper=1> rho;
  real<lower=2> above;
  real<upper=0> below;
  real<offset=1, multiplier=2> shifted;
  simplex[3] w;
  cov_matrix[2] S;
  ordered[2] cut;
}
model { }
"""


@pytest.mark.usefixtures("_stanc")
class TestStanModel:
    def test_the_given_slots_are_the_typed_data_block_entries(self, regression_file):
        model = StanModel("regression", regression_file)
        assert isinstance(model, ConditionalDistribution)
        assert dict(model.given_spec) == {
            "N": NumericArraySpec((), _INT),
            "K": NumericArraySpec((), _INT),
            "X": NumericArraySpec(("N", "K"), _FLOAT),
            "y": NumericArraySpec(("N",), _FLOAT),
        }

    def test_the_event_is_the_parameter_record_with_dtypes_and_supports(self, regression_file):
        model = StanModel("regression", regression_file)
        assert model.event_spec == OutputSpec(
            RecordSpec(
                beta=NumericArraySpec(("K",), _FLOAT, real),
                sigma=NumericArraySpec((), _FLOAT, positive),
            )
        )

    def test_each_parameter_carries_its_declared_constraint(self, tmp_path):
        path = tmp_path / "constrained.stan"
        path.write_text(_CONSTRAINED)
        assert dict(StanModel("constrained", str(path)).supports) == {
            "rho": interval(-1.0, 1.0),
            "above": greater_than(2.0),
            "below": None,
            "shifted": real,
            "w": simplex,
            "S": positive_definite,
            "cut": None,
        }

    def test_a_program_stanc_rejects_raises_value_error(self, tmp_path):
        path = tmp_path / "broken.stan"
        path.write_text("parameters { real x } model { }")
        with pytest.raises(ValueError, match="stanc"):
            StanModel("broken", str(path))

    def test_it_claims_the_conditional_unnormalized_density_alone(self, regression_file):
        model = StanModel("regression", regression_file)
        assert isinstance(model, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(model, SupportsConditionalLogProb)
        assert not _kernel_is_normalized(model)

    def test_data_given_at_construction_curry_the_program(self, regression_file):
        model = StanModel("regression", regression_file, data={"N": 3, "K": 2})
        assert list(model.given_spec) == ["X", "y"]
        assert model.event_spec.spec["beta"].shape == (2,)

    def test_binding_every_entry_returns_the_unnormalized_posterior(self, regression_file):
        posterior = StanModel("regression", regression_file)._condition_on(_data())
        assert isinstance(posterior, Distribution)
        assert isinstance(posterior, SupportsUnnormalizedLogProb)
        assert not _is_normalized(posterior)
        assert posterior.event_spec.spec["beta"] == NumericArraySpec((2,), _FLOAT, real)

    def test_a_construction_that_binds_every_entry_is_the_posterior(self, regression_file):
        posterior = StanModel("regression", regression_file, data=_data())
        assert isinstance(posterior, Distribution)
        assert not isinstance(posterior, ConditionalDistribution)

    def test_the_posterior_carries_its_program_and_data(self, regression_file):
        posterior = StanModel("regression", regression_file, data=_data())
        assert posterior.stan_file == regression_file
        assert set(posterior.data) == {"N", "K", "X", "y"}

    def test_the_posterior_pickles(self, regression_file):
        posterior = StanModel("regression", regression_file, data=_data())
        restored = pickle.loads(pickle.dumps(posterior))
        assert (restored.name, restored.spec, restored.stan_file) == (
            posterior.name,
            posterior.spec,
            posterior.stan_file,
        )

    def test_an_entry_the_data_block_does_not_declare_raises(self, regression_file):
        with pytest.raises(KeyError, match="data-block"):
            StanModel("regression", regression_file, data={"M": 3})

    def test_a_program_without_a_data_block_is_its_posterior(self, tmp_path):
        path = tmp_path / "prior.stan"
        path.write_text("parameters { array[2] real<lower=-1, upper=1> z; } model { }")
        law = StanModel("prior", str(path))
        assert isinstance(law, Distribution)
        assert law.event_spec.spec["z"].shape == (2,)

    def test_the_stan_methods_read_the_posterior(self, regression_file):
        posterior = StanModel("regression", regression_file, data=_data())
        registry = probpipe.inference_method_registry
        for name in ("cmdstan_nuts", "nutpie_nuts"):
            if name in registry.list_methods():
                assert type(posterior) in (registry.get_method(name).supported_types()), name

    @pytest.mark.usefixtures("_stan_toolchain")
    def test_the_density_is_bridgestans_without_the_jacobian(self, regression_file):
        posterior = StanModel("regression", regression_file, data=_data())
        value = probpipe.Record("value", {"beta": jnp.array([0.1, -0.2]), "sigma": 1.5})
        expected = (
            st.norm.logpdf([0.1, -0.2]).sum()
            + st.norm.logpdf(np.zeros(3), np.ones((3, 2)) @ np.array([0.1, -0.2]), 1.5).sum()
        )
        difference = float(posterior._unnormalized_log_prob(value)) - expected
        other = probpipe.Record("value", {"beta": jnp.array([0.4, 0.3]), "sigma": 0.7})
        expected_other = (
            st.norm.logpdf([0.4, 0.3]).sum()
            + st.norm.logpdf(np.zeros(3), np.ones((3, 2)) @ np.array([0.4, 0.3]), 0.7).sum()
        )
        assert float(posterior._unnormalized_log_prob(other)) - expected_other == pytest.approx(
            difference, abs=1e-6
        )


def _normal_model(y=None):
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=y)
    return model


def _regression(x=None, y=None):
    x = np.zeros(3) if x is None else np.asarray(x)
    import pymc as pm

    with pm.Model() as model:
        beta = pm.Normal("beta", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", beta * x, sigma, observed=y)
    return model


def _constrained_model(y=None):
    import pymc as pm

    with pm.Model() as model:
        pm.Normal("a", 0, 1)
        pm.HalfCauchy("b", 1.0)
        pm.Uniform("c", -1.0, 2.0)
        pm.Beta("d", 1.0, 1.0)
        pm.Dirichlet("e", np.ones(3))
        pm.Poisson("y", 2.0, observed=y)
    return model


def _flat_prior(y=None):
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Flat("mu")
        pm.Normal("y", mu, 1.0, observed=y)
    return model


def _with_potential(y=None):
    import pymc as pm

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Potential("penalty", -(mu**2))
        pm.Normal("y", mu, 1.0, observed=y)
    return model


def _half_flat_prior(y=None):
    import pymc as pm

    with pm.Model() as model:
        sigma = pm.HalfFlat("sigma")
        pm.Normal("y", 0.0, sigma, observed=y)
    return model


def _flat_regression(x=None, y=None):
    x = np.zeros(3) if x is None else np.asarray(x)
    import pymc as pm

    with pm.Model() as model:
        beta = pm.Flat("beta")
        pm.Normal("y", beta * x, 1.0, observed=y)
    return model


def _penalized_regression(x=None, y=None):
    x = np.zeros(3) if x is None else np.asarray(x)
    import pymc as pm

    with pm.Model() as model:
        beta = pm.Normal("beta", 0, 1)
        pm.Potential("penalty", -(beta**2))
        pm.Normal("y", beta * x, 1.0, observed=y)
    return model


class TestPyMCModel:
    @pytest.fixture(autouse=True)
    def _pymc(self):
        pytest.importorskip("pymc")

    def test_it_is_the_joint_law_of_its_free_variables(self):
        model = PyMCModel("normal", _normal_model)
        assert isinstance(model, Distribution)
        assert tuple(model.event_spec.components) == ("mu", "sigma", "y")

    def test_it_claims_sampling_and_a_normalized_density(self):
        model = PyMCModel("normal", _normal_model)
        assert isinstance(model, SupportsSampling)
        assert isinstance(model, SupportsLogProb)
        assert _is_normalized(model)

    def test_its_density_is_the_joint_density_of_the_free_variables(self):
        model = PyMCModel("normal", _normal_model)
        value = probpipe.Record("value", {"mu": 0.3, "sigma": 1.2, "y": 0.5})
        expected = (
            st.norm.logpdf(0.3, 0, 10) + st.halfnorm.logpdf(1.2) + st.norm.logpdf(0.5, 0.3, 1.2)
        )
        assert float(model._log_prob(value)) == pytest.approx(expected, rel=1e-6)

    def test_the_event_carries_the_variables_dtypes_and_supports(self):
        spec = PyMCModel("m", _constrained_model).event_spec.spec
        assert spec["a"] == NumericArraySpec((), _FLOAT, real)
        assert spec["b"] == NumericArraySpec((), _FLOAT, positive)
        assert spec["c"].support == interval(-1.0, 2.0)
        assert spec["d"].support == unit_interval
        assert spec["e"] == NumericArraySpec((3,), _FLOAT, simplex)
        assert spec["y"] == NumericArraySpec((), _INT, None)

    def test_the_posterior_record_carries_the_dtypes_and_supports(self):
        model = PyMCModel("m", _constrained_model)
        conditioned = model._pymc_model(data={"y": np.array(2)})
        record = model._parameter_record_for(conditioned, ("a", "b", "e"))
        assert record == RecordSpec(
            a=NumericArraySpec((), _FLOAT, real),
            b=NumericArraySpec((), _FLOAT, positive),
            e=NumericArraySpec((3,), _FLOAT, simplex),
        )

    def test_it_samples_the_prior_predictive_of_every_free_variable(self):
        model = PyMCModel("normal", _normal_model)
        draws = model._sample(jax.random.PRNGKey(0), (4,))
        assert {name: np.shape(draws[name]) for name in ("mu", "sigma", "y")} == {
            "mu": (4,),
            "sigma": (4,),
            "y": (4,),
        }
        again = model._sample(jax.random.PRNGKey(0), (4,))
        assert np.allclose(np.asarray(draws["mu"]), np.asarray(again["mu"]))

    @pytest.mark.parametrize("model_fn", [_flat_prior, _with_potential])
    def test_a_potential_or_an_improper_prior_leaves_the_density_unnormalized(self, model_fn):
        model = PyMCModel("model", model_fn)
        assert isinstance(model, SupportsUnnormalizedLogProb)
        assert not isinstance(model, SupportsLogProb)

    @pytest.mark.parametrize("model_fn", [_flat_prior, _half_flat_prior, _with_potential])
    def test_a_potential_or_an_improper_prior_claims_no_sampling(self, model_fn):
        model = PyMCModel("model", model_fn)
        assert not isinstance(model, SupportsSampling)
        assert not _is_normalized(model)

    @pytest.mark.parametrize("model_fn", [_flat_regression, _penalized_regression])
    def test_a_kernel_with_a_potential_or_an_improper_prior_claims_no_sampling(self, model_fn):
        kernel = PyMCModel("regression", model_fn)
        assert isinstance(kernel, ConditionalDistribution)
        assert isinstance(kernel, SupportsConditionalUnnormalizedLogProb)
        assert not isinstance(kernel, SupportsConditionalSampling)
        assert not _kernel_is_normalized(kernel)
        law = probpipe.operations._condition.condition_on.with_options(method="unnormalized")(
            kernel, {"x": np.linspace(0, 1, 3)}
        )
        assert not _is_normalized(law)

    def test_a_model_with_a_covariate_is_a_kernel_over_it(self):
        kernel = PyMCModel("regression", _regression)
        assert isinstance(kernel, ConditionalDistribution)
        assert list(kernel.given_spec) == ["x"]
        assert tuple(kernel.event_spec.components) == ("beta", "sigma", "y")
        assert isinstance(kernel, SupportsConditionalSampling)
        assert _kernel_is_normalized(kernel)

    def test_binding_the_covariate_returns_the_joint_law_there(self):
        kernel = PyMCModel("regression", _regression)
        law = probpipe.operations._condition.condition_on(kernel, {"x": np.linspace(0, 1, 5)})
        assert isinstance(law, PyMCModel)
        assert law.event_spec.spec["y"].shape == (5,)

    def test_it_pickles(self):
        model = PyMCModel("normal", _normal_model)
        restored = pickle.loads(pickle.dumps(model))
        assert (restored.name, restored.spec) == (model.name, model.spec)


class TestUnnormalizedDistribution:
    def _law(self):
        return UnnormalizedDistribution(
            "u", lambda x: -0.5 * jnp.sum(x**2), OutputSpec(x=NumericArraySpec((2,)))
        )

    def test_it_claims_the_unnormalized_density_alone(self):
        law = self._law()
        assert isinstance(law, SupportsUnnormalizedLogProb)
        assert not isinstance(law, SupportsLogProb)
        assert not isinstance(law, SupportsSampling)
        assert not _is_normalized(law)

    def test_its_density_is_the_users(self):
        assert float(self._law()._unnormalized_log_prob(jnp.ones(2))) == pytest.approx(-1.0)

    def test_a_log_density_that_is_not_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            UnnormalizedDistribution("u", 1.0, OutputSpec(x=NumericArraySpec((2,))))

    def test_the_package_exports_it(self):
        assert probpipe.UnnormalizedDistribution is UnnormalizedDistribution
