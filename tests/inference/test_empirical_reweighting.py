"""The exact inference method ``empirical_reweighting``: Bayes' rule for an empirical prior.

The posterior of a joint whose prior is an empirical law keeps the prior's
atoms and multiplies each atom's weight by the likelihood of the observed
values there, renormalized. The tests compare it with that computation done by
hand, and check when the method applies.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Normal,
    NumericArraySpec,
    OutputSpec,
    Poisson,
    condition_on,
    conditional_distribution,
    workflow_run,
)
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.distributions import ConditionalDistribution
from probpipe.distributions._capabilities import SupportsConditionalLogProb
from probpipe.inference import inference_method_registry

REAL = NumericArraySpec(())
GRID = jnp.linspace(-2.0, 2.0, 41)
Y = jnp.array([0.4, 0.9, -0.1])


def _grid_prior(weights: Any = None) -> EmpiricalDistribution:
    """An empirical law of ``mu`` on a grid, uniform unless *weights* are given."""
    return EmpiricalDistribution("mu", GRID, weights)


def _normal_kernel() -> ConditionalDistribution:
    """``y_i ~ Normal(mu, 1)`` for the three observations of ``Y``."""
    return conditional_distribution(
        "y", lambda mu: Normal("y", mu * jnp.ones(3), 1.0), given_spec={"mu": REAL}
    )


def _expected_weights(prior_weights: np.ndarray) -> np.ndarray:
    """The posterior weights on the grid, computed by hand."""
    log_likelihood = np.array(
        [jax.scipy.stats.norm.logpdf(Y, mu, 1.0).sum() for mu in np.asarray(GRID)]
    )
    weights = prior_weights * np.exp(log_likelihood - log_likelihood.max())
    return weights / weights.sum()


class _NumpyKernel(ConditionalDistribution, SupportsConditionalLogProb):
    """``y ~ Normal(mu, 1)`` whose density converts its given value to a Python float, so it does not trace."""

    def __init__(self) -> None:
        super().__init__("y", {"mu": REAL}, NumericArraySpec((3,)))

    def _condition_on(self, given: Any, /, **options: Any) -> Any:
        return Normal("y", float(given["mu"]) * jnp.ones(3), 1.0)

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        mu = float(given["mu"])
        return jnp.asarray(np.sum(-0.5 * (np.asarray(value) - mu) ** 2 - 0.5 * np.log(2 * np.pi)))


class TestTheRoute:
    def test_condition_on_selects_the_method_and_reports_it_exact(self):
        report = condition_on.check(_normal_kernel() * _grid_prior(), {"y": Y})
        assert report.selected.method_name == "inference_methods/empirical_reweighting"
        assert report.selected.exact is True

    def test_exact_only_admits_the_method(self):
        posterior = condition_on.with_options(exact_only=True)(
            _normal_kernel() * _grid_prior(), {"y": Y}
        )
        assert posterior.provenance.metadata["method"] == "empirical_reweighting"
        assert posterior.provenance.metadata["exact"] is True

    def test_the_method_is_registered_as_exact(self):
        method = inference_method_registry.get_method("empirical_reweighting")
        assert method.exact is True


class TestThePosterior:
    def test_the_posterior_keeps_the_models_label_and_holds_the_data_fixed(self):
        model = (_normal_kernel() * _grid_prior()).with_label("model")
        posterior = condition_on(model, {"y": Y})
        assert posterior.label == "model"
        assert str(posterior) == posterior.notation == "model(mu; y)"

    def test_the_atoms_are_labeled_by_the_posteriors_components(self):
        prior = EmpiricalDistribution("prior", GRID, event_spec=OutputSpec(mu=None))
        posterior = condition_on(_normal_kernel() * prior, {"y": Y})
        assert posterior.atoms.label == "mu"
        assert prior.atoms.label == "prior"

    def test_the_weights_are_the_prior_weights_times_the_likelihood(self):
        posterior = condition_on(_normal_kernel() * _grid_prior(), {"y": Y})
        np.testing.assert_allclose(
            posterior.weights, _expected_weights(np.ones(41) / 41), atol=1e-6
        )
        np.testing.assert_allclose(np.asarray(posterior.atoms["mu"].values), np.asarray(GRID))

    def test_a_weighted_prior_keeps_its_weights_in_the_product(self):
        prior_weights = np.linspace(1.0, 3.0, 41)
        prior_weights /= prior_weights.sum()
        posterior = condition_on(
            _normal_kernel() * _grid_prior(jnp.asarray(prior_weights)), {"y": Y}
        )
        np.testing.assert_allclose(posterior.weights, _expected_weights(prior_weights), atol=1e-6)

    def test_a_record_prior_keeps_its_fields(self):
        rng = np.random.default_rng(0)
        atoms = NumericRecordBatch(
            "particles",
            {
                "phi": jnp.asarray(rng.uniform(0.4, 0.8, 200), jnp.float32),
                "N": jnp.asarray(rng.uniform(300.0, 500.0, 200), jnp.float32),
            },
            ("particle",),
            axes_per_level=(1,),
        )
        particles = EmpiricalDistribution("particles", atoms)
        observe = conditional_distribution(
            "count",
            lambda phi, N: Poisson("y", phi * N),
            given_spec=particles.event_spec.components,
        )
        posterior = condition_on(observe * particles, {"y": 250.0})
        assert posterior.event_spec == particles.event_spec
        rate = np.asarray(atoms["phi"].values * atoms["N"].values)
        log_likelihood = np.asarray(jax.scipy.stats.poisson.logpmf(250.0, rate))
        expected = np.exp(log_likelihood - log_likelihood.max())
        # The rates near 250 leave float32 log-likelihoods about 1e-4 relative precision.
        np.testing.assert_allclose(posterior.weights, expected / expected.sum(), rtol=2e-4)

    def test_a_likelihood_that_does_not_trace_is_evaluated_at_each_atom(self):
        posterior = condition_on(_NumpyKernel() * _grid_prior(), {"y": Y})
        np.testing.assert_allclose(
            posterior.weights, _expected_weights(np.ones(41) / 41), atol=1e-6
        )

    def test_an_optional_slot_of_the_likelihood_takes_its_default(self):
        kernel = conditional_distribution(
            "y",
            lambda mu, scale=2.0: Normal("y", mu * jnp.ones(3), scale),
            given_spec={"mu": REAL},
        )
        posterior = condition_on(kernel * _grid_prior(), {"y": Y})
        log_likelihood = np.array(
            [jax.scipy.stats.norm.logpdf(Y, mu, 2.0).sum() for mu in np.asarray(GRID)]
        )
        expected = np.exp(log_likelihood - log_likelihood.max())
        np.testing.assert_allclose(posterior.weights, expected / expected.sum(), atol=1e-6)

    def test_observed_values_in_an_xarray_dataarray_give_the_weights_of_the_array(self):
        xr = pytest.importorskip("xarray")
        observed = xr.DataArray(np.asarray(Y), dims="observation")
        posterior = condition_on(_normal_kernel() * _grid_prior(), {"y": observed})
        np.testing.assert_allclose(
            posterior.weights, _expected_weights(np.ones(41) / 41), atol=1e-6
        )

    def test_the_annotations_record_the_method(self):
        posterior = condition_on(_normal_kernel() * _grid_prior(), {"y": Y})
        assert posterior.annotations.attrs["method"] == "empirical_reweighting"

    def test_zero_likelihood_at_every_atom_raises(self):
        prior = EmpiricalDistribution("rate", jnp.array([1.0, 2.0, 3.0]))
        observe = conditional_distribution(
            "count",
            lambda rate: Poisson("y", rate),
            given_spec=prior.event_spec.components,
        )
        with pytest.raises(ValueError, match="zero likelihood at every atom"):
            condition_on.with_options(method="empirical_reweighting")(observe * prior, {"y": -1.0})


class TestWhenItApplies:
    def test_a_parametric_prior_takes_another_method(self):
        model = _normal_kernel() * Normal("mu", 0.0, 1.0)
        info = inference_method_registry.check(
            condition_on.with_options(method="unnormalized")(model, {"y": Y})
        )
        assert info.method_name != "empirical_reweighting"
        report = inference_method_registry.get_method("empirical_reweighting").check(
            condition_on.with_options(method="unnormalized")(model, {"y": Y})
        )
        assert report.feasible is False
        assert "must be an EmpiricalDistribution" in report.description

    def test_a_likelihood_without_a_density_does_not_apply(self):
        class _SamplingOnly(ConditionalDistribution):
            def __init__(self) -> None:
                super().__init__("y", {"mu": REAL}, NumericArraySpec((3,)))

            def _condition_on(self, given: Any, /, **options: Any) -> Any:
                return Normal("y", float(given["mu"]) * jnp.ones(3), 1.0)

        target = condition_on.with_options(method="unnormalized")(
            _SamplingOnly() * _grid_prior(), {"y": Y}
        )
        report = inference_method_registry.get_method("empirical_reweighting").check(target)
        assert report.feasible is False
        assert "SupportsConditionalLogProb" in report.description

    def test_an_empirical_prior_of_draws_is_reweighted(self):
        draws = jax.random.normal(jax.random.PRNGKey(0), (500,))
        prior = EmpiricalDistribution("mu", draws)
        with workflow_run(seed=0):
            posterior = condition_on(_normal_kernel() * prior, {"y": Y})
        assert posterior.num_atoms == 500
        assert float(np.sum(posterior.weights)) == pytest.approx(1.0, rel=1e-5)
