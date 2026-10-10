"""Constructors take the component of a whole-term event and an optional label (II.4, III.7)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    EmpiricalDistribution,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    OutputSpec,
    conditional_distribution,
    distribution,
)
from probpipe.families import BernoulliFamily, KDEDistribution, MixtureDistribution, glm_likelihood

_SCALAR = NumericArraySpec(())


class TestAFamily:
    def test_its_label_defaults_to_its_class_name(self):
        law = Normal("mu", 0.0, 1.0)
        assert (law.label, list(law.event_spec.components), law.notation) == (
            "Normal",
            ["mu"],
            "Normal(mu)",
        )

    def test_a_given_label_names_the_law(self):
        law = Normal("mu", 0.0, 1.0, label="prior")
        assert (law.label, law.notation) == ("prior", "prior(mu)")

    def test_an_event_spec_of_another_component_raises(self):
        with pytest.raises(ValueError, match="has the component 'mu', but its event_spec names"):
            Normal("mu", 0.0, 1.0, event_spec=OutputSpec(theta=None))

    def test_a_component_that_is_not_a_string_raises(self):
        with pytest.raises(TypeError, match="component of its event as its first argument"):
            Normal(0.0, 0.0, 1.0)

    def test_an_empty_label_raises(self):
        with pytest.raises(TypeError, match="label must be a non-empty string"):
            Normal("mu", 0.0, 1.0, label="")


class TestAnEmpiricalLaw:
    def test_array_atoms_take_the_component_and_the_label_p(self):
        law = EmpiricalDistribution(jnp.arange(3.0), component="theta")
        assert (law.label, list(law.event_spec.components)) == ("p", ["theta"])
        assert (law.atoms.label, law.atoms.level_names) == ("theta", ("theta",))

    def test_array_atoms_without_a_component_raise(self):
        with pytest.raises(TypeError, match="needs the component of its event"):
            EmpiricalDistribution(jnp.arange(3.0))

    def test_record_atoms_refuse_a_component(self):
        atoms = NumericRecordBatch("atoms", {"a": jnp.arange(3.0)}, "draw")
        with pytest.raises(TypeError, match="takes no component"):
            EmpiricalDistribution(atoms, component="a")
        assert list(EmpiricalDistribution(atoms).event_spec.components) == ["a"]

    def test_an_event_spec_names_the_component_in_place_of_component(self):
        law = EmpiricalDistribution(jnp.arange(3.0), event_spec=OutputSpec(theta=None))
        assert list(law.event_spec.components) == ["theta"]

    def test_a_kde_follows_the_empirical_law(self):
        kde = KDEDistribution(jnp.array([0.0, 1.0, 3.0]), 0.5, component="x")
        assert (kde.label, list(kde.event_spec.components)) == ("KDEDistribution", ["x"])
        with pytest.raises(TypeError, match="needs the component"):
            KDEDistribution(jnp.array([0.0, 1.0, 3.0]), 0.5)


class TestALawFromFunctions:
    def test_a_bare_spec_takes_the_component(self):
        law = distribution(
            sample=lambda key: jax.random.normal(key), event_spec=_SCALAR, component="z"
        )
        assert (law.label, law.notation) == ("p", "p(z)")

    def test_a_bare_spec_without_a_component_raises(self):
        with pytest.raises(TypeError, match="needs a component"):
            distribution(sample=lambda key: jax.random.normal(key), event_spec=_SCALAR)

    def test_an_output_spec_refuses_a_component(self):
        with pytest.raises(TypeError, match="names its own components"):
            distribution(
                sample=lambda key: jax.random.normal(key),
                event_spec=OutputSpec(z=_SCALAR),
                component="z",
            )


class TestAKernel:
    def test_a_lambda_is_labeled_p_and_a_def_by_its_name(self):
        def y_given_mu(mu):
            return Normal("y", mu, 1.0)

        assert conditional_distribution(y_given_mu, given_spec={"mu": _SCALAR}).label == (
            "y_given_mu"
        )
        kernel = conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), given_spec={"mu": _SCALAR}
        )
        assert (kernel.label, kernel.notation) == ("p", "p(y | mu)")

    def test_a_label_first_raises(self):
        with pytest.raises(TypeError, match="takes the function first"):
            conditional_distribution("lik")

    def test_a_glm_likelihood_takes_its_component_and_the_label_p(self):
        glm = glm_likelihood("damage", BernoulliFamily())
        assert (glm.label, list(glm.event_spec.components)) == ("p", ["damage"])


def test_a_mixture_takes_its_label_as_a_keyword():
    laws = [Normal("x", 0.0, 1.0), Normal("x", 1.0, 1.0)]
    assert MixtureDistribution(laws, jnp.array([0.5, 0.5])).label == "MixtureDistribution"
    assert MixtureDistribution(laws, jnp.array([0.5, 0.5]), label="mix").label == "mix"
