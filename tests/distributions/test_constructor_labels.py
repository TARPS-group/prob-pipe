"""Constructors take the output of a whole-term event and an optional label (II.4, III.7)."""

from __future__ import annotations

import re

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
from tests._families import FAMILIES

_SCALAR = NumericArraySpec(())


class TestAFamily:
    def test_its_label_defaults_to_its_class_name(self):
        law = Normal("mu", 0.0, 1.0)
        assert (law.label, list(law.event_spec.outputs), law.notation) == (
            "Normal",
            ["mu"],
            "Normal(mu)",
        )

    @pytest.mark.parametrize("make", list(FAMILIES.values()), ids=list(FAMILIES))
    def test_each_family_defaults_its_label_to_its_class_name(self, make):
        law = make()
        name = type(law).__name__
        assert (law.label, law.notation) == (name, f"{name}(x)")
        assert "label=" not in repr(law)

    def test_a_given_label_names_the_law(self):
        law = Normal("mu", 0.0, 1.0, label="prior")
        assert (law.label, law.notation) == ("prior", "prior(mu)")

    def test_an_event_spec_of_another_output_raises(self):
        with pytest.raises(ValueError, match="has the output 'mu', but its event_spec names"):
            Normal("mu", 0.0, 1.0, event_spec=OutputSpec(theta=None))

    def test_an_output_that_is_not_a_string_raises(self):
        with pytest.raises(TypeError, match="output of its event as its first argument"):
            Normal(0.0, 0.0, 1.0)

    def test_an_empty_label_raises(self):
        with pytest.raises(TypeError, match="label must be a non-empty string"):
            Normal("mu", 0.0, 1.0, label="")


class TestAnEmpiricalLaw:
    def test_array_atoms_take_the_output_and_the_generic_label(self):
        law = EmpiricalDistribution(jnp.arange(3.0), output_name="theta")
        assert (law.label, list(law.event_spec.outputs)) == ("ℙ", ["theta"])
        assert (law.atoms.label, law.atoms.level_names) == ("theta", ("theta",))

    def test_array_atoms_without_an_output_raise(self):
        with pytest.raises(TypeError, match="needs the output of its event"):
            EmpiricalDistribution(jnp.arange(3.0))

    def test_record_atoms_refuse_an_output(self):
        atoms = NumericRecordBatch(
            {"a": jnp.arange(3.0)},
            "draw",
            label="atoms",
        )
        with pytest.raises(TypeError, match="takes no output"):
            EmpiricalDistribution(atoms, output_name="a")
        assert list(EmpiricalDistribution(atoms).event_spec.outputs) == ["a"]

    def test_an_event_spec_names_the_output_in_place_of_output(self):
        law = EmpiricalDistribution(jnp.arange(3.0), event_spec=OutputSpec(theta=None))
        assert list(law.event_spec.outputs) == ["theta"]

    @pytest.mark.parametrize("constructor", [EmpiricalDistribution, KDEDistribution])
    def test_a_label_passed_first_raises_with_the_call_to_write(self, constructor):
        call = f"{constructor.__name__}(atoms, label='theta')"
        with pytest.raises(TypeError, match=re.escape(call)):
            constructor("theta", jnp.arange(3.0))

    def test_a_kde_follows_the_empirical_law(self):
        kde = KDEDistribution(jnp.array([0.0, 1.0, 3.0]), 0.5, output_name="x")
        assert (kde.label, list(kde.event_spec.outputs)) == ("KDEDistribution", ["x"])
        with pytest.raises(TypeError, match="needs the output"):
            KDEDistribution(jnp.array([0.0, 1.0, 3.0]), 0.5)


class TestALawFromFunctions:
    def test_a_bare_spec_takes_the_output(self):
        law = distribution(
            sample=lambda key: jax.random.normal(key), event_spec=_SCALAR, output_name="z"
        )
        assert (law.label, law.notation) == ("ℙ", "ℙ(z)")

    def test_a_bare_spec_without_an_output_raises(self):
        with pytest.raises(TypeError, match="needs an output"):
            distribution(sample=lambda key: jax.random.normal(key), event_spec=_SCALAR)

    def test_an_output_spec_refuses_an_output(self):
        with pytest.raises(TypeError, match="names its own outputs"):
            distribution(
                sample=lambda key: jax.random.normal(key),
                event_spec=OutputSpec(z=_SCALAR),
                output_name="z",
            )


class TestAKernel:
    def test_a_lambda_uses_the_generic_label_and_a_def_by_its_name(self):
        def y_given_mu(mu):
            return Normal("y", mu, 1.0)

        assert conditional_distribution(y_given_mu, given_spec={"mu": _SCALAR}).label == (
            "y_given_mu"
        )
        kernel = conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), given_spec={"mu": _SCALAR}
        )
        assert (kernel.label, kernel.notation) == ("ℙ", "ℙ(y | mu)")

    def test_a_label_first_raises_with_the_call_to_write(self):
        with pytest.raises(
            TypeError, match=re.escape("conditional_distribution(function, label='lik')")
        ):
            conditional_distribution("lik")

    def test_a_glm_likelihood_takes_its_output_and_the_generic_label(self):
        glm = glm_likelihood("damage", BernoulliFamily())
        assert (glm.label, list(glm.event_spec.outputs)) == ("ℙ", ["damage"])


def test_a_mixture_takes_its_label_as_a_keyword():
    laws = [Normal("x", 0.0, 1.0), Normal("x", 1.0, 1.0)]
    assert MixtureDistribution(laws, jnp.array([0.5, 0.5])).label == "mixture"
    assert MixtureDistribution(laws, jnp.array([0.5, 0.5]), label="mix").label == "mix"
    with pytest.raises(
        TypeError, match=re.escape("MixtureDistribution(components, weights, label='mix')")
    ):
        MixtureDistribution("mix", laws)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Normal("mu", 0.0, 1.0, label=3),
        lambda: EmpiricalDistribution(jnp.arange(3.0), output_name="mu", label=3),
        lambda: distribution(
            sample=lambda key: jax.random.normal(key), event_spec=_SCALAR, output_name="z", label=3
        ),
        lambda: conditional_distribution(
            lambda mu: Normal("y", mu, 1.0), given_spec={"mu": _SCALAR}, label=3
        ),
    ],
    ids=["family", "empirical", "distribution", "conditional_distribution"],
)
def test_every_label_check_names_the_bad_value(build):
    with pytest.raises(TypeError, match="label must be a non-empty string, got 3"):
        build()
