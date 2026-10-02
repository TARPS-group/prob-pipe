"""Contracts of the inference-produced distributions (VII.7): ordinary members with a record.

A result is a member of the family that realizes it, it keeps the target's
event declaration under its own label, and its provenance names the method,
the target, and the inputs.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import EmpiricalDistribution, MultivariateNormal
from probpipe.distributions._capabilities import SupportsMean, SupportsSampling, SupportsVariance
from probpipe.inference import rwmh


@pytest.fixture(scope="module")
def target():
    return MultivariateNormal("z", jnp.zeros(2), cov=jnp.eye(2))


@pytest.fixture(scope="module")
def posterior(target):
    return rwmh(dist=target, num_results=40, num_warmup=20, step_size=0.5, random_seed=0)


def test_an_mcmc_result_is_empirical(posterior):
    assert isinstance(posterior, EmpiricalDistribution)


def test_the_result_has_the_capabilities_of_its_family(posterior):
    for protocol in (SupportsSampling, SupportsMean, SupportsVariance):
        assert isinstance(posterior, protocol), protocol.__name__


def test_the_provenance_names_the_method_and_the_target(posterior, target):
    assert posterior.provenance.operation == "blackjax_rwmh"
    (parent,) = posterior.provenance.parents
    assert (parent.type_name, parent.label) == (type(target).__name__, target.label)


def test_the_result_keeps_the_target_component(posterior):
    assert list(posterior.event_spec.components) == ["z"]


def test_the_result_keeps_the_target_packaging(posterior, target):
    assert posterior.event_spec == target.event_spec
