"""Contracts of the program-defined families (VII.9), checked where the adapters are defined today."""

from __future__ import annotations

import pytest

from probpipe.distributions import Distribution
from probpipe.distributions._capabilities import SupportsLogProb, SupportsUnnormalizedLogProb
from probpipe.modeling._pymc import PyMCModel
from probpipe.modeling._stan import StanModel


@pytest.mark.parametrize("adapter", [StanModel, PyMCModel])
def test_an_adapter_is_a_distribution(adapter):
    assert issubclass(adapter, Distribution)


@pytest.mark.pending(
    reason="a Stan adapter declares an unnormalized density unless normalization is established",
    raises=AssertionError,
)
def test_the_stan_adapter_declares_an_unnormalized_density():
    assert issubclass(StanModel, SupportsUnnormalizedLogProb)
    assert not issubclass(StanModel, SupportsLogProb)
