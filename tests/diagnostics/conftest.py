"""Shared fixtures for tests/diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from probpipe import EmpiricalDistribution, NumericArraySpec, OutputSpec, RecordSpec
from probpipe.inference._approximate_distribution import make_posterior


def _posterior(
    param_names: list[str], n_chains: int = 2, n_draws: int = 200, seed: int = 0
) -> EmpiricalDistribution:
    """An inference result with one scalar parameter per name, of standard-normal draws.

    Its atoms lie on the levels ``chain`` and ``draw``, as every inference method's do.
    """
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.standard_normal((n_chains, n_draws)) for _ in param_names], axis=-1)
    event = OutputSpec(RecordSpec(**{p: NumericArraySpec(()) for p in param_names}))
    return make_posterior(list(draws), (), "test", event_spec=event)


@pytest.fixture
def posterior():
    """Two-chain posterior with parameters 'alpha' and 'beta'."""
    return _posterior(["alpha", "beta"])


@pytest.fixture
def posterior_single_chain():
    """Single-chain posterior — triggers NotComputed paths for R-hat."""
    return _posterior(["alpha", "beta"], n_chains=1)


@pytest.fixture
def posterior_3params():
    """Three-parameter posterior for broader coverage."""
    return _posterior(["mu", "sigma", "nu"], n_chains=4, n_draws=500)
