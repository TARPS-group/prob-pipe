"""Shared fixtures for tests/diagnostics."""

from __future__ import annotations

import numpy as np
import pytest


class _FakeRecord(dict):
    """Dict subclass that also supports attribute-style and .fields access."""

    @property
    def fields(self):
        return list(self.keys())

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakePosterior:
    """Minimal ApproximateDistribution stand-in for diagnostics tests.

    Parameters
    ----------
    param_names : list[str]
        Named parameters.
    n_chains : int
        Number of chains.
    n_draws : int
        Draws per chain.
    seed : int
        RNG seed for reproducible draws.
    """

    def __init__(
        self,
        param_names: list[str],
        n_chains: int = 2,
        n_draws: int = 200,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._param_names = param_names
        self._n_chains = n_chains
        self._n_draws = n_draws
        # shape: (n_chains, n_draws) per param
        self._data = {p: rng.standard_normal((n_chains, n_draws)) for p in param_names}
        self._annotations = None

    # ---- ApproximateDistribution protocol expected by diagnostics ----

    @property
    def fields(self) -> list[str]:
        return self._param_names

    @property
    def num_chains(self) -> int:
        return self._n_chains

    @property
    def chains(self) -> list[np.ndarray]:
        # Each element is (n_draws, n_params) — shape doesn't matter for
        # the chain-count; only len(chains) is used in some paths.
        return [np.zeros((self._n_draws, len(self._param_names))) for _ in range(self._n_chains)]

    def draws(self, *, chain: int | None = None) -> _FakeRecord:
        if chain is None:
            return _FakeRecord({p: self._data[p].ravel() for p in self._param_names})
        return _FakeRecord({p: self._data[p][chain] for p in self._param_names})

    def _sample(self, key, shape):
        rng = np.random.default_rng(0)
        return _FakeRecord({p: rng.standard_normal(shape) for p in self._param_names})


@pytest.fixture
def posterior():
    """Two-chain posterior with parameters 'alpha' and 'beta'."""
    return _FakePosterior(["alpha", "beta"])


@pytest.fixture
def posterior_single_chain():
    """Single-chain posterior — triggers NotComputed paths for R-hat."""
    return _FakePosterior(["alpha", "beta"], n_chains=1)


@pytest.fixture
def posterior_3params():
    """Three-parameter posterior for broader coverage."""
    return _FakePosterior(["mu", "sigma", "nu"], n_chains=4, n_draws=500)
