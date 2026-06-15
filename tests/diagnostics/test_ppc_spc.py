"""Tests for probpipe.diagnostics._ppc_spc."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest
import xarray as xr

from probpipe.diagnostics._ppc_spc import (
    _observed_data_to_dataset,
    _replicated_data_to_dataset,
    _replicated_statistics_summary,
    add_ppc,
    add_spc,
)
from probpipe.diagnostics._views import DiagnosticsView, PPCView

# conftest.py provides: posterior, posterior_3params


# ---------------------------------------------------------------------------
# Fake GenerativeLikelihood (pure NumPy — no JAX required)
# ---------------------------------------------------------------------------


class _NumpyLikelihood:
    """Generates i.i.d. Normal(0, 1) replicated data.

    Does NOT accept a ``key`` keyword so _predictive_check_loop is used.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def generate_data(self, params, n_samples: int) -> np.ndarray:
        return self._rng.standard_normal(n_samples)


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------


def _mean(y: np.ndarray) -> float:
    return float(np.mean(y))


def _std(y: np.ndarray) -> float:
    return float(np.std(y))


# ---------------------------------------------------------------------------
# _observed_data_to_dataset
# ---------------------------------------------------------------------------


class TestObservedDataToDataset:
    def test_scalar_input(self):
        ds = _observed_data_to_dataset(3.14)
        assert "y" in ds.data_vars
        assert ds["y"].shape == ()

    def test_1d_input(self):
        ds = _observed_data_to_dataset(np.array([1.0, 2.0, 3.0]))
        assert ds["y"].dims == ("obs",)
        assert ds["y"].shape == (3,)

    def test_2d_input(self):
        ds = _observed_data_to_dataset(np.ones((4, 5)))
        assert ds["y"].ndim == 2

    def test_custom_var_name(self):
        ds = _observed_data_to_dataset(np.array([1.0]), var_name="x")
        assert "x" in ds.data_vars

    def test_returns_dataset(self):
        assert isinstance(_observed_data_to_dataset(np.array([1.0])), xr.Dataset)


# ---------------------------------------------------------------------------
# _replicated_data_to_dataset
# ---------------------------------------------------------------------------


class TestReplicatedDataToDataset:
    def test_scalar_gets_chain_draw_dims(self):
        ds = _replicated_data_to_dataset(np.float64(1.0))
        assert "chain" in ds["y"].dims
        assert "draw" in ds["y"].dims

    def test_1d_adds_chain_dim(self):
        ds = _replicated_data_to_dataset(np.ones(10))
        assert ds["y"].shape == (1, 10)

    def test_2d_adds_chain_dim(self):
        ds = _replicated_data_to_dataset(np.ones((5, 3)))
        assert ds["y"].shape == (1, 5, 3)

    def test_3d_kept_as_is(self):
        ds = _replicated_data_to_dataset(np.ones((2, 5, 3)))
        assert ds["y"].shape == (2, 5, 3)
        assert ds["y"].dims == ("chain", "draw", "obs")


# ---------------------------------------------------------------------------
# _replicated_statistics_summary
# ---------------------------------------------------------------------------


class TestReplicatedStatisticsSummary:
    def test_basic_summary(self):
        stats = {"mean_fn": np.array([0.1, 0.2, 0.3])}
        result = _replicated_statistics_summary(stats)
        assert result is not None
        assert "replicated_stat_mean" in result
        assert "replicated_stat_sd" in result

    def test_empty_dict_returns_none(self):
        assert _replicated_statistics_summary({}) is None

    def test_none_values_produce_nan(self):
        result = _replicated_statistics_summary({"fn": None})
        assert result is None  # all-NaN → no available stats

    def test_empty_array_produces_nan(self):
        result = _replicated_statistics_summary({"fn": np.array([])})
        assert result is None

    def test_multiple_fns(self):
        stats = {
            "mean_fn": np.ones(100),
            "std_fn": np.ones(100) * 2,
        }
        result = _replicated_statistics_summary(stats)
        assert result is not None
        assert len(result["replicated_stat_mean"]) == 2

    def test_quantiles_present(self):
        stats = {"fn": np.linspace(0, 1, 100)}
        result = _replicated_statistics_summary(stats)
        assert "replicated_stat_q05" in result
        assert "replicated_stat_q95" in result


# ---------------------------------------------------------------------------
# add_ppc
# ---------------------------------------------------------------------------


class TestAddPpc:
    def test_writes_ppc_group(self, posterior):
        observed = np.random.default_rng(0).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        assert posterior._auxiliary is not None
        ppc_ds = posterior._auxiliary["diagnostics"]["runs"]["ppc"].to_dataset()
        assert "p_value" in ppc_ds.data_vars

    def test_p_value_in_range(self, posterior):
        observed = np.random.default_rng(1).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=50,
        )
        view = PPCView(
            posterior._auxiliary["diagnostics"]["runs"]["ppc"]
        )
        p = view.p_values.get("_mean")
        if isinstance(p, float):
            assert 0.0 <= p <= 1.0

    def test_multiple_test_fns(self, posterior):
        observed = np.random.default_rng(2).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=[_mean, _std],
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        view = PPCView(
            posterior._auxiliary["diagnostics"]["runs"]["ppc"]
        )
        assert set(view.p_values.keys()) == {"_mean", "_std"}

    def test_observed_stored(self, posterior):
        observed = np.random.default_rng(3).standard_normal(30)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        view = PPCView(
            posterior._auxiliary["diagnostics"]["runs"]["ppc"]
        )
        obs = view.observed.get("_mean")
        if isinstance(obs, float):
            assert np.isfinite(obs)

    def test_returns_none(self, posterior):
        observed = np.ones(10)
        result = add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=5,
        )
        assert result is None

    def test_raises_without_observed_data(self, posterior):
        with pytest.raises(ValueError):
            add_ppc(
                posterior,
                test_fns=_mean,
                observed_data=None,
                generative_likelihood=_NumpyLikelihood(),
                n_replications=5,
            )

    def test_diagnostics_view_integration(self, posterior):
        observed = np.random.default_rng(4).standard_normal(40)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        assert view.ppc.exists


# ---------------------------------------------------------------------------
# add_spc
# ---------------------------------------------------------------------------


class TestAddSpc:
    def _make_posteriors(self, n: int):
        """Return n independent _FakePosterior instances."""
        import numpy as np

        class _FakeRecord(dict):
            @property
            def fields(self):
                return list(self.keys())

        class _FakePosterior:
            def __init__(self, seed):
                rng = np.random.default_rng(seed)
                self._data = {
                    "alpha": rng.standard_normal((2, 200)),
                    "beta": rng.standard_normal((2, 200)),
                }
                self._auxiliary = None
                self.fields = ["alpha", "beta"]
                self.num_chains = 2

            @property
            def chains(self):
                return [np.zeros((200, 2)) for _ in range(2)]

            def draws(self, *, chain=None):
                if chain is None:
                    return _FakeRecord({p: self._data[p].ravel() for p in self.fields})
                return _FakeRecord({p: self._data[p][chain] for p in self.fields})

            def _sample(self, key, shape):
                rng = np.random.default_rng(0)
                return _FakeRecord({p: rng.standard_normal(shape) for p in self.fields})

        return [_FakePosterior(seed=i) for i in range(n)]

    def _observed_seq(self, n: int, size: int = 30) -> list[np.ndarray]:
        return [np.random.default_rng(i).standard_normal(size) for i in range(n)]

    def test_writes_spc_to_final_dist(self):
        dists = self._make_posteriors(3)
        obs_seq = self._observed_seq(3)
        add_spc(
            dists,
            test_fns=_mean,
            observed_data_sequence=obs_seq,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=10,
        )
        final = dists[-1]
        spc_ds = final._auxiliary["diagnostics"]["runs"]["spc"].to_dataset()
        assert "p_value" in spc_ds.data_vars

    def test_spc_p_value_shape(self):
        n = 4
        dists = self._make_posteriors(n)
        obs_seq = self._observed_seq(n)
        add_spc(
            dists,
            test_fns=_mean,
            observed_data_sequence=obs_seq,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=10,
        )
        ds = dists[-1]._auxiliary["diagnostics"]["runs"]["spc"].to_dataset()
        # shape: (n_test_fns, n_time_steps)
        assert ds["p_value"].shape == (1, n)

    def test_each_dist_gets_ppc(self):
        dists = self._make_posteriors(3)
        obs_seq = self._observed_seq(3)
        add_spc(
            dists,
            test_fns=_mean,
            observed_data_sequence=obs_seq,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=10,
        )
        for dist in dists:
            assert dist._auxiliary is not None

    def test_returns_none(self):
        dists = self._make_posteriors(2)
        obs_seq = self._observed_seq(2)
        result = add_spc(
            dists,
            test_fns=_mean,
            observed_data_sequence=obs_seq,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=5,
        )
        assert result is None

    def test_raises_length_mismatch(self):
        dists = self._make_posteriors(3)
        obs_seq = self._observed_seq(2)
        with pytest.raises(ValueError, match="same length"):
            add_spc(
                dists,
                test_fns=_mean,
                observed_data_sequence=obs_seq,
                generative_likelihood=_NumpyLikelihood(),
                n_replications=5,
            )

    def test_raises_empty_distributions(self):
        with pytest.raises(ValueError):
            add_spc(
                [],
                test_fns=_mean,
                observed_data_sequence=[],
                generative_likelihood=_NumpyLikelihood(),
                n_replications=5,
            )

    def test_multiple_test_fns(self):
        dists = self._make_posteriors(2)
        obs_seq = self._observed_seq(2)
        add_spc(
            dists,
            test_fns=[_mean, _std],
            observed_data_sequence=obs_seq,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=10,
        )
        ds = dists[-1]._auxiliary["diagnostics"]["runs"]["spc"].to_dataset()
        assert ds["p_value"].shape[0] == 2  # 2 test functions
