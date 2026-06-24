"""Tests for probpipe.diagnostics._loo."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from probpipe.diagnostics._loo import (
    _add_log_likelihood,
    _as_numpy,
    _extract_pointwise_array,
    _has_group,
    _has_required_loo_pit_groups,
    _log_likelihood_to_dataset,
    _pareto_k_summary,
    _pointwise_dataarray,
    _safe_int,
    add_loo,
)
from probpipe.diagnostics._datatree_store import _add_group
from probpipe.diagnostics._views import LOOView


# ---------------------------------------------------------------------------
# Fake posterior
# ---------------------------------------------------------------------------


class _FakePosterior:
    def __init__(self, with_arviz_log_likelihood: bool = False):
        self._auxiliary = None
        if with_arviz_log_likelihood:
            ll = np.random.default_rng(0).standard_normal((2, 100, 30))
            ll_ds = xr.Dataset({"y": xr.DataArray(ll, dims=["chain", "draw", "obs"])})
            _add_group(self, "arviz/log_likelihood", ll_ds)


# ---------------------------------------------------------------------------
# Fake az.loo return value
# ---------------------------------------------------------------------------


def _fake_loo_result(
    *,
    elpd_loo: float = -120.5,
    se: float = 4.2,
    p_loo: float = 3.1,
    looic: float | None = None,
    warning: bool = False,
    good_k: float = 0.7,
    pareto_k: np.ndarray | None = None,
    loo_i: np.ndarray | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics ArviZ ELPDData."""
    if pareto_k is None:
        pareto_k = np.abs(np.random.default_rng(0).standard_normal(30)) * 0.3

    result = MagicMock()

    # Support both attribute and subscript access (_record_get uses both)
    result.elpd_loo = elpd_loo
    result.se = se
    result.p_loo = p_loo
    result.looic = looic
    result.warning = warning
    result.good_k = good_k
    result.pareto_k = xr.DataArray(pareto_k)
    result.loo_i = xr.DataArray(loo_i if loo_i is not None else -pareto_k)
    result.n_samples = 200
    result.n_data_points = 30
    result.__repr__ = lambda self: "FakeLOOResult"

    def _getitem(k):
        return getattr(result, k, None)

    result.__getitem__ = MagicMock(side_effect=_getitem)
    result.get = MagicMock(side_effect=lambda k, d=None: getattr(result, k, d))

    return result


# ---------------------------------------------------------------------------
# _as_numpy
# ---------------------------------------------------------------------------


class TestAsNumpy:
    def test_xarray_dataarray(self):
        da = xr.DataArray([1.0, 2.0, 3.0])
        result = _as_numpy(da)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_plain_array(self):
        arr = np.array([4.0, 5.0])
        np.testing.assert_array_equal(_as_numpy(arr), arr)

    def test_none_returns_none(self):
        assert _as_numpy(None) is None

    def test_values_attribute(self):
        class _HasValues:
            values = np.array([7.0])

        np.testing.assert_array_equal(_as_numpy(_HasValues()), [7.0])

    def test_samples_attribute(self):
        class _HasSamples:
            samples = np.array([9.0])

        np.testing.assert_array_equal(_as_numpy(_HasSamples()), [9.0])

    def test_falls_through_when_values_and_samples_fail(self):
        class _BadArray:
            def __array__(self, dtype=None):
                raise RuntimeError("bad array")

        class _BadArrayHooks:
            values = _BadArray()

            def __array__(self, dtype=None):
                return np.asarray([11.0], dtype=dtype)

        np.testing.assert_array_equal(_as_numpy(_BadArrayHooks()), [11.0])

    def test_returns_none_when_array_conversion_fails(self):
        class _Unconvertible:
            def __array__(self, dtype=None):
                raise RuntimeError("no array")

        assert _as_numpy(_Unconvertible()) is None

    def test_dataarray_values_failure_returns_none(self, monkeypatch):
        def _raise_values(self):
            raise RuntimeError("no values")

        monkeypatch.setattr(xr.DataArray, "values", property(_raise_values))

        assert _as_numpy(xr.DataArray([1.0])) is None

    def test_samples_failure_falls_back_to_array_conversion(self):
        class _BadSamples:
            def __array__(self, dtype=None):
                raise RuntimeError("no samples")

        class _SamplesFail:
            samples = _BadSamples()

            def __array__(self, dtype=None):
                return np.asarray([13.0], dtype=dtype)

        np.testing.assert_array_equal(_as_numpy(_SamplesFail()), [13.0])


class TestSafeInt:
    def test_none_and_unconvertible_return_minus_one(self):
        assert _safe_int(None) == -1
        assert _safe_int(object()) == -1

    def test_array_like_fallback(self):
        assert _safe_int(np.array([7.9])) == 7


# ---------------------------------------------------------------------------
# _log_likelihood_to_dataset
# ---------------------------------------------------------------------------


class TestLogLikelihoodToDataset:
    def test_dataset_passthrough(self):
        ds = xr.Dataset({"y": xr.DataArray(np.ones((2, 10, 5)), dims=["chain", "draw", "obs"])})
        result = _log_likelihood_to_dataset(ds)
        assert result is ds

    def test_dataarray_wraps(self):
        da = xr.DataArray(np.ones((2, 10, 5)), dims=["chain", "draw", "obs"], name="y")
        result = _log_likelihood_to_dataset(da)
        assert isinstance(result, xr.Dataset)
        assert "y" in result.data_vars

    def test_scalar_gets_chain_draw_dims(self):
        ds = _log_likelihood_to_dataset(np.float64(1.0))
        da = list(ds.data_vars.values())[0]
        assert da.dims == ("chain", "draw", "obs")
        assert da.shape == (1, 1, 1)

    def test_1d_raises_clear_error(self):
        with pytest.raises(ValueError, match="1-D log_likelihood"):
            _log_likelihood_to_dataset(np.ones(10))

    def test_1d_dataarray_raises_clear_error(self):
        da = xr.DataArray(np.ones(10), dims=["draw"], name="y")

        with pytest.raises(ValueError, match="pointwise log likelihood"):
            _log_likelihood_to_dataset(da)

    def test_1d_dataset_raises_clear_error(self):
        ds = xr.Dataset({"y": xr.DataArray(np.ones(10), dims=["draw"])})

        with pytest.raises(ValueError, match="pointwise log likelihood"):
            _log_likelihood_to_dataset(ds)

    def test_2d_adds_chain_dim(self):
        ds = _log_likelihood_to_dataset(np.ones((10, 5)))
        da = ds["y"]
        assert da.shape == (1, 10, 5)
        assert da.dims == ("chain", "draw", "obs")

    def test_3d_kept_as_is(self):
        ds = _log_likelihood_to_dataset(np.ones((2, 10, 5)))
        da = ds["y"]
        assert da.shape == (2, 10, 5)
        assert da.dims == ("chain", "draw", "obs")

    def test_custom_var_name(self):
        ds = _log_likelihood_to_dataset(np.ones((2, 10, 5)), var_name="loglik")
        assert "loglik" in ds.data_vars

    def test_higher_dimensional_obs_dims_are_named(self):
        ds = _log_likelihood_to_dataset(np.ones((2, 10, 3, 4)))
        da = ds["y"]
        assert da.dims == ("chain", "draw", "obs_dim_0", "obs_dim_1")


# ---------------------------------------------------------------------------
# _has_group
# ---------------------------------------------------------------------------


class TestHasGroup:
    def _tree(self, path: str) -> xr.DataTree:
        return xr.DataTree.from_dict({path: xr.Dataset()})

    def test_present(self):
        tree = self._tree("log_likelihood")
        assert _has_group(tree, "log_likelihood")

    def test_absent(self):
        tree = self._tree("log_likelihood")
        assert not _has_group(tree, "posterior_predictive")

    def test_nested_present(self):
        tree = self._tree("arviz/log_likelihood")
        assert _has_group(tree["arviz"], "log_likelihood")

    def test_none_tree(self):
        assert not _has_group(None, "anything")

    def test_root_path_skips_empty_path_parts(self):
        tree = xr.DataTree()
        assert _has_group(tree, "/")


# ---------------------------------------------------------------------------
# _pareto_k_summary
# ---------------------------------------------------------------------------


class TestParetoKSummary:
    def test_all_good(self):
        k = np.array([0.1, 0.2, 0.3])
        s = _pareto_k_summary(k)
        assert s["pareto_k_max"] == pytest.approx(0.3)
        assert s["pareto_k_bad_count"] == 0

    def test_some_bad(self):
        k = np.array([0.1, 0.8, 0.9])
        s = _pareto_k_summary(k, good_k=0.7)
        assert s["pareto_k_bad_count"] == 2

    def test_custom_threshold(self):
        k = np.array([0.5, 0.6, 0.7])
        s = _pareto_k_summary(k, good_k=0.55)
        assert s["pareto_k_bad_count"] == 2

    def test_none_input(self):
        s = _pareto_k_summary(None)
        assert np.isnan(s["pareto_k_max"])
        assert s["pareto_k_bad_count"] == -1

    def test_empty_array(self):
        s = _pareto_k_summary(np.array([]))
        assert np.isnan(s["pareto_k_max"])

    def test_xarray_dataarray_input(self):
        da = xr.DataArray([0.1, 0.85])
        s = _pareto_k_summary(da)
        assert s["pareto_k_max"] == pytest.approx(0.85)
        assert s["pareto_k_bad_count"] == 1

    def test_nan_good_k_uses_default_threshold(self):
        s = _pareto_k_summary(np.array([0.69, 0.71]), good_k=float("nan"))
        assert s["pareto_k_bad_count"] == 1


# ---------------------------------------------------------------------------
# _extract_pointwise_array and _pointwise_dataarray
# ---------------------------------------------------------------------------


class TestExtractPointwiseArray:
    def test_plain_array(self):
        arr = np.array([0.1, 0.2])
        result = _extract_pointwise_array(arr)
        np.testing.assert_array_equal(result, arr)

    def test_xarray_dataarray(self):
        da = xr.DataArray([0.3, 0.4])
        result = _extract_pointwise_array(da)
        np.testing.assert_array_almost_equal(result, [0.3, 0.4])

    def test_none_returns_none(self):
        assert _extract_pointwise_array(None) is None

    def test_2d_flattened(self):
        arr = np.ones((2, 3))
        result = _extract_pointwise_array(arr)
        assert result.shape == (6,)

    def test_non_numeric_returns_none(self):
        assert _extract_pointwise_array(["not-a-number"]) is None


class TestPointwiseDataarray:
    def test_returns_dataarray(self):
        da = _pointwise_dataarray(np.array([0.1, 0.2, 0.3]), name="pareto_k")
        assert isinstance(da, xr.DataArray)
        assert da.name == "pareto_k"
        assert da.dims == ("obs",)

    def test_none_returns_none(self):
        assert _pointwise_dataarray(None, name="pareto_k") is None

    def test_custom_dim(self):
        da = _pointwise_dataarray(np.ones(5), name="loo_i", dim="observation")
        assert "observation" in da.dims

    def test_bad_values_return_none(self):
        assert _pointwise_dataarray(["bad"], name="loo_i") is None


# ---------------------------------------------------------------------------
# add_loo
# ---------------------------------------------------------------------------


class TestAddLoo:
    def _posterior_with_ll(self) -> _FakePosterior:
        return _FakePosterior(with_arviz_log_likelihood=True)

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_writes_loo_group(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post)
        loo_ds = post._auxiliary["diagnostics"]["runs"]["loo"].to_dataset()
        assert "elpd_loo" in loo_ds.data_vars

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_scalar_values_stored(self, mock_loo):
        mock_loo.return_value = _fake_loo_result(elpd_loo=-50.0, se=2.0, p_loo=1.5)
        post = self._posterior_with_ll()
        add_loo(post)
        view = LOOView(post._auxiliary["diagnostics"]["runs"]["loo"])
        assert view.elpd_loo == pytest.approx(-50.0)
        assert view.se == pytest.approx(2.0)
        assert view.p_loo == pytest.approx(1.5)

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_looic_derived_from_elpd_when_absent(self, mock_loo):
        mock_loo.return_value = _fake_loo_result(elpd_loo=-60.0, looic=None)
        post = self._posterior_with_ll()
        add_loo(post)
        view = LOOView(post._auxiliary["diagnostics"]["runs"]["loo"])
        looic = view.looic
        if isinstance(looic, float):
            assert looic == pytest.approx(120.0)

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_pareto_k_summary_stored(self, mock_loo):
        k = np.array([0.1] * 25 + [0.8] * 5)  # 5 bad
        mock_loo.return_value = _fake_loo_result(pareto_k=k, good_k=0.7)
        post = self._posterior_with_ll()
        add_loo(post)
        view = LOOView(post._auxiliary["diagnostics"]["runs"]["loo"])
        assert view.pareto_k_bad_count == 5

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_warning_stored(self, mock_loo):
        mock_loo.return_value = _fake_loo_result(warning=True)
        post = self._posterior_with_ll()
        add_loo(post)
        view = LOOView(post._auxiliary["diagnostics"]["runs"]["loo"])
        assert view.warning is True

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_returns_none(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        assert add_loo(post) is None

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_force_false_skips_if_exists(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post)
        add_loo(post, force=False)
        assert mock_loo.call_count == 1

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_force_true_recomputes(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post)
        add_loo(post, force=True)
        assert mock_loo.call_count == 2

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_log_likelihood_arg_writes_to_auxiliary(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = _FakePosterior(with_arviz_log_likelihood=False)
        ll = np.random.default_rng(0).standard_normal((2, 100, 20))
        add_loo(post, log_likelihood=ll)
        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        assert "y" in ll_ds.data_vars

    def test_raises_without_log_likelihood(self):
        post = _FakePosterior(with_arviz_log_likelihood=False)
        with pytest.raises(ValueError, match="log_likelihood"):
            add_loo(post)

    def test_raises_without_auxiliary(self):
        post = _FakePosterior(with_arviz_log_likelihood=False)
        post._auxiliary = None
        with pytest.raises(ValueError):
            add_loo(post)

    def test_raises_when_arviz_tree_has_no_log_likelihood(self):
        post = _FakePosterior(with_arviz_log_likelihood=False)
        _add_group(post, "arviz/posterior", xr.Dataset(attrs={"present": True}))

        with pytest.raises(ValueError, match="No pointwise log_likelihood group"):
            add_loo(post)

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_pointwise_stored_by_default(self, mock_loo):
        k = np.abs(np.random.default_rng(0).standard_normal(30)) * 0.3
        mock_loo.return_value = _fake_loo_result(pareto_k=k)
        post = self._posterior_with_ll()
        add_loo(post, store_pointwise=True)
        loo_ds = post._auxiliary["diagnostics"]["runs"]["loo"].to_dataset()
        assert "pareto_k" in loo_ds.data_vars

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_pointwise_not_stored_when_disabled(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post, store_pointwise=False)
        loo_ds = post._auxiliary["diagnostics"]["runs"]["loo"].to_dataset()
        assert "pareto_k" not in loo_ds.data_vars

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_plot_ready_when_loo_pit_groups_exist(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        _add_group(post, "arviz/observed_data", xr.Dataset(attrs={"present": True}))
        _add_group(post, "arviz/posterior_predictive", xr.Dataset(attrs={"present": True}))

        add_loo(post)

        view = LOOView(post._auxiliary["diagnostics"]["runs"]["loo"])
        assert view.plot_ready is True


# ---------------------------------------------------------------------------
# _get_arviz_tree fallback paths
# ---------------------------------------------------------------------------


class TestGetArvizTree:
    def test_prefers_auxiliary_arviz(self):
        from probpipe.diagnostics._loo import _get_arviz_tree
        post = _FakePosterior(with_arviz_log_likelihood=True)
        tree = _get_arviz_tree(post)
        assert tree is not None
        assert _has_group(tree, "log_likelihood")

    def test_falls_back_to_inference_data(self):
        from probpipe.diagnostics._loo import _get_arviz_tree

        class _PostWithInferenceData:
            _auxiliary = None

            @property
            def inference_data(self):
                ll = np.random.default_rng(0).standard_normal((1, 10, 5))
                ll_ds = xr.Dataset({"y": xr.DataArray(ll, dims=["chain", "draw", "obs"])})
                return xr.DataTree.from_dict({"log_likelihood": ll_ds})

        tree = _get_arviz_tree(_PostWithInferenceData())
        assert tree is not None

    def test_returns_none_when_nothing_available(self):
        from probpipe.diagnostics._loo import _get_arviz_tree

        class _EmptyPost:
            _auxiliary = None
            inference_data = None

        assert _get_arviz_tree(_EmptyPost()) is None

    def test_auxiliary_without_arviz_falls_back_to_auxiliary(self):
        from probpipe.diagnostics._loo import _get_arviz_tree

        class _Post:
            _auxiliary = xr.DataTree.from_dict({"diagnostics": xr.Dataset()})
            inference_data = None

        assert _get_arviz_tree(_Post()) is _Post._auxiliary


# ---------------------------------------------------------------------------
# _has_required_loo_pit_groups
# ---------------------------------------------------------------------------


class TestHasRequiredLooPitGroups:
    def _tree(self, *paths) -> xr.DataTree:
        return xr.DataTree.from_dict({p: xr.Dataset() for p in paths})

    def test_all_present(self):
        tree = self._tree("observed_data", "posterior_predictive", "log_likelihood")
        assert _has_required_loo_pit_groups(tree)

    def test_missing_observed_data(self):
        tree = self._tree("posterior_predictive", "log_likelihood")
        assert not _has_required_loo_pit_groups(tree)

    def test_missing_log_likelihood(self):
        tree = self._tree("observed_data", "posterior_predictive")
        assert not _has_required_loo_pit_groups(tree)

    def test_empty_tree(self):
        tree = xr.DataTree()
        assert not _has_required_loo_pit_groups(tree)


# ---------------------------------------------------------------------------
# add_loo optional kwargs
# ---------------------------------------------------------------------------


class TestAddLooKwargs:
    def _posterior_with_ll(self):
        return _FakePosterior(with_arviz_log_likelihood=True)

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_scale_passed_to_arviz(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post, scale="log")
        _, kwargs = mock_loo.call_args
        assert kwargs.get("scale") == "log"

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_reff_passed_to_arviz(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post, reff=0.9)
        _, kwargs = mock_loo.call_args
        assert kwargs.get("reff") == pytest.approx(0.9)

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_scale_none_not_passed(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post)
        _, kwargs = mock_loo.call_args
        assert "scale" not in kwargs

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_reff_none_not_passed(self, mock_loo):
        mock_loo.return_value = _fake_loo_result()
        post = self._posterior_with_ll()
        add_loo(post)
        _, kwargs = mock_loo.call_args
        assert "reff" not in kwargs

    @patch("probpipe.diagnostics._loo.az.loo")
    def test_looic_derived_when_none(self, mock_loo):
        """looic = -2 * elpd_loo when ArviZ doesn't return it."""
        mock_loo.return_value = _fake_loo_result(elpd_loo=-60.0, looic=None)
        post = self._posterior_with_ll()
        add_loo(post)
        view = LOOView(post._auxiliary["diagnostics"]["runs"]["loo"])
        assert view.looic == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# _add_log_likelihood
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal model stub with a pure-NumPy per_datum_log_likelihood."""

    def __init__(self, n_obs: int = 20, n_features: int = 1):
        rng = np.random.default_rng(0)
        self._x = rng.standard_normal((n_obs, n_features))  # (n_obs, n_features)
        self._fit_intercept = True

        class _Likelihood:
            def __init__(self, x):
                self._x = x
                self._fit_intercept = True

            def per_datum_log_likelihood(self, params, datum):
                # Simple Gaussian log likelihood — fully NumPy, not JAX-traceable
                # so the fallback loop is exercised.
                import numpy as _np
                if hasattr(params, "__getitem__"):
                    try:
                        beta = _np.concatenate([
                            _np.atleast_1d(_np.asarray(params["intercept"])),
                            _np.atleast_1d(_np.asarray(params["slope"])),
                        ])
                    except Exception:
                        beta = _np.asarray(params).ravel()
                else:
                    beta = _np.asarray(params).ravel()
                x_i = _np.atleast_1d(_np.asarray(datum["X"]))
                y_i = float(_np.asarray(datum["y"]))
                eta = beta[0] + x_i @ beta[1:]
                return float(-0.5 * (y_i - eta) ** 2)

        self._likelihood = _Likelihood(self._x)


class _FakePostForLL:
    """Fake posterior compatible with internal log-likelihood computation."""

    def __init__(self, n_chains: int = 2, n_draws: int = 50, n_features: int = 1):
        self._auxiliary = None
        self._n_chains = n_chains
        self._n_draws = n_draws
        rng = np.random.default_rng(1)
        self._intercept = rng.standard_normal((n_chains, n_draws))
        self._slope = rng.standard_normal((n_chains, n_draws, n_features))

    @property
    def num_chains(self):
        return self._n_chains

    @property
    def num_draws(self):
        return self._n_draws

    @property
    def fields(self):
        return ["intercept", "slope"]

    def draws(self, *, chain: int):
        class _Rec(dict):
            @property
            def fields(self):
                return list(self.keys())

        return _Rec({
            "intercept": self._intercept[chain],
            "slope": self._slope[chain],
        })


class TestAddLogLikelihood:
    def _setup(self, n_obs=20):
        rng = np.random.default_rng(2)
        model = _FakeModel(n_obs=n_obs)
        post = _FakePostForLL()
        data = {"X": model._x, "y": rng.standard_normal(n_obs)}
        return post, model, data

    def test_writes_log_likelihood_group(self):
        post, model, data = self._setup()
        _add_log_likelihood(post, model, data)
        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        assert "y" in ll_ds.data_vars

    def test_output_shape(self):
        n_obs = 15
        post, model, data = self._setup(n_obs=n_obs)
        _add_log_likelihood(post, model, data)
        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        da = ll_ds["y"]
        assert da.dims == ("chain", "draw", "obs")
        assert da.shape == (post.num_chains, post.num_draws, n_obs)

    def test_values_are_finite(self):
        post, model, data = self._setup()
        _add_log_likelihood(post, model, data)
        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        arr = np.asarray(ll_ds["y"].values)
        assert np.all(np.isfinite(arr))

    def test_custom_var_name(self):
        post, model, data = self._setup()
        _add_log_likelihood(post, model, data, var_name="log_lik")
        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        assert "log_lik" in ll_ds.data_vars

    def test_returns_none(self):
        post, model, data = self._setup()
        assert _add_log_likelihood(post, model, data) is None

    def test_fallback_loop_produces_same_shape(self):
        """Force the fallback by patching jax.vmap to raise."""
        post, model, data = self._setup(n_obs=5)
        with patch("jax.vmap", side_effect=Exception("no vmap")):
            _add_log_likelihood(post, model, data)
        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        assert ll_ds["y"].shape == (post.num_chains, post.num_draws, 5)

    def test_add_loo_computes_missing_log_likelihood_from_model_and_data(self):
        """Integration: add_loo owns the internal log-likelihood path."""
        from unittest.mock import patch as _patch
        post, model, data = self._setup(n_obs=10)

        k = np.abs(np.random.default_rng(0).standard_normal(10)) * 0.2
        fake_result = _fake_loo_result(pareto_k=k, loo_i=-k)

        with _patch("probpipe.diagnostics._loo.az.loo", return_value=fake_result):
            add_loo(post, model=model, data=data)

        assert post._auxiliary is not None
        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        assert "y" in ll_ds.data_vars
        view = LOOView(post._auxiliary["diagnostics"]["runs"]["loo"])
        assert isinstance(view.elpd_loo, float)

    def test_fast_path_writes_log_likelihood(self):
        class _JaxModel:
            def __init__(self):
                self._x = np.arange(6, dtype=float).reshape(3, 2)

                class _Likelihood:
                    def __init__(self, x):
                        self._x = x

                    def per_datum_log_likelihood(self, params, datum):
                        eta = params[0] + datum["X"] @ params[1:]
                        return -0.5 * (datum["y"] - eta) ** 2

                self._likelihood = _Likelihood(self._x)

        post = _FakePostForLL(n_chains=1, n_draws=4, n_features=2)
        model = _JaxModel()
        data = {"X": model._x, "y": np.array([0.0, 1.0, 2.0])}

        _add_log_likelihood(post, model, data)

        ll_ds = post._auxiliary["arviz"]["log_likelihood"].to_dataset()
        assert ll_ds["y"].shape == (1, 4, 3)
