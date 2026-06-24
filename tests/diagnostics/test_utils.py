"""Tests for probpipe.diagnostics._utils."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

import probpipe.diagnostics._utils as utils
from probpipe.diagnostics._utils import (
    _as_numpy,
    _component_name,
    _dataset_values,
    _json_dumps_safe,
    _record_get,
    _resolve_generative_likelihood,
    _safe_float,
)

# ---------------------------------------------------------------------------
# _record_get
# ---------------------------------------------------------------------------


class TestRecordGet:
    def test_plain_dict(self):
        assert _record_get({"a": 1}, "a") == 1

    def test_missing_key_returns_default(self):
        assert _record_get({"a": 1}, "b", default=99) == 99

    def test_none_obj_returns_default(self):
        assert _record_get(None, "x") is None

    def test_subscript_access(self):
        class _Sub:
            def __getitem__(self, k):
                return 42 if k == "z" else KeyError(k)

        assert _record_get(_Sub(), "z") == 42

    def test_attribute_fallback(self):
        class _Attr:
            foo = "bar"

        assert _record_get(_Attr(), "foo") == "bar"

    def test_get_method_fallback(self):
        class _GetObj:
            def get(self, k, d=None):
                return {"x": 7}.get(k, d)

        assert _record_get(_GetObj(), "x") == 7
        assert _record_get(_GetObj(), "missing", default=0) == 0

    def test_completely_unknown_returns_default(self):
        assert _record_get(object(), "nonexistent", default="sentinel") == "sentinel"

    def test_get_method_exception_returns_default(self):
        class _ExplodingGet:
            def __getitem__(self, key):
                raise KeyError(key)

            def __getattr__(self, key):
                raise AttributeError(key)

            def get(self, key, default=None):
                raise RuntimeError("boom")

        assert _record_get(_ExplodingGet(), "x", default="fallback") == "fallback"


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_plain_float(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_zero(self):
        assert _safe_float(0.0) == 0.0

    def test_none_is_nan(self):
        assert np.isnan(_safe_float(None))

    def test_numpy_scalar(self):
        assert _safe_float(np.float32(2.5)) == pytest.approx(2.5, abs=1e-4)

    def test_numpy_array_scalar(self):
        assert _safe_float(np.array(1.5)) == pytest.approx(1.5)

    def test_numpy_array_1d_takes_first(self):
        assert _safe_float(np.array([9.0, 1.0, 2.0])) == pytest.approx(9.0)

    def test_unconvertible_is_nan(self):
        assert np.isnan(_safe_float("not-a-number"))

    def test_negative(self):
        assert _safe_float(-7.0) == pytest.approx(-7.0)


# ---------------------------------------------------------------------------
# _as_numpy
# ---------------------------------------------------------------------------


class TestAsNumpy:
    def test_plain_array(self):
        arr = np.array([1.0, 2.0])
        result = _as_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_none_returns_none(self):
        assert _as_numpy(None) is None

    def test_list_converts(self):
        result = _as_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_samples_attribute(self):
        class _HasSamples:
            samples = np.array([4.0, 5.0])

        result = _as_numpy(_HasSamples())
        np.testing.assert_array_equal(result, np.array([4.0, 5.0]))

    def test_samples_attribute_failure_falls_back(self):
        class _BadSamples:
            def __array__(self, dtype=None):
                raise RuntimeError("no samples")

        class _SamplesFail:
            samples = _BadSamples()

            def __array__(self, dtype=None):
                return np.asarray([8.0], dtype=dtype)

        np.testing.assert_array_equal(_as_numpy(_SamplesFail()), [8.0])

    def test_unconvertible_returns_none(self):
        class _Bad:
            def __array__(self):
                raise ValueError("no")

        assert _as_numpy(_Bad()) is None


# ---------------------------------------------------------------------------
# _json_dumps_safe
# ---------------------------------------------------------------------------


class TestJsonDumpsSafe:
    def test_list(self):
        out = _json_dumps_safe([1, 2, 3])
        assert json.loads(out) == [1, 2, 3]

    def test_dict(self):
        out = _json_dumps_safe({"a": 1})
        assert json.loads(out) == {"a": 1}

    def test_string(self):
        out = _json_dumps_safe("hello")
        assert json.loads(out) == "hello"

    def test_non_serializable_falls_back_to_str(self):
        out = _json_dumps_safe(object())
        # Must be valid JSON even if it looks like "<object ...>"
        parsed = json.loads(out)
        assert isinstance(parsed, str)

    def test_empty_list(self):
        assert json.loads(_json_dumps_safe([])) == []

    def test_double_json_failure_returns_empty_object(self, monkeypatch):
        calls = []

        def _always_fails(obj):
            calls.append(obj)
            if len(calls) == 1:
                raise TypeError("not serializable")
            raise RuntimeError("still not serializable")

        monkeypatch.setattr(utils.json, "dumps", _always_fails)

        assert _json_dumps_safe(object()) == "{}"


# ---------------------------------------------------------------------------
# Diagnostic dataset flattening
# ---------------------------------------------------------------------------


class TestDiagnosticDatasetValues:
    def test_component_name_formats_scalar_and_indexed_names(self):
        assert _component_name("alpha", ()) == "alpha"
        assert _component_name("beta", (1, 2)) == "beta[1, 2]"

    def test_dataset_values_flattens_scalar_vector_and_matrix_values(self):
        ds = xr.Dataset(
            {
                "alpha": xr.DataArray(np.array(1.5)),
                "beta": xr.DataArray(np.array([2.0, 3.0]), dims=["dim_0"]),
                "omega": xr.DataArray(
                    np.array([[4.0, 5.0], [6.0, 7.0]]),
                    dims=["dim_0", "dim_1"],
                ),
            }
        )

        assert _dataset_values(ds) == {
            "alpha": 1.5,
            "beta[0]": 2.0,
            "beta[1]": 3.0,
            "omega[0, 0]": 4.0,
            "omega[0, 1]": 5.0,
            "omega[1, 0]": 6.0,
            "omega[1, 1]": 7.0,
        }


# ---------------------------------------------------------------------------
# _resolve_generative_likelihood
# ---------------------------------------------------------------------------


class _FakeLikelihood:
    def generate_data(self, params, n_samples, *, key=None):
        return np.zeros(n_samples)


class TestResolveGenerativeLikelihood:
    def test_explicit_arg_wins(self):
        gl = _FakeLikelihood()
        assert _resolve_generative_likelihood(None, gl) is gl

    def test_subscript_data_path(self):
        gl = _FakeLikelihood()

        class _Model:
            def __getitem__(self, k):
                return gl if k == "data" else KeyError(k)

        result = _resolve_generative_likelihood(_Model())
        assert result is gl

    def test_likelihood_attribute(self):
        gl = _FakeLikelihood()

        class _Posterior:
            _likelihood = gl

        result = _resolve_generative_likelihood(_Posterior())
        assert result is gl

    def test_generative_likelihood_attribute(self):
        gl = _FakeLikelihood()

        class _Posterior:
            generative_likelihood = gl

        result = _resolve_generative_likelihood(_Posterior())
        assert result is gl

    def test_raises_when_nothing_found(self):
        with pytest.raises(ValueError, match="generative likelihood"):
            _resolve_generative_likelihood(object())

    def test_subscript_without_generate_data_is_skipped(self):
        """distribution["data"] exists but has no generate_data — skip it."""

        class _BadModel:
            def __getitem__(self, k):
                return "not-a-likelihood"

        with pytest.raises(ValueError):
            _resolve_generative_likelihood(_BadModel())
