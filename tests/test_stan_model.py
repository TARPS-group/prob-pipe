"""Tests for StanModel.

Uses mocks to test all code paths without requiring a compiled Stan model.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from probpipe import SupportsLogProb, SupportsConditionableComponents
from probpipe.modeling._stan import StanModel, _UnconstrainedStanView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_bs_model(num_params=3, param_names=("alpha", "beta", "sigma")):
    """Create a mock BridgeStan model."""
    mock = MagicMock()
    mock.param_unc_num.return_value = num_params
    mock.param_names.return_value = list(param_names)
    mock.log_density.return_value = -5.0
    mock.param_constrain.return_value = np.array([1.0, 2.0, 3.0])
    mock.param_unconstrain.return_value = np.array([0.5, 1.0, 1.5])
    return mock


def _make_stan_model(num_params=3, param_names=("alpha", "beta", "sigma"), name="test"):
    """Create a StanModel with a mocked BridgeStan backend."""
    mock_bs = _make_mock_bs_model(num_params, param_names)
    model = object.__new__(StanModel)
    model._stan_file = "test.stan"
    model._stan_data = None
    model._name_str = name
    model._bs_model = mock_bs
    model._num_params = num_params
    return model


# ---------------------------------------------------------------------------
# StanModel protocol compliance
# ---------------------------------------------------------------------------


class TestStanModelProtocols:
    def test_supports_log_prob(self):
        assert issubclass(StanModel, SupportsLogProb)

    def test_supports_conditionable_components(self):
        model = _make_stan_model()
        assert isinstance(model, SupportsConditionableComponents)


# ---------------------------------------------------------------------------
# StanModel with mocked backend
# ---------------------------------------------------------------------------


class TestStanModelMocked:
    @pytest.fixture
    def model(self):
        return _make_stan_model()

    def test_name(self, model):
        assert model.name == "test"

    def test_event_shape(self, model):
        assert model.event_shape == (3,)

    def test_component_names(self, model):
        assert model.component_names == ("alpha", "beta", "sigma")

    def test_parameter_names(self, model):
        assert model.parameter_names == ("alpha", "beta", "sigma")

    def test_conditionable_components(self, model):
        cc = model.conditionable_components
        assert cc == {"data": True}

    def test_required_observations(self, model):
        assert model.required_observations == ("data",)

    def test_getitem_valid(self, model):
        assert model["alpha"] == "alpha"

    def test_getitem_invalid(self, model):
        with pytest.raises(KeyError, match="Unknown component"):
            model["nonexistent"]

    def test_repr(self, model):
        r = repr(model)
        assert "StanModel" in r
        assert "test.stan" in r
        assert "num_params=3" in r

    def test_log_prob(self, model):
        x = jnp.array([1.0, 2.0, 3.0])
        lp = model._log_prob(x)
        assert jnp.isfinite(lp)
        model._bs_model.param_unconstrain.assert_called()
        model._bs_model.log_density.assert_called()

    def test_unnormalized_log_prob(self, model):
        x = jnp.array([1.0, 2.0, 3.0])
        ulp = model._unnormalized_log_prob(x)
        assert jnp.isfinite(ulp)

    def test_unnormalized_prob(self, model):
        x = jnp.array([1.0, 2.0, 3.0])
        up = model._unnormalized_prob(x)
        assert jnp.isfinite(up)
        assert float(up) > 0

    def test_prob(self, model):
        x = jnp.array([1.0, 2.0, 3.0])
        p = model._prob(x)
        assert jnp.isfinite(p)
        assert float(p) > 0

    def test_param_constrain(self, model):
        unc = jnp.array([0.5, 1.0, 1.5])
        result = model.param_constrain(unc)
        model._bs_model.param_constrain.assert_called()
        assert result.shape == (3,)

    def test_param_unconstrain(self, model):
        x = jnp.array([1.0, 2.0, 3.0])
        result = model.param_unconstrain(x)
        model._bs_model.param_unconstrain.assert_called()
        assert result.shape == (3,)

    def test_as_unconstrained_distribution(self, model):
        view = model.as_unconstrained_distribution()
        assert isinstance(view, _UnconstrainedStanView)

    def test_bridgestan_model_no_data(self, model):
        result = model._bridgestan_model()
        assert result is model._bs_model

    def test_bridgestan_model_with_data(self, model):
        mock_bs = MagicMock()
        mock_bs.StanModel.from_stan_file.return_value = MagicMock()
        with patch.dict("sys.modules", {"bridgestan": mock_bs}):
            result = model._bridgestan_model(data={"N": 10})
            mock_bs.StanModel.from_stan_file.assert_called_once_with("test.stan", data={"N": 10})


# ---------------------------------------------------------------------------
# _UnconstrainedStanView
# ---------------------------------------------------------------------------


class TestUnconstrainedStanView:
    @pytest.fixture
    def model(self):
        return _make_stan_model(name="mymodel")

    @pytest.fixture
    def view(self, model):
        return model.as_unconstrained_distribution()

    def test_name_with_base(self, view):
        assert view.name == "mymodel_unconstrained"

    def test_name_without_base(self):
        model = _make_stan_model(name=None)
        view = model.as_unconstrained_distribution()
        assert view.name == "unconstrained"

    def test_event_shape(self, view):
        assert view.event_shape == (3,)

    def test_log_prob(self, view):
        x = jnp.array([0.5, 1.0, 1.5])
        lp = view._log_prob(x)
        assert jnp.isfinite(lp)

    def test_unnormalized_log_prob(self, view):
        x = jnp.array([0.5, 1.0, 1.5])
        ulp = view._unnormalized_log_prob(x)
        np.testing.assert_allclose(float(ulp), float(view._log_prob(x)))

    def test_unnormalized_prob(self, view):
        x = jnp.array([0.5, 1.0, 1.5])
        up = view._unnormalized_prob(x)
        assert jnp.isfinite(up)

    def test_prob(self, view):
        x = jnp.array([0.5, 1.0, 1.5])
        p = view._prob(x)
        np.testing.assert_allclose(float(p), float(jnp.exp(view._log_prob(x))))

    def test_repr(self, view):
        r = repr(view)
        assert "UnconstrainedStanView" in r
        assert "StanModel" in r


# ---------------------------------------------------------------------------
# StanModel._condition_on
# ---------------------------------------------------------------------------


class TestStanModelConditionOn:
    def test_condition_on_falls_back_to_rwmh(self):
        """When nutpie import fails, falls back to RWMH."""
        import sys
        model = _make_stan_model()
        # Remove cached module so the import inside _condition_on fails
        saved = sys.modules.pop("probpipe.inference._nutpie", None)
        try:
            with patch.dict("sys.modules", {"probpipe.inference._nutpie": None}):
                with patch("probpipe.inference._rwmh._rwmh_impl") as mock_rwmh:
                    mock_rwmh.return_value = MagicMock()
                    model._condition_on({"y": [1, 2, 3]}, num_results=10)
                    mock_rwmh.assert_called_once()
        finally:
            if saved is not None:
                sys.modules["probpipe.inference._nutpie"] = saved


# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------


class TestStanModelImportError:
    def test_missing_bridgestan(self):
        """StanModel raises ImportError with install instructions when bridgestan missing."""
        with patch.dict("sys.modules", {"bridgestan": None}):
            with pytest.raises(ImportError, match="pip install bridgestan"):
                StanModel("test.stan")
