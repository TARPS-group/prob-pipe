"""Tests for StanModel.

Uses mocks to test all code paths without requiring a compiled Stan model.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from probpipe import SupportsLogProb
from probpipe.modeling._stan import StanModel, _UnconstrainedStanView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_bs_model(num_params=3, param_names=("alpha", "beta", "sigma")):
    """Create a mock BridgeStan model.

    constrain(x) = x + 1, unconstrain(x) = x - 1: a genuine inverse pair
    so the round-trip test can verify value preservation.
    """
    mock = MagicMock()
    mock.param_unc_num.return_value = num_params
    mock.param_names.return_value = list(param_names)
    mock.log_density.return_value = -5.0
    mock.param_constrain.side_effect = lambda x: np.asarray(x) + 1.0
    mock.param_unconstrain.side_effect = lambda x: np.asarray(x) - 1.0
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

    def test_supports_named_components(self):
        model = _make_stan_model()
        assert hasattr(model, 'fields')


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

    def test_fields(self, model):
        assert model.fields == ("alpha", "beta", "sigma")

    def test_parameter_names(self, model):
        assert model.parameter_names == ("alpha", "beta", "sigma")

    def test_getitem_returns_name_placeholder(self, model):
        """StanModel['alpha'] returns the parameter name — BridgeStan doesn't
        expose sub-distributions, so __getitem__ is a placeholder that merely
        validates the key. See the comment in StanModel.__getitem__.
        """
        assert model["alpha"] == "alpha"
        assert model["beta"] == "beta"
        assert model["sigma"] == "sigma"

    def test_getitem_unknown_key_raises(self, model):
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

    def test_param_constrain_applies_transform(self, model):
        """param_constrain should apply the underlying bridgestan transform."""
        unc = jnp.array([0.5, 1.0, 1.5])
        result = model.param_constrain(unc)
        # Mock defines constrain(x) = x + 1.
        np.testing.assert_allclose(np.asarray(result), np.asarray(unc) + 1.0)

    def test_param_unconstrain_applies_transform(self, model):
        """param_unconstrain should apply the underlying bridgestan transform."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = model.param_unconstrain(x)
        # Mock defines unconstrain(x) = x - 1.
        np.testing.assert_allclose(np.asarray(result), np.asarray(x) - 1.0)

    def test_constrain_unconstrain_are_inverses(self, model):
        """constrain(unconstrain(x)) == x: the wrapper must preserve values."""
        x = jnp.array([1.0, 2.0, 3.0])
        roundtrip = model.param_constrain(model.param_unconstrain(x))
        np.testing.assert_allclose(np.asarray(roundtrip), np.asarray(x), atol=1e-12)

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
# StanModel conditioning via registry
# ---------------------------------------------------------------------------


class TestStanModelConditionOn:
    def test_condition_on_delegates_to_registry(self):
        """condition_on routes StanModel through the inference registry."""
        from probpipe import condition_on

        model = _make_stan_model()
        with patch("probpipe.inference._registry.inference_method_registry.execute") as mock_exec:
            mock_exec.return_value = MagicMock()
            condition_on(model, {"y": [1, 2, 3]}, num_results=10)
            mock_exec.assert_called_once()


# ---------------------------------------------------------------------------
# CmdStan inference method tests
# ---------------------------------------------------------------------------


class TestCmdStanInferenceMethod:
    """Test CmdStan inference via the registry method."""

    def test_ensure_cmdstanpy_missing(self):
        from probpipe.inference._cmdstan_method import _ensure_cmdstanpy

        with patch.dict("sys.modules", {"cmdstanpy": None}):
            with pytest.raises(ImportError, match="pip install probpipe"):
                _ensure_cmdstanpy()

    def test_ensure_cmdstanpy_present(self):
        from probpipe.inference._cmdstan_method import _ensure_cmdstanpy

        mock_cmdstanpy = MagicMock()
        with patch.dict("sys.modules", {"cmdstanpy": mock_cmdstanpy}):
            result = _ensure_cmdstanpy()
            assert result is mock_cmdstanpy


# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------


class TestStanModelImportError:
    def test_missing_bridgestan(self):
        """StanModel raises ImportError with install instructions when bridgestan missing."""
        with patch.dict("sys.modules", {"bridgestan": None}):
            with pytest.raises(ImportError, match="pip install bridgestan"):
                StanModel("test.stan")


# ---------------------------------------------------------------------------
# Real BridgeStan integration (requires a compiled Stan program)
# ---------------------------------------------------------------------------


_bs = pytest.importorskip("bridgestan")


@pytest.fixture(scope="module")
def tmp_stan_model(tmp_path_factory):
    """Compile a trivial Stan program once per module for integration tests.

    The model is Normal(mu, 1) with a unit-variance prior on mu.  This
    gives an analytically tractable log_density for verifying StanModel
    behaviour against first principles.
    """
    tmp_dir = tmp_path_factory.mktemp("stan_models")
    stan_file = tmp_dir / "normal_mean.stan"
    stan_file.write_text(
        """
        parameters { real mu; }
        model { mu ~ normal(0, 1); }
        """
    )
    try:
        bs_model = _bs.StanModel(str(stan_file))
    except Exception as exc:
        pytest.skip(f"Stan compilation unavailable: {exc}")
    model = object.__new__(StanModel)
    model._stan_file = str(stan_file)
    model._stan_data = None
    model._name_str = "normal_mean"
    model._bs_model = bs_model
    model._num_params = 1
    return model


class TestStanModelIntegration:
    """Minimal real-backend checks against a compiled Stan program."""

    def test_event_shape(self, tmp_stan_model):
        assert tmp_stan_model.event_shape == (1,)

    def test_log_prob_matches_analytical(self, tmp_stan_model):
        """Stan's log_density for mu ~ N(0, 1) equals -0.5 * mu^2 + const."""
        for mu in [-1.0, 0.0, 0.5, 2.0]:
            lp = float(tmp_stan_model._log_prob(jnp.asarray([mu])))
            # Stan drops the normalizing constant (log(2*pi)/2), so lp = -mu^2/2
            np.testing.assert_allclose(lp, -0.5 * mu * mu, atol=1e-5)

    def test_constrain_identity_for_real(self, tmp_stan_model):
        """For unconstrained real parameters, constrain is the identity."""
        x = np.array([1.23])
        constrained = tmp_stan_model._bs_model.param_constrain(x)
        np.testing.assert_allclose(constrained, x)
