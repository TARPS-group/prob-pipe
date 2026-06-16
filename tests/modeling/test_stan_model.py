"""Tests for StanModel.

Uses mocks to test all code paths without requiring a compiled Stan model.
"""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import SupportsLogProb
from probpipe.modeling._stan import StanModel, _UnconstrainedStanView, _param_blocks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_bs_model(
    num_params=3, param_names=("alpha", "beta", "sigma"), param_unc_names=None,
):
    """Create a mock BridgeStan model.

    constrain(x) = x + 1, unconstrain(x) = x - 1: a genuine inverse pair
    so the round-trip test can verify value preservation. ``param_unc_names``
    defaults to ``param_names`` (no constrained/unconstrained difference).
    """
    mock = MagicMock()
    mock.param_unc_num.return_value = num_params
    mock.param_names.return_value = list(param_names)
    mock.param_unc_names.return_value = list(param_unc_names or param_names)
    mock.log_density.return_value = -5.0
    mock.param_constrain.side_effect = lambda x: np.asarray(x) + 1.0
    mock.param_unconstrain.side_effect = lambda x: np.asarray(x) - 1.0
    return mock


def _make_stan_model(
    num_params=3, param_names=("alpha", "beta", "sigma"), name="test",
    param_unc_names=None,
):
    """Create a StanModel with a mocked BridgeStan backend."""
    mock_bs = _make_mock_bs_model(num_params, param_names, param_unc_names)
    model = object.__new__(StanModel)
    model._stan_file = "test.stan"
    model._stan_data = None
    model._name = name if name else "StanModel"
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
        with patch.dict("sys.modules", {"bridgestan": mock_bs}):
            result = model._bridgestan_model(data={"N": 10})
        # The dict is handed straight to BridgeStan's constructor, which
        # serializes it (via stanio) — we don't pre-encode it ourselves.
        mock_bs.StanModel.assert_called_once_with("test.stan", data={"N": 10})
        assert result is mock_bs.StanModel.return_value


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
        # StanModel without an explicit name now falls back to the
        # class name ("StanModel") to satisfy the Distribution
        # metaclass's non-empty-name requirement; the view name
        # composes accordingly.
        model = _make_stan_model(name=None)
        view = model.as_unconstrained_distribution()
        assert view.name == "StanModel_unconstrained"

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
# Tier 2: per-Stan-parameter blocks (fields, shapes, keyword _pack_value)
# ---------------------------------------------------------------------------


class TestParamBlocks:
    """_param_blocks groups BridgeStan's flat, 1-indexed names into shaped
    blocks. BridgeStan flattens matrices column-major (L.1.1, L.2.1, ...)."""

    def test_all_scalars(self):
        blocks = _param_blocks(["mu", "sigma"])
        assert [(b.name, b.shape) for b in blocks] == [("mu", ()), ("sigma", ())]

    def test_vector(self):
        blocks = _param_blocks(["theta.1", "theta.2", "theta.3"])
        assert [(b.name, b.shape) for b in blocks] == [("theta", (3,))]

    def test_matrix(self):
        blocks = _param_blocks(["L.1.1", "L.2.1", "L.1.2", "L.2.2"])
        assert [(b.name, b.shape) for b in blocks] == [("L", (2, 2))]

    def test_mixed_blocks(self):
        names = ["mu", "theta.1", "theta.2", "theta.3",
                 "L.1.1", "L.2.1", "L.1.2", "L.2.2"]
        assert [(b.name, b.shape) for b in _param_blocks(names)] == [
            ("mu", ()), ("theta", (3,)), ("L", (2, 2))]

    def test_empty(self):
        assert _param_blocks([]) == ()


class TestStanModelTier2:
    """StanModel and its view expose one field per Stan parameter *block*,
    while ``parameter_names`` keeps BridgeStan's flat per-scalar names."""

    @pytest.fixture
    def model(self):
        names = ["mu", "theta.1", "theta.2", "theta.3",
                 "L.1.1", "L.2.1", "L.1.2", "L.2.2"]
        return _make_stan_model(num_params=len(names), param_names=names)

    def test_fields_are_blocks(self, model):
        assert model.fields == ("mu", "theta", "L")

    def test_parameter_names_stay_per_scalar(self, model):
        assert model.parameter_names == (
            "mu", "theta.1", "theta.2", "theta.3",
            "L.1.1", "L.2.1", "L.1.2", "L.2.2")

    def test_record_template_shapes(self, model):
        assert {f: model.record_template[f] for f in model.fields} == {
            "mu": (), "theta": (3,), "L": (2, 2)}

    def test_getitem_uses_block_fields(self, model):
        assert model["theta"] == "theta"
        with pytest.raises(KeyError, match="Unknown component"):
            model["theta.1"]  # per-scalar names are no longer components

    def test_pack_value_assembles_column_major(self, model):
        flat = model._pack_value(
            mu=0.5, theta=jnp.array([1.0, 2.0, 3.0]),
            L=jnp.array([[10.0, 30.0], [20.0, 40.0]]))
        # L is packed column-major to match BridgeStan: [10, 20, 30, 40].
        assert jnp.allclose(
            flat, jnp.array([0.5, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0]))

    def test_pack_value_missing_block_raises(self, model):
        with pytest.raises(TypeError, match="missing"):
            model._pack_value(mu=0.5, theta=jnp.array([1.0, 2.0, 3.0]))

    def test_pack_value_unexpected_block_raises(self, model):
        with pytest.raises(TypeError, match="unexpected"):
            model._pack_value(mu=0.5, theta=jnp.array([1.0, 2.0, 3.0]),
                              L=jnp.zeros((2, 2)), zzz=1.0)

    def test_pack_value_wrong_shape_raises(self, model):
        with pytest.raises(TypeError, match=r"shape \(2, 2\)"):
            model._pack_value(mu=0.5, theta=jnp.array([1.0, 2.0, 3.0]),
                              L=jnp.array([1.0, 2.0, 3.0, 4.0]))  # flat, not (2,2)

    def test_view_uses_unconstrained_blocks(self):
        # simplex[3] p: constrained names p.1..p.3 (size 3); the unconstrained
        # parametrization drops one degree of freedom -> p.1, p.2 (size 2).
        model = _make_stan_model(
            num_params=3, param_names=["mu", "p.1", "p.2", "p.3"],
            param_unc_names=["mu", "p.1", "p.2"])
        assert model.fields == ("mu", "p")
        assert model.record_template["p"] == (3,)
        view = model.as_unconstrained_distribution()
        assert view.fields == ("mu", "p")
        assert view.record_template["p"] == (2,)
        flat = view._pack_value(mu=0.5, p=jnp.array([0.1, 0.2]))
        assert jnp.allclose(flat, jnp.array([0.5, 0.1, 0.2]))


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

    def test_import_cmdstanpy_missing(self):
        from probpipe.inference._cmdstan_method import _import_cmdstanpy

        with (
            patch.dict("sys.modules", {"cmdstanpy": None}),
            pytest.raises(ImportError, match="pip install probpipe"),
        ):
            _import_cmdstanpy()

    def test_import_cmdstanpy_present(self):
        from probpipe.inference._cmdstan_method import _import_cmdstanpy

        mock_cmdstanpy = MagicMock()
        with patch.dict("sys.modules", {"cmdstanpy": mock_cmdstanpy}):
            result = _import_cmdstanpy()
            assert result is mock_cmdstanpy


# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------


class TestStanModelImportError:
    def test_missing_bridgestan(self):
        """StanModel raises ImportError with install instructions when bridgestan missing."""
        with (
            patch.dict("sys.modules", {"bridgestan": None}),
            pytest.raises(ImportError, match="pip install bridgestan"),
        ):
            StanModel("test.stan")


# ---------------------------------------------------------------------------
# Real BridgeStan integration (requires a compiled Stan program)
#
# These exercise the path the mocks above cannot: the real BridgeStan
# constructor signature and its float64-ndarray boundary.  The gate lives in
# the fixtures (not at module scope) so the mocked tests above still run when
# BridgeStan is absent.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _stan_toolchain(tmp_path_factory):
    """Skip the integration tests unless BridgeStan can compile here.

    Compiling a trivial, data-free probe separates "is the C++ toolchain
    present?" (a legitimate skip) from "does StanModel construct correctly?"
    (a regression that must fail loudly, not skip) — so the model fixture
    below can construct without a try/except that would swallow real bugs.
    """
    bridgestan = pytest.importorskip("bridgestan")
    probe = tmp_path_factory.mktemp("stan_probe") / "probe.stan"
    probe.write_text("parameters { real x; } model { x ~ normal(0, 1); }")
    try:
        bridgestan.StanModel(str(probe))
    except Exception as exc:
        pytest.skip(f"Stan compilation unavailable: {exc}")


@pytest.fixture(scope="module")
def tmp_stan_model(_stan_toolchain, tmp_path_factory):
    """A real StanModel for ``y ~ Normal(mu, 1)`` with a unit prior on ``mu``,
    built through the public constructor.

    Construction therefore exercises the BridgeStan boundary end to end —
    ``.stan`` compilation plus serialization of the ``data`` dict.  The
    toolchain is known good from ``_stan_toolchain``, so any failure here is a
    real bug, not a missing compiler.  The model is conjugate, so its
    log-density is known in closed form.
    """
    stan_file = tmp_path_factory.mktemp("stan_models") / "normal_mean.stan"
    stan_file.write_text(
        """
        data {
          int<lower=0> N;
          vector[N] y;
        }
        parameters { real mu; }
        model {
          mu ~ normal(0, 1);
          y ~ normal(mu, 1);
        }
        """
    )
    return StanModel(str(stan_file), data={"N": 3, "y": [1.0, 2.0, 3.0]},
                     name="normal_mean")


class TestStanModelIntegration:
    """Real-backend checks against a compiled Stan program."""

    # Observations baked into tmp_stan_model's data.
    _Y = np.array([1.0, 2.0, 3.0])

    def _expected_log_density(self, mu):
        # Stan accumulates target = log N(mu; 0, 1) + sum_i log N(y_i; mu, 1)
        # and drops every normalizing constant (propto), so the -0.5*log(2*pi)
        # terms vanish and mu (an unconstrained real) carries no Jacobian.
        return -0.5 * mu**2 - 0.5 * float(np.sum((self._Y - mu) ** 2))

    def test_construct_exposes_parameters(self, tmp_stan_model):
        assert isinstance(tmp_stan_model, StanModel)
        assert tmp_stan_model.event_shape == (1,)
        assert tmp_stan_model.parameter_names == ("mu",)

    def test_log_prob_matches_analytical(self, tmp_stan_model):
        """``_log_prob`` matches the closed-form log-density.

        A JAX array is passed in, so this also covers the JAX -> float64
        ndarray conversion required at ``param_unconstrain`` / ``log_density``.
        """
        for mu in [-1.0, 0.0, 0.5, 2.0]:
            lp = float(tmp_stan_model._log_prob(jnp.asarray([mu])))
            np.testing.assert_allclose(lp, self._expected_log_density(mu), atol=1e-5)

    def test_param_transforms_round_trip(self, tmp_stan_model):
        """constrain/unconstrain are identity for an unconstrained real and
        survive the JAX-array boundary in both directions."""
        x = jnp.asarray([1.23])
        constrained = np.asarray(tmp_stan_model.param_constrain(x))
        np.testing.assert_allclose(constrained, [1.23], atol=1e-6)
        round_trip = tmp_stan_model.param_constrain(tmp_stan_model.param_unconstrain(x))
        np.testing.assert_allclose(np.asarray(round_trip), [1.23], atol=1e-6)
