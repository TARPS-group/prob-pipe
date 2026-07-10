"""Tests for StanModel.

The behaviour tests compile real Stan programs through BridgeStan. The
``_stan_toolchain`` fixture gates them with ``importorskip`` plus a probe
compile, so they run in the dedicated ``stan`` CI job (which installs the
``stan`` extra) and skip cleanly elsewhere. The pure tests — the
parameter-name parser, the bridgestan-missing import guard, and the CmdStan
import shim — need no backend and run in the main matrix.

Models are compiled once per module (BridgeStan caches the shared object, so
re-instantiating the same ``.stan`` is cheap).
"""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import SupportsLogProb, log_prob
from probpipe.core.event_template import ArraySpec
from probpipe.modeling._stan import StanModel, _param_blocks, _UnconstrainedStanView

# ---------------------------------------------------------------------------
# Pure tests — no BridgeStan backend required (run in the main matrix)
# ---------------------------------------------------------------------------


class TestStanModelProtocols:
    def test_supports_log_prob(self):
        assert issubclass(StanModel, SupportsLogProb)


class TestParamBlocks:
    """``_param_blocks`` groups BridgeStan's flat, 1-indexed names into shaped
    blocks. BridgeStan flattens matrices column-major (L.1.1, L.2.1, ...). The
    name lists here are hand-written; ``test_param_blocks_match_real_names``
    cross-checks the same parser against a compiled model's real names.
    """

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
        names = ["mu", "theta.1", "theta.2", "theta.3", "L.1.1", "L.2.1", "L.1.2", "L.2.2"]
        assert [(b.name, b.shape) for b in _param_blocks(names)] == [
            ("mu", ()),
            ("theta", (3,)),
            ("L", (2, 2)),
        ]

    def test_empty(self):
        assert _param_blocks([]) == ()


class TestCmdStanInferenceMethod:
    """The CmdStan inference method's import shim (cmdstanpy, not BridgeStan)."""

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


class TestStanModelImportError:
    def test_missing_bridgestan(self):
        """StanModel raises ImportError with install instructions when bridgestan
        is missing — this *must* simulate bridgestan's absence, so it patches
        ``sys.modules`` rather than using a real backend."""
        with (
            patch.dict("sys.modules", {"bridgestan": None}),
            pytest.raises(ImportError, match="pip install bridgestan"),
        ):
            StanModel("test.stan")


# ---------------------------------------------------------------------------
# Real BridgeStan backend
#
# The model fixtures below depend on the shared ``_stan_toolchain`` fixture
# (tests/conftest.py), which ``importorskip``s bridgestan and probe-compiles the
# C++ toolchain — so these tests skip together when the backend is absent, while
# the pure tests above still run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conjugate_stan_file(_stan_toolchain, tmp_path_factory):
    """Path to a compiled conjugate model ``y ~ Normal(mu, 1)`` with a unit
    prior on ``mu`` — a single unconstrained real, so its log-density is
    closed-form and constrain/unconstrain are the identity. Returned as a path
    so tests can also build name-less / no-data instances cheaply (the compiled
    shared object is reused)."""
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
    return str(stan_file)


@pytest.fixture(scope="module")
def conjugate_model(conjugate_stan_file):
    """The conjugate model instantiated with data and an explicit name."""
    return StanModel(conjugate_stan_file, data={"N": 3, "y": [1.0, 2.0, 3.0]}, name="normal_mean")


@pytest.fixture(scope="module")
def structured_model(_stan_toolchain, tmp_path_factory):
    """A model exercising every parameter kind: a scalar, a vector, a matrix
    (column-major flattening), and a simplex — whose unconstrained
    parametrisation has one fewer dimension, so the view's blocks differ from
    the constrained model's."""
    stan_file = tmp_path_factory.mktemp("stan_models") / "structured.stan"
    stan_file.write_text(
        """
        parameters {
          real mu;
          vector[3] theta;
          matrix[2, 2] L;
          simplex[3] p;
        }
        model {
          mu ~ normal(0, 1);
          theta ~ normal(0, 1);
          to_vector(L) ~ normal(0, 1);
          p ~ dirichlet(rep_vector(1.0, 3));
        }
        """
    )
    return StanModel(str(stan_file), name="structured")


class TestStanModelSurface:
    """Distribution / ProbabilisticModel surface on a real conjugate model."""

    def test_supports_named_components(self, conjugate_model):
        assert hasattr(conjugate_model, "fields")

    def test_name(self, conjugate_model):
        assert conjugate_model.name == "normal_mean"

    def test_event_shape(self, conjugate_model):
        assert conjugate_model.event_shape == (1,)

    def test_fields(self, conjugate_model):
        assert conjugate_model.fields == ("mu",)

    def test_parameter_names(self, conjugate_model):
        assert conjugate_model.parameter_names == ("mu",)

    def test_getitem_returns_name_placeholder(self, conjugate_model):
        # BridgeStan doesn't expose sub-distributions, so __getitem__ is a
        # placeholder that merely validates the key.
        assert conjugate_model["mu"] == "mu"

    def test_getitem_unknown_key_raises(self, conjugate_model):
        with pytest.raises(KeyError, match="Unknown component"):
            conjugate_model["nonexistent"]

    def test_repr(self, conjugate_model):
        r = repr(conjugate_model)
        assert "StanModel" in r
        assert "normal_mean.stan" in r
        assert "num_params=1" in r

    def test_as_unconstrained_distribution(self, conjugate_model):
        assert isinstance(conjugate_model.as_unconstrained_distribution(), _UnconstrainedStanView)

    def test_name_defaults_to_class_name(self, conjugate_stan_file):
        # Without an explicit name, StanModel falls back to the class name to
        # satisfy the Tracked metaclass's non-empty-name requirement.
        model = StanModel(conjugate_stan_file, data={"N": 3, "y": [1.0, 2.0, 3.0]})
        assert model.name == "StanModel"


class TestStanModelDensity:
    """Density ops against the conjugate model's closed-form log-density.

    Tolerances cover float64 round-off on a deterministic closed form (Stan's
    log_density carries no RNG), so they are tight fixed constants, not
    seed-spread bounds.
    """

    _Y = np.array([1.0, 2.0, 3.0])  # baked into conjugate_model's data

    def _expected_log_density(self, mu):
        # Stan accumulates target = log N(mu; 0, 1) + sum_i log N(y_i; mu, 1)
        # and drops every normalizing constant (propto), so the -0.5*log(2*pi)
        # terms vanish and mu (an unconstrained real) carries no Jacobian.
        return -0.5 * mu**2 - 0.5 * float(np.sum((self._Y - mu) ** 2))

    def test_log_prob_matches_analytical(self, conjugate_model):
        """A JAX array is passed in, so this also covers the JAX -> float64
        ndarray conversion required at ``param_unconstrain`` / ``log_density``."""
        for mu in [-1.0, 0.0, 0.5, 2.0]:
            lp = float(conjugate_model._log_prob(jnp.asarray([mu])))
            np.testing.assert_allclose(lp, self._expected_log_density(mu), atol=1e-5)

    def test_unnormalized_log_prob_equals_log_prob(self, conjugate_model):
        # Stan's log_density is already the (unnormalized) target, so the two agree.
        x = jnp.asarray([0.5])
        np.testing.assert_allclose(
            float(conjugate_model._unnormalized_log_prob(x)),
            float(conjugate_model._log_prob(x)),
            atol=1e-6,
        )

    def test_prob_is_exp_log_prob(self, conjugate_model):
        x = jnp.asarray([0.5])
        np.testing.assert_allclose(
            float(conjugate_model._prob(x)), float(jnp.exp(conjugate_model._log_prob(x))), rtol=1e-6
        )

    def test_unnormalized_prob_positive(self, conjugate_model):
        up = conjugate_model._unnormalized_prob(jnp.asarray([0.5]))
        assert jnp.isfinite(up) and float(up) > 0

    def test_log_prob_accepts_float32_input(self, conjugate_model):
        # `_to_f64` exists so a float32 array (JAX's default dtype) can cross the
        # BridgeStan boundary, which rejects anything but float64. Passing one
        # explicitly exercises that coercion and checks the value survives it.
        lp = float(conjugate_model._log_prob(jnp.asarray([0.5], dtype=jnp.float32)))
        np.testing.assert_allclose(lp, self._expected_log_density(0.5), atol=1e-5)


class TestStanModelTransforms:
    """constrain / unconstrain across the JAX <-> float64 boundary. The
    conjugate model's single real is the identity in both directions; the
    structured model's simplex exercises a genuine (non-identity) transform."""

    def test_param_constrain_identity_for_real(self, conjugate_model):
        x = jnp.asarray([1.23])
        np.testing.assert_allclose(
            np.asarray(conjugate_model.param_constrain(x)), [1.23], atol=1e-6
        )

    def test_param_unconstrain_identity_for_real(self, conjugate_model):
        x = jnp.asarray([1.23])
        np.testing.assert_allclose(
            np.asarray(conjugate_model.param_unconstrain(x)), [1.23], atol=1e-6
        )

    def test_constrain_unconstrain_round_trip(self, conjugate_model):
        x = jnp.asarray([1.23])
        round_trip = conjugate_model.param_constrain(conjugate_model.param_unconstrain(x))
        np.testing.assert_allclose(np.asarray(round_trip), [1.23], atol=1e-6)

    def test_constrain_yields_valid_simplex(self, structured_model):
        # param_constrain applies the real (non-identity) transform: the simplex
        # block of the constrained vector must sum to 1 with positive entries —
        # a known invariant the identity-only conjugate model can't exercise.
        constrained = np.asarray(structured_model.param_constrain(jnp.zeros(10)))
        p = constrained[-3:]  # blocks pack in order; the simplex is last
        np.testing.assert_allclose(float(p.sum()), 1.0, atol=1e-5)
        assert (p > 0).all()

    def test_constrain_unconstrain_round_trip_simplex(self, structured_model):
        # constrain o unconstrain is the identity on a valid constrained point.
        # The simplex must sum to 1 exactly in float (0.25 + 0.25 + 0.5) for
        # BridgeStan's strict simplex_free to accept it.
        constrained = jnp.array([0.5, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 4.0, 0.25, 0.25, 0.5])
        round_trip = structured_model.param_constrain(
            structured_model.param_unconstrain(constrained)
        )
        np.testing.assert_allclose(np.asarray(round_trip), np.asarray(constrained), atol=1e-5)


class TestBridgestanModel:
    """``_bridgestan_model`` returns the compiled model, or builds a fresh one
    bound to new data."""

    def test_no_data_returns_compiled_model(self, conjugate_model):
        assert conjugate_model._bridgestan_model() is conjugate_model._bs_model

    def test_with_data_builds_new_model(self, conjugate_model):
        rebuilt = conjugate_model._bridgestan_model(data={"N": 2, "y": [0.0, 1.0]})
        assert rebuilt is not conjugate_model._bs_model
        assert rebuilt.param_unc_num() == 1  # still one parameter, `mu`


class TestStanModelParameters:
    """One field per Stan parameter *block*, while ``parameter_names`` keeps
    BridgeStan's flat per-scalar names. Exercised on a model with a scalar, a
    vector, a matrix, and a simplex."""

    def test_fields_are_blocks(self, structured_model):
        assert structured_model.fields == ("mu", "theta", "L", "p")

    def test_event_shape_is_unconstrained_size(self, structured_model):
        # mu(1) + theta(3) + L(4) + p simplex unconstrained(2) = 10.
        assert structured_model.event_shape == (10,)

    def test_parameter_names_stay_per_scalar(self, structured_model):
        assert structured_model.parameter_names == (
            "mu",
            "theta.1",
            "theta.2",
            "theta.3",
            "L.1.1",
            "L.2.1",
            "L.1.2",
            "L.2.2",
            "p.1",
            "p.2",
            "p.3",
        )

    def test_event_template_shapes(self, structured_model):
        # leaf_shapes is keyed by leaf path; this template is flat, so the
        # paths are the field names.
        assert structured_model.event_template.leaf_shapes == {
            "mu": (),
            "theta": (3,),
            "L": (2, 2),
            "p": (3,),
        }

    def test_param_blocks_match_real_names(self, structured_model):
        # The pure name-parser, fed BridgeStan's real param_names(), recovers
        # the declared shapes — guards against BridgeStan changing its
        # flattening convention out from under _param_blocks.
        blocks = _param_blocks(structured_model._bs_model.param_names())
        assert [(b.name, b.shape) for b in blocks] == [
            ("mu", ()),
            ("theta", (3,)),
            ("L", (2, 2)),
            ("p", (3,)),
        ]

    def test_getitem_uses_block_fields(self, structured_model):
        assert structured_model["theta"] == "theta"
        with pytest.raises(KeyError, match="Unknown component"):
            structured_model["theta.1"]  # per-scalar names are not components

    def test_pack_value_assembles_column_major(self, structured_model):
        flat = structured_model._pack_value(
            mu=0.5,
            theta=jnp.array([1.0, 2.0, 3.0]),
            L=jnp.array([[10.0, 30.0], [20.0, 40.0]]),
            p=jnp.array([0.2, 0.3, 0.5]),
        )
        # L is packed column-major to match BridgeStan: [10, 20, 30, 40].
        assert jnp.allclose(
            flat, jnp.array([0.5, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 0.2, 0.3, 0.5])
        )

    def test_pack_value_missing_block_raises(self, structured_model):
        with pytest.raises(TypeError, match="missing"):
            structured_model._pack_value(
                mu=0.5, theta=jnp.array([1.0, 2.0, 3.0]), L=jnp.zeros((2, 2))
            )  # no p

    def test_pack_value_unexpected_block_raises(self, structured_model):
        with pytest.raises(TypeError, match="unexpected"):
            structured_model._pack_value(
                mu=0.5,
                theta=jnp.array([1.0, 2.0, 3.0]),
                L=jnp.zeros((2, 2)),
                p=jnp.array([0.2, 0.3, 0.5]),
                zzz=1.0,
            )

    def test_pack_value_wrong_shape_raises(self, structured_model):
        with pytest.raises(TypeError, match=r"shape \(2, 2\)"):
            structured_model._pack_value(
                mu=0.5,
                theta=jnp.array([1.0, 2.0, 3.0]),
                L=jnp.array([1.0, 2.0, 3.0, 4.0]),  # flat, not (2, 2)
                p=jnp.array([0.2, 0.3, 0.5]),
            )

    def test_keyword_form_equals_positional(self, structured_model):
        # Both forms route through the same _pack_value, so this guards the
        # keyword->positional dispatch wiring; the column-major packing *order*
        # is validated by test_pack_value_assembles_column_major and
        # test_param_blocks_match_real_names. 0.25/0.25/0.5 are exact in float32,
        # so the simplex sums to 1 and BridgeStan's param_unconstrain accepts it.
        kw = dict(
            mu=0.5,
            theta=jnp.array([0.1, 0.2, 0.3]),
            L=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            p=jnp.array([0.25, 0.25, 0.5]),
        )
        lp_kw = float(jnp.asarray(log_prob(structured_model, **kw)))
        lp_pos = float(jnp.asarray(log_prob(structured_model, structured_model._pack_value(**kw))))
        np.testing.assert_allclose(lp_kw, lp_pos, atol=1e-6)


class TestUnconstrainedStanView:
    """The unconstrained view. Its blocks follow ``param_unc_names``, so a
    simplex appears with one fewer dimension than in the constrained model."""

    def test_name_with_base(self, structured_model):
        assert structured_model.as_unconstrained_distribution().name == "structured_unconstrained"

    def test_name_without_base(self, conjugate_stan_file):
        model = StanModel(conjugate_stan_file, data={"N": 3, "y": [1.0, 2.0, 3.0]})
        view = model.as_unconstrained_distribution()
        assert view.name == "StanModel_unconstrained"

    def test_event_shape_matches_model(self, structured_model):
        view = structured_model.as_unconstrained_distribution()
        assert view.event_shape == structured_model.event_shape == (10,)

    def test_blocks_follow_unconstrained_names(self, structured_model):
        view = structured_model.as_unconstrained_distribution()
        assert view.fields == ("mu", "theta", "L", "p")
        # The simplex is unconstrained in (n-1) free coordinates.
        assert structured_model.event_template["p"] == ArraySpec((3,))
        assert view.event_template["p"] == ArraySpec((2,))

    def test_log_prob_finite_and_unnormalized_agrees(self, structured_model):
        view = structured_model.as_unconstrained_distribution()
        # An all-zeros unconstrained point is always valid (maps to the centre
        # of every constrained support).
        x = jnp.zeros(view.event_shape)
        lp = view._log_prob(x)
        assert jnp.isfinite(lp)
        np.testing.assert_allclose(float(view._unnormalized_log_prob(x)), float(lp), atol=1e-6)

    def test_log_prob_finite_difference_in_mu(self, structured_model):
        # Exact value check: shifting the unconstrained mu coordinate by m moves
        # the log-density by exactly -0.5*m^2 (the N(0, 1) prior on mu), with the
        # simplex Jacobian and the other blocks held fixed.
        view = structured_model.as_unconstrained_distribution()
        x0 = jnp.zeros(view.event_shape)
        for m in (0.5, 1.0):
            delta = float(view._log_prob(x0.at[0].set(m))) - float(view._log_prob(x0))
            np.testing.assert_allclose(delta, -0.5 * m**2, atol=1e-5)

    def test_prob_is_exp_log_prob(self, structured_model):
        view = structured_model.as_unconstrained_distribution()
        x = jnp.zeros(view.event_shape)
        np.testing.assert_allclose(
            float(view._prob(x)), float(jnp.exp(view._log_prob(x))), rtol=1e-6
        )

    def test_repr(self, structured_model):
        r = repr(structured_model.as_unconstrained_distribution())
        assert "UnconstrainedStanView" in r
        assert "structured" in r


class TestStanModelConditionOn:
    def test_condition_on_delegates_to_registry(self, conjugate_model):
        """condition_on routes a StanModel through the inference registry
        (the registry is patched, so no actual MCMC runs)."""
        from probpipe import condition_on

        with patch("probpipe.inference._registry.inference_method_registry.execute") as mock_exec:
            mock_exec.return_value = MagicMock()
            condition_on(conjugate_model, {"y": [1, 2, 3]}, num_results=10)
            mock_exec.assert_called_once()
