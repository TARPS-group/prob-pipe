"""Tests for StanModel: the kernel from a Stan program's data to its unnormalized posterior.

The parameter-name parser's tests run everywhere. A StanModel reads its
program with BridgeStan's stanc, so the ``_stanc`` fixture gates the tests that
construct one, and the ``_stan_toolchain`` fixture gates those that evaluate a
density through BridgeStan with a probe compile; both run where the ``stan``
extra is installed and skip elsewhere.
"""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Record, SupportsLogProb, SupportsUnnormalizedLogProb, unnormalized_log_prob
from probpipe.families import StanModel
from probpipe.families._programs import _param_blocks
from probpipe.operations._condition import condition_on

# ---------------------------------------------------------------------------
# Pure tests — no BridgeStan backend required
# ---------------------------------------------------------------------------


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


def _program(tmp_path, text: str) -> str:
    path = tmp_path / "program.stan"
    path.write_text(text)
    return str(path)


@pytest.mark.usefixtures("_stanc")
class TestTheProgramText:
    """The given slots and the parameter record are read from stanc and the program text."""

    def test_bounds_arrays_and_matrix_types_are_read(self, tmp_path):
        stan_file = _program(
            tmp_path,
            """
            data {
              int<lower=0> N;       // the observations
              array[N] real<lower=0, upper=10> x;  /* a bounded array */
            }
            transformed data { real scale = 2; }
            parameters {
              cholesky_factor_corr[3] L;
              array[2] vector<lower=0>[N] z;
              simplex[4] p;
              matrix<offset=0, multiplier=2>[N, 2] B;
            }
            model { }
            """,
        )
        model = StanModel(stan_file, label="program")
        assert list(model.given_spec) == ["N", "x"]
        shapes = {name: spec.shape for name, spec in model.event_spec.spec.children.items()}
        assert shapes == {"L": (3, 3), "z": (2, "N"), "p": (4,), "B": ("N", 2)}

    def test_a_scalar_entry_binds_the_sizes_it_names(self, tmp_path):
        stan_file = _program(tmp_path, "data { int K; } parameters { vector[K] b; } model { }")
        assert StanModel(stan_file, data={"K": 3}, label="p").event_spec.spec["b"].shape == (3,)

    def test_a_size_expression_is_a_symbolic_dimension(self, tmp_path):
        stan_file = _program(
            tmp_path, "data { int K; vector[K] y; } parameters { vector[K - 1] b; } model { }"
        )
        assert StanModel(stan_file, label="p").event_spec.spec["b"].shape == ("b_0",)

    def test_an_unreadable_declaration_raises(self, tmp_path):
        stan_file = _program(tmp_path, "parameters { tuple(real, real) t; } model { }")
        with pytest.raises(ValueError, match="Stan"):
            StanModel(stan_file, label="p")

    def test_a_program_without_parameters_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no parameters"):
            StanModel(_program(tmp_path, "data { int N; } model { }"), label="p")


class TestTheDensityBackend:
    def test_construction_needs_bridgestans_stanc(self, tmp_path):
        stan_file = _program(tmp_path, "parameters { real nu; } model { nu ~ normal(0, 1); }")
        with (
            patch.dict("sys.modules", {"bridgestan": None, "bridgestan.compile": None}),
            pytest.raises(ImportError, match="pip install bridgestan"),
        ):
            StanModel(stan_file, label="model")

    @pytest.mark.usefixtures("_stanc")
    def test_the_density_needs_bridgestan(self, tmp_path):
        """The density raises ImportError with install instructions when bridgestan
        is missing — this *must* simulate bridgestan's absence, so it patches
        ``sys.modules`` rather than using a real backend."""
        posterior = StanModel(
            _program(tmp_path, "parameters { real mu; } model { }"), label="model"
        )
        with (
            patch.dict("sys.modules", {"bridgestan": None}),
            pytest.raises(ImportError, match="pip install bridgestan"),
        ):
            posterior._unnormalized_log_prob(jnp.zeros(1))


# ---------------------------------------------------------------------------
# Real BridgeStan backend
#
# The model fixtures below depend on the shared ``_stan_toolchain`` fixture
# (tests/conftest.py), which ``importorskip``s bridgestan and probe-compiles the
# C++ toolchain — so these tests skip together when the backend is absent.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conjugate_stan_file(_stan_toolchain, tmp_path_factory):
    """Path to the conjugate program ``y ~ Normal(mu, 1)`` with a unit prior on ``mu``.

    ``mu`` is a single unconstrained real, so the log-density is closed-form
    and the constraining transform is the identity.
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
    return str(stan_file)


@pytest.fixture(scope="module")
def conjugate_model(conjugate_stan_file):
    """The conjugate program's posterior at three observations."""
    return StanModel(conjugate_stan_file, data={"N": 3, "y": [1.0, 2.0, 3.0]}, label="normal_mean")


@pytest.fixture(scope="module")
def structured_model(_stan_toolchain, tmp_path_factory):
    """A program with a scalar, a vector, a column-major matrix, and a simplex.

    The simplex's unconstrained parametrisation has one fewer dimension, so
    the unconstrained view's blocks differ from the posterior's.
    """
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
    return StanModel(str(stan_file), label="structured")


class TestTheModelLibrary:
    def test_a_loaded_model_leaves_jax_able_to_allocate_empty_arrays(self, conjugate_model):
        """The model library links TBB without its malloc proxy.

        The proxy replaces the process allocator, after which JAX's zero-size
        allocations fail with RESOURCE_EXHAUSTED for the rest of the process.
        """
        conjugate_model._bridgestan_model()
        empty = jnp.zeros((0,)) + 1.0
        assert empty.shape == (0,)
        assert float(jnp.ones((3,)).sum()) == 3.0


class TestStanPosteriorDensity:
    """The posterior's density against the conjugate program's closed form.

    Stan's log density drops every normalizing constant, and ``mu`` carries no
    Jacobian, so the density is ``-mu²/2 - Σ(yᵢ - mu)²/2`` exactly.
    """

    _Y = np.array([1.0, 2.0, 3.0])

    def _expected(self, mu):
        return -0.5 * mu**2 - 0.5 * float(np.sum((self._Y - mu) ** 2))

    def test_it_claims_only_the_unnormalized_density(self, conjugate_model):
        assert isinstance(conjugate_model, SupportsUnnormalizedLogProb)
        assert not isinstance(conjugate_model, SupportsLogProb)

    def test_the_density_matches_the_closed_form(self, conjugate_model):
        for mu in [-1.0, 0.0, 0.5, 2.0]:
            value = Record("value", {"mu": jnp.asarray(mu)})
            lp = float(conjugate_model._unnormalized_log_prob(value))
            np.testing.assert_allclose(lp, self._expected(mu), atol=1e-5)

    def test_the_flat_vector_is_accepted_in_float32(self, conjugate_model):
        lp = float(conjugate_model._unnormalized_log_prob(jnp.asarray([0.5], dtype=jnp.float32)))
        np.testing.assert_allclose(lp, self._expected(0.5), atol=1e-5)


class TestStanPosteriorBlocks:
    def test_the_declared_record_shapes(self, structured_model):
        assert structured_model.event_spec.spec.leaf_shapes == {
            "mu": (),
            "theta": (3,),
            "L": (2, 2),
            "p": (3,),
        }

    def test_param_blocks_match_real_names(self, structured_model):
        blocks = _param_blocks(structured_model._bridgestan_model().param_names())
        assert [(b.name, b.shape) for b in blocks] == [
            ("mu", ()),
            ("theta", (3,)),
            ("L", (2, 2)),
            ("p", (3,)),
        ]

    def test_pack_value_assembles_column_major(self, structured_model):
        flat = structured_model._pack_value(
            mu=0.5,
            theta=jnp.array([1.0, 2.0, 3.0]),
            L=jnp.array([[10.0, 30.0], [20.0, 40.0]]),
            p=jnp.array([0.2, 0.3, 0.5]),
        )
        assert jnp.allclose(
            flat, jnp.array([0.5, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 0.2, 0.3, 0.5])
        )

    def test_pack_value_wrong_shape_raises(self, structured_model):
        with pytest.raises(TypeError, match=r"shape \(2, 2\)"):
            structured_model._pack_value(
                mu=0.5,
                theta=jnp.array([1.0, 2.0, 3.0]),
                L=jnp.array([1.0, 2.0, 3.0, 4.0]),
                p=jnp.array([0.2, 0.3, 0.5]),
            )

    def test_the_mapping_form_equals_the_record_form(self, structured_model):
        kw = dict(
            mu=0.5,
            theta=jnp.array([0.1, 0.2, 0.3]),
            L=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            p=jnp.array([0.25, 0.25, 0.5]),
        )
        lp_kw = float(jnp.asarray(unnormalized_log_prob(structured_model, kw)))
        lp_record = float(structured_model._unnormalized_log_prob(Record("value", kw)))
        np.testing.assert_allclose(lp_kw, lp_record, atol=1e-6)


class TestUnconstrainedStanView:
    def test_blocks_follow_unconstrained_names(self, structured_model):
        view = structured_model.as_unconstrained_distribution()
        assert view.label == "structured_unconstrained"
        assert view.event_spec.spec.leaf_shapes["p"] == (2,)

    def test_the_density_is_finite_and_unnormalized(self, structured_model):
        view = structured_model.as_unconstrained_distribution()
        assert isinstance(view, SupportsUnnormalizedLogProb)
        assert not isinstance(view, SupportsLogProb)
        assert jnp.isfinite(view._unnormalized_log_prob(jnp.zeros(10)))


class TestStanModelConditionOn:
    def test_binding_the_data_curries_then_a_stan_method_normalizes(self, conjugate_stan_file):
        report = condition_on.check(
            StanModel(conjugate_stan_file, label="normal_mean"), {"N": 3, "y": [1.0, 2.0, 3.0]}
        )
        assert report.route == "curry"
        assert report.method in ("nutpie_nuts", "cmdstan_nuts", "blackjax_rwmh")
