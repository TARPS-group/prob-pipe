"""Tests for the nutpie Function.

These tests require nutpie (and pymc, for the PyMC integration path) to
be installed.  Helper / error-path tests that don't require a compiled
model are isolated in ``TestHelpers``.
"""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

nutpie = pytest.importorskip("nutpie")

from probpipe.inference import ApproximateDistribution
from probpipe.inference._nutpie import (
    _compile_for_nutpie,
    _extract_chains,
    condition_on_nutpie,
)
from tests.inference._harness import validate_method

# ---------------------------------------------------------------------------
# Helpers (no model compilation needed)
# ---------------------------------------------------------------------------


class _CompiledStanModel:
    """A stand-in for nutpie's CompiledStanModel, whose data ``with_data`` sets."""

    def __init__(self, filename, data=None):
        self.filename, self.data = filename, data

    def with_data(self, *, seed=None, **updates):
        return _CompiledStanModel(self.filename, {**(self.data or {}), **updates})


def _compile_stan_model(*, code=None, filename=None, **kwargs):
    """nutpie 0.16's compile_stan_model, which takes every argument by keyword."""
    return _CompiledStanModel(filename)


class TestCompileForNutpie:
    """_compile_for_nutpie dispatch — still uses mocks since we only test
    which nutpie function is called, not that it produces a runnable model."""

    def test_a_stan_posterior_compiles_from_its_file_with_its_data(self, tmp_path):
        """A Stan posterior compiles through nutpie.compile_stan_model from its
        program's file, and nutpie's compiled model takes its data."""
        from probpipe.families import StanModel
        from probpipe.families._programs import _StanPosterior

        program = tmp_path / "program.stan"
        program.write_text("data { int N; } parameters { real mu; } model { }")
        posterior = StanModel("program", str(program), data={"N": 3})
        with (
            patch.object(_StanPosterior, "_bridgestan_model", lambda self: "bs_model"),
            patch.object(nutpie, "compile_stan_model", _compile_stan_model),
        ):
            compiled, pymc_build = _compile_for_nutpie(posterior, data=None)
        assert (compiled.filename, compiled.data) == (str(program), {"N": 3})
        assert pymc_build is None  # Stan target — no PyMC build to thread

    def test_a_stan_kernel_curries_to_its_posterior_at_the_data(self, tmp_path):
        """A StanModel given its remaining data curries to the posterior first,
        whose data are the construction data and the conditioning data together."""
        from probpipe.families import StanModel

        program = tmp_path / "program.stan"
        program.write_text("data { int N; vector[N] y; } parameters { real mu; } model { }")
        kernel = StanModel("program", str(program), data={"N": 2})
        with patch.object(nutpie, "compile_stan_model", _compile_stan_model):
            compiled, _ = _compile_for_nutpie(kernel, data={"y": [1.0, 2.0]})
        assert compiled.data == {"N": 2, "y": [1.0, 2.0]}

    def test_a_stan_posterior_keeps_its_parameter_record(self, tmp_path):
        """The posterior holds the parameter blocks alone, each in its own shape."""
        import xarray as xr

        from probpipe.families import StanModel

        program = tmp_path / "program.stan"
        program.write_text(
            "data { int N; } parameters { real mu; vector[2] theta; } model { } "
            "generated quantities { real twice = 2 * mu; }"
        )
        posterior = StanModel("program", str(program), data={"N": 3})
        mu = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
        trace = xr.DataTree.from_dict(
            {
                "posterior": xr.Dataset(
                    {
                        "mu": (("chain", "draw"), mu),
                        "theta": (("chain", "draw", "dim"), np.stack([100 + mu, 1000 + mu], -1)),
                        "twice": (("chain", "draw"), 2 * mu),
                    }
                )
            }
        )
        with (
            patch.object(nutpie, "compile_stan_model", _compile_stan_model),
            patch.object(nutpie, "sample", return_value=trace),
        ):
            result = condition_on_nutpie.apply(posterior, num_results=3, num_chains=2)
        assert tuple(result.event_spec.components) == ("mu", "theta")
        np.testing.assert_array_equal(
            np.asarray(result.chains[1]), [[10, 110, 1010], [11, 111, 1011], [12, 112, 1012]]
        )

    def test_a_posterior_builds_its_bridgestan_model_once(self, tmp_path):
        """The posterior's BridgeStan model is built at its data on first use and reused."""
        from probpipe.families import StanModel

        program = tmp_path / "program.stan"
        program.write_text("data { int N; } parameters { real mu; } model { }")
        posterior = StanModel("program", str(program), data={"N": 3})
        bridgestan = MagicMock()
        with patch.dict("sys.modules", {"bridgestan": bridgestan}):
            first = posterior._bridgestan_model()
            second = posterior._bridgestan_model()
        assert first is second
        bridgestan.StanModel.assert_called_once_with(
            str(program), data={"N": 3}, make_args=["TBB_LIBRARIES=tbb"]
        )

    def test_pymc_path(self):
        """Models with _pymc_model use nutpie.compile_pymc_model and
        return the conditioned build, from which the parameter record is read."""
        model = MagicMock(spec=[])
        model._pymc_model = MagicMock(return_value="pm_model")
        with patch.object(nutpie, "compile_pymc_model", return_value="compiled") as compile_pymc:
            compiled, pymc_build = _compile_for_nutpie(model, data={"y": [1, 2]})
        compile_pymc.assert_called_once_with("pm_model")
        model._pymc_model.assert_called_once_with(data={"y": [1, 2]})
        assert compiled == "compiled"
        assert pymc_build == "pm_model"

    def test_unsupported_model_raises(self):
        model = MagicMock(spec=[])
        with pytest.raises(TypeError, match="does not support"):
            _compile_for_nutpie(model, data=None)

    def test_a_stan_posterior_conditioned_on_a_parameter_is_declined(self, tmp_path):
        """nutpie samples a Stan program at its data, so it cannot fix a parameter."""
        from probpipe.families import StanModel
        from probpipe.inference._nutpie import NutpieNutsMethod
        from probpipe.operations._condition import condition_on

        program = tmp_path / "program.stan"
        program.write_text(
            "data { int N; vector[N] y; } parameters { real mu; real<lower=0> sigma; } "
            "model { y ~ normal(mu, sigma); }"
        )
        posterior = StanModel("program", str(program), data={"N": 2, "y": [1.0, 2.0]})
        target = condition_on.with_options(method="unnormalized")(posterior, {"mu": 0.3})
        report = NutpieNutsMethod().check(target)
        assert report.feasible is False
        assert "parameter" in report.description
        assert NutpieNutsMethod().check(posterior).feasible is True


class TestImportError:
    """When nutpie is missing, condition_on_nutpie raises a helpful
    ImportError.  This path is exercised by temporarily hiding nutpie."""

    def test_import_error_message(self):
        with (
            patch.dict("sys.modules", {"nutpie": None}),
            pytest.raises(ImportError, match="pip install nutpie"),
        ):
            condition_on_nutpie.apply(MagicMock(), num_results=10)


# ---------------------------------------------------------------------------
# _extract_chains — exercised with real ArviZ trace structure
# ---------------------------------------------------------------------------


class TestExtractChains:
    """Tests against mock arviz-like objects whose `values` attribute is a
    real numpy array, matching nutpie's actual trace shape."""

    def test_scalar_params_two_chains(self):
        mock_trace = MagicMock()
        mu_vals = np.random.randn(2, 10)
        sigma_vals = np.random.randn(2, 10)
        mu_var = MagicMock()
        mu_var.values = mu_vals
        sigma_var = MagicMock()
        sigma_var.values = sigma_vals

        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["mu", "sigma"]
        mock_posterior.__getitem__ = lambda self, k: {"mu": mu_var, "sigma": sigma_var}[k]
        mock_trace.posterior = mock_posterior

        chains, param_names = _extract_chains(mock_trace, num_chains=2)

        assert param_names == ["mu", "sigma"]
        assert len(chains) == 2
        assert chains[0].shape == (10, 2)
        np.testing.assert_allclose(chains[0][:, 0], mu_vals[0])
        np.testing.assert_allclose(chains[1][:, 1], sigma_vals[1])

    def test_multidim_params(self):
        mock_trace = MagicMock()
        beta_vals = np.random.randn(1, 5, 3)
        beta_var = MagicMock()
        beta_var.values = beta_vals

        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["beta"]
        mock_posterior.__getitem__ = lambda self, k: beta_var
        mock_trace.posterior = mock_posterior

        chains, _ = _extract_chains(mock_trace, num_chains=1)
        assert chains[0].shape == (5, 3)

    def test_no_posterior_raises(self):
        mock_trace = MagicMock(spec=[])
        with pytest.raises(TypeError, match="Cannot extract chains"):
            _extract_chains(mock_trace, num_chains=1)

    def test_keep_names_overrides_data_vars_order(self):
        """keep_names selects and orders columns explicitly, overriding
        the alphabetical posterior.data_vars order.

        nutpie sorts data_vars alphabetically; without an explicit order
        the concatenated columns would not line up with the PyMC template
        field order (declaration order), silently mislabeling draws.
        """
        mock_trace = MagicMock()
        a = np.full((1, 4), 1.0)
        m = np.full((1, 4), 2.0)
        z = np.full((1, 4), 3.0)
        va = MagicMock()
        va.values = a
        vm = MagicMock()
        vm.values = m
        vz = MagicMock()
        vz.values = z
        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["alpha", "mu", "zeta"]  # nutpie's sorted order
        mock_posterior.__getitem__ = lambda self, k: {"alpha": va, "mu": vm, "zeta": vz}[k]
        mock_trace.posterior = mock_posterior

        chains, names = _extract_chains(
            mock_trace,
            num_chains=1,
            keep_names=["zeta", "alpha", "mu"],
        )
        assert names == ["zeta", "alpha", "mu"]
        # Columns concatenated in keep_names order: zeta=3, alpha=1, mu=2.
        np.testing.assert_array_equal(chains[0][0], [3.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Real integration: nutpie + StanModel (requires a BridgeStan toolchain)
#
# Uses the shared ``_stan_toolchain`` fixture (tests/conftest.py).
# ---------------------------------------------------------------------------


class TestNutpieStanIntegration:
    """nutpie sampling of a StanModel preserves construction-time data."""

    def test_construction_data_survives_conditioning(self, _stan_toolchain, tmp_path_factory):
        """A StanModel built with fixed data (N, x) samples via nutpie when
        conditioned on y.  Without merging the construction data, the rebuilt
        BridgeStan model would lack N and x and fail to instantiate — so
        reaching a finite posterior pulled toward the data-generating beta is
        the regression signal.  nutpie's inference accuracy itself is covered
        by the PyMC integration tests below.
        """
        from probpipe.modeling import StanModel

        stan_file = tmp_path_factory.mktemp("stan_models") / "linreg.stan"
        stan_file.write_text(
            """
            data {
              int<lower=0> N;
              vector[N] x;
              vector[N] y;
            }
            parameters {
              real alpha;
              real beta;
            }
            model {
              alpha ~ normal(0, 1);
              beta ~ normal(0, 1);
              y ~ normal(alpha + beta * x, 1);
            }
            """
        )
        N = 20
        rng = np.random.default_rng(0)
        x = rng.normal(size=N)
        y = 0.5 + 1.5 * x + rng.normal(size=N)
        model = StanModel(
            "linreg",
            str(stan_file),
            data={"N": N, "x": x.tolist(), "y": y.tolist()},
        )

        result = condition_on_nutpie.apply(
            model,
            data={"y": y.tolist()},
            num_results=200,
            num_warmup=200,
            num_chains=2,
            random_seed=0,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.num_chains == 2
        assert result.algorithm == "nutpie_nuts"
        post = result.inference_data.posterior
        assert "alpha" in post and "beta" in post
        beta_mean = float(np.asarray(post["beta"]).mean())
        assert np.isfinite(beta_mean)
        # Data uses beta = 1.5; the posterior should be pulled toward it and
        # away from the N(0, 1) prior mean of 0 (a tolerance-free directional
        # check, since this run can't be re-seeded here to measure a bound).
        assert abs(beta_mean - 1.5) < abs(beta_mean - 0.0)


# ---------------------------------------------------------------------------
# Real integration: nutpie + PyMCModel
# ---------------------------------------------------------------------------


pm = pytest.importorskip("pymc")

from probpipe.modeling import PyMCModel


def _gaussian_pymc_fn(y=None):
    """Known-conjugate Gaussian model.  Posterior mean for mu is
    (n * y_bar / sigma^2) / (1/tau_0^2 + n/sigma^2) with tau_0=10,
    sigma=1, known from closed form."""
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        pm.Normal("y", mu, 1.0, observed=y)
    return m


class TestNutpieIntegration:
    """End-to-end sampling via real nutpie + PyMC compilation."""

    def test_samples_recover_posterior(self):
        """Nutpie recovers the analytical posterior mean for a simple Gaussian."""
        np.random.seed(0)
        y_obs = np.array([1.2, 0.8, 1.1, 0.9, 1.0], dtype=float)
        model = PyMCModel("gaussian", _gaussian_pymc_fn)
        result = condition_on_nutpie.apply(
            model,
            data={"y": y_obs},
            num_results=500,
            num_warmup=200,
            num_chains=2,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.num_chains == 2
        assert result.algorithm == "nutpie_nuts"
        assert result.provenance is not None
        assert result.provenance.operation == "nutpie_nuts"
        # Analytical posterior: prior N(0, 10), likelihood N(mu, 1) with n=5
        #   Precision: 1/100 + 5 = 5.01  ->  var = 0.1996
        #   Mean:      5.0 * y_bar / 5.01
        y_bar = float(y_obs.mean())
        post_mean = 5.0 * y_bar / (1.0 / 100.0 + 5.0)
        post_sd = np.sqrt(1.0 / (1.0 / 100.0 + 5.0))
        # PyMCModel declares one field per PyMC RV, so draws() returns a
        # NumericRecordBatch keyed by RV name. The only parameter is `mu`,
        # with event_shape ().
        draws = result.draws()
        assert draws.event_template.fields == ("mu",)
        mu_draws = jnp.asarray(draws["mu"])
        assert mu_draws.shape == (1000,)  # 2 chains × 500 draws, flattened
        # With 1000 draws total, MC SE for mean ~ post_sd / sqrt(1000) ~ 0.014
        np.testing.assert_allclose(float(jnp.mean(mu_draws)), post_mean, atol=0.05)
        np.testing.assert_allclose(float(jnp.std(mu_draws)), post_sd, atol=0.05)

    def test_the_registry_method_names_its_target_as_the_parent(self):
        """The posterior's provenance names the target it normalized, as pymc_nuts' does."""
        from probpipe.inference._nutpie import NutpieNutsMethod
        from probpipe.operations._condition import condition_on

        model = PyMCModel("gaussian", _gaussian_pymc_fn)
        target = condition_on.with_options(method="unnormalized")(
            model, {"y": np.array([0.0, 1.0])}
        )
        result = NutpieNutsMethod().execute(
            target, num_results=30, num_warmup=30, num_chains=1, random_seed=0
        )
        (parent,) = result.provenance.parents
        assert (parent.type_name, parent.provenance) == (
            "_UnnormalizedConditional",
            target.provenance,
        )

    def test_annotations_trace_attached(self):
        model = PyMCModel("gaussian", _gaussian_pymc_fn)
        y_obs = np.array([0.0, 1.0], dtype=float)
        result = condition_on_nutpie.apply(
            model,
            data={"y": y_obs},
            num_results=50,
            num_warmup=50,
            num_chains=1,
            random_seed=0,
        )
        assert result.inference_data is not None
        # arviz-like trace exposes posterior as an xarray Dataset/DataTree
        assert hasattr(result.inference_data, "posterior")

    def test_multiparam_draws_not_mislabeled(self):
        """Draws are labeled by the model's parameter order, not nutpie's
        alphabetical ``posterior.data_vars`` order.

        Declares ``zeta``, ``alpha``, ``mu`` (non-alphabetical) with
        distinct tight priors and a near-flat likelihood, so each
        posterior stays near its prior. If chain columns were taken in
        alphabetical order while the template uses declaration order,
        the means would be assigned to the wrong fields.
        """

        def model_fn(y=None):
            with pm.Model() as m:
                zeta = pm.Normal("zeta", 100.0, 1.0)
                alpha = pm.Normal("alpha", 0.0, 1.0)
                mu = pm.Normal("mu", -100.0, 1.0)
                pm.Normal("y", mu=zeta + alpha + mu, sigma=1000.0, observed=y)
            return m

        model = PyMCModel("ordering", model_fn)
        result = condition_on_nutpie.apply(
            model,
            data={"y": np.zeros(4, dtype=float)},
            num_results=300,
            num_warmup=300,
            num_chains=1,
            random_seed=0,
        )
        draws = result.draws()
        assert draws.event_template.fields == ("zeta", "alpha", "mu")
        for field, prior_mean in [("zeta", 100.0), ("alpha", 0.0), ("mu", -100.0)]:
            got = float(jnp.mean(jnp.asarray(draws[field])))
            np.testing.assert_allclose(got, prior_mean, atol=10.0)

    def test_partial_conditioning_draws_not_mislabeled(self):
        """Partial conditioning via nutpie: an unsupplied observed variable
        is inferred and its draws are labeled correctly.

        ``X`` is declared ``observed=X``; conditioning on ``y`` alone
        leaves it free, so the posterior covers ``mu`` and ``X``. ``X``
        sorts before ``mu`` in nutpie's alphabetical ``data_vars`` while
        the param order is ``(mu, X)``, so distinct priors catch any
        column mislabeling.
        """

        def model_fn(X=None, y=None):
            with pm.Model() as m:
                mu = pm.Normal("mu", 100.0, 0.5)
                X_rv = pm.Normal("X", -100.0, 0.5, observed=X)
                pm.Normal("y", mu=mu + X_rv, sigma=1000.0, observed=y)
            return m

        model = PyMCModel("partial", model_fn)
        result = condition_on_nutpie.apply(
            model,
            data={"y": np.zeros(5, dtype=float)},
            num_results=200,
            num_warmup=200,
            num_chains=1,
            random_seed=0,
        )
        draws = result.draws()
        assert set(draws.event_template.fields) == {"mu", "X"}
        np.testing.assert_allclose(float(jnp.mean(jnp.asarray(draws["mu"]))), 100.0, atol=10.0)
        np.testing.assert_allclose(float(jnp.mean(jnp.asarray(draws["X"]))), -100.0, atol=10.0)


# ---------------------------------------------------------------------------
# The canonical cases of the cross-method validation harness
# ---------------------------------------------------------------------------

test_nutpie_nuts_canonical_pymc = validate_method("nutpie_nuts", representation="pymc")
test_nutpie_nuts_canonical_stan = validate_method("nutpie_nuts", representation="stan")
