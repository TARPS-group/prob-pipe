"""Tests for the nutpie workflow function.

These tests require nutpie (and pymc, for the PyMC integration path) to
be installed.  Helper / error-path tests that don't require a compiled
model are isolated in ``TestHelpers``.
"""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

nutpie = pytest.importorskip("nutpie")

from probpipe.inference import ApproximateDistribution  # noqa: E402
from probpipe.inference._nutpie import (  # noqa: E402
    _compile_for_nutpie,
    _extract_chains,
    condition_on_nutpie,
)

# ---------------------------------------------------------------------------
# Helpers (no model compilation needed)
# ---------------------------------------------------------------------------


class TestCompileForNutpie:
    """_compile_for_nutpie dispatch — still uses mocks since we only test
    which nutpie function is called, not that it produces a runnable model."""

    def test_bridgestan_path(self):
        """Stan targets use nutpie.compile_stan_model, with the conditioning
        data merged on top of the construction-time data (``_stan_data``)."""
        model = MagicMock()
        model._stan_data = {"N": 10, "x": [1, 2, 3]}
        model._bridgestan_model.return_value = "bs_model"
        with patch.object(nutpie, "compile_stan_model",
                          return_value="compiled") as compile_stan:
            compiled, pymc_build = _compile_for_nutpie(model, data={"y": [4, 5, 6]})
        compile_stan.assert_called_once_with("bs_model")
        # Construction data (N, x) is preserved, not dropped for the observed y.
        model._bridgestan_model.assert_called_once_with(
            data={"N": 10, "x": [1, 2, 3], "y": [4, 5, 6]}
        )
        assert compiled == "compiled"
        assert pymc_build is None  # Stan target — no PyMC build to thread

    def test_bridgestan_observed_overrides_construction_data(self):
        """A conditioning value wins over a construction-time value of the
        same name (matches the CmdStan method's merge order)."""
        model = MagicMock()
        model._stan_data = {"N": 10, "y": [0.0, 0.0]}
        model._bridgestan_model.return_value = "bs_model"
        with patch.object(nutpie, "compile_stan_model", return_value="compiled"):
            _compile_for_nutpie(model, data={"y": [1.0, 2.0]})
        model._bridgestan_model.assert_called_once_with(data={"N": 10, "y": [1.0, 2.0]})

    def test_bridgestan_no_observed_reuses_cached_model(self):
        """With no conditioning data, ``_bridgestan_model`` is called with
        ``data=None`` so the model built at construction is reused, not
        rebuilt."""
        model = MagicMock()
        model._stan_data = {"N": 10}
        model._bridgestan_model.return_value = "bs_model"
        with patch.object(nutpie, "compile_stan_model", return_value="compiled"):
            _compile_for_nutpie(model, data=None)
        model._bridgestan_model.assert_called_once_with(data=None)

    def test_pymc_path(self):
        """Models with _pymc_model use nutpie.compile_pymc_model and
        return the conditioned build for event_template derivation."""
        model = MagicMock(spec=[])
        model._pymc_model = MagicMock(return_value="pm_model")
        with patch.object(nutpie, "compile_pymc_model",
                          return_value="compiled") as compile_pymc:
            compiled, pymc_build = _compile_for_nutpie(model, data={"y": [1, 2]})
        compile_pymc.assert_called_once_with("pm_model")
        model._pymc_model.assert_called_once_with(data={"y": [1, 2]})
        assert compiled == "compiled"
        assert pymc_build == "pm_model"

    def test_unsupported_model_raises(self):
        model = MagicMock(spec=[])
        with pytest.raises(TypeError, match="does not support"):
            _compile_for_nutpie(model, data=None)


class TestImportError:
    """When nutpie is missing, condition_on_nutpie raises a helpful
    ImportError.  This path is exercised by temporarily hiding nutpie."""

    def test_import_error_message(self):
        with (
            patch.dict("sys.modules", {"nutpie": None}),
            pytest.raises(ImportError, match="pip install nutpie"),
        ):
            condition_on_nutpie._func(MagicMock(), num_results=10)


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
        mock_posterior.__getitem__ = (
            lambda self, k: {"mu": mu_var, "sigma": sigma_var}[k]
        )
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
        mock_posterior.__getitem__ = (
            lambda self, k: {"alpha": va, "mu": vm, "zeta": vz}[k]
        )
        mock_trace.posterior = mock_posterior

        chains, names = _extract_chains(
            mock_trace, num_chains=1, keep_names=["zeta", "alpha", "mu"],
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

    def test_construction_data_survives_conditioning(
        self, _stan_toolchain, tmp_path_factory
    ):
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
            str(stan_file),
            data={"N": N, "x": x.tolist(), "y": y.tolist()},
            name="linreg",
        )

        result = condition_on_nutpie._func(
            model, data={"y": y.tolist()},
            num_results=200, num_warmup=200, num_chains=2, random_seed=0,
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

from probpipe.modeling import PyMCModel  # noqa: E402


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
        model = PyMCModel(_gaussian_pymc_fn, name="gaussian")
        result = condition_on_nutpie._func(
            model, data={"y": y_obs},
            num_results=500, num_warmup=200, num_chains=2, random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.num_chains == 2
        assert result.algorithm == "nutpie_nuts"
        assert result.source is not None
        assert result.source.operation == "nutpie_nuts"
        # Analytical posterior: prior N(0, 10), likelihood N(mu, 1) with n=5
        #   Precision: 1/100 + 5 = 5.01  ->  var = 0.1996
        #   Mean:      5.0 * y_bar / 5.01
        y_bar = float(y_obs.mean())
        post_mean = 5.0 * y_bar / (1.0 / 100.0 + 5.0)
        post_sd = np.sqrt(1.0 / (1.0 / 100.0 + 5.0))
        # PyMCModel now provides an event_template (one field per PyMC RV),
        # so draws() returns a NumericRecordArray keyed by RV name. The
        # only parameter is `mu`, with event_shape ().
        draws = result.draws()
        assert draws.fields == ("mu",)
        mu_draws = jnp.asarray(draws["mu"])
        assert mu_draws.shape == (1000,)  # 2 chains × 500 draws, flattened
        # With 1000 draws total, MC SE for mean ~ post_sd / sqrt(1000) ~ 0.014
        np.testing.assert_allclose(float(jnp.mean(mu_draws)), post_mean, atol=0.05)
        np.testing.assert_allclose(float(jnp.std(mu_draws)), post_sd, atol=0.05)

    def test_auxiliary_trace_attached(self):
        model = PyMCModel(_gaussian_pymc_fn, name="gaussian")
        y_obs = np.array([0.0, 1.0], dtype=float)
        result = condition_on_nutpie._func(
            model, data={"y": y_obs},
            num_results=50, num_warmup=50, num_chains=1, random_seed=0,
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

        model = PyMCModel(model_fn, name="ordering")
        result = condition_on_nutpie._func(
            model, data={"y": np.zeros(4, dtype=float)},
            num_results=300, num_warmup=300, num_chains=1, random_seed=0,
        )
        draws = result.draws()
        assert draws.fields == ("zeta", "alpha", "mu")
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

        model = PyMCModel(model_fn, name="partial")
        result = condition_on_nutpie._func(
            model, data={"y": np.zeros(5, dtype=float)},
            num_results=200, num_warmup=200, num_chains=1, random_seed=0,
        )
        draws = result.draws()
        assert set(draws.fields) == {"mu", "X"}
        np.testing.assert_allclose(
            float(jnp.mean(jnp.asarray(draws["mu"]))), 100.0, atol=10.0
        )
        np.testing.assert_allclose(
            float(jnp.mean(jnp.asarray(draws["X"]))), -100.0, atol=10.0
        )
