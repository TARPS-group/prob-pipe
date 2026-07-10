"""Tests for probpipe.diagnostics._sensitivity."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from probpipe.diagnostics._datatree_store import _add_group
from probpipe.diagnostics._sensitivity import (
    _compute_log_prior,
    _dataset_to_param_dict,
    add_sensitivity,
)
from probpipe.diagnostics._view_base import NotComputed
from probpipe.diagnostics._views import SensitivityView

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRecord(dict):
    """Dict subclass that also supports attribute-style and .fields access."""

    @property
    def fields(self):
        return list(self.keys())

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _StandardNormalPrior:
    """Minimal SupportsLogProb stub: independent standard-normal log-density.

    Sums ``-0.5 * x**2 - 0.5 * log(2*pi)`` over every leaf in the value —
    the closed-form log-density of an independent N(0, 1) prior on each
    field. Used instead of a real ProbPipe distribution so tests stay fast
    and dependency-light; the independent-baseline test below recomputes
    this same closed form directly with NumPy, not by calling this stub.
    """

    def _log_prob(self, value):
        total = 0.0
        for v in value.values():
            arr = np.asarray(v, dtype=float)
            total += float(np.sum(-0.5 * arr**2 - 0.5 * np.log(2 * np.pi)))
        return total

    _unnormalized_log_prob = _log_prob


class _FakePosterior:
    """Minimal ApproximateDistribution stand-in for sensitivity tests.

    Parameters
    ----------
    draws : np.ndarray, shape (n_chain, n_draw)
        Draws for the single scalar parameter ``theta``.
    with_arviz_posterior : bool
        If True, writes the ``arviz/posterior`` group so ``add_sensitivity``
        finds posterior draws without extra setup.
    """

    def __init__(self, draws: np.ndarray, with_arviz_posterior: bool = True):
        self._theta = np.asarray(draws, dtype=float)
        self._auxiliary = None
        if with_arviz_posterior:
            ds = xr.Dataset({"theta": xr.DataArray(self._theta, dims=["chain", "draw"])})
            _add_group(self, "arviz/posterior", ds)

    @property
    def fields(self) -> list[str]:
        return ["theta"]

    @property
    def num_chains(self) -> int:
        return self._theta.shape[0]

    def draws(self, *, chain: int | None = None) -> _FakeRecord:
        if chain is None:
            return _FakeRecord({"theta": self._theta.ravel()})
        return _FakeRecord({"theta": self._theta[chain]})


# ---------------------------------------------------------------------------
# _compute_log_prior
# ---------------------------------------------------------------------------


class TestComputeLogPrior:
    def test_shape_matches_chain_and_draw(self):
        rng = np.random.default_rng(0)
        draws = rng.standard_normal((2, 50))
        post = _FakePosterior(draws)
        log_prior = _compute_log_prior(post, _StandardNormalPrior())
        assert log_prior.shape == (2, 50)

    def test_values_match_closed_form(self):
        rng = np.random.default_rng(1)
        draws = rng.standard_normal((3, 20))
        post = _FakePosterior(draws)
        log_prior = _compute_log_prior(post, _StandardNormalPrior())
        expected = -0.5 * draws**2 - 0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(log_prior, expected)


# ---------------------------------------------------------------------------
# _dataset_to_param_dict
# ---------------------------------------------------------------------------


class TestDatasetToParamDict:
    def test_scalar_variables(self):
        ds = xr.Dataset({"mu": xr.DataArray(0.2), "sigma": xr.DataArray(0.05)})
        result = _dataset_to_param_dict(ds)
        assert result == {"mu": pytest.approx(0.2), "sigma": pytest.approx(0.05)}

    def test_vector_variable_uses_component_names(self):
        ds = xr.Dataset({"beta": xr.DataArray([0.1, 0.2], dims=["beta_dim_0"])})
        result = _dataset_to_param_dict(ds)
        assert result == {"beta[0]": pytest.approx(0.1), "beta[1]": pytest.approx(0.2)}


# ---------------------------------------------------------------------------
# add_sensitivity
# ---------------------------------------------------------------------------


class TestAddSensitivity:
    def _posterior(self, n_chain=2, n_draw=200, seed=0):
        rng = np.random.default_rng(seed)
        draws = rng.standard_normal((n_chain, n_draw))
        return _FakePosterior(draws)

    def test_writes_sensitivity_group_prior_only(self):
        post = self._posterior()
        add_sensitivity(post, _StandardNormalPrior())
        ds = post._auxiliary["diagnostics"]["runs"]["sensitivity"].to_dataset()
        assert "prior_sensitivity" in ds.data_vars
        assert "theta" in ds["prior_sensitivity"].coords["param"].values

    def test_prior_only_has_no_likelihood_sensitivity(self):
        post = self._posterior()
        add_sensitivity(post, _StandardNormalPrior())
        view = SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"])
        assert view.has_likelihood is False
        # NaN is read back as NotComputed, matching every other view's convention
        # (e.g. LOOView.pareto_k_max when LOO hasn't run).
        assert isinstance(view.likelihood_sensitivity["theta"], NotComputed)

    def test_with_likelihood_writes_both_sensitivities(self):
        rng = np.random.default_rng(2)
        post = self._posterior(seed=2)
        log_likelihood = rng.normal(loc=-1.0, scale=0.1, size=(2, 200, 5))
        add_sensitivity(post, _StandardNormalPrior(), log_likelihood=log_likelihood)
        view = SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"])
        assert view.has_likelihood is True
        assert not np.isnan(view.prior_sensitivity["theta"])
        assert not np.isnan(view.likelihood_sensitivity["theta"])

    def test_log_likelihood_is_explicit_only_no_fallback(self):
        """A log_likelihood group left by an earlier call must not be reused."""
        rng = np.random.default_rng(3)
        post = self._posterior(seed=3)
        log_likelihood = rng.normal(size=(2, 200, 4))

        add_sensitivity(post, _StandardNormalPrior(), log_likelihood=log_likelihood, force=True)
        assert SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"]).has_likelihood

        # Second call omits log_likelihood; must not silently reuse the group
        # written above even though it's still present in the arviz tree.
        add_sensitivity(post, _StandardNormalPrior(), force=True)
        view = SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"])
        assert view.has_likelihood is False

    def test_returns_none(self):
        post = self._posterior()
        assert add_sensitivity(post, _StandardNormalPrior()) is None

    def test_force_false_skips_if_exists(self):
        post = self._posterior()
        add_sensitivity(post, _StandardNormalPrior())
        first = (
            post._auxiliary["diagnostics"]["runs"]["sensitivity"].to_dataset().attrs["timestamp"]
        )
        add_sensitivity(post, _StandardNormalPrior(), force=False)
        second = (
            post._auxiliary["diagnostics"]["runs"]["sensitivity"].to_dataset().attrs["timestamp"]
        )
        assert first == second

    def test_force_true_recomputes(self):
        post = self._posterior()
        add_sensitivity(post, _StandardNormalPrior())
        first = (
            post._auxiliary["diagnostics"]["runs"]["sensitivity"].to_dataset().attrs["timestamp"]
        )
        add_sensitivity(post, _StandardNormalPrior(), force=True)
        second = (
            post._auxiliary["diagnostics"]["runs"]["sensitivity"].to_dataset().attrs["timestamp"]
        )
        assert first <= second

    def test_raises_without_posterior_group(self):
        post = _FakePosterior(np.zeros((2, 10)), with_arviz_posterior=False)
        with pytest.raises(ValueError, match="No ArviZ-compatible"):
            add_sensitivity(post, _StandardNormalPrior())

    def test_diagnosis_flags_prior_data_conflict(self):
        """A posterior shifted far from the prior, with a tight likelihood,
        should be flagged as prior-sensitive."""
        rng = np.random.default_rng(4)
        draws = rng.normal(loc=5.0, scale=0.3, size=(2, 300))
        post = _FakePosterior(draws)
        log_likelihood = rng.normal(loc=-1.0, scale=0.05, size=(2, 300, 8))
        add_sensitivity(post, _StandardNormalPrior(), log_likelihood=log_likelihood)
        view = SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"])
        assert view.prior_sensitivity["theta"] >= view.threshold

    def test_matches_direct_arviz_psense(self):
        """Independent baseline per STYLE_GUIDE Sec 8.6: build the expected
        result via a fresh az.from_dict + az.psense_summary call, not by
        reusing add_sensitivity's own dataset-construction path."""
        import arviz as az

        rng = np.random.default_rng(123)
        posterior_draws = rng.normal(size=(2, 150))
        log_prior = -0.5 * posterior_draws**2 - 0.5 * np.log(2 * np.pi)
        log_likelihood = rng.normal(loc=-1.0, scale=0.1, size=(2, 150, 6))

        expected = az.psense_summary(
            az.from_dict(
                {
                    "posterior": {"theta": posterior_draws},
                    "log_prior": {"log_prior": log_prior},
                    "log_likelihood": {"y": log_likelihood},
                }
            ),
        )

        post = _FakePosterior(posterior_draws)
        add_sensitivity(post, _StandardNormalPrior(), log_likelihood=log_likelihood)

        view = SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"])
        assert view.prior_sensitivity["theta"] == pytest.approx(
            float(expected.loc["theta", "prior"]), abs=1e-3
        )
        assert view.likelihood_sensitivity["theta"] == pytest.approx(
            float(expected.loc["theta", "likelihood"]), abs=1e-3
        )


# ---------------------------------------------------------------------------
# SensitivityView
# ---------------------------------------------------------------------------


class TestSensitivityView:
    def test_not_computed_repr(self):
        view = SensitivityView(None)
        assert view.exists is False
        assert repr(view) == "SensitivityView(not computed)"

    def test_repr_after_computed(self):
        post = _FakePosterior(np.random.default_rng(0).standard_normal((2, 50)))
        add_sensitivity(post, _StandardNormalPrior())
        view = SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"])
        assert "theta" in repr(view)

    def test_warnings_empty_when_not_sensitive(self):
        # Built directly rather than from add_sensitivity() on random draws:
        # power-scaling sensitivity has a nonzero noise floor even for a
        # well-matched prior/posterior at finite sample sizes, so asserting
        # "no warnings" on sampled data would be statistically fragile.
        ds = xr.Dataset(
            {
                "prior_sensitivity": xr.DataArray(
                    [0.01], dims=["param"], coords={"param": ["theta"]}
                ),
                "likelihood_sensitivity": xr.DataArray(
                    [0.01], dims=["param"], coords={"param": ["theta"]}
                ),
            }
        )
        ds.attrs["diagnosis_json"] = '{"theta": "✓"}'
        ds.attrs["threshold"] = 0.05
        view = SensitivityView(xr.DataTree(dataset=ds))
        assert view.warnings == []

    def test_warnings_populated_when_prior_sensitive(self):
        rng = np.random.default_rng(4)
        draws = rng.normal(loc=5.0, scale=0.3, size=(2, 300))
        post = _FakePosterior(draws)
        add_sensitivity(post, _StandardNormalPrior())
        view = SensitivityView(post._auxiliary["diagnostics"]["runs"]["sensitivity"])
        assert any("theta" in w for w in view.warnings)
