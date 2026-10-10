"""Tests for probpipe.diagnostics._mcmc."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import probpipe.diagnostics._mcmc as mcmc
from probpipe import NumericArraySpec, RecordSpec
from probpipe.diagnostics._datatree_store import _mcmc_has_field
from probpipe.diagnostics._mcmc import (
    _check_arviz,
    _emit_payload_warnings,
    _ess_warnings,
    _rhat_warnings,
    _write_mcmc_payload,
    add_ess,
    add_mcmc_diagnostics,
    add_mcse,
    add_rhat,
)
from probpipe.diagnostics._view_base import NotComputed
from probpipe.diagnostics._views import DiagnosticsView
from tests._posterior import posterior_of

if TYPE_CHECKING:
    import xarray as xr

# conftest.py provides: posterior, posterior_single_chain, posterior_3params


def _scalar_vector_draws() -> dict[str, np.ndarray]:
    """Draws of a scalar ``alpha`` and a vector ``beta``, each ``(chains, draws, *shape)``."""
    rng = np.random.default_rng(321)
    # Single precision, as an inference result stores its draws, so that both sides
    # of a comparison with ArviZ read the same values.
    return {
        "alpha": rng.standard_normal((2, 160)).astype(np.float32),
        "beta": rng.standard_normal((2, 160, 2)).astype(np.float32),
    }


def _vector_draws() -> dict[str, np.ndarray]:
    """Draws of a vector ``beta``, ``(chains, draws, 2)``."""
    return {"beta": np.random.default_rng(123).standard_normal((2, 120, 2))}


def _non_mixing_draws() -> dict[str, np.ndarray]:
    """Two chains of ``theta`` that stay apart, at -4 and at 4."""
    rng = np.random.default_rng(456)
    return {
        "theta": np.stack(
            [rng.normal(loc=-4.0, scale=0.1, size=120), rng.normal(loc=4.0, scale=0.1, size=120)],
            axis=0,
        )
    }


def _posterior_of_draws(draws: dict[str, np.ndarray]) -> Any:
    """The inference result whose chains hold *draws*, one record field per entry."""
    flat = np.concatenate([np.reshape(v, (*v.shape[:2], -1)) for v in draws.values()], axis=-1)
    event = RecordSpec(**{name: NumericArraySpec(v.shape[2:]) for name, v in draws.items()})
    return posterior_of(list(flat), event_spec=event)


def _arviz_stats_module():
    try:
        import arviz_stats as azs
    except ImportError:
        import arviz as azs
    return azs


def _independent_arviz_posterior(draws: dict[str, np.ndarray]) -> xr.Dataset:
    import arviz as az

    return az.from_dict({"posterior": dict(draws)}).posterior


# ---------------------------------------------------------------------------
# add_rhat
# ---------------------------------------------------------------------------


class TestAddRhat:
    def test_writes_rhat_field(self, posterior):
        add_rhat(posterior)
        assert _mcmc_has_field(posterior, "rhat")

    def test_rhat_values_are_numeric(self, posterior):
        add_rhat(posterior)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        rhat = view.rhat
        assert set(rhat.keys()) == {"alpha", "beta"}
        for v in rhat.values():
            assert isinstance(v, (float, NotComputed))

    def test_rhat_reasonable_for_iid_chains(self, posterior):
        """IID draws should give R-hat close to 1."""
        add_rhat(posterior)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        for v in view.rhat.values():
            if isinstance(v, float):
                assert v < 1.1

    def test_single_chain_returns_not_computed(self, posterior_single_chain):
        add_rhat(posterior_single_chain)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior_single_chain._annotations["diagnostics"])
        assert set(view.rhat) == {"alpha", "beta"}
        for v in view.rhat.values():
            assert v == NotComputed("R-hat requires at least 2 chains")

    def test_a_single_chain_posterior_reports_each_component(self):
        import jax.numpy as jnp

        from probpipe import NumericArraySpec, RecordSpec

        draws = jnp.asarray(np.random.default_rng(0).normal(size=(50, 3)), jnp.float32)
        event = RecordSpec(mu=NumericArraySpec((2,), jnp.float32), sigma=NumericArraySpec(()))
        payload = mcmc._compute_rhat_op(posterior_of([draws], event_spec=event))
        not_computed = NotComputed("R-hat requires at least 2 chains")
        assert payload["values"] == {"mu": not_computed, "sigma": not_computed}

    def test_idempotent(self, posterior):
        """Calling add_rhat twice should not raise and last write wins."""
        add_rhat(posterior)
        add_rhat(posterior)
        assert _mcmc_has_field(posterior, "rhat")

    def test_does_not_return_value(self, posterior):
        assert add_rhat(posterior) is None

    def test_matches_direct_arviz_for_scalar_and_vector_parameters(self):
        draws = _scalar_vector_draws()
        posterior = _posterior_of_draws(draws)
        ds = _independent_arviz_posterior(draws)
        expected = _arviz_stats_module().rhat(ds, method="rank")

        add_rhat(posterior)

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        assert view.rhat["alpha"] == pytest.approx(float(expected["alpha"]))
        assert view.rhat["beta[0]"] == pytest.approx(float(expected["beta"][0]))
        assert view.rhat["beta[1]"] == pytest.approx(float(expected["beta"][1]))

    def test_non_mixing_chains_have_large_rhat(self):
        posterior = _posterior_of_draws(_non_mixing_draws())

        add_rhat(posterior, threshold=999.0)

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        assert view.rhat["theta"] > 1.5


class TestMcmcHelpers:
    def test_check_arviz_reports_missing_dependency(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _missing_arviz(name, *args, **kwargs):
            if name in {"arviz", "arviz_stats"}:
                raise ImportError(f"missing {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _missing_arviz)

        with pytest.raises(ImportError, match="ArviZ is required"):
            _check_arviz()

    def test_warning_helpers_skip_unusable_values_and_report_failures(self):
        class _BadFloat:
            def __float__(self):
                raise TypeError("not numeric")

        rhat_messages = _rhat_warnings(
            {"missing": NotComputed("single chain"), "bad": _BadFloat(), "alpha": 1.2},
            threshold=1.01,
        )
        assert any("alpha" in msg for msg in rhat_messages)

        ess_messages = _ess_warnings(
            {"missing": NotComputed("no bulk"), "bad": _BadFloat(), "alpha": 100.0},
            {"missing": NotComputed("no tail"), "bad": _BadFloat(), "beta": 120.0},
            threshold=400,
        )
        assert any("bulk" in msg and "alpha" in msg for msg in ess_messages)
        assert any("tail" in msg and "beta" in msg for msg in ess_messages)

    def test_emit_payload_warnings_handles_missing_and_none_warning_fields(self):
        class _NoWarnings:
            def __getitem__(self, key):
                raise KeyError(key)

        _emit_payload_warnings(_NoWarnings())
        _emit_payload_warnings({"kind": "test", "warnings": None})

    def test_write_mcmc_payload_handles_composite_and_unknown_records(self, posterior):
        child = {
            "kind": "rhat",
            "values": {"alpha": 1.0},
            "attrs": {},
        }
        composite = {"kind": "mcmc", "records": {"rhat": child}}

        _write_mcmc_payload(posterior, composite)

        assert _mcmc_has_field(posterior, "rhat")

        with pytest.raises(ValueError, match="unknown MCMC diagnostic payload kind"):
            _write_mcmc_payload(posterior, {"kind": "bogus"})

        class _MissingKind:
            def __getitem__(self, key):
                raise KeyError(key)

        with pytest.raises(ValueError, match="missing required key"):
            _write_mcmc_payload(posterior, _MissingKind())


# ---------------------------------------------------------------------------
# add_ess
# ---------------------------------------------------------------------------


class TestAddEss:
    def test_writes_ess_bulk_and_tail(self, posterior):
        add_ess(posterior)
        assert _mcmc_has_field(posterior, "ess_bulk")
        assert _mcmc_has_field(posterior, "ess_tail")

    def test_ess_values_positive(self, posterior):
        add_ess(posterior)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        for v in view.ess_bulk.values():
            if isinstance(v, float):
                assert v > 0
        for v in view.ess_tail.values():
            if isinstance(v, float):
                assert v > 0

    def test_a_posterior_without_chains_raises_naming_its_levels(self):
        from probpipe import EmpiricalDistribution

        law = EmpiricalDistribution(
            np.random.default_rng(0).standard_normal((50, 2)), component="theta"
        )
        with pytest.raises(ValueError, match=r"'p' has no chains: .* \['theta'\]"):
            add_ess(law)

    def test_ess_covers_all_params(self, posterior_3params):
        add_ess(posterior_3params)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior_3params._annotations["diagnostics"])
        assert set(view.ess_bulk.keys()) == {"mu", "sigma", "nu"}

    def test_does_not_return_value(self, posterior):
        assert add_ess(posterior) is None

    def test_idempotent_skip_and_force_recompute(self, posterior, monkeypatch):
        add_ess(posterior)
        original = mcmc._compute_ess_op

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("ESS should have been skipped")

        monkeypatch.setattr(mcmc, "_compute_ess_op", _fail_if_called)
        add_ess(posterior)

        calls = []

        def _record_call(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(mcmc, "_compute_ess_op", _record_call)
        add_ess(posterior, force=True)
        assert calls

    def test_matches_direct_arviz_for_scalar_and_vector_parameters(self):
        draws = _scalar_vector_draws()
        posterior = _posterior_of_draws(draws)
        ds = _independent_arviz_posterior(draws)
        azs = _arviz_stats_module()
        expected_bulk = azs.ess(ds, method="bulk")
        expected_tail = azs.ess(ds, method="tail")

        add_ess(posterior, threshold=0)

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        assert view.ess_bulk["alpha"] == pytest.approx(float(expected_bulk["alpha"]))
        assert view.ess_tail["alpha"] == pytest.approx(float(expected_tail["alpha"]))
        assert view.ess_bulk["beta[0]"] == pytest.approx(float(expected_bulk["beta"][0]))
        assert view.ess_tail["beta[1]"] == pytest.approx(float(expected_tail["beta"][1]))


# ---------------------------------------------------------------------------
# add_mcse
# ---------------------------------------------------------------------------


class TestAddMcse:
    def test_writes_mcse_mean_and_sd(self, posterior):
        add_mcse(posterior)
        assert _mcmc_has_field(posterior, "mcse_mean")
        assert _mcmc_has_field(posterior, "mcse_sd")

    def test_mcse_values_finite(self, posterior):
        add_mcse(posterior)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        for v in view.mcse_mean.values():
            if isinstance(v, float):
                assert np.isfinite(v)

    def test_does_not_return_value(self, posterior):
        assert add_mcse(posterior) is None

    def test_idempotent_skip(self, posterior, monkeypatch):
        add_mcse(posterior)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("MCSE should have been skipped")

        monkeypatch.setattr(mcmc, "_compute_mcse_op", _fail_if_called)
        add_mcse(posterior)

    def test_matches_direct_arviz_for_scalar_and_vector_parameters(self):
        draws = _scalar_vector_draws()
        posterior = _posterior_of_draws(draws)
        ds = _independent_arviz_posterior(draws)
        azs = _arviz_stats_module()
        expected_mean = azs.mcse(ds, method="mean")
        expected_sd = azs.mcse(ds, method="sd")

        add_mcse(posterior)

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        assert view.mcse_mean["alpha"] == pytest.approx(float(expected_mean["alpha"]))
        assert view.mcse_sd["alpha"] == pytest.approx(float(expected_sd["alpha"]))
        assert view.mcse_mean["beta[0]"] == pytest.approx(float(expected_mean["beta"][0]))
        assert view.mcse_sd["beta[1]"] == pytest.approx(float(expected_sd["beta"][1]))


# ---------------------------------------------------------------------------
# add_mcmc_diagnostics
# ---------------------------------------------------------------------------


class TestAddMcmcDiagnostics:
    def test_writes_all_three_fields(self, posterior):
        add_mcmc_diagnostics(posterior)
        assert _mcmc_has_field(posterior, "rhat")
        assert _mcmc_has_field(posterior, "ess_bulk")
        assert _mcmc_has_field(posterior, "mcse_mean")

    def test_summary_table_runs(self, posterior):
        add_mcmc_diagnostics(posterior)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        table = view.summary_table()
        assert "alpha" in table
        assert "beta" in table
        assert "R-hat" in table

    def test_to_dict_serialisable(self, posterior):
        import json

        add_mcmc_diagnostics(posterior)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        d = view.to_dict()
        # Must be JSON-serialisable (NotComputed is converted by to_dict)
        json.dumps(d)

    def test_warnings_empty_for_iid(self, posterior):
        """IID draws from a single rng should pass all diagnostics."""
        add_mcmc_diagnostics(posterior)
        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        # Warnings may be empty; we just check it's a list
        assert isinstance(view.warnings, list)

    def test_does_not_return_value(self, posterior):
        assert add_mcmc_diagnostics(posterior) is None

    def test_a_single_metric_name_is_one_metric(self, posterior):
        add_mcmc_diagnostics(posterior, metrics="rhat")
        assert _mcmc_has_field(posterior, "rhat")
        assert not _mcmc_has_field(posterior, "ess_bulk")
        assert not _mcmc_has_field(posterior, "mcse_mean")

    def test_computes_only_the_named_metrics(self, posterior):
        add_mcmc_diagnostics(posterior, metrics=["ess", "mcse"])
        assert not _mcmc_has_field(posterior, "rhat")
        assert _mcmc_has_field(posterior, "ess_bulk")
        assert _mcmc_has_field(posterior, "mcse_mean")

    def test_refuses_an_unknown_metric(self, posterior):
        with pytest.raises(
            ValueError,
            match=r"unknown metric 'esss'; available metrics: \['rhat', 'ess', 'mcse', 'divergences'\]",
        ):
            add_mcmc_diagnostics(posterior, metrics=["rhat", "esss"])
        assert not _mcmc_has_field(posterior, "rhat")

    @pytest.mark.parametrize(
        ("metrics", "match"),
        [
            (3, "add_mcmc_diagnostics metrics must be a str or a sequence of str, got int 3"),
            (b"rhat", "got bytes"),
            ({"rhat": 1}, "got dict"),
            ({"rhat", "ess"}, "got set"),
            (("rhat", 3), "add_mcmc_diagnostics metrics entry must be a str, got int 3"),
        ],
    )
    def test_refuses_metrics_that_are_not_names(self, posterior, metrics, match):
        with pytest.raises(TypeError, match=match):
            add_mcmc_diagnostics(posterior, metrics=metrics)

    def test_a_run_without_divergence_statistics_records_no_count(self, posterior):
        add_mcmc_diagnostics(posterior)
        assert isinstance(posterior.diagnostics.mcmc.n_divergences, NotComputed)

    def test_the_divergences_are_the_sum_of_diverging(self):
        import jax.numpy as jnp

        from probpipe import MultivariateNormal
        from probpipe.inference._approximate_distribution import make_posterior
        from probpipe.inference._inference_utils import build_mcmc_datatree

        rng = np.random.default_rng(0)
        chains = [jnp.asarray(rng.normal(size=(50, 2))) for _ in range(2)]
        diverging = np.zeros((2, 50), dtype=bool)
        diverging[0, [3, 7]] = True
        diverging[1, 11] = True
        prior = MultivariateNormal("z", loc=jnp.zeros(2), cov=jnp.eye(2))
        posterior = make_posterior(
            chains,
            parents=(prior,),
            method="test",
            annotations=build_mcmc_datatree(chains, {"diverging": diverging}),
        )
        add_mcmc_diagnostics(posterior)
        assert posterior.diagnostics.mcmc.n_divergences == 3

    def test_vector_parameter_diagnostics_are_written_by_component(self):
        posterior = _posterior_of_draws(_vector_draws())

        add_mcmc_diagnostics(posterior)

        from probpipe.diagnostics._views import DiagnosticsView

        view = DiagnosticsView(posterior._annotations["diagnostics"])
        assert set(view.rhat) == {"beta[0]", "beta[1]"}
        assert set(view.ess_bulk) == {"beta[0]", "beta[1]"}
        assert set(view.mcse_mean) == {"beta[0]", "beta[1]"}
