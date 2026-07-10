"""Concrete ProbPipe diagnostic view classes.

These classes are domain-specific views built on top of the generic
DataTree view base classes from ``_view_base.py``.
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from ._view_base import (
    DatasetView,
    DataTreeView,
    DiagnosticRunView,
    NotComputed,
    read_json_attr,
)

__all__ = [
    "DiagnosticRunView",
    "DiagnosticsView",
    "LOOView",
    "MCMCView",
    "NotComputed",
    "PPCView",
    "SensitivityView",
]


# ---------------------------------------------------------------------------
# MCMCView
# ---------------------------------------------------------------------------


class MCMCView(DatasetView):
    """Structured read-only accessor for ``/diagnostics/mcmc``."""

    @property
    def rhat(self) -> dict[str, float | NotComputed]:
        return self.indexed("rhat", dim="param")

    @property
    def ess_bulk(self) -> dict[str, float | NotComputed]:
        return self.indexed("ess_bulk", dim="param")

    @property
    def ess_tail(self) -> dict[str, float | NotComputed]:
        return self.indexed("ess_tail", dim="param")

    @property
    def mcse_mean(self) -> dict[str, float | NotComputed]:
        return self.indexed("mcse_mean", dim="param")

    @property
    def mcse_sd(self) -> dict[str, float | NotComputed]:
        return self.indexed("mcse_sd", dim="param")

    @property
    def n_divergences(self) -> int | NotComputed:
        ds = self.dataset()
        if ds is None:
            return NotComputed("mcmc diagnostics not yet computed")

        raw = ds.attrs.get("n_divergences")

        if raw is None:
            return NotComputed("not recorded by this backend")

        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict) and "not_computed" in parsed:
                return NotComputed(parsed["not_computed"])
            return int(parsed)
        except Exception:
            try:
                return int(raw)
            except Exception:
                return NotComputed("could not parse n_divergences")

    @property
    def warnings(self) -> list[str]:
        ds = self.dataset()
        if ds is None:
            return []

        msgs: list[str] = []

        for key in ("rhat_warnings", "ess_warnings", "mcse_warnings"):
            raw = ds.attrs.get(key, "[]")
            with suppress(Exception):
                msgs.extend(json.loads(raw))

        return msgs

    def __repr__(self) -> str:
        params = list(self.rhat.keys()) or list(self.ess_bulk.keys())
        return f"MCMCView(params={params}, warnings={len(self.warnings)})"


# ---------------------------------------------------------------------------
# PPCView
# ---------------------------------------------------------------------------


class PPCView(DatasetView):
    """Structured read-only accessor for ``/diagnostics/runs/ppc``."""

    @property
    def p_values(self) -> dict[str, float | NotComputed]:
        return self.indexed("p_value", dim="test_fn")

    @property
    def observed(self) -> dict[str, float | NotComputed]:
        # Current writer uses "observed".
        out = self.indexed("observed", dim="test_fn")
        if out:
            return out

        # Backward compatibility with older dataset variable name.
        return self.indexed("observed_statistic", dim="test_fn")

    @property
    def replicated_stat_mean(self) -> dict[str, float | NotComputed]:
        return self.indexed("replicated_stat_mean", dim="test_fn")

    @property
    def replicated_stat_sd(self) -> dict[str, float | NotComputed]:
        return self.indexed("replicated_stat_sd", dim="test_fn")

    @property
    def result(self) -> dict[str, dict[str, float | NotComputed]]:
        """PPC result keyed by test function.

        Example
        -------
        {
            "var_mean_ratio": {
                "p_value": 0.43,
                "observed": 3.2,
            }
        }
        """
        p_vals = self.p_values
        obs = self.observed
        keys = set(p_vals) | set(obs)

        return {
            key: {
                "p_value": p_vals.get(key, NotComputed("missing p_value")),
                "observed": obs.get(key, NotComputed("missing observed")),
            }
            for key in sorted(keys)
        }

    @property
    def plot_ready(self) -> bool:
        return bool(self.attr("plot_ready", False))

    @property
    def timestamp(self) -> str:
        return str(self.attr("timestamp", ""))

    def __repr__(self) -> str:
        if not self.exists:
            return "PPCView(not computed)"
        return f"PPCView(test_fns={list(self.p_values.keys())}, plot_ready={self.plot_ready})"


# ---------------------------------------------------------------------------
# LOOView
# ---------------------------------------------------------------------------


class LOOView(DatasetView):
    """Structured read-only accessor for ``/diagnostics/runs/loo``."""

    @property
    def elpd_loo(self) -> float | NotComputed:
        return self.scalar("elpd_loo")

    @property
    def se(self) -> float | NotComputed:
        return self.scalar("se")

    @property
    def p_loo(self) -> float | NotComputed:
        return self.scalar("p_loo")

    @property
    def looic(self) -> float | NotComputed:
        return self.scalar("looic")

    @property
    def pareto_k_max(self) -> float | NotComputed:
        return self.scalar("pareto_k_max")

    @property
    def pareto_k_mean(self) -> float | NotComputed:
        return self.scalar("pareto_k_mean")

    @property
    def pareto_k_bad_count(self) -> int | NotComputed:
        v = self.scalar("pareto_k_bad_count")
        return int(v) if isinstance(v, float) else v

    @property
    def warning(self) -> bool:
        v = self.scalar("warning")
        return bool(v) if not isinstance(v, NotComputed) else False

    @property
    def warnings(self) -> list[str]:
        msgs: list[str] = []

        if self.warning:
            msgs.append("ArviZ LOO reliability warning — some Pareto-k values may be too high.")

        pk_max = self.pareto_k_max
        if isinstance(pk_max, float) and pk_max > 0.7:
            msgs.append(
                f"Pareto-k max = {pk_max:.3f} > 0.7 — LOO estimate "
                "may be unreliable for some observations."
            )

        return msgs

    @property
    def plot_ready(self) -> bool:
        return bool(self.attr("plot_ready", False))

    @property
    def plot_fn(self) -> str:
        return str(self.attr("plot_fn", ""))

    def __repr__(self) -> str:
        if not self.exists:
            return "LOOView(not computed)"
        return (
            f"LOOView(elpd_loo={self.elpd_loo}, se={self.se}, "
            f"looic={self.looic}, pareto_k_max={self.pareto_k_max})"
        )


# ---------------------------------------------------------------------------
# SensitivityView
# ---------------------------------------------------------------------------


class SensitivityView(DatasetView):
    """Structured read-only accessor for ``/diagnostics/runs/sensitivity``."""

    @property
    def prior_sensitivity(self) -> dict[str, float | NotComputed]:
        return self.indexed("prior_sensitivity", dim="param")

    @property
    def likelihood_sensitivity(self) -> dict[str, float | NotComputed]:
        return self.indexed("likelihood_sensitivity", dim="param")

    @property
    def has_likelihood(self) -> bool:
        return bool(self.attr("has_likelihood", False))

    @property
    def threshold(self) -> float:
        return float(self.attr("threshold", 0.05))

    @property
    def diagnosis(self) -> dict[str, str]:
        return read_json_attr(self.attrs, "diagnosis_json", {})

    @property
    def warnings(self) -> list[str]:
        return [
            f"'{param}' shows {label} (threshold={self.threshold})."
            for param, label in self.diagnosis.items()
            if label.startswith("potential") or label == "prior sensitive"
        ]

    def __repr__(self) -> str:
        if not self.exists:
            return "SensitivityView(not computed)"
        return (
            f"SensitivityView(params={list(self.prior_sensitivity.keys())}, "
            f"has_likelihood={self.has_likelihood})"
        )


# ---------------------------------------------------------------------------
# DiagnosticsView
# ---------------------------------------------------------------------------


class DiagnosticsView(DataTreeView):
    """Structured read-only accessor over the ``/diagnostics`` subtree."""

    # ── child-node helpers -------------------------------------------------

    def _child_or_none(self, *path: str):
        node = self._tree

        if node is None:
            return None

        for part in path:
            children = getattr(node, "children", {}) or {}
            if part not in children:
                return None
            try:
                node = node[part]
            except Exception:
                return None

        return node

    # ── structured subviews ------------------------------------------------

    @property
    def mcmc(self) -> MCMCView:
        return MCMCView(self._child_or_none("mcmc"))

    @property
    def ppc(self) -> PPCView:
        return PPCView(self._child_or_none("runs", "ppc"))

    @property
    def loo(self) -> LOOView:
        return LOOView(self._child_or_none("runs", "loo"))

    @property
    def sensitivity(self) -> SensitivityView:
        return SensitivityView(self._child_or_none("runs", "sensitivity"))

    # ── convenience MCMC passthroughs -------------------------------------

    @property
    def rhat(self):
        return self.mcmc.rhat

    @property
    def ess_bulk(self):
        return self.mcmc.ess_bulk

    @property
    def ess_tail(self):
        return self.mcmc.ess_tail

    @property
    def mcse_mean(self):
        return self.mcmc.mcse_mean

    @property
    def mcse_sd(self):
        return self.mcmc.mcse_sd

    @property
    def n_divergences(self):
        return self.mcmc.n_divergences

    @property
    def warnings(self) -> list[str]:
        return self.mcmc.warnings + self.loo.warnings + self.sensitivity.warnings

    # ── generic run list ---------------------------------------------------

    @property
    def runs(self) -> list[DiagnosticRunView]:
        runs_node = self._child_or_none("runs")
        if runs_node is None:
            return []

        children = getattr(runs_node, "children", {}) or {}

        return [DiagnosticRunView(name, runs_node[name]) for name in children]

    # ── summaries ----------------------------------------------------------

    def summary_table(self) -> str:
        """Pretty-printed table of MCMC metrics for notebook display."""
        rhat = self.rhat
        ess_b = self.ess_bulk
        ess_t = self.ess_tail
        mcse_m = self.mcse_mean

        params = list(rhat.keys()) or list(ess_b.keys())
        if not params:
            return "No MCMC diagnostics computed yet."

        def _fmt(d: dict, k: str, width: int) -> str:
            v = d.get(k, "—")
            if isinstance(v, NotComputed):
                s = "N/A"
            elif isinstance(v, float):
                s = f"{v:.4f}"
            else:
                s = str(v)
            return s.ljust(width)

        col_w = [14, 10, 12, 12, 12]

        header = (
            "Parameter".ljust(col_w[0])
            + "R-hat".ljust(col_w[1])
            + "ESS bulk".ljust(col_w[2])
            + "ESS tail".ljust(col_w[3])
            + "MCSE mean".ljust(col_w[4])
        )

        sep = "-" * len(header)
        rows = [header, sep]

        for p in params:
            rows.append(
                p.ljust(col_w[0])
                + _fmt(rhat, p, col_w[1])
                + _fmt(ess_b, p, col_w[2])
                + _fmt(ess_t, p, col_w[3])
                + _fmt(mcse_m, p, col_w[4])
            )

        rows.append("")

        warns = self.warnings
        if warns:
            rows.extend(f"⚠  {w}" for w in warns)
        else:
            rows.append("✓  All available diagnostics passed.")

        return "\n".join(rows)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of all diagnostics."""

        def _ser(v: Any) -> Any:
            if isinstance(v, NotComputed):
                return {"not_computed": v.reason}
            if isinstance(v, dict):
                return {k: _ser(vv) for k, vv in v.items()}
            return v

        return {
            "mcmc": {
                "rhat": _ser(self.rhat),
                "ess_bulk": _ser(self.ess_bulk),
                "ess_tail": _ser(self.ess_tail),
                "mcse_mean": _ser(self.mcse_mean),
                "mcse_sd": _ser(self.mcse_sd),
                "n_divergences": _ser(self.n_divergences),
                "warnings": self.mcmc.warnings,
            },
            "ppc": _ser(self.ppc.result),
            "loo": {
                "elpd_loo": _ser(self.loo.elpd_loo),
                "se": _ser(self.loo.se),
                "p_loo": _ser(self.loo.p_loo),
                "looic": _ser(self.loo.looic),
                "pareto_k_max": _ser(self.loo.pareto_k_max),
                "warnings": self.loo.warnings,
            },
            "sensitivity": {
                "prior_sensitivity": _ser(self.sensitivity.prior_sensitivity),
                "likelihood_sensitivity": _ser(self.sensitivity.likelihood_sensitivity),
                "diagnosis": self.sensitivity.diagnosis,
                "warnings": self.sensitivity.warnings,
            },
            "runs": [
                {
                    "name": r.name,
                    "result": _ser(r.result),
                    "plot_fn": r.plot_fn,
                    "plot_ready": r.plot_ready,
                    "plot_groups": r.plot_groups,
                    "timestamp": r.timestamp,
                }
                for r in self.runs
            ],
        }

    def __repr__(self) -> str:
        return (
            f"DiagnosticsView("
            f"params={list(self.rhat.keys())}, "
            f"warnings={len(self.warnings)}, "
            f"runs={len(self.runs)})"
        )
