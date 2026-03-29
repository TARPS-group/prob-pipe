"""MCMC and inference diagnostics."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..custom_types import Array

__all__ = ["InferenceDiagnostics", "MCMCDiagnostics", "extract_arviz_diagnostics"]


class InferenceDiagnostics:
    """Diagnostics from an inference run.

    All diagnostics are stored in a single dict-like container.
    Access any diagnostic by name::

        diag["log_accept_ratio"]   # get
        diag["diverging"] = arr    # set
        "diverging" in diag        # membership
        list(diag)                 # iterate keys

    The only built-in attribute is :pyattr:`algorithm`.  Everything else
    — acceptance ratios, step sizes, divergences, tree depths, etc. —
    lives in the dict and is accessed uniformly via ``diag[key]``.

    Convenience properties :pyattr:`accept_rate` and
    :pyattr:`final_step_size` are computed from stored diagnostics.

    Parameters
    ----------
    algorithm : str
        Name of the algorithm that produced these diagnostics.
    **kwargs
        Initial diagnostics to store.  Common keys:

        - ``log_accept_ratio`` — per-sample log MH acceptance ratio
        - ``step_size`` — final adapted step size(s)
        - ``is_accepted`` — per-sample boolean accept/reject flags
        - ``diverging`` — per-sample divergence flags
        - ``tree_depth`` — per-sample tree depths
        - ``energy`` — per-sample Hamiltonian energies
    """

    def __init__(self, algorithm: str = "unknown", **kwargs: Any):
        self.algorithm: str = algorithm
        self._data: dict[str, Any] = dict(kwargs)

    # -- dict-style access ----------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Get a diagnostic by name."""
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(
                f"No diagnostic named {key!r}. "
                f"Available: {list(self._data)}"
            ) from None

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a diagnostic by name."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check whether a diagnostic exists."""
        return key in self._data

    def __iter__(self):
        """Iterate over diagnostic keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of stored diagnostics."""
        return len(self._data)

    def keys(self):
        """Return diagnostic keys."""
        return self._data.keys()

    def values(self):
        """Return diagnostic values."""
        return self._data.values()

    def items(self):
        """Return diagnostic (key, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a diagnostic, returning *default* if absent."""
        return self._data.get(key, default)

    # -- convenience properties -----------------------------------------------

    @property
    def accept_rate(self) -> float:
        """Empirical acceptance rate.

        Uses ``_accept_rate_override`` if set, otherwise computes from
        ``is_accepted`` or ``log_accept_ratio``.
        """
        if "_accept_rate_override" in self._data:
            return float(self._data["_accept_rate_override"])
        is_accepted = self._data.get("is_accepted")
        if is_accepted is not None:
            return float(jnp.mean(jnp.asarray(is_accepted)))
        log_ar = self._data.get("log_accept_ratio")
        if log_ar is not None:
            return float(jnp.mean(jnp.exp(jnp.minimum(jnp.asarray(log_ar), 0.0))))
        return float("nan")

    @property
    def final_step_size(self) -> float:
        """Mean final adapted step size."""
        ss = self._data.get("step_size")
        if ss is None:
            return float("nan")
        return float(jnp.mean(jnp.asarray(ss)))

    def summary(self) -> str:
        """One-line summary of diagnostics."""
        parts = [f"algorithm={self.algorithm}"]
        # Include accept_rate and step_size if available
        if "log_accept_ratio" in self._data or "is_accepted" in self._data:
            parts.append(f"accept_rate={self.accept_rate:.3f}")
        if "step_size" in self._data:
            parts.append(f"final_step_size={self.final_step_size:.4f}")
        # Remaining keys (skip internal/already-shown keys)
        _skip = {"log_accept_ratio", "is_accepted", "step_size",
                 "_accept_rate_override"}
        for key, val in self._data.items():
            if key in _skip:
                continue
            if isinstance(val, (int, float)):
                parts.append(f"{key}={val}")
            elif hasattr(val, "shape"):
                parts.append(f"{key}=Array{tuple(val.shape)}")
            else:
                parts.append(f"{key}={type(val).__name__}")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return f"InferenceDiagnostics({self.summary()})"


# Backward-compatible alias
MCMCDiagnostics = InferenceDiagnostics


def extract_arviz_diagnostics(
    trace: Any,
    algorithm: str,
    num_results: int,
    num_chains: int,
) -> InferenceDiagnostics:
    """Extract diagnostics from an ArviZ ``InferenceData`` trace.

    Works with traces produced by nutpie, CmdStanPy, and PyMC — all of
    which store sampler statistics in ``trace.sample_stats``.

    Parameters
    ----------
    trace : InferenceData
        ArviZ trace object with an optional ``.sample_stats`` group.
    algorithm : str
        Algorithm name to record (e.g. ``"nutpie_nuts"``, ``"cmdstan_nuts"``).
    num_results : int
        Number of post-warmup draws per chain.
    num_chains : int
        Number of chains.

    Returns
    -------
    InferenceDiagnostics
        Populated diagnostics with any available fields.
    """
    n_total = num_results * num_chains

    stats = getattr(trace, "sample_stats", None)

    if stats is None:
        return InferenceDiagnostics(
            algorithm=algorithm,
            log_accept_ratio=jnp.zeros(n_total),
            step_size=0.0,
        )

    kwargs: dict[str, Any] = {}

    # -- Acceptance rate / log acceptance ratio --------------------------------
    # CmdStanPy/nutpie use "acceptance_rate"; PyMC uses "mean_tree_accept".
    accept_rate = None
    for ar_key in ("acceptance_rate", "mean_tree_accept"):
        if ar_key in stats:
            accept_rate = jnp.asarray(stats[ar_key].values).reshape(-1)
            kwargs["log_accept_ratio"] = jnp.log(
                jnp.clip(accept_rate, 1e-10, 1.0)
            )
            break
    else:
        kwargs["log_accept_ratio"] = jnp.zeros(n_total)

    # -- Step size -------------------------------------------------------------
    if "step_size" in stats:
        kwargs["step_size"] = jnp.asarray(stats["step_size"].values).reshape(-1)
    elif "step_size_bar" in stats:
        kwargs["step_size"] = jnp.asarray(stats["step_size_bar"].values).reshape(-1)

    # -- Extra diagnostics ----------------------------------------------------
    _EXTRA_FIELDS = {
        "diverging": jnp.bool_,
        "tree_depth": None,
        "max_tree_depth": None,
        "n_steps": None,
        "energy": None,
        "energy_error": None,
        "lp": None,
    }

    for field_name, dtype in _EXTRA_FIELDS.items():
        if field_name in stats:
            arr = stats[field_name].values
            if dtype is not None:
                kwargs[field_name] = jnp.asarray(arr, dtype=dtype).reshape(-1)
            else:
                kwargs[field_name] = jnp.asarray(arr).reshape(-1)

    # Convenience: count divergences
    if "diverging" in kwargs:
        kwargs["n_divergences"] = int(jnp.sum(kwargs["diverging"]))

    diag = InferenceDiagnostics(algorithm=algorithm, **kwargs)

    if accept_rate is not None:
        diag["_accept_rate_override"] = float(jnp.mean(accept_rate))

    return diag
