"""MCMC diagnostics dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from ..custom_types import Array

__all__ = ["MCMCDiagnostics"]


@dataclass
class MCMCDiagnostics:
    """Post-sampling diagnostics from an MCMC run.

    Core fields capture the most common diagnostics.  Additional
    backend-specific diagnostics (divergences, tree depth, energy, etc.)
    are stored in :pyattr:`extra` and accessible via dict-style access::

        diag["diverging"]          # get
        diag["diverging"] = arr    # set
        "diverging" in diag        # membership
        list(diag)                 # iterate extra keys

    Parameters
    ----------
    log_accept_ratio : Array
        Per-sample log Metropolis–Hastings acceptance ratio.
    step_size : Array or float
        Final adapted step size(s).
    is_accepted : Array or None
        Per-sample boolean accept/reject flags (if available).
    algorithm : str
        Name of the algorithm that produced these diagnostics.
    extra : dict
        Additional backend-specific diagnostics.
    """

    log_accept_ratio: Array
    step_size: Array | float
    is_accepted: Array | None = None
    algorithm: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    # -- dict-style access to extra diagnostics ------------------------------

    def __getitem__(self, key: str) -> Any:
        """Get an extra diagnostic by name."""
        try:
            return self.extra[key]
        except KeyError:
            raise KeyError(
                f"No diagnostic named {key!r}. "
                f"Available extras: {list(self.extra)}"
            ) from None

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an extra diagnostic by name."""
        self.extra[key] = value

    def __contains__(self, key: str) -> bool:
        """Check whether an extra diagnostic exists."""
        return key in self.extra

    def __iter__(self):
        """Iterate over extra diagnostic keys."""
        return iter(self.extra)

    def keys(self):
        """Return extra diagnostic keys."""
        return self.extra.keys()

    def values(self):
        """Return extra diagnostic values."""
        return self.extra.values()

    def items(self):
        """Return extra diagnostic (key, value) pairs."""
        return self.extra.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get an extra diagnostic, returning *default* if absent."""
        return self.extra.get(key, default)

    @property
    def accept_rate(self) -> float:
        """Empirical acceptance rate."""
        if hasattr(self, "_numpy_accept_rate"):
            return self._numpy_accept_rate
        if self.is_accepted is not None:
            return float(jnp.mean(self.is_accepted))
        return float(
            jnp.mean(jnp.exp(jnp.minimum(self.log_accept_ratio, 0.0)))
        )

    @property
    def final_step_size(self) -> float:
        """Mean final adapted step size."""
        return float(jnp.mean(jnp.asarray(self.step_size)))

    def summary(self) -> str:
        """One-line summary of diagnostics."""
        parts = [
            f"algorithm={self.algorithm}",
            f"accept_rate={self.accept_rate:.3f}",
            f"final_step_size={self.final_step_size:.4f}",
        ]
        for key, val in self.extra.items():
            if isinstance(val, (int, float)):
                parts.append(f"{key}={val}")
            elif hasattr(val, "shape"):
                parts.append(f"{key}=Array{tuple(val.shape)}")
            else:
                parts.append(f"{key}={type(val).__name__}")
        return ", ".join(parts)
