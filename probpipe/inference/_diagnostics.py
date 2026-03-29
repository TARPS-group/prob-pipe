"""MCMC diagnostics dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..custom_types import Array

__all__ = ["MCMCDiagnostics"]


@dataclass
class MCMCDiagnostics:
    """Post-sampling diagnostics from an MCMC run.

    Attributes
    ----------
    log_accept_ratio : Array
        Per-sample log Metropolis-Hastings acceptance ratio.
    step_size : Array or float
        Final adapted step size(s).
    is_accepted : Array or None
        Per-sample boolean accept/reject flags (if available).
    algorithm : str
        Name of the algorithm that produced these diagnostics.
    """

    log_accept_ratio: Array
    step_size: Array | float
    is_accepted: Array | None = None
    algorithm: str = "unknown"

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
        return (
            f"algorithm={self.algorithm}, "
            f"accept_rate={self.accept_rate:.3f}, "
            f"final_step_size={self.final_step_size:.4f}"
        )
