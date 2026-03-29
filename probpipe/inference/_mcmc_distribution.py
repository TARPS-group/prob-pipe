"""MCMC-aware empirical distribution with chain structure and diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from ..core.distribution import EmpiricalDistribution
from ..custom_types import Array
from ._diagnostics import InferenceDiagnostics

__all__ = ["MCMCApproximateDistribution"]


class MCMCApproximateDistribution(EmpiricalDistribution):
    """Empirical distribution from MCMC with chain structure and diagnostics.

    Wraps one or more MCMC chains as an
    :class:`~probpipe.core.distribution.EmpiricalDistribution` while
    preserving per-chain structure, optional warmup samples, and
    :class:`InferenceDiagnostics`.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each of shape ``(num_draws, *event_shape)``.
    diagnostics : InferenceDiagnostics or None
        Diagnostics from the MCMC run.
    warmup_samples : list of Array or None
        Per-chain warmup (burn-in) samples, same shapes as *chains*.
    weights : Array or None
        Optional per-sample importance weights (across all chains).
    name : str or None
        Distribution name for provenance.
    """

    def __init__(
        self,
        chains: list[Array],
        *,
        diagnostics: InferenceDiagnostics | None = None,
        warmup_samples: list[Array] | None = None,
        weights: Array | None = None,
        name: str | None = None,
    ):
        if not chains:
            raise ValueError("Must provide at least one chain")

        self._chains = [jnp.asarray(c) for c in chains]
        self._warmup_samples = (
            [jnp.asarray(w) for w in warmup_samples]
            if warmup_samples is not None
            else None
        )
        self._diagnostics = diagnostics

        # Concatenate all chains for the parent EmpiricalDistribution
        all_samples = jnp.concatenate(self._chains, axis=0)
        super().__init__(all_samples, weights=weights, name=name)

    # -- Chain access ---------------------------------------------------------

    @property
    def chains(self) -> list[Array]:
        """Per-chain sample arrays."""
        return self._chains

    @property
    def num_chains(self) -> int:
        """Number of MCMC chains."""
        return len(self._chains)

    @property
    def num_draws(self) -> int:
        """Number of draws per chain (assumes equal-length chains)."""
        return self._chains[0].shape[0]

    @property
    def diagnostics(self) -> InferenceDiagnostics | None:
        """MCMC diagnostics, if available."""
        return self._diagnostics

    @property
    def warmup_samples(self) -> list[Array] | None:
        """Per-chain warmup samples, if stored."""
        return self._warmup_samples

    def draws(
        self,
        chain: int | None = None,
        *,
        include_warmup: bool = False,
    ) -> Array:
        """Access draws from specific chains.

        Parameters
        ----------
        chain : int or None
            Chain index.  If ``None``, returns concatenated draws from all
            chains.
        include_warmup : bool
            If ``True`` and warmup samples are available, prepend them.

        Returns
        -------
        Array
            Sample array of shape ``(num_draws, *event_shape)`` (single chain)
            or ``(total_draws, *event_shape)`` (all chains).
        """
        if chain is not None:
            samples = self._chains[chain]
            if include_warmup and self._warmup_samples is not None:
                samples = jnp.concatenate(
                    [self._warmup_samples[chain], samples], axis=0
                )
            return samples

        # All chains
        parts = []
        if include_warmup and self._warmup_samples is not None:
            for w, c in zip(self._warmup_samples, self._chains):
                parts.append(jnp.concatenate([w, c], axis=0))
        else:
            parts = list(self._chains)
        return jnp.concatenate(parts, axis=0)

    def __repr__(self) -> str:
        diag = ""
        if self._diagnostics is not None:
            diag = f", {self._diagnostics.summary()}"
        return (
            f"MCMCApproximateDistribution("
            f"num_chains={self.num_chains}, "
            f"num_draws={self.num_draws}, "
            f"event_shape={self.event_shape}"
            f"{diag})"
        )
