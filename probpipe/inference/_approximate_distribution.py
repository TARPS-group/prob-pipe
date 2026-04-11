"""Approximate empirical distribution with chain structure and auxiliary DataTree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

import jax
import jax.numpy as jnp

from ..core.distribution import ArrayEmpiricalDistribution, Distribution, ValuesDistribution, _unflatten_batched
from ..core._values_distribution import _ValuesDistributionView
from ..core.provenance import Provenance
from ..core.values import Values
from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights

__all__ = ["ApproximateDistribution", "make_posterior"]


# ---------------------------------------------------------------------------
# ApproximateDistribution
# ---------------------------------------------------------------------------


class ApproximateDistribution(ArrayEmpiricalDistribution, ValuesDistribution):
    """Empirical distribution with chain structure.

    Stores per-chain sample arrays for chain-structured access via
    :meth:`draws`.  Algorithm metadata, sample statistics, warmup
    samples, and the ArviZ ``InferenceData`` object live in
    ``dist.auxiliary`` (a DataTree on the Distribution base class),
    not as attributes of this class.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each of shape ``(num_draws, *event_shape)``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Optional per-sample importance weights (across all chains).
    name : str or None
        Distribution name for provenance.
    """

    def __init__(
        self,
        chains: list[Array],
        *,
        weights: ArrayLike | Weights | None = None,
        name: str | None = None,
    ):
        if not chains:
            raise ValueError("Must provide at least one chain")

        self._chains = [jnp.asarray(c) for c in chains]
        self._concatenated: Array | None = None

        super().__init__(self._concat_chains(), weights=weights, name=name)

    def _concat_chains(self) -> Array:
        """Lazily concatenated view of all chains."""
        if self._concatenated is None:
            self._concatenated = jnp.concatenate(self._chains, axis=0)
        return self._concatenated

    # -- Chain access ---------------------------------------------------------

    @property
    def chains(self) -> list[Array]:
        """Per-chain sample arrays."""
        return self._chains

    @property
    def num_chains(self) -> int:
        """Number of chains."""
        return len(self._chains)

    @property
    def num_draws(self) -> int:
        """Number of draws per chain (assumes equal-length chains)."""
        return self._chains[0].shape[0]

    @property
    def algorithm(self) -> str:
        """Name of the inference algorithm (read from provenance)."""
        src = self.source
        if src is not None:
            return src.metadata.get("algorithm", src.operation)
        return "unknown"

    @property
    def inference_data(self) -> DataTree | None:
        """The auxiliary DataTree, for ArviZ compatibility.

        Alias for ``self.auxiliary``.  Use ArviZ functions for diagnostics::

            import arviz as az
            az.summary(posterior.inference_data)
        """
        return self.auxiliary

    @property
    def warmup_samples(self) -> list[Array] | None:
        """Per-chain warmup samples extracted from auxiliary DataTree."""
        aux = self.auxiliary
        if aux is None or "warmup" not in aux.children:
            return None
        warmup = aux["warmup"]["params"]
        n_chains = warmup.sizes.get("chain", 1)
        return [jnp.asarray(warmup.sel(chain=i).values) for i in range(n_chains)]

    def draws(
        self,
        chain: int | None = None,
        *,
        include_warmup: bool = False,
    ) -> Array | Values:
        """Access draws, optionally named via values_template.

        Parameters
        ----------
        chain : int or None
            Chain index.  If ``None``, concatenates all chains.
        include_warmup : bool
            If ``True`` and warmup samples are in the auxiliary DataTree,
            prepend them.

        Returns
        -------
        Array or Values
            If ``values_template`` is set, returns a :class:`~probpipe.Values`
            with named fields.  Otherwise returns a raw array.
        """
        if chain is not None:
            samples = self._chains[chain]
            if include_warmup:
                warmup = self.warmup_samples
                if warmup is not None:
                    samples = jnp.concatenate([warmup[chain], samples], axis=0)
        else:
            parts = list(self._chains)
            if include_warmup:
                warmup = self.warmup_samples
                if warmup is not None:
                    parts = [jnp.concatenate([w, c], axis=0)
                             for w, c in zip(warmup, parts)]
            samples = jnp.concatenate(parts, axis=0)

        if self.values_template is not None:
            return _unflatten_batched(samples, self.values_template)
        return samples

    def __repr__(self) -> str:
        return (
            f"ApproximateDistribution("
            f"algorithm={self.algorithm!r}, "
            f"num_chains={self.num_chains}, "
            f"num_draws={self.num_draws}, "
            f"event_shape={self.event_shape})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_posterior(
    chains: list[Array],
    parents: tuple[Distribution, ...],
    algorithm: str,
    *,
    auxiliary: DataTree | None = None,
    values_template: Values | None = None,
    **meta: Any,
) -> ApproximateDistribution:
    """Build an ApproximateDistribution with provenance.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each shaped ``(num_draws, *event_shape)``.
    parents : tuple of Distribution
        Parent distributions for provenance tracking.
    algorithm : str
        Inference algorithm name (e.g. ``"tfp_nuts"``, ``"rwmh"``).
    auxiliary : DataTree or None
        Pre-built auxiliary DataTree (diagnostics, sample stats, warmup).
        Inference methods are responsible for building this.
    values_template : Values or None
        If provided, ``draws()`` returns named ``Values``.
    **meta
        Additional metadata stored in provenance.

    Returns
    -------
    ApproximateDistribution
        Posterior with chain structure, auxiliary DataTree, and provenance.
    """
    result = ApproximateDistribution(chains, name="posterior")

    if auxiliary is not None:
        result._auxiliary = auxiliary

    if values_template is not None:
        result._values_template = values_template

    result.with_source(
        Provenance(algorithm, parents=parents, metadata={"algorithm": algorithm, **meta})
    )
    return result
