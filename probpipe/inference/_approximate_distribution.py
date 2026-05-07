"""Approximate empirical distribution with chain structure and auxiliary DataTree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

import jax.numpy as jnp

from ..core.distribution import RecordEmpiricalDistribution, Distribution
from ..core._record_distribution import _RecordDistributionView
from ..core.provenance import Provenance
from ..core.record import Record, RecordTemplate
from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights

__all__ = ["ApproximateDistribution", "make_posterior"]


# ---------------------------------------------------------------------------
# ApproximateDistribution
# ---------------------------------------------------------------------------


class ApproximateDistribution(RecordEmpiricalDistribution):
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
        record_template: RecordTemplate | None = None,
    ):
        if not chains:
            raise ValueError("Must provide at least one chain")

        self._chains = [jnp.asarray(c) for c in chains]
        self._concatenated: Array | None = None

        # When a record_template with multiple fields is provided,
        # slice the concatenated chain into per-field arrays so the
        # underlying ``RecordEmpiricalDistribution`` exposes the named
        # fields directly. Single-field user-supplied templates are
        # also honored (rename the auto-wrap field to match).
        flat = self._concat_chains()
        # Track whether the user explicitly supplied a template; we use
        # this in ``draws()`` to decide whether to wrap the output.
        self._user_template = record_template is not None
        # Multi-field template with all-flat-leaf fields → split per
        # field. Nested-template fields fall back to the single-field
        # auto-wrap path so ``draws()`` can lazily unflatten using the
        # full template (which preserves the nesting structure).
        all_flat_leaves = (
            record_template is not None
            and len(record_template.fields) > 1
            and not any(
                isinstance(record_template[f], RecordTemplate)
                for f in record_template.fields
            )
        )
        if all_flat_leaves:
            from ..core.record import Record
            from .._utils import prod
            offset = 0
            fields: dict[str, Array] = {}
            for field_name in record_template.fields:
                spec = record_template[field_name]
                shape = spec if spec is not None else ()
                size = prod(shape) if shape else 1
                chunk = flat[..., offset : offset + size]
                fields[field_name] = chunk.reshape(*flat.shape[:-1], *shape)
                offset += size
            super().__init__(Record(fields), weights=weights, name=name or "posterior")
            self._record_template = record_template
        else:
            # Single-field path: ``name`` (default ``"posterior"``)
            # becomes the auto-wrapped field name. If the user passed a
            # single-field template, rename to honor it.
            field_name = name or "posterior"
            if record_template is not None and len(record_template.fields) == 1:
                field_name = record_template.fields[0]
            super().__init__(flat, weights=weights, name=field_name)
            if record_template is not None:
                self._record_template = record_template

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
        """Per-chain warmup samples extracted from auxiliary data."""
        aux = self.auxiliary
        if aux is None:
            return None
        # arviz 1.x DataTree uses .children; arviz 0.x InferenceData uses .groups()
        has_warmup = (
            ("warmup" in aux.children) if hasattr(aux, "children")
            else hasattr(aux, "warmup")
        )
        if not has_warmup:
            return None
        warmup = aux["warmup"]["params"]
        n_chains = warmup.sizes.get("chain", 1)
        return [jnp.asarray(warmup.sel(chain=i).values) for i in range(n_chains)]

    def draws(
        self,
        chain: int | None = None,
        *,
        include_warmup: bool = False,
    ) -> Array | Record:
        """Access draws, optionally named via record_template.

        Parameters
        ----------
        chain : int or None
            Chain index.  If ``None``, concatenates all chains.
        include_warmup : bool
            If ``True`` and warmup samples are in the auxiliary DataTree,
            prepend them.

        Returns
        -------
        Array or Record
            If ``record_template`` is set, returns a :class:`~probpipe.Record`
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

        # Honor any user-supplied template (single-field or multi-field).
        # Without one, return the raw concatenated array — matches the
        # historical behaviour of single-field auto-wrap empiricals
        # under the previous numeric-array hierarchy.
        if getattr(self, "_user_template", False):
            from ..core._record_array import NumericRecordArray
            return NumericRecordArray.unflatten(samples, template=self.record_template)
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
    record_template: RecordTemplate | None = None,
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
    record_template : RecordTemplate or None
        If provided, ``draws()`` returns named ``Record``.
    **meta
        Additional metadata stored in provenance.

    Returns
    -------
    ApproximateDistribution
        Posterior with chain structure, auxiliary DataTree, and provenance.
    """
    result = ApproximateDistribution(
        chains, name="posterior", record_template=record_template,
    )

    if auxiliary is not None:
        result._auxiliary = auxiliary

    result.with_source(
        Provenance(algorithm, parents=parents, metadata={"algorithm": algorithm, **meta})
    )
    return result
