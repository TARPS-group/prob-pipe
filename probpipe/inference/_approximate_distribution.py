"""Approximate empirical distribution with chain structure and auxiliary DataTree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

import jax.numpy as jnp

from ..core.constraints import real
from ..core.distribution import ArrayDistribution, ArrayEmpiricalDistribution, Distribution
from ..core.protocols import SupportsMean, SupportsSampling, SupportsVariance
from ..core.provenance import Provenance
from ..core.values import Values
from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights

__all__ = ["ApproximateDistribution", "make_posterior", "_ValuesDistributionView"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unflatten_draws(flat_draws: Array, template: Values) -> Values:
    """Unflatten a (num_draws, flat_size) array into a Values with a draw axis.

    Each field in the returned Values has shape ``(num_draws, *event_shape)``
    where ``event_shape`` is taken from the corresponding field in *template*.
    """
    fields: dict[str, jnp.ndarray | Values] = {}
    offset = 0
    for name in template.fields():
        tval = template[name]
        if isinstance(tval, Values):
            size = tval.flat_size
            child_flat = flat_draws[:, offset:offset + size]
            fields[name] = _unflatten_draws(child_flat, tval)
            offset += size
        else:
            size = tval.size
            event_shape = tval.shape
            chunk = flat_draws[:, offset:offset + size]
            fields[name] = chunk.reshape(flat_draws.shape[0], *event_shape)
            offset += size
    return Values(fields)


# ---------------------------------------------------------------------------
# _ValuesDistributionView — lightweight ref to a named field
# ---------------------------------------------------------------------------


class _ValuesDistributionView(ArrayDistribution, SupportsSampling, SupportsMean, SupportsVariance):
    """Lightweight reference to a single named field of a Values-based distribution.

    The Values-world analog of
    :class:`~probpipe.core._joint.DistributionView`.  Preserves
    correlation when multiple views from the same parent are used in
    :class:`~probpipe.core.node.WorkflowFunction` broadcasting.

    Parameters
    ----------
    parent : Distribution
        A distribution with ``values_template`` set.
    key : str
        Field name in the parent's ``values_template``.
    """

    _sampling_cost = "low"
    _preferred_orchestration = None

    def __init__(self, parent: Distribution, key: str):
        template = parent.values_template
        if template is None or key not in template:
            raise KeyError(
                f"No field {key!r} in values_template "
                f"(available: {template.fields() if template else ()})"
            )
        self._parent = parent
        self._key = key
        self._key_path = (key,)  # duck-type contract with DistributionView
        self._template_field = template[key]

    # -- ArrayDistribution interface ----------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        f = self._template_field
        return f.shape if not isinstance(f, Values) else ()

    @property
    def support(self):
        return real

    @classmethod
    def _default_support(cls):
        return real

    # -- SupportsSampling ---------------------------------------------------

    def _sample_one(self, key: PRNGKey) -> Array:
        structured = self._parent._sample(key)
        return self._extract(structured)

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        structured = self._parent._sample(key, sample_shape)
        return self._extract(structured)

    # -- SupportsMean / SupportsVariance ------------------------------------

    def _mean(self) -> Array:
        if isinstance(self._parent, SupportsMean):
            m = self._parent._mean()
            if isinstance(m, Values):
                return m[self._key]
        # Fallback: empirical mean from draws
        return self._field_draws().mean(axis=0)

    def _variance(self) -> Array:
        if isinstance(self._parent, SupportsVariance):
            v = self._parent._variance()
            if isinstance(v, Values):
                return v[self._key]
        return self._field_draws().var(axis=0)

    # -- Internals ----------------------------------------------------------

    def _extract(self, structured: Array) -> Array:
        """Extract this field from a parent sample (flat array or Values)."""
        if isinstance(structured, Values):
            return structured[self._key]
        template = self._parent.values_template
        if structured.ndim == 1:
            return Values.unflatten(structured, template=template)[self._key]
        return _unflatten_draws(structured, template)[self._key]

    def _field_draws(self) -> Array:
        """All draws for this field (requires parent to have a ``draws()`` method)."""
        draws = self._parent.draws()
        if isinstance(draws, Values):
            return jnp.asarray(draws[self._key])
        return jnp.asarray(
            _unflatten_draws(draws, self._parent.values_template)[self._key]
        )

    def __repr__(self) -> str:
        return (
            f"_ValuesDistributionView(parent={type(self._parent).__name__}, "
            f"field={self._key!r})"
        )


# ---------------------------------------------------------------------------
# ApproximateDistribution
# ---------------------------------------------------------------------------


class ApproximateDistribution(ArrayEmpiricalDistribution):
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

    # -- Named component access (SupportsNamedComponents) --------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        """Field names from the values_template, or empty tuple."""
        tpl = self.values_template
        return tpl.fields() if tpl is not None else ()

    def __getitem__(self, key: str) -> _ValuesDistributionView:
        return _ValuesDistributionView(self, key)

    def select(self, *fields: str, **mapping: str) -> dict[str, _ValuesDistributionView]:
        """Select named fields as views for workflow function broadcasting.

        Positional args use the field name as the argument name.
        Keyword args remap: ``select(x="field_name")``.

        Usage::

            predict(**posterior.select("r", "K", "phi"), x=x_grid)
        """
        result: dict[str, _ValuesDistributionView] = {}
        for f in fields:
            result[f] = self[f]
        for arg_name, field_name in mapping.items():
            result[arg_name] = self[field_name]
        return result

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
            return _unflatten_draws(samples, self.values_template)
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
