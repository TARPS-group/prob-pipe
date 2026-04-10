"""Approximate empirical distribution with chain structure and ArviZ InferenceData."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..core.distribution import ArrayEmpiricalDistribution, Distribution
from ..core.provenance import Provenance
from ..core.values import Values
from ..custom_types import Array, ArrayLike
from .._weights import Weights

__all__ = ["ApproximateDistribution", "make_posterior"]


class ApproximateDistribution(ArrayEmpiricalDistribution):
    """Empirical distribution with chain structure and ArviZ InferenceData.

    Wraps one or more sample chains as an
    :class:`~probpipe.core.distribution.EmpiricalDistribution` while
    preserving per-chain structure, optional warmup samples, and an
    ArviZ ``InferenceData`` object for downstream diagnostics.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each of shape ``(num_draws, *event_shape)``.
    algorithm : str
        Name of the inference algorithm (e.g. ``"tfp_nuts"``, ``"rwmh"``).
    inference_data : InferenceData or None
        ArviZ ``InferenceData`` object from the inference run.  Contains
        ``posterior``, ``sample_stats``, and other groups depending on
        the backend.  Use ArviZ functions for diagnostics::

            import arviz as az
            az.summary(posterior.inference_data)
            az.plot_trace(posterior.inference_data)

    warmup_samples : list of Array or None
        Per-chain warmup (burn-in) samples, same shapes as *chains*.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Optional per-sample importance weights (across all chains).
    values_template : Values or None
        If provided, :meth:`draws` returns a :class:`~probpipe.Values`
        object with named fields instead of a raw array.  The template
        carries field names and shapes for unflattening the flat draw
        vectors.
    name : str or None
        Distribution name for provenance.
    """

    def __init__(
        self,
        chains: list[Array],
        *,
        algorithm: str = "unknown",
        inference_data: Any | None = None,
        warmup_samples: list[Array] | None = None,
        weights: ArrayLike | Weights | None = None,
        values_template: Values | None = None,
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
        self._algorithm = algorithm
        self._inference_data = inference_data
        self._values_template = values_template

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
        """Number of chains."""
        return len(self._chains)

    @property
    def num_draws(self) -> int:
        """Number of draws per chain (assumes equal-length chains)."""
        return self._chains[0].shape[0]

    @property
    def algorithm(self) -> str:
        """Name of the inference algorithm."""
        return self._algorithm

    @property
    def inference_data(self) -> Any | None:
        """ArviZ InferenceData, if available.

        Contains ``posterior``, ``sample_stats``, and other groups
        depending on the backend.  ``None`` only when inference was
        interrupted or the backend could not produce one.
        """
        return self._inference_data

    @property
    def warmup_samples(self) -> list[Array] | None:
        """Per-chain warmup samples, if stored."""
        return self._warmup_samples

    @property
    def values_template(self) -> Values | None:
        """The ``Values`` template used for named draws, if available."""
        return self._values_template

    def draws(
        self,
        chain: int | None = None,
        *,
        include_warmup: bool = False,
    ) -> Array | Values:
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
        Array or Values
            If a ``values_template`` was provided at construction, returns
            a :class:`~probpipe.Values` with named fields where each leaf
            has a leading draw dimension.  Otherwise returns a raw array
            of shape ``(num_draws, *event_shape)``.
        """
        if chain is not None:
            samples = self._chains[chain]
            if include_warmup and self._warmup_samples is not None:
                samples = jnp.concatenate(
                    [self._warmup_samples[chain], samples], axis=0
                )
        else:
            parts = []
            if include_warmup and self._warmup_samples is not None:
                for w, c in zip(self._warmup_samples, self._chains):
                    parts.append(jnp.concatenate([w, c], axis=0))
            else:
                parts = list(self._chains)
            samples = jnp.concatenate(parts, axis=0)

        if self._values_template is not None:
            return _unflatten_draws(samples, self._values_template)
        return samples

    def __repr__(self) -> str:
        return (
            f"ApproximateDistribution("
            f"algorithm={self._algorithm!r}, "
            f"num_chains={self.num_chains}, "
            f"num_draws={self.num_draws}, "
            f"event_shape={self.event_shape})"
        )


def _unflatten_draws(flat_draws: Array, template: Values) -> Values:
    """Unflatten a (num_draws, flat_size) array into a Values with a draw axis.

    Each field in the returned Values has shape ``(num_draws, *event_shape)``
    where ``event_shape`` is taken from the corresponding field in *template*.
    """
    fields: dict[str, Any] = {}
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


def make_posterior(
    chains: list[Array],
    parents: tuple[Distribution, ...],
    algorithm: str,
    *,
    inference_data: Any | None = None,
    warmup_samples: list[Array] | None = None,
    values_template: Values | None = None,
    **meta: Any,
) -> ApproximateDistribution:
    """Wrap chains into an ApproximateDistribution with provenance.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each shaped ``(num_draws, *event_shape)``.
    parents : tuple of Distribution
        Parent distributions for provenance tracking.
    algorithm : str
        Inference algorithm name (e.g. ``"tfp_nuts"``, ``"rwmh"``).
    inference_data : InferenceData or None
        ArviZ ``InferenceData`` from the inference run.
    warmup_samples : list of Array or None
        Per-chain warmup samples, same shapes as *chains*.
    values_template : Values or None
        If provided, ``draws()`` returns named ``Values`` instead of
        raw arrays.
    **meta
        Additional metadata stored in provenance.

    Returns
    -------
    ApproximateDistribution
        Posterior with chain structure, InferenceData, and provenance.
    """
    result = ApproximateDistribution(
        chains, algorithm=algorithm, inference_data=inference_data,
        warmup_samples=warmup_samples, values_template=values_template,
        name="posterior",
    )
    result.with_source(
        Provenance(algorithm, parents=parents, metadata={"algorithm": algorithm, **meta})
    )
    return result
