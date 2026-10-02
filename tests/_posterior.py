"""Readers of an inference result for the tests: its draws, chains, warmup, and annotations.

An inference result is an ``EmpiricalDistribution`` whose atoms lie on the levels
``chain`` and ``draw``, and whose annotations record the method's name and its
ArviZ-compatible groups. These readers give the tests the draws in the flat layout
the methods produce them in.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from probpipe.core._numeric_record import _reconstruct_from_vector
from probpipe.core._specs import RecordSpec, _components_record
from probpipe.inference._approximate_distribution import _flat_chains, _num_chains, make_posterior


def posterior_of(
    chains: list[Any],
    *,
    label: str | None = None,
    event_spec: Any = None,
    field_order: list[str] | None = None,
    weights: Any = None,
    method: str = "test",
) -> Any:
    """The inference result an inference method builds from *chains*, labeled *label*."""
    return make_posterior(
        chains,
        (),
        method,
        event_spec=event_spec,
        field_order=field_order,
        weights=weights,
        label=label or "posterior",
    )


def flat_chains(posterior: Any) -> Any:
    """The draws as one array ``(chains, draws, d)`` in the target's flat layout."""
    return _flat_chains(posterior)


def num_chains(posterior: Any) -> int:
    """The number of chains."""
    return _num_chains(posterior)


def num_draws(posterior: Any) -> int:
    """The number of draws per chain."""
    return int(posterior.atoms.batch_shape[1])


def arviz_data(posterior: Any) -> Any:
    """The ArviZ-compatible ``DataTree`` under ``arviz`` of the annotations, or ``None``."""
    annotations = posterior.annotations
    if annotations is None or "arviz" not in annotations.children:
        return None
    return annotations["arviz"]


def method_of(posterior: Any) -> str:
    """The name of the inference method that the annotations record."""
    return posterior.annotations.attrs["method"]


def warmup_samples(posterior: Any) -> list[Any] | None:
    """The per-chain warmup draws in the annotations, or ``None``."""
    tree = arviz_data(posterior)
    if tree is None or "warmup" not in tree.children:
        return None
    warmup = tree["warmup"]["params"]
    count = warmup.sizes.get("chain", 1)
    return [jnp.asarray(warmup.sel(chain=i).values) for i in range(count)]


def law_draws(law: Any, count: int = 500) -> dict[str, Any]:
    """*count* draws of *law* under the caller's workflow, each leaf an array ``(count, *shape)``.

    A record event gives one entry per leaf path, and any other event one entry
    under its component.
    """
    import numpy as np

    from probpipe import sample
    from probpipe.distributions._empirical import _flat_rows

    rows = _flat_rows(sample(law, sample_shape=(count,)))
    if isinstance(rows, dict):
        return {path: np.asarray(column) for path, column in rows.items()}
    (component,) = law.event_spec.components
    return {component: np.asarray(rows)}


def flat_draws(posterior: Any, chain: int | None = None, *, include_warmup: bool = False) -> Any:
    """The draws of one chain, or of all chains in order, optionally after the warmup.

    A result whose event is a whole term under its own label, as a result built
    without a target is, gives the array of flat draws. Any other gives a record
    whose fields are its components.
    """
    chains = list(_flat_chains(posterior))
    warmup = warmup_samples(posterior) if include_warmup else None
    if warmup is not None:
        chains = [jnp.concatenate([w, c], axis=0) for w, c in zip(warmup, chains)]
    samples = chains[chain] if chain is not None else jnp.concatenate(chains, axis=0)
    spec = posterior.event_spec
    if not isinstance(spec.spec, RecordSpec) and list(spec.components) == [posterior.label]:
        return samples
    return _reconstruct_from_vector(
        posterior.label, _components_record(posterior.event_spec), samples
    )
