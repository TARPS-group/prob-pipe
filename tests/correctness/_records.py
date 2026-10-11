"""Small generators of record schemas and of atoms over them, for parametrized record tests.

A schema is a ``RecordSpec`` of float32 array leaves, nested to a stated depth,
and its atoms are a ``NumericRecordBatch`` on named levels whose leaves share a
latent factor, so that every pair of leaves is correlated. The helpers return
the leaves of a batch as plain arrays along one atom axis, in the row-major
order of the batch axes, which is the order an empirical law weights its atoms.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import NumericArraySpec, NumericRecordBatch, RecordSpec
from probpipe.distributions._factored import _raw_record

__all__ = [
    "LAYOUTS",
    "SCHEMAS",
    "atoms",
    "columns",
    "group_paths",
    "leaf_paths",
    "node_of",
]


def _leaf(*shape: int) -> NumericArraySpec:
    return NumericArraySpec(tuple(shape), jnp.float32)


#: The schemas the parametrized tests range over, by id: a flat record, a
#: record of two groups, and a record three levels deep with a vector beside it.
SCHEMAS: Mapping[str, RecordSpec] = {
    "flat": RecordSpec(a=_leaf(), b=_leaf(3)),
    "two-groups": RecordSpec(
        population=RecordSpec(mu=_leaf(), tau=_leaf()), groups=RecordSpec(theta=_leaf(4))
    ),
    "deep": RecordSpec(
        model=RecordSpec(theta=RecordSpec(mu=_leaf(), sd=_leaf(2)), noise=_leaf()), y=_leaf(3)
    ),
}

#: The level layouts of the atoms, by id: each the level names and their sizes.
LAYOUTS: Mapping[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "one-level": (("atom",), (12,)),
    "two-levels": (("chain", "draw"), (3, 8)),
    "three-levels": (("a", "b", "c"), (2, 3, 4)),
}


def leaf_paths(spec: RecordSpec, prefix: str = "") -> list[str]:
    """The paths of *spec*'s array leaves, in its canonical order."""
    paths: list[str] = []
    for name, child in spec.children.items():
        path = f"{prefix}{name}"
        if isinstance(child, RecordSpec):
            paths.extend(leaf_paths(child, f"{path}/"))
        else:
            paths.append(path)
    return paths


def group_paths(spec: RecordSpec, prefix: str = "") -> list[str]:
    """The paths of *spec*'s interior nodes, outermost first."""
    paths: list[str] = []
    for name, child in spec.children.items():
        if isinstance(child, RecordSpec):
            path = f"{prefix}{name}"
            paths.append(path)
            paths.extend(group_paths(child, f"{path}/"))
    return paths


def node_of(spec: RecordSpec, path: str) -> Any:
    """The spec of the node at *path* of *spec*."""
    return spec.at_path(*path.split("/"))


def _nest(flat: Mapping[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        *heads, last = path.split("/")
        node = nested
        for head in heads:
            node = node.setdefault(head, {})
        node[last] = value
    return nested


def atoms(
    spec: RecordSpec, layout: tuple[tuple[str, ...], tuple[int, ...]], seed: int = 0
) -> NumericRecordBatch:
    """Atoms of *spec* on the levels of *layout*, each leaf a shared latent plus its own noise.

    Every coordinate is ``z + 0.5 e`` for one standard-normal latent ``z`` per
    atom and independent standard-normal noise ``e``, shifted by the leaf's
    index, so the coordinates are correlated (0.8) and the leaves have
    distinct means.
    """
    names, sizes = layout
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaf_paths(spec)) + 1)
    latent = jax.random.normal(keys[0], sizes)
    flat: dict[str, Any] = {}
    for index, (path, key) in enumerate(zip(leaf_paths(spec), keys[1:])):
        shape = node_of(spec, path).shape
        noise = jax.random.normal(key, (*sizes, *shape))
        expanded = jnp.reshape(latent, (*sizes, *([1] * len(shape))))
        flat[path] = (index + expanded + 0.5 * noise).astype(jnp.float32)
    return NumericRecordBatch(
        _nest(flat),
        names,
        element_spec=spec,
        axes_per_level=(1,) * len(names),
        label="atoms",
    )


def columns(batch: Any, spec: RecordSpec) -> dict[str, np.ndarray]:
    """Each leaf of *batch* as a float64 array ``(atoms, *leaf_shape)``, the batch axes merged.

    *batch* is a batch of records or the nested mapping of its raw columns.
    """
    raw = _raw_record(batch)
    result: dict[str, np.ndarray] = {}
    for path in leaf_paths(spec):
        node = raw
        for segment in path.split("/"):
            node = node[segment]
        values = np.asarray(node, dtype=np.float64)
        shape = node_of(spec, path).shape
        count = math.prod(values.shape[: values.ndim - len(shape)])
        result[path] = values.reshape(count, *shape)
    return result


def cases() -> Iterator[tuple[str, str]]:
    """Every pair of a schema id and a layout id."""
    for schema in SCHEMAS:
        for layout in LAYOUTS:
            yield schema, layout
