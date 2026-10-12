"""Importable callables annotated with type aliases, for callable-anchor tests.

The module leaves annotations unpostponed, so each function's annotations hold
the alias objects themselves rather than strings.
"""

from typing import Annotated, Any

import numpy as np
import typing_extensions

from probpipe import Normal, NumericArrayBatch

type Rows = NumericArrayBatch
type Maybe[T] = T | None
# Stands in for NumPy 2.5's ``numpy.typing.NDArray``, which CI does not install.
type Array[Scalar: np.generic] = np.ndarray[tuple[Any, ...], np.dtype[Scalar]]
type Tree = list[Tree] | int
ExtensionRows = typing_extensions.TypeAliasType(  # noqa: UP040 - exercises typing_extensions
    "ExtensionRows", NumericArrayBatch
)
type Undefined = NameDefinedNowhere  # noqa: F821 - the alias's value cannot be evaluated


def annotated_rows(x: Rows) -> float:
    return 0.0


def annotated_maybe(x: Maybe[Normal]) -> float:
    return 0.0


def annotated_array(x: Array[np.float64]) -> float:
    return 0.0


def annotated_tree(x: Tree) -> float:
    return 0.0


def annotated_extension_rows(x: ExtensionRows) -> float:
    return 0.0


def annotated_metadata(x: Annotated[Rows, "rows"]) -> float:
    return 0.0


def annotated_undefined(x: Undefined) -> float:
    return 0.0
