"""Importable callables with postponed type alias annotations for replay tests."""

from __future__ import annotations

from typing import Annotated, TypeAliasType

import jax

type Values = jax.Array
type Maybe[T] = T | None


def values_alias(name: str, value: type) -> TypeAliasType:
    return TypeAliasType(name, value)


def shift(x: Values, noise):
    return x + noise + (1.0 if isinstance(x, jax.Array) else 100.0)


def optional(x: Values | None):
    return 0.0


def generic(x: Maybe[Values]):
    return 0.0


def metadata(x: Annotated[Values, "values"]):
    return 0.0


def returned(x) -> Values:
    return x
