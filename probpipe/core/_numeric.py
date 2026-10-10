"""Numeric — the flat-vector interface of the numeric kinds.

See design II.3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import jax
import numpy as np

if TYPE_CHECKING:
    from ._spec_base import NumericSpec

__all__ = ["Numeric"]


class Numeric(ABC):
    """The flat-vector interface of the numeric kinds.

    :class:`~probpipe.NumericArray` and :class:`~probpipe.NumericRecord` implement it.

    A numeric value lays itself out as one flat vector in canonical order.
    :meth:`to_vector` returns that vector, :attr:`vector_size` is its length,
    and :meth:`from_vector` rebuilds a value from one. The base supplies the
    coordinate protocols, which present the vector to NumPy and JAX, so their
    functions return bare arrays. A batch form is not ``Numeric``, since its
    elements are the numeric values.

    Notes
    -----
    :class:`~probpipe.NumericArray` restates the coordinate protocols to present
    its array itself, whose row-major order is its vector's.
    :class:`~probpipe.NumericRecord` restates them to present its sole field, an
    interim implementation detail until a record no longer presents a single
    field.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """The length of this value's flat vector."""

    @abstractmethod
    def to_vector(self) -> jax.Array:
        """This value's coordinates, one flat vector in canonical order."""

    @classmethod
    @abstractmethod
    def from_vector(cls, spec: NumericSpec, vec: Any, *, label: str | None = None) -> Self:
        """Rebuild the value that *spec* declares from its flat vector *vec*.

        It inverts :meth:`to_vector`. A kind whose data cannot identify the value
        requires *label*, as :meth:`NumericArray.from_vector
        <probpipe.NumericArray.from_vector>` does, and a record derives a default
        from its top-level fields.

        Parameters
        ----------
        spec : NumericSpec
            The declaration of the value, which supplies its layout.
        vec : Any
            The flat vector, of length ``spec.vector_size``.
        label : str or None
            Keyword-only. The label of the rebuilt value; ``None``, the default,
            derives it where the kind can.

        Returns
        -------
        Self
            The value whose :meth:`to_vector` equals *vec*.
        """

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        vector = np.asarray(self.to_vector(), dtype=dtype)
        return vector.copy() if copy else vector

    # JAX reads this for ``jnp.asarray``; the NumPy protocol would win otherwise
    # and drop tracing.
    def __jax_array__(self) -> jax.Array:
        return self.to_vector()
