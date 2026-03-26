"""
Random function abstractions for ProbPipe.

Provides:
  - ``RandomFunction``        – Distribution over functions f: X → Y.
  - ``ArrayRandomFunction``   – Specialization for X = Array, Y = Array.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from ..utils import prod
from .distribution import Distribution

X = TypeVar('X')
Y = TypeVar('Y')


# ---------------------------------------------------------------------------
# RandomFunction[X, Y]
# ---------------------------------------------------------------------------


class RandomFunction(Distribution[Callable[[X], Y]]):
    """A distribution over functions f: X → Y.

    The primary interface is :meth:`__call__`. Calling the random 
    function on a set of inputs returns a distribution representing
    the (joint) distribution over the corresponding function outputs.
    In other words, calling returns the finite-dimensional distributions
    of the stochastic process. :meth:`log_prob` is typically not 
    implemented as random functions do not have densities in the 
    standard sense.    

    Sampling a random function means sampling an entire functional 
    trajectory. For infinite-dimensional models (e.g., Gaussian processes), 
    drawing an entire function realization may be impossible or require
    approximation. Therefore :meth:`_sample` and :meth:`sample` raise
    ``NotImplementedError`` by default.  Finite-dimensional subclasses
    (where a function is determined by a finite parameter vector) may
    override both to return callables. Infinite-dimensional subclasses
    may opt to implement this method using an approximate functional
    sampling approach.

    This class is generic in ``X`` (input type) and ``Y`` (output type).
    """

    # -- Distribution[T] contract -------------------------------------------

    def _sample(self, key: PRNGKey) -> Callable:
        """Draw a single function realization.

        Raises ``NotImplementedError`` by default.  Override in
        finite-dimensional subclasses where drawing a function reduces
        to drawing finite parameters, or in infinite-dimensional classes
        that opt to use an approximate sampling method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support sampling function "
            f"realizations. Use __call__(x) to obtain the predictive "
            f"distribution at specific inputs."
        )

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Callable:
        """Draw function realization(s).

        The default ``jax.vmap``-based implementation from
        :class:`Distribution` cannot batch over Python callables.
        This override raises ``NotImplementedError`` by default.

        Subclasses should override both :meth:`_sample` and :meth:`sample`.  
        The returned callable should produce outputs with leading 
        ``sample_shape`` dimensions.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support sampling function "
            f"realizations. Use __call__(x) to obtain the predictive "
            f"distribution at specific inputs."
        )

    # -- Fundamental interface ----------------------------------------------

    @abstractmethod
    def __call__(self, x: X) -> Distribution[Y]:
        """Return the distribution over outputs at input *x*.

        This is the fundamental interface of a random function.
        """
        ...

    # -- Optional shape properties ------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Shape of a single input point (array-valued case)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define input_shape"
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of a single output point (array-valued case)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define output_shape"
        )


# ---------------------------------------------------------------------------
# ArrayRandomFunction
# ---------------------------------------------------------------------------


class ArrayRandomFunction(RandomFunction[Array, Array]):
    """A random function mapping arrays to arrays.

    Follows the shape semantics as implemented in ``ArrayDistribution``.
    Given prediction input ``X`` with shape ``(*extra_batch, n, *input_shape)``, 
    where ``n`` is the number of input points:

    +----------------+-----------------+------------------------+-----------------------------------+
    | joint_inputs   | joint_outputs   | event_shape            | batch_shape                       |
    +================+=================+========================+===================================+
    | False          | False           | ()                     | (*extra_batch, n, *output_shape)  |
    +----------------+-----------------+------------------------+-----------------------------------+
    | True           | False           | (n,)                   | (*extra_batch, *output_shape)     |
    +----------------+-----------------+------------------------+-----------------------------------+
    | False          | True            | (*output_shape,)       | (*extra_batch, n)                 |
    +----------------+-----------------+------------------------+-----------------------------------+
    | True           | True            | (n, *output_shape)     | (*extra_batch,)                   |
    +----------------+-----------------+------------------------+-----------------------------------+

    In all modes a sample has the same total shape:
    ``(*sample_shape, *extra_batch, n, *output_shape)``.
    The flags only change which axes are jointly modeled (event) vs
    independent (batch).

    Parameters
    ----------
    input_shape : tuple of int
        Shape of a single input point, e.g. ``(3,)`` for 3-D inputs.
    output_shape : tuple of int
        Shape of a single output, e.g. ``(2,)`` for two outputs, ``()``
        for a scalar output.
    """

    # -- Capability flags (override in subclasses) --------------------------
    supports_joint_inputs: bool = False
    supports_joint_outputs: bool = False

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...] = (),
    ) -> None:
        self._input_shape = tuple(input_shape)
        self._output_shape = tuple(output_shape)

    # -- Properties ---------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Shape of a single input point."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of a single output point."""
        return self._output_shape

    # -- Callable interface -------------------------------------------------

    def __call__(
        self,
        X: ArrayLike,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Distribution:
        """Return a predictive distribution over outputs at input points *X*.

        Validates inputs, parses shapes, and delegates to :meth:`predict`.

        Parameters
        ----------
        X : array-like
            Input points with shape ``(*extra_batch, n, *input_shape)``.
        joint_inputs : bool, optional
            If True, the ``n`` axis is part of the event (predictions are
            correlated across input points).  Default: False.
        joint_outputs : bool, optional
            If True, ``output_shape`` is part of the event (predictions are
            correlated across outputs).  Default: False.

        Returns
        -------
        ArrayDistribution
            A distribution whose ``batch_shape`` and ``event_shape`` follow
            the shape table in the class docstring.
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        self._validate_joint_request(joint_inputs, joint_outputs)
        self._validate_X(X)
        return self.predict(X, joint_inputs=joint_inputs, joint_outputs=joint_outputs)

    # -- Abstract method ----------------------------------------------------

    @abstractmethod
    def predict(
        self,
        X: Array,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Distribution:
        """Subclass implementation of prediction.

        When this method is called, ``X`` has already been validated and
        converted to a JAX array.  Subclasses should return a distribution
        whose ``batch_shape`` and ``event_shape`` conform to the shape
        table in the :class:`ArrayRandomFunction` class docstring.

        Parameters
        ----------
        X : Array
            Validated input points, shape ``(*extra_batch, n, *input_shape)``.
        joint_inputs : bool
            Whether predictions should be joint across input points.
        joint_outputs : bool
            Whether predictions should be joint across outputs.
        """
        ...

    # -- Helpers ------------------------------------------------------------

    def _parse_X(self, X: Array) -> tuple[tuple[int, ...], int]:
        """Extract extra_batch shape and n from X.

        Parameters
        ----------
        X : Array
            Input with shape ``(*extra_batch, n, *input_shape)``.

        Returns
        -------
        extra_batch : tuple of int
        n : int
        """
        ndim_input = len(self._input_shape)
        n = X.shape[-(ndim_input + 1)]
        extra_batch = tuple(X.shape[:-(ndim_input + 1)])
        return extra_batch, n

    def _validate_X(self, X: Array) -> None:
        """Check that X has sufficient rank and matching trailing dims."""
        ndim_input = len(self._input_shape)
        min_ndim = ndim_input + 1  # at least (n, *input_shape)
        if X.ndim < min_ndim:
            raise ValueError(
                f"X must have at least {min_ndim} dimensions "
                f"(n, *input_shape) where input_shape={self._input_shape}, "
                f"but got X.shape={tuple(X.shape)}"
            )
        trailing = tuple(X.shape[-ndim_input:]) if ndim_input > 0 else ()
        if trailing != self._input_shape:
            raise ValueError(
                f"Trailing dimensions of X {trailing} do not match "
                f"input_shape={self._input_shape}"
            )

    def _validate_joint_request(
        self, joint_inputs: bool, joint_outputs: bool
    ) -> None:
        """Check that the requested joint mode is supported."""
        if joint_inputs and not self.supports_joint_inputs:
            raise ValueError(
                f"{type(self).__name__} does not support joint_inputs=True. "
                f"This model can only return marginals over input points."
            )
        if joint_outputs and not self.supports_joint_outputs:
            raise ValueError(
                f"{type(self).__name__} does not support joint_outputs=True. "
                f"This model can only return independent outputs."
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"input_shape={self._input_shape}, "
            f"output_shape={self._output_shape})"
        )
