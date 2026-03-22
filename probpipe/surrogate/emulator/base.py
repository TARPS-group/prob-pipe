"""
Base emulator abstraction for ProbPipe.

Provides:
  - ``Emulator`` - Abstract base class for probabilistic emulators.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import jax.numpy as jnp

from ...custom_types import Array, ArrayLike, PRNGKey
from ...distributions import Distribution


def _prod(shape: tuple[int, ...]) -> int:
    """Product of a shape tuple, returning 1 for empty tuple."""
    return math.prod(shape) if shape else 1


class Emulator(ABC):
    """Abstract base class for probabilistic emulators.

    An emulator wraps a probabilistic model that maps inputs to a predictive
    distribution over outputs.  The design follows TFP-style shape semantics
    (``sample_shape``, ``batch_shape``, ``event_shape``).

    Shape Contract
    --------------
    Given prediction input ``X`` with shape ``(*extra_batch, n, *input_shape)``:

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

    # -- Capability flags (override in subclasses) ----------------------------
    supports_joint_inputs: bool = False
    supports_joint_outputs: bool = False

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...] = (),
    ) -> None:
        self._input_shape = tuple(input_shape)
        self._output_shape = tuple(output_shape)

    # -- Properties -----------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Shape of a single input point."""
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of a single output point."""
        return self._output_shape

    # -- Callable interface ---------------------------------------------------

    def __call__(
        self,
        X: ArrayLike,
        *,
        joint_inputs: bool = False,
        joint_outputs: bool = False,
    ) -> Distribution:
        """Return a predictive distribution over outputs at input points *X*.

        This method validates inputs, parses shapes, and delegates to the
        subclass :meth:`predict` method.

        Parameters
        ----------
        X : array-like
            Input points with shape ``(*extra_batch, n, *input_shape)``.
            The trailing ``1 + len(input_shape)`` axes are ``(n, *input_shape)``;
            any leading axes are extra batch dimensions passed through to the
            returned distribution's ``batch_shape``.
        joint_inputs : bool, optional
            If True, the ``n`` axis is part of the event (predictions are
            correlated across input points).  Requires
            ``supports_joint_inputs=True``.  Default: False.
        joint_outputs : bool, optional
            If True, ``output_shape`` is part of the event (predictions are
            correlated across outputs).  Requires
            ``supports_joint_outputs=True``.  Default: False.

        Returns
        -------
        Distribution
            A distribution whose ``batch_shape`` and ``event_shape`` follow
            the shape table in the class docstring.

        Raises
        ------
        ValueError
            If ``joint_inputs=True`` and ``supports_joint_inputs`` is False.
            If ``joint_outputs=True`` and ``supports_joint_outputs`` is False.
            If ``X`` does not have enough dimensions or its trailing axes
            do not match ``input_shape``.
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        self._validate_joint_request(joint_inputs, joint_outputs)
        self._validate_X(X)
        return self.predict(X, joint_inputs=joint_inputs, joint_outputs=joint_outputs)

    # -- Abstract method ------------------------------------------------------

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
        object whose ``batch_shape`` and ``event_shape`` conform to the
        shape table in the :class:`Emulator` class docstring for the given
        joint flags.

        Parameters
        ----------
        X : Array
            Validated input points, shape ``(*extra_batch, n, *input_shape)``.
        joint_inputs : bool
            Whether predictions should be joint across input points.
        joint_outputs : bool
            Whether predictions should be joint across outputs.

        Returns
        -------
        Distribution
        """
        ...

    # -- Trajectory sampling --------------------------------------------------

    def sample_trajectory(
        self,
        key: PRNGKey,
        n_trajectories: int,
    ) -> Callable[[ArrayLike], Array]:
        """Draw random function realizations from the emulator.

        This is an optional method that concrete subclasses may implement.
        It views the emulator as a random function and draws ``n_trajectories``
        realizations.  The returned callable represents these realizations:
        calling it at different input locations evaluates the *same*
        underlying function draws, guaranteeing consistency across calls.

        Parameters
        ----------
        key : PRNGKey
            JAX PRNG key for sampling.
        n_trajectories : int
            Number of independent trajectories (function realizations)
            to draw.

        Returns
        -------
        g : callable
            Accepts ``X`` with shape ``(*extra_batch, n, *input_shape)``
            and returns an array of shape
            ``(n_trajectories, *extra_batch, n, *output_shape)``
            representing the ``n_trajectories`` function realizations
            evaluated at the input points.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement trajectory sampling.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample_trajectory."
        )

    # -- Helpers --------------------------------------------------------------

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
                f"This emulator can only return marginals over input points."
            )
        if joint_outputs and not self.supports_joint_outputs:
            raise ValueError(
                f"{type(self).__name__} does not support joint_outputs=True. "
                f"This emulator can only return independent outputs."
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"input_shape={self._input_shape}, "
            f"output_shape={self._output_shape})"
        )
