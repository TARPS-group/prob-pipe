"""
Emulator mixin for ProbPipe.

Provides:
  - ``EmulatorMixin`` – Mixin class adding training-data-aware
    capabilities (fit, update, data access) to random functions.
"""

from __future__ import annotations

from ..custom_types import Array, ArrayLike


class EmulatorMixin:
    """Mixin for random functions that are trained on observed data.

    Adds an optional interface for fitting, updating, and inspecting
    training data. All methods raise ``NotImplementedError`` by default;
    subclasses implement whichever ones are applicable. By mixing into 
    an existing random function class, the random function is interpreted
    as a probabilistic predictive model.

    This mixin is designed to be composed with any
    :class:`~probpipe.RandomFunction` subclass::

        class MyEmulator(GaussianRandomFunction, EmulatorMixin):
            ...

    The mixin is intentionally thin — it defines the *interface* for
    data-aware models without imposing any storage requirements.
    Subclasses that wrap external models (e.g., a pre-trained GP library
    object) can delegate to the wrapped model's own storage rather than
    duplicating training data.
    """

    # -- Training interface -------------------------------------------------

    def fit(self, X: ArrayLike, Y: ArrayLike) -> None:
        """Fit the model to training data.

        Parameters
        ----------
        X : array-like
            Training inputs.
        Y : array-like
            Training responses (targets).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fit"
        )

    def update(self, X_new: ArrayLike, Y_new: ArrayLike) -> None:
        """Update the model with new observations.

        Parameters
        ----------
        X_new : array-like
            New input observations.
        Y_new : array-like
            New response observations.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement update"
        )

    # -- Data access --------------------------------------------------------

    @property
    def training_inputs(self) -> Array:
        """Training input data used to fit this model."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose training_inputs"
        )

    @property
    def training_responses(self) -> Array:
        """Training response data used to fit this model."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose training_responses"
        )
