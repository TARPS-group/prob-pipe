"""Emulator (surrogate model) interfaces for ProbPipe."""

from probpipe.surrogate.emulator.base import Emulator
from probpipe.surrogate.emulator.gaussian import (
    GaussianEmulator,
    LinCombGaussianWeights,
    LinearGaussianRegressor,
)

__all__ = [
    "Emulator",
    "GaussianEmulator",
    "LinCombGaussianWeights",
    "LinearGaussianRegressor",
]
