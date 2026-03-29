"""Modeling interfaces for ProbPipe.

Provides abstract likelihood interfaces, iterative forecasting, and
concrete probabilistic model classes that wrap external PPL backends
(Stan, PyMC) as first-class ProbPipe distributions.
"""

from ._base import ProbabilisticModel
from ._likelihood import GenerativeLikelihood, IterativeForecaster, Likelihood
from ._simple import SimpleModel

__all__ = [
    "Likelihood",
    "GenerativeLikelihood",
    "IterativeForecaster",
    "ProbabilisticModel",
    "SimpleModel",
]

# Optional backends — available when their dependencies are installed.


def __getattr__(name: str):
    if name == "StanModel":
        from ._stan import StanModel

        return StanModel
    if name == "PyMCModel":
        from ._pymc import PyMCModel

        return PyMCModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
