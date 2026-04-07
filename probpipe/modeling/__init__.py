"""Modeling interfaces for ProbPipe.

Provides likelihood protocols, incremental conditioning, and
concrete probabilistic model classes that wrap external PPL backends
(Stan, PyMC) as first-class ProbPipe distributions.
"""

from ._base import ProbabilisticModel
from ._glm import GLMLikelihood
from ._likelihood import GenerativeLikelihood, IncrementalConditioner, Likelihood
from ._simple import SimpleModel
from ._simple_generative import SimpleGenerativeModel

__all__ = [
    "GLMLikelihood",
    "Likelihood",
    "GenerativeLikelihood",
    "IncrementalConditioner",
    "ProbabilisticModel",
    "SimpleModel",
    "SimpleGenerativeModel",
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
