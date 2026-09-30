"""Modeling interfaces for ProbPipe.

Provides likelihood protocols, incremental conditioning, and concrete
probabilistic model classes. ``StanModel`` and ``PyMCModel``, the program-defined
families of ``probpipe.families``, are available here as well.
"""

from ._base import ProbabilisticModel
from ._glm import GLMLikelihood
from ._likelihood import (
    ConditionallyIndependentLikelihood,
    GenerativeLikelihood,
    IncrementalConditioner,
    Likelihood,
)
from ._simple import SimpleModel
from ._simple_generative import SimpleGenerativeModel

__all__ = [
    "ConditionallyIndependentLikelihood",
    "GLMLikelihood",
    "GenerativeLikelihood",
    "IncrementalConditioner",
    "Likelihood",
    "ProbabilisticModel",
    "SimpleGenerativeModel",
    "SimpleModel",
]

# Optional backends — available when their dependencies are installed.


def __getattr__(name: str):
    if name in ("PyMCModel", "StanModel"):
        from ..families import _programs

        return getattr(_programs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
