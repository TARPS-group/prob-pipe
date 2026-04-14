"""
Core distribution abstractions for ProbPipe.

Public API facade that re-exports symbols from the internal submodules:

  - :mod:`._distribution_base`       – ``Distribution[T]`` + minimal helpers
  - :mod:`._array_distributions`     – Array hierarchy + ``BootstrapDistribution``
  - :mod:`._empirical`               – Empirical + BootstrapReplicate distributions
  - :mod:`._broadcast_distributions` – ``BroadcastDistribution`` + marginals

Import from this module (e.g., ``from probpipe.core.distribution import
Distribution``) rather than from the internal submodules directly.
"""

from __future__ import annotations

# -- _distribution_base -----------------------------------------------------
from . import _distribution_base as _base
from ._distribution_base import (
    Distribution,
    set_default_num_evaluations,
    set_return_approx_dist,
)


# Mutable globals: delegate attribute access to _distribution_base so that
# mutations via set_default_num_evaluations / set_return_approx_dist are
# visible through this facade module.
_MUTABLE_GLOBALS = {"DEFAULT_NUM_EVALUATIONS", "RETURN_APPROX_DIST"}


def __getattr__(name: str):
    if name in _MUTABLE_GLOBALS:
        return getattr(_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# -- _record_distribution ---------------------------------------------------
from ._record_distribution import (
    RecordDistribution,
    _RecordDistributionView,
    _unflatten_batched,
)

# -- _array_distributions ---------------------------------------------------
from ._array_distributions import (
    TFPRecordDistribution,
    ArrayDistribution,
    BootstrapDistribution,
    FlattenedView,
    TFPShapeMixin,
    _mc_expectation,
    _vmap_sample,
)

# -- _empirical -------------------------------------------------------------
from ._empirical import (
    ArrayEmpiricalDistribution,
    ArrayBootstrapReplicateDistribution,
    EmpiricalDistribution,
    BootstrapReplicateDistribution,
    TFPEmpiricalDistribution,
)

# -- _broadcast_distributions -----------------------------------------------
from ._broadcast_distributions import (
    BroadcastDistribution,
    MarginalizedBroadcastDistribution,
    _ArrayMarginal,
    _ListMarginal,
    _MixtureMarginal,
    _make_marginal,
    _make_mixture_marginal,
)

# -- _random_functions ------------------------------------------------------
from ._random_functions import (
    RandomFunction,
    ArrayRandomFunction,
)

__all__ = [
    # Base
    "Distribution",
    # Global settings
    "DEFAULT_NUM_EVALUATIONS",
    "RETURN_APPROX_DIST",
    "set_default_num_evaluations",
    "set_return_approx_dist",
    # Helpers
    "_vmap_sample",
    "_mc_expectation",
    # Record distribution
    "RecordDistribution",
    "_RecordDistributionView",
    "_unflatten_batched",
    # Array hierarchy
    "TFPRecordDistribution",
    "ArrayDistribution",
    "BootstrapDistribution",
    "FlattenedView",
    # Empirical
    "EmpiricalDistribution",
    "ArrayEmpiricalDistribution",
    # Joint bootstrap
    "BootstrapReplicateDistribution",
    "ArrayBootstrapReplicateDistribution",
    # Random functions
    "RandomFunction",
    "ArrayRandomFunction",
    # Broadcast
    "BroadcastDistribution",
    "MarginalizedBroadcastDistribution",
    "_ArrayMarginal",
    "_MixtureMarginal",
    "_ListMarginal",
    "_make_marginal",
    "_make_mixture_marginal",
]
