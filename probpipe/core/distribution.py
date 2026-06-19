"""
Core distribution abstractions for ProbPipe.

Public API facade that re-exports symbols from the internal submodules:

  - :mod:`._distribution_base`       – ``Distribution[T]`` + minimal helpers
  - :mod:`._numeric_record_distribution`     – Array hierarchy + ``BootstrapDistribution``
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
# -- _broadcast_distributions -----------------------------------------------
from ._broadcast_distributions import (
    BroadcastDistribution,
    MarginalizedBroadcastDistribution,
    _ListMarginal,
    _make_marginal,
    _make_mixture_marginal,
    _MixtureMarginal,
    _RecordMarginal,
)

# -- _empirical -------------------------------------------------------------
from ._empirical import (
    BootstrapReplicateDistribution,
    EmpiricalDistribution,
    RecordBootstrapReplicateDistribution,
    RecordEmpiricalDistribution,
)

# -- _numeric_record_distribution ---------------------------------------------------
from ._numeric_record_distribution import (
    BootstrapDistribution,
    FlatNumericRecordDistribution,
    FlattenedDistributionView,
    NumericRecordDistribution,
    NumericRecordDistributionView,
    _mc_expectation,
    _vmap_sample,
)

# -- _random_functions ------------------------------------------------------
from ._random_functions import (
    ArrayRandomFunction,
    RandomFunction,
)

# -- _random_measures -------------------------------------------------------
from ._random_measures import (
    NumericRandomMeasure,
    RandomMeasure,
)
from ._record_distribution import (
    RecordDistribution,
    _RecordDistributionView,
)

__all__ = [
    # Global settings
    "DEFAULT_NUM_EVALUATIONS",  # noqa: F822 — served dynamically via module __getattr__
    "RETURN_APPROX_DIST",  # noqa: F822 — served dynamically via module __getattr__
    "ArrayRandomFunction",
    "BootstrapDistribution",
    # Bootstrap replicate
    "BootstrapReplicateDistribution",
    # Broadcast
    "BroadcastDistribution",
    # Base
    "Distribution",
    # Empirical
    "EmpiricalDistribution",
    "FlatNumericRecordDistribution",
    "FlattenedDistributionView",
    "MarginalizedBroadcastDistribution",
    "NumericRandomMeasure",
    # Array hierarchy
    "NumericRecordDistribution",
    "NumericRecordDistributionView",
    # Random functions
    "RandomFunction",
    # Random measures
    "RandomMeasure",
    "RecordBootstrapReplicateDistribution",
    # Record distribution
    "RecordDistribution",
    "RecordEmpiricalDistribution",
    "_ListMarginal",
    "_MixtureMarginal",
    "_RecordDistributionView",
    "_RecordMarginal",
    "_make_marginal",
    "_make_mixture_marginal",
    "_mc_expectation",
    # Helpers
    "_vmap_sample",
    "set_default_num_evaluations",
    "set_return_approx_dist",
]
