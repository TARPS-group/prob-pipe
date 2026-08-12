"""Model validation utilities for ProbPipe.

Predictive checking (prior and posterior), posterior-vs-reference comparison
metrics that score an approximation against a trusted reference (analytic,
long-NUTS, or sandwich), and method self-consistency checks (simulation-based
calibration and interval coverage) — for validating inference methods and as the
unit-test layer beneath the ``probpipe-benchmark`` suite. Distinct from per-fit
convergence diagnostics, which diagnose a single fitted posterior.
"""

from ._calibration import SBCResult, interval_coverage, simulation_based_calibration
from ._comparison import (
    Reference,
    ksd,
    mmd,
    relative_cov_error,
    score_posterior,
    sliced_wasserstein,
    standardized_mean_error,
    std_ratios,
)
from ._predictive_check import predictive_check

__all__ = [
    "Reference",
    "SBCResult",
    "interval_coverage",
    "ksd",
    "mmd",
    "predictive_check",
    "relative_cov_error",
    "score_posterior",
    "simulation_based_calibration",
    "sliced_wasserstein",
    "standardized_mean_error",
    "std_ratios",
]
