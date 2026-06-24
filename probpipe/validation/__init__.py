"""Model validation utilities for ProbPipe.

Predictive checking (prior and posterior) plus posterior-vs-reference comparison
metrics that score an approximation against a trusted reference (analytic,
long-NUTS, or sandwich) — for validating inference methods and as the unit-test
layer beneath the ``probpipe-benchmark`` suite. Distinct from per-fit
convergence diagnostics, which diagnose a single fitted posterior.
"""

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
    "ksd",
    "mmd",
    "predictive_check",
    "relative_cov_error",
    "score_posterior",
    "sliced_wasserstein",
    "standardized_mean_error",
    "std_ratios",
]
