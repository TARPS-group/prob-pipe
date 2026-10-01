"""The operation model's operations and the new layers' unexported classes, resolved in one place.

``probpipe.condition_on``, ``probpipe.sample``, and the other top-level names
are still the earlier implementations, and ``probpipe.EmpiricalDistribution``
is still the earlier empirical class. The tests of the new layers import the
operations and the classes from this module, so switching the package's
exports changes only this file.
"""

from __future__ import annotations

from probpipe.distributions import FieldView
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.operations._condition import condition_on, inference_method_registry
from probpipe.operations._convert import convert
from probpipe.operations._density import log_prob, unnormalized_log_prob
from probpipe.operations._evaluate import evaluate
from probpipe.operations._joint import joint
from probpipe.operations._marginal import factor, marginal
from probpipe.operations._moments import cov, expectation, mean, quantile, variance
from probpipe.operations._sample import sample

__all__ = [
    "EmpiricalDistribution",
    "FieldView",
    "condition_on",
    "convert",
    "cov",
    "evaluate",
    "expectation",
    "factor",
    "inference_method_registry",
    "joint",
    "log_prob",
    "marginal",
    "mean",
    "quantile",
    "sample",
    "unnormalized_log_prob",
    "variance",
]
