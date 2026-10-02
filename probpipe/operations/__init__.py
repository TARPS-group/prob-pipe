"""The operations: one call per operation, whatever the operand and whichever route realizes it.

Each operation is declared in the module of its section and registered in
:data:`operation_registry`, whose ``list()`` and ``describe()`` report its
operands and its routes in selection order.
"""

from ._condition import condition_on, inference_method_registry
from ._convert import convert
from ._density import (
    log_prob,
    prob,
    random_log_prob,
    random_unnormalized_log_prob,
    unnormalized_log_prob,
    unnormalized_prob,
)
from ._evaluate import evaluate
from ._inverse import inverse, log_det_jacobian
from ._joint import joint
from ._marginal import factor, marginal
from ._mixture import mixture
from ._moments import (
    cov,
    expectation,
    mean,
    quantile,
    variance,
)
from ._operation import (
    BoundCall,
    OperandSummary,
    OperationRegistry,
    OperationRoute,
    OperationSummary,
    RouteSource,
    RouteSummary,
    operation,
    operation_registry,
)
from ._sample import sample

__all__ = [
    "BoundCall",
    "OperandSummary",
    "OperationRegistry",
    "OperationRoute",
    "OperationSummary",
    "RouteSource",
    "RouteSummary",
    "condition_on",
    "convert",
    "cov",
    "evaluate",
    "expectation",
    "factor",
    "inference_method_registry",
    "inverse",
    "joint",
    "log_det_jacobian",
    "log_prob",
    "marginal",
    "mean",
    "mixture",
    "operation",
    "operation_registry",
    "prob",
    "quantile",
    "random_log_prob",
    "random_unnormalized_log_prob",
    "sample",
    "unnormalized_log_prob",
    "unnormalized_prob",
    "variance",
]
