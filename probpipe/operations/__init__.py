"""The operations: one call per operation, whatever the operand and whichever route realizes it.

Each operation is declared in the module of its section and registered in
:data:`operation_registry`, whose ``list()`` and ``describe()`` report its
operands and its routes in selection order. The ``expectation`` exported here
takes its method, budget, and key as arguments and is backed by
:data:`expectation_method_registry`.
"""

from . import (  # each module registers its operations
    _condition,
    _convert,
    _density,
    _evaluate,
    _inverse,
    _joint,
    _marginal,
    _mixture,
    _sample,
)
from ._moments import ExpectationMethod, expectation_method_registry
from ._moments import _expectation_function as expectation
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

__all__ = [
    "BoundCall",
    "ExpectationMethod",
    "OperandSummary",
    "OperationRegistry",
    "OperationRoute",
    "OperationSummary",
    "RouteSource",
    "RouteSummary",
    "expectation",
    "expectation_method_registry",
    "operation",
    "operation_registry",
]
