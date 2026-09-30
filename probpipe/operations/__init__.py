"""The operations: one call per operation, whatever the operand and whichever method realizes it."""

from ._moments import ExpectationMethod, expectation, expectation_method_registry

__all__ = ["ExpectationMethod", "expectation", "expectation_method_registry"]
