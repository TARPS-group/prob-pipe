"""Core abstractions for ProbPipe.

This package contains the foundational building blocks: the distribution
classes built on the base in ``probpipe.distributions``, ``Record`` /
``RecordBatch`` / ``RecordSpec`` value types,
workflow-graph primitives (``Function``, ``Module``), ops, protocols,
constraints, and the provenance system. Internal modules are prefixed with
``_``; users should import the public API from the top-level ``probpipe``
package or the documented submodules (``probpipe.core.record``,
``probpipe.core.ops``, ...).
"""
