"""Core abstractions for ProbPipe.

This package contains the foundational building blocks: the ``Distribution``
hierarchy, ``Record`` / ``RecordArray`` / ``RecordTemplate`` value types,
workflow-graph primitives (``WorkflowFunction``, ``Module``), ops, protocols,
constraints, and the provenance system. Internal modules are prefixed with
``_``; users should import the public API from the top-level ``probpipe``
package or the documented submodules (``probpipe.core.distribution``,
``probpipe.core.record``, ``probpipe.core.ops``, ...).
"""
