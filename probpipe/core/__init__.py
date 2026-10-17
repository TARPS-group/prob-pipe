"""Core abstractions for ProbPipe.

This package contains the foundational building blocks: the distribution
classes built on the base in ``probpipe.distributions``, ``Record`` /
``RecordBatch`` / ``RecordSpec`` value types, protocols, constraints, and the
provenance system. ``Function`` is defined in ``probpipe.values``; its call
engine lives in ``probpipe.functions``. Internal modules are prefixed with
``_``; users should import the public API from the top-level ``probpipe``
package or the documented submodules, such as ``probpipe.core.record``.
"""
