"""Model validation utilities for ProbPipe.

Provides predictive checking (prior and posterior) as a first-class
:class:`~probpipe.core.node.WorkflowFunction`.
"""

from ._predictive_check import predictive_check

__all__ = ["predictive_check"]
