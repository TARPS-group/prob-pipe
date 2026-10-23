"""Public errors raised when work enters a workflow run, and by replay."""

from __future__ import annotations


class UnmanagedConcurrentWorkflowEntryError(RuntimeError):
    """An unmanaged thread or task attempted to enter a copied workflow context.

    Run the work through a ProbPipe call, such as one that names an executor,
    to take part in the parent run, or start an independent ``workflow_run`` in
    the concurrent worker.
    """


class ReplayCompatibilityError(RuntimeError):
    """Recorded workflow controls are incompatible with a replay attempt."""


class ReplayUnsupportedCallableError(TypeError):
    """A callable lacks the strong definition anchor required for replay."""
