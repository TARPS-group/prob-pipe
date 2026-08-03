"""Public errors raised by managed workflow execution and replay."""


class UnmanagedConcurrentWorkflowEntryError(RuntimeError):
    """A copied workflow context entered from an unmanaged thread or task."""


class ReplayCompatibilityError(RuntimeError):
    """Recorded workflow controls are incompatible with the replay attempt."""


class ReplayUnsupportedCallableError(TypeError):
    """A callable does not have the strong anchor required for replay."""
