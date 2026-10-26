"""Continue default random streams across IPython cells."""

from __future__ import annotations

import inspect
import sys
import threading
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextvars import Context

    from ._context import _DefaultWorkflowState, _WorkflowOwner


def _cell_on_stack(shell: Any) -> bool:
    """Whether this shell's cell coroutine is on the current Python stack."""
    from IPython.core.interactiveshell import InteractiveShell

    code = InteractiveShell.run_cell_async.__code__
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_code is code and frame.f_locals["self"] is shell:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


class _NotebookDefaultScope:
    """Persist an IPython shell's default from each completed cell's context.

    Parameters
    ----------
    shell : IPython shell
        The shell whose cell events delimit default-scope ownership.
    """

    def __init__(self, shell: Any):
        from . import _context

        self._shell = weakref.ref(shell)
        self._state: _DefaultWorkflowState | None = None
        self._host: _WorkflowOwner = _context._current_workflow_owner()
        self._cell_active = False
        self._cell_context: Context | None = None
        shell.events.register("pre_execute", self.before_cell)
        shell.events.register("post_execute", self.after_cell)

    def before_cell(self) -> None:
        """Bind the saved default to a new cell, leaving nested executions alone."""
        from . import _context

        shell = self._shell()
        if shell is None or self._cell_active:
            return
        if not _cell_on_stack(shell):
            return
        owner = _context._current_workflow_owner()
        if (
            owner.process_id != self._host.process_id
            or owner.thread_ref() is not self._host.thread_ref()
        ):
            return
        self._cell_active = True
        task = owner.task_ref() if owner.task_ref is not None else None
        self._cell_context = task.get_context() if task is not None else None
        if self._state is not None:
            self._state.owner = owner
        _context._DEFAULT_WORKFLOW_STATE.set(self._state)

    def after_cell(self) -> None:
        """Save the completed cell's default and release its execution context."""
        from . import _context

        shell = self._shell()
        if shell is None or not self._cell_active:
            return
        owner = _context._current_workflow_owner()
        if (
            owner.process_id != self._host.process_id
            or owner.thread_ref() is not self._host.thread_ref()
        ):
            return
        # A nested run_cell's completion still has the outer cell on its stack.
        if _cell_on_stack(shell):
            return
        if self._cell_context is not None:
            self._state = self._cell_context.get(_context._DEFAULT_WORKFLOW_STATE)
        elif owner.task_ref is None:
            # Synchronous cells keep their setting in the callback's context.
            self._state = _context._DEFAULT_WORKFLOW_STATE.get()
        self._cell_active = False
        self._cell_context = None


def _remember_notebook_default() -> None:
    """Bind the current IPython cell for saving its default at completion."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return
    shell = ipython.get_ipython()
    if shell is None:
        return
    adapter = getattr(shell, "_probpipe_default_scope", None)
    if adapter is None:
        if threading.current_thread() is not threading.main_thread():
            return
        if not _cell_on_stack(shell):
            return
        adapter = _NotebookDefaultScope(shell)
        shell._probpipe_default_scope = adapter
    if not adapter._cell_active:
        from . import _context

        # Initial installation can happen after pre_execute fired.
        current = _context._DEFAULT_WORKFLOW_STATE.get()
        adapter.before_cell()
        _context._DEFAULT_WORKFLOW_STATE.set(current)
