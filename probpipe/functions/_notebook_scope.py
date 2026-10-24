"""Continue default random streams across IPython kernel cell tasks."""

from __future__ import annotations

import inspect
import sys
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._context import _DefaultWorkflowState, _WorkflowOwner


def _cell_info(shell: Any) -> Any:
    """Find this shell's cell execution in the current task's Python stack."""
    code = shell.run_cell_async.__func__.__code__
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_code is code and frame.f_locals.get("self") is shell:
                return frame.f_locals.get("info")
            frame = frame.f_back
        return None
    finally:
        del frame


class _NotebookDefaultScope:
    """Hold one shell's default, admitting only the task executing its current cell."""

    def __init__(self, shell: Any):
        self._shell = weakref.ref(shell)
        self._state: _DefaultWorkflowState | None = None
        self._owner: _WorkflowOwner | None = None
        self._host: _WorkflowOwner | None = None
        self._info: Any = None
        shell.events.register("pre_execute", self.before_cell)
        shell.events.register("post_execute", self.after_cell)

    def before_cell(self) -> None:
        """Bind the saved default to a new cell, leaving nested executions alone."""
        from . import _context

        shell = self._shell()
        if shell is None or self._info is not None:
            return
        info = _cell_info(shell)
        if info is None:
            return
        owner = _context._current_workflow_owner()
        if self._host is not None and (
            owner.process_id != self._host.process_id
            or owner.thread_ref() is not self._host.thread_ref()
        ):
            return
        self._host = owner
        self._owner = owner
        self._info = info
        if self._state is not None:
            self._state.owner = owner
        _context._DEFAULT_WORKFLOW_STATE.set(self._state)

    def after_cell(self) -> None:
        """Release the cell owner after completion, including an error or cancellation."""
        shell = self._shell()
        if shell is None:
            return
        result = shell.last_execution_result
        if result is not None and result.info is self._info:
            self._info = None
            self._owner = None

    def remember(self, state: _DefaultWorkflowState | None) -> None:
        """Persist a cell's setting without persisting a child task's local default."""
        from . import _context

        shell = self._shell()
        if shell is None:
            return
        if self._info is None:
            # The first set/clear can install the hooks after pre_execute ran.
            # A child task has no shell execution on its stack and cannot claim it.
            current = _context._DEFAULT_WORKFLOW_STATE.get()
            self.before_cell()
            _context._DEFAULT_WORKFLOW_STATE.set(current)
        if self._owner is not None and _context._is_current_workflow_owner(self._owner):
            self._state = state


def _remember_notebook_default(state: _DefaultWorkflowState | None) -> None:
    """Save a notebook cell's default without importing an optional IPython runtime."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return
    shell = ipython.get_ipython()
    if shell is None or getattr(shell, "kernel", None) is None:
        return
    adapter = getattr(shell, "_probpipe_default_scope", None)
    if adapter is None:
        # A first setting in a worker is local to that worker. Install shell
        # hooks only from a cell, rather than racing to register them in workers.
        if _cell_info(shell) is None:
            return
        adapter = _NotebookDefaultScope(shell)
        shell._probpipe_default_scope = adapter
    adapter.remember(state)
