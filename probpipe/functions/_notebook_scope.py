"""Continue default random streams across IPython cells."""

from __future__ import annotations

import inspect
import sys
import threading
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextvars import Context
    from types import FrameType

    from ._context import _DefaultWorkflowState, _WorkflowOwner


def _cell_on_stack(shell: Any) -> bool:
    """Whether this shell's cell coroutine runs on the current stack.

    A task the cell creates runs on a stack of its own, without the cell
    coroutine, so a setting it makes stays local to it.
    """
    from IPython.core.interactiveshell import InteractiveShell

    return _shell_code_on_stack(
        shell, (InteractiveShell.run_cell_async.__code__,), inspect.currentframe()
    )


def _cell_running_in_thread(shell: Any, frame: FrameType) -> bool:
    """Whether another thread's stack, from *frame*, is running a cell of this shell.

    A terminal cell suspended at ``await`` keeps only IPython's synchronous
    runner on its thread's stack, so the runner counts as well as the cell
    coroutine.
    """
    from IPython.core.interactiveshell import InteractiveShell

    codes = tuple(
        method.__code__
        for method in (
            InteractiveShell.run_cell_async,
            getattr(InteractiveShell, "_run_cell", None),
        )
        if method is not None
    )
    return _shell_code_on_stack(shell, codes, frame)


def _shell_code_on_stack(shell: Any, codes: tuple[Any, ...], frame: FrameType | None) -> bool:
    """Whether a frame running one of *codes* for this shell is on the stack from *frame*."""
    try:
        while frame is not None:
            if frame.f_code in codes and frame.f_locals.get("self") is shell:
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
        if not _cell_on_stack(shell):
            return
        kernel_thread_id = _kernel_loop_thread_id(shell)
        if kernel_thread_id is not None:
            if kernel_thread_id != threading.get_ident():
                return
        # In-process and terminal shells can run without a kernel event loop.
        # A worker's nested cell leaves installation to the outer shell thread.
        elif threading.current_thread() is not threading.main_thread() and any(
            thread_id != threading.get_ident() and _cell_running_in_thread(shell, frame)
            for thread_id, frame in sys._current_frames().items()
        ):
            return
        adapter = _NotebookDefaultScope(shell)
        shell._probpipe_default_scope = adapter
    if not adapter._cell_active:
        from . import _context

        # Initial installation can happen after pre_execute fired.
        current = _context._DEFAULT_WORKFLOW_STATE.get()
        adapter.before_cell()
        _context._DEFAULT_WORKFLOW_STATE.set(current)


def _kernel_loop_thread_id(shell: Any) -> int | None:
    """The thread running the event loop of *shell*'s kernel, or ``None`` when that is unknown.

    A shell with no kernel, a kernel without an asyncio event loop, and a loop
    that is not running give ``None``.
    """
    kernel_loop = getattr(getattr(shell, "kernel", None), "io_loop", None)
    asyncio_loop = getattr(kernel_loop, "asyncio_loop", None)
    return getattr(asyncio_loop, "_thread_id", None)
