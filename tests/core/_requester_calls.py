"""The place in a call from which the engine asks for a set of evaluation points, for tests to build one there."""

from contextlib import contextmanager

from probpipe import function
from probpipe.core._call_context import _open_call_context, current_call_context
from probpipe.functions._execution import _requester_workflow_scope
from probpipe.values._function_base import _child_context_for_call


@contextmanager
def requester_call(function_view):
    """The engine's place in a call of *function_view*: the call's context open, with a key broker of its own."""
    child_context = _child_context_for_call(
        function_view, current_call_context(), entry_method_name="__call__"
    )
    with (
        _open_call_context(child_context, entry_method_name="__call__"),
        _requester_workflow_scope(),
    ):
        yield current_call_context()


def requester(**controls):
    """A Function to stand as the requester, with *controls* as its own."""
    requesting_function = function(lambda: None, label="requester")
    return requesting_function.using(**controls) if controls else requesting_function
