"""Notebook defaults persist across cells without authorizing user child tasks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SETUP = """
import asyncio, numpy as np
from probpipe import workflow_run, sample, Normal, replay_run, UnmanagedConcurrentWorkflowEntryError
def draw(): return sample(Normal('x',0.,1.), sample_shape=(4,))
with workflow_run(17): expected = [np.asarray(draw()) for _ in range(16)]
position = 0
def next_draw():
    global position
    r = draw(); np.testing.assert_array_equal(r, expected[position]); position += 1
    return r
"""

# Each entry is a real kernel execution request, not a simulated task switch.
_CELLS = [
    (
        "setup-and-first-use",
        _SETUP + "workflow_run.set_default(seed=17)\na = next_draw()\n",
        None,
    ),
    (
        "ordinary-next-cell",
        """b = next_draw()
""",
        None,
    ),
    (
        "top-level-await",
        """await asyncio.sleep(0)
c = next_draw()
""",
        None,
    ),
    (
        "normal-after-await",
        """d = next_draw()
""",
        None,
    ),
    (
        "copied-context-set",
        """
from contextvars import copy_context
copy_context().run(workflow_run.set_default, seed=99)
next_draw()
""",
        None,
    ),
    ("after-copied-context-set", "next_draw()", None),
    (
        "copied-context-clear",
        """
copy_context().run(workflow_run.clear_default)
next_draw()
""",
        None,
    ),
    ("after-copied-context-clear", "next_draw()", None),
    (
        "concurrent-children",
        """
async def rejected():
    try: draw()
    except UnmanagedConcurrentWorkflowEntryError: return True
    raise AssertionError('child used parent stream')
assert await asyncio.gather(rejected(), rejected()) == [True, True]
e = next_draw()
""",
        None,
    ),
    (
        "to-thread-copy",
        """
try: await asyncio.to_thread(draw)
except UnmanagedConcurrentWorkflowEntryError: pass
else: raise AssertionError('thread used parent stream')
f = next_draw()
""",
        None,
    ),
    (
        "child-local-default",
        """
async def independent():
    workflow_run.set_default(seed=17)
    np.testing.assert_array_equal(draw(), expected[0])
    workflow_run.clear_default()
await asyncio.create_task(independent())
g = next_draw()
""",
        None,
    ),
    (
        "child-spans-cells-start",
        """
gate = asyncio.Event()
async def later():
    await gate.wait()
    return await rejected()
pending = asyncio.create_task(later())
await asyncio.sleep(0)
""",
        None,
    ),
    (
        "child-spans-cells-finish",
        """
gate.set()
assert await pending
h = next_draw()
""",
        None,
    ),
    (
        "explicit-and-replay",
        """
with workflow_run(17): np.testing.assert_array_equal(draw(), expected[0])
with replay_run(a.provenance): np.testing.assert_array_equal(draw(), a)
i = next_draw()
""",
        None,
    ),
    (
        "reset-in-await-cell",
        """
await asyncio.sleep(0)
workflow_run.set_default(seed=17)
position = 0
next_draw()
""",
        None,
    ),
    (
        "after-reset",
        """next_draw()
""",
        None,
    ),
    (
        "clear-in-await-cell",
        """
await asyncio.sleep(0)
workflow_run.clear_default()
""",
        None,
    ),
    (
        "after-clear",
        """
from probpipe.functions import _context
assert _context._DEFAULT_WORKFLOW_STATE.get() is None
from unittest.mock import patch
with patch.object(_context, '_os_urandom', return_value=bytes(8)) as entropy:
    draw(); draw()
    assert entropy.call_count == 2
""",
        None,
    ),
    (
        "anonymous-set",
        """
workflow_run.set_default()
with patch.object(_context, '_os_urandom', return_value=bytes(8)) as entropy:
    draw()
    assert entropy.call_count == 1
anonymous_state = _context._DEFAULT_WORKFLOW_STATE.get()
""",
        None,
    ),
    (
        "anonymous-continue-await",
        """
await asyncio.sleep(0)
with patch.object(_context, '_os_urandom', side_effect=AssertionError('new entropy')):
    draw()
assert _context._DEFAULT_WORKFLOW_STATE.get() is anonymous_state
""",
        None,
    ),
    (
        "exception",
        """draw(); raise ValueError('cell failed')
""",
        "ValueError",
    ),
    (
        "after-exception",
        """assert _context._DEFAULT_WORKFLOW_STATE.get() is anonymous_state
draw()
""",
        None,
    ),
    (
        "cancelled",
        """await asyncio.sleep(0); draw(); raise asyncio.CancelledError()
""",
        "CancelledError",
    ),
    (
        "after-cancelled",
        """await asyncio.sleep(0)
assert _context._DEFAULT_WORKFLOW_STATE.get() is anonymous_state
draw()
""",
        None,
    ),
    (
        "nested-cell",
        """get_ipython().run_cell('draw()')
draw()
""",
        None,
    ),
    (
        "after-nested-cell",
        """assert _context._DEFAULT_WORKFLOW_STATE.get() is anonymous_state
draw()
""",
        None,
    ),
    (
        "silent-cell",
        "assert _context._DEFAULT_WORKFLOW_STATE.get() is anonymous_state; draw()",
        None,
    ),
    (
        "after-silent-cell",
        "assert _context._DEFAULT_WORKFLOW_STATE.get() is anonymous_state; draw()",
        None,
    ),
]


@pytest.fixture
def kernel_client():
    jupyter = pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    manager = jupyter.KernelManager()
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    manager.start_kernel(cwd=root, env=env)
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=90)
        yield client
    finally:
        client.stop_channels()
        manager.shutdown_kernel()


def _execute_cell(client, code, expected_error=None, *, silent=False):
    message_id = client.execute(code, silent=silent)
    errors = []
    while True:
        message = client.get_iopub_msg(timeout=90)
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        content = message["content"]
        if message["msg_type"] == "error":
            errors.append(content)
        if message["msg_type"] == "status" and content["execution_state"] == "idle":
            break
    assert [error["ename"] for error in errors] == (
        [] if expected_error is None else [expected_error]
    ), (code, errors)


@pytest.mark.parametrize("first_cell", ["ordinary", "await", "child"])
def test_default_stream_across_kernel_cells(kernel_client, first_cell):
    for index, (name, code, expected_error) in enumerate(_CELLS):
        if index == 0 and first_cell == "await":
            code = "import asyncio\nawait asyncio.sleep(0)\n" + code
        elif index == 0 and first_cell == "child":
            # The first default is local to the child, before cell hooks exist.
            code = code.replace(
                "workflow_run.set_default(seed=17)",
                "async def first_child():\n"
                "    workflow_run.set_default(seed=99)\n"
                "    draw()\n"
                "await asyncio.create_task(first_child())\n"
                "from probpipe.functions import _context\n"
                "assert _context._DEFAULT_WORKFLOW_STATE.get() is None\n"
                "workflow_run.set_default(seed=17)",
                1,
            )
        _execute_cell(kernel_client, code, expected_error, silent=name == "silent-cell")


@pytest.mark.parametrize(
    "first_cell",
    [
        pytest.param(
            """
from contextvars import copy_context
copy_context().run(workflow_run.set_default, seed=99)
""",
            id="copied-context",
        ),
        pytest.param(
            """
import threading
errors = []
def background():
    result = get_ipython().run_cell('workflow_run.set_default(seed=99)')
    errors.append(result.error_in_exec)
worker = threading.Thread(target=background)
worker.start()
worker.join()
assert errors == [None]
""",
            id="background-thread",
        ),
        pytest.param(
            """
result = await asyncio.to_thread(
    get_ipython().run_cell, 'workflow_run.set_default(seed=99)'
)
assert result.error_before_exec is None
assert result.error_in_exec is None
""",
            id="async-background-thread",
        ),
        pytest.param(
            """
from types import MethodType
shell = get_ipython()
original_run_cell_async = shell.run_cell_async
async def wrapped(self, *args, **kwargs):
    return await original_run_cell_async(*args, **kwargs)
shell.run_cell_async = MethodType(wrapped, shell)
""",
            id="wrapped-cell-coroutine",
        ),
    ],
)
def test_first_default_stays_in_its_context(kernel_client, first_cell):
    _execute_cell(kernel_client, _SETUP + first_cell)
    _execute_cell(
        kernel_client,
        """
from probpipe.functions import _context
assert _context._DEFAULT_WORKFLOW_STATE.get() is None
workflow_run.set_default(seed=17)
next_draw()
""",
    )
    _execute_cell(kernel_client, "await asyncio.sleep(0)\nnext_draw()")


@pytest.mark.parametrize("first_cell", ["ordinary", "await"])
def test_default_stream_across_terminal_cells(first_cell, tmp_path):
    pytest.importorskip("IPython")
    script = """
import sys
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from tests.functions.test_notebook_scope import _CELLS

shell = TerminalInteractiveShell.instance()
for index, (name, code, expected_error) in enumerate(_CELLS):
    if index == 0 and sys.argv[1] == 'await':
        code = 'import asyncio\\nawait asyncio.sleep(0)\\n' + code
    result = shell.run_cell(code, silent=name == 'silent-cell')
    error = result.error_before_exec or result.error_in_exec
    assert (None if error is None else type(error).__name__) == expected_error, (name, error)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_cell],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ | {"IPYTHONDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("first_default", ["shell", "worker", "async-worker"])
def test_default_stream_in_application_thread(first_default, tmp_path):
    pytest.importorskip("ipykernel")
    script = """
import sys
from concurrent.futures import ThreadPoolExecutor
from ipykernel.inprocess.manager import InProcessKernelManager
from tests.functions.test_notebook_scope import _CELLS

def run_cells():
    manager = InProcessKernelManager()
    manager.start_kernel()
    try:
        for index, (name, code, expected_error) in enumerate(_CELLS):
            if index == 0:
                code = 'import asyncio\\nawait asyncio.sleep(0)\\n' + code
                if sys.argv[1] != 'shell':
                    code = code.replace(
                        'workflow_run.set_default(seed=17)',
                        "import threading\\n"
                        "worker_errors = []\\n"
                        "def configure_worker():\\n"
                        "    result = get_ipython().run_cell('workflow_run.set_default(seed=99)')\\n"
                        "    worker_errors.append(result.error_before_exec or result.error_in_exec)\\n"
                        "worker = threading.Thread(target=configure_worker)\\n"
                        "worker.start()\\nworker.join()\\n"
                        "assert worker_errors == [None]\\n"
                        "workflow_run.set_default(seed=17)",
                        1,
                    )
                    if sys.argv[1] == 'async-worker':
                        code = code.replace(
                            'worker.start()\\nworker.join()',
                            'await asyncio.to_thread(configure_worker)',
                        )
            result = manager.kernel.shell.run_cell(code, silent=name == 'silent-cell')
            error = result.error_before_exec or result.error_in_exec
            assert (None if error is None else type(error).__name__) == expected_error, (name, error)
    finally:
        manager.shutdown_kernel()

with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(run_cells).result(timeout=90)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_default],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ | {"IPYTHONDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_subshell_completion_preserves_main_default(kernel_client):
    ipykernel = pytest.importorskip("ipykernel")
    if int(ipykernel.__version__.split(".")[0]) < 7:
        pytest.skip("kernel subshells require ipykernel 7")
    _execute_cell(
        kernel_client,
        _SETUP
        + """
import threading
ready = threading.Event()
finished = threading.Event()
interleave_completion = False

def wait_for_subshell():
    global interleave_completion
    if threading.current_thread() is threading.main_thread() and interleave_completion:
        interleave_completion = False
        ready.set()
        assert finished.wait(30), 'subshell did not finish'
    elif threading.current_thread() is not threading.main_thread() and ready.is_set():
        finished.set()

# Register before the adapter so the subshell completes before its cleanup.
get_ipython().events.register('post_execute', wait_for_subshell)
workflow_run.set_default(seed=17)
next_draw()
""",
    )
    request = kernel_client.session.send(
        kernel_client.control_channel.socket, "create_subshell_request", content={}
    )
    while True:
        reply = kernel_client.get_control_msg(timeout=90)
        if reply.get("parent_header", {}).get("msg_id") == request["msg_id"]:
            subshell_id = reply["content"]["subshell_id"]
            break
    main_id = kernel_client.execute("interleave_completion = True\nnext_draw()")
    subshell_request = kernel_client.session.msg(
        "execute_request",
        content={
            "code": "assert ready.wait(30)\npass",
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
    )
    subshell_request["header"]["subshell_id"] = subshell_id
    kernel_client.shell_channel.send(subshell_request)
    pending = {main_id, subshell_request["msg_id"]}
    errors = []
    while pending:
        message = kernel_client.get_iopub_msg(timeout=90)
        parent_id = message.get("parent_header", {}).get("msg_id")
        if parent_id not in pending:
            continue
        if message["msg_type"] == "error":
            errors.append(message["content"])
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            pending.remove(parent_id)
    assert not errors
    _execute_cell(kernel_client, "next_draw()")
