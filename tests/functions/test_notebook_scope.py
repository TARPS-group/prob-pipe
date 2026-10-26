"""Notebook defaults persist across cells without authorizing user child tasks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Each entry is a real kernel execution request, not a simulated task switch.
_CELLS = [
    (
        "setup-and-first-use",
        """
import asyncio, numpy as np
from probpipe import workflow_run, sample, Normal, replay_run, UnmanagedConcurrentWorkflowEntryError
def draw(): return sample(Normal('x',0.,1.), sample_shape=(4,))
with workflow_run(17): expected = [np.asarray(draw()) for _ in range(12)]
position = 0
def next_draw():
    global position
    r = draw(); np.testing.assert_array_equal(r, expected[position]); position += 1
    return r
workflow_run.set_default(seed=17)
a = next_draw()
""",
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


@pytest.mark.parametrize("first_cell", ["ordinary", "await", "child"])
def test_default_stream_across_kernel_cells(first_cell):
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
        for index, (name, code, expected_error) in enumerate(_CELLS):
            if index == 0 and first_cell == "await":
                code = "import asyncio\nawait asyncio.sleep(0)\n" + code
            elif index == 0 and first_cell == "child":
                # The first default is installed in a child. It must not become
                # the notebook default, even before the cell hooks exist.
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
            message_id = client.execute(code, silent=name == "silent-cell")
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
            ), (name, errors)
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
