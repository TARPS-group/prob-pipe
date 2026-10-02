"""Format the Python file that a Claude Code edit wrote.

Claude Code runs this script after each Edit, MultiEdit, or Write, with the tool
call as JSON on standard input. A Python file of the repository is formatted by
the ``ruff-format`` hook of ``.pre-commit-config.yaml``, so the formatter is the
ruff version that file pins. The script always exits with status 0, so an
editor without ``pre-commit`` keeps the file as the edit wrote it.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    try:
        call = json.load(sys.stdin)
    except ValueError:
        return 0
    path = Path(call.get("tool_input", {}).get("file_path", "")).resolve()
    if path.suffix != ".py" or not path.is_file() or not path.is_relative_to(ROOT):
        return 0
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(
            ["pre-commit", "run", "ruff-format", "--files", str(path)],
            cwd=ROOT,
            capture_output=True,
            check=False,
            timeout=120,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
