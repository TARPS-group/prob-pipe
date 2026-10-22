"""Compatibility entry point for the minibatch execution benchmarks.

Use ``python -m scripts.benchmarks run --group minibatch`` for the quick set.
This older entry point selects the full minibatch matrix and accepts ``gpu``
as an alias for ``cuda``. Results use the common JSONL format and local output
policy described in the benchmark README.
"""

from __future__ import annotations

import sys

from scripts.benchmarks.__main__ import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    for i, argument in enumerate(arguments):
        if argument == "--device=gpu":
            arguments[i] = "--device=cuda"
        elif argument == "gpu" and i and arguments[i - 1] == "--device":
            arguments[i] = "cuda"
    sys.argv = [sys.argv[0], "run", "--group", "minibatch", "--suite", "full", *arguments]
    sys.exit(main())
