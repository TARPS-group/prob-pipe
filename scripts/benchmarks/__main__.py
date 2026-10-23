"""Run with ``python -m scripts.benchmarks`` from the repository root."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.benchmarks.cases import GROUPS, SCENARIOS, catalog
from scripts.benchmarks.runner import ROOT, compare, git_metadata, launch, request_for, worker_main


def main() -> int:
    """List, measure, or compare execution workloads."""
    if sys.argv[1:] == ["_worker"]:
        return worker_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "run", "compare"))
    parser.add_argument("files", nargs="*")
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--group", action="append", choices=GROUPS)
    parser.add_argument("--case", action="append", help="Exact case name from list")
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", help="JSONL path, or - for stdout")
    parser.add_argument(
        "--timeout", type=float, help="Worker time limit in seconds (default: unlimited)"
    )
    args = parser.parse_args()
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be finite and positive")
    defaults = (2, 5, 1) if args.suite == "quick" else (3, 20, 3)
    for name, default in zip(("warmups", "repeats", "processes"), defaults, strict=True):
        if getattr(args, name) is None:
            setattr(args, name, default)
        if getattr(args, name) < (0 if name == "warmups" else 1):
            parser.error(f"Invalid --{name}")
    if args.command == "compare":
        if len(args.files) != 2:
            parser.error("compare requires two JSONL files")
        for row in compare(*args.files):
            print(json.dumps(row, allow_nan=False))
        return 0
    cases = [
        c
        for c in catalog(args.suite)
        if (not args.group or c.group in args.group) and (not args.case or c.name in args.case)
    ]
    if not cases:
        parser.error("No cases matched")
    if args.case and set(args.case) - {c.name for c in cases}:
        parser.error("Unknown or excluded case name")
    if args.command == "list":
        from dataclasses import asdict

        for case in cases:
            print(json.dumps({"name": case.name, **asdict(case), "scenarios": case.scenarios}))
        return 0
    try:
        git_metadata()
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"Benchmark execution requires a Git checkout: {exc}")
    output = (
        args.output or ROOT / ".benchmark-results" / f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.jsonl"
    )
    if output == "-":
        stream = sys.stdout
    else:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("x")
        print(f"Results: {path}", file=sys.stderr)
    failed = False
    try:
        for case in cases:
            for scenario in args.scenario or case.scenarios:
                for i in range(args.processes):
                    print(
                        f"Running {case.name}: {scenario} ({i + 1}/{args.processes})",
                        file=sys.stderr,
                    )
                    row = launch(request_for(case, args, scenario, i))
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    failed |= row["status"] == "error"
    finally:
        if stream is not sys.stdout:
            stream.close()
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
