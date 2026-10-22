"""Process-isolated execution measurements and structured comparisons."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from scripts.benchmarks.cases import Case, arrays, prepare, validate

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


def summary(samples):
    """Summarize measured calls while retaining their raw observations."""
    times = sorted(item["milliseconds"] for item in samples)
    quartiles = (
        statistics.quantiles(times, n=4, method="inclusive") if len(times) > 1 else times * 3
    )
    return {
        "median_ms": statistics.median(times),
        "q1_ms": quartiles[0],
        "q3_ms": quartiles[2],
        "compilation_events": sum(item["compilation_events"] for item in samples),
    }


def measure(call, synchronize, counter):
    """Time exactly one invocation and completion, including its compile events."""
    with counter() as events:
        start = time.perf_counter()
        result = call()
        synchronize(result)
        elapsed = (time.perf_counter() - start) * 1000
    return result, {"milliseconds": elapsed, "compilation_events": events.count}


def git_metadata():
    """Identify the checkout without changing its files or refs."""

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def worker(request):
    """Measure one workload and scenario in the caller's fresh process."""
    import jax

    from tests._compilation import count_compilations

    jax.config.update("jax_enable_compilation_cache", False)
    backend = "gpu" if request["device"] == "cuda" else "cpu"
    device = jax.devices(backend)[0]
    if request["device"] == "cuda" and "cuda" not in device.client.platform_version.lower():
        raise RuntimeError("CUDA was requested, but the selected GPU backend is not CUDA")
    # Import workloads before timing preparation, without executing them.
    for module in (
        "probpipe",
        "probpipe.inference._blackjax_sgmcmc",
        "probpipe.inference._minibatch",
    ):
        importlib.import_module(module)
    case = Case(**request["case"])
    count = 1 + request["warmups"] + request["repeats"]
    with jax.default_device(device):
        start = time.perf_counter()
        calls, construction_ms = prepare(case, count, request["scenario"], request["seed"])
        setup_ms = (time.perf_counter() - start) * 1000
        records = []
        # Validation waits until all timed calls finish, to avoid compiling
        # helper operations between the cold and warm measurements.
        outputs = []
        for i, (call, size) in enumerate(calls):
            result, record = measure(
                call, lambda r: jax.block_until_ready(arrays(r)), count_compilations
            )
            record.update(
                index=i,
                size=size,
                phase="first" if i == 0 else ("warmup" if i <= request["warmups"] else "measured"),
            )
            records.append(record)
            outputs.append((result, size))
        for result, size in outputs:
            validate(case, result, size)
    env = {
        "system": platform.system(),
        "host": platform.node(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "device": device.device_kind,
        "backend": device.platform,
        "platform_version": device.client.platform_version,
        "x64": jax.config.jax_enable_x64,
        "persistent_cache": False,
        "packages": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "blackjax")},
        "jax_flags": {
            k: v for k, v in os.environ.items() if k.startswith(("JAX_", "XLA_", "CUDA_"))
        },
    }
    return {
        "environment": env,
        "setup_ms": setup_ms,
        "owner_preparation_ms": construction_ms,
        "samples": records,
        "summary": summary([r for r in records if r["phase"] == "measured"]),
        "timing": {
            "external_key_preparation": False,
            "input_placement": False,
            "model_construction": False,
            "result_synchronization": True,
            "compile_counter": "/jax/core/compile/backend_compile_duration",
        },
    }


def launch(request):
    """Run one independent child and turn failures into structured records."""
    record = {"schema_version": SCHEMA_VERSION, **request, **git_metadata()}
    env = dict(
        os.environ,
        JAX_ENABLE_COMPILATION_CACHE="false",
        JAX_PLATFORMS="cuda" if request["device"] == "cuda" else "cpu",
    )
    command = [sys.executable, "-m", "scripts.benchmarks", "_worker"]
    process = subprocess.run(
        command, input=json.dumps(request), text=True, capture_output=True, cwd=ROOT, env=env
    )
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    try:
        payload = json.loads(process.stdout)
    except (ValueError, TypeError):
        payload = {"status": "error", "error": process.stdout or "Worker returned no JSON"}
    record.update(payload)
    if process.returncode:
        record.update(status="error", exit_code=process.returncode)
    return record


def worker_main():
    """Reserve stdout for a single JSON object, even if libraries print."""
    request = json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            payload = {"status": "ok", **worker(request)}
    except Exception as exc:
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(payload, allow_nan=False))
    return int(payload["status"] != "ok")


def compare(before, after):
    """Compare aligned measurements only when their conditions match."""

    def grouped(path):
        groups = {}
        with open(path) as stream:
            for line in stream:
                row = json.loads(line)
                case = row["case"]
                identity = {k: v for k, v in case.items() if k != "version"}
                key = json.dumps([identity, row["scenario"]], sort_keys=True)
                groups.setdefault(key, []).append(row)
        return groups

    left, right = grouped(before), grouped(after)
    results = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key, []), right.get(key, [])
        row = {"case": (a or b)[0]["case"], "scenario": (a or b)[0]["scenario"]}
        if not a or not b:
            results.append({**row, "status": "missing"})
            continue
        if any(r["status"] != "ok" for r in a + b):
            results.append({**row, "status": "error"})
            continue
        conditions = [
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "schema_version",
                        "environment",
                        "case",
                        "warmups",
                        "repeats",
                        "seed",
                        "timing",
                    )
                },
                sort_keys=True,
            )
            for r in a + b
        ]
        if len(set(conditions)) != 1:
            results.append({**row, "status": "incompatible"})
            continue
        phases = {}
        for phase in ("first", "warmup", "measured"):
            sa = [s for r in a for s in r["samples"] if s["phase"] == phase]
            sb = [s for r in b for s in r["samples"] if s["phase"] == phase]
            if not sa or not sb:
                continue
            aa, bb = summary(sa), summary(sb)
            phases[phase] = {
                "before": aa,
                "after": bb,
                "time_ratio": bb["median_ms"] / aa["median_ms"] if aa["median_ms"] else None,
                "compilation_events_per_call_delta": bb["compilation_events"] / len(sb)
                - aa["compilation_events"] / len(sa),
            }
        results.append(
            {
                **row,
                "status": "comparable",
                "phases": phases,
                "before_commits": sorted({r["commit"] for r in a}),
                "after_commits": sorted({r["commit"] for r in b}),
            }
        )
    return results


def request_for(case, args, scenario, process_index):
    return {
        "case": asdict(case),
        "scenario": scenario,
        "device": args.device,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "process_index": process_index,
    }
