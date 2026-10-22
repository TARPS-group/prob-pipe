"""Measurement boundaries, process isolation, and comparable benchmark records."""

import json
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts.benchmarks import runner
from scripts.benchmarks.cases import GROUPS, catalog


def test_catalog_covers_all_groups_and_full_set_contains_quick():
    quick, full = catalog(), catalog("full")
    assert {c.group for c in quick} == set(GROUPS)
    assert {c.name for c in quick} <= {c.name for c in full}
    assert len({c.name for c in full}) == len(full)


def test_measure_waits_inside_counter_without_repeating_call():
    events = []

    @contextmanager
    def counter():
        events.append("start")
        yield SimpleNamespace(count=2)
        events.append("end")

    def call():
        events.append("call")
        return "pending"

    def sync(result):
        assert result == "pending"
        events.append("sync")

    result, sample = runner.measure(call, sync, counter)
    assert result == "pending"
    assert events == ["start", "call", "sync", "end"]
    assert sample["compilation_events"] == 2
    assert sample["milliseconds"] >= 0


def test_launch_uses_new_interpreter_explicit_backend_and_cache_policy(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"ok"}', stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "git_metadata", lambda: {"commit": "abc", "dirty": False})
    for _ in range(2):
        assert runner.launch({"device": "cpu"})["status"] == "ok"
    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[0] == sys.executable
        assert command[-1] == "_worker"
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
        assert kwargs["env"]["JAX_ENABLE_COMPILATION_CACHE"] == "false"
        assert json.loads(kwargs["input"]) == {"device": "cpu"}


def test_worker_failure_is_not_a_timing_result(monkeypatch):
    monkeypatch.setattr(runner, "git_metadata", lambda: {})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="device unavailable\n"),
    )
    row = runner.launch({"device": "cuda"})
    assert row["status"] == "error"
    assert "samples" not in row
    assert row["exit_code"] == 1


def _record(time=1.0):
    return {
        "schema_version": 1,
        "case": {"group": "family", "version": 1},
        "scenario": "same_object",
        "status": "ok",
        "environment": {"device": "cpu"},
        "warmups": 0,
        "repeats": 1,
        "seed": 0,
        "timing": {},
        "commit": "abc",
        "samples": [{"phase": "measured", "milliseconds": time, "compilation_events": 0}],
    }


@pytest.mark.parametrize("change", ["environment", "version", "seed", "error", "none"])
def test_comparison_refuses_incompatible_conditions(tmp_path, change):
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    a, b = _record(), _record(2.0)
    if change == "environment":
        b["environment"]["device"] = "gpu"
    elif change == "version":
        b["case"]["version"] = 2
    elif change == "seed":
        b["seed"] = 1
    elif change == "error":
        b["status"] = "error"
    before.write_text(json.dumps(a) + "\n")
    after.write_text(json.dumps(b) + "\n")
    (result,) = runner.compare(before, after)
    if change == "none":
        assert result["status"] == "comparable"
        assert result["phases"]["measured"]["time_ratio"] == 2
    else:
        assert result["status"] == ("error" if change == "error" else "incompatible")
        assert "phases" not in result


def test_listing_is_json_and_does_not_require_device():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.benchmarks", "list", "--group", "inference"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1
    assert rows[0]["operation"] == "rwmh"


def test_preparation_never_calls_operation_and_reuses_structural_owners(monkeypatch):
    from scripts.benchmarks import cases

    built, executed = [], []

    def build(case, size, shift):
        built.append(size)

        def call(*args):
            executed.append(size)
            return size

        return call

    monkeypatch.setattr(cases, "build", build)
    case = cases.Case("family", "sample", 32, variant="normal")
    calls, construction = cases.prepare(case, 4, "structure_change", 2026)
    assert executed == []
    assert built == [32, 33]
    assert len(construction) == 2
    assert [call() for call, size in calls] == [32, 33, 32, 33]


def test_conditional_density_keeps_the_declared_record_order():
    from scripts.benchmarks.cases import Case, prepare, validate

    case = Case("joint", "density", 2, variant="conditional")
    calls, _ = prepare(case, 1, "same_object", 2026)
    call, size = calls[0]
    validate(case, call(), size)
