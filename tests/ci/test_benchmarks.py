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
        assert kwargs["cwd"] == runner.ROOT
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
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import runpy
import sys
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.'):
            raise AssertionError('Listing must not import JAX')
sys.meta_path.insert(0, NoJax())
sys.argv = ['benchmarks', 'list', '--group', 'inference']
runpy.run_module('scripts.benchmarks', run_name='__main__')
""",
        ],
        cwd=runner.ROOT,
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


@pytest.mark.parametrize("status", ["missing", "not_applicable"])
def test_comparison_reports_absent_measurements(tmp_path, status):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(_record()) + "\n")
    row = {**_record(), "status": status, "reason": "no owner"}
    row.pop("samples")
    b.write_text("" if status == "missing" else json.dumps(row) + "\n")
    (result,) = runner.compare(a, b)
    assert result["status"] == status
    assert "phases" not in result


def test_unavailable_scenario_does_not_launch_worker(monkeypatch):
    from dataclasses import asdict

    case = next(c for c in catalog() if c.operation == "indices")
    monkeypatch.setattr(runner, "git_metadata", lambda: {})
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("worker started"))
    result = runner.launch({"case": asdict(case), "scenario": "new_object"})
    assert result["status"] == "not_applicable"
    assert result["reason"]
    assert "samples" not in result and "summary" not in result
    assert "new_object" not in case.scenarios


def test_worker_timeout_is_structured_error(monkeypatch):
    def run(command, **kwargs):
        assert kwargs["timeout"] == 0.1
        raise subprocess.TimeoutExpired(command, 0.1)

    monkeypatch.setattr(runner, "git_metadata", lambda: {})
    monkeypatch.setattr(runner.subprocess, "run", run)
    row = runner.launch({"device": "cpu", "timeout": 0.1})
    assert row["status"] == "error"
    assert row["error_type"] == "timeout"
    assert "samples" not in row


def test_real_worker_json_phases_and_compilation_counts():
    from dataclasses import asdict

    case = next(c for c in catalog() if c.operation == "indices")
    request = {
        "case": asdict(case),
        "device": "cpu",
        "scenario": "same_object",
        "warmups": 1,
        "repeats": 2,
        "seed": 2026,
        "timeout": 120,
    }
    row = runner.launch(request)
    assert row["status"] == "ok", row
    assert row["environment"]["backend"] == "cpu"
    assert [s["phase"] for s in row["samples"]] == ["first", "warmup", "measured", "measured"]
    assert row["samples"][0]["compilation_events"] > 0
    assert all(s["compilation_events"] == 0 for s in row["samples"][1:])
    assert all(s["milliseconds"] >= 0 for s in row["samples"])
    assert row["timing"]["result_synchronization"]
    assert json.loads(json.dumps(row, allow_nan=False)) == row


@pytest.mark.parametrize("case", catalog(), ids=lambda c: c.name)
def test_quick_workload_smoke(case):
    from dataclasses import replace

    import jax

    from scripts.benchmarks.cases import arrays, measurement_context, prepare, validate

    # Cover every quick workload, with tiny dimensions and one invocation.
    case = replace(case, size=2, n=8 if case.n else 0)
    with measurement_context(case):
        calls, _ = prepare(case, 1, "same_object", 2026)
        call, size = calls[0]
        result = call()
        jax.block_until_ready(arrays(result))
        validate(case, result, size)


def test_gradient_reference_context_covers_first_trace_and_restores(monkeypatch):
    from probpipe.inference import _minibatch
    from scripts.benchmarks import cases

    original = _minibatch._draw_indices
    traced = []
    reference = cases.permutation

    def sampling(*args, **kwargs):
        traced.append(True)
        return reference(*args, **kwargs)

    monkeypatch.setattr(cases, "permutation", sampling)
    case = cases.Case("minibatch", "gradient", 2, n=8, variant="permutation")
    with cases.measurement_context(case):
        calls, _ = cases.prepare(case, 3, "same_object", 2026)
        assert not traced
        for call, _ in calls:
            call().block_until_ready()
        assert len(traced) == 1
        assert _minibatch._draw_indices is sampling
    assert _minibatch._draw_indices is original


def test_cli_output_paths_and_git_preflight(monkeypatch, tmp_path):
    from scripts.benchmarks import __main__ as cli

    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "git_metadata", lambda: {})
    monkeypatch.setattr(cli, "launch", lambda request: {"status": "ok"})
    options = ["benchmarks", "run", "--group", "inference", "--scenario", "same_object"]
    monkeypatch.setattr(sys, "argv", options)
    assert cli.main() == 0
    assert len(list((root / ".benchmark-results").glob("*.jsonl"))) == 1
    assert not (tmp_path / ".benchmark-results").exists()
    monkeypatch.setattr(sys, "argv", [*options, "--output", "explicit.jsonl"])
    assert cli.main() == 0
    assert (tmp_path / "explicit.jsonl").exists()

    def no_git():
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(cli, "git_metadata", no_git)
    monkeypatch.setattr(sys, "argv", [*options, "--output", "absent.jsonl"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert not (tmp_path / "absent.jsonl").exists()


def test_cli_default_skips_inapplicable_scenarios(monkeypatch, capsys):
    from scripts.benchmarks import __main__ as cli

    requests = []
    monkeypatch.setattr(cli, "git_metadata", lambda: {})

    def launch(request):
        requests.append(request)
        return {"status": "ok"}

    monkeypatch.setattr(cli, "launch", launch)
    case = next(c for c in catalog() if c.operation == "indices")
    monkeypatch.setattr(
        sys, "argv", ["benchmarks", "run", "--case", case.name, "--output", "-", "--warmups", "0"]
    )
    assert cli.main() == 0
    assert [r["scenario"] for r in requests] == list(case.scenarios)
    assert all(r["warmups"] == 0 for r in requests)
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_worker_context_and_validation_surround_timed_calls(monkeypatch):
    from dataclasses import asdict

    import jax
    import numpy as np

    from scripts.benchmarks.cases import Case

    events = []

    @contextmanager
    def context(case):
        events.append("enter")
        yield
        events.append("exit")

    def prepare(*args):
        events.append("prepare")

        def call():
            events.append("call")
            return np.ones(2)

        return [(call, 2)] * 3, [1.0]

    def measure(call, synchronize, counter):
        events.append("timer_start")
        result = call()
        synchronize(result)
        events.append("timer_end")
        return result, {"milliseconds": 1, "compilation_events": 0}

    monkeypatch.setattr(runner, "measurement_context", context)
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "measure", measure)
    monkeypatch.setattr(runner, "validate", lambda *args: events.append("validate"))
    cache_enabled = jax.config.jax_enable_compilation_cache
    try:
        runner.worker(
            {
                "case": asdict(Case("family", "sample", 2)),
                "device": "cpu",
                "scenario": "same_object",
                "seed": 0,
                "warmups": 1,
                "repeats": 1,
            }
        )
    finally:
        jax.config.update("jax_enable_compilation_cache", cache_enabled)
    assert events == ["enter", "prepare"] + ["timer_start", "call", "timer_end"] * 3 + [
        "validate"
    ] * 3 + ["exit"]
