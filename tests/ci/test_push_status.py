"""The tracking issue's status, as plans/status.py writes it from the PRs' labels."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "plans" / "status.py"
if not _SCRIPT.exists():
    pytest.skip("no push is running: plans/status.py is absent", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("push_status", _SCRIPT)
status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(status)

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
BODY = """Intro.

## Waiting on the maintainer

- Something by hand.

## Steps

- [ ] **S0**: design text
- [ ] **S1**: one path reader — **draft** #1
- [ ] **S3**: the merge — not started
- [ ] **E8**: the JAX settings — **cut**
- [ ] Tutorials migrated
"""


def _pr(number, *labels, state="OPEN", draft=False, merged=None, base="dev/overhaul", title="t"):
    return {
        "number": number,
        "title": title,
        "state": state,
        "isDraft": draft,
        "mergedAt": merged,
        "baseRefName": base,
        "labels": [{"name": name} for name in labels],
    }


def _comment(body, created, number=1):
    return {
        "body": body,
        "createdAt": created,
        "url": f"https://example/issues/601#issuecomment-{number}",
    }


def _line(body, step):
    return next(line for line in body.splitlines() if f"**{step}**" in line)


class TestStepLines:
    def test_a_step_merged_into_the_base_is_ticked(self):
        prs = [_pr(5, "step:S0", state="MERGED", merged="2026-10-20T21:17:01Z")]
        body = status.render(BODY, prs, [], NOW)
        assert _line(body, "S0") == "- [x] **S0**: design text — **merged** #5 (2026-10-20)"

    def test_a_merge_into_another_branch_does_not_count(self):
        prs = [_pr(5, "step:S0", state="MERGED", merged="2026-10-20T21:17:01Z", base="dev/stacked")]
        assert (
            _line(status.render(BODY, prs, [], NOW), "S0")
            == "- [ ] **S0**: design text — not started"
        )

    def test_an_open_pr_beside_a_merged_one_leaves_the_box_empty(self):
        prs = [
            _pr(5, "step:S0", state="MERGED", merged="2026-10-20T21:17:01Z"),
            _pr(9, "step:S0"),
        ]
        line = _line(status.render(BODY, prs, [], NOW), "S0")
        assert line.startswith("- [ ]") and line.endswith(
            "**merged** #5 (2026-10-20); **in review** #9"
        )

    def test_the_previous_status_is_replaced_and_a_draft_is_marked(self):
        prs = [_pr(1, "step:S1", draft=True), _pr(2, "step:S3")]
        body = status.render(BODY, prs, [], NOW)
        assert _line(body, "S1") == "- [ ] **S1**: one path reader — **draft** #1"
        assert _line(body, "S3") == "- [ ] **S3**: the merge — **in review** #2"

    def test_a_cut_step_and_lines_without_a_step_are_kept(self):
        body = status.render(BODY, [_pr(3, "step:E8")], [], NOW)
        assert "- [ ] **E8**: the JAX settings — **cut**" in body
        assert "- [ ] Tutorials migrated" in body
        assert "- Something by hand." in body


class TestGeneratedSections:
    def test_only_comments_naming_a_plan_file_after_the_last_checkpoint_are_listed(self):
        prs = [_pr(7, "push:checkpoint", state="MERGED", merged="2026-10-20T23:00:00Z")]
        comments = [
            _comment("**Decided** for `decisions.md`.", "2026-10-20T22:00:00Z", 1),
            _comment("**Later decision**, for `decisions.md`.", "2026-10-21T01:00:00Z", 2),
            _comment("**Status only.**", "2026-10-21T02:00:00Z", 3),
            _comment("**Postponed**, for `deferred.md`.", "2026-10-21T03:00:00Z", 4),
        ]
        body = status.render(BODY, prs, comments, NOW)
        assert "(#7)" in body
        assert (
            "[Later decision, for `decisions.md`.](https://example/issues/601#issuecomment-2)"
            in body
        )
        assert "issuecomment-4" in body
        assert "issuecomment-1" not in body and "issuecomment-3" not in body

    def test_plans_prs_are_listed_and_closed_ones_left_out(self):
        prs = [
            _pr(
                11,
                "push:plans",
                state="MERGED",
                merged="2026-10-20T10:00:00Z",
                title="docs(plans): a",
            ),
            _pr(12, "push:plans", state="CLOSED", title="docs(plans): b"),
        ]
        body = status.render(BODY, prs, [], NOW)
        assert "- #11, merged 2026-10-20: docs(plans): a" in body
        assert "#12" not in body

    def test_rendering_twice_gives_the_same_body(self):
        prs = [
            _pr(5, "step:S0", state="MERGED", merged="2026-10-20T21:17:01Z"),
            _pr(11, "push:plans"),
        ]
        comments = [_comment("For `deferred.md`.", "2026-10-21T01:00:00Z")]
        once = status.render(BODY, prs, comments, NOW)
        assert status.render(once, prs, comments, NOW) == once
        assert once.endswith("_Written by `plans/status.py` at 2026-10-21 12:00 UTC._\n")
