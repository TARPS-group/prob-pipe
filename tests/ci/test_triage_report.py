"""The weekly triage report, as scripts/triage/report.py writes it from canned ``gh`` JSON."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "triage" / "report.py"
_spec = importlib.util.spec_from_file_location("triage_report", _SCRIPT)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

NOW = datetime(2026, 10, 26, 11, 0, tzinfo=UTC)
RECENT = "2026-10-20T12:00:00Z"
OLD = "2026-06-01T12:00:00Z"


def _issue(number, *labels, milestone=None, updated=RECENT, created=None, title="t"):
    """An issue as ``gh issue list --json`` returns it."""
    return {
        "number": number,
        "title": title,
        "labels": [{"name": name} for name in labels],
        "milestone": {"number": 1, "title": milestone, "dueOn": None} if milestone else None,
        "createdAt": created or f"2026-01-{number:02d}T00:00:00Z",
        "updatedAt": updated,
    }


def _milestone(title, due):
    """A milestone as ``gh api repos/{owner}/{repo}/milestones`` returns it."""
    return {"title": title, "due_on": due, "html_url": f"https://example/milestone/{title}"}


TRIAGED_BUG = ("type:bug", "severity:S3", "priority:P2", "area:core")


class TestMissing:
    def test_a_triaged_bug_misses_nothing(self):
        assert report.missing(_issue(1, *TRIAGED_BUG)) == []

    @pytest.mark.parametrize(
        ("labels", "gap"),
        [
            (("priority:P2", "area:core"), "no `type:` label"),
            (("type:bug", "priority:P2", "area:core"), "a bug without a severity"),
            (("type:feature", "area:core"), "no priority"),
            (("type:docs", "priority:P3"), "no `area:` label"),
            (("type:feature", "priority:P1", "area:core"), "a P0 or P1 issue without a milestone"),
        ],
    )
    def test_each_required_label_is_reported_missing(self, labels, gap):
        assert report.missing(_issue(1, *labels)) == [gap]

    @pytest.mark.parametrize("kind", ["type:tracking", "type:question"])
    def test_a_tracking_issue_or_a_question_needs_no_priority(self, kind):
        assert report.missing(_issue(1, kind, "area:docs")) == []

    def test_a_p0_with_a_milestone_misses_nothing(self):
        issue = _issue(1, "type:bug", "severity:S1", "priority:P0", "area:core", milestone="0.2.1")
        assert report.missing(issue) == []


class TestViolations:
    def test_two_type_labels_are_a_violation(self):
        issue = _issue(1, "type:bug", "type:docs", "severity:S3", "priority:P2", "area:core")
        assert report.violations(issue) == ["2 `type:` labels: type:bug, type:docs"]

    def test_a_p3_with_a_milestone_is_a_violation(self):
        issue = _issue(1, "type:feature", "priority:P3", "area:core", milestone="0.3.0")
        assert report.violations(issue) == ["a P3 issue with a milestone"]

    def test_a_severity_on_a_feature_is_a_violation(self):
        issue = _issue(1, "type:feature", "severity:S2", "priority:P2", "area:core")
        assert report.violations(issue) == ["a severity on an issue that is not a bug"]


class TestSections:
    def test_an_issue_labeled_needs_triage_is_untriaged_even_when_complete(self):
        issues = [_issue(1, *TRIAGED_BUG, "status:needs-triage"), _issue(2, *TRIAGED_BUG)]
        assert [f.issue["number"] for f in report.untriaged(issues)] == [1]

    def test_an_issue_missing_a_label_is_untriaged_and_not_a_violation(self):
        issues = [_issue(1, "type:bug", "type:docs", "area:core")]
        (finding,) = report.untriaged(issues)
        assert finding.problems == (
            "a bug without a severity",
            "no priority",
            "2 `type:` labels: type:bug, type:docs",
        )
        assert report.rule_violations(issues) == []

    def test_a_triaged_issue_that_breaks_a_rule_is_a_violation(self):
        issues = [_issue(1, "type:docs", "priority:P3", "area:docs", milestone="0.3.0")]
        assert report.untriaged(issues) == []
        (finding,) = report.rule_violations(issues)
        assert finding.problems == ("a P3 issue with a milestone",)


class TestRetriage:
    def test_a_p1_or_p2_without_activity_for_90_days_is_due(self):
        issues = [
            _issue(1, *TRIAGED_BUG, updated=OLD),
            _issue(2, *TRIAGED_BUG),
            _issue(3, "type:feature", "priority:P3", "area:core", updated=OLD),
        ]
        (finding,) = report.due_for_retriage(issues, [], NOW)
        assert finding.issue["number"] == 1
        assert finding.problems == ("no activity for 146 days",)

    def test_the_open_issues_of_a_milestone_due_within_a_week_are_due(self):
        milestones = [_milestone("0.2.1", "2026-10-30T00:00:00Z"), _milestone("0.3.0", None)]
        issues = [
            _issue(1, "type:feature", "priority:P2", "area:core", milestone="0.2.1"),
            _issue(2, "type:feature", "priority:P2", "area:core", milestone="0.3.0"),
        ]
        (finding,) = report.due_for_retriage(issues, milestones, NOW)
        assert finding.issue["number"] == 1
        assert finding.problems == ("open in 0.2.1, due 2026-10-30",)

    def test_an_issue_due_for_two_reasons_is_listed_once(self):
        milestones = [_milestone("0.2.1", "2026-10-20T00:00:00Z")]
        issues = [_issue(1, *TRIAGED_BUG, milestone="0.2.1", updated=OLD)]
        (finding,) = report.due_for_retriage(issues, milestones, NOW)
        assert len(finding.problems) == 2

    def test_an_issue_in_the_triage_queue_is_left_out(self):
        issues = [_issue(1, *TRIAGED_BUG, "status:needs-triage", updated=OLD)]
        assert report.due_for_retriage(issues, [], NOW) == []


MILESTONES = (_milestone("0.3.0", "2026-12-18T00:00:00Z"), _milestone("0.2.1", None))
ISSUES = (
    _issue(1, "type:bug", "severity:S1", "priority:P0", "area:core", milestone="0.2.1"),
    _issue(2, "type:feature", "priority:P1", "area:core", milestone="0.2.1", title="A P1"),
    _issue(3, "type:feature", "priority:P2", "area:core", milestone="0.3.0"),
    _issue(4, "priority:P2", title="Unlabeled"),
)


class TestRender:
    def test_the_release_gate_lists_the_blockers_and_the_p1_issues(self):
        body = report.render(list(ISSUES), list(MILESTONES), NOW)
        gate = body[body.index("### [0.2.1]") :]
        assert "2 open issues, **blocked**: 1 P0 or S1, 1 P1 to close or move." in gate
        assert "- #1 t" in gate and "- #2 A P1" in gate

    def test_a_milestone_without_blockers_is_clear(self):
        body = report.render(list(ISSUES), list(MILESTONES), NOW)
        assert "1 open issues, **clear**." in body[: body.index("### [0.2.1]")]

    def test_milestones_with_a_due_date_come_first(self):
        body = report.render(list(ISSUES), list(MILESTONES), NOW)
        assert body.index("### [0.3.0]") < body.index("### [0.2.1]")

    def test_the_body_lists_untriaged_issues_and_ends_with_the_footer(self):
        body = report.render(list(ISSUES), list(MILESTONES), NOW)
        untriaged = body[body.index("## Untriaged") : body.index("## Rule violations")]
        assert "- #4 Unlabeled: no `type:` label, no `area:` label" in untriaged
        assert body.endswith("_Written by `scripts/triage/report.py` at 2026-10-26 11:00 UTC._\n")

    def test_an_empty_section_says_none(self):
        body = report.render([], [], NOW)
        assert "## Rule violations\n\nIssues that carry every required label" in body
        assert body.count("None.") == 3
        assert "No open milestones." in body
