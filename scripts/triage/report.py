"""Write the weekly triage report of the open issues from their labels and milestones.

Run ``python3 scripts/triage/report.py`` from the repository root to print the report,
or ``python3 scripts/triage/report.py --post 700`` to write it as the body of issue 700.
``--dry-run`` prints the body that ``--post`` would write. The script reads GitHub
through the ``gh`` command, so ``gh`` must be logged in.

The report applies the rules of CONTRIBUTING.md § Issue triage and has four sections:

1. **Untriaged**: each issue labeled ``status:needs-triage``, and each issue that
   misses a label or a milestone that a triaged issue carries.
2. **Rule violations**: each other issue whose labels break a triage rule, such as
   two type labels or a P3 issue with a milestone.
3. **Due for re-triage**: each P1 or P2 issue with no activity for
   :data:`STALE_DAYS` days, and each open issue of a milestone due within
   :data:`REVIEW_DAYS` days or overdue.
4. **Release gates**: for each open milestone, its open P0 and S1 issues, which
   block the release, and its open P1 issues, which are closed or moved first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

TYPE_PREFIX = "type:"
SEVERITY_PREFIX = "severity:"
PRIORITY_PREFIX = "priority:"
AREA_PREFIX = "area:"

TYPE_BUG = "type:bug"
TYPE_TRACKING = "type:tracking"
TYPE_QUESTION = "type:question"
#: The types that carry no priority.
UNPRIORITIZED_TYPES = frozenset({TYPE_TRACKING, TYPE_QUESTION})

SEVERITY_S1 = "severity:S1"
PRIORITY_P0 = "priority:P0"
PRIORITY_P1 = "priority:P1"
PRIORITY_P2 = "priority:P2"
PRIORITY_P3 = "priority:P3"

NEEDS_TRIAGE = "status:needs-triage"

#: The days without activity after which a P1 or P2 issue is due for re-triage.
STALE_DAYS = 90
#: The days before a milestone's due date from which its open issues are due for review.
REVIEW_DAYS = 7

FOOTER = "_Written by `scripts/triage/report.py` at "


class Finding(NamedTuple):
    """An issue and what the triage rules find wrong with it."""

    issue: dict[str, Any]
    problems: tuple[str, ...]


def _gh(*args: str) -> Any:
    """Run ``gh`` with *args* and return its JSON output."""
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _labels(issue: dict[str, Any]) -> set[str]:
    return {label["name"] for label in issue["labels"]}


def _with_prefix(labels: set[str], prefix: str) -> list[str]:
    return sorted(label for label in labels if label.startswith(prefix))


def _time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _needs_priority(labels: set[str]) -> bool:
    return not labels & UNPRIORITIZED_TYPES


def missing(issue: dict[str, Any]) -> list[str]:
    """Return what *issue* lacks of the labels and the milestone a triaged issue carries."""
    labels = _labels(issue)
    gaps = []
    if not _with_prefix(labels, TYPE_PREFIX):
        gaps.append("no `type:` label")
    if TYPE_BUG in labels and not _with_prefix(labels, SEVERITY_PREFIX):
        gaps.append("a bug without a severity")
    if _needs_priority(labels) and not _with_prefix(labels, PRIORITY_PREFIX):
        gaps.append("no priority")
    if labels & {PRIORITY_P0, PRIORITY_P1} and not issue["milestone"]:
        gaps.append("a P0 or P1 issue without a milestone")
    if not _with_prefix(labels, AREA_PREFIX):
        gaps.append("no `area:` label")
    return gaps


def violations(issue: dict[str, Any]) -> list[str]:
    """Return the triage rules that the labels of *issue* break, other than a missing label."""
    labels = _labels(issue)
    found = []
    for prefix in (TYPE_PREFIX, SEVERITY_PREFIX, PRIORITY_PREFIX):
        if len(several := _with_prefix(labels, prefix)) > 1:
            found.append(f"{len(several)} `{prefix}` labels: {', '.join(several)}")
    if _with_prefix(labels, SEVERITY_PREFIX) and TYPE_BUG not in labels:
        found.append("a severity on an issue that is not a bug")
    if PRIORITY_P3 in labels and issue["milestone"]:
        found.append("a P3 issue with a milestone")
    return found


def untriaged(issues: list[dict[str, Any]]) -> list[Finding]:
    """Return each issue labeled ``status:needs-triage`` or missing a required label."""
    findings = []
    for issue in issues:
        gaps = missing(issue)
        if gaps or NEEDS_TRIAGE in _labels(issue):
            findings.append(Finding(issue, (*gaps, *violations(issue))))
    return findings


def rule_violations(issues: list[dict[str, Any]]) -> list[Finding]:
    """Return each issue that :func:`untriaged` leaves out and whose labels break a rule."""
    findings = []
    for issue in issues:
        if missing(issue) or NEEDS_TRIAGE in _labels(issue):
            continue
        if found := violations(issue):
            findings.append(Finding(issue, tuple(found)))
    return findings


def milestones_due(milestones: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Return the open milestones due within :data:`REVIEW_DAYS` days or overdue, soonest first."""
    cutoff = now + timedelta(days=REVIEW_DAYS)
    due = [m for m in milestones if m.get("due_on") and _time(m["due_on"]) <= cutoff]
    return sorted(due, key=lambda m: m["due_on"])


def due_for_retriage(
    issues: list[dict[str, Any]], milestones: list[dict[str, Any]], now: datetime
) -> list[Finding]:
    """Return each issue due for re-triage, with the reasons.

    An issue is due when it is a P1 or P2 issue with no activity for
    :data:`STALE_DAYS` days, or when it is open in a milestone that
    :func:`milestones_due` returns. An issue labeled ``status:needs-triage`` is
    left out, since it is in the triage queue already.
    """
    reasons: dict[int, list[str]] = {}
    cutoff = now - timedelta(days=STALE_DAYS)
    for issue in issues:
        labels = _labels(issue)
        if NEEDS_TRIAGE in labels:
            continue
        if labels & {PRIORITY_P1, PRIORITY_P2} and _time(issue["updatedAt"]) < cutoff:
            days = (now - _time(issue["updatedAt"])).days
            reasons.setdefault(issue["number"], []).append(f"no activity for {days} days")
    for milestone in milestones_due(milestones, now):
        for issue in _in_milestone(issues, milestone["title"]):
            if NEEDS_TRIAGE not in _labels(issue):
                reason = f"open in {milestone['title']}, due {_due_date(milestone)}"
                reasons.setdefault(issue["number"], []).append(reason)
    by_number = {issue["number"]: issue for issue in issues}
    return [Finding(by_number[number], tuple(found)) for number, found in reasons.items()]


def _in_milestone(issues: list[dict[str, Any]], title: str) -> list[dict[str, Any]]:
    return [issue for issue in issues if (issue["milestone"] or {}).get("title") == title]


def _sort_oldest_first(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda finding: finding.issue["createdAt"])


def _issue_line(issue: dict[str, Any], problems: tuple[str, ...] = ()) -> str:
    line = f"- #{issue['number']} {issue['title']}"
    return f"{line}: {', '.join(problems)}" if problems else line


def _section(heading: str, lead: str, lines: list[str]) -> list[str]:
    return [heading, "", lead, "", *(lines or ["None."]), ""]


def _due_date(milestone: dict[str, Any]) -> str:
    return milestone["due_on"][:10] if milestone.get("due_on") else "no due date"


def _gate_lines(milestone: dict[str, Any], issues: list[dict[str, Any]]) -> list[str]:
    members = _in_milestone(issues, milestone["title"])
    blockers = [i for i in members if _labels(i) & {PRIORITY_P0, SEVERITY_S1}]
    to_move = [i for i in members if PRIORITY_P1 in _labels(i) and i not in blockers]
    if blockers or to_move:
        state = f"**blocked**: {len(blockers)} P0 or S1, {len(to_move)} P1 to close or move"
    else:
        state = "**clear**"
    lines = [
        f"### [{milestone['title']}]({milestone['html_url']}), due {_due_date(milestone)}",
        "",
        f"{len(members)} open issues, {state}.",
    ]
    if blockers:
        lines += ["", "Open P0 and S1 issues:", "", *(_issue_line(i) for i in blockers)]
    if to_move:
        lines += ["", "Open P1 issues:", "", *(_issue_line(i) for i in to_move)]
    return [*lines, ""]


def render(issues: list[dict[str, Any]], milestones: list[dict[str, Any]], now: datetime) -> str:
    """Return the Markdown body of the triage report of the open *issues* and *milestones*."""
    lines = [
        "The weekly triage report of the open issues, by the rules of "
        "CONTRIBUTING.md § Issue triage. A workflow rewrites this body each week, "
        "so an edit to it is overwritten.",
        "",
    ]
    lines += _section(
        "## Untriaged",
        f"Issues labeled `{NEEDS_TRIAGE}` or missing a required label, oldest first.",
        [_issue_line(f.issue, f.problems) for f in _sort_oldest_first(untriaged(issues))],
    )
    lines += _section(
        "## Rule violations",
        "Issues that carry every required label but break another triage rule.",
        [_issue_line(f.issue, f.problems) for f in _sort_oldest_first(rule_violations(issues))],
    )
    lines += _section(
        "## Due for re-triage",
        f"P1 and P2 issues with no activity for {STALE_DAYS} days, and the open issues "
        f"of each milestone due within {REVIEW_DAYS} days or overdue.",
        [
            _issue_line(f.issue, f.problems)
            for f in _sort_oldest_first(due_for_retriage(issues, milestones, now))
        ],
    )
    lines += ["## Release gates", ""]
    ordered = sorted(milestones, key=lambda m: (m.get("due_on") or "9999", m["title"]))
    for milestone in ordered:
        lines += _gate_lines(milestone, issues)
    if not milestones:
        lines += ["No open milestones.", ""]
    return "\n".join([*lines, f"{FOOTER}{now:%Y-%m-%d %H:%M} UTC._", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--post", type=int, metavar="ISSUE", help="write the report as the body of this issue"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the body instead of writing it"
    )
    args = parser.parse_args()
    fields = "number,title,labels,milestone,createdAt,updatedAt"
    issues = _gh("issue", "list", "--state", "open", "--limit", "1000", "--json", fields)
    milestones = _gh("api", "repos/{owner}/{repo}/milestones?state=open&per_page=100")
    body = render(issues, milestones, datetime.now(UTC))
    if args.post is None or args.dry_run:
        print(body)
        return
    subprocess.run(
        ["gh", "issue", "edit", str(args.post), "--body-file", "-"],
        input=body,
        text=True,
        check=True,
    )
    print(f"Updated #{args.post}.")


if __name__ == "__main__":
    main()
