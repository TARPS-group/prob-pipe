"""Check a pull request's title, branch, and body against the repository's conventions.

``CONTRIBUTING.md`` § Opening the PR and § Branch naming state the conventions,
and ``.github/PULL_REQUEST_TEMPLATE.md`` is the body every PR starts from. A
finding is one of these:

1. a title that does not read ``<type>(<scope>): <subject>``;
2. a branch that is not ``dev/<short-kebab-case-description>``;
3. a body that lacks a ``## `` section or a checklist item of the template.

Usage::

    PR_TITLE=... PR_BRANCH=... PR_BODY=... python3 scripts/ci/pr_hygiene.py

The script prints each finding as a GitHub warning annotation, appends the
findings to ``$GITHUB_STEP_SUMMARY`` when that variable is set, and exits with
status 1 while a finding remains. It is stdlib-only, like the other CI helpers.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"

_TITLE = re.compile(r"[a-z]+\([^()\s]+\)!?: \S.*")
_BRANCH = re.compile(r"dev/[a-z0-9]+(?:-[a-z0-9]+)*")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_ITEM = re.compile(r"^\s*- \[[ xX]\] (.+?)\s*$", re.M)


def title_findings(title: str) -> list[str]:
    """The findings for a PR title: one unless it reads ``<type>(<scope>): <subject>``."""
    if _TITLE.fullmatch(title.strip()):
        return []
    return [f"the title {title.strip()!r} does not read <type>(<scope>): <subject>"]


def branch_findings(branch: str) -> list[str]:
    """The findings for a PR branch: one unless it is ``dev/`` and a kebab-case description."""
    if _BRANCH.fullmatch(branch.strip()):
        return []
    return [f"the branch {branch.strip()!r} is not dev/<short-kebab-case-description>"]


def _sections(text: str) -> list[str]:
    return [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]


def _items(text: str) -> list[str]:
    return [" ".join(item.split()) for item in _ITEM.findall(_COMMENT.sub("", text))]


def body_findings(body: str, template: str) -> list[str]:
    """The findings for a PR body: each section and checklist item of *template* it lacks."""
    findings = [
        f"the body lacks the section '## {section}'"
        for section in _sections(template)
        if section not in _sections(body)
    ]
    present = set(_items(body))
    findings += [
        f"the body lacks the checklist item '{item}'"
        for item in _items(template)
        if item not in present
    ]
    return findings


def main() -> int:
    findings = [
        *title_findings(os.environ.get("PR_TITLE", "")),
        *branch_findings(os.environ.get("PR_BRANCH", "")),
        *body_findings(os.environ.get("PR_BODY", ""), TEMPLATE.read_text()),
    ]
    for finding in findings:
        print(f"::warning title=PR hygiene::{finding}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as out:
            out.write("### PR hygiene (advisory — does not gate merges)\n\n")
            out.writelines(f"- {finding}\n" for finding in findings)
            if not findings:
                out.write("The title, branch, and body follow CONTRIBUTING.md and the template.\n")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
