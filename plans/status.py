"""Rewrite the status of the contracts push on its tracking issue from the PRs' labels.

Run ``python3 plans/status.py`` from the repository root to rewrite the issue's body,
or ``python3 plans/status.py --dry-run`` to print the body it would write. It reads
GitHub through the ``gh`` command, so ``gh`` must be logged in.

The script owns three parts of the body and keeps the rest as it finds it:

- **Each step's line**, ``- [ ] **S0**: <description> — <status>``. The status is
  read from the PRs labeled ``step:S0``: merged, for a PR merged into
  ``dev/overhaul``; in review or draft, for an open PR; or not started. The box is
  ticked once a PR is merged and none is open. A line whose status is ``**cut**``
  is kept, since only the maintainer cuts a step. The description stops at the
  first `` — ``, so a description never contains one.
- **The section "Decided since the last checkpoint"**: the issue's comments that
  name ``decisions.md`` or ``deferred.md``, posted after the last PR labeled
  ``push:checkpoint`` merged. The checkpoint PR moves them into those files.
- **The section "Plans and design PRs"**: the PRs labeled ``push:plans`` or
  ``push:checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from typing import Any

ISSUE = 601
BASE = "dev/overhaul"
SEPARATOR = " — "
STEP_LINE = re.compile(r"^- \[[ x]\] \*\*(?P<step>[A-Za-z0-9-]+)\*\*: (?P<rest>.*)$")
DECIDED = "## Decided since the last checkpoint"
PLANS = "## Plans and design PRs"
FOOTER = "_Written by `plans/status.py` at "


def _gh(*args: str) -> Any:
    """Run ``gh`` with *args* and return its JSON output."""
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _labels(pr: dict[str, Any]) -> set[str]:
    return {label["name"] for label in pr["labels"]}


def _merged(pr: dict[str, Any]) -> bool:
    return bool(pr["mergedAt"]) and pr["baseRefName"] == BASE


def _step_status(prs: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return whether a step's box is ticked, and its status text, from its PRs."""
    parts = []
    merged = sorted((pr for pr in prs if _merged(pr)), key=lambda pr: pr["mergedAt"])
    if merged:
        listed = ", ".join(f"#{pr['number']} ({pr['mergedAt'][:10]})" for pr in merged)
        parts.append(f"**merged** {listed}")
    still_open = sorted((pr for pr in prs if pr["state"] == "OPEN"), key=lambda pr: pr["number"])
    for pr in still_open:
        parts.append(f"{'**draft**' if pr['isDraft'] else '**in review**'} #{pr['number']}")
    return bool(merged) and not still_open, "; ".join(parts) or "not started"


def _step_lines(body: str, prs: list[dict[str, Any]]) -> str:
    lines = []
    for line in body.splitlines():
        match = STEP_LINE.match(line)
        if match is None or line.endswith(f"{SEPARATOR}**cut**"):
            lines.append(line)
            continue
        step = match["step"]
        description = match["rest"].split(SEPARATOR)[0]
        ticked, status = _step_status([pr for pr in prs if f"step:{step}" in _labels(pr)])
        lines.append(f"- [{'x' if ticked else ' '}] **{step}**: {description}{SEPARATOR}{status}")
    return "\n".join(lines)


def _decided_section(comments: list[dict[str, Any]], prs: list[dict[str, Any]]) -> str:
    checkpoints = [pr["mergedAt"] for pr in prs if "push:checkpoint" in _labels(pr) and _merged(pr)]
    since = max(checkpoints, default="")
    items = []
    for comment in comments:
        if comment["createdAt"] <= since:
            continue
        if "decisions.md" not in comment["body"] and "deferred.md" not in comment["body"]:
            continue
        first = comment["body"].strip().splitlines()[0].replace("**", "").strip()
        title = first if len(first) <= 110 else first[:107].rstrip() + "..."
        items.append(
            f"- [{title}]({comment['url']}), {comment['createdAt'][:16].replace('T', ' ')} UTC"
        )
    after = f" (#{_last_checkpoint_number(prs)})" if since else ""
    lead = (
        f"Comments that name `decisions.md` or `deferred.md`, posted since the last checkpoint PR merged{after}. "
        "A new session reads these as well as `decisions.md`, and the next checkpoint PR moves them there."
    )
    return "\n".join([DECIDED, "", lead, "", *(items or ["None."])])


def _last_checkpoint_number(prs: list[dict[str, Any]]) -> int:
    merged = [pr for pr in prs if "push:checkpoint" in _labels(pr) and _merged(pr)]
    return max(merged, key=lambda pr: pr["mergedAt"])["number"]


def _plans_section(prs: list[dict[str, Any]]) -> str:
    rows = []
    for pr in sorted(prs, key=lambda pr: pr["number"], reverse=True):
        if not _labels(pr) & {"push:plans", "push:checkpoint"} or pr["state"] == "CLOSED":
            continue
        state = (
            f"merged {pr['mergedAt'][:10]}"
            if pr["mergedAt"]
            else ("draft" if pr["isDraft"] else "in review")
        )
        rows.append(f"- #{pr['number']}, {state}: {pr['title']}")
    return "\n".join([PLANS, "", *(rows or ["None."])])


def _replace_section(body: str, heading: str, section: str) -> str:
    """Replace the section under *heading*, up to the next heading or the footer, or add it at the end."""
    lines = body.splitlines()
    if heading not in lines:
        return body.rstrip() + "\n\n" + section
    start = lines.index(heading)
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].startswith("## ") or lines[i].startswith(FOOTER)
        ),
        len(lines),
    )
    return "\n".join([*lines[:start], *section.splitlines(), "", *lines[end:]])


def render(
    body: str, prs: list[dict[str, Any]], comments: list[dict[str, Any]], now: datetime
) -> str:
    """Return the issue body with its step lines and generated sections rewritten."""
    body = "\n".join(line for line in body.splitlines() if not line.startswith(FOOTER)).rstrip()
    body = _step_lines(body, prs)
    body = _replace_section(body, DECIDED, _decided_section(comments, prs))
    body = _replace_section(body, PLANS, _plans_section(prs))
    return f"{body.rstrip()}\n\n{FOOTER}{now:%Y-%m-%d %H:%M} UTC._\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="print the new body instead of writing it"
    )
    args = parser.parse_args()
    issue = _gh("issue", "view", str(ISSUE), "--json", "body,comments")
    fields = "number,title,state,isDraft,mergedAt,baseRefName,labels"
    prs = _gh("pr", "list", "--state", "all", "--limit", "500", "--json", fields)
    body = render(issue["body"], prs, issue["comments"], datetime.now(UTC))
    if args.dry_run:
        print(body)
        return
    subprocess.run(
        ["gh", "issue", "edit", str(ISSUE), "--body-file", "-"], input=body, text=True, check=True
    )
    print(f"Updated #{ISSUE}.")


if __name__ == "__main__":
    main()
