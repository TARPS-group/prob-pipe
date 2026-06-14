---
name: review-all
description: Run three independent PR reviews (/review-pr, /review, /code-review) in parallel and merge them into one deduplicated, severity-tiered report. Use for a thorough multi-lens review before merging a non-trivial PR.
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash(gh *) Bash(git *) Agent
argument-hint: [pr-number]
---

# ProbPipe Consolidated PR Review

Review pull request **$ARGUMENTS** (if no number is given, detect the PR for the
current branch with `gh pr view`) by running three independent reviews in
parallel and merging them into one report.

This is READ-ONLY. **Do not edit any files.** Present the consolidated report
and wait for the user to decide what to act on.

The three lenses are complementary, and running them as separate agents keeps
their judgments independent. A finding raised by two lenses is high-confidence;
a disagreement between lenses is itself signal.

## Step 1: Gather context

```
gh pr view <number> --repo TARPS-group/prob-pipe --json title,baseRefName,headRefName,files,state
gh pr diff <number> --repo TARPS-group/prob-pipe
```

Note the head/base branches, the changed files, and a one-line statement of what
the PR is meant to do. Pass this to all three agents so none re-derives it.

## Step 2: Spawn the three reviews as parallel agents

Spawn **all three in a single message** (three `Agent` tool calls, `general-purpose`
type) so they run concurrently. Each agent **invokes the corresponding skill**
on the PR and returns its report. In every prompt, include the Step 1 context
and require: READ-ONLY; **do not spawn further sub-agents**; return a
**self-contained structured report with no preamble or sign-off** (it goes to a
coordinator for consolidation).

- **Agent 1** — invoke the `review-pr` skill (ProbPipe conventions, docs, tests,
  duplicate code, AI-artifact comments).
- **Agent 2** — invoke the `review` skill (general correctness, conventions,
  performance, test coverage, security).
- **Agent 3** — invoke the `code-review` skill (focused, high-effort correctness
  bug-hunt of the diff; it may run code read-only to confirm/refute findings).

If a skill is unavailable in the agent's context, tell it to fall back to
performing that review directly per the lens description above.

## Step 3: Consolidate

Merge the three reports into ONE — do not concatenate.

1. **Deduplicate** findings that describe the same underlying issue, even when
   worded or located differently. Note where lenses agree independently.
2. **Re-tier by consensus severity** (your judgment, informed by how many lenses
   flagged it and how load-bearing it is): **Must fix** / **Should fix** /
   **Minor / polish**.
3. **Attribute** each finding to the lens(es) that raised it; keep `file:line`.
4. **Surface disagreements** explicitly, with a reconciliation grounded in
   `STYLE_GUIDE.md` / `CONTRIBUTING.md`.
5. **Keep a "Verified correct" section** carrying forward what the bug-hunt lens
   checked and found sound.
6. **End with a recommendation** — what you would do, in order — and offer to
   implement it.

### Output format

```
# Consolidated Review: PR #<n> — <title>

## Verdict
## Must fix
## Should fix
## Minor / polish
## Contested (lenses disagreed)
## Verified correct
## Recommendation
```

## Notes

- Three agents is real token cost. For a quick look, `/review-pr` alone is
  enough; reach for `/review-all` before merging a non-trivial PR.
- Explicit-invocation only — general "review this PR" requests route to
  `/review-pr`.
