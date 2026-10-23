# Memory format

The `design-pipeline` skill keeps two memory files so that it behaves
consistently across sessions: one about the **person**, shared by all their
projects, and one per **project**. Memory is what lets a later session skip
questions already answered, keep the same explanation style, and pick up a
design where it left off.

## Locations

| File | Default path | Scope |
|---|---|---|
| Person memory | `$PROBPIPE_DESIGN_HOME/user.md`, else `~/.probpipe-design/user.md` | Everything true of the person regardless of project. |
| Project memory | `<workspace>/memory.md` (workspace defaults to `./probpipe-design/<project-slug>/`) | Everything specific to one analysis. |

The home directory doesn't survive in ephemeral environments (cloud
sessions, containers). If you're in one, say so when you first create person
memory and suggest setting `PROBPIPE_DESIGN_HOME` to a persistent location,
or keeping a copy in the user's own repository.

## Rules

- **Ask once before creating person memory.** Say where it will live, what
  goes in it, and that the user can read, edit, or delete it at any time. If
  they decline, keep person-level facts in project memory only, and don't ask
  again in this project (record the refusal there).
- **Record provenance.** Tag every fact with its source and date:
  `[stated 2026-10-23]` (the user said it), `[confirmed 2026-10-23]` (you
  proposed it and the user agreed), or `[inferred 2026-10-23]` (your reading
  of the evidence, not yet confirmed). Never present an inferred fact to the
  user as settled.
- **Corrections win.** When the user corrects something, replace the old
  entry; don't keep both. When they say "forget X", delete it.
- **Store conclusions, not transcripts.** One line per fact. Keep person
  memory under about 80 lines and project memory under about 150; merge or
  prune stale entries when you update.
- **Don't store** data values or excerpts, credentials, file contents, or
  sensitive personal information (health, finances, demographics) beyond
  what the user explicitly asks you to remember. Pointers to data (paths,
  URLs) are fine.
- **Write at checkpoints:** after the background assessment is confirmed,
  at the end of each phase, after every design revision, and before the
  session ends (or whenever the user signals they're stopping).
- **Read before you write.** Another session may have changed the file;
  merge, don't overwrite.

## Person memory template

```markdown
# ProbPipe design assistant: person memory
Last updated: <date>

## About
- Name / how to address them, role, field, institution — only what they shared.

## Background profile
| Dimension | Level | Confidence | Basis |
|---|---|---|---|
| Domain science (<field>) | Expert | high | [confirmed 2026-10-23] ... |
| Probability and statistics | ... | ... | ... |
(dimensions and levels from BACKGROUND-RUBRIC.md)

## Working preferences
- Explanation depth, code style, how many questions at once, preferred
  terminology, preferred backends (e.g. "prefers PyMC over Stan"), anything
  they asked you to do or avoid. Each with provenance.

## Environment
- OS, Python setup (uv/conda/pip), ProbPipe version and installed extras,
  compute available (laptop, GPU, cluster), other tools they use.

## Projects
| Slug | Workspace | One-line summary | Status | Last session |
|---|---|---|---|---|
```

## Project memory template

```markdown
# Project memory: <project name>
Last updated: <date> · Phase: <current phase> · Design version: <n>

## Where we left off
- Two or three lines: what was last done and the agreed next step.

## Key facts
- Problem, estimands, and data in a few lines each (details live in brief.md).

## Decisions
- <date> — <decision> — <why> — [confirmed]/[provisional]

## Project-specific preferences and background
- Anything that differs from person memory for this project (e.g. "new to
  survival analysis even though fluent in regression").

## Tried and ruled out
- Approaches that failed or were rejected, and why, so they aren't re-proposed.

## Open threads
- Questions waiting on the user, a colleague, data, or an experiment.
```
