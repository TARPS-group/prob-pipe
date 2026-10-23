# Memory format

The `design-pipeline` skill keeps two memory files so that it behaves
consistently across sessions: one about the **person**, which all their
projects share, and one for each **project**. With memory, a later session
skips the questions already answered, keeps the same explanation style, and
resumes a design where it stopped.

## Locations

| File | Default path | Scope |
|---|---|---|
| Person memory | `$PROBPIPE_DESIGN_HOME/user.md`, else `~/.probpipe-design/user.md` | Everything true of the person in every project. |
| Project memory | `<workspace>/memory.md`, in the project's workspace directory | Everything specific to one analysis. |

The home directory is lost when an ephemeral environment, such as a cloud
session or a container, ends. In such an environment, say so when you first
create person memory, and suggest setting `PROBPIPE_DESIGN_HOME` to a
persistent location or keeping a copy in the user's own repository.

## Rules

- **Ask once before you create person memory.** Say where it will be stored,
  what goes in it, and that the user can read, edit, or delete it at any time.
  If the user declines, keep person-level facts in project memory only, record
  the refusal there, and ask no more in this project.
- **Record the source of each fact.** Tag every fact with its source and date:
  - `[stated 2026-10-23]`: the user said it;
  - `[confirmed 2026-10-23]`: you proposed it, and the user agreed;
  - `[inferred 2026-10-23]`: your reading of the evidence, which the user has
    not confirmed.

  Present an inferred fact to the user as unconfirmed.
- **A correction replaces the entry.** When the user corrects a fact, replace
  the old entry with the corrected one. When they say "forget X", delete the
  entry.
- **Store conclusions.** Write one line per fact. Keep person memory under
  about 80 lines and project memory under about 150, and merge or prune stale
  entries at each update.
- **Store pointers to data.** Store file paths and URLs. Leave out data values
  and excerpts, credentials, and file contents, and leave out sensitive
  personal information, such as health, finances, or demographics, beyond
  what the user explicitly asks you to remember.
- **Write at four checkpoints:**
  1. after the user confirms the background assessment;
  2. at the end of each phase;
  3. after each design revision;
  4. before the session ends, or whenever the user signals that they are
     stopping.
- **Read before you write.** Another session may have changed the file, so
  merge your changes into it.

## Person memory template

```markdown
# ProbPipe design assistant: person memory
Last updated: <date>

## About
- Name or how to address them, role, discipline, institution — only what they shared.

## Background profile
| Dimension | Rating | Confidence | Basis |
|---|---|---|---|
| Domain science (<discipline>) | Expert | high | [confirmed 2026-10-23] ... |
| Probability and statistics | ... | ... | ... |
(dimensions and ratings from BACKGROUND-RUBRIC.md)

## Working preferences
- Explanation depth, code style, how many questions at once, preferred
  terminology, preferred backends (e.g. "prefers PyMC over Stan"), anything
  they asked you to do or avoid. Each with its source tag.

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
- Problem, estimands, and data in a few lines each (details are in brief.md).

## Decisions
- <date> — <decision> — <why> — [confirmed]/[provisional]

## Project-specific preferences and background
- Anything that differs from person memory for this project (e.g. "new to
  survival analysis even though fluent in regression").

## Tried and ruled out
- Approaches that failed or were rejected, and why, so that they are not proposed again.

## Open threads
- Questions waiting on the user, a colleague, data, or an experiment.
```
