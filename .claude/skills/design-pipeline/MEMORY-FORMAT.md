# Memory format

The design-pipeline assistant keeps two memory files so that it behaves
consistently across sessions: one about the **person**, which all their
projects share, and one for each **project**. With memory, a later session
skips the questions already answered, keeps the same explanation style, and
resumes a design where it stopped.

## Locations

| File | Default path | Scope |
|---|---|---|
| Person memory | `$PROBPIPE_DESIGN_HOME/user.md`, else `~/.probpipe-design/user.md` | Everything true of the person in every project. |
| Project memory | `<workspace>/memory.md`, in the project's workspace directory | Everything specific to one analysis. |

The person-memory directory also holds the corpus, in its directory `corpus`,
which `build_corpus.py` writes there by default.

The home directory is lost when an ephemeral environment, such as a cloud
session or a container, ends. In such an environment, say so when you first
create person memory, and suggest setting `PROBPIPE_DESIGN_HOME` to a
persistent location or keeping a copy in the user's own repository.

## Workspace

A project's workspace directory holds the project memory and the files of the
design. Its default is `~/probpipe-projects/<short-project-slug>/`, and a user
who has a project directory of their own can keep it there instead, such as
`<their project>/probpipe-design/`. A workspace is outside the ProbPipe
checkout unless the user asks for it there.

Before you create a workspace, check whether its directory is inside a git
repository, with `git -C <directory> rev-parse --show-toplevel`. Inside a
repository, check whether git ignores the workspace, with
`git -C <repository> check-ignore -q <workspace>`. When git does not ignore it,
tell the user that git would track the memory and the design files, and offer
two choices: add the workspace to that repository's `.gitignore`, or move the
workspace outside the repository. When the user keeps a workspace that git
tracks, its project memory holds no facts about the person. Leave out the
section *Project-specific preferences and background* of its template, and
keep those facts in person memory.

A workspace holds these files:

- `<workspace>/memory.md`: the project memory.
- `<workspace>/brief.md`: the problem, the goals, the data, and the existing
  models, written in Phases 1 and 2 by the [brief template](#brief-template).
- `<workspace>/data-notes.md`: what you found when you inspected each dataset.
- `<workspace>/model-notes.md`: the models extracted from papers, code, or
  descriptions.
- `<workspace>/pipeline-design.md`: the current design, which follows
  [DESIGN-DOC-TEMPLATE.md](DESIGN-DOC-TEMPLATE.md), with a version history.
- `<workspace>/pipeline.py`: the runnable skeleton of the current design
  version, or the notebook `<workspace>/pipeline.ipynb` in its place.
- `<workspace>/report.py` and `<workspace>/sim_data.py`: the modules of
  printing and plotting helpers and of simulated data, which the pipeline
  imports, by
  [phases/4-proposal.md](phases/4-proposal.md#write-the-code).
- `<workspace>/pipeline.<stage>.ipynb`: the executed copy of the notebook
  from the last run of each stage, which
  [phases/4-proposal.md](phases/4-proposal.md#check-and-run-the-code)
  describes.
- `<workspace>/validation.json`: the grades and the table of the validation
  on realistic simulated data, which only the validate stage writes and the
  gate of the fit stage checks.
- The scripts that inspect the user's files, the log file of each run in the
  background, and the probe scripts with their log in `<workspace>/probes/`,
  by rules 3, 5, and 7 of the
  [rules for user materials](INSTRUCTIONS.md#user-materials-and-code).

## Rules

- **Ask once before you create person memory.** Say where it will be stored,
  what goes in it, and that the user can read, edit, or delete it at any time.
  After you create it, tell the user its path. If the user declines, record the
  refusal in project memory and ask no more in this project. Keep person-level
  facts in project memory when git does not track the workspace, and only in
  the conversation when git tracks it.
- **Keep these facts in these files.** Write the facts of this assistant only
  to person memory, project memory, and the workspace. Keep them out of your
  agent's own memory and instruction files, such as `CLAUDE.md`, `AGENTS.md`,
  or the agent's automatic memory.
- **Project memory decides for its project.** When person memory and project
  memory disagree about the current project, follow project memory.
- **Confirm old and inferred facts.** A fact tagged `[inferred]`, or dated more
  than about six months ago, is unconfirmed. Confirm it with the user before
  you rely on it.
- **Record the source of each fact.** Tag every fact with its source and date:
  - `[stated 2026-10-23]`: the user said it;
  - `[confirmed 2026-10-23]`: you proposed it, and the user agreed;
  - `[inferred 2026-10-23]`: your reading of the evidence, which the user has
    not confirmed.

  Present an inferred fact to the user as unconfirmed.
- **A correction replaces a fact.** When the user corrects a fact, such as a
  rating of the profile or a description of the data, replace the old entry
  with the corrected one. When they say "forget X", delete the facts about the
  person that X names, from person memory and from the section
  *Project-specific preferences and background* of project memory. A project's
  decisions stay by the next rule until the user asks to delete that project's
  memory. When they say "forget me", delete the person-memory file, tell them
  its path, and ask whether to delete the project memories that its index
  lists.
- **Project decisions are append-only.** Add each decision as a new entry of
  *Decisions*. When a later decision reverses an earlier one, keep the earlier
  entry and mark it `[superseded <date> by <decision>]`. Keep every entry of
  *Tried and ruled out*, because together the two sections record why the
  design has its current form.
- **Store conclusions.** Write one line per fact. Keep person memory under
  about 80 lines and project memory under about 150, and merge or prune stale
  entries at each update. The sections *Decisions*
  and *Tried and ruled out* are exempt from the limit and from pruning.
- **Store pointers to data.** Store file paths and URLs. Never store
  credentials, data values or excerpts, or file contents. Store sensitive
  personal information, such as health, finances, or demographics, only when
  the user explicitly asks you to remember it.
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

## Brief template

Phase 1 writes sections 1 and 2 of `<workspace>/brief.md`, and Phase 2 writes
sections 3 and 4.

```markdown
# Brief: <project name>
Last updated: <date>

## 1. Problem
- The system, phenomenon, or decision, and the question to answer.

## 2. Goals and constraints
- The scientific or applied goals, the kinds of analysis, and the estimands.
- Deadlines, compute, required software and reporting standards, and scale.

## 3. Data
- Each dataset: a pointer to it, how it was collected, and a summary from
  data-notes.md.

## 4. Existing models
- Each model: its form and source, and a summary from model-notes.md.
- Each mismatch between the data and the models, as an open question.
```

## Project memory template

```markdown
# Project memory: <project name>
Last updated: <date> · Phase: <current phase> · Design version: <n> · Mode: <quick start or full workflow>
Corpus: <corpus directory> · commit <commit> · ProbPipe <version>
Python: <python> · Kernel: <kernel>

## Where we left off
- Two or three lines: what was last done and the agreed next step.

## Key facts
- Problem, estimands, and data in a few lines each (details are in brief.md).

## Decisions
- <date> — <decision> — <why> — [confirmed]/[provisional]/[override]
- <date> — <decision> — <why> — [confirmed] [superseded <date> by <decision>]

## Project-specific preferences and background
- Anything that differs from person memory for this project (e.g. "new to
  survival analysis even though fluent in regression").

## Tried and ruled out
- Approaches that failed or were rejected, and why, so that they are not proposed again.

## Open threads
- Questions waiting on the user, a colleague, data, or an experiment.
```
