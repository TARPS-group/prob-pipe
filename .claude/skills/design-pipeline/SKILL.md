---
name: design-pipeline
description: Guide a user from a scientific or applied problem to a ProbPipe data analysis pipeline. The assistant searches the ProbPipe documentation and Bayesian workflow guidance, interviews the user about their problem, data, and models, agrees with them how to work together, and proposes and revises a pipeline with them, remembering the person and their projects across sessions. Use when someone wants help to design, plan, build, or resume an analysis with ProbPipe.
argument-hint: [problem description or project to resume]
disable-model-invocation: true
---

# ProbPipe pipeline design assistant

Read [INSTRUCTIONS.md](INSTRUCTIONS.md) and follow it for the rest of the
session. Those instructions are written for any agent, and this file maps them
to Claude Code.

The user's opening message is `$ARGUMENTS` when it is non-empty: a description
of their problem, or the name of a project to resume. When it is empty, the
user's next message is the opening message.

Re-read INSTRUCTIONS.md whenever the context is compacted.

The checkout is the root of this session's repository, which is the
`<checkout>` of the instructions, and `<python>` is the Python of the
environment where ProbPipe is installed in it, such as the `.venv` of the
checkout. The instructions name capabilities, and these Claude Code tools
provide them:

| Capability | Tool |
|---|---|
| Read a file, or a range of its lines | `Read` |
| Write a file, or change part of one | `Write` and `Edit` |
| Search the corpus or the source | `Grep` |
| Run a command with a timeout | `Bash`, with its timeout |
| Fetch a URL | `WebFetch` |
| Ask a multiple-choice question | `AskUserQuestion` |

Build the corpus in the default directory that `build_corpus.py` uses, so that
a later session finds it.

The repository's contributor files, such as `AGENTS.md`, `CONTRIBUTING.md`, and
`STYLE_GUIDE.md`, govern changes to ProbPipe itself. INSTRUCTIONS.md governs
this session's design, code, and messages for the user.
