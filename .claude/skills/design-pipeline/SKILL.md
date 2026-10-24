---
name: design-pipeline
description: Guide a user from a scientific or applied problem to a ProbPipe data analysis pipeline. The assistant searches the ProbPipe documentation and Bayesian workflow guidance, interviews the user about their problem, data, and models, confirms their background, and proposes and revises a pipeline with them, remembering the person and their projects across sessions. Use when someone wants help to design, plan, build, or resume an analysis with ProbPipe.
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

The checkout is the root of this session's repository, which is the
`<checkout>` of the instructions. The instructions name capabilities, and these
Claude Code tools provide them:

| Capability | Tool |
|---|---|
| Read a file, or a range of its lines | `Read` |
| Search the corpus or the source | `Grep` |
| Run a command with a timeout | `Bash`, with its timeout |
| Write or change a file | `Write` and `Edit` |
| Ask a multiple-choice question | `AskUserQuestion` |

Build the corpus in the default directory that `build_corpus.py` uses, so that
a later session finds it.
