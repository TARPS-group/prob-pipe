> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Labels and provenance

Each tracked term carries a label and a write-once provenance record, which names the operation that produced the term and its parents.
This page documents the identity of a tracked term, the provenance record and its traversal, and the modes of how much history a provenance chain keeps.
The settings `config.provenance` and `config.notation` are on [Settings](config.md).
Replay from a provenance record is on [Workflows and reproducibility](workflows.md).

## Labels and annotations

::: probpipe.TrackedTerm

::: probpipe.Annotated

::: probpipe.core.tracked.auto_label
    options:
      show_root_full_path: true

## Provenance records

::: probpipe.Provenance

::: probpipe.ParentInfo

::: probpipe.provenance_ancestors

::: probpipe.provenance_dag

## Provenance modes

`config.provenance.mode` takes one of these modes, as [Settings](config.md) states.
A `workflow_run` keeps the mode in effect when it opens, and changing the mode inside the run raises `RuntimeError`.

::: probpipe.ProvenanceMode

A [default workflow scope](workflows.md) fixes provenance mode for each top-level
call separately. Changing the mode between calls leaves its random stream
unchanged. Full recording cannot recover history omitted by earlier calls.
