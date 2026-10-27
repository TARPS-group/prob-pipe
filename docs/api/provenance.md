> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Labels and provenance

A label names a tracked term. A provenance record names the operation that produced a term and its parents.
`print(x.provenance)` prints the lineage of `x`, one step per line from the loaded data or other roots down to `x`, as `Provenance.__str__` states.
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

`config.provenance.mode` selects the data retained by new provenance records.
[Workflow scopes](workflows.md#workflow-scopes) determine when each call's mode is fixed.

::: probpipe.ProvenanceMode
