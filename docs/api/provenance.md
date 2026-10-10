> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Labels and provenance

Each tracked term carries a label and a write-once provenance record, which names the operation that produced the term and its parents.
This page documents the identity of a tracked term, the provenance record and its traversal, the setting of how much history a provenance chain keeps, and the setting of how many nested levels a label or a notation shows.
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

## Provenance settings

::: probpipe.ProvenanceMode

::: probpipe.provenance_config

::: probpipe.core.config.ProvenanceConfig
    options:
      show_root_full_path: true

## Notation settings

::: probpipe.notation_config

::: probpipe.core.config.NotationConfig
    options:
      show_root_full_path: true
