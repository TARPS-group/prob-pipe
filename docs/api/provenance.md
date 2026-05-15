# Provenance

Every `Distribution` and `Record` returned by a `WorkflowFunction` carries a
`Provenance` link describing how it was produced. The ancestor and DAG
helpers walk the chain backwards from a value.

::: probpipe.Provenance

::: probpipe.provenance_ancestors

::: probpipe.provenance_dag
