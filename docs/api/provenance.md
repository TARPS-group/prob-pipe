# Provenance

Every `Distribution` or `Record` returned by a workflow function carries
a `Provenance` record linking it to its inputs and the op that produced
it. The result is a directed acyclic graph: each node is a value, each
edge points from a value to one of its inputs.

`provenance_ancestors(value)` returns the transitive set of values that
went into producing `value`. `provenance_dag(value)` returns the same
information as a `dict` describing the full DAG — useful for debugging or
for rendering the lineage with `graphviz`.

::: probpipe.Provenance

::: probpipe.provenance_ancestors

::: probpipe.provenance_dag
