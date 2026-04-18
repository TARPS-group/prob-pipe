# Internals

These classes and helpers are **not part of the stable public API** — they're
documented here for contributors, advanced users who need to understand the
broadcasting / protocol-dispatch machinery, and anyone debugging a failing
`isinstance(obj, SupportsX)` check. Names start with `_` to reflect this.
Signatures may change without deprecation warnings between PRs.

If you find yourself reaching for something on this page in user code, there's
probably a public replacement — check [Operations](operations.md),
[Distributions](distributions.md), or [Joint Distributions](joint.md) first,
and open an issue if there isn't.

## Record-distribution views

`dist["field"]` on a [`RecordDistribution`][probpipe.core._record_distribution.RecordDistribution]
returns a `_RecordDistributionView` — a lightweight reference that preserves
correlation when multiple views from the same parent are used in
[`@workflow_function`][probpipe.core.node.workflow_function] broadcasting. The
view's protocol membership (`SupportsSampling`, `SupportsMean`,
`SupportsVariance`, `SupportsLogProb`, `SupportsCovariance`) is computed
dynamically from the parent's capabilities, so
`isinstance(view, SupportsMean)` is true iff the parent is.

::: probpipe.core._record_distribution._RecordDistributionView

## Flat-view helpers

`FlattenedView` is a public wrapper that exposes a distribution with
structured samples as a flat
[`NumericRecordDistribution`][probpipe.core._array_distributions.NumericRecordDistribution],
useful for interop with algorithms that expect flat vectors. It uses the same
dynamic-protocol pattern as `_RecordDistributionView` — only the protocols the
base distribution supports are attached to the view.

::: probpipe.core._array_distributions.FlattenedView

## Sampling primitive

::: probpipe.core._array_distributions._vmap_sample

::: probpipe.core._array_distributions._mc_expectation
