# Internals

These classes and helpers are **not part of the stable public API** — they're
documented here for contributors, advanced users who need to understand the
broadcasting / protocol-dispatch machinery, and anyone debugging a failing
`isinstance(obj, SupportsX)` check. Names start with `_` to reflect this.
Signatures may change without deprecation warnings between PRs.

If you find yourself reaching for something on this page in user code, there's
probably a public replacement — check [Operations](operations.md),
[Distributions](distributions/index.md), [Records](records.md), or
[Extending ProbPipe](extending.md) first, and open an issue if there isn't.

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

## Flat / Record view helpers

`FlattenedDistributionView` and `NumericRecordDistributionView` are the public
view classes for the flat ↔ Record-keyed bridge. Both follow the same
dynamic-protocol pattern as `_RecordDistributionView` — only the protocols
the base distribution supports are attached to the view.

`FlattenedDistributionView` is a [`FlatNumericRecordDistribution`][probpipe.core._numeric_record_distribution.FlatNumericRecordDistribution]:
it exposes any distribution as flat (single field, `event_shape=(N,)`),
for interop with algorithms that expect a flat parameter vector.
Construct via [`as_flat_distribution`][probpipe.core._numeric_record_distribution.NumericRecordDistribution.as_flat_distribution].

`NumericRecordDistributionView` is the inverse: it takes a
`FlatNumericRecordDistribution` and a `NumericEventTemplate`, and presents
the distribution under the template's named-field structure. Construct via
[`as_record_distribution`][probpipe.core._numeric_record_distribution.FlatNumericRecordDistribution.as_record_distribution].

::: probpipe.core._numeric_record_distribution.FlatNumericRecordDistribution

::: probpipe.core._numeric_record_distribution.FlattenedDistributionView

::: probpipe.core._numeric_record_distribution.NumericRecordDistributionView

## Sampling primitive

::: probpipe.core._numeric_record_distribution._vmap_sample

::: probpipe.core._numeric_record_distribution._mc_expectation
