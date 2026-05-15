# Extending ProbPipe

Base classes, protocols, and registry contracts for adding new
distributions, inference methods, converters, bijectors, and auxiliary
metadata.

## Distribution base classes

`Distribution` is the abstract root. `RecordDistribution` and
`NumericRecordDistribution` specialise it for distributions whose
`_sample()` returns a `Record` or `NumericRecord` respectively.
`TFPDistribution` wraps an existing TFP `Distribution`.

::: probpipe.Distribution

::: probpipe.RecordDistribution

::: probpipe.NumericRecordDistribution

::: probpipe.TFPDistribution

## Protocols

Protocols define capabilities that distributions may support. Each
protocol is `@runtime_checkable`; compliance is checked via `isinstance`
at dispatch time. External types satisfy a protocol structurally by
implementing the underscore method (`_sample`, `_log_prob`, ...) — no
inheritance required.

::: probpipe.SupportsSampling

::: probpipe.SupportsExpectation

::: probpipe.SupportsLogProb

::: probpipe.SupportsUnnormalizedLogProb

::: probpipe.SupportsRandomLogProb

::: probpipe.SupportsRandomUnnormalizedLogProb

::: probpipe.SupportsMean

::: probpipe.SupportsVariance

::: probpipe.SupportsCovariance

::: probpipe.SupportsConditioning

`SupportsArrayBackend` is the only **class-level** protocol: its declared
method (`_make_array_backend`) is a `@classmethod`, so the runtime check
is `isinstance(MyDistribution, SupportsArrayBackend)` against the class
itself, not an instance.

::: probpipe.SupportsArrayBackend

## Custom inference methods

`Method` subclasses register with `inference_method_registry` and declare
`supported_types`, a `priority`, and `check()` / `execute()` methods. The
built-ins are listed under
[Modeling and inference → Inference methods](inference.md#inference-methods).

::: probpipe.core._registry.MethodRegistry

::: probpipe.core._registry.Method

::: probpipe.core._registry.MethodInfo

## Custom converters

Subclass `Converter`, implement `check()` / `convert()`, and register with
`converter_registry.register(...)`. The built-in priorities and registry
handle are documented under [Conversion and interop](converters.md).

## Custom bijectors

`register_bijector` overrides the canonical bijector returned by
`bijector_for(c)` for a given `Constraint` `c` (or class). See
[Constraints → Bijectors](constraints.md#bijectors-for-unconstrained-reparameterization).

## Custom auxiliary metadata

`register_aux` extends the `Record` ↔ `NumericRecord` round-trip to a new
array-like type. See
[Records → Auxiliary-metadata registry](records.md#auxiliary-metadata-registry).

## Broadcasting internals (exposed for extension)

`DistributionArray` is the shape-indexed container produced by
parameter-sweep workflow functions whose inner call returns a
`Distribution`. `BroadcastDistribution` is the joint container produced
by `@workflow_function` when `include_inputs=True`.

::: probpipe.DistributionArray

::: probpipe.BroadcastDistribution

The truly private machinery (`_RecordDistributionView`, `FlattenedView`,
`_vmap_sample`, `_mc_expectation`) lives on [Internals](internals.md).
