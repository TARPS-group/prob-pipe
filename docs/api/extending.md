# Extending ProbPipe

ProbPipe's extension surface is small and grouped by capability. The
table below maps each kind of extension to the contract you implement
against and the registry (if any) you register with. Each row links to
the section on this page that covers it in detail.

| To add a... | Implement | Register with |
|---|---|---|
| New distribution family | Subclass of [`Distribution`](#distribution-base-classes), `RecordDistribution`, `NumericRecordDistribution`, or `TFPDistribution` | (none — capability is detected by `isinstance` against the matching [protocol](#protocols)) |
| New op support on an existing distribution | The matching underscore method (`_sample`, `_log_prob`, `_mean`, ...) on the class | (none — see [Protocols](#protocols) for which method backs which op) |
| New inference method (custom sampler, optimiser, ...) | Subclass of `Method` declaring `supported_types`, `priority`, `check()`, and `execute()` | `inference_method_registry.register(...)` — see [Custom inference methods](#custom-inference-methods) |
| New distribution-to-distribution converter | Subclass of `Converter` with `check()` / `convert()` | `converter_registry.register(...)` — see [Custom converters](#custom-converters) |
| New canonical bijector for a `Constraint` | A factory returning a TFP bijector | `register_bijector(constraint_or_class, factory)` — see [Custom bijectors](#custom-bijectors) |
| New auxiliary-metadata adapter (custom array-like) | `capture` and `restore` callables | `register_aux(leaf_type, capture, restore)` — see [Custom auxiliary metadata](#custom-auxiliary-metadata) |

The two remaining sections — [Broadcasting internals](#broadcasting-internals-exposed-for-extension)
and the [Internals](internals.md) page — document classes that an
extension rarely constructs directly but may need to reference.

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

`Method` subclasses register with `inference_method_registry` and
declare `supported_types`, a `priority`, and `check()` / `execute()`
methods. When [`condition_on`](operations.md#conditioning) runs, the
registry tries methods in descending priority order and the first whose
`check()` reports feasibility wins. The built-in methods table is on
[Modeling and inference → Inference methods](inference.md#inference-methods).

::: probpipe.core._registry.MethodRegistry

::: probpipe.core._registry.Method

::: probpipe.core._registry.MethodInfo

## Custom converters

Subclass `Converter`, implement `check()` / `convert()`, and register
with `converter_registry.register(...)`. The built-in priorities and the
registry handle itself are documented under
[Conversion and interop](converters.md).

## Custom bijectors

`register_bijector` overrides the canonical bijector returned by
`bijector_for(c)` for a given `Constraint` instance or class. See
[Constraints → Bijectors](constraints.md#bijectors-for-unconstrained-reparameterization).

## Custom auxiliary metadata

`register_aux` extends the `Record` ↔ `NumericRecord` round-trip to a
new array-like type. See
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
