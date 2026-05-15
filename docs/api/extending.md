# Extending ProbPipe

The surface you implement against when adding a new distribution, inference
method, converter, bijector, or auxiliary-metadata adapter.

If you are only using ProbPipe's built-ins, you can skip this page — every
public verb lives on [Operations](operations.md) and every concrete container
on [Distributions](distributions/index.md), [Records](records.md), or
[Modeling and inference](inference.md).

## Distribution base classes

`Distribution` is the abstract root. `RecordDistribution` is the base for
distributions whose `_sample()` returns a [`Record`](records.md);
`NumericRecordDistribution` is the numeric specialisation backed by
`NumericRecord`. `TFPDistribution` delegates to an internal
`tensorflow_probability.distributions.Distribution` instance and is the
quickest path to wrapping a new TFP-backed family.

::: probpipe.Distribution

::: probpipe.RecordDistribution

::: probpipe.NumericRecordDistribution

::: probpipe.TFPDistribution

## Protocols

Protocols define capabilities that distributions may support. Each protocol
is `@runtime_checkable`, so compliance is checked via `isinstance` at
dispatch time. External distribution types can satisfy a protocol through
structural subtyping without inheriting from any ProbPipe class — implement
the underscore method (`_sample`, `_log_prob`, ...) on your class.

Protocol methods use an underscore prefix to distinguish the primitive
implementation from the public [op](operations.md).

The protocols below are grouped by capability — sampling and expectations,
density evaluation, moments, conditioning, and the one class-level protocol
for array backends.

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

`SupportsArrayBackend` is the only **class-level** protocol — its declared
method (`_make_array_backend`) is a `@classmethod`, so the runtime check is
`isinstance(MyDistribution, SupportsArrayBackend)` against the class itself,
not an instance.

::: probpipe.SupportsArrayBackend

## Custom inference methods

The inference dispatch system is a priority-ordered registry of `Method`
subclasses. Each method declares `supported_types`, a `priority`, and
`check()` / `execute()` methods. When `condition_on(model, data)` is
called, the registry tries methods in descending priority order; the first
whose `check()` returns `feasible=True` wins. Models do not implement
`_condition_on` directly — conditioning is handled entirely by registered
methods.

The shipped registry is `inference_method_registry` (see
[Modeling and inference → Inference methods](inference.md#inference-methods)
for the built-ins table and override examples).

::: probpipe.core._registry.MethodRegistry

::: probpipe.core._registry.Method

::: probpipe.core._registry.MethodInfo

## Custom converters

To add a new conversion (e.g. PyTorch ↔ ProbPipe), subclass `Converter`,
implement `check()` / `convert()`, and register the subclass with
`converter_registry.register(...)`. See
[Conversion and interop](converters.md) for the registry, the built-in
priorities, and the surrounding dataclasses.

## Custom bijectors

`bijector_for(c)` returns the canonical TFP bijector for a `Constraint`
`c`. `register_bijector` overrides defaults — for example, preferring
`Softplus` over `Exp` for `positive`. See
[Constraints → Bijectors for unconstrained reparameterization](constraints.md#bijectors-for-unconstrained-reparameterization).

## Custom auxiliary metadata

`register_aux` extends the `Record` ↔ `NumericRecord` round-trip to your own
array-like type. See
[Records → Auxiliary-metadata registry](records.md#auxiliary-metadata-registry).

## Broadcasting internals (exposed for extension)

`DistributionArray` is the shape-indexed container produced by parameter-sweep
workflow functions whose inner call returns a `Distribution`.
`BroadcastDistribution` is the joint container produced by
`@workflow_function` when `include_inputs=True`. Both are intentionally
narrow surfaces — most extension code should not need to construct either
directly, but anything implementing custom broadcasting (e.g., a new
parameter-sweep design) will refer to them.

::: probpipe.DistributionArray

::: probpipe.BroadcastDistribution

For the truly private machinery underneath (`_RecordDistributionView`,
`FlattenedView`, `_vmap_sample`, `_mc_expectation`), see
[Internals](internals.md).
