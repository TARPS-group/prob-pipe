# Extending ProbPipe

ProbPipe's extension surface is small and grouped by capability. The
table below maps each kind of extension to the contract you implement
against and the registry (if any) you register with. Each row links to
the section on this page that covers it in detail.

| To add a... | Implement | Register with |
|---|---|---|
| New distribution family | Subclass of [`Distribution`](#distribution-base-classes), `RecordDistribution`, `NumericRecordDistribution`, or `TFPDistribution` | (none — capability is detected by `isinstance` against the matching [protocol](#protocols)) |
| New op support on an existing distribution | The matching underscore method (`_sample`, `_log_prob`, `_mean`, ...) on the class | (none — see [Protocols](#protocols) for which method backs which op) |
| New inference method (custom sampler, optimiser, ...) | Subclass of `InferenceMethod` declaring `supported_types`, `priority`, `check()`, and `execute()` | `inference_method_registry.register(...)` — see [Custom inference methods](#custom-inference-methods) |
| New distribution-to-distribution converter | Subclass of `Converter` with `check()` / `convert()` | `converter_registry.register(...)` — see [Custom converters](#custom-converters) |
| New canonical bijector for a `Constraint` | A factory returning a TFP bijector | `register_bijector(constraint_or_class, factory)` — see [Custom bijectors](#custom-bijectors) |
| New array backend (custom array-like leaf type) | An `ArrayBackend` (shape / dtype / conversion hooks) | `register_array_backend(leaf_type, backend)` — see [Custom array backends](#custom-array-backends) |

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

::: probpipe.SupportsExactConditioning

::: probpipe.SupportsApproximateConditioning

`SupportsArrayBackend` is the only **class-level** protocol: its declared
method (`_make_array_backend`) is a `@classmethod`, so the runtime check
is `isinstance(MyDistribution, SupportsArrayBackend)` against the class
itself, not an instance.

::: probpipe.SupportsArrayBackend

## Custom inference methods

`InferenceMethod` subclasses register with
`inference_method_registry` and declare `supported_types`, whether they are
`exact`, a `priority`, and `check()` / `execute()` methods. When
[`condition_on`](operations.md#conditioning) runs, the registry tries the
methods in selection order and runs the first whose `check()` reports
feasibility. The built-in methods table is on
[Modeling and inference → Inference methods](inference.md#inference-methods).

### Exactness, then rank

A method declares two things about where it stands, and they are separate.

- **`exact`** says whether the result denotes the requested mathematical
  object or stands in for it. Exact methods are always tried before
  approximate ones, and `exact_only=True` on a call excludes the
  approximate ones. Every built-in inference method is approximate:
  a finite MCMC, SG-MCMC, slice, ABC, or variational output stands in for
  the conditional law whatever its asymptotic guarantee. `InferenceMethod`
  declares `exact = False` for you; a method that returns a representation
  of the conditional law itself overrides it.
- **`priority`** ranks methods of the same exactness, higher first; it is
  only a rank and carries no other meaning. `None`, the default, is
  **opt-in only**: the registry skips the method during auto-dispatch and
  it runs only when named via `method="..."`. A method that does not
  override `priority` is opt-in until a contributor ranks it, so registering
  one never changes what runs.

Ties go to the method whose declared types are closest to the argument's
class, then to registration order. `inference_method_registry.set_priorities(...)`
re-ranks at runtime, by keyword or by a mapping for names that are not
identifiers; it cannot change whether a method is exact.

#### Choosing a rank

Rank among the approximate methods with these axes in mind, roughly in
order of weight:

1. **Robustness when applicable** — how often the method gives a usable
   answer without per-model tuning, conditional on `check()` passing.
2. **Computational cost per effective sample (or per converged result)**.
   Two kinds of cost advantage deserve separate consideration:
   *algorithmic* specialisation that exploits model structure for an
   asymptotic speedup (Kalman, INLA, conjugate updates), and
   *engineering* specialisation — same algorithm, faster backend
   (nutpie's Rust-backed NUTS vs. BlackJAX's; Stan's compiled gradients
   vs. JAX traces).
3. **Approximation quality** — controlled-error approximations >
   asymptotically-exact MCMC > intrinsic approximations. These are
   guarantees a method documents, not a further exactness level.
4. **Diagnostic richness** — methods that fail silently rank below
   methods with built-in failure signals, all else equal.
5. **Model-class breadth** as a tiebreaker only. A broader-applicability
   method does not need a higher rank than a narrow one; whichever
   applies wins via `check()`.

The built-in ranks are anchors: `nutpie_nuts` 88, `blackjax_nuts` 85,
`cmdstan_nuts` and `pymc_nuts` 82, `blackjax_elliptical_slice` 75,
`blackjax_rwmh` 55, `blackjax_sgld` 45, `pyabc_smcabc` 6. Place a new
method relative to the nearest of these.

#### Setting `priority` on an `InferenceMethod` subclass

```python
class MySelfTuningMethod(InferenceMethod):
    @property
    def priority(self) -> int | None:
        # Self-tuning and broadly applicable: beside blackjax_elliptical_slice.
        return 75
```

A method that should not auto-dispatch — perhaps it's experimental, has
sharp failure modes, or exists only for `method=` testing — leaves
`priority` at the inherited default of `None`.

::: probpipe.core._dispatch.BaseDispatchRegistry

::: probpipe.core._dispatch.UnaryDispatchRegistry

::: probpipe.core._dispatch.BinaryDispatchRegistry

::: probpipe.core._dispatch.BaseDispatchMethod

::: probpipe.core._dispatch.UnaryDispatchMethod

::: probpipe.core._dispatch.BinaryDispatchMethod

::: probpipe.core._dispatch.Feasibility

::: probpipe.core._dispatch.MethodInfo

::: probpipe.core._dispatch.ResolutionError

::: probpipe.core._dispatch.MathematicalDomainError

## Custom converters

Subclass `Converter`, implement `check()` / `convert()`, and register
with `converter_registry.register(...)`. The built-in priorities and the
registry handle itself are documented under
[Conversion and interop](converters.md).

## Custom bijectors

`register_bijector` overrides the canonical bijector returned by
`bijector_for(c)` for a given `Constraint` instance or class. See
[Constraints → Bijectors](constraints.md#bijectors-for-unconstrained-reparameterization).

## Custom array backends

`register_array_backend` makes a new array-like container a first-class
numeric leaf: recognised, promoted, converted at the compute boundary, and
fingerprinted. See
[Records → Array-backend registry](records.md#array-backend-registry).

## Broadcasting internals (exposed for extension)

`DistributionArray` is the shape-indexed container produced by
parameter-sweep `Function` calls whose inner call returns a
`Distribution`. `BroadcastDistribution` is the joint container produced
by `Function` when called with `workflow.with_options(include_inputs=True)(...)`.

::: probpipe.DistributionArray

::: probpipe.BroadcastDistribution

The truly private machinery (`_RecordDistributionView`, `_vmap_sample`,
`_mc_expectation`) lives on [Internals](internals.md), alongside the
public-but-rarely-constructed `FlattenedDistributionView` and
`NumericRecordDistributionView`.
