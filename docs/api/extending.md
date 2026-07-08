# Extending ProbPipe

ProbPipe's extension surface is small and grouped by capability. The
table below maps each kind of extension to the contract you implement
against and the registry (if any) you register with. Each row links to
the section on this page that covers it in detail.

For the mental model behind the dispatch registries and the catalog —
when to reach for a `UnaryDispatchRegistry` vs a `BinaryDispatchRegistry`,
how to create a new registry, and how plugin registration works — see the
[dispatch registries contributor guide](../contributor/dispatch-conventions.md).
This page is the API reference it points back to.

| To add a... | Implement | Register with |
|---|---|---|
| New distribution family | Subclass of [`Distribution`](#distribution-base-classes), `RecordDistribution`, `NumericRecordDistribution`, or `TFPDistribution` | (none — capability is detected by `isinstance` against the matching [protocol](#protocols)) |
| New op support on an existing distribution | The matching underscore method (`_sample`, `_log_prob`, `_mean`, ...) on the class | (none — see [Protocols](#protocols) for which method backs which op) |
| New inference method (custom sampler, optimiser, ...) | Subclass of `InferenceMethod` declaring `supported_types`, `priority`, `check()`, and `execute()` | `inference_method_registry.register(...)` — see [Custom inference methods](#custom-inference-methods) |
| New distribution-to-distribution converter | Subclass of `Converter` with `check()` / `convert()` | `converter_registry.register(...)` — see [Custom converters](#custom-converters) |
| New canonical bijector for a `Constraint` | A factory returning a TFP bijector | `register_bijector(constraint_or_class, factory)` — see [Custom bijectors](#custom-bijectors) |
| New auxiliary-metadata adapter (custom array-like) | `capture` and `restore` callables | `register_aux(leaf_type, capture, restore)` — see [Custom auxiliary metadata](#custom-auxiliary-metadata) |
| New registry (cataloging surface) | Pass `name="..."` to a `BaseDispatchRegistry` subclass, *or* expose `name`/`description`/`kind` + `entry_summaries()`/`describe_entry()` on a non-conforming registry | Self-registers (named dispatch registries) *or* `registry_catalog.register(...)` (adapter pattern) — see [Registry catalog](#registry-catalog) |

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

`InferenceMethod` subclasses register with
`inference_method_registry` and declare `supported_types`, a
`priority`, and `check()` / `execute()` methods. When [`condition_on`](operations.md#conditioning) runs, the
registry tries methods in descending priority order and the first whose
`check()` reports feasibility wins. The built-in methods table is on
[Modeling and inference → Inference methods](inference.md#inference-methods).

### Setting priority for a new method

The integer returned by `priority` carries semantics: it tells the registry
whether your method should auto-dispatch, and if so, where it ranks
against the alternatives.

- **`priority > 50`** — *exact*: auto-dispatched, higher = preferred
  among exact alternatives.
- **`0 < priority <= 50`** — *inexact*: auto-dispatched, higher =
  preferred among inexact alternatives. The `50` break is documentary;
  the registry walks every positive priority uniformly.
- **`priority == 0`** — *opt-in only*: the registry skips the method
  during auto-dispatch. The method is reachable by name via
  `method="..."`. This is the default; an `InferenceMethod` subclass
  that doesn't override `priority` gets opt-in behaviour automatically.

#### Selection criteria

Choose a number with these axes in mind, roughly in order of weight:

1. **Robustness when applicable** — how often the method gives a usable
   answer without per-model tuning, conditional on `check()` passing.
2. **Computational cost per effective sample (or per converged result)**.
   Two kinds of cost advantage deserve separate consideration:
   *algorithmic* specialisation that exploits model structure for an
   asymptotic speedup (Kalman, INLA, conjugate updates), and
   *engineering* specialisation — same algorithm, faster backend
   (nutpie's Rust-backed NUTS vs. BlackJAX's; Stan's compiled gradients
   vs. JAX traces).
3. **Approximation quality** — analytical exact > controlled-error
   approximations > asymptotically-exact MCMC > intrinsic approximations.
4. **Diagnostic richness** — methods that fail silently rank below
   methods with built-in failure signals, all else equal.
5. **Model-class breadth** as a tiebreaker only. A broader-applicability
   method does not need a higher priority than a narrow one; whichever
   applies wins via `check()`.

#### Tier ranges — exact (51–100)

Five tiers, each 10 wide. Criteria are stated as positive properties of
the method.

| Range | Criterion |
|---|---|
| 91–100 | Per-call cost in a strictly better complexity class than general-purpose alternatives; the speedup comes from exploiting model structure. |
| 81–90 | Optimised implementation of a more general algorithm; lower constant-factor cost than the reference implementation within its applicable model class. |
| 71–80 | Self-tuning; converges robustly without per-model hyperparameter selection. |
| 61–70 | Well-understood with strong convergence theory; performs well once hand-tuned. |
| 51–60 | Slow per effective sample or unreliable in typical use. |

#### Tier ranges — inexact (1–50)

Four named tiers ordered by the strength of the asymptotic-to-exact
story. The slot at 11–20 is intentionally reserved for methods with
intermediate guarantees that don't fit cleanly into a named tier.

| Range | Criterion |
|---|---|
| 41–50 | Asymptotically exact under algorithmic refinement; bias is a knob the user can tighten (step size, mini-batch size). |
| 31–40 | Particle-based approximation refinable by particle count; quality improves with more particles, though convergence may be slow or unstable. |
| 21–30 | Parametric posterior approximation; error bounded by family expressiveness or by regularity conditions on the posterior shape. |
| 11–20 | *(reserved for methods with intermediate guarantees not covered by neighbouring tiers)* |
| 1–10 | No asymptotic-to-exact guarantee in practice; quality bounded by intrinsic information loss (summary statistics, fixed tolerance, learned representations). |

#### Setting `priority` on an `InferenceMethod` subclass

```python
class MyNutsMethod(InferenceMethod):
    @property
    def priority(self) -> int:
        # Tier 71-80 (self-tuning, broadly applicable).
        return 75
```

A method that should not auto-dispatch — perhaps it's experimental, has
sharp failure modes, or exists only for `method=` testing — leaves
`priority` at the inherited default of `0`. The registry will exclude
it from the auto-dispatch walk; users can still invoke it explicitly by
name.

::: probpipe.core._registry.BaseDispatchRegistry

::: probpipe.core._registry.UnaryDispatchRegistry

::: probpipe.core._registry.BinaryDispatchRegistry

::: probpipe.core._registry.BaseDispatchMethod

::: probpipe.core._registry.UnaryDispatchMethod

::: probpipe.core._registry.BinaryDispatchMethod

::: probpipe.core._registry.MethodInfo

## Registry catalog

The global :data:`registry_catalog` is a name-indexed view of every
registry in the process. It supplements dispatch (it does not replace
per-registry singletons like `inference_method_registry` or
`converter_registry`) and exists so contributors can answer two
questions at the REPL:

- *What registries exist?* — `print(probpipe.registry_catalog)` prints
  a table; `probpipe.registry_catalog.list()` returns one
  `RegistryInfo` per registry.
- *What's in this one?* — `probpipe.registry_catalog.describe("kl")`
  prints the per-method records (priority, supported types, module
  path) for the registry by that name.

Two ways to participate:

- **Conforming dispatch registries** (any subclass of
  `BaseDispatchRegistry`) self-register when given a non-empty
  `name=`. The default `register_in_catalog` is auto-derived from the
  name; tests can opt out with `register_in_catalog=False` to construct
  isolated registries that don't pollute the global singleton.
- **Non-conforming registries** (e.g. the `ConverterRegistry`, the
  bijector facade) — those whose dispatch shape doesn't fit
  `Unary`/`Binary` — satisfy `SupportsRegistryCataloging` directly by
  exposing `name`, `description`, `kind`, `entry_summaries()`, and
  `describe_entry()`. They register themselves at module load via
  `registry_catalog.register(...)`.

`SupportsRegistryCataloging` is intentionally *weaker* than the
dispatch contract: it only describes identity and introspection.
Empty-name guard and duplicate-name guard live on
`RegistryCatalog.register()`, so a structurally-conforming registry
without a name cannot accidentally land in the catalog.

::: probpipe.core._registry_catalog.RegistryCatalog

::: probpipe.core._registry_catalog.SupportsRegistryCataloging

::: probpipe.core._registry_catalog.EntrySummary

::: probpipe.core._registry_catalog.RegistryInfo

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
by `WorkflowFunction` when called with `workflow.with_options(include_inputs=True)(...)`.

::: probpipe.DistributionArray

::: probpipe.BroadcastDistribution

The truly private machinery (`_RecordDistributionView`, `_vmap_sample`,
`_mc_expectation`) lives on [Internals](internals.md), alongside the
public-but-rarely-constructed `FlattenedDistributionView` and
`NumericRecordDistributionView`.
