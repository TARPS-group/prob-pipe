# Package Structure

The package layout realizes the design reference as an import architecture: one package per layer of the reference, imports pointing strictly downward, and registries carrying capability upward. Like the rest of the reference, it describes the target state, but its module boundaries follow the seams the implementation already has, so the layout is reachable by moving modules rather than rewriting them.

This document uses the **`Function`** vocabulary. A `Function` is the universal function representation: a tracked term carrying an input and an output template and wrapping any Python callable. `LinOp` is its linear subtype, and invertibility is a capability a `Function` may claim. The value-layer specs stay callable-generic: a `FunctionSpec` admits any callable — a `Function` is one such, never the required type — and a `FunctionBatch` holds a collection of them. The base class is `Function` itself, in the value layer, carrying templates, identity, and plain evaluation; the Part IV engine is installed on it at import, registration flowing upward. This document fixes where its pieces live.

### Principles

- **One package per layer.** Packages mirror the reference's parts in dependency order, and a module realizes one section or one coherent piece of one.
- **Imports point downward.** Each package imports only from packages above it in the tree below. There are no import cycles, and no lazy imports to dodge one.
- **Registration flows upward.** A lower layer defines a registry and higher layers populate it at import time: the families register evaluation rules and converters, and the inference methods register themselves. Capability reaches the operations without the operations importing their providers.
- **A spec lives with the type it admits.** `ArraySpec`, `OpaqueSpec`, and `FunctionSpec` live in the value layer with the value kinds they describe, and `DistributionSpec` and `ConditionalDistributionSpec` live with the distribution classes. The same placement rule covers each type's batch form.
- **Modules are private, packages are public.** Every module is underscore-prefixed. A package's `__init__` exports its public names, and the top-level `probpipe` namespace re-exports the curated user surface, which is the only import a user needs.
- **Tests mirror the tree**, as `tests/<package>/test_<module>.py`.

### Rationale

The layout makes the reference's dependency order mechanical: what a part may depend on is what its package may import, so the document and the code cannot drift on layering. Upward registration is `D2 – Generality first` in the import graph, since the supported set grows by adding a provider package rather than by widening a lower layer. The single curated namespace serves `C3 – Computational detail hidden by default, available on demand`: module paths stay free to change, and a user's imports do not. Module boundaries on the implementation's existing seams keep the reorganization honest: each target module names work that is already one coherent unit.

### The tree

```
probpipe/
├── __init__.py                # the curated public API
├── core/                      # Part II — shared abstractions
│   ├── _named_tree.py         #   NamedTree (II.1)
│   ├── _identity.py           #   Tracked, Annotated, Provenance, fingerprints (II.2)
│   ├── _batch.py              #   Batch: axis groups, level names, select (II.3)
│   ├── _dispatch.py           #   dispatch methods and registries (II.4)
│   ├── _catalog.py            #   EntrySummary, RegistryCatalog (II.4)
│   └── _config.py             #   library configuration
├── values/                    # the value layer (III.1–III.4)
│   ├── _constraints.py        #   Constraint and the constraint factories (III.1)
│   ├── _specs.py              #   ValueSpec, ArraySpec, OpaqueSpec (III.1)
│   ├── _event_template.py     #   EventTemplate, NumericEventTemplate, unification (III.1)
│   ├── _function_base.py      #   Function itself (templates, identity, controls and with_options,
│   │                          #     plain evaluation), FunctionSpec, and the function capabilities
│   ├── _function_batch.py     #   FunctionBatch (III.2)
│   ├── _opaque_batch.py       #   OpaqueBatch (III.2)
│   ├── _record.py             #   Record, NumericRecord (III.3)
│   └── _record_batch.py       #   RecordBatch, NumericRecordBatch (III.4)
├── linalg/                    # LinOp, the linear Function subtype (III.5)
│   ├── _linop.py              #   LinOp: the action, the queries, flags
│   ├── _structured.py         #   Dense / Diagonal / Triangular / Cholesky / Root …
│   ├── _composites.py         #   Product / Sum / Scaled / Transpose — the operator algebra
│   └── _batch.py              #   LinOpBatch
├── distributions/             # the distribution layer (III.6–III.14)
│   ├── _distribution.py       #   Distribution, NumericDistribution, DistributionSpec (III.6)
│   ├── _views.py              #   FieldView (III.6–III.7)
│   ├── _capabilities.py       #   the Supports* protocols (III.7)
│   ├── _conditional.py        #   ConditionalDistribution, its markers and spec (III.8)
│   ├── _batches.py            #   DistributionBatch, ConditionalDistributionBatch (III.9)
│   ├── _factored.py           #   SupportsFactors and the factored classes (III.10)
│   ├── _composition.py        #   the * engine behind __mul__ (III.11)
│   ├── _empirical.py          #   EmpiricalDistribution (VI.2) — the closure family the
│   │                          #     lift and the Monte Carlo fallbacks construct
│   ├── _conversion.py         #   Converter, ConverterRegistry (III.13)
│   └── _reparameterization.py #   Constraint → invertible Function (III.14)
├── functions/                 # Part IV — Function and its engine
│   ├── _function.py           #   the engine installed on Function at import; the wrapping decorator
│   ├── _call.py               #   argument classification: the lifting trigger (IV.2)
│   ├── _plan.py               #   broadcast planning, root-ancestor grouping (IV.2)
│   ├── _rules.py              #   the evaluation-rule registry: consulted by the engine,
│   │                          #     populated upward by the families (V.6)
│   ├── _broadcast.py          #   the sampling lift over distributions, include_inputs (IV.2)
│   ├── _sweep.py              #   the batch sweep (IV.2)
│   ├── _keys.py               #   the structural key split from seed (IV.3)
│   ├── _execution.py          #   jax / sequential / thread dispatch (IV.4)
│   ├── _orchestration.py      #   optional tracing (IV.4)
│   └── _result.py             #   output wrapping, identity, provenance (IV.1, V.0)
├── operations/                # Part V — the operations
│   ├── _moments.py            #   mean, variance, cov, quantile, expectation (V.1)
│   ├── _sample.py             #   sample (V.2)
│   ├── _density.py            #   log_prob, unnormalized_log_prob (V.3)
│   ├── _condition.py          #   condition_on, predictive, the inference registry (V.4)
│   ├── _joint.py              #   joint (V.5)
│   ├── _evaluate.py           #   evaluate and its rule registry (V.6)
│   └── _marginal.py           #   marginal, factor (V.7)
├── families/                  # Part VI — the distribution catalog
│   ├── _backend.py            #   TFPDistribution, the backend adapter (VI.1)
│   ├── _continuous.py         #   Normal, Gamma, … (VI.1)
│   ├── _discrete.py           #   Bernoulli, Poisson, … (VI.1)
│   ├── _multivariate.py       #   MultivariateNormal, Dirichlet, … (VI.1)
│   ├── _resampling.py         #   bootstrap forms, KDE and its kernels (VI.2)
│   ├── _mixture.py            #   MixtureDistribution (VI.3)
│   ├── _transformed.py        #   the evaluation-result families (VI.4)
│   ├── _random_functions.py   #   RandomFunction, RandomMeasure (VI.5)
│   ├── _gaussian.py           #   the Gaussian algebra (VI.6)
│   ├── _conditional.py        #   LinearGaussianConditional, the GLM assembly (VI.8)
│   └── _converters.py         #   the shipped converters (III.13)
├── inference/                 # the registered inference methods (V.4)
├── diagnostics/               # diagnostics over inference results
└── validation/                # predictive checks and model comparison
```

### The layers

- **`core/`** is Part II verbatim: generic, type-agnostic, and importable by everything.
- **`values/`** is the value layer of Part III, covering every leaf kind. The function kind's base lives here: `Function` itself, the class a `FunctionSpec` admits and a `FunctionBatch` holds, together with its capability protocols. The base carries the *representation* only: templates, identity, and plain evaluation. `LinOp` subclasses it and the spec references it, both below the distribution layer.
- **`linalg/`** is the linear subtype and its operator algebra, kept as its own package because the structured subclasses and composites are a coherent domain of their own.
- **`distributions/`** is the distribution layer of Part III, through composition, conversion, and reparameterization. `EmpiricalDistribution` lives here rather than with the other families: it is the closure family that the lift and every Monte Carlo fallback construct, so it must sit below the machinery that uses it. Its Part VI entry is unchanged, and the placement is the single exception to part-per-package.
- **`functions/`** is the `Function` engine, installed on the III.2 base at import, one package because it is one machine. Argument classification, planning and grouping, the sampling lift, the batch sweep, the key split, execution dispatch, orchestration, and result wrapping are the stages of one call path, and they change together. It sits above `distributions/` because lifting samples distributions and materializes empirical results.
- **`operations/`** is thin by design, matching what the operations are: capability-dispatched definitions wrapped by the decorator, one module per operation section (V.1–V.7). V.0's operation model is the wrapping itself, and V.8's batching is the engine's sweep, so neither is a module here. The inference-method registry is defined here with `condition_on` and populated from above; the evaluation-rule registry lives with the engine (`functions/_rules.py`), which consults it, with `evaluate` as its operation form.
- **`families/`** implements the catalog: constructors and capability implementations, registering its evaluation rules and converters upward at import.
- **`inference/`**, **`diagnostics/`**, and **`validation/`** sit outside the reference's parts: inference methods register into the V.4 registry, and diagnostics and validation are application layers over the public operations.

A handful of private helper modules (dtypes, array utilities) support the packages and carry no design contract.

### Correspondence to the implementation

The load-bearing moves, for orientation; the target contracts above are authoritative.

| Today | Target |
|---|---|
| `core/node.py` (`WorkflowFunction`, the decorator, `with_options`) | `functions/_function.py` |
| `core/_workflow_call.py` | `functions/_call.py` |
| `core/_workflow_plan.py` | `functions/_plan.py` |
| `core/_workflow_distribution_broadcast.py` | `functions/_broadcast.py` |
| `core/_workflow_sweep.py` | `functions/_sweep.py` |
| `core/_workflow_execution.py` | `functions/_execution.py` |
| `core/_workflow_result.py`, `core/_workflow_distribution_normalization.py` | `functions/_result.py` |
| `core/ops.py` | `operations/`, one module per operation section (V.1–V.7) |
| `core/distribution.py`, `core/_distribution_base.py` | `distributions/_distribution.py` |
| `core/protocols.py` | `distributions/_capabilities.py` |
| `core/_distribution_array.py`, `core/_broadcast_distributions.py` | `distributions/_batches.py` |
| `core/_empirical.py` | `distributions/_empirical.py` |
| `inference/_registry.py` (the registry object, today imported upward by `core/ops.py`) | `operations/_condition.py`; the methods stay in `inference/`, and the edge points downward |
| `core/named_tree.py`, `core/tracked.py`, `core/provenance.py`, `core/_registry.py` | `core/`, one module per II section |
| `core/event_template.py`, `core/record.py`, `core/constraints.py`, `core/_record_array.py` | `values/`, one module per III section |

### Open points

- *Model-construction helpers.* The GLM assembly lands in the catalog; whether the remaining model-building conveniences earn a package is settled by the catalog consolidation.
