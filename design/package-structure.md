# Package Structure

The package layout realizes the design reference as an import architecture: one package per layer of the reference, imports pointing strictly downward, and registries carrying capability upward. Like the rest of the reference, it describes the target state, but its module boundaries follow the divisions the implementation already has, so the layout is produced by moving modules rather than rewriting them.

`Function` (III.3) is the value-layer base, and the Part IV engine is installed on it at import, so its pieces span two packages; this document fixes where each is.

### Principles

- **One package per layer.** Packages mirror the reference's parts in dependency order, and a module realizes one section or one coherent piece of one.
- **Imports point downward.** Each package imports only from packages above it in the tree below. There are no import cycles, and no lazy imports to avoid one.
- **Registration is upward** (II.7). A lower layer defines a registry and higher layers populate it at import time, so capability becomes available to the operations without the operations importing their providers.
- **A spec is placed with the type it admits.** `FunctionSpec` is in the value layer with the kind it describes, and `DistributionSpec` and `ConditionalDistributionSpec` are with the distribution classes. The rule bends only where layering forbids it, since `core/` cannot import the value layer: `TermSpec` is what the tracked base stores, `RecordSpec` is the schema the shared layer itself uses, `InputSpec` and `OutputSpec` are the declarations the kind specs carry, and `Numeric` and `NumericSpec` are the interface pair the numeric kinds and their specs implement, so all of them are in `core/` with `NumericArraySpec`, `OpaqueSpec`, and `Constraint`. The same placement rule covers each type's batch form.
- **Modules are private, packages are public.** Every module is underscore-prefixed. A package's `__init__` exports its public names, and the top-level `probpipe` namespace re-exports the curated public API, which is the only import a user needs.
- **Tests mirror the tree**, as `tests/<package>/test_<module>.py`.

### Rationale

The layout makes the reference's dependency order mechanical: what a part may depend on is what its package may import, so the document and the code cannot drift on layering. Upward registration is `D2 – Generality first` in the import graph, since the supported set grows by adding a provider package rather than by widening a lower layer. The single curated namespace serves `C3 – Computational detail hidden by default, available on demand`: module paths stay free to change, and a user's imports do not. Module boundaries on the implementation's existing divisions keep the reorganization concrete: each target module names work that is already one coherent unit.

### The tree

```
probpipe/
├── __init__.py                # the curated public API
├── core/                      # Part II — shared abstractions
│   ├── _named_tree.py         #   NamedTree (II.6)
│   ├── _constraints.py        #   Constraint and the constraint factories (II.3)
│   ├── _specs.py              #   TermSpec, NumericArraySpec, OpaqueSpec (II.1), InputSpec, OutputSpec (II.2)
│   ├── _numeric.py            #   Numeric and its spec-side mixin NumericSpec (II.3)
│   ├── _array_backend.py      #   the array-backend registry for native numeric leaves (II.3)
│   ├── _record_spec.py        #   RecordSpec, NumericRecordSpec, unification (III.5)
│   ├── _identity.py           #   TrackedTerm, Provenance, fingerprints (II.4)
│   ├── _batch.py              #   Batch, BatchSpec: axis groups, level names, at_levels (II.5)
│   ├── _dispatch.py           #   dispatch methods and registries, Fidelity, MethodInfo, ResolutionError (II.7)
│   ├── _catalog.py            #   EntrySummary, RegistryCatalog (II.7)
│   └── _config.py             #   library configuration
├── values/                    # the value layer (III.1–III.6; LinOp, III.4, is in linalg/)
│   ├── _numeric_array.py      #   NumericArray (III.1)
│   ├── _numeric_array_batch.py  #   NumericArrayBatch (III.1)
│   ├── _opaque.py             #   Opaque (III.2)
│   ├── _function_base.py      #   Function itself (declared sides, identity, controls and with_options,
│   │                          #     plain evaluation, install_call_engine), FunctionSpec, the function
│   │                          #     capabilities, and is_differentiable
│   ├── _object_batch.py       #   object-array storage the two batch forms share
│   ├── _function_batch.py     #   FunctionBatch (III.3)
│   ├── _opaque_batch.py       #   OpaqueBatch (III.2)
│   ├── _record.py             #   Record, NumericRecord (III.5)
│   ├── _record_batch.py       #   RecordBatch (III.6)
│   └── _numeric_record_batch.py  #   NumericRecordBatch (III.6)
├── linalg/                    # LinOp, the linear Function subtype (III.4)
│   ├── _linop.py              #   LinOp: the action, the queries, flags
│   ├── _structured.py         #   Dense / Diagonal / Triangular / Cholesky / Root …
│   ├── _composites.py         #   Product / Sum / Scaled / Transpose — the operator algebra
│   └── _batch.py              #   LinOpBatch
├── distributions/             # the distribution layer (III.7–III.15)
│   ├── _distribution.py       #   Distribution, NumericDistribution, DistributionSpec (III.7)
│   ├── _views.py              #   FieldView (III.7–III.8)
│   ├── _capabilities.py       #   the Supports* protocols (III.8)
│   ├── _conditional.py        #   ConditionalDistribution, its markers and spec (III.9)
│   ├── _batches.py            #   DistributionBatch, ConditionalDistributionBatch (III.10)
│   ├── _factored.py           #   SupportsFactors and the factored classes (III.11)
│   ├── _composition.py        #   the * engine behind __mul__ (III.12)
│   ├── _empirical.py          #   EmpiricalDistribution (VI.2) — the closure family the
│   │                          #     lift and the Monte Carlo fallbacks construct
│   ├── _conversion.py         #   Converter, ConverterRegistry (III.14)
│   └── _reparameterization.py #   bijector_for, register_bijector, is_invertible (III.15)
├── functions/                 # Part IV — Function and its engine
│   ├── _function.py           #   the engine installed on Function at import; the decorator; with_options (IV.1, IV.2)
│   ├── _call.py               #   binding, and normalization as wrap, convert, admit; ApplicabilityError (IV.3, IV.4)
│   ├── _plan.py               #   lift classification, root-ancestor grouping, and the result declaration (IV.5, IV.6)
│   ├── _rules.py              #   the evaluation-rule registry: consulted by the engine,
│   │                          #     populated upward by the families (IV.7)
│   ├── _broadcast.py          #   the sampling lift over distributions, include_inputs (IV.9, IV.10)
│   ├── _sweep.py              #   the batch sweep (IV.9, IV.10)
│   ├── _rng.py                #   structural event identity, the versioned key derivation (IV.8)
│   ├── _context.py            #   workflow scopes and frames: workflow_run (IV.8)
│   ├── _replay.py             #   replay_run, replay records and anchors; the cache key (IV.8)
│   ├── _broker.py             #   managed work items: keys across threads, tasks, and flows (IV.8, IV.9)
│   ├── _execution.py          #   the jax / sequential / thread dispatch modes, the execution contract (IV.9)
│   ├── _orchestration.py      #   optional tracing (IV.9)
│   └── _result.py             #   assembly, declaration enforcement, kind-directed wrap, identity, provenance (IV.10)
├── operations/                # Part V — the operations
│   ├── _operation.py          #   the @operation decorator: roles, conditions, and result rule;
│   │                          #     OperationRoute and its four helpers, route resolution, and the
│   │                          #     operation registry (V.0)
│   ├── _evaluate.py           #   evaluate and its rule registry (V.1)
│   ├── _inverse.py            #   inverse, log_det_jacobian (V.2)
│   ├── _sample.py             #   sample (V.3)
│   ├── _density.py            #   log_prob, unnormalized_log_prob (V.4)
│   ├── _moments.py            #   mean, variance, cov, quantile, expectation (V.5)
│   ├── _condition.py          #   condition_on, the inference registry (V.6)
│   ├── _joint.py              #   joint (V.7)
│   ├── _marginal.py           #   marginal, factor (V.8)
│   ├── _mixture.py            #   mixture (V.9)
│   └── _convert.py            #   convert (V.10)
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
│   ├── _programs.py           #   StanModel, PyMCModel: program-defined joints (VI.9)
│   └── _converters.py         #   the shipped converters (III.14)
├── designs/                   # designs: batches materialized from per-field candidate sets, over any element spec
├── inference/                 # the registered inference methods (V.6)
├── diagnostics/               # diagnostics over inference results
└── validation/                # predictive checks and model comparison
```

### The layers

- **`core/`** is Part II plus `RecordSpec` (III.5), which the shared layer needs: generic, type-agnostic, and importable by everything.
- **`values/`** is the value layer of Part III, covering every leaf kind, `Function`'s base included (III.3); `LinOp` subclasses it and the spec references it, both below the distribution layer.
- **`linalg/`** is the linear subtype and its operator algebra, kept as its own package because the structured subclasses and composites are a coherent domain of their own.
- **`distributions/`** is the distribution layer of Part III, through composition, conversion, and reparameterization. `EmpiricalDistribution` is here rather than with the other families: it is the closure family that the lift and every Monte Carlo fallback construct, so it must be below the code that uses it. Its Part VI entry is unchanged, and the placement is the single exception to part-per-package.
- **`functions/`** is the `Function` engine, installed on the III.3 base at import, one package because it is one machine. Controls, binding, normalization, lift classification, planning, rule resolution, the workflow scopes and structural keys, replay and caching, execution under a dispatch mode, orchestration, and return are the steps of one stack (IV.1), and they change together. It is above `distributions/` because lifting samples distributions and materializes empirical results.
- **`operations/`** is thin by design, matching what the operations are: a declaration wrapped by the decorator with its routes registered beside it, one module per operation section (V.1–V.10) above `_operation.py`, which holds V.0's decorator, route protocol, resolution, and registry. V.11's batching is the engine's sweep, so it is no module here. The inference-method registry is defined here with `condition_on` and populated from above; the evaluation-rule registry is defined with the engine (`functions/_rules.py`), which consults it, with `evaluate` as its operation form.
- **`families/`** implements the catalog: constructors and capability implementations, registering its evaluation rules and converters upward at import.
- **`inference/`**, **`diagnostics/`**, and **`validation/`** are outside the reference's parts: inference methods register into the V.6 registry, and diagnostics and validation are application layers over the public operations.
- **`designs/`** builds a `Batch` of any element kind from per-field candidate sets combined by a rule, the full factorial being the Cartesian product, on a level named `design`; a design is a distinct concept from the batch it produces, and its section is to be written.
- **Experimental, placement to be decided.** `Module`, `AbstractModule`, `workflow_method`, `abstract_workflow_method`, and `Module.dag()` in `core/node.py` form a container of `Function`s with shared inputs and a Graphviz view of their graph. They stay outside the reference until their role is settled, at low priority.

A handful of private helper modules (dtypes, array utilities) support the packages and carry no design contract.

### Correspondence to the implementation

The main moves, for orientation; the target contracts above are authoritative.

| Today | Target |
|---|---|
| `core/node.py` (`Function`, the decorator, `with_options`) | `functions/_function.py` |
| `core/node.py` (`Module`, `AbstractModule`, `workflow_method`, `abstract_workflow_method`, `Module.dag()`) | experimental; placement to be decided |
| `core/_workflow_call.py`, `core/_workflow_distribution_normalization.py` | `functions/_call.py` |
| `core/_workflow_plan.py` | `functions/_plan.py` |
| `core/_function_contract.py` | split: construction-time validation of the declared sides to `values/_function_base.py`; per-call binding and the result declaration to `functions/_plan.py`; output validation and declared wrapping to `functions/_result.py` |
| `core/_workflow_distribution_broadcast.py` | `functions/_broadcast.py` |
| `core/_workflow_sweep.py` | `functions/_sweep.py` |
| `core/_workflow_rng.py` | `functions/_rng.py` |
| `core/_workflow_context.py` | `functions/_context.py` |
| `core/_workflow_replay.py`, `core/_workflow_recipe.py` | `functions/_replay.py` |
| `core/_workflow_broker.py`, `core/_workflow_managed.py` | `functions/_broker.py` |
| `core/_workflow_execution.py`, `core/_workflow_execution_contract.py` | `functions/_execution.py` |
| `core/_workflow_result.py` | `functions/_result.py` |
| `core/ops.py` | `operations/`, one module per operation section (V.1–V.10), plus `_operation.py` for the declaration, route, and registry code |
| `core/distribution.py`, `core/_distribution_base.py` | `distributions/_distribution.py` |
| `core/protocols.py` | `distributions/_capabilities.py` |
| `core/_distribution_array.py`, `core/_broadcast_distributions.py` | `distributions/_batches.py` |
| `core/_empirical.py` | `distributions/_empirical.py` |
| `inference/_registry.py` (the registry object, today imported upward by `core/ops.py`) | `operations/_condition.py`; the methods stay in `inference/`, and the edge points downward |
| `core/named_tree.py`, `core/tracked.py`, `core/provenance.py`, `core/_registry.py` | `core/`, one module per II section |
| `core/_numeric_array.py`, `core/_opaque.py`, `core/record.py`, and their batch modules | `values/`, one module per III section |
| `core/event_template.py`, `core/constraints.py` | split in place: `core/_specs.py`, `core/_record_spec.py`, `core/_numeric.py`, `core/_constraints.py` (II.1–II.3, III.5) |
| `record/design.py` | `designs/`, generalized from `RecordBatch` to any element spec |
| `core/_record_distribution.py` | `distributions/_views.py` (`FieldView`, III.7); `RecordDistribution` is retired (III.13) |
| `core/_numeric_record_distribution.py` | the numeric marker to `distributions/_distribution.py` (III.7) and `BootstrapDistribution` to `families/_resampling.py` (VI.2); `FlatNumericRecordDistribution`, `FlattenedDistributionView`, and `NumericRecordDistributionView` are retired, the flat view being `evaluate(to_vector, d)` (III.7) |
| `modeling/_stan.py`, `modeling/_pymc.py` | `families/_programs.py` (VI.9) |
| `modeling/_glm.py` | `families/_conditional.py` (VI.8) |
| `modeling/_base.py`, `modeling/_simple.py`, `modeling/_simple_generative.py`, and `Likelihood`, `ConditionallyIndependentLikelihood`, `GenerativeLikelihood` in `core/protocols.py` | retired: a model is a program-defined family (VI.9) or a factored joint (III.11), and a learned likelihood is a `ConditionalDistribution` (III.9) |
| `modeling/_likelihood.py` (`IncrementalConditioner`) | retired as a class; a fold of `condition_on` over data batches, settled with `iterate` |
| `converters/_registry.py`, `converters/_protocol.py` | `distributions/_conversion.py` (III.14): `ConversionMethod` becomes `Fidelity`, `Converter.convert` becomes `execute`, and the protocol resolver becomes protocol targets |
| `converters/_probpipe.py`, `converters/_scipy.py`, `converters/_tfp.py` | `families/_converters.py` (III.14) |
| `expectation`'s `return_dist` and `set_return_approx_dist` (`core/ops.py`, `core/_distribution_base.py`) | retired: the error of a Monte Carlo estimate is taken explicitly through the bootstrap (VI.2); `set_default_num_evaluations` becomes the sample-count default in `core/_config.py` (IV.2) |

### Open points

- *Incremental conditioning.* `IncrementalConditioner` (`modeling/_likelihood.py`) is a fold of `condition_on` over data batches; whether it is written as a derived operation or as a workflow recipe is settled together with `iterate`.
