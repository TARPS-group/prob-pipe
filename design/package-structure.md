# Package Structure

The package layout realizes the design reference as an import architecture: one package per layer of the reference, imports pointing strictly downward, and registries carrying capability upward. Like the rest of the reference, it describes the target state, but its module boundaries follow the divisions the implementation already has, so the layout is produced by moving modules rather than rewriting them.

`Function` (III.3) is the value-layer base, and the Part V engine is installed on it at import, so its pieces span two packages; this document fixes where each is.

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
│   ├── _specs.py              #   TermSpec, NumericArraySpec, OpaqueSpec (II.1), InputSpec, OutputSpec and component projection contracts (II.2)
│   ├── _kinds.py              #   the kind table: register_kind, term_class_for_spec, batch_class_for_spec (II.1)
│   ├── _numeric.py            #   Numeric and its spec-side mixin NumericSpec (II.3)
│   ├── _array_backend.py      #   the array-backend registry for native numeric leaves (II.3)
│   ├── _record_spec.py        #   RecordSpec, NumericRecordSpec, unification (III.5)
│   ├── _identity.py           #   TrackedTerm with annotations on the base, Immutable, Provenance, fingerprints, the provenance traversal (II.4)
│   ├── _batch.py              #   Batch, BatchSpec: axis groups, level names, at_levels (II.5)
│   ├── _dispatch.py           #   dispatch methods and registries, Fidelity, MethodInfo, ResolutionError, MathematicalDomainError (II.7)
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
├── distributions/             # the distribution layer (III.7–IV.3)
│   ├── _distribution.py       #   Distribution, NumericDistribution, DistributionSpec (III.7)
│   ├── _views.py              #   FieldView (III.7–III.8)
│   ├── _capabilities.py       #   the Supports* protocols (III.8)
│   ├── _conditional.py        #   ConditionalDistribution, its markers and spec (III.9)
│   ├── _batches.py            #   DistributionBatch, ConditionalDistributionBatch (III.10)
│   ├── _factored.py           #   SupportsFactors and the factored classes (IV.1)
│   ├── _composition.py        #   the * engine behind __mul__ (IV.2)
│   ├── _empirical.py          #   EmpiricalDistribution (VII.2) — the closure family the
│   │                          #     lift and the Monte Carlo fallbacks construct
│   └── _conversion.py         #   Converter, ConverterRegistry (IV.3)
├── functions/                 # Part V — Function and its engine
│   ├── _function.py           #   the engine installed on Function at import; the decorator; with_options (V.1, V.2)
│   ├── _call.py               #   binding, wrap, conversion planning, admission; ApplicabilityError (V.3, V.4)
│   ├── _plan.py               #   lift classification, root-ancestor grouping, and the result declaration (V.5, V.6)
│   ├── _rules.py              #   the evaluation-rule registry: consulted by the engine,
│   │                          #     populated upward by the families (V.7)
│   ├── _broadcast.py          #   the sampling lift over distributions, include_inputs (V.9, V.10)
│   ├── _sweep.py              #   the batch sweep (V.9, V.10)
│   ├── _rng.py                #   structural event identity, the versioned key derivation (V.8)
│   ├── _context.py            #   workflow scopes and frames: workflow_run (V.8)
│   ├── _replay.py             #   replay_run, replay records and anchors; the cache key (V.8)
│   ├── _broker.py             #   managed work items: keys across threads, tasks, and flows (V.8, V.9)
│   ├── _execution.py          #   the jax / sequential / thread dispatch modes, the execution contract (V.9)
│   ├── _orchestration.py      #   optional tracing (V.9)
│   ├── _result.py             #   assembly, output inference, declaration enforcement and errors, kind-directed wrap, identity (V.10)
│   └── _reparameterization.py #   bijector_for, register_bijector (V.12)
├── operations/                # Part VI — the operations
│   ├── _operation.py          #   the @operation decorator: roles, conditions, and result rule;
│   │                          #     OperationRoute and its four helpers, route resolution, and the
│   │                          #     operation registry (VI.0)
│   ├── _evaluate.py           #   evaluate (VI.1); its registry is functions/_rules.py
│   ├── _inverse.py            #   inverse, log_det_jacobian (VI.2)
│   ├── _sample.py             #   sample (VI.3)
│   ├── _density.py            #   log_prob, unnormalized_log_prob (VI.4)
│   ├── _moments.py            #   mean, variance, cov, quantile, expectation (VI.5)
│   ├── _condition.py          #   condition_on, the inference registry (VI.6)
│   ├── _joint.py              #   joint (VI.7)
│   ├── _marginal.py           #   marginal, factor (VI.8)
│   ├── _mixture.py            #   mixture (VI.9)
│   └── _convert.py            #   convert (VI.10)
├── families/                  # Part VII — the distribution catalog
│   ├── _backend.py            #   TFPDistribution, the backend adapter (VII.1)
│   ├── _continuous.py         #   Normal, Gamma, … (VII.1)
│   ├── _discrete.py           #   Bernoulli, Poisson, … (VII.1)
│   ├── _multivariate.py       #   MultivariateNormal, Dirichlet, … (VII.1)
│   ├── _resampling.py         #   bootstrap forms, KDE and its kernels (VII.2)
│   ├── _mixture.py            #   MixtureDistribution (VII.3)
│   ├── _transformed.py        #   the evaluation-result families (VII.4)
│   ├── _random_functions.py   #   RandomFunction, RandomMeasure (VII.5)
│   ├── _gaussian.py           #   the Gaussian algebra (VII.6)
│   ├── _conditional.py        #   LinearGaussianConditional, the GLM assembly (VII.8)
│   ├── _programs.py           #   StanModel, PyMCModel: backend models with explicit given/event contracts (VII.9)
│   └── _converters.py         #   the shipped converters (IV.3)
├── designs/                   # designs: batches materialized from per-field candidate sets, over any element spec
├── inference/                 # the registered inference methods (VI.6)
├── diagnostics/               # diagnostics over inference results
└── validation/                # predictive checks and model comparison
```

### The layers

- **`core/`** is Part II plus `RecordSpec` (III.5), which the shared layer needs: generic, type-agnostic, and importable by everything.
- **`values/`** is the value layer of Part III, covering every leaf kind, `Function`'s base included (III.3); `LinOp` subclasses it and the spec references it, both below the distribution layer.
- **`linalg/`** is the linear subtype and its operator algebra, kept as its own package because the structured subclasses and composites are a coherent domain of their own.
- **`distributions/`** is the distribution layer of Parts III and IV, through composition and conversion. `EmpiricalDistribution` is here rather than with the other families: it is the closure family that the lift and every Monte Carlo fallback construct, so it must be below the code that uses it. Its Part VII entry is unchanged, and the placement is the single exception to part-per-package.
- **`functions/`** is the `Function` engine, installed on the III.3 base at import, one package because it is one machine. Controls, binding, normalization, lift classification, planning, route resolution, the workflow scopes and structural keys, replay and caching, execution under a dispatch mode, orchestration, and return are the steps of one stack (V.1), and they change together. The constraint-to-bijector factory is here too, since a bijector is a `Function` (V.12). It is above `distributions/` because lifting samples distributions and materializes empirical results.
- **`operations/`** is thin by design, matching what the operations are: a declaration wrapped by the decorator with its routes registered beside it, one module per operation section (VI.1–VI.10) above `_operation.py`, which holds VI.0's decorator, route protocol, resolution, and registry. VI.11's batching is the engine's sweep, so it is no module here. The inference-method registry is defined here with `condition_on` and populated from above; the evaluation-rule registry is defined with the engine (`functions/_rules.py`), which consults it, with `evaluate` as its operation form.
- **`families/`** implements the catalog: constructors and capability implementations, registering its evaluation rules and converters upward at import.
- **`inference/`**, **`diagnostics/`**, and **`validation/`** are outside the reference's parts: inference methods register into the VI.6 registry, and diagnostics and validation are application layers over the public operations.
- **`designs/`** builds a `Batch` of any element kind from per-field candidate sets combined by a rule, the full factorial being the Cartesian product, on a level named `design`; a design is a distinct concept from the batch it produces, and its section is to be written.
- **Experimental, placement to be decided.** `Module`, `AbstractModule`, `workflow_method`, `abstract_workflow_method`, and `Module.dag()` in `core/node.py` form a container of `Function`s with shared inputs and a Graphviz view of their graph. They stay outside the reference until their role is settled, at low priority.

A handful of private helper modules (dtypes, array utilities) support the packages and carry no design contract.

### Correspondence to the implementation

Every module with a design contract, with where it goes; the target contracts above are authoritative.

| Today | Target |
|---|---|
| `core/node.py` (`Function`, the decorator, `with_options`) | `functions/_function.py` |
| `core/node.py` (`Module`, `AbstractModule`, `workflow_method`, `abstract_workflow_method`, `Module.dag()`) | experimental; placement to be decided |
| `core/_workflow_call.py`, `core/_workflow_distribution_normalization.py` | `functions/_call.py`; conversion executes later under the IV.3 plan |
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
| `core/ops.py` | `operations/`, one module per operation section (VI.1–VI.10), plus `_operation.py` for the declaration, route, and registry code |
| `core/distribution.py`, `core/_distribution_base.py` | `distributions/_distribution.py` |
| `core/protocols.py` | `distributions/_capabilities.py` |
| `core/_distribution_array.py` | `distributions/_batches.py` (III.10) |
| `core/_broadcast_distributions.py` | split: `BroadcastDistribution` is retired, the lift's joint result being an `EmpiricalDistribution` (V.10, VII.2); the row aggregator `_make_stack` to `functions/_result.py` (V.10); the mixture and record marginals to `operations/_marginal.py` and `families/_mixture.py` (VI.8, VII.3) |
| `core/_empirical.py` | `distributions/_empirical.py` |
| `inference/_registry.py` (the registry object, today imported upward by `core/ops.py`) | `operations/_condition.py`; the methods stay in `inference/`, and the edge points downward |
| `core/named_tree.py`, `core/tracked.py`, `core/provenance.py`, `core/_registry.py` | `core/`, one module per II section; `Annotated` folds into `TrackedTerm` (II.4) |
| `core/_numeric_array.py`, `core/_opaque.py`, `core/record.py`, and their batch modules | `values/`, one module per III section |
| `core/event_template.py`, `core/constraints.py` | split in place: `core/_specs.py`, `core/_record_spec.py`, `core/_numeric.py`, `core/_constraints.py` (II.1–II.3, III.5) |
| `record/design.py` | `designs/`, generalized from `RecordBatch` to any element spec |
| `core/_record_distribution.py` | `distributions/_views.py` (`FieldView`, III.7); `RecordDistribution` is retired, since draw structure is declared in `event_spec` (III.7) |
| `core/_numeric_record_distribution.py` | the numeric marker to `distributions/_distribution.py` (III.7) and `BootstrapDistribution` to `families/_resampling.py` (VII.2); `FlatNumericRecordDistribution`, `FlattenedDistributionView`, and `NumericRecordDistributionView` are retired, the flat view being `evaluate(to_vector, d)` (III.7) |
| `modeling/_stan.py`, `modeling/_pymc.py` | `families/_programs.py` (VII.9), retaining separate data inputs and declared event variables |
| `modeling/_glm.py` | `families/_conditional.py` (VII.8) |
| `modeling/_base.py`, `modeling/_simple.py`, `modeling/_simple_generative.py`, and `Likelihood`, `ConditionallyIndependentLikelihood`, `GenerativeLikelihood` in `core/protocols.py` | retired: a model is a program-defined family (VII.9) or a factored joint (IV.1), and a learned likelihood is a `ConditionalDistribution` (III.9) |
| `modeling/_likelihood.py` (`IncrementalConditioner`) | retired as a class; a fold of `condition_on` over data batches, settled with `iterate` |
| `converters/_registry.py`, `converters/_protocol.py` | `distributions/_conversion.py` (IV.3): `ConversionMethod` becomes `Fidelity`, `Converter.convert` becomes `execute`, and the protocol resolver becomes protocol targets |
| `converters/_probpipe.py`, `converters/_scipy.py`, `converters/_tfp.py` | `families/_converters.py` (IV.3) |
| `expectation`'s `return_dist` and `set_return_approx_dist` (`core/ops.py`, `core/_distribution_base.py`) | retired: the error of a Monte Carlo estimate is taken explicitly through the bootstrap (VII.2); `set_default_num_evaluations` becomes the sample-count default in `core/_config.py` (V.2) |
| `core/_kinds.py`, `core/_array_backend.py` | `core/`, in place: the kind table (II.1) and the array-backend registry (II.3) |
| `core/_immutable.py`, `core/_fingerprint.py` | `core/_identity.py` (II.4) |
| `core/config.py` | `core/_config.py` |
| `core/_workflow_callable.py`, `core/_workflow_descendants.py` | `functions/_replay.py` for the callable anchors and `functions/_plan.py` for the root-ancestor capture (V.5, V.8) |
| `core/_workflow_errors.py` | `functions/`, each error beside the step that raises it (V.1) |
| `core/_random_functions.py`, `core/_random_measures.py` | `families/_random_functions.py` (VII.5) |
| `distributions/_product.py`, `distributions/_sequential_joint.py`, `distributions/_joint_utils.py`, `distributions/joint.py` | `distributions/_factored.py` (IV.1): `ProductDistribution` and `SequentialJointDistribution` become `FactoredDistribution` |
| `distributions/_joint_gaussian.py`, `distributions/gaussian_random_function.py` | `families/_gaussian.py` (VII.6): `JointGaussian` becomes `FactoredMultivariateGaussian` |
| `distributions/_joint_empirical.py` | `distributions/_empirical.py` (VII.2): an empirical joint is an `EmpiricalDistribution` over a record event, and its conditioning, if kept, is a capability route of `condition_on` |
| `distributions/_tfp_base.py` | `families/_backend.py` (VII.1) |
| `distributions/continuous.py`, `distributions/discrete.py`, `distributions/multivariate.py` | `families/_continuous.py`, `families/_discrete.py`, `families/_multivariate.py` (VII.1) |
| `distributions/kde.py` | `families/_resampling.py` (VII.2) |
| `distributions/transformed.py` | `families/_transformed.py` (VII.4) |
| `distributions/_bijector_dispatch.py` | `functions/_reparameterization.py` (V.12) |
| `linalg/linear_operator.py`, `linalg/operations.py`, `linalg/utils.py` | `linalg/_linop.py`, `linalg/_structured.py`, `linalg/_composites.py`; the free-function queries become `LinOp` methods (III.4) |
| `inference/_approximate_distribution.py`, `inference/_minibatch.py` | `inference/`, in place: `ApproximateDistribution` becomes an `EmpiricalDistribution` carrying provenance and annotations (VII.7), and `MinibatchedDistribution` is a `RandomMeasure` member (VII.5) |
| `core/transition.py` (`iterate`, `with_conversion`, `with_resampling`) | open, with the incremental-conditioning point below |
| `_weights.py`, `_array_utils.py`, `_dtype.py`, `_utils.py` | private helpers, unchanged |
| `diagnostics/`, `validation/` | unchanged |

### Open points

- *Incremental conditioning.* `IncrementalConditioner` (`modeling/_likelihood.py`) is a fold of `condition_on` over data batches; whether it is written as a derived operation or as a workflow recipe is settled together with `iterate`.
