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

The layout makes the reference's dependency order mechanical: what a part may depend on is what its package may import, so the document and the code cannot drift on layering. Upward registration is `D2 – Generality first` in the import graph, since the supported set grows by adding a provider package rather than by widening a lower layer. The single curated namespace serves `C3 – Computational detail hidden by default, available on demand`: module locations stay free to change, and a user's imports do not. Module boundaries on the implementation's existing divisions keep the reorganization concrete: each target module names work that is already one coherent unit.

### The tree

```
probpipe/
├── __init__.py                # the curated public API
├── core/                      # Part II — shared abstractions
│   ├── _named_tree.py         #   NamedTree (II.6)
│   ├── _constraints.py        #   Constraint and the constraint factories (II.3)
│   ├── _shapes.py             #   the reading of shape, level-name, and axis-count arguments (II.1, II.5)
│   ├── _spec_base.py          #   TermSpec and dimension unification (II.1), NumericSpec (II.3), NumericArraySpec, OpaqueSpec (III.1–III.2)
│   ├── _specs.py              #   InputSpec, OutputSpec and component projection contracts (II.2)
│   ├── _kinds.py              #   the kind table: register_kind, term_class_for_spec, batch_class_for_spec (II.1)
│   ├── _numeric.py            #   Numeric, the flat-vector interface of the numeric kinds (II.3)
│   ├── _array_backend.py      #   the array-backend registry for native numeric leaves (II.3)
│   ├── _record_spec.py        #   RecordSpec, NumericRecordSpec, unification (III.5)
│   ├── _identity.py           #   TrackedTerm with annotations on the base, Immutable, Provenance, fingerprints, the provenance traversal (II.4)
│   ├── _batch.py              #   Batch, BatchSpec: axis groups, level names, at_levels (II.5)
│   ├── _dispatch.py           #   dispatch methods and registries, Feasibility, MethodInfo, ResolutionError, MathematicalDomainError (II.7)
│   ├── _catalog.py            #   EntrySummary, RegistryCatalog (II.7)
│   └── _config.py             #   library configuration
├── values/                    # the value layer (III.1–III.6; LinOp, III.4, is in linalg/)
│   ├── _numeric_array.py      #   NumericArray (III.1)
│   ├── _numeric_array_batch.py  #   NumericArrayBatch (III.1)
│   ├── _opaque.py             #   Opaque (III.2)
│   ├── _function_base.py      #   Function itself (declared sides, identity, controls and with_options,
│   │                          #     plain evaluation, install_call_engine), FunctionSpec, the function
│   │                          #     capabilities, and is_differentiable
│   ├── _binding.py            #   the frozen signature and the argument binding apply performs
│   │                          #     without the engine (III.3, V.3)
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
├── distributions/             # the distribution layer (III.7–IV.4)
│   ├── _distribution.py       #   Distribution, NumericDistribution, DistributionSpec (III.7)
│   ├── _views.py              #   FieldView (III.7–III.8)
│   ├── _capabilities.py       #   the Supports* protocols (III.8)
│   ├── _conditional.py        #   ConditionalDistribution, its markers and spec (III.9), conditional_distribution (IV.4)
│   ├── _from_functions.py     #   distribution: a law from a sampler, a density, or both (IV.4)
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
│   ├── _resolution.py         #   the selection among a Function's own routes, as an operation's (V.7, VI.0)
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
│   ├── _density.py            #   log_prob, unnormalized_log_prob, prob, unnormalized_prob,
│   │                          #     random_log_prob, random_unnormalized_log_prob (VI.4)
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
│   ├── _programs.py           #   StanModel, PyMCModel: program-defined laws (VII.9)
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

Every module with a design contract that is not yet where the tree places it, with where it goes; the target contracts above are authoritative. A module at the path the tree gives it is in place and has no row, as every module of `distributions/`, `families/`, and `operations/` is.

The spec implementation in `core/` is divided into three files: `core/_spec_base.py` defines `TermSpec`, `NumericSpec`, `NumericArraySpec`, `OpaqueSpec`, and shared dimension unification; `core/_record_spec.py` defines `RecordSpec` and `NumericRecordSpec`; and `core/_specs.py` defines `InputSpec` and `OutputSpec` and re-exports the public spec types of `core/`. `FunctionSpec` is defined beside `Function` in `values/_function_base.py`, and `DistributionSpec` beside `Distribution` in `distributions/_distribution.py`.

| Today | Target |
|---|---|
| `core/named_tree.py` | `core/_named_tree.py` (II.6) |
| `core/constraints.py` | `core/_constraints.py` (II.3) |
| `core/tracked.py`, `core/provenance.py`, `core/_immutable.py`, `core/_fingerprint.py`, `core/_repr.py` | `core/_identity.py` (II.4); `Annotated` folds into `TrackedTerm` |
| `core/config.py` | `core/_config.py` |
| `core/_numeric_array.py`, `core/_numeric_array_batch.py` | `values/_numeric_array.py`, `values/_numeric_array_batch.py` (III.1) |
| `core/_opaque.py`, `core/_opaque_batch.py`, `core/_object_batch.py` | `values/_opaque.py`, `values/_opaque_batch.py`, `values/_object_batch.py` (III.2) |
| `core/_function_batch.py` | `values/_function_batch.py` (III.3) |
| `core/record.py`, `core/_numeric_record.py` | `values/_record.py` (III.5) |
| `core/_record_batch.py`, `core/_numeric_record_batch.py` | `values/_record_batch.py`, `values/_numeric_record_batch.py` (III.6) |
| `core/protocols.py` (`SupportsArrayBackend`) | `distributions/_capabilities.py` |
| `core/node.py` (`Node`, `InputFrozenError`), `functions/_module.py` (`Module`, `AbstractModule`, `workflow_method`, `abstract_workflow_method`, `Module.dag()`) | experimental; placement to be decided |
| `core/transition.py` (`iterate`, `with_conversion`, `with_resampling`) | `inference/`: sequential updating, which folds a step such as `condition_on` over the data |
| `functions/_normalization.py` | `functions/_call.py`; conversion executes later under the IV.3 plan (V.4) |
| `functions/_contract.py`, `functions/_descendants.py` | `functions/_plan.py`: the per-call binding of the declared inputs and the root-ancestor capture (V.5, V.6) |
| `functions/_callable.py`, `functions/_recipe.py` | `functions/_replay.py`: the callable anchors and the recorded recipes (V.8) |
| `functions/_managed.py` | `functions/_broker.py` (V.8, V.9) |
| `functions/_execution_contract.py` | `functions/_execution.py` (V.9) |
| `functions/_errors.py` | `functions/`, each error beside the step that raises it (V.1) |
| `linalg/linear_operator.py`, `linalg/operations.py`, `linalg/utils.py` | `linalg/_linop.py`, `linalg/_structured.py`, `linalg/_composites.py`; the free-function queries become `LinOp` methods (III.4) |
| `record/design.py` | `designs/`, generalized from `RecordBatch` to any element spec |
| `inference/_minibatch.py` | `inference/`, in place: `MinibatchedDistribution` is a `RandomMeasure` member (VII.5) |
| `_weights.py`, `_array_utils.py`, `_dtype.py`, `_messages.py`, `custom_types.py` | private helpers, unchanged |
| `diagnostics/`, `validation/` | in place; a predictive check takes the kernel of the observations and a law over its given slots, and reads its replications from their composition, as `mixture` does (VI.9) |
