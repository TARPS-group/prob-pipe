# Package Structure

The package layout realizes the design reference as an import architecture: one package per layer of the reference, imports pointing strictly downward, and registries carrying capability upward. Like the rest of the reference, it describes the target state.

This document uses the **`Function`** vocabulary: the universal function representation, a tracked term carrying an input and an output template, of which `LinOp` is the linear subtype and invertibility is a capability. A `FunctionSpec` admits a `Function` and a `FunctionBatch` holds them. Its detailed contract is unsettled (see Open points); this document fixes only its place in the layering.

### Principles

- **One package per layer.** Packages mirror the reference's parts in dependency order, and a module realizes one section or one coherent piece of one.
- **Imports point downward.** Each package imports only from packages above it in the tree below. There are no import cycles, and no lazy imports to dodge one.
- **Registration flows upward.** A lower layer defines a registry and higher layers populate it at import time: the families register pushforward rules and converters, and the inference methods register themselves, so capability reaches the operations without the operations importing their providers.
- **A spec lives with the type it admits.** `ArraySpec` and `OpaqueSpec` live in the value layer, `FunctionSpec` with `Function`, and `DistributionSpec` and `ConditionalDistributionSpec` with the distribution classes. The same placement rule covers each type's batch form.
- **Modules are private, packages are public.** Every module is underscore-prefixed. A package's `__init__` exports its public names, and the top-level `probpipe` namespace re-exports the curated user surface, which is the only import a user needs.
- **Tests mirror the tree**, as `tests/<package>/test_<module>.py`.

### Rationale

The layout makes the reference's dependency order mechanical: what a part may depend on is what its package may import, so the document and the code cannot drift on layering. Upward registration is `D2 – Generality first` in the import graph, since the supported set grows by adding a provider package rather than by widening a lower layer. The single curated namespace serves `C3 – Computational detail hidden by default, available on demand`: module paths stay free to change, and a user's imports do not.

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
│   ├── _record.py             #   Record, NumericRecord (III.3)
│   ├── _record_batch.py       #   RecordBatch, NumericRecordBatch (III.4)
│   └── _opaque_batch.py       #   OpaqueBatch (III.2)
├── functions/                 # the Function representation
│   ├── _function.py           #   Function, FunctionSpec, the wrapping decorator
│   ├── _capabilities.py       #   function capabilities: invertibility with log-det-Jacobian, …
│   ├── _batch.py              #   FunctionBatch (III.2)
│   └── _reparameterization.py #   Constraint → invertible Function (III.14)
├── linalg/                    # LinOp, the linear Function subtype (III.5)
│   ├── _linop.py              #   LinOp: the action, the queries, flags
│   ├── _structured.py         #   Dense / Diagonal / Triangular / Cholesky / Root …
│   ├── _composites.py         #   Product / Sum / Scaled / Transpose — the operator algebra
│   └── _batch.py              #   LinOpBatch
├── distributions/             # the distribution layer (III.6–III.13)
│   ├── _distribution.py       #   Distribution, NumericDistribution, DistributionSpec (III.6)
│   ├── _views.py              #   FieldView (III.6–III.7)
│   ├── _capabilities.py       #   the Supports* protocols (III.7)
│   ├── _conditional.py        #   ConditionalDistribution, its markers and spec (III.8)
│   ├── _batches.py            #   DistributionBatch, ConditionalDistributionBatch (III.9)
│   ├── _factored.py           #   SupportsFactors and the factored classes (III.10)
│   ├── _composition.py        #   the * engine behind __mul__ (III.11)
│   └── _conversion.py         #   Converter, ConverterRegistry (III.13)
├── operations/                # Parts IV–V — lifting and the operations
│   ├── _lifting.py            #   the sampling lift: grouping, broadcast, include_inputs (IV.2)
│   ├── _controls.py           #   controls, with_options, the structural key split (IV.3)
│   ├── _execution.py          #   jax / sequential / thread dispatch (IV.4)
│   ├── _orchestration.py      #   optional tracing (IV.4)
│   ├── _moments.py            #   mean, variance, cov, quantile, expectation (V.1)
│   ├── _sample.py             #   sample (V.2)
│   ├── _density.py            #   log_prob, unnormalized_log_prob (V.3)
│   ├── _condition.py          #   condition_on, predictive, the inference registry (V.4)
│   ├── _joint.py              #   joint (V.5)
│   ├── _pushforward.py        #   pushforward and its rule registry (V.6)
│   └── _marginal.py           #   marginal, factor (V.7)
├── families/                  # Part VI — the distribution catalog
│   ├── _backend.py            #   TFPDistribution, the backend adapter (VI.1)
│   ├── _continuous.py         #   Normal, Gamma, … (VI.1)
│   ├── _discrete.py           #   Bernoulli, Poisson, … (VI.1)
│   ├── _multivariate.py       #   MultivariateNormal, Dirichlet, … (VI.1)
│   ├── _empirical.py          #   empirical, bootstrap, KDE and its kernels (VI.2)
│   ├── _mixture.py            #   MixtureDistribution (VI.3)
│   ├── _transformed.py        #   the pushforward-result families (VI.4)
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
- **`values/`** is the value layer of Part III. Its specs are `ArraySpec` and `OpaqueSpec` only, by the placement rule above.
- **`functions/`** holds the `Function` **representation**: the class, its spec and batch form, its capabilities, and the constraint reparameterization, which produces invertible functions from constraints. It sits below `distributions/` because distributions consume function-typed objects: `_cov` returns a `LinOp`, a function-valued event holds `Function`s, and the reparameterization serves inference. The **lifting machinery is not here**: it needs distributions and `sample`, so it is realized in `operations/`.
- **`linalg/`** is the linear subtype and its operator algebra, kept as its own package because the structured subclasses and composites are a coherent domain of their own.
- **`distributions/`** is the distribution layer of Part III, through composition and conversion.
- **`operations/`** realizes Parts IV and V together: the lift, controls, and execution dispatch are the behavioral half of `Function`, and the operations are themselves functions built on that machinery. The inference-method registry is defined here and populated from `inference/`.
- **`families/`** implements the catalog: constructors and capability implementations, registering its pushforward rules and converters upward at import.
- **`inference/`**, **`diagnostics/`**, and **`validation/`** sit outside the reference's parts: inference methods register into the V.4 registry, and diagnostics and validation are application layers over the public operations.

A handful of private helper modules (dtypes, array utilities) support the packages and carry no design contract.

### Open points

- *The `Function` contract.* The decorator, the call behavior, and how the lifting engine attaches to `Function` are fixed by the forthcoming `Function` design. This document constrains only the placement: the representation in `functions/`, the lifting in `operations/`.
- *Model-construction helpers.* The GLM assembly lands in the catalog; whether the remaining model-building conveniences earn a package is settled by the catalog consolidation.
