# API Reference

A task-to-section map for the public API. Use the search box (top right)
for known-name lookups.

## What are you trying to do?

| Goal | Look here |
|---|---|
| Sample, evaluate density, compute moments, condition | [Operations](operations.md) |
| Use a built-in distribution family | [Distributions](distributions/index.md) |
| Compose, transform, or stack distributions | [Distributions → Composite and joint](distributions/composite.md) |
| Work with empirical samples or bootstrap replicates | [Distributions → Empirical and bootstrap](distributions/empirical.md) |
| Hold structured, immutable data | [Records and data](records.md) |
| Define a probabilistic model | [Modeling and inference → Models](inference.md#models) |
| Run inference (MCMC, VI, SBI) | [Modeling and inference → Inference methods](inference.md#inference-methods) |
| Check a posterior with predictive simulations | [Modeling and inference → Predictive checks](inference.md#predictive-checks) |
| Scale to Prefect / Ray / Dask | [Workflows and orchestration](workflows.md) |
| Constrain parameters / move to unconstrained space | [Constraints and bijectors](constraints.md) |
| Interop with TFP, scipy, xarray, or pandas | [Conversion and interop](converters.md) |
| Inspect provenance of computed quantities | [Identity & provenance](provenance.md) |
| Write a new distribution, inference method, or converter | [Extending ProbPipe](extending.md) |

## Pages

- **[Operations](operations.md)** — `sample`, `log_prob`, `prob`,
  `unnormalized_log_prob`, `unnormalized_prob`, `random_log_prob`,
  `random_unnormalized_log_prob`, `mean`, `variance`, `cov`,
  `expectation`, `condition_on`, `from_distribution`.
- **[Distributions](distributions/index.md)** — continuous, discrete,
  multivariate, composite and joint, empirical and bootstrap, random
  functions.
- **[Records and data](records.md)** — `Record`, `NumericRecord`, the
  `RecordArray` family, `Weights`, parameter-sweep `Design`s, and the
  auxiliary-metadata registry.
- **[Modeling and inference](inference.md)** — model and likelihood
  classes, the inference-method registry and built-ins, iterative
  transformations, predictive checks.
- **[Workflows and orchestration](workflows.md)** — `WorkflowFunction`,
  `Module`, the workflow decorators, and Prefect-orchestration
  configuration.
- **[Constraints and bijectors](constraints.md)** — `Constraint`
  singletons and factories, the `bijector_for` map for reparameterization.
- **[Conversion and interop](converters.md)** — `converter_registry`,
  `Converter`, the conversion-info dataclasses.
- **[Identity & provenance](provenance.md)** — `Tracked`, `Annotated`, `Provenance`, `provenance_ancestors`,
  `provenance_dag`.
- **[Extending ProbPipe](extending.md)** — base classes, protocols, and
  extension contracts.
- **[Internals](internals.md)** — implementation details that may move
  between releases.
