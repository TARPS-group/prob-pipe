# API Reference

ProbPipe is organized around a small set of generic verbs ([Operations]) that
act on a few container abstractions ([Distributions], [Records], and
[Models]). This page is a map from "what are you trying to do" to the
section that documents it; if you already know the name of the symbol you
want, the search box (top right) is the fastest path.

## What are you trying to do?

| Goal | Look here |
|---|---|
| Sample, evaluate density, compute moments, condition | [Operations](operations.md) |
| Use a built-in distribution family | [Distributions](distributions.md) |
| Compose, transform, or stack distributions | [Distributions → Composite and joint](distributions.md#composite-and-joint) |
| Work with empirical samples or bootstrap replicates | [Distributions → Empirical and bootstrap](distributions.md#empirical-and-bootstrap) |
| Hold structured, immutable data | [Records and data](records.md) |
| Define a probabilistic model | [Modeling and inference → Models](inference.md#models) |
| Run inference (MCMC, VI, SBI) | [Modeling and inference → Inference methods](inference.md#inference-methods) |
| Check a posterior with predictive simulations | [Modeling and inference → Predictive checks](inference.md#predictive-checks) |
| Scale to Prefect / Ray / Dask | [Workflows and orchestration](workflows.md) |
| Constrain parameters / move to unconstrained space | [Constraints and bijectors](constraints.md) |
| Interop with TFP, scipy, xarray, or pandas | [Conversion and interop](converters.md) |
| Inspect provenance of computed quantities | [Provenance](provenance.md) |
| Write a new distribution, inference method, or converter | [Extending ProbPipe](extending.md) |

## Pages

- **[Operations](operations.md)** — the verbs: `sample`, `log_prob`, `prob`,
  `unnormalized_log_prob`, `unnormalized_prob`, `mean`, `variance`, `cov`,
  `expectation`, `condition_on`, `from_distribution`,
  `random_log_prob`, `random_unnormalized_log_prob`.
- **[Distributions](distributions.md)** — continuous, discrete, multivariate,
  composite (product / sequential / transformed / joint), empirical and
  bootstrap, random functions.
- **[Records and data](records.md)** — `Record`, `NumericRecord`, the
  `RecordArray` family, `Weights`, parameter-sweep `Design`s, and the
  auxiliary-metadata registry.
- **[Modeling and inference](inference.md)** — model / likelihood classes,
  the inference-method registry and built-in methods, iterative
  transformations, and predictive checks.
- **[Workflows and orchestration](workflows.md)** — `WorkflowFunction`,
  `Module`, the workflow decorators, and Prefect-orchestration configuration.
- **[Constraints and bijectors](constraints.md)** — `Constraint` singletons
  and factories, plus the `bijector_for` map used for reparameterization.
- **[Conversion and interop](converters.md)** — `converter_registry`,
  `Converter`, and the conversion-info dataclasses.
- **[Provenance](provenance.md)** — `Provenance`, `provenance_ancestors`,
  `provenance_dag`.
- **[Extending ProbPipe](extending.md)** — base classes, protocols, and the
  extension points for distributions, inference methods, converters,
  bijectors, and auxiliary metadata.
- **[Internals](internals.md)** — implementation details that may move
  between releases; useful when debugging a `isinstance(obj, SupportsX)`
  surprise.

[Operations]: operations.md
[Distributions]: distributions.md
[Records]: records.md
[Models]: inference.md
