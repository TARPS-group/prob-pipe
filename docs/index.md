ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core design principle is **simplification via abstraction** — making probabilistic inference feel more mathematical via a functional, registry-based system built around just three main types you need to know about.

## Why ProbPipe?

Most workflows for probabilistic inference can be described in terms of **distributions**, **fixed values** (data, hyperparameters, covariates), and **operations** that transform distributions. But implementing these workflows is harder than describing them because math has to be translated into computation:

- **Algorithmic challenges.** There are many possible algorithms for common operations, with varying trade-offs that need to be explored in a problem-specific manner. A posterior could be approximated using a variety of MCMC algorithms, variational inference methods, or sequential Monte Carlo, or might require more specialized methods such as those for amortized and simulation-based inference. 
- **Representational challenges.** Algorithms expect (and produce) specific formats for distributions and fixed values, and those formats are not always compatible with other parts of the workflow. Fixed values may be named parameter vectors, covariate matrices, or structured observations, and different algorithms expect different representations.

In practice, these issues make it hard to explore the full design space of available methods or to build more complex workflows that many algorithms for different steps. ProbPipe addresses these challenges through a single design principle: **simplification via abstraction**. There are just three core types:

1. **`Distribution`**: the universal representation of random quantities (priors, posteriors, data-generating processes). A distribution's capabilities are declared via protocols (`SupportsSampling`, `SupportsLogProb`, ...), and ProbPipe converts between representations as needed.
2. **`Record`**: the universal container for non-random structured data (observed datasets, hyperparameters, design matrices). `Record` is the deterministic counterpart of `Distribution`.
3. **`WorkflowFunction`**: Usually construction by decorating a function with  `@workflow_function`. Pass the declared types of values, the workflow function runs normally. But pass a `Distribution` where a concrete value is expected, and ProbPipe propagates uncertainty automatically, returning a `Distribution` over the functions declared result type. Similarly, array-valued inputs (a `RecordArray`) broadcast across fixed values (e.g., for hyperparameter sweeps). To ensure composability and modularity, all returned values from a workflow function are wrapped as an appropriate `Record` / `Distribution`. 

`Distribution` and `Record` share a single interface for named-field access (`fields`, `select(...)`, `select_all()`) and passing components into a `WorkflowFunction`, so they are interchangeable as arguments to workflow function. 

## Built-in operations

ProbPipe provides a set of built-in **ops**, which are workflow functions that can support specalized features to streamline pipeline construction:

- **`condition_on`**: condition a model on observed data, automatically selecting the best inference algorithm (or specify one with `method=`).
- **`mean`**, **`variance`**, **`cov`**, **`expectation`**: compute distributional summaries, with automatic Monte Carlo fallback when exact computation is unavailable.
- **`sample`**, **`log_prob`**: draw samples or evaluate densities through a uniform interface.
- **`from_distribution`**: convert between distribution representations via a customizable converter registry.
- **`predictive_check`**: built-in prior and posterior predictive model checking.

## Next Steps

- [Getting Started tutorial](tutorials/getting_started.ipynb) -- iterative Bayesian model building following the Bayesian Workflow (Gelman et al., 2020)
- [User Guide](user_guide.md) -- in-depth coverage of specific ProbPipe features
- [API Reference](api/index.md) -- full class and function documentation
