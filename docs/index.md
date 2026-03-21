ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core organizing principle is **distributions in, distributions out**: every node in a pipeline can consume and emit probability distributions, enabling principled uncertainty propagation across the entire workflow.

## Why ProbPipe?

Scientific discovery and real-world decision-making increasingly depend on complex, end-to-end inferential pipelines that integrate heterogeneous data, fit probabilistic models, propagate uncertainty, and validate predictions. Yet high-quality uncertainty quantification (UQ) is rarely achieved at scale because such pipelines are difficult to construct in a way that is simultaneously flexible, reliable, and scalable.

ProbPipe provides general-purpose abstractions for probabilistic *workflows* -- making composability, scalability, and reproducibility the default. Its design is driven by five requirements:

- **Reusable inferential components.** Workflows are expressed in terms of modular, swappable statistical units rather than low-level orchestration primitives.
- **Interoperability with the Python ecosystem.** Modules can wrap existing ML and probabilistic libraries, and automatic conversion among distributional representations removes brittleness.
- **End-to-end uncertainty propagation.** When a workflow node expects a concrete value but receives a distribution, ProbPipe automatically broadcasts over samples -- users write deterministic functions and get UQ for free.
- **Seamless scalability.** The same pipeline scales computationally (JAX vectorization) and operationally (Prefect orchestration) without code changes.
- **Provenance and reproducibility.** Every distribution records how it was created, enabling full lineage tracing from any result back to its inputs.

## Quick Example

Suppose you want to estimate total revenue, but both the unit price and customer demand are uncertain. Write the business logic as a plain function, pass distributions for the uncertain inputs, and ProbPipe propagates the uncertainty automatically:

```python
from probpipe import Normal, Workflow

price = Normal(loc=50.0, scale=5.0, name="price")
demand = Normal(loc=1000.0, scale=100.0, name="demand")

def total_revenue(price, demand):
    return price * demand

wf = Workflow(func=total_revenue)
revenue = wf(price=price, demand=demand)

revenue.mean()       # ~50000 (mean revenue in dollars)
revenue.variance()   # captures joint uncertainty from price and demand
revenue.source       # Provenance('broadcast', parents=[price, demand])
```

The result is an `EmpiricalDistribution` -- a first-class distribution object that can be passed into downstream workflow nodes, triggering further uncertainty propagation.

## Next Steps

- [Getting Started](getting-started.md) -- installation and first steps
- [Tutorials](tutorials.md) -- guided notebooks covering distributions, transforms, joint models, and more
- [API Reference](api/distributions.md) -- full class and function documentation
