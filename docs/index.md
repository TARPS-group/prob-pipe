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

```python
from probpipe import Normal, Workflow

def simulate(mu: float, sigma: float) -> float:
    return mu + sigma * 0.1

wf = Workflow(func=simulate)
result = wf(mu=Normal(loc=0.0, scale=1.0), sigma=Normal(loc=1.0, scale=0.1))
# result is an EmpiricalDistribution with 128 samples
result.mean()  # ~0.1
```

## Next Steps

- [Getting Started](getting-started.md) -- installation and first steps
- [Tutorials](examples/01_distributions/) -- guided notebooks covering distributions, transforms, joint models, and more
- [API Reference](api/distributions.md) -- full class and function documentation
