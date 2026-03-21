# Getting Started

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

Core dependencies: JAX, NumPy, SciPy, TensorFlow Probability (nightly).

### Optional extras

```bash
pip install .[dev]       # pytest, jupyter, matplotlib, graphviz, docs
pip install .[prefect]   # Prefect orchestration backend
pip install .[docs]      # MkDocs documentation tools
```

## Your First Pipeline

### 1. Create a distribution

ProbPipe wraps 20+ TensorFlow Probability distributions with a uniform interface:

```python
from probpipe import Normal
import jax

prior = Normal(loc=0.0, scale=1.0)
samples = prior.sample(jax.random.PRNGKey(0), (1000,))
prior.log_prob(0.5)  # -1.0439...
```

### 2. Transform it

Apply bijectors to create transformed distributions:

```python
from probpipe import TransformedDistribution
import tensorflow_probability.substrates.jax.bijectors as tfb

positive_prior = TransformedDistribution(prior, tfb.Exp())
positive_prior.support  # positive
```

### 3. Propagate uncertainty through a workflow

When a workflow node receives a distribution where it expects a scalar, ProbPipe automatically broadcasts:

```python
from probpipe import Workflow

def simulate(mu: float, sigma: float) -> float:
    return mu + sigma * 0.1

wf = Workflow(func=simulate)
result = wf(mu=Normal(loc=0.0, scale=1.0), sigma=Normal(loc=1.0, scale=0.1))
# result is an EmpiricalDistribution
result.mean()  # ~0.1
```

### 4. Track provenance

Every distribution records its lineage:

```python
from probpipe import provenance_ancestors

result.source  # Provenance('broadcast', parents=[Normal, Normal], ...)
provenance_ancestors(result)  # [Normal(name='mu'), Normal(name='sigma')]
```

## Next Steps

Explore the [tutorials](examples/01_distributions.ipynb) for in-depth coverage of distributions, joint models, conditioning, automatic differentiation, and modular inference pipelines.
