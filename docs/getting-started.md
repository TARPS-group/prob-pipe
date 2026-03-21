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

We'll build a simple pharmacokinetic model: given an uncertain drug half-life, predict the concentration remaining after a fixed time.

### 1. Create a distribution

ProbPipe wraps 20+ TensorFlow Probability distributions with a uniform interface. All distributions follow TFP shape semantics and are fully JAX-differentiable:

```python
from probpipe import Normal
import jax

half_life = Normal(loc=4.0, scale=0.5, name="half_life")

half_life.mean()              # 4.0
half_life.log_prob(4.0)       # -0.2258 (log-density at the mean)

samples = half_life.sample(jax.random.PRNGKey(0), (1000,))
samples.shape                 # (1000,)
```

### 2. Transform it

Apply bijectors to create transformed distributions. For example, enforce positivity with an exponential transform:

```python
from probpipe import TransformedDistribution
import tensorflow_probability.substrates.jax.bijectors as tfb

pos_half_life = TransformedDistribution(
    Normal(loc=1.5, scale=0.3), tfb.Exp(), name="pos_half_life"
)
pos_half_life.support         # positive
```

### 3. Propagate uncertainty through a workflow

Write the science as a plain function. When ProbPipe sees a distribution where a scalar is expected, it automatically draws samples and returns an `EmpiricalDistribution` over the outputs:

```python
from probpipe import Workflow

initial_dose = 100.0  # mg (known)
t = 8.0               # hours (known)

def concentration(half_life):
    return initial_dose * (0.5 ** (t / half_life))

wf = Workflow(func=concentration)
conc = wf(half_life=half_life)

conc.mean()                   # ~25.0 (expected concentration after 8h)
conc.source                   # Provenance('broadcast', parents=[half_life])
```

### 4. Track provenance

Every distribution records its lineage automatically. You can traverse the provenance chain or serialize it to JSON:

```python
from probpipe import provenance_ancestors

provenance_ancestors(conc)    # [Normal(name='half_life', event_shape=())]
conc.source.to_dict()         # {'operation': 'broadcast',
                              #  'parents': [{'type': 'Normal', 'name': 'half_life'}],
                              #  'metadata': {'vectorize': 'jax', ...}}
```

### 5. Differentiate through distributions

Since everything is built on JAX, you can compute gradients of distribution operations -- useful for sensitivity analysis, MLE, and variational inference:

```python
import jax

# Score function: d/d(x) log p(x)
score = jax.grad(half_life.log_prob)(4.0)   # 0.0 (zero at the mean)
score = jax.grad(half_life.log_prob)(3.0)   # 4.0 (positive below the mean)
```

## Next Steps

Explore the [tutorials](examples/01_distributions/) for in-depth coverage of distributions, joint models, conditioning, automatic differentiation, and modular inference pipelines.
