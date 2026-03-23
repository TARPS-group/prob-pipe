# Getting Started

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

Core dependencies: JAX and TensorFlow Probability (nightly).

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
from probpipe import WorkflowFunction

initial_dose = 100.0  # mg (known)
t = 8.0               # hours (known)

def concentration(half_life):
    return initial_dose * (0.5 ** (t / half_life))

wf = WorkflowFunction(func=concentration)
conc = wf(half_life=half_life)

conc.mean()                   # ~25.0 (expected concentration after 8h)
conc.source                   # Provenance('broadcast', parents=[half_life])
```

### 4. Compute expectations with error tracking

The `expectation(f)` method computes `E[f(X)]`. On empirical distributions (like broadcast results), this is exact. On parametric distributions, it uses Monte Carlo and returns a `BootstrapDistribution` that tracks the sampling error:

```python
# Exact expectation on the broadcast result (EmpiricalDistribution)
conc.expectation(lambda x: x)     # Array(~25.0) -- exact weighted sum

# MC expectation on the parametric prior, with error tracking
ex = half_life.expectation(lambda x: 0.5 ** (8.0 / x), num_evaluations=5000)
ex.mean()       # point estimate of E[conc(half_life)]
ex.variance()   # MC error variance -- decreases with more evaluations
```

### 5. Track provenance

Every distribution records its lineage automatically. You can traverse the provenance chain or serialize it to JSON:

```python
from probpipe import provenance_ancestors

provenance_ancestors(conc)    # [Normal(name='half_life', event_shape=())]
conc.source.to_dict()         # {'operation': 'broadcast',
                              #  'parents': [{'type': 'Normal', 'name': 'half_life'}],
                              #  'metadata': {'vectorize': 'jax', ...}}
```

### 6. Interoperate with TFP and scipy

The converter registry handles automatic bidirectional conversion between ProbPipe, raw TFP, and scipy.stats distributions. Workflows accept any recognized distribution type as input:

```python
from probpipe import converter_registry
import tensorflow_probability.substrates.jax.distributions as tfd
import scipy.stats as ss

# Raw TFP and scipy distributions work directly in workflows
conc_from_tfp = wf(half_life=tfd.Normal(loc=4.0, scale=0.5))
conc_from_scipy = wf(half_life=ss.norm(loc=4.0, scale=0.5))

# Inspect a conversion before performing it
info = converter_registry.check(tfd.Normal(loc=4.0, scale=0.5), Normal)
info.method   # ConversionMethod.EXACT
info.cost     # 0.01

# Convert ProbPipe distributions to scipy for use with scipy tools
from scipy.stats._distn_infrastructure import rv_frozen
sp = converter_registry.convert(half_life, rv_frozen)
sp.ppf(0.975)  # 95th percentile via scipy
```

### 7. Differentiate through distributions

Since everything is built on JAX, you can compute gradients of distribution operations -- useful for sensitivity analysis, MLE, and variational inference:

```python
import jax

# Score function: d/d(x) log p(x)
score = jax.grad(half_life.log_prob)(4.0)   # 0.0 (zero at the mean)
score = jax.grad(half_life.log_prob)(3.0)   # 4.0 (positive below the mean)
```

## Next Steps

Explore the [tutorials](tutorials.md) for in-depth coverage of distributions, joint models, conditioning, automatic differentiation, and modular inference pipelines.
