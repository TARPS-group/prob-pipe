# ProbPipe

[![CI](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml/badge.svg)](https://github.com/TARPS-group/prob-pipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TARPS-group/prob-pipe/branch/main/graph/badge.svg)](https://codecov.io/gh/TARPS-group/prob-pipe)
[![docs](https://img.shields.io/badge/docs-tarps--group.github.io%2Fprob--pipe-blue)](https://tarps-group.github.io/prob-pipe/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20683559-blue)](https://doi.org/10.5281/zenodo.20683559)

**[Installation Guide](#installation)** | **[Getting Started](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)** | **[Full Documentation](https://tarps-group.github.io/prob-pipe/)** | **[Help](https://tarps-group.github.io/prob-pipe/help/)**

ProbPipe is a Python framework for building scalable probabilistic pipelines with automated uncertainty quantification.

### Why ProbPipe?

Most workflows for probabilistic inference can be described in terms of **distributions**, **fixed values** (data, hyperparameters, covariates), and **operations** that transform distributions. But implementing these workflows is harder than describing them because math has to be translated into computation:

- **Algorithmic challenges.** There are many possible algorithms for common operations, with varying trade-offs that need to be explored in a problem-specific manner. A posterior could be approximated using a variety of MCMC algorithms, variational inference methods, or sequential Monte Carlo, or might require more specialized methods such as those for amortized and simulation-based inference.
- **Representational challenges.** Algorithms expect (and produce) specific formats for distributions and fixed values, and those formats are not always compatible with other parts of the workflow. Fixed values may be named parameter vectors, covariate matrices, or structured observations, and different algorithms expect different representations.

In practice, these issues make it hard to explore the full design space of available methods or to build more complex workflows that combine many algorithms for different steps. ProbPipe addresses these challenges through a single design principle: **simplification via abstraction**. There are just three core types:

1. **`Distribution`**: the universal representation of random quantities (priors, posteriors, data-generating processes). A distribution's capabilities are declared via protocols (`SupportsSampling`, `SupportsLogProb`, ...), and ProbPipe converts between representations as needed.
2. **`Record`**: the universal container for non-random structured data (observed datasets, hyperparameters, design matrices). `Record` is the deterministic counterpart of `Distribution`.
3. **`WorkflowFunction`**: Usually constructed by decorating a function with `@workflow_function`. Pass the declared types of values and the workflow function runs normally. But pass a `Distribution` where a concrete value is expected, and ProbPipe propagates uncertainty automatically, returning a `Distribution` over the function's declared result type. Similarly, array-valued inputs (a `RecordArray`) broadcast across fixed values (e.g., for hyperparameter sweeps). To ensure composability and modularity, all returned values from a workflow function are wrapped as an appropriate `Record` / `Distribution`.

`Distribution` and `Record` share a single interface for named-field access (`fields`, `select(...)`, `select_all()`) and passing components into a `WorkflowFunction`, so they are interchangeable as arguments to workflow functions.

### Built-in operations

ProbPipe provides a set of built-in **ops**, which are workflow functions that can support specialized features to streamline pipeline construction:

- **`condition_on`**: condition a model on observed data, automatically selecting the best inference algorithm (or specify one with `method=`).
- **`mean`**, **`variance`**, **`cov`**, **`expectation`**: compute distributional summaries, with automatic Monte Carlo fallback when exact computation is unavailable.
- **`sample`**, **`log_prob`**: draw samples or evaluate densities through a uniform interface.
- **`from_distribution`**: convert between distribution representations via a customizable converter registry.
- **`predictive_check`**: built-in prior and posterior predictive model checking.

## Quick Example

Logistic regression on the [Challenger O-ring data](https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster) (Dalal, Fowlkes & Hoadley, 1989): 23 launches, each with a launch temperature and a binary O-ring damage indicator. Fit a Bayesian logistic regression, then propagate posterior uncertainty through a predicted damage probability at the Challenger launch temperature (31°F) and a typical launch temperature (65°F).

```python
import jax, jax.numpy as jnp
import pandas as pd
import tensorflow_probability.substrates.jax.glm as tfp_glm
from probpipe import (
    Normal, ProductDistribution, GLMLikelihood, SimpleModel,
    workflow_function, condition_on,
)

# --- Load data ---
df = pd.read_csv("docs/tutorials/data/challenger.csv")
temperature = jnp.asarray(df["temperature"].values, dtype=jnp.float32)
damage = jnp.asarray(df["damage"].values, dtype=jnp.float32)

# --- 1. Build a model with named parameters ---
likelihood = GLMLikelihood(tfp_glm.Bernoulli(), temperature)
prior = ProductDistribution(
    intercept=Normal(loc=0.0, scale=10.0, name="intercept"),
    slope=Normal(loc=0.0, scale=1.0, name="slope"),
)
model = SimpleModel(prior, likelihood)

# --- 2. Condition on data (auto-selects NUTS) ---
posterior = condition_on(model, damage, random_seed=0)
draws = posterior.draws()            # NumericRecordArray(intercept=..., slope=...)
float(draws["intercept"].mean())     #  ~12.8 — high baseline log-odds…
float(draws["slope"].mean())         #  ~-0.2 — …attenuated by temperature

# --- 3. Propagate posterior uncertainty through a prediction ---
@workflow_function
def predict_prob(intercept, slope, x):
    return jax.nn.sigmoid(intercept + slope * x)

predictive = predict_prob(**posterior.select('intercept', 'slope'),
                          x=jnp.array([31.0, 65.0]))
# predictive is a Distribution over (P(damage|T=31°F), P(damage|T=65°F))
# posterior mean: ~0.98 at 31°F, ~0.46 at 65°F
```

Since `predict_prob` is constructed to be a workflow function, ProbPipe samples from the posterior and evaluates the function for each draw, yielding a Monte Carlo approximation to the full predictive distribution. The two posterior fields are extracted from a single parent, so their values are sampled jointly, ensuring each `(intercept, slope)` pair stays correlated. From there, we can easily plot the full predictive curve across temperatures:

```python
import numpy as np, matplotlib.pyplot as plt

t_grid = jnp.linspace(30, 85, 50)  # grid of temperature values
# predictive distribution at each temperature
curves = predict_prob(**posterior.select('intercept', 'slope'), x=t_grid)
# curves is a Distribution, so extract samples for plotting
S = np.array(curves.samples)         # (n_draws, 50)
# compute 90% credible intervals, then plot
lo, hi = np.percentile(S, [5, 95], axis=0)
plt.fill_between(np.array(t_grid), lo, hi, alpha=0.3, label='90% credible interval')
plt.plot(np.array(t_grid), S.mean(axis=0), lw=2, label='Posterior mean')
plt.scatter(np.array(temperature), np.array(damage), s=20, label='Data')
plt.axvline(31, ls='--', color='red', label='Challenger launch (31°F)')
plt.xlabel('Temperature (°F)'); plt.ylabel('P(O-ring damage)'); plt.legend()
```

![Posterior predictive](docs/assets/images/readme_logistic.png)

## Key Features

- **Protocol-based dispatch.** A distribution's capabilities are declared via `@runtime_checkable` protocols (`SupportsSampling`, `SupportsLogProb`, `SupportsMean`, ...). Operations like `condition_on` and `from_distribution` use these protocols to auto-select the best algorithm from a pluggable registry. Override with `method=` when you want control.
- **Multiple backends.** The inference registry spans BlackJAX (NUTS by default; HMC, SGLD, SGHMC opt-in), nutpie / CmdStan / PyMC NUTS for `StanModel` / `PyMCModel` targets, gradient-free RWMH, TFP NUTS/HMC as opt-in alternates, and PyMC ADVI. Swap backends without changing model code.
- **Automatic distribution conversion.** A converter registry converts between distribution representations (e.g., MCMC samples to KDE) as needed, using protocol-based dispatch analogous to `condition_on`.
- **JAX-native.** Distributions and workflow functions are compatible with JAX (`vmap`, `jit`, `grad`), with built-in support for TFP distributions.
- **Provenance tracking.** Each distribution records how it was created (algorithm, parents, metadata), enabling full lineage tracing from any result back to its inputs.
- **Prefect orchestration.** Distribute pipeline steps across machines and CPUs without code changes.

## Installation

Requires Python >= 3.12 (tested on 3.12, 3.13, and 3.14). ProbPipe is not yet
on PyPI, so installation is from source.

### New to Python?

You do **not** need an existing Python installation. [uv](https://docs.astral.sh/uv/)
manages both Python and the environment for you. First
[install uv](https://docs.astral.sh/uv/getting-started/installation/) (on
macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
uv venv                 # create an isolated environment (uv fetches a compatible Python)
uv pip install .        # install ProbPipe into it
source .venv/bin/activate   # activate it; now `python` and `probpipe` are available
```

Then work through the
**[Getting Started tutorial](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)**.
(Prefer not to activate? Prefix commands with `uv run`, e.g. `uv run python`.)

### Already have a Python environment?

Install from source with pip into your active environment:

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

uv users can substitute `uv pip install .` (into an active `uv venv`), or
`uv sync` for a lockfile-managed dev environment — see
[CONTRIBUTING.md](CONTRIBUTING.md#installation).

### Dependencies and optional extras

Core dependencies: JAX, [BlackJAX](https://blackjax-devs.github.io/blackjax/), and TensorFlow Probability. JAX provides arrays and autodiff; BlackJAX is the default backend for gradient MCMC (`condition_on(model, data)` auto-dispatches to `blackjax_nuts`); TFP supplies the distribution implementations the wrappers in `probpipe.Normal`, `probpipe.Gamma`, etc. delegate to. ProbPipe uses [tfp-nightly](https://pypi.org/project/tfp-nightly/), which is the [recommended approach](https://github.com/tensorflow/probability/issues/1994#issuecomment-3129033043) for TFP on JAX since stable TFP releases are tied to TensorFlow and often lag behind JAX.

Optional extras (append to the `pip install .` / `uv pip install .` above, e.g. `pip install ".[dev]"`):

```bash
pip install ".[dev]"        # pytest, jupyter, matplotlib, graphviz
pip install ".[prefect]"    # Prefect orchestration backend
pip install ".[stan]"       # Stan models via BridgeStan + CmdStanPy
pip install ".[pymc]"       # PyMC model integration
pip install ".[nutpie]"     # nutpie Markov chain Monte Carlo (MCMC) sampler
pip install ".[bayesflow]"  # BayesFlow amortized simulation-based inference (Python 3.12-3.13)
```

### Ray via Prefect

ProbPipe can dispatch Prefect-orchestrated `WorkflowFunction` tasks to Ray via
Prefect-Ray:

```bash
pip install "probpipe[prefect]"
pip install "prefect[ray]"
```

The local demo uses a persistent Ray head:

```bash
prefect server start
ray start --head
python example_scripts/run_ray_demo.py
```

See the [Ray via Prefect guide](https://tarps-group.github.io/prob-pipe/orchestration/ray/)
for setup details, deployment notes, and current support boundaries.

## Next Steps

- For a more detailed overview of ProbPipe, see the **[Getting Started Tutorial](https://tarps-group.github.io/prob-pipe/tutorials/getting_started/)**
- For all the details of the ProbPipe API, see the **[Reference Documentation](https://tarps-group.github.io/prob-pipe/)**
- For getting started as a ProbPipe developer, see the **[Contributing Guide](CONTRIBUTING.md)**.

## Citing ProbPipe

If you use ProbPipe in your research, please cite it:

```bibtex
@software{probpipe,
  author  = {Huggins, Jonathan and Roberts, Andrew and Lim, Yongho and
             Erozer, Can and Zhu, Jiaqiang},
  title   = {{ProbPipe}: Probabilistic pipelines with automated uncertainty quantification},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.20683559},
  url     = {https://github.com/TARPS-group/prob-pipe}
}
```

See the **[citation page](https://tarps-group.github.io/prob-pipe/cite/)** for
version-specific DOIs and how to cite the inference backends ProbPipe builds on.


