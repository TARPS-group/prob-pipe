ProbPipe is a Python framework for building probabilistic pipelines with automated uncertainty quantification. Its core design principle is **simplification via abstraction** — making probabilistic inference feel more mathematical via a functional, registry-based system built around just three main types you need to know about.

## Why ProbPipe?

Most workflows for probabilistic inference can be described in terms of **distributions**, **fixed values** (data, hyperparameters, covariates), and **operations** that transform distributions. But implementing these workflows is harder than describing them because math has to be translated into computation:

- **Algorithmic challenges.** There are many possible algorithms for common operations, with varying trade-offs that need to be explored in a problem-specific manner. A posterior could be approximated using a variety of MCMC algorithms, variational inference methods, or sequential Monte Carlo, or might require more specialized methods such as those for amortized and simulation-based inference. 
- **Representational challenges.** Algorithms expect (and produce) specific formats for distributions and fixed values, and those formats are not always compatible with other parts of the workflow. Fixed values may be named parameter vectors, covariate matrices, or structured observations, and different algorithms expect different representations.

In practice, these issues make it hard to explore the full design space of available methods or to build more complex workflows that many algorithms for different steps. ProbPipe addresses these challenges through the **simplification via abstraction** design principle. For example, there are just three core types:

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

## Installation

ProbPipe requires Python ≥ 3.12 (tested on 3.12, 3.13, and 3.14). ProbPipe is
not yet on PyPI, so installation is from source.

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

Then work through the [Getting Started tutorial](tutorials/getting_started.ipynb).
(Prefer not to activate? Prefix commands with `uv run`, e.g. `uv run python`.)

### Already have a Python environment?

Install from source with pip into your active environment:

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install .
```

uv users can substitute `uv pip install .` (into an active `uv venv`), or
`uv sync` for a lockfile-managed dev environment (see
[CONTRIBUTING.md](https://github.com/TARPS-group/prob-pipe/blob/main/CONTRIBUTING.md#installation)).

### Two distributions: `probpipe` and `probpipe-core`

ProbPipe ships as two distributions that share the same `probpipe` import name:

| Install | What you get |
|---|---|
| `pip install probpipe` | **Recommended.** The lean core plus the inference backends the docs use — PyMC, nutpie, and BayesFlow — so every example and tutorial runs out of the box. |
| `pip install probpipe-core` | **Lean.** The JAX base only (JAX, BlackJAX, TFP, ArviZ); add backends as extras, e.g. `pip install "probpipe-core[pymc]"`. |

`probpipe` already bundles PyMC, nutpie, and BayesFlow. Any remaining optional extra can be added on top with either name — `pip install "probpipe[prefect]"` (also `[viz]`, `[stan]`) — and lean `probpipe-core` users add any backend the same way, e.g. `pip install "probpipe-core[pymc]"`. On **Python 3.14** `probpipe` omits BayesFlow (its neural-SBI backend caps `<3.14`) until upstream lifts the cap; everything else is unaffected.

> Publishing to PyPI is pending; for now install from source as shown above (the repository root builds `probpipe-core` — add the extras you need).

### Dependencies and optional extras

Core dependencies: JAX and TensorFlow Probability. ProbPipe uses [tfp-nightly](https://pypi.org/project/tfp-nightly/), which is the [recommended approach](https://github.com/tensorflow/probability/issues/1994#issuecomment-3129033043) for TFP on JAX since stable TFP releases are tied to TensorFlow and often lag behind JAX.

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

ProbPipe can dispatch Prefect-orchestrated `WorkflowFunction` tasks to Ray via Prefect-Ray:

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

See the [Ray via Prefect guide](orchestration/ray.md) for setup details, deployment notes, and current support boundaries.

## Next Steps

Once you've installed ProbPipe, check out the **[Getting Started Tutorial](tutorials/getting_started.ipynb)**. You can also check out:

- **[User Guide](user_guide.md)**
- **[API Reference](api/index.md)**
- **[Contributing Guide](https://github.com/TARPS-group/prob-pipe/blob/main/CONTRIBUTING.md)**
