> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Installation

ProbPipe requires Python 3.12 or later, and it installs JAX as a dependency.
ProbPipe is not yet on PyPI, so every command on this page installs it from GitHub.
Until a release, the commands install the `dev/overhaul` branch, which is the version this documentation describes.
pip and uv clone the repository to install it, so they need [Git](https://git-scm.com/downloads).

## The two distributions

ProbPipe has two distributions, and both install the same import package, `probpipe`:

| Distribution | What it installs | When to use it |
|---|---|---|
| `probpipe-core` | ProbPipe with JAX, BlackJAX, TensorFlow Probability, and ArviZ | A minimal install, to which you add the extras you need |
| `probpipe` | `probpipe-core` with the extras `pymc`, `nutpie`, and `pyabc`, and with `bayesflow` on Python 3.12 and 3.13 | Running the tutorials |

The `probpipe` distribution depends on `probpipe-core==0.1.0`, which is not yet on PyPI.
Until it is, install the tutorials' set of extras through `probpipe-core`, as the commands below do.

## Google Colab

[Google Colab](https://colab.research.google.com/) runs Python in your browser, so it needs no install of Python.
To install the minimal set, run this in a code cell:

```text
%pip install "probpipe-core @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

To install the set the tutorials use, run this instead:

```text
%pip install "probpipe-core[pymc,nutpie,pyabc,bayesflow] @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

If Colab's Python is 3.14, leave `bayesflow` out of the list, as `!python --version` shows.
The quickstart and the tutorials open with a cell that installs ProbPipe when they run on Colab.

## pip

Create a virtual environment, activate it, and install ProbPipe into it:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install "probpipe-core @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

To install the set the tutorials use, name the extras in brackets:

```bash
pip install "probpipe-core[pymc,nutpie,pyabc,bayesflow] @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

On Python 3.14, leave `bayesflow` out of the list.

## uv

In a project that [uv](https://docs.astral.sh/uv/) manages, `uv add` installs ProbPipe and records it as a dependency in `pyproject.toml`:

```bash
uv add "probpipe-core @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

In a virtual environment that is not a uv project, `uv pip install` takes the same arguments as `pip install`:

```bash
uv venv
uv pip install "probpipe-core @ git+https://github.com/TARPS-group/prob-pipe.git@dev/overhaul"
```

Both commands take extras in brackets, as pip does.

## Extras

The extras add optional backends and tools:

| Extra | What it adds | When you need it |
|---|---|---|
| `stan` | BridgeStan and CmdStanPy | To write a model as a Stan program with `StanModel`, and to sample it with CmdStan through the method `cmdstan_nuts` |
| `pymc` | PyMC and matplotlib | To write a model in PyMC with `PyMCModel`, and to fit it with PyMC's methods `pymc_nuts` and `pymc_advi` |
| `nutpie` | nutpie | To sample a Stan or PyMC model with nutpie's NUTS sampler through the method `nutpie_nuts`, which `condition_on` selects first for those models |
| `bayesflow` | BayesFlow and Keras | For amortized neural simulation-based inference with `learn_amortized_posterior`, `learn_amortized_likelihood`, and `learn_amortized_ratio` |
| `pyabc` | pyabc | For simulation-based inference by sequential Monte Carlo approximate Bayesian computation (SMC-ABC), the method `pyabc_smcabc` |
| `prefect` | Prefect | To run a pipeline's functions as Prefect tasks and flows, and on a Ray cluster through Prefect, as [Ray via Prefect](../user_guide/ray.md) describes |
| `viz` | matplotlib and graphviz | To draw the plots of the quickstart and the tutorials, and a provenance graph with `provenance_dag` |

Two extras need more than their packages:

1. **`stan`:** CmdStan and a C++ toolchain, which compile Stan programs. CmdStanPy's command `install_cmdstan` installs CmdStan, and the [CmdStanPy installation guide](https://mc-stan.org/cmdstanpy/installation.html) lists the toolchain each operating system needs.
2. **`bayesflow`:** Python 3.12 or 3.13, and the JAX backend of Keras. ProbPipe sets the environment variable `KERAS_BACKEND` to `jax` when it first imports BayesFlow, unless the variable is already set. A program that imports Keras before then, or sets the variable to another backend, must set `KERAS_BACKEND=jax` before its first import of Keras.

The extras `docs` and `dev` are for contributors, and the [contributing guide](https://github.com/TARPS-group/prob-pipe/blob/main/CONTRIBUTING.md) uses them to set up a development environment.

## New to Python?

Neither of these two ways needs Python installed beforehand:

- **In the browser:** open the [quickstart](quickstart.ipynb) in Google Colab from the badge at its top, which needs only a Google account.
- **On your computer:** install uv by following its [installation guide](https://docs.astral.sh/uv/getting-started/installation/). uv downloads a suitable Python when it needs one. The command `uv init my-analysis` creates a project in the folder `my-analysis`, and the `uv add` command of the [uv section](#uv) run in that folder installs ProbPipe into it. The command `uv run python` then starts Python in the project's environment.

The [Python tutorial](https://docs.python.org/3/tutorial/) introduces the language itself.

## Check the installation

This script imports ProbPipe and draws from a normal distribution with mean 3 and standard deviation 1:

```python
import jax.numpy as jnp

import probpipe
from probpipe import Normal, sample, workflow_run

with workflow_run(seed=0):
    draws = sample(Normal("x", 3.0, 1.0), sample_shape=1000)

print("ProbPipe version:", probpipe.__version__)
print("number of draws:", draws.shape[0])
print("mean of the draws:", round(float(jnp.mean(draws.values)), 2))
print("standard deviation of the draws:", round(float(jnp.std(draws.values)), 2))
```

It prints:

```text
ProbPipe version: 0.1.0
number of draws: 1000
mean of the draws: 3.0
standard deviation of the draws: 1.0
```

## Next steps

- The [quickstart](quickstart.ipynb) fits a first model, checks it, and forecasts with it in about ten minutes.
- The [first tutorial](../tutorials/01_first_analysis.ipynb) begins a forecasting problem that the tutorials follow from a first posterior to simulation-based inference.
