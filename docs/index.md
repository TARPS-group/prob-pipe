> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# ProbPipe

--8<-- "README.md:intro"

## What you can do

--8<-- "README.md:capabilities"

## How it works

ProbPipe keeps the mathematics separate from the computation.
A model states only the mathematics, and each operation chooses how to compute its result, so changing the algorithm never changes the model.
Six ideas make this work:

1. **Mathematical objects:** distributions, conditional distributions, functions, and values are ProbPipe objects, and each one names its components, such as the coefficients and the response of a regression. A batch holds several objects of one kind, such as a set of scenarios, on named axes.
2. **One vocabulary of operations:** `*` composes a conditional distribution with a distribution into their joint distribution, `condition_on` conditions, and summaries such as `mean` and `quantile` describe a law. Each operation applies to every object that supports it mathematically and returns another ProbPipe object, so results compose.
3. **Computation from capabilities:** an operation computes its result from what its inputs can do, by a closed form where one exists, and otherwise by an exact algorithm or an approximate method from a registry of backends such as BlackJAX, Stan, and PyMC. The choice is automatic, `check` reports it before a call runs, and `with_options` overrides it.
4. **Lifting:** an ordinary Python function applied to distributions returns the distribution of its output, and applied to a batch it returns the batch of its outputs.
5. **Traceable, reproducible results:** every result records the operation, the route, and the inputs that produced it, and `workflow_run(seed=...)` makes its random draws reproducible.
6. **Native Python and existing packages:** functions are plain Python, typically JAX code, and data can arrive in pandas or xarray objects. A model written in PyMC or Stan becomes a ProbPipe object, and a sampler's draws are available as ArviZ data.

## A short example

--8<-- "README.md:quick-example"

## Where to go next

- [Installation](get_started/installation.md) gives the install commands, and the [quickstart](get_started/quickstart.ipynb) works through the example above in ten minutes.
- The tutorials follow one analysis of a moose population, from a first model to forecasts that update as each count arrives, and they end with a model that can only be simulated. They start with [Tutorial 1: A model of the moose counts](tutorials/01_first_analysis.ipynb).
- The [API reference](api/index.md) documents every public name, grouped by topic.
- [Ray via Prefect](user_guide/ray.md) runs a pipeline's independent tasks on a Ray cluster.
- [Cite](cite.md) gives the citation, and [Help](help.md) says where to ask questions.
