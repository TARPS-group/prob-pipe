# Documentation

## Overview of the Project & Goals
ProbPipe is a Python-based workflow management system for probabilistic modeling and uncertainty quantification (UQ). Its core vision is to enable scientists and engineers to construct, compose, and execute probabilistic data-analysis pipelines in a modular and trustworthy way. Treating "distributions in → distributions out" as the central organizing principle.

### Motivation
Modern scientific and engineering workflows increasingly rely on quantifying and propagating uncertainty, especially in domains such as data assimilation, state-space modeling, and Bayesian inference. Moreover, these workflows are growing increasingly complex. There are many tools for specific aspects of the workflow, but putting them all together is challenging. While there are many ML/AI workflow management tools available (Apache Airflow, Orchestra, Prefect, Flyte, etc.), they are low-level frameworks designed for SWEs, not scientists. ProbPipe aims to provide a workflow management system specifically for probabilistic modeling, learning, and prediction, where users can take advantage of existing tools for statistical inference, validation, and learning.  It provides a simple, module-based interface for creating workflows abstracts away complexity while remaining fully extensible for advanced users. 
Ultimately, ProbPipe aims to serve as a general foundation for uncertainty-aware computation pipelines, bridging probabilistic modeling, data assimilation, and scientific machine learning.

### Design Goals
- **Uncertainty Quantification Built-In:** ProbPipe explicitly tracks uncertainty propagation, enabling distribution-aware computation across all modules.
- **Scalable and Multi-Scale:** The architecture enforces compositionality, where complex workflows can be built from simpler, independent components while providing automatic parallelization where possible. 
- **Easy to Deploy:** ProbPipe can seemlessly deploy across platforms, whether it's a laptop for initial development, a HPC cluster for academic projects, or a cloud service provide for scalability and redundency. 
- **Flexible yet User-Friendly:** Users can operate at a high level of abstraction, while advanced functionality remains accessible for expert customization.
- **Trustworthy Execution:** Validity of the workflow is checked at the start, reducing runtime errors and improving reliability.

### Key Features
- **Composable Architecture:** Every computational unit ("module") is built from smaller probabilistic primitives, allowing infinite composability of models and workflows.
- **Distributions as First-Class Objects:** Modules can consume and emit probability distributions, even if they were implemented using scalar inputs or outputs, makign it easy to propogate and account for uncertainty across complete workflows. 
- **Algorithmic-Level Operation:** Workflows are defined in terms of algorithmic components (sampling, inference, transformation), improving scalability and conceptual clarity.
- **Seamless Conversion and Data Handling:** The system automatically manages conversions between distributional representations preferred by different algorithms, streamlining code and improving clarity.

## Installation Instructions
### Prerequisites
Before installing ProbPipe, make sure you have:
- Conda (Anaconda or Miniconda) installed
- Python ≥ 3.8 (the recommended version for full compatibility)

You can verify Conda is available by running:

```bash
conda --version
```

If you don’t have Conda yet, download and install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Anaconda](https://www.anaconda.com/download).

### Create a Conda Environment
We recommend creating an isolated environment for ProbPipe to avoid dependency conflicts with other Python projects.

```bash
conda create -n probpipe python=<python_version>
conda activate probpipe
```
Note: python_version has to be ≥ 3.8

### Clone and Install from Source
Next, clone the repository and install the package inside your Conda environment:

```bash
git clone https://github.com/TARPS-group/prob-pipe.git
cd prob-pipe
pip install -e .
```

This installs the core dependencies listed in ```setup.py```:
- ```numpy ≥ 1.20```
- ```scipy ≥ 1.7```
- ```prefect ≥ 3.4```
- ```makefun ≥ 1.16```

### (Optional) Install Developer Tools

```bash
pip install -e .[dev]
```

This will include:
- **pytest** – for testing
- **sphinx, nbsphinx** – for documentation
- **black, flake8** – for linting and formatting

### Verify Installation
You can check whether the installation succeeded by running:

```bash
python -c "import probpipe; print(probpipe.__version__)"
```

Expected output:
```
0.1.0
```

### Updating Dependencies
If your environment is missing system-level libraries (e.g., libffi, libgcc, etc.), Conda can easily fix these:

```bash
conda install -c conda-forge numpy scipy prefect makefun
```

Then re-run:
```
pip install -e .
```

## Example Usage
This example demonstrates how to combine Prior, Likelihood, and MetropolisHastings modules with the MCMC class to estimate a posterior distribution over a scalar parameter θ: the mean of a Normal distribution with known standard deviation.

### Step 1: Generating synthetic data

```python
import numpy as np
from mcmc import Likelihood, Prior, MetropolisHastings, MCMC
from multivariate import Normal1D

np.random.seed(42)
true_mu = 2.5
sigma = 1.0
data = np.random.normal(loc=true_mu, scale=sigma, size=50)
```

### Step 2: Defining prior and likelihood distributions

```python
# Prior: Normal(0, 5)
prior_dist = Normal1D(mu=0.0, sigma=5.0)
# Likelihood base distribution: Normal(μ, 1.0)
likelihood_dist = Normal1D(mu=0.0, sigma=sigma)
```

### Step 3: Creating module instances
Each component (Prior, Likelihood, Sampler) is instantiated independently, then composed inside the MCMC module.

```python
prior = Prior(distribution=prior_dist)
likelihood = Likelihood(distribution=likelihood_dist)
sampler = MetropolisHastings()
```

### Step 4: Build and run the MCMC workflow
The MCMC module internally builds a log_target function combining prior and likelihood log-densities.

```python
mcmc = MCMC(prior=prior, likelihood=likelihood, sampler=sampler)

posterior = mcmc.calculate_posterior(
    num_samples=2000,
    initial_param=0.0,
    data=data,
    proposal_std=0.5,
)

# Summarizing posterior results
print(f"Posterior mean (estimate of μ): {posterior.mu:.3f}")
print(f"Posterior std: {posterior.sigma:.3f}")
```

After sampling, it returns a Normal1D summary object representing the posterior mean and uncertainty of the parameter.

**Expected outcome:**
With synthetic data generated around μ=2.5, you should see a posterior mean close to 2.5 and a standard deviation reflecting posterior uncertainty.

## Pointer to Further Documentation 
to be created soon...















