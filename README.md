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
- **Standalone Workflow Nodes:** Any Python callable can be turned into a workflow node, with explicit inputs and outputs.
- **Abstract Modules:** Modules define high-level algorithms (e.g., inference, forecasting, simulation) and internally manage their required workflow nodes.
- **Composable Architecture:** Every computational unit ("module") is built from smaller probabilistic primitives, allowing infinite composability of models and workflows.
- **Distributions as First-Class Objects:** Nodes can consume and emit probability distributions, enabling principled uncertainty propagation across the entire pipeline.
- **Automatic Type & Representation Handling:** The system manages conversions between different distribution representations used by different algorithms.
- **DAG-Based Execution:** Workflows are represented as explicit computational graphs, enabling inspection, reuse, and orchestration.

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
- ```numpy ≥ 2.0```
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
This example demonstrates a simple Bayesian-style pipeline where uncertainty is propagated through workflow nodes.

### Step 1: Define workflow nodes

```python
import numpy as np
from probpipe.core.node import wf
from probpipe.distributions.real_vector.gaussian import Gaussian
```

```python
@wf
def load_data() -> np.ndarray:
    np.random.seed(42)
    return np.random.normal(loc=2.5, scale=1.0, size=50)
```

```python
@wf
def fit_mean(data: np.ndarray) -> Gaussian:
    mu_hat = float(np.mean(data))
    sigma_hat = float(np.std(data) / np.sqrt(len(data)))
    return Gaussian(mu=mu_hat, cov=sigma_hat)
```

### Step 2: Compose a workflow

```python
data_node = load_data()
posterior_node = fit_mean(data=data_node)

posterior = posterior_node()
print(posterior)
```
- ```load_data``` and ```fit_mean``` are standalone workflow nodes.
- The output of ```fit_mean``` is a distribution, not a scalar.
- Uncertainty is explicitly represented and preserved.

### Step 3: Using abstract modules (conceptually)
More complex algorithms (e.g., Bayesian regression, forecasting, filtering) are implemented as modules, which internally assemble multiple workflow nodes.

```python
from probpipe.modules import BayesianLinearRegression

blr = BayesianLinearRegression()
posterior = blr(data=X, targets=y)
```

## Pointer to Further Documentation 
to be created soon...















