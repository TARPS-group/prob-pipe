# Documentation

## Overview of the Project & Goals
ProbPipe is a Python framework designed to make probabilistic modeling and uncertainty quantification (UQ) a first-class citizen within workflow management systems. Its core vision is to enable scientists and engineers to construct, compose, and execute probabilistic data-analysis pipelines in a modular and trustworthy way. Treating "distributions in → distributions out" as the central organizing principle.

### Motivation
Modern scientific and engineering workflows increasingly rely on quantifying and propagating uncertainty, especially in domains such as data assimilation, state-space modeling, and Bayesian inference. While existing systems specialize in data assimilation algorithms, they often expose users to implementation complexity or rigid interfaces. ProbPipe aims to provide a cleaner, more flexible, and scalable Python interface that abstracts this complexity while remaining fully extensible for advanced users. In the nearer term, the framework is targeted at state-space modeling and data assimilation tasks, where the goal is to recursively estimate latent states or parameters of dynamical systems from noisy and possibly streaming observations. Ecological applications are a key motivating example, specifically, calibration and data assimilation for models of the terrestrial carbon cycle.

### Design Goals
- **Uncertainty Quantification Built-In:** ProbPipe explicitly tracks uncertainty propagation, enabling distribution-aware computation across all modules.
- **Scalable and Multi-Scale:** The architecture enforces compositionality—complex workflows can be built from simpler, independent components, naturally scaling to multi-scale systems.
- **Flexible yet User-Friendly:** Users can operate at a high level of abstraction, while advanced functionality remains accessible for expert customization.
- **Trustworthy Execution:** Compatibility between computational modules is verified at compile time, reducing runtime errors and improving reliability.

### Key Features
- **Composable Architecture:** Every computational unit ("module") is built from smaller probabilistic primitives, allowing infinite composability of models and workflows.
- **Distributions as First-Class Objects:** Modules consume and emit probability distributions, rather than point estimates—enabling automatic provenance tracking and uncertainty decomposition.
- **Algorithmic-Level Operation:** Workflows are defined in terms of algorithmic components (sampling, inference, transformation), improving scalability and conceptual clarity.
- **Automatic Compatibility Checks:** Type and shape compatibility between modules are validated early, promoting both trust and developer productivity.
- **Seamless Conversion and Data Handling:** The system automatically manages conversions between distributional representations preferred by different algorithms, reducing cognitive load on the user.

### Long-Term Vision
Ultimately, ProbPipe aims to serve as a general foundation for uncertainty-aware computation pipelines, bridging probabilistic modeling, data assimilation, and scientific machine learning. By emphasizing compositionality, provenance, and clarity, it aspires to provide a robust ecosystem for reproducible Bayesian workflows—from ecological calibration studies to scalable probabilistic data science.

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
This minimal module computes the posterior for a Normal likelihood with known σ and a Normal prior on μ. It returns a Normal1D (a distribution out, not a point estimate).

```python
import numpy as np
from probpipe import Module, Normal1D, EmpiricalDistribution

class GaussianPosterior(Module):
    """
    Posterior for: y_i ~ Normal(mu, sigma^2), prior mu ~ Normal(mu0, var0)
    Returns posterior Normal1D(mu_n, var_n).

    NOTE: This module expects a Normal1D prior at runtime.
    If the user passes an EmpiricalDistribution (or other supported types),
    the Module base class auto-converts it to Normal1D before this method runs.
    """

    def __init__(self, **deps):
        super().__init__(**deps)

        # Controls for auto-conversion (see notes below)
        self._conv_num_samples = 2048      # resample size for conversions
        self._conv_by_kde = False          # default: moment matching
        self._conv_fit_kwargs = {}         # kwargs forwarded to KDE fitting (if enabled)

        # Register the internal function as the callable
        self.run_func(self._calculate_posterior, name="calculate_posterior", as_task=True)

    def _calculate_posterior(self, *, prior: Normal1D, y: np.ndarray, sigma: float = 1.0) -> Normal1D:
        """
        Compute posterior Normal1D given Normal likelihood with known sigma,
        prior Normal1D (mu0, var0), and data y.
        (By this point, 'prior' has been auto-converted to Normal1D if needed.)
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        n = y.size
        var = float(sigma) ** 2

        # Prior parameters (Normal1D interface)
        mu0 = float(prior.mean())
        var0 = float(prior.cov())  # variance

        # Conjugate update
        var_n = 1.0 / (n / var + 1.0 / var0)
        mu_n  = var_n * (y.sum() / var + mu0 / var0)

        return Normal1D(mu_n, np.sqrt(var_n))
```


### Case A — Normal prior (conjugate)

```python
gp = GaussianPosterior()
prior_norm = Normal1D(mu=0.0, sigma=5.0)

post = gp.calculate_posterior(
    prior=prior_norm,
    y=np.array([1.2, 0.7, -0.3, 0.4]),
    sigma=1.0
)
```

### Case B — Empirical prior (auto-converted to Normal1D)

```python
emp_prior = EmpiricalDistribution(
    samples=np.array([1.3, 2.0, 3.3, 4.1, 5.1]).reshape(-1, 1)
)

post2 = gp.calculate_posterior(
    prior=emp_prior,     # <-- auto-conversion happens under the hood
    y=np.array([1.2, 0.7, 0.3, 0.4]),
    sigma=1.0
)
```

**Default path:**
Moment matching (uses the empirical mean/variance of the provided distribution to form a Normal1D prior).

**KDE path:**
If you prefer smoothing before matching, enable KDE:

```python
gp._conv_by_kde = True
gp._conv_num_samples = 4096                   # (optional) more resamples
gp._conv_fit_kwargs = {"bandwidth": "scott"}  # (optional) KDE tuning
```

## Pointer to Further Documentation 
to be created soon...















