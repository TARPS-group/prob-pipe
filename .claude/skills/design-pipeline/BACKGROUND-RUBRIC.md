# Background assessment rubric

Phase 3 of the design-pipeline assistant rates the user's background with this
rubric. Rate each dimension that matters for the user's problem, and always
rate domain science, probability and statistics, and Python and software.
Leave out a dimension that does not apply, such as scientific computing and
simulation for a model with no simulator or numerical solver.

## Ratings

| Rating | Meaning |
|---|---|
| **New** | Has little or no exposure, and needs each concept introduced from the beginning. |
| **Familiar** | Knows the ideas and the vocabulary, with limited hands-on use. |
| **Working** | Uses it routinely and can judge the standard choices, but may miss subtleties. |
| **Expert** | Has deep, current knowledge, and can critique methods and spot edge cases. |

Give each rating a **confidence**:

- *high*: direct evidence, such as the user's own statement or demonstrated
  use;
- *medium*: consistent indirect signals;
- *low*: little evidence, so the rating is mostly a default.

Frame a low-confidence rating as a question for the user.

## Dimensions

| Dimension | What it covers | Signals |
|---|---|---|
| **Domain science** | The user's discipline: its mechanisms, measurement, conventions, and literature. | Uses precise terminology and cites papers. Talks about mechanisms and measurement error, and anticipates confounders. |
| **Probability and statistics** | Random variables, distributions, likelihood, estimation, uncertainty, regression. | Distinguishes parameters from data and variance from uncertainty. Talks about likelihoods, standard errors, and intervals. |
| **Bayesian modeling and inference** | Priors, posteriors, hierarchical models, MCMC and VI, diagnostics, predictive checks. | Mentions priors, pooling, R-hat, divergences, or posterior predictive checks. Has used Stan, PyMC, or similar software. |
| **ML engineering** | Training neural networks and other predictive models: validation splits, early stopping, hyperparameter tuning, GPUs. | Talks about train/validation splits, overfitting, architectures, or learning rates. Has used PyTorch, JAX, or scikit-learn. |
| **Approximate inference** | Variational inference, normalizing flows, simulation-based and amortized inference, Gaussian processes, stochastic-gradient MCMC, and how each one fails. | Mentions the ELBO, NPE or NLE, coverage or SBC of an approximate posterior, or kernels and length scales. Has used BayesFlow or GPyTorch. |
| **Scientific computing and simulation** | Simulators, ODE/PDE models, numerical methods, HPC. | Has a simulator, and discusses solvers, run times, or clusters. |
| **Python and software** | Python fluency, fluency in other languages such as R, packages, environments, version control, testing. | Shares or describes code, and mentions virtual environments, notebooks, or git. Names R packages or writes R model formulas. |
| **JAX** | Arrays, `jit`, `vmap`, `grad`, pure functions, PRNG keys. | Mentions JAX or its idioms, and writes functional, array-oriented code. |
| **ProbPipe** | ProbPipe itself. | Has run the tutorials, and uses ProbPipe vocabulary, such as `Record` and `condition_on`. |

## Adaptation

The profile shapes the session by the rows of this table. It decides how much
you explain, which choices you make for the user, and which you put to them.
The problem alone decides which inference method you recommend, by
[RESOURCES.md](RESOURCES.md) stages E and I. Combine the rows of several
dimensions, so that a domain expert who is new to Bayesian methods gets deep
scientific questions and gentle statistical explanations.

| If the user is... | Then... |
|---|---|
| New or Familiar in Bayesian modeling | Explain each modeling choice in plain language with a one-line *why*. Pick sensible defaults, such as the priors and the inference method, yourself, and say so. Use MCMC when it is feasible, and propose an approximate inference method only with the validation step that [RESOURCES.md](RESOURCES.md) stage E requires, shown to the user. Introduce diagnostics as "how we know the answer is trustworthy". |
| Working or Expert in Bayesian modeling | Use standard terminology without definitions. Put the real choices to them, such as the priors, the parameterization, the pooling, and the inference method, with their trade-offs. Discuss diagnostics and pathologies directly. |
| Expert in the domain | Ask detailed questions about mechanism, measurement, and prior knowledge. Ask them to justify the priors and the structure. |
| New or Familiar in the domain | Ask about the domain more carefully, and flag your own domain assumptions explicitly. Suggest that they check the key assumptions with a colleague. |
| New or Familiar in Python | Give complete, runnable code, with a comment on each stage and explicit instructions to install and run it. Use plain idioms. Describe the design in words, and name ProbPipe classes only in the code and its comments. |
| Working or Expert in Python | Show concise code, and skip the setup basics. |
| Fluent in R | Relate each model to the R code they know when it has an R counterpart, such as `glm(count ~ year, family = poisson)`, `lme4::glmer(count ~ year + (1 \| site), family = poisson)`, or a `brms` or `rstan` model. Give the pandas equivalent of each data step that they would write with dplyr or tidyr. When they prefer, let them prepare the data in R and export a CSV for the pipeline. |
| New to JAX | Keep JAX-specific details inside ProbPipe calls where possible. When JAX appears, as in `jnp`, the purity of a `@function`, or random seeds, explain it briefly. |
| Working or Expert in ML engineering | Draw analogies to ML practice where they help, such as held-out validation and early stopping. Put the training choices to them, such as the architecture, the simulation budget, and the stopping rule. |
| New or Familiar in ML engineering | Choose the training settings of a neural estimator or an emulator yourself, say so, and explain each validation check as a test on simulations or simulator runs that the training did not use. |
| Working or Expert in approximate inference | Discuss the failure modes of the approximation directly, such as overconfidence and variance underestimation, and put to them the choices that the problem leaves open, such as the approximating family or the number of simulations, with their trade-offs. |
| New or Familiar in approximate inference | Explain why an approximate method needs its own validation, and show each check of RESOURCES.md stages E and I as part of the result. |
| New to ProbPipe | Point to the specific tutorial or user-guide section for each ProbPipe concept you use. |
