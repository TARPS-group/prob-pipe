# Background assessment rubric

Phase 3 of the `design-pipeline` skill rates the user's background with this
rubric. Rate each dimension that matters for the user's problem, and always
rate domain science, probability and statistics, and Python and software.
Leave out a dimension that does not apply, such as scientific computing and
simulation for a pure regression problem.

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
| **Domain science** | The user's discipline: its mechanisms, measurement, conventions, and literature. | Precise terminology; talks about mechanisms and measurement error; cites papers; anticipates confounders. |
| **Probability and statistics** | Random variables, distributions, likelihood, estimation, uncertainty, regression. | Distinguishes parameters from data, variance from uncertainty; talks about likelihoods, standard errors, intervals. |
| **Bayesian modeling and inference** | Priors, posteriors, hierarchical models, MCMC and VI, diagnostics, predictive checks. | Mentions priors, pooling, R-hat, divergences, posterior predictive checks; has used Stan, PyMC, or similar. |
| **Machine learning** | Predictive modeling, neural networks, training and validation, amortized or simulation-based inference. | Talks about train/test splits, overfitting, architectures; has used PyTorch, JAX, or scikit-learn. |
| **Scientific computing and simulation** | Simulators, ODE/PDE models, numerical methods, HPC. | Has a simulator; discusses solvers, run times, clusters. |
| **Python and software** | Python fluency, packages, environments, version control, testing. | Shares or describes code; mentions virtual environments, notebooks, git. |
| **JAX** | Arrays, `jit`, `vmap`, `grad`, pure functions, PRNG keys. | Mentions JAX or its idioms; writes functional, array-oriented code. |
| **ProbPipe** | ProbPipe itself. | Has run the tutorials; uses ProbPipe vocabulary, such as `Record` and `condition_on`. |

## Adaptation

The profile shapes the session by the rows of this table. Combine the rows of
several dimensions, so that a domain expert who is new to Bayesian methods gets
deep scientific questions and gentle statistical explanations.

| If the user is... | Then... |
|---|---|
| New or Familiar in Bayesian modeling | Explain each modeling choice in plain language with a one-line *why*. Pick sensible defaults, such as the priors and the inference method, yourself, and say so. Introduce diagnostics as "how we know the answer is trustworthy". |
| Working or Expert in Bayesian modeling | Use standard terminology without definitions. Put the real choices to them, such as the priors, the parameterization, the pooling, and the inference method, with their trade-offs. Discuss diagnostics and pathologies directly. |
| Expert in the domain | Ask detailed questions about mechanism, measurement, and prior knowledge. Ask them to justify the priors and the structure. |
| New or Familiar in the domain | Ask about the domain more carefully, and flag your own domain assumptions explicitly. Suggest that they check the key assumptions with a colleague. |
| New or Familiar in Python | Give complete, runnable code, with a comment on each stage and explicit instructions to install and run it. Use plain idioms. |
| Working or Expert in Python | Show concise code, and skip the setup basics. |
| New to JAX | Keep JAX-specific details inside ProbPipe calls where possible. When JAX appears, as in `jnp`, the purity of a `@function`, or random seeds, explain it briefly. |
| Working or Expert in ML | Draw analogies to ML practice where they help, as for amortized inference and validation. Consider options from ML, such as emulators and neural SBI, when they fit. |
| New to ProbPipe | Point to the specific tutorial or user-guide section for each ProbPipe concept you use. |
