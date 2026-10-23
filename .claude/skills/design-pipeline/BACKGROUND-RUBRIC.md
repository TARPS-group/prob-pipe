# Background assessment rubric

Used in Phase 3 of the `design-pipeline` skill. Rate each dimension that
matters for this user's problem. Leave out the ones that don't apply (for
example, simulators for a pure regression problem), but always include
domain, probability and statistics, and Python.

## Levels

| Level | Meaning |
|---|---|
| **New** | Little or no exposure. Needs concepts introduced from the ground up. |
| **Familiar** | Knows the ideas and vocabulary; limited hands-on use. |
| **Working** | Uses it routinely; can judge standard choices; may miss subtleties. |
| **Expert** | Deep, current knowledge; can critique methods and spot edge cases. |

Also give a **confidence**: *high* (direct evidence, such as their own
statement or demonstrated use), *medium* (consistent indirect signals),
or *low* (little evidence, mostly a default). A low-confidence rating is a
question for the user, so frame it that way.

## Dimensions

| Dimension | What it covers | Signals |
|---|---|---|
| **Domain science** | The user's field: mechanisms, measurement, conventions, literature. | Precise terminology; talks about mechanisms and measurement error; cites papers; anticipates confounders. |
| **Probability and statistics** | Random variables, distributions, likelihood, estimation, uncertainty, regression. | Distinguishes parameters from data, variance from uncertainty; talks about likelihoods, standard errors, intervals. |
| **Bayesian modeling and inference** | Priors, posteriors, hierarchical models, MCMC and VI, diagnostics, predictive checks. | Mentions priors, pooling, R-hat, divergences, posterior predictive; has used Stan, PyMC, or similar. |
| **Machine learning** | Predictive modeling, neural networks, training and validation, amortized or simulation-based inference. | Talks about train/test splits, overfitting, architectures; has used PyTorch, JAX, or scikit-learn. |
| **Scientific computing and simulation** | Simulators, ODE/PDE models, numerical methods, HPC. | Has a simulator; discusses solvers, run times, clusters. |
| **Python and software** | Python fluency, packages, environments, version control, testing. | Shares or describes code; mentions virtual environments, notebooks, git. |
| **JAX** | Arrays, `jit`, `vmap`, `grad`, pure functions, PRNG keys. | Mentions JAX or its idioms; code is functional and array-oriented. |
| **ProbPipe** | ProbPipe itself. | Has run the tutorials; uses ProbPipe vocabulary (`Record`, `condition_on`). |

## Adaptation

How the profile shapes the session. Combine dimensions; for example, a
domain expert who is new to Bayesian methods gets deep scientific questions
and gentle statistical explanations.

| If the user is... | Then... |
|---|---|
| New/Familiar in Bayesian modeling | Explain each modeling choice in plain language with a one-line *why*. Pick sensible defaults (priors, inference method) yourself and say so, instead of asking them to choose. Introduce diagnostics as "how we know the answer is trustworthy". |
| Working/Expert in Bayesian modeling | Use standard terminology without definitions. Put real choices to them (priors, parameterization, pooling, inference method) with trade-offs. Discuss diagnostics and pathologies directly. |
| Expert in the domain | Ask detailed questions about mechanism, measurement, and prior knowledge. Lean on them to justify priors and structure. Don't explain their field to them. |
| New/Familiar in the domain | Ask about the domain more carefully and flag your own domain assumptions explicitly. Suggest they check key assumptions with a colleague. |
| New/Familiar in Python | Give complete, runnable code with comments on each stage and explicit setup instructions (install, run). Avoid clever idioms. |
| Working/Expert in Python | Show concise code; skip setup basics. |
| New to JAX | Hide JAX-specific details behind ProbPipe where possible. When it shows up (`jnp`, `@function` purity, random seeds), explain it briefly. |
| Working/Expert in ML | Draw analogies to ML practice where they help (e.g. amortized inference, validation). Consider ML-flavored options such as emulators and neural SBI when they fit. |
| New to ProbPipe | Point to the specific tutorial or user-guide section for each ProbPipe concept you use. |
