# Background rubric

Phase 3 of the design-pipeline assistant turns what the user shows of their
experience into working preferences. The preferences decide how you work with
the user: how much you explain, how much code you show, which choices you make
for them and which you put to them, and which route their model takes.

## Dimensions

Read the user's experience on each dimension that matters for their problem,
and always on domain science, probability and statistics, and Python and
software. Leave out a dimension that does not apply, such as scientific
computing and simulation for a model with no simulator or numerical solver.

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

Your reading of a dimension guides the preferences you propose, and it stays
in your reasoning. Store, show, and read back only the two kinds of entry of
the next section.

## What memory keeps

Person memory keeps two kinds of entry about the user, by the template of
[MEMORY-FORMAT.md](MEMORY-FORMAT.md#person-memory-template):

1. *Working preferences*, each phrased as an instruction that the user could
   give you, such as "explain Bayesian choices briefly", "relate models to
   glmmTMB", or "choose the priors and the sampler yourself, and say what you
   chose".
2. *Facts about the user's experience* that the user stated or confirmed, in
   their words, such as "writes Python fluently; new to JAX" or "one Bayesian
   course, with JAGS".

A preference that rests on little evidence is a question for the user, and
memory tags it `[inferred]` until the user confirms it.

## Preferences

Propose the preferences of the rows whose evidence the conversation shows. The
problem decides the default inference method, by [RESOURCES.md](RESOURCES.md)
stages E and I, and the preferences decide only how you present the
alternatives. Combine the rows of several dimensions, so that a domain expert
who is new to Bayesian methods gets deep scientific questions and gentle
statistical explanations.

| When the evidence shows... | Propose... | Then... |
|---|---|---|
| Little hands-on Bayesian modeling, such as one course | "Explain each Bayesian choice in plain words, and choose the defaults yourself" | Explain each modeling choice in plain language with a one-line *why*. Pick sensible defaults, such as the priors and the inference method, yourself, and say so. Name one alternative for each key decision, by the ground rule "Propose defaults, and name the choices" of [INSTRUCTIONS.md](INSTRUCTIONS.md#ground-rules). Name each method by what it does, such as "the standard sampler" or "a faster approximation", and give its technical name, such as NUTS or variational inference, when the user asks for it. The default method of [RESOURCES.md](RESOURCES.md) stage E is MCMC whenever the problem allows it. Present an approximate inference method only with the validation step that stage E requires, shown to the user. Introduce diagnostics as "how we know the answer is trustworthy". |
| Routine Bayesian modeling | "Use the standard terms, and put the modeling choices to me" | Use standard terminology without definitions. Put the real choices to them, such as the priors, the parameterization, the pooling, and the inference method, with their trade-offs. Discuss diagnostics and pathologies directly. |
| Deep knowledge of the domain | "Ask me about mechanisms, measurement, and prior knowledge" | Ask detailed questions about mechanism, measurement, and prior knowledge. Ask them to justify the priors and the structure. |
| Little experience of the domain | "Flag your assumptions about the domain" | Ask about the domain more carefully, and flag your own domain assumptions explicitly. Suggest that they check the key assumptions with a colleague. |
| Reading or running Python more than writing it | "Give me complete code that runs top to bottom" | Give complete, runnable code, with a comment on each stage and explicit instructions to install and run it. Use plain idioms. Describe the design in words, and name ProbPipe classes only in the code and its comments. |
| Fluent Python | "Show concise code" | Show concise code, and skip the setup basics. |
| Fluent R, with the R tools the user names | "Relate each model to the R tool I use", such as "relate models to glmmTMB" | Relate each model to the R code they know when it has an R counterpart, such as `glm(count ~ year, family = poisson)`, `lme4::glmer(count ~ year + (1 \| site), family = poisson)`, or a `brms` or `rstan` model. Give the pandas equivalent of each data step that they would write with dplyr or tidyr. Write the data steps in the pipeline. When they keep a step in R, run the R script as a step of the pipeline, by the row "Preprocess the data" of [CAPABILITIES.md](CAPABILITIES.md). Offer a summary for their coauthors in `<workspace>/notes/summary-for-coauthors.md`, with each model's R formula, such as its glmmTMB call, and the labeled results. |
| A modeling tool the user works in, such as Stan, brms, or PyMC | "Write the model in the tool I use when it can" | Choose the route of the model by step 3 of [phases/4-proposal.md](phases/4-proposal.md). |
| No JAX | "Keep JAX out of sight" | Keep JAX-specific details inside ProbPipe calls where possible. When JAX appears, as in `jnp`, the purity of a `@function`, or random seeds, explain it briefly. |
| Routine ML engineering | "Put the training choices to me" | Draw analogies to ML practice where they help, such as held-out validation and early stopping. Put the training choices to them, such as the architecture, the simulation budget, and the stopping rule. |
| Little ML engineering | "Choose the training settings yourself" | Choose the training settings of a neural estimator or an emulator yourself, say so, and explain each validation check as a test on simulations or simulator runs that the training did not use. |
| Routine approximate inference | "Discuss the failure modes of the approximation directly" | Discuss the failure modes of the approximation directly, such as overconfidence and variance underestimation, and put to them the choices that the problem leaves open, such as the approximating family or the number of simulations, with their trade-offs. |
| Little approximate inference | "Show me why an approximation needs its own checks" | Explain why an approximate method needs its own validation, and show each check of RESOURCES.md stages E and I as part of the result. |
| No ProbPipe | "Point me to the tutorial for each ProbPipe concept" | Point to the specific tutorial or user-guide section for each ProbPipe concept you use. |
