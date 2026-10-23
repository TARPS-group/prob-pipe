# Workflow guidance and resources

Used by the `design-pipeline` skill to make pipelines statistically and
computationally sound. Part 1 is an operational checklist, distilled mainly
from *Bayesian Workflow* (Gelman, Vehtari, McElreath et al.), with each step
mapped to the ProbPipe tools that implement it. Part 2 is an annotated
bibliography: what each resource is good for and when to point a user to it.

Apply Part 1 from your own knowledge of these sources; you don't need network
access. If you can reach the web and need detail beyond what you know (a
specific diagnostic threshold, say), consult the source. If a link is dead,
search for the title. Cite only resources from this list, or ones you have
actually opened in this session.

---

## Part 1 — Workflow checklist

A pipeline design is sound when every stage below has been considered and
either addressed or consciously skipped with a reason recorded in the design
doc. The workflow is iterative: problems found late often send you back to
the model, which is normal and should be planned for, not treated as failure.

### A. Before writing a model

- **Start from the question.** Tie every model component to the estimands. A
  model that can't express the quantity of interest is the wrong model,
  however well it fits.
- **Understand the measurement process.** How was the data collected? Censoring,
  truncation, selection, measurement error, and missingness all belong in the
  model, or in an explicit assumption.
- **Start simple, then expand.** Begin with the simplest model that addresses
  the question, then add structure (hierarchy, covariates, nonlinearity) one
  step at a time, checking at each step. Plan the sequence of models up front.
  Keep earlier models: they're baselines for comparison and debugging.

### B. Priors

- Choose priors on interpretable scales, using domain knowledge where it
  exists. Weakly informative priors (ruling out implausible magnitudes) beat
  flat priors almost always.
- The prior only means something in the context of the likelihood: judge it
  through its **prior predictive distribution**, not in isolation.
- Rescale or standardize parameters so that priors and the sampler both work
  on order-one scales.
- *ProbPipe:* distribution families, `ProductDistribution`,
  `SequentialJointDistribution`, constraints and bijectors.

### C. Prior predictive checks

- Simulate data from the prior and ask the domain expert whether it looks
  possible. Implausible simulations (negative counts, astronomically large
  values) mean the priors or the model structure need work.
- *ProbPipe:* `predictive_check` on the prior model; `sample`.

### D. Validate the computation on simulated data

- **Parameter recovery:** simulate data from known parameter values, fit, and
  check the posterior covers them. Cheap, and catches many bugs.
- **Simulation-based calibration (SBC):** repeat over many prior draws and
  check rank uniformity; use data-dependent test quantities (such as the log
  likelihood) for more sensitivity. Worth doing for new or custom models and
  any approximate inference method; may be too expensive for slow models.
- *ProbPipe:* generative models (`SimpleGenerativeModel`), `@function`
  broadcasting over prior draws, `probpipe.validation` metrics.

### E. Fit and diagnose the computation

- Use the default algorithm (NUTS through `condition_on`) unless there's a
  reason not to. Run multiple chains.
- Check R-hat (rank-normalized, target < 1.01), bulk and tail ESS, divergences,
  and tree depth. Don't interpret a posterior with failing diagnostics.
- **Folk theorem:** computational trouble usually signals a model problem.
  Before tuning the sampler, look for weak identifiability, bad parameterization
  (try non-centered for hierarchical models), poorly scaled parameters, or a
  model that disagrees with the data.
- For approximate methods (VI, amortized or simulation-based inference), check
  their accuracy: compare with MCMC on a subset, use Pareto-k̂ diagnostics, or
  posterior error bounds.
- Fit fast first: subsample the data, use fewer draws, or start with a
  simpler model, then scale up.
- *ProbPipe:* `condition_on` with the inference registry (`method=`),
  `probpipe.diagnostics` (`add_mcmc_diagnostics`, `add_rhat`, `add_ess`),
  `posterior.diagnostics`, `posterior.arviz_data`.

### F. Check the model against the data

- **Posterior predictive checks:** compare replicated data with the observed
  data, choosing test statistics that target the scientific question and the
  model's weakest assumptions, not just the mean. Use graphical checks.
- **Predictive performance:** PSIS-LOO cross-validation; check the Pareto-k̂
  diagnostics; use k-fold CV when LOO is unreliable.
- *ProbPipe:* `predictive_check`, `add_ppc`, `add_loo`.

### G. Compare, expand, and combine models

- Compare models on predictive performance (LOO differences with standard
  errors), not only on fit. Prefer stacking to selecting a single model when
  several are comparably good.
- When models may be misspecified, check whether conclusions are stable:
  bagged posteriors (BayesBag) give more reproducible model selection and
  uncertainty that reflects misspecification.
- *ProbPipe:* broadcasting `condition_on` over models or datasets,
  `BootstrapReplicateDistribution`, the bagged-posteriors user guide.

### H. Understand the fitted model

- Propagate the full posterior into the estimands and report their
  uncertainty, keeping joint draws together.
- **Sensitivity analysis:** how do conclusions change under alternative
  priors, likelihoods, or data exclusions? Report the ones that matter.
- *ProbPipe:* `@function` propagation with `select(...)`, broadcasting over
  prior or hyperparameter settings, `mean`, `quantile`, `expectation`.

### I. Simulators and expensive models

- For simulators without tractable likelihoods: simulation-based inference
  (ABC, neural likelihood/posterior/ratio estimation), always validated with
  SBC-style coverage checks.
- For expensive models: emulators (Gaussian processes) with their own
  uncertainty propagated, and an accounting for model discrepancy.
- *ProbPipe:* `learn_amortized_posterior`, `learn_amortized_likelihood`,
  `learn_amortized_ratio`, `GaussianRandomFunction`, the random functions
  and emulators user guide.

### J. Reproducibility and communication

- Fix random seeds, record software versions, keep every model in the
  sequence, and write down each decision and its reason (the design doc's
  changelog serves this purpose).
- *ProbPipe:* provenance tracking, workflow RNG scopes.

---

## Part 2 — Annotated bibliography

When suggesting reading, match it to the user's profile: textbooks for those
new to Bayesian modeling, papers for working and expert users.

### The workflow as a whole

- **Gelman, Vehtari, McElreath, et al., *Bayesian Workflow*** (CRC Press,
  2026). The book-length treatment of iterative model building, checking,
  computational troubleshooting, and simulated-data experiments, with worked
  examples. The primary reference for Part 1.
- **Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry,
  Bürkner & Modrák, "Bayesian Workflow"** (2020), arXiv:2011.01808. The
  free article that preceded the book; the best single thing to read first.
  https://arxiv.org/abs/2011.01808
- **Betancourt, "Towards a Principled Bayesian Workflow"** (case study,
  2018–2020). Prior predictive checks, computational calibration, and
  posterior checks organized around explicit questions about the model.
  https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html
- **Schad, Betancourt & Vasishth, "Toward a principled Bayesian workflow in
  cognitive science"** (Psychological Methods, 2021), arXiv:1904.12765. The
  same workflow worked through on an applied example; good for scientists.

### Textbooks

- **McElreath, *Statistical Rethinking*** (2nd ed., CRC, 2020). Best
  introduction for scientists new to Bayesian modeling; strong on causal
  thinking and building models from scientific mechanisms.
- **Gelman, Hill & Vehtari, *Regression and Other Stories*** (Cambridge,
  2020). Applied regression with a workflow emphasis. Book site with code and
  examples: https://avehtari.github.io/ROS-Examples/
- **Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data Analysis***
  (3rd ed., CRC, 2013). The standard graduate reference; free PDF from the
  authors. https://sites.stat.columbia.edu/gelman/book/

### Priors

- **Gelman, Simpson & Betancourt, "The prior can often only be understood in
  the context of the likelihood"** (Entropy, 2017), arXiv:1708.07487.
- **Stan prior choice recommendations** (Stan wiki). Practical defaults by
  model type. https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations

### Validating computation

- **Talts, Betancourt, Simpson, Vehtari & Gelman, "Validating Bayesian
  inference algorithms with simulation-based calibration"** (2018),
  arXiv:1804.06788.
- **Modrák et al., "Simulation-based calibration checking for Bayesian
  computation: The choice of test quantities shapes sensitivity"** (Bayesian
  Analysis, 2023), arXiv:2211.02383. Data-dependent test quantities.
- **Vehtari, Gelman, Simpson, Carpenter & Bürkner, "Rank-normalization,
  folding, and localization: An improved R-hat"** (Bayesian Analysis, 2021),
  arXiv:1903.08008. Current R-hat and ESS.
- **Betancourt, "A conceptual introduction to Hamiltonian Monte Carlo"**
  (2017), arXiv:1701.02434. Why divergences matter and what they mean.
- **Yao, Vehtari, Simpson & Gelman, "Yes, but did it work?: Evaluating
  variational inference"** (ICML, 2018), arXiv:1802.02538. PSIS diagnostics
  for VI.
- **Huggins, Kasprzak, Campbell & Broderick, "Validated variational inference
  via practical posterior error bounds"** (AISTATS, 2020), arXiv:1910.04102.

### Model checking, comparison, and robustness

- **Gabry, Simpson, Vehtari, Betancourt & Gelman, "Visualization in Bayesian
  workflow"** (JRSS A, 2019), arXiv:1709.01449. Graphical prior and posterior
  predictive checks.
- **Vehtari, Gelman & Gabry, "Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC"** (Statistics and Computing, 2017),
  arXiv:1507.04544.
- **Vehtari, Simpson, Gelman, Yao & Gabry, "Pareto smoothed importance
  sampling"** (JMLR, 2024), arXiv:1507.02646.
- **Yao, Vehtari, Simpson & Gelman, "Using stacking to average Bayesian
  predictive distributions"** (Bayesian Analysis, 2018), arXiv:1704.02030.
- **Huggins & Miller, "Reproducible model selection using bagged
  posteriors"** (Bayesian Analysis, 2023), arXiv:2007.14845. BayesBag, the idea
  behind ProbPipe's bagged-posteriors user guide.

### Simulators and emulators

- **Cranmer, Brehmer & Louppe, "The frontier of simulation-based inference"**
  (PNAS, 2020), arXiv:1911.01429.
- **Kennedy & O'Hagan, "Bayesian calibration of computer models"** (JRSS B,
  2001). Emulators and model discrepancy.
