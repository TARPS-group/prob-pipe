# Workflow guidance and resources

The `design-pipeline` skill uses this file to make each pipeline statistically
and computationally sound. Part 1 is an operational checklist, distilled
mainly from *Bayesian Workflow* by Gelman, Vehtari, McElreath, and coauthors,
which maps each step to the ProbPipe tools that implement it. Part 2 is an
annotated bibliography, which says what each resource is good for and when to
point a user to it.

Apply Part 1 from your own knowledge of these sources, which needs no network
access. When you have web access and need detail beyond what you know, such as
a specific diagnostic threshold, consult the source. When a link is dead,
search for the title. Cite only the resources of this list and the ones you
have opened in this session.

---

## Part 1 — Workflow checklist

A pipeline design is sound when it addresses each stage of this checklist, or
skips the stage for a reason recorded in the design document. The workflow is
iterative, so a problem found late often sends you back to the model. Plan for
that return.

### A. Before writing a model

- **Start from the question.** Tie each part of the model to the estimands. A
  model that cannot express the quantity of interest is the wrong model,
  however well it fits.
- **Understand the measurement process.** Learn how the data was collected.
  Censoring, truncation, selection, measurement error, and missingness each
  belong in the model or in an explicit assumption.
- **Start simple, then expand.** Begin with the simplest model that addresses
  the question, then add structure, such as hierarchy, covariates, or
  nonlinearity, one step at a time, and check each step. Plan the sequence of
  models up front. Keep the earlier models as baselines for comparison and
  debugging.

### B. Priors

- Choose priors on interpretable scales, with domain knowledge where it
  exists. Weakly informative priors, which rule out implausible magnitudes,
  are almost always better than flat priors.
- A prior has meaning only in the context of the likelihood, so judge it
  through its **prior predictive distribution**.
- Rescale or standardize the parameters so that the priors and the sampler
  both work on scales of order one.
- *ProbPipe:* distribution families merged with `&`, a
  `conditional_distribution` for a prior that depends on other parameters,
  and constraints and bijectors (`positive`, `bijector_for`).

### C. Prior predictive checks

- Simulate data from the prior, and ask the domain expert whether it looks
  possible. Implausible simulations, such as negative counts or astronomically
  large values, mean that the priors or the model structure need revision.
- *ProbPipe:* `predictive_check` with the prior as its law; `sample` of the
  joint model.

### D. Validate the computation on simulated data

- **Parameter recovery:** simulate data from known parameter values, fit the
  model, and check that the posterior covers the values. The check is cheap,
  and it catches many bugs.
- **Simulation-based calibration (SBC):** repeat the recovery over many prior
  draws, and check that the ranks are uniform. Data-dependent test quantities,
  such as the log likelihood, make the check more sensitive. SBC is worth its
  cost for a new or custom model and for any approximate inference method, and
  it can be too expensive for a slow model.
- *ProbPipe:* `sample` of the joint model to simulate data, `condition_on`
  on the simulated data, and `probpipe.validation`
  (`simulation_based_calibration`, `interval_coverage`).

### E. Fit and diagnose the computation

- Use the default algorithm of `condition_on`, which is NUTS, unless a reason
  calls for another. Run several chains.
- Check the rank-normalized R-hat against a target below 1.01, the bulk and
  tail ESS, the divergences, and the tree depth. Interpret a posterior only
  after its diagnostics pass.
- **Folk theorem:** computational trouble usually signals a model problem.
  Before you tune the sampler, look for weak identifiability, a poor
  parameterization, poorly scaled parameters, or a model that disagrees with
  the data. A non-centered parameterization often helps a hierarchical model.
- Check the accuracy of an approximate method, such as VI or amortized or
  simulation-based inference, by a comparison with MCMC on a subset, by
  Pareto-k̂ diagnostics, or by posterior error bounds.
- Fit fast first: subsample the data, use fewer draws, or start with a simpler
  model, then scale up.
- *ProbPipe:* `condition_on` with the inference registry
  (`condition_on.using(method=...)`), `probpipe.diagnostics`
  (`add_mcmc_diagnostics`, `add_rhat`, `add_ess`), `posterior.diagnostics`,
  and the ArviZ data `posterior.annotations["arviz"]`.

### F. Check the model against the data

- **Posterior predictive checks:** compare replicated data with the observed
  data. Choose test statistics that target the scientific question and the
  model's weakest assumptions, beyond the mean, and use graphical checks.
- **Predictive performance:** estimate it by PSIS-LOO cross-validation, and
  check its Pareto-k̂ diagnostics. Use k-fold cross-validation when LOO is
  unreliable.
- *ProbPipe:* `predictive_check`, `add_ppc`, `add_loo`.

### G. Compare, expand, and combine models

- Compare models on predictive performance, by LOO differences with their
  standard errors, as well as on fit. When several models are comparably good,
  prefer stacking to the selection of one model.
- When the models may be misspecified, check whether the conclusions are
  stable. Bagged posteriors, or BayesBag, give a more reproducible model
  selection and an uncertainty that reflects the misspecification.
- *ProbPipe:* a sweep of `condition_on` over a `DistributionBatch` of models
  or a `RecordBatch` of datasets, which returns a `DistributionBatch` of
  posteriors; `BootstrapReplicateDistribution` for bootstrap replicates of the
  data (`docs/user_guide/08_conversion_and_updating.ipynb`).

### H. Understand the fitted model

- Propagate the full posterior into the estimands with joint draws of the
  parameters, and report the uncertainty of the estimands.
- **Sensitivity analysis:** measure how the conclusions change under
  alternative priors, likelihoods, or data exclusions, and report the changes
  that matter.
- *ProbPipe:* a `@function` called on coupled marginals of the posterior,
  such as `f(posterior["a"], posterior["b"])`, which the lift draws jointly; a
  sweep of `condition_on` over a `DistributionBatch` of models under
  alternative priors; `mean`, `quantile`, `expectation`.

### I. Simulators and expensive models

- For a simulator without a tractable likelihood, use simulation-based
  inference, such as ABC or a neural estimate of the likelihood, the
  posterior, or the likelihood ratio. Validate it with SBC-style coverage
  checks.
- For an expensive model, use an emulator, such as a Gaussian process,
  propagate the emulator's own uncertainty, and account for model discrepancy.
- *ProbPipe:* `learn_amortized_posterior`, `learn_amortized_likelihood`,
  `learn_amortized_ratio`, approximate Bayesian computation through
  `condition_on` (`docs/tutorials/06_simulating.ipynb`), and
  `GaussianRandomFunction` (`docs/api/random_functions.md`).

### J. Reproducibility and communication

- Fix the random seeds, record the software versions, keep every model of the
  sequence, and record each decision with its reason in the changelog of the
  design document.
- *ProbPipe:* the provenance of each result (`provenance_ancestors`), and
  workflow scopes for random draws (`workflow_run`, `replay_run`).

---

## Part 2 — Annotated bibliography

Match each suggested reading to the user's profile, with textbooks for a user
new to Bayesian modeling and papers for a working or expert user.

### The workflow as a whole

- **Gelman, Vehtari, McElreath, et al., *Bayesian Workflow*** (CRC Press,
  2026). The book treats iterative model building, checking, computational
  troubleshooting, and simulated-data experiments at length, with worked
  examples. It is the primary reference for Part 1.
- **Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry,
  Bürkner & Modrák, "Bayesian Workflow"** (2020), arXiv:2011.01808. This free
  article preceded the book, and it is the best single thing to read first.
  https://arxiv.org/abs/2011.01808
- **Betancourt, "Towards a Principled Bayesian Workflow"** (case study,
  2018–2020). It organizes prior predictive checks, computational calibration,
  and posterior checks around explicit questions about the model.
  https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html
- **Schad, Betancourt & Vasishth, "Toward a principled Bayesian workflow in
  cognitive science"** (Psychological Methods, 2021), arXiv:1904.12765. It
  works through the same workflow on an applied example, which suits
  scientists.

### Textbooks

- **McElreath, *Statistical Rethinking*** (2nd ed., CRC, 2020). It is the best
  introduction for scientists new to Bayesian modeling, and it is strong on
  causal thinking and on building models from scientific mechanisms.
- **Gelman, Hill & Vehtari, *Regression and Other Stories*** (Cambridge,
  2020). It teaches applied regression with an emphasis on the workflow. The
  book's site has code and examples: https://avehtari.github.io/ROS-Examples/
- **Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, *Bayesian Data Analysis***
  (3rd ed., CRC, 2013). It is the standard graduate reference, with a free PDF
  from the authors. https://sites.stat.columbia.edu/gelman/book/

### Priors

- **Gelman, Simpson & Betancourt, "The prior can often only be understood in
  the context of the likelihood"** (Entropy, 2017), arXiv:1708.07487.
- **Stan prior choice recommendations** (Stan wiki). It gives practical
  defaults by model type.
  https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations

### Validating computation

- **Talts, Betancourt, Simpson, Vehtari & Gelman, "Validating Bayesian
  inference algorithms with simulation-based calibration"** (2018),
  arXiv:1804.06788.
- **Modrák et al., "Simulation-based calibration checking for Bayesian
  computation: The choice of test quantities shapes sensitivity"** (Bayesian
  Analysis, 2023), arXiv:2211.02383. It introduces data-dependent test
  quantities.
- **Vehtari, Gelman, Simpson, Carpenter & Bürkner, "Rank-normalization,
  folding, and localization: An improved R-hat"** (Bayesian Analysis, 2021),
  arXiv:1903.08008. It defines the current R-hat and ESS.
- **Betancourt, "A conceptual introduction to Hamiltonian Monte Carlo"**
  (2017), arXiv:1701.02434. It explains why divergences matter and what they
  mean.
- **Yao, Vehtari, Simpson & Gelman, "Yes, but did it work?: Evaluating
  variational inference"** (ICML, 2018), arXiv:1802.02538. It gives PSIS
  diagnostics for VI.
- **Huggins, Kasprzak, Campbell & Broderick, "Validated variational inference
  via practical posterior error bounds"** (AISTATS, 2020), arXiv:1910.04102.

### Model checking, comparison, and robustness

- **Gabry, Simpson, Vehtari, Betancourt & Gelman, "Visualization in Bayesian
  workflow"** (JRSS A, 2019), arXiv:1709.01449. It covers graphical prior and
  posterior predictive checks.
- **Vehtari, Gelman & Gabry, "Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC"** (Statistics and Computing, 2017),
  arXiv:1507.04544.
- **Vehtari, Simpson, Gelman, Yao & Gabry, "Pareto smoothed importance
  sampling"** (JMLR, 2024), arXiv:1507.02646.
- **Yao, Vehtari, Simpson & Gelman, "Using stacking to average Bayesian
  predictive distributions"** (Bayesian Analysis, 2018), arXiv:1704.02030.
- **Huggins & Miller, "Reproducible model selection using bagged
  posteriors"** (Bayesian Analysis, 2023), arXiv:2007.14845. It introduces
  BayesBag.

### Simulators and emulators

- **Cranmer, Brehmer & Louppe, "The frontier of simulation-based inference"**
  (PNAS, 2020), arXiv:1911.01429.
- **Kennedy & O'Hagan, "Bayesian calibration of computer models"** (JRSS B,
  2001). It covers emulators and model discrepancy.
