# Workflow guidance and resources

The design-pipeline assistant uses this file to make each pipeline
statistically and computationally sound. Part 1 is an operational checklist,
distilled mainly from *Bayesian Workflow* by Gelman, Vehtari, McElreath, and
coauthors, and [CAPABILITIES.md](CAPABILITIES.md) names the ProbPipe tools for
each of its stages. Part 2 is an
annotated bibliography, which says what each resource is good for and when to
point a user to it.

Apply Part 1 from your own knowledge of these sources, which needs no network
access. Part 1 states the thresholds of the diagnostics. When you have web
access and need detail beyond what you know, such as the derivation of a
diagnostic, consult the source. When a link is dead,
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
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage B.

### C. Prior predictive checks

- Simulate data from the prior, and ask the domain expert whether it looks
  possible. Implausible simulations, such as negative counts or astronomically
  large values, mean that the priors or the model structure need revision.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage C.

### D. Validate the computation on realistic simulated data

Validate the computation on simulated data with a known ground truth before
you trust a fit to the real data. Every design includes this validation, and
the real-data fit is interpreted only after it passes.

1. **Simulate at the real design.** Simulate datasets from the model with the
   actual covariates, sample sizes, grouping structure, and missingness
   pattern of the real data. Take the parameter values from several draws of
   the prior, and add values that the user judges realistic.
2. **Fit with the planned method.** Fit each simulated dataset with the
   inference method and the settings planned for the real data.
3. **Check recovery and coverage of the estimands.** Across the simulated
   datasets, the posterior intervals of each estimand cover its true value at
   about their nominal rate. For each parameter, compute the posterior z-score,
   which is the posterior mean minus the true value over the posterior
   standard deviation, and the posterior contraction, which is one minus the
   ratio of the posterior variance to the prior variance. A large z-score marks
   bias, and a contraction near zero marks a parameter that the data does not
   inform. Betancourt's case study "Towards a Principled Bayesian Workflow"
   describes both.
4. **Run SBC when the model or the method calls for it.** Run
   simulation-based calibration (Talts et al. 2018) for a custom likelihood and
   for any approximate method. Run it always for an amortized method, because
   its posterior at each simulated dataset costs one network evaluation. Scale
   the number of replications to the budget, since even 100 to 200
   replications reveal gross errors. Data-dependent test quantities, such as
   the log likelihood, make the check more sensitive (Modrák et al. 2023).
   When you skip SBC, record the reason in the design document.

- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage D.

### E. Fit and diagnose the computation

- **Choose the method by the model.** Prefer NUTS when its conditions hold:
  continuous parameters and a differentiable log density. Marginalize discrete
  latent variables, such as mixture labels, change points, or the states of a
  hidden Markov model, out of the likelihood, so that NUTS applies. When a
  discrete variable cannot be marginalized, say so, and choose a method that
  samples it.
- **Diagnose each MCMC run** by these criteria, and interpret a posterior only
  after it passes them. Vehtari et al. (2021) define the R-hat and the ESS.
  1. Chains: run at least four, from dispersed initial values, so that
     separate modes show up as disagreeing chains.
  2. R-hat: the rank-normalized split-R-hat is below 1.01 for every parameter
     and every estimand. For many short chains, use the nested R-hat instead.
     TODO(source): Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari &
     Gelman, "Nested R-hat: assessing the convergence of Markov chain Monte
     Carlo when running many short chains" (Bayesian Analysis, 2024).
  3. ESS: the bulk ESS and the tail ESS exceed 100 per chain.
  4. MCSE: the Monte Carlo standard error of each estimand is small against
     the precision that the question needs.
  5. Divergences: each divergent transition after warmup needs investigation,
     because it marks a region that the sampler cannot explore reliably.
  6. E-BFMI: a value below about 0.3 signals that the momentum resampling
     explores the energy distribution poorly.
  7. Tree depth: trajectories that stop at the maximum tree depth make a run
     inefficient, and a higher maximum or a reparameterization reduces the
     cost. The posterior stays valid when the other criteria pass.
- **Folk theorem:** computational trouble usually signals a model problem.
  Before you tune the sampler, look for weak identifiability, a poor
  parameterization, poorly scaled parameters, or a model that disagrees with
  the data. A non-centered parameterization often helps a hierarchical model.
- **Validate an approximate method before you report its results.** An
  approximate method, such as VI, amortized inference, or ABC, is validated in
  a step that the user sees. Mean-field VI typically underestimates the
  posterior variance. Compare the approximation with an MCMC fit to the same
  data, and use a smaller simulated problem of the same structure when the
  full MCMC fit is too slow. A posterior conditioned on a subset of the data
  is a different posterior, so it serves as no reference. Use the Pareto-k̂
  diagnostic of PSIS (Yao et al. 2018) or posterior error bounds (Huggins et
  al. 2020) when the likelihood is tractable, and the checks of stage D
  otherwise.
- Fit fast first: subsample the data, use fewer draws, or start with a simpler
  model, then scale up.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage E.

### F. Check the model against the data

- **Posterior predictive checks:** compare replicated data with the observed
  data. Choose test statistics that target the scientific question and the
  model's weakest assumptions, beyond the mean, and use graphical checks.
- **Predictive performance:** estimate it by the cross-validation scheme that
  matches the prediction task of the estimands:
  1. new observations of exchangeable units: PSIS-LOO (Vehtari, Gelman &
     Gabry 2017);
  2. new groups of a hierarchical model: leave-one-group-out;
  3. future values of a time series: leave-future-out or rolling-origin
     cross-validation. TODO(source): Bürkner, Gabry & Vehtari, "Approximate
     leave-future-out cross-validation for Bayesian time series models"
     (Journal of Statistical Computation and Simulation, 2020);
  4. new locations of spatial data: cross-validation that leaves out blocks
     of nearby locations. TODO(source): Roberts et al., "Cross-validation
     strategies for data with temporal, spatial, hierarchical, or phylogenetic
     structure" (Ecography, 2017).
- **Pareto k̂:** check the k̂ of each left-out unit. A k̂ above 0.7, or above
  the threshold min(1 − 1/log₁₀ S, 0.7) for S posterior draws, makes the PSIS
  estimate for that unit unreliable (Vehtari et al. 2024). For those units,
  use moment matching, refit the model without them, or use k-fold
  cross-validation.
- **Calibration:** check the leave-one-out predictive distributions with
  LOO-PIT, whose values are uniform for a calibrated model (Gabry et al.
  2019).
- **Valid likelihoods only:** LOO, WAIC, and model comparison use the
  normalized pointwise log likelihood. A likelihood known only up to a
  constant that does not depend on the parameters, such as a learned
  likelihood-to-evidence ratio, therefore gives no valid value for them.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage F.

### G. Compare, expand, and combine models

- Compare models on predictive performance, by the differences of their
  expected log predictive density (elpd) with standard errors, as well as on
  fit. The standard error of a difference is unreliable in three cases: the
  difference is below about 4, the number of observations is small, or the
  models are clearly misspecified. TODO(source): Sivula, Magnusson, Matamoros
  & Vehtari, "Uncertainty in Bayesian leave-one-out cross-validation based
  model comparison" (arXiv:2008.10296).
- Stacking weights the predictive distributions of several models (Yao et al.
  2018). When several models predict comparably well, prefer stacking for
  prediction to the selection of one model. Stacking combines predictive
  distributions only, because a parameter can mean different things in
  different models.
- When the models may be misspecified, check whether the conclusions are
  stable. Bagged posteriors, or BayesBag, give a more reproducible model
  selection and an uncertainty that reflects the misspecification. The
  bootstrap of BayesBag assumes exchangeable units, so resample whole groups
  or blocks when the data are grouped or dependent.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage G.

### H. Understand the fitted model

- Propagate the full posterior into the estimands with joint draws of the
  parameters, and report the uncertainty of the estimands.
- **Sensitivity analysis:** measure how the conclusions change under
  alternative priors, likelihoods, or data exclusions, and report the changes
  that matter.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage H.

### I. Simulators and expensive models

- For a simulator without a tractable likelihood, use simulation-based
  inference, such as ABC or a neural estimate of the likelihood, the
  posterior, or the likelihood ratio. Validate it by stage D, which requires
  SBC for an amortized method.
- For an expensive model, use an emulator, such as a Gaussian process,
  propagate the emulator's own uncertainty, and account for model discrepancy.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage I.

### J. Reproducibility and communication

- Fix the random seeds, record the software versions, keep every model of the
  sequence, and record each decision with its reason in the changelog of the
  design document.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage J.

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
