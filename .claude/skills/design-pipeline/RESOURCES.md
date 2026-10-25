# Workflow guidance and resources

The design-pipeline assistant uses this file to make each pipeline
statistically and computationally sound. Part 1 is an operational checklist,
distilled mainly from *Bayesian Workflow* (Gelman et al. 2026), and [CAPABILITIES.md](CAPABILITIES.md) names the ProbPipe tools for
each of its stages. Part 2 is an
annotated bibliography, which says what each resource is good for and when to
point a user to it.

Apply Part 1 from your own knowledge of these sources, which needs no network
access. Part 1 states the thresholds of the diagnostics. When you have web
access and need detail beyond what you know, such as the derivation of a
diagnostic, consult the source. When a link is dead,
search for the title. Cite only the resources of Part 2 and the ones you
have opened in this session.

---

## Part 1 — Workflow checklist

A pipeline design is sound when it addresses each stage of this checklist, or
skips the stage for a reason recorded in the design document. A stage that the
quick start mode of the [ground rules](INSTRUCTIONS.md#ground-rules) defers is
recorded as `deferred: quick start`, and it stays among the next steps. The
workflow is iterative, so a problem found late often sends you back to the
model. Plan for that return.

### A. Before writing a model

- **Start from the question.** Tie each part of the model to the estimands. A
  model that cannot express the quantity of interest is the wrong model,
  however well it fits.
- **Understand the measurement process.** Learn how the data was collected.
  Censoring, truncation, selection, measurement error, and missingness each
  belong in the model or in an explicit assumption.
- **Start simple, then expand.** The default is to begin with the simplest
  model that addresses the question, then add structure, such as hierarchy,
  covariates, or nonlinearity, one step at a time, and check each step. A user
  who brings an established model can start from it instead, with a simpler
  model fitted beside it as a baseline. Plan the sequence of models up front.
  Keep the earlier models as baselines for comparison and debugging.

### B. Priors

- Choose priors on interpretable scales, with domain knowledge where it
  exists. Weakly informative priors, which rule out implausible magnitudes,
  are almost always better than flat priors.
- A prior has meaning only in the context of the likelihood, so judge it
  through its **prior predictive distribution** (Gelman et al. 2017).
- Rescale or standardize the parameters so that the priors and the sampler
  both work on scales of order one.
- **Default priors when the user cannot give numbers.** Standardize each
  covariate, and choose weakly informative priors, such as normal priors of
  scale one on the scale of the linear predictor. Then judge them by the
  prior predictive check of stage C. A simulated value is implausible when it
  lies beyond a bound that the user can name, such as a physical limit or the
  largest value on record, or else beyond ten times the range of the observed
  outcome. Tighten the scales of the priors until at most 10% of the
  simulated datasets hold an implausible value, which is the default share.
  Record in section 3 of the design document that this check set the priors,
  with the bound and the share.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage B.

### C. Prior predictive checks

- Simulate data from the prior, and ask the domain expert whether it looks
  possible. Implausible simulations, such as negative counts or astronomically
  large values, mean that the priors or the model structure need revision.
- Show the expert one or two summaries in the units of the data, so that they
  judge the simulations by what they know. An example is the share of
  simulated datasets whose largest value exceeds the largest value they would
  believe, beside the range of a typical value.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage C.

### D. Validate the computation on realistic simulated data

Validate the computation on simulated data with a known ground truth before
you trust a fit to the real data. Every design includes this validation, and
the real-data fit is interpreted only after the validation's grade, by step 6,
is *passed* or *passed with notes*.

1. **Simulate at the real design.** Simulate datasets from the model with the
   actual covariates, sample sizes, grouping structure, and missingness
   pattern of the real data. Take the parameter values from several draws of
   the prior, and add one or two values that the user judges realistic, with
   several datasets at each, 20 by default, because step 6 grades recovery by
   their mean. When the user cannot name realistic values, validate on the
   prior draws alone, and say so to the user and in the design document.
2. **Fit with the planned method.** Fit each simulated dataset with the
   inference method and the settings planned for the real data, and record
   the settings.
3. **Diagnose each fit.** Record the divergences and the R-hat of each fit by
   the criteria of stage E. Coverage averages over the fits, so it can look
   nominal while many fits are broken, and the grade of step 6 therefore
   judges each fit.
4. **Check coverage over the prior draws and recovery at the chosen values.**
   Bayesian intervals are calibrated on average over the prior, so the
   coverage check uses the datasets simulated from prior draws. Across them,
   the posterior intervals of each estimand cover its true value at about
   their nominal rate. With n datasets, the coverage at the level p has the
   Monte Carlo standard error √(p(1 − p)/n). At each value θ that the user
   chose, check recovery instead, by three quantities of each parameter in
   each fit:
   1. the posterior z-score, which is the posterior mean minus the true value
      over the posterior standard deviation;
   2. the posterior contraction, which is one minus the ratio of the posterior
      variance to the prior variance, and whose value near zero marks a
      parameter that the data does not inform;
   3. the pull of the prior, which is Σ_post Σ_prior⁻¹ (μ_prior − θ) over the
      posterior standard deviation, where Σ_post and Σ_prior are the
      posterior and the prior covariance and μ_prior is the prior mean.

   Betancourt (2020) describes the first two. An exact posterior at a fixed
   value shrinks toward the prior mean. In a linear model with normal noise
   and a normal prior, the z-scores of a parameter at θ therefore have the
   pull as their mean and a standard deviation of at most one, and the same
   holds approximately for a posterior close to normal. The *centered
   z-score*, which is the z-score minus the pull, has mean zero for an exact
   method, so a mean centered z-score far from zero marks bias. Over the
   prior draws, the z-score of an exact method has mean zero and variance one,
   so it is its own centered z-score there. Also check that a wide interval,
   such as the 95% interval, holds the true value.
5. **Run SBC when the model or the method calls for it.** Run
   simulation-based calibration (Talts et al. 2018) for a custom likelihood and
   for any approximate method. Run it always for an amortized method, because
   its posterior at each simulated dataset costs one network evaluation. Scale
   the number of replications to the budget, since even 100 to 200
   replications reveal gross errors. Data-dependent test quantities, such as
   the log likelihood, make the check more sensitive (Modrák et al. 2025).
   Read the ranks from an ECDF plot with simultaneous confidence bands
   (Säilynoja et al. 2022). The replications of SBC can use cheaper settings
   than the real fit, such as fewer chains or shorter runs, because the fits
   of steps 2 and 3 use the real settings. Record the settings of SBC beside
   its result. When you skip SBC, record the reason in the design document.
6. **Grade the validation.** Grade the validation of each model after the
   remedies of the bullet "Remedy divergences in this order" of stage E, by
   the criteria below. Three of them grade many results at once, so their
   bounds widen with the number of results M that each grades. The bound of
   *passed* is b₂(M), the normal quantile whose two-sided tail is 0.0455/M,
   and the bound of *failed* is b₃(M), whose two-sided tail is 0.0027/M.
   `scipy.stats.norm.isf(0.0455 / (2 * M))` computes b₂(M). For one result
   the bounds are 2 and 3, and for eight results they are 2.77 and 3.58. When
   each result is close to normal, an exact method crosses b₂(M) on some
   result with probability at most 0.0455, and b₃(M) with probability at most
   0.0027, whatever the correlation between the results. A single bound of ±3 on each of 30
   z-scores, in contrast, fails an exact method in about one run in 13. The
   bias and the single misses use the n datasets of each chosen value, or the
   prior draws as one group of n datasets when step 1 has no chosen value.
   The three criteria are these:
   - *coverage:* the distance of the coverage of each estimand at each level
     from the level, in Monte Carlo standard errors, with M the number of
     estimands times the number of levels;
   - *bias:* the mean centered z-score of each parameter over the n datasets
     of each group, in standard errors of 1/√n, with M the number of
     parameters times the number of groups;
   - *single misses:* the largest absolute centered z-score over every
     parameter and every dataset of the groups, with M the number K of those
     z-scores.

   These bounds, and the 20 datasets at each chosen value, are defaults. A
   project that sets others records them as a decision with its reason. Two
   named options are *strict single z-scores*, which grades each z-score by
   the bounds 2 and 3, and *notes fail*, which counts every note as a
   failure.
   1. *Passed:* every fit passes the criteria of stage E with no divergence,
      and the coverage, the bias, and the single misses each lie within
      b₂(M).
   2. *Passed with notes:* no criterion of *failed* holds, and some result
      misses *passed*. Each note names its result. Examples are these:
      - a few fits, at most 5% of them and at least one, whose divergences or
        whose diagnostics just past their thresholds, such as an R-hat up to
        1.05 or a bulk ESS of at least 50 per chain, disappear when those
        fits are rerun with a remedy, which the fit to the real data then
        uses;
      - a coverage between b₂(M) and b₃(M) standard errors below its level,
        or more than b₂(M) above it, which marks intervals wider than they
        need to be;
      - a bias or a single miss between b₂(M) and b₃(M);
      - a contraction below 0.1, which marks an estimand that the data barely
        inform;
      - a mean pull beyond ±1, which marks a chosen value whose estimate the
        prior moves by more than one posterior standard deviation.
   3. *Failed:* divergences or failed diagnostics that the remedies leave, or
      that occur in more fits; a coverage more than b₃(M) standard errors
      below its level; a bias or a single miss beyond b₃(M); or an SBC rank
      ECDF that leaves its simultaneous band.

   Beside the grade, list each centered z-score beyond ±3 with its parameter
   and its dataset, and the number that chance gives an exact method, which
   is 0.0027 K, so that no mean hides a large miss.

   Report each grade with one sentence, in the user's terms, on what it means
   for the answers:
   - *passed:* "The method recovered known answers from data like yours, so
     its intervals can be trusted.";
   - *passed with notes:* the same sentence, with the caveat of each note;
   - *failed:* "The method did not reliably recover known answers from data
     like yours, so its estimates cannot be trusted yet."

   In the quick start mode, the validation simulates few datasets, so its
   coverage has a wide Monte Carlo band. Its grade then rests on the
   diagnostics of each fit, the bias, and the single misses, and its coverage
   is reported without a grade.

   Estimate the cost of the validation before you propose it, as the number
   of fits times the time of one fit at the planned settings, scaled from the
   most recent smoke run. When the cost exceeds the user's budget, propose a
   pilot, which fits fewer datasets. Tell the user plainly that a pilot
   checks the diagnostics of each fit and gross bias, and that it cannot
   establish coverage. Record that limit in the status of each answer of the
   pilot's model, in section 0 of the design document.

- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage D.

### E. Fit and diagnose the computation

- **Choose the method by the model.** The default is NUTS when its conditions
  hold: continuous parameters and a differentiable log density. The
  alternatives fit other models:
  1. a density without a gradient: a gradient-free method, such as
     random-walk Metropolis;
  2. a fit too slow for MCMC: an approximate method, validated as the bullets
     below require;
  3. a model without a tractable likelihood: the methods of stage I.

  Discrete latent variables, such as mixture labels, change points, or the
  states of a hidden Markov model, can be marginalized out of the likelihood
  so that NUTS applies (Stan User's Guide; Zucchini et al. 2016). When a discrete variable cannot be marginalized, say
  so, and choose a method that samples it.
- **Diagnose each MCMC run** by these criteria, and interpret a posterior only
  after it passes them. Vehtari et al. (2021) define the R-hat and the ESS.
  The thresholds are the conventional defaults. A project that sets another
  threshold records it as a decision with its reason.
  1. Chains: run at least four, from dispersed initial values, so that
     separate modes show up as disagreeing chains. The replications of SBC in
     stage D can run fewer.
  2. R-hat: the rank-normalized split-R-hat is below 1.01 for every parameter
     and every estimand. For many short chains, use the nested R-hat instead
     (Margossian et al. 2025).
  3. ESS: the bulk ESS and the tail ESS exceed 100 per chain.
  4. MCSE: the Monte Carlo standard error of each estimand is small against
     the precision that the question needs.
  5. Divergences: each divergent transition after warmup needs investigation,
     because it marks a region that the sampler cannot explore reliably
     (Betancourt 2017).
  6. E-BFMI: a value below about 0.3 signals that the momentum resampling
     explores the energy distribution poorly.
  7. Tree depth: trajectories that stop at the maximum tree depth make a run
     inefficient, and a higher maximum or a reparameterization reduces the
     cost. The posterior stays valid when the other criteria pass.
  8. Values: each criterion has a value. A criterion reported as not
     computed, such as an R-hat of N/A, counts as failed. A chain that never
     leaves its initial point, as when the log density is NaN there, gives
     such an R-hat, an MCSE of zero, and a divergence at every transition.

  Judge a run by these criteria themselves, because a summary that reports
  the run as passing can leave some of them unchecked.
- **Folk theorem:** computational trouble usually signals a model problem.
  Before you tune the sampler, look for weak identifiability, a poor
  parameterization, poorly scaled parameters, or a model that disagrees with
  the data.
- **Remedy divergences in this order.** When a fit has divergent transitions
  or fails another criterion above, in the validation of stage D or on the
  real data, try these remedies in order, and rerun the fits after each
  one:
  1. Reparameterize the model, by the folk theorem, such as in the
     non-centered form of the next bullet.
  2. Raise the target acceptance rate of the step-size adaptation, such as
     from 0.8 to 0.95. A higher target adapts a smaller step size, so each
     draw takes longer. The row "Condition on the data" of
     [CAPABILITIES.md](CAPABILITIES.md) names the methods that take this
     setting.
  3. Lengthen the warmup, such as from 500 to 2000 iterations, so that the
     adaptation of the step size and the mass matrix settles.

  Each remedy computes the same model another way. When the prior predictive
  check of stage C shows implausible values, a narrower prior is the
  alternative to remedy 1. It changes the model, so the user decides it.
- **Parameterize a hierarchical model by how well the data inform its
  groups.** For a normal hierarchical model with few groups, or with groups
  that the data inform weakly, the default is the non-centered
  parameterization, which writes each group effect as the population mean
  plus the population scale times a standard normal variable. When the data
  inform every group well, the centered parameterization is the alternative,
  because it then samples better (Papaspiliopoulos et al. 2007).
- **Validate an approximate method before you report its results.** An
  approximate method, such as VI, amortized inference, or ABC, is validated in
  a step that the user sees. Mean-field VI typically underestimates the
  posterior variance. Compare the approximation with an MCMC fit to the same
  data, and use a smaller simulated problem of the same structure when the
  full MCMC fit is too slow. A posterior conditioned on a subset of the data
  is a different posterior, so it serves as no reference. When the
  likelihood is tractable and the approximation has a density, also use the
  Pareto k̂ that the bullet "Validate a variational fit" defines, or posterior
  error bounds (Huggins et al. 2020). Otherwise, use the checks of stage D.
- **Validate a variational fit.** A variational method fits its
  approximation by optimization, so its validation adds three checks to the
  comparison with MCMC:
  1. Restarts: run the optimization from several initializations or seeds.
     Restarts that end at different objective values or different estimands
     mark a local optimum or an optimization that has not converged.
  2. Convergence: the trace of the objective over the iterations is flat at
     the end of the optimization.
  3. Pareto k̂: weight each draw of the approximation by the ratio of the
     unnormalized posterior density to the density of the approximation, and
     compute the k̂ of PSIS from these weights (Yao et al. 2018a). A k̂ above
     0.7 means that importance sampling cannot correct the approximation, so
     report neither its intervals nor its tail quantities. With many
     parameters, the k̂ of a useful approximation is often above 0.7 too, so
     a high k̂ there leaves the accuracy of the approximation undecided. Then
     compare the approximation with an MCMC fit to a smaller problem of the
     same structure, by the bullet "Validate an approximate method before you
     report its results".
- **Many observations.** Stochastic-gradient MCMC, such as stochastic-gradient
  Langevin dynamics, estimates the gradient of the log density from a random
  subset of the observations at each step, so each step costs a fraction of a
  full-data gradient (Welling & Teh 2011). With a constant step size and no
  Metropolis–Hastings correction, its draws are biased, and the bias grows
  with the step size (Nemeth & Fearnhead 2021). R-hat and ESS measure how well the chain
  mixes around its own stationary distribution, which differs from the
  posterior by that bias. Use full-batch NUTS when the data fit in memory and
  a full-data gradient is affordable. Otherwise, validate stochastic-gradient
  MCMC in three steps:
  1. Reference: fit full-batch NUTS and the stochastic-gradient method to the
     same subsample of the data, or to a simulated dataset of the same
     structure. Both fits condition on the same data, so the NUTS fit is a
     reference for the other. Use the step size and the ratio of the batch
     size to the number of observations that are planned for the full data.
     Compare the two fits as the bullet "Validate an approximate method before you report
     its results" requires.
  2. Step size: refit at two or three step sizes, and treat a change in the
     estimands as bias.
  3. Kernel Stein discrepancy: compute it for the draws of each full-data
     fit. It needs only the gradient of the log posterior density at the
     draws, and a smaller value means draws closer to the posterior. It has
     no absolute scale, so compare its values across the step sizes
     (Gorham & Mackey 2017).

  The reference of step 1 checks the method on a smaller problem, and the
  error of the method can grow with the number of observations, so the error
  that it finds is a lower bound for the full data. The step sizes of step 2
  and the kernel Stein discrepancy of step 3, both computed on the full data,
  are the evidence about the full-data fit.

  The default way to tune the step size and the batch size is the comparison
  of step 2. Large-sample analyses of stochastic-gradient algorithms are
  among the alternatives, because they predict what a tuning gives before a
  long run. Negrea et al. (2022) show that the average of the iterates at a
  large constant step size is robust to the choice of the tuning parameters,
  and that its covariance approaches the covariance of the sampling
  distribution of the maximum likelihood estimator. Their
  Bernstein–von Mises-like theorem guides the tuning, generalized posteriors
  that are robust to misspecification included. Wang et al. (2026) predict
  the stationary covariance, the covariance of the iterate average, and the
  integrated autocorrelation time of stochastic-gradient Langevin dynamics and
  stochastic-gradient descent, with and without momentum, when the batch
  size is large or the model is misspecified. A tuning chosen by either
  analysis passes the three steps above.
- **Numerical precision:** run MCMC in double precision when the log density
  sums many terms or the posterior is poorly scaled, because single
  precision then loses the accuracy that the gradient and the acceptance step
  need. The row "Condition on the data" of
  [CAPABILITIES.md](CAPABILITIES.md) says how to set the precision.
- Fit fast first: subsample the data, use fewer draws, or start with a simpler
  model, then scale up.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage E.

### F. Check the model against the data

- **Predictive checks:** compare replicated data with the observed data.
  Choose test statistics that target the scientific question and the
  model's weakest assumptions, beyond the mean, and use graphical checks.
  The posterior predictive check is the default. It uses the data both to
  fit the model and to check it, so its p-values are conservative. When the
  user needs calibrated p-values, the alternatives include the LOO-PIT of
  the bullet "Calibration" and a split predictive check, which conditions
  the model on a training part of the data and checks its predictions of the
  held-out part (Li & Huggins 2026). A single split has more power with
  little data or a substantial misspecification. Divided split predictive
  checks have more power as the data grow or when the misspecification is
  subtle.
- **Predictive performance:** estimate it by the cross-validation scheme that
  matches the prediction task of the estimands:
  1. new observations of exchangeable units: PSIS-LOO (Vehtari et al.
     2017);
  2. new groups of a hierarchical model: leave-one-group-out. Leaving out a
     group changes the posterior more than leaving out one observation, so
     PSIS often gives a high k̂ for a group. Check the k̂ of each group, and
     refit the model without each group whose k̂ is high, or use grouped
     k-fold cross-validation;
  3. future values of a time series: leave-future-out or rolling-origin
     cross-validation (Bürkner et al. 2020);
  4. new locations of spatial data: cross-validation that leaves out blocks
     of nearby locations (Roberts et al. 2017).
- **Pareto k̂:** check the k̂ of each left-out unit. A k̂ above the threshold
  min(1 − 1/log₁₀ S, 0.7) for S posterior draws makes the PSIS estimate for
  that unit unreliable (Vehtari et al. 2024). For those units,
  use moment matching, refit the model without them, or use k-fold
  cross-validation.
- **Calibration:** check the leave-one-out predictive distributions with
  LOO-PIT, whose values are uniform for a calibrated model of a continuous
  outcome (Gabry et al. 2019). For a discrete outcome, such as a count, the
  PIT values are uniform only in the randomized PIT or the nonrandomized PIT
  for discrete data, so use one of those, or check the counts with a
  rootogram.
- **Forecasts:** score a forecast of a time series or of new locations with
  a proper scoring rule, such as the log score or the continuous ranked
  probability score, and check its calibration with the probability integral
  transform (Gneiting & Raftery 2007; Gneiting et al. 2007).
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
  models are clearly misspecified (Sivula et al. 2020).
- Stacking weights the predictive distributions of several models (Yao et al.
  2018b). When several models predict comparably well, prefer stacking for
  prediction to the selection of one model. Stacking combines predictive
  distributions only, because a parameter can mean different things in
  different models.
- When the models may be misspecified, check whether the conclusions are
  stable. Bagged posteriors, or BayesBag, give a more reproducible model
  selection and an uncertainty that reflects the misspecification (Huggins &
  Miller 2023; Huggins & Miller 2024). The
  bootstrap of BayesBag assumes exchangeable units, so resample whole groups
  or blocks when the data are grouped or dependent.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage G.

### H. Understand the fitted model

- Propagate the full posterior into the estimands with joint draws of the
  parameters, and report the uncertainty of the estimands.
- **Audit the uncertainty of a forecast.** List which sources of uncertainty
  the forecast propagates and which it holds fixed (Dietze 2017):
  1. the parameters;
  2. the initial conditions;
  3. the parameters fixed at one value, such as uncalibrated ones;
  4. the drivers, such as forecast weather;
  5. the process error;
  6. the observation error.

  Say whether its intervals are for the latent quantity or for new
  observations. An interval for the latent quantity leaves out the
  observation error, so its coverage of observed values falls below its
  level. Propose each source that the forecast holds fixed as an addition.
- **Sensitivity analysis:** measure how the conclusions change under
  alternative priors, likelihoods, or data exclusions, and report the changes
  that matter. Power-scaling the prior and the likelihood measures this
  sensitivity from the draws of one fit (Kallioinen et al. 2024).
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage H.

### I. Simulators, amortized inference, and emulators

- **Choose the method by the problem.** The likelihood, the cost of the
  model, and the number of datasets decide the method:
  1. a tractable likelihood whose code can be differentiated: MCMC, by
     stage E;
  2. a simulator without a tractable likelihood: simulation-based inference,
     which is ABC or a neural estimate of the posterior, the likelihood, or
     the likelihood-to-evidence ratio (Cranmer et al. 2020);
  3. many datasets analyzed with one model: neural posterior estimation,
     whose trained network gives the posterior of each new dataset at the
     cost of one network evaluation (Radev et al. 2022). A summary network
     lets it take datasets of varying size, and ProbPipe's learners take no
     summary network, so in ProbPipe the datasets share one size;
  4. datasets of varying size whose observations are conditionally
     independent: a neural estimate of the likelihood or of the
     likelihood-to-evidence ratio of one observation, followed by MCMC on
     each dataset, with the restriction of stage F on a learned ratio;
  5. an expensive deterministic simulator: a Gaussian-process emulator
     (Rasmussen & Williams 2006) trained on a design of simulator runs.
- **Budget the simulations first.** The budget bounds the method, because ABC
  spends its simulations on each dataset and a neural estimator spends them
  once, in training. Record four quantities before you choose:
  1. the cost of one simulator call, and whether calls can be vectorized over
     parameter values;
  2. the total simulation budget, in calls and in wall-clock time;
  3. the dimensions of the parameters and of the data;
  4. the summary statistics, which are chosen from domain knowledge or
     learned by a summary network when the data have many dimensions.
- **Train a neural estimator with held-out simulations:**
  1. hold out a set of validation simulations, and stop training when the
     loss on them stops decreasing;
  2. fix the seed of each training run, and train an ensemble of several
     networks with different seeds when the budget allows, since the
     disagreement between them measures the estimation error of one network;
  3. record the architecture, the numbers of simulations and of epochs, the
     hyperparameters that were tuned, and the wall-clock time in the design
     document.
- **Calibrate every amortized posterior.** Stage D requires SBC and the
  coverage of the estimands for an amortized method. Compute the coverage over
  held-out simulations at several credibility levels from 50% to 95%, which is
  the expected-coverage check, and grade it by step 6 of stage D, with n the
  number of held-out simulations. Report the band p ± 2√(p(1 − p)/n) beside
  each coverage. Coverage below the band means intervals that are too narrow,
  which is the common failure of neural posterior estimates, and coverage
  above it is conservative (Hermans et al. 2022). The coverage of each
  parameter's interval misses errors in the joint posterior, which the TARP
  test checks (Lemos et al. 2023). When a reference posterior exists, such as
  an MCMC fit of a tractable version of the model, compare the two by the
  validation of approximate methods in stage E.
- **Check that the observed data resemble the simulations.** Compare the
  summary statistics of the observed data with their prior predictive
  distribution, by stage C. An observed summary far in the tail of the
  simulations means that the simulator does not produce data like the
  observed data. An amortized estimator evaluated there extrapolates beyond
  its training data, so revise the simulator or the prior before you
  interpret its posterior (Schmitt et al. 2024).
- **Validate an emulator on held-out simulator runs.** The emulator is valid for inference when it passes five checks:
  1. Training design: the inputs of the training runs fill the region of the
     parameter space that the prior supports, as a Latin hypercube does, and
     sequential runs add inputs where the emulator's predictive variance is
     large (Gramacy 2020).
  2. Held-out runs: the standardized error of each held-out run, which is the
     emulator's prediction error over its predictive standard deviation, is
     approximately standard normal. Many standardized errors beyond ±2 mean
     that the emulator is overconfident (Bastos & O'Hagan 2009).
  3. Propagation: the likelihood includes the emulator's predictive variance,
     so that the posterior reflects the emulator's error. Roberts et al.
     2026a review how surrogate uncertainty is propagated and how active
     learning chooses new simulator runs. Roberts et al. 2026b compare
     approximations of the posterior under an uncertain surrogate and show
     where the expected unnormalized posterior breaks down.
  4. Discrepancy: a calibration that ignores the difference between the
     simulator and the real system can give biased and overconfident
     parameter estimates (Kennedy & O'Hagan 2001). The default is a
     discrepancy term in the model with an informative prior, from the
     user's knowledge of where the simulator falls short, because a
     discrepancy with a vague prior cannot be separated from the parameters
     (Brynjarsdóttir & O'Hagan 2014). When no such knowledge exists, the
     alternative is a model without a discrepancy term. Its estimands are
     then conditional on the simulator, and a sensitivity analysis refits
     the model with a discrepancy term to show how much they change.
  5. Precision: the Gaussian process is computed in double precision,
     because its covariance matrix is often ill-conditioned and its Cholesky
     factor fails or loses accuracy in single precision. The row "Emulate an
     expensive model" of [CAPABILITIES.md](CAPABILITIES.md) says how to set
     the precision.
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage I.

### J. Reproducibility and communication

- Fix the random seeds, record the software versions, keep every model of the
  sequence, and record each decision with its reason in the changelog of the
  design document.
- Report the analysis with the items of the Bayesian analysis reporting
  guidelines (Kruschke 2021).
- *ProbPipe:* the rows of [CAPABILITIES.md](CAPABILITIES.md) for stage J.

---

## Part 2 — Annotated bibliography

Each entry starts with its citation key in bold, and Part 1 and the design
document cite the entry by that key. A key names the first author, both
authors of a two-author work, or the first author with "et al." for three or
more, followed by the year. A letter after the year separates two works with
the same key. "Free" marks a copy of the full text that the authors or the
publisher host at no cost.

Choose readings by the kind of model and by the user's background. For a user
new to Bayesian modeling, recommend at most one book and one short applied
paper, from the starting points or from the shelf of the user's field, and
prefer free sources in a language the user knows. For other users, choose the
entries of the shelf that matches the model and the hardest part of the
design.

### Starting points for newcomers

- **Johnson et al. 2022.** Johnson, Ott & Dogucu, *Bayes Rules! An
  Introduction to Applied Bayesian Modeling* (Chapman & Hall/CRC, 2022).
  doi:10.1201/9780429288340. Free: https://www.bayesrulesbook.com/. It
  introduces Bayesian modeling with R at an undergraduate level, so it suits
  a user who has never fit a Bayesian model.
- **Martin et al. 2021.** Martin, Kumar & Lao, *Bayesian Modeling and
  Computation in Python* (Chapman & Hall/CRC, 2021).
  doi:10.1201/9781003019169. Free: https://bayesiancomputationbook.com/. Its
  examples use PyMC, TensorFlow Probability, and ArviZ, so it suits a user who
  works in Python.
- **McElreath 2020.** McElreath, *Statistical Rethinking: A Bayesian Course
  with Examples in R and Stan* (2nd ed., Chapman & Hall/CRC, 2020).
  doi:10.1201/9780429029608. It builds models from scientific mechanisms and
  causal diagrams, which suits a scientist. Its chapter 15 treats measurement
  error and missing data.
- **van de Schoot et al. 2021.** van de Schoot, Depaoli, King, Kramer,
  Märtens, Tadesse, et al., "Bayesian statistics and modelling," *Nature
  Reviews Methods Primers* 1:1. doi:10.1038/s43586-020-00001-2. This short
  primer covers the steps from the prior to the report, and it serves as the
  one short paper for a newcomer from any field.

### The workflow as a whole

- **Gelman et al. 2026.** Gelman, Vehtari & McElreath, with Simpson,
  Margossian, Yao, Kennedy, Gabry, Bürkner, Modrák & Leos Barajas, *Bayesian
  Workflow* (Chapman & Hall/CRC, 2026). Free:
  https://avehtari.github.io/Bayesian-Workflow/ (PDF and the code of the case
  studies). It treats iterative model building, checking, and computational
  troubleshooting through case studies in R and Stan. It is the primary reference for Part 1.
- **Gelman et al. 2020a.** Gelman, Vehtari, Simpson, Margossian, Carpenter,
  Yao, et al., "Bayesian workflow," arXiv:2011.01808. This article is the
  shorter overview of the workflow that the book expands, and it suits a
  working user who wants the whole workflow in one paper.
- **Betancourt 2020.** Betancourt, "Towards a principled Bayesian workflow"
  (case study, April 2020), listed on the author's index as "Principled Model
  Building Workflow".
  https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html.
  It organizes the workflow around four questions about the model, from the
  consistency of the prior with domain expertise to the adequacy of the
  model, and it defines the posterior z-score and contraction of stage D.
  The page is large and loads slowly. Schad, Betancourt & Vasishth, "Toward a
  principled Bayesian workflow in cognitive science," *Psychological Methods*
  26(1):103–126 (2021), doi:10.1037/met0000275, arXiv:1904.12765, applies the
  same workflow to reading-time data, which suits an applied scientist.
- **Gelman et al. 2013.** Gelman, Carlin, Stern, Dunson, Vehtari & Rubin,
  *Bayesian Data Analysis* (3rd ed., Chapman & Hall/CRC, 2013).
  doi:10.1201/b16018. Free: https://sites.stat.columbia.edu/gelman/book/ (PDF
  for non-commercial use). It is the standard graduate reference, with
  chapters on hierarchical models, model checking, and missing data, and it
  suits a user with a background in statistics.

### Priors and elicitation

- **Gelman et al. 2017.** Gelman, Simpson & Betancourt, "The prior can often
  only be understood in the context of the likelihood," *Entropy* 19(10):555.
  doi:10.3390/e19100555, arXiv:1708.07487. It argues that a prior is judged by
  the data that it predicts, which is the basis of the prior predictive check
  of stage C.
- **Gelman 2006.** Gelman, "Prior distributions for variance parameters in
  hierarchical models (comment on article by Browne and Draper)," *Bayesian
  Analysis* 1(3):515–534. doi:10.1214/06-BA117A. It shows that an
  inverse-gamma(ε, ε) prior on a group-level variance is informative when the
  variance is small, and it recommends uniform or half-t priors on the
  standard deviation. Point a user here when they choose the prior of a
  hierarchical model.
- **Simpson et al. 2017.** Simpson, Rue, Riebler, Martins & Sørbye,
  "Penalising model component complexity: A principled, practical approach to
  constructing priors," *Statistical Science* 32(1):1–28.
  doi:10.1214/16-STS576. It defines penalized-complexity priors, which shrink
  a model component toward a simpler base model. R-INLA uses them for the
  precision and the range of spatial effects.
- **Mikkola et al. 2024.** Mikkola, Martin, Chandramouli, Hartmann, Abril Pla,
  Thomas, et al., "Prior knowledge elicitation: The past, present, and
  future," *Bayesian Analysis* 19(4):1129–1161. doi:10.1214/23-BA1381,
  arXiv:2112.01380. It reviews methods that turn an expert's judgments into a
  prior, and it suits a project whose priors come from a domain expert.
- **Stan wiki 2025.** Stan Development Team, "Prior choice recommendations"
  (wiki, last edited April 2025).
  https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations.
  It gives default priors by model and parameter type. Some entries are dated, such as
  the older rstanarm defaults and a Cauchy prior that the page itself no
  longer recommends, so check a choice against Gelman et al. 2026.

### Regression, GLMs, and hierarchical models

- **Gelman et al. 2020b.** Gelman, Hill & Vehtari, *Regression and Other
  Stories* (Cambridge University Press, 2020). doi:10.1017/9781139161879.
  Free: https://avehtari.github.io/ROS-Examples/ (PDF and code). It teaches
  linear and generalized linear regression with simulation, checking, and
  causal questions, and it suits a user whose model is a regression.
- **Gelman & Hill 2007.** Gelman & Hill, *Data Analysis Using Regression and
  Multilevel/Hierarchical Models* (Cambridge University Press, 2007).
  doi:10.1017/CBO9780511790942. Its multilevel chapters treat varying
  intercepts and slopes and multilevel GLMs, which suit a user with grouped
  data.
- **Bürkner 2017.** Bürkner, "brms: An R package for Bayesian multilevel
  models using Stan," *Journal of Statistical Software* 80(1).
  doi:10.18637/jss.v080.i01. It maps R model formulas to Stan models, which
  helps an R user state the model that the pipeline will express.
- **Papaspiliopoulos et al. 2007.** Papaspiliopoulos, Roberts & Sköld, "A
  general framework for the parametrization of hierarchical models,"
  *Statistical Science* 22(1):59–73. doi:10.1214/088342307000000014. It
  explains when the centered or the non-centered parameterization mixes
  better, which depends on how strongly the data inform each group.

### Time series, state-space models, and forecasting

- **West & Harrison 1997.** West & Harrison, *Bayesian Forecasting and
  Dynamic Models* (2nd ed., Springer, 1997). doi:10.1007/b98971. It is the
  authoritative treatment of Bayesian dynamic linear models and sequential
  forecasting.
- **Durbin & Koopman 2012.** Durbin & Koopman, *Time Series Analysis by State
  Space Methods* (2nd ed., Oxford University Press, 2012).
  doi:10.1093/acprof:oso/9780199641178.001.0001. It covers the Kalman filter
  and smoother, non-Gaussian state-space models, and simulation smoothing.
- **Särkkä & Svensson 2023.** Särkkä & Svensson, *Bayesian Filtering and
  Smoothing* (2nd ed., Cambridge University Press, 2023).
  doi:10.1017/9781108917407. Free:
  https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf (personal
  use). It derives the Kalman, the nonlinear Gaussian, and the particle
  filters and smoothers, and it suits a user whose latent states are
  estimated as the data arrive.
- **Hyndman & Athanasopoulos 2021.** Hyndman & Athanasopoulos, *Forecasting:
  Principles and Practice* (3rd ed., OTexts, 2021). Free:
  https://otexts.com/fpp3/. It is an accessible guide to forecasting models,
  rolling-origin evaluation, and forecast accuracy. Its methods are
  classical, so use it for the evaluation design and for baseline forecasts.
- **Gneiting & Raftery 2007.** Gneiting & Raftery, "Strictly proper scoring
  rules, prediction, and estimation," *Journal of the American Statistical
  Association* 102(477):359–378. doi:10.1198/016214506000001437. It defines
  proper scoring rules, such as the log score and the continuous ranked
  probability score, which score a predictive distribution as a whole.
- **Gneiting et al. 2007.** Gneiting, Balabdaoui & Raftery, "Probabilistic
  forecasts, calibration and sharpness," *Journal of the Royal Statistical
  Society Series B* 69(2):243–268. doi:10.1111/j.1467-9868.2007.00587.x. It
  states the goal of maximal sharpness subject to calibration, and it checks
  calibration with the probability integral transform.

Hidden Markov models are under mixtures and latent variables, and
leave-future-out cross-validation is under model checking.

### Spatial models

- **Banerjee et al. 2014.** Banerjee, Carlin & Gelfand, *Hierarchical
  Modeling and Analysis for Spatial Data* (2nd ed., Chapman & Hall/CRC, 2014).
  doi:10.1201/b17115. It is the authoritative Bayesian treatment of
  point-referenced and areal data, including spatial GLMs and spatially
  varying coefficients.
- **Moraga 2019.** Moraga, *Geospatial Health Data: Modeling and
  Visualization with R-INLA and Shiny* (Chapman & Hall/CRC, 2019).
  doi:10.1201/9780429341823. Free: https://www.paulamoraga.com/book-geospatial/.
  It fits areal and geostatistical disease models in R-INLA, and it suits an
  applied user new to spatial models.
- **Besag et al. 1991.** Besag, York & Mollié, "Bayesian image restoration,
  with two applications in spatial statistics," *Annals of the Institute of
  Statistical Mathematics* 43(1):1–20. doi:10.1007/BF00116466. It introduces
  the BYM model of areal counts, which adds an intrinsic conditional
  autoregressive (ICAR) effect to an unstructured effect.
- **Riebler et al. 2016.** Riebler, Sørbye, Simpson & Rue, "An intuitive
  Bayesian spatial model for disease mapping that accounts for scaling,"
  *Statistical Methods in Medical Research* 25(4):1145–1165.
  doi:10.1177/0962280216660421. It reparameterizes the BYM model as BYM2,
  whose scaled ICAR effect and mixing parameter take interpretable priors.
- **Morris et al. 2019.** Morris, Wheeler-Martin, Simpson, Mooney, Gelman &
  DiMaggio, "Bayesian hierarchical spatial models: Implementing the Besag York
  Mollié model in Stan," *Spatial and Spatio-temporal Epidemiology* 31:100301.
  doi:10.1016/j.sste.2019.100301. It writes the ICAR and BYM2 log densities as
  sums over neighboring pairs, which NUTS can use, so it suits a user who
  brings an areal model.
- **Rue et al. 2009.** Rue, Martino & Chopin, "Approximate Bayesian inference
  for latent Gaussian models by using integrated nested Laplace
  approximations," *Journal of the Royal Statistical Society Series B*
  71(2):319–392. doi:10.1111/j.1467-9868.2008.00700.x. It defines INLA, the
  deterministic approximation that R-INLA computes. A user who brings an
  R-INLA model can compare its fit with an MCMC fit in ProbPipe.
- **Lindgren et al. 2011.** Lindgren, Rue & Lindström, "An explicit link
  between Gaussian fields and Gaussian Markov random fields: the stochastic
  partial differential equation approach," *Journal of the Royal Statistical
  Society Series B* 73(4):423–498. doi:10.1111/j.1467-9868.2011.00777.x. It
  represents a Matérn field by a sparse Gaussian Markov random field on a
  mesh, which makes geostatistical models of many locations tractable.
- **Datta et al. 2016.** Datta, Banerjee, Finley & Gelfand, "Hierarchical
  nearest-neighbor Gaussian process models for large geostatistical
  datasets," *Journal of the American Statistical Association*
  111(514):800–812. doi:10.1080/01621459.2015.1044091. It defines the
  nearest-neighbor Gaussian process (NNGP), whose cost grows linearly in the
  number of locations.
- **Heaton et al. 2019.** Heaton, Datta, Finley, Furrer, Guinness, Guhaniyogi,
  et al., "A case study competition among methods for analyzing large spatial
  data," *Journal of Agricultural, Biological and Environmental Statistics*
  24(3):398–425. doi:10.1007/s13253-018-00348-w. It compares scalable spatial
  methods on common datasets, which helps a user choose one for a large
  dataset.

Gaussian processes in general are under emulators, and spatial
cross-validation is under model checking.

### Spatiotemporal models

- **Cressie & Wikle 2011.** Cressie & Wikle, *Statistics for Spatio-Temporal
  Data* (Wiley, 2011). ISBN 978-0-471-69274-4. It is the authoritative
  treatment of descriptive and dynamic spatiotemporal models in a
  hierarchical framework.
- **Wikle et al. 2019.** Wikle, Zammit-Mangion & Cressie, *Spatio-Temporal
  Statistics with R* (Chapman & Hall/CRC, 2019). doi:10.1201/9781351769723.
  Free: https://spacetimewithr.org/. It is the applied companion of Cressie &
  Wikle 2011, with R code for descriptive and dynamic models and their
  evaluation.

### Mixtures, hidden Markov models, and latent variables

- **Frühwirth-Schnatter et al. 2019.** Frühwirth-Schnatter, Celeux & Robert
  (eds.), *Handbook of Mixture Analysis* (Chapman & Hall/CRC, 2019).
  doi:10.1201/9780429055911. It covers finite and infinite mixtures, label
  switching, and the choice of the number of components.
- **Zucchini et al. 2016.** Zucchini, MacDonald & Langrock, *Hidden Markov
  Models for Time Series: An Introduction Using R* (2nd ed., Chapman &
  Hall/CRC, 2016). doi:10.1201/b20790. It computes the likelihood with the
  forward algorithm, which marginalizes the hidden states as stage E
  requires, and it checks models with pseudo-residuals.
- **Stan User's Guide.** Stan Development Team, *Stan User's Guide*, chapter
  "Latent discrete parameters."
  https://mc-stan.org/docs/stan-users-guide/latent-discrete.html.
  It marginalizes change points and other discrete parameters out of a
  likelihood with log-sum-exp, which lets NUTS fit the model.

### Survival and event-history models

- **Ibrahim et al. 2001.** Ibrahim, Chen & Sinha, *Bayesian Survival
  Analysis* (Springer, 2001). doi:10.1007/978-1-4757-3447-8. It is the
  authoritative treatment of Bayesian parametric, semiparametric, and
  cure-rate models.
- **Brilleman et al. 2020.** Brilleman, Elci, Novik & Wolfe, "Bayesian
  survival analysis using the rstanarm R package," arXiv:2002.09633. It fits
  parametric and spline-based hazard models with censoring, delayed entry, and
  time-varying effects, and it suits an applied user.

### Missing data and measurement error

- **Little & Rubin 2019.** Little & Rubin, *Statistical Analysis with Missing
  Data* (3rd ed., Wiley, 2019). doi:10.1002/9781119482260. It defines the
  mechanisms of missingness and the likelihood-based and Bayesian methods for
  each. McElreath 2020 and Gelman et al. 2013 give shorter Bayesian
  treatments.
- **Carroll et al. 2006.** Carroll, Ruppert, Stefanski & Crainiceanu,
  *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.,
  Chapman & Hall/CRC, 2006). doi:10.1201/9781420010138. It treats classical
  and Berkson error in covariates, with a chapter on Bayesian methods.

### Causal inference

- **Hernán & Robins 2020.** Hernán & Robins, *Causal Inference: What If*
  (Chapman & Hall/CRC, 2020). Free: https://miguelhernan.org/whatifbook. It
  defines causal effects by counterfactuals and treats confounding, selection
  bias, and time-varying treatments, so it suits a user whose estimand is
  causal.
- **Oganisian & Roy 2021.** Oganisian & Roy, "A practical introduction to
  Bayesian estimation of causal effects: Parametric and nonparametric
  approaches," *Statistics in Medicine* 40(2):518–551. doi:10.1002/sim.8761.
  It computes the posterior of an average treatment effect from the posterior
  of an outcome model, which is the propagation of stage H.

### Simulators and simulation-based inference

- **Cranmer et al. 2020.** Cranmer, Brehmer & Louppe, "The frontier of
  simulation-based inference," *Proceedings of the National Academy of
  Sciences* 117(48):30055–30062. doi:10.1073/pnas.1912789117,
  arXiv:1911.01429. It reviews ABC and the neural estimates of the posterior,
  the likelihood, and the likelihood ratio, and it suits a user who chooses
  among them.
- **Sisson et al. 2018.** Sisson, Fan & Beaumont (eds.), *Handbook of
  Approximate Bayesian Computation* (Chapman & Hall/CRC, 2018).
  doi:10.1201/9781315117195. It is the authoritative treatment of ABC,
  including summary statistics and sequential Monte Carlo ABC.
- **Radev et al. 2022.** Radev, Mertens, Voss, Ardizzone & Köthe, "BayesFlow:
  Learning complex stochastic models with invertible neural networks," *IEEE
  Transactions on Neural Networks and Learning Systems* 33(4):1452–1466.
  doi:10.1109/TNNLS.2020.3042395, arXiv:2003.06281. It defines amortized
  posterior estimation with a learned summary network, which BayesFlow
  implements and ProbPipe's amortized posterior uses.
- **Hermans et al. 2022.** Hermans, Delaunoy, Rozet, Wehenkel, Begy & Louppe,
  "A trust crisis in simulation-based inference? Your posterior approximations
  can be unfaithful," *Transactions on Machine Learning Research* (2022).
  arXiv:2110.06581, https://openreview.net/forum?id=LHAbHkt6Aq. It shows that
  the common neural and ABC methods can give overconfident posteriors, and it
  measures this with the expected-coverage check of stage I.
- **Lemos et al. 2023.** Lemos, Coogan, Hezaveh & Perreault-Levasseur,
  "Sampling-based accuracy testing of posterior estimators for general
  inference," *Proceedings of the 40th International Conference on Machine
  Learning*, PMLR 202:19256–19273. arXiv:2302.03026. It defines TARP, a
  coverage test of the joint posterior that needs only draws from the
  estimator.
- **Schmitt et al. 2024.** Schmitt, Bürkner, Köthe & Radev, "Detecting model
  misspecification in amortized Bayesian inference with neural networks," in
  *Pattern Recognition: DAGM GCPR 2023*, Lecture Notes in Computer Science
  14264:541–557 (Springer, 2024). doi:10.1007/978-3-031-54605-1_35,
  arXiv:2112.08866. It detects observed data unlike the training simulations
  by a distance in the space of the learned summary statistics.

ODE and compartmental models are on the epidemiology shelf, with Grinsztajn
et al. 2021.

### Emulators and calibration of computer models

- **Kennedy & O'Hagan 2001.** Kennedy & O'Hagan, "Bayesian calibration of
  computer models," *Journal of the Royal Statistical Society Series B*
  63(3):425–464. doi:10.1111/1467-9868.00294. It defines the calibration of a
  simulator with a Gaussian-process emulator and a model-discrepancy term.
- **Brynjarsdóttir & O'Hagan 2014.** Brynjarsdóttir & O'Hagan, "Learning about
  physical parameters: the importance of model discrepancy," *Inverse
  Problems* 30(11):114007. doi:10.1088/0266-5611/30/11/114007. It shows that a
  calibration without a discrepancy term gives biased and overconfident
  parameters, and that the discrepancy term needs an informative prior to be
  separated from the parameters.
- **Bastos & O'Hagan 2009.** Bastos & O'Hagan, "Diagnostics for Gaussian
  process emulators," *Technometrics* 51(4):425–438.
  doi:10.1198/TECH.2009.08019. It defines the diagnostics of an emulator on
  held-out runs, such as the standardized prediction errors and their
  Mahalanobis distance.
- **Gramacy 2020.** Gramacy, *Surrogates: Gaussian Process Modeling, Design,
  and Optimization for the Applied Sciences* (Chapman & Hall/CRC, 2020).
  doi:10.1201/9780367815493. Free: https://bobby.gramacy.com/surrogates/. It
  covers space-filling and sequential designs of simulator runs, the fitting
  of Gaussian processes, and calibration, with R code.
- **Roberts et al. 2026a.** Roberts, Dietze & Huggins, "Surrogate-based
  Bayesian inference: Uncertainty quantification and active learning"
  (2026). arXiv:2603.13646. It reviews the use of surrogates, also called
  emulators, in Bayesian inference, with attention to propagating the
  surrogate's uncertainty and to choosing new simulator runs by active
  learning. Point a user here for an overview before building an emulator.
- **Roberts et al. 2026b.** Roberts, Dietze & Huggins, "Propagating
  surrogate uncertainty in Bayesian inverse problems" (2026).
  arXiv:2601.03532. It proposes the expected posterior as the baseline for a
  posterior that accounts for the surrogate's uncertainty, shows that the
  expected unnormalized posterior can break down when that uncertainty is
  far from uniform, and gives an MCMC algorithm for Gaussian-process
  surrogates. Point a user here when an emulator replaces an expensive
  likelihood or forward model.
- **Rasmussen & Williams 2006.** Rasmussen & Williams, *Gaussian Processes for
  Machine Learning* (MIT Press, 2006). doi:10.7551/mitpress/3206.001.0001.
  Free: https://gaussianprocess.org/gpml/. It defines Gaussian-process
  regression, covariance functions, and the fitting of their
  hyperparameters, which emulators and geostatistical models share.

### Approximate inference and machine-learning models

- **Blei et al. 2017.** Blei, Kucukelbir & McAuliffe, "Variational inference:
  A review for statisticians," *Journal of the American Statistical
  Association* 112(518):859–877. doi:10.1080/01621459.2017.1285773,
  arXiv:1601.00670. It derives the evidence lower bound and mean-field VI, and
  it explains why mean-field VI underestimates the posterior variance.
- **Yao et al. 2018a.** Yao, Vehtari, Simpson & Gelman, "Yes, but did it
  work?: Evaluating variational inference," *Proceedings of the 35th
  International Conference on Machine Learning*, PMLR 80:5581–5590.
  arXiv:1802.02538. It defines the Pareto k̂ diagnostic of a variational fit,
  which stage E uses, and a variational form of SBC.
- **Huggins et al. 2020.** Huggins, Kasprzak, Campbell & Broderick, "Validated
  variational inference via practical posterior error bounds," *Proceedings
  of the 23rd International Conference on Artificial Intelligence and
  Statistics*, PMLR 108:1792–1802. arXiv:1910.04102. It bounds the error of
  the means and variances of a variational approximation, and it suits a user
  who reports VI results without an MCMC reference.
- **Papamakarios et al. 2021.** Papamakarios, Nalisnick, Rezende, Mohamed &
  Lakshminarayanan, "Normalizing flows for probabilistic modeling and
  inference," *Journal of Machine Learning Research* 22(57):1–64.
  arXiv:1912.02762. https://jmlr.org/papers/v22/19-1028.html.
  It reviews normalizing flows, the density estimators of amortized inference and of
  flexible variational families.
- **Welling & Teh 2011.** Welling & Teh, "Bayesian learning via stochastic
  gradient Langevin dynamics," *Proceedings of the 28th International
  Conference on Machine Learning*, 681–688.
  https://icml.cc/2011/papers/398_icmlpaper.pdf.
  It defines stochastic-gradient Langevin dynamics.
- **Nemeth & Fearnhead 2021.** Nemeth & Fearnhead, "Stochastic gradient Markov
  chain Monte Carlo," *Journal of the American Statistical Association*
  116(533):433–450. doi:10.1080/01621459.2020.1847120, arXiv:1907.06986. It
  reviews the stochastic-gradient MCMC methods, their bias at a constant step
  size, and their diagnostics.
- **Negrea et al. 2022.** Negrea, Yang, Feng, Roy & Huggins, "Tuning
  stochastic gradient algorithms for statistical inference via large-sample
  asymptotics," arXiv:2207.12395. Its first version is titled "Statistical
  inference with stochastic gradient algorithms." It analyzes
  stochastic-gradient algorithms for optimization and sampling as the step
  size and the number of observations grow together, and it gives the
  iterate-averaging result and the tuning guidance of stage E. It suits a
  user who tunes a stochastic-gradient method for statistical inference,
  generalized posteriors included.
- **Wang et al. 2026.** Wang, Ding & Huggins, "Accurate large-sample
  uncertainty quantification using stochastic gradient Markov chain Monte
  Carlo," *Proceedings of the 43rd International Conference on Machine
  Learning*, PMLR 306. arXiv:2606.00293. It gives the discrete-time
  approximations of stage E, with non-asymptotic bounds on their error. It
  suits a user who tunes stochastic-gradient MCMC for accurate uncertainty
  when the batch size is large or the model is misspecified.
- **Wilson & Izmailov 2020.** Wilson & Izmailov, "Bayesian deep learning and
  a probabilistic perspective of generalization," *Advances in Neural
  Information Processing Systems* 33. arXiv:2002.08791. It treats Bayesian
  model averaging for neural networks and relates deep ensembles to it, so it
  suits a user whose model is a neural network.
- **Murphy 2023.** Murphy, *Probabilistic Machine Learning: Advanced Topics*
  (MIT Press, 2023). Free: https://probml.github.io/pml-book/book2.html
  (draft, CC BY-NC-ND). It is a reference for deep generative models,
  Bayesian neural networks, and approximate inference from the view of
  machine learning.

### Validating and diagnosing computation

- **Talts et al. 2018.** Talts, Betancourt, Simpson, Vehtari & Gelman,
  "Validating Bayesian inference algorithms with simulation-based
  calibration," arXiv:1804.06788. It defines SBC by the uniformity of the
  ranks of the true parameters among the posterior draws.
- **Modrák et al. 2025.** Modrák, Moon, Kim, Bürkner, Huurre, Faltejsková,
  Gelman & Vehtari, "Simulation-based calibration checking for Bayesian
  computation: The choice of test quantities shapes sensitivity," *Bayesian
  Analysis* 20(2):461–488 (online 2023). doi:10.1214/23-BA1404,
  arXiv:2211.02383. It shows that data-dependent test quantities, such as the
  log likelihood, detect errors that the ranks of the parameters miss.
- **Säilynoja et al. 2022.** Säilynoja, Bürkner & Vehtari, "Graphical test
  for discrete uniformity and its applications in goodness-of-fit evaluation
  and multiple sample comparison," *Statistics and Computing* 32:32.
  doi:10.1007/s11222-022-10090-6, arXiv:2103.10522. It gives the simultaneous
  confidence bands of the ECDF plot of SBC ranks and of LOO-PIT values.
- **Vehtari et al. 2021.** Vehtari, Gelman, Simpson, Carpenter & Bürkner,
  "Rank-normalization, folding, and localization: An improved R̂ for
  assessing convergence of MCMC (with discussion)," *Bayesian Analysis*
  16(2):667–718. doi:10.1214/20-BA1221, arXiv:1903.08008. It defines the
  rank-normalized split-R̂, the bulk and tail ESS, and their thresholds.
- **Margossian et al. 2025.** Margossian, Hoffman, Sountsov, Riou-Durand,
  Vehtari & Gelman, "Nested R̂: Assessing the convergence of Markov chain
  Monte Carlo when running many short chains," *Bayesian Analysis*
  20(4):1587–1614. doi:10.1214/24-BA1453, arXiv:2110.13017. It defines the
  nested R̂, which suits many short chains run in parallel on a GPU.
- **Betancourt 2017.** Betancourt, "A conceptual introduction to Hamiltonian
  Monte Carlo," arXiv:1701.02434. It explains the geometry of HMC, and it
  shows why a divergent transition marks a region that the sampler cannot
  explore.
- **Stan diagnostics guide.** Stan Development Team, "How to diagnose and
  resolve convergence problems."
  https://mc-stan.org/learn-stan/diagnostics-warnings.html.
  It explains each warning of a NUTS run, from divergences and tree depth to E-BFMI, R̂, and
  ESS, with the usual remedies. It suits a practitioner who meets a warning.
- **Gorham & Mackey 2017.** Gorham & Mackey, "Measuring sample quality with
  kernels," *Proceedings of the 34th International Conference on Machine
  Learning*, PMLR 70:1292–1301. arXiv:1703.01717. It defines the kernel Stein
  discrepancy, which measures how far a set of draws is from the posterior
  with only the gradient of the log density.

### Model checking, comparison, and cross-validation

- **Gabry et al. 2019.** Gabry, Simpson, Vehtari, Betancourt & Gelman,
  "Visualization in Bayesian workflow," *Journal of the Royal Statistical
  Society Series A* 182(2):389–402. doi:10.1111/rssa.12378, arXiv:1709.01449.
  It gives the graphical prior and posterior predictive checks and the
  LOO-PIT check.
- **Li & Huggins 2026.** Li & Huggins, "Calibrated model criticism using split
  predictive checks," *Journal of the American Statistical Association*
  (2026). doi:10.1080/01621459.2026.2649585, arXiv:2203.15897. It defines
  single and divided split predictive checks, which split the data into a
  training part and a held-out part, and it develops their asymptotic theory.
  It shows that these checks are calibrated where posterior predictive checks
  are conservative, and when each of the two variants has more power. It
  suits a user who reads the p-value of a check as evidence against the
  model.
- **Vehtari et al. 2017; Vehtari et al. 2024.** Vehtari, Gelman & Gabry,
  "Practical Bayesian model evaluation using leave-one-out cross-validation
  and WAIC," *Statistics and Computing* 27(5):1413–1432.
  doi:10.1007/s11222-016-9696-4, arXiv:1507.04544. Vehtari, Simpson, Gelman,
  Yao & Gabry, "Pareto smoothed importance sampling," *Journal of Machine
  Learning Research* 25(72):1–58 (2024), arXiv:1507.02646. The first defines
  PSIS-LOO and its use in model comparison. The second defines the Pareto k̂
  and its threshold, which depends on the number of draws.
- **Vehtari CV-FAQ.** Vehtari, "Cross-validation FAQ."
  https://users.aalto.fi/~ave/CV-FAQ.html.
  It answers the practical questions of LOO, such as when LOO is unreliable and how to read the standard error
  of an elpd difference, and it suits a working user.
- **Sivula et al. 2020.** Sivula, Magnusson, Matamoros & Vehtari, "Uncertainty
  in Bayesian leave-one-out cross-validation based model comparison,"
  arXiv:2008.10296 (version 6, 2025). It shows when the standard error of an
  elpd difference is too small, which is the basis of the cases of stage G.
- **Bürkner et al. 2020.** Bürkner, Gabry & Vehtari, "Approximate
  leave-future-out cross-validation for Bayesian time series models,"
  *Journal of Statistical Computation and Simulation* 90(14):2499–2523.
  doi:10.1080/00949655.2020.1783262, arXiv:1902.06281. It approximates
  leave-future-out cross-validation with PSIS, so a time-series model is
  refitted only when the k̂ is too high.
- **Roberts et al. 2017.** Roberts, Bahn, Ciuti, Boyce, Elith,
  Guillera-Arroita, et al., "Cross-validation strategies for data with
  temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*
  40(8):913–929. doi:10.1111/ecog.02881. It shows that random folds
  overestimate predictive performance on dependent data, and it recommends
  blocked folds.
- **Yao et al. 2018b.** Yao, Vehtari, Simpson & Gelman, "Using stacking to
  average Bayesian predictive distributions (with discussion)," *Bayesian
  Analysis* 13(3):917–1007. doi:10.1214/17-BA1091, arXiv:1704.02030. It
  weights the predictive distributions of several models by their LOO
  performance.
- **Huggins & Miller 2023; Huggins & Miller 2024.** Huggins & Miller,
  "Reproducible model selection using bagged posteriors," *Bayesian Analysis*
  18(1):79–104. doi:10.1214/21-BA1301, arXiv:2007.14845. Huggins & Miller,
  "Reproducible parameter inference using bagged posteriors," *Electronic
  Journal of Statistics* 18(1):1549–1585 (2024). doi:10.1214/24-EJS2237. The
  first introduces BayesBag for stable model selection under misspecification.
  The second shows that the bagged posterior of a parameter gives an
  uncertainty that reflects the misspecification.
- **Kallioinen et al. 2024.** Kallioinen, Paananen, Bürkner & Vehtari,
  "Detecting and diagnosing prior and likelihood sensitivity with
  power-scaling," *Statistics and Computing* 34:57.
  doi:10.1007/s11222-023-10366-5, arXiv:2107.14054. It measures how the
  posterior changes when the prior or the likelihood is raised to a power,
  from the draws of one fit, which gives the sensitivity analysis of stage H
  without refitting.

### Reporting

- **Kruschke 2021.** Kruschke, "Bayesian analysis reporting guidelines,"
  *Nature Human Behaviour* 5(10):1282–1291. doi:10.1038/s41562-021-01177-7
  (open access). It lists what a report of a Bayesian analysis states, from
  the model and the priors to the diagnostics and the sensitivity analysis,
  and it is the checklist of the reporting section of a pipeline.

### Software that ProbPipe works with

Cite each of these in a report whose results depend on it.

- **Kumar et al. 2019.** Kumar, Carroll, Hartikainen & Martin, "ArviZ a
  unified library for exploratory analysis of Bayesian models in Python,"
  *Journal of Open Source Software* 4(33):1143. doi:10.21105/joss.01143. ProbPipe
  computes the R̂, the ESS, and PSIS-LOO through ArviZ.
- **Cabezas et al. 2024.** Cabezas, Corenflos, Lao & Louf, "BlackJAX:
  Composable Bayesian inference in JAX," arXiv:2402.10797. BlackJAX runs the
  default gradient-based MCMC of ProbPipe.
- **Abril-Pla et al. 2023.** Abril-Pla, Andreani, Carroll, Dong, Fonnesbeck,
  Kochurov, et al., "PyMC: a modern, and comprehensive probabilistic
  programming framework in Python," *PeerJ Computer Science* 9:e1516.
  doi:10.7717/peerj-cs.1516. Cite it for a model that a user brings in PyMC.
- **Carpenter et al. 2017.** Carpenter, Gelman, Hoffman, Lee, Goodrich,
  Betancourt, et al., "Stan: A probabilistic programming language," *Journal
  of Statistical Software* 76(1). doi:10.18637/jss.v076.i01. Cite it for a
  model that a user brings in Stan.
- **Radev et al. 2023.** Radev, Schmitt, Schumacher, Elsemüller, Pratz,
  Schälte, et al., "BayesFlow: Amortized Bayesian workflows with neural
  networks," *Journal of Open Source Software* 8(89):5702.
  doi:10.21105/joss.05702. BayesFlow trains the neural estimators of ProbPipe's
  amortized inference.

### Shelves by field

Ecology:

- **Kéry & Royle 2016.** Kéry & Royle, *Applied Hierarchical Modeling in
  Ecology: Analysis of Distribution, Abundance and Species Richness in R and
  BUGS*, vol. 1 (Academic Press, 2016). doi:10.1016/C2013-0-19160-X. It
  builds occupancy and abundance models that separate the ecological process
  from imperfect detection, and it suits an ecologist with survey data.
- **Hobbs & Hooten 2025.** Hobbs & Hooten, *Bayesian Models: A Statistical
  Primer for Ecologists* (2nd ed., Princeton University Press, 2025). ISBN
  978-0-691-25012-0. It teaches model building from probability and
  likelihood to hierarchical models with little code, and it is the one book
  for an ecologist new to Bayesian modeling.
- **Hooten & Hobbs 2015.** Hooten & Hobbs, "A guide to Bayesian model
  selection for ecologists," *Ecological Monographs* 85(1):3–28.
  doi:10.1890/14-0661.1. It explains predictive model comparison,
  cross-validation, and regularization for ecologists, and it serves as their
  short applied paper.
- **Dietze 2017.** Dietze, *Ecological Forecasting* (Princeton University
  Press, 2017). doi:10.1515/9781400885459. It treats state-space models, data
  fusion, and the propagation and partitioning of forecast uncertainty for
  ecological systems.

Epidemiology and public health:

- **Grinsztajn et al. 2021.** Grinsztajn, Semenova, Margossian & Riou,
  "Bayesian workflow for disease transmission modeling in Stan," *Statistics
  in Medicine* 40(27):6209–6234. doi:10.1002/sim.9164, arXiv:2006.02985. It
  fits ODE compartmental models, from a simple SIR model to a model of
  SARS-CoV-2, and uses simulations to check the model and the priors. It
  suits any user whose model is a system of ODEs.
- **Gelman & Carpenter 2020.** Gelman & Carpenter, "Bayesian analysis of tests
  with unknown specificity and sensitivity," *Journal of the Royal Statistical
  Society Series C* 69(5):1269–1283. doi:10.1111/rssc.12435. It estimates a
  prevalence from a survey with an imperfect test, with hierarchical priors on
  the test's accuracy and poststratification, and it is a short applied
  example of measurement error.
- Also on other shelves: Moraga 2019 and Morris et al. 2019 for disease
  mapping, and Hernán & Robins 2020 for causal questions.

Social sciences and education: Gelman et al. 2013, whose chapter 5 fits a
hierarchical model to the eight-schools study of coaching for a standardized
test, and Gelman & Hill 2007, whose multilevel chapters model grouped data from
the social sciences, such as election surveys.

Cognitive science: Schad, Betancourt & Vasishth 2021, in the entry Betancourt
2020.
