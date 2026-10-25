# Phase 4 — Proposal for an existing pipeline

Follow this file in place of the numbered steps of
[phases/4-proposal.md](4-proposal.md) when the user brings a pipeline that
already runs and wants to bring it into ProbPipe or to migrate it to ProbPipe.
Such a pipeline is, for example, a PyMC or Stan model with its fitting script,
a custom pipeline in JAX, NumPy, or SciPy, or a simulator with the script that
fits it. The sections "Write the code", "Check and run the code", and "Present
the proposal" of phases/4-proposal.md apply to it, and so do the
[rules for revisions](5-feedback.md#rules-for-revisions).

1. **Take an inventory of the pipeline.** Read its code in full, by the
   [rules for user materials](../INSTRUCTIONS.md#user-materials-and-code), and
   record what it does in `<workspace>/notes/model-notes.md`, in five parts:
   1. inputs: the data files and each preprocessing step;
   2. model: the six things that the section "Models" of
      [Phase 2](2-materials.md#models) extracts;
   3. inference: the method, its settings, and its seeds;
   4. outputs: the estimates, tables, and figures, with the estimands they
      report;
   5. checks: the diagnostics and the checks that it runs.

   Note for each part the route of the section "Bring any model" of
   [CAPABILITIES.md](../CAPABILITIES.md) by which it enters ProbPipe. Skip
   the elicitation of plausible values, by the section "Data" of
   [Phase 2](2-materials.md#data).
2. **Offer integration, migration, or a verified re-implementation.** Put
   the options to the user as a choice, with their trade-offs:
   1. *Integrate the pipeline as it stands.* The model enters by its route,
      such as `PyMCModel` for a PyMC model or `StanModel` for a Stan program,
      the other code runs as `function` steps that call it, and draws that it
      computes enter by the row "Use draws from another program". The user's
      code stays the reference, and the change is smallest. What ProbPipe can
      add depends on the route, by the last column of its row.
   2. *Migrate it step by step.* Rewrite one part at a time in ProbPipe, such
      as the data steps as `function` steps, then the model by a route, then
      the inference with `condition_on`, and repeat the reproduction check of
      step 3 after each part. It takes more work, and it gives every check of
      the workflow and the provenance of each step.
   3. *Re-implement the model in JAX and verify it against the original.*
      This option serves a user who wants gradients or speed from a model
      whose code JAX cannot trace, such as an R function or a compiled
      simulator. Wrapped as it stands, such a model has no gradient. Only a
      gradient-free method such as `blackjax_rwmh` then fits it, and such a
      method mixes slowly with many correlated parameters. Transcribe the
      model line by line into JAX, integrate the rest of the pipeline by
      option 1, and keep the original model as the reference. Before any
      fit, a deterministic agreement check runs the original and the
      transcription on the same inputs and compares their outputs and their
      log densities at many parameter sets, with a tolerance stated before
      the check, such as a relative difference of 1e-8 in double precision.
      The sets include these:
      - draws over all the parameters, the fixed ones included;
      - each parameter at the bounds of its prior's support;
      - values on both sides of each threshold in the code, such as a
        dry-soil switch or the day of a phenology switch.

      A log-density difference that is one constant at every set, such as
      one from another log-gamma routine, leaves the posterior unchanged.
      Record the check in section 4f of the design document.

   Recommend integration by default. Recommend migration when the user asks
   for it or needs a check that the route of integration rules out, and
   recommend option 3 when the user wants gradients or speed from a model
   that JAX cannot trace. Record the choice as a decision.
   Keep each interface as simple as the ground rule "Offer to build an
   interface to a missing component" of
   [INSTRUCTIONS.md](../INSTRUCTIONS.md#ground-rules) requires.

   A pipeline in R takes the options that the ground rule gives for an R
   tool, because ProbPipe plans no interface to R:
   - re-implement the model by a route, such as a brms model as its Stan
     program, or a model written in R by option 3;
   - wrap the user's case, such as an R script that runs as a `function`
     step;
   - bring in the draws that the R pipeline computes.
3. **Reproduce the results before you change anything.** Show that ProbPipe
   reproduces the original pipeline's results before any change to its model,
   its data steps, or its inference:
   1. Take the original's results from its saved outputs, from a run of the
      original that the user consents to by rule 5 of the rules for user
      materials, or from a run by the user. Keep its draws when it computes
      draws.
   2. Write the reproduction in `<workspace>/reproduce.ipynb`, or in
      `<workspace>/reproduce.py` for a user who works in scripts. It runs the
      integrated or migrated pipeline on the same data, with the same priors,
      and with the same inference method and settings where ProbPipe has the
      method, and it writes its results to `<workspace>/runs/reproduce/`.
      When ProbPipe has no counterpart of the original's method, such as a
      differential-evolution sampler, fit with the default method of
      [RESOURCES.md](../RESOURCES.md) stage E, and compare the two posteriors
      by substep 3, which judges the draws of any method.
      Check it and run it by the steps of
      [Check and run the code](4-proposal.md#check-and-run-the-code), with a
      smoke run on simulated data first. Its run on the real data is the first
      fit to the real data, which needs the user's consent by rule 5, and
      which rule 6 allows before the validation.
   3. Compare the estimands. For each one, compute the median and the 2.5% and
      97.5% quantiles of the draws of each run, and compute the correlation of
      each pair of parameters. The error of each value is approximately the
      mean of its influence function over the draws, so the Monte Carlo
      standard error (MCSE) of the value is the MCSE of that mean. The
      influence functions are these:
      - the quantile q_p: (p − 1{x ≤ q_p})/f(q_p), with f a density estimate
        of x at q_p;
      - the correlation r of x and y: z_x z_y − r(z_x² + z_y²)/2, with z_x and
        z_y the standardized draws.

      The MCSE then depends on which chains are independent:
      1. Chains that run independently, as NUTS runs them: ArviZ's `mcse` on
         the draws arranged as (chain, draw) with `method="quantile"` and
         `prob=` gives the MCSE of a quantile, and `mcse` with
         `method="mean"` on the influence-function series gives the MCSE of a
         correlation.
      2. Chains that share information within a run, as in a population
         sampler such as DEzs or an ensemble sampler such as emcee: the runs
         are the independent units. Average the influence-function series
         over the chains of each run at each iteration, compute the MCSE of
         that average from its autocorrelation or by batch means, and combine
         the runs as independent. With one run, the batch means of that run
         give the MCSE.

      The z-score of a comparison is the difference of the two values over
      the square root of the sum of their squared MCSEs. Judge the M z-scores
      together by the bounds b₂(M) and b₃(M) of
      [RESOURCES.md](../RESOURCES.md) stage D step 6. The runs agree when
      every |z| is within b₂(M). Each |z| between b₂(M) and b₃(M) is a note
      that names its estimand, and an |z| beyond b₃(M) is a disagreement.
      Report the mean, the standard deviation, and the largest |z| beside the
      verdict, and report a set within the bounds as consistent with Monte
      Carlo error. The bounds are defaults, and a project that sets others
      records them as a decision. A deterministic step, such as an optimizer
      or a wrapped function, agrees when its outputs match within the
      tolerance of the original's own solver.
   4. Show the comparison as a table of one row per estimand, with both
      results in the form of rule 4 of
      [Messages](../INSTRUCTIONS.md#messages) and whether they agree, and give
      the diagnostics of stage E of each run beside it. Record it in section
      4f of the design document, and in the fit log with the stage
      `reproduce`. When an estimand disagrees, find the cause before you
      change anything, such as another prior, parameterization, data step, or
      method, or a run that has not converged.
4. **Add what ProbPipe brings, one piece at a time.** Add the pieces that the
   user's goals need, one per design version, in this order by default:
   1. the validation on realistic simulated data of
      [RESOURCES.md](../RESOURCES.md) stage D;
   2. the diagnostics of stage E;
   3. the predictive checks of stage F;
   4. the propagation of the posterior into the downstream quantities. For
      a forecast, this piece starts with the audit of the original's forecast
      by the bullet "Audit the uncertainty of a forecast" of stage H;
   5. the provenance of the data steps.

   An addition that leaves the model as it is belongs to the class *added
   after the fit* of the rules for revisions, because the original pipeline
   has already fitted the real data. After an addition that changes the
   model, the data steps, or the inference, repeat the reproduction check, or
   say why the estimates should now differ. From the first addition on, the
   pipeline runs in the stages of
   [Write the code](4-proposal.md#write-the-code), and its fit stage waits for
   the validation.
