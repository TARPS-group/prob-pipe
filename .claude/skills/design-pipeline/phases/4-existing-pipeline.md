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
   [CAPABILITIES.md](../CAPABILITIES.md) by which it enters ProbPipe.
2. **Offer integration or migration.** Put the two to the user as a choice,
   with their trade-offs:
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

   Recommend integration, unless the user asks to migrate or needs a check
   that the route of integration rules out. Record the choice as a decision.
   Keep each interface as simple as the ground rule "Offer to build an
   interface to a missing component" of
   [INSTRUCTIONS.md](../INSTRUCTIONS.md#ground-rules) requires.

   A pipeline in R takes the options that the ground rule gives for an R
   tool, because ProbPipe plans no interface to R:
   - re-implement the model by a route, such as a brms model as its Stan
     program;
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
      Check it and run it by the steps of
      [Check and run the code](4-proposal.md#check-and-run-the-code), with a
      smoke run on simulated data first. Its run on the real data is the first
      fit to the real data, which needs the user's consent by rule 5, and
      which rule 6 allows before the validation.
   3. Compare the estimands. For each one, compute the median and the 2.5% and
      97.5% quantiles of the draws of each run, and the Monte Carlo standard
      error (MCSE) of each, with ArviZ's `mcse` on the draws arranged as
      (chain, draw), with `method="median"` or with `method="quantile"` and
      `prob=`. Two values agree when their difference is at most three times
      the square root of the sum of their squared MCSEs. The bound is a
      default, and a project that sets another bound records it as a
      decision. A deterministic step, such as an optimizer or a wrapped
      function, agrees when its outputs match within the tolerance of the
      original's own solver.
   4. Show the comparison as a table of one row per estimand, with both
      results in the form of rule 4 of
      [Messages](../INSTRUCTIONS.md#messages) and whether they agree. Record
      it in section 4f of the design document, and in the fit log with the
      stage `reproduce`. When an estimand disagrees, find the cause before you
      change anything, such as another prior, parameterization, data step, or
      method, or a run that has not converged.
4. **Add what ProbPipe brings, one piece at a time.** Add the pieces that the
   user's goals need, one per design version, in this order by default:
   1. the validation on realistic simulated data of
      [RESOURCES.md](../RESOURCES.md) stage D;
   2. the diagnostics of stage E;
   3. the predictive checks of stage F;
   4. the propagation of the posterior into the downstream quantities;
   5. the provenance of the data steps.

   An addition that leaves the model as it is belongs to the class *added
   after the fit* of the rules for revisions, because the original pipeline
   has already fitted the real data. After an addition that changes the
   model, the data steps, or the inference, repeat the reproduction check, or
   say why the estimates should now differ. From the first addition on, the
   pipeline runs in the stages of
   [Write the code](4-proposal.md#write-the-code), and its fit stage waits for
   the validation.
