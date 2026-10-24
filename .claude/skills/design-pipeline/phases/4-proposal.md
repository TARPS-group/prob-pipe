# Phase 4 — Initial pipeline proposal

Read [RESOURCES.md](../RESOURCES.md) in full before you start. Design the
first version of the pipeline by
[DESIGN-DOC-TEMPLATE.md](../DESIGN-DOC-TEMPLATE.md) and the workflow checklist
of RESOURCES.md Part 1. Follow the decisions and preferences in memory, and
propose nothing that memory records as ruled out. The ProbPipe calls for each
step are in the rows of [CAPABILITIES.md](../CAPABILITIES.md) for the step's
checklist stage. Take the steps in this order, because each step constrains
the next:

1. **Estimands.** Restate the scientific goals as the specific quantities to
   estimate or predict, with their uncertainty.
2. **Model sequence.** Plan a sequence that starts from the simplest model
   that addresses the question and lists the expansions you expect to need.
   Design version 1 in detail around the first model or two.
3. **Generative model.** Describe the data-generating process, including the
   measurement process: the parameters, the priors, and the likelihood. Justify
   each prior, ideally from domain knowledge or the user's papers. Map each
   part to ProbPipe objects by the rows of CAPABILITIES.md for stages A and B:
   the independent priors, a prior that depends on other parameters, the
   likelihood, and the joint model, or a wrapped Stan or PyMC model.
4. **Fast first fits.** Plan the first fits to be fast, as RESOURCES.md
   stage E advises: the first model of the sequence, run in the smoke mode
   that [Write the code](#write-the-code) defines. A smoke run shows that the
   code runs, and step 4 of [Check and run the code](#check-and-run-the-code)
   says how to label its results.
5. **Checks before fitting.** Plan the prior predictive checks, by the rows
   for stage C, and the validation on realistic simulated data that
   RESOURCES.md stage D defines, by the rows for stage D. Plan to run the
   validation before the fit to the real data.
6. **Inference.** Choose the method by RESOURCES.md stage E. Find the method
   that the conditioning selects by default and the reason each other method
   declines, by the row "Condition on the data". State whether to keep the
   default or to name another method, and why. Consider an alternative, such
   as a gradient-free, variational, or amortized method, when the likelihood
   is intractable, non-differentiable, or too expensive. For many
   observations, choose by the bullet "Many observations" of stage E, and for
   a simulator or an expensive model, by stage I, from the scale that
   [Phase 1](1-intake.md) records. Plan the validation that stage E requires
   for an approximate method, and the validation that stage I requires for
   an amortized method or an emulator.
7. **Computational diagnostics.** Plan the diagnostics of RESOURCES.md stage
   E, by the row "Diagnose MCMC", and what to do when each one fails.
8. **Model criticism.** Plan three kinds of check:
   - posterior predictive checks, with test statistics chosen for this
     problem;
   - the cross-validation scheme that matches the prediction task, by
     RESOURCES.md stage F;
   - the validation metrics that apply.
9. **Downstream quantities.** Plan how posterior uncertainty propagates into
   the estimands, by the row "Compute downstream quantities": a function
   called on coupled marginals of the posterior, which draws them jointly.
10. **Robustness and extensions.** Plan those that the goals call for:
    - sensitivity analysis;
    - model comparison or stacking;
    - bagged posteriors for misspecification;
    - sequential updating for streaming data;
    - emulators for expensive simulators;
    - Prefect or Ray for scale.

When the diagnostics of step 7 or the checks of step 8 fail, return to step 3
and revise the model before you tune the algorithm, by the folk theorem of
RESOURCES.md stage E. Each return is a revision, which follows the
[rules for revisions](5-feedback.md#rules-for-revisions).

## Write the code

Write `<workspace>/pipeline.py` as a runnable skeleton in the style of the
tutorials, or a notebook when the user works in notebooks. It loads the user's
data or simulates it, and each pipeline stage is a labeled section that matches
the design document.

The code has a smoke mode for fast trial runs. It reads the environment
variable `SMOKE`, and when `SMOKE=1` it uses small budgets: a few draws,
chains, simulations, and training epochs, and a subsample of the data. A
notebook sets the budgets in its first code cell from the same variable.

## Check and run the code

Take these steps before you show the code, and again after each revision:

1. Run `python <assistant>/check_symbols.py <workspace>/pipeline.py`, or the
   same command on the notebook, with the Python of the ProbPipe environment,
   and fix each problem that it prints. It
   checks that each ProbPipe name resolves and that each call's arguments bind
   to the signature, so read the docstring of each call whose meaning you are
   unsure of.
2. If ProbPipe imports here, ask the user whether you may run the pipeline.
   Run it first with `SMOKE=1` and a timeout, such as ten minutes, and fix
   what breaks. Otherwise, say that the code has not been run.
3. Record the inference method that ran, as the row "Condition on the data" of
   CAPABILITIES.md says to read it, in the design document and in project
   memory. When it differs from the method you planned, say so, and revise the
   plan or name the method.
4. Label each result of a smoke run "smoke run, not for inference" wherever
   you show it or record it, the fit log of the design document included.

## Present the proposal

Present the proposal at the depth that the profile calls for. Lead with a
short overview, which is a stage diagram in text or Mermaid with one sentence
per stage. Then give the key design decisions, each with the alternative you
considered and the reason for your choice. List the assumptions that the user
should check, and the open questions. Point to the files for the full detail,
and to one or two resources of [RESOURCES.md](../RESOURCES.md) that suit the
user's background and the hardest part of the design. End with targeted
feedback questions about the decisions you are least sure of. Update project
memory.
