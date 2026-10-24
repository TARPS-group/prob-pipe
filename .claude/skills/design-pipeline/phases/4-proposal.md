# Phase 4 — Initial pipeline proposal

Read Part 1 of [RESOURCES.md](../RESOURCES.md) in full before you start, and
read the section of its Part 2 for the kind of model when you choose a resource
to cite or to point the user to. Design the first version of the pipeline by
[DESIGN-DOC-TEMPLATE.md](../DESIGN-DOC-TEMPLATE.md) and the workflow checklist
of RESOURCES.md Part 1. Follow the decisions and preferences in memory, and
propose nothing that memory records as ruled out. The ProbPipe calls for each
step are in the rows of [CAPABILITIES.md](../CAPABILITIES.md) for the step's
checklist stage. Take the steps in this order by default, because each step
constrains the next. When the user's materials already settle a later step,
such as an established model or a required method, start from them and check
the earlier steps against them:

1. **Estimands.** Restate the scientific goals as the specific quantities to
   estimate or predict, with their uncertainty.
2. **Model sequence.** Plan the sequence of models of RESOURCES.md stage A,
   and design version 1 in detail around the first model or two.
3. **Generative model.** Describe the data-generating process, including the
   measurement process: the parameters, the priors, and the likelihood or the
   simulator. Justify each prior, ideally from domain knowledge or the user's
   papers. Then choose the route of the section "Bring any model" of
   CAPABILITIES.md by which the model enters ProbPipe:
   - The default is the route of the code the user already has, such as their
     PyMC model, Stan program, or simulator.
   - A model written for this project defaults to the route in JAX and
     ProbPipe's laws, which ProbPipe's gradient methods and predictive checks
     apply to. When no tutorial or user-guide example shows its structure,
     PyMC or Stan is an alternative whose own documentation shows it, and the
     missing example is a ProbPipe gap.
   - Tell the user what each candidate route allows and rules out, from its
     last column, when the routes differ for this model.

   Map each part to ProbPipe objects by the rows of CAPABILITIES.md for stages
   A and B, and record in the design document each check that the chosen
   route rules out.
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
   default or to name another method that applies, and why, and tell the user
   which alternatives they have. Consider an alternative, such
   as a gradient-free, variational, or amortized method, when the likelihood
   is intractable, non-differentiable, or too expensive. For many
   observations, choose by the bullet "Many observations" of stage E, and for
   a simulator or an expensive model, by stage I, from the scale that
   [Phase 1](1-intake.md) records. Plan the validation that stage E requires
   for an approximate method, and the validation that stage I requires for
   an amortized method or an emulator.
7. **Computational diagnostics.** Plan the diagnostics of RESOURCES.md stage
   E, by the rows of CAPABILITIES.md for stage E, such as "Diagnose MCMC" for
   an MCMC fit, and what to do when each one fails.
8. **Model criticism.** Plan three kinds of check:
   - posterior predictive checks, with test statistics chosen for this
     problem;
   - the cross-validation scheme that matches the prediction task, by
     RESOURCES.md stage F;
   - the validation metrics that apply.
9. **Downstream quantities.** Plan how posterior uncertainty propagates into
   the estimands and into the user's own downstream code, such as a
   prediction or a decision rule, by the row "Compute downstream quantities".
10. **Robustness and extensions.** Plan those that the goals call for:
    - sensitivity analysis;
    - model comparison or stacking;
    - bagged posteriors for misspecification;
    - sequential updating for streaming data;
    - emulators for expensive simulators;
    - Prefect or Ray for scale.

On the fast path of the [ground rules](../INSTRUCTIONS.md#ground-rules), step
8 is one posterior predictive check, and the cross-validation of step 8 and all
of step 10 are deferred.

When the diagnostics of step 7 or the checks of step 8 fail, return to step 3
and revise the model before you tune the algorithm, by the folk theorem of
RESOURCES.md stage E. Each return is a revision, which follows the
[rules for revisions](5-feedback.md#rules-for-revisions).

## Write the code

Write the pipeline as a runnable skeleton in the style of the tutorials, in the
form the user works in: the notebook `<workspace>/pipeline.ipynb` for a user
who works in notebooks, and `<workspace>/pipeline.py` otherwise. A notebook
starts with an install cell in the style of the tutorials' Colab cell, which
installs ProbPipe when the notebook runs on Google Colab. The pipeline loads
the user's data or simulates it, and each pipeline stage is a labeled section
that matches the design document. Each stage shows a figure where one helps,
such as the prior predictive draws or the posterior predictive check, with a
one-sentence comment on how to read it.

The code has a smoke mode for fast trial runs. It reads the environment
variable `SMOKE`, and when `SMOKE=1` it uses small budgets: a few draws,
chains, simulations, and training epochs, and a subsample of the data. A
notebook sets the budgets from the same variable in its first code cell after
the install cell.

## Check and run the code

Take these steps before you show the code, and again after each revision:

1. Run `<python> <assistant>/check_symbols.py <workspace>/pipeline.py`, or the
   same command on the notebook, and fix each problem that it prints. It
   checks that each ProbPipe name resolves and that each call's arguments bind
   to the signature, so read the docstring of each call whose meaning you are
   unsure of.
2. If ProbPipe imports here, run the pipeline with `SMOKE=1` by the
   [rules for running code](../INSTRUCTIONS.md#user-materials-and-code), and
   fix what breaks. Otherwise, say that the code has not been run.
3. Record the inference method that ran, as the row "Condition on the data" of
   CAPABILITIES.md says to read it, in the design document and in project
   memory. When it differs from the method you planned, say so, and revise the
   plan or name the method.
4. Label each result of a smoke run "smoke run, not for inference" wherever
   you show it or record it, the fit log of the design document included.

## Present the proposal

Present the proposal at the depth that the profile calls for. Draw a Mermaid
diagram only in an interface that renders it. Give the user the steps of
section 11 of the design document, so that they can run the code.

For a user new to Bayesian modeling, open with two plain sentences on what the
analysis will tell them. Then list the stages as a numbered list with one
sentence each, and give at most three key design decisions. The design
document is your record of the work, so point the user to it in place of
presenting it.

For other users, lead with a short overview, which is a stage diagram with one
sentence per stage. Then give the key design decisions, each with the
alternative you considered and the reason for your choice. List the assumptions that the user
should check, and the open questions. Point to the files for the full detail,
and to one or two resources of [RESOURCES.md](../RESOURCES.md) that suit the
user's background and the hardest part of the design. End with targeted
feedback questions about the decisions you are least sure of. Update project
memory.
