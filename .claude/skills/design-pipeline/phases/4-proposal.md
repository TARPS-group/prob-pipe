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
   estimate or predict, with their uncertainty. Mark each estimand that
   compares two models, such as whether overdispersion changes a conclusion,
   because step 2 then plans both models.
2. **Model sequence.** Plan the sequence of models of RESOURCES.md stage A,
   and design version 1 in detail around the first model or two.
3. **Generative model.** Describe the data-generating process, including the
   measurement process: the parameters, the priors, and the likelihood or the
   simulator. Justify each prior, ideally from domain knowledge or the user's
   papers. Then choose the route of the section "Bring any model" of
   CAPABILITIES.md by which the model enters ProbPipe:
   - The default is the route of the code the user already has, such as their
     PyMC model, Stan program, or simulator.
   - A model of standard statistical structure written for this project
     defaults to PyMC or Stan. Examples are a regression with random effects,
     such as a generalized linear mixed model, and a spatial model with a CAR,
     ICAR, or BYM prior. Choose the one the user is more comfortable with, by
     the background profile: Stan for a user of Stan or brms, and PyMC for a
     user who works in Python or whose computer cannot compile Stan's C++
     programs.
   - A custom, mechanistic, or simulator-based model, or a model whose code
     is already in JAX, defaults to the route in JAX and ProbPipe's laws.
   - Name the other routes to the user, with what each allows and rules out,
     from the last column of its row.

   Map each part to ProbPipe objects by the rows of CAPABILITIES.md for stages
   A and B, and record in the design document each check that the chosen
   route rules out.
4. **Fast first fits.** Plan the first fits to be fast, as RESOURCES.md
   stage E advises: the first model of the sequence, run in the smoke stage
   that [Write the code](#write-the-code) defines. A smoke run shows that the
   code runs, and step 4 of [Check and run the code](#check-and-run-the-code)
   says how to label its results.
5. **Checks before fitting.** Plan the prior predictive checks, by the rows
   for stage C, and the validation on realistic simulated data that
   RESOURCES.md stage D defines, by the rows for stage D. The validation
   runs in the validate stage of [Write the code](#write-the-code), which
   gates the fit to the real data.
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

When a step needs a component that ProbPipe does not connect to, such as a
Gaussian-process package or a domain simulator, offer the options of the
ground rule "Offer to build an interface to a missing component" of
[INSTRUCTIONS.md](../INSTRUCTIONS.md#ground-rules).

In the quick start mode of the [ground rules](../INSTRUCTIONS.md#ground-rules),
step 2 plans the baseline and the elaboration that the mode names, and step 8
is one posterior predictive check of each model. The mode decides which other
stages to defer, such as the cross-validation of step 8 and step 10.

When the diagnostics of step 7 fail in a run of the validate stage, take the
remedies of the bullet "Remedy divergences in this order" of RESOURCES.md
stage E yourself, in order, before you report the grade *failed*. Write each
remedy into the pipeline as a computational revision, and rerun the validate
stage by rule 5 of the
[rules for user materials](../INSTRUCTIONS.md#user-materials-and-code). A
remedy that is a setting, such as the target acceptance rate, can apply to the
failed fits alone, as a sweep over their datasets, and the fit to the real
data then uses it. Report each remedy you tried with its result, and grade the
validation by RESOURCES.md stage D after the last one.

When the checks of step 8 fail, or the diagnostics of a fit to the real data
fail, return to step 3 and revise the model before you tune the algorithm, by
the folk theorem of RESOURCES.md stage E. Each remedy and each return is a
revision, which follows the
[rules for revisions](5-feedback.md#rules-for-revisions). Only the validate
and the fit stage decide a revision, because the budgets of a smoke run are
too small to judge the diagnostics and the checks.

## Write the code

Write the pipeline as a runnable skeleton in the style of the tutorials, in the
form the user works in: the notebook `<workspace>/pipeline.ipynb` for a user
who works in notebooks, and `<workspace>/pipeline.py` otherwise. A notebook
starts with an install cell in the style of the tutorials' Colab cell. On
Google Colab, the cell installs ProbPipe at the commit that
`<corpus>/MANIFEST.md` records, or at its branch when that commit is not on
GitHub, so that Colab runs the code that `check_symbols.py` checked.
`git -C <checkout> branch -r --contains <commit>` lists the branches on GitHub
that hold the commit, as of the last `git fetch`. The cell
then asks the user to upload the data file with `google.colab.files.upload()`,
and the pipeline reads each data file by a path relative to the notebook. The
notebook's metadata names the project's kernel, which
[Check and run the code](#check-and-run-the-code) registers. The pipeline loads
the user's data or simulates it, and each pipeline stage is a labeled section
that matches the design document. Each stage shows a figure where one helps,
such as the prior predictive draws or the posterior predictive check, with a
one-sentence comment on how to read it.

Write the code close to the mathematics, and in batches:

1. Write the model as its generative equations, with the notation of the
   design document as the variable names, such as `mu`, `tau`, and `z` for
   μ, τ, and z_j.
2. Batch each repeated computation. Draw all the simulated datasets of the
   validation in one batched draw, fit them in one sweep, and compute the
   coverage, the z-scores, and the summaries as array operations, by the
   batched recipe of the row "Validate the computation on realistic simulated
   data" of [CAPABILITIES.md](../CAPABILITIES.md). Use a Python loop only to
   present results, such as to print the rows of a table or to draw the
   panels of a figure.
3. Keep the auxiliary code in two modules beside the pipeline, which the
   pipeline imports. `<workspace>/report.py` holds the helpers that print,
   plot, and write results. `<workspace>/sim_data.py` draws the simulated
   data at the real design from the model that the pipeline passes to it, and
   masks it to the real pattern of missing values. The model, the inference,
   and the checks stay in the pipeline itself.

The code runs in three stages, which the environment variable `STAGE`
selects:

1. `STAGE=smoke` is the default and runs every section at small budgets,
   such as a few draws, chains, or training epochs. It fits data simulated
   from the model at the structure of the real data, such as its sizes, its
   groups, and its pattern of missing values, in place of the real data, so
   the code fixes that the smoke runs need stay out of the record of
   revisions. A smoke run on a subsample of the real data is an
   option that the user approves, and never for data that
   [Phase 2](2-materials.md#data) found sensitive.
2. `STAGE=validate` runs the sections before the fit to the real data at
   their full budgets, such as the prior predictive check and the validation
   on realistic simulated data. The validation writes its table, its verdict,
   and the design version to `<workspace>/validation.json`.
3. `STAGE=fit` runs the fit to the real data and the sections after it. Its
   first step reads `<workspace>/validation.json` and stops with an error
   unless the validation of the current design version passed.

Every stage runs the data step first. A notebook sets the stage and the
budgets in its first code cell after the install cell, so a user who runs it
in Jupyter or on Colab sets the stage there. Each stage writes its results,
such as the diagnostics and the estimates, to files in the workspace as it
computes them, so a run that a failed step stops keeps the results computed
before it.

## Check and run the code

Take these steps before you show the code, and again after each revision:

1. Run `<python> <assistant>/check_symbols.py <workspace>/pipeline.py`, or the
   same command on the notebook, and fix each problem that it prints. It
   checks the file and each module beside it that the file imports, such as
   `<workspace>/report.py`. It checks that each ProbPipe name resolves, that
   each call's arguments bind to the signature, and that the code calls no
   export that is not implemented. It checks no meaning, so read the docstring
   of each call whose meaning you are unsure of.
2. If ProbPipe imports here, run the smoke stage by the
   [rules for running code](../INSTRUCTIONS.md#user-materials-and-code), and
   fix what breaks. Otherwise, say that the code has not been run. The
   commands of a stage are these, shown for the validate stage:
   - a script: `STAGE=validate <python> <workspace>/pipeline.py`;
   - a notebook:
     `STAGE=validate <python> -m nbconvert --to notebook --execute <workspace>/pipeline.ipynb --ExecutePreprocessor.kernel_name=<kernel> --output pipeline.validate`,
     which writes the executed copy `<workspace>/pipeline.validate.ipynb`.

   nbconvert writes no executed copy of a run that a failed cell stops. The
   smoke and validate stages fit no real data, so to read the error of a
   failed run of one of them, rerun it with `--allow-errors` and search its
   executed copy for the outputs whose type is `error`. Run the fit stage
   without `--allow-errors`, because that flag runs each cell after a failed
   one, the fit after a failed check of the validation included.

   Name the kernel `<kernel>` of the project `probpipe-<project-slug>`, and
   record it in project memory. Before the first notebook run, list the kernels with `<python> -m jupyter kernelspec list`,
   and choose another name when that one is taken. Then register the kernel
   with `<python> -m ipykernel install --user --name <kernel>`, after the user
   approves, and add `--env PYTHONPATH <checkout>` when ProbPipe imports
   through `PYTHONPATH`. When nbconvert or ipykernel does not import, ask
   before you install it.
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

For a user whose rating in Bayesian modeling is New or Familiar, open with two
plain sentences on what the analysis will tell them. Then list the stages as a
numbered list with one sentence each, and give at most three key design
decisions. The design
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
