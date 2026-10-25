# Phase 4 — Initial pipeline proposal

When the user brings a pipeline that already runs, follow
[phases/4-existing-pipeline.md](4-existing-pipeline.md) in place of the
numbered steps below.

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
   papers, and otherwise by the default priors of RESOURCES.md stage B. Then
   choose the route of the section "Bring any model" of
   CAPABILITIES.md by which the model enters ProbPipe:
   - The default is the route of the code the user already has, such as their
     PyMC model, Stan program, or simulator.
   - A model written for this project that `glm_likelihood` expresses
     defaults to it, by the row "Use the
     built-in generalized linear model" of CAPABILITIES.md. Such a model is a
     regression on fixed covariates with a normal, Bernoulli, or Poisson
     response and an invertible link. ProbPipe's predictive checks, LOO, and
     stochastic-gradient MCMC apply to it directly, whereas the PyMC and Stan
     routes compute the first two in the pipeline's own code.
   - A model of standard statistical structure beyond `glm_likelihood`
     written for this project defaults to PyMC or Stan. Examples are a
     regression with random effects, such as a generalized linear mixed
     model, a count model with an exposure offset or a negative binomial
     response, and a spatial model with a CAR, ICAR, or BYM prior. A computer that cannot compile Stan programs rules
     Stan out, so a user of brms whose compiler fails gets PyMC. Otherwise
     choose the one the user is more comfortable with, by the working
     preferences and the experience in person memory: Stan for a user who
     works in Stan or brms, and PyMC for a user who works in Python. Stan needs its backend and a C++ toolchain, by the row "A Stan
     program" of CAPABILITIES.md. When the backends that
     [Phase 0](0-load.md) recorded lack it, ask before you install it, by
     rule 8 of the
     [rules for user materials](../INSTRUCTIONS.md#user-materials-and-code),
     or choose PyMC.
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
is one posterior predictive check of each model and the cross-validation that
the mode keeps for an estimand.

When the diagnostics of step 7 fail in a run of the validate stage, take the
remedies of the bullet "Remedy divergences in this order" of RESOURCES.md
stage E yourself, in order, before you report the grade *failed*. Write each
remedy into the pipeline as a computational revision, and rerun the validate
stage by rule 5 of the
[rules for user materials](../INSTRUCTIONS.md#user-materials-and-code).
Report each remedy you tried with its result, and grade the
validation by RESOURCES.md stage D after the last one. A time budget that the
user states, such as "about an hour", allows one validate run and at most one
remedy. When the grade after them is *failed*, the user chooses at once
between more time, with its estimate, and the provisional results of
RESOURCES.md stage D.

When the checks of step 8 fail, or the diagnostics of a fit to the real data
fail, return to step 3 and revise the model before you tune the algorithm, by
the folk theorem of RESOURCES.md stage E. Each remedy and each return is a
revision, which follows the
[rules for revisions](5-feedback.md#rules-for-revisions). Only the validate
and the fit stage decide a revision, because the budgets of a smoke run are
too small to judge the diagnostics and the checks.

## Write the code

Write the pipeline in the form the user works in: the notebook
`<workspace>/pipeline.ipynb` for a user who names notebooks or Jupyter as the
way they work, and the script `<workspace>/pipeline.py` for one who names
scripts. When the user's answers name neither, ask in one line.

Start from the worked example `<assistant>/templates/pipeline_skeleton.py`
and its modules `pipeline_support.py`, `report.py`, and `sim_data.py`. Its
comments give the reason for each part, and they say when to keep each block
tagged `# OPTIONAL: <name>`. Pare it down in four steps:

1. Copy the example and its modules to the workspace.
2. Keep each optional block that the design document calls for, and delete
   the others: each line that ends in the tag, and each section from its
   tagged header to the next `# ---` header, in the pipeline and in
   `report.py`.
3. Replace the example's data, models, and estimands with those of the
   design.
4. Run `check_symbols.py` by [Check and run the code](#check-and-run-the-code),
   and delete what only a deleted block used, such as an import.

A notebook starts with an install cell in the style of the tutorials' Colab
cell. On Google Colab, the cell installs ProbPipe at the commit that
`<corpus>/MANIFEST.md` records, or at its branch when that commit is not on
GitHub, so that Colab runs the code that `check_symbols.py` checked.
`git -C <checkout> branch -r --contains <commit>` lists the branches on GitHub
that hold the commit, as of the last `git fetch`. The cell then asks the user
to upload the data file and the modules beside the notebook with
`google.colab.files.upload()`, and the pipeline reads each file by a path
relative to the notebook. A script reads each file by a path relative to
`Path(__file__).parent`, so that it runs from any directory. The notebook's
metadata names the project's kernel, which
[Check and run the code](#check-and-run-the-code) registers, with the language
`python`, so that Colab runs the notebook on its own Python kernel. Each step
of the pipeline matches a subsection of section 4 of the design document, and
shows a figure where one helps, with a one-sentence comment on how to read it.
The notebook's first text cell, or the docstring of a script, says where to
change each choice, such as "the priors are in `build_model()`", so that the
user can find, defend, and edit each choice in code they did not write.

Structure a script as a short program whose top function reads as the
pipeline (Wilson et al. 2017):

1. **`main()` first.** The script defines `main()` right after its imports and
   settings, and it ends with `if __name__ == "__main__": main()`. `main()`
   reads as the pipeline, one or two lines per step, and each step is a
   function whose name says what it does and that fits on one screen, of
   about 60 lines.
2. **One call for the stage.** `main()` starts with one call of
   `resolve_stage()` of `<workspace>/pipeline_support.py`, which returns the
   settings of the stage: the budgets, the steps that run, the models to fit
   with their labels, and the output folder. That module holds the gate, the
   records of the run, and the grade of the validation, so the pipeline shows
   the data, the models with their priors, and the steps, and it defines no
   classes. `main()` runs the random steps in one `workflow_run(seed=SEED)`,
   so a rerun reproduces each result.
3. **Pure steps.** A step function takes its inputs as arguments and returns
   values. The reads and writes of files happen in `main()`, in
   `pipeline_support.py`, and in `<workspace>/report.py`, which holds every
   print, table, and figure and formats each interval by rule 4 of
   [Messages](../INSTRUCTIONS.md#messages). A `report.show_*` call shows a
   step's named result and computes nothing that a later step uses.
4. **Checked data.** The data step checks the data right after the read: the
   expected columns, their types and ranges, and the codes of missing values.
   It stops at the first problem with a message that names the column.
5. **Units in names.** A variable that holds a quantity with units names the
   unit, such as `rain_mm`, and each unit conversion is one small named
   function.
6. **Close to the mathematics.** The model is written as its generative
   equations, with the design document's notation as names. Each repeated
   computation is batched, such as the validation's datasets in one sweep by
   the recipe of [CAPABILITIES.md](../CAPABILITIES.md). A Python loop
   presents results or runs steps that differ in structure, such as models
   with different parameters. `<workspace>/sim_data.py` draws the simulated data
   at the real design and masks them to the real pattern of missing values.

A notebook mirrors this structure and explains the analysis to its reader
(Rule et al. 2019). Its first code cell holds the settings and the call of
`resolve_stage()`, and each later code cell runs one step. A short text cell
before each code cell says why the step is there and what to look at in its
output. The step functions may be in `<workspace>/steps.py`, which the
notebook imports, while the models and their priors stay in a cell of the
notebook. The notebook runs from top to bottom in a fresh kernel.

The code runs in three stages, each with its budgets in the settings.
`resolve_stage()` reads the stage from the environment variable `STAGE`,
`smoke` by default. A notebook's first code cell sets it with
`os.environ.setdefault("STAGE", "smoke")`, whose default a user on Jupyter or
Colab edits.

1. `STAGE=smoke` shows that the code runs, at small budgets, such as a few
   draws, chains, or training epochs. It runs the data step, the checks of
   the pipeline's own code, the prior predictive draws, and one fit of each
   model with the steps after it, and no validation. Each fit is to one
   dataset simulated from the model at the structure of the real data, such as its sizes, its
   groups, and its pattern of missing values, so the code fixes that the
   smoke runs need stay out of the record of revisions. A smoke run on a
   subsample of the real data is an option that the user approves, and never
   for data that [Phase 2](2-materials.md#data) found sensitive. Under a
   hold of rule 5 of the
   [rules for user materials](../INSTRUCTIONS.md#user-materials-and-code),
   run it with `HOLD_SIMULATED_FITS` set to the date of the hold, and it runs
   only the data step and the checks of the code.
   The checks of the code are a few asserts in one step function, such as
   `check_code()` of the example (Wilson et al. 2014). Each unit conversion
   matches a known value, the log density matches a hand computation at one
   point, and a model re-implemented from the user's code agrees with the
   original by the agreement check of option 3 of
   [phases/4-existing-pipeline.md](4-existing-pipeline.md). A check of random
   draws passes within a statistical tolerance, such as four standard errors.
2. `STAGE=validate` runs the prior predictive check and the validation on
   realistic simulated data at their full budgets, and grades the validation
   of each model by RESOURCES.md stage D. Under a hold it stops at once. It
   deletes `<workspace>/runs/validation.json` when it starts, and
   `record_validation()` writes it as each model's validation ends, with
   these entries:
   - `stage`, which is `"validate"`;
   - the design version;
   - the hash of what the validation depends on: the data file, the data
     step, the model definitions, the simulation of datasets, the fit
     function with its methods and budgets, and the grading code of stage D. The pipeline
     declares these in `VALIDATED`, and a step after the fit, such as an
     answer, a check, or a comparison, and `report.py` leave the hash
     unchanged;
   - for each model, its grade with its sentence and notes, whether it
     needed the remedy of a rerun, and its table.

   No other stage writes the file. The validation step ends by printing its
   verdict in one line of plain words, which is the grade of RESOURCES.md
   stage D with its sentence, such as "Validation passed: the
   method recovered known answers from data like yours, so its intervals can
   be trusted."
3. `STAGE=fit` runs the fit to the real data and the steps after it. Its
   gate is part of `resolve_stage()`, so it runs before any step. The gate
   reads `<workspace>/runs/validation.json`, and a file whose stage, design
   version, and hash match the current code is a validation of that code.
   The gate lists the models to fit, which are those whose grade is *passed*
   or *passed with notes*, and it marks in the settings object that the
   observed values may be used. A model without such a grade is fitted only
   for the provisional results of the paragraph after this list, and the gate
   stops with an error when no model is left to fit.
   The step that returns the observed values reads that mark, and in the
   smoke stage it returns simulated data. A run whose gate fails therefore
   fits nothing, even under nbconvert's `--allow-errors`. Colab deletes
   the files of `/content` when its runtime resets, so run the validate and
   the fit stage in one session, or upload the `<workspace>/runs/validation.json`
   from a validate run of the same design version.

The user chooses provisional results after a failed validation or by the
choices of RESOURCES.md stage D, and you record the choice as a decision with
the user's reason, by rule 4 of the
[rules for revisions](5-feedback.md#rules-for-revisions). `resolve_stage()`
reads the reason from the environment variable `ALLOW_UNVALIDATED`, which the
user sets, as for `STAGE`. When it holds a reason, the gate
also fits each model that no validation of the current code passed, with
one of two labels:

1. "PROVISIONAL: validation failed" when its validation graded it *failed*;
2. "PROVISIONAL: not validated" when no validation of the current code
   ended, as when the user declined or held it.

The label appears in each printed answer, each figure title, each file name
in `<workspace>/outputs/`, and each estimate in the design document, and the
methods paragraph of the design document states the status.

Every run resolves the stage and then runs the data step. Each stage writes
its results, such as the diagnostics and the estimates, as it computes them,
so a run that a failed step stops keeps the results computed before it. The
smoke and the validate stage write to `<workspace>/runs/smoke/` and
`<workspace>/runs/validate/`, and only the fit stage writes to
`<workspace>/outputs/`, so a result of a smoke run never takes the file name of
a deliverable. [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#workspace) describes the
layout of the workspace.

Every number in section 0 and in the methods paragraph of the design document
is copied from a file that the fit stage wrote to `<workspace>/outputs/`
(Sandve et al. 2013). `report.py` saves each figure with a CSV file of the
numbers it plots. `record_run()` writes into each output folder `<folder>`
`<folder>/provenance.json`, with the seed, the ProbPipe version and checkout
commit, the design version, and the code hash, and
`<folder>/environment.txt`, with the versions of Python and of each installed
package.

## Check and run the code

Take these steps before you show the code, and again after each revision:

1. Run `<python> <assistant>/check_symbols.py <workspace>/pipeline.py`, or the
   same command on the notebook, and fix each problem that it prints. It
   checks the names and the call arguments of the file and of each module
   beside it that the file imports, by its docstring. It checks no meaning,
   so read the docstring of each call whose meaning you
   are unsure of.
2. If ProbPipe imports here, run the smoke stage by the
   [rules for running code](../INSTRUCTIONS.md#user-materials-and-code), and
   fix what breaks. Otherwise, say that the code has not been run. The
   commands of a stage are these, shown for the validate stage:
   - a script: `STAGE=validate <python> <workspace>/pipeline.py`;
   - a notebook:
     `STAGE=validate <python> -m nbconvert --to notebook --execute <workspace>/pipeline.ipynb --ExecutePreprocessor.kernel_name=<kernel> --output pipeline.validate --output-dir <workspace>/runs`,
     which writes the executed copy `<workspace>/runs/pipeline.validate.ipynb`
     and runs the notebook from the workspace, so its relative paths hold.

   For another stage, replace `validate` with the stage. Run the smoke and
   the validate stage of a notebook with `--allow-errors`, so that one run
   shows each failing cell. With that flag, nbconvert exits with status 0 and
   prints no traceback, so search the executed copy for the outputs whose type
   is `error`. Run the fit stage without the flag. A failed cell then stops
   the run, and nbconvert exits with a nonzero status, prints the cell's
   traceback to standard error, and writes no executed copy.

   The project's kernel `<kernel>` is named `probpipe-<short-project-slug>`.
   Record its name in project memory. Before the first notebook run, list the
   kernels with `<python> -m jupyter_client.kernelspecapp list`, and choose
   another name when that one is taken. Then register the kernel with
   `<python> -m ipykernel install --user --name <kernel>`, which is an install
   by rule 8 of the
   [rules for user materials](../INSTRUCTIONS.md#user-materials-and-code), and add `--env PYTHONPATH <checkout>` when ProbPipe imports
   through `PYTHONPATH`. When nbconvert or ipykernel does not import, ask
   before you install it.
3. Record the inference method that ran, as the row "Condition on the data" of
   CAPABILITIES.md says to read it, in the design document and in project
   memory. When it differs from the method you planned, say so, and revise the
   plan or name the method.
4. Label each result of a smoke run "smoke run, not for inference" wherever
   you show it or record it, the fit log of the design document included.

## Present the proposal

Open the proposal's message with the one line of
[Phase 3](3-background.md) that confirms the working preferences, and show the
result of the first smoke run. Draw a Mermaid diagram only in an interface
that renders it. Point to the design document for the full detail and to its
section 11 for how to run the code.

For a user whose preferences ask for Bayesian choices in plain words, the
message is a summary of at most about 150 words: two plain sentences on what
the analysis will tell them, the smoke result, and at most three key design
decisions. The stages and the rest are in the design document.

For other users, give a stage diagram with one sentence per stage, the key
design decisions with the alternative you considered and your reason, the
assumptions to check, and the open questions. Point to one or two resources
of [RESOURCES.md](../RESOURCES.md) that suit the user's background and the
hardest part of the design. End with feedback questions about the decisions
you are least sure of. Update project memory.
