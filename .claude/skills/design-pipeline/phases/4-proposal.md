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
   papers. Then choose the route of the section "Bring any model" of
   CAPABILITIES.md by which the model enters ProbPipe:
   - The default is the route of the code the user already has, such as their
     PyMC model, Stan program, or simulator.
   - A model of standard statistical structure written for this project
     defaults to PyMC or Stan. Examples are a regression with random effects,
     such as a generalized linear mixed model, and a spatial model with a CAR,
     ICAR, or BYM prior. A computer that cannot compile Stan programs rules
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
the mode keeps for an estimand. The mode decides which other stages to defer,
such as step 10.

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
and the pipeline reads each data file by a path relative to the notebook. A
script reads each file by a path relative to `Path(__file__).parent`, so that
it runs from any directory. The notebook's metadata names the project's
kernel, which [Check and run the code](#check-and-run-the-code) registers,
with the language `python`, so that Colab, which lacks that kernel, runs the
notebook on its own Python kernel. The pipeline loads
the user's data or simulates it, and each pipeline stage is a labeled section
that matches the design document. Each stage shows a figure where one helps,
such as the prior predictive draws or the posterior predictive check, with a
one-sentence comment on how to read it. The notebook's first text cell, or a
comment at the top of a script, says where to change each choice, such as
"the priors are in section 3, Priors, and the number of draws of each stage is
in the first code cell", so that the user can find, defend, and edit each
choice in code they did not write.

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
   plot, and write results, and it formats each interval by rule 4 of
   [Messages](../INSTRUCTIONS.md#messages). `<workspace>/sim_data.py` draws the simulated
   data at the real design from the model that the pipeline passes to it, and
   masks it to the real pattern of missing values. The model, the inference,
   and the checks stay in the pipeline itself.

The code runs in three stages. The first code cell after the install cell,
or the top of a script, reads the stage with
`STAGE = os.environ.get("STAGE", "smoke")` and sets the budgets of each stage,
so a user who runs the notebook in Jupyter or on Colab sets the stage by
editing that default.

1. `STAGE=smoke` runs every section at small budgets, such as a few draws,
   chains, or training epochs. It fits data simulated from the model at the
   structure of the real data, such as its sizes, its groups, and its
   pattern of missing values, in place of the real data, so the code fixes
   that the smoke runs need stay out of the record of revisions. A smoke run
   on a subsample of the real data is an option that the user approves, and
   never for data that [Phase 2](2-materials.md#data) found sensitive.
2. `STAGE=validate` runs the sections before the fit to the real data at
   their full budgets, such as the prior predictive check and the validation
   on realistic simulated data, and grades the validation of each model by
   RESOURCES.md stage D. It deletes `<workspace>/runs/validation.json` when it
   starts and writes it when the validation ends, with these entries:
   - `stage`, which is `"validate"`;
   - the design version;
   - a hash of the code: the source of the code cells of the notebook, or of
     the script, and of each module beside it that it imports, with each line
     that reads `os.environ` left out, so that setting the stage keeps the
     hash;
   - the grade of each model, with its sentence;
   - the table of the validation.

   No other stage writes the file. Each cell of the validation ends by
   printing its verdict in one line of plain words, which is the grade of
   RESOURCES.md stage D with its sentence, such as "Validation passed: the
   method recovered known answers from data like yours, so its intervals can
   be trusted."
3. `STAGE=fit` runs the fit to the real data and the sections after it. Its
   gate is the cell right after the data step. The gate reads
   `<workspace>/runs/validation.json` and checks its stage, its design version, and
   its hash against the current code. It stops with an error when a check
   fails or when no model passed. Otherwise it binds the observed values, and
   it lists the models to fit, which are those whose grade is *passed* or
   *passed with notes*. Only the gate binds the observed values, after its
   checks, so a run whose gate fails fits nothing, even under nbconvert's
   `--allow-errors`. On Colab, the session's files hold no copy of the
   notebook, so the gate there skips the hash and says so. Colab also deletes
   the files of `/content` when its runtime resets, so run the validate and
   the fit stage in one session, or upload the `<workspace>/runs/validation.json`
   from a validate run of the same design version.

Every stage runs the data step first. Each stage writes its results, such as
the diagnostics and the estimates, as it computes them, so a run that a
failed step stops keeps the results computed before it. The smoke and the
validate stage write to `<workspace>/runs/smoke/` and
`<workspace>/runs/validate/`, and only the fit stage writes to
`<workspace>/outputs/`, so a result of a smoke run never takes the file name of
a deliverable. [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#workspace) describes the
layout of the workspace.

When the validation of a model fails and the user still wants its answers,
they can ask for provisional results. Offer them only on that request. Record
the request as a decision with the user's reason, by rule 4 of the [rules for
revisions](5-feedback.md#rules-for-revisions). The first code cell reads the
reason with `ALLOW_UNVALIDATED = os.environ.get("ALLOW_UNVALIDATED", "")`, and
the user sets the variable or edits that default. When it holds a reason, the
gate also fits each model whose grade is *failed*, and each output of such a
model carries the label "PROVISIONAL: validation failed": each printed answer,
each figure title, each column name or file name of a CSV file, and each
estimate in the design document. The methods paragraph of the design document
states the failure.

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
[Phase 3](3-background.md) that confirms the working preferences. Present the
proposal at the depth that the preferences call for. Draw a Mermaid diagram
only in an interface that renders it. Give the user the steps of
section 11 of the design document, so that they can run the code.

For a user whose preferences ask for Bayesian choices in plain words, open
with two plain sentences on what the analysis will tell them. Then list the stages as a
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
