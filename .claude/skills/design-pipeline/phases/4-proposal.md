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
     ICAR, or BYM prior. Choose the
     one the user is more comfortable with, by the background profile: Stan
     for a user of Stan or brms, and PyMC for a user who works in Python or
     whose computer cannot compile Stan's C++ programs.
   - A custom, mechanistic, or simulator-based model, or a model whose code
     is already in JAX, defaults to the route in JAX and ProbPipe's laws.
   - Name the other routes to the user, with what each allows and rules out,
     from the last column of its row.

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

When a step needs a component that ProbPipe does not connect to, such as a
Gaussian-process package or a domain simulator, offer the options of the
ground rule "Offer to build an interface to a missing component" of
[INSTRUCTIONS.md](../INSTRUCTIONS.md#ground-rules).

In the quick start mode of the [ground rules](../INSTRUCTIONS.md#ground-rules),
step 2 plans the baseline and the elaboration that the mode names, and step 8
is one posterior predictive check of each model. The mode decides which other
stages to defer, such as the cross-validation of step 8 and step 10.

When the diagnostics of step 7 or the checks of step 8 fail in a full run,
return to step 3 and revise the model before you tune the algorithm, by the
folk theorem of RESOURCES.md stage E. Each return is a revision, which follows
the [rules for revisions](5-feedback.md#rules-for-revisions). Only a full run
decides a return, because the budgets of a smoke run are too small to judge
the diagnostics and the checks.

## Write the code

Write the pipeline as a runnable skeleton in the style of the tutorials, in the
form the user works in: the notebook `<workspace>/pipeline.ipynb` for a user
who works in notebooks, and `<workspace>/pipeline.py` otherwise. A notebook
starts with an install cell in the style of the tutorials' Colab cell. On
Google Colab, the cell installs ProbPipe at the commit that
`<corpus>/MANIFEST.md` records, or at its branch when that commit is not on
GitHub, so that Colab runs the code that `check_symbols.py` checked. The cell
then asks the user to upload the data file with `google.colab.files.upload()`,
and the pipeline reads each data file by a path relative to the notebook. The
notebook's metadata names the kernel `probpipe`, which
[Check and run the code](#check-and-run-the-code) registers. The pipeline loads
the user's data or simulates it, and each pipeline stage is a labeled section
that matches the design document. Each stage shows a figure where one helps,
such as the prior predictive draws or the posterior predictive check, with a
one-sentence comment on how to read it.

The code has a smoke mode for fast trial runs. It reads the environment
variable `SMOKE`, and when `SMOKE=1` it uses small budgets, such as a few
draws, chains, or training epochs. By default, a smoke run fits data simulated
from the model at the structure of the real data, such as its sizes, its
groups, and its pattern of missing values, in place of the real data. The real
data then stays out of the session, and the code fixes that the smoke runs
need stay out of the record of revisions. A smoke run on a subsample of the
real data is an option that the user approves, and never for data that
[Phase 2](2-materials.md#data) found sensitive. A notebook sets the budgets
from the same variable in its first code cell after the install cell.

## Check and run the code

Take these steps before you show the code, and again after each revision:

1. Run `<python> <assistant>/check_symbols.py <workspace>/pipeline.py`, or the
   same command on the notebook, and fix each problem that it prints. It
   checks that each ProbPipe name resolves and that each call's arguments bind
   to the signature, so read the docstring of each call whose meaning you are
   unsure of.
2. If ProbPipe imports here, run the pipeline with `SMOKE=1` by the
   [rules for running code](../INSTRUCTIONS.md#user-materials-and-code), and
   fix what breaks. Otherwise, say that the code has not been run. The
   commands are these:
   - a script: `SMOKE=1 <python> <workspace>/pipeline.py`;
   - a notebook:
     `SMOKE=1 <python> -m jupyter nbconvert --to notebook --execute <workspace>/pipeline.ipynb --ExecutePreprocessor.kernel_name=probpipe`,
     which writes the executed copy `<workspace>/pipeline.nbconvert.ipynb`.

   The kernel `probpipe` runs `<python>`. Before the first notebook run, list
   the kernels with `<python> -m jupyter kernelspec list`, because the
   registration replaces a kernel of the same name. Then register the kernel
   with `<python> -m ipykernel install --user --name probpipe`, after the user
   approves. When nbconvert or ipykernel does not import, ask before you
   install it.
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
