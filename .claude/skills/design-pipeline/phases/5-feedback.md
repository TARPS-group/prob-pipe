# Phase 5 — Feedback loop

Repeat these steps until the user says the design is done. Propose that it is
done when the model is adequate for the estimands: the validation on simulated
data, the diagnostics, and the checks pass, and section 6 of the design
document lists each deficiency that remains.

1. Take the feedback. Classify each point as scientific, modeling, inference,
   computational, or presentation, and restate each point you had to
   interpret.
2. When a point conflicts with something earlier, such as the goals, a
   recorded decision, or a ProbPipe constraint, say so, and propose a
   resolution for the user to confirm.
3. Revise the design by the [rules for revisions](#rules-for-revisions). Bump
   the version in `<workspace>/notes/pipeline-design.md`, and add a changelog entry
   that lists what changed, why, and the class of the revision. Update the
   pipeline (`<workspace>/pipeline.py` or `<workspace>/pipeline.ipynb`) to
   match, and check and run it again as
   [Phase 4](4-proposal.md#check-and-run-the-code) describes.
4. Record the decisions and the rejected options in project memory, which
   keeps them by the append-only rule of
   [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#rules). Record each new general
   preference, such as "always show me the math", in person memory.
5. Show a compact summary of what changed. Raise again each open question
   that is still unresolved.
6. When the fit stage has run on the real data, run by you or by the user,
   write section 0 of the design document from its outputs, and read them
   with the user: what the diagnostics say, what each estimate and interval
   means for the science, and what the checks found.
7. Ask what to do next, and suggest the most valuable next step, such as "run
   the prior predictive check on your real covariates" or "decide between
   partial and complete pooling".

## Rules for revisions

A revision prompted by a fit to the real data can fit the noise of that data,
and the analysis then overstates its certainty. Every revision of a design, in
any phase, follows these rules:

1. **Classify each revision** into one of four classes:
   - *before any fit to your data:* a change made before any fit to the real
     data;
   - *data-driven:* a change of the model prompted by a fit to the real data,
     a predictive check, or a cross-validation result;
   - *computational:* the same model computed another way, such as a
     reparameterization or other settings of the sampler;
   - *added after the fit:* an analysis added after a fit to the real data and
     prompted by something other than that fit, such as a reviewer's request
     for another model or a prior sensitivity analysis. It runs beside the
     main model as a sensitivity analysis, and the main model stays as it
     is.

   A fix of the code after a smoke run implements the design as it stands, so
   it needs no new version.
2. **Keep each prior independent of its own posterior.** When the user asks to
   change a prior after seeing the posterior it produced, say that the change
   uses the data twice, and propose a sensitivity analysis over both priors
   instead.
3. **Keep the failed checks and the observations.** A check that fails stays in
   the design, with its result. An observation is excluded only for a modeling
   reason, such as a recorded measurement failure, which the design document
   states.
4. **Record each override.** When the user overrides a check or a rule of the
   workflow, record the override as a decision with the user's reason.
5. **Prefer nesting and stacking.** Prefer an expansion that contains the
   earlier model as a special case, or stacking of the candidate models, to the
   selection of the variant that fits best.
6. **Log every fit.** Record each fit, smoke runs included, in the fit log of
   the design document.

## Closing the session

Before the closing summary of a session in which the fit stage ran, rerun
the fit stage in a fresh process from the raw data files: the script with
`STAGE=fit`, or the notebook with a restarted kernel and every cell run. Check
that the answers of the rerun match those that the summary reports, and report
each difference with its likely cause, such as a draw outside `workflow_run`
(Zeng et al. 2025).

When the user is done or stops for now, update both memory files. Project
memory records where the project stopped and the agreed next step, and the
index of person memory records the project's status. Close with a summary:

- the design version;
- every model fit, from the fit log of the design document, with the outcome
  of each fit;
- the files, and how to run the pipeline, from section 11 of the design
  document;
- the open questions and the assumptions still to check;
- the ProbPipe gaps you found, with an offer to draft GitHub issues for them;
- the suggested next steps, such as a fit on the full data, a sensitivity
  analysis, or a write-up.

A draft issue describes the gap with a synthetic or public example. It holds
none of the user's data, file contents, file paths, or background. Give each
draft to the user as text. File it only when the user asks you to, after they
approve its final text.

Tell the user how to resume the project: in Claude Code with
`/design-pipeline resume <short-project-slug>`, and in another agent with the
setup prompt of the ProbPipe documentation page "Design assistant", whose last
line is then "resume <short-project-slug>".
