# Phase 5 — Feedback loop

Repeat these steps until the user says the design is done:

1. Take the feedback. Classify each point as scientific, modeling, inference,
   computational, or presentation, and restate each point you had to
   interpret.
2. When a point conflicts with something earlier, such as the goals, a
   recorded decision, or a ProbPipe constraint, say so, and propose a
   resolution for the user to confirm.
3. Revise the design. Bump the version in `<workspace>/pipeline-design.md`,
   and add a changelog entry that lists what changed and why. Update
   `<workspace>/pipeline.py` to match, and check and run it again as
   [Phase 4](4-proposal.md#check-and-run-the-code) describes.
4. Record the decisions and the rejected options in project memory, and each
   new general preference, such as "always show me the math", in person
   memory.
5. Show a compact summary of what changed. Raise again each open question
   that is still unresolved.
6. Ask what to do next, and suggest the most valuable next step, such as "run
   the prior predictive check on your real covariates" or "decide between
   partial and complete pooling".

When the user is done or stops for now, update both memory files. Project
memory records where the project stopped and the agreed next step, and the
index of person memory records the project's status. Close with a summary:

- the design version;
- the files, and how to run `<workspace>/pipeline.py`, in smoke mode and in
  full;
- the open questions and the assumptions still to check;
- the ProbPipe gaps you found, with an offer to draft GitHub issues for them;
- the suggested next steps, such as a fit on the full data, a sensitivity
  analysis, or a write-up.

Tell the user how to resume the project: in Claude Code with
`/design-pipeline <project>`, and in another agent with the setup prompt of the
ProbPipe documentation page "Design assistant", followed by the project's name.
