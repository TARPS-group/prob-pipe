# Phase 1 — Problem and goals intake

Learn what the user wants to achieve, in their own terms. Cover four topics:

- **The problem:** the system, phenomenon, or decision the analysis is about,
  and the question the user wants answered.
- **Scientific or applied goals:** what would count as success, and what the
  results are for, such as a paper, a decision, or a forecast.
- **Analysis goals:** which kinds of analysis the user is after. Probe for the
  *estimands*, which are the specific quantities whose uncertainty matters. The kinds
  are these:
  - estimation of specific quantities;
  - prediction;
  - model comparison;
  - hypothesis assessment;
  - uncertainty propagation into a downstream calculation;
  - experimental design;
  - emulation of an expensive simulator.
- **Constraints:** the deadlines and the compute budget, the required software
  and reporting standards, and who else must understand or maintain the
  pipeline.
  Skip what person memory already records, such as the user's compute
  environment.

Once you can name the project, propose a workspace directory by
[MEMORY-FORMAT.md](../MEMORY-FORMAT.md#workspace), and confirm it with the
user. If person memory does not exist yet, ask now whether you may create it,
as [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#rules) describes. Create the project
memory, record the corpus in it as [Phase 0](0-load.md) step 6 says, and add
the project to the index of person memory.

Write sections 1-2 of `<workspace>/brief.md`. Then summarize the problem and
the goals for the user in a few sentences, and ask them to confirm the summary
before you continue.
