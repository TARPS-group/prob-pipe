# Phase 1 — Problem and goals intake

Learn what the user wants to achieve, in their own terms.

Until [Phase 3](3-background.md) confirms the user's background, use plain
words. Ask "what quantities do you want to estimate, and how sure do you need
to be about them?" in place of naming estimands. Ask about the collection of
the data in the user's terms, such as "were some sites not surveyed in some
years?". Among the first questions, ask one about the user's background, such
as "Have you fitted a Bayesian model before, and do you usually work in
Python, R, or something else?". Choose the vocabulary of Phases 1 and 2 by the
answer and by the *Adaptation* section of [BACKGROUND-RUBRIC.md](../BACKGROUND-RUBRIC.md).

Cover these topics:

- **The problem:** the system, phenomenon, or decision the analysis is about,
  and the question the user wants answered.
- **Scientific or applied goals:** what would count as success, and what the
  results are for, such as a paper, a decision, or a forecast.
- **Analysis goals:** which kinds of analysis the user is after. Ask
  open-endedly, such as "what do you want the analysis to tell you?", and offer
  examples only when the user asks for them. Classify the answer by the kinds
  below, and probe for the *estimands*, which are the specific quantities whose
  uncertainty matters. The kinds are these:
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
- **Scale,** when the data are large or the model is a simulator. These
  quantities choose the inference method by
  [RESOURCES.md](../RESOURCES.md) stages E and I:
  - the size of the data, in observations, variables, and bytes;
  - the cost of one simulator call, whether the simulator is stochastic, and
    whether its calls can be vectorized over parameter values;
  - the simulation budget;
  - how many datasets the model will analyze;
  - the hardware, such as a GPU, and the wall-clock time available.

When the project starts, ask whether to use quick start mode, by the
[ground rules](../INSTRUCTIONS.md#ground-rules), and recommend it. Ask it
with the first group of questions, and describe it in plain words, such as
"a first complete answer soon: a simple model and the model your questions
need, checks that the analysis can be trusted, and your estimates with their
uncertainty. Comparing models by their predictions and testing how much the
answers depend on the assumptions come next, unless you name one of them as a
concern now."

Once you can name the project, propose a workspace directory by
[MEMORY-FORMAT.md](../MEMORY-FORMAT.md#workspace), and confirm it with the
user. If person memory does not exist yet, ask now whether you may create it,
as [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#rules) describes. Create the project
memory, record the corpus in it as [Phase 0](0-load.md) step 6 says, and add
the project to the index of person memory.

Write sections 1-2 of `<workspace>/notes/brief.md`. Then summarize the problem and
the goals for the user in a few sentences, and ask them to confirm the summary
before you continue.
