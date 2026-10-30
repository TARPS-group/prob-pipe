# Phase 1 — Problem and goals intake

Learn what the user wants to achieve, in their own terms.

Until [Phase 3](3-background.md) chooses the working preferences, use plain
words. Ask "what quantities do you want to estimate, and how sure do you need
to be about them?" in place of naming estimands. Ask about the collection of
the data in the user's terms, such as "were some sites not surveyed in some
years?". Among the first questions, ask one about the user's background, such
as "Have you fitted a Bayesian model before, and do you usually work in
Python scripts, notebooks, R, or something else?". Choose the vocabulary of
Phases 1 and 2 by the answer and by the table of preferences of
[BACKGROUND-RUBRIC.md](../BACKGROUND-RUBRIC.md#preferences).

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

Ask the questions of this phase over several messages, by the rules of
[Messages](../INSTRUCTIONS.md#messages). The first message announces the
quick start mode of the [ground rules](../INSTRUCTIONS.md#ground-rules) and
the way to leave it, in plain words, such as "I'll get you a first complete
answer first: a simple model and the model your questions need, checks that
the analysis can be trusted, and your estimates with their uncertainty. Say
'full workflow' at any time."
Its three questions are, in this order, what the user wants to learn, where
the data is, and the background question, so that the summary of the data of
[Phase 2](2-materials.md#data) comes early. Skip each question that the
opening message already answers, and ask in its place the most important
topic of this phase that remains open, such as the user's existing models or
the deadline. Ask the other topics after the summary of the data.

The inspection of Phase 2 writes its script in the workspace, so propose the
workspace directory in the message that asks to read the data. Name its
default by [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#workspace), with a project
name taken from the opening message, and say that the user can name another
directory. If person memory does not exist yet, ask whether you may create
it, as [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#rules) describes. Put the
consents of that message to the user as one question, which counts once
toward the three questions of a message of
[Messages](../INSTRUCTIONS.md#messages): the consent to read the data with
its question on sensitive data, by [Phase 2](2-materials.md#data), the
standing approval of rule 5 of the
[rules for user materials](../INSTRUCTIONS.md#user-materials-and-code), the
workspace, and person memory. It states the default of each, and the user
answers it with one yes or no, or names what to change. Create the workspace
after the user replies, at the default when the reply does not name one.
Create the project memory, record the corpus in it as [Phase 0](0-load.md)
step 6 says, and add the project to the index of person memory.

Write sections 1-2 of `<workspace>/notes/brief.md`. Then summarize the problem and
the goals for the user in a few sentences, and ask them to confirm the summary
before you continue.
