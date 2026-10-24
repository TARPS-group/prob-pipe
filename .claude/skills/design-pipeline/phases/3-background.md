# Phase 3 — Working preferences

Agree with the user how you will work together, as working preferences that
[BACKGROUND-RUBRIC.md](../BACKGROUND-RUBRIC.md) defines: how much you explain,
how much code you show, and which choices you make for them and which you put
to them.

**If person memory already has confirmed preferences,** follow them. Propose a
new or changed preference only for what this project adds:

- a dimension that this project newly depends on, such as a new discipline or
  a first simulator;
- a preference that memory tags `[inferred]`, or that
  [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#rules) counts as unconfirmed;
- anything that the conversation suggests has changed.

**Otherwise,** propose preferences by the rubric's table of preferences, from
**evidence in the conversation so far**. The evidence includes:

- the user's vocabulary and the questions they asked;
- how they described the data and the models, and the materials they gave;
- anything they said about themselves, such as their answer to the background
  question of [Phase 1](1-intake.md).

Base each preference on what the user said about their own experience or
work, such as "you mentioned that you work mostly in R", and quote none of the
user's questions as evidence.

Confirm the preferences in one line at the top of the proposal of
[Phase 4](4-proposal.md#present-the-proposal), rather than in a message of
their own, such as "I'll pitch this as: Bayesian choices in plain words, each
model beside its glmer formula, and a notebook you run top to bottom. Correct
me if that's wrong." Record them in two steps:

1. Save the preferences to person memory when you choose them, tagged
   `[inferred]`, with the facts about their experience that the user stated.
2. When the user's reply corrects a preference, replace it with the user's
   words, tagged `[stated]`. A reply that goes on without correcting the line
   accepts it, so tag the other preferences `[confirmed]`. Put a preference
   that holds only for this project, such as "explain survival models from
   the start", in project memory, except in a workspace that git tracks, as
   [MEMORY-FORMAT.md](../MEMORY-FORMAT.md#workspace) says.

The preferences can change. When later answers show that a preference no
longer fits, update it, tag it `[inferred]`, and tell the user in one sentence
when the change alters how you work, such as how much you explain.
