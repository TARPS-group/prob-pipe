# Phase 3 — Background assessment

**If person memory already has a confirmed profile,** say briefly how you will
work with the user, and ask only about what this project adds:

- the dimensions that this project newly depends on, such as a new discipline
  or a first simulator;
- the ratings that are inferred or have low confidence;
- anything that the conversation suggests has changed.

Then save the changes, as step 2 of the full assessment does.

**Otherwise,** estimate the user's background from **evidence in the
conversation so far**, with the dimensions and the ratings of
[BACKGROUND-RUBRIC.md](../BACKGROUND-RUBRIC.md). The evidence includes:

- the user's vocabulary and the questions they asked;
- how they described the data and the models, and the materials they gave;
- anything they said about themselves.

Present the assessment as how you will work with the user, by the rubric's
*Adaptation* section: how much you will explain, how much code you will show,
and which choices you will make for the user and which you will put to them.
Base each point on what the user said about their own experience or work, such
as "you mentioned that you work mostly in R", and quote none of the user's
questions as evidence. Show the table of one row per dimension, with the
rating, your confidence, and the evidence, only when the user asks for it. For
a user new to both JAX and ProbPipe, record those two ratings in memory and
leave them out of the presentation. Be candid and respectful, because the
assessment calibrates how you work together. Then take two steps:

1. Ask the user to correct anything wrong, and to add context that the
   conversation did not reveal, such as "I've used Stan for years but never
   JAX".
2. Save the confirmed profile, with the user's corrections, to person memory.
   Put anything true only of this project, such as "new to survival analysis",
   in project memory.

The profile can change. When later answers show a different rating, update
memory and mark the entry `[inferred]`. When the change alters how you work,
such as how much you explain, tell the user in one sentence.
