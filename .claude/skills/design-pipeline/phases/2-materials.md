# Phase 2 — Data and model materials

## Data

Ask what data the user has, and ask for **pointers to it**, such as local file
paths, URLs, or repositories, or for a description and a sample when the data
cannot be shared. Also ask how the data was collected:

- the design and the sampling;
- the measurement instruments and the units;
- known biases;
- the mechanisms of missingness.

Before you open any dataset, ask for consent to read it, by rule 5 of the
[rules for user materials](../INSTRUCTIONS.md#user-materials-and-code), such
as "reads your CSV file; a few seconds; each value it prints goes to the model
provider". In the same message, ask whether it contains personal or health
information, sensitive locations such as the sites of protected species, or
data under a use agreement. Aggregated statistics that identify no person,
such as published case counts per district, count as non-personal data. Ask
whether a small-cell rule applies to them, such as a rule that suppresses
counts below five, and follow it in what you show and record. If the dataset
holds sensitive data, or the user is unsure, open none of its rows. Write a
short script that prints only each column's name, type, and number of missing
values. Ask the user to run it, and work from its output.
Record only that schema in `<workspace>/notes/data-notes.md`.

For each dataset you can open:

- Inspect it yourself, by the
  [rules for user materials](../INSTRUCTIONS.md#user-materials-and-code). Load
  a file of a few megabytes or less in full, and read the header and a sample
  of the rows of a larger one. Use `pandas` if it is installed, and the
  standard library otherwise.
- Record these features in `<workspace>/notes/data-notes.md`:
  - the format and the size;
  - the variables, with their types, units, and ranges;
  - missingness;
  - grouping or hierarchy, such as subjects, sites, or time;
  - censoring or truncation;
  - outliers;
  - anything that contradicts the user's description;
  - for counts and rates, six more features:
    - the true zeros and the values that were not observed, including
      sentinel codes such as -99, blanks, or a 0 that means "not surveyed";
    - whether the table is wide, with one column per time, or long, with one
      row per observation;
    - the exposure or effort of each count, such as survey effort, area, or
      person-time, which enters the model as an offset;
    - overdispersion, which shows as a variance well above the mean within
      comparable groups;
    - zero inflation and imperfect detection: more zeros than the count model
      predicts, as at sites where a species is absent, and counts below the
      true number, as when observers miss individuals. Repeated visits to a
      site separate the two, by the occupancy and N-mixture models of
      Kéry & Royle 2016 in [RESOURCES.md](../RESOURCES.md);
    - whether the missing values are in the outcome, which the likelihood can
      omit when the missingness depends only on observed quantities, or in a
      covariate, which needs a model or an imputation that you discuss with
      the user.
- Note which variables look like outcomes, covariates, grouping variables, or
  known constants, and confirm that reading with the user.
- Propose how the data maps onto ProbPipe objects in the form the user keeps
  it, such as tables, images, or the outputs of another pipeline, by the row
  "Hold the data" of
  [CAPABILITIES.md](../CAPABILITIES.md). Present it by the rows on Python of the
  [table of preferences](../BACKGROUND-RUBRIC.md#preferences) that the answer
  to the background question of [Phase 1](1-intake.md) suggests.
- Plan the preprocessing as steps of the pipeline that start from the raw files,
  by the row "Preprocess the data" of [CAPABILITIES.md](../CAPABILITIES.md).
  ProbPipe then records each step in the provenance of the data, and a rerun
  reproduces the data from the raw files. This is the default for cleaning,
  reshaping, and derived variables. When the user keeps a step outside
  ProbPipe, record where it runs and where its script is kept in the design
  document.

After the inspection, show the user one quick summary of the data, such as the
counts by group and time and where values are missing, and ask whether it
matches what they expected.

If the data is not available yet, ask for a description detailed enough to
simulate synthetic data of the same structure. Tell the user that the pipeline
runs on synthetic data until the real data arrives.

## Models

Ask for any existing models in any form, such as papers, code, or informal
descriptions like "growth slows when resources run low". The code can be in any
language, such as PyMC, Stan, or JAX, or it can be a simulator. From each
model, extract six things:

1. the structure;
2. the parameters and their meaning;
3. the priors or the parameter ranges;
4. the likelihood, the noise model, or the random draws of a simulator;
5. the assumptions;
6. the inference method used.

Each kind of material needs one more step:

- **Papers**, as PDFs or URLs: note the page and equation numbers.
- **Code:** read it, and note the route of the section "Bring any model" of
  [CAPABILITIES.md](../CAPABILITIES.md) by which it enters ProbPipe as it is,
  and what that route allows downstream. Code in a language with no route,
  such as R, enters as wrapped code or as the draws it computes, by the same
  file. A brms model enters as the Stan program that `brms::stancode()`
  writes.
- **Informal descriptions:** turn each one into a candidate mathematical model,
  and show it to the user to check.

Record each model in `<workspace>/notes/model-notes.md`, and summarize the models in
sections 3-4 of `<workspace>/notes/brief.md`. List each mismatch between the data and
the models as an open question, such as a variable that a model needs and the
data lacks. Update project memory.
