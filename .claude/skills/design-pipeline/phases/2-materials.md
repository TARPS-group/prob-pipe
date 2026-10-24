# Phase 2 — Data and model materials

## Data

Ask what data the user has, and ask for **pointers to it**, such as local file
paths, URLs, or repositories, or for a description and a sample when the data
cannot be shared. Also ask how the data was collected:

- the design and the sampling;
- the measurement instruments and the units;
- known biases;
- the mechanisms of missingness.

Before you open any dataset, ask whether it contains personal or health
information, sensitive locations such as the sites of protected species, or
data under a use agreement. If it does, or the user is unsure, open none of its
rows. Write a short script that prints only the column names, the types, the
counts, and the missingness of each column, ask the user to run it, and work
from its output.

For each dataset you can open:

- Inspect it yourself. Load a small file in full, and read the header and a
  sample of a large one. Use `pandas` if it is installed, and the standard
  library otherwise.
- Record these features in `<workspace>/data-notes.md`:
  - the format and the size;
  - the variables, with their types, units, and ranges;
  - missingness;
  - grouping or hierarchy, such as subjects, sites, or time;
  - censoring or truncation;
  - outliers;
  - anything that contradicts the user's description;
  - for counts and rates, five more features:
    - the true zeros and the values that were not observed, including
      sentinel codes such as -99, blanks, or a 0 that means "not surveyed";
    - whether the table is wide, with one column per time, or long, with one
      row per observation;
    - the exposure or effort of each count, such as survey effort, area, or
      person-time, which enters the model as an offset;
    - overdispersion, which shows as a variance well above the mean within
      comparable groups;
    - whether the missing values are in the outcome, which the likelihood can
      omit when the missingness depends only on observed quantities, or in a
      covariate, which needs a model or an imputation that you discuss with
      the user.
- Note which variables look like outcomes, covariates, grouping variables, or
  known constants, and confirm that reading with the user.
- Propose how the data maps onto ProbPipe objects, by the row "Hold the
  observed data and the covariates" of [CAPABILITIES.md](../CAPABILITIES.md),
  and present it by the Python rows of the *Adaptation* section of
  [BACKGROUND-RUBRIC.md](../BACKGROUND-RUBRIC.md).

After the inspection, show the user one quick summary of the data, such as the
counts by group and time and where values are missing, and ask whether it
matches what they expected.

If the data is not available yet, ask for a description detailed enough to
simulate synthetic data of the same structure. Tell the user that the pipeline
runs on synthetic data until the real data arrives.

## Models

Ask for any existing models in any form, such as papers, code, or informal
descriptions like "growth slows when resources run low". The code can be in any
language, such as Stan, PyMC, or R, or it can be a simulator. From each model,
extract six things:

1. the structure;
2. the parameters and their meaning;
3. the priors or the parameter ranges;
4. the likelihood or the noise model;
5. the assumptions;
6. the inference method used.

Each kind of material needs one more step:

- **Papers**, as PDFs or URLs: note the page and equation numbers.
- **Code:** read it, and note whether it can be wrapped directly, by the rows
  "Use an existing Stan or PyMC model" and "Wrap a simulator" of
  [CAPABILITIES.md](../CAPABILITIES.md).
- **Informal descriptions:** turn each one into a candidate mathematical model,
  and show it to the user to check.

Record each model in `<workspace>/model-notes.md`, and summarize the models in
sections 3-4 of `<workspace>/brief.md`. List each mismatch between the data and
the models as an open question, such as a variable that a model needs and the
data lacks. Update project memory.
