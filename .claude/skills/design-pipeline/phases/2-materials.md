# Phase 2 — Data and model materials

## Data

Ask what data the user has, and ask for **pointers to it**, such as local file
paths, URLs, or repositories, or for a description and a sample when the data
cannot be shared. Also ask how the data was collected:

- the design and the sampling;
- the measurement instruments and the units;
- known biases;
- the mechanisms of missingness.

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
  - anything that contradicts the user's description.
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
