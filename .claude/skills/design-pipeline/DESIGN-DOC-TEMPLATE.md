# Pipeline design document template

The design-pipeline assistant writes `<workspace>/notes/pipeline-design.md` with this
structure. Leave out each section that does not apply, and keep the numbers of
the other sections, so that versions can be compared. Match the depth of
explanation to the working preferences in person memory.

```markdown
# <Project name>: ProbPipe pipeline design

**Version:** <n> · **Date:** <date> · **Status:** draft | agreed

## 0. Answers and status
One row per question of the user, in the user's words and units, written from
the last run of the fit stage. A result of a smoke or a validate run has no
row.
| Question | Estimate | Interval: median [2.5%, 97.5%] units | Status | Caveats | Date | Design version |
|---|---|---|---|---|---|---|
The status of a row is the grade of the validation of its model: passed,
passed with notes, or "PROVISIONAL: validation failed" for the provisional
results of phases/4-proposal.md. The status of a grade from a pilot
validation, which RESOURCES.md stage D step 6 describes, adds "pilot: coverage
not established". The caveats name each note of the grade and each check of
the fit that failed.

## 1. Goals and estimands
- Scientific goal, in the user's words.
- Estimands: the quantities to estimate or predict, with units and the form
  of uncertainty needed (e.g. "posterior for the decay rate; 90% predictive
  interval for next season's count").

## 2. Data
- Each dataset: source, size, key variables, how it maps onto ProbPipe
  objects.
- Preprocessing steps, each a step of the pipeline from the raw files, and why.
  Any step kept outside ProbPipe, where it runs, and where its script is kept.
- Known data issues and how the pipeline handles them.

## 3. Generative model
- Model in math (parameters, priors, and the likelihood or the simulator),
  with the meaning of each parameter.
- Justification for each prior (domain knowledge, a cited paper, or a
  weakly informative default, said so explicitly).
- ProbPipe mapping: the route of CAPABILITIES.md "Bring any model" by which
  the model enters, the ProbPipe objects of each piece, and the checks that
  the route rules out.

## 4. Pipeline stages
Text or Mermaid diagram of the stages, then one subsection per stage:
- **Purpose**, **ProbPipe calls**, **inputs → outputs**, **what to look at**
  in the output, and **failure signs**.
The conditioning stage also records the inference method planned, the
method that ran, and the stage of that run (smoke, validate, fit, or
reproduce).
Typical stages: data loading → prior predictive check → validation on
realistic simulated data → conditioning → diagnostics → posterior predictive
check → downstream quantities → robustness/sensitivity.

## 4a. Model sequence
The planned progression from the simplest model to the expansions expected,
and which one this version implements.

## 4b. Workflow checklist
One line per stage of the RESOURCES.md checklist (A-J): how it is addressed,
why it is skipped, or `deferred: quick start`.

## 4c. Compute plan
- Hardware: CPU or GPU, memory, and the wall-clock time available.
- Precision: single or double, and why (RESOURCES.md stages E and I).
- Estimated cost of each stage of section 4, such as the simulations,
  the training, the fits, and the validation on simulated data.
- Smoke and full settings of each stage:
| Stage | Smoke setting (STAGE=smoke) | Full setting (STAGE=validate or fit) |
|---|---|---|

## 4d. Learned components
One subsection per neural estimator or emulator, with its training and
validation (RESOURCES.md stage I):
- **Training budget and data:** the number of simulations or simulator runs,
  and the training design.
- **Validation split and stopping rule:** the held-out simulations or runs,
  and when training stops.
- **Architecture and hyperparameters:** the network or the kernel, which
  hyperparameters were tuned, and how.
- **Ensemble size and seeds.**
- **Calibration evidence:** SBC, coverage at each credibility level, or the
  standardized errors of held-out runs, with the fit-log entry of each.

## 4e. Interfaces built
One line per interface that the design adds for a component that ProbPipe
does not connect to.
| Component | Route of CAPABILITIES.md | The user's case it covers | Test and its fit-log entry |
|---|---|---|---|

## 4f. Reproduction of the original pipeline
For a pipeline the user brought, by phases/4-existing-pipeline.md: the
comparison of the original results with ProbPipe's, before any change. For a
model re-implemented by option 3 of its step 2, the deterministic agreement
check comes first, with each kind of parameter set and its count, the largest
difference of the outputs and of the log density, and the tolerance. In the
table of the reproduction, a row is a parameter, with its three quantiles, or
a pair of parameters, with its correlation r. Below the table, give the number
M of z-scores, the bounds b₂(M) and b₃(M), and the mean, the standard
deviation, and the largest |z|.
| Estimand | Original: median [2.5%, 97.5%], or r | ProbPipe: median [2.5%, 97.5%], or r | z of 2.5% / 50% / 97.5%, or of r | Agree? |
|---|---|---|---|---|

## 5. Key design decisions
Each decision, including each override of a check by the user.
| Decision | Choice | Alternatives considered | Why |
|---|---|---|---|

## 6. Assumptions to check
Scientific and statistical assumptions the user should verify, each with how
to check it.

## 7. Open questions
Unresolved questions, each with who can answer it (the user, data, a
colleague, an experiment).

## 8. ProbPipe gaps
### 8a. Missing capabilities
Foundational capabilities that ProbPipe lacks.
| Need | Stage | Workaround used | Worth a GitHub issue? |
|---|---|---|---|

### 8b. Missing interfaces
Tools that ProbPipe does not connect to, each with the interface of section
4e or the other workaround used.
| Tool | Need | Workaround used | Worth a GitHub issue? |
|---|---|---|---|

## 9. Changelog
| Version | Changes | Reason (feedback or finding) | Class (before any fit to your data, data-driven, computational, or added after the fit) |
|---|---|---|---|

## 10. Fit log
Every fit, in order, smoke runs and fits to simulated data included.
| Date | Design version | Model | Data (real or simulated) | Stage (smoke, validate, fit, or reproduce) | Method and settings (chains, draws, seed) | Diagnostics | Checks passed and failed | Outcome, such as the grade of a validation |
|---|---|---|---|---|---|---|---|---|

## 11. Getting it running
- Install: the command from `docs/get_started/installation.md` for the
  backends this design uses, at the commit or branch that the corpus manifest
  records, and the Python version it needs, which is 3.12 or later, or 3.12
  or 3.13 for the extra `bayesflow`. pandas installs with ArviZ, which
  every install includes, and matplotlib with the extra `pymc` or `viz`.
- Open and run: how to run the code locally and on Google Colab, stage by
  stage, with `STAGE` set to `smoke`, `validate`, and then `fit`. For a
  notebook, the `nbconvert` command of each stage and the project's kernel
  that it runs, the first code cell that sets the stage in Jupyter or on
  Colab, and on Colab, the upload of the data file.
- Expected runtime of each stage, from section 4c.
- What a successful run prints and shows.
- Where the code was validated, and whether that environment differs from the
  user's.

## 12. Reporting
- A draft methods paragraph, written only from the rows of the fit log whose
  checks passed: the model, the priors, the software and its versions, the
  inference method and its settings, the diagnostics, and the grade of the
  validation. It states each check that failed and each provisional
  result.
- A table of the priors, each with its justification from section 3.
- The figures to include, each with what it shows.
- How to cite ProbPipe, from `docs/cite.md`.
```
