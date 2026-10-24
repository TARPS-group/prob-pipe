# Pipeline design document template

The design-pipeline assistant writes `<workspace>/pipeline-design.md` with this
structure. Leave out each section that does not apply, and keep the numbers of
the other sections, so that versions can be compared. Match the depth of
explanation to the background profile in person memory.

```markdown
# <Project name>: ProbPipe pipeline design

**Version:** <n> · **Date:** <date> · **Status:** draft | agreed

## 1. Goals and estimands
- Scientific goal, in the user's words.
- Estimands: the quantities to estimate or predict, with units and the form
  of uncertainty needed (e.g. "posterior for the decay rate; 90% predictive
  interval for next season's count").

## 2. Data
- Each dataset: source, size, key variables, how it maps onto ProbPipe
  objects.
- Preprocessing steps, and why.
- Known data issues and how the pipeline handles them.

## 3. Generative model
- Model in math (parameters, priors, likelihood), with the meaning of each
  parameter.
- Justification for each prior (domain knowledge, a cited paper, or a
  weakly informative default, said so explicitly).
- ProbPipe mapping: which classes implement each piece.

## 4. Pipeline stages
Text or Mermaid diagram of the stages, then one subsection per stage:
- **Purpose**, **ProbPipe calls**, **inputs → outputs**, **what to look at**
  in the output, and **failure signs**.
The conditioning stage also records the inference method planned, the
method that ran, and whether that run was a smoke run.
Typical stages: data loading → prior predictive check → validation on
realistic simulated data → conditioning → diagnostics → posterior predictive
check → downstream quantities → robustness/sensitivity.

## 4a. Model sequence
The planned progression from the simplest model to the expansions expected,
and which one this version implements.

## 4b. Workflow checklist
One line per stage of the RESOURCES.md checklist (A-J): how it is addressed,
why it is skipped, or `deferred: fast path`.

## 4c. Compute plan
- Hardware: CPU or GPU, memory, and the wall-clock time available.
- Precision: single or double, and why (RESOURCES.md stages E and I).
- Estimated cost of each stage of section 4, such as the simulations,
  the training, the fits, and the validation on simulated data.
- Smoke and full settings of each stage:
| Stage | Smoke setting (SMOKE=1) | Full setting |
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
Needs that ProbPipe does not meet, the workaround used, and whether each is
worth a GitHub issue.

## 9. Changelog
| Version | Changes | Reason (feedback or finding) | Pre-data or data-driven |
|---|---|---|---|

## 10. Fit log
Every fit, in order, smoke runs and fits to simulated data included.
| Date | Design version | Model | Data (real, simulated, or smoke) | Method and settings (chains, draws, seed) | Diagnostics | Checks passed and failed | Outcome |
|---|---|---|---|---|---|---|---|

## 11. Getting it running
- Install: the command from `docs/get_started/installation.md` for the
  backends this design uses, and the Python version it needs.
- Open and run: how to run the code locally and on Google Colab, first with
  `SMOKE=1` and then in full.
- Expected runtime of each stage, from section 4c.
- What a successful run prints and shows.
- Where the code was validated, and whether that environment differs from the
  user's.

## 12. Reporting
- A draft methods paragraph: the model, the priors, the software and its
  versions, the inference method and its settings, and the diagnostics.
- A table of the priors, each with its justification from section 3.
- The figures to include, each with what it shows.
- How to cite ProbPipe, from `docs/cite.md`.
```
