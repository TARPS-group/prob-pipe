# Pipeline design document template

The `design-pipeline` skill writes `<workspace>/pipeline-design.md` with this
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
Typical stages: data loading → prior predictive check → (synthetic
recovery / SBC) → conditioning → diagnostics → posterior predictive check →
downstream quantities → robustness/sensitivity.

## 4a. Model sequence
The planned progression from the simplest model to the expansions expected,
and which one this version implements.

## 4b. Workflow checklist
One line per stage of the RESOURCES.md checklist (A-J): how it is addressed,
or why it is skipped.

## 5. Key design decisions
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
| Version | Changes | Reason (feedback or finding) |
|---|---|---|
```
