> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# API reference

The API reference documents each public name of ProbPipe on one page, grouped by what a user looks the name up for.
A name whose heading is a bare name, such as `Normal`, imports from `probpipe`.
A name whose heading is a dotted path, such as `probpipe.families.GaussianProcess`, imports from the module that path names.

| Page | What it documents |
|---|---|
| [Values and records](values.md) | The value kinds, such as records and numeric arrays, their batches, and the linear operators. |
| [Declarations](declarations.md) | The term specs, the input and output declarations, and the constraints of a numeric value. |
| [Distributions and families](distributions.md) | The distribution classes, their capabilities, and the families ProbPipe ships. |
| [Random functions](random_functions.md) | The distributions over functions and over distributions. |
| [Functions](functions.md) | `Function` and its decorator, the capabilities a function claims, and the errors of a call. |
| [Operations](operations.md) | The operations, such as `sample`, `mean`, and `condition_on`, and the errors of their resolution. |
| [Inference methods](inference.md) | The inference methods, the functions of amortized and sequential inference, and the inference-method registry. |
| [Conversion](conversion.md) | The `convert` operation and the converter registry. |
| [Workflows and reproducibility](workflows.md) | Workflow scopes and replay, the orchestration settings, and the default sample count. |
| [Labels and provenance](provenance.md) | The identity of a tracked term, its provenance record, and the provenance settings. |
| [Diagnostics](diagnostics.md) | The MCMC, predictive, and leave-one-out diagnostics, and the views that read them. |
| [Validation](validation.md) | Predictive checks, scores against a reference posterior, and calibration. |
| [Registries for extensions](extending.md) | The dispatch registries and the base classes that an extension implements and registers. |
