# Tutorial migration

This table lists every user-visible rename and behaviour change of the push's never-cut steps, old → new, with the step that makes it, so the workshop's tutorials on `dev/workshop-tutorials` can be brought onto the freeze tag (README § Workshop materials).
It covers P0, the PR #575 that merged before the push, and the never-cut steps of README § Steps; S0 and E1 change only design text, and an off-ramp's changes are added only if it merges.
Each announced step extends the table in its own PR with any rename it makes that a row does not list.
A change is **silent** when the same code still runs and returns something else; the step's CHANGELOG entry shows the same call before and after.
The rows restate `declarations-and-addressing.md` C.2, C.3, and C.5 and `execution-layer.md` § 13.1, which own the full lists.

In the examples, `x` is a law over `{a, b, c, sigma}`, `ke` a kernel `ke(e | sigma)`, and `d` a law over `{a, g: {b, c}}`.

## Merged before the push

| Old | New | Step | Silent? |
|---|---|---|---|
| `t.raw()`, `t.raw(path)` | `t.raw`, a property; `t[path].raw` | #575 | no |

## P0: #586 and #587

| Old | New | Step | Silent? |
|---|---|---|---|
| `n_broadcast_samples` | `max_evaluations` for lifting, `num_samples` for Monte Carlo summaries | P0 | no |
| `set_default_n_broadcast_samples(n)` | `config.sampling` | P0 | no |
| the settings objects `*_config` | `config.*` | P0 | no |
| the default Monte Carlo count 256 | 1024 | P0 | yes |

## The call context

| Old | New | Step | Silent? |
|---|---|---|---|
| a route run by `f.apply(x)` inside `g.using(num_samples=7)` took 7 draws | it takes `f`'s own count; `check` likewise | P0b | yes |
| a `config` scope exited out of order was discarded | it raises `RuntimeError`, naming both scopes | P0b | no |
| `with workflow_run(): with config.provenance.using(mode=OFF): ...` recorded provenance while the block read OFF | it raises, naming the run's mode; set the mode before the run opens | PV | no |
| a session assignment made while a threaded call ran reached its later worker threads | it reaches no evaluation point of that call | PV | yes |

## Paths and selection

| Old | New | Step | Silent? |
|---|---|---|---|
| a tuple of `str` is one nested path: `r["g", "b"]` is `r["g/b"]` | a tuple or list is a list of paths; a nested path is written `"g/b"` | S1 | yes, where each name is also a top-level path |
| `sample(d)["g"]` raises "use at_path()" | `sample(d)["g"]` is the group | S1 | no |
| `d["a", "g/b"]` is a law over `{a, b}` | a law over `{a, g: {b}}`, labelled `(a, g/b)` | S1 | yes |
| `marginal(d, field)`; several factors in the order of the paths | `marginal(d, paths)`; the law's order | S1 | the order: yes |
| `select` keys are full paths | `select` keys are last names; a shared last name raises | S1 | yes |
| `samples[()]` is the whole batch | it raises | S1 | no |
| `without` accepts a repeated path, or one inside another | it raises | S1 | no |
| `in` tests fields on a record and iterates a batch | `"g" in sample(d)` is true, `1.0 in samples` false | S1 | yes |

## The final words: input, output, and the specs

| Old | New | Step | Silent? |
|---|---|---|---|
| `x.event_spec.spec`, `output_spec.spec`, `with_spec` | `x.event_spec.term_spec`, `output_spec.term_spec`, `with_term_spec` | S2a | no |
| `exposes_record` | removed | S2a | no |
| `x.event_spec.components`, `batch.components` | `x.event_spec.outputs`; batches have no `components` | S2a | no |
| `component=`, `factor(d, component_name)`, `OutputSpec.default(component=)` | `output_name=`, `factor(d, output_name)`, `OutputSpec.default(output_name=)` | S2a | no |
| "slot", "component", "input declaration", "output declaration", "event declaration" in messages and docs | "input", "output", "input spec", "output spec", "event spec"; "component" keeps only a mixture's component laws | S2a | no |
| the page `docs/api/declarations.md` | `docs/api/input_and_output_specs.md` | S2a | no |
| a family's first parameter `component`, as `Normal(component="x", ...)`; `EmpiricalDistribution(..., component=)`, `distribution(..., component=)` | `output_name`, as `Normal(output_name="x", ...)`; the positional `Normal("x", 0.0, 1.0)` is unchanged | S2a | no |
| `mean.using(method="by_component")`, likewise `variance` and `quantile`; the route `"by_component"` in `provenance.metadata["route"]` | `method="by_output"`; the route `"by_output"` | S2a | the provenance route: yes |
| the route `"bind_given_slots"` of `condition_on` on a kernel, in `provenance.metadata["route"]` and a method report's `route` | `"bind_inputs"` | S2a | yes |
| the key `"components"` of each block of `provenance.metadata["blocks"]` under the route `by_output` | `"outputs"` | S2a | yes |
| `output_spec.with_spec(spec=...)`; `OutputSpec.default(spec=..., ...)` | `output_spec.with_term_spec(...)`, positional only; `OutputSpec.default(term_spec=..., output_name=...)` | S2a | no |

## Output specs and inputs

| Old | New | Step | Silent? |
|---|---|---|---|
| `{"sigma": x.event_spec.components["sigma"]}` | `x.event_spec["sigma"]`; `OutputSpec[]` takes a path or a list of paths, and `in` tests paths | S2b-out | no |
| `x.event_spec.spec.without("sigma")` | `x.event_spec.without("sigma")` | S2b-out | no |
| `{**inputs, "e": ke.event_spec.components["e"]}` | `kept & ke.event_spec`, for `kept = x.event_spec[["a", "b", "c"]]`; it raises where `kept` already has `e` | S2b-out | no |
| `list(x.event_spec)` iterates | it raises `TypeError`; `list(x.event_spec.outputs)` | S2b-out | no |
| an output spec over a law or kernel type has a sample's fields as outputs | it has none, and a merge refuses a law positional over a law, kernel, function, or batch | S2b-out | the empty outputs: yes |
| `merge` on records, record specs, and record batches is deep | shallow: a shared top-level name raises | S2b-out | no |
| `given_spec=x.event_spec.components` | `given_spec=x.event_spec`; an `InputSpec`, a `RecordSpec`, an `OutputSpec`, or a mapping declares inputs, with the size shorthand everywhere | S2b-in | no |
| `InputSpec.without` ignores an unknown name | it raises `KeyError` | S2b-in | no |
| a `RecordSpec` given as `given_spec` or `input_spec` is read by its leaf paths | it is read by its children | S2b-in | no: the old reading raises where they differ |
| a `Function`'s optional inputs are read from `input_spec` alone | a parameter with a default is optional; marking optional a parameter without one raises `ValueError` | S2b-in | the optional inputs: yes |
| a record in another field order conforms to a record type | it is refused with `ValueError`; a raw mapping given for an input or as a kernel's conditioning value is laid out in the declared order | S2b-in | no |

## Merge and rename

| Old | New | Step | Silent? |
|---|---|---|---|
| `lik * prior` between laws or kernels | `lik & prior`; `*` between laws or kernels raises `TypeError` naming `&` | S3 | no |
| `lik and prior` returned `prior` | `bool()` on a law or kernel raises | S3 | no |
| `record_a.merge(record_b)` on `NamedTree`, `Record`, `RecordSpec`, `RecordBatch` | `record_a & record_b` | S3 | no |
| `with_path_names` | `rename` | S3 | no |
| `FieldView` | `CoupledMarginal` | S3 | no |
| the `joint` route `"compose"` in `provenance.metadata["route"]` | `"merge"`; a recipe recorded before cannot be replayed | S3 | yes |

## Batches

| Old | New | Step | Silent? |
|---|---|---|---|
| `batch.spec.element_spec`, a type | `batch.spec.element_spec.term_spec`; `element_spec` is an output spec, naming base elements `element` | S4a | no |
| `BatchSpec(element_spec, ...)` | `BatchSpec(element, ...)`, taking an `OutputSpec` or a type | S4a | no |
| `NumericArrayBatch(..., element_spec=...)`, likewise `RecordBatch`, `NumericRecordBatch`, `OpaqueBatch`, `FunctionBatch`, `DistributionBatch`, `ConditionalDistributionBatch`, and `RecordBatch.stack` | `element=...`, taking the elements' spec; a batch's repr shows `element=` | S4a | no |

## Execution

| Old | New | Step | Silent? |
|---|---|---|---|
| `method="identity"` | `method="derived"` | E0 | no |
| `@operation(identity_check=..., identity_route=...)` | `@operation(derived_route_check=..., derived_route=...)` | E0 | no |
| `@operation(conditions=...)` | `@operation(applicability_checks=...)` | E0 | no |
| `op.identity`, `op.summary().identity` | `op.formula`, `op.summary().formula` | E0 | no |
| `metadata["route"] == "identity"` in a derived operation's provenance | `metadata["route"] == "derived"` | E0 | yes |
| a seeded draw inside an evaluation point | another value, once; a replay record written before cannot be replayed | E0 | yes |
| `is_jax_traceable(f, x)` | `f.check(x)` reports whether `f` traces | E2 | no |
| `except ValueError:` around a lifted call whose evaluation point fails | `except EvaluationError as error:`, reading `error.failures` | E3b, E4 | no, but an `except` of the user's own exception type stops catching |
| an output that does not match its output spec failed later; a `float64` output under a `float32` spec lost precision silently | `EvaluationError` naming the evaluation point; the cast warns | E3a | no |
| a replay recipe recorded `requested_dispatch` | it records `executor`; an old recipe is refused | E3b | no |
| a compiled JAX program that failed while it ran, such as a joint's mapping of a kernel, ran again eagerly | the error propagates, with no eager rerun | E3a | no |
| a kernel in a joint that read `num_samples` while it was traced kept the first call's value | it reads the value of the call that runs it | E3a | yes |
| `f.using(dispatch="thread", max_workers=8)` | `f.using(executor="thread", executor_options={"max_workers": 8})` | E3b | no |
| `f.using(dispatch="sequential")` ran a function JAX can trace one draw at a time | `jax` maps it whatever executor is named; `@function(traceable=False)` runs it one draw at a time | E3b | yes, up to the last bit of rounding |
| a `float64` output spec with 64-bit floats off silently gave `float32` | an error that says to set `jax_enable_x64` | E4 | no |
| an inference method reads `run_seed` and imports `ModelFactors` privately | `InferenceMethod.execute(target, /, *, key, **options)` with an `InferenceTarget`; `ModelFactors` and `make_posterior` are public | E6 | no |
