# Deferred work

This file lists work that the push postpones until after the workshop, and problems found along the way that are outside its scope.
Each entry says where it came from, so it can become a GitHub issue with its context when the push ends.
Postponed work leaves the current behaviour unchanged (README directive 8).
A source written "gaps" and an item, such as "gaps D8", names an item of `gaps-and-conflicts.md`, which was settled on 2026-10-18 and deleted; it is in git history at commit `dac760cf`.

## Planned for after the workshop

### Declarations and addressing

| Work | Source |
|---|---|
| Grammar B for random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets, each wrapping its draws under the name its constructor takes; then the refusal of the interim whole form, and the private event default goes (step G2) | `declarations-and-addressing.md` B.6; gaps D4 |
| Paths through every kind, `Function[]`, the accessor `t.output(c)`, and coupled marginals at held fields with no wrap (step G3) | `declarations-and-addressing.md` § 4.1, B.6 |
| `condition_on`'s result with the event spec `d.event_spec.without(*fixed)`, `law.without`, paths into a record input of an `InputSpec`, and the rule that only merge and rename keep factors (step G4) | `declarations-and-addressing.md` § 6.2, § 15, § 4.3, § 10, § 5.2, B.6 (T32) |
| `[]` on batches of laws, on batches of functions, and on kernels (step G5) | `declarations-and-addressing.md` § 6.3, § 16, § 6.5, § 18, B.6 |
| `&` on `Function`, where functions feed as point-mass kernels, and on batches other than `RecordBatch`, element by element | `declarations-and-addressing.md` § 6.4, § 17, B.6 |
| `without` on a `Function`, a `ConditionalDistribution`, a batch of laws, and through a law held in a record field, so I10 holds for every kind | `declarations-and-addressing.md` § 7.1, B.4 (row T30) |
| `&` on `DistributionSpec`, `ConditionalDistributionSpec`, `FunctionSpec`, and `BatchSpec`, and `InputSpec.rename`, so a function that builds a model declares its output | `declarations-and-addressing.md` B.6 |
| A lifting rule inside a merge, so an input that expects a value can be fed a law- or function-valued output | `declarations-and-addressing.md` § 5.6, B.6 item 4 |
| Aligning the levels of two batches by name, with broadcasting, in a batch merge | `declarations-and-addressing.md` § 6.5, § 18, B.6 item 2, B.3 (VI.11) |
| The requester restores each input's output spec from the stored input structure (O4, option 1), once #506 lets a body ask for tracked terms | `declarations-and-addressing.md` O4 |

From Appendix C.4 of the inputs and outputs proposal, merged into `declarations-and-addressing.md` on 2026-10-18; the proposal, `inputs-and-outputs.md`, is in git history at commit `dac760cf`, and "fact N" and "Appendix D" below are its own:

| Work | Source |
|---|---|
| The point-mass reading of merges across categories: a record as the law of one draw, and a function as the kernel whose laws are point masses, so `lik & Record({"mu": 0.3})` and `f & d` have a meaning | `declarations-and-addressing.md` § 5.3 |
| Whether coupling survives binding: whether `k["a"](w)` and `k["b"](w)`, two coupled marginals of a kernel bound at one input value, draw together | `declarations-and-addressing.md` A.2, item 52 |
| `marginal` on batches and functions, which would equal `[]`, since neither kind's selection keeps a dependence that a standalone form drops | `declarations-and-addressing.md` A.2, item 52 |
| A merge that sorts the flattened factors topologically, so a kernel can be added between two merged operands | `declarations-and-addressing.md` E, item 25; the proposal's Appendix D, item 18 |
| The joint's spec for a shared input becomes the unification `U` instead of the first operand's spec; one test asserts today's choice | `declarations-and-addressing.md` § 2.1, § 5.3, D.2 Q4(d) |
| A bug: `condition_on` of a joint kernel ignores an optional input that a merge feeds, as in `condition_on(lik_opt * s_prior, {"mu": 0.0})`; S3's rename carries it unchanged, and G4's rule that only merge and rename keep factors does not fix it, since the fed input does not remain | `declarations-and-addressing.md` § 5.5, C.3; fact 34 |
| Two different dimension names at one axis become one dimension, which changes II.1 and `_unify_specs` | the proposal's Appendix D, item 14 |
| The long-term rule for optional inputs across merges; recommended: a merge that leaves optional inputs shows them, as a kernel whose inputs are all optional | `declarations-and-addressing.md` § 5.2, A.1 item 1, D.2 Q4(b) |
| The unification of kernel and function types as a true meet: inputs take the widest type both accept, events and returns the unification | `declarations-and-addressing.md` § 2.1, A.1 item 4 |
| The dimension scopes of merge results: store a scope on every result, input specs and functions included, with `with_dim_names` renaming stored names, or substitute and forget everywhere | `declarations-and-addressing.md` § 5.7, D.2 Q6 |

### The execution layer

From `execution-layer.md` §13, "After the workshop":

| Work | Source |
|---|---|
| `evaluate_at_points` at the remaining sites: LOO, predictive checks, KSD, minibatching, empirical reweighting, SBC | §13 item 1 |
| Failure options: dropping failed evaluation points, the common executor options `retries`, `retry_errors`, and `bisect`, `cancel_on_failure`, interrupts | §6.3 rule 2, §6.4, §13 item 2 |
| Compiled MCMC that tolerates failures, through a success flag per lane that crosses the boundary next to each output | §6.4, §8 |
| The memo across calls: the `memo` control, `memo=False` for the run table, the `version` parameter, and `DiskMemo` keyed by `version`; a content-fingerprinted `DiskMemo` after #504 | §6.2 item 4, §6.5, Q6 |
| A mapping from a `Function`'s label to an executor on an operation's call | Q3, option C |
| Whether V.8's and #554's "random event" is renamed, for example to "random draw" | Q16 |
| Supplied derivatives (`vjp`, `jacobian`, `jvp`) after #522, and the `jacobian` operation with `grad`. Their docstrings, V.13, and the user guide must state clearly that a backward pass run after its call closed (an eager `grad` or a kept `vjp`) uses the default context, with an example (decision M18) | §6.9, Q11; § 6.16, "Lifecycle", registration rule 7 |
| The opt-in executors `futures` and `prefect` | §6.8, Q9 |
| The rest of the compiled layer: compiled lifting if it did not join the core, the mixture sampler, the persistent cache, public `count_compilations`, which `plans/deferred/execution-introspection.md` replaces with `execution_log()` (its Q3) | §6.10 |
| pyabc and BayesFlow on sets of evaluation points and executors; see jhuggins's lighter pyabc route below | §6.12 |
| Tests on a GPU or TPU | §8 |
| The remaining design text, and a User Guide notebook on opaque code, executors, failures, and the memo | §11 |

A plan written on 2026-10-25, after the code freeze, is in `plans/deferred/`, not yet approved. It lets a user see how a call will run and how it ran, and it lists its open questions in its Appendix D, which are reviewed before any of it is implemented.

| Work | Source |
|---|---|
| Reports of how a call runs: `check` describes each function a call will evaluate (its traceability, whether it is called back from a compiled JAX program, its executor and how it was chosen, whether it is vectorized or compiled, and the probe result it read) and each capability of a distribution the selected route runs; `ProbeResult` is public; `execution_status()` on a `Function`, and on a `Distribution` or `ConditionalDistribution` per capability; provenance records the runs of JAX programs; `execution_log()` in `probpipe.diagnostics` | `plans/deferred/execution-introspection.md` |
| Three bugs the plan found against rules the design already states: an operation's `check` reports no embedded function (#679), an operation's provenance leaves out its evaluations (#680), and a rejected executor's reason prints JAX's raw error text (#681) | `plans/deferred/execution-introspection.md`, Appendix A |

### Numeric terms

A plan written on 2026-10-27, after the workshop's code freeze, in `plans/deferred/` and not yet approved; it lists its open questions, which are reviewed before any of it is implemented. The vector-interface plan below builds on it, and its § 6 lists the edits that plan needs.

| Work | Source |
|---|---|
| One definition of "numeric" for every kind: a numeric value is an array or a record of arrays, and a law, kernel, function, or batch is numeric when its events, inputs, and outputs hold numeric values; `TermSpec.is_numeric` in place of ten private predicates; one `VectorLayout` that every vector conversion uses; `to_vector()` on every numeric kind, returning a term of the same kind, with one `from_vector(spec, vec)`; `NumericFunction`, `NumericConditionalDistribution` meaning both sides numeric, and six markers removed; and `free_dims`, `with_dim_sizes`, and `with_dim_names` on `Function` (steps N1 to N5) | `plans/deferred/numeric-contract.md` |
| Arithmetic on dimension names, so a vector of a `("n",)` array and a `(2,)` array has the symbolic size `n + 2` | `plans/deferred/numeric-contract.md` § 4.5 |
| How a symbolic shape crosses a JAX trace, and how a constructor that evaluates a Python function abstractly, as `conditional_distribution` does, accepts an input of symbolic shape or an opaque input | `plans/deferred/numeric-contract.md` § 4.5, § 5.5 |

### The inference interface and vectors

Two plans written during step E6 (#618) and deferred by the maintainer on 2026-10-21, so as to add no work before the workshop. Both are in `plans/deferred/`, not yet approved; each lists its open questions, which are reviewed before any of it is implemented.

| Work | Source |
|---|---|
| One law-level vector interface for numeric laws, built on the fixed value-level `to_vector`/`from_vector` contract and V.12: `to_vector_map(spec)`, `vector_law(d)` (design III.7's flat-vector law, never built), `vector_bijector(spec)`, `unconstrained_law(d)`, and `log_prob_function(d)`, the raw density for compiled code beside #554's `sampler(d)`; `NumericArrayBatch.to_vector`/`from_vector`; then about twenty private helpers that each lay out laws on their own are deleted (steps V1 to V3, and V4 within P2) | `plans/deferred/vector-interface.md` |
| Every built-in inference method's adapter onto that interface, with `make_posterior(..., coordinates=)`, `initial_vectors`, one key split per run, a start per chain, `factors()`'s order check, and checks that draw and compile nothing (step P2). It changes seeded draws once, a silent change. It rewrites the runners E4 makes module-level, so it reads E4's merged runners first | `plans/deferred/inference-interface.md` |
| Whether `sampler(d)` (#554 § 7) ships with the vector interface or with #554 | `plans/deferred/vector-interface.md` § 4.5 |
| No "array pytree" view of a law for BlackJAX: BlackJAX ravels a pytree position for its mass matrix and covariances anyway, a benchmark found no consistent speed difference (fixed-work parity within 10%), ProbPipe's own records fail as BlackJAX positions, and a `dict` position reorders the vector. A per-leaf "parts" view for TFP or BayesFlow, derived from the same order, waits until a backend needs it | `plans/deferred/vector-interface.md` § 6 |
| A public reading of a target's traceability for inference methods written outside ProbPipe, with the public execution claims of E3a; `plans/deferred/execution-introspection.md` § 4.4 proposes it as `execution_status()` on a law, with a traceability per capability | #618; decided 2026-10-21 |

### Batches

A plan written on 2026-10-27, after the workshop's code freeze, in `plans/deferred/` and not yet approved; it lists its open questions, which are reviewed before any of it is implemented.

| Work | Source |
|---|---|
| One contract for every batch: every element conforms to one element spec in one dimension scope, so a `FunctionBatch` refuses functions of different specs as a `DistributionBatch` refuses laws of different event specs; the element spec inferred as the common spec of the elements, independent of their order; one storage interface and `Batch.stack` in place of the `isinstance` checks of fourteen modules; and `output`, `rename`, `without`, `&`, and equality by value on every batch by one lifted-operation rule, which takes over the batch parts of S4b, G5, and the deferred `&` and `without` entries above (steps B1 to B6) | `plans/deferred/batch-contract.md` |

### Annotations

A plan written on 2026-10-28, in `plans/deferred/` and not yet approved; it lists its open questions, which are reviewed before any of it is implemented.

| Work | Source |
|---|---|
| One rule for when an annotation follows a term: each operation declares its result's relation to its source (same, renamed, part, or new), each entry declares its propagation (term, run, or by path), and one function carries the entries, so a marginal of an MCMC posterior keeps its draws' ArviZ data and the run's sampler statistics; annotations on every term, batches included, written through one method `annotate` (steps A1 to A5) | `plans/deferred/annotations.md` |
| An inference result's ArviZ `posterior` group derived from its atoms rather than stored, so the draws are stored once | `plans/deferred/annotations.md` § 6.1, decision 2 |

### Derivatives

A plan written on 2026-10-28, in `plans/deferred/` and not yet approved; it lists its open questions, which are reviewed before any of it is implemented. It builds on the numeric-contract and vector-interface plans, and it answers the execution plan's Q11.

| Work | Source |
|---|---|
| One contract for derivatives: a continuous leaf is a leaf of a real floating dtype whose support `Constraint.is_discrete` does not call discrete; the derivative of a map is a `LinOp`, its Jacobian operator; every `Function` has a `Differentiability` (inputs, modes, order, source), claimed with paths or `False` and otherwise presumed from traceability; supplied `vjp`, `jvp`, and `jacobian` are installed as JAX rules; a law's derivatives are those of `log_prob_function(d)` and `sampler(d)`; the operations `jacobian`, `grad`, and `hessian`, with `inverse` and `log_det_jacobian` implemented; and gradient methods dispatched on `is_differentiable` rather than on traceability (steps D1 to D6) | `plans/deferred/differentiability.md` |
| An `optimize` operation with Newton, gradient, and gradient-free routes, and the mode and the Laplace approximation of a law derived from it | `plans/deferred/differentiability.md` § 6.2 |

### Controls and options

A plan written on 2026-10-28, in `plans/deferred/` and not yet approved; it lists its open questions, which are reviewed before any of it is implemented. Every operation takes controls, so its § 6 lists the edits the other deferred plans need, the inference-interface plan's first.

| Work | Source |
|---|---|
| One contract for `using`: controls (ProbPipe's names), options (a registered entry's names), and settings (`config`), each declared once by an `Option` that validation, defaults, provenance, the catalog, and the documentation all read; option names checked against every method a call could select; a warning, `UnusedControlWarning`, for each given control or option a call does not read (#675); one vocabulary of common options, which renames `num_results` to `num_draws`; `backend_options` for backend arguments; `describe_using()` and `CallReport.unused`; and the hand-kept lists of control names replaced by one `passes` field (steps C1 to C5) | `plans/deferred/controls-and-options.md` |

### Arrays

A plan written on 2026-10-31, in `plans/deferred/` and not yet approved; it lists its open questions, which are reviewed before any of it is implemented. Its § 9 lists the edits the numeric, batch, and paths plans need.

| Work | Source |
|---|---|
| One contract for arrays: optional dimension names on every axis, kept by `with_dim_sizes`, with a batch level as one named dimension; coordinates held by terms and bound in the dimension scope; every numeric array stored as one JAX array, converted once at the wrap through an array-host registry that reads and exports xarray and pandas; operators that broadcast by position and check names; `isel`, `sel`, `rename_dims`, and `with_coords` apart from paths; one `to_xarray` export in place of three (steps A1 to A6) | `plans/deferred/array-contract.md` |

## Open issues the push changes

Each row says what the push does for an open issue before the workshop, and what the issue keeps for after it.
When the push ends, each issue's text is updated to match, and none closes until its remaining work is done.

| Issue | Done in the push, before the workshop | Kept for after the workshop | Source |
|---|---|---|---|
| [#524](https://github.com/TARPS-group/prob-pipe/issues/524) Finish the function engine | Two module moves, in step E0: `_execution_contract.py` into `_execution.py`, and `_managed.py` with `_broker.py` into `_evaluation_keys.py` | The return step's `ResultKindError`; a lifted body that returns a batch; the four lift departures from V.5 and V.6; the related defects #406, #452, #232 | gaps D8 |
| [#554](https://github.com/TARPS-group/prob-pipe/issues/554) Explicit-key randomness | §8 only, in step E6: an inference method receives its run key through its own `key` parameter, which replaces the private `run_seed` | Everything else: `key` parameters on `@function` callables (`random_key`) and `Supports*` methods, stacked keys for `vectorized` callables, random-event subdivision, `share_keys_across`, versions per sampling algorithm, `sampler(d)`, the guard on workflow draws inside JAX transformations a user opens, and the execution plan's `numpy_generator` and examples 7.5 and 7.7. The refusal of workflow draws inside an embedded evaluation point does not wait for it (`execution-layer.md` § 6.9, item 2) | gaps D8 |
| [#506](https://github.com/TARPS-group/prob-pipe/issues/506) What a function body receives | Sets of evaluation points present raw values to the callable (step E3a) | Raw forms by default for every body, annotations that ask for tracked terms, one converter registry for every kind, `ArrayBackend.axes` | gaps D8 |
| [#522](https://github.com/TARPS-group/prob-pipe/issues/522) `LinOp` and the differentiability claim | Nothing | All of it, then the execution plan's supplied derivatives (`vjp`, `jacobian`, `jvp`) and the `jacobian` operation on top of it (Q11) | gaps D8 |
| [#451](https://github.com/TARPS-group/prob-pipe/issues/451) One n-ary dispatch registry | Nothing: the executor registry uses today's unary registry | All of it; the executor registry moves to the n-ary class with the others | gaps D8 |
| [#571](https://github.com/TARPS-group/prob-pipe/issues/571) Declarations | Steps S2a, which renames "slot" and "component" to "input" and "output"; S2b-out, S2b-in, S3, S4a, S4b, G1a, and G1b | Step G2, and the deferred `&` kinds listed above; the issue points to the merged plan, `declarations-and-addressing.md` | gaps D1–D4 |
| [#573](https://github.com/TARPS-group/prob-pipe/issues/573) Field paths and field selections | Step S1 | Steps G3–G5 | gaps D1, D4 |
| [#580](https://github.com/TARPS-group/prob-pipe/issues/580) Execution layer | Steps E0–E8 | The execution-layer work listed above | `execution-layer.md` §13 |
| [#585](https://github.com/TARPS-group/prob-pipe/issues/585) An exact result lost to a budget | Nothing | All of it; its text needs `max_evaluations` after #586 | `execution-layer.md` Q12 |

## Found along the way

| Date | Problem or idea | Found by | Notes |
|---|---|---|---|
| 2026-10-21 | `blackjax_rwmh` drops the data of a model conditioned on data it does not declare as fields: its `check` passes, but its density receives a bare vector, so it raises, or with an event of matching size would read the parameters as the data. | E6 research, #618 | Fixed by P2 (`plans/deferred/inference-interface.md`, appendix item 1). Deferred with the plans on 2026-10-21. |
| 2026-10-21 | A law over a `(2, 2)` array crashes `blackjax_rwmh` and `tfp_nuts`, and a non-family law over a constrained array crashes every density method, while each method's `check` reports it feasible: five routes evaluate a density on a vector, and only some reshape. | E6 research, #618 | Fixed by the vector interface and P2. |
| 2026-10-21 | The run's key is used twice, for the initial state and for the chains, in BlackJAX NUTS and HMC, random-walk Metropolis, elliptical slice, and SG-MCMC, and every chain of a run starts from one shared state. | E6 research, #618 | Fixed by P2; the fix changes seeded draws once. |
| 2026-10-21 | `blackjax_rwmh` crashes on a scalar law that cannot sample: `get_init_state`'s fallback start is 0-d. | E6 research, #618 | Fixed by P2's `initial_vectors`. |
| 2026-10-21 | SG-MCMC (`blackjax_sgld`, `blackjax_sghmc`) runs in constrained space, so its draws can leave the support. | E6 research, #618 (from reading the code) | Fixed by P2. |
| 2026-10-21 | Elliptical slice assembles the prior's mean and covariance in factor order, and ESS, SG-MCMC, empirical reweighting, and pyabc lay chains out in the prior's order while `make_posterior` reads the target's; the orders agree in every model tried, but nothing enforces it. | E6 research, #618 | P2 enforces it in `factors()`. |
| 2026-10-21 | `flat_vector` lays a mapping out in the mapping's key order, not its spec's, so a nested draw given in another key order is scrambled; a record event in the keyword form is refused by the layout helpers; `unconstrained_coordinates` raises `int('n')` on a symbolic shape. | E6 research, #618 | Replaced by the vector interface. |
| 2026-10-21 | `NumericRecord.__jax_array__` and `__array__` present a record's only field and raise for more than one, where II.3 says they present `to_vector()`; a `NumericRecord` also fails as a JAX pytree that JAX rebuilds with abstract leaves, as a BlackJAX position. | E6 research, #618 | The value level, outside the vector plan's scope. |
| 2026-10-20 | The fingerprint of a `DistributionSpec` includes the name of the output at the root (`core/_fingerprint.py`, its `component=` field), which the decision on equality of 2026-10-20 leaves out. | S0, #605 | With S4b's name-free spec, or after the workshop if S4b is cut. |
| 2026-10-20 | `RecordBatch.stack(..., element_spec: RecordSpec)` names a parameter `element_spec` that holds a record spec, while `element_spec` is now an output spec. | S0, #605 | S4a decides whether to rename it. |
| 2026-10-20 | `functions/_recipe.py` and `functions/_replay.py` both define `_RNG_RECIPE_ABI`, `_STOCHASTIC_PLAN_ABI`, and `_REPLAY_ANCHOR_ABI`, and `functions/_function.py` and `_descendants.py` hold literal sampling and provider versions; each should have one definition in `_evaluation_keys.py`. | E0, #598 | Predates the push. |
| 2026-10-20 | Six near-identical helpers in `core/`, `diagnostics/`, `families/`, `inference/`, `validation/`, and `operations/` wrap `_evaluation_keys._resolve_automatic_key` across a package boundary. | E0, #598 | Predates the push. |
| 2026-10-20 | Two records of a call's Monte Carlo draws coexist: `provenance.metadata["num_samples"]` from #586, and `functions/_recipe.monte_carlo_draws`, which the table footer and `summarize` read. One should derive from the other. | E0, #598; P0, #586 | |
| 2026-10-20 | `.test_durations` lists moved tests under their old names. | E0, #598 | Harmless until the file is refreshed. |
| 2026-10-20 | In `functions/_evaluation_keys.py`, the broker section keeps `_broker.py`'s order, with private helpers before `_AutomaticKeyBroker`, which has a one-line docstring; `functions/_execution.py` re-exports `issue_evaluation_points` and the segment functions, and `_broadcast.py`, `_sweep.py`, and `_function.py` import them through it. | review of E0, #598 | Predates #598. |
| 2026-10-20 | `EvaluationPointOutcome`, one point's outcome in a Prefect payload, exists beside the planned `EvaluationOutcomes`. | review of E0, #598 | E3a and E3b keep the two distinct or merge them. |
| 2026-10-20 | V.9 declares `EvaluationPoints.claims: ExecutionClaims` and `shared_payload() -> SharedPayload`, and `design/` declares neither type. | review of E0, #598 | E2 and E3a declare them. `ExecutionClaims` is declared in V.13 once #596 adds V.13, and #610 adds it when it updates. |
| 2026-10-20 | A probe restores NumPy's and Python's global random states, so a thread that draws from them during a probe loses those draws. | E2, #610 | A probe runs only while a call is planned; a lock or a per-thread check could close this. |
| 2026-10-20 | A route does not reject a `method_options` entry it does not read: `mean.using(method="monte_carlo", method_options={"nonsense": 1})(Normal("x", 0.0, 1.0))` runs, although V.2 says the method rejects it when the call runs. | E1, #596 | Check each route's validation against V.2. |
| 2026-10-20 | Private helpers and parameters still say "declaration" while they hold output specs, such as `_summary_declaration`, `_joint_declaration`, `_event_declaration`, and parameters named `declaration`; "packaged joint" stays in private docstrings. | S2a, #607 | Renamed with the steps that rewrite these functions; "packaged joint" goes in G1a. |
| 2026-10-20 | Prefect's `ProcessPoolTaskRunner` cannot run a ProbPipe task: `TypeError: cannot pickle 'weakref.ReferenceType' object`, since Prefect copies the submitting thread's context and the workflow frame holds a weakref to its thread (`execution-layer.md` § 6.16, "The problem", row 7). | PV, #608 | With the Prefect analysis deferred after the workshop (M21); #608 tests the worker by a pickle round trip into a fresh thread. |
| 2026-10-20 | **A thorough review of the plans and of the code merged in the push.** The maintainer could not review with his usual scrutiny before the workshop, and the design documents and the controls, the amended `/review-pr` among them, were relied on to catch significant errors (decision N13). Review `plans/` and every PR merged in the push against the design, expecting some design and implementation changes, and record them as issues. | maintainer, decision N13 | After the workshop, before `plans/` is deleted. |
| 2026-10-20 | **A ProbPipe writing rule against announcing sentences and fragments, and a sweep of the whole project.** Every sentence states a definition, fact, rule, or reason; no sentence only announces, summarizes, or points to what follows ("Three requirements decide which.", "so the design names it", "Here is how", "This matters because"); no empty asides; no contrastive tails unless they prevent a misreading. `STYLE_GUIDE.md` §10 states it as rule 11, with rules 12 and 13 and §10.1 on explanatory documents; what remains is a candidate detector in `scripts/design/prose.py` and a sweep of `design/`, `docs/`, docstrings, and the plans. | maintainer | After the workshop. |
| 2026-10-20 | **Finalize the declarations design document** (`plans/declarations-and-addressing.md`) after the editing was cut off for implementation: finish the maintainer's review, resolve Appendix A.2's open items, then merge its final wording into the documentation — the conceptual model (Part I) into a new advanced page, "How ProbPipe works", and the code design (Part II) into `design/` — before `plans/` is deleted. The rendered page is built by `plans/render/render_plan.py`. | maintainer | After the workshop; implementation proceeds from the document as it stands. |
| 2026-10-16 | The Rocq model of the component interface does not cover renames by path or paths inside slots. | addressing plan | Extend before the grammar-B design text is final. |
| 2026-10-16 | The Gaussian route of `condition_on` accepts a given record with a missing or extra field. | addressing plan | A bug, separate from the push. |
| 2026-10-16 | Inference on a law in an interim whole form over a record fails with a wrong message. Reproduced 2026-10-18: `condition_on(k * marginal(post, "params"), {"y": ...})`, where `post` is a posterior over a record field `params`, says the draws "must be a numeric array or a record of numeric arrays", though they are a record of numeric arrays; `condition_on.check` also chose empirical reweighting as exact before the call raised. A hand-built law in an interim whole form over a record conditions correctly. | addressing plan; jhuggins on #593 | Grammar B removes the cause (step G1a). G1a is the fifth off-ramp; if it is cut, this stays broken until after the workshop. No workshop notebook reaches this path. |
| 2026-10-16 | A lighter pyabc route: a `pyabc.sampler.Sampler` subclass seeds NumPy's global generator per evaluation from `fold_in(fold_in(run_key, t), i)` and sends the seeds to the executor as one set of evaluation points. A 60-line prototype gave bitwise-identical populations run sequentially, with chunks in reverse order, on 4 processes, and with two batch sizes. The thread executor cannot run these evaluation points. | jhuggins, comment on the execution Doc | Would replace the plan's pyabc reimplementation. |
| 2026-10-17 | Rename `with_level_names` and `with_dim_names` to `rename_levels` and `rename_dims`, so all three renames read alike. | addressing plan; `declarations-and-addressing.md` O13 | Not part of the push. |
| 2026-10-17 | Rename a mixture's `components=` parameter, so "component" keeps one sense in the API. | `declarations-and-addressing.md` O14 | Not part of the push; less pressing once S2a's rename leaves "component" with only this sense. |
| 2026-10-18 | Rename `LinOp` to `LinearOperator`, with its module and every use, as a clean break. | maintainer | Not part of the push. |
| 2026-10-18 | A comprehensive analysis of ProbPipe's Prefect support, and possibly a refactor: how orchestration, task runners, caching, and the planned `prefect` executor fit together, and what a nested call in a worker should do. | maintainer, decision M21 | After the workshop. |
| 2026-10-16 | Issues to open: deterministic fields of a joint (Q4) and a Gibbs method (Q10). | execution plan | |
| 2026-10-16 | An exact route ruled out only by a budget should be reported, and the call should warn. | execution plan, Q12 | Tracked in #585; its text needs `max_evaluations` after #586. |
