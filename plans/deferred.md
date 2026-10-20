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
| Paths through every kind, `Function[]`, the accessor `t.output(c)`, and views at held fields with no wrap (step G3) | `declarations-and-addressing.md` § 4.1, B.6 |
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
| Refusing a merge of two views of one parent, or returning their joint view, since today `hier[["s1"]] & hier[["s2"]]` is the product of the marginals | `declarations-and-addressing.md` § 5.6 |
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
| The rest of the compiled layer: compiled lifting if it did not join the core, the mixture sampler, the persistent cache, public `count_compilations` | §6.10 |
| pyabc and BayesFlow on sets of evaluation points and executors; see jhuggins's lighter pyabc route below | §6.12 |
| Tests on a GPU or TPU | §8 |
| The remaining design text, and a User Guide notebook on opaque code, executors, failures, and the memo | §11 |

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
