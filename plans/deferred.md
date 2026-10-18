# Deferred work

This file lists work that the push postpones until after the workshop, and problems found along the way that are outside its scope.
Each entry says where it came from, so it can become a GitHub issue with its context when the push ends.
Postponed work leaves the current behaviour unchanged (README directive 8).

## Planned for after the workshop

### Declarations and addressing

| Work | Source |
|---|---|
| Grammar B for random measures, random functions and Gaussian processes, bootstrap laws, empirical laws of laws, and minibatched targets; then the refusal of a whole declaration over a spec with parts (step G2) | `declarations-and-addressing.md` § 19.8; gaps D4 |
| Paths through every kind, `Function[]`, and the public `component` accessor (step G3) | `declarations-and-addressing.md` § 9, § 19.8 |
| `condition_on` declared as `d.without(*fixed)`, and `law.without` (step G4) | `declarations-and-addressing.md` § 12, § 19.8 |
| `[]` on batches of laws and on kernels (step G5) | `declarations-and-addressing.md` § 9, § 19.8 |
| `&` on `Function`, `FunctionSpec`, `Batch`, and `BatchSpec`, and lifting by postcomposition, so a consumer can meet a function's parts | `declarations-and-addressing.md` § 14.4 |
| `without` on a `Function`, a `ConditionalDistribution`, a batch of laws, and through a law held in a record field, so I10 holds for every kind | `declarations-and-addressing.md` § 3, § 16 |
| `&` on `DistributionSpec` and `ConditionalDistributionSpec`, and `InputSpec` renames, so a function that builds a model declares its output | `declarations-and-addressing.md` § 14.4 |

### The execution layer

From `execution-layer.md` §13, "After the workshop":

| Work | Source |
|---|---|
| `evaluate_at_points` at the remaining sites: LOO, predictive checks, KSD, minibatching, empirical reweighting, SBC | §13 item 1 |
| Failure options: dropping failed evaluation points, retries, `cancel_on_failure`, interrupts | §6.4, §13 item 2 |
| The memo across calls, and `DiskMemo` keyed by `version`; a content-fingerprinted `DiskMemo` after #504 | §6.5, Q6 |
| Supplied derivatives (`vjp`, `jacobian`, `jvp`) after #522, and the `jacobian` operation with `grad`. Their docstrings, V.13, and the user guide must state clearly that a backward pass run after its call closed (an eager `grad` or a kept `vjp`) uses the default context, with an example (decision M18) | §6.9, Q11; § 6.16 rule 7 |
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
| [#554](https://github.com/TARPS-group/prob-pipe/issues/554) Explicit-key randomness | §8 only, in step E6: an inference method receives its run key through its own `key` parameter, which replaces the private `run_seed` | Everything else: `key` parameters on `@function` callables and `Supports*` methods, random-event subdivision, `share_keys_across`, versions per sampling algorithm, `sampler(d)`, the guard on workflow draws inside JAX transformations | gaps D8 |
| [#506](https://github.com/TARPS-group/prob-pipe/issues/506) What a function body receives | Sets of evaluation points present raw values to the callable (step E3a) | Raw forms by default for every body, annotations that ask for tracked terms, one converter registry for every kind, `ArrayBackend.axes` | gaps D8 |
| [#522](https://github.com/TARPS-group/prob-pipe/issues/522) `LinOp` and the differentiability claim | Nothing | All of it, then the execution plan's supplied derivatives (`vjp`, `jacobian`, `jvp`) and the `jacobian` operation on top of it (Q11) | gaps D8 |
| [#451](https://github.com/TARPS-group/prob-pipe/issues/451) One n-ary dispatch registry | Nothing: the executor registry uses today's unary registry | All of it; the executor registry moves to the n-ary class with the others | gaps D8 |
| [#571](https://github.com/TARPS-group/prob-pipe/issues/571) Declarations | Steps S2a, S2b, S3, S4, G1a, and G1b | Step G2 and the deferred `&` kinds listed above; the issue points to the reconciled plan | gaps D1–D4 |
| [#573](https://github.com/TARPS-group/prob-pipe/issues/573) Field paths and subscripts | Step S1 | Steps G3–G5 | gaps D1, D4 |
| [#580](https://github.com/TARPS-group/prob-pipe/issues/580) Execution layer | Steps E0–E8 | The execution-layer work listed above | `execution-layer.md` §13 |
| [#585](https://github.com/TARPS-group/prob-pipe/issues/585) An exact result lost to a budget | Nothing | All of it; its text needs `max_evaluations` after #586 | `execution-layer.md` Q12 |

## Found along the way

| Date | Problem or idea | Found by | Notes |
|---|---|---|---|
| 2026-10-16 | The Rocq model of the component interface does not cover renames by path or paths inside slots. | addressing plan | Extend before the grammar-B design text is final. |
| 2026-10-16 | The Gaussian route of `condition_on` accepts a given record with a missing or extra field. | addressing plan | A bug, separate from the push. |
| 2026-10-16 | Inference on a law declared whole over a record fails with a wrong message. Reproduced 2026-10-18: `condition_on(k * marginal(post, "params"), {"y": ...})`, where `post` is a posterior over a record field `params`, says the draws "must be a numeric array or a record of numeric arrays", though they are a record of numeric arrays; `condition_on.check` also chose empirical reweighting as exact before the call raised. A hand-built law declared whole over a record conditions correctly. | addressing plan; jhuggins on #593 | Grammar B removes the cause (step G1a). G1a is the fifth off-ramp; if it is cut, this stays broken until after the workshop. No workshop notebook reaches this path. |
| 2026-10-16 | A lighter pyabc route: a `pyabc.sampler.Sampler` subclass seeds NumPy's global generator per evaluation from `fold_in(fold_in(run_key, t), i)` and sends the seeds to the executor as one set of evaluation points. A 60-line prototype gave bitwise-identical populations run sequentially, with chunks in reverse order, on 4 processes, and with two batch sizes. The thread executor cannot run these evaluation points. | jhuggins, comment on the execution Doc | Would replace the plan's pyabc reimplementation. |
| 2026-10-17 | Rename `with_level_names` and `with_dim_names` to `rename_levels` and `rename_dims`, so all three renames read alike. | addressing plan; `declarations-and-addressing.md` O13 | Not part of the push. |
| 2026-10-17 | Rename a mixture's `components=` parameter, so "component" keeps one sense in the API. | `declarations-and-addressing.md` O14 | Not part of the push. |
| 2026-10-18 | Rename `LinOp` to `LinearOperator`, with its module and every use, as a clean break. | maintainer | Not part of the push. |
| 2026-10-18 | A comprehensive analysis of ProbPipe's Prefect support, and possibly a refactor: how orchestration, task runners, caching, and the planned `prefect` executor fit together, and what a nested call in a worker should do. | maintainer, decision M21 | After the workshop. |
| 2026-10-16 | Issues to open: deterministic fields of a joint (Q4) and a Gibbs method (Q10). | execution plan | |
| 2026-10-16 | An exact route ruled out only by a budget should be reported, and the call should warn. | execution plan, Q12 | Tracked in #585; its text needs `max_evaluations` after #586. |
