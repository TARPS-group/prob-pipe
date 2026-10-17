# Deferred work

This file lists work that the push postpones until after the workshop, and problems found along the way that are outside its scope.
Each entry says where it came from, so it can become a GitHub issue with its context when the push ends.
Postponed work leaves the current behaviour unchanged (README directive 8).

## Planned for after the workshop

### Declarations and addressing

| Work | Source |
|---|---|
| Grammar-B enforcement and wrapping (step G1) | `addressing.md`: The grammar rule |
| Paths through every kind, `Function[]`, and the public `component` accessor (step G2) | `addressing.md`: Indexing each kind |
| `condition_on` declared as `d.without(*fixed)`, and `law.without` (step G3) | `addressing.md`: Operations that take paths |
| `[]` on batches of laws and on kernels (step G4) | `addressing.md`: Indexing each kind |
| `&` on `Function`, `FunctionSpec`, `Batch`, and `BatchSpec` | `declarations.md` Part 2 §3 |
| `&` on `DistributionSpec`, `ConditionalDistributionSpec`, and `FunctionSpec`, and `InputSpec` renames, so a function that builds a model declares its output | `declarations.md` Part 2 §4 |

### The execution layer

From `execution-layer.md` §13, "After the workshop":

| Work | Source |
|---|---|
| `map_points` at the remaining sites: LOO, predictive checks, KSD, minibatching, empirical reweighting, SBC | §13 item 1 |
| Failure options: dropping failed points, retries, `cancel_on_failure`, per-point output checks, interrupts | §6.4, §13 item 2 |
| The memo across calls, and `DiskMemo` keyed by `version`; a content-fingerprinted `DiskMemo` after #504 | §6.5, Q6 |
| Supplied derivatives (`vjp`, `jacobian`, `jvp`) after #522, and the `jacobian` operation with `grad` | §6.9, Q11 |
| The opt-in executors `futures` and `prefect` | §6.8, Q9 |
| The rest of the compiled layer: inference runners, the mixture sampler, the persistent cache, public `count_compilations` | §6.10 |
| pyabc and BayesFlow on point sets and executors; see jhuggins's lighter pyabc route below | §6.12 |
| Tests on a GPU or TPU | §8 |
| The remaining design text, and a User Guide notebook on opaque code, executors, failures, and the memo | §11 |
| The rest of #524, #554, #506, and #522, beyond the pieces the core absorbs | `gaps-and-conflicts.md` D8 |

## Found along the way

| Date | Problem or idea | Found by | Notes |
|---|---|---|---|
| 2026-10-16 | The Rocq model of the component interface does not cover renames by path or paths inside slots. | addressing plan | Extend before the grammar-B design text is final. |
| 2026-10-16 | The Gaussian route of `condition_on` accepts a given record with a missing or extra field. | addressing plan | A bug, separate from the push. |
| 2026-10-16 | Inference on a law declared whole over a record fails with a wrong message. | addressing plan | Grammar B removes the cause (step G1). |
| 2026-10-16 | A lighter pyabc route: a `pyabc.sampler.Sampler` subclass seeds NumPy's global generator per evaluation from `fold_in(fold_in(run_key, t), i)` and sends the seeds to the executor as one point set. A 60-line prototype gave bitwise-identical populations run sequentially, with chunks in reverse order, on 4 processes, and with two batch sizes. The thread executor cannot run these points. | jhuggins, comment on the execution Doc | Would replace the plan's pyabc reimplementation. |
| 2026-10-16 | Issues to open: deterministic fields of a joint (Q4) and a Gibbs method (Q10). | execution plan | |
| 2026-10-16 | An exact route ruled out only by a budget should be reported, and the call should warn. | execution plan, Q12 | Tracked in #585; its text needs `max_evaluations` after #586. |
