# Part IV — Workflow Functions

A **workflow function** is an ordinary function lifted into ProbPipe's world of distributions and values. The user writes a plain function over its "natural" values, and wrapping it makes that function (i) **lift** automatically over distribution- and batch-valued arguments and (ii) a tracked node in a computation graph, so its result carries provenance. The operations of Part V are themselves workflow functions, which is why this part comes first: `sample`, `log_prob`, and `condition_on` inherit the lifting, tracking, dispatch, and orchestration defined here.

## IV.0 — Overview: what a workflow function adds

Wrapping a function `f` layers four things on top of calling `f`, each defined in a section below:

| §     | Concern                  | What it adds to `f`                                                                                                                                                                                     |
| ----- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| III.1 | the wrapper              | `f` becomes a tracked node in a computation graph; a plain call runs `f` and returns a `Tracked` result with provenance.                                                                                |
| III.2 | lifting                  | a distribution over a value that is passed where a value is expected is sampled and `f` is applied per draw (an empirical distribution over outputs); a batch is swept; correlated arguments co-sample. |
| III.3 | controls vs. arguments   | ProbPipe controls (sample count, seed, …) are kept in a namespace separate from that of the arguments to `f`                                                                                            |
| III.4 | dispatch & orchestration | *how* the per-draw calls run computationally and *whether* they are traced for lineage.                                                                                                                 |
## IV.1 — `WorkflowFunction`

### Contract

A `WorkflowFunction` wraps exactly one function and presents it as a node in a computation graph. It is created with the `@workflow_function` decorator:

```python
@workflow_function
def predict(theta, x): ...                    # an ordinary function over concrete values

@workflow_function(seed=42, dispatch="jax")    # optional construction-time controls
def predict(theta, x): ...
```

Calling it runs the wrapped function and returns a `Tracked` result: the output is wrapped as a value or distribution carrying `Provenance` that records this workflow function and its tracked inputs. A call whose arguments are all ordinary values is one invocation of `f` followed by that wrap. A distribution- or batch-valued argument triggers lifting instead.

A workflow function is a node in a directed graph: arguments that are themselves workflow objects become graph **dependencies**, and the rest are plain **inputs**. That graph is what provenance and orchestration traverse. A workflow function may also belong to a *module* that supplies some of its inputs and dependencies, but the unit of execution is always the single wrapped function.

### Rationale

This makes `C1 – Uniform interface to distributions and values` and `C4 – Function lifting via pushforward` operational: a user writes mathematics as an ordinary, testable function, and ProbPipe lifts it to act on distributions and values without the function being rewritten. Making every workflow function a graph node is what delivers `C6 – Traceable and reproducible workflows` — each result records how it was produced — and what later lets a whole workflow be traced or re-run. Because the wrapper changes only invocation and tracking, the operations can be *defined* as workflow functions and inherit all of it.

## IV.2 — Lifting over distributions and batches

### Contract

A workflow function compares each argument against the type its function expects, and lifts where they differ:

- **A distribution where a value is expected → broadcast.** The distribution is sampled `n` times and the function is applied to each draw. The result is an **empirical distribution** over the outputs, the pushforward of the input through `f`. If the function genuinely expects a distribution, as its annotation indicates, the distribution passes through unlifted.
- **A batch where one element is expected → sweep.** The function is mapped over the batch's elements, returning a batch of outputs.
- **Both at once → a nested sweep of broadcasts.** The function is mapped over the batch's elements, with a broadcast performed within each.
- **Neither → a plain call.**

**Correlation is preserved.** Arguments that are *views of one parent* (sibling field views) are co-sampled from a single parent draw, so dependence between them flows through `f` rather than being broken by independent sampling. (Operationally, broadcast arguments are grouped by parent identity, and each group draws once.)

By default the result holds only the outputs, but setting `include_inputs = True` instead returns the **joint** distribution over the sampled inputs *and* the outputs.

### Rationale

This is `C4 – Function lifting via pushforward` realized: replacing any argument of `f` with a distribution over that argument's type leaves `f` well-defined, and the result is the pushforward. Doing it by sampling keeps the contract fully general (`D2 – Generality first`) — it works for any `f` and any distribution, with closed-form shortcuts reserved for specific cases — and leaves the user's function untouched. Co-sampling by parent is what makes the lift *correct* rather than merely type-correct: it is the same correlation-preserving mechanism the field views rest on, so passing sibling views through a function transports their joint law.

### Open points

- *Default sample count.* How many draws a broadcast takes by default (a speed-versus-accuracy ceiling, with an explicit per-call override always available) is unsettled. The default should signal "rough estimate," not "tuned."

## IV.3 — Controls vs. arguments

### Contract

A workflow function keeps two namespaces strictly apart:

- **The wrapped function's arguments.** Every positional and keyword argument of a call binds to the wrapped function. 
- **ProbPipe controls.** The sample count, PRNG seed, and the `include_inputs` switch are set on the decorator (construction time) or, for a single call, through a `with_options` view:

```python
predict.with_options(n_broadcast_samples=1000, seed=7)(theta=prior, x=x_obs)
#   controls go to with_options;  theta and x bind to the wrapped function
```

### Rationale

A workflow function must wrap an *ordinary* function with no naming restrictions (`C5 – Naming for unambiguous meaning`): a user should never have to rename a `seed` parameter because the framework wanted that word. Holding the control plane in a separate namespace removes the collision entirely, while keeping the bare decorator and a single call site ergonomic.

## IV.4 — Dispatch and orchestration

### Contract

Two orthogonal computational concerns sit beneath a lifted call, both with defaults so a user need not touch them:

- **Dispatch — *how* the per-draw / per-element calls run.** `jax` vectorizes them (one `vmap`); `sequential` runs them one at a time; `thread` runs them on a thread pool; `auto` probes whether the call is array-traceable and picks `jax`, falling back to `sequential`. Under `jax`, a lifted call is differentiable end-to-end.
- **Orchestration — *whether* the call is traced.** Off by default; a workflow function can instead run as a traced task or flow, recording the computation graph for lineage and scheduling. Tracing is opt-in and never changes the result.

### Rationale

Dispatch and orchestration are `C3 – Computational detail hidden by default, available on demand` in action: the algorithm that realizes a lifted call, and whether its graph is recorded, are computational concerns — handled automatically by default, exposed for users who need control. Keeping them orthogonal (how a call runs is independent of whether it is traced) lets the fast vectorized path and full lineage tracking compose rather than trade off.

### Open points

- *Non-array backends.* Lifting and dispatch are array-native, built for a differentiable array backend; first-class support for other tensor frameworks (e.g., a Torch model as the wrapped function, with conversion at the boundary) is not yet settled. This should be feasible using, e.g., Keras. 
