> **AI-generated.** An AI assistant drafted this plan and ran its measurements, and no maintainer has reviewed it yet. Please comment on the pull request that adds it.

# Compiled execution of ProbPipe's JAX work: a plan

ProbPipe runs most of its numerical work in JAX, but it calls JAX in a way that makes XLA compile the same code again on every call.
A warm `sample` of the first tutorial's model spends more than a second compiling, and a repeated `condition_on` spends several.
Issue #531 reported the problem for `sample` and prototyped two fixes.
This plan measures the problem across the package, states the principles a fix should follow, specifies the fix of each slow path, and plans the documentation that guides users who write JAX functions for ProbPipe.

The plan has four parts:
1. **Principles:** how ProbPipe should handle its JAX backend.
2. **The slow paths and their fixes:** each path's cause, its change, and the measurement behind both.
3. **User-side decisions:** the choices a user makes when writing a JAX function for ProbPipe, and the documentation that guides them.
4. **Order of work, tests, and decisions:** the sequence of pull requests, the tests that guard the result, and the questions for the maintainers.

---

## Background

### The model the measurements use

Most measurements use the model of Tutorial 1 (`docs/tutorials/01_first_analysis.ipynb`).
It has three priors and one likelihood:
- **the priors:** a carrying capacity `K ~ LogNormal`, a growth rate `r ~ LogNormal`, and a detection rate `phi ~ Beta`;
- **the likelihood:** a conditional distribution of 22 yearly counts `y ~ Poisson(phi * N)`, where `N` follows the Ricker recursion `N[t+1] = N[t] * exp(r * (1 - N[t] / K))`, written with `jax.lax.scan`.

The joint is `model = likelihood * prior`, a factored distribution.
`sample(model, n)` draws the priors and then maps the likelihood's function over the `n` drawn values with `jax.vmap`.

### Why the current code compiles on every call

Outside `jax.jit`, a `lax.while_loop`, `lax.scan`, or `lax.cond` traces its body, lowers it, and compiles it.
JAX caches that work by the identity of the body callable.
ProbPipe hands JAX a new callable on every call in three ways:
- **TFP's rejection samplers:** Gamma, Beta, Poisson, Binomial, StudentT, and VonMises, and the families built on them, define their loop bodies as closures inside `sample`, and ProbPipe calls `sample` without a jit.
- **Fresh lambdas under `jax.vmap`:** a joint maps each conditional factor with `jax.vmap` over a lambda built during the call, so the factor's loops, both TFP's and the user's `lax.scan`, are traced and compiled again.
- **Fresh closures in inference:** each `condition_on` builds a new target density, kernel, and scan body.

Simple element-wise operations are cached per primitive and shape, so they do not recompile.
Only two places in `probpipe/` call `jax.jit` (`inference/_pyabc.py:180` and `inference/_blackjax_sgmcmc.py:235`), and neither reuses its compiled code across calls.

`tfp_eager_vs_jit.py`, taken from #531, measures TFP alone; the times are seconds for calls 1, 2, and 3, each with a new key:

| Family | eager | `jax.jit` with the backend as an argument |
|---|---|---|
| LogNormal (closed form) | 0.282 / 0.002 / 0.001 | 0.129 / 0.000 / 0.000 |
| Beta (rejection) | 1.631 / 0.477 / 0.448 | 0.536 / 0.001 / 0.000 |
| Gamma (rejection) | 0.247 / 0.236 / 0.254 | 0.387 / 0.001 / 0.000 |
| Poisson (rejection) | 2.193 / 0.678 / 0.712 | 0.693 / 0.002 / 0.002 |

At one key the jitted Beta and Poisson draws equal the eager draws, and the jitted LogNormal and Gamma draws differ by a relative 5e-7, because XLA fuses `exp` differently when it compiles.

### Measurement setup

All measurements were taken on `dev/overhaul` at `d6053137`, with Python 3.14.4, JAX and jaxlib 0.10.1, and TFP 0.26.0-dev20260609.
They ran on the CPU backend of an 8-core Apple-silicon laptop with 16 GB of memory.
Each benchmark ran in its own process, and no process exceeded 1.9 GB of resident memory.

Wall-clock times on a shared laptop are noisy, so the tables also count the XLA compilations and the jaxpr traces of each call.
The counts come from `jax.monitoring` listeners on the events `/jax/core/compile/backend_compile_duration` and `/jax/core/compile/jaxpr_trace_duration`.
A count does not depend on machine load.
In the tables, `c=` is the number of compilations, with their total seconds in parentheses, and `t=` is the number of traces.
A "warm" call is the third call of the same operation in one process.

The scripts are in `review/jax-performance-bench/`, and the appendix lists them.
Each script takes a mode, which selects the code under test:
- **`baseline`:** the code as it is;
- **`sampler`:** `TFPDistribution._sample` routed through one module-level jitted function that takes the TFP distribution as an argument;
- **`kernel`:** the `sampler` change, plus a jit of the vmapped function that draws a conditional factor in `_conditional_draw`, cached per factor.

The `sampler` and `kernel` modes are the two prototypes of #531, applied as monkeypatches.
They are measurement tools, and Part 2 specifies the changes the real code needs.

---

## Summary

1. **One cause slows many operations.**
   Every warm call of these operations compiles code again:

   | Operation | warm time | compilations |
   |---|---|---|
   | `sample(model, 50)` | 1.25 s | 5 |
   | `mean(model)` (Monte Carlo) | 1.30 s | 5 |
   | `quantile` of a Beta | 0.21 s | 1 |
   | a `@function` lifted over a prior with a Beta component | 0.37 s | 2 |
   | `condition_on.check(model, observed)` | 0.75 s | 5 |
   | `condition_on` with NUTS, repeated | 5.1 s, of which 3.6 s compiles | 19 |
   | `sample` of a mixture of two Betas | 0.85–2.6 s | 4–51 |

2. **Compiled programs with stable identities remove the repeated cost.**
   The `kernel` mode reduces a warm `sample(model, 50)` from 1.25 s to 0.006 s.
   The code cells of Tutorial 1 then run in 8.7 s instead of 19.9 s, without counting imports.
   A jitted runner that takes a `jax.tree_util.Partial` over a stable function was confirmed to compile once and then reuse its program when only the data change.
3. **Two hazards must be fixed before any jit ships.**
   - **The mixture sampler is not stable under jit.**
     It takes a different algorithm when its input is traced, so its eager and jitted draws at one key differ by up to 0.52.
     Under the `sampler` change it would also compile on every call, because the number of draws from each component changes from call to call.
   - **A cache keyed by `id()` is unsafe.**
     The `kernel` prototype keys its cache on `id(factor)`, and Python reuses an id once its object is freed.
4. **JAX's persistent compilation cache complements the fix.**
   With the cache enabled, a cold call in a second process is about 5 times faster, both on the current code and on the fixed code.
   ProbPipe should offer the cache as an opt-in, and CI should enable it.
5. **Users should not need to jit.**
   Under this design a user writes a pure function in `jax.numpy` and `lax`, and ProbPipe compiles it.
   Under the `sampler` change alone, a user's `jax.jit` still changes the warm `sample(model, 50)` 7-fold (0.068 s against 0.010 s); under the full design it changes nothing (0.006 s either way).
   Other choices matter more once the fix is merged:
   - **loops:** a 200-step Python loop takes 9.2 s to compile, against 0.6 s for `lax.scan`;
   - **rebuilding a model:** a model built again for every call costs 0.62 s a call, against 0.02 s for a reused model;
   - **purity, fixed shapes, and the first-call cost**, which Part 3 explains.
6. **The existing tests tolerate the change.**
   Under the `kernel` mode, `tests/families`, `tests/distributions`, and `tests/operations` pass except for one assertion, whose default tolerance of `rtol=1e-7` is below the fusion difference of a relative 2.3e-7.

---

## Part 1. Principles

These principles state how ProbPipe handles its JAX backend.
Each later section refers to them by number.

**J1. Python orchestrates, XLA computes, and the jit boundary is at the raw layer.**
An operation such as `sample`, `log_prob`, or `condition_on` has two layers:
- **orchestration:** binding the arguments, selecting a route, deriving keys, building records, and recording provenance;
- **computation:** the raw methods `_sample(key, sample_shape)` and `_log_prob(value)`, their conditional twins, a lifted function's body, and an inference method's chain runner.

The orchestration layer stays eager Python and runs once per call.
The computation layer runs as compiled programs.
The boundary is below the derivation of keys, for two reasons:
- **the key guard:** `design/05-functions.md` § V.8 ("Keys") refuses a workflow-owned random event inside a transformation the caller opens, so a jit above the key derivation would raise;
- **term identity:** `design/02-shared-abstractions.md` makes identity boundary-attached, so labels and provenance never enter a trace, and terms already enter a trace as their raw representation with their spec as the only static data.

The design therefore adds no contract to the operations.
It changes how the raw methods execute.

**J2. Every compiled program has a stable identity, and per-call data enter as arguments.**
A program ProbPipe compiles is either a module-level jitted function or a jitted function held by an owner that persists across calls, such as a kernel, a `Function`, or a model.
It is never a closure built during the call.
Values that change between calls are pytree arguments:
- keys;
- parameters;
- drawn and fixed givens;
- observed data.

Configuration is a static hashable argument:
- a sample shape;
- a number of steps;
- an algorithm name.

JAX's cache is then keyed by structure: tree definition, shapes, dtypes, and static arguments.
It is never keyed by values.
The invariant is that a warm call with the same structure compiles nothing, and Part 4 tests it by count.

**J3. Eager and compiled execution follow one code path.**
A raw method never branches on whether its input is traced in a way that changes its result.
The traced algorithm is the canonical one, and an eager call runs the same program.
At a fixed key the draws are then the same whether ProbPipe or the user compiles the call, up to the floating-point effects of fusion (J8).
A branch that only skips a check or a cache write for a traced input is allowed, because it does not change results.

**J4. Compilation is an optimization, and it is never a requirement.**
When a computation cannot be traced, ProbPipe runs it eagerly or one draw at a time, as it does today.
Three examples:
- a function written in NumPy or SciPy;
- a PyMC or Stan backend;
- givens that are not numeric.

ProbPipe decides once for each owner and input structure, and stores the decision with the owner, so a call never retries a trace that already failed.
The existing reports, `.check()` and `CallReport`, show the route a call takes.
A call does not warn each time it runs eagerly.

**J5. A function a user passes to ProbPipe is pure.**
Principle C2 (`design/01-design-principles.md`) applies to operations.
This design extends it to every callable a user passes in:
- a `@function`;
- the function of `conditional_distribution`;
- the sampler and density of `distribution`;
- the statistics of `predictive_check`;
- the integrand of `expectation`.

Such a callable returns a result determined by its arguments, and it has no side effects that matter.
ProbPipe may trace it once and reuse the trace.
A global or a closed-over value is therefore read when ProbPipe first traces the callable, and a later change to it is not seen until the user builds the object again.
This is the behavior `jax.jit` has.
The user does not need to jit the callable, and jitting it is harmless (Part 3).

**J6. Importing ProbPipe does not change global JAX settings.**
ProbPipe leaves `jax.config` unchanged on import, because every setting there belongs to the user's whole process.
Examples are the persistent compilation cache, `jax_enable_x64`, and the debug flags.
ProbPipe offers the persistent cache through an explicit call, and CI and the documentation build enable it.

**J7. Caches are safe and bounded.**
A compiled program is held in one of two places:
- **a per-owner store**, which is dropped on pickle and released with its owner;
- **a module-level jitted function**, whose own JAX cache is keyed by tree structure.

No cache is keyed by `id()`.
No module-level dictionary grows with the number of objects a session creates.

**J8. Reproducibility holds within one environment.**
A seeded result is reproducible for a fixed ProbPipe version, JAX version, backend, and device count.
Compiling a computation may change results in the last float32 digit, a relative difference near 5e-7, because XLA fuses operations differently.
A test asserts against an independent baseline with a measured tolerance (`STYLE_GUIDE.md` §8.6), and never bit for bit against stored draws.

**J9. Compilation is measured by counts.**
A test asserts the number of compilations of a warm call rather than its wall-clock time, because the count does not depend on machine load.
The same counter is offered to users as a diagnostic (Part 3).

---

## Part 2. The slow paths and their fixes

### 2.1 Measurements of the tutorials

`tut1.py` builds the Tutorial 1 model, calls each operation three times, and counts compilations.

| Call | baseline, call 1 | baseline, warm | `sampler`, warm | `kernel`, warm |
|---|---|---|---|---|
| `sample(K, 1000)` (LogNormal) | 0.25 s, c=8 | 0.003 s, c=0 | 0.002 s | 0.002 s, c=0 |
| `sample(prior, 100)` | 2.02 s, c=48 | 0.52 s, c=2 | 0.003 s | 0.003 s, c=0 |
| `sample(model["r"], 10)` | 3.97 s, c=90 | 1.09 s, c=5 | 0.05 s | 0.003 s, c=0 |
| `sample(model, 50)` | 4.29 s, c=83 | 1.25 s, c=5 | 0.07 s | 0.006 s, c=0 |
| `sample(condition_on(model, ...), 50)` | 1.87 s, c=45 | 0.51 s, c=2 | 0.002 s | 0.002 s, c=0 |
| `mean(model)` (Monte Carlo) | 4.37 s, c=92 | 1.30 s, c=5 | 0.09 s | 0.04 s, c=0 |

The Beta factor's sampler accounts for 2 compilations per warm call, and the Poisson likelihood under `vmap` accounts for the other 3.

`nbprofile.py` runs a notebook's code cells in one process and records the time and compilations of each cell.
The wall times below exclude the cell that imports ProbPipe, which takes about 5 s.

| Notebook | time of the cells | compile time | slowest cells |
|---|---|---|---|
| Tutorial 1, baseline | 19.9 s | 16.6 s | the first `sample` of the model, 5.0 s with 109 compilations; the informed prior, 5.6 s; the quantiles, 5.3 s |
| Tutorial 1, `kernel` | 8.7 s | 7.4 s | the same cells take 2.0 s, 1.9 s, and 2.6 s |
| Tutorial 2, baseline | 34.7 s | 26.7 s | NUTS `condition_on`, 9.7 s with 147 compilations; `condition_on.check`, 3.8 s with 111 compilations; the short NUTS run, 7.9 s |

In Tutorial 1 under the `kernel` mode, 85% of the remaining time is first-call compilation, which only the persistent cache (2.9) reduces.

### 2.2 The parametric families

**Where.** `probpipe/families/_backend.py`:
- `TFPDistribution._sample` (line 370) and `TFPDistribution._log_prob` (374);
- `_backend_quantile` (142);
- `_TFPArrayBackend._sample` and `_log_prob` (569–577), which delegate to the first two.

**Change.**
1. Add module-level jitted functions that take the TFP distribution as a pytree argument.
   The names must differ from the existing `_backend_quantile`:

   ```python
   @functools.partial(jax.jit, static_argnames="sample_shape")
   def _jitted_sample(backend, key, sample_shape):
       return backend.sample(seed=key, sample_shape=sample_shape)

   @jax.jit
   def _jitted_log_prob(backend, value):
       return backend.log_prob(value)

   @jax.jit
   def _jitted_quantile(backend, q):
       return backend.quantile(q)
   ```

2. Route `_sample`, `_log_prob`, and the quantile capability through them.
3. Leave the closed-form moments (`_backend_mean`, `_backend_variance`, `_backend_cov`) eager.
   A warm `variance(phi)` compiles nothing and takes 1 ms, so jitting them gains nothing.
4. Call each through the fallback helper of 2.8.
   TFP's own distributions are pytrees, but `TFPDistribution` accepts any `tfd.Distribution`, and a subclass that is not a pytree must run eagerly (J4).
   The helper keys that decision on the backend's type.

**Why here.**
- **The cache key:** it is the backend's tree structure (family, parameter shapes, dtypes) plus the sample shape.
  A new parameter value of the same shape reuses the compiled program, and so do a GLM's per-call laws and every law a kernel builds.
- **Inside an enclosing trace:** in a kernel's `vmap` or an MCMC `scan`, a nested `jax.jit` is inlined, so the change costs nothing there.
- **Quantiles:** they are included because TFP's `betaincinv` loops.
  A warm `quantile(phi, levels)` takes 0.21 s with one compilation today, while LogNormal's closed-form quantile takes 0.002 s.

**Measured effect.** Under the `sampler` mode the warm `sample(prior, 100)` drops from 0.52 s to 0.003 s.

**TFP argument validation.**
ProbPipe never passes `validate_args=True`, so no TFP runtime assertion is lost under jit.
A comment beside `_jitted_sample` should state that dependency.

### 2.3 The mixture sampler

**Where.** `probpipe/families/_mixture.py:113–148`, in `_mixture_sample`.

**Problem.**
When the component choice is concrete, the sampler copies it to the host with `np.asarray(choice)` and draws `len(where)` values from each component.
When the choice is traced, it draws `sample_shape` values from every component and selects among them.
`mixture.py` measures two consequences on a mixture of two Betas:
- **The draws differ.** At one key, the eager and jitted draws differ by up to 0.52, which violates J3.
- **It recompiles on every call.** The number of draws from each component changes from call to call.
  Today a warm call takes 0.85–2.6 s with 4–51 compilations.
  Under the `sampler` mode it still takes 1.0 s with 3 compilations, because each new count is a new static sample shape.

**Change.**
Use the traced algorithm, `_drawn_from_every_component`, for every call, and delete the eager branch.
Each of the K components then draws the full sample shape, which costs little next to a compilation for the small K a mixture has in practice.
When every component belongs to one family, a gather over stacked parameters could replace the K draws later.

**Consequence.** The seeded draws of every mixture change, so their baselines must be regenerated (J8).

### 2.4 Conditional factors of a joint

**Where.**
- `probpipe/distributions/_factored.py`: `_conditional_draw` (735–762), `_each_value` (642–656), and `_score` and `_conditional_score` (765–805);
- `probpipe/distributions/_conditional.py`: `_FunctionKernel` (1002–1179), whose `_law` (1149–1179) calls the user's function.

**Problem.**
`_conditional_draw` passes a new lambda to `_each_value`, which applies `jax.vmap` without `jax.jit`.
The mapped function is therefore traced, and its loops compiled, on every call.
In the Tutorial 1 model these are the Poisson rejection sampler and the user's `lax.scan`.
`_score` and `_conditional_score` have the same structure for densities.
The score path costs time even without a loop.
A warm `log_prob(prior, 64 draws)` takes 44 ms, and the profile shows about 150 primitive dispatches per call through JAX's batching interpreter in Python.
One jitted call replaces them.

**Change.**
1. **Give each kernel a program store.**
   A `_FunctionKernel` creates one `ProgramCache` (2.8) in `__init__`.
   Every copy of the kernel shares that store by reference:
   - the curried copies `_condition_on` makes;
   - the copies `with_label` makes;
   - the dimension-bound copies of the factor graph.

   A pickle drops the store.
   The kernel is the right owner because a factor's program depends only on the user's function, the slots, and the bound values.
   A model rebuilt from the same kernel object reuses its programs.
   A kernel built again, as when a notebook cell that decorates the function runs again, compiles fresh programs.
   The design rejects `id()` keys (J7).
   It also rejects weak references to the user's function, because a bound method is recreated on each attribute access.
2. **Compile the mapped draw.**
   In `_conditional_draw`, when `_numeric_givens(factor, inner)` holds, get or build this program:

   ```python
   def one(values, fixed, bound, subkey):
       given = _factor_given(factor, values, fixed)
       return factor._sample_with_bound(bound, given, subkey)

   program = jax.jit(jax.vmap(one, in_axes=(0, None, None, 0)))
   ```

   `_sample_with_bound` is a new private method of `_FunctionKernel`.
   It builds the law from `bound` passed in, rather than from `self._bound`, and draws from it.
   Three rules govern the program:
   - **Changing values are arguments.** The drawn givens, the fixed givens, and the curried values in `bound` enter as arguments, never as closure constants.
     The program is therefore reused across curried copies and new data.
   - **The key is structural.** It holds the inner slots, the factor's given and event specs (dimension-bound copies may differ in their dimensions), and the tree structure of `fixed` and `bound`.
   - **Other values are static or fall back.** A `fixed` or `bound` value that is not an array goes into the key when it is hashable.
     Otherwise the call falls back to today's eager `vmap` (J4).
3. **Compile the scores the same way.** `_score` and `_conditional_score` get programs for `_conditional_log_prob` and `_conditional_unnormalized_log_prob`.
4. **Simplify `_each_value`.** It takes a stable callable and the program store, so the lambdas it receives today disappear.
   The draw and the score then share one implementation of step 2.
5. **Leave the root factors alone.** `_ancestral_sample` draws them through `_sample`, which 2.2 already compiles.
   Compiling the whole ancestral sample as one program would save a few dispatches.
   Per-factor programs are the reliable unit, because each factor can fall back on its own (J4).
   Whole-joint compilation should be measured only if the dispatch overhead appears in a profile.

**Measured effect.**
Under the `kernel` mode, the warm `sample(model, 50)` drops from 1.25 s to 0.006 s with no compilation, and the warm `sample(model["r"], 10)` drops from 1.09 s to 0.003 s.
#531 reports that the draws of `sample(model, 50)` under the `kernel` mode equal those under the `sampler` mode bit for bit, and Part 4 turns that into a test.

**Questions #531 raised about this change, answered by experiment.**
- **Must a kernel's function be pure?**
  Under this design a kernel's function is traced once for each input structure.
  J5 states the requirement, and Part 3 documents it.
  Because the program store belongs to the kernel object, running the decorator again gives a fresh trace, which is what a notebook user expects.
- **What happens when a kernel's function calls a random operation such as `sample`?**
  `concrete.py` defines a kernel whose function calls `sample(Normal(...))`.
  Both the baseline and the `kernel` mode raise when the kernel is built, because `_probe` runs the function under `jax.eval_shape` (`_conditional.py:1309`).
  A jit therefore changes nothing.
  The message is misleading, though, because it blames a transformation "opened by the caller" when ProbPipe opened it.
  The change should run `_probe` and the kernel programs inside `_workflow_jax_runtime_guard`.
  It should raise a message that names the cause: the function of a conditional distribution returns a law and cannot draw with a random operation.
- **Can every kernel on the vectorized path be jitted?**
  `concrete2.py` defines a kernel that calls `int()` on a given.
  Both modes refuse it when it is built, because the probe traces every required slot abstractly.
  Every kernel that takes the vectorized path therefore already traces with all of its required givens abstract.
  One case remains: an optional slot that the probe fills with its default but a later call fixes to another value.
  The fallback of step 2 covers it.

### 2.5 Laws built from functions, transformed laws, and the empirical expectation

**Where.**
- `probpipe/distributions/_from_functions.py:88–104` (`_draw`) and `108–131` (`_score`);
- `probpipe/families/_transformed.py:124–153` (`_per_point`, `_transformed_sample`, `_change_of_variables`) and `168–177` (`_affine_jacobian`);
- `probpipe/distributions/_empirical.py:666–693` (`_expectation`).

**Change.**
Each of these sites applies `jax.vmap` to a fresh lambda, the structure of 2.4.
Give each owner a program store and jit the mapped body, with the per-call values as arguments.
The owners are the law `distribution` builds and the transformed law.
A user's sampler built on `jax.random` alone does not recompile today, so the gain is largest for samplers that build TFP families or use `lax` control flow.

`EmpiricalDistribution._expectation` receives a new integrand `f` on each call.
Its program can be cached on the `Function` that wraps `f` when there is one, and is otherwise built per call.

These sites have lower priority than 2.2–2.4, because no tutorial cell measured here is slowed by them.

### 2.6 Function lifting and the dispatch probe

**Where.**
- `probpipe/functions/_broadcast.py:670–735` (`_mapped_evaluator`) and `1021–1039` (`mapped_draw_body`);
- `probpipe/functions/_sweep.py:429–471`;
- `probpipe/functions/_function.py:727–754` (`resolve_dispatch`) and `903–1048` (the traceability probe).

**Problem.**
A lifted call, such as Tutorial 2's `capacity_count(prior["K"], prior["phi"])`, builds a new `single_call` closure and applies `jax.vmap` to it without a jit.
Under `dispatch="auto"` it first runs `jax.make_jaxpr(jax.vmap(...))` to decide whether the call traces, and it keeps that decision only for the call.
A warm call of `capacity_count` takes 0.37 s, with 2 compilations (the Beta sampler) and 3 traces.
After 2.2 the compilations disappear, but each call still traces twice and dispatches its operations one at a time.

**Change.**
1. Give each `Function` a program store, shared by its copies that keep the same callable.
2. Build the mapped program once per key.
   The key has three parts:
   - the `broadcast_args` references;
   - the tree structure of the lifted values;
   - for each other argument, its tree structure and shapes when it is an array tree, or the value itself when it is hashable.

   An argument that is neither makes the call fall back to today's eager `vmap` (J4).
3. Pass the lifted values and the array-valued other arguments to `jax.jit(jax.vmap(one_draw, in_axes=...))`.
   The other arguments enter with `in_axes=None`.
4. Store the `auto` probe's outcome under the same key.
   Traceability depends on the structure of the inputs, not on their values, so a later call with that structure skips the probe.
5. Keep `_workflow_jax_runtime_guard` around the call that traces.
   A warm call does not trace, so the guard costs nothing there.
6. Keep the Prefect task and flow wrappers, which wrap the call to the program and need no change.

**A check before merging.**
Under `vmap` today, `one_draw` runs the engine's per-point wrapper (binding, input normalization, and output checks) once per call.
Under jit it runs once per trace.
The change must confirm that the wrapper has no side effect needed on every call.
Two facts suggest it has none:
- **Provenance:** its creation is skipped inside the guard.
- **The output's support check:** it is skipped when traced and repeated after the map (`values/_function_base.py:497–561`).

### 2.7 Inference runners

**Where.**
- `probpipe/inference/_blackjax_mcmc.py:150–256` (`_run_blackjax_chains`);
- `probpipe/inference/_inference_utils.py`: `is_jax_traceable` (191–204), `get_init_state` (691–790), `build_target_log_prob` (809–836), `unconstrained_chain` (1054–1106), `run_chain_scan` (1163–1190), and `parallel_chain_map` (1198–1234);
- `_tfp_mcmc.py:36–90`, `_blackjax_rwmh.py:226–384`, `_blackjax_ess.py:123–174`, and `_blackjax_sgmcmc.py:226–243`.

**Problem.**
Every `condition_on` call builds a new target closure, a new kernel, and new scan bodies, then runs the warmup and sampling scans without a jit.
`hotpaths.py` runs NUTS with 2 chains of 200 warmup and 200 sampling steps on the Tutorial 1 model, twice in one process:

| | call 1 | call 2 |
|---|---|---|
| `baseline` | 8.7 s, c=184 (6.5 s) | 5.1 s, c=19 (3.6 s) |
| `kernel` | 8.0 s, c=157 (6.0 s) | 2.8 s, c=7 (1.7 s) |

Even under the `kernel` mode, a repeated `condition_on` spends 1.7 s of its 2.8 s compiling.
Repeated calls are common:
- **sequential updating:** Tutorial 5 conditions one model on new data as counts arrive;
- **simulation-based calibration:** `validation/_calibration.py:507–520` calls `condition_on` once per simulation in a Python loop, which multiplies the compile cost;
- **notebooks:** a user runs a fitting cell again.

`condition_on.check` also costs 0.75 s per warm call.
Most of that is `get_init_state`, which draws from the prior eagerly, and 2.2 fixes it.
The rest is `is_jax_traceable`, which traces the density with `make_jaxpr` on every call.

**Change.**
1. **Stable target densities.**
   `build_target_log_prob` returns `jax.tree_util.Partial(density, observed=...)`.
   Here `density` is a function `(params, observed) -> log density` held in the model's program store.
   The observed data and the fixed conditioning values are arguments, never closure constants.
   A `Partial` is a pytree whose static data is its function, so a jitted runner that takes one as an argument is reused whenever the model is the same object.
   `partial_cache.py` confirms the mechanism.
   A jitted chain runner given `Partial(stable_fn, data=d)` compiles 5 programs on the first call and none on the next two calls with new data.
   The same runner given a fresh closure compiles on every call.
2. **Module-level jitted chain runners, one per algorithm.**
   - **Static arguments:** `num_results`, `num_warmup`, `num_chains`, and the algorithm.
   - **Array arguments:** the step size, the chain keys, and the initial state.
   - **Contents:** the warmup scan and the sampling scan, plus the map over chains when `parallel_chain_map` takes its `vmap` branch.

   Its `pmap` branch calls a module-level `jax.pmap` of a function that takes the `Partial` as a broadcast argument, so it is cached in the same way.
3. **The same treatment for TFP NUTS, RWMH, ESS, and SGMCMC.**
   RWMH builds a new `step` closure for each adaptation window (`_blackjax_rwmh.py:268–280`), so each window compiles.
   Its runner takes the window's tuning values as arguments instead.
4. **A memoized traceability check.** `is_jax_traceable` stores its result in the model's program store, keyed by the structure of the observed data.

**Expected effect.**
A repeated `condition_on` on the same model compiles nothing.
On the measured configuration, a warm call would drop from 2.8 s to about 1.1 s.
The first call is unchanged except for fusion, which may shorten it.
This change was not prototyped end to end, so its pull request must measure it.

**Reproducibility.**
`parallel_chain_map` uses `pmap` when there are at least as many devices as chains, and `vmap` otherwise.
Its docstring states that the draws differ between the two branches, so the device count is part of the environment in J8.
That behavior exists today and does not change.

### 2.8 Shared helpers

Add `probpipe/core/_compilation.py`.
It is placed in `core` because every higher package uses it (`design/package-structure.md`).

1. **`ProgramCache`:** a small dictionary from a structural key to a jitted callable or a fallback marker.
   - **Ownership:** it belongs to an owner, and the owner's shallow copies share it.
   - **Lifetime:** it is dropped on pickle and released with the owner (J7).
   - **Placement:** it is a declared attribute of the owner, not a `transient_memo` entry, because `transient_memo` is dropped on every copy and the kernel copies in a factor graph must share their store.
2. **`compiled_or_eager(cache, key, build, *args)`:** on a cache miss, it builds the jitted callable and calls it.
   - **Trace errors:** if the call raises one, it records a fallback marker under the key and runs the eager form, and later calls skip the attempt.
     The trace errors are `ConcretizationTypeError`, `TracerBoolConversionError`, `TracerIntegerConversionError`, `TracerArrayConversionError`, and `UnexpectedTracerError`.
   - **Other exceptions:** they propagate unchanged.

   It generalizes `_compiled` in `inference/_pyabc.py:170–197`, which jits a function on its first call and falls back to eager execution if that fails.
3. **`count_compilations()`:** a context manager built on the `jax.monitoring` listeners.
   It yields the number of compilations, their seconds, and the number of traces.
   It is public in `probpipe.diagnostics`, because users need it (Part 3) and the tests use it (Part 4).
4. **The opt-in persistent cache** of 2.9.

### 2.9 The persistent compilation cache

JAX can write compiled programs to disk and read them in a later process.
`tut1.py` ran twice per mode, in two processes, with `JAX_COMPILATION_CACHE_DIR` set and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`:

| Call | `baseline`, process 1 | `baseline`, process 2 | `kernel`, process 1 | `kernel`, process 2 |
|---|---|---|---|---|
| `sample(model, 50)`, call 1 | 4.12 s | 0.78 s | 1.90 s | 0.39 s |
| `sample(model, 50)`, warm | 0.24 s | 0.26 s | 0.006 s | 0.006 s |
| `mean(model)`, call 1 | 4.23 s | 0.77 s | 1.83 s | 0.40 s |

Three facts follow:
- **A cold call in a later process is about 5 times faster**, on the current code and on the fixed code.
- **The cache softens today's recompilation without replacing the fix.**
  The program a TFP sampler compiles again is identical each time, so even in the first process the warm baseline call drops from 1.25 s to 0.24 s.
  The fixed code is still 40 times faster than that.
- **The cache is small.** The four runs wrote 377 entries, 3.5 MB in all.

**Change.**
1. Add `compilation_config.enable_persistent_cache(directory=None)` to `probpipe/core/config.py`, beside the existing configuration objects, and export it.
   With no directory, it uses a ProbPipe directory in the user's cache folder.
   It sets `jax_compilation_cache_dir` and a minimum compile time.
2. Read the environment variable `PROBPIPE_COMPILATION_CACHE`, as the other configuration objects read theirs.
   A `JAX_COMPILATION_CACHE_DIR` the user sets keeps priority.
3. Leave the cache disabled on import (J6).
4. Measure the minimum compile time before choosing its default.
   The runs above used 0, and JAX's default of 1 s would skip most of ProbPipe's programs, which compile in 0.1–1.7 s.
   The pull request compares 0 and 0.1 s by cache size and hit rate.
5. In CI, set the variable for the test and notebook jobs, and keep the directory between runs with `actions/cache`, keyed on `uv.lock`.

### 2.10 Field views draw only their ancestors

`sample(model["r"], 10)` draws the whole joint, including the 22 Poisson counts of the likelihood.
`_ancestral_sample` splits one key per factor index (`_factored.py:684–703`).
Drawing only the ancestors of a view's paths, with the same split, therefore gives the same values as drawing the whole joint and selecting.
After 2.4 a full draw of the Tutorial 1 model costs 3 ms warm, so the gain is small, except for a likelihood that is expensive to simulate.
The change belongs in `_capture_field_view` (`functions/_descendants.py:341`).
It changes how a field view is captured, so it needs a design decision and is optional.

### 2.11 Costs out of scope

These costs are not recompilation, or they are off the paths the tutorials take.
The plan lists them so that each pull request can state they are out of scope.

- **Per-draw keys under `vmap`.**
  A conditional draw maps the kernel over one key per draw.
  For the Poisson likelihood, 256 mapped draws with their own keys take 41 ms of XLA time, while one batched draw at the same 256 × 22 rates takes 3.8 ms.
  This is most of the 40 ms a warm `mean(model)` takes under the `kernel` mode.
  Drawing from one batched law would change which key produces which draw, so it changes seeded results and is a separate decision.
- **Python loops over draws.** Bootstrap draws loop in Python (`families/_resampling.py:448–471`), and a nested sweep runs one lift per row (`functions/_sweep.py:158–166`).
- **Host synchronizations:**
  - the provenance fingerprint copies arrays to the host (`core/_fingerprint.py:476–527`);
  - `bool(jnp.all(...))` checks block on the device (`families/_continuous.py:124`, `families/_multivariate.py:95`);
  - the PyMC sampler calls `int(jax.random.randint(...))` (`families/_programs.py:1240`).
- **Expectations over a finite support.** The Binomial and Categorical expectations apply `jax.vmap(f)` over the support (`families/_discrete.py:81–92`).

---

## Part 3. User-side decisions and their documentation

### 3.1 The questions a user faces

A user who writes a JAX function for ProbPipe faces these questions:
1. **Jitting:** should the function be wrapped in `jax.jit`?
2. **Traceability:** what may the function contain, so that ProbPipe can compile and vectorize it?
3. **Loops:** should a recursion over time be a Python loop or a `lax.scan`?
4. **Changing values:** where do data and settings that change between calls go?
5. **Reuse:** how often should a model be built, and may it be built inside a loop?
6. **The first call:** why is it slow, and can the cost be kept between sessions?
7. **Diagnosis:** how does one see what ProbPipe compiled, and why a call is slow?
8. **Code that is not JAX:** how should a simulator written in NumPy or SciPy be wrapped?

### 3.2 Measurements of user-side choices

`userstyle.py` builds the Tutorial 1 model with four versions of the Ricker likelihood and times `sample(model, 100)`.
All versions are built in one process, so the first row's call 1 also compiles the priors, which later rows reuse.
Each cell gives call 1 and the warm call.

| Likelihood written with | `baseline`, 22 years | `kernel`, 22 years | `kernel`, 200 years |
|---|---|---|---|
| `lax.scan` | 4.15 s / 1.05 s (c=5) | 1.48 s / 0.019 s (c=0) | 1.55 s / 0.061 s (c=0) |
| a Python `for` loop | 1.14 s / 1.02 s (c=4) | 0.80 s / 0.019 s (c=0) | **9.41 s** / 0.061 s (c=0) |
| `lax.scan` inside the user's `jax.jit` | 1.06 s / 1.01 s (c=4) | 0.63 s / 0.019 s (c=0) | 0.68 s / 0.060 s (c=0) |
| `lax.scan`, with the model built again for each call | 1.08 s / 1.11 s (c=5) | 0.68 s / **0.62 s** (c=1) | 0.67 s / **0.62 s** (c=1) |

`tut1.py` with `USERJIT=1` compares the warm `sample(model, 50)` with and without a user jit of the Ricker function:

| Ricker function | `baseline` | `sampler` | `kernel` |
|---|---|---|---|
| plain `lax.scan` | 1.25 s, c=5 | 0.068 s, c=0 | 0.006 s, c=0 |
| `jax.jit(ricker, static_argnames="years")` | 1.35 s, c=4 | 0.010 s, c=0 | 0.006 s, c=0 |

Four conclusions follow:
- **A user's jit matters only while ProbPipe does not compile the kernel.**
  JAX caches a jitted inner function by its identity even under an eager `vmap`, so a user's jit helps under the `sampler` mode.
  Under the `kernel` mode it changes nothing, and the design should not depend on it.
- **A Python loop unrolls into the program.**
  Its warm time equals the scan's, but at 200 years it takes 9.2 s to compile against 0.6 s for the scan, and the gap grows with the number of steps.
- **Building a model again for every call compiles one program per call.**
  It costs 0.62 s a call against 0.02 s for a reused model.
  In the baseline every row costs about 1 s a call, so this mistake becomes visible only once the fix is merged.
- **Warm compute scales with the work.** At 200 years every version takes 0.06 s warm.

### 3.3 The guidance

These are the answers the documentation gives, with the evidence for each.

1. **Do not jit for ProbPipe's sake.**
   ProbPipe compiles every function it calls on a hot path.
   A `jax.jit` on the function is harmless, and it helps when the user also calls the function directly.
   A jitted function declares its shape-determining arguments static, as `static_argnames="years"` does.
2. **Write the function in `jax.numpy` and `lax`, as a pure function.**
   - **No host conversions of an input:** no `float()`, `int()`, `bool()`, NumPy call, or Python `if` on an input's value.
     ProbPipe refuses such a kernel when it is built (2.4), and runs such a `@function` one draw at a time.
   - **No side effects:** a `print` runs only when ProbPipe traces, so use `jax.debug.print`.
     Do not write to globals.
   - **No randomness of the function's own:** a distribution draws through ProbPipe, and a function that needs randomness takes a key argument.
   - **No globals that change:** ProbPipe reads a global or a closed-over array when it first traces the function (J5).
     A value that changes between calls belongs in an argument.
3. **Use `lax.scan` or `lax.fori_loop` for long recursions.**
   A Python loop unrolls into the program, so its compile time grows with the number of steps.
   At 200 steps the loop compiled in 9.2 s and the scan in 0.6 s, with equal warm times (3.2).
4. **Keep shapes fixed.**
   Each new shape compiles a new program, whether it is the sample size, the number of years, or the length of the data.
   A session that calls `sample(model, n)` for 20 values of `n` compiles 20 programs, so use one sample size where possible.
   A shape-determining setting is a Python `int`, not an array.
5. **Build a model once and reuse it.**
   ProbPipe stores compiled programs with the kernel and the model.
   A kernel built inside a loop, for example by decorating a function in the loop's body, compiles again on each pass.
   That cost 0.62 s a call against 0.02 s for a reused model (3.2).
6. **Expect the first call to compile, and keep the result between sessions.**
   A first call compiles every program it needs, which takes 0.2–4 s for the tutorial models.
   `probpipe.compilation_config.enable_persistent_cache()` makes the first calls of later sessions about 5 times faster (2.9).
7. **Measure with `count_compilations()`.**
   - **A warm call that compiles anything** rebuilds a function or changes a shape.
   - **`operation.check(...)`** shows the route and the dispatch mode a call takes.
   - **`jax.disable_jit()`** runs everything eagerly, so ordinary debugging works; it also disables ProbPipe's compilation, so it is slower.
8. **Wrap code that is not JAX, and accept a cost per draw.**
   ProbPipe calls a NumPy or SciPy function once per draw.
   For speed, vectorize the function over a leading axis, or call it through `jax.pure_callback` with `vmap_method="expand_dims"`, so that it is called once per batch.
9. **Use float32 unless float64 is needed.**
   ProbPipe follows `jax_enable_x64`, and float64 roughly doubles the cost of a memory-bound program.
10. **Choose the number of devices for chains.**
    On CPU, `XLA_FLAGS=--xla_force_host_platform_device_count=N` runs N chains in parallel through `pmap`, and the draws then differ from a `vmap` run at the same seed (2.7).

### 3.4 The documentation

1. **A user-guide notebook, `docs/user_guide/performance.ipynb`, titled "Writing fast JAX functions for ProbPipe".**
   The `user_guide` CI leg executes it with the `dev` and `nutpie` extras only, so it uses no other backend.
   Its sections follow 3.3, and each runs a small experiment with `count_compilations()`:
   1. *What ProbPipe compiles, and when:* one cold and one warm call of `sample` on a small model, with their compilations.
   2. *Writing the function:* a pure Ricker function, then three failing variants and their messages:
      - a `float()` of an input;
      - a `print`;
      - a global that changes.
   3. *Loops:* a Python loop and a `lax.scan` at 22 and 200 steps.
   4. *Shapes and sample sizes:* the compilations of `sample(model, n)` over a few values of `n`.
   5. *Build once, call many times:* a model rebuilt in a loop, then built once.
   6. *Should I jit my function?* The warm time with and without a user jit, and when a jit still helps.
   7. *The first call:* the persistent cache, and what it stores.
   8. *Code that is not JAX:* a NumPy simulator, its sequential route in `.check()`, and its vectorized form through `jax.pure_callback`.
   9. *A checklist:* one line per rule of 3.3.

   The notebook follows the rules of `CONTRIBUTING.md` § Documentation: labeled output, public names, and no design citations.
   It carries the AI-generated label.
   Timings vary between machines, so it prints compilation counts as its evidence and times only as illustrations.
2. **Docstrings.**
   The Notes sections of `conditional_distribution`, `function`, `distribution`, `predictive_check`, and `expectation` state the purity and traceability contract of J5 and point to the guide.
   `count_compilations` and `compilation_config.enable_persistent_cache` get full docstrings, and `docs/api/diagnostics.md` and the configuration page list them.
3. **Tutorial 1.**
   One sentence after the likelihood cell says that the first `sample` compiles the model, and links to the guide.
   The tutorial's printed draws change in the last digit under 2.2, so its outputs are regenerated in the same pull request.
4. **`mkdocs.yml`:** the notebook is added under "User guide".
5. **`design/`:**
   - `design/05-functions.md` gains a "Compiled execution" paragraph beside V.8 and V.9 that states J1–J5 and J8 as the target contract;
   - C2 in `design/01-design-principles.md` gains one sentence saying that a callable passed into ProbPipe is pure in the same sense.

   The `design-check` skill covers both edits.

---

## Part 4. Order of work, tests, and decisions

### 4.1 Order of work

Each item is one pull request against `dev/overhaul`, and each depends on the ones before it.

1. **The compilation helpers and the counter** (2.8): `ProgramCache`, `compiled_or_eager`, and `count_compilations`, with unit tests.
2. **The family backend and the mixture** (2.2, 2.3).
   They ship together, because the backend change makes the mixture compile on every call.
3. **Conditional factors of a joint** (2.4), with the corrected message for a random operation inside a kernel.
4. **The persistent-cache option and its CI setup** (2.9).
5. **Function lifting and the dispatch probe** (2.6), and the laws of 2.5.
6. **Inference runners** (2.7), starting with BlackJAX NUTS, the default method.
7. **The documentation** (3.4).
   Its notebook can be drafted after item 3, since its experiments need only that.
8. **Optional:** field views (2.10) and the costs of 2.11.

### 4.2 Tests

Tests mirror the package tree, as `tests/<package>/test_<module>.py`.

1. **A warm call compiles nothing.**
   Each test calls a path twice, with new keys or new values of the same structure, and asserts `count_compilations().compiles == 0` on the second call.
   The paths are:
   - `sample`, `log_prob`, and `quantile` of each family whose sampler loops;
   - `sample` and `log_prob` of a joint with a kernel that uses `lax.scan`;
   - `sample` of a mixture;
   - a lifted `@function`;
   - NUTS `condition_on` on a second data set.

   Counts do not depend on load, so these tests are stable under `pytest-xdist` (J9).
2. **Eager and compiled execution agree (J3).**
   - **Families:** `_sample(key, shape)` called directly equals `jax.jit` of the same call, within a relative 1e-6.
   - **The mixture:** the two are equal.
   - **A joint:** the draws through the kernel program equal the draws through the eager fallback at one key.
3. **The fallback works (J4).**
   - A backend that is not a pytree samples eagerly.
   - A kernel with an unhashable fixed value runs eagerly.
   - A `@function` that calls NumPy runs one draw at a time.

   Each gives the same values as before the change.
4. **The program store behaves as owned (J7).**
   - A shallow copy of a kernel shares its programs.
   - A kernel built again compiles again.
   - A pickled and restored model works and compiles again.
   - Releasing a model releases its programs, which a weak reference to the store shows.
5. **The baselines follow J8.**
   Seeded LogNormal and Gamma draws change by a relative 5e-7, and the mixture's draws change completely.
   `tests/families`, `tests/distributions`, and `tests/operations` ran with 4 workers in both modes, with `pp_patch_plugin.py` applying the `kernel` mode:

   | | passed | failed | skipped | xfailed |
   |---|---|---|---|---|
   | `baseline` | 3991 | 0 | 45 | 31 |
   | `kernel` | 3990 | 1 | 45 | 31 |

   The failure is `tests/distributions/test_tfp_array_backend.py::TestBatchedOpsMatchTFPNative::test_sample_with_sample_shape_matches_native`.
   It compares a Normal backend's draws with TFP's eager draws at `assert_allclose`'s default `rtol=1e-7`.
   One of 30 elements differs by a relative 2.3e-7, which is the fusion effect of J8.
   The fix is a measured tolerance of `rtol=1e-6`, with a comment, as `STYLE_GUIDE.md` §8.6 requires.
   The skip counts are equal, so no failure is hidden by a skip, which is the check `AGENTS.md` § Verify asks for.
   The prototypes do not cover the mixture (2.3), lifting (2.6), or inference (2.7).
   Their pull requests need the same comparison over `tests/inference`, `tests/functions`, `tests/validation`, and the notebooks as well.

### 4.3 Decisions for the maintainers

1. **Extending C2 (J5).**
   Should every callable passed to ProbPipe be pure, so that ProbPipe may trace it once and reuse the trace?
2. **The mixture's draws (2.3).**
   Should the seeded draws of every mixture change, in exchange for one algorithm under eager and compiled execution?
3. **The program store (2.8).**
   Should it be an attribute that shallow copies share, as proposed, or should `transient_memo` gain a mode that copies share?
4. **The persistent cache (2.9).**
   Should it stay opt-in, as proposed (J6)?
   Should its minimum compile time default to 0 or to 0.1 s?
5. **Field views (2.10) and batched kernel draws (2.11).**
   Should either be pursued?
   Batched kernel draws change seeded results.

---

## Appendix. The benchmark scripts

The scripts in `review/jax-performance-bench/` run from the repository root as `uv run python review/jax-performance-bench/<script> ...`.
A script that takes `MODE` accepts `baseline`, `sampler`, or `kernel`.
Run one script at a time, because each compiles XLA code and several at once use a lot of memory.

| Script | What it measures |
|---|---|
| `compilecount.py` | the counter of compilations and traces that the other scripts import |
| `patches.py` | the `sampler` and `kernel` prototypes, as monkeypatches |
| `tut1.py MODE` | the Tutorial 1 calls of 2.1; `USERJIT=1` jits the Ricker function |
| `nbprofile.py NOTEBOOK [MODE]` | the time and compilations of each cell of a notebook |
| `hotpaths.py MODE [cheap\|nuts]` | quantiles, lifting, `log_prob`, `condition_on.check`, and NUTS |
| `mixture.py MODE` | a mixture's recompilation, and its eager against its jitted draws |
| `concrete.py MODE`, `concrete2.py MODE` | kernels that convert a given to a Python value, and a kernel that calls `sample` |
| `partial_cache.py` | the reuse of a jitted runner given a `Partial` |
| `userstyle.py MODE YEARS` | the user-side comparison of 3.2 |
| `pp_patch_plugin.py` | a pytest plugin that applies `patches.py` in the mode `PP_PATCH` names |
| `tfp_eager_vs_jit.py` | TFP alone, eager and jitted, for the table of the background (from #531) |
