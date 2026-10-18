# The call context: one mechanism for per-call state

> **Status: superseded on 2026-10-18** by [`execution-layer.md`](execution-layer.md) § 6.16, into which it was folded after the maintainer approved it. It is kept until this PR merges for its revision history (§ 13).
> It answered item D7 of [`gaps-and-conflicts.md`](gaps-and-conflicts.md) and follows the two decisions of 2026-10-17 in [`decisions.md`](decisions.md): one call context implemented once, and no name that uses "record" for anything but `Record`.
>
> **Revised twice on 2026-10-17**, after a first review and after an adversarial review (§ 13 lists the revisions). The first revision:
> 1. a call registers a token, and waits for its JAX programs, only when it runs a JAX program that stages an embedded call, so every other call keeps JAX's asynchronous dispatch;
> 2. the key of a JAX program ProbPipe opens includes only the settings that traced code may read, not every `config` block (OD6);
> 3. the carrier is named `_ContextBoundFunction`.
>
> The second revision fixes two blockers the adversarial review found: no token was passed to embedded calls inside the inference runners or past an inner `jax.jit`, and the detection of JAX programs with embedded calls was neither specified soundly nor built by any step before the workshop. It also moves the shipping of JAX settings into E3a, gives the executor choice one home, and adds OD7 and OD8.

- **For:** the maintainer, and through § 12 for jhuggins on PR #587.
- **Date:** 2026-10-17.
- **Code base:** `dev/overhaul` at `a703f15e` (this worktree), and PR #587 at `5ef1c6b2`, stacked on PR #586 at `a2f6e7fe`. JAX 0.10.1, Python 3.13.9, Prefect 3.7.4, on CPU.
- **Verification:** every claim marked ✅ was checked by running a script. § 10 lists the scripts, which are kept outside the repository (gaps W4).

---

## How to read this document

Section 1 states the design in ten lines.
Sections 2 to 4 give the background: the two designs as they stand, with their code, then the principles and the terms.
Section 5 is the contract, section 6 the names, and section 7 says where the code goes and what changes in #587 against what the execution steps add later.
Section 8 works three examples through, section 9 tries to break the design, and section 10 is the verification log.
Section 11 lays out the open decisions, and section 12 drafts the comment for #587.

Code labelled **dev/overhaul** runs on `dev/overhaul`, code labelled **#587** runs on PR #587, and code labelled **Proposed** is the target.
Each change from today is marked **Change**.
The steps E3a, E4, E7, and E8 are those of [`README.md`](README.md) § Steps.

---

## Contents

1. [Summary](#1-summary)
2. [The problem](#2-the-problem)
3. [Principles](#3-principles)
4. [Definitions](#4-definitions)
5. [The contract](#5-the-contract)
6. [Names](#6-names)
7. [Where the code goes, and what changes when](#7-where-the-code-goes-and-what-changes-when)
8. [Worked examples](#8-worked-examples)
9. [Trying to break the design](#9-trying-to-break-the-design)
10. [Verification log](#10-verification-log)
11. [Open decisions](#11-open-decisions)
12. [Draft comment for PR #587](#12-draft-comment-for-pr-587)
13. [Revisions](#13-revisions)

---

## 1. Summary

A ProbPipe call runs code in many places:
- the calling thread;
- the engine's worker threads and Prefect tasks;
- spawned processes;
- host callbacks inside compiled JAX programs.

That code must read the same per-call state wherever it runs, such as an active `config.using` block or the executor named on the call.
Today three context variables carry that state, and the execution plan adds a fourth structure keyed by a call token.
This proposal replaces them with one.

The design in eleven statements:
1. **One value.** One frozen `_CallContext` in one context variable carries every piece of per-call state. Each piece has one field as its home.
2. **Who opens one.** Every call of a `Function`, operations included, opens a child call context when it starts and closes it when it ends. A `config.using` block and a derived route's scope derive a changed copy of the current context, and open no call.
3. **What a child takes.** A child keeps its parent's `config` blocks and the controls an enclosing derived route passes. It resolves its own controls and starts its own `num_samples_used`.
4. **Precedence.** An explicit choice on the call wins, then a choice an enclosing call passes down, then the innermost `config.using` block, then the session's value, then the framework's default. `exact_only` accumulates.
5. **Evaluation points.** An evaluation point runs under its call's *evaluation context*, which drops the executor choice, so a call made inside an evaluation point uses automatic selection (Q8).
6. **Transport.** Within a thread the context variable carries the context. Across a thread, a Prefect task, or a process, the work function carries the evaluation context as plain data. Into a compiled JAX program, an integer call token carries it: the host side looks the context up in a thread-safe table of active calls.
7. **Tracing.** ProbPipe resolves settings in Python, when a call opens. Code that ProbPipe traces may read only the settings declared traced, which are part of the JAX program's cache key; reading any other setting while ProbPipe traces raises. A value that only steers the host side travels by token, and every other value stays in Python.
8. **Errors.** A call context closes in a `finally` clause: it releases its token if it registered one, merges its `num_samples_used` into its parent's only on success, and restores the context variable. The engine raises a callback's recorded failures in place of whatever exception JAX raises.
9. **Tokens only where embedded calls are.** A call registers a token only before it runs a JAX program that stages an embedded call, and only such a call waits for its JAX programs before it closes. Every other call registers nothing and keeps JAX's asynchronous dispatch.
10. **Names.** `_CallContext` in a new module `core/_call_context.py`. #587's `record` becomes `num_samples_used`, `_MonteCarloRecord` becomes `_NumSamplesUsed`, and `_CarriedContext` becomes `_ContextBoundFunction`. The token table is `_ActiveCalls`.
11. **Order.** #587 creates the one context with the fields it needs before it merges. E3a adds the executor choice, its reset where each evaluation point starts, and the shipped JAX settings; E4 the embedded-call state (the token and the failures); E7 the evaluation run table; and E8 the check of the JAX settings, each as a field or a rule of the same type.

**What #587 changes before it merges** (§ 7.2):
1. its three context variables become the fields of one value, which its carried wrapper carries;
2. the renames of § 6 apply;
3. a new rule replaces the sentence of V.2 on tracing.

**Open decisions** (§ 11): where `config` blocks are kept, where the provenance mode is kept, what a JAX program the user opens gets, whether process workers return their `num_samples_used`, whether #587 or a follow-up PR does the unification, what the cache key includes, whether embedded calls inside a compiled set of evaluation points use the call's executor, and what the program store holds (OD1–OD8).

---

## 2. The problem

### 2.1 Per-call state on `dev/overhaul`

A derived operation, such as `standard_deviation = sqrt(variance)`, passes some of its controls to the calls its formula makes (VI.0, "Controls inside a derived route").
PR #547 carries those controls into the engine's worker threads.

**dev/overhaul** (`values/_function_base.py:713-805`):

```python
_INHERITED_CONTROLS: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "probpipe_inherited_controls", default=None
)

class _CarriedControls:                     # a work function under the controls of its caller
    def __init__(self, func, inherited): ...
    def __call__(self, *args, **kwargs):
        token = _INHERITED_CONTROLS.set(MappingProxyType(self.inherited))
        try:
            return self.func(*args, **kwargs)
        finally:
            _INHERITED_CONTROLS.reset(token)

def _carrying_inherited_controls(func):     # used by execute_many before dispatching
    inherited = _INHERITED_CONTROLS.get()
    return func if inherited is None else _CarriedControls(func, inherited)
```

### 2.2 What #586 and #587 add

PR #586 records the Monte Carlo draws a call takes, so its provenance can report `num_samples`.
It adds a second context variable and puts the record into the carried wrapper.
PR #587 groups the settings under `probpipe.config`, adds temporary `using` blocks kept in a third context variable, and carries them too.

**#587** (`core/config.py`, `values/_function_base.py`):

```python
_TEMPORARY: ContextVar[Mapping[str, Mapping[str, Any]]] = ContextVar(...)   # the active config.using blocks
_INHERITED_CONTROLS: ContextVar[Mapping[str, Any] | None] = ContextVar(...)   # exact_only and the two counts
_MONTE_CARLO: ContextVar[_MonteCarloRecord | None] = ContextVar(...)          # the running call's draws

@dataclass
class _MonteCarloRecord:
    num_samples: int                  # the call's resolved num_samples, which a converter reads
    counts: set[int]                  # the draw count of each Monte Carlo estimate
    collects_calls: bool = False      # whether inner calls report their counts here

class _CarriedContext:
    def __init__(self, func, inherited, record, settings): ...
    def __call__(self, /, *args, **kwargs):
        token = _INHERITED_CONTROLS.set(inherited)
        record_token = _MONTE_CARLO.set(self.record)
        try:
            with _applying_settings(self.settings):      # installs the carried blocks
                return self.func(*args, **kwargs)
        finally:
            _MONTE_CARLO.reset(record_token)
            _INHERITED_CONTROLS.reset(token)
```

```python
# #587, as a user writes it
with probpipe.config.sampling.using(num_samples=4096):
    mean.using(method="monte_carlo")(y)     # provenance: num_samples=4096
```

#587's V.2 text states the precedence "a view's control from `f.using` wins over a block, which wins over the session", and adds: "A setting read while JAX traces a function is its value at trace time."

### 2.3 What the execution plan adds

The execution plan (§6.3, § "How the settings travel from a call to a callable") opens a **call context** when a call starts: the resolved execution settings, the list of failures, and a fresh call token.
It sets the context in a context variable, which the engine's worker threads inherit, and it enters the context in a table of active calls under the token.
A compiled JAX program takes the token as an argument, and a host callback looks the call up (§6.9):

```python
# execution plan §6.9, pseudocode
def host_side(call_token, theta, *, argument_ranks):
    call_context = active_calls.get(int(call_token))   # the call's executor, options, controls, and failures
    evaluation_points = flatten_to_evaluation_points(forward, theta, argument_ranks)
    evaluation_outcomes = select_executor(call_context).execute(evaluation_points)
    ...
```

The token is needed for two reasons, both confirmed again here:
1. Under `pmap`, a host callback runs on XLA's own threads, and a context variable set by the caller is not visible there (✅ `v02`).
2. A value read while tracing becomes a constant of the compiled JAX program, so a cached JAX program serves a later call with the first call's value (✅ `v03`: two calls with different settings both saw the first).

### 2.4 Where the two designs overlap or conflict

| # | Overlap or conflict | Evidence |
|---|---|---|
| 1 | Three context variables (`_TEMPORARY`, `_INHERITED_CONTROLS`, `_MONTE_CARLO`) and the execution plan's call context all carry per-call state, each with its own capture and install code. | #587 code; plan §6.3 |
| 2 | A callback on an XLA thread finds the execution plan's call context by token, but not #587's blocks or record, so it would read the session's values and note its draws nowhere. | ✅ `v02`; gaps D7 item 1 |
| 3 | #587 lets a setting read while tracing keep its trace-time value. A program store that reuses one compiled JAX program across calls then leaks one call's setting into the next. | ✅ `v03`; ✅ `v09` item 4 |
| 4 | The call's resolved `num_samples` has two homes: the `Function`'s options and `_MonteCarloRecord.num_samples`. | #586 code |
| 5 | The provenance mode has two homes: the workflow frame copies it when a `workflow_run` opens, and #587's block sets it. Inside an open `workflow_run`, a `config.provenance.using(mode=OFF)` block reads OFF but provenance is still recorded. | ✅ `v09` item 1 |
| 6 | The requested execution (`requested_dispatch`, `requested_workflow_kind`) is kept on the key broker, which owns randomness, and replay reads it from there. | `functions/_broker.py:1216-1246` |
| 7 | #587's text says a plain `contextvars.copy_context` would not carry the blocks into a Prefect task. Prefect's thread runner does copy the submitting context. The real reasons to carry explicitly are process runners, which need pickling, and the workflow frame, which a copy would also carry. | ✅ `v10`, ✅ `v12` |

---

## 3. Principles

These rules hold for every piece of per-call state.
They are numbered CC1 to CC8, so they do not clash with the design principles C1 to C5 or the execution principles P1 to P11.

**CC1. One home per piece of state.**
Each piece of per-call state is one field of one value, and no other variable holds a copy that code reads.
A copy made to cross a boundary, such as a pickled context, is a transport copy and is installed before anything reads it.

**CC2. One capture, one install.**
One function captures what crosses a boundary, and one function installs it on the other side.
Threads, Prefect, processes, and callbacks differ only in how the captured value travels.

**CC3. Per-call values are data at every boundary.**
This is P11 of the execution plan applied to settings: a per-call value enters a compiled JAX program, a callback, or a worker as an argument or as carried data, never as a captured constant or as thread-local state the receiver cannot see.

**CC4. Settings are read in Python, and traced code reads only settings that cannot change a computed value or are part of the key.**
A call resolves its controls once, when it opens, and the code the call runs reads the resolved values from its context.
Each `config` setting declares one of two kinds:
1. **traced:** it can change what a compiled JAX program computes, such as `sampling.num_samples`. While ProbPipe traces a JAX program it opens, code may read it, and it is part of the JAX program's cache key (OD6).
2. **presentation:** it changes only labels, messages, or provenance, such as `notation.max_depth` and `provenance.mode`. Code may read it at any time, since a value baked into a JAX program cannot change a computed result, and it never enters a cache key. Labelled terms built inside traced kernel and route code read `notation.max_depth` today (`TrackedTerm._store_expression`), and this keeps them working.

Reading a setting of neither kind while ProbPipe traces one of its JAX programs raises an error that names the setting, so a new setting must declare its kind.

**CC5. An explicit choice beats an ambient one.**
A value set on a call beats a value passed down by an enclosing call, which beats a `config.using` block, which beats the session's value.

**CC6. Inheritance stops at a scope's edge.**
A count a derived route passes applies to the calls its formula makes directly, and an executor choice applies up to the nearest evaluation point.

**CC7. A value never captures a call's settings.**
A term built inside a call keeps no reference to the call's context.
A `Function` keeps only its own explicit controls, so later work on a term runs under the settings of the later call.

**CC8. Closing always restores.**
A call context, a block, and an installed evaluation context each restore the previous context on exit, an error included, on every thread.

---

## 4. Definitions

- **Call:** one invocation of a `Function` through `__call__`, `apply`, or `check`. An operation is a `Function`, so an operation's invocation is a call.
- **Evaluation point:** one evaluation of a callable at one set of arguments (execution plan §5). A call that evaluates its body many times, as a lift or a sweep does, runs one evaluation point per evaluation.
- **JAX program:** a function that JAX traces and compiles, through `jax.jit` alone or with `vmap` or `pmap` inside it. Not every ProbPipe call produces one, so this document always writes "JAX program", never "program" alone.
- **Call context:** the `_CallContext` value that holds the per-call state of § 5.1. The **current call context** is the value of the context variable `_CALL_CONTEXT` in the running thread. Outside every call and block, it is the default context.
- **Default context:** the `_CallContext` with every field at its default: no blocks, no passed controls, no call, automatic selection.
- **`config` block:** the temporary values that `config.<section>.using(...)` or `config.using(...)` sets for the extent of a `with` statement (#587).
- **Passed controls:** the `exact_only`, `max_evaluations`, and `num_samples` that a derived route passes to the calls its formula makes (VI.0). Today they are the value of `_INHERITED_CONTROLS`.
- **Evaluation context:** the call context that an evaluation point of a call runs under, which `_CallContext.evaluation_context()` returns. It is the call's context without the executor choice, the embedded-call state, and the evaluation run table.
- **Call token:** a `numpy.uint32` that names one open call context in the table of active calls (execution plan §5). Token 0 names no call. A call registers a token only before it runs a JAX program that stages an embedded call.
- **JAX program with embedded calls:** a JAX program ProbPipe opens whose trace staged at least one embedded call. The program store finds this by searching the JAX program's jaxpr when it lowers the JAX program, before the JAX program first runs (§ 5.2).
- **ProbPipe is tracing:** `_TRACED_CALL_TOKEN` is set, which holds while ProbPipe traces a JAX program it opens.
- **A trace is active:** JAX is tracing anything, a JAX program ProbPipe opens or one the user opens. JAX exposes no public test for this, so one helper, `_jax_tracing_active()`, wraps the private check, and a test pins its behaviour.
- **Table of active calls:** `_ACTIVE_CALLS`, a thread-safe map from call token to call context.
- **JAX program ProbPipe opens:** a compiled JAX program that ProbPipe's code builds and runs and that runs user code or route code, such as the `jax` executor's lifted JAX program, a joint's compiled sampler, or an inference method's chain runner. Its first argument is the call token. A family's own sampler, which runs no user code, takes no token (OD6).
- **JAX program the user opens:** a compiled JAX program that the user's code builds, such as their own `jax.jit` around ProbPipe code. It takes no token.
- **Traced token:** while ProbPipe traces a JAX program it opens, the tracer that stands for that JAX program's token argument. An embedded call staged during the trace passes it to its callback.
- **`num_samples` used:** the number of draws each Monte Carlo estimate of a call actually took, held in `num_samples_used`; the call's provenance reports it as `num_samples` (#586). It differs from the `num_samples` control, which says how many draws an estimate should take: a call that ran no Monte Carlo estimate reports none, and the calls a derived route's formula makes may use their own counts. In `method_options`, an MCMC method's draw counts are a different thing.

---

## 5. The contract

### 5.1 What the call context holds

`_CallContext` is a frozen dataclass.
Its mutable parts are separate objects that it references, each with its own lock, so that copying a context never copies an accumulator by accident.

**Proposed** (`core/_call_context.py`):

```python
@dataclass(frozen=True)
class _CallContext:
    """The per-call state that every piece of code a call runs reads, wherever it runs.

    One value of this class is current in each thread, held by the context variable
    `_CALL_CONTEXT`; `current_call_context()` returns it. Outside every call and
    `config` block it is the default context, with every field at its default.

    Every piece of per-call state has exactly one field here as its home, and no
    other variable keeps a copy that code reads.

    Lifecycle:

    - Every call of a `Function` (`__call__`, `apply`, `check`) opens a child with
      `opening_call` when it starts, and the previous context is restored when it
      ends, an error included.
    - A `config` block derives a changed copy with `with_config_block`, and an
      operation's derived route's scope with `with_passed_controls`; neither opens a call.
    - An evaluation point runs under `evaluation_context()`, the call's context
      without its executor choice, its embedded-call state, and its run table, so a
      ProbPipe call made inside the evaluation point uses automatic selection.

    Transport:

    - Within a thread, the context variable carries it.
    - Into worker threads, Prefect tasks, and spawned processes, the work carries
      `evaluation_context()` as data and installs it before running. The value
      pickles; its read-only mappings travel as plain dicts.
    - Into a compiled JAX program, the call token in `embedded_calls` carries it: a
      host callback looks the context up in `_ACTIVE_CALLS`.

    Reading settings while JAX traces: code may read a `config` setting declared
    traced, which is part of the JAX program's cache key, or one declared
    presentation, which cannot change a computed value. Reading any other setting
    while ProbPipe traces one of its JAX programs raises.

    The dataclass is frozen. Its mutable parts (`num_samples_used`, `evaluation_log`,
    `embedded_calls`, `evaluation_run_table`) are separate objects that it references, each
    with its own lock, so copying a context never copies an accumulator.

    Attributes
    ----------
    config_blocks : Mapping[str, Mapping[str, Any]]
        The active `config.using` blocks, by section and then setting, the
        innermost value winning.
    passed_controls : Mapping[str, Any] or None
        The controls an enclosing derived operation's identity passes to the calls
        it makes directly: `exact_only`, `max_evaluations`, and `num_samples`. A
        child keeps only `exact_only`.
    label : str or None
        The running call's `Function` label, for messages.
    controls : Mapping[str, Any]
        The running call's resolved controls, without `executor` and
        `executor_options`, whose home is `execution_choice`.
    num_samples_used : _NumSamplesUsed or None
        The numbers of Monte Carlo draws the call's routes take, which its
        provenance reports as `num_samples`. Merged into the parent's on success
        when the parent collects calls.
    execution_choice : _ExecutionChoice
        The executor, its options, and whether the choice was named on this call,
        inherited, or automatic. The only place code reads the executor choice.
    evaluation_log : _EvaluationLog or None
        What the call ran: each set of evaluation points, its executor, and its
        failures, for provenance and replay. Merged into the parent's on close.
    jax_settings : Mapping[str, Any]
        The four JAX settings shipped to worker processes (`jax_enable_x64`,
        `jax_default_prng_impl`, `jax_threefry_partitionable`,
        `jax_default_matmul_precision`), as they were when the root call opened.
    embedded_calls : _EmbeddedCallState or None
        The call token, registered only before the call first runs a JAX program
        with embedded calls; the failures that host callbacks record; and the
        outputs the call waits for before it releases its token. `None` for a call
        opened while JAX traces.
    evaluation_run_table : _EvaluationRunTable or None
        The evaluation points the call has run, so an opaque `Function` runs at
        most once per distinct evaluation point within the call.
    """

    # --- from #587 ------------------------------------------------------------------------
    config_blocks: Mapping[str, Mapping[str, Any]] = _NO_BLOCKS
    passed_controls: Mapping[str, Any] | None = None
    label: str | None = None
    controls: Mapping[str, Any] = _NO_CONTROLS
    num_samples_used: _NumSamplesUsed | None = None
    # --- added by E3a ---------------------------------------------------------------------
    execution_choice: _ExecutionChoice = AUTOMATIC_SELECTION
    evaluation_log: _EvaluationLog | None = None
    jax_settings: Mapping[str, Any] = _NO_JAX_SETTINGS
    # --- added by E4 ----------------------------------------------------------------------
    embedded_calls: _EmbeddedCallState | None = None
    # --- added by E7 ----------------------------------------------------------------------
    evaluation_run_table: _EvaluationRunTable | None = None

    def with_config_block(self, config_values: Mapping[str, Mapping[str, Any]]) -> _CallContext: ...
    def with_passed_controls(self, passed_controls: Mapping[str, Any] | None) -> _CallContext: ...
    def opening_call(self, *, label: str, controls: Mapping[str, Any], execution_choice: _ExecutionChoice) -> _CallContext: ...
    def evaluation_context(self) -> _CallContext: ...
    def is_default(self) -> bool: ...
```

Field by field:

| Field | Type | Written by | Read by | Step | Today's home | Change |
|---|---|---|---|---|---|---|
| `config_blocks` | mapping of section to setting to value | `config.<section>.using` and `config.using`, through `with_config_block` | `_Section._value_in_effect` in `core/config.py` | #587 | `_TEMPORARY` (#587) | **Change:** the variable goes; its value is a field |
| `passed_controls` | mapping or `None` | `_passing_down` in a derived route's scope; `opening_call`, which clears the counts and keeps `exact_only` (today's `_EXACT_ONLY_BELOW`) | `opening_call` through `Function._inheriting` | #587 | `_INHERITED_CONTROLS` (#547, #586) | **Change:** the variable goes; its value is a field |
| `label` | `str` or `None` | `opening_call` | messages and warnings | #587 | none | new |
| `controls` | mapping | `opening_call`: the view's `options` after `_inheriting`, without `executor` and `executor_options`, whose one home is `execution_choice` | `_num_samples_in_effect`, executors (E3a), requesters that read `method_options["on_failure"]` | #587 | `_MonteCarloRecord.num_samples` and `Function.options` | **Change:** `_MonteCarloRecord.num_samples` goes |
| `num_samples_used` | `_NumSamplesUsed` | routes, through `_note_num_samples_used`; `opening_call` and `_collecting_num_samples_used` create one | the call's provenance; the parent's counts on success, when the parent collects calls | #587 | `_MONTE_CARLO` (#586) | **Change:** renamed; the variable goes |
| `execution_choice` | `_ExecutionChoice(executor, executor_options, source)`, the one home of the executor choice | `opening_call`, from the call's own `executor` controls, else the parent's choice; `evaluation_context` resets it | the executor dispatcher; the host side through the token | E3a | the controls `dispatch` and `max_workers` | new |
| `evaluation_log` | `_EvaluationLog` | the executor dispatcher and the host side; a closing child merges its log into its parent's, on success and on error | provenance; replay's `executor` | E3a | the key broker's `requested_dispatch` | **Change:** moves off the broker |
| `embedded_calls` | `_EmbeddedCallState`: the call token (`None` until registered), the failure list, and the outputs of the JAX programs with embedded calls the call ran | `opening_call` creates it, empty, outside a trace; the engine registers the token before it first runs a JAX program with embedded calls; the host side writes failures | JAX programs with embedded calls; `_ACTIVE_CALLS`; the engine at close | E4 | none | new |
| `evaluation_run_table` | `_EvaluationRunTable` | the host side and the dispatcher | the same, before running an evaluation point | E7 | none | new |
| `jax_settings` | mapping of the shipped settings: `jax_enable_x64`, `jax_default_prng_impl`, `jax_threefry_partitionable`, `jax_default_matmul_precision` | `opening_call` at a root call; inherited by children | the worker initializer of the `process` executor (E3a); the check at close and on the host side, and provenance (E8) | E3a | none | new |

Three pieces of state that the task listed have their home elsewhere, so they are not fields:
1. **The failure rule** is a requester's `method_options={"on_failure": "drop"}`, so it is in `controls`. An embedded call always raises (Q7).
2. **The traced token** exists only while a JAX program is being traced and never crosses a boundary, so it has its own context variable, `_TRACED_CALL_TOKEN`, in `functions/_boundary.py` (E4).
3. **The workflow frame and the key broker** carry randomness, which § 5.7 keeps out of the call context.

`_ExecutionChoice.source` is `"call"` when the executor was named on this call or on the `Function` itself, `"inherited"` when it came from an enclosing call, and `"automatic"` when nobody named one.
The execution plan's operation rule 6 needs it: a refusing executor named on the call raises, and an inherited one falls back with a warning.

### 5.2 Lifecycle

**Opening.** Every call of a `Function` opens a child call context before anything else runs.
The decorator that wraps `check`, `apply`, and `__call__` today, `_inheriting_controls`, becomes the one place that opens it.

**Proposed** (`values/_function_base.py`):

```python
def _opening_call_context(entry_method):
    """*entry_method* of a Function (`check`, `apply`, or `__call__`), run in a call context
    of its own, opened from the current one."""
    @wraps(entry_method)
    def run_in_child_call_context(self, /, *args, **kwargs):
        parent_context = current_call_context()
        function_view = (
            self if parent_context.passed_controls is None
            else self._inheriting(parent_context.passed_controls)
        )
        child_context = parent_context.opening_call(
            label=function_view.label,
            controls=function_view.options,     # explicit, then passed, then block, then session
            execution_choice=_resolve_execution_choice(function_view, parent_context),   # E3a; see § 5.3
        )
        with _call_context_open(child_context, parent_context):   # releases any token (E4) and restores on exit
            return entry_method(function_view, *args, **kwargs)
    return run_in_child_call_context
```

`opening_call` builds the child:
1. **`config_blocks`:** the parent's mapping, the same object.
2. **`passed_controls`:** the parent's passed `exact_only` alone, or `None`, as `_EXACT_ONLY_BELOW` does today. The passed counts therefore apply only to the calls a derived route's formula makes directly.
3. **`controls`, `label`, `execution_choice`:** as resolved above.
4. **`num_samples_used`:** a new, empty `_NumSamplesUsed`.
5. **`embedded_calls` and `evaluation_run_table`:** new and empty, when the call opens while no trace is active (E4, E7). Creating them registers nothing. A call opened while a trace is active gets neither, because its JAX programs become part of the traced JAX program and run later under that JAX program's token.
6. **`jax_settings`:** the parent's snapshot, or a new snapshot of the four shipped settings when the parent has no call (E3a).

Two methods differ from `__call__`:
- **`check`** opens a context without `num_samples_used` and without embedded-call state, since it draws nothing and runs no JAX program.
- **`apply`** shares its parent's `num_samples_used` instead of starting its own, so the draws an `apply` makes count toward the enclosing call, as they do today, where only `__call__` starts its own `num_samples_used` (`functions/_function.py:313`).

**Registering a token.** A token is registered only for a JAX program with embedded calls, and only when no trace is active:
1. **Detection.** The program store lowers a JAX program before its first run (execution plan §6.10, item 3, which step E3a builds). It traces the JAX program with `jax.jit(f).trace(...)`, searches the resulting jaxpr, sub-jaxprs included, and then lowers it. (A `Lowered` object carries no jaxpr.) The search looks for ProbPipe's own embedded-call callback, matched by the host-side function object, so `jax.debug.print` and a user's own callbacks do not count. A Python flag set while staging is not enough: a nested `jax.jit` that is already cached does not run its Python again when a second outer JAX program traces it (review test R1). The search found the callback through a nested `jit`, `custom_vjp` (its primal and under `grad`), `custom_jvp`, `checkpoint`, and `scan` (review test R1).
2. **Validity.** The search also checks that the token operand of every embedded call is the JAX program's own token argument. A literal token, or a tracer from another trace, means a `jax.jit` between the JAX program and the embedded call captured the token, such as a user's jitted helper around an opaque `Function`, which otherwise sends the wrong call's settings or raises `UnexpectedTracerError` (review test R2). The program store raises an error that names the jitted helper and says to remove the inner `jit` or to call the opaque code through a ProbPipe call.
3. **Registration.** Before the engine runs a JAX program with embedded calls, it registers the call's context in `_ACTIVE_CALLS`, if the call holds no token yet, and passes the token. Every other JAX program receives token 0, which it never reads.
4. **Inside a trace.** When a trace is active, the engine registers nothing and waits for nothing. It passes `_TRACED_CALL_TOKEN` when ProbPipe is tracing, and 0 otherwise. Without this rule a JAX program the engine runs inside an outer trace is staged into the outer JAX program with its token as a literal, and the cached outer JAX program later sends a released token to the host (review test R3).

A JAX program that ProbPipe opens while tracing another is part of the outer JAX program, so its embedded calls make the outer JAX program one with embedded calls.

**Deriving without a call.** Two scopes change the current context without opening a call:
- **A `config` block** derives `current_context.with_config_block(config_values)`: the enclosing blocks merged with the new values, an inner value winning.
- **A derived route's scope** derives `current_context.with_passed_controls(passed_controls)` for the extent of the derived route's check and run.

Each installs the derived context with `_under_call_context`, which resets the variable on exit.

**Evaluation points.** An evaluation point opens no call.
Wherever it runs, it runs under its call's `call_context.evaluation_context()`, installed by `_under_call_context`.
A ProbPipe call the evaluation point makes then opens a child of the evaluation context.

**Closing.** `_call_context_open` closes the context in this order:
1. **Wait for JAX programs with embedded calls.** When the call holds a token, the engine blocks on the outputs of every JAX program with embedded calls it ran (`jax.block_until_ready`) before the call returns. JAX dispatches a JAX program asynchronously, so without the wait a callback could look up a released token. On CPU a JAX program with a callback returned only after the callback ran, and a JAX program without one returned before its result was ready (✅ `v04`). The design relies on neither. A call that ran no JAX program with embedded calls holds no token and waits for nothing, so its JAX programs keep running asynchronously after it returns.
2. **Release the token** (E4), if the call registered one, in a `finally` clause.
3. **Merge `num_samples_used`** into the parent's on success, when the parent collects calls, as #586 does.
4. **Merge what ran:** the call's `evaluation_log` merges into its parent's, on success and on error (§ 5.1).
5. **Check the JAX settings** at a root call (E8, Q18). A change raises on success. On error, a note naming the change is added to the original exception, so the original error is not hidden.
6. **Restore the context variable**, in the same `finally` clause.

**On error.** An exception raised during the call propagates after steps 2, 4, and 6.
`num_samples_used` is not merged.
When the failures in the call's `embedded_calls` hold entries recorded by a callback, the engine raises `EvaluationError` with them, chained to the exception JAX raised.
It decides by the recorded failures, not by the exception's type: a failing callback raised `JaxRuntimeError` in one JAX program and as `ValueError("INTERNAL: CpuCallback error ...")` in another (✅ `v07`, `v07b`).
The execution plan's §6.4 rule 2 now states this.

### 5.3 Precedence

For each control of a call, `opening_call` takes the first of these sources that gives a value:
1. **The call's own explicit setting:** the `Function`'s `using` view, which overrides its decorator or constructor.
2. **A value an enclosing call passes down:**
   - the counts that an enclosing derived route passes to the calls its formula makes directly (VI.0);
   - the executor choice of the enclosing call, for an opaque callable, up to the nearest evaluation point (execution plan §6.3, operation rules 1 to 5).
3. **The innermost `config.using` block** that sets the control's default.
4. **The session's value** in `config`.
5. **The framework's default**, which is the session's initial value.

`exact_only` is the one exception.
It restricts route selection, so it is true when the call sets it or an enclosing derived route passes it, whatever the call's own view says.

**Embedded calls inside a compiled set of evaluation points.** Q8 applies to a ProbPipe call that an evaluation point makes in Python, which opens a child of the evaluation context and so uses automatic selection. An embedded call inside a JAX program the `jax` executor compiles is not such a call: it is part of the call's own JAX program, and its host side finds the call's context by token, executor choice included. So `growth.using(executor="slurm")(prior)`, whose traceable body calls an opaque `forward`, sends `forward`'s evaluation points to `slurm` (OD7).

This one rule contains each of the three rules it reconciles:

| Source | Its rule | How it follows from the order above |
|---|---|---|
| #587 | "a view's control from `f.using` wins over a block, which wins over the session" | sources 1, 3, and 4 |
| Execution plan, operation rule 4 | "the innermost explicit choice wins": an executor set on the `Function` wins over an inherited one | source 1 before source 2 |
| Execution plan, Q8 | "a call inside an evaluation point uses automatic selection" | source 2 stops at an evaluation point, because `evaluation_context()` drops the executor choice, and no `config` section holds an executor default, so source 5 applies |
| VI.0 | an inherited count applies "unless that call's view sets its own" | source 1 before source 2 |

One case needs a sentence of its own.
A count passed by a derived route beats a `config` block even when the block is opened inside the derived route's scope, because the passed count is an explicit choice on the derived call.
In practice this case does not arise. The calls that take a passed count are the ones the formula makes directly, and the formula is ProbPipe's code, so no user's block opens between the derived route and those calls.

A setting that is not a control, such as `config.notation.max_depth`, resolves from sources 3 to 5 at the moment it is read.

### 5.4 Transport

Each boundary has one mechanism, and every mechanism installs with the same function, `_under_call_context`.

| Boundary | Mechanism | What crosses | Evidence |
|---|---|---|---|
| A nested call in the same thread | the context variable; the child derives from the current context | everything; the child computes its own controls, `num_samples_used`, and token | ✅ `v01` |
| The engine's worker threads (today `dispatch="thread"`; E3a's `thread` executor) | the work function carries `current_context.evaluation_context()`: `_ContextBoundFunction(work_function, call_context)` today, the `EvaluationChunk` from E3a | the evaluation context; `num_samples_used` by reference | ✅ `v01`, `v09` item 2, `v12` |
| Prefect tasks and flows (`workflow_kind` `TASK` or `FLOW`) | the same wrapper, inside Prefect's payload | the evaluation context; `num_samples_used` is copied under a process runner | ✅ `v10` |
| Spawned worker processes (E3a's `process` executor) | the pool's worker initializer applies `jax_settings` before the worker unpickles any chunk; the `EvaluationChunk` pickles the evaluation context | the evaluation context; `num_samples_used` comes back with the outcomes (OD4) | ✅ `v05` |
| Host callbacks in JAX programs ProbPipe opens, under `jit`, `vmap`, and `pmap` | the call token, an argument of the JAX program; the host side calls `_ACTIVE_CALLS.lookup(call_token)` | the whole context to the host side, which runs each evaluation point under its `evaluation_context()` | ✅ `v02`, `v03`, `v06`, `v13` |
| A JAX program ProbPipe opens while tracing another | the enclosing traced token, passed as the inner JAX program's token argument | as above | ✅ `v06` |
| A JAX program the user opens with their own `jax.jit` | no token: an embedded call stages token 0, and the host side runs under the default context | nothing | ✅ `v06`; OD3 |

Four facts decide these mechanisms:
1. **A worker thread sees nothing unless something carries it.** A thread of `ThreadPoolExecutor` starts with an empty `contextvars` context, so the call context reads as the default context. `copy_context().run` gives it a copy, but one copied `Context` cannot be entered by two threads at once (✅ `v01`).
2. **Copying the whole context is wrong.** A copy also carries the workflow frame, which belongs to the thread that opened it, so a draw in the worker raises `UnmanagedConcurrentWorkflowEntryError` (✅ `v12`). Carrying only the call context is what CC1 asks for anyway.
3. **The context must pickle.** A `MappingProxyType` does not pickle (✅ `v05`), so the context pickles its mappings as plain dicts and restores the read-only views when it is unpickled. A spawned worker starts with JAX's default settings, so it applies the carried ones (✅ `v05`: float32 and x64 off without them, float64 with them).
4. **The token arrives with extra axes.** Under `vmap` with `vmap_method="expand_dims"`, an unbatched token arrives at the host with shape `(1,)`, and under two `vmap`s with shape `(1, 1)` (✅ `v14`). The host reads the token as the one value of that array. Under `pmap` the host runs on several XLA threads at once, so `_ACTIVE_CALLS` takes a lock for each lookup, and never holds it while user code runs (✅ `v06`).

**The token table.**

```python
class _ActiveCalls:
    """The open call contexts that compiled programs may call back into, by call token."""
    def register(self, call_context: _CallContext) -> int: ...   # a token no open context holds, never 0;
                                                                  # called before a program with embedded calls first runs
    def lookup(self, call_token: int) -> _CallContext: ...        # 0 gives the default context; an unknown token raises
    def release(self, call_token: int) -> None: ...

_ACTIVE_CALLS = _ActiveCalls()
```

A token is a `numpy.uint32`, because a Python `int` of `2**31` or more overflows a jitted argument when 64-bit types are off (✅ `v03`).
Tokens count up and wrap, skipping 0 and every open token.
A token that is not open raises an error that names it, which can happen only if a JAX program outlived its call, and the wait of § 5.2 rules that out.

### 5.5 Tracing

#587's sentence, "A setting read while JAX traces a function is its value at trace time", is replaced by this rule.

**The rule.**
Every setting a call uses falls into one of three classes:
- **(a) Python only.** The value steers ProbPipe's orchestration and no compiled JAX program computes with it. The call reads it from its context, in Python.
- **(b) Static for a compiled JAX program.** The value changes what a JAX program ProbPipe opens computes. It enters the JAX program as a static argument, so it is part of the program-store key (execution plan §6.10).
- **(c) Run-time data by token.** The value only steers what the host side does when a JAX program calls back. The host side reads it from the context it finds by token, and the JAX program never contains it.

A per-call value that the device computation itself uses, such as a random key or a parameter, is an ordinary array argument of the JAX program (P11), so it is none of the three.

**The test an implementer applies to a new setting.**
Ask whether two calls that differ only in this setting may share one compiled JAX program.
- If they may, and no compiled code reads the value, the setting is class (a).
- If they may, and the host side reads the value, it is class (c): add it to the call context and read it there.
- If they may not, it is class (b): pass it as a static argument of the JAX program, and never read it from the context inside the traced function.

**Every setting today, classified.**

| Setting | Where it is read | Class |
|---|---|---|
| `config.sampling.max_evaluations` and `num_samples`, and the controls of those names | planning a lift; the size of a Monte Carlo estimate | (a), and (b) for a JAX program whose shapes they set, where they are part of its input structure |
| `config.notation.max_depth` | rendering a label | (a); declared presentation, so readable while tracing (CC4) |
| `config.provenance.mode` | building provenance | (a); declared presentation (CC4) |
| `config.prefect.workflow_kind` and `task_runner` | choosing Prefect's transport | (a) |
| `exact_only`, `method`, `method_options`, `conversions`, `include_inputs`, `raw`, the `workflow_kind` control | route selection and planning | (a) |
| the failure rule, `method_options["on_failure"]` | a requester in Python | (a) |
| `executor` and `executor_options` for a set of evaluation points built in Python | the dispatcher | (a) |
| the executor and options an embedded call inherits from the call that runs the JAX program | the host side | (c) |
| an executor set on the `Function` itself, or named on a call that opens while a trace is active | the host side, from the `Function` that the staged callback holds | (c): an executor's options can hold objects with no hash, such as a pool, so they never enter a JAX program key |
| the `jax` executor's `batch_size` | compiling a lifted JAX program | (b) |
| `memo` | the staging of an embedded call | (b), as the execution plan §6.10 item 2 states |
| the failures in `embedded_calls` and the evaluation run table | the host side | (c) |
| the traced settings: `passed_controls`, and the `config` settings declared traced (today `sampling.max_evaluations` and `sampling.num_samples`), as ProbPipe code traced inside a JAX program ProbPipe opens reads them | the trace | (b): part of the cache key of every JAX program ProbPipe opens (OD6); reading any other setting while ProbPipe traces raises (CC4) |
| JAX's global settings: `jax_enable_x64`, `jax_default_prng_impl`, `jax_threefry_partitionable`, `jax_default_matmul_precision` | tracing and XLA | (b), through `jax.jit`'s own cache (OD8), and checked on the host side as (c) |

Three consequences follow.
1. **A call that opens while tracing.** A `Function` called inside a kernel that a joint compiles opens its context during the trace. Its explicit executor choice travels with the `Function` that the staged callback holds, and the host side reads it from there. It inherits no executor from its parent: its embedded calls pass the traced token, so the call that runs the JAX program decides at run time.
2. **JAX's settings and the program store.** JAX's own `jit` cache traces again after `jax_enable_x64`, `jax_threefry_partitionable`, or the default matmul precision changes (✅ `v08`). A program store that keeps jitted functions inherits that keying. A store that keeps compiled executables, as `jax.jit(f).lower(...).compile()` returns, does not: a stored executable ran with float32 after 64-bit types were turned on (✅ `v08`). **Change** to the execution plan §6.10: the program store holds `jax.jit` objects, which inherit JAX's keying, rather than compiled executables (OD8). Lowering and then calling a `jax.jit` object compiles once (review test R1).
3. **A JAX program the user opens.** ProbPipe code inside the user's own `jax.jit` runs its Python while JAX traces it, under the context of the thread that traces it. Its decisions, such as a route chosen under `exact_only`, become part of the user's JAX program, which JAX caches by input shapes (✅ `v09` item 4: a `config` value read inside a user's `jit` kept its first value inside a later block). ProbPipe states this in V.2 and makes no per-call promise for such a JAX program.

### 5.6 Provenance and replay

Provenance reports each setting once, at the call whose work used it, under the setting's own name:
1. **Route metadata.** A Monte Carlo route reports `num_samples` from `num_samples_used.provenance_num_samples`, and a lift reports `max_evaluations`, as #586 does. A derived operation's call reports the draws of its formula's calls, because its counts collect theirs.
2. **Execution** (E3a, E4, E7). A call reports, from `evaluation_log`, each set of evaluation points it ran: the executor, its options, the number of evaluation points and of runs, and the failed or dropped evaluation points. An embedded call's sets of evaluation points are reported by the call that owns the token, never by a call that opened while tracing.
3. **No block as such.** A `config` block is never recorded. Its effect is recorded through the resolved values that change a result, which items 1 and 2 already report. A setting that changes no result, such as `config.notation.max_depth`, is not recorded.
4. **JAX's settings** (E8). A call that records a replay recipe records the root's `jax_settings` snapshot with the JAX version and the device count. It reads the snapshot from its context and never reads `jax.config` again, so the recorded values are the ones the call ran under.
5. **Replay.** The recipe's `requested_dispatch` becomes `executor` (execution plan §6.8), read from the context's `execution_choice`. **Change:** the key broker's `set_requested_execution` and its `requested_dispatch` and `requested_workflow_kind` fields go, so the broker keeps only randomness.

### 5.7 State that stays outside the call context

| State | Home | Why it stays |
|---|---|---|
| The workflow frame (`_ACTIVE_WORKFLOW_FRAME`) | `functions/_context.py` | Randomness (V.8). An evaluation point gets a new child frame, not a copy, and its admission rules differ from the call context's. The code that runs an evaluation point enters both. |
| The key broker and the managed attempt | `functions/_evaluation_keys.py` (decided Q16) | Randomness, with its own coordination protocol across Prefect. |
| The probe guard and the JAX runtime guard (`_STOCHASTIC_PROBE_STATE`, `_JAX_RUNTIME_GUARD`) | `functions/_context.py` | They mark a trace in progress and never cross a boundary. |
| Whether a `check` is probing (`_call._CHECKING`) | `functions/_call.py` | Per call, so it could become a field later. Neither design carries it, and it never crosses a boundary. |
| The traced token | `functions/_boundary.py` (E4) | Trace-time only. |
| Replay state (`_ACTIVE_REPLAY_STATE`, `_REPLAY_FUNCTION_DEPTH`, `_REMOTE_REPLAY_CLAIMS`) | `functions/_replay.py` | Replay of randomness, which the key broker's coordination protocol carries across boundaries, like the frame. |
| Whether a family keeps its backend's batch axes as separate laws (`_BATCHED_INIT_BYPASS`) | `families/_backend.py` | A construction-local flag for the extent of one `with` block inside a family's code; no call's code reads it across a boundary. |

The code that runs an evaluation point, today `_execute_work_item` and from E3a `EvaluationChunk.run`, is the one place that enters both the point's workflow scope and its evaluation context, so an evaluation point needs one entry, not two.

---

## 6. Names

Each name says what the thing is, and none uses "record" for anything but `Record`.

| #586 and #587 | Proposed | Reason |
|---|---|---|
| (three context variables) | `_CallContext`, held in `_CALL_CONTEXT` | "call context" is the execution plan's term (§5) and is free in the glossary |
| `_CarriedContext` | `_ContextBoundFunction(work_function, call_context)` | a function bound to the call context it runs under |
| `_CarriedContext.record` | `_CallContext.num_samples_used` | it holds the `num_samples` each Monte Carlo estimate used |
| `_CarriedContext.inherited` | `_CallContext.passed_controls` | VI.0 says a derived route *passes* controls to the calls its formula makes |
| `_CarriedContext.settings` | `_CallContext.config_blocks` | it holds the active `config.using` blocks, not every setting |
| `_MonteCarloRecord` | `_NumSamplesUsed` | |
| `_MonteCarloRecord.num_samples` | removed; `_CallContext.controls["num_samples"]` | a second home of the resolved control |
| `_MonteCarloRecord.recorded` | `_NumSamplesUsed.provenance_num_samples` | the `num_samples` value the call's provenance reports |
| `_MonteCarloRecord.collects_calls` | unchanged | it matches the route attribute of the same name |
| `_recording_monte_carlo` | `_collecting_num_samples_used` | the sub-scope that `_ByComponent` opens for each block, and that a call opens |
| `_note_monte_carlo` | `_note_num_samples_used` | |
| `_MONTE_CARLO`, `_INHERITED_CONTROLS`, `_TEMPORARY` | removed | their values are fields |
| `_temporary_settings`, `_applying` | removed | carrying the context replaces them |
| `_temporarily` | `_CallContext.with_config_block` and `_under_call_context` | |
| `_inheriting_controls` | `_opening_call_context` | it opens the call's context, and inheriting is one part of that |
| `_carrying_call_context` | unchanged | it says what it does |
| the local `record_token`, and `record` in `_run_call` and `_ByComponent` | `num_samples_used`, and `block_entry` for `_ByComponent`'s per-block metadata | |
| (execution plan) table of active calls | `_ActiveCalls`, instance `_ACTIVE_CALLS` | |
| (execution plan) traced token variable | `_TRACED_CALL_TOKEN` | |
| (execution plan) resolved execution settings | `_ExecutionChoice` | the executor, its options, and where the choice came from |

Names introduced by later steps follow the same pattern: `_EvaluationLog` (E3a), `_EmbeddedCallState` (E4), and `_EvaluationRunTable` (E7).
**Glossary:** "call context" and "evaluation context" are new terms, which the PR that adds `_CallContext` adds to `design/glossary.md` (directive 11).

---

## 7. Where the code goes, and what changes when

### 7.1 Modules

The layering of `design/package-structure.md` and the decided Q5 place the code by what it depends on:

| Module | Contents | Why there |
|---|---|---|
| `core/_call_context.py` (new) | `_CallContext`, `_CALL_CONTEXT`, `current_call_context`, `_under_call_context`, `_NumSamplesUsed`, `_collecting_num_samples_used`, `_note_num_samples_used`, `_ContextBoundFunction`, `_carrying_call_context`; later `_ExecutionChoice` (E3a), `_ActiveCalls` (E4), the JAX settings snapshot (E3a) | `core/config.py` reads `config_blocks`, and `core/_evaluation_points.py` (E3a) carries the context, so it must be in `core/`. It imports nothing outside `core/`. |
| `core/config.py` | the sections read `current_call_context().config_blocks`; `using` derives a context | as #587 |
| `values/_function_base.py` | `_opening_call_context`, the decorator on `check`, `apply`, and `__call__`; `_passing_down` | the call's entry, as `_inheriting_controls` is today |
| `operations/_operation.py` | `_derived_route_scope` calls `_passing_down`, unchanged | as today |
| `functions/_execution.py` | `execute_many` wraps with `_carrying_call_context` until E3a replaces it with `EvaluationChunk` | as today |
| `core/_evaluation_points.py`, `core/_executors.py` (E3a) | `EvaluationChunk` carries `evaluation_context()`; the dispatcher reads `execution_choice` | execution plan §6.14 |
| `functions/_boundary.py` (E4) | `_TRACED_CALL_TOKEN`, the staging of embedded calls, the host side | needs the `Function` engine |

`_ContextBoundFunction` and `_carrying_call_context` move from `values/_function_base.py` to `core/_call_context.py`, because E3a's `EvaluationChunk` in `core/` uses them and `core/` cannot import `values/`.

### 7.2 What #587 changes before it merges

#587 is the PR that creates the one context, so that no later step rewrites its code (OD5).
The changes, in the order a reviewer reads them:

1. **The module.** Add `core/_call_context.py` with `_CallContext` holding the five #587 fields of § 5.1, the context variable, `current_call_context`, and `_under_call_context`.
2. **`config` blocks.** `_Section._current` becomes `_Section._value_in_effect`, which reads `current_call_context().config_blocks`. `_Section.using` and `Config.using` install `current_context.with_config_block(checked_config_values)`. `_TEMPORARY`, `_temporarily`, `_temporary_settings`, and `_applying` are removed.

   ```python
   # #587
   def _current(self, setting):
       block = _TEMPORARY.get().get(self._name)
       ...

   # Proposed: renamed, since "current" alone does not say what it returns (README directive 5)
   def _value_in_effect(self, setting_name):
       section_block = current_call_context().config_blocks.get(self._name)
       ...
   ```

3. **Passed controls.** `_passing_down` installs `current_context.with_passed_controls(...)`. `_INHERITED_CONTROLS` is removed, and `_inheriting_controls` becomes `_opening_call_context` (§ 5.2).
4. **`num_samples` used.** `_recording_monte_carlo` becomes `_collecting_num_samples_used`, which installs a context with fresh `num_samples_used`. `_opening_call_context` uses it, so `_call_function_in_context` no longer starts a `num_samples_used` of its own. `_ByComponent._run_block` uses it for each block, as today. `_current_num_samples` becomes `_num_samples_in_effect`, which reads `current_call_context().controls["num_samples"]` inside a call, and the block's or the session's value outside every call. `_MONTE_CARLO` is removed.
5. **The carrier.** `_carrying_call_context(work_function)` returns `_ContextBoundFunction(work_function, current_call_context().evaluation_context())`, or `work_function` itself when the current context is the default context. In #587 `evaluation_context()` returns the context unchanged, because the fields it drops arrive with E3a and E4.

   ```python
   # #587
   return _CarriedContext(func, inherited, record, settings)

   # Proposed
   current_context = current_call_context()
   if current_context.is_default():
       return work_function
   return _ContextBoundFunction(work_function, current_context.evaluation_context())
   ```

6. **Pickling.** `_CallContext` pickles its mappings as plain dicts, as `_CarriedContext` does today.
7. **Renames.** The names of § 6, in #586 if it has not merged, otherwise in #587, with every docstring that says "the call's record".
8. **Design text.**
   - V.2: replace "A setting read while JAX traces a function is its value at trace time" with the class (a) sentence of § 5.5, and with the user's-JAX program sentence of § 5.5, consequence 3. State the precedence of § 5.3 for the controls #587 covers.
   - VI.0: "The scope covers the calls made in the calling thread and in the workers the engine dispatches to" names the call context as the carrier.
   - The glossary gains "call context".
9. **PR text.** #587's body says a plain `contextvars.copy_context` would not carry the blocks into a Prefect task. The accurate reason is that Prefect's process runners need pickled data and that a copy would carry the workflow frame (§ 2.4, row 7).
10. **Tests.** #587's tests of blocks in threads and Prefect pass unchanged. Two tests are added: a block opened inside one evaluation point of a thread sweep applies to that evaluation point alone (✅ `v09` item 3 shows today's behaviour, which stays), and a carried context pickles and applies in a spawned process (✅ `v05`).

What stays in #587 unchanged:
- the sections, their checks, and their environment variables;
- `config.reset()` and the signature of `Config.using`;
- the precedence;
- `config.prefect`, with its auto-detection.

### 7.3 What the later steps add

Each step adds fields and behaviour to the same type and the same functions, and none adds a second carrier.

| Step | Adds | Uses from #587 |
|---|---|---|
| E3a | `execution_choice` and `_ExecutionChoice`; `evaluation_context()` drops `execution_choice`; `evaluation_log` and its merge into the parent; `jax_settings`, shipped to process workers through the pool's initializer; `EvaluationChunk` carries `evaluation_context()` and returns its `num_samples_used` (OD4); the program store, holding `jax.jit` objects, tracing and lowering each JAX program before its first run (execution plan §6.10, item 3), with the passed controls and the traced settings in its cache key, and the declaration of each `config` setting's kind (OD6, CC4); replay reads `executor` from the context; the broker's requested-execution fields go | `_under_call_context`, `evaluation_context`, `_ContextBoundFunction` (or `EvaluationChunk.run` in its place) |
| E4 | `embedded_calls` and `_EmbeddedCallState`, `_ActiveCalls`; `evaluation_context()` drops `embedded_calls`; `_TRACED_CALL_TOKEN` and `_jax_tracing_active()`; detection and validity by a search of the jaxpr; lazy registration, the rule inside a trace, and the wait before release; raising recorded failures; the token threaded through every JAX program ProbPipe opens that runs user or route code, the inference runners included (execution plan §6.10, item 4, moves into E4); the output check per evaluation point and its cast warning; the error for reading a setting of neither kind while ProbPipe traces | the same |
| E7 | `evaluation_run_table`; `evaluation_context()` drops it | the same |
| E8 | the check of the JAX settings at close and on the host side; their provenance | the same |

### 7.4 Prefect until the `prefect` executor

`config.prefect.resolve_task_runner()` keeps auto-detecting Ray and then Dask, unchanged, until the `prefect` executor replaces it after the workshop (Q9).
Before then, `workflow_kind=TASK` or `FLOW` keeps selecting today's Prefect transport in `functions/_execution.py`, as it does now.
E3a's executor registry holds no Prefect entry and no adapter for one.
The call context crosses into Prefect tasks through the same carrier as into threads.
When a call names an executor and `workflow_kind` selects Prefect, Prefect runs the work and the call warns, as today's warning about `dispatch` and `max_workers` does, with the new control names.
None of this is bridging code: it is today's behaviour, unchanged, with today's carrier renamed.

---

## 8. Worked examples

### 8.1 A `config` block around `condition_on`, with an opaque simulator inside a compiled sampler under `pmap`

```python
@function(traceable=False, output_spec=NumericArraySpec((40,), "float32"))
def forward(theta): ...                                     # runs a solver through subprocess

model = likelihood & prior                                  # likelihood's function calls forward(theta)

with workflow_run(seed=1), config.sampling.using(num_samples=4096):
    posterior = condition_on.using(
        method="blackjax_rwmh",
        method_options={"num_chains": 4, "chain_map": "pmap"},
        executor="slurm", executor_options={"n_jobs": 4},
    )(model, observed)
```

What happens, step by step:
1. **The block.** `config.sampling.using` installs a context whose `config_blocks` is `{"sampling": {"num_samples": 4096}}`. No call is open.
2. **The call opens.** `condition_on`'s context takes the blocks, resolves its controls (its own `method`, `executor`, and `executor_options`; `num_samples` 4096 from the block), and gets `execution_choice = _ExecutionChoice("slurm", {"n_jobs": 4}, "call")`. It registers no token yet.
3. **The sampler compiles once.** The method opens a compiled JAX program whose first argument is the call token. While the store lowers it, `_TRACED_CALL_TOKEN` holds the traced token. The kernel's function calls `forward(theta)` on traced values, so `forward` becomes an embedded call staged with the traced token, and the store records a JAX program with embedded calls. Before running it, the engine registers `condition_on`'s context and gets token 7. The JAX program's cache key includes the traced settings, so `num_samples` 4096 is part of it (OD6).
4. **Four XLA threads.** Under `pmap`, each chain's callback runs on its own XLA thread and sees no context variable (✅ `v02`). Each host side looks up token 7 and finds `slurm`, its options, and the blocks. It runs its evaluation points under the evaluation context, so code inside an evaluation point reads `num_samples` 4096.
5. **A call inside an evaluation point.** If the solver's Python makes a ProbPipe call, that call opens a child of the evaluation context, which has no executor choice, so it uses automatic selection (Q8).
6. **The end.** The call holds a token, so the engine blocks on the sampler's outputs, releases token 7, and records in provenance the executor, its options, and the evaluation points that ran. The block is not recorded.
7. **A second call.** A later `condition_on` outside the block, with no executor, gets a new token when it runs the sampler. Its `num_samples` differs, so under OD6 the sampler compiles once more for it. A later call inside an equal block reuses the first JAX program, and its callbacks find their own call through its token.

✅ `v13` runs this case in miniature, with a JAX program keyed by shapes alone, so it shows the run-time half of the design: four XLA threads each saw `slurm` and 4096 for the first call, a call inside an evaluation point saw no executor and 4096, and the second call's callbacks saw `sequential` and 1024 through the same compiled sampler.

### 8.2 A sweep on a thread executor

```python
@function
def loss(theta):
    return mean.using(method="monte_carlo")(model_at(theta))   # a Monte Carlo summary inside the body

grid = NumericArrayBatch(thetas, level_names=("theta",), label="grid")

with workflow_run(seed=3), config.sampling.using(num_samples=4096):
    losses = loss.using(executor="thread", executor_options={"max_workers": 8})(grid)
```

**#587 today** (`dispatch="thread"` in place of the executor): the wrapper carries the inherited controls, the record, and the blocks, so each worker reads 4096 (✅ `v09` item 2).

**Proposed:**
1. `loss`'s call opens a context with the block, `execution_choice = ("thread", {"max_workers": 8}, "call")`, and its own `num_samples_used`.
2. Each cell is an evaluation point. The thread executor gives each worker `_ContextBoundFunction(evaluate_cell, call_context.evaluation_context())`, from E3a an `EvaluationChunk` holding the same evaluation context. One copied `Context` cannot serve two threads (✅ `v01`), and copying the whole context would carry the workflow frame (✅ `v12`), so carrying the named value is the mechanism.
3. In a worker, `mean`'s call opens a child of the evaluation context. It inherits the block (4096) and no executor, so a Monte Carlo summary runs where the cell runs. Its `num_samples_used` is its own, and `loss`'s provenance does not collect them, because `loss` is not a derived operation.
4. A `config.using` block that the body opens in one cell applies to that cell alone (✅ `v09` item 3: evaluation point 0 read 7, the other evaluation points and the caller read 1024).

### 8.3 A spawned process executor

```python
jax.config.update("jax_enable_x64", True)

with workflow_run(seed=4), config.sampling.using(num_samples=2048):
    y = forward.using(executor="process", executor_options={"max_workers": 4})(design)
```

1. `forward`'s call opens a context with the block, the executor choice, and a `jax_settings` snapshot of the four shipped settings, here with `jax_enable_x64` true (E3a).
2. The `process` executor checks that the chunks can be shipped, then ships each `EvaluationChunk` with the evaluation context pickled inside it.
3. The pool starts each worker with an initializer that applies the shipped JAX settings before the worker unpickles anything. `EvaluationChunk.run` then installs the evaluation context, enters each evaluation point's workflow scope, and runs the evaluation points. Without the settings the worker draws in float32 (✅ `v05`), and settings applied after unpickling are too late: a float64 array in the payload had already become float32 (review test R4). `jax_num_cpu_devices` cannot be changed in a running process, so it is not shipped.
4. A ProbPipe call inside an evaluation point opens a child of the evaluation context, so it sees the block and uses automatic selection.
5. The chunk's outcomes carry the `num_samples_used` its evaluation points noted, and the parent merges them into the call's (OD4). Today a process worker notes into a copy that the call never receives (#586's docstring).

---

## 9. Trying to break the design

Each case was tried against the contract as a skeptical reviewer would.

1. **Nested calls.** `standard_deviation.using(num_samples=500)(y)` opens a context whose derived route's scope derives `passed_controls = {exact_only: False, num_samples: 500, ...}`. `variance` opens a child, takes 500 unless its own view sets a count, and clears the passed counts for its own inner calls. Its `num_samples_used` merges into `standard_deviation`'s on success, because the derived route collects calls. This is today's behaviour, with one variable in place of two.
2. **Threads inside a callback.** A host side that runs the `thread` executor gives each worker the evaluation context of the token's call. The worker's ProbPipe calls open their own contexts and tokens. ✅ `v06` ran a thread pool inside a callback, under `jit` and under `pmap`.
3. **An exception mid-call.** The `finally` clause releases the token and restores the variable, and `num_samples_used` is not merged (✅ `v07`: the variable was empty and the table was empty afterwards). An exception inside a worker evaluation point restores that worker's variable through `_under_call_context`, so a reused pool thread starts clean.
4. **Two concurrent calls on different threads.** Each thread has its own context variable, each call its own token, and the compiled JAX program is shared. ✅ `v07`: two threads made five calls each through one compiled JAX program, and every callback found its own call.
5. **Re-entrant calls.** An evaluation point run by a host side opens a ProbPipe call, which opens its own JAX program with its own token while the outer JAX program waits. ✅ `v06`: under `jit` and under `pmap`, each inner call found its own context with no executor, and the table was empty at the end. The lock of `_ACTIVE_CALLS` is held only for the lookup, so re-entry cannot deadlock on it.
6. **A `config.using` block opened inside an evaluation point.** It derives from the evaluation context and is reset on exit. ✅ `v09` item 3 for evaluation points on threads, and ✅ `v11` for embedded evaluation points on XLA threads: inside the block an evaluation point read 7, and after it 1024. In `v11` JAX started new XLA threads for each `pmap` call, so a leak across calls did not appear. JAX documents no such rule, so the reset on exit is required, not optional.
7. **A JAX program that outlives its call.** JAX dispatches asynchronously, so a call could return while its JAX program still runs. A JAX program with embedded calls is waited for before its token is released (§ 5.2), so no callback finds a released token (✅ `v04`). A JAX program without callbacks may outlive its call, which is harmless because it never looks a token up.
8. **A call that opens while tracing.** A `Function` called inside a traced kernel opens a context with no embedded-call state, so it never registers a token. Its embedded calls pass the traced token, so the call that runs the JAX program decides the executor at run time (✅ `v06`, nested JAX programs). Its explicit choice, if any, is static, which is correct because it is part of the traced code.
9. **A JAX program the user opens.** It passes token 0, so its callbacks run under the default context on every thread. Under plain `jit` the callback runs on the calling thread and could see that thread's context, while under `pmap` it could not (✅ `v02`). Using the default context makes the two agree (OD3).
10. **A term used after its call.** A law returned by one call holds no context (CC7). A later `sample` of it runs under the later call's settings, which is #587's stated rule that "a setting governs the functions constructed before it".
11. **`check` against the call.** `check` opens a context with the same resolution, so it reports what the call would do (P4). It registers no token, because it runs no JAX program.
12. **A block holding an object that does not pickle.** A `config.prefect.using(task_runner=...)` block is carried with the evaluation context, so a process runner pickles the task runner into every task. E3a's `shipping_check` reports a value that does not pickle before any evaluation point runs. The behaviour is #587's, and § 11 does not change it.
13. **The provenance mode inside a `workflow_run`.** Today the block is ignored for provenance while `config.provenance.mode` reads the block's value (✅ `v09` item 1). OD2 settles it.

---

## 10. Verification log

The scripts are kept by the author outside the repository, beside the execution plan's (gaps W4), in the gitignored `.claude/plans/call-context-verification/` of the main checkout.
Beside them, `ctxproto.py` is a miniature of the design, which tests it and is not proposed code.
They ran with the Python of #587's worktree, `uv sync --extra dev`.
Scripts marked † import #587's code and run from its worktree.
Scripts marked ‡ need `XLA_FLAGS=--xla_force_host_platform_device_count=4`.

| Script | What it checks | Result |
|---|---|---|
| `v01_threads_contextvar.py` | a context variable in `ThreadPoolExecutor` workers, with and without `copy_context`; one `Context` entered by two threads | not visible without a copy; visible with one; a set in the worker stays there; a shared object is shared; the second entry raises `RuntimeError` |
| `v02_callback_threads.py` ‡ | the thread and the context of a `pure_callback` under `jit`, `jit(vmap)`, `pmap`, and `jit` from a worker | the calling thread, with the variable visible, under `jit`; four XLA threads, with the variable not visible, under `pmap` |
| `v03_token_argument.py` ‡ | a token argument under `jit(vmap)` and `pmap`, one compiled JAX program serving two calls; a setting read while tracing; Python `int` tokens | one trace and two different settings seen, under both; the setting read while tracing gave the first call's value twice; an `int` of `2**31` overflowed |
| `v04_async_dispatch.py` | whether a JAX program can call back after the Python call returns | a JAX program with a callback returned after the callback; a JAX program without one returned before its result was ready |
| `v05_spawn_process.py` | pickling a carried context; a spawned worker with and without the parent's JAX settings | `MappingProxyType` does not pickle; the context with dicts pickles; the worker read the block; float32 without the settings, float64 with them |
| `v06_nested_reentrant.py` ‡ | a JAX program opened while another is traced; a user's `jit`; a callback that opens its own call and JAX program; a thread pool inside a callback; the same under `pmap` | the right call each time, one cache entry; token 0 for the user's `jit`; inner calls had no executor; the table was empty at the end |
| `v07_concurrent_and_errors.py` | two threads calling one compiled JAX program with different tokens; a failing callback; an exception mid-call | each call saw its own settings, one cache entry; the recorded failure was raised in place of JAX's `ValueError`; the variable and the table were restored |
| `v07b_error_type.py` | the type of the exception a failing callback raises, in a minimal JAX program | `JaxRuntimeError` |
| `v08_jax_settings_cache.py` | whether `jit` traces again after a JAX setting changes; a stored compiled executable after an x64 change | traced again for x64, the matmul precision, and `jax_threefry_partitionable`; the stored executable kept float32 |
| `v09_pr587_behaviour.py` † | #587: a provenance block inside `workflow_run`; a block in thread workers; a block inside one evaluation point; a setting read in a user's `jit` | provenance kept although the mode read OFF; workers read 4096; only the evaluation point that opened the block read 7; the user's `jit` kept 1024 inside a block of 10 |
| `v10_prefect_context.py` † | whether a Prefect task sees the submitting context; #587 under `workflow_kind=TASK` in a block | Prefect's thread runner copied the context; the tasks read 4096 |
| `v11_xla_thread_leak.py` ‡ | a variable set on an XLA thread without a reset; a block inside an embedded evaluation point | new threads for each `pmap` call, so no leak appeared; inside the block 7, after it 1024 |
| `v12_copy_context_frame.py` † | a ProbPipe draw in a worker that runs a copied context inside `workflow_run` | `UnmanagedConcurrentWorkflowEntryError` |
| `v13_worked_example_pmap.py` ‡ | worked example 8.1 in miniature | as § 8.1 states |
| `v14_token_shape.py` | the shape of an unbatched token at the host under `jit`, `vmap`, and two `vmap`s | `()`, `(1,)`, `(1, 1)`, dtype `uint32` |

**Not verified:**
- Behaviour on a GPU or a TPU, where callbacks run on runtime threads and dispatch is asynchronous in more cases (execution plan §8 already lists this).
- Ray and Dask task runners for Prefect, which are not installed. The design assumes they pickle the payload, as a process does.
- Why the failing callback of `v07` arrived as `ValueError` and the one of `v07b` as `JaxRuntimeError`. The design does not depend on the type.
- Free-threaded Python, where the set in `num_samples_used` needs its lock. The design gives it one.

---

## 11. Open decisions

### OD1. Where `config` blocks are kept

**Background.** #587 keeps the active blocks in their own context variable, `_TEMPORARY`. The token table must give a callback the blocks too (§ 2.4, row 2), so the value registered under a token must hold them.

**Options.**
- **A.** The blocks are the `config_blocks` field of the one call context. A block derives a new context. Registering the call context registers the blocks.
- **B.** The blocks keep their own variable, and the carrier and the token table capture it beside the call context.

Under B the token table's entry holds a copy of the blocks taken when the call opened, while the variable holds the current value, so the blocks have two homes, which CC1 forbids. B also keeps two capture functions.

**Recommendation: A.**

### OD2. Where the provenance mode is kept

**Background.** A `workflow_run` copies `config.provenance.mode` into its frame when it opens, and a Prefect payload carries that copy. #587's block then sets the mode for every read except inside an open `workflow_run`, where the frame's copy wins (✅ `v09` item 1).

```python
with workflow_run(seed=0):
    with config.provenance.using(mode=ProvenanceMode.OFF):
        r = double(1.0)
config.provenance.mode     # reads OFF inside the block
r.provenance               # a Provenance: the frame's LIGHTWEIGHT applied
```

**Options.**
- **A.** The mode is a setting read when a call opens, like every other. The frame and the Prefect payload stop carrying a copy, and the call context carries the block.
- **B.** The mode is fixed for a whole `workflow_run`. A `config.provenance.using` block inside an open `workflow_run` raises, naming the run.
- **C.** Keep both homes and document that a block inside an open run has no effect.

A changes `workflow_run`'s documented rule that "a top-level workflow takes the mode in effect when it is entered". B keeps that rule and makes the conflict visible. C leaves a silent conflict, which P10 rules out.

**Recommendation: A**, in a small PR of its own after #587 and before E3a, because it removes a home and a transported field. B is acceptable if the per-run rule matters to replay.

### OD3. What a JAX program the user opens gets

**Background.** A JAX program the user opens passes no token. Under plain `jit` its callbacks run on the calling thread, which may hold a block. Under `pmap` they run on XLA threads, which hold nothing (✅ `v02`).

**Options.**
- **A.** Token 0 makes the host side run under the default context, on every thread: session values, automatic selection, and a failure that raises at once.
- **B.** The host side runs under whatever context its thread has.

Under B the same code reads a block under `jit` and the session's value under `pmap`, so the device count changes what runs. A gives one answer, and a user who wants a block or an executor applied runs the opaque code through a ProbPipe call.

**Recommendation: A.**

### OD4. `num_samples` used in process workers

**Background.** #586 notes that a worker process counts its draws into a copy that the call never receives, so the call's provenance can miss them.

**Options.**
- **A.** From E3a, an `EvaluationChunk` returns the `num_samples_used` its evaluation points noted with its outcomes, and the executor's merge adds them to the call's counts.
- **B.** Keep #586's behaviour and document it.

**Recommendation: A**, in E3a, which builds the chunk anyway.

### OD5. Which PR unifies the context

**Options.**
- **A.** #587 creates `_CallContext` and removes its three variables before it merges (§ 7.2).
- **B.** #587 merges with the renames and the design-text change only, and a follow-up PR before E3a replaces its carrier and its variables.

B rewrites #587's `_CarriedContext`, `_TEMPORARY`, and their helpers soon after they merge, which is building twice. A costs #587 about a day of work, mostly in `values/_function_base.py` and `core/config.py`, and leaves its user-facing behaviour and tests unchanged.

**Recommendation: A**, agreed with jhuggins. B is the fallback if #587 must merge first for other PRs.

### OD6. What the cache key of a JAX program ProbPipe opens includes

**Background.** ProbPipe code traced inside a JAX program ProbPipe opens can read the passed controls and some `config` settings, for example when a kernel's function calls an operation whose routes depend on `exact_only`, or a Monte Carlo summary whose draw count is `num_samples`. A value read there is baked into the JAX program (✅ `v03`), so it must be part of the JAX program's cache key.

Not every setting can be part of a cache key. A `config.prefect.using(task_runner=RayTaskRunner())` block holds an object with no meaningful hash, and a cache key by object identity is forbidden (J7). Settings such as `config.notation.max_depth` and `config.provenance.mode` steer only Python and never change a compiled JAX program.

**Options.**
- **A.** The cache key of every JAX program ProbPipe opens includes every active `config` block and the passed controls.
- **A′.** Each `config` setting declares its kind, traced or presentation (CC4). The cache key of every JAX program ProbPipe opens includes the passed controls and the current values of the settings declared traced; presentation settings never enter it. A traced setting's value must be hashable, which its section's check enforces. Today the traced settings are `sampling.max_evaluations` and `sampling.num_samples`; `notation` and `provenance` are presentation settings; `prefect` is read only in Python orchestration, never in traced code.
- **B.** Each owner of a JAX program lists the settings its traced code reads, and only those enter its key.
- **C.** ProbPipe traces its JAX programs under the default context, so traced code never sees blocks or passed controls.

A breaks on a block that holds an unhashable object, and it recompiles for changes that cannot affect a JAX program. B depends on every owner listing every read, and a missed one is a silent wrong result. C changes what traced code computes, since `exact_only` would stop restricting routes chosen inside a kernel. A′ puts the declaration where the setting is defined, once, so a new setting states its class when it is added (§ 5.5's test), and a changed traced setting costs one compilation, which is rare.

**Recommendation: A′.** This replaces the first draft's recommendation of A.
Only JAX programs that run user code or route code take the token and the traced settings in their cache key. A family's own sampler runs neither, so a sweep over `num_samples` does not recompile it.

### OD7. Embedded calls inside a compiled set of evaluation points

**Background.** Q8 says a call made inside an evaluation point uses automatic selection, so a cluster job does not submit cluster jobs. A traceable lifted body that calls an opaque `Function` is compiled by the `jax` executor, and the opaque call becomes an embedded call inside that one JAX program. The same body run by `sequential` runs each evaluation point in Python, where a ProbPipe call opens a child of the evaluation context.

```python
@function
def growth(r, K):                       # traceable
    return forward(r * K)               # forward is opaque

growth.using(executor="slurm")(prior)   # where do forward's evaluation points run?
```

**Options.**
- **A.** An embedded call inside a compiled set of evaluation points belongs to the call that runs the JAX program, and uses its executor. `forward`'s evaluation points go to `slurm`.
- **B.** The `jax` executor registers the evaluation context for its JAX programs, so embedded calls inside them use automatic selection, as Q8 reads literally.

Under B the executor of nested work depends on whether the outer callable traces, and `growth.using(executor="slurm")` would never send `forward` to the cluster, which is the case the executor exists for.

**Recommendation: A.** Q8's text then says it applies to ProbPipe calls that an evaluation point makes in Python.

### OD8. What the program store holds

**Background.** JAX's own `jit` cache traces again when a global setting such as `jax_enable_x64` changes (✅ `v08`). A compiled executable, as `jax.jit(f).lower(...).compile()` returns, keeps the settings it was compiled under, and ran in float32 after 64-bit types were turned on (✅ `v08`).

**Options.**
- **A.** The store holds `jax.jit` objects. Lowering one and then calling it compiles once (review test R1), JAX keys it by the settings, and `.trace()` gives the jaxpr that detection searches.
- **B.** The store holds compiled executables and adds the JAX settings to its key by hand.

**Recommendation: A.**

Settled elsewhere and listed for completeness: there is no `config.execution` section for now (gaps D7, option 1), and Prefect's auto-detection stays until the `prefect` executor (§ 7.4).

---

## 12. Draft comment for PR #587

> Not posted. For jhuggins, from the maintainer.

---

Thanks for #587. The `config` sections and the `using` blocks are what we want, and the user-facing behaviour stays as you wrote it. I'm asking for one structural change before it merges, because of how it meets the execution-layer plan (#580). It is about a day of work, and I'm glad to push the commits to your branch myself if you prefer.

**Why.** The execution plan needs per-call state to follow a call into host callbacks inside compiled JAX programs. Under `pmap` those callbacks run on XLA's own threads, which see no context variable, so the plan passes a small integer "call token" into every compiled JAX program and looks the call up in a table. Whatever the callback needs must be in the value registered under that token: the executor, and also your `config` blocks, the inherited `exact_only`, and the `num_samples` its Monte Carlo estimates used. Today #587 keeps these in three context variables (`_TEMPORARY`, `_INHERITED_CONTROLS`, `_MONTE_CARLO`) and carries them with `_CarriedContext`. If the execution plan added its own call context next to them, we would have two carriers for the same state.

**The proposal.** One frozen `_CallContext`, in one context variable, in a new module `core/_call_context.py`. Your three variables become three of its fields, and the execution steps add their fields to the same type later. The full design is in `plans/call-context-proposal.md`.

**Concrete changes in #587:**
1. Add `core/_call_context.py` with `_CallContext` (fields `config_blocks`, `passed_controls`, `label`, `controls`, `num_samples_used`), the context variable, `current_call_context()`, and `_under_call_context(call_context)`.
2. `_Section._current` becomes `_Section._value_in_effect` and reads `current_call_context().config_blocks`, and `using` installs `current_context.with_config_block(...)`. `_TEMPORARY`, `_temporarily`, `_temporary_settings`, and `_applying` go.
3. `_passing_down` installs `current_context.with_passed_controls(...)`, and `_INHERITED_CONTROLS` goes. `_inheriting_controls` becomes `_opening_call_context`, which opens a child context for every call.
4. `_recording_monte_carlo` becomes `_collecting_num_samples_used`, and `_MONTE_CARLO` goes. The call's resolved `num_samples` is read from `call_context.controls`, so `_MonteCarloRecord.num_samples` goes.
5. `_carrying_call_context(work_function)` returns `_ContextBoundFunction(work_function, current_call_context().evaluation_context())`. In this PR `evaluation_context()` returns the context unchanged.
6. Renames, so no name uses "record" for anything but `Record`: `_MonteCarloRecord` → `_NumSamplesUsed`, the field `record` → `num_samples_used`, `.recorded` → `.provenance_num_samples`, `_note_monte_carlo` → `_note_num_samples_used`, `_CarriedContext` → `_ContextBoundFunction`, and the locals named `record`. If #586 is still open, these belong there.
7. In V.2, replace "A setting read while JAX traces a function is its value at trace time." A compiled JAX program is reused across calls, so a value baked in while tracing leaks into the next call. The replacement says that ProbPipe reads settings when a call starts, in Python, and that ProbPipe code inside a user's own `jax.jit` keeps the values in effect while that JAX program was traced.
8. Add "call context" to the glossary.
9. In the PR body, the reason for carrying explicitly is that Prefect's process runners need pickled data and that `copy_context` would also copy the workflow frame. Prefect's thread runner does copy the context.

Your tests should pass unchanged. Two to add: a block opened inside one evaluation point of a thread sweep applies to that evaluation point alone, and a carried context pickles and applies in a spawned process.

**One question for you.** Inside an open `workflow_run`, `config.provenance.using(mode=OFF)` reads OFF, but the run's frame keeps the mode it copied at entry, so provenance is still recorded. I'd make the mode a per-call setting like the others and drop the frame's copy, in a small follow-up PR. Does the per-run rule matter for anything you know of?

Nothing else changes. The sections, their checks and environment variables, `reset()`, and the precedence stay as they are, and so does `config.prefect` with its Ray and Dask detection.

---

## 13. Revisions

### The first revision

| # | Change | Why |
|---|---|---|
| 1 | A call registers a token only before it runs a program with embedded calls, and only such a call waits for its programs at close (§ 5.2). The fields `token` and `failures` became one field, `embedded_calls`, holding a `_EmbeddedCallState`. | The first draft registered a token and waited at the end of every call, which would remove JAX's asynchronous dispatch from every call. Only a program that can call back needs its token kept open. |
| 2 | OD6 recommends A′: the program key includes the passed controls and the settings declared traced, not every `config` block. | A block can hold an unhashable object, such as a Prefect task runner, and most settings never change a compiled program. |
| 3 | `_InCallContext` is renamed `_ContextBoundFunction`. | The name says what the object is: a function bound to the call context it runs under. |
| 4 | The draft comment for #587 offers that the maintainer pushes the changes to the branch. | The unification is a day of work on a PR that is otherwise ready, and should not read as handing it back. |

### The second revision, after the adversarial review

| # | Change | Why |
|---|---|---|
| 5 | E4 threads the token through every program ProbPipe opens that runs user or route code, the inference runners included, which moves the execution plan's §6.10 item 4 into E4. | The runners ran eager `vmap`, `pmap`, and `scan` over closures, so their embedded calls staged token 0, and `condition_on.using(method="blackjax_rwmh", executor="slurm")` would silently run sequentially. |
| 6 | Detection of a program with embedded calls is a search of the lowered jaxpr for ProbPipe's own callback, and the same search refuses a token operand that is not the program's token argument (§ 5.2). E3a builds the lowering. | A flag set while staging misses a cached inner `jit` (review test R1). A user's jitted helper captured a stale or literal token (review test R2). No step before the workshop built the lowering. |
| 7 | When any trace is active, the engine registers no token and waits for nothing, and passes the traced token or 0 (§ 5.2). | A program run inside an outer trace had its token baked in, and the cached outer program later sent a released token (review test R3). |
| 8 | JAX settings are shipped from E3a, through the pool's worker initializer, from an allowlist of four settings. E8 keeps the check and provenance. | Settings applied after unpickling are too late (review test R4), the cut order removed E8, and no public API lists every setting's default. |
| 9 | CC4 says traced code reads only traced settings, and reading another while ProbPipe traces raises. | The first text said ProbPipe never reads a setting while tracing, while OD6 said traced code reads some. |
| 10 | `execution` is the one home of the executor choice, and a `Function`'s own executor is read by the host side from the `Function`, never from a program key. | `controls` also held the executor controls, and executor options can hold a pool or a task runner with no hash. |
| 11 | OD7: an embedded call inside a compiled point set uses the call's executor. | Q8 read literally made the executor depend on whether the outer callable traces. |
| 12 | OD8: the program store holds `jax.jit` objects. | JAX keys them by its settings, and lowering then calling compiles once. |
| 13 | `point_sets_run` merges into the parent on close; `check` opens no draw counts and `apply` shares its parent's; `is_empty()` is in the sketch; § 5.7 lists the replay variables and `_BATCHED_INIT_BYPASS`; "requester" replaces "consumer"; "program with embedded calls" replaces "program with callbacks". | Gaps and text errors the review found. |
| 14 | Not adopted: releasing tokens by sweeping finished programs in place of the lazy scheme. | The search of the jaxpr makes the lazy scheme sound, and the sweep keeps every call registered. |
| 15 | CC4 distinguishes traced settings from presentation settings, which stay readable while tracing; the OD6 key and the declaration of each setting's kind move into E3a, which builds the program store; detection traces with `jax.jit(f).trace(...)` and then lowers; the closing order merges `point_sets_run`. | The final review: labelled terms read `notation.max_depth` inside traced code today, a store without its full key until E4 could carry a value across calls, and a `Lowered` object has no jaxpr. |
| 16 | (2026-10-18) *Point* is renamed *evaluation point*: *point context* became *evaluation context*, `for_point()` became `evaluation_context()`, *point boundary* became *evaluation boundary*, the point-run table became the run table (`_RunTable`, field `evaluation_runs`), `point_sets_run` and `_PointSetLog` became `evaluations_run` and `_EvaluationLog`, `PointChunk` became `EvaluationChunk`, and `functions/_point_keys.py` became `functions/_evaluation_keys.py`. | The maintainer and jhuggins found "point" too generic. |
| 17 | (2026-10-18) "JAX program" replaces "program" wherever a compiled JAX program is meant, with a definition in § 4; *empty context* became *default context* and `is_empty()` became `is_default()`; *evaluation boundary* is dropped, since it named the place an evaluation point starts and no boundary; `_tracing_active()` became `_jax_tracing_active()`. | The maintainer's review. |
| 18 | (2026-10-18) "Cache key" replaces "key" for what a cache looks a compiled JAX program up by, so "key" keeps its random-key and mapping senses; `_CallContext` gets a full docstring; the fields `evaluations_run` and `evaluation_runs`, easy to confuse, became `evaluation_log` and `run_table`. | The maintainer's review. |
| 19 | (2026-10-18) `run_table` became `evaluation_run_table`, and `_RunTable` became `_EvaluationRunTable`. | The maintainer's review. |
| 20 | (2026-10-18) A derived operation's "identity" is its **formula**, and the identity route is the **derived route**, so `passed_controls` are the controls a derived route passes to the calls its formula makes, and the identity scope is the derived route's scope (`_derived_route_scope`). | The maintainer's review. |
| 21 | (2026-10-18) The record of the Monte Carlo draws a call took is named after the public control it reports: `draw_counts` became `num_samples_used`, `_MonteCarloDrawCounts` became `_NumSamplesUsed`, `_note_monte_carlo_draws` became `_note_num_samples_used`, and `_counting_monte_carlo_draws` became `_collecting_num_samples_used`; `execution` became `execution_choice`. | The maintainer's review: one name for what the user sees as `num_samples`, and "draw counts" also named an MCMC method's draws. |
| 22 | (2026-10-18) The wrapper inside `_opening_call_context` is named `run_in_child_call_context`, not `entered`. | The maintainer's review. |
| 23 | (2026-10-18) Vague names in proposed code replaced under README directive 5's naming rules: `current` became `current_context`, `parent` became `parent_context`, `view` became `function_view`, the opened `context` became `child_context` and a looked-up one `call_context`, `method` became `entry_method`, `values` and `passed` became `config_values` and `passed_controls`, `func` became `work_function`, `token` became `call_token` in signatures, `_execution_choice` became `_resolve_execution_choice`, and `_ExecutionChoice`'s `options` became `executor_options`; in the quoted host side, `call`, `points`, `outcomes`, and `ranks` became `call_context`, `evaluation_points`, `evaluation_outcomes`, and `argument_ranks`. | The maintainer found `entered` and `current` too vague. |
| 24 | (2026-10-18) Four mentions of a derived operation's "identity" missed by row 20 now say "derived route" or "formula"; #587's `_Section._current` becomes `_value_in_effect(setting_name)` and #586's `_current_num_samples` becomes `_num_samples_in_effect`, under README directive 5's rule 5; a "record" noun removed. | The maintainer's review. |
