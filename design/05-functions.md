# Part V — Functions

A **`Function`** (III.3) wraps an ordinary Python callable. This part describes its **engine**, the call semantics beyond plain evaluation that are installed on the base at import. The user writes a plain function over its "natural" values, and wrapping it makes that callable (i) **lift** automatically over distribution- and batch-valued arguments and (ii) **act** as a tracked node in a computation graph, so its result carries provenance. `Function`s therefore compose into a workflow.

Wrapping a callable `f` as a `Function` runs one **stack** of steps on every call. V.1 states the stack and what it reads from the `Function`; the sections that follow describe each step in the order it runs, with what it requires and what happens when that fails; the last three state the construction-time claims that the operations and the executors read, and the reparameterization factory.

| §     | Step or concern   | What it covers                                                                                                  |
| ----- | ----------------- | --------------------------------------------------------------------------------------------------------------- |
| V.1  | the stack         | the `Function`, its engine, the three declarations the engine reads, and the steps in order                     |
| V.2  | controls          | how the framework's controls are set and resolved apart from the arguments of `f` (step 1)                      |
| V.3  | binding           | how arguments bind to the signature and become dependencies or inputs (step 2)                                  |
| V.4  | normalization     | wrapping raw hosts into their kinds, converting distributions, and admitting each argument (step 3)             |
| V.5  | lifting           | which arguments are lifted, how they group, and the regime that results (step 4)                                |
| V.6  | planning          | checking that the known specs conform in one dimension scope, and deferring what only the return can settle (step 5) |
| V.7  | resolution        | the evaluation-rule registry and the selection of the route that realizes the call (step 6)             |
| V.8  | randomness        | workflow scopes and the structural keys the evaluation points consume                                           |
| V.9  | execution         | running sets of evaluation points on an executor, carrying the call's settings, and recording (step 7)          |
| V.10 | return            | assembling, validating, wrapping, and identifying the result, or detaching it under `raw=True` (step 8)          |
| V.11 | differentiability | the construction-time claim of which inputs gradients propagate through                                          |
| V.12 | reparameterization | the constraint-to-bijector factory                                                                            |
| V.13 | execution claims  | how a callable may be run, and the boundary between JAX code and opaque code                                    |

## V.1 — The `Function` and its engine

### Contract

A `Function` wraps exactly one callable and presents it as a node in a computation graph. It is created with the `@function` decorator:

```python
@function
def predict(theta, x): ...                    # an ordinary callable over concrete values

@function(max_evaluations=500, executor="thread")   # optional construction-time controls
def predict(theta, x): ...
```

**The engine.** The engine is one callable installed as the base's call handler (III.3), once, at import. On concrete values it agrees with plain evaluation, adding only the wrap and the provenance. Every call runs the stack below in order, and a failure ends the call at its step. The stack reads three declarations from the `Function` it runs: what each parameter **accepts**, the **result's output spec**, and the **realization**. For a `@function` the three are as follows. A parameter accepts a value that conforms to its input's spec, or any value where it is unannotated. The result's output spec is the `output_spec` given at construction, completed per call (II.2), or is read from the return when none was given. The realization is the body, or, for a lifted application, the route the evaluation registry selects (V.7).

1. **Configure** (V.2). The effective controls are resolved by the precedence that V.2 states, which takes the controls a derived operation passes down (VI.0) and an enclosing call's executor choice (V.9) into account.
2. **Bind** (V.3). The arguments bind to the wrapped function's signature, tracked arguments as dependencies and the rest as inputs.
3. **Normalize** (V.4). Each argument is wrapped into its kind, a conversion is planned where its parameter names another distribution class, and the result is admitted against what the parameter accepts.
4. **Classify the lift** (V.5). The lifted arguments are found and grouped, resulting in one of the following lifting regimes: a plain call, a broadcast, a sweep, or a nested sweep.
5. **Plan** (V.6). The available specs are checked to conform in one dimension scope, the result's output spec is derived, and the checks only the return can settle are deferred.
6. **Resolve** (V.7). The route that realizes the call is selected among the feasible candidates.
7. **Execute** (V.9). Planned conversions execute, then the selected route runs with its keys (V.8), and its evaluation points run on the executor the call's controls choose.
8. **Return** (V.10). The outputs are assembled, validated against the completed output spec, wrapped, and given identity, or detached under `raw=True`.

**Checking feasibility.** `f.check(...)` runs steps 1 to 6 without numerical execution or random events. It raises a failure of steps 1 to 5 as the call does, for example the binding `TypeError` or an `ApplicabilityError`, and a conversion plan that lacks target information is reported unresolved instead (V.4). Each route reports feasible, infeasible with a reason, or unresolved with the declarations it needs (II.7). `check` returns these reports in a `CallReport`, which it fills without invoking the body on values or constructing a conversion; the probe of a callable that states nothing may run its Python up to the first operation that fails on a tracer (V.13). `using(...).check(...)` probes under the same controls as the corresponding call.

```python
@dataclass(frozen=True)
class CallReport:
    routes:      tuple[MethodInfo, ...]         # one per route, in selection order
    selected:    MethodInfo | None              # None when selection cannot be decided
    deferred:    tuple[str, ...]                # the checks left to return
    result:      OutputSpec | None              # the result's planned output spec (V.6); None when only the return settles it
    lifted:      tuple[str, ...]                # the parameters whose arguments lift or sweep (V.5)
    conversions: Mapping[str, ConversionInfo]   # the planned conversion of each parameter that converts (V.4)
    execution:   ExecutionReport                # how each callable's evaluation points will run (V.9)
```

### Rationale

This makes `C1 – Uniform interface to functions, distributions, and values` and `C4 – Function lifting` operational: a user writes mathematics as an ordinary, testable function, and ProbPipe lifts it to act on distributions and values without the function being rewritten. Making every `Function` a graph node delivers `C6 – Traceable and reproducible workflows`: each result records how it was produced, and a whole workflow can later be traced or re-run. Because the wrapper changes only invocation and tracking, the operations can be *defined* as `Function`s and inherit all of it. The stack is the boundary principles made mechanical: normalization is `B1 – Either presentation in`, execution over raw types is `B2 – Representations only inside`, and return is `B3 – Tracked forms out by default`; planning and the lift are what the engine promises beyond them.

## V.2 — Controls (step 1)

### Contract

A `Function` keeps two namespaces apart: the wrapped function's arguments, which are every positional and keyword argument of a call, and the framework's **controls**, which select realization, execution, or presentation. The construction metadata `label`, `output_label`, and `output_spec` are fixed at construction, apart from the controls. The controls are `max_evaluations`, `num_samples`, `include_inputs`, `method`, `exact_only`, the per-parameter `conversions`, `method_options`, `raw`, the executor selectors `executor` and `executor_options` (V.9), and the orchestration selector `workflow_kind`. `method_options` is one mapping of the numerical budgets the selected method reads, such as an MCMC method's warmup and draw counts, and the selected method validates its entries when it runs. The name that `method` gives resolves among an operation's routes and the methods of its registry routes (VI.0). Every ProbPipe-caused draw's key comes from the workflow scope (V.8), so a call takes no framework key or seed. A wrapped function's own `key` or `seed` parameter binds as an ordinary argument.

**Setting a control.** A call resolves its controls once, when it opens, and the code it runs reads the resolved values from its call context (V.9). Each control takes its value from the first of these sources that gives one:
1. the call's own explicit setting, which is the `Function`'s controls from its decorator or constructor and from the view that `using` returns, the view's value winning;
2. a value an enclosing call passes down: the `max_evaluations` and `num_samples` that a derived operation passes to the calls its formula makes directly (VI.0), and the executor choice of an enclosing call, by the rules of V.9;
3. the innermost active `config.using` block that sets the control's default;
4. the session's value in `config`;
5. the framework's default, which is the session's initial value.

`exact_only` is the one exception: it restricts route selection, so it holds when the call sets it or an enclosing derived operation passes it, whatever the call's own view sets. A count that a derived operation passes therefore wins over a `config.using` block opened inside the derived call, and loses to the count that the called `Function` sets itself. The view leaves the original `Function` unchanged. Every control has a default, and a view that sets a control to `None` removes its explicit setting, whatever the decorator set, so the control takes its value from sources 2 to 5. `using(executor=None)` therefore restores the inheritance of an enclosing call's executor (V.9). Per-call controls go through `using` for both ordinary functions and operations, and direct call keywords always bind to the authored signature. Thus a wrapped function's own `raw` parameter remains an argument, distinct from the framework's control of that name.

A control is set in one of three ways, from the narrowest to the widest:

| Way | Example | What it governs |
|---|---|---|
| on a `Function` | `mean.using(num_samples=500)(posterior)` | every call of the configured copy that `using` returns |
| as a block | `with config.sampling.using(num_samples=500):` | every call inside the block for which neither its own setting nor a value an enclosing call passes down gives one |
| as the session's value | `config.sampling.num_samples = 500` | every call for which no other source gives a value |

No `config` section holds `executor` or `executor_options`, so an executor has no block form and no session value: it is named on a `Function`, by its decorator or its `using`.

**The two counts.** Two controls count the work of an approximation, and each has one role:
1. `max_evaluations` is the maximum number of times a call lifted over laws evaluates its function (V.5). The lift enumerates the empirical arguments whose combinations of atoms number at most `max_evaluations` (V.7), and it draws the other arguments jointly: `max_evaluations` draws when it enumerates none, and otherwise `max_evaluations // n` draws for each of the `n` combinations, at least one. Its default is 256.
2. `num_samples` is the number of Monte Carlo draws that a route estimating a summary takes, as the Monte Carlo `mean` does (VI.5), and that a converter takes where it draws, unless the converter's own `num_samples` option is given (VI.10). The call's provenance records it whenever a route of the call draws it. Its default is 1024.

The session's defaults of the two counts are the fields of `config.sampling`, which a call reads when it runs, so a setting governs the functions constructed before it. The environment variables `PROBPIPE_SAMPLING_MAX_EVALUATIONS` and `PROBPIPE_SAMPLING_NUM_SAMPLES` set their initial values.

**The settings.** `config` holds the global settings in four sections: `sampling` for the two counts, `notation` for the depth a rendering shows (II.4), `provenance` for the provenance mode, and `prefect` for the orchestration selector and the Prefect task runner. Assigning a setting sets the session's value. A `using` block sets temporary values, for one section as `config.sampling.using(num_samples=4096)` and for several as `config.using(sampling={...}, notation={...})`. `f.using(...)` returns a reusable configured copy of `f`, while `config.using(...)` is a context manager. A block applies to every read inside it, by the work a call dispatches to worker threads or Prefect workers included, and to no other thread. Blocks nest, an inner value wins, and the enclosing values apply again when a block exits, an error included. An assignment or a `reset` inside a block changes the session's values, which apply once the block exits. The provenance mode is fixed for a run: the outermost `workflow_run` takes the mode in effect when it opens (V.8), and a `config.provenance.using` block that sets another mode inside the open run raises an error that names the run's mode and says to set the mode before the run opens; a block that sets the run's own mode is accepted. Blocks close in the reverse order of opening, and a block that exits out of order raises, as the call context's rules state (V.9). The blocks are part of the call context, which carries them into the work a call sends to other threads and processes (V.9). ProbPipe code inside a user's own `jax.jit` keeps the values in effect while JAX traced it. Each setting is one of two kinds, which its docstring states. A **static** setting can change what a compiled JAX program computes, because ProbPipe code reads it while JAX traces the program, so a JAX program ProbPipe opens is compiled for its value and a call under another value compiles its own (V.9): `sampling.max_evaluations` and `sampling.num_samples`. An **inert** setting changes no computed value, so no compiled JAX program depends on it: every setting of `notation`, `provenance`, and `prefect`. *Requires:* each name a block gives is a section or a setting of its section, and each value is admissible as an assignment admits it. *On failure:* a `TypeError` naming the sections or the settings, or the assignment's error, at the call of `using`.

```python
class SamplingConfig:                         # the session's defaults of the two counts
    @property
    def max_evaluations(self) -> int: ...     # 256 unless PROBPIPE_SAMPLING_MAX_EVALUATIONS sets another
    @property
    def num_samples(self) -> int: ...         # 1024 unless PROBPIPE_SAMPLING_NUM_SAMPLES sets another
    def reset(self) -> None: ...              # the defaults, reading both variables again
    def using(self, **settings: Any) -> AbstractContextManager[SamplingConfig]: ...   # every section has one

class Config:                                 # the global settings, grouped in sections
    @property
    def sampling(self) -> SamplingConfig: ...
    @property
    def notation(self) -> NotationConfig: ...     # II.4
    @property
    def provenance(self) -> ProvenanceConfig: ... # its mode is a ProvenanceMode
    @property
    def prefect(self) -> PrefectConfig: ...       # its workflow_kind is a WorkflowKind
    def using(self, **sections: Mapping[str, Any]) -> AbstractContextManager[Config]: ...
    def reset(self) -> None: ...              # every section's defaults

config: Config                                # the settings in effect, exported as probpipe.config
```

```python
predict.using(max_evaluations=1000)(theta=prior, x=x_obs)
sample.using(raw=True)(posterior, sample_shape=(100,))
mean.using(method="monte_carlo", num_samples=4096)(posterior)
config.sampling.num_samples = 4096
# sample_shape is an argument, and raw and the two counts are controls
```

*Requires:* every control set is one the framework defines, and its value is admissible, for example a positive `num_samples`; the selected method rejects a `method_options` entry it does not read when the call runs, and the executor errors of V.9 are raised by the call too, since only the call selects an executor. *On failure:* a `TypeError` or `ValueError` at the decorator or at the call of `using`, before any call, for an unknown control or an inadmissible value; the `method_options` and executor errors at the call.

### Rationale

A `Function` must wrap an *ordinary* function with no naming restrictions (`C5 – Naming for unambiguous meaning`): a user should never have to rename a `seed` parameter because the framework wanted that word. Holding the controls in a separate namespace removes the collision while keeping the bare decorator and a single call site convenient, and with seeding scoped rather than carried (V.8), ProbPipe claims no `seed` argument.

## V.3 — Binding (step 2)

### Contract

The arguments bind to the wrapped function's signature by Python's own rules (III.3). A parameter the call leaves unbound takes its value from the construction bindings `bind`, and otherwise from its default, so a call argument overrides a construction binding. An entry of `bind` that names no parameter goes to the variadic keyword parameter. Without an `input_spec`, the arguments bind by the signature alone, and the parameter annotations still determine conversion targets (V.4) and lifting (V.5). An argument that is a tracked term becomes a graph **dependency** and any other a plain **input**; the dependencies are the result's provenance parents and the inputs are recorded by parameter name (V.10). *Requires:* every argument binds to a parameter. *On failure:* Python's binding error, naming the parameter, and a `TypeError` for an entry of `bind` that names no parameter when the signature has no variadic keyword parameter.

### Rationale

Binding by the ordinary signature lets the wrapped function stay ordinary (`C1 – Uniform interface to functions, distributions, and values`), and classifying the arguments at binding gives every result its lineage (`C6 – Traceable and reproducible workflows`).

## V.4 — Normalization (step 3)

### Contract

Normalization wraps each bound argument and plans any required conversion, then admits the resulting kind. A planned conversion stands for its declared target until execution (IV.3). The three sub-steps run in order for every argument.

**3a. Wrap.** A raw argument is wrapped into the kind it is by the **kind-directed** table, which also wraps a return (V.10): a tracked term is kept as it is; a raw callable becomes a `Function`; a raw mapping becomes a `Record` (III.5); a raw array becomes a `NumericArray`; a backend distribution passes to 3b; and a value no other kind admits becomes an `Opaque` (III.2), as a list, a tuple, or a set does. A numeric host is recognized through the array-backend registry (II.3), registry first and duck typing second: a registered container supplies its event shape and dtype without being converted, its named dimensions bind the spec's symbolic dimensions (II.1), and its remaining metadata, such as coordinates and attributes, is carried as annotations (II.4). *Requires:* the host constructs as its kind. *On failure:* that kind's construction error, for example a mapping keyed by an empty name.

**3b. Plan conversion.** A distribution-shaped argument whose class differs from the one its parameter names is planned through the converter registry's `check`, which determines what `execute` would do (IV.3). A parameter annotated with a union of several distribution classes or capability protocols names no one target, so an argument of one of them passes as it is, and any other raises `ApplicabilityError`, naming the parameter, the classes, and what arrived. Conversion planning covers backend distributions entering ProbPipe and conversions between ProbPipe representations. The parameter's declared spec supplies a backend object's event spec, and at an unannotated parameter the adapter derives the event spec from the object, with the output name defaulting as III.7 states. The parameter's `conversions` control selects the converter, whether it must be exact, and converter-specific options, as one mapping per parameter: `conversions={"d": {"method": "tfp", "exact_only": True, **options}}`. The registry forwards the options to the converter's `check` and `execute`, as II.7 forwards a call's keywords. A backend object's declaration arrives as the option `event_spec`, and the `convert` operation passes its options to the registry the same way (VI.10). Admission and planning read the target spec and promised capabilities without constructing the converted law. Conversion executes in step 7 and records its source and local fidelity. *Requires:* a converter satisfying the requested constraints. *On failure:* `ResolutionError`; a probe with insufficient target information reports unresolved requirements (V.1).

**3c. Admit.** The wrapped argument's kind, or the planned conversion's target kind, is checked against what the parameter accepts, which is the kind of its declared spec for a `@function`, any kind where the parameter is unannotated, and the kinds its role names for an operation (VI.0). A `Distribution` or a `Batch` over an accepted kind is admitted for lifting (V.5), whereas a `ConditionalDistribution` at a value parameter is refused, since a kernel has no marginal law to lift over. Every kind's term is a `TrackedTerm`, so a bare object, which 3a wrapped as an `Opaque`, passes only a parameter that accepts `Opaque`, whatever methods it carries. *Requires:* the kind is accepted, directly or as the element kind of a lifted argument. *On failure:* `ApplicabilityError`, naming the parameter, what it accepts, and what arrived.

```python
class ApplicabilityError(TypeError): ...  # arguments or declarations violate the call contract
```

### Rationale

Normalization is `B1 – Either presentation in` made mechanical, in the order the information becomes available: a raw host says what kind it is, a conversion declares its target, and only then can the parameter's acceptance be judged. Conversion as a step of its own keeps the array-backend and converter registries visible as registries (`D2 – Generality first`), and reading a container's named dimensions into the declaration is `D5 – Explicit, carried structure` taken from the data rather than declared by hand.

## V.5 — Lifting (step 4)

### Contract

A `Function` compares each admitted argument against the kind its parameter expects and lifts where they differ. A `Distribution` where a value is expected induces a **broadcast**, whose result is the pushforward through `f`. Its generic route applies the function to `max_evaluations` draws (V.2), and exact routes may realize the law directly (V.7). A `Batch` where one element is expected is **swept**: the function is mapped over its elements. Both at once give a **nested** sweep of broadcasts, one broadcast within each element, and neither gives a plain call.

**The trigger.** An input with a declared spec expects the kind its spec names (V.1): an input whose spec is a `DistributionSpec` or a `ConditionalDistributionSpec` consumes the law or kernel passed to it, and an input of a value kind lifts a law. For a parameter with no declared spec, the annotation decides:
- unannotated, or a value type: the parameter expects a value, so a distribution passed there is lifted;
- `Distribution` or a subclass of it, or a distribution capability protocol of III.8: the function consumes the distribution itself, which passes through unlifted;
- `Function` (III.3) or a function capability: the parameter expects a value, as for a value type;
- `Any`: the parameter accepts every kind as it arrives, so a law or a batch passes whole, and an author who wants the lift leaves the parameter unannotated.

Per draw, the function receives the draw as `sample` returns it (VI.3), at the kind the law's event spec names, except for a batch-valued draw. A batch-valued draw is passed to a parameter whose annotation names a batch class, such as `NumericArrayBatch` or `Batch`, as that batch, and to any other parameter in its raw form: the array, or for a batch of records the nested mapping of columns that `raw` returns. A function written for arrays therefore takes a bootstrap replicate unchanged.

A parameter annotated with a class of raw values, such as `pd.Series` or `jax.Array`, receives the raw form of an argument, or of each element or draw that a lift passes to it, when that form is an instance of the class, so a function written for pandas receives the `Series` that a record field holds.

Explicit argument binding reads the draw's term spec, and the keyword names the receiving parameter: in `predict(theta=prior)`, the parameter `theta` receives the draws even when the prior's output is `beta`. Matching by name is the merge's contract (IV.2), and a merge never lifts.

**Grouping and correlation.** The lifted arguments are grouped by **root ancestor**, transitively: coupled marginals of one parent, the same distribution passed twice, and a parent passed alongside its own coupled marginal all fall in one group, and the laws of one group are **coupled** (III.7). Each group contributes one joint draw per repetition, so dependence between its members is preserved through `f`. A coupled marginal lifts by sampling its parent, so its parent must itself sample. Groups with no common ancestor draw independently: the lift samples the **product law**, and, as a corollary, standalone marginals of one joint lift independently while its coupled marginals co-sample. A law that `rename`, `with_label`, `with_dim_names`, or `with_dim_sizes` returns draws together with the law it is made from, since each method returns the same law under a new event spec or a new label (III.7). For example, `f(d, d["x"])` forms one group, and each repetition evaluates `f` on a joint draw and its own projection, while `f(d1, d2)` for unrelated `d1` and `d2` samples the product of their laws. The number of lifted arguments changes only the grouping.

**Alignment.** Swept batches align by level name, as VI.11 states for batched operations, and the classification fixes the regime and the groups without reading any value. *Requires:* the aligned levels have broadcast-compatible shapes. *On failure:* `ApplicabilityError`, naming the level.

### Rationale

This is `C4 – Function lifting` realized in both of its cases: replacing an argument of `f` with a distribution over that argument's type leaves `f` well-defined and returns the pushforward, and replacing one with a batch over that type leaves it equally well-defined and returns a batch: one substitution rule, differing only in whether the multiplicity is a law or a collection. Providing sampling and elementwise sweep as the generic routes keeps the contract general (`D2 – Generality first`): it works for any `f`, any number of lifted arguments, and any distribution that samples, with exact routes registering above those floors, and it leaves the user's function body unchanged. The annotation trigger makes the lifting boundary explicit in the signature, where the author already states intent. Co-sampling by root ancestor makes the lift *correct* rather than merely type-correct: it is the mechanism that keeps coupled marginals coupled, so passing coupled marginals of one law through a function transports their joint law.

## V.6 — Planning (step 5)

### Contract

Planning checks that the available specs conform in one dimension scope, and computes the result's **output spec** as far as they determine it. It reads argument specs, the event or element spec of a lifted argument, declared capabilities and representation metadata, the parameters the result rule reads, such as paths, and controls. It also reads a map argument's declared signature: its parameter annotations, its roles, and its `input_spec`. It never evaluates the body or reads traced array contents. What the declarations leave open becomes a check deferred to return. Resolution chooses the representation that realizes the output spec (V.7).

**The declared output.** `output_spec=` is optional and accepts an `OutputSpec` or a bare term spec, which reads as `OutputSpec.default` under the function's output name (III.3): a base spec is the keyword form under that name, a record is positional, a batch of raw base values takes the name, and a law, a kernel, a function, or any other batch is positional. Without an output spec, the spec of the returned value is read by the same rule, including during distribution broadcasting. `OutputSpec(mean=None)` instead names the output `mean` at the root and leaves its spec unspecified; at return, `with_term_spec` completes it for that call (II.2). Display aliases never determine outputs. Dimensions shared with inputs bind before execution; output-only dimensions bind from the returned term, and all bindings are checked together at return. A route that needs the output spec before the call requires it to be declared. Under compilation, a value-dependent shape that the backend cannot represent is refused.

```python
@function(output_spec=OutputSpec(rate=NumericArraySpec(("obs",), float32, positive)))
def rate(x):
    return jnp.exp(x)
# inference alone would read the support as real; the output spec carries support=positive
# and binds "obs" to the actual output length on each call
```

**The result of a lift.** Planning declares the result of a lifted call from the function's output spec. The route that resolution selects then chooses the representation that realizes it (V.7). The kind of the result depends on the lift's regime (V.5):

1. **A broadcast** returns a law over the function's outputs. When the function returns base values or records, the law's event spec is the function's output spec. When it returns laws, kernels, functions, or batches, a law over them would have no outputs and could not be merged (IV.2), so the lift wraps each result under the function's output name: for `def f(x): return Normal("y", x, 1.0)`, `f(Normal("x", 0.0, 1.0))` is a law over `{f: law}`. The lift builds this law and names its record, as `include_inputs=True` does, while the function's own result is never wrapped, since its output spec states its outputs (II.2). A closed-form rule may return a parametric law, such as a Gaussian, and the sampling lift returns an `EmpiricalDistribution`. An empirical result holds the outputs as its atoms, on one level named by the function's output name (III.3). An output that is itself a batch keeps its own levels within each atom, after the atom level (VII.2). Display aliases do not change that level. A collision with an event level raises `ResultSchemaError`; an explicit `OutputSpec` can supply a distinct output name.
2. **A sweep** returns a batch of the output kind on the swept levels, with one element for each element swept, whose element spec is the function's completed output spec (II.5).
3. **A nested sweep** returns a `DistributionBatch` on the swept levels, whose element at each position is the law that the broadcast at that position returns.

Without an output spec, the result's output spec completes at return from the outputs, by the rule that **The declared output** states. The elements of a sweep must then complete to one output spec, and elements that disagree raise `ResultSchemaError` (V.10). An empty sweep has no output to complete from, so its output spec must already be complete, with every spec stated and every dimension bound.

```python
@function
def growth(interest, years):
    return (1.0 + interest) ** years

interest = Normal("interest", 0.05, 0.02)
horizons = NumericArrayBatch(jnp.array([5.0, 10.0, 20.0]), "horizon", label="years")

growth(interest, 10.0)      # a broadcast: an EmpiricalDistribution whose atoms lie on the level growth
growth(0.05, horizons)      # a sweep: a NumericArrayBatch on the level horizon
growth(interest, horizons)  # a nested sweep: a DistributionBatch on horizon, of laws whose atoms lie on growth
```

**Including the inputs.** With `include_inputs=True`, the sampling lift returns a joint empirical law over inputs and outputs. Each lifted parameter contributes one field holding its complete draw, named by the parameter. A record draw remains nested even when it has one field. The output contributes its outputs (II.2), and a returned law, kernel, function, or batch is wrapped under the output name as a broadcast wraps it. Plain inputs contribute no fields, since provenance records them. A collision between a parameter's name and an output name raises at planning when known and at return otherwise. This control selects only routes that produce the joint of inputs and outputs. Grouping determines co-sampling, and the parameters determine the layout.

```python
# posterior.event_spec.term_spec == RecordSpec(beta=NumericArraySpec(shape=(5,), dtype=float32, support=real))

@function(output_label="prediction", output_spec=OutputSpec(mean=None),
          include_inputs=True, max_evaluations=200)
def predict(theta, x):
    return x @ theta["beta"]      # theta arrives as the Record a posterior draw is

result = predict(theta=posterior, x=X_new)   # X_new: a plain (20, 5) array, not lifted

# result: empirical over 200 atoms, each one joint draw (theta_s, predict(theta_s, X_new)):
#   RecordSpec(
#       theta=RecordSpec(beta=NumericArraySpec(shape=(5,), dtype=float32, support=real)),
#       mean=NumericArraySpec(shape=(20,), dtype=float32, support=real),
#   )
# so the fields are theta/beta and mean; X_new is recorded in provenance, not in the law
```

The result label is `prediction`, independently of its output field `mean` and the function label `predict`. Each atom is one joint draw, so the result couples every sampled input with its own output, as a predictive check or a sensitivity analysis needs.

*Requires:* the arguments conform in one dimension scope, the declared applicability checks hold, and every declaration needed before execution is available; a check may be deferred to return only when the selected route allows it. *On failure:* `ApplicabilityError` for invalid arguments or conflicting declarations; a missing implementation is a resolution failure (V.7), and mathematical undefinedness is distinct (II.7).

### Rationale

Carrying known structure forward and validating deferred output information at its producer realizes `D5 – Explicit, carried structure` without requiring inference of an arbitrary Python function's result. Separating the output kind from its chosen family is `C3 – Computational detail hidden by default, available on demand`: representation selection cannot change the declared mathematical interface.

## V.7 — Resolution (step 6)

### Contract

Resolution selects the route that realizes the call. Candidates are checked on the static information admitted by V.6, including capability guards. Selection follows II.7, including registry priority; operation routes use the convention in VI.0. The controls `method` and `exact_only` select or restrict the route (V.2). A plain call of a plain function has its body as its one candidate, and an operation resolves among its routes (VI.0). The body is exact, so `exact_only=True` admits it, and a name given to `method` matches no route of a plain call and raises `ResolutionError`, as an unregistered name does (II.7). A lifted application, which is the direct call `f(d)` or `f(batch)`, resolves through the **evaluation-rule registry**, which the package exports as `evaluation_rule_registry` beside `converter_registry`. It is a `BinaryDispatchRegistry` keyed on the map's and the operand's types whose methods are **evaluation rules**, each a route of the lifted application. A rule's `check` and `execute` take `(f, operand, /, *, parameter, fixed_args, controls)`: `parameter` names the parameter the operand binds, `fixed_args` holds the map's other arguments, and `controls` holds the resolved controls, kept apart from the map's arguments as VI.1 requires. `pushforward` (VI.1) exposes the registry's rules on laws as an operation, so the direct call `f(d)` of a map that takes draws and `pushforward(d, f)` take the same route. The built-in rule families are:
- **Closed-form rules** return an exact parametric result. For example, `A @ d` for a Gaussian `d` is again Gaussian, with mean `A @ mean(d)` and covariance `A Σ Aᵀ` built lazily through the operator algebra.
- **Change of variables** applies when the map is invertible and carries the Jacobian claim (`is_invertible` and `SupportsLogDetJacobian`, III.3), returning a transformed distribution whose `log_prob` is exact via the log-determinant of the Jacobian.
- **The sampling lift** is the generic rule for an operand that samples, subject to the call's controls and deferred checks: draws from `d` are pushed through the map, returning an empirical distribution over the outputs, with the number of draws set by `max_evaluations` (V.2). It is the route every plain callable takes, and grouped, multi-distribution lifts always take it, as co-sampling requires (V.5).
- **The empirical enumeration** is exact for an empirical operand: the map is applied at every atom, and the result carries the atoms' weights. On the `jax` executor (V.9), every combination of atoms is evaluated in one compiled map, as the sampling lift's draws are. A lift enumerates each empirical argument whose atoms, multiplied by those of the arguments it enumerates already, number at most `max_evaluations`, taking the smallest first, and it samples the others. The lift evaluates the map at most `max_evaluations` times, as V.2 states. The sampling lift draws an empirical argument's atoms with no more variance than independent draws give: equally weighted atoms without replacement, and weighted atoms by stratified resampling, in a shuffled order.
- **The elementwise sweep** is the batch counterpart: the rule at the generic pair for a batch operand. A fused batched implementation, such as an operator's matrix–matrix routine or a single vectorized call over array-backed elements, registers above it.

An **integration rule**, such as quadrature or quasi-Monte Carlo, registers beside these and returns a weighted empirical law whose atoms are its nodes and whose weights are the nodes' weights (VII.2).

A **floor** is a fallback on its stated domain, with the lowest rank there: the sampling lift on samplable distribution operands and the elementwise sweep on supported batch elements. Each declares `is_floor`, which is the first criterion of II.7's selection order, so that rank holds whatever the exactness of the other rules: a registered rule of either exactness takes precedence over a floor where both apply, and selection among the other rules follows II.7. An integration rule or an unscented transform therefore ranks above the empirical lift (II.7). Floors remain subject to fidelity controls, output requirements, and capability guards. The generic lazy linear-pushforward rule (VII.4) represents the image law exactly and preserves derivable moments. The selected route records its representation and local fidelity in provenance. *Requires:* a feasible candidate under the requested controls. *On failure:* `ResolutionError`, naming missing requirements; `check` may report an unresolved selection (V.1).

### Rationale

Dispatching over pairs of map and operand types realizes `C3 – Computational detail hidden by default, available on demand`, since a pair with a known closed form or a fused batched routine gets it automatically, while every other pair still resolves through the floors. Registration grows the exact set without changing call sites (`D2 – Generality first`), and recording the producing route makes the approximation explicit (`D1 – Mathematical fidelity`).

## V.8 — Randomness: workflow scopes and structural keys

### Contract

Randomness is scoped: a **workflow scope** owns every ProbPipe-caused draw inside it, and each such **workflow-owned random event** receives a key derived structurally from the scope's root seed, so no `Function` and no operation holds a seed.

```python
with workflow_run(seed=42):              # a seeded scope: same seed, same structure ⇒ same draws
    result = predict(theta=posterior, x=X_new)

with replay_run(result.provenance):      # re-runs one recorded call on its recorded draws
    replayed = predict(theta=posterior, x=X_new)
```

**Scopes.** `workflow_run(seed)` opens a scope whose root the seed fixes; entering it again with the same stochastic structure reproduces its workflow-owned events. `workflow_run()` with no seed opens an anonymous scope rooted in fresh entropy, and a bare call outside any scope gets its own equivalent ephemeral scope, so unscoped code is fresh rather than repeatable. Scopes nest, a nested scope extending the enclosing structure rather than restarting it, and a scope fixes the run's provenance mode (II.4) at entry. A scope is thread- and task-local: work crosses into another thread, task, or process only as evaluation points (V.9), and entering a copied scope outside an evaluation point raises rather than silently forking the stream.

**Keys.** A fixed stream is obtained by fixing the scope's seed, and an implementer's `_sample` receives the derived key (III.8). A workflow-owned random event claimed inside a JAX transformation that the caller opens, such as `jax.jit`, `jax.grad`, or `jax.vmap`, raises before an event is committed or a key is derived. A compiled call can capture a key drawn during tracing, and an externally mapped call has no workflow-defined identity for each lane. Explicit-key numerical functions can be transformed by the caller, and the engine's own transformations use its planned random events.

**Structural event identity.** A workflow-owned key is a pure function of the scope's root seed and the event's identity, three structural coordinates:
1. the **occurrence path**: the invocation's position in the workflow, extended by nesting and by evaluation points, with repeated or recursive invocations of the same call distinguished by a deterministic logical ordinal;
2. the **stochastic source**: which co-sampling group of V.5 is drawing;
3. the **logical unit**: which broadcast repetition or sweep cell consumes the draw.

Identity follows the workflow's logical structure: ordinals are fixed by program order, no key is drawn twice, and the same call produces the same draws under any executor, thread count, or orchestration. Each draw in the body of an `apply` evaluation (III.3) is its own workflow-owned event, as in a plain call.

**Consequences.** Because keys attach to structure, perturbing an input reuses the same keys, preserving common random numbers: comparisons across nearby inputs and reparameterization gradients keep a low variance. A fresh estimate or an independent stream is obtained by changing the seed. The one cost is that the streams are tied to the program's structure, so restructuring the computation reshuffles them.

**The derivation contract.** The derivation from seed and identity to key is a versioned contract: it is keyed and domain-separated, so distinct events cannot collide, and its version is recorded with every execution and every replay record, so a reproduced run either reproduces under the same contract or refuses. The backend key type is produced through an adapter whose behavior is certified before first use, and a deviation raises. The design fixes the version string, and the hash primitive may be anything that honors the contract.

**Replay.** `replay_run(provenance)` re-executes exactly one recorded `Function` call with its recorded workflow-owned events. Replay validates: the recorded plan, the version of the execution contract (V.9), which includes the derivation version, and the events observed must all match, and an incompatibility raises. The executor is recorded and need not match (V.9). The function matches its recorded definition when its code, its signature, and the types of its declarations do. A rename of the function, of its output label, or of its outputs changes no value, so a renamed function replays the call to the same draws under its new names.

**Caching.** The same structural identity that makes a run reproducible makes it cacheable: a call whose function, operand fingerprints, resolved controls, and random-event identities all match a recorded call must produce the same result, so the result may be served from a cache instead of recomputed. Caching is therefore a workflow-layer option with no semantic effect, off by default. What it can serve is bounded by what provenance records: under the lightweight mode the identity tier is structural, so a cache keyed on it assumes unchanged inputs, while the full mode's content-verifiable fingerprints let a hit be checked rather than assumed. A function's fingerprint records the definition that replay matches, so a cached result is stored as values and takes the names of the function whose call it serves.

### Rationale

Scoped, structural randomness is `C6 – Traceable and reproducible workflows` made mechanical: one seed reproduces a workflow, one provenance record replays a call, and identity-derived keys make the result independent of execution order, so reproducibility survives parallelism and orchestration rather than trading off against them. Key management is `C3 – Computational detail hidden by default, available on demand` applied to randomness: draws are fresh by default and reproducible on demand through the scope's seed. Versioning the derivation makes reproduction exact: a run reproduces under the contract that produced it or refuses.

### Open points

- *The caching interface.* Caching is deferred past the first release; its intended form is a `cache=` parameter of `workflow_run`, keyed on the structural identities the scope holds.

## V.9 — Execution (step 7)

### Contract

Execution first constructs the planned conversions in dependency order and validates their promised declarations, and any check that awaited those declarations is made before its route runs. It then runs the selected route. An exact distributional rule may construct a law directly, and a route that evaluates a callable many times, such as the sampling lift or the elementwise sweep, evaluates it at a set of evaluation points with keys as required (V.8). Numerical work begins here, including converter sampling. What a call computes, its random keys included, is fixed before anything runs. An **executor** then decides only how the evaluations run, such as where they run and how many run at once. Changing the executor therefore changes no result beyond the last bit of rounding.

**Evaluation points.** An **evaluation point** is one evaluation of one callable at one set of arguments, with its random key when the callable draws. The evaluation points of one callable that one request asks for form one **set of evaluation points**, an `EvaluationPoints`. The code that asks for it is its **requester**, such as a lift, a joint, or an inference method. A set of evaluation points comes from one of these sources:

| Source | Its evaluation points |
|---|---|
| a broadcast by the sampling lift (V.5, V.7) | one per draw, `max_evaluations` of them |
| a broadcast by the empirical enumeration (V.7) | one per combination of atoms |
| a sweep over a batch (V.5) | one per element |
| a nested sweep (V.5) | one per element and draw |
| an embedded call under `vmap` (V.13) | one per lane |
| an inference method's batch | one per particle, chain, or proposal |

Each evaluation point has three coordinates:
1. **its point index:** its position within its set of evaluation points;
2. **its evaluation point identity:** its unique name within the call, made of the callable, the ordinal of its set among the sets the call requested, and its point index, as in "draw 17 of the first set of evaluation points of `forward` in this call", which reports and errors use. For an embedded call the ordinal counts the callback's invocations, which JAX may repeat (V.13), so it names an evaluation point within one run and is not a stable name across compilations;
3. **its random key:** derived from its position in the workflow (V.8), so different evaluation points usually have different keys.

A set of evaluation points arrives at the executor registry by one of two entries:
1. **From Python.** A requester that runs in Python, such as a lifted call or a validation loop, builds a set of evaluation points and passes it to the registry.
2. **From inside a compiled JAX program.** An opaque `Function` called on traced values becomes an embedded call (V.13), whose host side builds a set of evaluation points from the batch it receives. A joint's compiled sampler and an inference method's chain runner use the registry this way, with no change to their code.

`evaluate_at_points` is the one function through which a requester evaluates a callable at many arguments. `varying` maps each argument that differs between evaluation points to its values stacked on a leading axis, and `shared` maps each argument that every evaluation point receives unchanged. `origin` names the requester's callable or site for messages and reports, and `output_spec` is the callable's output spec, which each output is checked against. With traced arguments the function calls the callable inline, which makes an opaque `Function` an embedded call. With concrete arguments it builds one set of evaluation points, whose keys come from each evaluation point's workflow scope, and runs it on the executor the call chooses, under the current call context. It returns each evaluation point's output in point order, and raises `EvaluationError` naming every failed evaluation point (**Failures** below). A set of evaluation points describes its members without building each one, so a sweep of a million cells builds no data per evaluation point until a chunk runs. Its `claims` hold the callable's execution claims (V.13), and its `shared_payload()` holds the arguments every evaluation point shares, which an executor ships once per worker.

```python
def evaluate_at_points(fn: Function, /, *, varying: Mapping[str, Any], shared: Mapping[str, Any],
                       origin: str, output_spec: OutputSpec | None) -> tuple[Any, ...]: ...

class EvaluationPoints:
    origin: str                          # the Function or the site, for messages and reports
    claims: ExecutionClaims              # traceable, vectorized, differentiable (V.13)
    output_spec: OutputSpec | None       # the callable's output spec, when a host callback can return its value (V.13)
    def __len__(self) -> int: ...
    def chunks(self, *, chunk_size: int | None = None,
               num_chunks: int | None = None) -> list[EvaluationChunk]: ...
    def shared_payload(self) -> SharedPayload: ...   # the arguments every evaluation point shares
    def shipping_check(self) -> Feasibility: ...     # whether the chunks and the payload pickle; cached per call

class EvaluationChunk:                   # a contiguous slice of evaluation points, which pickles
    start: int
    stop: int
    def run(self, shared_payload: SharedPayload) -> EvaluationOutcomes: ...   # one outcome per evaluation point, in order

@dataclass(frozen=True)
class SharedPayload:                     # what an executor passes to EvaluationChunk.run without reading it
    fn: Callable[..., Any]                # the callable
    shared_arguments: Mapping[str, Any]  # the arguments every evaluation point shares
```

**Running a chunk.** An executor runs a set of evaluation points by its **chunks**, and `EvaluationChunk.run` does the same wherever it runs:
1. **Configuration.** In another process, the pool's worker initializer has applied the parent's JAX settings `jax_enable_x64`, `jax_default_prng_impl`, `jax_threefry_partitionable`, and `jax_default_matmul_precision` before the worker unpickles anything, since a spawned interpreter starts with JAX's defaults.
2. **Context.** It installs the call context carried in the chunk ("The call context" below).
3. **Scope.** For an evaluation point built in Python, it enters the evaluation point's workflow scope (V.8), so a ProbPipe call inside the callable derives its keys from that evaluation point. In another process, the workflow scope and the record of the draws it registers travel inside the chunk, and the draws the worker registers return with the outcomes, so a draw nested in an evaluation point gets the same key in a worker process as under `sequential`, and a replay reproduces it. An evaluation point of an embedded call has no workflow scope (V.13).
4. **Arguments.** It presents each argument as its raw value. A draw inside the callable takes its key from the evaluation point's workflow scope (step 3), and the callable receives no key argument.
5. **Calls.** It calls the callable once per evaluation point, or once for the chunk when the callable claims `vectorized` (V.13). When a vectorized call raises, every evaluation point of the chunk fails with that error.
6. **Outcomes.** It catches each evaluation point's `Exception` as that evaluation point's failure, checks each output against the output spec, and continues with the next evaluation point.

The varying arguments, the keys, and the point indices are private to the set of evaluation points, so an executor cannot change them. The arguments every evaluation point shares travel once per worker for each set of evaluation points, and an executor that runs chunks in the calling process passes them without copying. Shipping a chunk to a worker is a transfer for execution, and a chunk is never stored. An executor that ships chunks calls `shipping_check()` before it runs anything.

**Executors.** An executor is a dispatch method (II.7) of the **executor registry**, `executor_registry`, which the registry catalog lists. Every set of evaluation points has the type `EvaluationPoints`, so the structural pre-filter admits every executor, and `check` and priority decide.

```python
class EvaluationExecutor(UnaryDispatchMethod):
    name: str
    exact = True                         # an executor returns the requested result itself
    priority: int | None = None          # None is opt-in only (II.7)
    description: str                     # one line, for the registry catalog
    def supported_types(self) -> UnarySupportedTypes: ...   # (EvaluationPoints,)
    def check(self, evaluation_points: EvaluationPoints, /, **executor_options: Any) -> Feasibility: ...
    def execute(self, evaluation_points: EvaluationPoints, /, **executor_options: Any) -> EvaluationOutcomes: ...

executor_registry: UnaryDispatchRegistry[EvaluationExecutor]   # the global instance
```

1. **`check`** reads the claims of the evaluation points and the environment, and runs none of them. Its outcome is cached per call, so an MCMC step does not repeat it.
2. **`execute`** returns one outcome per evaluation point, in the order of the evaluation points.
3. **Options** arrive as keywords, and an executor rejects an option it does not read.

The built-in executors:

| Executor | Priority | Runs | Options |
|---|---|---|---|
| `jax` | 20 | traceable callables, compiled as one mapped JAX program | `batch_size`, which maps in chunks with `lax.map` to bound memory |
| `sequential` | 0 | every callable, one evaluation point at a time | `chunk_size`, for a vectorized callable |
| `thread` | opt-in | every callable, on a thread pool | `max_workers`, `chunk_size` |
| `process` | opt-in | sets whose chunks can be shipped, in spawned worker processes | `max_workers`, `chunk_size`, `start_method` |
| `futures` | opt-in | sets whose chunks can be shipped, on any `concurrent.futures.Executor` | `pool` |
| `prefect` | opt-in | sets whose chunks can be shipped, as Prefect tasks | `task_runner`, `retries` |

Every executor also reads four common options: `retries`, `retry_errors`, and `cancel_on_failure` (**Failures** below), and `bisect`, which splits a failing vectorized chunk to find its failing evaluation points. **Automatic selection** is the registry's selection order (II.7) applied to a set of evaluation points: it tries the executors from the highest priority down, skips the opt-in ones, and runs the first that can run the set. A traceable callable, one that JAX can trace and compile, therefore gets `jax`, and an opaque one, which JAX cannot trace, gets `sequential`. `thread` is opt-in because user code may not be thread-safe. ProbPipe ships no cluster executor, since clusters differ in partitions, accounts, and file systems: `futures` runs chunks on any standard pool, such as the one Dask's `client.get_executor()` gives over a Slurm cluster, and a user registers an executor of their own, which `check_executor` checks against the contract below.

```python
def check_executor(executor: EvaluationExecutor) -> None: ...   # in probpipe.testing; raises AssertionError on a broken rule
```

**Choosing an executor.** The controls `executor` and `executor_options` (V.2) name an executor and its options, as `method` and `method_options` name a route and its budgets:

```python
forward.using(executor="thread", executor_options={"max_workers": 8})(prior)
forward.using(executor_options={"chunk_size": 1000})(prior)   # options for the automatically selected executor
condition_on.using(method="pyabc_smcabc", executor="slurm",
                   executor_options={"n_jobs": 50})(model, observed)
```

The choice follows seven rules:
1. **Options travel with their executor.** `executor_options` belong to the executor they were named with, which validates them. On a `Function`'s own call, options given without `executor` go to the executor that automatic selection picks for that call. On an operation, whose work fans out to many callables and so has no one automatically selected executor, `executor_options` given without `executor` raise `ValueError`.
2. **One unit.** `executor` and `executor_options` resolve together, by the precedence of V.2, so options never pair with an executor named in another layer. Options given without an executor are the `Function`'s own choice, so they are never inherited and they block inheritance: inside `condition_on.using(executor="slurm")`, a kernel that calls `epidemic.using(executor_options={"chunk_size": 1000})` runs `epidemic` on `sequential` with chunks of 1000. `using(executor=None)` removes the call's own choice, which restores inheritance, and a call opts out of an inherited executor by naming one, such as `"sequential"`.
3. **What a choice covers.** An executor named on a call applies to every set of evaluation points of an opaque user callable made during the call, at any depth, the call's own callable included when it is opaque. It applies in the calling thread, in the engine's worker threads, and in embedded calls through the call token (V.13). A requester that builds a set of evaluation points of its own, such as simulation-based calibration's inference runs, runs it on the executor named on the requester's call.
4. **Traceable callables compile.** A traceable callable is always compiled by `jax`, whatever executor is named, and receives none of the named options. The `jax` executor's JAX program runs under the call's own context, so the opaque callables it calls take the call's executor. ProbPipe's own numerical methods, such as a family's sampler, run no user callable, so no executor governs them.
5. **The innermost explicit choice wins.** An executor set on a `Function` itself, by its decorator or its own `using`, wins over one inherited from an enclosing call, and it takes only its own options. A cheap opaque statistic declared with `@function(executor="sequential")` therefore stays local while the call sends its simulator to a cluster.
6. **A choice stops at an evaluation point.** A ProbPipe call that an evaluation point makes in Python uses automatic selection unless it names an executor itself, so a cluster job submits no cluster jobs. An embedded call inside a JAX program that the `jax` executor compiles belongs to the call that runs that JAX program, and takes that call's executor.
7. **Refusals.** An executor named for a callable whose evaluation points it cannot run raises `ResolutionError`, naming the executor and the reason, whether the evaluation points arrive from Python or through a call token. It is named for the callable when it is set on the callable's own `Function` or named on the call whose own callable it is. An inherited executor that cannot run a set of evaluation points, for example because a closure does not pickle, leaves the set to automatic selection without the inherited options, and warns once per call. An executor named on a call that runs no set of evaluation points of an opaque callable warns, since the name was probably meant for something else.

Each message follows one template:

```text
unknown executor 'threads'; the registered executors are futures, jax, prefect, process, sequential, thread
executor 'thread' does not read the option 'n_jobs'; its options are bisect, cancel_on_failure, chunk_size,
    max_workers, retries, retry_errors
executor 'process' cannot run the evaluation points of 'forward': its closure cannot be pickled
    (PicklingError: ...); name another executor, or define forward at module level
executor 'slurm', inherited from condition_on, cannot run the evaluation points of 'summary'
    (its closure cannot be pickled); they run on 'sequential' instead
condition_on named executor='thread', but every callable in the call is traceable and was
    compiled by jax, so 'thread' ran nothing
condition_on was given executor_options without an executor; an operation's options belong to
    the executor named with them, so add executor='thread' (or another registered executor)
```

Each is raised or warned by the call, when it resolves its executor, and never by `using`. The first and the third raise `ResolutionError`, as an unknown or infeasible `method` does, the second and the last raise `ValueError`, and the fourth and the fifth are warnings.

For the model `likelihood & prior`, whose likelihood calls an opaque `forward` and whose statistic `summary` declares its own executor, `sample.using(executor="slurm", executor_options={"n_jobs": 50})(model, 200)` runs as follows:

```text
sample's call: executor slurm, n_jobs 50
├─ the prior's draws: the family's own compiled sampler, which runs no user callable
├─ the likelihood, mapped over 200 givens: one compiled JAX program, which receives sample's call token
│    └─ forward(theta): opaque, so an embedded call; its host side finds slurm and runs 200 evaluation points
└─ summary(y), declared with @function(executor="sequential"): its own choice wins, so sequential
```

**The call context.** The **call context** is the one value that carries the per-call state that a call's code reads wherever it runs: the call's resolved controls, its executor choice, the active `config.using` blocks (V.2), the controls a derived route passes down (VI.0), and the records of what the call ran. Every piece of per-call state that crosses a thread, a process, or a compiled JAX program has one home in it. It follows five rules:
1. **Opening.** Every call of a `Function`, through `__call__`, `apply`, or `check`, opens a child of the current call context when it starts, operations included. The child keeps its parent's `config` blocks and only the passed `exact_only`, so the counts a derived operation passes apply to the calls its formula makes directly. The child resolves its own controls by the precedence of V.2 and its executor choice by the rules above. A `config.using` block and a derived operation's scope derive a changed copy of the current call context, and open no call.
2. **Evaluation points.** An evaluation point opens no call. It runs under its call's context without the executor choice, so a call made inside it uses automatic selection (rule 6). Inside an evaluation point, under every executor, a setting that no block sets reads the session's value in effect when the outermost call opened. A session assignment made while a call runs therefore changes none of the call's evaluation points, and every executor gives a nested call the same settings.
3. **Transport.** Within a thread, a context variable holds the current call context. Work sent to a worker thread, a Prefect task, or a worker process carries its call's context as data and installs it before it runs. The context pickles, with its records of what ran arriving empty and coming back with the outcomes. The `prefect` section of `config` is never shipped to another process, so a call there runs without Prefect orchestration, and any other setting that does not pickle travels as a placeholder that raises `UnshippableSettingError` only when the worker reads it. A JAX program that ProbPipe opens receives a call token in place of the context (V.13).
4. **Closing.** A call closes its context when it ends, an error included, and restores the context that was current when it opened. Calls and `config.using` blocks close in the reverse order of opening. A block that a generator holds open can break that order in two ways, and each restores the context and then raises `RuntimeError`: a block closed while a call opened inside it still runs keeps applying until the call ends, and a call or a block that ends while a block opened inside it is still open closes that block with it. Either way no closed block applies afterwards. A term built inside a call keeps no reference to the call's context, so later work on the term runs under the settings of the later call.
5. **Reporting.** A call's provenance reports what the call ran itself: each set of evaluation points, its executor and options, the number of evaluation points and of runs, and the failed or dropped ones. When an evaluation point succeeds, what it noted merges into its call's report, under every executor, and the notes of a failed or retried evaluation point are dropped. A call merges into its parent's report only when the parent is a derived operation's call, so a derived operation also reports what its formula's calls ran.

```python
class UnshippableSettingError(RuntimeError): ...   # a setting that could not be sent to a worker process, read there
```

**Failures.** An executor reports every evaluation point's outcome, and the requester decides what a failure means. No executor drops an evaluation point or replaces its value.

```python
@dataclass(frozen=True)
class EvaluationFailure:
    point_index: int                                      # its position within its evaluation points
    point_identity: tuple[str, int, int]                  # the callable, the ordinal of its set, and its point index
    kind: Literal["error", "invalid", "lost", "timeout"]  # user code raised; its output failed the check;
                                                          # the infrastructure lost it; it ran out of time
    error_type: str                                       # the exception's qualified name, which survives shipping
    message: str
    error: Exception | None                               # None when the exception could not be shipped back

@dataclass(frozen=True)
class EvaluationOutcomes:
    start: int                                            # the point index of the first evaluation point covered
    outputs: tuple[Any, ...]                              # in the order of the evaluation points; None where one failed
    failures: Mapping[int, EvaluationFailure]             # by point index
    @classmethod
    def lost(cls, chunk: EvaluationChunk, reason: str) -> EvaluationOutcomes: ...   # every evaluation point of the chunk lost
    @classmethod
    def merge(cls, chunk_outcomes: Iterable[EvaluationOutcomes]) -> EvaluationOutcomes: ...   # sorted by start

class EvaluationError(RuntimeError):
    failures: tuple[EvaluationFailure, ...]
    outputs: tuple[Any, ...]                              # the outputs that succeeded, kept for diagnosis
```

A failure is handled by seven rules:
1. **A failure makes the call raise.** The unit is the set of evaluation points: its executor runs every evaluation point and keeps the outputs that succeeded, and the call then raises `EvaluationError` naming every failed evaluation point. The error is chained to the first failure's exception when that exception is available in the calling process, so a bug in the user's own function keeps the traceback that points into it. Its message names the callable, the number of failed evaluation points out of the total, and up to five of them in the requester's words, such as "draw" or "grid point": `evaluation of 'forward' failed at 1 of 200 evaluation points: draw 17: CalledProcessError: solver exited with status 3`. `EvaluationError` is a `RuntimeError`, so code that catches the user's own exception type around a lifted call catches `EvaluationError` instead and reads its `failures`. The common executor option `cancel_on_failure=True` cancels the evaluation points not yet started instead of running them.
2. **Inside a compiled JAX program, a failure stops the JAX program.** The callback's set of evaluation points completes, and the callback then records its failures on its call's context, which it finds by the call token, and raises. The JAX program stops at once, and the engine raises `EvaluationError` with the recorded failures, chained to whatever JAX raised. The engine decides by the recorded failures, since JAX delivers a host exception under more than one type. The host side records every other exception it raises the same way, such as a `ResolutionError` for a refusing executor named for the callable or an executor's own error, and the engine raises that exception, chained to JAX's, in preference to `EvaluationError`, so the user receives it with its own type.
3. **Only a set built in Python can drop failed evaluation points.** A requester that builds a set of evaluation points in Python, such as ABC over an opaque simulator, drops failed evaluation points when the user opts in with `method_options={"on_failure": "drop"}`. Dropping changes the target to the region where the callable succeeds, so `check` reports it and provenance records every dropped evaluation point. A failure of an opaque callable embedded in a compiled JAX program stops that JAX program (rule 2), so `check` refuses `on_failure="drop"` for it and names the callable.
4. **Dropping covers user-code errors only.** A `lost` or `timeout` evaluation point is retried by the executor or raises, so a flaky worker never changes the target.
5. **Retries.** The option `retries` retries `lost` and `timeout` evaluation points with the same key, so a retry computes the same value, and `retry_errors=True` also retries evaluation points whose code raised.
6. **Interrupts.** A chunk catches `Exception` and never `KeyboardInterrupt`. An interrupt cancels the evaluation points not yet started, stops a compiled JAX program through its callback, and is raised again as `KeyboardInterrupt`.
7. **Outputs are checked per evaluation point.** `EvaluationChunk.run` checks each output against the output spec, its tree structure, shape, and dtype, before the outcome leaves the chunk. A mismatch is an `invalid` failure of that evaluation point alone. A host `float64` output under a `float32` output spec is cast, and the cast warns once.

**The executor contract.** An executor that follows these rules cannot change a result:
1. **Order:** `execute` returns one outcome per evaluation point, in the order of the evaluation points, whatever order they finished in.
2. **Keys:** an executor leaves each evaluation point's key unchanged and creates no randomness of its own.
3. **Retries:** an executor may run an evaluation point again after an infrastructure failure, and the fixed key and identity make the retry compute the same value.
4. **Blocking:** `execute` returns when every evaluation point has an outcome, and it may submit and poll internally.
5. **Interrupts:** an interrupt cancels the evaluation points that have not started, and `execute` raises it again.
6. **Outputs:** an executor returns the outcomes its chunks produced, whose outputs `EvaluationChunk.run` has checked, and return then validates the assembled result (V.10).
7. **Nesting:** a chunk that calls ProbPipe uses automatic selection inside (rule 6).
8. **Configuration:** a chunk run in another process runs under the parent's JAX settings and its call's context.
9. **Conformance:** `check_executor` checks rules 1 to 8 on fixed sets of evaluation points that include failures, a vectorized callable, and a nested call that reads the call's settings, under a parent with 64-bit floats and a non-default `jax_threefry_partitionable`. Every built-in executor passes it.

**Replay.** The versioned execution contract that replay checks (V.8) versions what changes draws: the version of the key derivation, the key format, and the version of each sampling algorithm. An executor changes no draw, so replay records the executor and its options with the environment, as it records the JAX version, and replays under any executor that passes `check_executor`. Replay refuses a recipe whose execution-contract version differs from the current one, with a message that says to record it again.

**Orchestration: recording apart from running.** Orchestration records a call and never changes how it runs. The control `workflow_kind` records a call as a Prefect task or flow, for lineage and scheduling, and it records whichever executor runs the call's evaluation points. Running work as Prefect tasks, on the task runner the user names, is the `prefect` executor. Recording never changes the result, and evaluation points carry their keys across every thread, task, and process boundary (V.8), so orchestrated and distributed runs draw from the same structural stream as local ones.

**What `check` reports.** `check` reports how a call will run in the `execution` field of its `CallReport` (V.1), with one `EvaluationReport` per callable whose evaluation points the call evaluates:

```python
@dataclass(frozen=True)
class EvaluationReport:
    callable_name: str
    num_evaluation_points: int | None              # None when the requester is adaptive or the callable is embedded
    traceable: bool
    traceability_source: Literal["claimed", "probed"]   # V.13
    executor: MethodInfo                           # the executor chosen
    rejected: tuple[MethodInfo, ...]               # each higher-priority executor that could not run, with its reason
    failure_rule: Literal["raise", "drop"]

@dataclass(frozen=True)
class ExecutionReport:
    evaluations: tuple[EvaluationReport, ...]      # one call may send one callable to jax and another to a cluster
```

For a set of evaluation points built in Python, `check` reports the callable the requester will run. For a route that opens a JAX program, `check` traces that JAX program, as the call does, without running it, and reports each embedded `Function` it finds, with `num_evaluation_points=None` and the executor its host side would choose. The `CallReport`'s representation shows the execution report only when a callable is opaque or an executor other than `jax` was chosen, so the report of an all-JAX model shows no execution part. Provenance records the executor and its options, the number of evaluation points, the failed or dropped evaluation points, and whether each JAX program compiled or fell back. A fallback that `check` cannot foresee warns when it happens. A traceable `Function` that names an executor and is compiled inline warns, since its executor then runs nothing.

**Compiled JAX programs.** A JAX program that ProbPipe opens to run user or route code, such as a joint's mapping of a kernel over the draws of its inputs, is kept in its owner's **program store**, which holds one `jax.jit` object per body and compiles it once for each input structure and each value of the call's static settings. The static settings are the call's resolved counts, the controls a derived route passes, and the values in effect of the static `config` settings (V.2). The traced body receives them as a static argument, so they are part of JAX's own cache key too, and a call under another `num_samples` compiles its own JAX program rather than reusing one traced under the first. A JAX program is traced before it first runs: a body that does not trace runs eagerly, and the owner records that once, while an error raised while a traced JAX program compiles or runs propagates, with no eager retry. What the traced code notes on the running call, such as the draws of a Monte Carlo estimate, is kept with the store's entry and applied to the running call after each run, cold or warm, so a warm call reports the provenance a cold call reports. A note may depend only on the entry's cache key, and a trace that notes otherwise raises.

**Frameworks other than JAX.** A callable written with NumPy, PyTorch, or an external program is opaque, and everything above applies to it. It runs on concrete values through the same API under every executor but `jax`, and inside a compiled JAX program as an embedded call (V.13), where a derivative it supplies, such as a VJP from PyTorch's autograd, carries gradients across.

*Requires:* the route completes, and every evaluation point has an outcome. *On failure:* `EvaluationError` naming each failed evaluation point, `ResolutionError` for an unknown executor or one named for a callable it cannot run, `ValueError` for an option the executor does not read or for options without an executor on an operation, or the route's own error.

### Rationale

Fixing what a call computes, and every random key, before an executor runs anything (`P1 – Meaning is settled before execution`) makes the executor a computational detail with no effect on the result (`P3 – Results do not depend on the executor`), which is `C3 – Computational detail hidden by default, available on demand`: a call runs on a default executor, and a user who needs a cluster names one on the call without rewriting the model. Running every set of evaluation points through one registry, with the priorities, opt-in entries, and selection by name of II.7, is `D2 – Generality first`: an executor a user registers serves lifting, joints, and inference without a change to any of them. Keys that are data at every boundary make the results identical across executors up to rounding, so reproducibility survives parallelism and orchestration (`C6 – Traceable and reproducible workflows`). Reporting each failure per evaluation point, and letting only the requester drop one, keeps the target of a call fixed unless the user opts in, which is `D1 – Mathematical fidelity`. One call context, with one home for each piece of per-call state, is `D6 – Single source of truth`: every thread, process, and callback that runs a call's code reads the same settings.

### Open points

- *Failures inside compiled MCMC.* Rejecting a proposal whose evaluation point failed needs a success flag per lane that crosses the boundary next to each output, so a compiled sampler over a simulator that sometimes fails stops at the first failure.
- *Streaming results.* `execute` blocks until every evaluation point has an outcome. An adaptive requester, such as ABC stopping early, could use outcomes as they complete, in an extension that keeps the order of the evaluation points in the result.

## V.10 — Return (step 8)

### Contract

Return binds deferred output dimensions, infers undeclared output structure where permitted, and validates the produced terms. It assembles lifted results according to their declarations and the selected routes. The kind-directed wrap reads the completed return term spec, which is `OutputSpec.term_spec` where declared: an array named `beta` stays an array, while a one-field record stays a record (II.2). A tracked return keeps its kind under the call's fresh identity.

The result carries the function application expression unless an explicit `output_label` supplies an alias, for a lift the outer law or batch included. A law or a kernel that a call on values returns takes the function's label, or the alias, and displays its own signature (II.4). An operation's result carries the label II.4 derives from its inputs, and a result's outputs come from its output spec, whatever its label. The law of a broadcast displays as the function applied to draws of its inputs, as `f(beta ~ model; y)` (II.4). The result receives provenance recording the function, dependencies, named inputs, resolved controls, and the selected route's fidelity and target (II.7). A route's result that carries a record of its own, as an inference method's posterior names the method and its target, is a parent too, under the expression the result boundary gives the result, so the parent of a posterior of `model` reads `model` and keeps the method's record. With `using(raw=True)` the result is returned detached (II.4), without constructing the outer identity. A declared support is a value-dependent check, which a traced value cannot take: a mapped call checks it on the stacked results once the map returns, and a trace the caller opens, such as `jax.jit` around `apply`, skips it. *Requires:* the result conforms to the completed output spec. *On failure:* `ResultKindError` for a wrong returned kind, or `ResultSchemaError` for incompatible structure, dimensions, support, or outputs other than the declared ones. Both are return-contract defects, not caller admission failures.

```python
class ResultKindError(TypeError): ...
class ResultSchemaError(ValueError): ...
```

### Rationale

Return is `B3 – Tracked forms out by default` at the one boundary every call crosses, and attaching provenance there makes every result record how it was produced (`C6 – Traceable and reproducible workflows`).

## V.11 — Differentiability claims

### Contract

The decorator's `differentiable` argument declares which inputs gradients propagate through: a non-empty `NumericSpec` covering exactly those values, which are array-native with no gradient-breaking operations. Omitting the argument makes no claim, and there is no shorthand for claiming every input, so a claim never widens as the function changes. The constructed `Function` then carries `SupportsDifferentiation` with that schema, read through `is_differentiable` wherever gradients are required. Like the declared sides, the declaration is fixed at construction: a claim, not a control. An opaque callable claims it only with a derivative it supplies (V.13).

```python
@runtime_checkable
class SupportsDifferentiation(Protocol):
    @property
    def differentiable_template(self) -> NumericSpec: ...
    # exactly the values gradients propagate through: a sub-schema of the numeric
    # inputs (maps) or of the numeric event schema (distributions)

def is_differentiable(x: Any, values: NamedTree | None = None) -> bool: ...
# True when every value named in `values` lies in x's differentiable template;
# with no `values`, when the template covers every numeric value. False when x
# does not declare the capability.
```

The capability is cross-kind: the `differentiable_template` is a sub-schema of the numeric inputs for a map and of the numeric event schema for a distribution, so a linear operator claims its whole input schema and a distribution family claims the event values its sampling reparameterizes. The differentiability claim composes: a coupled marginal restricts its parent's schema to its path, a joint assembles its factors' schemas under their field names, and a value is differentiable through a chain of steps exactly when every step claims it. An operation that needs gradients checks `is_differentiable` for the values it differentiates and names the first step that fails, before a backend trace runs; the executor is a separate control, so the `jax` executor compiles a traceable call whether or not the object differentiates.

### Rationale

Differentiability as a declared claim is `D3 – Capability-based operations` applied to gradients: support is promised by the object and checked before a backend trace, never inferred from a value being numeric.

### Open points

- *Differentiability of sampling-based routes.* Whether a Monte Carlo fallback differentiates through its sampler's reparameterization is unsettled. So is the eventual `grad` operation the claims feed, with registered routes: a custom gradient method where an object supplies one, the automatic-differentiation route gated by the declared template, and finite differences as the approximate fallback. Both are left to a dedicated pass.

## V.12 — Constraint reparameterization

### Contract

Many inference algorithms, for example gradient-based optimization and Hamiltonian Monte Carlo, operate on an unconstrained space ℝᵈ, so a constrained support must be reparameterized. The **constraint-to-bijector factory** maps a `Constraint`, which is the support a `NumericArraySpec` carries, to a *bijector*: a `Function` that takes `ℝⁿ` onto that support and claims the inverse and Jacobian capabilities of III.3. `bijector_for(constraint)` returns the canonical one, and `register_bijector` plugs in a factory for a constraint type or a specific instance, instance registrations taking precedence.

An input checks the claims it needs at construction and raises `ResolutionError` when unavailable (II.7): the bijector of a transformed distribution requires both `SupportsInverse` and `SupportsLogDetJacobian` (III.3), the link of a GLM likelihood `is_invertible` alone.

```python
def bijector_for(constraint: Constraint) -> Function: ...   # the canonical map ℝⁿ → support(constraint)
def register_bijector(key: type[Constraint] | Constraint,
                      factory: Callable[[Constraint], Function], *,
                      description: str = "") -> None: ...   # the description is the catalog's
```

The factory keys on constraint instances and types rather than dispatching on argument types alone, so it is not a dispatch registry; it still satisfies `SupportsRegistryCataloging` and appears in the registry catalog alongside the dispatch registries.

### Rationale

A bijector for every constraint lets inference run in an unconstrained space while a model stays stated in its natural, constrained one, which is `C3 – Computational detail hidden by default, available on demand`: the reparameterization an algorithm needs is supplied for it rather than written into the model. Keeping the factory open through `register_bijector` is `D2 – Generality first`: a new constrained support becomes inference-ready by registering its reparameterization, without changing the distributions that use it.

### Open points

- *Round-trip fidelity.* The forward map (bijector to support) and this inverse map (support to bijector) are not strict inverses for every constraint, so a reparameterized support can drift to a coarser one. Whether to unify the two is unsettled.

## V.13 — Execution claims and the boundary

### Contract

A `Function` states at construction how it may be run, and ProbPipe reads these statements instead of guessing. Code that JAX can trace is compiled, vectorized, and differentiated. Code that JAX cannot trace runs on concrete values through the same API and gives the same results, more slowly, so a user never rewrites working NumPy code to use ProbPipe. The two kinds of code join at one boundary, the embedded call.

**Traceable and opaque.** A callable is **traceable** for its input shapes when `jax.jit` and `jax.vmap` both succeed on it. A traceable callable is pure and acts on array inputs only through `jax.numpy` and `lax`: it never converts an input to Python or NumPy, never branches in Python on an input's value, and its output shapes depend only on its input shapes. Any other callable is **opaque**, such as one that calls a solver through `subprocess`, a SciPy routine, or a PyTorch network. Traceable says nothing about gradients, which the differentiability claim states (V.11).

**Execution claims.** An **execution claim** is a statement a `Function` makes when it is built about how it may be run. It is a declaration, fixed for the `Function`'s life, and never a control. There are three:
1. **`traceable`** takes three values. `True` states that the callable is traceable. `False` states that it is opaque, so ProbPipe never traces or probes it. `None` states nothing and is the default.
2. **`vectorized=True`** states that the callable accepts every array argument with an extra leading axis of evaluation points (V.9), and returns its outputs stacked on that axis, with output row `i` depending only on input row `i`. An executor may then evaluate a chunk of evaluation points in one call.
3. **`differentiable`** is the claim of V.11, a `NumericRecordSpec` whose field names are the callable's input names, checked against its `input_spec`. JAX differentiates a traceable callable itself, and an opaque callable is differentiable only through a derivative it supplies, as `vjp=`, `jacobian=`, or `jvp=`:

| Derivative | Its signature | What it serves |
|---|---|---|
| `vjp` | `vjp(*args, cotangent) -> tuple`, one cotangent per input the claim names, in signature order | reverse mode, which gradient methods use |
| `jacobian` | `jacobian(*args) -> tuple`, one Jacobian matrix per input the claim names | both modes, since ProbPipe applies it as `J @ t` in JAX |
| `jvp` | `jvp(args, tangents) -> Array`, returning the output's tangent | forward mode only, since JAX cannot transpose a host computation |

A `vectorized` callable's derivative receives stacked rows too. An opaque callable that claims `differentiable` without a derivative raises `TypeError` when it is built, and a traceable callable may supply one to replace JAX's own rule, for example for numerical stability.

```python
@function(traceable=True)                      # compiled, vectorized, and differentiated by JAX
def ricker(r, K, n0): ...

@function(traceable=False,                     # never traced or probed
          output_spec=NumericArraySpec((40,), "float32"))
def forward(theta): ...

@function(traceable=False, vectorized=True,    # one call takes a stack of evaluation points
          output_spec=NumericArraySpec((40,), "float32"))
def forward_batch(thetas): ...                 # thetas has shape (n, 2)

@function(traceable=False,                     # differentiable through the derivative it supplies
          output_spec=NumericArraySpec((40,), "float32"),
          differentiable=NumericRecordSpec(theta=NumericArraySpec((2,))),
          vjp=forward_adjoint)
def forward_with_adjoint(theta): ...
```

The claims a `Function` holds are one value, which V.9's sets of evaluation points carry:

```python
@dataclass(frozen=True)
class ExecutionClaims:
    traceable: bool | None = None                # True, False, or None for no statement
    vectorized: bool = False
    differentiable: NumericSpec | None = None    # the claim of V.11
```

**The probe.** A **probe** is a trace made only to learn whether a callable traces, with `jax.make_jaxpr` on stand-ins for its inputs. A probe runs the callable's Python up to the first operation that fails on a tracer, so a callable with side effects claims `traceable=False`. ProbPipe keeps one **probe result** per `Function` and **input structure**, which is the tree structure that ProbPipe's canonical flatten gives for a call's arguments, with their shapes and dtypes, and the values of its non-numeric hashable arguments:

| Claim | First call or first `check` for a new input structure | Later |
|---|---|---|
| `True` | no probe; a trace that fails raises, because the claim is false | no probe |
| `False` | no probe | no probe |
| `None` | one probe, whose outcome is stored | the stored outcome |

Four rules keep the probe result sound:
1. **Python numbers.** A Python `float` argument is a weakly typed array, so a sweep over `n0=10.0, 11.0, ...` reuses one probe result and one JAX program. An integer scalar, a Python `int` or a NumPy integer, is treated the same way when the probe succeeds with it as data. When the probe fails because the callable uses an `int` as a loop count or a shape, it runs once more with every integer scalar static, and the probe result records which form traced. The two attempts are one probe, so a callable is still probed at most once per input structure. A static `int`, a string, and any other hashable non-number are part of the input structure, and a lookup tries the input structure with `int`s as data first.
2. **NumPy's global generator.** The probe compares the global states of NumPy's and Python's random generators before and after, and puts them back. A callable that drew from either is recorded as opaque, with the warning `'simulate' draws from NumPy's or Python's global random generator, so ProbPipe does not compile it with JAX. Draw from a JAX key instead, or pass traceable=False`. A NumPy `Generator` that the callable creates or holds is not caught, so such a callable claims `traceable=False`.
3. **Copies share the probe result.** The probe result belongs to the wrapped callable and its construction bindings, so the copies `using` makes share it. A law built from functions inside a kernel, which the kernel builds anew for every given value, shares the probe results of its functions with every closure of the same code whose captured values have the same structure, so one probe serves every application of the kernel. The probe traces such a closure with its captured arrays as data, so its result holds for any captured arrays of the same shapes and dtypes. A closure that fails so because it reads a captured value, as `jnp.zeros(int(n))` does, is probed with its captured values as they are, and shares that result only with closures that capture equal values. A captured hashable value counts by its value, and a captured `Function` by its identity, since its claims are fixed. Any other captured value, a plain callable included, prevents sharing, so the closure is probed for itself.
4. **`check` probes too.** `f.check(...)` runs the probe the call would run and stores its outcome, so `check` and the call choose the same executor and method. `check` may therefore run the callable's Python once.

A false `traceable=True` claim is caught where the `Function` is called, even inside a larger trace, a probe of another callable included. ProbPipe catches JAX's tracing errors and raises its own `TypeError`, chained to JAX's:

```text
'forward' claims traceable=True, but tracing it failed: it converts a traced value to NumPy
(TracerArrayConversionError). Remove the claim, or claim traceable=False
```

A false claim that traces without an error, such as a callable that reads a global, cannot be caught this way.

**Three states.** The claims and the output spec put every `Function` in one of three states:

| State | Condition | Called inside a JAX program | Called in Python on a batch |
|---|---|---|---|
| **traceable** | claimed, or found by a probe | traced inline, as part of that JAX program | the `jax` executor compiles a `vmap` of it |
| **embeddable** | opaque, with an output spec that describes a value a host callback can return | an embedded call | any executor but `jax` |
| **host-only** | opaque, without such an output spec | refused, with an error naming the rule its output spec fails | any executor but `jax` |

A host callback must return a value whose structure, shapes, and dtypes are known before the JAX program runs. An output spec describes such a value when its type is set, and that type is a `NumericArraySpec`, or a `RecordSpec` whose every leaf is a `NumericArraySpec`, each with its dtype set and each dimension an integer or a name that the `Function`'s declared `input_spec` binds from the shapes of the traced arguments (II.1). In short, the value is a numeric array or a record of them, with dtypes, and shapes bound by the declared inputs. For example:
1. `NumericArraySpec((40,), "float32")` describes such a value;
2. `NumericArraySpec(("n",), "float32")` describes one when the `Function` declares an `input_spec` whose shapes name `n`, and does not without one, since then no input binds `n`;
3. a missing `output_spec`, and an output spec whose type is unspecified, such as `OutputSpec(y=None)`, do not;
4. a leaf without a dtype, such as `NumericArraySpec((40,))`, and an `OpaqueSpec` do not;
5. a type that holds the type of a law, a kernel, a function, or a batch anywhere does not, so a callable that returns one is host-only.

The rule reads the output spec's type rather than its outputs, since a function that returns a law offers no outputs, and "every output has a shape" would hold of it vacuously.

Claims combine as the differentiability claim does (V.11):
1. a view of a law at some of its paths keeps only the claims of the parts it views;
2. a joint, a kernel, or a law built from functions traces when each of its JAX parts traces and each opaque `Function` inside it is embeddable;
3. a value is differentiable through a chain of steps when every step on its path is, and an opaque step is differentiable only through a supplied derivative.

A law or a kernel has a **traceability**: traceable, embeddable, or host-only, the state of its density, or of its sampler when it has no density, since that is what an inference method runs (VI.6). It derives it from its parts the first time it is read and keeps it:

| Law or kernel | Its traceability |
|---|---|
| a family | traceable |
| a joint | the least of its factors' |
| a view or a rename | its parent's |
| a law built from functions | its density's probe result, or its sampler's when it has no density |
| the kernel of a function | the traceability of the law its function returned when the kernel was built, read by the construction's abstract evaluation (IV.4) |
| a conditional at given values | its joint's |
| any other law or kernel | its density probed once at a stand-in of one value, each dimension not yet bound of size one, or its sampler at one key when it has no density |

The least of several is the least permissive in the order traceable, embeddable, host-only.

**The embedded call.** When a `Function` that is not traceable is called and at least one of its arguments is a JAX tracer, ProbPipe makes an **embedded call**, which runs the opaque code from inside the compiled JAX program:
1. **Requirement.** The `Function` must be embeddable, so that the JAX program can be compiled before the callable runs. A host-only `Function` raises while it is traced, and the error names the rule its output spec fails, as in `'forward' cannot run inside a JAX program: its output spec sets no dtype; give NumericArraySpec((40,), "float32")`.
2. **Staging.** ProbPipe stages a host callback, `jax.pure_callback`, with the declared output shape and `vmap_method="expand_dims"`, inside one `jax.custom_jvp` or `jax.custom_vjp` wrapper (**Gradients across the boundary** below). Under a `vmap` over N lanes, the host therefore receives all N lanes in one call. Nested `vmap`s add several leading axes, which the host flattens into evaluation points by each argument leaf's rank per evaluation point, read from the input's term spec. The ranks are bound into the staged callback rather than stored on the `Function`, since one `Function` can be staged at several ranks.
3. **The host side.** When the JAX program runs, the host side finds the call by the call token it receives (**Call tokens** below), builds one set of evaluation points from the batch, serves from the call's records the evaluation points it has already run, and runs the rest on the executor that V.9's rules choose. The evaluation points it runs keep their original point indices, so failures and reports name the evaluation points the requester asked for. It takes the executor set on the `Function` itself first, then a choice that a call opened within the same trace bound into the staged callback, and then the executor choice of the call it finds. It checks each output (V.9) and returns the outputs stacked. Its table of active calls is safe for several threads, because under `pmap` the host side runs on several of JAX's own threads at once.
4. **Failures.** A failed evaluation point stops the JAX program at once, and the call raises `EvaluationError` naming it (V.9).

JAX may skip a callback whose result is unused, or run one twice for one value, and both are correct because a callable a user passes to ProbPipe must be pure. In a batched `while_loop` or `lax.cond`, JAX also evaluates lanes whose results it discards, such as the finished chains of a NUTS tree. Their evaluation points run too, and a failure there still stops the call.

Abstract evaluation, with `jax.eval_shape` or `jax.make_jaxpr`, records an embedded call without running it, so building a kernel whose function calls an embeddable `Function` runs nothing (IV.4). A kernel whose function calls opaque code that no `Function` wraps still fails, and the error names the fix:

```text
the function of 'likelihood' converts a traced given to NumPy, so it cannot be traced.
Wrap the code that is not JAX in @function(traceable=False, output_spec=...)
```

**Call tokens.** A value read while JAX traces becomes a constant of the compiled JAX program, and under `pmap` a callback runs on JAX's own threads, where the caller's context variables are not visible. An embedded call therefore finds its call by a **call token**: a `numpy.uint32` that names one open call context in a thread-safe table of active calls, where token 0 names no call. Every JAX program that ProbPipe opens to run user code or route code takes the call token as its first argument, an inference method's chain runner included, and an embedded call passes the traced token to its callback. One cached JAX program therefore serves every call, and each call's token arrives at the host. A family's own sampler runs no user code and takes no token. Tokens follow seven rules:
1. **Registration.** A call registers its token only before it runs a **JAX program with embedded calls**, which is a JAX program ProbPipe opens whose trace staged at least one embedded call, found by a search of the JAX program's jaxpr before it first runs. Every other JAX program receives token 0, and a call that holds no token keeps JAX's asynchronous dispatch.
2. **Waiting.** A call that holds a token waits for the outputs of its JAX programs with embedded calls before it releases the token, so no callback finds a released token.
3. **Inside a trace.** While ProbPipe traces a JAX program it opens, an inner call's body runs inline in the outer trace, under the inner call's own context, rather than in a JAX program of its own. The inner call's executor choice is therefore staged into the outer JAX program, its embedded calls pass the outer traced token, and only the outermost JAX program is stored. A JAX staging trace is active while JAX records a JAX program rather than running it: under `jax.jit`, `jax.make_jaxpr`, or `jax.eval_shape`, and inside the body of a primitive that stages its body even outside `jax.jit`, such as `lax.map`, `lax.scan`, `lax.cond`, or `jax.checkpoint`; an eager `jax.vmap` or `jax.grad` is not one (rule 5). A call opened while any JAX staging trace is active registers nothing and runs inline.
4. **A captured token.** A `jax.jit` placed between a JAX program and an embedded call, such as a user's jitted helper around an opaque `Function`, captures the token as a constant when JAX traced it while ProbPipe was not tracing. ProbPipe refuses it with an error that names the inner `jax.jit` and says to remove it or to call the opaque code through a ProbPipe call.
5. **Under an eager transformation.** Under a `jax.vmap` or `jax.grad` called outside any `jax.jit`, a callback runs at once on the calling thread, so the embedded call registers its call's token then and passes it as a concrete value.
6. **A JAX program the user opens.** A JAX program that the user's own `jax.jit` builds passes token 0, so its embedded calls run under no call's settings: the session's values, automatic selection, and a failure that raises at once from the callback rather than becoming `NaN`. A `config.using` block around the user's `jax.jit` therefore does not apply on the host side of its embedded calls, which read the session's values. To choose an executor for opaque code, a user runs it through a ProbPipe call, which opens its own JAX programs and passes its token. An embedded call staged inside a staging trace the user opened, such as a `lax.map` called during a call that names an executor, warns once per call, naming the executor.
7. **A backward pass after its call closed.** Under an eager `jax.grad` or `jax.vjp`, a supplied backward rule runs after the call that made the forward pass has closed, so it runs under no call's settings, and the forward pass never keeps its token. Inside a JAX program that ProbPipe opens, the backward rule is traced with the forward pass and passes the traced token like any embedded call:

```python
loss = lambda theta: jnp.sum(forward_with_adjoint.apply(theta) ** 2)
jax.grad(loss)(theta0)       # eager: forward_adjoint runs after the call closed, under automatic selection

with workflow_run(seed=0):
    posterior = condition_on.using(method="blackjax_nuts", executor="slurm")(model, observed)
    # NUTS differentiates inside the JAX program it opens, so forward_adjoint runs on slurm
```

**What an embedded evaluation point may not do.** JAX decides how many times an embedded call runs, so a random event inside one would have no stable identity:
1. **It is untracked.** An embedded call is a raw evaluation, like `apply` (III.3). The call that owns the token reports what it ran, such as its executor, its evaluation points, and its failures, and a call opened while tracing reports none of it.
2. **It draws no workflow randomness.** A ProbPipe call inside an embedded evaluation point that draws workflow-owned randomness raises (V.8), on whatever thread runs it, since the refusal travels with the evaluation point's call context.
3. **Orchestration runs inline.** A call orchestrated by Prefect that opens while a JAX staging trace is active runs inline as part of the traced JAX program, with no recording.

**Gradients across the boundary.** JAX invokes the rule of an embedded call's derivative wrapper only when a gradient flows into the call:

| What is differentiated | What happens |
|---|---|
| a value whose path avoids the opaque call | works, and the rule is never invoked |
| a value through an opaque call that supplies no derivative | raises, as in `gradients cannot pass through 'forward', which is not traceable and supplies no derivative` |
| a value through an opaque call with `vjp` | works in reverse mode, under `vmap` too |
| a value through an opaque call with `jacobian` | works in both modes |
| a value through an opaque call with only `jvp` | works in forward mode, and reverse mode raises |
| an input outside the callable's `differentiable` claim, through a call with a derivative | raises, naming the input |

A gradient in `σ` of a likelihood `Normal(forward(θ), σ)` therefore needs no derivative of `forward`. ProbPipe registers the rule with symbolic zeros, so it sees which inputs carry a gradient and raises for one outside the claim, where JAX would otherwise return a zero gradient silently. A gradient method checks the claims before it traces (V.11), so `check` reports the second and the last rows before any work runs.

*Requires:* every opaque `Function` called inside a JAX program is embeddable, and every gradient that crosses an opaque call has a supplied derivative for each input it flows through. *On failure:* an error that names the missing declaration, the false claim, or the callable or input that a gradient cannot cross, and `TypeError` for a differentiability claim without a derivative.

### Rationale

Claims stated at construction and read by every requester are `D3 – Capability-based operations` applied to execution: whether a callable compiles, vectorizes, or differentiates is promised by the callable and checked before a trace runs, never guessed from a failure. Probing a callable that states nothing, once per input structure and in `check` as in the call, keeps that promise for code written without claims, and `check` and the call always agree (`C3 – Computational detail hidden by default, available on demand`). Running opaque code on raw values, through the same API, is `B2 – Representations only inside` extended to code JAX cannot compile, so a user's working NumPy simulator enters a compiled model unchanged (`C1 – Uniform interface to functions, distributions, and values`). The embedded call is the one place opaque code and JAX code meet (`P7 – Opaque code joins compiled code at one boundary`), so a compiled sampler or a kernel uses a user's executor without being rewritten (`D2 – Generality first`). Passing the call token as an argument, never as a captured constant or a thread's own state, lets one compiled JAX program serve every call and still find each call's settings (`C6 – Traceable and reproducible workflows`). The output spec that makes a callable embeddable is `D5 – Explicit, carried structure`: the JAX program is compiled from the declared shape before the callable runs.

### Open points

- *Accelerators.* On a GPU or a TPU, callbacks run on runtime threads, and calling back into JAX from a callback may block, so an embedded evaluation point that uses JAX could hang. The design was checked on CPU only.
- *The structure of a supplied derivative.* What a `jvp` returns and what a `jacobian`'s matrices are, for a record-valued output or an input of more than one dimension, is unsettled, and so is when a callable that claims `differentiable` without a derivative raises if it states `traceable=None` and its probe finds it opaque.
- *The probe and batching.* The probe traces with `jax.make_jaxpr`, so it cannot catch a callable that traces but has no batching rule, such as a callback without a `vmap_method`.
- *Reverse mode from a JVP.* A callable with only a `jvp` and a low-dimensional input could serve `grad` through one JVP per input, run as one set of evaluation points.
- *An adjoint solver's state.* `vjp(*args, cotangent)` makes an adjoint solver recompute its forward state; letting `vjp` receive state saved by the forward call would halve its cost for some solvers.
- *Deterministic fields of a joint.* A joint that stores `forward(θ)` as a deterministic field of its event would let posterior draws carry it, let reuse skip the need for bitwise-equal arguments, and let a host-only callable take part in a joint. It changes what a joint's event holds, so it belongs with Part IV.
