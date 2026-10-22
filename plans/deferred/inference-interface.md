# Plan: from `condition_on` to an inference backend

> **Status: deferred until after the workshop; not approved.**
> Written on 2026-10-21, during step E6 (#618), against `dev/overhaul` as of #618, and deferred by the maintainer the same day so as to add no work before the workshop of 2026-10-27.
> Review it, and answer its questions, before implementing anything from it. Before then, re-check it against the code: E4 rewrites the inference runners this plan also changes.
> `plans/deferred.md` lists it under "The inference interface and vectors".
> Read [`vector-interface.md`](vector-interface.md) alongside: this plan uses its vector laws and does not define any layout of its own. Once approved, its decisions move into `design/` (VI.6).

**Decisions this plan asks for, all open** (gathered in § 9):

1. The order of the work relative to E4 (§ 8), now settled by the deferral.
2. The new helper `initial_vectors` and the new `make_posterior(..., coordinates=)` (§ 5.2, § 7).
3. That seeded draws change once, to stop reusing the run's key (§ 6.1).

---

## 1. The journey in one picture

A user writes a model, conditions it on data, and gets a posterior:

```python
likelihood = glm_likelihood("y", GaussianFamily(), X=X)          # a kernel: y given beta and dispersion
prior = MultivariateNormal("beta", jnp.zeros(2), 5 * jnp.eye(2)) * HalfNormal("dispersion", 1.0)
model = likelihood * prior                                         # a merged law over y, dispersion, beta

with workflow_run(seed=0):
    posterior = condition_on.using(method="blackjax_nuts")(model, {"y": y})
```

Between `condition_on` and the bytes BlackJAX receives, there are six layers. Each exists for one reason.

```
 (1) the model             a merged law: likelihood * prior
        │  condition_on(model, {"y": y})
 (2) the exact stage       what is the conditional, mathematically?
        │                  → an exact law (returned as is), or the unnormalized conditional
 (3) the normalization     which algorithm normalizes it, and with which random key?
        │  stage           → the inference-method registry selects a method by its check
 (4) the target            what may the method read?  → an InferenceTarget
        │
 (5) the method's adapter  turn the target into what this backend takes, run it,
        │                  and turn its output into chains of vectors
 (6) the result            make_posterior → an EmpiricalDistribution on (chain, draw)
```

**Layers 1 to 4 and 6 are ProbPipe's, and the same for every method. Layer 5 is the only backend-specific part**, and it is where most of today's duplicated code sits. The vector-interface plan supplies the one set of tools every adapter uses in layer 5.

The traced example, as it runs today:
- the model's outputs are `y, dispersion, beta`;
- the target's event is `{dispersion: (), beta: (2,)}`, with supports `non_negative` and `real`;
- the vector size is 3;
- `factors()` gives the prior over `dispersion, beta`, the GLM likelihood, and the observed `y`, of shape `(30,)`.

## 2. Layers 1 to 3: from a model to a selected method

### 2.1 The model

`condition_on` needs nothing special of the model. It is any law or kernel.
- Merged models (`likelihood * prior`, written `&` after S3) matter because their **factors** let some methods use structure, such as a Gaussian prior or a likelihood that can be subsampled (§ 3.3).
- Programs (a Stan or PyMC model) matter because a backend can run them natively.

### 2.2 The exact stage: what is the conditional?

`condition_on` first asks a purely mathematical question: *what law results from fixing these paths?* It answers it without approximation where it can (VI.6):
- **Binding an input** applies a kernel.
- **Slicing** keeps the factors downstream of the fixed fields, as fixing `beta` in `p(y | beta) p(beta)` leaves the kernel `p(y | beta)`.
- **Exact conditioning** uses a law's own `_condition_on`, as a `MultivariateNormal` has.

Where none applies, the stage forms the **unnormalized conditional**: the law of the unconditioned fields whose unnormalized log-density is the joint's at the given values. It is the posterior known up to a constant, `p(beta, dispersion | y) ∝ p(y | beta, dispersion) p(beta) p(dispersion)`.

**Why a separate stage.** It keeps *what* is computed apart from *how* it is approximated.
- An exact answer never runs an algorithm.
- `method="unnormalized"` returns the unnormalized conditional for a user who runs their own algorithm.
- Provenance records the two stages separately (C6).

### 2.3 The normalization stage: which algorithm, with which key?

A result that is not normalized goes to the **inference-method registry**.

**Selection is by capability, not by type.**
- Every method lists `Distribution` in `supported_types`.
- Its `check` reads what it needs from the target:
  - an unnormalized density and its traceability (E2) for a gradient method;
  - a Gaussian prior for elliptical slice;
  - a program in `target.joint` for Stan.
- Automatic selection runs the highest-priority method whose `check` passes. `exact_only` keeps only exact methods (II.7).

**The key.** The registry derives one random key per run of a method from the enclosing `workflow_run` and passes it as `execute(target, *, key)`, so `workflow_run(seed=...)` reproduces the run. The method draws all its randomness from that key and nothing else (#554 § 8; #618).

**Three cases the registry also handles,** so that methods never see them:
- **Renamed laws:** `_held_target` normalizes the law a rename holds, then renames the result.
- **Kernels left with inputs:** `_PerValueNormalization` normalizes each law the kernel yields, once its inputs are bound.
- **Bare unnormalized laws:** the registry hands them over as the target of conditioning on nothing (§ 3.1).

None of this changes in this plan; § 4 states it precisely.

## 3. Layer 4: the target, which is the one thing a method receives

### 3.1 What a target is

**An `InferenceTarget` is a law: a model's unobserved quantities at its observed values.**
- It is a `Distribution`, with the event spec, label, and capabilities of one.
- It has an unnormalized density exactly when the model has one.

It also carries the model's **structure**, for methods that use more than a density:

| Member | What it is | Who needs it, and why |
|---|---|---|
| the law itself: `event_spec`, `label`, capabilities | the parameters' law, up to a constant | every method: the event fixes what is inferred, and the density drives MCMC, VI, and importance sampling |
| `joint` | the model | program methods (Stan, PyMC, nutpie) run the model natively, so they read the program from it |
| `given` | the observed values | program methods pass them as the program's data; a simulation method compares simulations with them |
| `factors()` → `ModelFactors(prior, likelihood, observed)` | the model split into prior and likelihood, when it is a merge | methods that use structure: elliptical slice (Gaussian prior), SG-MCMC (subsampled likelihood), empirical reweighting (prior atoms), ABC (simulate the likelihood), BayesFlow (train on prior and simulator) |

**Three forms, which `given` tells apart:**
1. a model at values of its fields (`given` is a `Record`);
2. a model at data it does not declare as fields, such as a program's data block (`given` is that data);
3. a law that is itself unnormalized, which is the target of conditioning it on nothing (`given` is an empty `Record`, and `joint` is the law).

Conditioning on nothing is the identity, so that target records no `condition_on`, and a posterior from it names the law.

**What a target is not.**
- **It has no vector interface.** Every vector view of it is one of the vector-interface plan's laws: `vector_law(target)` and `unconstrained_law(target)`.
- **It is not specific to densities.** A simulation method reads `factors()` and never touches the density.

### 3.2 Why a protocol, and why one implementation

`InferenceTarget` is a protocol, so a method types its argument by what it reads, not by a private class. `_UnnormalizedConditional` is its one implementation.

The alternative is to make that class public. The protocol keeps the class free to change, which mattered in this very step when the bare-law case was folded in. Recommendation: keep the protocol.

### 3.3 `ModelFactors` and one invariant it must keep

Methods that use `factors()` lay out vectors with the **prior's** spec, while `make_posterior` reads the **target's** event. The two agree today in every model tried, but nothing enforces it.

This plan makes it a contract: **`factors().prior.event_spec.outputs` equals `target.event_spec.outputs`, in order.** `factors()` checks it and returns `None` for a model that breaks it, so the method's `check` refuses that model instead of mislabelling draws.

## 4. Layer 5: the method's adapter, by family of method

Every adapter does three things:
1. **Read** the target into its backend's input.
2. **Run** the backend with the run's key.
3. **Return** chains for `make_posterior`.

The vector-interface plan provides the reading and the return for every method that works on vectors. What remains is backend glue.

### 4.1 Density methods on ℝᵈ: NUTS, HMC, random-walk Metropolis, TFP NUTS, SG-MCMC, VI

```python
u = unconstrained_law(target)                               # ℝᵈ′, density includes log |det J|
log_density = log_prob_function(u, normalized=False)        # raw: jit, vmap, grad
init_key, run_key = jax.random.split(key)
initial = initial_vectors(target, init_key, num_chains)     # (num_chains, d′), see § 5.2
chains = run_backend(log_density, initial, run_key, ...)    # BlackJAX / TFP, on flat vectors
return make_posterior(target, chains, method=..., coordinates="unconstrained", annotations=...)
```

- **One density route** for every one of these methods, whatever the target's form (keyed, data, or a bare law). This fixes random-walk Metropolis, which today drops a program's data from the density (reproduced), and the `(2, 2)`-array crashes.
- **Unconstrained coordinates for every one,** SG-MCMC included. Today SG-MCMC runs in constrained space and can leave the support.
- **The backend receives flat vectors,** as today. The pytree question is settled in the vector-interface plan, § 6.
- **SG-MCMC's gradient** sums:
  - the prior's unconstrained density (`log_prob_function(unconstrained_law(factors.prior))`);
  - the minibatched likelihood at `from_vector(spec, b(z))`.

### 4.2 Methods that use the model's structure

- **Elliptical slice** needs a Gaussian prior on the vector.
  - **Check:** the prior is Gaussian on the vector, which is the case when `vector_law(factors.prior)` is a `MultivariateNormal`, `Normal`, or merge of them.
  - **Prior moments:** `mean(vector_law(prior))` and `cov(vector_law(prior))`. These are in the canonical layout by construction; today they are assembled in factor order.
  - **Likelihood:** `log_prob_function(factors.likelihood, normalized=False)` at `given=from_vector(spec, v)` and the observed value.
- **Empirical reweighting** works on the prior's atoms directly. It vmaps the likelihood's `log_prob_function` over the atom batch's columns, so no flattening is needed (today it flattens and immediately unflattens).
- **ABC (pyabc)** draws parameters with `sampler(factors.prior)`, simulates with `sampler(factors.likelihood)`, and moves in `unconstrained_law(factors.prior)`.
  - Its parameters are named by their layout position (`beta[0]`, `dispersion`) instead of `p0`, `p1`.
  - Observed and simulated values are laid out with `to_vector` against their spec, which fixes the key-order bug of `flat_vector`.
  - This needs `sampler` (vector-interface decision 2). Until it ships, the built-in pyabc method keeps its private draws.
- **BayesFlow** trains on draws from `sampler(prior)` and `sampler(likelihood)`, moving each leaf with `vector_bijector`.

### 4.3 Program methods: Stan (CmdStan), PyMC, nutpie

The model is a program, and the backend runs it with its own transforms, initial values, and seeding.
- **Read:** the program from `target.joint` and its data from `target.given`.
- **Seed:** `integer_seed(key)`.
- **Return:** the backend's draws by variable name, concatenated in the target's order, and passed to `make_posterior(..., parameter_shapes=...)`, since the program fixes array sizes from its data.

No ProbPipe density or vector law is involved, and none should be.

### 4.4 What each backend receives

| Method | Backend entry | It receives | Built from |
|---|---|---|---|
| `blackjax_nuts`, `blackjax_hmc` | `blackjax.window_adaptation` + `nuts`/`dynamic_hmc`, `lax.scan` | flat `(d′,)` positions; `log_density` | `unconstrained_law(target)`, `initial_vectors` |
| `blackjax_rwmh` | `blackjax.normal_random_walk`, Welford adaptation | the same; `sigma` is `(d′, d′)` | the same |
| `tfp_nuts` | `tfp.mcmc.NoUTurnSampler` with step-size adaptation | a flat `(d′,)` state | the same |
| `blackjax_sgld`, `blackjax_sghmc` | `blackjax.sgld`/`sghmc` | a flat `(d′,)` position; a gradient estimator | `unconstrained_law(prior)` + the minibatched likelihood |
| `blackjax_elliptical_slice` | `blackjax.elliptical_slice` | a flat `(d,)` position; `mean (d,)`, `cov (d, d)`; the log-likelihood | `vector_law(prior)` moments + `log_prob_function(likelihood)` |
| `empirical_reweighting` | none (exact) | the prior's atom batch | `log_prob_function(likelihood)` vmapped |
| `pyabc_smcabc` | `pyabc.ABCSMC` | named scalars in unconstrained coordinates; a simulator | `unconstrained_law(prior)`, `sampler` |
| `cmdstan_nuts`, `pymc_nuts`, `pymc_advi`, `nutpie_nuts` | the program's own sampler | the program and its data | `target.joint`, `target.given` |

## 5. Layer 6: the result, and two helpers

### 5.1 `make_posterior`

`make_posterior(target, chains, *, method, coordinates="vector", ...)` turns a method's chains into the posterior:
1. maps the chains back to values (the vector-interface plan, § 5);
2. stacks them on the levels `("chain", "draw")`;
3. declares the target's event;
4. stores the method's diagnostics as ArviZ groups;
5. records provenance with the target as parent, or the law itself for the target of conditioning on nothing.

`coordinates="unconstrained"` is new in this plan, so no method writes the way back itself.

### 5.2 `initial_vectors`

**The problem today:**
- every chain of a run starts from **one** shared state, which weakens R-hat's check across chains;
- that state is drawn with the **same key** the chains then split, so the key is reused;
- the fallback box covers the whole vector, where VI.6 says it covers only the fields no sampling factor draws.

`initial_vectors(target, key, num_chains, *, coordinates="unconstrained")` gives one starting vector per chain, by VI.6's rule:
1. each chain's key draws from the factors of the target's joint that sample, restricted to the unconstrained fields;
2. any field no sampling factor draws starts from `Uniform(-2, 2)` in unconstrained coordinates, as Stan does;
3. a vector-coordinate start maps through `vector_bijector`.

## 6. Cross-cutting rules

### 6.1 Keys

- **One run, one key.** A method splits its key once into `init_key` and `run_key`, and never uses a key twice.
- **Consequence:** this changes every seeded draw of the BlackJAX, TFP, random-walk Metropolis, elliptical slice, and SG-MCMC methods once.
- **Breaking:** it is a silent change, so the CHANGELOG shows the same call before and after (decision 3).

### 6.2 Feasibility without work

A method's `check` reads capabilities, traceability (E2), and specs, and never draws or compiles. Today the NUTS and TFP checks build the target and sample the prior, and pyabc's compiles and samples. With the vector laws, `check` constructs `unconstrained_law(target)`, which is cheap and draws nothing.

### 6.3 Opaque code (E4)

E4 makes every runner a module-level JAX program that takes the call token, so an opaque `Function` inside the model runs on the call's executor.

The adapters of § 4.1 and § 4.2 are exactly those runners. So this plan's rewrite of the runners and E4's must not happen twice (§ 8).

## 7. The API, with docstrings

### 7.1 `InferenceTarget` (as in #618)

```python
@runtime_checkable
class InferenceTarget(Protocol):
    """The law an inference method normalizes: a model's parameters at its observed values.

    ``condition_on`` hands every inference method one target. The target is the
    law of the quantities left unobserved, called the parameters below, and its
    unnormalized log-density at a value of the parameters is the model's joint
    log-density at that value and the observed values. A method returns a
    normalized law over the target's event, usually through
    :func:`~probpipe.inference.make_posterior`.

    The observed values take one of three forms, which :attr:`given` holds:

    1. **Values of the model's fields.** ``condition_on(model, {"y": y})``
       gives a target whose ``given`` is a :class:`~probpipe.Record` keyed by
       the fields it fixes, and whose event is the model's other fields.
    2. **Data the model takes but does not declare as fields**, such as the
       data block of a program. ``given`` is that data, as the model takes it.
    3. **Nothing.** A law that is itself unnormalized, such as one built from a
       log-density alone, is the target of conditioning on nothing. ``given``
       is an empty ``Record``, and :attr:`joint` is that law.

    A target is a law: it has the event spec, the label, and the capabilities
    of a :class:`~probpipe.Distribution`. It has an unnormalized density, read
    through :func:`~probpipe.unnormalized_log_prob`, exactly when the model has
    one, so a method states the capabilities it needs in its ``check``: a
    method that evaluates a density requires
    :class:`~probpipe.SupportsUnnormalizedLogProb`, and one that simulates
    reads :meth:`factors`. A method that works on vectors reads the target
    through :func:`~probpipe.vector_law` or
    :func:`~probpipe.unconstrained_law`.

    A target is immutable: a method reads it and never changes it.
    """

    @property
    def label(self) -> str:
        """The target's label, which is the model's."""

    @property
    def event_spec(self) -> OutputSpec:
        """The output spec of one value of the parameters: the event of the law a method returns."""

    @property
    def joint(self) -> Distribution:
        """The model: the joint law of the parameters and the observed quantities."""

    @property
    def given(self) -> Any:
        """The observed values: a ``Record`` of fields, the data a model takes, or an empty ``Record``."""

    def factors(self) -> ModelFactors | None:
        """The model's prior, likelihood, and observed value, or ``None`` when it has none.

        A model has them when it is a merge of factors, such as
        ``likelihood & prior``, conditioned on every field the likelihood
        factors produce and on no field of the prior factors, and when the
        prior's outputs are the target's, in the target's order.

        Returns
        -------
        ModelFactors or None
            The prior, the likelihood, and the observed value of the
            likelihood's event; ``None`` for any other model.
        """
```

The only change from #618 is the last clause of `factors()` (§ 3.3), and the pointer to the vector laws.

### 7.2 `ModelFactors` (as in #618)

```python
class ModelFactors(NamedTuple):
    """The prior and the likelihood of a model, and the observed value of the likelihood's event.

    :meth:`InferenceTarget.factors` returns it for a model merged from factors,
    such as ``likelihood & prior``, conditioned on the fields its likelihood
    produces. The prior's outputs are the target's, in the target's order.

    Attributes
    ----------
    prior : Distribution
        The law of the parameters: the merge of the factors that produce no
        observed field.
    likelihood : Distribution or ConditionalDistribution
        The law of the observed fields given the parameters: the merge of the
        factors that produce them, a kernel whose inputs are fields of the
        prior, or a law when it reads none.
    observed : Any
        The observed value of the likelihood's event.
    """
```

### 7.3 `InferenceMethod.execute` (as in #618)

```python
@abstractmethod
def execute(self, target: InferenceTarget, /, *, key: PRNGKey, **options: Any) -> Distribution:
    """Normalize *target*: return a normalized law over its event.

    ``condition_on`` calls it once per run of the method, after ``check``
    admitted the target, with a random key it derives for that run from the
    enclosing ``workflow_run``, so ``workflow_run(seed=...)`` reproduces the
    run and two runs in one workflow draw differently. A method draws all of
    its randomness from *key*, splitting it as it needs and never using a key
    twice, and none from a global random generator; a method that draws
    nothing ignores it. Code that calls a method directly passes a key of its
    own choice.

    Parameters
    ----------
    target : InferenceTarget
        The law to normalize: a model's parameters at its observed values.
    key : PRNGKey
        The run's random key.
    **options : Any
        The call's ``method_options``, which the method validates with
        :meth:`_check_options`.

    Returns
    -------
    Distribution
        A normalized law over ``target.event_spec``, usually built with
        :func:`~probpipe.inference.make_posterior`.
    """
```

### 7.4 `make_posterior` (#618, plus `coordinates`)

```python
def make_posterior(
    target: InferenceTarget,
    chains: list[Array],
    *,
    method: str,
    coordinates: Literal["vector", "unconstrained"] = "vector",
    annotations: DataTree | None = None,
    parameter_shapes: Mapping[str, tuple[int, ...]] | None = None,
    weights: ArrayLike | Weights | None = None,
    label: str = "posterior",
    **meta: Any,
) -> EmpiricalDistribution:
    """The posterior an inference method returns: the empirical law of its draws of the target's parameters.

    The result is an :class:`~probpipe.EmpiricalDistribution` labeled *label*,
    whose atoms are the draws on the levels ``chain`` and ``draw``, labeled by
    the target's outputs, as ``beta`` or ``(K, r, phi)``. Its event spec is the
    target's, so a record of parameters keeps its fields' supports, shapes, and
    nested groups. Each chain holds one vector per draw, in the coordinates
    *coordinates* names, which the result reads back as draws of the
    parameters. Its provenance names the method, with the target as its
    parent, or the law itself for the target of conditioning a law on nothing.
    Its annotations are a ``DataTree`` whose root attribute ``method`` is
    *method*, and which stores the method's diagnostics, sample statistics,
    and warmup draws as ArviZ-compatible groups under ``arviz/``.

    Parameters
    ----------
    target : InferenceTarget
        The target the method ran on.
    chains : list of Array
        Per-chain draws, each of shape ``(num_draws, size)``; the chains have
        equal lengths.
    method : str
        The inference method's name, by which the ``method`` control selects it.
    coordinates : {"vector", "unconstrained"}
        ``"vector"``: row ``i`` is ``to_vector()`` of draw ``i``, of size
        ``vector_size``. ``"unconstrained"``: row ``i`` is the point ``z`` of
        :func:`~probpipe.unconstrained_law`, of its size, and the result maps it
        through :func:`~probpipe.vector_bijector`.
    annotations : DataTree or None
        The method's diagnostics, sample statistics, and warmup draws, as
        ArviZ groups of a ``DataTree``, such as ``sample_stats``. A
        ``posterior`` or ``warmup`` group whose one variable ``params`` holds
        draws in *coordinates* is stored with one variable per leaf of the
        result, named by its path with ``.`` between the parts.
    parameter_shapes : Mapping of str to tuple of int, or None
        The shape of each parameter as the run fixed it, for a target whose
        event leaves shapes for a program to fix when it runs. It names every
        output of the target and nothing else, and needs ``coordinates="vector"``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Per-draw importance weights across all chains in chain order.
    label : str
        The result's label.
    **meta
        Further metadata recorded in the provenance.

    Returns
    -------
    EmpiricalDistribution
        The posterior, with its atoms on the levels ``chain`` and ``draw``.

    Raises
    ------
    ValueError
        If *chains* is empty or the chains differ in length, a draw's width is
        not the size *coordinates* needs, or *parameter_shapes* misses an output
        of the target or names one it does not have.
    TypeError
        If the target's event has a leaf that is not a numeric array, or
        *parameter_shapes* is given for a target whose event is not a record.
    """
```

### 7.5 `initial_vectors` (new)

```python
def initial_vectors(target: InferenceTarget, key: PRNGKey, num_chains: int, *,
                    coordinates: Literal["vector", "unconstrained"] = "unconstrained") -> Array:
    """One starting vector per chain for a method that runs on *target*'s vectors.

    Each chain starts from its own key. A field that a factor of the target's
    joint draws, and that the target leaves unconditioned, starts at that
    factor's draw; any other field starts uniformly in ``(-2, 2)`` in
    unconstrained coordinates, as Stan does. Every start lies in the target's
    support.

    Parameters
    ----------
    target : InferenceTarget
        The target the chains run on.
    key : PRNGKey
        The key of the starts, split once per chain.
    num_chains : int
        The number of chains.
    coordinates : {"vector", "unconstrained"}
        Whether the starts are vectors (``to_vector`` of a value) or points of
        :func:`~probpipe.unconstrained_law`.

    Returns
    -------
    Array
        The starts, of shape ``(num_chains, size)`` in *coordinates*.

    Raises
    ------
    TypeError
        If the target's event is not numeric.
    ValueError
        If *num_chains* is not a positive integer, or the event has a symbolic
        dimension.
    """
```

It lives in `probpipe.inference`, beside `make_posterior`.

### 7.6 The registry (as in #618, no change)

- `inference_method_registry` holds `InferenceMethod`s only, and refuses anything else.
- Its `execute(target_or_model, data=None, *, method=None, exact_only=False, **options)` forms the target, selects a method by `check`, derives the run's key, and calls `execute(target, key=key, **options)`.
- `check` takes the same arguments, receives no key, and runs nothing.

## 8. Steps and order

| Step | Content | Breaks user code? | Builds on |
|---|---|---|---|
| **#618** (E6) | the target, the key, `make_posterior` | yes (signatures) | — |
| **V1** | vector maps: `to_vector_map`, `vector_bijector`, `NumericArrayBatch` vectors | no (additive) | — |
| **V2** | `vector_law`, `unconstrained_law` | no | V1 |
| **V3** | `log_prob_function`, and `sampler` if the vector plan's decision 2 puts it there | no | V2 |
| **P2** | every built-in method's adapter onto V1 to V3 (§ 4); `initial_vectors`; `make_posterior(coordinates=)`; key split and per-chain starts; `factors()`'s order check; the reproduced bugs fixed; the private helpers deleted (vector-interface plan § 7) | **yes, silent**: seeded draws change once (§ 6.1) | #618, V3, and E4 as merged |

**Order against E4.** Deferring this plan settles its first question: E4 goes ahead before the workshop on today's runners, and P2 rewrites those runners after it, keeping E4's call token. P2 must then read E4's merged runners first.

## 9. Questions for the maintainer

1. **Order with E4.** Settled by the deferral: E4 first, then V1 → V2 → V3 → P2 after the workshop (§ 8).
2. **`initial_vectors` and `make_posterior(coordinates=)` as public names.** Recommended: yes, since every outside MCMC method needs both.
3. **The one-time change of seeded draws** to stop reusing the key and to give each chain its own start. Recommended: yes, in P2, listed as a silent change.
4. **Keep `InferenceTarget` a protocol** with one private implementation. Recommended: yes (§ 3.2).
5. **The order invariant of `factors()`** (§ 3.3). Recommended: enforce it, refusing a model that breaks it.

## Appendix: bugs this work fixes (all reproduced except where noted)

1. **Random-walk Metropolis drops a program's data.** Its `check` passes on a model whose density reads `(params, data)`, but the density receives a bare vector.
2. **Random-walk Metropolis crashes on a scalar law that cannot sample.** The fallback initial state is 0-d.
3. **A law over a `(2, 2)` array crashes `blackjax_rwmh` and `tfp_nuts`.** On a non-family constrained array law, every density method crashes, while `check` reports it feasible.
4. **The run's key is reused** for the initial state and for the chains, in BlackJAX NUTS/HMC, random-walk Metropolis, elliptical slice, and SG-MCMC.
5. **Every chain starts from one state.**
6. **SG-MCMC runs in constrained space** and can leave the support (from reading the code).
7. **Elliptical slice's prior moments are in factor order,** which only coincides with the canonical order.
8. **`flat_vector` lays a mapping out in the mapping's key order** (affects pyabc's comparison; inferred).
9. **A record event in the keyword form is refused by the layout helpers.**
10. **Checks draw and compile** (NUTS, TFP, pyabc).
