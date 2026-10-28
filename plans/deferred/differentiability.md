# Plan: one contract for derivatives

> **Status: proposed; not approved.**
> Written on 2026-10-28 against `main` at `a45cd20f`.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Derivatives".
> It builds on `plans/deferred/numeric-contract.md` and `plans/deferred/vector-interface.md`, and it answers the execution plan's open question Q11; § 7 lists the edits the other plans need.
> Once approved, its rules move into `design/` (III.3, III.4, III.8, V.11, V.13, VI.2, and a new section of Part VI) and its steps into #522 and #109.

**Decisions this plan asks for, all open** (each explained where it arises, and gathered in § 9):

1. The derivative of a map at a point is a `LinOp` from the layout of the differentiated inputs to the layout of the output (§ 4.3).
2. A differentiability claim names input paths, not a `NumericSpec` (§ 4.5).
3. Code that traces is presumed differentiable in its continuous inputs, which states the accepted Q14 as a rule, and `differentiable=False` states the opposite (§ 4.5).
4. Every `Function` has a differentiability, which replaces the `SupportsDifferentiation` protocol (§ 4.4).
5. A law's derivatives are the derivatives of its capability maps, `log_prob_function(d)` and `sampler(d)`, which become `Function`s (§ 4.7).
6. The operations `jacobian`, `grad`, and `hessian`, evaluated at a point, with finite differences selected only by name (§ 5.1).
7. A value that a ProbPipe call returns inside a JAX transformation is raw, for operations as for plain calls (§ 4.8).

---

## 1. Why this plan exists

Three properties of a term decide what ProbPipe can do with it:

1. **numeric**: its values are arrays of numbers, so each has a vector (`numeric-contract.md` § 4.1);
2. **traceable**: JAX can trace its code (V.13);
3. **differentiable**: ProbPipe can obtain a derivative of it.

Each can hold without the others.
An integer array is numeric and has no derivative.
A function that calls `jnp.round` traces, and its derivative is zero almost everywhere.
A Fortran solver with an adjoint does not trace, and it has a derivative.

The design states the first two with care.
The third has one section, V.11, which leaves most of it open, and the code implements none of it: the claim raises `NotImplementedError`, and gradient methods are chosen because a density traces.
A model with a count parameter shows the result:

```python
lik = conditional_distribution(lambda k: Normal("y", k, 1.0),
                               given_spec={"k": NumericArraySpec(())}, label="lik")
model = lik & Poisson("k", 3.0)
condition_on.check(model, {"y": jnp.array(2.0)}).selected
# MethodInfo(True, method_name='inference_methods/blackjax_nuts', exact=False)
```

NUTS, which moves `k` continuously, is selected for a posterior over a count.

The goal is one contract for derivatives, built on the numeric contract and beside the execution claims:

- one definition of what may be differentiated, read from the spec and the claims;
- one representation of a derivative, which every derivative operation and every gradient method uses;
- one predicate on which methods dispatch.

## 2. The design today

### 2.1 What `design/` states

| Section | What it states |
|---|---|
| I, P7 | Opaque code joins compiled code at one boundary, and a gradient crosses it only through a derivative the callable supplies. |
| III.3 | A map claims `SupportsInverse` and `SupportsLogDetJacobian`, and `is_invertible` reads the first with its guard. |
| III.4 | A `LinOp` is a `Function` between flat numeric spaces. A square one claims both map capabilities, and its queries return scalar arrays, "keeping the queries differentiable". A Hessian is named as an endomorphism. |
| III.7 | The flat-vector law is `pushforward(d, to_vector)`, whose map claims its inverse and a unit Jacobian. |
| V.7 | Change of variables applies when a map is invertible and claims the Jacobian. |
| V.8 | Structural keys keep common random numbers, so "reparameterization gradients keep a low variance". |
| V.11 | `differentiable=` takes a `NumericSpec` template of the inputs gradients pass through. The `Function` then claims `SupportsDifferentiation`, read by `is_differentiable(x, values)`. A distribution claims "the event values its sampling reparameterizes". The claim is never inferred from a value being numeric. A `grad` operation is an open point. |
| V.12 | `bijector_for(constraint)` returns a map from ℝⁿ onto a support, claiming its inverse and its Jacobian. |
| V.13 | `differentiable` is one of three execution claims. An opaque callable claims it only with `vjp=`, `jacobian=`, or `jvp=`, and a gradient through an opaque call without one raises. The structure of a supplied derivative for a record is an open point. |
| VI.2 | `inverse(f)` and `log_det_jacobian(f, x)`, each with a capability route, beside which a numerical route may register. |

Two questions of `plans/execution-layer.md` bear on this plan:

- **Q14, accepted on 2026-10-17:** a traceable model may use gradient methods without a claim, and claims only narrow this.
- **Q11, open:** one `jacobian` operation, with `grad` as its case for a scalar output, and three routes (automatic differentiation, a supplied derivative, finite differences).

### 2.2 What the design leaves open

1. **Differentiable in what.** V.11 gives a law one template, but a law has three derivatives: of its density in the value, of its density in a kernel's given, and of its draws in the given, which is the pathwise derivative. One template cannot say which it means.
2. **Two sources of truth.** A claim restates the shapes that `input_spec` declares, as in `differentiable=NumericRecordSpec(theta=NumericArraySpec((2,)))` beside `input_spec=InputSpec(theta=(2,))`.
3. **What may be differentiated.** No section says that an integer or discrete leaf has no derivative.
4. **Modes and orders.** A supplied `jvp` serves forward mode only, and a supplied derivative has no second derivative. V.11's template records neither, so a method that needs reverse mode or a Hessian cannot check for it.
5. **What a derivative is.** No section states the value a derivative is when an input or the output is a record.
6. **V.11 against Q14.** V.11 says a claim is never inferred, and Q14 lets traceable code use gradients without one. No text states the rule that reconciles them.
7. **Operations.** VI.2 has two operations, and no section has `jacobian`, `grad`, or a Hessian.

## 3. The code today

### 3.1 Where the code agrees with the design

- `ExecutionClaims` holds `differentiable` beside `traceable` and `vectorized` (`core/_execution_claims.py`).
- `bijector_for` and `register_bijector` follow V.12, and each bijector claims `SupportsInverse` and `SupportsLogDetJacobian`.
- `BijectorTransformedDistribution` computes its density by change of variables, from the bijector's `_inverse` and `_log_det_jacobian`.
- The embedded call refuses a gradient with V.13's message (`functions/_boundary.py`).
- A plain `Function` call inside a JAX transformation returns its raw result, so `jax.grad(f)` works on a traceable `f`.

### 3.2 Where the code departs from the design

**G1. The claim cannot be made, and nothing is differentiable.**

```python
function(lambda x: x ** 2, differentiable=NumericRecordSpec(x=NumericArraySpec(())))
# NotImplementedError: Function(differentiable=...) is not supported yet; omit differentiable

f = function(lambda x: jnp.sin(x))
jax.grad(f)(1.0)          # Array(0.5403023)
is_differentiable(f)      # False
```

**G2. `inverse` and `log_det_jacobian` are stubs**, although every bijector implements both capabilities:

```python
b = bijector_for(positive)
b._log_det_jacobian(jnp.array(0.3))    # Array(0.3)
log_det_jacobian(b, jnp.array(0.3))    # NotImplementedError: log_det_jacobian is not implemented yet
inverse(b)                             # NotImplementedError: inverse is not implemented yet
```

**G3. Gradient methods are gated by traceability alone.**
`blackjax_nuts`, `blackjax_hmc`, and `tfp_nuts` check `traces_in_jax(model)`, which the gradient-free `blackjax_rwmh` and `blackjax_elliptical_slice` also check.
The SG-MCMC methods call `jax.grad` and check neither.
Three consequences follow, each reproduced:

1. The count model of § 1 selects NUTS.
2. A likelihood through `jnp.round` selects NUTS, whose gradient drops the likelihood without an error:

   ```python
   lik = conditional_distribution(lambda mu: Normal("y", jnp.round(mu), 1.0),
                                  given_spec={"mu": NumericArraySpec(())}, label="lik")
   model = lik & Normal("mu", 0.0, 3.0)
   jax.grad(lambda mu: model._unnormalized_log_prob({"mu": mu, "y": jnp.array(2.0)}))(0.3)
   # -0.0333: the prior's gradient, -mu / 9, alone
   ```

3. A callable that traces through `lax.while_loop` passes `check`, and fails when NUTS differentiates it:

   ```python
   f = function(lambda x: lax.while_loop(lambda c: c[1] < 3,
                                         lambda c: (c[0] * x, c[1] + 1), (x, 0))[0])
   jax.jvp(f.apply, (1.5,), (1.0,))   # (5.06, 13.5): forward mode works
   jax.grad(f)(1.5)
   # ValueError: Reverse-mode differentiation does not work for lax.while_loop ...
   ```

**G4. The pathwise derivative of a discrete law is zero without an error.**

```python
jax.grad(lambda r: Poisson("k", r)._sample(key))(3.0)   # Array(0.)
```

**G5. Jacobians and log-determinants are computed in seven places**, which Appendix A lists.
`BijectorTransformedDistribution` takes the Jacobian of an affine map with its own `jax.jacfwd` over its own ravel, and `LinOp.logdet` is a query that no `log_det_jacobian` reads.
`vector-interface.md` § 1.2 already counts four of the seven among its unconstrained-coordinate copies.

**G6. Only the bijector registry knows which supports are discrete.**
`Constraint` has no attribute that states it.
`bijector_for(non_negative_integer)` raises "the support is discrete" because `_register_no_smooth_bijector` registers a factory that raises, so the one test of a discrete support is a request for its bijector.

**G7. An operation inside `jax.grad` fails**, because an operation returns a term where a plain call returns its raw form:

```python
jax.grad(lambda m: log_prob(Normal("x", m, 1.0), 0.5))(0.0)
# ValueError: Triggering __jax_array__() during abstractification is no longer supported. ...
jax.grad(lambda m: log_prob(Normal("x", m, 1.0), 0.5).raw)(0.0)   # Array(0.5)
```

**G8. `LinOp` is not a `Function`** (#522), so it claims neither map capability, and the rest of ProbPipe does not read its `logdet` as a log-Jacobian.

## 4. The proposed design: the core contract

The contract is built in five levels, each reading only the levels before it:

1. **Specs** decide which leaves may be differentiated (§ 4.2).
2. **The Jacobian operator** is what a derivative is (§ 4.3).
3. **Maps** state or derive their differentiability, and compose it (§ 4.4 to § 4.6).
4. **Laws and kernels** have the differentiability of their capability maps (§ 4.7).
5. **Operations and methods** read the levels below through one predicate (§ 5 and § 6).

### 4.1 Three properties, three levels

| Property | A property of | Decided by | Stated in |
|---|---|---|---|
| numeric | a spec's value positions | the spec alone | `numeric-contract.md` § 4.1 |
| traceable | a callable at an input structure | a claim, or a probe | V.13 |
| differentiable | a map, in a set of its inputs | the spec, a claim or a presumption, and a source of the derivative | this plan |

Three rules relate them, and § 4.2 to § 4.5 make each precise:

1. **Differentiable implies numeric.** The differentiated inputs and the output are numeric, and the differentiated inputs are continuous (§ 4.2).
2. **Tracing is one source of derivatives, not a requirement.** JAX differentiates traceable code, and an opaque callable is differentiable through a derivative it supplies.
3. **Traceable does not imply differentiable.** `jnp.round`, `stop_gradient`, and an integer cast trace and give a wrong derivative, and `lax.while_loop` traces and has no reverse mode.

### 4.2 Continuous leaves

A derivative exists only where a value can move by small amounts.

> [!IMPORTANT]
> **Definition (continuous leaf).** A leaf of a numeric value spec is *continuous* when its dtype is a real floating dtype, or is unset, and its support is not discrete.
> A constraint states whether it is discrete as `Constraint.is_discrete`: `boolean`, `non_negative_integer`, and `integer_interval` are discrete, and every other built-in constraint is not.

Consequences:

1. **Read from the spec.** The numeric contract's `VectorLayout` marks each leaf with `is_continuous`, so a layout states which of its segments may be differentiated.
2. **Two exclusions.** A `Poisson` event has dtype float32 and is excluded by its discrete support. An `int32` leaf is excluded by its dtype.
3. **Continuous is not unconstrained.** A leaf on the sphere is continuous, and no bijector maps ℝⁿ onto it (V.12). A density on the sphere has a gradient in the ambient coordinates, and a method that moves in ℝᵈ refuses it through its bijector, not through this definition.
4. **Complex leaves are excluded** until a derivative for them is designed (Appendix B, item 3).

### 4.3 What a derivative is

The derivative of a map at a point is the linear map from small changes of its inputs to the resulting changes of its output.
ProbPipe has one kind for linear maps between numeric spaces, the `LinOp` (III.4), whose two sides each have a layout, so the derivative is a `LinOp`.

> [!IMPORTANT]
> **Definition (Jacobian operator).** Let `f` be a map, `W` a set of its continuous input paths, and `x` a point of its inputs. The *Jacobian operator* `J_W f(x)` is the `LinOp` with the following parts:
> 1. its input spec is the spec of `f`'s inputs at `W`, and its output spec is `f`'s output spec;
> 2. `matvec(t)` is the change of the output along a change `t` of the inputs at `W`, which JAX calls the JVP;
> 3. `rmatvec(c)` is the pullback of an output change `c` onto the inputs at `W`, which JAX calls the VJP;
> 4. `to_dense()` is its matrix, in the vector layouts of its two sides.

Consequences:

1. **One structure for every derivative.** A tangent has the spec of the inputs it changes, and a cotangent has the spec of the output, as in JAX. This settles V.13's open point on records.
2. **The other derivatives are queries of it:**
   - the gradient of a map with a scalar output is `J.rmatvec(1.0)`, which is a record of the spec of the inputs at `W`;
   - the log-Jacobian of a map with a square Jacobian is `J.logdet()`, through which `log_det_jacobian` falls back (§ 5.2);
   - the Hessian of a map with a scalar output is the Jacobian operator of its gradient, an endomorphism, as III.4 names it.
3. **No matrix unless asked.** A Jacobian operator built from a JVP or a VJP forms no matrix. `to_dense()` costs `d` JVPs or `m` VJPs, for `d` inputs and `m` outputs in the layouts, and a supplied `jacobian` gives the matrix at once.
4. **The modes are the actions.** A map with only a JVP gives an operator whose `rmatvec` raises, and a map with only a VJP gives one whose `matvec` raises.

```python
f = function(lambda x, y: {"s": x + y, "p": x * y},
             input_spec=InputSpec(x=(2,), y=()),
             output_spec=OutputSpec(RecordSpec(s=(2,), p=(2,))))
J = jacobian(f, {"x": jnp.array([1.0, 2.0]), "y": 3.0})
J.shape                                             # (4, 3): the output's vector by the inputs'
J.matvec({"x": jnp.array([1.0, 0.0]), "y": 0.0})    # the record s=[1, 0], p=[3, 0]
```

### 4.4 The differentiability of a map

> [!IMPORTANT]
> **Definition (differentiability).** The *differentiability* of a map `f` states which Jacobian operators of `f` ProbPipe can build:
> 1. `wrt`: the input paths gradients pass through, each a continuous leaf (§ 4.2), possibly none;
> 2. `modes`: the actions available, which are forward (`matvec`), reverse (`rmatvec`), or both;
> 3. `max_order`: the highest order of derivative available, or `None` for any order;
> 4. `source`: how ProbPipe obtains the derivative, which is *automatic* when JAX differentiates the callable, *supplied* when the callable provides the derivative, and *none* otherwise;
> 5. `claimed`: whether a claim states the differentiability, or ProbPipe presumes it (§ 4.5).
>
> A map is *differentiable in `W`, in a mode, to an order* when `W` is contained in `wrt`, the mode is one of `modes`, and the order is at most `max_order`.

```python
@dataclass(frozen=True)
class Differentiability:
    wrt: tuple[str, ...]                               # input paths, in declared order; () for none
    modes: frozenset[Literal["forward", "reverse"]]
    max_order: int | None                              # None for any order
    source: Literal["automatic", "supplied", "none"]
    claimed: bool                                      # a claim states it, or ProbPipe presumes it

def differentiability(f: Function) -> Differentiability: ...
def is_differentiable(f: Function, wrt: str | Sequence[str] | None = None, *,
                      mode: Literal["forward", "reverse"] = "reverse", order: int = 1) -> bool: ...
# wrt=None asks for every continuous input of f
```

Consequences:

1. **Every map has one.** A map that cannot be differentiated has `wrt=()` and the source `none`. `differentiability` therefore replaces the `SupportsDifferentiation` protocol, whose membership could not distinguish a claim from a presumption (decision 4).
2. **The source fixes the modes and the order.** An automatic source gives both modes to any order, unless the probe of § 4.5 finds a mode missing. A supplied source gives the modes of the supplied derivative, to order 1.
3. **Finite differences are not a source.** They approximate a derivative, so they are a route of the `jacobian` operation (§ 5.1), and they never make a map differentiable.

### 4.5 Claims and presumptions

A `Function` states its differentiability when it is built, as it states `traceable`, and the statement is fixed for its life.

| `differentiable=` | What it states |
|---|---|
| omitted, or `None` | nothing: ProbPipe presumes differentiability from traceability, by the definition below |
| `False` | no input is differentiable, whatever JAX would compute |
| a path, or a list of paths | those inputs are differentiable, and no others |

A derivative is supplied as `vjp=`, `jvp=`, or `jacobian=`, with V.13's signatures, and each takes and returns tangents and cotangents in the specs of § 4.3.
A `Function` given one claims the matching capability, `SupportsVJP`, `SupportsJVP`, or `SupportsJacobian`, which III.3 lists beside `SupportsInverse`.

```python
@function(traceable=False, output_spec=NumericArraySpec((40,), "float32"),
          differentiable="theta", vjp=forward_adjoint)
def forward(theta, settings): ...       # gradients pass through theta, and never through settings

@function(differentiable=False)
def binned(x):
    return jnp.round(x)                 # traces; every gradient method refuses it
```

> [!IMPORTANT]
> **Definition (presumed differentiability).** A `Function` with no statement and no supplied derivative is *presumed* differentiable, with the source automatic, in every continuous input, when it is traceable at the input structure (V.13). An opaque `Function` with no statement is not differentiable.

Consequences:

1. **Q14 as a rule.** Tutorial models keep their gradient methods, and V.11's "never inferred" now reads "never inferred from a value being numeric".
2. **A claim only narrows.** A path claim restricts `wrt`, and `False` empties it. A claim of a path that is not a continuous leaf raises `TypeError` when the `Function` is built, naming the path.
3. **The differentiation probe.** When a requester needs a mode of a presumed map, `check` traces that mode once on stand-ins, with `jax.make_jaxpr` of `jax.linearize` or of `jax.vjp`, and stores the outcome with the probe result (V.13). `check` then reports the `while_loop` of G3. A claimed map is not probed, and a false claim raises where the map is differentiated, as a false `traceable=True` raises where it is traced.
4. **What no probe finds.** `jnp.round` and `stop_gradient` give a derivative without an error, so a user states `differentiable=False` for code that uses them (Appendix B, item 2).
5. **Paths, not a spec.** The claim names paths of the declared inputs, read with II.6's path reader, so `input_spec` alone states the shapes (decision 2). A path into a record input, such as `"theta/beta"`, claims part of that input.

### 4.6 Composition

A map built from other maps derives its differentiability from theirs, as V.13 derives a law's traceability from its parts.

> [!IMPORTANT]
> **Rule (chain rule over differentiability).** A value is differentiable through a chain of steps when every step on its path is differentiable in the input that carries it.
> 1. A composition `g ∘ f` is differentiable in `W` when `f` is, and `g` is differentiable in every input that `f`'s output feeds.
> 2. Its modes are the intersection of the steps' modes, and its order is the least of their orders.
> 3. Its source is automatic: JAX differentiates the composition, and applies each supplied step's derivative through the rule § 4.8 installs.
> 4. A step whose output does not depend on `W` imposes nothing, as V.13 already says of an opaque call off a gradient's path.

Construct by construct:

1. **Selection and merge.** `f[p]` keeps `f`'s differentiability. `f & g` is differentiable in an input when every operand that reads the input is.
2. **Inverse.** `inverse(f)` of an invertible `f` that is differentiable in its one input is differentiable in its one input.
3. **Vector forms.** The numeric contract's `vector_map(spec)` is linear and claims every input, so `f.to_vector()` is differentiable at the vector segments of `f`'s `wrt`.
4. **Bijectors.** Each map `bijector_for` returns claims its one input, with the source automatic, both modes, and any order.
5. **`LinOp`.** A `LinOp` claims its one input to any order, and its Jacobian operator at every point is itself.

### 4.7 Laws and kernels

A law is not a map, and each of its capabilities is one.
A law's derivatives are therefore the derivatives of its capability maps, and no further notion is needed.

| Capability map | Its inputs | What its derivative is, and what uses it |
|---|---|---|
| `log_prob_function(d)` | `value` | the density's gradient in the value: HMC, NUTS, MALA, and a mode search |
| `log_prob_function(k)` | `value`, `given` | in `given`, the score in the parameters: maximum likelihood, variational inference |
| `sampler(k)` | `key`, `given` | in `given`, the pathwise derivative of a draw: reparameterized variational inference, the sensitivity of a simulation |

The two functions come from other plans: `log_prob_function` from `vector-interface.md` § 4.5, and `sampler` from #554 § 7.
Both plans have each return a raw callable, which carries no claims, so a method could not check it.
This plan has each return a `Function` whose `apply` is that raw callable and whose claims are the law's (decision 5).

A parameter of a law is not one of its inputs, so a derivative in a parameter is a derivative of the kernel whose given holds it.
`Normal("x", mu, 1.0)` therefore has no derivative in `mu`, and `conditional_distribution(lambda mu: Normal("x", mu, 1.0), ...)` has one in its given `mu`.

Each law and kernel derives the differentiability of its capability maps once, through a private `_differentiability(capability)` beside `_traceability()`:

| Law or kernel | Its density, in the value and the given | Its sampler, in the given |
|---|---|---|
| a family | automatic, at the continuous leaves | automatic when TFP states the family `FULLY_REPARAMETERIZED`, and none otherwise |
| a joint | its factors', path by path, by the chain rule | its factors', path by path |
| a view or a rename | its parent's, restricted to its paths | its parent's, restricted |
| a law from functions | its density function's | its sampling function's |
| the kernel of a function | the function's, composed with the returned law's | the same composition |
| a transformed law | the base's, composed with the bijector's inverse | the base's, composed with the bijector |
| an empirical law | none, since it has no density | none, since a draw selects an atom |
| a KDE | automatic in the value | none |
| a Stan program's unconstrained view | supplied: BridgeStan's `log_density_gradient` | none |
| any other law | none, unless its class states one | none |

Consequences:

1. **G4 is fixed.** TFP does not state `Poisson` fully reparameterized, so its sampler is not differentiable in its rate, and the gradient that was zero is refused.
2. **A discrete event has no density gradient.** The density of the count model of § 1 has a `wrt` without `k`, so a method that needs the gradient in every event leaf refuses it (§ 6.1).
3. **Introspection reports it.** `execution-introspection.md` § 4.4 organizes a law's status by capability, and each capability's status gains its differentiability.

### 4.8 Derivatives inside JAX

JAX is the one mechanism that differentiates inside a compiled program, and ProbPipe installs every supplied derivative into it:

1. **Supplied derivatives become JAX rules.** The embedded call (V.13) stages a supplied `vjp` with `jax.custom_vjp`, and a `jvp` with `jax.custom_jvp`. A supplied `jacobian` is staged with `jax.custom_jvp` as `J @ t`, which JAX transposes for reverse mode. A traceable `Function` that supplies a derivative is staged the same way, and the supplied derivative replaces JAX's own rule. So `jax.grad(f.apply)` uses a supplied derivative wherever one applies, and the `jacobian` operation's automatic route is JAX's transform of `f.apply` (§ 5.1).
2. **A route without a derivative refuses one.** An approximate route that is not differentiable, such as an MCMC run, stages its result behind a rule that raises when a gradient flows into it, as the embedded call refuses a gradient today (V.13). No derivative is ever zero without an error.
3. **An input outside a claim raises**, through `symbolic_zeros`, as V.13 states, so a claim narrower than JAX's derivative is enforced.
4. **Values cross raw.** A value that any ProbPipe call returns inside a JAX transformation is its raw form, as a plain call's already is. A law, kernel, or function result stays a term. G7's example then works without `.raw` (decision 7).

### 4.9 Where the contract is implemented

Each concept has one module, and each module reads only the concepts above it in this table.

| Concept | Module | Replaces |
|---|---|---|
| `Constraint.is_discrete` | `core/constraints.py` | the raising bijector factories as the test of discreteness |
| `Leaf.is_continuous` in `VectorLayout` | `core/_vector_layout.py`, from the numeric contract | — |
| the claim, `Differentiability`, and the protocols `SupportsVJP`, `SupportsJVP`, `SupportsJacobian` | `core/_execution_claims.py`, `values/_function_base.py` | `SupportsDifferentiation`, and the two stubs of G1 |
| presumption, the differentiation probe, and composition | `functions/_differentiation.py`, new | `traces_in_jax` as a gradient gate |
| supplied derivatives as JAX rules | `functions/_boundary.py` | the refusing `custom_jvp`, which stays as the rule of a map without a derivative |
| `JacobianLinOp` | `linalg/` | `_affine_jacobian` |
| `jacobian`, `grad`, `hessian`, and the routes of `inverse` and `log_det_jacobian` | `operations/_derivatives.py`, `operations/_inverse.py` | the stubs of G2 |
| a law's `_differentiability(capability)` | each law and kernel class, beside `_traceability()` | — |

## 5. The operations

### 5.1 `jacobian`, `grad`, and `hessian`

```python
def jacobian(f: Function, x, *, wrt: str | Sequence[str] | None = None) -> LinOp: ...
def grad(f: Function, x, *, wrt=None) -> Numeric: ...     # derived: jacobian(f, x, wrt=wrt).rmatvec(1.0)
def hessian(f: Function, x, *, wrt=None) -> LinOp: ...    # derived: the jacobian, in wrt, of grad(f, ·, wrt=wrt) at x
```

`x` is a record of `f`'s inputs, or a bare value when `f` has one input, and `wrt=None` means every continuous input.

| Route of `jacobian` | Route source (VI.0) | Exact | Feasible when |
|---|---|---|---|
| `supplied` | capability: `SupportsJacobian`, else `SupportsJVP` or `SupportsVJP` | yes | `f` supplies a derivative, and `wrt` lies in its claim |
| `automatic` | structural: JAX transforms `f.apply` | yes | `f` is differentiable in `wrt` with the source automatic |
| `finite_differences` | fallback, selected only by name | no | `f`'s inputs at `wrt` are continuous; its `2d` evaluations run as one set of evaluation points (V.9) |

Consequences:

1. **Q11 is answered.** One `jacobian` operation is primitive, and `grad` and `hessian` are derived from it (VI.0), so both take its routes and its fidelity.
2. **Lifting needs no code.** `grad(f, points)` over a batch of points sweeps, and `grad(f, d)` over a law is the law of the gradients, by V.5.
3. **Finite differences only by name.** A gradient method given an approximate derivative it did not ask for would return a wrong answer without an error. `method="finite_differences"` therefore selects the route, and `exact_only=True` refuses it (decision 6).
4. **A density's gradient** is `grad(log_prob_function(d), x)`. A shorthand, such as `score(d, x)`, is left open (§ 9, question 6).

### 5.2 `inverse` and `log_det_jacobian`

VI.2's contract stays, and the two operations gain their implementations and one route:

- `inverse(f)` runs its capability route, which returns a `Function` that claims `f` as its inverse and carries `f`'s Jacobian claim (VI.2).
- `log_det_jacobian(f, x)` runs `SupportsLogDetJacobian` when `f` claims it. Otherwise, when `f` is differentiable in its one input and its Jacobian is square, it returns `jacobian(f, x).logdet()`. This route is exact, and VI.2 already allows it.

```python
log_det_jacobian(bijector_for(positive), 0.3)    # 0.3, through the capability
g = function(lambda x: jnp.tanh(x) + x)            # claims no log-Jacobian
log_det_jacobian(g, jnp.array([0.1, 0.2]))         # through jacobian(g, x).logdet()
```

A `LinOp` claims `SupportsLogDetJacobian` with `logdet` as its implementation (III.4), so G8's query and the operation compute one quantity in one place.

## 6. Particular cases

### 6.1 Dispatching inference methods

Each method states the derivative it needs from the target, and its `check` reads it through `is_differentiable` before any trace.

| Method | Needs | Of |
|---|---|---|
| `blackjax_nuts`, `blackjax_hmc`, `tfp_nuts` | reverse mode, order 1, inside a JAX program | `log_prob_function(unconstrained_law(target))`, in `vec` |
| `blackjax_sgld`, `blackjax_sghmc` | the same | the prior's density and the minibatched likelihood's |
| a Laplace approximation or a Newton step (future) | order 2 | the same as NUTS |
| `blackjax_rwmh`, `blackjax_elliptical_slice` | no derivative; a density that traces or embeds | — |
| `cmdstan_nuts`, `pymc_nuts`, `nutpie_nuts`, `pymc_advi` | nothing from ProbPipe, since the program differentiates itself | — |
| `empirical_reweighting`, `pyabc_smcabc` | no derivative | — |

"Inside a JAX program" admits both sources, automatic and supplied, since § 4.8 installs both into JAX.

Consequences:

1. **G3's three cases are refused in `check`.** The count model fails at `k`, a likelihood claiming `differentiable=False` fails at its input, and the `while_loop` fails the reverse-mode probe. SG-MCMC gains the check it lacks.
2. **Opaque models can use NUTS.** A model whose simulator supplies a VJP is differentiable, so NUTS runs it through embedded calls, as `execution-layer.md` § 7.3 shows.
3. **One message,** following `STYLE_GUIDE.md` § 9.3:

   ```text
   blackjax_nuts needs the gradient of the log density of 'model' in 'k', but 'k' is discrete
   (non_negative_integer). Use a method that needs no gradients, such as method='blackjax_rwmh'
   ```

### 6.2 Optimization (future)

An `optimize` operation finds a minimizer of a map with a scalar output, and it dispatches on the map's differentiability as the inference methods do:

| Route | Needs | Example backends |
|---|---|---|
| `newton` | order 2 | `jax.scipy.optimize`, a trust-region method |
| `gradient` | reverse mode, order 1 | `optax`, `scipy.optimize.minimize` with `jac` |
| `gradient_free` | nothing, so a host-only map is admitted | Nelder–Mead, CMA-ES |

Two operations derive from it:

- **the mode of a law:** `optimize` of the negative log density of `unconstrained_law(d)`, mapped back to `d`'s values;
- **a Laplace approximation:** a `MultivariateNormal` at that mode whose precision is the `hessian` there, a `LinOp`, which VII.6 accepts.

A host optimizer may take a finite-difference gradient by name, since it runs outside JAX.

### 6.3 Opaque code and external gradients

Every derivative computed outside JAX enters ProbPipe as a supplied derivative:

1. **A user's adjoint solver** supplies `vjp=` (V.13).
2. **A backend program** supplies its own gradient. A Stan program's unconstrained view supplies BridgeStan's `log_density_gradient` as its VJP, so BlackJAX's NUTS can run a Stan model through an embedded call, beside CmdStan's own NUTS.
3. **Finite differences on a cluster.** `jacobian.using(method="finite_differences", executor="slurm")(f, x)` runs its `2d` evaluations as one set of evaluation points.

### 6.4 Change of variables and reparameterization

V.7, V.12, and the vector-interface plan each use § 4 and § 5 and keep no derivative code of their own:

1. A bijector is a map that claims its inverse, its log-Jacobian, and its differentiability (§ 4.6).
2. `BijectorTransformedDistribution` reads its log-Jacobian through `log_det_jacobian`, and its affine covariance through `jacobian(f, mean).to_dense()`, so `_affine_jacobian` goes.
3. `vector_bijector(spec)` sums its leaves' log-Jacobians through `log_det_jacobian`, and refuses a leaf whose support `is_discrete` (`vector-interface.md` § 4.4, rule 3).

### 6.5 Pathwise derivatives and sensitivity analysis

#109 asks to differentiate through a workflow, as in the sensitivity of a posterior summary to a prior's scale.
Whether a derivative passes through an operation depends on the route the operation selects:

1. **An exact route differentiates** by § 4.6. The variance of `A @ d`, which a closed-form rule computes, has an exact derivative in a scale of `d`.
2. **A Monte Carlo route differentiates pathwise** when its sampler is differentiable in the given (§ 4.7). Structural keys hold the draws fixed as the input moves (V.8), so the result estimates the derivative of the expectation.
3. **An MCMC route does not differentiate**, since a chain's accept steps are discrete. `grad` through `condition_on` with an MCMC method raises, naming the method (§ 4.8, item 2).

### 6.6 Batches

A claim is not part of a `FunctionSpec`, so the common element spec of `batch-contract.md` does not fix it.
A batch of maps has the differentiability its elements share, which is the intersection of theirs, and `jacobian` over a batch of maps sweeps by VI.11.

## 7. Edits to the other plans and to `design/`

| Document | Edit |
|---|---|
| `numeric-contract.md` § 4.2 | `Leaf` gains `is_continuous` (§ 4.2). |
| `numeric-contract.md` § 4.3 | `vector_map` claims every input, with the source automatic and any order (§ 4.6). |
| `vector-interface.md` § 4.5, § 8.5 | `log_prob_function` returns a `Function` whose `apply` is the raw callable and whose claims are the law's (decision 5). |
| `vector-interface.md` § 4.4 | Rule 3 reads `Constraint.is_discrete`. |
| `inference-interface.md` § 2.3, § 4.4 | A gradient method's `check` reads `is_differentiable` (§ 6.1), and the table states what each method needs. |
| `execution-introspection.md` § 4.4 | Each capability's status gains its differentiability, and `check` reports the probe of § 4.5 beside the traceability probe. |
| `execution-layer.md` Q11 | Answered by § 5.1. |
| `design/` III.3, III.4 | The protocols `SupportsVJP`, `SupportsJVP`, and `SupportsJacobian`, and the Jacobian operator. |
| `design/` III.8, V.11, V.13 | V.11 is rewritten from § 4.2 to § 4.7, V.13's claim takes paths or `False`, and III.8 points to § 4.7's table. |
| `design/` VI.2, and a new section of Part VI | The routes of § 5.2, and `jacobian`, `grad`, and `hessian`. |
| `design/glossary.md` | continuous leaf, Jacobian operator, differentiability, presumed differentiability, and derivative source. |

## 8. Steps

Each step is one PR.

1. **D1, specs**, after the numeric contract's N1: `Constraint.is_discrete`, and `Leaf.is_continuous`.
2. **D2, claims**: `differentiable=` with paths or `False`; the three supplied derivatives as JAX rules; `Differentiability`, `differentiability`, and `is_differentiable`; presumption; and the differentiation probe. This step is #522's claim work and the execution layer's deferred supplied derivatives.
3. **D3, operators and operations**: `LinOp` as a `Function` (#522), `JacobianLinOp`, `jacobian`, `grad`, and `hessian`, with `inverse` and `log_det_jacobian` implemented.
4. **D4, laws**, after the vector interface's V3: `_differentiability(capability)` on every law and kernel, and `log_prob_function` and `sampler` as `Function`s.
5. **D5, dispatch**: every gradient method's `check` reads its requirement, with the message of § 6.1.
6. **D6, optimization**, later: `optimize`, the mode of a law, and the Laplace approximation.

**Tests.**

- One conformance test, parametrized over the sources, asserts that `jacobian(f, x).to_dense()` agrees with central finite differences, and that `matvec` and `rmatvec` are transposes of each other.
- A table test asserts the differentiability of every row of § 4.7.
- Each example of G1 to G4 and G7 becomes a test of the new behaviour.

**What users see.**

- **Breaking:** the gradient methods refuse a model with a discrete parameter, a step that claims `differentiable=False`, or a reverse-mode failure, where today they run or crash. A claim takes paths where V.11 takes a spec, and no released code makes a claim, since the claim raises today.
- **Fixed:** `inverse` and `log_det_jacobian` work, `is_differentiable` answers, a discrete law's pathwise gradient is refused, and an operation works inside `jax.grad`.
- **Added:** `jacobian`, `grad`, `hessian`, supplied derivatives, and NUTS on an opaque model that supplies a VJP.

## 9. Questions for the maintainer

1. **The derivative is a `LinOp`.** Recommended: `J_W f(x)` is a `LinOp` between the two layouts, and every derivative is a query of it (§ 4.3). Alternative: a dense array, or a record of records. Then the Hessian, the log-Jacobian, and the derivatives that form no matrix each need a form of their own.
2. **Paths, not a spec.** Recommended: `differentiable="theta"` (§ 4.5). Alternative: V.11's `NumericSpec`, which restates the shapes of `input_spec`.
3. **Presumption.** Recommended: traceable code with no statement is presumed differentiable in its continuous inputs, and `differentiable=False` opts out, which states the accepted Q14 as a rule (§ 4.5). Alternative: strict V.11, which removes the gradient methods from every tutorial model.
4. **No `SupportsDifferentiation`.** Recommended: every `Function` has a `Differentiability`, possibly empty, read by `differentiability(f)` (§ 4.4). Alternative: keep the protocol, and report a presumption apart from it.
5. **Capability maps as `Function`s.** Recommended: `log_prob_function(d)` and `sampler(d)` return `Function`s that carry the law's claims (§ 4.7). Alternative: raw callables, with a method reading the law's differentiability separately.
6. **The operations.** Recommended: `jacobian`, `grad`, and `hessian` evaluated at a point, with finite differences selected only by name (§ 5.1). Alternatives: operations that return a function of the point, as `jax.grad` does, or finite differences as an automatic fallback. A shorthand for a density's gradient, such as `score(d, x)`, is also open.
7. **Values cross JAX raw.** Recommended: inside a JAX transformation every call returns a value result in its raw form (§ 4.8). Alternative: operations keep returning terms, and the documentation tells a user to read `.raw`.

## Appendix A. Where derivatives are computed today

| Site | What it computes |
|---|---|
| `functions/_reparameterization._BackendBijector._log_det_jacobian` | a TFP bijector's log-Jacobian, summed over every axis |
| `inference/_inference_utils._coordinates`, as `UnconstrainedCoordinates.log_jacobian` | the sum of the leaves' log-Jacobians, over a layout it builds itself |
| `inference/_bayesflow_posteriors._field_bijectors` and `_flow_log_density` | a bijector per field, and the log-Jacobians of the fields |
| `inference/_pymc_method`'s transform classes, with `_forward_log_det_jacobian` | PyMC's transforms, rebuilt as TFP bijectors |
| `families/_programs._UnconstrainedStanView` | Stan's own Jacobian, through BridgeStan |
| `families/_transformed._affine_jacobian` | an affine map's Jacobian, by `jax.jacfwd` over its own ravel |
| `linalg.LinOp.logdet` | an operator's log-determinant, which nothing reads as a log-Jacobian |

One further site takes a gradient with no check: `inference/_blackjax_sgmcmc._build_grad_estimator` calls `jax.grad` on a minibatch density.

The gates on traceability are `traces_in_jax`, in `_blackjax_mcmc`, `_tfp_mcmc`, `_blackjax_ess`, and `_empirical_reweighting`, and `_target_traces`, in `_blackjax_rwmh`.
Of these, `_blackjax_mcmc` and `_tfp_mcmc` gate gradient methods.

## Appendix B. Readings this plan adopts

1. **V.11's template for a distribution** is read as the pathwise derivative of its sampler ("the event values its sampling reparameterizes"), which § 4.7 states as the differentiability of `sampler(k)` in its given.
2. **Operations with a zero derivative.** No probe finds `jnp.round` or `stop_gradient`. A scan of the traced program for them could warn later, and until then the plan relies on `differentiable=False`.
3. **Complex leaves** are not continuous leaves, since a real gradient of a complex input needs a convention, holomorphic or Wirtinger, that no section states.
4. **An unresolved answer is `False`.** `is_differentiable` returns `False` when a presumption needs a probe it cannot run, as for a `Function` without a declared input structure, as `is_invertible` treats an unresolved guard (III.3).
