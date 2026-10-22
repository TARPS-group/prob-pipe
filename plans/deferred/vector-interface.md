# Plan: one vector interface for numeric laws

> **Status: deferred until after the workshop; not approved.**
> Written on 2026-10-21, during step E6 (#618), against `dev/overhaul` as of #618, and deferred by the maintainer the same day so as to add no work before the workshop of 2026-10-27.
> Review it, and answer its questions, before implementing anything from it. Before then, re-check it against the code: E4 rewrites the inference runners this plan also changes.
> `plans/deferred.md` lists it under "The inference interface and vectors".
> Once approved, its decisions move into `design/` (II.3, III.7, V.12) and its steps into the plans of the work that follows the workshop.

**Decisions this plan asks for, all open** (each explained where it arises, and gathered in § 10):

1. The names `to_vector_map`, `vector_law`, `vector_bijector`, `unconstrained_law`, and `log_prob_function` (§ 4, § 8).
2. Whether V3 ships #554's `sampler(d)` beside `log_prob_function`, or #554 ships it on its own schedule (§ 4.5).
3. Whether to add no pytree view, which is the recommendation (§ 6).

---

## 1. Why this plan exists

### 1.1 What an inference algorithm needs

Most inference algorithms are written for a vector of real numbers.
NUTS, HMC, and random-walk Metropolis move a point in ℝᵈ. Variational methods fit a density on ℝᵈ. A mass matrix or a proposal covariance is a d × d matrix.

ProbPipe's laws are richer. A posterior over `{beta: (3,), sigma: (), groups: {mu: (8,), tau: ()}}` has named, shaped, nested parts. Some parts are constrained: `sigma` and `tau` are positive.

So between a ProbPipe law and such an algorithm, four things must happen:

1. **A layout.** Fix an order in which a value's entries become the entries of one vector, and read a vector back as a value.
2. **A density on vectors.** Evaluate the law's density at a vector.
3. **Unconstrained coordinates.** Map ℝᵈ′ onto the vectors whose values lie in the law's support, so an algorithm that steps anywhere in ℝᵈ′ never leaves it. The density in these coordinates gains the log-determinant of the map's Jacobian (the change of variables).
4. **The way back.** Turn the algorithm's vectors, in either coordinate system, into draws of the original law.

### 1.2 What ProbPipe has today

Two of these four already have a clean, designed home.

- **The layout is fixed at the value level** (design II.3, III.5, III.6). Every numeric value has `to_vector()`, `vector_size`, and `from_vector(spec, vec, *, label)`. A record lays its leaves out in its spec's order, depth first, each raveled row-major. This contract is settled and stays as it is.
- **Unconstrained maps are designed per constraint** (V.12). `bijector_for(constraint)` returns a `Function` from ℝⁿ onto the support, claiming its inverse and its Jacobian.

The design also names the law-level object: III.7's **flat-vector law**, `pushforward(d, to_vector)`, "a numeric law's law over its coordinates". It was never built.

Instead, the law-level work is done by about twenty private helpers across ten modules. Each derives the layout from the event spec in its own way. A survey of the code found the following.

- **Density on a vector: five routes, which disagree.**
  - `flat_density` does not reshape a law over one array.
  - `build_target_log_prob_flat` reshapes only parametric families, which it recognizes by class.
  - `_flat_unflatten` reshapes by spec.

  As a result, on a law over a `(2, 2)` array, `blackjax_rwmh` and `tfp_nuts` crash while their `check` reports the law feasible. The same happens for every method on a non-family law over a constrained array. (All reproduced.)
- **Vector to value: five implementations.**
  - `_reconstruct_from_vector`, `NumericRecordBatch.from_vector`, `_record_atoms`, `_leaf_variables`, and the KDE's `_unflattened`.
  - `operations/` and the BayesFlow modules also import the private `_reconstruct_from_vector` instead of the public `from_vector`.
- **Value to vector: about ten implementations.**
  - Among them, `flat_vector` lays a mapping out in the mapping's own key order, not the spec's. A nested draw given in another key order comes out scrambled (reproduced).
- **The unconstrained map of a whole law: four implementations with different policies.**
  - The four are the inference helpers, BayesFlow, PyMC's transform table, and Stan's own view.
  - They disagree on policy: one keeps an undeclared or discrete leaf unchanged, another raises.
  - The two inference helpers also disagree with each other: `unconstrained_coordinates` raises on a law with no layout, while `unconstrained_chain` silently returns the identity.
- **"Can this law be laid out" is decided three ways:** by class (`TFPDistribution`), by output form (a positional record only, so a record in the keyword form is refused), and by spec. None reads `NumericDistribution`, which is the design's criterion.
- **The state a chain carries depends on the support.**
  - A `(2, 2)` law's initial state stays `(2, 2)` when no bijector applies, and becomes `(4,)` when one does.
  - `get_init_state`'s fallback box covers the whole vector, where VI.6 says it covers only the fields no sampling factor draws.
- **Bad names.** "flat view" means "is a parametric family". `_flat_record_spec` returns a spec, not anything flat. `_coordinates` means four different things in four modules. The parameter vector is called "position", "theta_flat", "flat state", "chain state", "coordinates", and "params".

The goal of this plan is one source of truth for the four needs of § 1.1. It reuses the fixed value-level contract and V.12, builds III.7's designed law, and deletes the private copies.

## 2. Principles

1. **The value-level contract is the layout.** There is exactly one way to lay out a numeric value as a vector: `to_vector`. Nothing in this plan defines another order.
2. **A law over vectors is a law.** Laying out a law's draws, or moving to unconstrained coordinates, gives another ProbPipe law, built by `pushforward` with a map that claims its inverse and Jacobian (V.7). So density, sampling, moments, and provenance work on it as on any law, and nothing new is invented for them.
3. **The support decides the unconstrained map, leaf by leaf, through V.12.** One rule, `bijector_for(leaf.support)` for each leaf, read from the spec. No class checks.
4. **Compiled code reads raw functions through public operations,** as #554's `sampler(d)` does for draws. An operation called outside compiled code returns a pure JAX function.
5. **One layer, one module.** Everything in this plan lives in `functions/` (maps) and `families/` (the laws), beside V.12 and the existing change of variables. Inference code calls it and contains no layout code.

## 3. The layers, bottom up

```
value  ──to_vector──▶  vector ∈ ℝᵈ  ◀──vector_bijector(spec)──  z ∈ ℝᵈ′
       ◀─from_vector──                ──inverse(...)─────────▶

law d  ──pushforward──▶ vector_law(d) ──pushforward──▶ unconstrained_law(d)
         (to_vector_map,                (inverse of vector_bijector,
          unit Jacobian)                 change of variables)
```

| Layer | What it is | Where it lives | Status |
|---|---|---|---|
| 0. Values ↔ vectors | `to_vector`, `from_vector`, `vector_size` | `core/` | **fixed**; one gap filled (§ 4.1) |
| 1. The vector map of a spec | `to_vector_map(spec)`: `to_vector` as a `Function` that claims its inverse and a zero log-Jacobian | `functions/_reparameterization.py` | new |
| 2. The vector law | `vector_law(d) = pushforward(d, to_vector_map(spec))`: III.7's flat-vector law | `families/_vector.py` | new (designed in III.7) |
| 3. The unconstrained bijector of a spec | `vector_bijector(spec)`: ℝᵈ′ onto the vectors of the spec's support, built leaf by leaf from V.12 | `functions/_reparameterization.py` | new (extends V.12 from a constraint to a spec) |
| 4. The unconstrained law | `unconstrained_law(d) = pushforward(vector_law(d), inverse(vector_bijector(spec)))` | `families/_vector.py` | new; reuses `BijectorTransformedDistribution` |
| 5. Raw functions for compiled code | `log_prob_function(d)`, and `sampler(d)` from #554 § 7 | `operations/` | new (`sampler` per #554) |

Here `spec` is always `d.event_spec.term_spec`.

## 4. The design, layer by layer

### 4.1 Values and vectors (fixed)

The contract of II.3 stays exactly as it is, with one gap filled.

- `NumericArray.to_vector()`, `NumericRecord.to_vector()`, and `NumericRecordBatch.to_vector()` give the coordinates. A batch gives `(*batch_shape, vector_size)`.
- `from_vector(spec, vec, *, label)` is the inverse, casting each leaf to its declared dtype.

**The gap.** `NumericArrayBatch` has neither `to_vector` nor `from_vector`, although II.3 states the interface for every numeric kind. This plan adds both, as `NumericRecordBatch` has them: `to_vector` gives `(*batch_shape, vector_size)`, and `from_vector(spec, vec, *, level_names, axes_per_level=None, label=None)` reads it back. This completes the fixed contract; it changes nothing in it.

**A known deviation, out of scope.** `NumericRecord.__jax_array__` presents a record's only field, and raises for more than one, where II.3 says it presents `to_vector()`. It belongs to the value level and is recorded for `deferred.md`.

### 4.2 The vector map of a spec

**What.** `to_vector_map(spec)` is a `Function` whose input is a value of `spec` and whose output is the vector `value.to_vector()`.

**Why a `Function`.** `pushforward` takes a `Function`, and V.7's change of variables applies to a map that claims `SupportsInverse` and `SupportsLogDetJacobian`. `to_vector` is a bijection between the spec's values and ℝᵈ that only rearranges entries, so its log-Jacobian is exactly 0. As a `Function`, it lets the vector law be built by the existing `pushforward`, not by a parallel mechanism.

**Its spec.** It takes the input `value` with spec `spec`. Its output spec is `OutputSpec(vec=NumericArraySpec((d,), dtype))`, where `d = spec.vector_size` and `dtype` is the result type of the leaves' dtypes (what `to_vector` gives). `vec` is the glossary's canonical name for a 1-D numeric serialization.

**Requirements.** `spec` is a `NumericSpec` (a `NumericArraySpec` or a `NumericRecordSpec`) whose shapes are concrete. A spec with a symbolic dimension raises `NumericSpec.vector_size`'s existing error, which names `with_dim_sizes`.

### 4.3 The vector law: III.7's flat-vector law, built

**What.** `vector_law(d)` is the law of `X.to_vector()` for `X ~ d`. It is defined as `pushforward(d, to_vector_map(d.event_spec.term_spec))`, and `vector_law` is the named shorthand for that one composition, so it has one source of truth.

**Its event.** `OutputSpec(vec=NumericArraySpec((d,), dtype, support))`.
- `support` is the leaves' common support when all share one, and otherwise none. A vector's entries may have different supports, and one `NumericArraySpec` holds one support.
- The per-leaf supports stay on the original spec, which is where § 4.4 reads them.

**Its capabilities**, by V.7 with a zero log-Jacobian:
- an unnormalized density exactly when `d` has one, equal to `d`'s at `from_vector(spec, vec)`;
- a normalized density when `d` has one;
- sampling when `d` samples, as `to_vector` of `d`'s draws.

**Moments come for free.** VI.5 already defines `cov(d)` as "a covariance operator over the flattened draw". So `mean(vector_law(d))` and `cov(vector_law(d))` are the moments in the canonical layout. Elliptical slice sampling needs exactly these. Today it assembles them in factor order, which only coincides with the canonical order.

**Implementation.**
- A family class `VectorLaw(Distribution)` in `families/_vector.py`, which the change-of-variables rule returns for `to_vector_map`.
- Its raw methods call `from_vector` and `to_vector` and nothing else. This replaces `_flat_unflatten`, `flat_density`, `_flat_view`, and the density half of `build_target_log_prob_flat`.
- `BijectorTransformedDistribution` stays for array-to-array maps.

### 4.4 Unconstrained coordinates

**What.** `vector_bijector(spec)` is a `Function` from ℝᵈ′ onto the set of vectors `v` with `from_vector(spec, v)` in `spec`'s support.

- **Construction:** it is built leaf by leaf, in the layout's order, from V.12. Leaf `i`, with constraint `cᵢ`, gets `bᵢ = bijector_for(cᵢ)`.
- **Forward map:** split `z` into segments of the leaves' unconstrained sizes, apply each `bᵢ`, and ravel each result.
- **Inverse:** split `v` by the leaves' sizes, then apply each `bᵢ`'s inverse.
- **Log-Jacobian:** the sum of the leaves' log-Jacobians.
- **Unconstrained size:** `d′` can be less than `d`. A 3-simplex leaf takes 2 coordinates.

**One policy for every leaf:**
1. A leaf whose support is `real`, or that declares none, maps by the identity. A leaf with no declared support ranges over ℝ, which is what the inference helpers assume today.
2. A leaf with a continuous constrained support maps by `bijector_for(support)`, and a registered bijector (`register_bijector`) is honoured.
3. A leaf with a discrete support, or an integer dtype, raises `MathematicalDomainError`, naming the leaf: no continuous reparameterization exists. Today one helper silently moves such a leaf continuously and truncates it on the way back, and another raises; the plan takes the honest one.

**Why on the spec and not on the law.** The supports are part of the spec (a `NumericArraySpec` carries its support), so the map is a function of the spec alone. That is how V.12 is already organized, keyed on constraints. A method that has a spec but no law, such as one reading a program's parameter record, can use it too.

**What.** `unconstrained_law(d)` is the law of `z = vector_bijector(spec)⁻¹(X.to_vector())`. It is defined as `pushforward(vector_law(d), inverse(vector_bijector(spec)))`.
- **Its event:** `OutputSpec(vec=NumericArraySpec((d′,), dtype, real))`.
- **Its density** is `d`'s density at `from_vector(spec, b(z))` plus `log |det J_b(z)|`, by the existing change of variables (`BijectorTransformedDistribution`, which already carries an unnormalized density when its base has one).
- **Back to values:** apply `b` and then `from_vector`.
- **Replaces:** `UnconstrainedCoordinates`, `unconstrained_coordinates`, `unconstrained_chain`, `_leaf_maps`, `_flat_leaves`, `_coordinates`, BayesFlow's `_field_bijectors`, `_unconstrained_shape`, and `_flow_log_density`, and the init-to-zero patch-up.

### 4.5 Raw functions for compiled code

**The problem.** A sampler's loop is a compiled JAX program, and J1 says compiled programs start at raw methods. ProbPipe's operations return tracked values. Inside `jax.jit`, `unnormalized_log_prob(law, z)` traces and runs, but it returns a `NumericArray`, and `jax.grad` refuses that (verified: "Triggering `__jax_array__()` during abstractification is no longer supported"). Today the built-in methods call private `_unnormalized_log_prob`, and an outside method has no public equivalent.

**The pattern already chosen.** #554 § 7 plans `sampler(d)`: "called outside compiled code and returns a pure function `draw(key, sample_shape=())`", whose output is "raw arrays, so it composes with `jit`, `vmap`, and `grad`". This plan adds its density counterpart:

- `log_prob_function(d, *, normalized=True)` returns a pure function `log_prob(value) -> Array`.
  - With `normalized=False` it reads the unnormalized density, which is what inference needs.
  - For a law over one array (every vector law and unconstrained law), `value` is a raw array.
  - For a conditional distribution, the function takes `given=` as well, as `sampler(K)` does.

**Decision 2: `sampler(d)` in V3, or with #554?**
- **Why it matters.** Without `sampler(d)`, an outside method that simulates (ABC, simulation-based calibration) has no public way to draw inside compiled code, and the built-in pyabc method keeps its private draws.
- **Recommendation.** Ship `sampler(d)` in V3, exactly as #554 § 7 specifies, so the raw draw and the raw density become public together.
- **Alternative.** Ship `log_prob_function` alone in V3, and leave `sampler` to #554.

**Naming (decision 1).**
- `log_prob_function` mirrors the operations `log_prob` and `unnormalized_log_prob`.
- `sampler` was chosen by #554.
- An alternative is a noun matching `sampler`, such as `scorer`. I recommend `log_prob_function`, because it names exactly what it returns.

## 5. Reading results back

An inference method's draws come back in one of two coordinate systems, and each has one route back. Both routes use only § 4.

| The method's chains hold | Route back to values |
|---|---|
| vectors (layer 2), `(num_draws, d)` | `from_vector(spec, chain)`, as a batch |
| unconstrained `z` (layer 4), `(num_draws, d′)` | apply `vector_bijector(spec)`, then `from_vector` |

`make_posterior` (plan 1) takes `coordinates="vector" | "unconstrained"` and applies the route itself, so no method writes it.
- **Columns.** Assembling a posterior's atoms uses `NumericRecordBatch.from_vector` or `NumericArrayBatch.from_vector` on the levels `("chain", "draw")`. This replaces `_record_atoms`, `_array_atoms`, and `_reconstruct_from_vector`'s batch branch.
- **ArviZ variables.** ArviZ's per-leaf variables come from the same batch's columns, which replaces `_leaf_variables`.

Going the other way, from a posterior to vectors, is `posterior.atoms.to_vector()`. This replaces `_flat_chains`, `distributions/_empirical._coordinates`, and `_flat_atoms`.

## 6. The pytree question

*Does flattening to a vector cost BlackJAX anything, and should ProbPipe offer an "array pytree" view?*

**What BlackJAX accepts** (read from the installed BlackJAX 1.5):
- NUTS, HMC, `window_adaptation`, random-walk Metropolis, elliptical slice, SGLD, and SGHMC all accept a position that is any pytree.
- Internally, each ravels the position with `jax.flatten_util.ravel_pytree` wherever it needs a vector:
  - the mass matrix is always a flat `(d,)` or `(d, d)` array in ravel order;
  - random-walk Metropolis's `sigma` and elliptical slice's `mean` and `cov` must be flat.

So the vector is BlackJAX's own internal currency.

**What ProbPipe does today:** a flat vector to every BlackJAX sampler.

**Measured** (warm, jitted `window_adaptation` with 500 steps, then 1,000 NUTS draws, one chain, CPU, JAX 0.10.1, BlackJAX 1.5; flat vector versus a dict of arrays, with the same density):

| Model | Wall time, dict / flat | Per-gradient cost, dict / flat | Fixed-work HMC, dict / flat |
|---|---|---|---|
| regression, 4 parameters | 0.51 | 0.52 | 1.02 |
| hierarchical, 107 parameters | 0.98 | 1.05 | 1.10 |
| hierarchical, 207 parameters | 1.19 | 1.11 | 1.04 |

- **Noise:** the machine was loaded (load average 15 to 18), and repeated runs scattered from 0.73 to 1.6 in both directions.
- **The fairest measure:** the fixed-work comparison shows parity within 2 to 10%.
- **Draws:** they agree in distribution, but they are not bitwise equal across the two layouts.
- **Mass matrix:** it comes out the same, per element, either way.

**Two hazards of a pytree view, both reproduced:**
1. **ProbPipe's own records cannot be BlackJAX positions.** A `Record` or `NumericRecord` position fails inside NUTS: its pytree flattening converts leaves, which breaks when JAX rebuilds it with abstract leaves. Only a plain container of arrays works.
2. **A plain `dict` reorders the vector.** `ravel_pytree` sorts its keys (`{"z": [1, 2], "a": 3}` ravels to `[3, 1, 2]`), so a mass matrix, an initial value, a `proposal_cov`, or an elliptical-slice covariance built in the canonical order would silently misalign. Only a tuple or an `OrderedDict` keeps the order.

**Recommendation: no pytree view now.**
- **No speed to gain.** The measurements show no consistent speed difference.
- **BlackJAX ravels anyway.** It turns every pytree into a vector for every matrix it holds.
- **One layout.** It removes a second layout and its ordering hazard, so `to_vector` stays the one layout in every algorithm.

If a backend later benefits from per-leaf parts, such as TFP's list of state parts or BayesFlow's named arrays, a "parts" view can be derived from the same spec order without a second source of truth. This is recorded in `deferred.md`, not planned now.

## 7. What goes, and what replaces it

| Today (private, in `inference/` unless noted) | Replaced by |
|---|---|
| `_has_flat_view`, `_flat_view`, `_reshape_to`, `_one_array_spec`, `_flat_record_spec`, `_flat_unflatten`, `_check_position_size`, `_no_flat_layout` (`operations/_inference_target.py`) | `to_vector_map`, `vector_law` |
| `flat_density`, `build_target_log_prob`, `build_target_log_prob_flat`, `likelihood_flat` | `log_prob_function(unconstrained_law(target))` or `log_prob_function(vector_law(target))`, and for a likelihood `log_prob_function(likelihood)` with `given=` |
| `flat_vector`, `_declared_vector`, the `to_vector` in `_joint_draw` | `value.to_vector()`; a mapping is first read into a `Record` by its spec (fixing the key-order bug) |
| `_flat_leaves`, `_LeafMap`, `UnconstrainedCoordinates`, `_leaf_maps`, `_segments`, `_coordinates`, `unconstrained_coordinates`, `unconstrained_chain` | `vector_bijector`, `unconstrained_law` |
| BayesFlow `_field_bijectors`, `_unconstrained_shape`, `_flow_log_density`, its use of `_reconstruct_from_vector` | `vector_bijector`, `from_vector` |
| `_array_atoms`, `_record_atoms`, `_leaf_variables`, `_flat_chains`; `distributions/_empirical._coordinates`; `_empirical_reweighting._flat_atoms` | `from_vector` and `to_vector` on batches (§ 5) |
| imports of private `_reconstruct_from_vector` outside `core/` | public `from_vector` |
| KDE's `_unflattened` and `_flattened` (`families/_resampling.py`), mixture `_flat` | `to_vector` and `from_vector` on the batch |

Kept, because each is backend-specific glue and not a second layout:
- `extract_chain_columns`, which reads a trace by name;
- the Stan bridge for BridgeStan's column-major matrices (`_pack_block_params`);
- PyMC's transform table (`_bijector_of`), which maps PyMC's own transforms rather than ProbPipe supports.

## 8. The API, with docstrings

### 8.1 `to_vector_map`

```python
def to_vector_map(spec: NumericSpec) -> Function:
    """The map from a value of *spec* to its vector, ``value.to_vector()``, as a ``Function``.

    The map is a bijection between the values of *spec* and vectors of length
    ``spec.vector_size``: its inverse is ``from_vector(spec, vec)``, and its
    log-determinant of the Jacobian is 0, since it only rearranges entries. It
    claims both, so ``pushforward(d, to_vector_map(spec))`` keeps ``d``'s density
    (:func:`vector_law`).

    Parameters
    ----------
    spec : NumericSpec
        A ``NumericArraySpec`` or ``NumericRecordSpec`` of concrete shape.

    Returns
    -------
    Function
        The map from the input ``value`` (spec *spec*) to the output ``vec``, a
        ``NumericArraySpec((spec.vector_size,), dtype)``, where ``dtype`` is the
        result type of the leaves' dtypes.

    Raises
    ------
    TypeError
        If *spec* is not a numeric spec.
    ValueError
        If a shape of *spec* has a symbolic dimension; the message names
        ``with_dim_sizes``.
    """
```

### 8.2 `vector_law`

```python
def vector_law(d: Distribution) -> Distribution:
    """The law of the vector of a draw of *d*: ``pushforward(d, to_vector_map(spec))``.

    For ``X ~ d``, the vector law is the law of ``X.to_vector()``, whose entries
    are ``X``'s leaves laid out in its spec's order, each raveled row-major. It
    has the density of *d* at ``from_vector(spec, vec)`` where *d* has one,
    samples where *d* samples, and has the moments of the laid-out draws, so
    ``cov(vector_law(d))`` is *d*'s covariance in that layout.

    Parameters
    ----------
    d : Distribution
        A law whose event is a numeric array or a record of numeric arrays, of
        concrete shape (``spec = d.event_spec.term_spec``).

    Returns
    -------
    Distribution
        The vector law, whose event is ``OutputSpec(vec=NumericArraySpec((d,),
        dtype, support))``; ``support`` is the leaves' common support, or none
        when they differ. Its provenance names ``pushforward`` and *d*.

    Raises
    ------
    TypeError
        If *d*'s event is not numeric.
    ValueError
        If *d*'s event has a symbolic dimension.

    Examples
    --------
    >>> prior = MultivariateNormal("beta", jnp.zeros(2), jnp.eye(2)) * Normal("mu", 0.0, 1.0)
    >>> vector_law(prior).event_spec
    OutputSpec(vec=NumericArraySpec(shape=(3,), dtype=float32, support=real))
    """
```

### 8.3 `vector_bijector`

```python
def vector_bijector(spec: NumericSpec) -> Function:
    """The bijector from unconstrained coordinates onto the vectors of *spec*'s support.

    It maps ``z`` in ℝᵈ′ to a vector ``v = b(z)`` of length ``spec.vector_size``
    whose value ``from_vector(spec, v)`` lies in *spec*'s support. Leaf by leaf,
    in the layout's order, it applies ``bijector_for(leaf.support)`` to the
    leaf's segment of ``z``: a leaf whose support is ``real``, or that declares
    none, maps by the identity, and a constrained leaf may take fewer
    coordinates than entries, as a simplex does. It claims its inverse and the
    log-determinant of its Jacobian, the sum of the leaves'.

    Parameters
    ----------
    spec : NumericSpec
        A ``NumericArraySpec`` or ``NumericRecordSpec`` of concrete shape.

    Returns
    -------
    Function
        The bijector, from the input ``z``, a ``NumericArraySpec((d′,))``, to the
        output ``vec``, a ``NumericArraySpec((spec.vector_size,))``.

    Raises
    ------
    MathematicalDomainError
        If a leaf has a discrete support or an integer dtype, which no
        continuous map reaches; the message names the leaf.
    ValueError
        If a shape of *spec* has a symbolic dimension.
    """
```

### 8.4 `unconstrained_law`

```python
def unconstrained_law(d: Distribution) -> Distribution:
    """The law of *d* in unconstrained coordinates.

    It is ``pushforward(vector_law(d), inverse(vector_bijector(spec)))``: the law
    of ``z`` with ``vector_bijector(spec)(z) = X.to_vector()`` for ``X ~ d``. Its
    density at ``z`` is *d*'s at ``from_vector(spec, b(z))`` plus
    ``log |det J_b(z)|``, by the change of variables, so an algorithm that moves
    anywhere in ℝᵈ′ never leaves *d*'s support. A draw ``z`` returns to a value
    of *d* as ``from_vector(spec, vector_bijector(spec)(z))``.

    Parameters
    ----------
    d : Distribution
        A law whose event is a numeric array or a record of numeric arrays, of
        concrete shape, with continuous supports.

    Returns
    -------
    Distribution
        The unconstrained law, whose event is ``OutputSpec(vec=NumericArraySpec(
        (d′,), dtype, real))``.

    Raises
    ------
    TypeError
        If *d*'s event is not numeric.
    ValueError
        If *d*'s event has a symbolic dimension.
    MathematicalDomainError
        If a leaf of *d*'s event has a discrete support or an integer dtype.
    """
```

### 8.5 `log_prob_function`

```python
def log_prob_function(d: Distribution | ConditionalDistribution, *,
                      normalized: bool = True) -> Callable[..., Array]:
    """A pure JAX function of a value that returns *d*'s log-density there.

    Called outside compiled code, it returns ``log_prob(value)`` (for a kernel,
    ``log_prob(value, *, given)``) that takes and returns raw arrays, so it
    composes with ``jax.jit``, ``jax.vmap``, and ``jax.grad``. It captures *d*'s
    parameters as constants. ``normalized=False`` reads the unnormalized density,
    which inference needs. It is the density counterpart of ``sampler(d)``.

    Parameters
    ----------
    d : Distribution or ConditionalDistribution
        The law or kernel.
    normalized : bool
        Whether the function returns the normalized log-density
        (``SupportsLogProb``) or the unnormalized one
        (``SupportsUnnormalizedLogProb``).

    Returns
    -------
    callable
        ``log_prob(value) -> Array`` of shape ``()`` for one value; the value is
        a raw array for a law over one array, and a ``Record`` or a mapping of
        raw arrays for a record law.

    Raises
    ------
    ResolutionError
        If *d* does not claim the density *normalized* asks for.

    Examples
    --------
    >>> u = unconstrained_law(target)
    >>> log_density = log_prob_function(u, normalized=False)
    >>> jax.grad(log_density)(jnp.zeros(u.event_spec.term_spec.shape))
    """
```

### 8.6 `NumericArrayBatch.to_vector` and `from_vector`

These have the signatures of `NumericRecordBatch`'s (III.6), with `to_vector() -> Array` of shape `(*batch_shape, vector_size)`.

### 8.7 `sampler` (if decision 2 puts it in V3)

Exactly as #554 § 7 states it: `sampler(d)` returns `draw(key, sample_shape=())`, and `sampler(K)` returns `draw(key, sample_shape=(), given=...)`. It requires `SupportsSampling`.

## 9. Steps, tests, and what users see

**Steps**, each one PR:
1. **V1, maps:** `to_vector_map`, `vector_bijector`, and `NumericArrayBatch.to_vector`/`from_vector`.
2. **V2, laws:** `vector_law` and `unconstrained_law`.
3. **V3, raw functions:** `log_prob_function`, and `sampler` if decision 2 puts it here.
4. **V4, migration:** every user of the private helpers in § 7 moves onto V1 to V3, and the helpers are deleted.
   - It belongs with plan 1's methods (P2), since those are where the helpers are used.
   - KDE, the mixture, and the empirical laws' coordinates move in V4 too.

**Tests that assert the contract:**
- **Round trips:** `from_vector(spec, to_vector_map(spec)(x)) == x` for arrays, nested records, and every dtype. `vector_bijector(spec)` and its inverse round-trip on the support, and the inverse refuses points outside it.
- **Densities:**
  - `vector_law(d)` at `x.to_vector()` equals `d` at `x`.
  - `unconstrained_law(d)` at `z` equals `d` at the mapped value plus the log-Jacobian, checked against the finite-difference Jacobian.
- **Leaf policy:** each case of § 4.4's policy, including a simplex leaf (`d′ < d`) and a discrete leaf that raises.
- **The reproduced bugs:**
  - A `(2, 2)` family law, a non-family `(2, 2)` law, and a constrained `(2, 2)` law run under NUTS, TFP, and random-walk Metropolis (today they crash).
  - A record in the keyword form is accepted.
  - A mapping given in another key order is laid out correctly.
- **Compiled code:** `log_prob_function` under `jit`, `vmap`, and `grad`.
- **Agreement:** a seeded NUTS run on a record model gives the same draws as today, where today's code is correct (a regression check on the common case).

**What users see.**
- **Added:** the five functions, and `NumericArrayBatch.to_vector`/`from_vector`. These are additive.
- **Behaviour:** a law over one array that is not a family now works under every density method, and a discrete leaf is refused by gradient methods instead of being silently moved and truncated. The CHANGELOG lists both as fixes.
- **No breaking change in this plan;** plan 1 lists its own.

## 10. Questions for the maintainer

1. **Names.**
   - Recommended: `to_vector_map`, `vector_law`, `vector_bijector`, `unconstrained_law`, and `log_prob_function`.
   - Alternatives: III.7's phrase "flat-vector law" for `vector_law`, and `unconstraining_bijector` for `vector_bijector`.
   - The plan avoids the word "flat", which the codebase overuses with four meanings.
2. **`sampler(d)` in V3.** Recommended: ship #554 § 7 there, as specified (§ 4.5).
3. **The pytree view.** Recommended: none now (§ 6).
4. **A leaf that declares no support.** Recommended: treat it as `real`, the identity map (§ 4.4, rule 1), which is today's inference behaviour. The alternative is to refuse it, as BayesFlow does today.
5. **Where the layers live.** Recommended: maps in `functions/_reparameterization.py` beside V.12, and the two laws in a new `families/_vector.py` beside the change of variables. `inference/` then holds no layout code.
