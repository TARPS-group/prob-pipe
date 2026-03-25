# Design Document: Generalized Distribution Hierarchy

**Status:** Draft | **Date:** March 2026 | **Package:** prob-pipe

---

## 1. Purpose and Audience

This document defines the high-level architecture for generalizing prob-pipe's distribution abstraction from array-valued distributions to distributions over arbitrary types (pytrees, functions, other distributions). It is intended as a guide for refactoring the prob-pipe codebase.

**Important context for implementers:** This document defines structure and goals. The prob-pipe codebase already has implementations of several related abstractions (product distributions, joint distributions, expectations, etc.). Integrating this design will require refactoring existing code — the specific changes are not fully enumerated here. The implementer should use this document as the architectural reference and adapt it to the particulars of the existing codebase.

---

## 2. Motivation

prob-pipe currently defines a distribution abstraction modeled on TensorFlow Probability (TFP): distributions over arrays with batch/event shape semantics. This works well for standard probabilistic models but does not naturally accommodate several important use cases:

- **Surrogate modeling in Bayesian inference.** When a surrogate model has internal randomness (e.g., a neural network with random weights), this induces a distribution over posterior distributions. Representing this as a distribution-valued distribution (a "random measure") provides a clean abstraction for quantifying and propagating surrogate uncertainty.

- **Random functions and stochastic processes.** Gaussian processes, neural emulators, and other function-valued random objects are naturally described as distributions over functions. Their finite-dimensional distributions are standard array-valued distributions, but the object itself benefits from a function-valued abstraction.

- **Structured parameter spaces.** Posteriors over pytree-structured parameters (e.g., neural network weight dictionaries) currently require manual flattening. A pytree-valued distribution provides a natural and direct representation.

The goal is to introduce a generalized distribution hierarchy that supports these use cases while preserving full backward compatibility for users who only need standard array-valued distributions.

---

## 3. Design Principles

- **Backward compatibility.** Existing array-valued distribution code should require only a rename (`Distribution` → `ArrayDistribution`) and continue to work. The generalized hierarchy should not complicate the standard case.

- **Minimal base, rich specializations.** The generic base class commits to almost nothing. Shape semantics, batch/event conventions, and domain-specific methods live in specialized subclasses.

- **Composition over inheritance.** Higher-level abstractions (random measures, random functions) compose with and project back to lower-level ones (array-valued distributions). Every "escape hatch" from a non-array distribution leads back to the familiar world of shaped arrays.

- **JAX-native where possible.** The sampling contract is designed so that `jax.vmap`-based batching works by default for pytree-representable types. However, prob-pipe does not require everything to be JAX-compatible, and the hierarchy accommodates non-JAX types at the upper layers.

---

## 4. Type Hierarchy Overview

```
Distribution[T]                           # Layer 0: most general base
│                                         # sample, optional log_prob
│
├── PyTreeDistribution[T]                 # Layer 0.5: treedef contract
│   │                                     # abstract batching, pytree structure
│   │
│   └── PyTreeArrayDistribution[T]        # Layer 1: array leaves, full shape semantics
│       │                                 # batch_shape, event_shapes, flatten/unflatten
│       │
│       ├── ArrayDistribution             # Layer 1.5: T = single Array
│       │   ├── Normal, Gamma, MVN, ...   # (renamed from current Distribution class)
│       │   └── TransformedDistribution
│       │
│       ├── JointDistribution[T]          # joint over pytree components
│       │   └── ProductDistribution[T]    # independent components (existing, generalized)
│       │
│       └── FlattenedView                 # wraps ArrayDistribution + treedef
│
├── RandomMeasure                         # Layer 2: T = Distribution
│   └── SurrogatePosterior, ...
│
└── RandomFunction[X, Y]                  # Layer 2: T = Callable
    └── GaussianProcess, Emulator, ...
```

The hierarchy has three conceptual layers. Layer 0 (`Distribution[T]`) is maximally generic. Layer 0.5 (`PyTreeDistribution`) adds pytree structure. Layer 1 (`PyTreeArrayDistribution` and below) adds full TFP-style shape semantics for array-backed pytrees. Layer 2 (`RandomMeasure`, `RandomFunction`) provides domain-specific abstractions that compose with Layer 1.

---

## 5. Layer 0: Distribution[T]

The root of the hierarchy. Parameterized by value type `T`. Commits only to a sampling contract and an optional log-density.

### 5.1 Interface

```python
T = TypeVar('T')

class Distribution(Generic[T], ABC):

    @abstractmethod
    def _sample(self, key: PRNGKey) -> T:
        """Draw a single sample. Subclasses implement this."""
        ...

    def sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> T:
        """Draw sample(s). Returns a single object of type T,
        where sample_shape is absorbed into the structure of T
        in a type-dependent manner.

        Default implementation uses vmap over _sample. This works
        for any JAX-pytree-representable T. Subclasses with
        non-pytree T or requiring custom batching should override.
        """
        if sample_shape == ():
            return self._sample(key)
        keys = jax.random.split(key, math.prod(sample_shape))
        flat_samples = jax.vmap(self._sample)(keys)
        return jax.tree.map(
            lambda x: x.reshape(*sample_shape, *x.shape[1:]),
            flat_samples
        )

    def log_prob(self, value: T) -> Array:
        """Log-density of value. Optional; raises by default.
        Takes one pytree value in, returns one scalar (or batch of scalars) out.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support log_prob"
        )
```

### 5.2 Design Decisions

- **`_sample` (single draw) is the abstract primitive; `sample` (with `sample_shape`) is derived.** This keeps the subclass contract simple. Subclasses override `sample` for efficiency where needed.

- **`sample_shape` is universal but type-dependent in meaning.** For arrays, it prepends dimensions. For distributions-as-values, it augments batch shape. For pytrees, it prepends dimensions to every leaf. The vmap-based default handles all JAX-pytree-representable types automatically.

- **`log_prob` is optional.** Stochastic processes have no density over function space. Random measures may only have tractable densities in special cases. Array-backed distributions (Layer 1 and below) make it required.

- **No shape properties at this layer.** `batch_shape` and `event_shape` are concepts specific to array/pytree-array-valued distributions. The base class does not know about them.

---

## 6. Layer 0.5: PyTreeDistribution[T]

An intermediate layer for distributions whose values are pytrees. This layer defines the treedef contract and the concept of pytree-structured values, but leaves batching abstract.

### 6.1 Interface

```python
class PyTreeDistribution(Distribution[T]):
    """Distribution over a pytree-structured value.

    Defines the structural contract: values must conform to a
    specific PyTreeDef. Batching semantics are left to subclasses.
    """

    @property
    @abstractmethod
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """The pytree structure of a single sample.
        All samples must conform to this structure.
        """
        ...
```

### 6.2 Design Decisions

- **This layer is lightweight — mostly an interface.** Most users will work with the array-backed subclasses. `PyTreeDistribution` exists to give a subclassing point for distributions over non-array pytrees (graphs, discrete structures, etc.) where JAX-native batching may not apply.

- **Batching is abstract.** `PyTreeDistribution` does not define how `sample_shape` interacts with the pytree structure. For array-leaf pytrees, the answer is "prepend dimensions to every leaf." For non-array-leaf pytrees, the answer is type-specific and left to subclasses (e.g., a Python list of samples).

- **`log_prob` semantics: one pytree in, one scalar out.** When `log_prob` is implemented on a `PyTreeDistribution`, it takes a single pytree value conforming to `treedef` and returns a scalar (or a batch-shaped array, if batching is applicable). The decomposition of `log_prob` into per-component contributions (if meaningful) is the concern of `JointDistribution`, not this layer.

- **Treedef matching is strict.** Two pytrees are considered structurally compatible only if their `PyTreeDef` objects are equal. This includes dictionary key ordering. A canonical ordering should be enforced and clearly documented. Implementers should choose a convention (e.g., sorted keys for dicts) and apply it consistently.

---

## 7. Layer 1: PyTreeArrayDistribution[T]

The workhorse layer. All leaf values are JAX arrays. Full TFP-style batch/event shape semantics apply, generalized to the pytree structure.

### 7.1 Interface

```python
class PyTreeArrayDistribution(PyTreeDistribution[T]):
    """Distribution over a pytree of arrays.

    All leaves are JAX arrays. batch_shape is shared across all leaves.
    Each leaf has its own event_shape. The full 'shape identity' of this
    distribution is: treedef + batch_shape + per-leaf event_shapes.
    """

    @property
    @abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        """Shared across all leaves."""
        ...

    @property
    @abstractmethod
    def event_shapes(self) -> T:
        """Per-leaf event shapes. Returns a pytree with the same
        structure as T, with shape tuples (tuple[int, ...]) at each
        leaf position."""
        ...

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """Event shapes as a flat list in canonical leaf order."""
        return jax.tree.leaves(self.event_shapes)

    @property
    def event_size(self) -> int:
        """Total number of scalar elements in one event (analogous to numpy.ndarray.size)."""
        return sum(math.prod(s) for s in self.flat_event_shapes)

    @abstractmethod
    def log_prob(self, value: T) -> Array:
        """Log-density. Takes a pytree of arrays, returns
        array of shape batch_shape."""
        ...

    # ── Flatten / unflatten ──────────────────────────────────

    def flatten_value(self, value: T) -> Array:
        """Flatten a pytree value to a 1D array.

        Each leaf array (of shape *event_shape) is raveled and the
        results are concatenated in canonical leaf order.
        For batched values (leaves have shape *batch_dims, *event_shape),
        returns array of shape (*batch_dims, event_size).
        """
        ...

    def unflatten_value(self, flat: Array) -> T:
        """Unflatten a 1D (or batched 2D) array back to the
        pytree structure, using treedef and event_shapes."""
        ...

    def as_flat_distribution(self) -> 'ArrayDistribution':
        """View this distribution as a distribution over flat arrays.

        Returns an ArrayDistribution with event_shape=(event_size,).
        Equivalent to applying flatten_value as a bijection.
        This enables interoperability with any algorithm that
        expects flat vectors (MCMC, optimizers, VI methods).
        """
        ...
```

### 7.2 Shape Semantics

The shape semantics follow TFP conventions, applied independently to each leaf:

- Each leaf array in a single sample has shape `(*event_shape_for_that_leaf,)`.
- In a batched context, each leaf has shape `(*batch_shape, *event_shape_for_that_leaf)`.
- Sampling with `sample_shape` prepends those dimensions to every leaf: `(*sample_shape, *batch_shape, *event_shape_for_that_leaf)`.
- `batch_shape` is shared across all leaves. This is enforced; per-leaf batch shapes are not supported (this could be generalized later if needed).
- `log_prob` consumes the event dimensions and returns an array of shape `(*batch_shape,)`.

### 7.3 Flatten / Unflatten

Flattening maps a pytree value to a 1D array by raveling each leaf and concatenating in canonical leaf order. This is not always trivial: an `ArrayDistribution` with `event_shape = (d, d)` (e.g., a distribution over matrices) flattens to dimension `d*d`. For batched values, flattening preserves leading batch dimensions and flattens only the event dimensions, yielding shape `(*batch_shape, event_size)`.

The `as_flat_distribution` method returns an `ArrayDistribution` with `event_shape = (event_size,)` that is mathematically equivalent. This is the primary interoperability mechanism: any algorithm written for `ArrayDistribution` works with `PyTreeArrayDistribution` via flattening.

---

## 8. Layer 1.5: ArrayDistribution

The current prob-pipe distribution class, renamed. A `PyTreeArrayDistribution` where the pytree has a single leaf (a single array).

### 8.1 Interface

```python
class ArrayDistribution(PyTreeArrayDistribution[Array]):
    """Distribution over a single array.

    This is the standard TFP-style distribution. Renamed from the
    current Distribution class.
    """

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single event. Alias for the single
        leaf in event_shapes."""
        ...

    @property
    def event_shapes(self) -> Array:
        """Inherited from PyTreeArrayDistribution.
        Returns the single event_shape (since there is one leaf)."""
        # derived from event_shape
        ...

    @abstractmethod
    def log_prob(self, value: Array) -> Array: ...

    # All existing methods: mean, variance, cdf, entropy, etc.
    # remain exactly as they are.
```

### 8.2 Migration Notes

- The current `Distribution` class is renamed to `ArrayDistribution`. The name `Distribution` is freed up for the generic base class.
- `event_shape` is the canonical property (preserving current prob-pipe and TFP conventions). `event_shapes` is derived from it.
- `flatten_value` inherited from `PyTreeArrayDistribution` ravels the event dimensions: an `event_shape = (d, d)` flattens to a 1D array of length `d*d`. A batched value with shape `(*batch_shape, d, d)` flattens to `(*batch_shape, d*d)`.
- All existing concrete distributions (`Normal`, `Gamma`, `MVN`, etc.) become subclasses of `ArrayDistribution`. Their implementations do not change.
- `TransformedDistribution` and related machinery continue to work at this layer.

---

## 9. Joint and Product Distributions (Pytree-Generalized)

prob-pipe already has `JointDistribution` and `ProductDistribution` implementations. These should be generalized so that joint distributions can be defined using pytrees, where different pytree nodes correspond to component distributions.

### 9.1 JointDistribution[T]

A `PyTreeArrayDistribution` whose pytree structure naturally decomposes into component distributions. Each component corresponds to a subtree (often a single leaf) of the full pytree.

Key capabilities:
- **Component-wise log_prob.** Evaluate the log-density contribution of individual components or groups of components. The total `log_prob` is the sum of component contributions (plus any cross-component coupling terms).
- **Marginal sampling.** Sample from the marginal distribution of individual components without sampling the full joint.
- **Structured construction.** Build a joint distribution by specifying the component distributions and the pytree structure that organizes them.

```python
class JointDistribution(PyTreeArrayDistribution[T]):
    """A joint distribution with pytree-structured components.

    Each component is identified by its position in the pytree.
    Supports component-wise log_prob evaluation and marginal sampling.
    """

    def component_log_prob(self, value: T) -> T:
        """Per-component log-density contributions.
        Returns a pytree with the same structure, scalars at leaves."""
        ...

    def marginal(self, component_path) -> ArrayDistribution:
        """Marginal distribution of a specific component."""
        ...
```

### 9.2 ProductDistribution[T]

A `JointDistribution` where all components are independent. The `log_prob` is the sum of per-component log-probs with no coupling terms. This is the generalization of prob-pipe's existing `ProductDistribution`.

```python
class ProductDistribution(JointDistribution[T]):
    """Joint distribution from independent component distributions.

    Constructed from a pytree of ArrayDistribution objects.
    """

    def __init__(self, distributions: T):
        """distributions: a pytree where each leaf is an
        ArrayDistribution. The resulting ProductDistribution
        has the same pytree structure, with values at each leaf.
        """
        ...
```

Common use cases:
- Combining independent parameter-group distributions into a structured posterior: `ProductDistribution({'weights': normal_dist, 'noise': gamma_dist})`.
- Defining independent priors over structured parameter spaces.
- Any situation where the user wants a single distribution object over a pytree of values, constructed from per-leaf distributions.

### 9.3 Implementation Guidance

The existing `JointDistribution` and `ProductDistribution` in prob-pipe should be refactored to fit this hierarchy. The core functionality likely already exists; the refactoring is primarily about: (a) making them inherit from `PyTreeArrayDistribution` rather than the current base, (b) generalizing the component structure from whatever it currently is to pytrees, and (c) ensuring the `event_shapes`, `batch_shape`, and `flatten/unflatten` contracts are satisfied. The implementer should examine the existing code and propose a design that preserves current functionality while fitting the new hierarchy.

---

## 10. Layer 2a: RandomMeasure

A distribution whose values are themselves distributions. The primary abstraction for surrogate uncertainty quantification.

### 10.1 Interface

```python
class RandomMeasure(Distribution['ArrayDistribution']):
    """A distribution over distributions (a random measure).

    sample() returns an ArrayDistribution (or batch thereof).
    """

    @abstractmethod
    def _sample(self, key: PRNGKey) -> ArrayDistribution:
        """Draw a single distribution from the random measure."""
        ...

    def marginal(self) -> ArrayDistribution:
        """Marginalize over the outer randomness.
        Returns p(x) = E_Q[q(x)] as a mixture."""
        raise NotImplementedError

    def predictive(self, key: PRNGKey, n_outer: int) -> ArrayDistribution:
        """Monte Carlo approximation to the marginal.
        Draws n_outer distributions and forms an equal-weight mixture."""
        components = self.sample(key, (n_outer,))
        return Mixture(components, weights=jnp.ones(n_outer) / n_outer)
```

### 10.2 Sampling Shape Contract

When `sample(key, (k,))` is called on a `RandomMeasure`, the result is a single `ArrayDistribution` object whose parameters have a leading batch dimension of `k`. This follows from the universal pytree batching rule: an `ArrayDistribution` is a pytree of parameters, and `sample_shape` is absorbed into the leading axes of every leaf. The result is an `ArrayDistribution` with `batch_shape = (k, *original_batch_shape)`.

This is consistent with TFP conventions: a "batch of distributions" is a single distribution with enriched batch dimensions, not a collection of separate objects.

---

## 11. Layer 2b: RandomFunction[X, Y]

A distribution over functions. The primary interface is callable: given inputs, the object returns a distribution over outputs. This generalizes the existing Emulator class in prob-pipe.

### 11.1 Interface

```python
X = TypeVar('X')  # input type
Y = TypeVar('Y')  # output type

class RandomFunction(Distribution[Callable[[X], Y]], Generic[X, Y]):
    """A distribution over functions f: X -> Y.

    The primary interface is __call__, not sample.
    Calling the object with inputs returns a distribution over outputs.
    """

    @abstractmethod
    def __call__(self, x: X) -> Distribution[Y]:
        """Return the distribution over outputs at input x.
        This is the fundamental interface."""
        ...

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Shape of a single input point (array-array case)."""
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of a single output point (array-array case)."""
        raise NotImplementedError
```

### 11.2 Calling Convention and Shape Contract (Array-Array Case)

For `RandomFunction[Array, Array]`, the callable interface follows a specific shape contract. Given input `X` with shape `(*extra_batch, n, *input_shape)`, the returned distribution produces samples of total shape `(*sample_shape, *extra_batch, n, *output_shape)`.

Two boolean flags (`joint_inputs`, `joint_outputs`) control how the `n` and `output_shape` dimensions are allocated between batch and event dimensions of the returned distribution:

| `joint_inputs` | `joint_outputs` | `event_shape`         | `batch_shape`                    |
|:--------------:|:---------------:|:---------------------:|:--------------------------------:|
| False          | False           | `()`                  | `(*extra_batch, n, *output_shape)` |
| True           | False           | `(n,)`                | `(*extra_batch, *output_shape)`    |
| False          | True            | `(*output_shape,)`    | `(*extra_batch, n)`                |
| True           | True            | `(n, *output_shape)`  | `(*extra_batch,)`                  |

In all modes, the total shape of a sample is invariant. The flags only control which axes are jointly modeled (event, meaning `log_prob` sums over them) versus treated as independent (batch).

This shape contract is specific to the `RandomFunction[Array, Array]` case. For generalized input/output types, the contract would need to be restated in terms of whatever structure `X` and `Y` have.

### 11.3 The sample Method on RandomFunction

The `sample` method is **not abstract** on `RandomFunction`. The base class does not define `_sample`. Calling `sample` on a `RandomFunction` would mean drawing an actual function from the distribution, which for infinite-dimensional models (e.g., Gaussian processes) requires either approximation or state-tracking for consistent path sampling.

Subclasses with finite-dimensional parameterization can naturally implement `_sample`, since drawing a function reduces to drawing finite parameters:

```python
class NeuralEmulator(RandomFunction[Array, Array]):
    def _sample(self, key):
        weights = self.weight_dist.sample(key)
        return lambda x: self.forward(weights, x)

    def __call__(self, x):
        # Returns distribution over outputs, marginalizing over weights
        ...
```

For batched function samples, `sample(key, (k,))` returns a callable whose outputs have a leading sample dimension of `k`, consistent with the vmap-based default.

### 11.4 Relationship to Existing Emulator

The existing `Emulator` class in prob-pipe already implements the callable interface and the shape contract described above. Refactoring to inherit from `RandomFunction[Array, Array]` should be largely structural. The implementer should examine the existing `Emulator` and determine the minimal changes needed to fit the new hierarchy while preserving all current functionality.

---

## 12. Universal Sampling Shape Contract

A single rule governs batched sampling across the entire hierarchy:

> **`sample_shape` is absorbed into the leading axes of every leaf of the pytree representation of the value T.**

| Value type T     | `sample(key)` returns                               | `sample(key, (k,))` returns                                   |
|:----------------:|:----------------------------------------------------:|:--------------------------------------------------------------:|
| Array            | Array of shape `(*event_shape,)`                     | Array of shape `(k, *event_shape)`                             |
| PyTree of arrays | Pytree; each leaf shaped per event_shapes            | Same pytree; each leaf gets leading dim `k`                    |
| Distribution     | A Distribution with `batch_shape`                    | A Distribution with `(k, *batch_shape)`                        |
| Callable         | A function `X → Y`                                  | A function `X → Y` with leading dim `k` in output             |

The unifying mechanism is `jax.vmap`: the default implementation vmaps the single-draw `_sample` over split keys, then reshapes the leading axis of every pytree leaf to match `sample_shape`. This works automatically for any JAX-pytree-representable type.

Types that are not pytree-representable (variable-length lists, graphs) cannot use the vmap-based default. For these, the subclass must override `sample`. This is an acceptable edge case in a JAX-centric package.

---

## 13. Implementation Plan

### Phase 1: Introduce Distribution[T] and rename existing class

Define `Distribution[T]` as the generic base with `_sample`/`sample`/`log_prob`. Rename the current `Distribution` class to `ArrayDistribution`, inheriting from the new base. All existing tests must continue to pass. This is the highest-priority phase, as it establishes the foundation for everything else.

### Phase 2: PyTreeDistribution and PyTreeArrayDistribution

Implement `PyTreeDistribution` (treedef contract, abstract batching) and `PyTreeArrayDistribution` (array leaves, full shape semantics, flatten/unflatten). Make `ArrayDistribution` inherit from `PyTreeArrayDistribution`. Implement `as_flat_distribution` and the inverse construction path. Enforce canonical treedef ordering. This is likely the most design-intensive phase.

### Phase 3: Generalize JointDistribution and ProductDistribution

Refactor existing `JointDistribution` and `ProductDistribution` to use pytree structure for components and to inherit from `PyTreeArrayDistribution`. Preserve all existing functionality while enabling pytree-structured joints. Add `component_log_prob` and marginal access.

### Phase 4: RandomMeasure

Implement the `RandomMeasure` base class and a concrete `SurrogatePosterior` subclass. This validates the distribution-valued sampling contract and the interaction between the base layer and array-valued distributions.

### Phase 5: Formalize RandomFunction from existing Emulator

Refactor the existing `Emulator` class to inherit from `RandomFunction[Array, Array]`. The callable interface and shape contract are already implemented; this phase adds the type-generic framing and ensures interoperability with `RandomMeasure`.

### Phase 6: GP and additional RandomFunction subclasses

Implement a `GaussianProcess` as a `RandomFunction` to validate the abstraction in the infinite-dimensional case, including the question of optional `sample` support.

---

## 14. Open Questions and Areas for Further Design

### 14.1 PyTreeDistribution: independence assumptions in JointDistribution

`ProductDistribution` assumes independence across components, making per-component `log_prob` and marginal sampling trivial. For `JointDistribution` with cross-component dependencies (e.g., a full-rank MVN unflattened into a pytree), `component_log_prob` and `marginal` require actual marginalization. The interface should be clear about when these operations are exact versus approximate, and what happens when they're intractable.

### 14.2 Treedef rigidity and canonical ordering

Treedef matching is strict: two pytrees are structurally compatible only if their `PyTreeDef` objects are equal, including dictionary key ordering. A canonical ordering convention must be chosen (e.g., sorted keys for dicts) and documented clearly. This convention should be established early and applied consistently across the codebase. Users should be warned that constructing values with inconsistent key ordering will cause errors.

### 14.3 RandomFunction: generality of the type signature

The shape contract (the `joint_inputs`/`joint_outputs` table) is specific to `RandomFunction[Array, Array]`. For generalized input/output types (graphs, structured inputs, multi-modal inputs), the contract would need to be restated. How much of this generality should be built into the base class versus left to subclasses? The current recommendation is to leave it to subclasses and keep the base minimal.

### 14.4 RandomFunction: sample semantics for infinite-dimensional models

For a Gaussian process, `sample(key)` should conceptually return a function, but consistent path sampling (where `f(x1)` and `f(x2)` are jointly consistent regardless of evaluation order) requires either: (a) maintaining internal state (a growing cache of evaluated points), which is stateful and un-JAX-like, or (b) committing to a finite representation (weight-space approximation, Karhunen-Loève truncation), which is lossy. The current resolution is to make `sample` optional on `RandomFunction`. The question of what to do when a user calls it on a GP-like object needs a clear answer — likely a helpful error message with guidance on alternatives.

### 14.5 RandomMeasure: log_prob tractability

When is `log_prob` meaningful on a `RandomMeasure`? If the inner distribution is parameterized by a finite-dimensional vector θ and the random measure is a distribution over θ, then `log_prob` is the density of θ evaluated at the parameters of the given distribution. This requires a canonical parameterization and a way to extract θ from a Distribution object. Whether `RandomMeasure` should provide machinery for this or leave it to subclasses is an open question.

### 14.6 Interaction with transformations and bijectors

TFP's `TransformedDistribution` applies a bijection to an array-valued base distribution. A generalized pushforward `Pushforward(base_dist, f)` where `f: T → S` would produce a `Distribution[S]` from a `Distribution[T]`. This is straightforward for `sample` (apply `f`), but `log_prob` requires the inverse and log-det-Jacobian, which may not exist for general `T` and `S`. The design of a general transformation layer is a potential future extension.

### 14.7 Serialization and pytree registration

All distribution objects need to be valid JAX pytrees for use with `jit`, `vmap`, and other JAX transformations. `Distribution[T]` for general `T` needs this treatment. The implications for custom types that may not be JAX-friendly should be considered. The existing prob-pipe approach to pytree registration should be extended to the new classes.

### 14.8 Naming conventions

The current proposal renames the existing `Distribution` to `ArrayDistribution` and uses `Distribution[T]` for the generic base. Alternative names for the base were considered (`DistributionBase`, `GenericDistribution`, `Stochastic[T]`, `Sampleable[T]`), but `Distribution` is the most natural and readable. `RandomMeasure` and `RandomFunction` are well-established terms in probability theory. Naming of intermediate layers (`PyTreeDistribution`, `PyTreeArrayDistribution`) is descriptive but somewhat verbose — shorter alternatives may be worth considering if they don't sacrifice clarity.
