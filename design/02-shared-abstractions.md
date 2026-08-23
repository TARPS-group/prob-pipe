# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on: generic, type-agnostic machinery, one piece at a time, in dependency order.

## II.0 — Overview: the shared abstractions

The shared abstractions, in dependency order. Each is generic and type-agnostic, defined once and reused throughout the library:

| § | Layer | Abstraction | Role |
|---|---|---|---|
| II.1 | Structure | `NamedTree` | The named, ordered tree addressed by path that every structured object is built on, owning the leaf-keyed mapping contract and navigation. |
| II.2 | Type | `TermSpec` | The term specifications every field and declaration is typed by — one spec per kind, with the `InputSpec` / `OutputSpec` declarations beside them — and the storage rule: a tracked term stores its type as its spec. |
| II.3 | Schema | `RecordSpec` | The named tree of term specifications that is at once the record kind's spec and the schema of one structured value, with numeric promotion and symbolic dimensions. |
| II.4 | Identity | `TrackedTerm` / `Annotated` / `Provenance` | The name, type (spec), lineage, and annotations an object carries beyond its raw representation. |
| II.5 | Multiplicity | `Batch` | The generic multiplicity axis: an indexed collection of *separate* objects, distinct from one object over a structured space. |
| II.6 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved. The shared mechanism behind converters, inference selection, and bijector factories. |

## II.1 — `NamedTree`

### Contract

Values in ProbPipe are represented as named, ordered trees.  ProbPipe uses the following standardized **terminology** to refer to components of these trees:
- A **field** is one named leaf — a single object in the collection.
- A **path** is a `/`-joined sequence which can address either a field or an interior node.
- A **key** is a path that specifically addresses a *field*. Every key is a path but a path for an interior node is not a key.
- A **child** of a node is an entry directly under that node.
- The **canonical order** of a tree is a depth-first walk visiting children in insertion order.

Naming, addressing, traversal, and structure-preserving transforms are defined once in the `NamedTree` class. Specifically, a `NamedTree` implements the `Mapping` interface, with keys exactly its leaf paths, in canonical order, so `keys()`, `values()`, `items()`, `len`, `in`, and `[]` all agree on that one key set, as for a plain `dict`. A leaf path may be equivalently written as a string (`"a/b"`) or a tuple (`("a", "b")`). Since interior nodes are *not* keys, `[]` raises on an interior path. Interior nodes are instead accessed via either the one-level `children` view or `at_path`, which can access any leaf or subtree. So, for example, the invariants  `x.children["a"].children["b"] == x.at_path("a", "b") == x.at_path("a/b")` hold. Sibling names are distinct, so every path identifies at most one node. Distinct subtrees may reuse a name, as in `a/c` and `b/c`, and a bare name is then ambiguous on its own.

```python
class NamedTree[L]:
    # mapping interface — keyed by FIELD path
    def __getitem__(self, key: str | tuple[str, ...]) -> L: ...
    def __contains__(self, key: str | tuple[str, ...]) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...  # field paths
    def __len__(self) -> int: ...             # field count
    def keys(self) -> Iterable[str]: ...      # field paths, canonical order
    def values(self) -> Iterable[L]: ...
    def items(self) -> Iterable[tuple[str, L]]: ...

    # tree navigation — ranges over ALL paths
    def at_path(self, *path: str) -> L | Self: ...    # at_path("a", "b") == at_path("a/b")
    @property
    def children(self) -> Mapping[str, L | Self]: ...
    def is_field(self, path: str | tuple[str, ...]) -> bool: ...
    @property
    def is_multi_field(self) -> bool: ...   # True when the tree holds more than one field

    # structure-preserving transforms — return the same family
    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename or move nodes, old -> new; keys are paths, or bare names when unambiguous;
    # a new name may itself be a path, which moves the node there (see Notes)
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str | tuple[str, ...], leaf: L | Self) -> Self: ...
    def merge(self, other: Self) -> Self: ...              # a key collision raises
    def without(self, path: str | tuple[str, ...]) -> Self: ...

    # serialization (export; the constructor reads a nested mapping back)
    def to_nested_dict(self) -> dict[str, Any]: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...         # the family's own node type
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...   # the family's declared leaf type
```

### Rationale

Using named paths is necessary to satisfy `C5 – Naming for unambiguous meaning`. Housing the collection contract in one shared class ensures `D7 – Single source of truth` is satisfied — the type- and value-level structures built on it cannot drift apart on how a field is named or a path is resolved.
### Notes

- *Same-family closure.* Every navigator and transform returns the same family, enforced by `_node_type()`.
- *Leaf type versus node type.* The parameter `L` declares the leaf type, the only axis on which tree families differ: it is what `values()`, `[]`, and `map` traffic in. The node type is not a second parameter, since interior nodes are always instances of the family's own class, which is what `_node_type()` reports. The runtime partition therefore uses the node type alone: a field value is an interior node when it is an instance of `_node_type()`, and a leaf otherwise. Validation is handled once in `NamedTree` rather than by each family: construction checks every leaf against the family's declared leaf type, reported by `_leaf_type()`, so a malformed tree fails at construction rather than at first navigation. A family whose leaves are arbitrary values declares `object`, making the check vacuous.
- *Navigation yields views.* `children`, `at_path`, and `[]` return a subtree or leaf that is a *view* into the same underlying store, derived on demand rather than copied out.
- *Mappings are never leaves.* Construction materializes a mapping-valued field into a subtree, so a nested mapping and its `to_nested_dict` export round-trip faithfully through the constructor — there is no separate nested-dict reader.
- *Mapping protocol, not `Mapping` ABC.* `NamedTree` implements the `Mapping` interface but is deliberately **not** a `collections.abc.Mapping` instance: construction detects tree structure with `isinstance(value, Mapping)`, which must capture a mapping-valued leaf to materialize while leaving a nested tree to nest as a child.
- *Abstract substrate.* `NamedTree` provides no constructor logic and is not directly instantiable; each concrete family's constructor owns storage and validation policy.
- *Bare-name reference.* `with_path_names` renames by `old="new"` pairs, where a key may be a bare name instead of a full path: a bare name resolves to the unique node so named and raises when the tree contains it more than once. Keyword pairs therefore cover the common case, while the positional mapping form addresses any path, as in `with_path_names({"group1/mu": "loc"})`. The `Mapping` interface itself is untouched, with `[]` and `keys()` keyed by full path.
- *Path-valued targets.* A rename target may be a path rather than a bare name: `with_path_names({"group/mu": "mu"})` moves the field to the top level, and `{"mu": "group/mu"}` moves it under `group`, creating intermediate nodes as needed. Rename and move are one operation because both relabel the same store: the leaves are untouched, and only the paths that address them change. A move onto an existing path raises, exactly as a rename onto an existing sibling does, and an interior node emptied by a move is dissolved, since a tree holds no empty subtrees.

## II.2 — Term specifications: `TermSpec`, `InputSpec`, `OutputSpec`

### Contract

A **term specification** types one term: what a single field may hold, and what an operation's result is declared to be. `TermSpec` is the base class and declares validation once. Every kind has exactly one spec class — `NumericArraySpec` and `OpaqueSpec` below, `RecordSpec` (II.3), `BatchSpec`, and the `FunctionSpec`, `DistributionSpec`, and `ConditionalDistributionSpec` completed in Part III — and **every term spec has exactly one tracked class and one batch form**: `NumericArraySpec` has `NumericArray` and `NumericArrayBatch`, `OpaqueSpec` has `Opaque` and `OpaqueBatch`, and `RecordSpec` has `Record` and `RecordBatch` (III.1). The spec's class is the kind.

```python
class TermSpec(ABC):
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...

class NumericArraySpec(TermSpec):  # a numeric array
    shape: tuple[int | str, ...]   # a str names a symbolic dimension (II.3)
    dtype: DType
    support: Constraint

class OpaqueSpec(TermSpec):        # the fallback spec; is_valid accepts any non-mapping value
    meta: Hashable
```

The support is a `Constraint`, an elementwise membership test. A constraint compares and hashes by value, so an instance can serve as a registry key.

```python
class Constraint(ABC):
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # elementwise membership
```

Every spec accepts a value in either presentation: the raw representation, such as a bare array at a `NumericArraySpec` position, or the tracked term of the spec's kind, which construction normalizes to the stored form. Access returns the tracked term and `raw` returns the stored representation (III.2, V.0), so the raw/tracked distinction is a storage detail rather than an interface one.

A batch is a tracked term too, so it has a spec of its own: `BatchSpec` types the *collection*, carrying the spec its elements satisfy together with the named multiplicity. A batch's own type is not its element's type, which is what keeps the storage rule below universal — `OpaqueBatch`, whose elements carry no structure of their own, still has one.

```python
class RecordSpec(NamedTree[TermSpec], TermSpec): ...   # the named tree of term specs — the schema (II.3)

class FunctionSpec(TermSpec):      # a callable; is_valid accepts any callable
    input_spec: InputSpec | None   # None: that side's structure unspecified
    output_spec: OutputSpec | None

class BatchSpec(TermSpec):         # a Batch; is_valid accepts a matching batch
    element_spec: TermSpec                          # the spec every element satisfies
    axis_groups: tuple[tuple[int | str, ...], ...]  # the multiplicity, tiled into levels (II.5);
                                                    #   a str names a symbolic dimension, as in NumericArraySpec
    level_names: tuple[str, ...]
# DistributionSpec and ConditionalDistributionSpec are the remaining kind specs (Part III).
```

The map-like kinds declare their two sides with a pair of declarations that are not term specs themselves — they type no field and have no tracked class or batch form — but appear inside `FunctionSpec` and `ConditionalDistributionSpec`:

```python
class InputSpec(Mapping[str, TermSpec]): ...   # named slots; keys are Python identifiers

class OutputSpec:
    name: str        # an identifier naming the produced term
    spec: TermSpec
```

An `InputSpec` is a flat mapping from names to term specs: the independently bindable slots a map-like term takes in, with a structured slot declared by a `RecordSpec` value. It is flat because slots are bound independently and met from different sources — the semantics of function arguments rather than of one jointly produced value — and because composition's calculus is set algebra on exactly this shape (III.10). An `OutputSpec` is a term spec plus a required name for the produced term. Every output declaration is named because every tracked term is named (II.4): constructors default the name, captured once from the constructor's `name` argument or the wrapped function's name, and never re-read from a later relabeling. For a record-shaped output the name labels the term only, the fields carrying the produced names; for any other kind the name doubles as the term's field name wherever one is required — in composition's produced slots and in a joint result's layout — so no field name is ever invented downstream.

A term spec plays one of two roles, fixed by its position. **As data**, at a named slot or field, it types what that position holds — a `Distribution` stored inside a record is such a field. **As result**, inside an `OutputSpec`, it states that the result *is* a term of that kind, and the stored class fixes the kind: a `RecordSpec` draws a `Record`, a `DistributionSpec` a `Distribution`, a `NumericArraySpec` a `NumericArray`, never the runtime type of an implementer's return. `Distribution("mu", NumericArraySpec(...))` therefore declares a scalar law whose draws are `NumericArray`s, and `Distribution(name, DistributionSpec(t))` a random measure whose draws are `Distribution`s. `Distribution(name, RecordSpec(x=DistributionSpec(t)))` declares instead a record law whose draws are `Record`s holding a distribution-valued field `x`: the two spaces are isomorphic, but the draw kinds differ, and the declaration alone decides which. A wrong-kind result raises the kind error, a schema mismatch its own.

A declaration is *stored* in its normalized form: a bare mapping is accepted wherever an `InputSpec` is meant, a bare term spec wherever an `OutputSpec` is meant (the name defaulted as above), and nested mapping data wherever a `RecordSpec` is meant. After construction only the stored form remains, so the declared kind is the stored spec's class. The same storage rule holds for the tracked types: each carries the spec of its kind as the single stored source of its type — the slot is `TrackedTerm.spec` (II.4), declared once on the tracked base and narrowed by each kind to its own spec class. Convenience accessors expose that spec's properties, and so differ by kind:

| Type | Accessors |
|---|---|
| `Record` | `spec` — its `RecordSpec` |
| `Function` | `input_spec`, `output_spec` |
| `Distribution` | `event_spec` — an `OutputSpec` |
| `ConditionalDistribution` | `given_spec` — an `InputSpec` — and `event_spec` |

Each accessor is a view on the one stored object, so none can disagree with it.

Two rules govern record-shaped positions, symmetric in what arrives. Mapping data materializes into the record's own structure, under derived identity; a supplied tracked term is stored and keeps its identity (name, provenance, capabilities). Both conform to the same spec, so structure and identity never disagree about what a field is.

A `FunctionSpec` types a callable by its two sides, either optional: `None` leaves a side unspecified, so a bare `FunctionSpec()` describes any callable. Validity is callability alone — the sides document the schema, enforced at the call boundary rather than by `is_valid` — and it is the spec's identity as a `FunctionSpec`, not `is_valid`, that tells the wrap boundary to wrap a raw callable result into a `Function`. The value layer stays **callable-generic**: a `FunctionSpec` admits any callable — a plain lambda, a NumPy function, a `Function` — the `Function` being one such, not the required type, and a `FunctionBatch` holds a collection of them. No operation branches on whether a callable arrived bare or wrapped.

### Rationale

One `is_valid` contract across the kinds keeps validation uniform (`C1 – Uniform interface to distributions and values`). Naming the base for the terms it types is `C5 – Naming for unambiguous meaning` applied to the library's own vocabulary: every spec types a tracked term, and *value* stays reserved for the mathematical kind. Storing every declaration as a spec, and the spec on the term itself, is `D7 – Single source of truth`: the declared kind is a stored class rather than an inference, and every accessor is a view that cannot drift. Requiring a name on every output declaration serves `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows` together: the produced term's name is fixed where the producer is declared, so model structure never rides on a relabelable string.

## II.3 — `RecordSpec`

### Contract

A `RecordSpec` is a `NamedTree` whose leaves are term specifications (II.2): the record kind's spec and the **schema** of one structured value — the shape of one event, such as a draw or a stored datum. One class serves both readings because they denote the same space. Nesting is just nesting: a record-shaped position inside a schema is a subtree, and whether the value arriving there materializes as structure or is stored as a term is decided by the value, per the symmetric rule of II.2, not by a second spec class.

When every leaf is a `NumericArraySpec`, the schema is fully numeric and construction auto-promotes it to a `NumericRecordSpec`. The promotion is re-derived whenever a transform constructs a new schema, so a replacement that removes the last non-numeric leaf promotes the result and one that introduces a non-numeric leaf demotes it: the numeric axis is an invariant of the current leaves, not of the object's history. Beyond the inherited `NamedTree` interface (with `L = TermSpec`), `RecordSpec` adds construction sugar, lossy inference from a value, and the numeric projection:

```python
class RecordSpec(NamedTree[TermSpec], TermSpec):
    def __init__(self, field_specs: Mapping[str, Any] | None = None, /,
                 **fields: TermSpec | Mapping | tuple[int, ...] | None) -> None: ...
    # sugar: a bare shape tuple means NumericArraySpec(shape) and None means OpaqueSpec();
    # the positional mapping form accepts "/"-path keys and names that collide with keywords

    @classmethod
    def infer_from(cls, value: Any) -> RecordSpec: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    @property
    def is_concrete(self) -> bool: ...                   # False when any dimension is symbolic
    @property
    def free_dims(self) -> frozenset[str]: ...           # the unbound symbolic dimensions
    def with_dims(self, **sizes: int) -> RecordSpec: ... # bind them; a new schema
    def numeric_subset(self) -> NumericRecordSpec: ...   # remove non-NumericArraySpec leaves
```

`infer_from` types a term-valued field at its own kind — a `Distribution`-valued field infers a `DistributionSpec`, a callable a `FunctionSpec` — and nested structure as nested structure, so inference never mistypes a term as the raw value it happens to resemble.

`NumericRecordSpec` further provides a flat (vectorized) layout of the leaves:

```python
class NumericRecordSpec(RecordSpec):
    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]: ...   # per-field array shapes, canonical order
    @property
    def vector_size(self) -> int: ...                          # total flat dimension; defined only when concrete
```

**Symbolic dimensions.** A shape entry may be a **named symbolic dimension** instead of an integer. `NumericArraySpec(shape=("obs", "features"))` fixes the rank and gives each dimension an identity while deferring its size. Within one schema a name refers to one dimension: fields `X: ("obs", "features")` and `coefficients: ("features",)` share the dimension `features`, an equality no pair of concrete integers can express. A schema with any symbolic entry is **polymorphic**, with `is_concrete` false and `free_dims` listing the unbound names. Schemas carry no scope object beyond the names themselves, so they serialize as plain data.

Checking a value against a polymorphic schema cannot be done leaf by leaf, because a symbolic name constrains several leaves at once. Validation therefore runs a single pass over all fields, resolving each name to one size: a name is bound the first time it is seen, every later occurrence must agree, and a disagreement raises. Bound names never rebind.

The work splits accordingly. A leaf's `is_valid` checks its own rank and dtype, and nothing else — a `NumericArraySpec`'s `support` is descriptive metadata, unchecked. Sizes belong to the one pass, since only it sees every occurrence of a name.

**Every spec reports, substitutes, and binds its own dimensions.** `TermSpec.free_dims` gives the names one spec declares, and a schema's are the union over its fields; an `InputSpec`'s are the union over its slots, sharing one scope with the output declaration beside it. A nested spec's schema therefore lies inside the scope: a name declared within a `DistributionSpec` is the same dimension as that name beside it, so it binds once and a disagreement raises. The outcome does not depend on the order the fields were declared in. Keeping all three with the spec is what reaches a spec the schema layer cannot name — a batch axis, declared one layer out — so every dimension a schema reports is one some spec can bind.

Binding returns a new schema rather than mutating the original, so refinement is monotone. `with_dims(**sizes)` binds explicitly, naming any dimension left unbound; a value binds by unification. Until every name is bound the schema is not concrete, so a `NumericRecordSpec` has no flat layout: an operation that needs sizes raises, naming the free dimensions.

### Rationale

As the *type layer*, a `RecordSpec` is the explicit structure that travels with a value and with the producers and consumers of values (`D5 – Explicit, carried structure`). One class for the record kind and the schema is `D7 – Single source of truth` at the level of concepts: the reference formerly kept a separate schema class over the same tree, converted it to the kind's spec at every construction site, and left the two able to disagree about whether a position was structure or a term-valued leaf — a distinction the value layer already decides. A symbolic dimension carries a dimension's identity, which is mathematical structure, while deferring its size to the data that determines it, so cross-field equalities travel with the term and sizes bind when their producer appears (`D5 – Explicit, carried structure`, `C3 – Computational detail hidden by default, available on demand`).

---

## II.4 — Identity, type & metadata: `TrackedTerm`, `Annotated`, `Provenance`

### Contract

Identity, type & metadata is the cross-cutting layer that lets any object carry, alongside its raw representation, four things: a **name** (what the object is called), a **spec** (the declaration of its type, II.2), a **provenance** (how it was produced), and free-form **annotations** (auxiliary information supplied by the user or an algorithm). The structure is provided by two mixins: `TrackedTerm` (name, spec, and provenance) and `Annotated` (annotations). Every first-class object, the kind an operation consumes and produces, must be a `TrackedTerm`, while structural helpers such as templates and specs are not. We call any such object a **tracked term**: a value, distribution, conditional distribution, linear operator, or batch that carries a name, its spec, and provenance.

Annotation metadata is a free-form mapping, and the **one exception to
immutability**: the store is written after construction by inference backends,
validators, and diagnostic operations, and mutated in place, so the channel is
append-only by convention — a writer adds under its own key and never overwrites
mathematical state or another writer's entries. Being written in place is also
why a copy takes its own container: otherwise a write on one term would show
through on the term it was copied from.

```python
class Annotated:
    annotations: Mapping[str, Any] | None
```

A tracked term's name must be provided by the user when constructed explicitly (as the required first argument to the constructor). When an operation produces an object, it must provide a meaningful, deterministic name derived from its inputs. The `name_is_auto` flag records which, because the two behave differently: a structure-changing transform re-derives an auto-derived name from its result, and composition into a larger object may rename it again, while a user-given name is preserved. A *nested* object (i.e., a sub-object of a `NamedTree`) takes its name from the field key it sits under. For example, a sub-object reached at `parameters` is itself named `parameters`. `with_name` renames the object itself, unlike `NamedTree.with_path_names`, which renames the fields within it. A name is a human label rather than an identity: nothing resolves an object by name, derived names need no escaping scheme, and two objects may share a name wherever field uniqueness does not force distinctness. The provenance of a tracked term stores pointers to descriptors of the parent objects that created it, along with the operation. Optionally, it can also provide references to the parents themselves.

The `spec` slot is the term's type, stored once (II.2). Each kind narrows it to its own spec class and exposes convenience accessors for its properties.

**A tracked term is immutable, and that is a property of being one** (`C2 – Functional interface over immutable objects`): assignment and deletion raise, naming the class the caller touched, and every transformation returns a new term. `TrackedTerm` therefore carries the immutability itself rather than each kind opting in — one guard, so a subclass cannot report a different rule than its base. Immutability obliges a second thing, since `pickle` and `copy` restore an object by assigning its state back: a term reconstructs by allocating its resolved class and restoring the state it actually holds, rather than by rebuilding through its constructor. So a reconstruction cannot re-derive a schema an explicit declaration had pinned, cannot re-decide a class from arguments the state no longer carries, and cannot omit a field — including one written after construction, which no constructor argument names. A term declares any *memo* it holds as transient, keeping a cache out of the round-trip, and any *store written in place* as decoupled, so a copy takes its own container rather than sharing one. Annotations are the one such store (below). Identity is **boundary-attached** under compiled execution: inside a `jit` or `vmap` trace a term presents as its raw representation with only its spec as static data — name, provenance, and annotations never enter a trace, so a name can never affect compilation-cache identity — and the tracked result is minted at the enclosing call boundary (III.2, V.0).

```python
class TrackedTerm:
    name:         str
    name_is_auto: bool
    spec:         TermSpec                       # the single stored source of the term's type (II.2)
    provenance:   Provenance | None              # write-once via with_provenance(...)
    def with_name(self, name: str) -> Self: ...  # shallow copy with name_is_auto = False
    def with_provenance(self, p: Provenance) -> Self: ...
    # immutable: __setattr__ / __delattr__ raise; state round-trips through the
    # attributes the term holds, so copy and pickle need nothing from the class

class Provenance:
    operation: str                       # the operation that produced the object
    parents:   tuple[ParentInfo, ...]    # descriptors of the tracked inputs
    controls:  Mapping[str, Any]         # the resolved controls: PRNG key, sample count, selected method, ...
    inputs:    Mapping[str, ParentInfo]  # plain (untracked) arguments, keyed by parameter name

class ParentInfo:
    type_name:   str
    name:        str
    fingerprint: str            # best-effort content hash
    fingerprint_is_weak: bool   # True when the fingerprint is only object identity
    parent:      Any | None     # optional reference to original parent
```

A provenance records everything its operation resolved: the tracked parents, the plain arguments, and the **controls** the run actually used, the PRNG key, sample counts, and any selected method among them, so a result can be reproduced from its record. Fingerprints are best-effort, in tiers — a content hash, the qualified name and code hash for a closure-free callable, or object identity, with `fingerprint_is_weak` marking the weakest — and **the provenance mode, not the operation, sets the tier**: the default lightweight mode records identity-tier descriptors for every operation's parents, with no content hashing; the full mode retains parent references and fingerprints content-verifiably, computable lazily at export since the objects are held; off records nothing. Every operation behaves identically under a given mode, so provenance cost is one user-visible dial rather than a property of what ran.

### Rationale

`TrackedTerm` serves the two non-mathematical principles: `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. Housing the spec on the tracked base is `D7 – Single source of truth` for a term's type: one slot, declared once, that every kind's accessors are views on. The guarantee behind `C6 – Traceable and reproducible workflows` is a single rule: **every object a ProbPipe operation natively returns is a tracked term** (whether or not it is also `Annotated`), so the provenance chain is never broken. Recording the resolved controls, not just the parents, is what turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived names keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`). Because identity and metadata are orthogonal to *what* an object is mathematically, they are defined uniformly across classes.

## II.5 — `Batch`

### Contract

A `Batch` is the generic `TrackedTerm` nd array of shape `batch_shape` that holds objects of a common element type. `batch_shape` is nonempty: a batch has at least one batch axis. It could also be `Annotated` if applications for it arise. A concrete batch implementation must specify how to store the elements. Since a batch is a *collection* of its elements, `len` / `iter` / `batch_shape` / `batch_size` operate only on the batch axes. The `batch_*` names are kept deliberately rather than a numpy-style `.shape` / `size`, which could ambiguously cover both the batch axes and the per-element content. A concrete batch adds whatever its element type affords in that element's own section, indexing into the elements' fields included.

**What `[]` dispatches on.** A key is either a **position** or a **name**, and the two namespaces never collide: an axis has no name, and a field no position. A position — an integer, a slice, or a tuple of those — addresses the batch axes, which `Batch` itself answers. A name — a string, or a tuple of strings for a path — addresses a field within every element, which only a batch whose elements have fields can answer; the default reports that they have none, and a batch of records supplies the reading. A tuple mixing the two addresses neither, and is refused as a mix rather than as a wrong number of indices.

```python
class Batch[E](TrackedTerm):
    @property
    def spec(self) -> BatchSpec: ...                     # the single stored source of the type
    @property
    def element_spec(self) -> TermSpec: ...              # view on spec: what every element satisfies
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_size(self) -> int: ...                    # total element count, prod(batch_shape)
    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]: ...   # batch_shape tiled into levels, outermost first
    @property
    def level_names(self) -> tuple[str, ...]: ...       # one name per level, aligned with axis_groups
    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # shallow copy renaming levels old -> new; shapes and elements unchanged, as with_path_names is for fields
    def __len__(self) -> int: ...                       # leading-axis size, batch_shape[0]
    def __iter__(self) -> Iterator[E | Self]: ...       # over the leading batch axis
    def __repr__(self) -> str: ...                      # the class, the name, and each level with its sizes
    def __getitem__(self, key: Any) -> Any: ...         # a position indexes the axes, a name the elements' fields
    def at_levels(self, /, **levels: int | slice | None | tuple[int | slice | None, ...]) -> E | Self: ...
    # index by named level (a view); unnamed levels kept whole, None means the whole axis (:)
    def raw(self) -> Any: ...
    # the storage view: the elements' raw values in their native stacked layout (III.1, III.3)
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**. `axis_groups` tiles `batch_shape` into contiguous groups, outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, flat vectorization above all, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and positional `[]` address the leading **axis**, `batch_shape[0]`, rather than the leading level. The two coincide when the outermost level holds one axis: iterating `(N,)` of `(S,)` then walks the `N`, yielding each inner batch of `S` as a view. When the outermost level spans several axes, the leading axis is only the first of them, so indexing drops that axis and leaves the level in place, one axis shorter. Nesting needs no dedicated classes: a batch is itself a tracked term, so a batch whose elements are batches is already admitted, and grouped storage presents the levels as views into one store.

**The batch's own type.** A batch stores a `BatchSpec` (II.2), which carries the element spec and the named multiplicity together; `element_spec`, `axis_groups`, and `level_names` are views on it. This keeps the storage rule reading the same way for a batch as for any other term — `spec` is the batch's *own* type, not its element's — which is what keeps `OpaqueBatch` well typed even though its elements carry no structure of their own. It also mirrors the model, where a batch inhabits the *family* kind over its element's kind rather than the element kind itself.

**The raw view.** `raw()` returns the storage view: the elements' raw values in their native stacked layout — one array for array-valued elements, an object array for callable- or opaque-valued ones, and the nested mapping of raw columns for record-valued ones (III.3). It is the batch counterpart of the `raw` opt-out on operations (V.0): the tracked batch is the default, and the raw view is the same data with no identity attached.

**A polymorphic multiplicity.** An axis size may be a symbolic dimension name instead of an integer, exactly as a `NumericArraySpec` shape entry may, so a *declaration* can fix the number of levels while deferring how many elements each holds — "returns a batch of `S` draws" before `S` is known. The names share one scope with the element's schema, so a batch of `("n",)` over arrays of shape `("n",)` is square by declaration. A declaration may be polymorphic; a live `Batch` may not, since it holds elements at positions, so construction refuses a spec with free dimensions and `batch_size` is undefined until they are bound.

**Level names.** Each level carries a name, listed in order by `level_names`, and names are unique within a batch. An operation that mints a level takes the name to give it; a name already in use raises, as does a rename onto one, so the caller supplies another. A suffixing rule was considered and rejected: `draw2` would state the order the levels were added in, a fact about the computation path rather than about meaning, and levels are named precisely so that operands align by meaning. `with_level_names` repins names on a shallow copy, shapes and elements unchanged, re-reading its own derived name under the new ones so that it and any view taken from it agree. Renaming a *view* is refused where the new name belongs to a level the view derives its name from but no longer carries, since the derived name would then be ambiguous; `with_name` re-roots it instead. Names must be identifiers, since `at_levels` addresses a level by keyword. Operations align batched operands by level name — two levels meant to correspond under different names are lined up by renaming one — and level names are independent of the field names within an element.

**View identity.** A view of a batch — an element or a sub-batch — derives its name from the batch it was taken from and the positions it selects, naming the level each selection addresses. Take a batch named `posterior`, with a `chain` level of `(4,)` over a `draw` level of `(1000,)`:

```python
posterior.at_levels(chain=0).name           # "posterior[chain=0]"          — a sub-batch of draws
posterior.at_levels(chain=0, draw=7).name   # "posterior[chain=0, draw=7]"  — an element
posterior.at_levels(draw=slice(1, 3)).name  # "posterior[draw=1:3]"         — both levels kept
```

The derived name is marked `name_is_auto`, and it states what was selected rather than how it was reached: levels selected whole are left out, so selecting all of a batch derives the batch's own name, and the levels that do appear are listed in the batch's own order rather than the order they were indexed in. Two routes to one selection therefore read alike, and two different selections never do. Lineage needs no node of its own, since selecting computes nothing — a view carries the lineage of the batch it came out of, and which position it was is already stated by the name. Storage matters at one point only, and there it governs name and lineage together. A batch that *materializes* an element, as a row of columnar storage that does not exist until it is built, gives that element the derived name and its own lineage. A batch that *stores* its elements hands the stored object back untouched, under the name and lineage it already carries: renaming it would mean returning a copy, and a batch that did not produce an object cannot truthfully claim its lineage. A *sub-batch* is derived either way, being the batch's own view rather than an object a caller put there. A stored bare value has no identity of its own to hand back, so selecting it materializes the tracked term of its kind under the derived name, exactly as a materializing batch does.

**Selecting by level.** `at_levels(**levels)` indexes a batch along its named levels and returns a view, the by-name counterpart of positional `[]`. It is the level analogue of `NamedTree.at_path` (II.1), and shares its shape: a path addresses a position and returns a leaf or a subtree, while named level indexers address positions and return an element or a sub-batch. The name is neither `select`, the field-splatting selector on `Record` (III.2) that a batch of records also carries, nor a name promising elements, which it returns only when the selection indexes down to one. Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order, where an integer drops its axis and a slice or `None` keeps it, `None` meaning the whole axis as `:` does. It means that here and only here: a keyword cannot take a `:` literal, so `None` is the spelling this form needs, while positional `[]` writes `:` and refuses `None`, which an unset argument would otherwise leave standing for *all of it*. A level spanning several axes takes one indexer per axis, and a shorter indexer fills the leading axes and leaves the rest whole, so a scalar `draw=i` on a two-axis `draw` level means `draw=(i, None)`. A level whose axes are all dropped is removed, so selecting a single-axis level by an integer — or every axis of a multi-axis level — yields the inner batch or element just as positional indexing and iteration do, while a level left unnamed is kept whole. This parallels xarray's `isel` in indexing by name and by position, with `level_names` in the role of xarray dimension names, and there is no label-based counterpart, since batch levels carry no coordinate labels. The tuple form departs from it: a tuple here addresses a level's axes in order, where `isel` reads one as several positions of a single dimension.

### Rationale

`Batch` is necessary to satisfy `D1 – Mathematical fidelity` by ensuring how many objects there are stays separate from what one object contains. The level structure extends the same fidelity to collections of collections: how many objects sit at each level is a mathematical distinction, so `N` laws with `S` draws each are `(N,)` of `(S,)` rather than one anonymous `(N, S)`. Naming the levels is `C5 – Naming for unambiguous meaning` on that axis, letting a user say which multiplicity is which and letting operations align batches by meaning rather than by position. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`). When those elements are array-backed and claim differentiation, the mapping is vectorized and differentiable end-to-end (`D6 – Differentiability as a capability`). To satisfy `D7 – Single source of truth`, indexing or iterating yields a *view*, the levels of a multi-level batch included. The obligation lands on the concrete batch, since only it knows its storage: the hook that presents a sub-batch shares the store rather than copying out of it, wherever the storage affords sharing. That is also why selecting all of a batch needs no special case — it reaches the same hook as any other selection, and a view costs nothing there.

## II.6 — Dispatch and registries

### Contract

Some operations have many possible implementations, and which one applies depends on the *types* of the objects involved rather than on an object's own class. A **dispatch registry** holds those implementations as named methods and selects one for a given call.

Each **dispatch method** declares a unique `name`, the types it applies to via `supported_types`, a `check` function that probes feasibility without doing significant computation, an `execute` function that performs it, and a `priority` that orders auto-selection. Dispatch is by argument type: a `UnaryDispatchRegistry` keys on the first argument's type, and a `BinaryDispatchRegistry` on the first two. The registry takes the matching methods in priority order and runs the first whose `check` reports feasible. Within one priority, the method whose declared types sit closest to the argument's class in method-resolution order wins, and any remaining tie falls to registration order. A caller can bypass auto-selection and name a method with `method="..."`. New methods are added by registration.

```python
class BaseDispatchMethod(ABC):
    name: str
    priority: int   # defaults to opt-in-only (0)

    @abstractmethod
    def check(self, *args, **kwargs) -> MethodInfo: ...
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any: ...

class UnaryDispatchMethod(BaseDispatchMethod):    # still abstract
    @abstractmethod
    def supported_types(self) -> tuple[type, ...]: ...
class BinaryDispatchMethod(BaseDispatchMethod):   # still abstract
    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]: ...   # (left, right) types

class MethodInfo:
    feasible:    bool
    method_name: str
    description: str

class BaseDispatchRegistry[M: BaseDispatchMethod](ABC):
    # the public interface is concrete; arity subclasses supply key extraction and matching
    def register(self, method: M) -> None: ...
    def execute(self, *args, method: str | None = None, **kwargs) -> Any: ...   # auto-select, or run the named method
    def check(self, *args, method: str | None = None, **kwargs) -> MethodInfo: ...
    def list_methods(self) -> list[str]: ...                           # names, in selection order

class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]): ...    # keys on one argument's type
class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]): ...  # keys on the first two
```

A single **catalog** makes every registry discoverable. It provides a list of registries, their entries with their priorities, and a one-line description, so a user can see which entries exist and how a given call will resolve. An **entry** is one registered item within a registry; the catalog uses this generic term rather than *method* because it spans registries whose items are not all type-dispatched methods. A registry can be added to the catalog if it implements `SupportsRegistryCataloging`. Satisfying the protocol is structural, while membership requires an explicit `register`.

```python
@dataclass(frozen=True)
class EntrySummary:
    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""
    @property
    def is_opt_in_only(self) -> bool: ...

@dataclass(frozen=True)
class RegistryInfo:            # the catalog's per-registry record
    name: str                  # unique within the catalog
    description: str           # one line
    kind: str                  # e.g., "dispatch", "factory", "converter"
    entry_count: int

@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    name: str
    description: str
    kind: str
    def entry_summaries(self) -> list[EntrySummary]: ...
    def describe_entry(self, name: str) -> EntrySummary: ...

class RegistryCatalog:
    def register(self, registry: SupportsRegistryCataloging) -> None: ...   # empty or duplicate names raise
    def names(self) -> list[str]: ...
    def list(self) -> list[RegistryInfo]: ...
    def __getitem__(self, name: str) -> SupportsRegistryCataloging: ...     # get the registry
    def __contains__(self, name: object) -> bool: ...                       # check if a registry exists
    def describe(self, name: str) -> str: ...                               # a readable summary of one registry
registry_catalog = RegistryCatalog()  # the global instance
```

### Rationale

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` reach operations whose implementation cannot be chosen from a single object alone. The `check` probe keeps auto-selection safe, while `method="..."` leaves the choice in the user's hands. New implementations join by registering a method, so the supported set grows without touching the call sites that use it, which is `D2 – Generality first`. Gathering every registry under one catalog serves `D7 – Single source of truth`: there is one place to see which implementations exist and how a call resolves.
