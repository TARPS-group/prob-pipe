# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on: generic, type-agnostic machinery, one piece at a time, in dependency order.

## II.0 — Overview: the shared abstractions

The shared abstractions, in dependency order. Each is generic and type-agnostic, defined once and reused throughout the library:

| § | Layer | Abstraction | Role |
|---|---|---|---|
| II.1 | Structure | `NamedTree` | The named, ordered tree addressed by path that every structured object is built on, owning the leaf-keyed mapping contract and navigation. |
| II.2 | Type | `ValueSpec` / `TermSpec` | The value specifications every field and declaration is typed by, and the storage rule: a tracked term stores its type as its spec. |
| II.3 | Schema | `EventTemplate` | The named tree of value specifications that gives one event its shape, with numeric promotion and symbolic dimensions. |
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
    # rename nodes, old -> new; keys are paths, or bare names when unambiguous
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

## II.2 — Value specifications: `ValueSpec`, `TermSpec`

### Contract

A **value specification** types one value: what a single field may hold, and what an operation's result is declared to be. The base class declares validation once:

```python
class ValueSpec(ABC):
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...
```

Value specs come in two families. A **raw-value spec** types a plain value that names no ProbPipe kind: a numeric array or an opaque Python object. An `ArraySpec` declares a shape, a dtype, and a support. An `OpaqueSpec` is the fallback, admitting any non-mapping value.

```python
class ArraySpec(ValueSpec):  # a numeric array leaf
    shape: tuple[int | str, ...]   # a str names a symbolic dimension (II.3)
    dtype: DType
    support: Constraint

class OpaqueSpec(ValueSpec):  # the fallback spec; is_valid accepts any non-mapping value
    meta: Hashable
```

The support is a `Constraint`, an elementwise membership test. A constraint compares and hashes by value, so an instance can serve as a registry key.

```python
class Constraint(ABC):
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # elementwise membership
```

A **term spec** types a tracked term, one concrete class per kind. `TermSpec` subclasses `ValueSpec` as a marker, since a tracked term is still one value that can occupy a leaf. Two consequences follow: `is_valid` stays declared once, on `ValueSpec`, and anywhere a leaf is accepted, a term spec is accepted unchanged.

A batch is a tracked term too, so it has a term spec of its own: `BatchSpec` types the *collection*, carrying the spec its elements satisfy together with the named multiplicity. It is what keeps the storage rule below universal — a batch's own type is not its element's type, and `OpaqueBatch`, whose elements name no kind at all, still has one.

```python
class TermSpec(ValueSpec): ...      # marker; adds nothing, is_valid inherited

class RecordSpec(TermSpec):  # a Record; is_valid accepts a matching Record
    event_template: EventTemplate

class FunctionSpec(TermSpec):  # a callable; is_valid accepts any callable
    input_template: EventTemplate | None   # None: that side's structure unspecified
    output_spec: ValueSpec | None          # the output declaration, stored as a spec;
                                           #   construction wraps an EventTemplate as RecordSpec(template)

class BatchSpec(TermSpec):     # a Batch; is_valid accepts a matching batch
    element_spec: ValueSpec               # the spec every element satisfies
    axis_groups: tuple[tuple[int, ...], ...]   # the multiplicity, tiled into levels (II.5)
    level_names: tuple[str, ...]
# DistributionSpec and ConditionalDistributionSpec are the other two term specs (Part III).
```

Specs and event templates are mutually recursive: a template's leaves are specs, and a kind spec's parameters are templates (II.3). Both are finite trees, so the recursion grounds.

A term spec plays one of two roles, fixed by its position.

- **As data.** At a named leaf of a template, a term spec types a field that holds a term. A `Distribution` stored inside a record is a leaf value.
- **As result.** As an output declaration, a term spec states that the result *is* a term of that kind. A fitted mapping declares that it returns a `Function` this way.

A `Distribution`'s event declaration is an output declaration: it types what `sample` returns. The two roles therefore settle every draw kind. `Distribution(name, DistributionSpec(t))` declares a random measure, whose draws are `Distribution`s. `Distribution(name, EventTemplate(x=DistributionSpec(t)))` declares a record law, whose draws are `Record`s holding a distribution-valued field `x`.

The two spaces are isomorphic, but the draw kinds differ. The declaration alone decides which, never the runtime type of `_sample`'s return. A wrong-kind result raises the kind error, a schema mismatch its own.

A declaration is *stored* as a spec. A bare `EventTemplate` is accepted wherever a record declaration is meant, and construction wraps it as `RecordSpec(template)`. The two forms denote the same space, and after construction only the spec remains, so the declared kind is the stored spec's class.

The same storage rule holds for the tracked types. Each carries the spec of its kind as the single stored source of its type: the slot is `TrackedTerm.spec` (II.4), declared once on the tracked base and narrowed by each kind to its own spec class. Convenience accessors expose that spec's properties, and so differ by kind:

| Type | Accessors |
|---|---|
| `Record` | `event_template` |
| `Function` | `input_template`, `output_spec` |
| `Distribution` | `event_spec` |
| `ConditionalDistribution` | `given_template`, `event_spec` |

Each accessor is a view on the one stored object, so none can disagree with it.

`RecordSpec(τ)` and the template `τ` denote the same space. The tag, not the denotation, fixes the kind and the operations. Two rules then govern record-valued positions. **Raw mappings are never leaves**: a raw `dict` flattens to nested tree structure. **A tracked term as a field value stays a term-valued leaf**, at every kind, so its identity (name, provenance, capabilities) is never dropped implicitly.

A `FunctionSpec` types a callable by its input and output structure, either side optional: `None` leaves it unspecified, so a bare `FunctionSpec()` describes any callable. The input side is an explicit `EventTemplate`, written out even for a single-field signature, so a function's field names are caller-chosen and meaningful. The output side accepts any value specification, so a function may declare a term result of any kind or a raw-value result, with a record output the common case. A term declaration names its kind by its class; a raw-value declaration types the value that the wrap boundary then places in a single-field `Record`, keyed by the function's name, so no field name is invented at the spec layer. The output side is unchecked — validity is callability alone — so nothing there is expressible but unsatisfiable, unlike an event declaration, which `DistributionSpec` checks and therefore keeps record-valued. Validity is callability alone: the value-layer specs stay callable-generic, and it is the spec's identity as a `FunctionSpec`, not `is_valid`, that tells the wrap boundary to wrap a raw callable result into a `Function`. The two sides are independent, so a callable may map a space to itself or between two spaces.

### Rationale

One `is_valid` contract across raw values and terms keeps validation uniform (`C1 – Uniform interface to distributions and values`). A term spec being a value spec is what lets a term occupy a field anywhere a plain value can.

Storing every declaration as a spec, and the spec on the term itself, is `D7 – Single source of truth`. The declared kind is a stored class rather than an inference, and every accessor is a view that cannot drift.

## II.3 — `EventTemplate`

### Contract

An `EventTemplate` is a `NamedTree` whose leaves are value specifications (II.2). It defines the *shape of one event*, such as a draw or a stored datum, and it is the *schema*: the structure that indexes a kind. It is never one of its own leaves.

When every leaf is an `ArraySpec`, the template is fully numeric and construction auto-promotes it to a `NumericEventTemplate`. The promotion is re-derived whenever a transform constructs a new template, so a replacement that removes the last non-numeric leaf promotes the result and one that introduces a non-numeric leaf demotes it: the numeric axis is an invariant of the current leaves, not of the object's history. Beyond the inherited `NamedTree` interface (with `L = ValueSpec`), `EventTemplate` adds construction, lossy template inference from a value, and projection to `NumericEventTemplate`:

```python
class EventTemplate(NamedTree[ValueSpec]):
    def __init__(self, field_specs: Mapping[str, Any] | None = None, /,
                 **fields: ValueSpec | EventTemplate | tuple[int, ...] | None) -> None: ...
    # sugar: a bare shape tuple means ArraySpec(shape) and None means OpaqueSpec();
    # the positional mapping form accepts "/"-path keys and names that collide with keywords

    @classmethod
    def infer_from(cls, value: Any) -> EventTemplate: ...   # best-effort, possibly lossy
    @property
    def is_numeric(self) -> bool: ...
    @property
    def is_concrete(self) -> bool: ...                      # False when any dimension is symbolic
    @property
    def free_dims(self) -> frozenset[str]: ...              # the unbound symbolic dimensions
    def numeric_subset(self) -> NumericEventTemplate: ...   # remove non-ArraySpec leaves
```

`NumericEventTemplate` further provides a flat (vectorized) layout of the leaves:

```python
class NumericEventTemplate(EventTemplate):
    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]: ...   # per-field array shapes, canonical order
    @property
    def vector_size(self) -> int: ...                          # total flat dimension; defined only when concrete
```

**Symbolic dimensions.** A shape entry may be a **named symbolic dimension** instead of an integer. `ArraySpec(shape=("obs", "features"))` fixes the rank and gives each dimension an identity while deferring its size. Within one template a name refers to one dimension: fields `X: ("obs", "features")` and `coefficients: ("features",)` share the dimension `features`, an equality no pair of concrete integers can express. A template with any symbolic entry is **polymorphic**, with `is_concrete` false and `free_dims` listing the unbound names. Templates carry no scope object beyond the names themselves, so they serialize as plain data.

Checking a value against a polymorphic template cannot be done leaf by leaf, because a symbolic name constrains several leaves at once. Validation therefore runs a single pass over all fields, resolving each name to one size: a name is bound the first time it is seen, every later occurrence must agree, and a disagreement raises. Bound names never rebind.

The work splits accordingly. A leaf's `is_valid` checks its own rank and dtype, and nothing else — an `ArraySpec`'s `support` is descriptive metadata, unchecked. Sizes belong to the one pass, since only it sees every occurrence of a name.

Binding returns a new template rather than mutating the original, so refinement is monotone. Until every name is bound the template is not concrete, so a `NumericEventTemplate` has no flat layout: an operation that needs sizes raises, naming the free dimensions.

### Rationale

As the *type layer*, an `EventTemplate` is the explicit structure that travels with a value and with the producers and consumers of values (`D5 – Explicit, carried structure`). It separates the structure of one event from the orthogonal axes of *multiplicity* and *identity*, keeping those distinctions explicit (`D1 – Mathematical fidelity`). A symbolic dimension carries a dimension's identity, which is mathematical structure, while deferring its size to the data that determines it, so cross-field equalities travel with the term and sizes bind when their producer appears (`D5 – Explicit, carried structure`, `C3 – Computational detail hidden by default, available on demand`).

---

## II.4 — Identity, type & metadata: `TrackedTerm`, `Annotated`, `Provenance`

### Contract

Identity, type & metadata is the cross-cutting layer that lets any object carry, alongside its raw representation, four things: a **name** (what the object is called), a **spec** (the declaration of its type, II.2), a **provenance** (how it was produced), and free-form **annotations** (auxiliary information supplied by the user or an algorithm). The structure is provided by two mixins: `TrackedTerm` (name, spec, and provenance) and `Annotated` (annotations). Every first-class object, the kind an operation consumes and produces, must be a `TrackedTerm`, while structural helpers such as templates and specs are not. We call any such object a **tracked term**: a value, distribution, conditional distribution, linear operator, or batch that carries a name, its spec, and provenance.

Annotation metadata is a free-form mapping:

```python
class Annotated:
    annotations: Mapping[str, Any] | None
```

A tracked term's name must be provided by the user when constructed explicitly (as the required first argument to the constructor). When an operation produces an object, it must provide a meaningful, deterministic name derived from its inputs. The `name_is_auto` flag records which, because the two behave differently: a structure-changing transform re-derives an auto-derived name from its result, and composition into a larger object may rename it again, while a user-given name is preserved. A *nested* object (i.e., a sub-object of a `NamedTree`) takes its name from the field key it sits under. For example, a sub-object reached at `parameters` is itself named `parameters`. `with_name` renames the object itself, unlike `NamedTree.with_path_names`, which renames the fields within it. A name is a human label rather than an identity: nothing resolves an object by name, derived names need no escaping scheme, and two objects may share a name wherever field uniqueness does not force distinctness. The provenance of a tracked term stores pointers to descriptors of the parent objects that created it, along with the operation. Optionally, it can also provide references to the parents themselves.

The `spec` slot is the term's type, stored once (II.2). Each kind narrows it to its own spec class and exposes convenience accessors for its properties.

```python
class TrackedTerm:
    name:         str
    name_is_auto: bool
    spec:         TermSpec                       # the single stored source of the term's type (II.2)
    provenance:   Provenance | None              # write-once via with_provenance(...)
    def with_name(self, name: str) -> Self: ...  # shallow copy with name_is_auto = False
    def with_provenance(self, p: Provenance) -> Self: ...

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

A provenance records everything its operation resolved: the tracked parents, the plain arguments, and the **controls** the run actually used, the PRNG key, sample counts, and any selected method among them, so a result can be reproduced from its record. Fingerprints are best-effort, in tiers: a content hash for arrays and records, the qualified name and code hash for a closure-free callable, and object identity otherwise, with `fingerprint_is_weak` marking that last tier.

### Rationale

`TrackedTerm` serves the two non-mathematical principles: `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. Housing the spec on the tracked base is `D7 – Single source of truth` for a term's type: one slot, declared once, that every kind's accessors are views on. The guarantee behind `C6 – Traceable and reproducible workflows` is a single rule: **every object a ProbPipe operation natively returns is a tracked term** (whether or not it is also `Annotated`), so the provenance chain is never broken. Recording the resolved controls, not just the parents, is what turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived names keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`). Because identity and metadata are orthogonal to *what* an object is mathematically, they are defined uniformly across classes.

## II.5 — `Batch`

### Contract

A `Batch` is the generic `TrackedTerm` nd array of shape `batch_shape` that holds objects of a common element type. `batch_shape` is nonempty: a batch has at least one batch axis. It could also be `Annotated` if applications for it arise. A concrete batch implementation must specify how to store the elements. Since a batch is a *collection* of its elements, `len` / `iter` / `batch_shape` / `batch_size` operate only on the batch axes. The `batch_*` names are kept deliberately rather than a numpy-style `.shape` / `size`, which could ambiguously cover both the batch axes and the per-element content. A concrete batch implementation adds whatever its element type affords in that element's own section — including, where useful, indexing into the elements' fields, since `[]` dispatches unambiguously on the key type.

**What `[]` dispatches on.** A key is either a **position** or a **name**, and the two namespaces never collide: an axis has no name and a field no position. A position — an integer, a slice, or a tuple of those — addresses the batch axes, which is `Batch`'s own business and stated once there. A name — a string, or a tuple of strings for a path — addresses a field within every element, which only a batch whose elements have fields can answer, so the base reports that these elements have none and a batch of records supplies it. A tuple mixing the two addresses neither and is refused by that reading rather than by a complaint about how many indices a tuple holds, which would blame the count for the wrong thing.

```python
class Batch[E](TrackedTerm):
    @property
    def spec(self) -> BatchSpec: ...                     # the single stored source of the type
    @property
    def element_spec(self) -> ValueSpec: ...             # view on spec: what every element satisfies
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
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**. `axis_groups` tiles `batch_shape` into contiguous groups, outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, flat vectorization above all, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and indexing operate on the leading axis, which is the outermost level exactly when that level holds one axis, and an element of a multi-level batch is the inner-level batch, as a view. Nesting needs no dedicated classes: a batch is itself a tracked term, so a batch whose elements are batches is already admitted, and grouped storage presents the levels as views into one store.

**The batch's own type.** A batch stores a `BatchSpec` (II.2), which carries the element spec and the named multiplicity together; `element_spec`, `axis_groups`, and `level_names` are views on it. This keeps the storage rule reading the same way for a batch as for any other term — `spec` is the batch's *own* type, not its element's — which is what makes `OpaqueBatch` well typed even though an `OpaqueSpec` names no kind. It also mirrors the model, where a batch inhabits the *family* kind over its element's kind rather than the element kind itself.

**Level names.** Each level carries a name, listed in order by `level_names`. An operation that produces a level takes the name to give it, and `with_level_names` repins it: a shallow copy over the same axes and elements, whose own derived name is re-read under the new names so that it and any view taken from it agree. Renaming a *view* is refused where the new name belongs to a level the view derives its name from but no longer carries, since the derived name would then be ambiguous; `with_name` is the way out, because a user-given name makes the batch the root its views derive from and discards the selection accumulated before it. Names are unique within a batch, and a duplicate is an error rather than something the framework resolves: an operation minting a level whose name is already present raises, as a rename onto an existing name does, so the caller supplies a name of its own. A suffixing rule was considered and rejected — a name like `draw2` states the order levels were added in, which is a fact about the computation path rather than about meaning, and levels are named precisely so that operands align by meaning. Names must be identifiers, since `at_levels` addresses a level by keyword. Operations align batched operands by level name, and two levels meant to correspond under different names are lined up by renaming one. Level names are independent of the field names within each element.

**View identity.** A view of a batch — an element or a sub-batch — derives its name from the batch it was taken from and the positions it selects, naming the level each selection addresses: `posterior[chain=0]` for a sub-batch, `posterior[chain=0, draw=7]` for an element, and `posterior[draw=1:3]` for a range. The name is marked `name_is_auto`, with provenance recording the indexing. Levels selected whole are left out, so selecting all of a batch derives the batch's own name; the levels that appear are listed in the batch's own order rather than the order they were indexed in. Naming the level is what makes the reading identify the object: a derived name is a function of what the view selects, so two routes to the same selection read alike and two different selections never do, whereas a positional `name[i]` cannot say which axis `i` addressed. A batch whose elements are bare values yields bare elements, which carry no identity to derive.

**Selecting by level.** `at_levels(**levels)` indexes a batch along its named levels and returns a view, the by-name counterpart of positional `[]`. It is the level analogue of `NamedTree.at_path` (II.1), and shares its shape: a path addresses a position and returns a leaf or a subtree, while named level indexers address positions and return an element or a sub-batch. The name is neither `select`, the field-splatting selector on `Record` (III.2) that a batch of records also carries, nor a name promising elements, which it returns only when the selection indexes down to one. Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order, where an integer drops its axis and a slice or `None` keeps it, `None` meaning the whole axis as `:` does. It means that here and only here: a keyword cannot take a `:` literal, so `None` is the spelling that form needs, while positional `[]` writes `:` and refuses `None` — a `None` left by an unset argument would otherwise be read as *all of it*, answering a question the caller never asked. A level spanning several axes takes one indexer per axis, and a shorter indexer fills the leading axes and leaves the rest whole, so a scalar `draw=i` on a two-axis `draw` level means `draw=(i, None)`. A level every axis of which is dropped is removed, so selecting a single-axis level by an integer — or every axis of a multi-axis level — yields the inner batch or element just as positional indexing and iteration do, while a level left unnamed is kept whole. This parallels xarray's `isel` in indexing by name and by position, with `level_names` in the role of xarray dimension names, and there is no label-based counterpart, since batch levels carry no coordinate labels. The tuple form departs from it: a tuple here addresses a level's axes in order, where `isel` reads one as several positions of a single dimension.

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
