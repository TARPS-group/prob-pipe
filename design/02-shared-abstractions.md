# Part II — Shared Abstractions

Part II introduces the shared abstractions the rest of the library is built on.

| § | Category | Abstraction | Role |
|---|---|---|---|
| II.1 | Typing | `TermSpec` | The term-specification base every field and declaration is typed by, with the symbolic-dimension protocol. |
| II.2 | Declarations | `InputSpec` / `OutputSpec` | The input and output declarations: the slots where a term receives values, and the components of the term it produces. |
| II.3 | Numeric values | `Numeric` / `NumericSpec` / `Constraint` | The flat-vector interface the numeric kinds share, its spec-side mixin, and the elementwise support constraint. |
| II.4 | Identity | `TrackedTerm` / `Provenance` | The label, type (spec), lineage, and annotations an object carries beyond its raw representation, with `raw` as its access to that representation. |
| II.5 | Multiplicity | `Batch` | An indexed collection of *separate* objects, with its `BatchSpec`. |
| II.6 | Structure | `NamedTree` | Abstract named, ordered tree: its paths, how `[]` reads them, and navigation. |
| II.7 | Dispatch | dispatch & registries | Registry-based multiple dispatch that selects an implementation by the types involved, and the catalog that makes every registry discoverable. |

## II.1 — Term specifications: `TermSpec`

### Contract

A **term specification** ("term spec") describes the typing information available for a mathematical term and validates whether an object satisfies it.

**Base and batch kinds.** The specs partition into the **base kinds** and the **batch kinds**. Every base kind has exactly one term spec, one **base form**, and one **batch form**. Since a batch of batches is a batch, the base form and batch form of a batch are identical. The correspondence is recorded in the **kind table**: `register_kind` records the tracked class and the batch form of a spec class, and `term_class_for_spec` and `batch_class_for_spec` look them up through the spec's class and its bases, so a spec subclass inherits its base's kind unless it registers its own. A second registration that disagrees raises.

**Symbolic dimensions.** A dimension size for a numeric value spec may be an integer or a **named symbolic dimension**. A dimension name is a Python identifier, such as `n_obs`, so that a call such as `with_dim_sizes(n_obs=100)` can name it; a reserved word such as `class` is an identifier too, and is named in a mapping, as `with_dim_sizes(**{"class": 100})`. A spec with any symbolic dimensions is **polymorphic**; one with none is **concrete**. A spec can *report* the names still unbound, *substitute* explicit sizes for names, *rename* a dimension, and *bind* names by checking that a value or another spec conforms to it (below), reading the sizes off that value's spec. In binding, a name takes its size from its first occurrence, and a later occurrence that disagrees raises. Names form one scope wherever specs meet: within a schema (III.5), between a batch and its element (II.5), across the slots of one map (V.6), and across the operands of a merge (IV.2). Two dimensions that are different quantities are renamed apart before they meet.

The base API is validation plus the dimension protocol:

```python
class TermSpec(ABC):
    @abstractmethod
    def is_valid(self, value: Any) -> bool: ...      # structural validity such as kind, rank, dtype

    # the symbolic-dimension protocol: report, substitute, rename, bind
    @property
    def free_dims(self) -> frozenset[str]: ...       # report: the unbound symbolic dimensions
    @property
    def is_concrete(self) -> bool: ...               # True when free_dims is empty
    def with_dim_sizes(self, **sizes: int) -> Self: ...   # substitute explicit sizes
    def with_dim_names(self, **names: str) -> Self: ...   # rename symbolic dimensions, old=new
    def bind_dims_from_value(self, value: Any) -> Self: ...   # bind by checking that a value conforms
    def bind_dims_from_spec(self, other: TermSpec) -> Self: ...   # bind by checking that another spec conforms

def register_kind(spec_type: type[TermSpec], *, term_class: type, batch_class: type) -> None: ...
def term_class_for_spec(spec: TermSpec) -> type: ...    # the tracked class of the spec's kind
def batch_class_for_spec(spec: TermSpec) -> type: ...   # its batch form
```

**Shape arguments.** An argument that takes a shape, such as `NumericArraySpec`'s `shape`, a level of a `BatchSpec` (II.5), or `sample_shape` (VI.3), takes a single int or str as one dimension and a sequence, such as a tuple, a list, a `range`, or a 1-D array, as one dimension per item, and it stores the tuple. So `3` is `(3,)` and `"n"` is `("n",)`, as NumPy reads `np.zeros(3)`. An iterator such as a generator, a set, `bytes`, a `memoryview`, and a mapping are refused: an iterator is used up by one reading, a set has no order, and the others iterate into byte values or keys. An integer of another type, such as `numpy.int64`, is stored as a Python `int`, a name of a `str` subclass is stored as a `str`, and a `bool` is refused. A shape that sizes draws or an array, such as `sample_shape`, takes integers only. Two arguments that hold sizes are not shape arguments: the size shorthand below takes a tuple only, and `with_dim_sizes` takes one non-negative int per name.

`with_dim_sizes` substitutes supplied sizes and leaves other dimensions symbolic. `with_dim_names` renames simultaneously. Binding reads concrete sizes in one shared scope and rejects disagreements; unobserved dimensions remain symbolic.

**The size shorthand.** Wherever a `RecordSpec` field (III.5), a slot of an `InputSpec` or a `given_spec`, or the keyword form of `OutputSpec` (II.2) takes a term spec, a tuple of sizes is the `NumericArraySpec` of that shape, so `RecordSpec(K=())` and `InputSpec(K=())` declare a scalar. The shorthand is a tuple only, so `K=3` is refused. A spec written in the shorthand sets no dtype and no support, so it conforms to, and unifies with, a spec of any dtype and support of the same shape.

**Conformance.** A term spec `A` **conforms** to a term spec `B` when every value of type `A` is one that `B` admits, in one dimension scope, and several specs **conform in one dimension scope** when each conforms with every name bound once across all of them. At one axis, two equal sizes agree, a name agrees with itself, a name and a size bind the name, and two different sizes or two different free names raise `ValueError`. The rule by kind:
1. **arrays:** equal ranks, each axis agreeing, and, when both set a dtype, `A`'s casting to `B`'s by NumPy's same-kind rule; supports are not compared, since a support is descriptive (III.5);
2. **opaque values:** `B`'s type is `None` or a superclass of `A`'s, and `B`'s `meta` is `None` or equal to `A`'s (III.2);
3. **records:** the same child names, each child of `A` conforming to the child of `B`;
4. **output declarations:** the same form and name, and `A`'s term spec conforming to `B`'s, a type hole in `B` admitting any term spec (II.2);
5. **laws:** `A`'s event declaration conforming to `B`'s;
6. **kernels and functions:** the same slot names, each slot spec of `B` conforming to `A`'s slot spec of the same name, a slot optional in `B` being optional in `A`, and `A`'s output declaration conforming to `B`'s; a side that `B` leaves `None` admits any side of `A`, and a side that only `A` leaves `None` does not conform;
7. **batches:** the same levels, and `A`'s element term spec conforming to `B`'s, a base element's name left uncompared (II.5).

Binding against a value or another spec is this check. Slots are contravariant and outputs covariant, as a kernel's spec states (III.9).

**Unification.** Two term specs **unify** when their unification `U(A, B)` exists, in one dimension scope. `U(A, B)` is the spec of the values that both admit, which is the spec of a slot that two operands of a merge share (IV.2):
1. **arrays:** equal ranks and agreeing axes; a dtype that only one sets, and, where both set one, the dtype of the spec that conforms to the other, the narrower by NumPy's safe cast when each conforms to the other, with no unification when neither conforms; a support only when both set the same one;
2. **opaque values:** equal types or one `None`, and equal `meta`s or one `None`, giving the known type and the set `meta` (III.2);
3. **records:** the same child names, unified child by child;
4. **output declarations:** the same form and name, and `U` of the term specs, a type hole taking the other's term spec;
5. **laws:** `U` of the event declarations;
6. **kernels and functions:** the same slot names, with equal slot specs and optional flags, and `U` of the output declarations, a side that one leaves `None` taking the other's;
7. **batches:** the same levels, and `U` of the element term specs.

Any other pair, such as an array spec with a record spec, does not unify. `U` is symmetric, and a spec conforms to `U(A, B)` exactly when it conforms to both `A` and `B`, provided no two different free dimension names meet at one axis.

```python
NumericArraySpec((3,), dtype="float32")   # conforms to NumericArraySpec(("p",)), binding p to 3
NumericArraySpec((), dtype="int32")       # conforms to NumericArraySpec((), dtype="float32"): a same-kind cast
NumericArraySpec((), dtype="float32")     # does not conform to NumericArraySpec((), dtype="int32")
NumericArraySpec(("m",))                  # does not conform to NumericArraySpec(("n",)): two different names
# U(NumericArraySpec((), dtype="float32"), NumericArraySpec((), dtype="float64")) is NumericArraySpec((), dtype="float32")
# U(OpaqueSpec(type=str), OpaqueSpec()) is OpaqueSpec(type=str)
```

### Rationale

One `is_valid` contract across the kinds keeps validation uniform (`C1 – Uniform interface to functions, distributions, and values`), and defining each concrete spec beside the kind it describes keeps this layer generic (`D2 – Generality first`). Naming the base for the terms it types is `C5 – Naming for unambiguous meaning` applied to the library's own vocabulary: every spec types a tracked term, and *value* stays reserved for the mathematical kind. The kind rule is `D2 – Generality first`: every result can be tracked and every collection of draws stacked, so everything an operation produces lies within the system. A symbolic dimension carries a dimension's identity, which is mathematical structure, while deferring its size to the data that determines it; hence cross-field equalities travel with the term, and sizes bind when their producer appears (`D5 – Explicit, carried structure`, `C3 – Computational detail hidden by default, available on demand`). Declaring the kind correspondence once, where each kind is defined, is `D6 – Single source of truth`: the wrap, the declaration enforcement, and the batch constructors read one table, so widening a kind widens all of them. One word for each relation between specs, conformance for the directional check that a value is admitted and unification for the symmetric spec of a shared slot, states each check once (`D6 – Single source of truth`), and the directional check is the one a fed value needs, since the receiving slot must admit every value it is given (`D1 – Mathematical fidelity`).

## II.2 — Input and output declarations: `InputSpec`, `OutputSpec`

### Contract

A term receives values at its slots and produces one term. An `InputSpec` declares the slots, and an `OutputSpec` declares the type of what the term produces and the components under which it offers it.

**The produced term.** The **produced term** of a term is what one evaluation of it yields:

| Term | Its produced term |
|---|---|
| an array, an opaque value, or a record | the term itself |
| a law `d` | a draw `X ~ d` |
| a kernel `k` | at each value `w` of its slots, a draw `X ~ k(w)` |
| a function `f` | at each value `w` of its slots, its return value `f(w)` |
| a batch | the batch itself |

Here `w` is a record with one field for each slot it binds. A law's produced term is a draw rather than the law, and a kernel's is a draw rather than the law it gives, so a joint law is a law over a joint draw (IV.2). The produced term of a law or a kernel is random, so a statement about it is a statement about the law of `X`.

**Base specs and specs with parts.** A **base spec** is a `NumericArraySpec` or an `OpaqueSpec`: the type of a term with no named parts, which is a **base value**. A **spec with parts** is one of five:
1. a `RecordSpec` with at least one child;
2. a `DistributionSpec`;
3. a `ConditionalDistributionSpec`;
4. a `FunctionSpec`;
5. a `BatchSpec`.

An empty `RecordSpec` is neither, and an output declaration refuses it. A law is a term with parts even when its draws are single arrays, as `Normal("x", 0.0, 1.0)`'s are.

**Slots.** A **slot** is a named place where a term receives one value. It has three attributes:
1. a name, which is a Python identifier;
2. a term spec, which types the values it admits;
3. a flag, which makes it required or optional.

A `Function`'s slots are its parameters (III.3), and a `ConditionalDistribution`'s slots are what it conditions on (III.9). **Binding** supplies values to slots by name, as a call does, and as `condition_on(k, given)` and the fused conditional calls do (III.9). A binding may omit an optional slot, and the term that owns the slot then uses its own default. A slot whose spec is a `RecordSpec` is one slot, which one record fills, and VI.6 states how a binding fills part of it.

**Input declarations.** An `InputSpec` is an ordered mapping from slot names to term specs, together with the set of optional slots. It stores no defaults, since the term that owns an optional slot holds its default. Two input declarations are equal when they have the same slots, specs, and optional slots, in any order, since binding reads slots by name. An input declaration may have no slots, and `InputSpec(RecordSpec())` is `InputSpec()`.

**The four sources of an input declaration.** Every place that takes a term's slots reads them through the `InputSpec` constructor, which accepts four sources:
1. an `InputSpec`, as it is;
2. a mapping from slot names to term specs;
3. a `RecordSpec`, read by its children, so a nested record declares one slot for each child;
4. an `OutputSpec`, read by its components, each slot typed by its component's spec, so the slots are those the components would feed (IV.2).

The constructor tests for a `RecordSpec` and an `OutputSpec` before a mapping, since a `RecordSpec` iterates its fields (II.6) and an `OutputSpec` is read by its components. `InputSpec(R)` equals `InputSpec(OutputSpec(R))` for every non-empty `RecordSpec` `R`. Every slot of a mapping, a `RecordSpec`, or an `OutputSpec` source is required, and a slot's spec may be written in the size shorthand of II.1. A name that is not an identifier raises `ValueError`. A type hole in an `OutputSpec` source, any other source, and a source given together with keywords raise `TypeError`.

Three arguments take every source: `conditional_distribution`'s `given_spec` (IV.4), `Function`'s `input_spec` (III.3), and `ConditionalDistributionSpec`'s `given_spec` (III.9). `conditional_distribution` and `Function` then mark optional each slot whose parameter has a default, and a source that marks optional a slot whose parameter has no default raises `ValueError`. A law's event declaration, cut down by `[]` or `without`, therefore declares a kernel's slots, as `given_spec=prior.event_spec.without("tau")` does.

**Operations on an input declaration.** An input declaration is indexed by the names of its slots:
1. `I[name]` is that slot's term spec, and a path into a record slot, such as `"g/c"`, gives the term spec at that path;
2. `I[names]`, for a list of names, is the input declaration of those slots, in `I`'s order, each optional slot staying optional; a list that holds a path into a record slot gives that slot holding the cut-down record, and the slot becomes required, since a default is a whole value;
3. `name in I` is true when `name` is a `str` and `I[name]` succeeds, so a list of names is never in `I`;
4. `I.without(*names)` is the other slots, each optional slot staying optional, and it may leave none; a path into a record slot leaves that slot holding the rest of the record, required;
5. `I.rename(...)` renames slots by the rule of II.6, restricted to names, since slots are flat;
6. `I & J` merges two input declarations (IV.2).

A name that is not a slot raises `KeyError`, and the path reader of II.6 states the other errors.

**Output declarations.** An `OutputSpec` states two things about a term's produced term: its term spec, read as `term_spec`, and the name under which the produced term is offered when that term spec is a base spec. The constructor has two forms:
1. **the keyword form**, `OutputSpec(x=T)`, for a base spec `T`, names the whole produced term `x`, its one component, which is **at the root**;
2. **the positional form**, `OutputSpec(T)`, for a spec with parts `T`, offers the components that `T`'s kind gives (below).

**The grammar rule.** The keyword form names only a base spec. A produced term with parts is offered by its parts, in the positional form, or the code that returns it **wraps** it in a one-field record that holds its name, declared `OutputSpec(RecordSpec(name=T))`. ProbPipe never wraps a value that a function body returns: the body returns the record, and the declaration checks it. A name outside a value therefore names only a base value, which has no paths for the name to disagree with, so the paths a term takes, its components, and the paths of its produced term agree (II.6). The form is decided by the term spec alone, so no flag records it.

```python
OutputSpec(beta=NumericArraySpec((5,)))             # keyword form: one component beta, at the root
OutputSpec(beta=(5,))                               # the same, in the size shorthand (II.1)
OutputSpec(beta=None)                               # a type hole: beta's type is filled at completion
OutputSpec(RecordSpec(beta=(5,), sigma=()))         # positional: components beta and sigma; a value is {beta, sigma}
OutputSpec(RecordSpec(params=RecordSpec(beta=(5,), sigma=())))
# wrapped: one component params; a value is {params: {beta, sigma}}
OutputSpec(DistributionSpec(OutputSpec(RecordSpec(mu=(), tau=()))))
# positional over a law: the produced term is a law, which offers no components
OutputSpec(params=RecordSpec(beta=(5,), sigma=()))
# TypeError: params would name a record; give OutputSpec(RecordSpec(...)) to offer its fields, or wrap it
```

**Components.** The components of an output declaration `O` are names, and `O.components` maps each to the spec of its component, in order:

| `O` | Its components | The spec of the component `c` |
|---|---|---|
| the keyword form, or a type hole, named `n` | `n`, at the root | `O.term_spec`, or `None` in a type hole |
| positional over a `RecordSpec` `R` | the names of `R`'s children, in `R`'s order | the child's spec |
| positional over a `FunctionSpec` or a `BatchSpec` `T` that holds the output declaration `E` | the components of `E`, and none when a `FunctionSpec` declares no output | `T` with `E` replaced by `E[c]`, or by `E` itself when `c` is `E`'s component at the root |
| positional over a `DistributionSpec` or a `ConditionalDistributionSpec` | none | — |

The output declaration a spec holds is a `DistributionSpec`'s or a `ConditionalDistributionSpec`'s `event_spec`, a `FunctionSpec`'s `output_spec`, and a `BatchSpec`'s `element_spec` (II.5). The spec of a component is the spec of the produced term's value at that component:
1. for a record, the child;
2. for a batch, the column (II.5);
3. for a function, the component function `x ↦` the value at `c` of its return value.

A function whose return values are records `{mean, var}` is the record of its component functions, and a batch of records is the record of its columns, so each offers its components. A law over records `{mu, tau}` is not the record of its marginals, so a produced term that is a law or a kernel offers no components, and neither does a batch of laws or of kernels. A law's own event declaration describes a draw rather than a law, so `prior.event_spec` offers `mu` and `tau`.

**Paths and indexing.** The paths of a positional declaration are the paths of its term spec (II.6), and a declaration in the keyword form has none, since the name of a component at the root is not a path. A term's paths are its output declaration's. `O[s]` takes a path or a list of paths, which the path reader of II.6 reads, and gives an output declaration:
1. **One path `p`** gives `O`'s declaration at `p`: a base node in the keyword form under the last name of `p`, and a node with parts in the positional form, so a group offers its children, and a held law, kernel, function, or batch is declared positionally over its own spec.
2. **A list of paths** gives the positional declaration over `O.term_spec` cut down to the chosen nodes. Each node keeps its own path, in the tree's order, and a path through a held term cuts that term down in its own kind.
3. **A list may also name the component at the root**, at the top level only, so for the keyword form named `n`, `O[[n]]` is the wrap `OutputSpec(RecordSpec(n=O.term_spec))`.

For a law, a kernel, a function, or a batch `t` with output declaration `O`, `O[s]` is the output declaration of `t[s]`, and for a record `r`, `r[s].spec` is `r.spec[s]` (II.6). `O[s]` adds no name that `O` does not hold, so a function's output declaration agrees with the value it returns. A `str` that is not a path raises `KeyError`. `path in O` is true when `path` is a `str` and `O[path]` succeeds. `O` is not a mapping and is not iterable, so iterating it raises `TypeError`, whose message says to list its components with `.components`.

`O.without(*paths)` removes the nodes at `paths` and keeps every other node at its path, so a list of paths and `without` partition the leaf paths (II.6). A path that is not a path of `O` raises `KeyError`, so every path given to the keyword form raises, and a removal that leaves nothing raises `ValueError`. `O.rename(...)` renames and moves nodes by their paths and renames the component at the root (II.6), and `O & P` merges two output declarations (IV.2). Indexing and removal keep every dimension name, bound or free, so a kept node stays in the dimension scope of the declaration it came from.

```python
O = OutputSpec(RecordSpec(a=(), g=RecordSpec(c=(), e=())))
O["a"]                    # OutputSpec(a=...)
O["g/c"]                  # OutputSpec(c=...)
O["g"]                    # OutputSpec(RecordSpec(c=..., e=...)): the components c and e
O[["g"]]                  # OutputSpec(RecordSpec(g=RecordSpec(c=..., e=...))): the component g
O[["g/e", "a", "g/c"]]    # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=..., e=...))): the tree's order
O.components["g"]         # RecordSpec(c=..., e=...): the spec of the component g
"g/c" in O                # True
O.without("g/e")          # OutputSpec(RecordSpec(a=..., g=RecordSpec(c=...)))
O["x"]                    # KeyError: x is not a path of O
```

**Completion.** A producer knows the type of the term it returns, and it **completes** its declaration with `with_term_spec`. A type hole takes the produced spec. A stated term spec is completed by the produced one: the declared dtype and support are kept, dimensions are bound from the produced term, and the produced value is checked to conform to the stored type (II.1). A produced spec that does not conform to the declared one raises. A type hole is filled by kind:
1. with a base spec, it stays in the keyword form;
2. with a batch whose elements carry only the default name `element` (II.5), it names the elements by the hole's name;
3. with any other spec with parts, it raises `TypeError`, whose message asks the caller to declare the record.

Only the keyword form admits a type hole. `None` means only a pending type, so an opaque field of a `RecordSpec` is declared as `OpaqueSpec()` (III.5).

**The default.** `OutputSpec.default(spec, component=name)` is the declaration a producer uses when it is given none, decided by the kind of `spec`:
1. a base spec: the keyword form under `name`;
2. a batch whose elements carry only the default name `element`: the positional form over that batch with its elements named `name`;
3. any other spec with parts: the positional form, so a record offers its children, a law or a kernel offers no components, and a function offers the components of what it returns.

A bare term spec given as a function's `output_spec` is read by the default, under the callable's default component (III.3, V.6). A law's `event_spec` keeps its own rule, which III.7 states.

**Equality.** Two output declarations are equal when they have the same form, the same component names in order, and equal term specs. `OutputSpec(params=R)` and `OutputSpec(RecordSpec(params=R))` have the same components and describe different produced terms, so they are unequal. Equality keeps order, because component order is the layout `to_vector` reads. An `OutputSpec` equals only another `OutputSpec`, and its hash agrees with its equality.

**The declarations of each term.** Each kind names its declarations by their roles:

| Term `t` | Its input declaration | Its output declaration |
|---|---|---|
| an array or an opaque value | none | none, since a base value carries no name and a label is not one (II.4) |
| a record `r` | none | `OutputSpec(r.spec)`, derived and never stored, and none for an empty record |
| a law `d` | none (III.7) | `d.event_spec`, which describes a draw |
| a kernel `k` | `k.given_spec`, with at least one slot (III.9) | `k.event_spec` |
| a function `f` | `f.input_spec`, or `None` when undeclared | `f.output_spec`, or `None` when undeclared |
| a batch `bt` | its elements' input declaration | `OutputSpec(bt.spec)`, derived, whose components are its columns for a batch of records or of functions, one component at the root for a batch of base values, and none for a batch of laws or of kernels (II.5) |

`bt.element_spec` is the output declaration of one element as a term (II.5). A kernel and the function that returns its laws are one map `S → P(T)` whose produced terms differ, a draw and a law, so the names `event_spec` and `output_spec` keep that difference visible, as `given_spec` names what a kernel is conditioned on (III.9). A kernel never has a name that is both a slot and a component (III.9).

**Names.** Component names are the only names a declaration carries, and a merge reads only them: a distribution labeled `regression_model` may declare `OutputSpec(beta=beta_spec)` or `OutputSpec(RecordSpec(beta=beta_spec))`, and either offers `beta`. A function that returns a law declares it positionally, which offers no components, or its body returns `{"posterior": law}` under the declaration `OutputSpec(RecordSpec(posterior=post_spec))`, which offers `posterior`. The object's label is set separately, by the constructor's `label=` keyword or by `with_label` (II.4). A component name is any non-empty string without `/`, which is the rule for a record's field names, and a component that feeds a slot must also be an identifier, which `InputSpec` checks.

```python
class InputSpec(Mapping[str, TermSpec]):   # named slots; keys are Python identifiers
    def __init__(self, slots: InputSpec | Mapping[str, TermSpec | tuple] | RecordSpec | OutputSpec | None = None,
                 /, **slots_by_name: TermSpec | tuple) -> None: ...   # one of the four sources
    @property
    def required(self) -> tuple[str, ...]: ...   # the slots a binding supplies, in slot order
    @property
    def optional(self) -> frozenset[str]: ...    # the slots a binding may omit
    def with_optional(self, *names: str) -> InputSpec: ...   # these slots optional as well
    @overload
    def __getitem__(self, name: str, /) -> TermSpec: ...     # a slot's spec, or the spec at a path into a record slot
    @overload
    def __getitem__(self, names: Sequence[str], /) -> InputSpec: ...   # those slots, in this declaration's order
    def __contains__(self, name: object, /) -> bool: ...     # False for a list or a tuple
    def get(self, name: str, default: TermSpec | None = None, /) -> TermSpec | None: ...
    def without(self, *names: str) -> InputSpec: ...         # the other slots, optional ones kept optional
    def __and__(self, other: InputSpec, /) -> InputSpec: ...  # the merge of IV.2
    def rename(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> InputSpec: ...

class OutputSpec:
    @overload
    def __init__(self, term_spec: TermSpec, /) -> None: ...  # the positional form, over a spec with parts
    @overload
    def __init__(self, **term: TermSpec | tuple | None) -> None: ...   # the keyword form: one keyword, over a base spec
    @property
    def term_spec(self) -> TermSpec | None: ...   # the produced term's type; None only while a type hole is pending
    @property
    def components(self) -> Mapping[str, TermSpec | None]: ...   # each component's spec, in order
    def __getitem__(self, paths: str | Sequence[str], /) -> OutputSpec: ...   # the declaration at paths
    def __contains__(self, path: object, /) -> bool: ...    # whether path is a str and self[path] succeeds
    def __iter__(self) -> NoReturn: ...                     # raises TypeError: indexed by path, not iterable
    def without(self, *paths: str) -> OutputSpec: ...       # the other nodes, each at its path
    def __and__(self, other: OutputSpec, /) -> OutputSpec: ...   # the merge of IV.2
    def with_term_spec(self, term_spec: TermSpec, /) -> OutputSpec: ...   # completion
    def rename(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> OutputSpec: ...
    # rename or move nodes by their paths, or rename the component at the root (II.6)

    @classmethod
    def default(cls, spec: TermSpec, *, component: str) -> OutputSpec: ...   # decided by the kind of spec
```

### Rationale

Separate object and component names serve `C5 – Naming for unambiguous meaning`: a model's label and its variables answer different questions. The grammar rule keeps every name inside the value it names, or on a base value that has no paths, so a term has one set of paths, which its draws, its views, and the keys of a rename share (`C5 – Naming for unambiguous meaning`). The term spec alone decides the form, so the declaration stores no flag that could disagree with its spec (`D6 – Single source of truth`). A produced law offers no components because a joint law is not the record of its marginals, and offering them would let a merge receive two marginal laws that have lost the dependence between them (`D1 – Mathematical fidelity`). One constructor reads every source of slots, and a `RecordSpec` has one reading, by its children, whether it declares components or slots (`D6 – Single source of truth`, `C1 – Uniform interface to functions, distributions, and values`). One completion operation serves every producer (`D6 – Single source of truth`).

### Notes

- *The interim whole form.* Until every distribution family wraps its draws under the name its constructor takes (VII.2, VII.5), the keyword form also accepts a spec with parts, `OutputSpec(n=T)`, as an interim whole form, and so does a type hole filled with one. It keeps the reading those families need: one component `n`, at the root, of spec `T`; the paths `n` and `n/q` for each path `q` of `T`; `O[n]` is `O`, and `O["n/q"]` is the declaration at `q` of `T`; `O.without(n)` raises `ValueError`, while a removal below `n` keeps the name; and a merge stores the produced term whole, as the field `n` (IV.2). The grammar rule's guarantee and the paths a merge keeps do not hold for it. Once the families wrap their draws, the keyword form refuses a spec with parts.

## II.3 — Numeric values: `Numeric`, `NumericSpec`, `Constraint`

### Contract

**The `Numeric` interface.** The numeric kinds share the `to_vector` interface, which lays a value out as one flat vector in canonical order. In addition, `vector_size` gives that vector's length, and `from_vector` rebuilds a value from a vector. The coordinate protocols expose the same layout to foreign libraries, so `np.*` and `jnp.*` functions apply to the numeric kinds (returning bare arrays). A bare array passed where a `Numeric` value is expected is promoted to the appropriate numeric type. Functions therefore act on a numeric value in two ways: a foreign function sees the coordinates and returns a bare array, while ProbPipe's own operators and entrywise `map` preserve structure and return tracked terms.

```python
class Numeric(ABC):                         # the flat-vector interface of the numeric kinds
    @property
    @abstractmethod
    def vector_size(self) -> int: ...       # total flat dimension; defined only when concrete
    @abstractmethod
    def to_vector(self) -> Array: ...       # the coordinates: one flat vector, canonical order
    @classmethod
    @abstractmethod
    def from_vector(cls, spec: NumericSpec, vec: Array, *,
                    label: str) -> Self: ...  # the inverse of to_vector

    # the coordinate protocols: NumPy and JAX read the value as to_vector(),
    # so their functions return bare arrays
    def __array__(self) -> np.ndarray: ...
    def __jax_array__(self) -> Array: ...
```

**The array-backend registry.** A numeric host that duck typing cannot read, such as an xarray `DataArray` or a pandas object, is recognized through a registry keyed on its type. A registered backend reports whether an instance holds numeric data, its event shape and dtype without converting it, its conversion to the backend array at the compute boundary, and the metadata it carries, which the wrap of V.4 reads. Backends register at import (II.7); NumPy and JAX arrays need none.

**The `NumericSpec` mixin.** A term spec includes `NumericSpec` when every value it admits implements `Numeric`. The mixin serves two purposes. As a type, it identifies numeric structure from declarations alone. As an interface, it adds `vector_size`, since the flat dimension is known from the spec's shapes alone. A spec that includes it keeps its own kind.

```python
class NumericSpec(TermSpec, ABC):   # mixin: the specs whose values implement Numeric
    @property
    def vector_size(self) -> int: ...   # total flat dimension; raises unless concrete
    @abstractmethod
    def _vector_size(self) -> int: ...  # the count itself; called only for a concrete spec
```

A numeric kind may also specify the **support** of its values with a `Constraint`, which compares and hashes by value.

```python
class Constraint(ABC):
    @abstractmethod
    def check(self, value: ArrayLike) -> Array: ...   # entrywise membership
```

A support that is a subset of the reals contains a complex entry only where its imaginary part is zero and its real part is in the support.

### Rationale

One flat-vector interface over the numeric kinds is `D2 – Generality first`: everything that consumes flat numeric values types against it once, and the coordinate protocols keep foreign array functions usable with no ProbPipe-specific code (`C3 – Computational detail hidden by default, available on demand`). The spec-side mixin is the same generality at the type level, whether the event is one array or a named tree of them. Both are abstract bases rather than protocols, which keeps the pair symmetric and follows the rule the library uses throughout: an interface a closed set of ProbPipe kinds implements is a base, while an interface any object may implement is a structural protocol. The base also holds the shared coordinate protocols once rather than in each kind (`D6 – Single source of truth`). A constraint is data: comparing and hashing by value lets a support key a registry, so the bijector factories select by the mathematics rather than by class identity (`D3 – Capability-based operations`).

## II.4 — Identity, type & metadata: `TrackedTerm`, `Provenance`

### Contract

Every tracked term carries four things through the one mixin `TrackedTerm`:
1. a **label**: what a reader calls the object;
2. a **spec**: the declaration of its type (II.1);
3. a **provenance**: how it was produced;
4. **annotations**: free-form auxiliary information supplied by the user or an algorithm.

Every term a constructor builds carries a description, the expression its label renders unless the caller gives an alias. Constructors take mathematical data first and the alias as the keyword `label=`. A label is required when the data cannot identify what the value represents: a raw `NumericArray`, `Opaque`, `NumericArrayBatch`, or `OpaqueBatch` value, and a function or a function batch member whose callable has no name, such as a `functools.partial`. Otherwise, construction derives a description from declared structure or described operands:

1. records and record batches use their ordered field names, except that a batch stacked from records takes the first record's label (III.6); an empty record requires a label;
2. functions use the callable's name, with `𝒻` for a lambda; lambda kernels use `ℙ`;
3. catalog laws use their family name and declared components; generic laws and kernels use `ℙ` and their declarations;
4. products, mixtures, and minibatched laws describe their operands, a mixture as `mixture([a(x), b(x)])` and a minibatched law as `minibatch(prior(beta), lik(y | beta), batch_size=40)`; object batches describe their members in row-major order over all their batch axes, showing at most eight and then `…`, and empty object batches require a label.

A managed function call derives an application expression from the function and arguments unless an explicit `output_label` supplies an alias. Unnamed array arguments appear by their parameter names. An operation derives its result's expression from its operands as its own rule specifies. Computational names belong to declarations: labels never supply components, fields, levels, fingerprints, or random-stream identity.

JAX pytree rebuilding carries numerical leaves and the specs needed to interpret them, without descriptions or provenance. A rebuilt term displays `<no description>` and remains fully usable, including inside `jit` and `vmap`, and the managed result boundary supplies its result expression.

The result of a user-defined function called on values carries its application description unless an explicit `output_label` provides an alias (III.3). A law or a kernel that the function returns is labeled by the function's label, or by the alias, and displays its own signature, as `predict(y, mu)` (IV.2). An operation's result carries the expression that the operation's rule builds from its operands' expressions, and an operation without a rule gives its result the expression of its primary operand, the first in its signature. A merge (IV.2), `condition_on` (VI.6), and `marginal` and `factor` (VI.8) return a part or a conditional of a law, so their rules label the result by what it is: a product of factors joins the factors' labels, and a conditional keeps the label of the law it conditions and records the paths it fixes, as `condition_on(schools, data)` is labeled `schools` and displays as `schools(theta; y)`. When conditioning leaves one factor of a product alone, the result is that factor's conditional and keeps the factor's label, even when the product carries an alias from `with_label`, so `condition_on(lik & prior, {"mu": 0.5})` displays as `lik(y; mu)` whatever the product's label.

A label is set once, at construction, and representation changes preserve it: a record with renamed fields, a realigned factor, or a converted law keeps the label it had, and `with_label` replaces it explicitly. A law converted from a law constructed without `label=` takes its own default label instead, since the source's default names only the source's class, so `convert(Laplace("g", 9.0, 1.0), Normal)` displays as `Normal(g)` (VI.10). `with_label` keeps the paths a law or a kernel holds fixed, so a posterior relabeled `fit` displays as `fit(mu; y)`. No lookup resolves an object by its label, and derived labels need no escaping scheme. A derived label may hold `~`, a space, `;`, or `/`, so no operation uses a label as a component name or a level name: a law an operation builds for its own computation takes a fixed name or its operand's components.

**Label, signature, and notation.** A label names an object for a reader, and each other mechanism that names or identifies something has a responsibility of its own:

| Mechanism | Responsibility | Section |
|---|---|---|
| label | names an object for a reader | II.4 |
| signature | states what a law, a kernel, or a function is over, read from its declaration | II.4 |
| component name | names a component of a produced term, which a merge and `component` match | II.2 |
| slot | names where an argument or a given value binds | II.2 |
| source identity, random-event identity, fingerprints, replay, and caching | identify objects, content, and random events, and decide reuse | V.5, V.8, II.4 |

The **signature** of a term is read from its declaration:
1. a law: its event components in declaration order, as `mu, tau`;
2. a kernel: its event components, then `|` and its given slots, as `y | beta`;
3. a function: its parameters, as `x, y`.

A given slot or a parameter that has a default reads `name=value`, as `counts(y | K, r, n0=50.0)` or `predict(x, scale=1.0)`. A default that is a number, a string, `None`, or an array of no axes shows its value, and any other default, or one the kernel cannot read, shows `name=…`. A slot with a default stays a given of each law and kernel that conditioning makes until a given binds it (VI.6), so `counts` at `K` and `r` reads `counts(y | n0=50.0; K, r)`. The call of a lifted function (below) shows the arguments of the call, so a parameter left at its default is not among them.

A law or a kernel that holds paths fixed at given values lists them after `;`, as `mu; y` for the posterior of `mu` given `y` and `y | sigma; beta` for a kernel applied at `beta`. A component rename changes the signature, and `with_label` changes the label. The **notation** is the label followed by the signature in parentheses, as `prior(mu)`, `glm(y | beta)`, or `predict(x, y)`, and the property `notation` returns it. `str()` shows the notation, and the repr shows the label and the component as the repr paragraph below states. The signature has no public attribute of its own, since `Function.signature` is the Python call signature that binding reads (III.3). No operation reads a label or a signature to decide anything, and two objects may share either.

**The expression.** Every tracked term carries one immutable **expression**, a tree that states what the term is, and its label, its notation, `str()`, and every label derived from it are renderings of that tree. A term constructed directly carries its label alone. The operation that makes a term builds the term's expression from its operands' expressions, so no operation builds a label of its own, and no operation reads an expression. A node of the tree is one of these:
1. a label, with the signature of a law, a kernel, or a function;
2. a product of factors without a label;
3. a law or a kernel conditioned at given values of some paths;
4. a law indexed at some of its paths;
5. a draw from a law;
6. a function applied to draws of laws and to values;
7. a summary of a law or a draw: an expectation, a variance, a covariance, a quantile, a score, or a density;
8. an operator applied to values;
9. a batch at some of its positions.

A term's label is the name part of the rendering:
1. a conditioned or an indexed law keeps its base's label;
2. a product without a label joins its factors' labels;
3. an applied function takes the function's label;
4. a value renders in full, as `mu ~ prior` or `𝔼[mu ~ prior]`.

The notation is the whole rendering, grouped by the rules below. A law's signature is read from its declaration, and the paths it holds fixed and whether a product was given a label are read from its expression (IV.2). `with_label` replaces the expression with the new label, so a user's label hides the derivation, which provenance still records. Provenance records how a term was computed, and the expression records what the term is, for a reader. Fingerprints and replay omit the expression, as they omit the label, and copies and pickles keep it.

A rendering shows at most `notation_config.max_depth` nested levels, eight by default and at most 64, and a part nested deeper renders as its label, the name of a law or a function, or as `…` for a value. A term's label is rendered once, when the term is built, at the setting in effect then, and the term keeps it. A notation is rendered each time it is shown, at the current setting. A stored expression keeps at most 64 levels, so a long derivation, such as a loop that adds to a value, never builds a tree too deep to copy or pickle. A node built over a part that nests 64 levels stores the whole part as its label or `…`, together with its signature and the paths it holds fixed, and it renders that way at any setting, which is why the setting is at most 64. A long derivation is therefore truncated whole each time it reaches 64 levels, and the label of the operation that truncates it shows that operation alone, as `… + 1`. Showing a term warns: `str()`, `repr()`, and `notation` warn with a `UserWarning` when what they show left a part out, and the warning names the setting where raising it shows more. A part that storage truncated warns only where its text shows less than the part in full, and that warning says that no setting shows the part.

**Values computed from a law.** A value computed from a law `d` is labeled by the value written in probability notation over `d`, and its components follow the result rule of VI.0:
1. **A draw** (VI.3) is labeled by its components, `~`, and `d`'s label, then `;` and the paths `d` holds fixed when it holds any, as `mu ~ prior`, `(y, mu) ~ model`, or `mu ~ model; y`, and it keeps `d`'s components.
2. **A score** (VI.4) is labeled `log` followed by `d`'s notation, as `log prior(mu)` or `log model(mu; y)`, and a density is labeled by `d`'s notation, as `prior(mu)`, since that notation denotes the density.
3. **A moment** (VI.5) is labeled `𝔼[draw]`, `𝕍[draw]`, or `ℂ[draw]`, and a quantile `ℚ[draw]`, as `𝔼[(y, mu) ~ model]`.
4. **An expectation** of `f` is labeled `𝔼[...]` around the label that a call of `f` on a draw takes, as `𝔼[f(mu ~ prior)]`, with `𝒻` for a lambda, so `expectation(d, f)` and `mean(f(d))` read alike.

A batch of draws or of scores has the label of one draw or one score and a level of its own, and its element is grouped before its positions, as `(mu ~ prior)[sample=0]`. A batch of laws reads as one law under its label, so a value computed from a batch `schools` of laws over `effect` is labeled `effect ~ schools`, `𝔼[effect ~ schools]`, or `log schools(effect)`, and `str()` of the batch shows that notation, `over`, and its levels, as `schools(effect) over school`. The atoms of an empirical law that `condition_on` or an inference method produces are labeled by the law's components, as `beta` or `(K, r, phi)`, and the atoms of an empirical law a user constructs keep the label they were given. A node of a value takes its path as its label, so `sample(model)["y"]` is labeled `y`. A value indexed by several paths is labeled by its paths, as `(a, g/c)`. An operator on values is labeled by its expression, as `2 * effect` or `2 * 𝔼[mu ~ prior]`.

**Lifted functions.** A law that lifting a function over laws produces (V.5) is labeled by the function's `output_label`, and it displays as the function applied to draws of its inputs, as `challenger_damage_probability(beta ~ oring_model; damage)`, since it is the law of that random quantity. Arguments that the lift draws together from one law share one draw, as `f((a, b) ~ model)`. A law that is not lifted appears by its notation, as `log_prob(g(g), q ~ q)`, any other tracked argument by its label, and any other argument that the caller passed by its value when it is a scalar, as `f(beta ~ model; y, 2.0)`, and by its parameter's name otherwise, as `f(beta ~ model; y, X)`. A batch of such laws, which a lift over a law and a sweep over a batch give together, carries the same expression, so its mean is labeled `𝔼[f(mu ~ prior, tau)]`. Its element keeps the label of its position, as `f[tau=3]`, and displays as its row's call, which shows a scalar of the swept batch by its value, as `f(mu ~ prior, 4.0)`; any other positions display as the batch's call and the levels they take, as `f(mu ~ prior, tau)[tau=1:3]`. A value computed from such a law takes the law's notation in place of a draw, as `𝔼[challenger_damage_probability(beta ~ oring_model; damage)]`.

**Grouping.** A label is grouped where another label is built from it, so it reads as one operand:
1. a **compound** label is parenthesized, and a label is compound when it has one of these forms:
   - an operator at its top level, as `effect + 1.0` or the draw `mu ~ prior`;
   - a product of labels, as `lik·prior`;
   - a unary operator or `log` at its start, as `-x` or the score `log prior(mu)`;
2. any other label of several words is bracketed, as `[other effect]`;
3. a label of one word is used as it is, and a call such as `prior(mu)` is one word.

So indexing a batch labeled `x·y` at some positions gives the label `(x·y)[sample=0:2]`, and a law derived from an unlabeled product displays as `(lik·prior)(y)`. A draw and a score are grouped before positions, as in `(mu ~ prior)[sample=0]` and `(log prior(mu))[sample=0]`, and the notation of a product without a label is grouped after `log`, as in `log (lik(y | mu)·prior(mu))`. Two rules join labels without grouping them. The right side of `~` is not grouped, since `~` binds most loosely, as in `(y, mu) ~ lik·prior`. Labels join associatively, so a product joins a further factor's label as it is, and `(lik & prior) & d` is labeled `lik·prior·d` (IV.2).

The `spec` slot is the term's type, stored once. Each kind narrows it to its own spec class and exposes convenience accessors for its properties.

Every tracked term exposes the property `raw` as the single access point to the representation layer. It is the term **detached** from the workflow. Detachment removes provenance, annotations, and any reference to a container or parent, and it keeps the spec and the label. A kind represented by an object from outside ProbPipe has a **raw host**, which is its `raw`, for example a backing array object or a wrapped callable. The `raw` of a kind whose representation is a ProbPipe object, such as a factored joint, is that object detached, and the `raw` of a view of a law is the raw form of its parent's marginal at its path (III.7). **Every read of `raw` gives the same raw form.** A term keeps the detached copy it builds on the first read, and the copy refuses provenance, since every reader shares it, while a relabeled or unpickled copy of it is a term of its own. A record keeps its resolved leaves and gives a new `dict` over them on each read, which the caller may change (III.5).

Accessing a container returns a **view**, for example a record field or a batch element. A container's view is a tracked term labeled from the accessor, which is the path for a record and the levels the positions take for a batch, and a batch element is a relabeled copy, so the term the batch stores keeps its own label; its provenance records the container and the source term where one was supplied. A batch's label is grouped before the positions, as in `(lik·prior)[dataset=0]`, so the positions read as applying to the whole label.

**The repr.** A law's or kernel's repr reads as a call of its public constructor, with any paths it holds fixed as its last argument. Other terms show diagnostic summaries of their label and structure, which are not constructor calls, so `NumericArray('x', shape=(3,), dtype=float32)` shows the label first although the constructor takes it as a keyword:
1. **the label**: a value, a record, or a batch displays it first in its diagnostic summary, as in `RecordBatch('schools', ...)`. A law or a kernel takes it as the keyword `label=` after its other arguments, and leaves it out exactly when the constructor, given the arguments the call shows and no label, would give that label again: its default, the class name or `ℙ`, and a label it derives from its arguments, such as a product's or a mixture's. Any other label is shown, including one an operation carries to another class, so `convert` of a mixture to `Normal` reads `Normal('x', ..., label='mixture')`. A label that a function's name gives is shown, as in `ConditionalDistribution('y', given=('mu',), label='ricker_counts')`, since the class does not state it;
2. **a batch's levels**: a mapping of level name to size, as in `levels={'chain': 4, 'draw': 500}`, with the tuple of its sizes for a level of several axes;
3. **the element's structure**, named for what it is:
   - a record: its field paths, as in `fields=('data/effect', 'data/se', 'label')`, which a numeric record lists on the lines below its summary instead;
   - an array: its shape and dtype, as in `NumericArray('x', shape=(3,), dtype=float32)`;
   - a batch of arrays or laws: its element spec;
   - a law or a kernel: the component at the root of its event and its other mathematical arguments, as its constructor takes them, and its event declaration as `event_spec=` where that differs from the default of III.7. So `Normal("mu", 0.0, 1.0)` reads `Normal('mu', loc=0.0, scale=1.0)`, and `Normal("mu", 0.0, 1.0, label="prior")` reads `Normal('mu', loc=0.0, scale=1.0, label='prior')`. An atoms-based law follows its own constructor's order, as `EmpiricalDistribution(atoms=..., component='theta')`.

A law or a kernel that holds paths fixed shows them as the call's last argument, `fixed=(...)`. A posterior of `oring_model` given `damage` therefore reads `EmpiricalDistribution(atoms=..., label='oring_model', fixed=('damage',))` and differs in its repr from a law of the same atoms that holds nothing fixed, and a kernel applied at `beta` reads `ConditionalDistribution('y', given=('sigma',), label='glm', fixed=('beta',))`. The repr of a family that holds nothing fixed, and whose parameters print in full, evaluates to an equal law: one of the same class, under the same label, with equal parameters. A parameter prints in full when it is a number, an array of at most eight entries, a law whose own repr evaluates, or a sequence of at most eight such laws, as a mixture's components are. A function, such as a bijector, prints by its label and parameters, so a law that holds one does not evaluate.

A spec reads as its own constructor call, with the attributes it sets, and so does a report, such as a `MethodInfo` or a `CallReport` (V.1). A name that is not a Python identifier, such as the component `my model`, cannot be a keyword, so a call with one writes its names in order inside one `**{...}` mapping, as in `OutputSpec(**{'my model': NumericArraySpec(shape=())})`. A private class reads as its public class or kind, so a law that renames another reads as the class of the law it renames. A repr longer than about 100 characters shows one argument per line. In a notebook a `CallReport` also displays as a table, with a row for each route and the selected route marked.

**A numeric term shows its values.** A numeric term, which is a `NumericArray`, a `NumericRecord`, or a batch of either, shows its values on the lines below the summary, with each line indented by two spaces. `str()` shows the values alone, at the margin. NumPy's printing formats each value, so `np.set_printoptions` sets the precision and the size past which an array is shortened with `...`. Only the entries shown are read from the device. A record and a batch of records show one line for each field path, such as `population/mu`, followed by its value, and a value shortened on such a line ends with its shape. The summary above these lines leaves out `fields=`, since the lines list the paths. A batch shows its stored array or its columns, whose leading axes are its levels. A summary shown alone keeps `fields=`, and a term shows its summary alone in three cases: inside another repr, as the atoms in a law's repr do; inside a JAX trace, where the values are tracers; and when no entry can be read. Its `str()` is then the summary too. In a notebook a record and a batch of records also display as a two-column table of paths and values.

**A record of summaries displays as a table.** A numeric record displays as a table under its label when its top-level fields are records over the same paths, with the same shape at each path, and at least one of them records how it was computed. The table has one row for each path, or for each element of an array path, as `y[0]`, and one column for each top-level field, with the table's lines indented by two spaces as a record's lines are. NumPy's print options format each column's entries together and shorten the rows of a large array path to its first and last elements. The line below the table states how each column was computed: `exact` for an exact route, `Monte Carlo` with the number of draws for a route that records its draws, and `approximate` for another approximate route, as each column's provenance records it. A record whose provenance records a description of each column under `computations`, as the record that `summarize` returns does (VI.5), takes those descriptions instead. `str()` gives the table without its label and that line, and in a notebook the table is an HTML table. The record that `summarize` returns displays this way, and so does one built by hand, as `Record({"mean": mean(d), "sd": standard_deviation(d)})`. A record of plain values, or the draw or the mean of a law whose groups share their field names, as `a/mu` and `b/mu`, records no computation and shows one line for each path, as does every record when provenance tracking is off.

A batch of records displays one table for each element when its provenance describes each element's columns, as the batch that `summarize` returns for a `DistributionBatch` does. Each table is titled by its element's levels, as `law=0`, with each title indented by two spaces and each table by four. The line that states how the columns were computed follows each table, or follows the last when every element's line is the same. The elements are shortened to their first and last ones when their rows number more than NumPy's `threshold`. In a notebook each table is an HTML table.

**A tracked term is immutable.** `TrackedTerm` carries an immutability guard automatically, so assignment and deletion raise an error. Immutability requires that every transformation, including each `with_*` method, returns a new term that shares the representation.

Identity is **boundary-attached** under compiled execution. Inside a `jit` or `vmap` trace a term presents as its raw representation with only its spec as static data, so label, provenance, and annotations never enter a trace and a label can never affect compilation-cache identity; the tracked result is minted at the enclosing call boundary. A value, a record, and a batch of either are pytrees whose static data is their spec alone, so two terms that differ only in their labels or their expressions have equal treedefs and share a compilation. A term rebuilt from its leaves displays `<no description>` until a managed result boundary gives it the call's expression or explicit output alias. Rebuilt terms remain usable during JAX tracing and direct rebuilding.

```python
class TrackedTerm(ABC):
    label:        str
    spec:         TermSpec                       # the single stored source of the term's type (II.1)
    provenance:   Provenance | None              # write-once via with_provenance(...)
    annotations:  Mapping[str, Any] | None       # free-form; the one store written after construction
    def with_label(self, label: str) -> Self: ...  # the one way a label changes
    def with_provenance(self, p: Provenance) -> Self: ...
    @property
    @abstractmethod
    def raw(self) -> Any: ...
    # the representation, detached from the workflow; each kind defines it, as its section states
    # immutable: __setattr__ / __delattr__ raise; state round-trips through the
    # attributes the term holds, so copy and pickle need nothing from the class

class Provenance:
    operation: str                       # the operation that produced the object
    parents:   tuple[ParentInfo, ...]    # descriptors of the tracked inputs
    controls:  Mapping[str, Any]         # the resolved controls, and the derived key of each draw: sample count, selected method, ...
    inputs:    Mapping[str, ParentInfo]  # plain (untracked) arguments, keyed by parameter name

class ParentInfo:
    type_name:   str
    name:        str
    fingerprint: str            # best-effort content hash
    fingerprint_is_weak: bool   # True when the fingerprint is only object identity
    parent:      Any | None     # optional reference to original parent

def provenance_ancestors(term: TrackedTerm) -> list[ParentInfo]: ...   # every ancestor reachable through parents
def provenance_dag(term: TrackedTerm) -> Any: ...                       # the chain rendered as a Graphviz graph

class NotationConfig:                  # how labels and notations render
    @property
    def max_depth(self) -> int: ...    # the nested levels a rendering shows: 8 by default, from 1 to 64
    def reset(self) -> None: ...       # the defaults, re-reading PROBPIPE_NOTATION_MAX_DEPTH

notation_config: NotationConfig        # the settings in effect
```

The **annotations** store is the one exception to immutability: it can be written after construction, so a diagnostic can attach its result to the object it examined, and by convention it is append-only. Annotations are otherwise inert: no operation reads them to decide behavior, they do not propagate to results because lineage is recorded in `provenance`, and they never enter a compiled trace.

Fingerprints are best-effort and tiered, from a content hash, through the code hash of a closure-free callable, down to object identity, with `fingerprint_is_weak` marking the weakest tier. A fingerprint records what a term computes, so a label, which names the term for display, is no part of it: a relabeled law or function keeps its fingerprint. **The provenance mode alone sets the tier.** The default `lightweight` mode records identity-tier descriptors for every operation's parents with no content hashing; the `full` mode retains parent references and content-verifiable fingerprints, computed lazily at export; `off` mode records nothing. Every operation behaves identically under a given mode. Two functions can access provenance information when available: `provenance_ancestors` returns every ancestor reachable through the parents while `provenance_dag` shows it as a graph.

### Rationale

`TrackedTerm` serves the two non-mathematical principles, `C5 – Naming for unambiguous meaning` and `C6 – Traceable and reproducible workflows`. Housing the spec on the tracked base is `D6 – Single source of truth` for a term's type: one slot, declared once, that every kind's accessors are views on. Recording the resolved controls, not just the parents, turns traceability into reproducibility: re-running the recorded operation on the recorded inputs with the recorded controls reproduces the result. Auto-derived labels keep every intermediate object identifiable without forcing the user to label it (`C5 – Naming for unambiguous meaning`), and boundary attachment keeps labels inert in computation, so a label never decides what gets compiled. Immutability is `C2 – Functional interface over immutable objects` embodied, and confining the one writable store to a container that no operation reads keeps that contract intact in substance. Carrying annotations on the base makes every tracked term annotatable, including a batch of draws. `raw` is `B3 – Tracked forms out by default` for a term already in hand: the representation is one attribute away. The repr serves `C5 – Naming for unambiguous meaning`, since it names the term's label, its public class, and each part by what it is, so a reader learns what a term is and holds from its repr alone. A numeric term's repr also shows its values, so a reader sees a result in the tracked form that `B3 – Tracked forms out by default` returns without reading `raw`. Reading only the entries shown keeps the display of a large array cheap. Reading the signature from the declaration is `D6 – Single source of truth`: the notation states what a term is over from the one place that fixes it, so a label chosen by the user cannot misstate it.

### Notes

- *Reconstruction.* `pickle` and `copy` restore a term by assigning its state back rather than through its constructor, so a schema fixed by an explicit declaration, a class decided at construction, and a field written afterward all survive the round-trip. A term declares any memo as transient and any store written in place (annotations) as decoupled, so a copy takes its own container.

Automatic summaries use these symbols:

| Summary | Symbol |
|---|---|
| Mean and expectation | `𝔼` |
| Variance | `𝕍` |
| Standard deviation | `√𝕍` |
| Covariance | `ℂ` |
| Quantile | `ℚ` |

Explicit labels remain literal, including ASCII `p`, `f`, `E`, and `Q`. Generated component names retain their ASCII spelling, and caller-supplied keys are unchanged. An anonymous function displays as `𝒻` while its default output component is `f`.

## II.5 — `Batch`

### Contract

A `Batch` is the generic `TrackedTerm` nd-array of shape `batch_shape`, holding objects of a common element type. `batch_shape` is nonempty. A concrete batch specifies how it stores its elements and adds whatever its element type affords, including indexing its elements by path. Since a batch is a *collection* of its elements, `len`, `iter`, `batch_shape`, and `batch_size` operate on the batch axes only. `BatchSpec` is the batch kind's term spec (II.1), carrying the output declaration of one element together with the named multiplicity:

```python
class BatchSpec(TermSpec):         # the batch kind's spec; is_valid accepts a matching batch
    element_spec: OutputSpec                        # the output declaration of one element as a term
    axis_groups: tuple[tuple[int | str, ...], ...]  # the multiplicity, tiled into levels below;
                                                    #   a str names a symbolic dimension (II.1)
    level_names: tuple[str, ...]
    def __init__(self, element: TermSpec | OutputSpec,
                 levels: Mapping[str, int | str | Iterable[int | str]] | None = None, /,
                 **level_shapes: int | str | Iterable[int | str]) -> None: ...
    # element: the elements' term spec, or their output declaration;
    # one keyword per level, outermost first, giving the shape of its axes
    @property
    def levels(self) -> Mapping[str, tuple[int | str, ...]]: ...   # level name -> its axis sizes
```

**Constructing a `BatchSpec`.** `BatchSpec(element, **levels)` names each level and gives the shape of its axes, outermost first, as in `BatchSpec(NumericArraySpec(()), chain=4, draw="S")`. `element` is an `OutputSpec` or a term spec: a bare base spec takes the default name `element` in the keyword form, and a bare spec with parts takes the positional form (II.2). Each level's shape is a shape argument (II.1), so `draw=4` is one axis of size 4 and `grid=(3, 4)` is one level of two axes. A level name that no keyword spells is given in a mapping passed positionally, as `BatchSpec(spec, {"my level": 2})`, and the two forms are not combined. `levels` returns that mapping, so `BatchSpec(other_spec, batch.spec.levels)` declares another element type over a batch's levels. A batch constructor takes `level_names`, where a single str names one level, and `axes_per_level`, where a single int is one level's count, since a live batch reads its sizes off its elements.

Construction checks every element against the elements' term spec, `element_spec.term_spec`, and reports the position that failed, since the batch asserts that spec of all of them. A constructor given no element spec infers it from the elements, as the element kind's constructor infers a term's spec from its value. The axes past those the levels hold are an array's event shape, so `NumericArrayBatch(jnp.arange(4.0), 'draw', label='draws')` holds four scalars. A batch of records infers each field's spec from its column in the same way.

**The element spec.** Every batch holds an **element spec**, `element_spec`: the output declaration of one element as a term (II.2), whose `term_spec` is the elements' type:
1. **a batch an operation produces** stores its producer's declaration of one element, the law's event declaration for draws (VI.3) and the completed output declaration for a sweep (V.6);
2. **a batch built from raw values** gets the default: a record element is positional; a law, kernel, or function element is positional over its own spec; and a base element is named `element`, in the keyword form, which the batch's repr leaves out;
3. **a column of a batch of records** stores the declaration of the view at its path (II.2), so the column at a field `a` is named `a`, and a group's column is positional over the group's record.

For a base element, the name is the one thing `element_spec` holds beyond the term spec, so a `BatchSpec` still stores one thing about its elements. For a batch of values, `element_spec` is the output declaration of each element, with the batch supplying a base element's name. For a batch of laws, kernels, or functions, each element's own output declaration is the one its term spec holds, as III.10's `event_spec` is. The batch's label never supplies a component.

```python
draws = sample(Normal("mu", 0.0, 1.0), sample_shape=4)
draws.element_spec                   # OutputSpec(mu=NumericArraySpec(shape=(), dtype=float32, support=real))
draws.element_spec.term_spec         # NumericArraySpec(shape=(), dtype=float32, support=real)
NumericArrayBatch(jnp.arange(4.0), "draw", label="xs").element_spec   # OutputSpec(element=...)
grouped_draws = sample(grouped, sample_shape=4)
grouped_draws["a"].element_spec      # OutputSpec(a=...)
grouped_draws["g"].element_spec      # OutputSpec(RecordSpec(c=..., e=...))
```

**A batch's components.** A batch is its own produced term, and its output declaration is `OutputSpec(bt.spec)`, derived (II.2). Its components depend on its elements:
1. a batch of records or of functions offers its columns, each typed by its column's `BatchSpec`;
2. a batch of base values offers one component at the root;
3. a batch of laws or of kernels offers none.

`bt.element_spec.components` lists the elements' components, and `bt.component(c)` is the column at `c`, or the batch itself for the elements' component at the root (II.6).

**Names stay out of a batch's equality and of JAX.** `BatchSpec` compares a base element's name in its own equality and hash. A batch's equality and hashing, its fingerprint, and every pytree flatten use a name-free canonical form of its spec instead, recursively, so a `BatchSpec` held inside another term's static data, such as the `RecordSpec` of a record whose field holds a batch, is flattened to its canonical form too. Draws of `Normal("x", 0.0, 1.0)` and of `Normal("y", 0.0, 1.0)` are therefore equal when their values are, they are one structure to JAX, and a compiled function compiles once for both. The names of a record element are part of its `RecordSpec`, which is static data already, so only a base element's name is dropped: a raw JAX transform such as `jax.jit(lambda b: b)(draws)` returns a batch whose elements are named `element`, as it drops a label (II.4). ProbPipe's own operations restore the names at their results, where they declare them anyway.

**`[]` dispatch.** The argument of `[]` is a **position**, which indexes the batch axes, or a path or a list of paths, which index every element, as the path reader of II.6 reads them. A position is an integer, a slice, or a tuple of those, and `Batch` itself handles it. A path indexes each element:
1. one path gives the column, the batch over the same levels of each element's node at the path, as a view (II.4) named from the path;
2. a list of paths gives the batch of each element's record of the chosen nodes, each at its own path;
3. a batch of laws gives the batch of the views of its laws (III.10), and a batch of functions the batch of each function indexed there.

Positions and paths commute, so `draws[3]["g"] == draws["g"][3]`. A batch of base values has no paths below its elements, so `draws["mu"]` raises `KeyError` for the draws of `Normal("mu", 0.0, 1.0)`, whose message says that `mu` names the elements as a whole; `draws.component("mu")` is the batch itself, and `draws[["mu"]]` is the batch of records `{mu}`. An empty tuple, an empty list, and a list of positions raise (II.6). `p in bt` tests paths and never elements, so a batch of base values answers `False` for every `str`, and `1.0 in draws` is `False`.

```python
class Batch[E](TrackedTerm):
    @property
    def spec(self) -> BatchSpec: ...
    @property
    def element_spec(self) -> OutputSpec: ...           # view on spec: the output declaration of one element
    @property
    def batch_shape(self) -> tuple[int, ...]: ...
    @property
    def batch_size(self) -> int: ...                    # total element count, prod(batch_shape)
    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]: ...   # batch_shape tiled into levels, outermost first
    @property
    def level_names(self) -> tuple[str, ...]: ...       # one name per level, aligned with axis_groups
    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename levels old -> new; shapes and elements unchanged, as rename is for nodes
    def __len__(self) -> int: ...                       # leading-axis size, batch_shape[0]
    def __iter__(self) -> Iterator[E | Self]: ...       # over the leading batch axis
    def __repr__(self) -> str: ...                      # the label, the levels, the elements' structure, and a numeric batch's values (II.4)
    def __getitem__(self, key: Any) -> Any: ...         # a position indexes the axes, a path or a list of paths every element
    def __contains__(self, path: object) -> bool: ...   # whether path is a str and self[path] succeeds
    def component(self, name: str, /) -> Batch: ...     # the column at name, or self for the component at the root
    def __and__(self, other: Batch, /) -> Batch: ...    # the merge of IV.2, element by element over equal levels
    def at_levels(self, /, **levels: int | slice | None | tuple[int | slice | None, ...]) -> E | Self: ...
    # index by named level (a view); unnamed levels kept whole, None means the whole axis (:)
    @property
    def raw(self) -> Any: ...
    # the storage view: the elements' raw values in their native stacked layout
```

**Axis groups.** A batch's axes are partitioned into ordered **levels**: `axis_groups` tiles `batch_shape` into contiguous groups with the outermost level first, and `batch_shape` stays their flat concatenation, so anything stated over `batch_shape`, such as flat vectorization, applies to a multi-level batch unchanged. A single-level batch has one group holding all its axes. `len`, `iter`, and positional `[]` address the leading **axis**, which is `batch_shape[0]`, rather than the leading level; the two coincide when the outermost level holds one axis, so iterating `(N,)` of `(S,)` runs over the `N` and yields each inner batch of `S` as a view. When the outermost level spans several axes, indexing drops the leading axis and leaves the level in place with one axis fewer.

**The batch's own type.** A batch stores its `BatchSpec`; `element_spec`, `axis_groups`, and `level_names` are views on it. `spec` is the batch's own type, which keeps `OpaqueBatch` well typed although its elements carry no structure.

**The raw view.** `raw` is the **storage view**, which holds the elements' raw values in their native stacked layout — for example, one array for array-valued elements and an object array for opaque-valued ones.

**A polymorphic multiplicity.** An axis size may be a symbolic dimension name instead of an integer, as a `NumericArraySpec` shape entry may, so a *declaration* can fix the number of levels while deferring how many elements each holds: "returns a batch of `S` draws" before `S` is known. The names share one scope with the element's schema, so a batch of `("n",)` over arrays of shape `("n",)` is square by declaration. A live `Batch` is concrete, since it holds elements at positions; construction refuses a spec with free dimensions, and `batch_size` is defined once they are bound.

**Level names.** Each level carries a name, listed in order by `level_names`. Names are unique within a batch and follow the rule for component names, so a level name is any non-empty string without `/` (II.2). `at_levels` takes a level's name as a keyword, and a name that is not a Python identifier is passed in a mapping, as `at_levels(**{"my model": 0})`. An operation names the level it mints after itself, and a constructor such as `stack` takes the name to give it. A name already in use raises, as does a rename onto one, so the caller renames first or supplies another. `with_level_names` renames levels while preserving the object's label, shapes, and elements; subsequent views use the renamed levels. Renaming a *view* is refused when the new name collides with a level of its root batch that it no longer carries; `with_label` gives it a new label and view root. Operations align batched operands by their level names (VI.11), which are independent of the field names within an element.

**View identity.** A view of a batch, whether an element or a sub-batch, derives its label from the batch it was taken from and the positions it takes, naming the level each position addresses. Take a batch labeled `posterior`, with a `chain` level of `(4,)` over a `draw` level of `(1000,)`:

```python
posterior.at_levels(chain=0).label          # "posterior[chain=0]"          — a sub-batch of draws
posterior.at_levels(chain=0, draw=7).label  # "posterior[chain=0, draw=7]"  — an element
posterior.at_levels(draw=slice(1, 3)).label # "posterior[draw=1:3]"         — both levels kept
```

The derived label states which positions the view takes. Levels taken whole are left out, so the view of all of a batch derives the batch's own label, and the levels that appear are listed in the batch's own order; hence two ways of indexing the same positions read alike, and different positions read differently. Whether a batch *stores* an element outright or *materializes* it on demand, as columnar storage builds a row, indexing returns a view (II.4); storage is invisible to access. A *sub-batch* is a view in the same way, being the batch at some of its positions.

**Indexing by level.** `at_levels(**levels)` indexes a batch along its named levels as the by-name counterpart of positional `[]`. It is distinct from `select`, which splats a `Record`'s fields, and it returns an element only when the indexers leave no axis. Each indexer is an integer, a slice, `None`, or a tuple of these addressing the level's axes in order: an integer drops its axis, and a slice or `None` keeps it. `None` stands for the whole axis here and only here, since a keyword cannot take a `:` literal and positional `[]` refuses `None`. A shorter tuple fills the leading axes and leaves the rest whole, so `draw=i` on a two-axis `draw` level means `draw=(i, None)`. A level whose axes are all dropped is removed, yielding the inner batch or element as positional indexing does; a level left unnamed is kept whole.

### Rationale

`Batch` satisfies `D1 – Mathematical fidelity` by keeping how many objects there are separate from what one object contains. Levels extend the same fidelity to collections of collections: how many objects are at each level is a mathematical distinction, so `N` laws with `S` draws each are `(N,)` of `(S,)` rather than one anonymous `(N, S)`. Naming the levels is `C5 – Naming for unambiguous meaning` on that axis: a user can say which multiplicity is which, and operations align batches by meaning rather than by position. A suffixing rule for a taken name, such as `draw2`, was rejected on the same ground: it would state the order the levels were added in, which is a fact about the computation rather than about meaning. The `batch_*` names are kept rather than a numpy-style `.shape` / `size`, which could cover the batch axes, the per-element content, or both. An operation broadcasts across a batch by mapping over its elements, so a batch supports an operation exactly when its elements do (`D3 – Capability-based operations`). Indexing and iteration yield views at every level, to satisfy `D6 – Single source of truth`. Keeping a base element's name out of the static data keeps names inert in computation, as labels are (II.4), so a compiled program is reused whatever its arguments are named (`C3 – Computational detail hidden by default, available on demand`). The obligation falls on the concrete batch, since only it knows its storage, and its sub-batch method shares the store wherever the storage affords it (`B4 – No copying at boundaries`). Hence indexing all of a batch needs no special case: it calls the same method as any other positions, and a view costs nothing there.

### Notes

- *Relation to xarray.* `at_levels` parallels `isel` in indexing by name and by position, with `level_names` in the role of dimension names; there is no label-based counterpart, since batch levels carry no coordinate labels, and a tuple here addresses a level's axes in order, where `isel` reads one as several positions of a single dimension.

## II.6 — `NamedTree`

### Contract

Structured values in ProbPipe are represented as named, ordered trees, and a path names a place inside a term's value. The following terms refer to the parts of a tree, shown on a draw `{a, g: {c, e}}`:

| Term | Meaning | In `{a, g: {c, e}}` |
|---|---|---|
| **path** | the `/`-joined names from the top of a term down to a node, always a `str` | `"a"`, `"g"`, `"g/c"` |
| **node** | a place in a term that a path names; the **root** is the term itself, which has no path | `a`, `g`, `g/c`, `g/e` |
| **child** | an entry directly under a node | `a` and `g` under the root |
| **field** | a node of a record, named through records only, that is not itself a record | `a`, `g/c`, `g/e` |
| **group** | a node of a record, other than the root, that is itself a record | `g` |

A field may hold a law, a kernel, a function, or a batch, which is then a **held term** at a **held field**. A path with no path below it is a leaf path. The **canonical order** of a tree is a depth-first walk visiting children in insertion order. A name is a non-empty `str` without `/`. Sibling names are distinct, so every path names at most one node, and distinct groups may reuse a name, as in `a/c` and `b/c`.

**Paths go through every kind.** A path continues past a held field into the held term's own paths:
1. a law's paths are the paths of its draws;
2. a kernel's are those of the draws of the law it gives;
3. a function's are those of what it returns;
4. a batch's are those of one element.

So the record `record_with_law = Record({"n": 3.0, "G": post})`, whose field `G` holds a law over `{mu, tau}`, has the paths `n`, `G`, `G/mu`, and `G/tau`, its fields are `n` and `G`, and `record_with_law["G/mu"]` is `record_with_law["G"]["mu"]`. Below a held term, the spec at a path is the spec of a term of the held term's kind whose produced term is the node at that path (II.2). The paths of every term are the paths of its output declaration (II.2).

**How `[]` reads its argument.** Every term indexed by path reads the argument of `[]` through one **path reader**:
1. a `str` is one path, and a `str` subclass such as `numpy.str_` is read as a `str`;
2. a list or a tuple of `str` is a list of paths;
3. an `int`, anything with `__index__`, a `slice`, or a tuple of those is a position, which only a batch takes (II.5).

`bool`, `None`, and `...` raise `TypeError`. The reader refuses a malformed argument before anything is looked up:

| Indexing | Error | Reason |
|---|---|---|
| `t[""]`, `t["g//c"]`, `t["/a"]` | `ValueError` | an empty path or an empty name |
| `t[[]]`, `t[()]` | `ValueError` | no paths, on a batch as well |
| `t["a", "a"]` | `ValueError` | the same path twice |
| `t["g", "g/c"]` | `ValueError` | a path inside another would hold one node twice |
| `t["a", 0]` | `TypeError` | a path and a position in one `[]` |
| `draws[[0, 2]]`, `draws[jnp.array([0, 2])]` | `TypeError` | a list of positions |
| `t[{"a"}]`, `t[iter(["a"])]` | `TypeError` | a set has no order, and an iterator can be read only once |

A path that does not exist raises `KeyError`, whose message lists the term's paths. When the missing path comes from a tuple whose `/`-joined form is a path of the term, the message adds that a tuple is a list of paths and that a nested path is written with `/`. Only `[]`, `marginal` (VI.8), and `OutputSpec[]` (II.2) take a list of paths. Every other argument takes one path as a `str`, which the same reader checks, so `without("a", "a")` and `without("g", "g/c")` raise as `[]` does.

**One path gives the node, and a list gives a record that holds the nodes.** `t[p]` is the node at `p`, which may be a group. `t[ps]`, for a list of paths, keeps each chosen node at its own path, in the tree's canonical order, whatever order the list gives. A list of one path therefore keeps the node's name, so `t["g"]` is the group's contents and `t[["g"]]` is the record `{g: ...}`, as pandas reads `df["a"]` and `df[["a"]]`. A path in a list that goes through a held field cuts the held term down to the rest of the path, in the held term's own kind. On a record, `t[s]` is the node itself (III.5). On a law, a kernel, a function, or a batch, `t[s]` is the term of the same kind whose produced term is the node at `s` of `t`'s produced term (III.7, III.9, III.3, II.5), and indexing never changes a term's slots. A value indexed by several paths is labeled by its paths, as `(a, g/c)` (II.4).

```python
draw = sample(grouped)    # Record {a, g: {c, e}}
draw["g/c"]               # NumericArray, labeled "g/c"
draw["g"]                 # Record {c, e}
draw["g"]["c"]            # the same as draw["g/c"]
draw["a", "g/c"]          # Record {a, g: {c}}, labeled "(a, g/c)"
draw["g/e", "a"]          # Record {a, g: {e}}: the tree's order
draw[["g"]]               # Record {g: {c, e}}
draw.keys()               # ("a", "g/c", "g/e"): the fields
```

**The component at the root.** A component at the root is a name and not a path (II.2), so `[]` never reads it. `t.component(c)` reads the term at the component `c`: `t` itself when `c` is at the root, and `t[c]` otherwise. A name that is not a component raises `KeyError`, so a term whose produced term is a law offers nothing to `component`.

**Membership.** `p in t` is true when `p` is a `str` and `t[p]` succeeds, and false otherwise. It holds on every term other than an array, and on `RecordSpec`, `OutputSpec`, and `InputSpec` (II.2), so a list of paths is never in a term.

**The mapping interface.** A `NamedTree` keeps Python's mapping methods over its fields: `keys()`, `values()`, `items()`, iteration, and `len` range over the fields in canonical order, while `[]` and `in` take every path. `"g" in draw` is therefore true while `"g"` is not in `draw.keys()`. `at_path("g", "c")` joins separate names with `/`, so it is `t["g/c"]`, and `children` is the one-level view of a node. Navigation returns views (II.4).

```python
class NamedTree[L]:
    # the mapping interface, keyed by FIELD path
    def __iter__(self) -> Iterator[str]: ...  # field paths
    def __len__(self) -> int: ...             # field count
    def keys(self) -> Iterable[str]: ...      # field paths, canonical order
    def values(self) -> Iterable[L]: ...
    def items(self) -> Iterable[tuple[str, L]]: ...

    # indexing, over every path
    def __getitem__(self, paths: str | Sequence[str]) -> L | Self: ...
    # the node at one path, or the tree of the nodes at a list of paths, each at its path
    def __contains__(self, path: object) -> bool: ...   # whether path is a str and self[path] succeeds
    def at_path(self, *segments: str) -> L | Self: ...  # at_path("a", "b") == self["a/b"]
    @property
    def children(self) -> Mapping[str, L | Self]: ...
    def is_field(self, path: str) -> bool: ...
    @property
    def is_multi_field(self) -> bool: ...   # True when the tree holds more than one field

    # structure-preserving transforms — return the same family
    def rename(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self: ...
    # rename or move nodes, old -> new, where old is the exact path of a node;
    # a new name may itself be a path, which moves the node there
    def map(self, f: Callable[[L], L], /, *args, **kwargs) -> Self: ...
    def map_with_keys(self, f: Callable[[str, L], L], /, *args, **kwargs) -> Self: ...
    def replace(self, path: str, leaf: L | Self) -> Self: ...
    def __and__(self, other: Self, /) -> Self: ...      # shallow: a top-level name in both raises
    def without(self, *paths: str) -> Self: ...         # the other nodes, each at its path

    # serialization (export; the constructor reads a nested mapping back)
    def to_nested_dict(self) -> dict[str, Any]: ...

    @classmethod
    def _node_type(cls) -> type[Self]: ...         # the family's own node type: every navigator and transform returns the same family
    @classmethod
    def _leaf_type(cls) -> type | UnionType: ...   # the family's declared leaf type
```

The parameter `L` declares the leaf type, which is what `values()` and `map` accept and return; groups are always the family's own class. Implementations should check leaves against the declared leaf type at construction.

**Removing nodes.** `t.without(*paths)` returns the tree with the nodes at `paths` removed and every other node at its path. A group that the removal empties goes, a record stays a record even when one node is left, and a removal that leaves nothing raises `ValueError`. A list of paths and `without` therefore partition the leaf paths: every leaf path of `t` is a leaf path of exactly one of `t[ps]` and `t.without(*ps)`.

**Merging.** `t & u` is the tree of `t`'s top-level children followed by `u`'s, and a top-level name in both raises `ValueError`, groups included, as the merge of IV.2 states. A merge that combines two groups of one name is written by hand, as `r1.without("g") & r2.without("g") & Record({"g": r1["g"] & r2["g"]})`.

`rename` renames the nodes *within* a tree, whereas `with_label` relabels the object itself (II.4). `at_path` has a level analogue in `Batch.at_levels` (II.5), and the two are alike: a path names a node and gives it, and named level indexers name positions and give an element or a sub-batch.

**Renaming.** `rename` renames nodes by `old="new"` pairs. A **rename** gives the node at path `old` the path `new`, and the node keeps the nodes below it. In each pair, `old` is the exact path of a node, so keyword pairs name top-level nodes and the positional mapping form names any node. The keys of a rename are the term's paths and the name of its component at the root, which the rename renames. Under the grammar rule of II.2, no key reads both as a name and as a path.

A target may itself be a path, so a rename can **move** a node to another parent: `rename({"group/mu": "mu"})` moves the field to the top level, and `{"mu": "group/mu"}` moves it under `group`, creating intermediate nodes as needed. A rename is *in place* when its target has the node's parent path and no ancestor of the node is itself renamed, so the node keeps its parent; every other rename moves the node. A group that the result leaves empty dissolves, since a tree holds no empty group, and a target may then take its path. The name of a component at the root cannot become a nested path, since the base value has no record to hold it, so `Normal("x", 0.0, 1.0).rename(x="g/y")` raises `ValueError`; a law merged into a joint first has a record to move the node within.

All renames apply simultaneously: each `old` resolves against the original tree, so swaps such as `{"a": "b", "b": "a"}` and simultaneous moves of a node and its ancestor are well-defined.

Every target is checked against the result, and three cases raise:
1. a collision, as when a node is renamed onto an existing sibling;
2. a move into the moved node's own group;
3. two targets of which one is a prefix of the other.

Ordering is deterministic. An in-place rename keeps the node's position, and a moved node is appended to its new parent's children in the order the renames are given, with the mapping's entries before the keywords. A group that one move empties and another refills keeps its position.

For example, in `{"g": "h", "g/mu": "g/m"}` the original `g` becomes `h`. The target `g/m` keeps the parent path of `mu`, but `g` itself is renamed, so `mu` moves and is appended as `m` to a new group `g`.

### Rationale

Named paths satisfy `C5 – Naming for unambiguous meaning`, and keying a rename by exact path gives every node one address, fixed by its position in the tree. One path reader for every term makes a path mean the same thing wherever it is passed, and paths that go through every kind give a law, its draws, its summaries, and its views one set of paths (`C1 – Uniform interface to functions, distributions, and values`). One path gives the node and a list gives a record that holds it, so the form of the argument, never the number of paths, decides the result's kind (`D1 – Mathematical fidelity`). Housing the collection contract in one shared class ensures the type- and value-level structures built on it cannot drift apart on how a node is named or a path is resolved (`C1 – Uniform interface to functions, distributions, and values`).

### Notes

- *Mappings are subtrees.* Construction should materialize a mapping value into a group. The need for this mapping check is why `NamedTree` keeps the mapping methods without registering as a `collections.abc.Mapping`, ensuring a nested mapping and its `to_nested_dict` export round-trip through the constructor.
- *The interim reading of the component at the root.* Until terms gain `component`, `[]` on a law, and on an output declaration in the keyword form, also accepts the name of the component at the root and returns the term itself, and `in` is true for that name. Until paths go through every kind, a path stops at a held field, and a second `[]` indexes the held term.

## II.7 — Dispatch and registries

### Contract

Some operations have many possible implementations, held outside the classes they apply to, and which one applies depends on the *types* of the objects involved. A **dispatch registry** holds those implementations as named methods and selects one for a given call.

Each **dispatch method** declares:
1. a unique `name`;
2. the types it applies to, via `supported_types`;
3. a `check` function that probes feasibility without significant computation and reports, as a `Feasibility`, whether the call is feasible, infeasible, or **unresolved**, which means the declarations the probe needs are not yet available;
4. an `execute` function that performs it;
5. whether it is **exact**: a method either returns a representation of the requested result or a stand-in for it, declared where the method is registered and fixed for the method's life;
6. whether it is a **floor**, `is_floor`: a fallback on its supported types, which ranks after every method that is not a floor, declared where the method is registered and fixed for the method's life, and `False` by default;
7. a **priority**: an integer rank among the methods of the same floor status and exactness, and the one thing about a method a deployment may change at runtime;
8. optionally, a one-line `description`, which the registry catalog shows.

Dispatch is by argument type: a `UnaryDispatchRegistry` keys on the first argument's type, and a `BinaryDispatchRegistry` on the first two. The registry takes the matching methods in **selection order** and runs the first whose `check` establishes feasibility. While a higher-ranked candidate is unresolved, the registry's `check` reports the call unresolved (V.1), and execution resolves prerequisite plans first or reports unavailable requirements (V.9). Selection order is the same in every registry:
1. every method that is not a floor before every floor, so auto-selection tries a floor only after every other candidate;
2. exact methods before approximate ones, so auto-selection runs an exact method whenever one that is not a floor is feasible;
3. priority among methods of the same floor status and exactness, higher first;
4. specificity, favoring the method whose declared types are closest to the arguments' classes in method-resolution order: a type that admits a class without appearing in its method-resolution order, such as a registered virtual subclass of an abstract base class, counts as least specific, and a binary registry sums the distances of its two arguments;
5. registration order.

The first criterion sets the floors apart as a tier of their own: a floor ranks after every other method a call admits, whatever their exactness and priority, as the sampling lift and the elementwise sweep do among the evaluation rules (V.7). The other four criteria order the methods within each tier in the same way.

**Priority.** A method whose priority is `None` is **opt-in-only**: auto-selection skips it, and a caller selects it by name. `None` is the default, so a method registered without a priority leaves auto-selection unchanged until a contributor ranks it. `set_priorities` re-ranks methods at runtime, by keyword or by a mapping, which accepts every method name. It keeps each method's declared exactness and floor status, so it moves a method only within its tier and its exactness, and it warns when a method moves into or out of opt-in-only.

**Listing.** `list_methods` returns the method names ranked by floor status, exactness, priority, and registration order. A call also ranks its candidates by specificity, before registration order, since specificity is measured against that call's arguments.

**Naming a method.** `method="..."` selects the named method in place of auto-selection. Naming skips the type pre-filter alone: the named method's `check` still decides feasibility, and a call with fewer positional arguments than the registry's arity raises `TypeError`, as it does without a name. A call with no feasible method raises `ResolutionError`, naming the methods tried and what each was missing, and an infeasible named method or an unregistered name raises the same. The registry's `check` and `execute` are the dispatch interface, so an unregistered name is a dispatch that cannot resolve, whereas `get_method` and `set_priorities` look a name up and raise `KeyError` for an unregistered one.

**Registration.** Methods are added by registration at import, by whichever layer owns the implementation, so a registry gains its providers without importing them. A registry reads a method's declarations once, at registration, and validates them before adding the method; thereafter it ranks and reports the declarations it read.

```python
type UnarySupportedTypes = tuple[type, ...]
type BinarySupportedTypes = tuple[tuple[type, ...], tuple[type, ...]]   # (left, right) types

class BaseDispatchMethod[SupportedTypesT](ABC):
    name: str
    exact: bool                   # declared at registration, fixed for the method's life
    is_floor: bool = False        # a fallback, ranked after every method that is not one; fixed for the method's life
    priority: int | None = None   # rank among methods of the same floor status and exactness, higher first; None is opt-in-only
    description: str = ""         # one line, shown by the registry catalog

    @abstractmethod
    def supported_types(self) -> SupportedTypesT: ...   # admitted by the registry's structural pre-filter
    @abstractmethod
    def check(self, *args, **kwargs) -> Feasibility: ...
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any: ...

class UnaryDispatchMethod(BaseDispatchMethod[UnarySupportedTypes]): ...     # the first argument determines admission
class BinaryDispatchMethod(BaseDispatchMethod[BinarySupportedTypes]): ...   # the first two arguments do

class Feasibility:                # what a method's check reports
    feasible:    bool | None   # None when required declarations are not yet available
    description: str           # why not, when infeasible
    pending:     tuple[str, ...]   # the missing declarations; non-empty exactly when feasible is None
    actionable:  bool          # infeasible on a detail of the call the caller can fix; a listing leads with it

class MethodInfo(Feasibility):    # what a registry's check reports
    method_name: str | None    # from the registration; both None exactly when no method was selected
    exact:       bool | None   # the selected method's declared exactness, never the check's

class ResolutionError(Exception): ...   # no available implementation under the requested controls
class MathematicalDomainError(ValueError): ...  # the mathematical operation is known to be undefined

class BaseDispatchRegistry[M: BaseDispatchMethod[Any]](ABC):
    # the public interface is concrete; arity subclasses supply key extraction and matching
    kind: str = "dispatch"        # with name and description, the SupportsRegistryCataloging identity
    def __init__(self, *, name: str = "", description: str = "") -> None: ...   # constructing does not catalog
    def register(self, method: M) -> None: ...
    def set_priorities(self, priorities: Mapping[str, int | None] | None = None, /,
                       **kwargs: int | None) -> None: ...   # ranks only; warns on a move into or out of opt-in-only
    def execute(self, *args, method: str | None = None, exact_only: bool = False, **kwargs) -> Any: ...   # auto-select, or run the named method
    def check(self, *args, method: str | None = None, exact_only: bool = False, **kwargs) -> MethodInfo: ...
    def list_methods(self) -> list[str]: ...                           # names, ranked by floor status, exactness, priority, and registration order
    def entry_summaries(self) -> list[EntrySummary]: ...               # one per method, in list_methods order
    def describe_entry(self, name: str) -> EntrySummary: ...           # KeyError for an unknown name

class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]): ...    # keys on one argument's type
class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]): ...  # keys on the first two
```

**Fidelity and validity.** Exact means that the returned representation denotes the requested mathematical result; for a draw-producing operation it means a draw exact in law, so two exact algorithms may draw differently. Approximate means a stand-in for that result. A sampling method may be either: a finite empirical posterior or pushforward is approximate, while sampling from that empirical law may be exact for the law it is. Approximate methods state their assumptions, such as integrability for a Monte Carlo mean. Finite MCMC output is approximate even when the chain has the desired invariant law, and asymptotic targets and convergence assumptions are recorded as method guarantees. Exactness is therefore a fact about a returned representation, so it is one boolean, `exact`, and `exact_only=True` excludes approximate methods.

Provenance records whether each step was exact and its target, preserving upstream approximation history. An earlier approximation stays recorded relative to its upstream target through exact downstream work, and a later exact calculation on the returned law stays locally exact. Derived routes and registry routes report the exactness of their selected implementation chain; a wrapper cannot upgrade it.

**Two failures.** Known mathematical undefinedness alone raises `MathematicalDomainError`, for example the mean of a `Cauchy`, whereas a moment that diverges is the extended real ±∞ (VII.1). Computational unavailability, including a missing capability and a failure to establish existence, raises `ResolutionError`, and a numerical execution failure propagates unchanged. Structural admission errors and return-contract defects are specified at their engine steps (V.4, V.10).

A single **catalog** makes every registry discoverable: it lists the registries and their entries, each entry with its exactness, its floor status, its priority, and a one-line description, so a user can see which entries exist and how a call will resolve. An **entry** is one registered item within a registry; the term is generic because the catalog spans registries of every `kind`, such as the dispatch registries and the bijector factory. A registry can be cataloged if it implements `SupportsRegistryCataloging`; satisfying the protocol is structural, and membership requires an explicit `register`. The operation vocabulary is cataloged the same way, so what ProbPipe can do and how a given call resolves are answered from one place.

```python
@dataclass(frozen=True)
class EntrySummary:
    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""
    exact: bool | None = None   # None when the registry declares no exactness per entry
    is_floor: bool = False      # a fallback, ranked after every entry that is not one
    @property
    def is_opt_in_only(self) -> bool: ...   # priority is None; meaningful only in a registry that ranks its entries

@dataclass(frozen=True)
class RegistryInfo:            # the catalog's per-registry record
    name: str                  # unique within the catalog
    description: str           # one line
    kind: str                  # e.g., "dispatch", "operation", "factory"
    entry_count: int

@runtime_checkable
class SupportsRegistryCataloging(Protocol):   # the identity members are read-only
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def kind(self) -> str: ...
    def entry_summaries(self) -> list[EntrySummary]: ...
    def describe_entry(self, name: str) -> EntrySummary: ...

class RegistryCatalog:
    def register(self, registry: SupportsRegistryCataloging) -> None: ...   # empty or duplicate names raise
    def names(self) -> list[str]: ...
    def list(self) -> list[RegistryInfo]: ...
    def __getitem__(self, name: str) -> SupportsRegistryCataloging: ...     # get the registry
    def __contains__(self, name: object) -> bool: ...                       # check if a registry exists
    def describe(self, name: str) -> str: ...                               # a readable summary of one registry
registry_catalog: RegistryCatalog   # the global instance
```

### Rationale

A registry is how `C3 – Computational detail hidden by default, available on demand` and `D3 – Capability-based operations` apply to operations whose implementation cannot be chosen from a single object alone. The guarded `check` probe keeps auto-selection safe, and `method="..."` leaves the choice in the user's hands. New implementations join by registration, so the supported set grows without changing the call sites (`D2 – Generality first`). Keeping undefinedness distinct from unavailability is `D1 – Mathematical fidelity`, and recording fidelity with its target serves `C6 – Traceable and reproducible workflows`. One catalog over every registry is `D6 – Single source of truth`: one place to see which implementations exist and how a call resolves.
