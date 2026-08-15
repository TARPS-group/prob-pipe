# Changelog

All notable changes to ProbPipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)

- **`ArraySpec` → `NumericArraySpec` (#434; design #443).** The public spec has
  been hard-renamed with no compatibility alias; update imports and type
  references to use `NumericArraySpec`. The bare backend-array alias remains
  `Array`. The design baseline now gives each raw value kind a corresponding
  tracked term and batch form, including `NumericArray` / `NumericArrayBatch`
  and `Opaque` / `OpaqueBatch`, and defines `Batch.raw()` as a shared storage
  view. Runtime implementations of those design contracts land in the
  subsequent stack.

- **Workflow-scoped structural RNG, co-sampling, and validated replay (#389).**
  Function-owned RNG controls—`Function(..., seed=...)`, the former reserved
  call-level RNG option, and `Function.with_options(seed=...)`—have been removed
  without a deprecation shim. Reproducible ProbPipe-owned randomness now belongs
  to a run:

  ```python
  from probpipe import workflow_run

  with workflow_run(seed=42):
      result = workflow(distribution_input)
  ```

  A bare omitted-key stochastic call receives a fresh ephemeral root; seeded,
  anonymous, and nested `workflow_run` scopes derive keys from stable call,
  source, and logical-unit identities. All omitted-key sampling, conversion,
  validation, and diagnostics routes use the same broker. Explicit sampling
  keys and inference `random_seed` arguments remain caller-owned, are passed
  through unchanged, and do not advance the workflow stream. A wrapped user
  callable's own `seed` parameter is still an ordinary input.

  `score_posterior(..., key=None)` no longer uses a fixed
  `jax.random.PRNGKey(0)` for sliced Wasserstein projections. It now follows
  the same ownership rule: a bare score receives a fresh ephemeral root, while
  benchmark scoring must run inside `workflow_run(seed=...)` (or pass an
  explicit `key=`) to remain reproducible.

  Omitted-key `predictive_check`, `simulation_based_calibration`, and `add_ppc`
  certify only the exact built-in `GLMLikelihood` data generator. Custom or
  otherwise opaque likelihoods, including subclasses, must pass `key=`
  explicitly; inheriting `generate_data` does not certify that a subclass's
  sampling still matches the built-in stochastic-effect descriptor. The
  omitted-key route also requires that exact likelihood to carry its stored
  design matrix.

  PPC test functions must have unique `__name__` values because those names
  label the returned statistics; use distinct named functions instead of
  multiple lambdas or same-named methods.

  Repeated aliases, record views, empirical weights, and the supported closed
  set of transformed descendants now co-sample from one root realization.
  Exact empirical eligibility likewise follows that recursive root, so calls
  such as `f(emp["x"], emp["y"])` enumerate and weight the original atoms
  instead of drawing unweighted rows. Because JAX dispatch does not implement
  exact enumeration, explicitly requesting `dispatch="jax"` for this case now
  raises `dispatch='jax' does not support exact empirical enumeration`; use
  `auto`, `sequential`, or `thread` instead.

  Managed thread/Prefect work items preserve logical RNG identity across
  scheduling and retries, while rejecting unmanaged copied concurrent
  contexts. JAX probing cannot consume workflow RNG state.

  Successful workflow-owned stochastic results store an exact provenance RNG
  recipe in FULL and LIGHTWEIGHT modes. `replay_run(provenance)` validates the
  callable, plan, execution capability, provider ABI, and expected events
  before re-deriving keys; OFF and legacy provenance without a recipe are not
  guessed. The new public failures are
  `UnmanagedConcurrentWorkflowEntryError`, `ReplayCompatibilityError`, and
  `ReplayUnsupportedCallableError`.

  A top-level workflow snapshots `provenance_config.mode` on entry. Nested
  scopes and managed workers inherit that value, and configuration changes
  made during execution apply only to the next top-level workflow.

  RNG ABI v1 has no call-local replacement that recreates the old exact
  sibling-realization behavior. Put related quantities in one joint Function
  call, retain the resulting joint distribution, or materialize and reuse
  samples explicitly when a shared realization is required.

### Added

- **`NumericArray` and `NumericArrayBatch` (#398).** The tracked class of the
  numeric-array kind and its batch form, so `NumericArraySpec` has the pair every
  other value spec has. Nothing returns them yet; the operations switch over in a
  later change.

  `NumericArray` holds one array and carries no batch axes, so its `shape` is the
  event shape. Construction **validates without converting** — the value is stored
  in its native form and materialises at most once, at the compute boundary — the
  rule `NumericRecord` already follows for its leaves. It carries the full array
  surface, and **arithmetic yields a bare value of the stored type**: identity is
  attached by operations, and arithmetic is not one. It is a registered pytree,
  which is what lets it cross a `jit` or `vmap` boundary; the boundary presents a
  bare array, and its spec is re-derived from what arrives, since a shape and a
  dtype state it exactly.

  `NumericArrayBatch` stores one array with the batch axes leading and splits
  them from the event axes by its element spec. Selection yields a `NumericArray`
  under the derived name, as `RecordBatch` yields a `Record`.

### Changed

- **`ArraySpec` is renamed `NumericArraySpec` (#398).** The spec now agrees with
  the class it names, as `RecordSpec`/`Record` does. `Array` remains the type
  alias for a bare backend array.

### Changed

### Removed (breaking)

- **`RecordArray` and `NumericRecordArray` are gone; the batch of records is
  `RecordBatch` / `NumericRecordBatch`.** A batched record was a `Record`
  subclass, which made `isinstance(x, Record)` true of a collection and put a
  batch's `len` and iteration in competition with a record's fields. The batch
  types are `Batch` subclasses now: they hold named levels, `len` and `iter`
  speak about the collection, and the field structure is read from
  `event_template` where it belongs. `RecordBatch.stack` replaces
  `RecordArray.stack`, `NumericRecordBatch.to_vector` / `from_vector` replace
  their array counterparts, and a producer that returned a `RecordArray` returns
  a `RecordBatch`.

  `_RecordArrayView` goes with them: a field selection off a batch is an ordinary
  batch, and sibling selections align by their shared level names rather than by
  a parent pointer. `Design` and `FullFactorialDesign` are batches.

  The batch types were built alongside the array ones and then took over, so the
  entries below describe the batch types throughout — this is the only entry that
  names the classes being removed.

### Fixed

- **Reading a distribution no longer modifies it.** `BroadcastDistribution`
  assigned its marginal on the first `marginalize()`, and a backend-delegated
  `DistributionArray` assigned its components on the first read, so a query
  changed the object a caller was holding — against `C2` and the §V.1 promise
  that an implementer's object is never modified. Each now fills a memo container
  assigned at construction, so the result is still computed once and the term's
  own fields stay as they were built. Both remain lazy.

- **Every dispatch presents a one-field draw the same way.** A one-field
  record-valued law — a `ProductDistribution` over a single distribution, say —
  draws a batch of records. The row-wise paths presented each draw as its bare
  leaf; the `vmap` path presented the record. Since the record shim carries
  conversions but deliberately no arithmetic, a body as ordinary as `x * 2`
  succeeded under `dispatch="sequential"` and crashed under the mapped
  executor. All four paths now present a draw through one rule.

  Design II.4 leaves the choice itself open, riding on the single-value
  coercion question `Record` poses. What it does not leave open is that the
  dispatches agree, which is what this restores; the bare-leaf presentation is
  the one three of the four paths already made.

- **A law that cannot report its `dtype` is probed rather than refused.** The
  trace probe read `event_shape` and `dtype` to size a synthetic dummy. Reading
  `dtype` was itself the refusal: `getattr(law, "dtype", None)` swallows only
  `AttributeError`, so a law raising anything else — a
  `SequentialJointDistribution` view raises `NotImplementedError` — failed the
  probe and was sent to row-wise dispatch for want of a placeholder. The probe
  now draws a sample instead, and the draw carries both.

  `dispatch="jax"` consequently accepts cases it used to reject, those views
  above all. They build each component from a Python callable, which is indeed
  not traceable, but that runs while sampling, before the map, so only the body
  is traced; the mapped result matches the row-wise one exactly. An empirical
  law is unaffected, still enumerated so its exact weights are preserved.

- **A body that returns a batch no longer crashes the marginalization path.**
  Calling a `Function` whose body returns a `RecordBatch` with a `Distribution`
  argument raised the pytree rank error out of `jax.vmap` instead of falling
  back to sequential dispatch.

  The trace probe that gates JAX dispatch models the transform its executor
  applies, so that a body which traces cleanly bare but cannot survive the
  transform is caught while a fallback is still available. It did that for the
  sweep executor and not for `_broadcast_jax`, which also maps — over the draw
  axis rather than over batch rows — so a batch-returning body passed the probe
  and then failed inside the executor, where nothing was left to fall back to.
  Both mapping executors are now probed under a map.

- **`copy` and `pickle` no longer drop a term's annotations (#409).** `Record`,
  `NumericRecord`, and `ProductDistribution` each reconstruct through a
  `__reduce__` that listed its state by hand, and none of them listed
  `_annotations`, so a copied or unpickled term came back with its annotations
  gone — the diagnostics and inference-backend payloads written into that store
  among them — and nothing raised. `__reduce__` governs `copy.copy` and `copy.deepcopy`
  as well as `pickle`, so all three paths lost them.

  The omission was systematic rather than careless: annotations are the one field
  written *after* construction — the documented exception to immutability — so a
  state list assembled from constructor arguments misses exactly this one.

  So reconstruction reads the term's own state instead of a list: nothing has to
  name a field for it to survive, and `TrackedTerm._restore_identity` — which
  wrote identity onto an already-constructed object, bypassing both the
  immutability guard and the write-once provenance rule for any caller who found
  it — **is deleted**.

  The container a reconstruction is handed is decoupled from the one it was built
  from, as `with_name` already does: entries are shared, the container is not, so
  a write on a copy does not show through on the original. Annotations still do
  not cross a JAX transform boundary — `tree_unflatten` rebuilds a bare term,
  unchanged.

- **`is_concrete` no longer reports a polymorphic template as concrete (#390).**
  A symbolic dimension declared inside a term spec — a `RecordSpec`'s schema, a
  `DistributionSpec`'s event declaration, a `FunctionSpec`'s either side — was
  invisible to `free_dims`, so `EventTemplate(law=DistributionSpec(x=("obs",)))`
  reported itself concrete. Design II.3 draws no line at a term-spec boundary:
  *any* symbolic entry makes a template polymorphic.

  Reporting a dimension, substituting it, and binding it are now three methods
  every `ValueSpec` answers, so the spec that declares a dimension resolves it.
  `EventTemplate.free_dims` is the union over its children, so a name is reported
  wherever it is declared. Three things follow. Substitution reaches through a
  term spec, so every dimension reported is bindable. **Unification binds through
  one too**: a spec's declaration unifies against the actual term's own, in the
  shared binding scope, so a name inside a `DistributionSpec` is the same
  dimension as that name beside it — it binds once, and a disagreement raises.
  And a `BatchSpec` axis size may now be a symbolic name in that same scope,
  bound from the actual `Batch` it is matched against, so a batch of `("n",)`
  over arrays of shape `("n",)` is square by declaration, and a batch that is not
  square is refused.

  Each spec owns its own binding, which is what reaches a spec the schema layer
  cannot name: `BatchSpec` lives in `_batch.py`, which imports from
  `event_template.py`, so a type test there could report a batch axis as free
  while nothing could bind it. Every spec that reports a dimension implements
  both binding methods — `NumericArraySpec` and `FunctionSpec` included, which the
  unification pass had special-cased — so the four methods are one contract
  rather than a rule with exceptions.

  A `FunctionSpec`'s output binds whatever kind it declares. Only a record
  declaration was read before, so a callable declaring an output that contradicted
  the input bound the input alone and reported an output schema that was wrong
  rather than merely unbound: input `("n",)` against a declared `(3,)` and an
  actual output of `(5,)` reported `(5,)` as `(3,)`. A non-record declaration
  describes the one value returned, so it now meets the sole leaf of the
  callable's output template, and several output fields do not match it.

  This brings the term specs into line with `NumericArraySpec`, which has always
  accepted a concrete value against a symbolic shape and left the sizes to the
  single pass, per II.3's division of labor. A polymorphic term-spec declaration
  was previously unsatisfiable: `is_valid` compared inner templates for exact
  equality, so a symbolic declaration never matched a concrete value.

  A live `Batch` still requires a concrete multiplicity — it holds elements at
  positions — so construction refuses a polymorphic `BatchSpec`, and
  `batch_size` raises until the dimensions are bound.

- **Aliased lifted arguments now co-sample (#388).** Within one lifted call, two
  references to the same law denote one random variable, so they must come from
  one draw. Passing the same `Distribution` to two arguments sampled it twice
  instead, so `f(d, d)` approximated `f(X1, X2)` — a silently wrong answer, with
  `difference(dist, dist)` returning a spread around zero rather than zero.

  Arguments were already grouped by root ancestor, as the co-sampling contract
  requires; the grouping was then discarded for plain distributions and honored
  only for field views. Each group is now drawn **once**, from its root, with
  every member taking its own value out of that draw. Two further cases follow
  from the same change: a parent passed alongside its own view no longer raises
  (it was projected as though the parent were a view), and an empirical passed
  twice contributes **one** enumeration axis rather than a squared grid — over
  three atoms, `f(e, e)` enumerates 3 points instead of 9, each weighted once
  instead of squared.

  Arguments with no common root are unaffected, down to the subkeys: a group of
  one consumes exactly one key split, as before. Only calls that were already
  returning wrong values change their output.

- **A record-valued law can be lifted.** Passing a record-valued
  `Distribution` as an argument raised `TypeError: ... is not array-like`, from
  two places that assumed every argument's samples were an array. Broadcast
  assembly read the row count from the samples' `shape`, which a record batch
  refuses unless it holds exactly one leaf; the count now comes from
  `batch_shape`, the one accessor that means the same thing for every batched
  value. Enumeration also stacked each argument's per-row values with
  `jnp.stack`, which a `Record` row is not; those now stack through
  `RecordBatch.stack`.

  The first of those is what kept `f(d, d["x"])` — a parent alongside its own
  view, the remaining co-sampling case above — from running end to end once its
  draws were shared. Record-valued laws now lift under `auto`, `sequential`,
  `thread`, and `jax` dispatch when the mapped body is JAX-traceable, including
  nested sampled records and repeated roots. Exactly enumerated empirical roots
  still report the exact-enumeration error described above under explicit
  `dispatch="jax"`.

  **The joint those lifts produce also resamples.** `include_inputs=True` keeps
  every input beside the output, and drawing from that joint gathers the same
  rows from each, which is what keeps a drawn tuple paired. A record-shaped
  component has fields rather than a shape, so handing it an array of rows raised
  `TypeError: key must be str, tuple, or int`. Every component now goes through
  one gather that reads the container it is given: an array indexes directly, a
  list of per-row objects gathers positionally, and a record is rebuilt from its
  gathered leaves. The rebuild is deliberate rather than a `jax.tree.map` — a
  `RecordBatch` stores its row count and a `Record` its event template, both in
  pytree aux data, so mapping over the leaves alone would have produced a batch
  quietly claiming the rows it started with. The same gather covers the output
  side, where a vectorized broadcast over a record-returning function leaves the
  output a batched `Record`. A single draw is unwrapped to one record rather than
  a one-row batch, its field names intact.

- **Value specs are fingerprinted by declaration, not identity (#381).** The
  spec hasher now covers `RecordSpec` and recurses into a stored declaration
  (`DistributionSpec.event_spec`, `FunctionSpec.output_spec`), which is a spec
  rather than a template. The generic hasher also routes any `ValueSpec` to it,
  so a spec reached other than as a template leaf — bare, or inside a tuple,
  list, or mapping — records its type and declaration fields instead of falling
  through to identity hashing. Previously such a spec hashed weakly, so two
  *equal* declarations produced different fingerprints and silently broke jit
  cache keys and provenance. Because a record declaration is now stored as a
  `RecordSpec`, the digest of a template carrying a `DistributionSpec`, or a
  `FunctionSpec` with a declared output, also changes value; fingerprints are
  in-memory jit cache keys and provenance only, never persisted.

### Added

- **A sweep whose body returns a batch now vectorizes (#405).** Such a body used
  to fail the JAX trace probe and drop to row-wise dispatch: `vmap` inserts an
  output axis and re-enters `RecordBatch`'s unflatten hook, which has no name to
  give the new level and refuses rather than guess — the pytree contract carries
  neither `in_axes` nor `out_axes`, so *a shape is not a provenance*.

  The refusal stands; the executor no longer routes through it. A body's
  returned batch is taken apart into raw columns for the crossing and rebuilt on
  the far side by the executor, which holds the level names the hook lacked. The
  result carries the sweep's levels followed by the body's, and equals what
  row-wise dispatch produced. The input side already worked this way, rebuilding
  each row's record from raw columns inside the traced call; this is the same
  move on the output side.

  A raw `jax.vmap` over a batch is refused exactly as before. Only an operation
  that knows which axis it added may name the level, which is what the executor
  knows and a bare transform does not.

- **`RecordBatch` / `NumericRecordBatch` — a batch of records, stored columnar.**
  A batch of records that all conform to one `EventTemplate`: the batched value a
  `Function` produces and consumes, such as the many draws a `sample` yields. It
  is a `Batch`, so `len`, `iter`, and `batch_shape` speak about the collection and
  never about what one record contains, and its own type is a `BatchSpec` over the
  `RecordSpec` its elements satisfy.

  **Storage is columnar and keyed by leaf path** — one column per *field*, not per
  top-level child, each shaped `(*batch_shape, *event_shape)`. A field access
  hands back that column directly, and an element is assembled from the columns on
  demand rather than stored twice. Keying by leaf path is what makes a nested field
  reachable: `batch["outer/a"]` is a column like any other and `batch["outer"]` is
  the sub-batch over the columns beneath it, so a nested record batches and reads
  back. A column comes back in the batch form its spec calls for: the array itself
  for an array field, a `FunctionBatch` or an `OpaqueBatch` for a field with no
  native stacked form. Either way it is a **view** — the object batch shares the
  column rather than copying it, so reading a field costs nothing per element.

  **A batch is a collection, not a named tree**, so there is no field-keyed
  `Mapping` protocol — no `keys()` / `values()` / `items()` / `children` /
  `at_path` — and the field structure is read from `event_template`, where it
  belongs. What `[]` does depends on the key: a position addresses the batch axes,
  a name addresses a field within every element. `select` / `select_all` survive as
  the field-splatting selector, resolving a path as `Record.select` does — a key
  gives a one-column view, a partial path the sub-batch under it — and returning
  batch *views* that carry the parent's level names, so an operation aligning
  operands by level name lines them up. `select_all` keys by top-level name, as
  the record's does, since a `/`-path could not bind to a parameter.

  An element is **materialized** rather than stored, which is the other side of the
  rule the batch base states: it takes the derived name (`"post[draw=1]"`), marked
  auto, and inherits the batch's provenance. It is built against the batch's own
  `element_spec`, so batch and element share one spec object — schema agreement is
  structural, and a row costs no declaration to build. `NumericRecordBatch` adds
  the batched flat layout: `to_vector` gives `(*batch_shape, vector_size)` with the
  flat dimension last and the levels kept as the leading axes, and `from_vector`
  inverts it, naming the levels it reconstructs so a multi-level batch round-trips
  and casting each field back to its declared dtype, which concatenating promoted.
  Its columns are the leaves `jit` / `vmap` / `grad` traverse; the batch itself
  is rebuilt only under the transforms the contract below states.

  **Raw pytree transformations have a stated contract**, because a batch cannot
  thread its declaration through a round trip the way a `Record` does: `vmap`
  removes an axis the stored spec still names, so rebuilding against that spec
  verbatim would give back an object whose `batch_shape` its own columns
  contradict, and every method reading that shape — `to_vector` among them — would
  be wrong. What arrives is the only evidence, and **a shape is not a
  provenance**: a no-op round trip and a transpose of a square batch arrive
  identically, and a dropped middle axis with a resized survivor imitates a
  dropped leading one. So two transformations are supported and the rest refused,
  rather than inferring which axis went.

  Supported: a transform that **preserves every batch axis** — the ordinary round
  trip through `jit`, `grad`, and a shape-preserving `tree_map` — which reuses the
  stored spec; and one that **removes every batch axis**, which yields a single
  `Record`, as `vmap` over a single-level batch does.

  Refused: a **partial** rank reduction, including `vmap` over one level of
  several, since no shape says which level survived; an **added** axis, which
  belongs to no level and which unflattening has no name to give one; a
  **resized** axis, which a per-level slice and a slice-composed-with-a-transpose
  reach alike; a column reporting **no shape**, which a stored column never does;
  columns left **disagreeing** on the batch axes, since a batch states one
  multiplicity for all its fields; and a **retyped element**, whose own axes and
  dtype are the element type's rather than the transform's — the kind is
  re-checked, not only a pinned dtype, so a numeric batch cannot come back holding
  objects.

  Preserving every batch axis is a **precondition**, not a check: an axis
  permutation that preserves the shape satisfies neither supported case and cannot
  be detected, so it is unsupported rather than refused. Mapping one level of a
  multi-level batch needs an operation that knows which level it consumed — the
  workflow sweep has that knowledge and never routes through the pytree hook,
  mapping raw columns and building each row explicitly. Indexing is likewise
  exact, since it is told which positions it keeps.

  The structural transforms re-derive the class from their result, as the record
  transforms do: an edit that removes the last non-numeric field promotes, one that
  introduces a non-numeric field demotes, and a mixed `merge` therefore gives the
  same answer whichever way round it is written. An edited field is typed the way
  template inference would type it, so a field of callables stays a function field,
  and `replace` accepts what field access hands back.

  A batch holds what it validated. Every field is checked against what it declares:
  an array field for a numeric dtype its declaration admits, by the same same-kind
  rule `NumericArraySpec.is_valid` applies to one value, and every other field value by
  value against its spec, naming the field and the position that failed. A field
  with no stacked form is stored as a frozen object array, so its entries are the
  values themselves and a caller keeping a handle cannot write in a value the spec
  refuses afterwards. The field's *spec* decides its stored form rather than its
  values, which is what keeps an opaque field opaque when its values happen to be
  numeric.

  This was additive: the batch types were built alongside what they replaced
  and still what the library uses, and the new classes are not yet exported.

- **`EventTemplate.with_dims(**sizes)`** binds symbolic dimensions explicitly,
  returning a new template so refinement stays monotone, and naming any
  dimension left unbound. It reaches through a term spec, and auto-promotes to
  `NumericEventTemplate` when the bound template is all-numeric, so a bound
  template gains its flat layout. The law-level `with_dims` design 03 names on
  `Distribution` will delegate to it.
- **A `Record` stores its `RecordSpec`.** `Record.spec` is the single stored
  source of a record's type, and `event_template` becomes a view on it, so the
  two cannot disagree. Construction accepts either form of a record
  declaration: a `RecordSpec` is stored verbatim and a bare `EventTemplate` is
  wrapped, the two denoting the same space. Everything that reads
  `event_template` is unaffected.

  This is the storage rule the tracked types share — a term carries the spec of
  its kind, and its schema accessors are views on that one object — reaching
  the record side. A `Distribution`'s `event_spec` and a `Function`'s
  `output_spec` follow with their own layers, and the slot moves onto the
  tracked base once every kind carries one. A batched record still subclasses
  `Record` and is not one record, so a batched record's `spec` raised rather than
  reporting an element's spec as the batch's own type: a batch's type specifies
  the collection. That override goes away with the subclassing, when the batch
  types become collections rather than records.

  The JAX pytree aux data is now the `(spec, name, name_is_auto)` triple rather
  than `(event_template, …)`, and pickled records serialize the spec. Aux stays
  hashable and equal for equal declarations, so treedefs still compare by value
  and a jit cache keyed on one is unaffected — including across the two
  declaration forms, which agree. A pickle written before this change still
  loads, its bare template accepted as the declaration it is.

- **`FunctionBatch` and `OpaqueBatch` — the batch forms that store objects.** A
  numeric array batches natively, with the batch axes leading, so it needs no
  class; a callable and an opaque object have no such form, so each gets a thin
  `Batch` that stores its elements and carries the one `element_spec` they all
  satisfy, adding no interface beyond it. Elements go in as a flat sequence or
  as an object array of any shape, with a name required for every level — a
  placeholder would read as meaning something while naming nothing, the same
  reason a level clash is not resolved by suffixing.

  Storage is a numpy object array, chosen for the contract rather than for
  arrays: numpy basic indexing returns a **view**, so a sub-batch shares its
  parent's store in every indexing form, and it honors a descending or stepped
  slice in the order given, which is the order a view's derived names are stated
  in. The store is frozen and a supplied array is copied — only the pointer
  array, so the elements stay shared — so a batch holds the elements it
  validated and a view cannot write through to its parent. Elements are never
  unpacked: a batch of arrays or of lists stays a batch of two things rather than
  becoming one 2-d array, and a container that iterates into its *parts* rather
  than into elements (a string, a mapping, a numeric array) is refused, since
  each would otherwise yield a batch of pieces of one object.

  An element comes back as the object that was stored, untouched — neither
  renamed nor given provenance. Identity and lineage are derived where a batch
  *materializes* an element per index; these store theirs, so what the caller put
  in is what comes out.

  `OpaqueBatch` is the case a batch's own spec exists for even though an
  `OpaqueSpec` accepts arbitrary non-mapping Python objects. Every element is
  checked against the shared spec at construction,
  reporting the position that failed, since a batch asserts that spec of *all*
  of them, and `axis_groups` must tile the shape the elements are stored in, so
  the spec cannot describe a shape the storage does not have.

- **`TermSpec` — the term-spec sub-hierarchy, and declarations stored as specs
  (#381).** `ValueSpec` now splits into *raw-value specs* (`NumericArraySpec`,
  `OpaqueSpec`), which describe the raw hosts held by corresponding tracked
  kinds, and *term specs*, whose concrete class identifies an already tracked
  kind. `TermSpec(ValueSpec)` is the marker
  `isinstance` reads; `is_valid` stays declared once on `ValueSpec`, so a term
  spec is accepted anywhere a leaf is. New `RecordSpec` completes the four
  corners beside `DistributionSpec`, `FunctionSpec`, and the conditional spec
  to come, and `DistributionSpec` / `FunctionSpec` are reparented under
  `TermSpec`.

  An **output declaration** is any value specification, matching the model's
  `Fun(σ, ρ)` with `ρ` a value specification: a callable may declare a term
  result of any kind or a raw-value result, the latter typing the raw host that
  the boundary wraps in its corresponding tracked kind. An **event** declaration is
  narrower, record-valued, because `DistributionSpec.is_valid` checks it.

  A *declaration* — of an event or an output — is now **stored as a spec**: a
  bare `EventTemplate` is accepted as construction sugar wherever a record
  declaration is meant and normalised to `RecordSpec(template)`, so after
  construction the declared kind is simply the stored spec's class. This makes
  an operation's result kind a structural test rather than an inference from a
  runtime value. A callable's `output_spec` accepts any kind; an *event*
  declaration is record-valued for now, since a `Distribution` exposes an
  `EventTemplate` and nothing that reports a term-valued draw kind, so a
  random-measure declaration is refused at construction rather than accepted
  and always reported invalid. `FunctionSpec` declares no check on its output,
  so nothing there is expressible-but-unsatisfiable.

- **`Batch[E]` — the generic multiplicity axis (#350).** New
  `probpipe.core._batch` module holding the tracked nd-collection ABC the
  concrete batch types will specialize. A batch says *how many* objects there
  are, separately from what one object contains, so `len` / `iter` /
  `batch_shape` / `batch_size` speak only about the batch axes and never about
  an element's structure.

  Axes are partitioned into ordered **levels**: `axis_groups` tiles
  `batch_shape` into contiguous groups, outermost first, with `batch_shape`
  their flat concatenation — so `N` laws of `S` draws each are `(N,)` of `(S,)`
  rather than one anonymous `(N, S)`, and anything stated over `batch_shape`
  applies to a multi-level batch unchanged. Each level carries a name
  (`level_names`, repinned by `with_level_names`), and names are unique within a
  batch: an operation minting a level takes the name to give it, and a name
  already present raises rather than being altered, exactly as a rename onto an
  existing name does.

  Indexing has two entry points. `at_levels(**levels)` takes one indexer per
  named level and returns a view — the by-name counterpart of positional `[]`,
  and the level analogue of `NamedTree.at_path`. `[]` itself dispatches on
  whether the key is a **position** or a **name**: a position (an integer, a
  slice, or a tuple of those) addresses the batch axes, while a name (a string,
  or a tuple of strings for a path) addresses a field within every element —
  which a batch of records will answer and a batch of anything else refuses. A
  tuple mixing the two addresses neither, and is refused as a mix rather than as
  a wrong number of indices. A whole axis is written `:` positionally; `None`
  spells it in `at_levels` alone, where a keyword cannot take a `:` literal.

  A view is **named by what it selects**, naming the level each selection
  addresses — `"posterior[chain=0]"` for a sub-batch,
  `"posterior[chain=0, draw=7]"` for an element, `"posterior[draw=1:3]"` for a
  range. Levels selected whole are left out, so selecting all of a batch derives
  the batch's own name, and the levels that appear are listed in the batch's own
  order. The selection is tracked against the batch the name is rooted in, so a
  derived name is a function of what the view selects: indexing two levels in one
  call, in two calls, or in the other order all read alike, and two different
  selections of one batch never do. A selection carries the *lineage* of the batch it came
  out of rather than a node recording the read: nothing is computed by reading one
  position out of a collection, and which position it was is what the name says.

  A batch's **specification is its own**, at the *family* kind: the new
  `BatchSpec` term spec carries the element's specification together with that
  named multiplicity, and a batch stores it as the single source of its type.
  `spec` therefore names the collection, just as any other term's spec names the
  term, while `element_spec`, `axis_groups`, `level_names`, `batch_shape`, and
  `batch_size` are views on it; the level invariants are the spec's own, checked
  when it is constructed. A batch of values naming no kind is specified all the
  same, a raw-value `element_spec` being as well formed as a term spec.

  Element storage is the concrete class's business — the only thing left to a
  subclass, through the `_element_at` and `_sub_batch_at` hooks. The second
  presents a *view* that shares the store rather than copying out of it, which is
  why selecting all of a batch needs no special case. A third hook, `_at_fields`,
  is supplied only where the elements have fields to address by name. Renaming a
  level touches no storage, so it defaults to a shallow copy.

  A batch is immutable, round-trips through `pickle` and `copy`, and reprs as its
  class, its name, and each level with its sizes, reading no element.
  `FunctionBatch`, `RecordBatch`, and `DistributionBatch` follow separately.

- **First-class, tracked `Function` values (#368).** `Function` is now an
  immutable `Node` / `TrackedTerm` / `Annotated` object with a construction-time
  Python `signature`, optional authoritative `input_template` and
  `output_template`, and a raw `apply(*args, **kwargs)` execution boundary.
  `NumericArraySpec` shapes accept symbolic dimension names; templates expose
  `free_dims` / `is_concrete`, and each invocation unifies input and output
  symbols without mutating declarations. Decorated and private-
  implementation-backed Functions share the same planner, invocation-local
  RNG and dispatch state, and output validation.
  Variadic arguments now participate in lifting and sweeps through stable
  per-element planner slots; `Any` on a variadic parameter remains
  non-restrictive rather than suppressing those behaviors. Authoritative nested
  outputs aggregate identically across sequential, threaded, Prefect, and JAX
  dispatch without changing the public stacking contract, and
  declared distribution sweeps expose their concrete schema through
  `DistributionArray.event_template`.
  Callable and private-implementation fingerprints encode frozen signatures
  and templates structurally; callable implementations additionally encode
  their code, defaults, and closure, while private implementations use stable
  opaque-default type fallbacks rather than address-bearing signature strings.
  Authoritative output validation requires field trees and concrete shapes to
  conform, uses same-kind dtype checks for bare values, mappings, and Records,
  and enforces their declared supports against concrete values. An existing
  Distribution must carry an `event_template` exactly equal to the concrete
  declaration; Function neither reconciles parallel `dtypes` / `supports`
  accessors nor rewrites the Distribution's intrinsic template. Consolidating
  Distribution schema ownership remains follow-up work. Support-bearing value
  outputs use row-wise execution under auto dispatch because their
  data-dependent checks cannot run while JAX traces; explicit JAX dispatch and
  direct `jax.jit(Function.apply)` report that limitation. `apply` preserves a
  returned container's original template, while the independent `__call__`
  result copy carries the concrete declared template for Records and retains
  an already-matching Distribution template.
  `LinOp.apply(x)` now delegates to `matvec(x)`, preserving existing operator
  structure and behavior. `Function._from_implementation(...)` is the internal
  construction entry point for dynamically produced ordinary Functions; #370
  will layer fitted-producer validation and attestations over that boundary.
  Result plans and `OperationRef` remain follow-up work.

- **`NamedTree` — the public name-keyed tree substrate (#338).** New
  `probpipe.core.named_tree` module holding the ordered, immutable,
  `/`-path-navigable tree that `EventTemplate` and `Record` now share as their
  common base (previously the private `_NamedTree`). The substrate owns the
  leaf-keyed mapping interface, nested-dict export (`to_nested_dict`) that the
  constructor reads back, and the
  structural edits `merge` / `without` / `replace`, and gains
  `with_path_names(old=new, ...)` — rename leaves or whole subtrees by path,
  or by bare name when unique in the tree. Each family declares its leaf type
  (`ValueSpec` for templates; arbitrary values for records), checked at
  construction, and **a `Mapping` is never a leaf**: a dict field value is
  always materialised into a nested subtree, never stored as an opaque leaf.
  Diagnostics payloads, previously carried as Records with dict-valued fields,
  are now plain dicts.

- **`TrackedTerm` / `Annotated` identity-and-metadata mixins (#336).** New
  `probpipe.core.tracked` module defining the shared identity attributes and methods every
  ProbPipe object carries: `TrackedTerm` (a `name`, a `name_is_auto` flag, and a
  write-once `provenance` attached via `with_provenance`, plus `with_name` for
  rename-as-copy) and `Annotated` (a free-form `annotations` mapping).
  `Distribution` and `Record` / `NumericRecord` inherit both; the batch types
  (`RecordBatch` / `NumericRecordBatch` / `DistributionArray`) are tracked
  terms too. `name_is_auto` records whether an object's name was auto-derived
  by the operation that produced it (`True`) or supplied by the user
  (`False`), so later composition can re-derive auto names while preserving
  user-given ones. The construction-time guarantee that every tracked term
  has a non-empty name is enforced by the mixin's metaclass, replacing the
  previous `Distribution`-only metaclass check and extending it to the
  `Record` family. Both mixins are exported from the top-level `probpipe`
  package.

- **pyabc SMC-ABC inference backend (#238).** A `pyabc_smcabc` method
  (priority 6) for `SimpleGenerativeModel`: it derives a joint pyabc prior from
  the prior's flattened parameter vector (so correlated and multivariate priors
  work, not just products of independent marginals), runs SMC-ABC against the
  observed data, and returns importance-weighted particles keyed by parameter
  name. Auto-dispatches via `condition_on` for a pure generative model. Ships as
  the `[pyabc]` extra (bundled by the `probpipe` metapackage); `make_posterior`
  gains a `weights=` argument so weighted particles flow through to the
  `ApproximateDistribution` without resampling.

- **`probpipe.diagnostics` posterior diagnostics subsystem.** Adds in-place
  mutator operations `add_rhat`, `add_ess`, `add_mcse`,
  `add_mcmc_diagnostics`, `add_ppc`, and `add_loo`; structured
  `posterior.diagnostics` views (`DiagnosticsView`, `MCMCView`, `PPCView`, and
  `LOOView`); and ArviZ-compatible interop through `posterior.arviz_data`.

- **`probpipe.validation` posterior-vs-reference comparison metrics.** A
  dependency-light scoring layer for validating inference methods against a
  trusted reference: `Reference` (a container for analytic / long-NUTS /
  sandwich references — high-precision `(mean, cov)`, `draws`, and/or a target
  `score_fn`), `standardized_mean_error` (Mahalanobis mean error
  `‖Σ_ref^{-1/2}(μ̂ − μ_ref)‖₂`), `relative_cov_error` (operator-norm whitened
  covariance error `‖I − Σ_ref^{-1/2} Σ̂ Σ_ref^{-1/2}‖₂`), `std_ratios`,
  `sliced_wasserstein`, `mmd` (unbiased
  RBF), `ksd` (IMQ kernel Stein discrepancy), and the `score_posterior`
  aggregator.

- **`probpipe.validation` calibration checks.** `simulation_based_calibration`
  (Talts et al. 2018) drives an inference method over many `(θ★, data,
  posterior)` replications and tests whether the rank of the truth among the
  posterior draws is uniform — returning an `SBCResult` with per-parameter rank
  histograms and a KS-to-uniform p-value. `interval_coverage` checks whether
  central credible intervals contain the truth at their nominal rate. Both run a
  backend-agnostic loop over `condition_on`.

- **`quantile` op and `SupportsQuantile` protocol.** `quantile(dist, q)` returns
  per-field quantile(s) at probability level(s) `q`, parallel to
  `mean`/`variance`/`cov`. `RecordEmpiricalDistribution` implements it
  weight-aware via the midpoint-CDF (Hazen, type-5) quantile, used for both
  uniform and non-uniform weights so the estimator is continuous in the
  weights.

- **`ProvenanceMode` enum and `provenance_config` singleton for lineage-tracking
  control.** Three modes are available: `FULL` retains live references to every
  parent distribution (good for interactive debugging); `LIGHTWEIGHT` (the new
  default) stores only `ParentInfo` descriptors — type name, distribution name,
  and the parent's own provenance chain — so parent data arrays are free to be
  garbage-collected once a workflow step completes; `OFF` skips provenance
  entirely for minimum overhead.  The mode is set once at startup:
  ```python
  import probpipe
  from probpipe import ProvenanceMode
  probpipe.provenance_config.mode = ProvenanceMode.FULL  # for debugging
  ```

- **`ParentInfo` descriptor** (new public export).  A frozen dataclass carrying
  `type_name`, `name`, `provenance` (the parent's own `Provenance`, kept in all
  non-OFF modes so the ancestry DAG remains traversable), `fingerprint` and
  `fingerprint_is_weak` (see below), and `parent` (the live parent object, set
  only in FULL mode).

- **`ParentInfo.fingerprint` — classified best-effort hashing for provenance.**
  Every `ParentInfo` descriptor now carries a 16-character best-effort digest
  plus `fingerprint_is_weak`. Known content-bearing values and closure-free
  Python functions receive portable structural fingerprints. Closure-bearing
  functions, bound methods, partials, callable instances, classes, builtins,
  and unsupported objects use process-local identity and are marked weak;
  weakness propagates through composite fingerprints. Large arrays (> 256 MB)
  are sampled at evenly-spaced offsets rather than read in full. Both the
  fingerprint and its classification are visible in `to_dict()` output, so a
  future cross-run `cache_key_fn` can fail closed on weak inputs.

- **`Provenance.create()` factory classmethod.**  Centralises mode-checking:
  reads `provenance_config.mode`, wraps each parent in a `ParentInfo`, and
  returns `None` in OFF mode.  All ~15 provenance assembly sites in the
  codebase now route through this single entry point, so mode behavior is
  uniform everywhere.

### Changed

- **Terms that build a result write it before handing it over.**
  `DistributionArray._from_backend`, `_make_distribution_array`,
  `TFPProductDistribution`'s combined-view build, `make_posterior`'s annotations
  store, and `SequentialJointDistribution`'s conditioning all populated a term
  after allocating it, using plain attribute assignment. They now write through
  `object.__setattr__`, the way a constructor does. No behavior changes — each
  wrote before the object reached a caller — but an assignment guard on every
  tracked term would refuse the old form, and `ApproximateDistribution`'s chain
  concatenation moves to the same memo container as the two lazy reads above.

- **Immutability is a property of being a tracked term (#395).** `TrackedTerm`
  inherits `Immutable`, so assignment and deletion raise on a record, a batch, a
  function, or a template once its constructor has returned — the design's `C2`
  and the §V.1 promise that an implementer's object is never modified, enforced
  rather than documented. Four classes enforced it individually before.

  **The distribution layer is exempt for now.** `Distribution` permits assignment
  and deletion, because the documented way to build an emulator is to subclass a
  random function and train it in place, and fitting has no contract yet that
  returns a new term instead. That is two overrides — `__setattr__` and
  `__delattr__` — and removing both turns the guard on for the other
  seventy-two classes; removing one would leave half an exemption. A test
  asserts `Distribution` is the *only* exempt class, so a second cannot appear
  quietly.

  Construction is unaffected: it runs inside a per-instance window the
  `TrackedTerm` metaclass opens, so a host's `__init__` assigns normally and no
  constructor needed converting. The window closes when `__init__` returns, and
  also when it raises, so a half-built term left behind by a failure is as
  immutable as a finished one. Code that allocates with `object.__new__` and then
  calls a constructor by hand opens the window itself — three sites in the
  package do.

  Nothing changes for a caller: the classes that refuse assignment are the same
  four families as before, and a distribution still accepts assignment *and*
  deletion, the exemption covering both. What changes is
  where the rule lives — in the term hierarchy rather than in four class bodies —
  and that turning it on for the rest is now a deletion.

- **Immutability is one mixin, and a term reconstructs from its state (#395).**
  Four classes spelled out the same guard — three of them hardcoding a class name,
  so `NumericRecord` reported `Record` and `NumericEventTemplate` reported
  `EventTemplate` — and answered the round-trip that immutability forces in five
  different ways. `Record`, `EventTemplate`, `Batch`, and `Function` now mix in
  one `Immutable`, which owns the guard (naming the class the caller touched) and
  the state round-trip.

  What the round-trip carries comes from the attributes the object holds, not
  from a list each class writes out: every assigned slot declared anywhere in the
  hierarchy, a bare-string `__slots__`, and a subclass's instance dictionary
  alike. That is what made the annotations bug (#409) possible — a hand-written
  list cannot name a field written after construction — and it is now impossible
  by construction rather than fixed once. A class names its memos in
  `_transient_state` to keep a cache out of the payload (`NumericRecord`'s lazy
  conversion cache), and a store written in place in `_decoupled_state` so a copy
  takes its own container (the annotations channel).

  Reconstruction allocates the resolved class and restores state, so it no longer
  re-runs a constructor: an `EventTemplate` keeps the class its specs were
  resolved to instead of re-deciding the numeric promotion, and a `Record` keeps
  the exact schema it was written with. `Function` now writes its own state
  through `object.__setattr__` like every other host, so the `_initializing`
  window its constructor used to open is gone and one guard covers every case.

  **Breaking:** a pickle written by an earlier version does not load. The
  reconstruction entry points it names (`_unpickle_record`,
  `_unpickle_numeric_record`, `_unpickle_event_template`) are gone, state being
  restored directly now. Re-generate any persisted records, templates, or
  batches.

- **A joint law draws a `RecordBatch`.** `_sample` on the four joint laws —
  product, sequential, Gaussian, and empirical — returns a `NumericRecordBatch`
  (a `RecordBatch` when a leaf is non-numeric) for a batched draw, over a single
  `draw` level spanning however many axes the `sample_shape` had. An unbatched
  draw is a `Record`, unchanged. The flat-vector reconstruction behind
  `unflatten_value` returns a batch for the same reason, and both classes are now
  exported: a value handed to a caller must be nameable by that caller.

  **A nested draw is one flat batch, not a batch per subtree.** A batch stores one
  column per *field*, so a nested product draws into one mapping over leaf paths
  and the result is a single batch whose `batch["outer"]` is a *view* over the
  columns beneath `outer`. Where the previous nested draw built a record-array per
  subtree, there is now one store, which is also what makes a nested field
  reachable by path.

  **A broadcast that stacks a batch per row names its own level.** Stacking rows
  that are each a batch puts the sweep in front of the levels a row already
  carried — the swept levels' own names in front of the row's — rather than
  refusing nested batched
  records as it used to. Since the columns are leaf-keyed, a nested element needs
  no special case.

  Three consequences for calling code. A batched draw is not a `Record`, so
  `isinstance(draw, Record)` is `False` where it used to be `True`; ask for
  `RecordBatch`, or for either. A batch is a collection, not a named tree, so a
  draw's fields are read from `draw.event_template` rather than `draw.fields` /
  `.items()` / `.at_path()`, while `draw["x"]` and `draw["outer/a"]` are
  unchanged. And a batch has no `mean` / `var` over its batch axis; reduce the
  column, as `jnp.mean(draw["x"], axis=0)`.

  An object-valued law draws a batch on the same terms as a numeric one: the class
  follows the leaves — `NumericRecordBatch` when the template is numeric, the
  permissive `RecordBatch` otherwise — but a batched draw is a batch either way.

  **A transform cannot add a level.** `vmap` strips the mapped axis on the way in,
  which unflattening handles by re-deriving which levels survived; on the way out
  it *adds* one, and an added axis belongs to no level. Unflattening has no name
  to give, so it now raises instead of keeping the stored spec — which returned a
  batch whose `batch_shape` its own columns contradicted, making every method that
  reads the shape quietly wrong. Map over a batch's columns, or build the batch
  where the axis is added.

  **The sweep addresses a multi-level batch by position** — one indexer per
  batch axis, where a flat index read the leading axis alone and ran off its end
  — and the aggregate carries the swept groups' own axis partition, so two
  independent sweeps followed by a batch-returning body mint one level per group
  rather than refusing. An empty declared sweep builds its aggregate from the
  output template, every declared field present at zero rows.

  **Automatic dispatch probes the vmap it is choosing.** A body that traces
  cleanly bare but cannot run under ``vmap`` — one returning a batch, whose
  added axis no level names — now resolves to sequential dispatch, which
  produces the same result by the dispatch-equivalence contract, instead of
  failing mid-call.

  **Levels align by name, and only whole.** Operands carrying the same level
  names zip; operands with no level in common form a product; a level shared by
  operands whose other levels differ is refused — aligning it would broadcast
  the rest, which is not built, and a product would read the shared name as two
  unrelated axes. Operands naming the same levels must also hold them on the
  same axes: the flat shape can agree while the partition does not. A parameter
  annotated `Batch` (or a generic alias such as `Batch[Record]`) takes the value
  whole, since it names a batched container.

  **A transform never resizes the element's own axes.** A per-column slice can
  pass the rank check while shrinking the event, and a transpose reads an event
  axis as a batch axis; both now raise, as does a reduction below the event
  rank. An empty sweep builds its declared fields only when zero rows were
  *expected* — an empty list where rows were expected is a missing-output error,
  not a fabrication. The dispatch probe states the flat batch size exactly as
  the executor does, so a zero-width event column passes under explicit
  `dispatch="jax"`.

  **Shape cannot recover axis provenance, so ambiguity refuses.** A removal
  whose size matches more than one level (``vmap(..., in_axes=1)`` over equal
  sizes) and a permutation of the batch axes (a transpose, which shape alone
  cannot tell from a per-axis resize) both raise rather than guess; a
  distinctly-sized removal now names the level that *survived*, not the
  leftmost that fits. A pinned dtype is held like the event axes, under the
  constructor's same-kind rule. And zero-row sweeps take one aggregation path
  under every dispatch, so the output schema does not depend on how rows would
  have been executed.

  **A same-rank transform cannot lie about sizes.** Slicing a batch's columns
  keeps every axis, so the levels carry over onto the sizes the columns actually
  have; columns left disagreeing about their batch axes are refused rather than
  papered over with the stored spec.

  **A batch is fingerprinted by its levels and its columns.** A multi-field batch
  failed fingerprinting outright, so provenance omitted it; a single-field one was
  hashed as its sole column, omitting the schema and the levels. Both are fixed:
  the spec, the level names and their axis groups, and the raw columns in leaf
  order all contribute.

  A declared `support` on a batched output is checked column by column. Walking a
  batch as a named tree found no children and asked a multi-field batch to convert
  to one array, so two valid columns raised.


- **A batch of records is recognized wherever a batched record was.** A
  `RecordBatch` is deliberately not a `Record`, so every place that recognized a
  batched value by `isinstance(x, Record)`, by a `RecordBatch` subclass check, or
  by duck-typing on `.fields` stopped recognizing one when a batch arrived. None
  of those gates raise — they take the other branch — so a batch would have been
  re-wrapped as a single opaque field, minibatched by its field count, or read as
  a bare array. Each now admits a batch and does with it what it did with a
  `RecordBatch`:

  the `Function` boundary keeps a returned batch as the batch it is and copies it
  into an independent result under the declared output template, validating each
  column against its field; broadcast planning treats a batch argument as a
  batched one and sweeps its rows, handing the body an *element*; the broadcast
  helpers count, gather, and unwrap a batch's rows, a gather keeping the levels it
  started with; a marginal peels a batch's rows axis into a record of batched
  leaves; a field view reads its column out of a batch; the flat-vector boundary,
  the joint log-densities, and the GLM design coercion accept one; minibatching
  reads its row count from `batch_shape` and gathers a field at a time; and the
  ArviZ bridge finds its variables through `event_template`, which a batch has and
  `.fields` is not.

  These are the gates a batch arrives at. Two
  paths are deliberately left for the cutover, each needing a decision rather than
  a wider gate: stacking a list of batched records from a broadcast, which has to
  name the levels the broadcast grid mints, and
  `RecordEmpiricalDistribution`, which requires a `Record` and is the subject of
  #340.

  Level alignment reads the levels an operand **has**. A batched operand carrying
  none of its own — a `DistributionArray`, which is swept by
  its `batch_shape` without being a `Batch` — has an anonymous multiplicity: it
  aligns with nothing by name and products with everything. Standing its parameter
  name in for the levels it lacks made a parameter named `draw` collide with a real
  `draw` level on another operand, refusing a call whose two axes are independent,
  over a level neither operand disagreed about. A `DistributionArray` stays
  levelless, so this is not tied to any one batched-record class.

- **Renamed, for the storage rule (#381):** `FunctionSpec.output_template` is
  now **`output_spec`**, storing any `ValueSpec` or `None`, and
  `DistributionSpec.event_template` is now **`event_spec`**, storing a
  `RecordSpec`. The names now carry their content: `*_template` is always a
  record schema, `*_spec` a declaration of any kind — which is why
  `RecordSpec.event_template` keeps its name.

  To migrate: **positional** construction is unchanged, and both constructors
  still accept an `EventTemplate` and wrap it, so `DistributionSpec(tau)` and
  `FunctionSpec(tau, tau)` keep working. **Keyword** construction moves to the
  new parameter name, and so does every **read** of the old attribute:

  ```python
  DistributionSpec(event_template=tau)    ->  DistributionSpec(event_spec=tau)
  FunctionSpec(tau, output_template=tau)  ->  FunctionSpec(tau, output_spec=tau)
  spec.event_template                     ->  spec.event_spec.event_template
  ```

  Each field is now declared at the type it *stores* — `event_spec: RecordSpec`,
  `output_spec: ValueSpec | None` — with the wider template sugar carried by the
  constructor signature, so a type checker and the generated API reference both
  read the post-construction guarantee. `NumericArraySpec` follows the same split, its
  `dtype` field declared as the `numpy.dtype` it stores rather than the
  `DTypeLike` spellings it accepts.

- **Function calls establish a new result identity and provenance boundary
  (#368, breaking).** Existing operations such as `condition_on` and
  `from_distribution` now record point-call operations as `workflow.<name>`,
  with the called Function as the first parent followed by tracked inputs.
  Resolved ordinary arguments are fingerprinted separately in
  `Provenance.inputs` and do not become ancestry nodes. When an implementation
  directly returns a `Record`, `RecordBatch`, or `Distribution`,
  `Function.__call__` returns a shallow independent result rather than the same
  object, clears the implementation result's provenance, and attaches only the
  current call provenance. Consequently, implementation-domain metadata such
  as `conditioned`, `ess`, or backend algorithm details is not propagated to
  the public call result; a plain point-call result carries `{"func": name}`
  while broadcast and sweep results retain their own execution metadata. Use
  `Function.apply()` when raw identity, provenance, or domain metadata is
  required. Existing operation controls remain provenance metadata. Other
  tracked return values remain event payloads until #369 adds explicit
  term-result planning.

- **`TrackedTerm` renamed from `Tracked` (breaking).** The mixin carrying a
  `name`, a `name_is_auto` flag, and a `provenance` is now `TrackedTerm`, the
  name the design reference uses for what it holds: the objects operations
  consume and produce are *tracked terms*, while templates and specs are
  structural helpers that are not. The private metaclass follows as
  `_TrackedTermMeta`. This is a hard rename with no compatibility aliases, and
  nothing about identity, provenance, or immutability behaviour changes.

- **`WorkflowFunction` renamed to `Function` (#377, breaking).** The public
  decorator is likewise renamed from `@workflow_function` to `@function`.
  This is a hard rename with no compatibility aliases; call, lifting,
  dispatch, RNG, provenance, and output-wrapping behavior are unchanged.

- **`from_nested_dict` and `_flatten_paths` removed — the constructor reads a
  nested mapping directly (breaking).** Under the *"a mapping is never a leaf"*
  invariant, `Record(name, data)` already materialises every nested mapping
  value into a subtree, so `Record.from_nested_dict` /
  `NamedTree.from_nested_dict` (and the private `NamedTree._flatten_paths`)
  added nothing the constructor lacked. Build from a nested mapping with
  `Record(name, data)` and round-trip via `Record(name, r.to_nested_dict())`.
  This also **tightens validation**: an input mixing a `/`-path key with a
  nested-dict value under the same prefix (e.g. `{"y/a": 1.0, "y": {"b": 2.0}}`)
  now raises, where `from_nested_dict` silently reshaped it. `Record.ensure`
  and the workflow-output wrap, which used `_flatten_paths` internally, are
  unaffected.

- **`FunctionSpec` requires explicit input/output `EventTemplate`s (#357, breaking).**
  Neither side accepts a bare `ValueSpec` anymore (previously wrapped into a
  single-field template keyed `"input"` / `"output"`); pass an `EventTemplate`
  or `None`, writing a single-field signature out explicitly —
  `FunctionSpec(EventTemplate(x=()), EventTemplate(out=()))`. This removes the
  `FunctionSpec` / `DistributionSpec` constructor asymmetry (both now take an
  explicit template) and keeps a function's field names caller-chosen rather
  than a fixed placeholder. `LinOp`'s design is aligned in the same spirit
  (its input/output templates carry meaningful names), with the deliberate
  exception that a self-adjoint operator's two sides may share one name.
- **`NamedTree` is generic over its leaf type — `NamedTree[L]` (#356).** The
  shared tree substrate now carries its leaf type as a type parameter, so
  `py.typed` consumers see the real leaf type rather than `Any` on the
  leaf-trafficking surface: `EventTemplate` binds `NamedTree[ValueSpec]` (so
  `template["x"]` type-checks as a `ValueSpec`) while `Record` binds
  `NamedTree[Any]`. The leaf accessors (`[]`, `values`, `items`, `map`) carry
  `L`, and the structure-preserving transforms (`without` / `merge` /
  `replace` / `with_path_names` / `map`) return `Self`.
  Annotations only — no runtime behavior change.

- **`NumericRecord` stores native-form leaves; conversion is lazy at the
  compute boundary (breaking).** Leaves are no longer coerced to `jax.Array`
  at construction: a numeric array or container (`jax` / `numpy` /
  `xarray.DataArray` / `pandas` / registered backends) is stored **verbatim**,
  validated from container metadata only, and converted at the compute
  boundary (JAX pytree flatten, `to_vector`, the single-field scalar shim)
  through a set-once per-leaf cache — so a lazy or disk-backed leaf is not
  materialised at construction, and `record["x"]` returns the native leaf
  (previously always a `jax.Array`). Backend metadata now survives structural
  transforms and pickling because data and metadata never separate;
  `NumericRecord.to_native()` is **removed** (leaves are already native —
  navigation is the export), and `to_numeric()` is the identity on a
  `NumericRecord` (on a plain `Record` it validates rather than
  converts). All-numeric records holding
  native containers **auto-promote** to `NumericRecord` (the previous
  backend-leaf exclusion is removed), and `EventTemplate.infer_from` infers
  `NumericArraySpec` for them. Native leaves are stored by reference (no defensive
  copies). A native container's metadata (an `xarray` leaf's coords / dims /
  attrs, a `pandas` leaf's index / columns) is **part of a record's identity**:
  `Record.__eq__` and `fingerprint()` distinguish it, so two records with equal
  values but different coords are unequal and fingerprint differently.
  `Record.__eq__`, `to_numpy()`, and content fingerprints route through the
  array-backend registry (so a registered non-numpy container is compared and
  materialised by its own hooks) and materialise lazy leaves on demand;
  `Record.__hash__` stays a coarse structural hash (shape + dtype, no values or
  metadata) that does not materialise. The capture/restore aux machinery is
  retired: `AuxHooks` / `register_aux` / `aux_for` and `NumericRecord.aux` are
  **removed**, replaced by the recognition/conversion registry `ArrayBackend` /
  `register_array_backend` / `array_backend_for` (with hooks
  `event_shape` / `numpy_dtype` / `is_numeric` / `to_jax` / `to_numpy` /
  `metadata`) — one registration makes a new container type recognised,
  validated, promoted, converted (including at the eager batch-stacking
  boundary), and content-fingerprinted everywhere at once. `fingerprint()`
  hashes native containers by type + materialised values + identity metadata
  instead of falling to the process-dependent `repr` tier. A pandas
  nullable / masked *numeric* column (`Int64`, `Float64`, `boolean`, Sparse /
  pyarrow numerics) is a first-class numeric leaf. It is stored verbatim, so
  its validity mask stays intact at rest. At the compute boundary it converts
  to its columns' common dense dtype with each NA encoded as `NaN` — the only
  missing-value form `jax` offers — so missing data round-trips as `NaN`. That
  dtype keeps a nullable float at its own width and a complex column complex
  (`Sparse[complex128]` → `complex128`); a nullable integer / boolean promotes
  to `float64` (no integer `NaN`). The generic duck-typing gate still rejects
  a *bare*
  extension dtype (it is not a dense numpy dtype); the pandas backend
  recognises its own masked dtypes. Non-numeric extension dtypes (categorical /
  string / datetime) are not numeric and leave the container a plain `Record`.

- **`Record` / `NumericRecord` construction is name-first, and all-numeric
  records auto-promote (#338, breaking).** The constructors are now
  `Record(name, fields=None, /, *, event_template=None, name_is_auto=False,
  **kw_fields)` — the
  record's name is a required first positional argument, and the old `name=`
  keyword and nameless forms are removed. `Record(...)` whose fields are all
  numeric (bare arrays and scalars, no backend metadata) returns a
  `NumericRecord`; passing an explicit non-numeric `event_template=` pins a
  plain `Record`. Structural transforms (`without` / `merge` / `replace` /
  `with_path_names`) re-derive the numeric axis the same way, and a nested
  record stored as a field is renamed to its field key. An operation that
  assembles a record supplies a meaningful, deterministic name derived from
  its inputs (the producing distribution's or model's name, or a domain term
  such as `"observed"` / `"data"`) and marks it `name_is_auto=True`. The
  pytree registration now carries
  the event template and identity in the treedef aux data, so
  `jax.tree_util.tree_map` over a `Record` preserves its template, name, and
  auto flag. Value-level (de)serialization entry points moved onto the value
  types: `Record.from_field_values(name, template, values)` replaces
  `EventTemplate.from_field_values(values)` (removed), and
  `NumericRecord.from_vector(name, template, vec)` replaces
  `NumericEventTemplate.from_vector` (removed) as the classmethod inverse of
  the value-level `NumericRecord.to_vector`. `Record.from_dict` likewise takes
  the name first. Construction now validates each
  leaf against its field spec's `is_valid` (structure only: shape and dtype,
  the latter by `numpy.can_cast` same-kind, so a cross-kind dtype raises). A
  `NumericArraySpec`'s `support` is descriptive metadata and is not checked by
  `is_valid` — a data-dependent check that is not `jax.jit`-traceable.

- **Leaf specs unified under a `ValueSpec` base with `is_valid` (#337,
  breaking).** `NumericArraySpec` / `OpaqueSpec` / `DistributionSpec` / `FunctionSpec`
  now subclass a common `ValueSpec` ABC, and every spec implements
  `is_valid(value) -> bool` — a structural check that a concrete value matches
  the spec (shape and dtype for arrays — a `NumericArraySpec`'s `support` is
  descriptive metadata and is **not** checked by `is_valid`, being
  data-dependent and not `jax.jit`-traceable; `OpaqueSpec` accepts any
  non-mapping value; a `DistributionSpec` requires a `Distribution` carrying an
  equal `event_template`; `FunctionSpec` requires a callable). A non-matching
  value
  returns `False` rather than raising, but an unexpected error from inspecting
  a malformed value is left to propagate rather than masked as invalid. The
  `LeafSpec` type alias is **removed**; use `ValueSpec` (exported from
  `probpipe`). **Renamed** `DistributionSpec.inner_template` →
  `event_template`. `FunctionSpec`'s `input_template` / `output_template` are
  now optional (default `None`, meaning "structure unspecified", so a bare
  `FunctionSpec()` describes any callable); either may still be given as a
  bare `ValueSpec`, wrapped in a single-field template (fields `input` /
  `output`). `NumericArraySpec` fixes: `dtype` is
  normalised to `numpy.dtype` at construction so equal dtypes compare and hash
  equal however they were spelled (the field is annotated `DTypeLike`
  accordingly), and a spec with an unset `dtype` no longer compares equal to
  one with a concrete dtype (the two previously compared equal but hashed
  apart).

- **Identity attributes and methods renamed to the design-reference vocabulary (#336,
  breaking).** The duplicated per-class naming/provenance/metadata code on
  `Distribution` and `Record` is replaced by the `TrackedTerm` / `Annotated`
  mixins, with a hard rename (no aliases): `source` → `provenance`,
  `with_source(...)` → `with_provenance(...)`, `renamed(...)` →
  `with_name(...)` (rename provenance now records the operation as
  `"with_name"`), and the `auxiliary` metadata store → `annotations`
  (`_auxiliary` → `_annotations`; a `DataTree` remains a valid value and the
  diagnostics accessors are unchanged). `make_posterior`'s `auxiliary=`
  keyword is now `annotations=`. `ParentInfo` fields follow the reference:
  `obj` → `parent` and `source` → `provenance` (the serialized provenance
  dict key changes accordingly). `BootstrapReplicateDistribution`'s internal
  source-distribution slot is renamed `_source_dist` and is now included in
  content fingerprints (previously it was unintentionally skipped, so
  replicates of different source distributions could fingerprint equal).

- **Leaf-keyed named-collection surface for `Record` / `EventTemplate` (#326,
  breaking).** The mapping protocol (`keys`, `values`, `items`, `__iter__`,
  `__len__`, `__contains__`, `__getitem__`) is now keyed by **leaves** — every
  field's full `/`-path in canonical first-appearance order — instead of the top
  level only. `record["x"]` therefore reaches a leaf and raises on an interior
  node; navigate to a leaf **or** a subtree with `record.at_path("x")`, and use
  `record.children` for the one-level view. **Removed** `EventTemplate.leaf_paths`
  (use `keys()`), `to_leaf_list` (use `list(values())`), `from_leaf_list` (use
  `from_field_values`), and `map_with_names` (use `map_with_keys`). `Record.fields`
  and `Record.to_dict` survive as **temporary** aliases for `children` and
  `to_nested_dict`. `RecordBatch` / `NumericRecordBatch` keep a top-level mapping
  for now, pending the batch-axis rework.

- **`EventTemplate` moved to its own module and `Record` now carries an
  authoritative `EventTemplate` (breaking changes to the value-model surface).**
  `EventTemplate` / `NumericEventTemplate` and the leaf specs now live in
  `probpipe.core.event_template` (the public `probpipe.*` exports are unchanged).
  A `Record` stores its `EventTemplate` rather than re-deriving it on access.
  Several methods moved or were removed:
  - **Removed** `EventTemplate.pack`, `numeric_fields`, `non_numeric_fields`,
    `event_shapes`, and `field_event_shape`; **removed** `Record.flatten` /
    `Record.unflatten` (use `jax.tree_util.tree_flatten` for the JAX-pytree path).
  - **Renamed** `EventTemplate.from_record` → `EventTemplate.infer_from`
    (best-effort, lossy inference). The value upcast is consolidated to
    `Record.to_numeric()`.
  - **Moved** `leaf_shapes` onto `NumericEventTemplate`; `numeric_leaf_shapes`
    is consolidated into `leaf_shapes`. (`to_vector` / `from_vector` are now
    value-level methods on `NumericRecord` / `NumericRecordBatch` — see the
    value-model entry above — not template methods.)
  - **Added** leaf-keyed (de)composition: the mapping protocol
    (`keys` / `values` / `items` / `__iter__`) enumerates every leaf by its
    canonical `/`-path. Reconstruction from a leaf list is now
    `Record.from_field_values` (the former `EventTemplate.from_field_values`
    was removed; see the value-model entry above).

- **User Guide notebooks moved from the former examples section.** The docs nav
  and grouped overview now list all 11 User Guide notebooks under
  `/user_guide/.../`, including the Prefect scalability guide.

- **Adopt `ruff format` for code formatting.** Formatting is now owned by
  `ruff format` (Black-style) rather than the previous manual horizontal-packing
  conventions: the source tree was reformatted in one mechanical sweep (recorded
  in `.git-blame-ignore-revs`), a `ruff-format` pre-commit hook reformats on
  commit, and `ruff format --check` is a **blocking** CI step. Notebooks are
  excluded so the docs' tutorial cells keep their hand layout; string quotes
  normalize to double. See
  [CONTRIBUTING.md § Code formatting](CONTRIBUTING.md#code-formatting).

- **`provenance_ancestors()` now returns `ParentInfo` descriptors, not live
  Distribution objects (breaking change).**  Under the previous always-live
  model, every element of the returned list was a `Distribution` or `Record`
  that could be sampled, inspected, etc.  Under the new LIGHTWEIGHT default,
  elements are `ParentInfo` instances:
  ```python
  # Before
  ancestor = provenance_ancestors(result)[0]   # Distribution
  ancestor.sample(key, (10,))                  # worked

  # After (LIGHTWEIGHT default)
  ancestor = provenance_ancestors(result)[0]   # ParentInfo
  ancestor.name                                # "prior"
  ancestor.obj                                 # None — parent may be GC'd

  # To restore live-object access, opt in to FULL mode
  probpipe.provenance_config.mode = ProvenanceMode.FULL
  ancestor = provenance_ancestors(result)[0]
  ancestor.obj                                 # live Distribution
  ancestor.obj.sample(key, (10,))              # works
  ```
  Code that checks `x in provenance_ancestors(result)` or accesses
  `.samples` / `.log_prob` on ancestors needs to be updated — either
  switch to FULL mode, or use `ancestor.name` / `ancestor.type_name` for
  identity checks.
- **Two-distribution packaging: `probpipe-core` (minimal) and `probpipe`
  (core + all backends) (#237).** The root distribution is renamed `probpipe-core` (minimal JAX base —
  every inference backend is an optional extra), and a new code-less `probpipe`
  metapackage (`packaging/probpipe/`) pins `probpipe-core` and bundles the
  backends the docs exercise — PyMC, nutpie, pyabc, and BayesFlow (the last
  marker-guarded `python_version < "3.14"`) — so `pip install probpipe` runs every example and
  tutorial (on Python 3.12–3.13; 3.14 omits BayesFlow until upstream lifts its
  cap). The `probpipe` **import** name is unchanged in both. Extras not
  already bundled (`prefect`, `viz`, `stan`) are re-exported on the metapackage,
  so `pip install "probpipe[<extra>]"` works alongside
  `pip install "probpipe-core[<extra>]"`. The package `authors` metadata is set
  to the ProbPipe Development Team, with the full contributor list in `AUTHORS`.
  Existing from-source installs should reinstall to pick up the renamed
  distribution. (CI to build and publish both distributions follows in a
  separate PR.)

- **Nested `ProductDistribution` support in the record layer (#262).**
  `RecordBatch` accepts slash-delimited paths in string indexing
  (`arr["outer/a"]`) and integer-indexes a nested array into a nested record
  element; `flatten` / `unflatten` recurse into nested record fields in
  depth-first leaf order; and a batched draw from a nested `ProductDistribution`
  is a canonical, flattenable nested record array.

- **Citation metadata and a "Cite" / "Help" docs section.** A
  `CITATION.cff` enables GitHub's "Cite this repository" button; the
  README gains a "Citing ProbPipe" section with a BibTeX entry; and the
  docs site adds a [Cite](https://tarps-group.github.io/prob-pipe/cite/)
  page (software citation + how to cite the inference backends) and a
  [Help](https://tarps-group.github.io/prob-pipe/help/) page (where to
  ask questions / file issues). The Zenodo DOI is minted from the first
  tagged release and is dropped into the BibTeX/CFF once available.

- **"Open in Colab" badges on the tutorials.** Both tutorial notebooks
  (Getting Started, Flexible Inference) gain an "Open in Colab" badge and a
  guarded setup cell that, *only when run on Colab*, installs ProbPipe with
  the extras the notebook uses (and, for Getting Started, fetches the
  dataset). The cell is a no-op in local Jupyter, the docs build, and CI, so
  notebook execution elsewhere is unaffected.

- **Keyword form for the `log_prob`-family ops (#228).** `log_prob`, `prob`,
  `unnormalized_log_prob`, `unnormalized_prob`, and the `random_*_log_prob`
  ops accept named field arguments —
  `log_prob(model, intercept=0.0, slope=0.5, X=X_obs, y=y_obs)` — built into a
  single draw via `Distribution._pack_value` (single-field → the bare field
  value; multi-field → a `Record`), whose general field validation and
  `Record` building is the new public `RecordTemplate.pack`. The ops stay plain
  `Function`s that resolve this in their body — the same shape as
  `condition_on`'s named data kwargs — so the positional form (including
  `value=`) is unchanged and still broadcasts, and per-call controls use
  `with_options` (`log_prob.with_options(seed=0)(dist, value)`). The keyword
  form is purely additive; the one case it cannot express is a distribution
  whose field name collides with the op's own `value`/`dist` parameter — for a
  multi-field distribution pass a positional `Record`, for a single-field one
  the bare positional value (mirroring `condition_on`'s `observed`).

- **`StanModel` participates in the `log_prob` keyword form with one field per
  Stan parameter (#228).** `StanModel` and its unconstrained view gain a
  `record_template` exposing one field per Stan parameter *block* — e.g.
  `theta` for `vector[3] theta`, `L` for `matrix[2, 2] L` — with each block's
  full multidimensional shape reconstructed from BridgeStan's flattened
  parameter names. The keyword form assembles the flat parameter vector
  BridgeStan consumes — `log_prob(model, mu=0.0, theta=theta_vec, L=chol)` —
  placing each scalar by its parsed index so matrices pack in BridgeStan's
  column-major order; a flat array may still be passed positionally.

### Changed

- **`RecordTemplate` → `EventTemplate` rename + leaf-spec representation
  (#235, Phase 1a).** `RecordTemplate` is now `EventTemplate`,
  `NumericRecordTemplate` is `NumericEventTemplate`, and
  `Distribution.record_template` is `event_template` (hard rename, **no
  deprecation alias** — pre-stable). Template leaves are now a closed sum of
  frozen, hashable specs (`NumericArraySpec` / `OpaqueSpec` / `DistributionSpec` /
  `FunctionSpec`) instead of `tuple[int, ...] | None`; construction-time sugar
  is preserved (`EventTemplate(x=(3,), label=None, sub=…)` still works) and
  `__getitem__` now returns the spec object (shape access stays on
  `leaf_shapes` / `event_shapes` / `field_event_shape`). Behavior-preserving
  otherwise.

- **`SupportsLogProb` / `SupportsUnnormalizedLogProb` are now generic in the
  sample type (#228)** — `SupportsLogProb[T]`. Annotation-level only; runtime
  behavior is unchanged.

- **`StanModel.fields` now returns one name per Stan parameter block (#228)**
  rather than one per scalar (`vector[3] theta` is the single field `theta`,
  not `theta.1` / `theta.2` / `theta.3`). BridgeStan's flat, per-scalar names
  remain available unchanged via `parameter_names`.

- **Amortized SBI learners accept nested priors (#262).**
  `learn_amortized_posterior` / `learn_amortized_likelihood` /
  `learn_amortized_ratio` now train on nested `ProductDistribution` priors,
  iterating the prior's numeric leaves; for NPE, per-leaf bijectors run at each
  leaf's native event shape, and posterior draws come back under their nested
  leaf names. Previously a nested prior was rejected up front.

- **Install docs: a "New to Python?" two-route fork.** The README and docs
  landing now split installation into a newcomer path (uv manages Python and
  the environment — no prior Python needed) and an experienced-user pip path,
  and note that ProbPipe installs from source (not yet on PyPI). The optional-
  extras list also gains the previously-missing `bayesflow` extra.

### Fixed

- **ml_dtypes arrays (bfloat16, float8, int4) now classify as numeric
  (#343).** The numeric-dtype gates previously keyed on numpy's
  `dtype.kind`, under which the ml_dtypes extension types JAX registers
  report `"V"` (void) — so a bfloat16 array failed `NumericArraySpec.is_valid`,
  inferred as an `OpaqueSpec`, and was rejected as a `NumericRecord` /
  `NumericRecordBatch` leaf. All five gates (template inference, spec
  validation, the two record-layer leaf checks, the broadcast-template
  builder, and the `Design` marginals probe) now route through one shared
  predicate that also admits ml_dtypes numerics; structured (record)
  dtypes remain non-numeric. The internal `_NUMERIC_DTYPE_KINDS` constant
  is removed in favor of the shared predicate.

- **Core container indexing and nested reductions.** `DistributionArray`
  integer indexing now raises `IndexError` for positive overflow and negatives
  past the axis bounds, while 0-d arrays accept only empty-tuple indexing.
  `NumericRecordBatch.mean()` and `.var()` now recurse through nested numeric
  record fields instead of treating nested records as arrays.

- **Linear-algebra and Gaussian-conditioning edge cases on the algebra bug-fix
  branch.** `RootLinOp.diag()` now squares diagonal roots; `CholeskyLinOp`
  keeps lower-root (`L @ L.T`) and upper-root (`U.T @ U`) representations
  consistent across `cholesky`, `to_cholesky_representation`, `matvec`,
  `rmatvec`, `matmat`, `rmatmat`, `diag`, `to_dense`, and `solve`;
  `JointGaussian.condition_on` uses linear solves instead of forming explicit
  covariance inverses; and `SumLinOp.matmat` / `rmatmat` preserve the `(n, 1)`
  matrix shape for single-column inputs.

- **Invalid log-space weights are rejected before normalization.**
  `Weights(log_weights=...)` now rejects `NaN` entries and zero-total-mass
  inputs such as all `-inf`, avoiding downstream `nan` normalized weights while
  still allowing individual `-inf` entries for zero-weight atoms.

- **`StanModel` now works against a real BridgeStan backend.** Two bugs at the
  BridgeStan boundary were hidden by the mocked tests: construction passed a
  `data=` keyword that `bridgestan.StanModel.from_stan_file` does not accept,
  and JAX arrays were handed to a ctypes interface that requires `float64`
  NumPy arrays. Construction now goes through BridgeStan's supported
  constructor — which takes the `.stan` path directly and serializes the data
  dict — and every value crossing into `param_constrain` / `param_unconstrain`
  / `log_density` is coerced to a `float64` ndarray, so `StanModel(stan_file)`
  and `log_prob(stan_model, ...)` succeed end to end. The `stan` extra now pins
  `bridgestan>=2.7` (the first release with that constructor), and a
  compile-gated integration test guards this boundary against future drift.

- **nutpie sampling of a `StanModel` keeps its construction-time data.** The
  nutpie path rebuilt the BridgeStan model from the conditioning data alone,
  dropping any data passed to `StanModel(file, data=...)` — so a model carrying
  fixed data (sizes, covariates) failed on the missing variables when sampled
  via nutpie, while the CmdStan path worked. The conditioning data is now merged
  on top of the construction-time data (conditioning values override), matching
  the CmdStan method.

- **`condition_on` no longer silently ignores a case-mismatched data kwarg
  (#228).** Passing `condition_on(model, x=...)` when the field is `X` used to
  route `x` to the inference parameters, where it was silently dropped (e.g. by
  NUTS) — a wrong result with no error. A kwarg that matches a field only up to
  case now raises a `TypeError` with the correct casing (`did you mean X=...?`);
  unknown kwargs that are *not* a case-variant of any field remain inference
  parameters.

- **Codecov no longer misreports coverage on targeted PRs (#261).**
  On a PR that ran only the changed-files test path, the main test job
  skipped its Codecov upload while the BayesFlow job still uploaded, so
  Codecov computed project/patch from the BayesFlow report alone —
  yielding spuriously low numbers and a "HEAD has 1 upload less than
  BASE" warning even though every Actions job passed. Now: the main
  test job uploads coverage on the targeted path too (tagged `unit`),
  so **patch** coverage is accurate and stays an enforced PR gate;
  Codecov **project** is `informational` on PRs (the real 88% floor is
  enforced in-CI on the full-suite run via `--cov-fail-under`); the
  BayesFlow leg is gated to run only on BayesFlow-relevant changes; and
  per-flag `carryforward` keeps the project number sane when a flag
  isn't uploaded.

- **Package license metadata corrected to Apache-2.0 (was MIT).**
  `pyproject.toml` declared `license = { text = "MIT" }` while the
  repository's `LICENSE` is Apache License 2.0 — and the metadata field is
  what PyPI displays. The field is now a PEP 639 SPDX expression
  (`license = "Apache-2.0"` with `license-files = ["LICENSE", "AUTHORS"]`),
  so built distributions carry `License-Expression: Apache-2.0` (core
  metadata 2.4). The setuptools build floor rises from 61 to 77.0.3 — PEP
  639 support landed in 77.0.0, which also deprecated the old
  `license = { text = ... }` table form, and 77.0.3 relaxed the new
  `license-files` validation from errors to warnings — and the redundant
  `wheel` build requirement is dropped (`bdist_wheel` ships inside
  setuptools since 70.1). Build-time changes only; runtime dependencies
  are unchanged.

### Added

- **Contributor conventions for comments, naming, tests, and PR hygiene.**
  CONTRIBUTING.md gains "Code comments & docstrings" (no process narration,
  no negative documentation, public docstrings describe behavior) and "Test
  quality" (tightest reliable tolerances, structured cases, dispatch-path
  equivalence) sections, a description-equals-final-state PR rule, and a
  docs-ship-with-the-change rule. STYLE_GUIDE.md gains §1.12 "Naming
  accuracy" (semantic accuracy, ecosystem alignment, symmetry, complete
  rename sweeps). The `review-pr` skill now checks all of these and reads
  the convention docs from the PR's base ref.

- **BayesFlow amortized-SBI backend (`[bayesflow]` extra).** New
  `learn_amortized_posterior(prior, simulator, method="npe"|"fmpe"|"cmpe",
  ...)` trains a jax-native (keras-on-JAX) amortized neural posterior
  estimator — NPE (coupling flow), FMPE (flow matching), or CMPE
  (consistency model) — and returns a `BayesFlowModel` bundling the joint
  model (prior + simulator, exposed as properties) with the trained
  estimator: `condition_on(model, observed)` draws from `p(theta | observed)`
  in a single network forward pass (no MCMC). This restores the amortized
  half of the SBI layer dropped with sbijax.
  - Training simulates `(theta, y)` offline (`prior` drawn via the `sample`
    op, `simulator.generate_data` for the data); the prior is used only to
    draw `theta` and needs no TFP translation. The trained estimator is
    amortized — the same instance conditions on any observation with no
    retraining — and its draws are named via the prior's `record_template`.
    The simulator receives the prior's native structured per-draw sample (named
    fields), matching the `GenerativeLikelihood` contract, and keras training is
    seeded for reproducibility.
  - Continuous priors with constrained supports — including matrix- and
    simplex-valued ones (positive, an interval, Dirichlet's simplex, Wishart's
    positive-definite matrices, …) — are handled by per-field `bijector_for`
    reparameterization applied at each field's native event shape: training runs
    in the unconstrained space and draws are mapped back to the support (identity
    for real-valued fields). NPE's coupling-flow minimum is counted in
    unconstrained dimensions. Discrete priors have no smooth bijector and are
    rejected with a clear error.
  - Training seeds keras for reproducibility but snapshots and restores the
    caller's global NumPy / Python RNG state, so a call does not perturb
    unrelated random streams.
  - The `[bayesflow]` extra is **Python 3.12–3.13 only** (BayesFlow 2.x caps
    `<3.14`); keras runs on the JAX backend (`KERAS_BACKEND=jax`) — no
    TensorFlow or PyTorch. The backend is imported lazily, so `import
    probpipe` does not load keras.

- **jax-native NLE and NRE (`[bayesflow]` extra).** New
  `learn_amortized_likelihood(prior, simulator, ...)` (neural likelihood
  estimation: a conditional coupling flow for `p(y | theta)`) and
  `learn_amortized_ratio(...)` (neural ratio estimation: an NRE-C classifier
  for the likelihood-to-evidence ratio) return `BayesFlowLikelihood` /
  `BayesFlowRatio` — `ConditionallyIndependentLikelihood` components whose
  `log_likelihood` is **jax.grad-transparent**, so
  `SimpleModel(prior, learned)` + `condition_on` samples the posterior with
  the existing BlackJAX/TFP NUTS machinery. No PyTorch: this replaces the
  planned sbi-torch default path (verified by the Step-6a spike — gradients
  finite-difference-exact and NUTS recovering analytic posteriors, including
  discrete-observation + constrained-parameter cases for NRE).
  - Per-row scores sum under conditional independence, so datasets of any
    size work natively (NPE's conditioning shape is fixed at training time),
    and `per_datum_log_likelihood` comes for free.
  - The networks take raw constrained `theta` as *input* (no bijector
    reparameterization needed on that side); discrete-valued parameter
    fields are accepted. NLE's default coupling flow needs observations with
    >= 2 dimensions and a reverse-differentiable density (adaptive-ODE
    networks such as `FlowMatching` integrate `log_prob` with a dynamic-bound
    `while_loop`, which JAX cannot reverse-differentiate); NRE's MLP
    classifier has neither restriction and handles discrete observations.
  - `learn_amortized_likelihood(dequantize=True)` supports integer-valued
    observations via uniform dequantization (Theis et al. 2016; Ho et al.
    2019, Flow++): training adds `U[0,1)` jitter to the simulated `y` and the
    wrapper scores integer data at the unit-cell midpoint `y + 1/2`. Without
    it, the continuous fit measurably overdisperses the posterior as
    observations concentrate on few atoms.
  - `BayesFlowRatio` values are log-ratios — valid for conditioning (the
    evidence constant cancels) but not for absolute-likelihood uses (model
    comparison, LOO/WAIC); the caveat is documented on the class.

- **`ProductDistribution.supports`** — per-field support constraints (each
  component's `support`), implementing the canonical `RecordDistribution`
  accessor that previously raised `NotImplementedError`.

- **Python 3.14 to the CI test matrix.** The matrix is now
  `[3.12, 3.13, 3.14]`. `requires-python = ">=3.12"` is unchanged.
- **Coverage floor enforced at 88%** on the full-suite CI run
  (`--cov-fail-under=88`). The changed-files-only PR path and local
  single-file runs are exempt (`--cov-fail-under=0`), since a global floor
  is only meaningful when the whole suite executes. Current measured
  coverage on `main` is ~91%; the floor is set conservatively within the
  beta plan's ≥85–90% commitment to leave headroom for normal fluctuation.
- **Concurrency cancellation on CI for PR pushes.** A new push to a PR
  branch cancels the prior in-progress CI run. Pushes to `main` are
  unaffected (no cancellation — the merge-history gate stays solid).
  Same pattern added to the docs build (PR builds cancel; pages deploys
  still serialize via the original `pages` group).
- **PR auto-labeling.** `.github/workflows/labeler.yml` +
  `.github/labeler.yml` apply `area:*` labels to PRs based on changed
  file paths. `kind:*` and `status:*` labels are still applied by
  humans.
- **Dependabot for GitHub Actions.** `.github/dependabot.yml` opens
  weekly PRs that bump pinned action versions (`actions/checkout`,
  `astral-sh/setup-uv`, `codecov/codecov-action`, `actions/labeler`).
  Auto-labeled `area:infrastructure`. Pip/uv dependency bumps are NOT
  enabled — the JAX/TFP resolver interaction means lockfile updates
  must be intentional.

### Changed

- **Pyright type checking (advisory).** A `typecheck (advisory)` CI job
  runs [pyright](https://microsoft.github.io/pyright/) over the `probpipe`
  package and reports type issues; it is **advisory for now** (does not
  gate merges) while the type-debt baseline — largely JAX/TFP
  untyped-attribute noise — is burned down. Config lives in
  `pyrightconfig.json` (`basic` mode, `reportMissingTypeStubs` off). Run
  locally with `uv run --with 'pyright[nodejs]' pyright`. To enforce
  later: drop the job's `continue-on-error` and tighten
  `typeCheckingMode`. See [CONTRIBUTING.md](CONTRIBUTING.md#type-checking).

- **Dev tooling: migrated to [uv](https://docs.astral.sh/uv/) for
  environment + dependency management.** `uv.lock` is committed and used
  by CI (`uv sync --frozen`), making the install reproducible and lifting
  the duplicated inline jax/jaxlib/tfp-nightly pins out of the workflow
  files. Local dev: `uv sync --extra dev --extra nutpie [--extra pymc]`,
  then `uv run pytest`. The pip path (`pip install -e ".[dev]"`) still
  works for contributors with an existing pip setup. See
  [CONTRIBUTING.md](CONTRIBUTING.md#installation).

- **Ruff linting + pre-commit hooks.** The `lint & format` CI job runs
  `ruff check` over the whole tree as a **blocking** gate. A
  `.pre-commit-config.yaml` (install with `uvx pre-commit install`) runs
  ruff (lint + format) plus file-hygiene hooks on staged files. The lint
  config (`[tool.ruff.lint]`) selects the `E`/`W`/`F`/`I`/`UP`/`B`/`SIM`/`RUF`
  families, ignores the ambiguous-unicode rules (`RUF001/2/3` — false
  positives on mathematical notation), and excludes notebooks (executed in
  CI instead). See [CONTRIBUTING.md](CONTRIBUTING.md#linting--pre-commit).

- **`pymc_nuts` reclaims multi-core sampling.** The method previously
  forced `cores=1` to avoid an `os.fork()` deadlock against JAX's worker
  threads. It now samples one worker per chain (capped at the CPU count,
  overridable via a `cores=` kwarg) using the **`spawn`** multiprocessing
  start method — clean worker processes with no inherited threads, so it
  is deadlock-free on every platform (POSIX `fork`, the deadlock-prone
  default on Linux, is never used). Empirically `cores=2` spawn is no
  slower than the old single-core path; a new test exercises the
  multi-core path after spinning up JAX's threads to reproduce the
  hazard.

- **Ecosystem cutover to arviz 1.x and pymc 6 (breaking).** The core
  `arviz` pin moves `>=0.13,<1.0` → `>=1.1,<2.0`, **dropping arviz 0.x
  entirely**, and the `[pymc]` extra moves `pymc>=5.28` → `pymc>=6`
  (pymc 5.x hard-caps `arviz<1.0`; pymc 6.0 is the first
  arviz-1.x-compatible release). ProbPipe now binds the arviz 1.x split
  packages **by name** — `arviz_base.from_dict` (`build_mcmc_datatree`),
  `arviz_base.from_cmdstanpy` (the CmdStan method), and `arviz_stats.*`
  — never bare `import arviz`; the runtime 0.x/1.x version probes are
  removed and the auxiliary is an arviz 1.x `xarray.DataTree`
  throughout (`ApproximateDistribution.warmup_samples` reads
  `aux.children`). `[pymc]` additionally requires `matplotlib` (the
  pymc 6 sampler progress bar imports it). Internal pymc-6 fix:
  `pm.sample_prior_predictive(samples=)` → `draws=` (the `samples=`
  kwarg was dropped in pymc 6). The
  ArviZ 1.0 defaults — credible interval 0.94 → 0.89 and HDI → ETI —
  are adopted as-is: no affected statistic is called in ProbPipe
  source, so no `rcParams` pin is needed, and a frozen-fixture suite
  (`tests/inference/test_arviz_regression.py`) locks the 0.89/ETI
  defaults plus golden `arviz_stats` values as a tripwire against
  future silent default drift.

- **Core jax / jaxlib floor raised to `>=0.9`; blackjax to `>=1.4`.**
  With sbijax gone (see *Removed*), the `<0.9` jax / jaxlib cap that
  existed solely to keep sbijax's hard `jax==0.8.1` pin satisfiable is
  lifted, and the floor moves up to `>=0.9` — the verified stack
  resolves to jax 0.10.1. `blackjax` moves `>=1.3` → `>=1.4` (held at
  `1.3` only to spare sbijax's jax 0.8.1 environment). Users pinned to
  jax 0.8.x must upgrade to `>=0.9`. Isolated as its own PR for
  bisectability given the broad RNG / numerics blast radius across
  every JAX backend. The arviz `<1.0` ceiling and the `pymc>=6` bump
  are intentionally *not* changed here — each lands in its own isolated
  PR (the arviz-1.x ceiling lift and the pymc 6 upgrade).

### Removed

- **sbijax dropped (breaking).** The `sbijax`-backed simulation-based
  inference (SBI) layer is removed in full, ahead of the PyMC 6 /
  ArviZ 1.0 ecosystem upgrade — `sbijax` constrains the jax / jaxlib
  floor and blocks the rest of the stack from moving forward. No
  replacement ships in this release; the SBI capability is being
  re-platformed onto **pyabc** (SMC-ABC), **BayesFlow** (amortized
  NPE / FMPE / CMPE), and **sbi** (NLE / NRE) in subsequent releases.
  Removed surface:
  - The **`[sbi]` extra** (`pip install probpipe[sbi]`) and its
    `sbijax>=0.3.6` dependency.
  - The public Functions **`sbi_learn_conditional`** and
    **`sbi_learn_likelihood`** (exported from both `probpipe` and
    `probpipe.inference`), the **`DirectSamplerSBIModel`** they
    returned (exported from `probpipe.inference`), their `method=`
    selectors (`npe` / `fmpe` / `cmpe` for the direct sampler,
    `nle` / `nre` for the emulated-likelihood path), and the
    `network_factory=` hook. `from probpipe import
    sbi_learn_conditional` now raises `ImportError` rather than
    returning an install-prompt stub.
  - The **`sbijax_smcabc`** inference method (`SbiSMCABCMethod`,
    priority 5) and its registration; `condition_on(generative_model,
    data, method="sbijax_smcabc", ...)` no longer resolves.
  - The internal `probpipe/inference/_sbijax.py` module, the `sbi`
    pytest marker, the `tests/inference/test_sbijax.py` suite, and the
    CI `--no-deps sbijax` install shims. The contract invariants those
    tests covered — posterior recovery, amortization, and SMC-ABC
    dispatch — are re-homed per backend as the replacements land,
    rather than in this removal.

  The jax / jaxlib `<0.9` and arviz `<1.0` version caps that `sbijax`
  forced are *retained* here and lifted in their own isolated PRs (the
  jax-0.10 floor bump and the arviz-1.x ceiling lift); this PR changes
  no runtime version pins. The `docs/tutorials/flexible_inference.ipynb`
  tutorial's SBI sections are flagged out of date until a replacement
  backend ships — its `condition_on` dispatch and NUTS material remain
  accurate.

### Added

- **BlackJAX-backed gradient-free MCMC.** Two new inference methods
  bundled with the BlackJAX MCMC migration:
  - **`blackjax_rwmh`** (priority 55) replaces the hand-rolled
    Python-loop RWMH. Two execution paths share the same BlackJAX
    kernel: a fast path (`jax.lax.scan` + `jax.vmap` across chains)
    when the target log-density is JAX-traceable, and an eager
    Python-loop fallback when it isn't (BridgeStan / scipy /
    external-simulator likelihoods — the case the hand-rolled loop
    existed to support). The default warmup is a Stan-style window
    adaptation: ``n_windows`` (default 4) geometrically-growing
    windows, each sampling with the current proposal Cholesky and
    accumulating Welford statistics on positions, refreshing the
    proposal at window boundaries. Production sigma is
    ``chol(Sigma_hat) * 2.38 / sqrt(d)`` per Roberts-Gelman-Gilks.
    Short warmups (``< 50`` steps) collapse to a single phase
    automatically. ``adapt=False`` falls back to the legacy
    ``step_size * I`` for parity with the prior behavior.
  - **`blackjax_elliptical_slice`** (priority 75, tier 71-80
    self-tuning) is new — restricted to `SimpleModel` targets with a
    Gaussian prior and a JAX-traceable likelihood. Recognises
    `Normal`, `MultivariateNormal`, `JointGaussian` (named multi-field
    Gaussian with cross-covariance), and `ProductDistribution`
    compositions via the new `_gaussian_prior_params` helper.
- New Function `probpipe.elliptical_slice(model, data, ...)`.

### Changed

- **`blackjax_hmc` randomizes its trajectory length.** Production now
  draws the number of leapfrog steps from a low-discrepancy Halton
  sequence (`blackjax.dynamic_hmc`) with mean `num_integration_steps`
  (default 10, unchanged), instead of a fixed count. A fixed trajectory
  length can resonate on near-Gaussian targets — the proposal returns
  near its start, giving high acceptance and zero divergences yet poor
  mixing and up to ~30% posterior-variance under-estimation. Jittering
  `L` around the same mean breaks the resonance (Neal 2011, sec. 4.2).
  Window adaptation tunes step size + mass matrix against the *same*
  randomized-`L` kernel, so dual-averaging's acceptance target is
  calibrated to the kernel that actually runs; `num_integration_steps`
  is now the *mean*. The drawn count is floored at 1 leapfrog step (the
  Halton range includes 0, a no-op trajectory). NUTS is unaffected.
- **Multi-chain BlackJAX MCMC dispatch picks `jax.pmap` when devices
  permit.** When `jax.local_device_count() >= num_chains` the per-chain
  runner is mapped with `pmap` (each chain on its own device,
  bit-identical to a single-chain sequential run at the same seed);
  otherwise the prior `vmap` path is used. Single-chain calls
  (``num_chains == 1``) short-circuit both, applying the runner
  directly. Default single-CPU-device behaviour is unchanged for
  ``num_chains == 1``; users with
  ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` get
  per-device parallelism and bit-identical-to-sequential draws
  (notably for NUTS) without code changes. The three BlackJAX
  modules (`_blackjax_mcmc.py`, `_blackjax_rwmh.py`, `_blackjax_ess.py`)
  route their multi-chain dispatch through the new
  `parallel_chain_map` helper in `_inference_utils.py`.

### Changed (breaking)

- **`Function` controls now live outside user call kwargs.**
  `@function(...)` configures definition-time controls, and
  `workflow.with_options(...)(...)` is the call-time override API for
  `seed`, `n_broadcast_samples`, and `include_inputs`. Wrapped
  functions may now declare and receive those names as ordinary
  parameters. Passing those names as call kwargs no longer configures
  ProbPipe controls; use `workflow.with_options(...)(...)` instead.
- **`Function.workflow_kind` and `Module.workflow_kind` now require
  `WorkflowKind` enum members.** String aliases such as `"task"` / `"flow"`
  and `None` are no longer accepted and now raise `TypeError`; use
  `WorkflowKind.TASK`, `WorkflowKind.FLOW`, or `WorkflowKind.OFF` explicitly.
  The old `parallel=` / `vectorize=` keyword guard on `Function` was
  also removed, so those names are no longer specially reserved by the
  constructor.
- **`tfp_rwmh` removed.** The hand-rolled Python-loop RWMH that sat
  behind ``method="tfp_rwmh"`` is gone; ``blackjax_rwmh`` is the only
  RWMH backend. Callers must rename ``method="tfp_rwmh"`` →
  ``method="blackjax_rwmh"``.
- **Sample-count / observation-count terminology unified
  across the codebase.** Several adjacent concepts had drifted into
  different naming styles (`.n`, `num_draws`, `n_samples`, `n_iter`,
  `n_simulations`, `n_replications`, `num_steps`). Audited and
  consolidated under three canonical names per concept:

  *Finite-sample distribution size.* `.n` is gone. Use
  **`num_atoms`** for any empirical-measure size (one atom = one
  stored realisation): `EmpiricalDistribution.num_atoms`,
  `RecordEmpiricalDistribution.num_atoms`,
  `JointEmpirical.num_atoms`, `BootstrapDistribution.num_atoms`,
  `KDEDistribution.num_atoms`, `BroadcastDistribution` family +
  marginals — all expose `num_atoms`. `ApproximateDistribution`
  inherits `num_atoms` (total chain×draw count) and additionally
  exposes `num_draws` (draws *per chain*).

  *Bootstrap replicate size.* Use **`replicate_size`** for the number
  of items in each bootstrap replicate:
  `BootstrapReplicateDistribution.replicate_size`,
  `RecordBootstrapReplicateDistribution.replicate_size`. The
  constructor kwarg changes from ``n=`` to ``replicate_size=``; the
  related ``source_n`` property becomes ``source_size``. Callers that
  previously wrote ``BootstrapReplicateDistribution(data, n=N)`` will
  now get a ``TypeError`` and must rename to ``replicate_size=N``.
  (`replicate_size`, not `num_observations`: the resampled items come
  from an arbitrary source — parameter samples, function values, etc. —
  so "observations" would overclaim.)

  *Generative-likelihood observation count.*
  ``generate_data(params, n_samples, ...)`` is now
  ``generate_data(params, num_observations, ...)`` across the
  `GenerativeLikelihood` protocol, `GLMLikelihood`,
  `SimpleGenerativeModel`, and `predictive_check` (the latter's
  `n_replications` kwarg also becomes `num_replications`).

- **Inference-method count kwargs unified under `num_*`.** Several
  inference methods exposed `n_*`-style kwargs out of sync with the
  rest of the registry (which uniformly used `num_results` /
  `num_warmup` / `num_chains`). Renamed:
  - `blackjax_sgld` / `blackjax_sghmc`: `num_steps=` → `num_results=`
    (SGMCMC produces one chain draw per step; the kwarg matches
    every other MCMC backend now).
  - `sbi_learn_conditional` / `sbi_learn_likelihood`: `n_iter=` →
    `num_iterations=`, `n_simulations=` → `num_simulations=`.
  - `sbi_learn_conditional` posterior-sampling default
    `n_samples=` → `num_results=`; `DirectSamplerSBIModel.__init__`
    and `condition_on(direct_sampler_model, ...,
    n_samples=...)` likewise.

  Internal `sbijax.simulate_data(..., n_simulations=...)` /
  `sbijax.fit(..., n_iter=...)` / `sbijax.sample_posterior(...,
  n_samples=...)` calls keep their native sbijax kwarg names —
  only the probpipe-facing surface changes.

  Bug fix bundled with the rename: `tests/test_sbijax.py` was
  calling `condition_on(nle_model, obs, method="tfp_nuts",
  n_samples=500, n_warmup=500, n_chains=2, ...)` — the MCMC backend
  silently ignored those kwargs (it expects `num_results=` /
  `num_warmup=` / `num_chains=`) and the test passed by accident.
  Fixed.

- **`condition_on` MCMC default switched from TFP to BlackJAX NUTS,
  plus inference-method priority re-anchoring.** Several entangled
  changes consolidated into a single migration:

  *Auto-dispatch winner switches to BlackJAX NUTS.* `blackjax_nuts`
  (priority 85, tier 81–90) wins auto-dispatch for any
  `SupportsLogProb` + JAX-traceable target — the canonical ProbPipe
  model class. `tfp_nuts` / `tfp_hmc` are demoted to the opt-in-only
  sentinel (`priority=0`); they stay registered and reachable via
  `method="tfp_nuts"` / `method="tfp_hmc"` for bit-pattern regression
  checks or side-by-side comparisons.

  *Structurally-unreachable methods demoted to `priority=0`.* Methods
  whose `check()` is identical to a higher-priority sibling can never
  win auto-dispatch — they're opt-in in effect. Made that explicit:
  `blackjax_hmc` (same `check()` as `blackjax_nuts`) and
  `blackjax_sghmc` (same `check()` as `blackjax_sgld`, which is also
  the simpler default — fewer tuning dials) are now opt-in only.

  *VI demoted to opt-in.* `pymc_advi` (was priority 25) is now
  `priority=0`. VI is a deliberate bias-for-speed tradeoff that users
  should pick explicitly via `method="pymc_advi"`; silently dispatching
  into it when (e.g.) `pymc_nuts` happens to fail would surface VI in
  MCMC's place.

  *NUTS-tier numbers retuned.* `nutpie_nuts` 85 → 88 (top of the
  optimised-backend tier — Rust gradients are the fastest of every
  registered NUTS backend); `pymc_nuts` 81 → 82 (ties with
  `cmdstan_nuts` at 82; the two apply to disjoint model classes so
  the tie is documentary).

  `tfp_rwmh` (gradient-free RWMH) is unchanged at priority 55 — the
  gradient-free-MCMC migration to BlackJAX is queued separately
  (`~/.claude/plans/bie-rwmh-blackjax-migration.md`).

  Migration: an existing `condition_on(model, data)` call that
  previously ran TFP NUTS now runs BlackJAX NUTS. The numerical
  posterior is asymptotically identical but the per-seed bit pattern
  differs. Pin `method="tfp_nuts"` for bit-pattern regression. The
  closed-form correctness gate (mean within ~3 σ_MC, variance within
  10% on a known 2-D Gaussian target) is tested under
  `tests/test_blackjax_mcmc.py`. Existing `condition_on(...,
  method="pymc_advi")` / `method="blackjax_hmc"` /
  `method="blackjax_sghmc"` calls continue to work — only the
  auto-dispatch path changes.

- **Distribution & Record hierarchy cleanup (#200).** Implements the
  integrated cleanup plan as six self-contained commits. The public-
  facing changes are:
  - **`Distribution.validation_results` is removed.**
    `predictive_check` now writes its per-invocation payload to
    `dist.auxiliary["predictive_check/check_N"]` (a wrapped
    `xarray.Dataset` under a numbered group). Future validation
    functions (LOO, WAIC, …) land under their own named groups in
    the same `DataTree`. Code that read `dist.validation_results`
    should read `dist.auxiliary["predictive_check"]` instead.
  - **`flatten_value` / `unflatten_value` are now `@staticmethod` with
    explicit kwargs.** Callers pass `event_shape=` /
    `template=` explicitly:
    `dist.flatten_value(value, event_shape=dist.event_shape)` and
    `dist.unflatten_value(flat, template=dist.record_template)`.
    The previous instance-method form (no kwargs) raises at runtime.
  - **`_default_support` classmethods are removed** from every
    concrete distribution (`Normal`, `Gamma`, `Poisson`, …; 24 in
    total). Support compatibility is now checked post-construction
    via `NumericRecordDistribution._check_support_compatible(source)`;
    downstream code that reached for the classmethod should use the
    instance `support` / `supports` properties.
  - **`SimpleModel.__init__` requires a `RecordDistribution` prior**
    (in addition to the pre-existing `SupportsLogProb` check). Priors
    that satisfy `SupportsLogProb` but aren't `RecordDistribution`
    raise `TypeError`. The type system can't express the intersection
    statically, so the runtime guard is the backstop.
  - **Default model names change from `None` to the class name.**
    `SimpleModel()`, `SimpleGenerativeModel()`, `PyMCModel()`,
    `StanModel()`, and `DirectSamplerSBIModel()` now default to
    `"SimpleModel"` / `"SimpleGenerativeModel"` / `"PyMCModel"` /
    `"StanModel"` / `"DirectSamplerSBIModel(<alg>)"` when no name is
    supplied. The metaclass invariant requires every `Distribution`
    instance to have a non-empty name.
  - **`NumericRecordDistribution.event_shape` is abstract** —
    raises `NotImplementedError` on the base. Single-leaf subclasses
    must override directly; multi-leaf subclasses (joints) set
    `_record_template` explicitly and never trigger the auto-build.
    Previously the default tried to derive from `event_shapes`,
    which looped back through `record_template`.
  - **`ProductDistribution` and `SequentialJointDistribution`
    conditionally mix in `NumericRecordDistribution`** based on
    their resolved leaves. Both stay rooted at the general
    `RecordDistribution` (their content is well-defined for
    non-numeric leaves too — sampling produces a `Record` keyed by
    component name, conditioning and named-component access always
    work). When *every* leaf is itself a `NumericRecordDistribution`,
    the dynamic class factory adds `NumericRecordDistribution` to the
    bases, so the joint also exposes the numeric API (`event_size`,
    `flatten_value` / `unflatten_value`, `as_flat_distribution`,
    `dtypes`, `supports`). For mixed or non-numeric leaves those
    methods are simply absent on the instance. Leaf type constraint
    relaxed from `NumericRecordDistribution` to `Distribution`.
    **Caller-visible consequence:**
    `isinstance(joint, NumericRecordDistribution)` is no longer
    guaranteed for `ProductDistribution` / `SequentialJointDistribution`
    instances — it returns `True` only when every resolved leaf is
    itself an NRD (the common case). Downstream code that branched on
    `isinstance(..., NumericRecordDistribution)` for these joints
    should verify the new dispatch matches its expectations, or
    switch to checking for the specific capability (e.g.,
    `hasattr(joint, "event_size")`).
  - **`NumericJointEmpirical` adds `NumericRecordDistribution` as a
    mixin** (previously implicit via `JointEmpirical` only). The
    sibling `JointEmpirical` stays on `RecordDistribution` and now
    builds a structural template from the stored samples (object-
    dtype leaves use `None` specs) to satisfy the metaclass
    invariant.

### Added

- **`RecordTemplate.event_shapes` and `RecordTemplate.field_event_shape(name)`**
  expose per-top-level-field event shapes (nested sub-templates and
  opaque leaves collapse to `()`). The previous helper
  `RecordDistribution._field_event_shape` is removed in favor of these
  template methods.

- **Metaclass-enforced invariants.** Every `Distribution` instance
  has a non-empty `name`; every `RecordDistribution` instance has a
  non-`None` `record_template`. The checks fire post-`__init__` via
  the `_DistributionMeta` / `_RecordDistributionMeta` metaclasses
  (derived from `typing._ProtocolMeta` to compose with
  `@runtime_checkable` protocols). Subclasses that forget either
  invariant raise `TypeError` at construction with a clear pointer.

### Changed

- **`GLMLikelihood` fits an intercept by default** (``fit_intercept=True``).
  The covariate matrix ``X`` carries only the covariates — no leading
  column of 1s. ``params`` flattens to ``(intercept, *slopes)`` and the
  likelihood computes ``eta = intercept + X @ slopes``. Pass
  ``fit_intercept=False`` for the classical "model matrix" convention
  where the user prepends the constant column to ``X`` themselves.
  Avoids the axis-position ambiguity of stacking the intercept slot
  into ``X``; matches the pattern in sklearn / statsmodels GLM APIs.

- **Dispatch-registry hierarchy split: `BaseDispatchRegistry`,
  `UnaryDispatchRegistry`, `BinaryDispatchRegistry`.** The
  arity-independent logic (registration, priority management, opt-in
  filtering, `check`/`execute` loop) lives on the new
  `BaseDispatchRegistry` abstract base. `UnaryDispatchRegistry` is the
  single-argument concrete subclass that replaces the previous
  `MethodRegistry` and now backs the inference method registry.
  `BinaryDispatchRegistry` adds two-argument dispatch on the joint type
  of the first two positional args. `BaseDispatchMethod` /
  `UnaryDispatchMethod` / `BinaryDispatchMethod` mirror the registry
  split on the method side. The previous `Method` / `MethodRegistry`
  aliases are removed — inference-method subclasses should subclass
  `UnaryDispatchMethod` (or the `InferenceMethod` re-export from
  `probpipe.inference`).

- **Inference-method registry priorities re-anchored with a semantic
  convention (issue #189).**
  - `priority > 50` marks *exact* methods; `0 < priority <= 50` marks
    *inexact* methods; `priority == 0` is the opt-in-only sentinel
    (selectable by name via `method="..."` but skipped during
    auto-dispatch). This is the new default value inherited from
    `Method.priority`.
  - Built-in priorities re-anchored: `nutpie_nuts` 80→85,
    `cmdstan_nuts` 70→82, `pymc_nuts` 60→81, `tfp_nuts` 100→75,
    `tfp_hmc` 90→65, `tfp_rwmh` 50→55, `blackjax_sgld` 30→45,
    `blackjax_sghmc` 25→42, `pymc_advi` 35→25, `sbijax_smcabc` 40→5.
    The relative ordering among exact methods is corrected so that
    optimised backends (`nutpie_nuts`, `cmdstan_nuts`, `pymc_nuts`)
    sit above the general-purpose `tfp_nuts`.
  - `MethodRegistry._find_methods()` now skips priority-0 methods
    during auto-dispatch. `MethodRegistry.set_priorities()` emits a
    `UserWarning` when an override crosses into or out of `0`;
    crossings of the documentary `50` break do not warn.
  - The `OPT_IN_ONLY_PRIORITY` sentinel is exported from
    `probpipe.core._registry` for use in `Method` subclasses that
    want to opt out of auto-dispatch by name.
  - The contributor-facing selection criteria and tier ranges for
    setting a new method's priority are documented under
    [Extending ProbPipe → Setting priority for a new method](docs/api/extending.md#setting-priority-for-a-new-method).
  - Migration: a `Method` subclass that previously relied on
    inheriting `priority = 0` from the base class while expecting
    auto-dispatch must now set a positive priority explicitly.
    `set_priorities` calls that stay within positive priorities are
    unaffected; calls that move a method to or from `0` emit a
    warning explaining the auto-dispatch participation change.

- **PyMC-backed posteriors now carry RV-keyed Record structure.**
  ``PyMCModel`` exposes a ``record_template`` property that pairs each
  free RV with its event shape (scalar RVs → ``()``; shape-`k` RVs →
  ``(k,)``). The PyMC NUTS, PyMC ADVI, and nutpie inference paths all
  thread this through to ``make_posterior``, so ``mean(post)`` returns
  a ``NumericRecord`` keyed by RV name and ``draws()`` returns a
  ``NumericRecordBatch``. Previously, PyMC posteriors had no field
  structure and ``draws()`` returned a flat ``(n_draws, n_params)``
  array. Models declared with multiple scalar RVs (e.g. separate
  ``intercept`` and ``slope`` ``pm.Normal`` calls) now produce a
  field-per-RV posterior matching the ``ProductDistribution``-prior
  workflow.

  Free RVs whose ``type.shape`` contains a ``None`` dimension are
  rejected with ``ValueError`` — silently dropping unknown dims would
  produce an under-shaped template.

- **`GLMLikelihood` no longer accepts stacked ``(X, y)`` arrays.** Both
  ``log_likelihood`` and ``per_datum_log_likelihood`` now require either
  ``data = Record(X=..., y=...)`` (canonical) or, for ``log_likelihood``
  only, a bare response array when ``X`` was supplied at construction
  time. Passing a single matrix whose last column was interpreted as
  the response is intentionally rejected — ProbPipe uses named Records
  to avoid axis-position ambiguity. Existing call sites that used a
  ``Record`` are unaffected.

- **`SimpleModel.prior` / `SimpleGenerativeModel.prior` type
  annotations tightened** from ``Distribution[P]`` /
  ``SupportsSampling[P]`` to the specific capability protocol
  (``SupportsLogProb[P]`` / ``SupportsSampling[P]``). Static type
  checkers now catch wrong-type priors at the call site; the
  construction-time ``isinstance`` check stays as a backstop. The two
  model wrappers are now parallel in both the input typing and the
  ``.prior`` property return type.

### Added

- **BlackJAX-backed SGMCMC methods** registered with
  ``inference_method_registry``:
  - ``blackjax_sgld`` — Stochastic Gradient Langevin Dynamics. Priority 45.
  - ``blackjax_sghmc`` — Stochastic Gradient Hamiltonian Monte Carlo. Priority 42.

  Both consume a `SimpleModel` whose `likelihood` satisfies
  `ConditionallyIndependentLikelihood`, plus a required `batch_size=`
  kwarg. Internally they wrap the model+data in a
  `MinibatchedDistribution` and feed BlackJAX's gradient estimator
  via the per-step random-measure draw — the kernel stays oblivious
  to the minibatching convention.

  ```python
  posterior = condition_on(
      model, data,
      method="blackjax_sgld",
      batch_size=64, num_results=2000, num_warmup=500, step_size=1e-3,
  )
  ```

  Priorities sit in the refinement-based MC tier (1–50), below every
  exact full-batch gradient method (`tfp_nuts=75`, `tfp_hmc=65`,
  `tfp_rwmh=55`). SGMCMC's `check()` also requires `batch_size=`, so
  it does not fire on a routine `condition_on(model, observed)` call —
  the user opts in by passing `batch_size=` (and typically the
  matching `method=`).

- **`MinibatchedDistribution`** (`probpipe.MinibatchedDistribution`)
  — a `RandomMeasure[Record]` over fixed-minibatch stochastic
  surrogates of the full-data unnormalized log-posterior. A draw is a
  `Distribution[Record]` with unnormalized log-density
  `log p(theta) + (N/b) * sum_{d in B} log p(d|theta)`, an unbiased
  stochastic surrogate (in expectation over the minibatch `B`) of the
  full-data target; the `N/b` rescaling makes the gradient an unbiased
  estimator.

  The constructor takes a prior and a conditionally-independent
  likelihood directly, mirroring `SimpleModel(prior, likelihood)` on
  the first two args. Consume the measure via
  `SupportsRandomUnnormalizedLogProb` to get the per-minibatch
  log-density callable that SGMCMC kernels feed `jax.grad`:

  ```python
  from probpipe import MinibatchedDistribution, Record, random_unnormalized_log_prob

  m = MinibatchedDistribution(prior, likelihood, Record(X=X, y=y), batch_size=64)

  rf = random_unnormalized_log_prob(m)
  target = rf._sample(k)                     # callable: theta -> log~D_B(theta)
  grad = jax.grad(target)(theta)             # unbiased gradient estimate
  ```

  This is the path stochastic-gradient MCMC kernels use under the
  hood; the BlackJAX SGLD / SGHMC dispatch builds a `MinibatchedDistribution`
  internally and threads `target` into the BlackJAX gradient
  estimator. Tempered SMC (future work) is expected to consume the
  same surface.

- **`ConditionallyIndependentLikelihood`** (`probpipe.ConditionallyIndependentLikelihood`)
  — a `Likelihood` subclass / Protocol whose observations factorise as
  `log p(D | theta) = sum_i log p(d_i | theta)`. Adds a
  `per_datum_log_likelihood(params, datum)` method on top of the base
  `Likelihood`'s `log_likelihood(params, data)`. Required by
  stochastic-gradient inference (the upcoming `MinibatchedDistribution`)
  and independently useful for held-out predictive log-likelihoods,
  leave-one-out cross-validation, and PSIS-LOO. The existing concrete
  likelihoods (`GLMLikelihood`, `_NLELikelihood`, `_NRELikelihood`) all
  satisfy the Protocol — `GLMLikelihood` via a direct family
  `log_prob` evaluation that skips the per-batch tile, the two
  sbijax-backed classes via a length-1-batch fallback.

  A standalone helper `_default_per_datum_log_likelihood(likelihood,
  params, datum)` provides the length-1-batch implementation for
  subclasses that want a default rather than an efficient override.

- **`SimpleModel.prior` / `SimpleModel.likelihood`** and
  **`SimpleGenerativeModel.prior` / `SimpleGenerativeModel.likelihood`**
  — public read-only properties that expose the underlying components
  without poking at private state. The two model wrappers stay
  symmetric: `SimpleModel.likelihood` is typed `Likelihood`,
  `SimpleGenerativeModel.likelihood` is typed `GenerativeLikelihood`.

- **`FlatNumericRecordDistribution`** (`probpipe.FlatNumericRecordDistribution`)
  — a `NumericRecordDistribution` subclass that enforces the flat
  contract (single field, `event_shape == (N,)`). Algorithms that
  operate on a flat parameter vector (MCMC kernels, optimisers,
  Hessian / curvature builders, variational families, Pathfinder /
  Laplace surrogates) can require this type rather than runtime
  shape probes. Carries the `flat_size: int` shortcut (=
  `event_shape[0]`) and the `as_record_distribution(template=...)`
  method.

  The natively-multivariate parametrics
  (`MultivariateNormal`, `Dirichlet`, `Multinomial`, `VonMisesFisher`)
  now inherit from `FlatNumericRecordDistribution` in addition to
  `TFPDistribution`. `FlattenedDistributionView` also implements the
  contract by construction. Scalar parametrics (`Normal`, `Beta`,
  `Bernoulli`, …) have `event_shape == ()` and do not satisfy the
  contract directly; call `.as_flat_distribution()` to obtain a
  `FlattenedDistributionView` with `event_shape == (1,)`.

- **`FlatNumericRecordDistribution.as_record_distribution(template=...)`**
  — inverse of `as_flat_distribution()`. Lifts a flat distribution to
  a Record-keyed view under a user-supplied `NumericRecordTemplate`.
  Sampling, log-prob, and moments delegate to the source and reshape
  via the template; capability protocols (`SupportsX`) match the
  source via dynamic isinstance dispatch. The view is a thin wrapper —
  no value copying.

  ```python
  from probpipe import MultivariateNormal, NumericRecordTemplate

  mvn = MultivariateNormal(                     # already a FlatNRD
      loc=jnp.array([1.0, 2.0, 3.0, 4.0]),
      cov=jnp.diag(jnp.array([0.5, 1.0, 1.5, 2.0])),
      name="theta",
  )
  template = NumericRecordTemplate(intercept=(), slope=(3,))
  posterior = mvn.as_record_distribution(template=template)
  draw = sample(posterior, key=k)         # NumericRecord(intercept, slope)
  mean(posterior)["slope"]                # vector mean of the slope block
  ```

### Changed

- **`FlattenedView` renamed to `FlattenedDistributionView`** and now
  inherits from `FlatNumericRecordDistribution` (formerly
  `NumericRecordDistribution`). The view's flat contract was always
  satisfied structurally; the new base class makes it explicit and
  enables receiver-type-driven dispatch for `as_record_distribution`.
- **`_RecordLiftedView` renamed to `NumericRecordDistributionView`** and
  made public. Constructed via
  `FlatNumericRecordDistribution.as_record_distribution(template=...)`.

### Migration

- Code that imports `FlattenedView`: rename to `FlattenedDistributionView`.
- Code that imports `_RecordLiftedView`: rename to
  `NumericRecordDistributionView`.
- Code that calls `as_record_distribution` on a non-flat distribution
  (e.g., a scalar `Normal`): chain via `.as_flat_distribution()` first.
  Calling it directly on a non-flat `NumericRecordDistribution` now
  raises `TypeError` instead of `ValueError` and points at the
  `as_flat_distribution()` chain.

- **`SupportsArrayBackend` capability protocol** (`probpipe.SupportsArrayBackend`)
  declares that a `Distribution` subclass can produce a fused storage
  backend for `DistributionArray`. Implemented by every TFP-backed
  concrete class (`Normal`, `Beta`, `Gamma`, `MultivariateNormal`,
  `Dirichlet`, …) via inheritance from `TFPDistribution`. Distribution
  classes that don't implement the protocol still work in a
  `DistributionArray` via the literal-array fallback path.

- **`DistributionArray.from_batched_params(dist_cls, *, name, batch_shape=None, **batched_params)`**
  factory + ergonomic per-class alias **`Distribution.from_batched_params(*, name, batch_shape=None, **batched_params)`**.
  Constructs a `DistributionArray` of homogeneous components,
  dispatching on `SupportsArrayBackend`: TFP-backed classes get a
  fused TFP-batched backend; other classes fall back to one
  `dist_cls` instance per cell. Per-cell names auto-suffix
  `f"{name}_{flat_index}"`. `batch_shape` is inferred from broadcast
  of array-valued params; classes with heterogeneous per-param event
  ranks (`MultivariateNormal`, `Dirichlet`) require explicit
  `batch_shape=...`.

  ```python
  # Recommended ergonomic form
  da = Normal.from_batched_params(loc=jnp.zeros(5), scale=1.0, name="x")
  da.batch_shape       # (5,)
  da[2].name           # "x_2"

  # Equivalent universal form
  da = DistributionArray.from_batched_params(
      Normal, loc=jnp.zeros(5), scale=1.0, name="x",
  )
  ```

### Changed (breaking)

- **Prefect orchestration is now opt-in** (#182). The shipped global
  default for `prefect_config.workflow_kind` is `WorkflowKind.OFF`
  instead of the prior `WorkflowKind.DEFAULT` (which auto-promoted to
  `TASK` whenever Prefect was importable). The old behaviour silently
  enabled Prefect for any environment with Prefect on `sys.path` —
  including environments where Prefect was pulled in as a transitive
  dependency — and produced a confusing `httpx.ConnectError` when no
  Prefect server was running. The new default produces no surprise
  network traffic; users who want orchestration opt in once per
  session or deployment:

  ```python
  import probpipe
  probpipe.prefect_config.workflow_kind = probpipe.WorkflowKind.TASK
  ```

  Or via the new `PROBPIPE_WORKFLOW_KIND` environment variable
  (`off` / `task` / `flow` / `default`, case-insensitive), which is
  read once at import time. Per-workflow overrides remain available via
  `@function(workflow_kind=probpipe.WorkflowKind.TASK)`; string
  aliases are no longer accepted in this release (see the
  `workflow_kind` breaking entry above).
  Migration: production callers that relied on the implicit
  "Prefect importable → tasks enabled" path must add the one-line
  assignment or env var above.

- **`NumericRecordDistribution.dtypes` is canonical; subclasses must
  override.** The base accessor previously returned
  ``{name: default_float_dtype()}`` for every field of the
  ``record_template`` (a silent lie for every integer-valued TFP
  distribution — ``Bernoulli`` / ``Categorical`` reported
  ``float32``). The base now raises ``NotImplementedError`` so the
  truth direction is unambiguous; concrete subclasses declare
  ``dtypes`` directly via the new
  ``_spread_to_fields(value)`` helper:

  ```python
  >>> from probpipe import Bernoulli
  >>> Bernoulli(probs=0.5, name="x").dtype
  jnp.int32   # was float32 (the lie)
  >>> Categorical(probs=jnp.array([0.5, 0.5]), name="x").dtype
  jnp.int32   # was float32
  ```

  Migration for custom subclasses: implement
  ``dtypes`` returning ``{field: dtype}`` aligned with
  ``record_template.fields``. The single-leaf shortcut for
  uniform-dtype subclasses is
  ``return self._spread_to_fields(my_dtype)``. The convenience
  ``dtype`` accessor derives automatically.

  Related cleanups landing in the same PR:

  - ``supports`` is also canonical now (raises if not overridden);
    ``support`` is a convenience that derives via
    ``_single_field_name``. Existing single-field ``support``
    overrides on concrete TFP-backed classes continue to work.
  - ``record_template`` auto-build (single-field
    ``RecordTemplate(**{name: event_shape})``) moved from
    ``TFPDistribution`` to the base, so any concrete subclass
    with a ``name=`` and ``event_shape`` gets a template
    automatically.
  - ``treedef`` derives from ``record_template`` (leaf for
    single-leaf, ``NumericRecord`` skeleton for multi-leaf) and
    is cached on first read.
  - ``flat_event_shapes`` tree-walks ``event_shapes`` rather than
    hardcoding ``[event_shape]``.
  - ``_check_support_compatible`` reads canonical ``supports``
    (per-field check on multi-leaf source, single-leaf message
    preserved).

- **`Distribution.batch_shape` removed.** The property is gone
  from `Distribution` and every subclass; reads now raise
  `AttributeError`. Collections of distributions live in
  `DistributionArray`, which retains its own `batch_shape` (the
  outer array shape).

  ```python
  >>> from probpipe import Normal
  >>> hasattr(Normal(loc=0.0, scale=1.0, name="x"), "batch_shape")
  False
  ```

  Migration: drop the read — once batched parameters were rejected,
  it was always `()`. `GaussianRandomFunction.predict` (and every
  `ArrayRandomFunction` subclass) now returns a `DistributionArray`
  rather than a single batched `Normal` / `MultivariateNormal`;
  per-cell `event_shape` is unchanged. Fully-joint predictions with
  no extra batch axes return a 0-d `DistributionArray`; ops
  (`sample`, `mean`, `log_prob`, …) auto-unwrap a 0-d DA to its
  single cell, so call sites stay unchanged.

- **`DistributionArray` container surface aligned with numpy / jax**
  (#178). `iter(da)` now walks the leading axis: a 1-D array yields
  its scalar cells (unchanged); a multi-d array yields
  ``DistributionArray`` slices of shape ``batch_shape[1:]``,
  mirroring ``iter(np.zeros((2, 3)))``. Use ``da.components`` for
  flat row-major access over every cell (the pre-#178 default).
  Adds ``DistributionArray.size`` returning ``prod(batch_shape)``,
  matching ``np.ndarray.size`` / ``jax.Array.size``.

- **`RecordDistribution.n` and `DistributionArray.n` removed.**
  STYLE_GUIDE §1.9 reserves `.n` for finite-sample distribution
  classes that hold a finite collection of samples / observations
  / components (`EmpiricalDistribution`, `BootstrapDistribution`,
  `BroadcastDistribution`, …). The two cases removed here did
  not fit the contract: parametric `Normal(0, 1)` does not "hold"
  any items, and `DistributionArray` is a positional collection of
  independent cells, not a finite-sample distribution. Migration:
  for `DistributionArray`, use `len(da)` (leading-axis size) or
  `prod(da.batch_shape)` (total cell count) — `__repr__` now shows
  `batch_shape=...`. For parametric distributions, drop the
  call — it always returned `1`. Finite-sample distributions
  retain `.n` (see STYLE_GUIDE §1.9 for the full table).

- **TFP-backed distribution constructors reject batched parameters.**
  `Normal(loc=jnp.zeros(5), scale=1.0, name="x")` (and the same
  pattern for every other TFP-backed class — `Beta`, `Gamma`,
  `MultivariateNormal`, `Pareto`, `TruncatedNormal`, `Binomial`, …)
  now raises `ValueError` whenever the parameters imply a non-empty
  TFP `batch_shape`. The framework hierarchy rule "one random
  variable per `Distribution`" (CONTRIBUTING.md) is enforced at
  construction time.

  ```text
  ValueError: Normal parameters imply batch_shape=(5,); wrap multiple
  distributions in a DistributionArray instead. See
  DistributionArray.from_batched_params(Normal, ...) (or the alias
  Normal.from_batched_params(...)) for the factory.
  ```

  Migration: route through the
  `DistributionArray.from_batched_params` factory (or its per-class
  alias) added in the previous release. The factory is
  performance-equivalent to the legacy form because the fused
  `_TFPArrayBackend` wraps the same TFP-batched distribution under
  the hood.

  ```python
  # Before (rejected)
  n = Normal(loc=jnp.zeros(5), scale=1.0, name="x")

  # After (recommended ergonomic form)
  da = Normal.from_batched_params(loc=jnp.zeros(5), scale=1.0, name="x")

  # After (universal entry point)
  da = DistributionArray.from_batched_params(
      Normal, loc=jnp.zeros(5), scale=1.0, name="x",
  )
  ```

  Removed associated tests that exercised the legacy form's
  per-element support checks: ``test_uniform_support_array_bounds``,
  ``test_half_cauchy_support_array_bounds``,
  ``test_pareto_support_array_bounds``,
  ``test_truncated_normal_support_array_bounds``,
  ``test_binomial_support_array_total_count``,
  ``test_repr_with_batch_shape``. Per-element support checks belong
  on `Constraint` directly; batched constructions migrate to
  `DistributionArray.from_batched_params`.

  Internal infrastructure that legitimately needs the batched form
  (the `_TFPArrayBackend` fused-storage backend, the
  `ProbPipeConverter` dispatch, sequential-joint sampling /
  log_prob, `GaussianRandomFunction.predict`) opts into a private
  bypass; user code is unaffected by the bypass and always sees the
  rejection.

- **Empirical / Bootstrap / Marginal class consolidation.** The
  generic-vs-numeric pair is collapsed into a generic ``[T]`` base
  plus a single Record-based specialisation:

  | Removed | Replacement |
  |---|---|
  | ``NumericEmpiricalDistribution`` | ``RecordEmpiricalDistribution`` |
  | ``ArrayBootstrapReplicateDistribution`` | ``RecordBootstrapReplicateDistribution`` |
  | ``_ArrayMarginal`` (private) | ``_RecordMarginal`` (private) |
  | ``_RecordEmpiricalDistribution`` (private) | ``RecordEmpiricalDistribution`` |
  | ``_RecordBootstrapReplicateDistribution`` (private) | ``RecordBootstrapReplicateDistribution`` |
  | ``_RecordArrayMarginal`` (private) | ``_RecordMarginal`` (private) |

  Migration: a numeric array auto-wraps as a single-field ``Record``
  keyed by the (now mandatory) ``name=`` kwarg.

  ```python
  # Before
  emp = EmpiricalDistribution(arr)                    # Worked
  emp = NumericEmpiricalDistribution(arr)             # Worked
  emp = ArrayBootstrapReplicateDistribution(arr)      # Worked

  # After
  emp = EmpiricalDistribution(arr, name="theta")       # ✓
  emp = EmpiricalDistribution(arr)                    # ValueError: name= required
  ```

  The ``name=`` becomes the field name of the auto-wrapped
  ``Record``; downstream code that does ``emp.samples["theta"]`` /
  ``emp["theta"]`` then has a meaningful key. If you want to keep the
  old call-site shape, wrap explicitly:
  ``EmpiricalDistribution(Record(theta=arr))``.
- **`BootstrapReplicateDistribution[T]` accepts a `SupportsSampling`
  source.** Each replicate is ``n`` i.i.d. draws from
  ``source._sample``. **``n`` is mandatory** when ``source`` is a
  ``SupportsSampling`` distribution (no canonical observation count);
  it remains optional for ``Record`` / numeric-array / ``Empirical``
  sources, where it defaults to the source's row count.
  ``BootstrapReplicateDistribution(Normal(0, 1, name="x"), n=50)``.
- **`NumericJointEmpirical` no longer claims `SupportsLogProb`.** The
  Gaussian-approximation log-density is gone — empirical distributions
  do not advertise a density. Migration:
  ``from_distribution(emp, KDEDistribution, ...)`` for a non-parametric
  density, or fit a parametric distribution and call ``log_prob`` on
  that.
- **Distributions are non-iterable.** Codified in STYLE_GUIDE §1.11
  with a regression test
  (``tests/test_iteration_protocol.py``). Finite-sample subclasses
  (see §1.9) expose stored samples via ``.samples`` / ``.draws()``
  and ``.n``; parametric distributions do not have ``.n``.

- **`Record` field ordering is now insertion-order**, not alphabetical.
  ``Record(z=1, a=2)`` now iterates ``("z", "a")``. Same change applies
  to ``RecordTemplate``, ``RecordBatch``, and every Record-based
  distribution that derives ``fields`` from the underlying store.
  Previous alphabetical ordering was an accident of
  ``OrderedDict(sorted(...))``.
- **`/` is reserved in `Record` and `RecordTemplate` field names.**
  Construction-time ``ValueError``. Used as the slash-delimited path
  separator in ``record["params/intercept"]`` style access.
- **`Record.to_datatree()` / `Record.from_datatree(...)` removed.**
  Use ``record.to_numeric().to_native()`` for a metadata-preserving
  round-trip via the aux registry, or ``xr.DataTree`` directly if you
  specifically want a DataTree.
- **`NumericRecord(...)` (and `Record.to_numeric()`) raise `TypeError`
  on non-coercible leaves** (strings, opaque objects). Today's
  implicit failure inside ``NumericRecord(...)`` becomes an explicit,
  well-messaged error at construction time.
- **`RecordTemplate.leaf_shapes` keys for nested templates use `/`**
  instead of ``.`` (e.g. ``"physics/force"`` instead of
  ``"physics.force"``) for consistency with ``Record["a/b"]`` path
  access.

### Added

- **Framework abstraction hierarchy** documented in CONTRIBUTING.md.
  Three rules: one random variable per ``Distribution``; two
  implementations per concept (generic + Record-based); iteration is
  a Record-family convention.

- **`RecordEmpiricalDistribution.flat_samples`** — flat ``(n, dim)``
  matrix view across all fields, where
  ``dim = sum(prod(event_shape_f) for f in fields)``. Field order is
  the dist's insertion order; multi-dim event shapes flatten
  row-major. Use ``.samples`` for the structured ``NumericRecord``
  view (per-field access via ``.samples[name]``) and ``.flat_samples``
  for stacked-matrix idioms — ``post.flat_samples.mean(axis=0)``,
  per-parameter posterior summaries, etc. Replaces hand-rolled
  ``np.column_stack([post.samples[f] for f in post.fields])``.

- **`Record.to_numeric()` / `NumericRecord.to_native()`** — explicit
  conversion to / from ProbPipe's native JAX-array form, with metadata
  round-trip via the aux registry. Backend metadata survives the structural
  edits (`without` / `merge` / `replace` / `with_path_names`) for leaves they
  leave unchanged, at any nesting depth, and a pickle round-trip (an
  aux-carrying record pickles through its native form, so `to_native` stays
  faithful across `pickle` / Ray transport); a value transform (`map`) or a
  JAX pytree round-trip drops it.
- **`probpipe.AuxHooks` / `register_aux(...)` / `aux_for(...)` /
  `aux_registry`** in :mod:`probpipe.core._array_backend` — a registry
  of ``(capture, restore)`` hooks for round-tripping backend-specific
  metadata across the ``Record`` ↔ ``NumericRecord`` boundary.
  Built-in registrations (gated on import) cover
  ``xarray.DataArray`` (dims / coords / attrs / name),
  ``pandas.Series`` (index / name / dtype), and ``pandas.DataFrame``
  (index / columns / dtypes).
- **`NumericRecord.aux`** property — captured backend metadata, keyed
  by field name. ``None`` when no field had a registered hook.
- **Slash-delimited path access** on nested ``Record``s:
  ``record["params/intercept"]`` is sugar for
  ``record["params", "intercept"]``. ``"a/b/c" in record`` works the
  same way.

### Changed

- **dtype handling** now follows JAX's rules. Distributions, weights, and
  empirical classes preserve user-supplied dtypes and honor
  ``jax.config.update("jax_enable_x64", True)`` end-to-end. Previously every
  TFP-backed constructor silently downcast its parameters to ``float32``,
  causing ``log_prob`` / ``sample`` / ``mean`` to raise ``TypeError`` under
  x64. Multi-parameter constructors now promote inputs to a common float
  dtype via ``jnp.result_type`` (integer inputs are promoted to JAX's
  default float, so ``Normal(loc=0, scale=1)`` still works). Internal
  helpers ``_default_float_dtype()`` and ``_promote_floats()`` live in
  ``probpipe/_dtype.py``. The float64-truncation warning filter previously
  in ``probpipe/__init__.py`` is removed.

### Added

- **Uniform `select_all()`** on ``Record`` / ``RecordBatch`` /
  ``RecordDistribution``. Splatting the result into a
  ``@function`` preserves correlation on the two batched variants
  and plain splats fields on scalar ``Record``.
- **Public `.parent` / `.field`** properties on ``_RecordDistributionView``,
  which say two views draw from one law.
- **Single-field `.shape` / `.ndim` shims** on ``RecordDistribution`` and
  ``_RecordDistributionView`` (mirror the existing shims on
  ``NumericRecord`` / ``NumericRecordBatch``). Multi-field distributions
  raise ``TypeError``.

### Changed (breaking)

- **`len(RecordBatch)`** now returns the **field count** (matching
  ``len(Record)``) instead of ``prod(batch_shape)``. For the flat batch
  size, use ``prod(ra.batch_shape)``.
- **`event_shapes`** now always returns ``dict[str, tuple[int, ...]]``.
  Untemplated (legacy) distributions return ``{}``; use the singular
  ``.event_shape`` for the whole-sample shape.
- **`component_names` → `fields`** on every Record-based distribution and
  model (``RecordDistribution``, ``ProductDistribution``, ``JointGaussian``,
  ``JointEmpirical``, ``SequentialJointDistribution``,
  ``BroadcastDistribution``, ``ProbabilisticModel``, ``SimpleModel``,
  ``SimpleGenerativeModel``, ``PyMCModel``, ``StanModel``). No backward
  alias.

## [0.1.0] - 2025-03-21

Initial release with TensorFlow Probability / JAX backend.

### Added

- **Distribution framework**: `Distribution` ABC with TFP shape semantics (`sample_shape + batch_shape + event_shape`), `TFPDistribution` mixin, `EmpiricalDistribution` (direct JAX).
- **23 distribution wrappers**: 14 continuous (Normal, Beta, Gamma, etc.), 5 discrete (Bernoulli, Binomial, Poisson, etc.), 4 multivariate (MultivariateNormal, Dirichlet, Multinomial, Wishart, VonMisesFisher).
- **Constraints**: `Constraint` base class with partial-order compatibility checking. Built-in singletons (`real`, `positive`, `unit_interval`, `simplex`, etc.) and factories (`interval()`, `greater_than()`).
- **Transformed distributions**: `TransformedDistribution` with TFP bijectors (Exp, Sigmoid, Softplus, Shift, Scale, Chain). Automatic support derivation.
- **Joint distributions**: `ProductDistribution` (independent), `SequentialJointDistribution` (autoregressive), `JointEmpirical` (weighted joint samples), `JointGaussian` (exact analytical conditioning). `DistributionView` for component access, `ConditionedComponent` for conditioning.
- **Workflows and broadcasting**: `WorkflowFunction` with automatic uncertainty propagation. Multi-backend broadcasting (`jax` vectorization, `loop`, `prefect` orchestration). Auto-detection of JAX traceability. Empirical enumeration with budget-aware cartesian product.
- **Bayesian inference**: `SimpleModel` with NUTS/HMC via TFP + auto-fallback to gradient-free RWMH. `Likelihood` base class. `IterativeForecaster` for sequential Bayesian updating. `ApproximateDistribution` with chain structure and diagnostics.
- **Provenance tracking**: Automatic lineage on all creation paths (transforms, broadcasting, conditioning, inference). `Provenance.to_dict()` / `from_dict()` serialization. `provenance_ancestors()` and `provenance_dag()` utilities.
- **Documentation**: MkDocs Material site with API reference, getting started guide, and 6 tutorial notebooks (distributions, transformations, joint distributions, broadcasting, autodiff, modular forecasting).
- **CI/CD**: GitHub Actions CI with pytest + coverage, Codecov integration, automated docs deployment to GitHub Pages.
