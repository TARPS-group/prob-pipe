# Dispatch registries and the registry catalog

This is the contributor-facing guide to ProbPipe's dispatch machinery:
the priority-based registries that pick an implementation from the types
you pass, and the catalog that indexes every registry in the process.
Read it before adding a new registry, a new dispatch method, or a new
operation that has to choose between several implementations.

It is the *narrative* companion to
[Extending ProbPipe](../api/extending.md), which carries the
mkdocstrings API reference and the detailed priority-tier criteria. This
page gives the mental model and the recipes; that page gives the exact
signatures. Cross-links point you to it at each step.

## The one mental model

Everything in ProbPipe that "picks an implementation based on the types
you passed" is a **priority-based dispatch registry**, or a
*non-conforming* cousin that the catalog still indexes. A registry is a
priority-ordered list of pluggable **methods**. On each call (without an
explicit `method=`) the registry:

1. Computes a **dispatch key** from the positional arguments — the type
   of the first argument for a unary registry, the `(type, type)` pair
   for a binary one.
2. **Pre-filters** the registered methods by `supported_types()` using a
   fast `issubclass` check (cached per key), and drops any method that is
   *opt-in only*.
3. **Walks** the survivors in descending effective priority and runs the
   first whose `check()` reports `feasible=True`.

That is the whole model. `condition_on` uses it (a `UnaryDispatchRegistry`
of inference methods); a future `kl_divergence` will use it (a
`BinaryDispatchRegistry` of divergence methods). Learn it once and every
registry in the codebase reads the same way.

## Choosing a registry shape

| Your operation dispatches on… | Use | Example |
|---|---|---|
| The type of **one** argument | `UnaryDispatchRegistry` | `inference_method_registry` (backs `condition_on`) |
| The **joint** type of **two** arguments | `BinaryDispatchRegistry` | divergences (KL, TV, Wasserstein), pushforwards |
| A shape that fits neither | a **non-conforming** registry + a catalog adapter | `converter_registry`, `bijector_registry` |

Arity is a property of the registry *type*, not a runtime flag: calling
`execute(p)` with a single argument on a `BinaryDispatchRegistry` is a
static type error, and `_cache_key` raises `TypeError` at runtime rather
than dispatching against a half-formed key.

Two things that are deliberately **not** registries:

- **Op-layer protocol dispatch** — `sample`, `log_prob`, `mean`, and the
  other ops in `core/ops.py` dispatch on a single `@runtime_checkable`
  protocol (`SupportsSampling`, `SupportsLogProb`, …). Each type has
  exactly one implementation — its own underscore method (`_sample`,
  `_log_prob`) — so there is no plurality to rank. Keep these as
  protocol checks; do not wrap them in a registry.
- **Non-conforming registries** — the converter and bijector registries
  dispatch on shapes that do not fit the standard model (see
  [Non-conforming registries](#non-conforming-registries-and-adapters)).
  They keep their own dispatch mechanics and only borrow the catalog's
  introspection surface.

## The method contract

A dispatch method subclasses the arity-appropriate base and supplies
five things. The bases live in `probpipe.core._registry`.

| Member | Kind | Purpose |
|---|---|---|
| `name` | property → `str` | Unique identifier within the registry; the value you pass as `method="…"`. |
| `priority` | property → `int` | Auto-dispatch rank. Defaults to `OPT_IN_ONLY_PRIORITY` (`0`) — see [Priority and opt-in](#priority-and-opt-in). |
| `description` | `ClassVar[str]` | Optional one-line blurb surfaced in the catalog. Defaults to `""`. |
| `supported_types()` | method | Fast `issubclass` pre-filter. Arity-specific return shape (below). |
| `check(*args, **kwargs)` | method → `MethodInfo` | Cheap feasibility probe. No heavy computation. |
| `execute(*args, **kwargs)` | method → result | Do the work. |

```python
from probpipe.core._registry import BinaryDispatchMethod, MethodInfo

class GaussianKL(BinaryDispatchMethod):
    name = "kl_normal_normal"          # a property is also fine
    priority = 90                      # exact, closed-form — see the tiers
    description = "Closed-form KL for two Normals."

    def supported_types(self):
        return ((Normal,), (Normal,))  # (left_types, right_types)

    def check(self, p, q, **_):
        return MethodInfo(feasible=True)

    def execute(self, p, q, **_):
        ...
```

`supported_types()` returns different shapes by arity:

- `UnaryDispatchMethod.supported_types() -> tuple[type, ...]`
- `BinaryDispatchMethod.supported_types() -> tuple[tuple[type, ...], tuple[type, ...]]`
  — the left slot and the right slot, each a tuple of accepted types.

!!! warning "`supported_types()` must return concrete classes"
    The pre-filter uses `issubclass`, which does not work reliably
    against protocols that carry non-method members. Return nominal
    classes here (`Normal`, `TFPDistribution`, `Distribution`). Put
    protocol-based gating — `isinstance(p, SupportsSampling)`,
    `isinstance(q, SupportsLogProb)` — inside `check()`, where it belongs.

`check()` returns a `MethodInfo(feasible: bool, method_name: str = "",
description: str = "")`. Return `feasible=False` (optionally with a
`description` explaining why) to have the registry skip to the next
candidate. Keep `check()` cheap: it runs on every candidate during the
dispatch walk.

For the full signatures and the rendered API, see
[Extending ProbPipe → the dispatch bases](../api/extending.md#custom-inference-methods).

## Priority and opt-in

`priority` is an integer whose magnitude carries meaning. The same scheme
applies to *every* registry, not just inference:

- **`priority > 50`** — *exact*: auto-dispatched, higher preferred.
- **`0 < priority <= 50`** — *inexact*: auto-dispatched, higher
  preferred. The `50` break is documentary; the registry walks every
  positive priority uniformly.
- **`priority == 0`** (`OPT_IN_ONLY_PRIORITY`) — *opt-in only*: excluded
  from the auto-dispatch walk, reachable only by name via `method="…"`.

`0` is the inherited default, so a method that forgets to set a priority
is *safe by default* — it never silently auto-fires. Assign a positive
priority once you have classified the method against the
[tier criteria in the extending guide](../api/extending.md#setting-priority-for-a-new-method).

!!! tip "Monte-Carlo fallbacks get a *low positive* priority, not `0`"
    A universal MC estimator should sit at a low positive priority
    (e.g. `5`) rather than at `OPT_IN_ONLY_PRIORITY`. Users who call
    `kl_divergence(p, q)` without naming a method expect a numerical
    answer whenever one is computable, not a `TypeError`. A low positive
    priority makes MC the last-resort auto-dispatch; `method="kl_mc"`
    still forces it for comparison. Reserve `0` for methods that are
    genuinely unsafe without explicit opt-in (a known-biased estimator,
    say).

Priorities can be re-ranked at runtime with
`registry.set_priorities(name=new_priority, …)` — useful for a deployment
that wants to demote a method (say, one whose compilation tax outweighs
its per-step speed in that environment) without forking the method class.
Moving a method *into* or *out of* `0` changes whether it participates in
auto-dispatch at all, so that crossing emits a `UserWarning`.

## Registering a method

Registration is a subclass plus a `register()` call. Names must be unique
within the registry; re-registering a name raises `ValueError`.

```python
kl_registry.register(GaussianKL())
kl_registry.register(TFPKL())
kl_registry.register(MCKL())
```

`register()` re-sorts the registry by effective priority and invalidates
the type cache, so registration order does not matter for correctness.
What *does* matter is that the registering module actually runs — see
[Side-effect imports](#side-effect-imports-for-plugin-authors).

## Named override

Both `check()` and `execute()` take an optional `method=` keyword that
bypasses the auto-dispatch walk and targets one method by name:

```python
kl_divergence(p, q)                      # auto-select (highest feasible priority)
kl_divergence(p, q, method="kl_mc")      # force the MC estimator
```

A named method still has its `check()` consulted; if it reports
infeasible, `execute()` raises `TypeError` rather than silently falling
back. Use `method=` for reproducibility (pin the implementation in a
saved workflow), for debugging (compare two implementations on the same
input), and to reach opt-in-only methods.

## Creating a new registry

Creating a registry is the explicit three-line constructor pattern —
there is deliberately no `create_registry()` helper (it would save
nothing and invite throwaway registries that clutter the catalog).

```python
# probpipe/validation/_kl_registry.py   (illustrative — the KL registry
# lands in a later stage of this effort; the shape is the template)
from __future__ import annotations

from ..core._registry import BinaryDispatchMethod, BinaryDispatchRegistry, MethodInfo

kl_registry = BinaryDispatchRegistry[BinaryDispatchMethod](
    name="kl",
    description="Kullback-Leibler divergence KL(p || q).",
)

class GaussianKL(BinaryDispatchMethod):
    name = "kl_normal_normal"
    priority = 90                        # exact, closed-form
    def supported_types(self): return ((Normal,), (Normal,))
    def check(self, p, q, **_): return MethodInfo(feasible=True)
    def execute(self, p, q, **_): ...

class MCKL(BinaryDispatchMethod):
    name = "kl_mc"
    priority = 5                         # last-resort fallback
    def supported_types(self): return ((Distribution,), (Distribution,))
    def check(self, p, q, **_):
        ok = (isinstance(p, SupportsSampling)
              and isinstance(p, SupportsLogProb)
              and isinstance(q, SupportsLogProb))
        return MethodInfo(feasible=ok)
    def execute(self, p, q, *, key, n=10_000, **_): ...

kl_registry.register(GaussianKL())
kl_registry.register(MCKL())

# The public op wraps the registry so broadcasting/orchestration is handled
# by the workflow layer, not the registry.
@workflow_function
def kl_divergence(p, q, *, method: str | None = None, **kw):
    return kl_registry.execute(p, q, method=method, **kw)
```

Decisions when creating a registry:

- **Name.** A short lowercase string, unique across the catalog. It is
  the catalog key (`registry_catalog["kl"]`) and the label in the
  printed table.
- **`kind`.** Inherited as `"dispatch"` for anything subclassing
  `BaseDispatchRegistry`; you only set it for a non-conforming adapter
  (see below).
- **`description`.** One line; shown in `registry_catalog.list()` and the
  table.
- **One registry per family.** Prefer `kl_registry`, `tv_registry`,
  `wasserstein_registry` as separate instances over a single tagged
  `discrepancy_registry`. Each is ~30 lines of boilerplate following the
  template above.

A registry constructed with a `name=` **self-registers** in the global
catalog. To construct an isolated registry that does *not* touch the
global singleton — the pattern the test suite uses — pass
`register_in_catalog=False`.

## The registry catalog

The catalog answers two discovery questions at the REPL without the user
having to know which module a registry lives in. It **supplements**
dispatch; it does not replace the per-registry singletons
(`inference_method_registry`, `converter_registry`, …), which stay the
canonical entry points for code that already knows which registry it
wants.

```python
import probpipe

probpipe.registry_catalog                       # prints a table of all registries
probpipe.registry_catalog.names()               # ['bijectors', 'converters', 'inference']
probpipe.registry_catalog.list()                # list[RegistryInfo]
probpipe.registry_catalog["inference"]          # the registry itself (KeyError if absent)
"inference" in probpipe.registry_catalog        # membership test
print(probpipe.registry_catalog.describe("inference"))
```

`describe(name)` renders the per-entry records — priority, supported
types, module path — and lists **opt-in-only entries in their own
section**, so a reader sees they exist but is not surprised when they do
not auto-fire. In a notebook, the catalog's `_repr_html_` renders as a
table.

The catalog's data records:

- `RegistryInfo(name, description, kind, entry_count)` — the *outside*
  view of a registry, one per `list()` element.
- `EntrySummary(name, priority, supported_types, description, module_path)`
  — one per registered entry, with `priority=None` for factory-style
  registries and an `is_opt_in_only` convenience property.

## Non-conforming registries and adapters

"Non-conforming" describes a registry's *dispatch shape*, not its code
quality. The existing converter and bijector registries are well-designed
for what they do but do not fit the standard priority + `check()` +
`supported_types()` model:

- **`converter_registry`** dispatches on `(source_type, target_type)`
  where the *target* is passed as a type, not a value
  (`convert(my_dist, target_type=Normal)`). It also carries a richer
  `ConversionInfo` / `ConversionMethod` vocabulary that does not
  generalize.
- **`bijector_registry`** dispatches on a constraint *instance* or class:
  precedence is exact-instance match first, then a walk of the
  constraint's type MRO. There is no `check()` step — it is a pure
  factory lookup, so its catalog entries carry `priority=None`.

Both stay exactly as they are at the dispatch level. To appear in the
catalog they satisfy the **`SupportsRegistryCataloging`** protocol, which
is intentionally *weaker* than the dispatch contract — it asks only for
identity and introspection:

```python
class SupportsRegistryCataloging(Protocol):
    name: str
    description: str
    kind: str                       # "dispatch" | "factory" | "converter" | "other"
    def entry_summaries(self) -> list[EntrySummary]: ...
    def describe_entry(self, name: str) -> EntrySummary: ...
```

A conforming dispatch registry satisfies this protocol for free by
inheriting `BaseDispatchRegistry`. A non-conforming registry satisfies it
with a small adapter: the `ConverterRegistry` adds
`name`/`description`/`kind = "converter"` plus `entry_summaries()` /
`describe_entry()` directly on the class, and the bijector dispatch — a
module-level dict plus two functions, not a class — gets a thin
`_BijectorRegistryFacade` (`kind = "factory"`) that walks the dict. Both
register explicitly at module load:

```python
converter_registry = ConverterRegistry()
registry_catalog.register(converter_registry)
```

The adapter is purely additive; `convert(...)` and `bijector_for(...)`
are unchanged.

!!! note "`kind` is a plain string"
    Documented values are `"dispatch"`, `"factory"`, `"converter"`, and
    `"other"`. It is a string rather than an enum so a plugin can
    introduce a new kind without a core change; use `"other"` if none
    fit.

## Side-effect imports for plugin authors

Registration happens **at import time**: a method or registry exists in
the catalog only once the module that calls `register()` has run.
`probpipe/__init__.py` imports every built-in registry module for its
side effects, which is why `import probpipe` is enough to populate the
catalog with `inference`, `converters`, and `bijectors`.

A third-party package that adds a method or a registry must make sure its
registration module is imported. The idiomatic options:

- Import the registration module from your package's `__init__.py`, so
  `import my_probpipe_plugin` triggers it.
- Or expose it through a packaging entry point that your package loads on
  import.

If the module never runs, the method silently will not dispatch — the
registry simply behaves as though it were never added. When a custom
method "isn't being picked", the first thing to check is whether its
module was imported.

## See also

- [Extending ProbPipe](../api/extending.md) — the API reference for the
  dispatch bases and the catalog, plus the full priority-tier criteria.
- [Internals](../api/internals.md) — the private machinery a registry
  rarely constructs directly.
- [Contributing](https://github.com/TARPS-group/prob-pipe/blob/main/CONTRIBUTING.md)
  — PR workflow, testing, and docs conventions.
