"""Priority-based dispatch registry hierarchy.

Provides a reusable base for registries that dispatch operations
to pluggable methods based on priority and feasibility.

Priority semantics:

- ``priority > 50`` — *exact* methods. Auto-dispatched in descending
  priority order.
- ``0 < priority <= 50`` — *inexact* methods. Auto-dispatched in
  descending priority order; the ``50`` break is documentary, not
  behavioural, so the registry walks every positive priority uniformly.
- ``priority == 0`` — *opt-in only.* Skipped during auto-dispatch;
  selectable by name via ``method="..."``. This is the default value
  inherited from :class:`BaseDispatchMethod`, so a newly-registered
  method gets the safe behaviour (opt-in) until a contributor
  classifies it explicitly.

See ``docs/api/extending.md`` for the tier criteria a contributor
should use when choosing a priority for a new method.

**Arity-typed subclasses.** :class:`UnaryDispatchRegistry` and
:class:`BinaryDispatchRegistry` differ only in ``supported_types()``
signature, cache-key construction, and the pre-filter predicate.  Arity
is a type-level property: calling ``execute(p)`` (a single positional
argument) on a :class:`BinaryDispatchRegistry` is a static type error
rather than a runtime one.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ._registry_catalog import EntrySummary

__all__ = [
    "OPT_IN_ONLY_PRIORITY",
    "BaseDispatchMethod",
    "BaseDispatchRegistry",
    "BinaryDispatchMethod",
    "BinaryDispatchRegistry",
    "MethodInfo",
    "UnaryDispatchMethod",
    "UnaryDispatchRegistry",
]

# Priority value for methods that should not auto-dispatch. The registry
# excludes any method whose effective priority equals this sentinel from
# the auto-selection walk; such a method is reachable only by name via
# ``method=...``.
OPT_IN_ONLY_PRIORITY: int = 0


@dataclass(frozen=True)
class MethodInfo:
    """Metadata describing whether a method is applicable.

    Returned by a method's ``check()`` to describe feasibility
    without performing the actual computation.
    """

    feasible: bool
    method_name: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Abstract method bases
# ---------------------------------------------------------------------------


class BaseDispatchMethod(ABC):
    """Abstract base for all pluggable dispatch methods.

    Subclasses declare a unique ``name``, a ``priority``, and implement
    ``check``/``execute``.  The ``supported_types`` method is **not**
    defined here — its return shape is arity-dependent and is declared
    by :class:`UnaryDispatchMethod` and :class:`BinaryDispatchMethod`.

    An optional ``description`` class attribute (default ``""``) lets a
    concrete method carry a one-line blurb that surfaces in
    :class:`~probpipe.core._registry_catalog.EntrySummary` records used
    by the registry catalog.
    """

    #: Optional one-line description shown in catalog summaries.
    description: ClassVar[str] = ""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this method (e.g., ``'tfp_nuts'``)."""
        ...

    @abstractmethod
    def check(self, *args: Any, **kwargs: Any) -> MethodInfo:
        """Probe whether this method is applicable (must be cheap)."""
        ...

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the method and return the result."""
        ...

    @property
    def priority(self) -> int:
        """Auto-dispatch ordering.

        Higher priority methods are tried first during auto-selection.
        Defaults to :data:`OPT_IN_ONLY_PRIORITY` — the opt-in-only
        sentinel.  A method that doesn't override this property is
        excluded from auto-dispatch and is reachable only by name via
        ``method="..."`` until a contributor assigns a positive
        priority.  See the module docstring for the
        ``> 50`` / ``<= 50`` / ``== 0`` convention contributors should
        follow when choosing a number.
        """
        return OPT_IN_ONLY_PRIORITY


class UnaryDispatchMethod(BaseDispatchMethod):
    """Abstract base for single-argument dispatch methods.

    Adds :meth:`supported_types`, which returns the nominal types this
    method handles.  Used by :class:`UnaryDispatchRegistry` as a fast
    ``issubclass`` pre-filter before ``check()`` is called.
    """

    @abstractmethod
    def supported_types(self) -> tuple[type, ...]:
        """Types this method can operate on (fast pre-filter).

        Must return concrete classes, not protocols with non-method
        members — ``issubclass`` does not work reliably with such
        protocols.  Protocol-based feasibility checks belong in
        ``check()``.
        """
        ...


class BinaryDispatchMethod(BaseDispatchMethod):
    """Abstract base for two-argument dispatch methods.

    Adds :meth:`supported_types`, which returns a pair of type-tuples
    ``((left_types, ...), (right_types, ...))`` used by
    :class:`BinaryDispatchRegistry` as a fast ``issubclass`` pre-filter.
    """

    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        """Left and right type-tuples for the pre-filter.

        Returns a 2-tuple of type-tuples: the first covers the left
        argument, the second the right.  Example::

            def supported_types(self):
                return ((Normal,), (Normal,))

        Must contain concrete classes, not protocols — see the note on
        :meth:`UnaryDispatchMethod.supported_types`.
        """
        ...


# ---------------------------------------------------------------------------
# Registry base and arity-typed subclasses
# ---------------------------------------------------------------------------


class BaseDispatchRegistry[M: BaseDispatchMethod](ABC):
    """Priority-based dispatch registry (arity-independent base).

    Holds all arity-independent logic: registration, priority
    management, opt-in filtering, and the ``check``/``execute`` loop.
    Arity-specific subclasses override :meth:`_cache_key`,
    :meth:`_find_methods`, and :meth:`_format_key`.

    Dispatch mechanics
    ------------------
    On every :meth:`check` / :meth:`execute` call (without an explicit
    ``method=...``) the registry:

    1. Computes a *dispatch key* from the positional arguments via
       :meth:`_cache_key`.  The key shape is arity-dependent — a single
       ``type`` for unary registries, a ``(type, type)`` pair for
       binary registries.
    2. Looks up the auto-dispatchable methods that match that key via
       :meth:`_find_methods`, using an internal ``_type_cache`` so the
       ``issubclass`` pre-filter only runs once per key shape.
    3. Walks the matches in descending effective-priority order and
       returns/runs the first whose ``check()`` reports feasibility.

    Methods whose effective priority equals :data:`OPT_IN_ONLY_PRIORITY`
    (``0``) are excluded from that auto-dispatch walk and are reachable
    only by name via ``method="..."``.

    Under-arity behaviour
    ---------------------
    :meth:`check` is a polite probe: when called with **no** positional
    arguments it returns an infeasible :class:`MethodInfo` (``"No
    arguments provided"``) rather than raising.  This is the natural
    "is there anything dispatchable here?" form.  If fewer arguments
    are given than the registry's dispatch shape requires — e.g. a
    single positional arg to a :class:`BinaryDispatchRegistry` —
    :meth:`_cache_key` raises :class:`TypeError`, and :meth:`check`
    propagates it.  The asymmetry is intentional: zero args is a
    no-target probe, partial args is an API-contract violation that
    should fail loudly.  :meth:`execute` raises in both cases.

    Priority overrides
    ------------------
    :meth:`set_priorities` mutates a per-registry override map without
    touching the method instances.  This lets a deployment re-rank
    methods at runtime (for example, demote ``blackjax_nuts`` for an
    environment where the JAX compilation tax outweighs its per-step
    speed advantage) without forking and re-registering the method
    class.  Overrides also provide the supported escape hatch for
    moving a method into or out of opt-in-only status — a transition
    that emits a :class:`UserWarning` because it changes whether the
    method participates in auto-dispatch at all.

    Catalog integration
    -------------------
    A dispatch registry constructed with a non-empty ``name`` self-registers
    in the global :data:`~probpipe.core._registry_catalog.registry_catalog`
    so it can be discovered via
    ``probpipe.registry_catalog["<name>"]``.  The default for
    ``register_in_catalog`` is derived from whether a ``name`` was
    supplied; passing ``register_in_catalog=False`` opts out explicitly
    (used by tests that want isolated registries).

    Two read APIs live on the registry:

    - :meth:`list_methods` returns the priority-ordered ``list[str]`` of
      method names.  This is the dispatch-side convenience that callers
      pass back as ``method="..."``; existing tutorial notebooks and
      tests depend on this shape.
    - :meth:`entry_summaries` returns ``list[EntrySummary]`` — the
      rich introspection records the catalog renders in
      ``describe(name)``.  This is the surface satisfying
      :class:`~probpipe.core._registry_catalog.SupportsRegistryCataloging`.
    """

    #: Catalog ``kind`` for dispatch registries.  Subclasses inherit.
    kind: ClassVar[str] = "dispatch"

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str = "",
        register_in_catalog: bool | None = None,
    ) -> None:
        """Create a registry, optionally registering it in the catalog.

        Parameters
        ----------
        name
            Identifier used as the catalog key.  ``None`` (the default)
            constructs an *unnamed* registry that does not appear in the
            catalog; this preserves the bare ``UnaryDispatchRegistry()``
            / ``BinaryDispatchRegistry()`` call form used throughout the
            test suite.
        description
            One-line human-readable purpose statement, shown in catalog
            descriptions.
        register_in_catalog
            ``None`` (default) → auto-derived as ``name is not None``.
            ``True`` requires a non-empty *name* or :class:`ValueError`
            is raised.  ``False`` skips catalog registration even when a
            name was given (used to construct named registries for
            catalog-round-trip tests without polluting the global
            singleton).

        Raises
        ------
        ValueError
            If ``register_in_catalog`` is ``True`` but no ``name`` was
            supplied.
        """
        if register_in_catalog is None:
            register_in_catalog = name is not None
        if register_in_catalog and not name:
            raise ValueError(
                "Cannot register a registry in the catalog without a name; "
                "either pass name='...' or set register_in_catalog=False."
            )
        self.name: str = name or ""
        self.description: str = description
        self._methods: list[M] = []
        self._name_index: dict[str, M] = {}
        self._priority_overrides: dict[str, int] = {}
        self._type_cache: dict[Any, list[M]] = {}
        if register_in_catalog:
            # Lazy import: the catalog imports OPT_IN_ONLY_PRIORITY from
            # this module, so doing the import in-function keeps
            # _registry_catalog -> _registry as the only module-load-time
            # edge.
            from ._registry_catalog import registry_catalog

            registry_catalog.register(self)

    # -- registration -------------------------------------------------------

    def register(self, method: M) -> None:
        """Register a method (invalidates the lookup cache).

        Parameters
        ----------
        method
            The dispatch method to register.  Must expose a non-empty
            ``name`` and a unique name across all currently-registered
            methods.

        Raises
        ------
        ValueError
            If ``method.name`` is empty/falsy, or if a method with the
            same name is already registered.
        """
        if not method.name:
            raise ValueError(f"Method.name must be a non-empty string; got {method.name!r}")
        if method.name in self._name_index:
            raise ValueError(f"Method name {method.name!r} is already registered")
        self._methods.append(method)
        self._name_index[method.name] = method
        self._sort_methods()

    # -- priority management ------------------------------------------------

    def _effective_priority(self, method: M) -> int:
        """Return the effective priority (override if set, else default)."""
        return self._priority_overrides.get(method.name, method.priority)

    def _sort_methods(self) -> None:
        """Re-sort methods by effective priority and clear caches."""
        self._methods.sort(key=self._effective_priority, reverse=True)
        self._type_cache.clear()

    def set_priorities(self, **name_to_priority: int) -> None:
        """Override the priority of one or more methods.

        Higher priority methods are tried first during auto-selection.
        Overrides are unrestricted: the new value can be any integer,
        including the opt-in-only sentinel ``0``. When an override
        moves a method *into* or *out of* ``0``, the registry emits a
        :class:`UserWarning` because that crossing changes whether the
        method participates in auto-dispatch at all.

        Parameters
        ----------
        **name_to_priority
            Keyword arguments mapping method names to new priorities.
            e.g., ``set_priorities(blackjax_rwmh=200, tfp_nuts=50)``

        Raises
        ------
        KeyError
            If a method name is not registered.
        """
        for name in name_to_priority:
            if name not in self._name_index:
                available = ", ".join(sorted(self._name_index)) or "(none)"
                raise KeyError(f"No method named {name!r}. Available: {available}")
        for name, new_priority in name_to_priority.items():
            old_priority = self._effective_priority(self._name_index[name])
            if (old_priority == OPT_IN_ONLY_PRIORITY) != (new_priority == OPT_IN_ONLY_PRIORITY):
                direction = (
                    "out of opt-in-only"
                    if old_priority == OPT_IN_ONLY_PRIORITY
                    else "into opt-in-only"
                )
                warnings.warn(
                    f"Priority override for {name!r} moves it {direction} "
                    f"({old_priority} -> {new_priority}); auto-dispatch "
                    f"participation changes accordingly.",
                    UserWarning,
                    stacklevel=2,
                )
        self._priority_overrides.update(name_to_priority)
        self._sort_methods()

    # -- query --------------------------------------------------------------

    def get_method(self, name: str) -> M:
        """Look up a method by name.  Raises ``KeyError`` if not found."""
        try:
            return self._name_index[name]
        except KeyError:
            available = ", ".join(sorted(self._name_index)) or "(none)"
            raise KeyError(f"No method named {name!r}. Available: {available}") from None

    def list_methods(self) -> list[str]:
        """Return method names in priority order (highest first).

        This is the dispatch-side convenience: each returned string is a
        valid value for ``method="..."`` in :meth:`check` / :meth:`execute`.
        For richer per-entry introspection (priority, supported types,
        module path) use :meth:`entry_summaries`.
        """
        return [m.name for m in self._methods]

    def entry_summaries(self) -> list[EntrySummary]:
        """Return one :class:`EntrySummary` per registered method.

        Walks the methods in priority order (matching
        :meth:`list_methods`).  Each summary carries the method's
        effective priority (override-aware), its
        :meth:`~UnaryDispatchMethod.supported_types` result, the
        :attr:`BaseDispatchMethod.description` if any, and the
        ``type(method).__module__`` for source-navigation in tooling.

        Surface targeted by
        :class:`~probpipe.core._registry_catalog.SupportsRegistryCataloging`.
        """
        # Local import: the catalog module imports from this module, so
        # keeping this import lazy prevents a load-order cycle.
        from ._registry_catalog import EntrySummary

        # ``supported_types`` is declared abstractly on
        # ``UnaryDispatchMethod`` and ``BinaryDispatchMethod`` rather
        # than on ``BaseDispatchMethod`` (its return shape is
        # arity-dependent).  At runtime every instantiable method is one
        # of those two; the access is safe.  Suppress the static-check
        # complaint here rather than weakening the base class.
        return [
            EntrySummary(
                name=m.name,
                priority=self._effective_priority(m),
                supported_types=m.supported_types(),  # type: ignore[attr-defined]
                description=type(m).description,
                module_path=type(m).__module__,
            )
            for m in self._methods
        ]

    def describe_entry(self, name: str) -> EntrySummary:
        """Return the :class:`EntrySummary` for the named method.

        Raises
        ------
        KeyError
            If *name* is not registered.  The error message lists the
            available names.
        """
        from ._registry_catalog import EntrySummary

        try:
            m = self._name_index[name]
        except KeyError:
            available = ", ".join(sorted(self._name_index)) or "(none)"
            raise KeyError(f"No method named {name!r}. Available: {available}") from None
        # See ``entry_summaries`` for the rationale behind the
        # ``# type: ignore`` on the ``supported_types`` access.
        return EntrySummary(
            name=m.name,
            priority=self._effective_priority(m),
            supported_types=m.supported_types(),  # type: ignore[attr-defined]
            description=type(m).description,
            module_path=type(m).__module__,
        )

    def _is_auto_dispatchable(self, m: M) -> bool:
        """True if *m* participates in auto-dispatch.

        A method whose effective priority equals
        :data:`OPT_IN_ONLY_PRIORITY` is excluded from the auto-dispatch
        walk.  It remains reachable by explicit ``method="..."``
        regardless.  Subclasses must call this inside
        :meth:`_find_methods` to keep filtering consistent.
        """
        return self._effective_priority(m) != OPT_IN_ONLY_PRIORITY

    def check(self, *args: Any, method: str | None = None, **kwargs: Any) -> MethodInfo:
        """Check feasibility.  Auto-selects or uses the named method."""
        if method is not None:
            m = self.get_method(method)
            return m.check(*args, **kwargs)

        if not args:
            return MethodInfo(feasible=False, description="No arguments provided")

        key = self._cache_key(args)
        for m in self._find_methods(key):
            info = m.check(*args, **kwargs)
            if info.feasible:
                return info

        return MethodInfo(
            feasible=False,
            description=f"No applicable method for {self._format_key(key)}",
        )

    def execute(self, *args: Any, method: str | None = None, **kwargs: Any) -> Any:
        """Execute using the best (or named) method.

        Raises ``TypeError`` if no method is applicable.
        Raises ``KeyError`` if *method* is not registered.
        """
        if method is not None:
            m = self.get_method(method)
            info = m.check(*args, **kwargs)
            if not info.feasible:
                raise TypeError(f"Method {method!r} is not applicable: {info.description}")
            return m.execute(*args, **kwargs)

        if not args:
            raise TypeError("No arguments provided for dispatch")

        key = self._cache_key(args)
        for m in self._find_methods(key):
            info = m.check(*args, **kwargs)
            if info.feasible:
                return m.execute(*args, **kwargs)

        raise TypeError(
            f"No method registered for {self._format_key(key)}. Available: {self.list_methods()}"
        )

    # -- internals (arity-specific overrides) --------------------------------

    @abstractmethod
    def _cache_key(self, args: tuple[Any, ...]) -> Any:
        """Compute the dispatch cache key from positional arguments."""
        ...

    @abstractmethod
    def _find_methods(self, key: Any) -> list[M]:
        """Return auto-dispatchable methods matching *key* (cached).

        Implementations must:

        1. Consult ``self._type_cache`` to avoid recomputation.
        2. Call ``self._is_auto_dispatchable(m)`` to exclude opt-in-only
           methods (Approach A: single point of truth for the filter).
        3. Apply an ``issubclass`` pre-filter via ``supported_types()``.
        """
        ...

    @abstractmethod
    def _format_key(self, key: Any) -> str:
        """Format *key* as a human-readable string for error messages."""
        ...


class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]):
    """Priority-based registry for single-argument dispatch.

    Dispatches on the type of the first positional argument.
    """

    def _cache_key(self, args: tuple[Any, ...]) -> type:
        return type(args[0])

    def _find_methods(self, key: type) -> list[M]:
        """Return auto-dispatchable methods matching *key* (cached).

        Applies an ``issubclass`` pre-filter on ``supported_types()``
        and excludes opt-in-only methods via
        :meth:`~BaseDispatchRegistry._is_auto_dispatchable`.

        Uses ``issubclass`` rather than ``isinstance`` because we are
        comparing *types*, not instances.  ``supported_types()`` must
        return concrete classes (not protocols with non-method members),
        since ``issubclass`` does not work reliably with such protocols.
        """
        if key not in self._type_cache:
            self._type_cache[key] = [
                m
                for m in self._methods
                if self._is_auto_dispatchable(m)
                and any(issubclass(key, st) for st in m.supported_types())
            ]
        return self._type_cache[key]

    def _format_key(self, key: type) -> str:
        return key.__name__


class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]):
    """Priority-based registry for two-argument dispatch.

    Dispatches on the joint type of the first two positional arguments.
    Each registered method's ``supported_types()`` must return a 2-tuple
    of type-tuples: ``((left_types, ...), (right_types, ...))``.
    """

    def _cache_key(self, args: tuple[Any, ...]) -> tuple[type, type]:
        if len(args) < 2:
            raise TypeError(
                "BinaryDispatchRegistry requires at least two positional "
                f"arguments; got {len(args)}"
            )
        return (type(args[0]), type(args[1]))

    def _find_methods(self, key: tuple[type, type]) -> list[M]:
        """Return auto-dispatchable methods matching the type pair *key* (cached).

        A method is included when the left arg type is a subclass of at
        least one type in ``supported_types()[0]`` **and** the right arg
        type is a subclass of at least one type in
        ``supported_types()[1]`` — i.e., both slots must independently
        match against the corresponding type-tuple via ``any(issubclass(...))``.
        Opt-in-only methods are excluded via
        :meth:`~BaseDispatchRegistry._is_auto_dispatchable`.
        """
        if key not in self._type_cache:
            tl, tr = key
            matches: list[M] = []
            for m in self._methods:
                if not self._is_auto_dispatchable(m):
                    continue
                st = m.supported_types()
                if any(issubclass(tl, lt) for lt in st[0]) and any(
                    issubclass(tr, rt) for rt in st[1]
                ):
                    matches.append(m)
            self._type_cache[key] = matches
        return self._type_cache[key]

    def _format_key(self, key: tuple[type, type]) -> str:
        return f"({key[0].__name__}, {key[1].__name__})"
