"""Dispatch methods and registries.

A dispatch registry holds named implementations of one operation and
selects among them by the types of the arguments and their relative
priorities. Every method declares whether it is **exact**. Selection order
is based on four criteria, in decreasing precedence:

1. exact methods before approximate ones;
2. priority among methods of the same exactness, higher first;
3. specificity, favoring the method whose declared types are closest to
   the arguments' classes in method-resolution order;
4. registration order.

A method whose ``priority`` is ``None`` is **opt-in-only**: automatic
selection skips it and it runs only when named through ``method="..."``.
``set_priorities`` re-ranks at runtime but never changes a method's
exactness.

A method's ``check`` returns a :class:`Feasibility`: whether the call is
feasible, if not then why, and which declarations are pending. The
registry's ``check`` returns a :class:`MethodInfo`, which adds the selected
method's name and declared exactness from the registration.

A :class:`ResolutionError` means there is no available implementation under
the requested controls. A :class:`MathematicalDomainError` means the
mathematical operation is known to be undefined.
"""

from __future__ import annotations

import abc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BaseDispatchMethod",
    "BaseDispatchRegistry",
    "BinaryDispatchMethod",
    "BinaryDispatchRegistry",
    "BinarySupportedTypes",
    "Feasibility",
    "MathematicalDomainError",
    "MethodInfo",
    "ResolutionError",
    "UnaryDispatchMethod",
    "UnaryDispatchRegistry",
    "UnarySupportedTypes",
]

type UnarySupportedTypes = tuple[type, ...]
type BinarySupportedTypes = tuple[tuple[type, ...], tuple[type, ...]]


class ResolutionError(Exception):
    """No available implementation under the requested controls.

    Raised by a registry when no registered method is feasible for the
    arguments, when the first candidate that is not infeasible is still
    unresolved, or when the method a caller named is infeasible or not
    registered.
    """


class MathematicalDomainError(ValueError):
    """The mathematical operation is known to be undefined.

    Raised by a method that can establish nonexistence, such as a
    requested mean that does not exist.
    """


@dataclass(frozen=True)
class Feasibility:
    """What a method's ``check`` reports about one call.

    Attributes
    ----------
    feasible : bool or None
        ``True`` if the method applies, ``False`` if it does not, and ``None``
        if the probe could not determine feasibility.
    description : str
        Describes why method does not apply if ``feasible`` is ``False``.
    pending : tuple of str
        One entry per missing declaration, each describing what is required
        for the method to be feasible. Non-empty exactly when ``feasible`` is
        ``None``.

    Raises
    ------
    TypeError
        If ``feasible`` is anything but a ``bool`` or ``None``. Registries
        compare it by identity, so a truthy or falsy stand-in such as ``1``
        or ``""`` would make ``check`` and ``execute`` disagree.
    ValueError
        If ``pending`` is empty while ``feasible`` is ``None``, or non-empty
        while it is not.
    """

    feasible: bool | None
    description: str = ""
    pending: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.feasible is not None and type(self.feasible) is not bool:
            raise TypeError(f"feasible must be a bool or None; got {self.feasible!r}")
        if self.feasible is None and not self.pending:
            raise ValueError("an unresolved Feasibility must name its pending declarations")
        if self.feasible is not None and self.pending:
            raise ValueError("only an unresolved Feasibility carries pending declarations")

    @property
    def unresolved(self) -> bool:
        """``True`` when ``feasible`` is ``None``."""
        return self.feasible is None


@dataclass(frozen=True)
class MethodInfo(Feasibility):
    """Used by a registry to describe a method's feasibility plus its registration information.

    The ``method_name`` and ``exact`` attributes are both ``None`` exactly when no method could
    be selected, which is the infeasible report that lists every method tried; a feasible or
    unresolved report names its method.

    Attributes
    ----------
    method_name : str or None
        The selected method's name.
    exact : bool or None
        The selected method's declared exactness.

    Raises
    ------
    ValueError
        If exactly one of ``method_name`` and ``exact`` is ``None``; if both
        are ``None`` while ``feasible`` is not ``False``; or on a condition
        :class:`Feasibility` rejects.
    """

    method_name: str | None = None
    exact: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.method_name is None) != (self.exact is None):
            raise ValueError("method_name and exact are set together or not at all")
        if self.method_name is None and self.feasible is not False:
            raise ValueError("a feasible or unresolved MethodInfo names its method")


# ---------------------------------------------------------------------------
# Abstract method bases
# ---------------------------------------------------------------------------


class BaseDispatchMethod[SupportedTypesT](ABC):
    """Abstract base for all pluggable dispatch methods.

    A subclass declares a unique ``name``, whether it is ``exact``, the
    ``supported_types`` the registry's structural pre-filter admits, and
    ``check`` / ``execute``; ``priority`` has a default. The type parameter
    is the shape of ``supported_types``. :class:`UnaryDispatchMethod` and
    :class:`BinaryDispatchMethod` fix it, so an implementation subclasses
    one of those and never spells the parameter.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this method, for example ``'blackjax_nuts'``."""
        ...

    @property
    @abstractmethod
    def exact(self) -> bool:
        """Whether the result denotes the requested mathematical object.

        ``True`` for an equivalent representation or an exact draw in law;
        ``False`` for a stand-in, such as a finite Monte Carlo output or a
        moment-matched family. Declared once and fixed for the method's life.
        """
        ...

    @abstractmethod
    def supported_types(self) -> SupportedTypesT:
        """Types admitted by the registry's structural pre-filter.

        Concrete classes, matched by ``issubclass``; a protocol with
        non-method members does not work there and belongs in ``check``.
        The shape is the type parameter: a tuple of classes for a unary
        method, a ``(left_types, right_types)`` pair of them for a binary
        one.
        """
        ...

    @abstractmethod
    def check(self, *args: Any, **kwargs: Any) -> Feasibility:
        """Probe feasibility for a call without significant computation."""
        ...

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the method and return the result."""
        ...

    @property
    def priority(self) -> int | None:
        """Rank among methods of the same exactness, higher first.

        ``None``, the default, is opt-in-only: excluded from automatic selection
        and reachable only by name.
        """
        return None


class UnaryDispatchMethod(BaseDispatchMethod[UnarySupportedTypes]):
    """Dispatch method whose first argument determines admission."""


class BinaryDispatchMethod(BaseDispatchMethod[BinarySupportedTypes]):
    """Dispatch method whose first two arguments determine admission."""


# ---------------------------------------------------------------------------
# Registry base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Registration[M]:
    """A method with the declarations the registry read from it at registration.

    The registry ranks, filters, and reports from these fields and never
    reads the method's attributes again, so a method mutated after
    registration changes nothing the registry does.
    """

    method: M
    name: str
    exact: bool
    priority: int | None
    supported_types: Any
    index: int


def _validated_priority(name: str, priority: Any) -> int | None:
    """``priority`` if it is an ``int`` or ``None``; ``TypeError`` otherwise, ``bool`` included."""
    if priority is None or (isinstance(priority, int) and not isinstance(priority, bool)):
        return priority
    raise TypeError(f"Method {name!r} priority must be an int or None; got {priority!r}")


class BaseDispatchRegistry[M: BaseDispatchMethod[Any]](ABC):
    """Registry of dispatch methods for one operation.

    :meth:`register` adds a method, :meth:`check` reports which method a call
    would run, :meth:`execute` runs it, :meth:`set_priorities` re-ranks at
    runtime, and :meth:`list_methods` lists the methods in selection order.
    The registry reads a method's ``name``, ``exact``, ``priority``, and
    ``supported_types()`` once, at registration, and validates them before it
    changes; ranking, filtering, and reporting then use that registration, so
    a method mutated afterwards changes nothing.

    Everything that does not depend on how many arguments select the method
    is implemented here, admission and ranking included; an arity subclass
    supplies the four hooks :meth:`_cache_key`,
    :meth:`_validate_supported_types`, :meth:`_distance`, and
    :meth:`_format_key`.
    """

    def __init__(self) -> None:
        self._registrations: list[_Registration[M]] = []
        self._by_name: dict[str, _Registration[M]] = {}
        self._priority_overrides: dict[str, int | None] = {}
        self._type_cache: dict[Any, list[_Registration[M]]] = {}
        self._cache_token = abc.get_cache_token()

    # -- registration -------------------------------------------------------

    def _available(self) -> str:
        return ", ".join(sorted(self._by_name)) or "(none)"

    def register(self, method: M) -> None:
        """Register a method.

        The method's ``name``, ``exact``, ``priority``, and
        ``supported_types()`` are read here, once, and validated before the
        registry changes, so a rejected method leaves it as it was.

        Parameters
        ----------
        method : M
            The method to register.

        Raises
        ------
        TypeError
            If ``method.name`` is not a ``str``; if ``method.exact`` is not a
            ``bool``; if ``method.priority`` is not an ``int`` or ``None``, a
            ``bool`` included; or if ``method.supported_types()`` does not
            have the registry's arity shape.
        ValueError
            If ``method.name`` is empty or already registered.
        """
        name = method.name
        if not isinstance(name, str):
            raise TypeError(f"Method.name must be a str; got {name!r}")
        if not name:
            raise ValueError("Method.name must be a non-empty string; got ''")
        if name in self._by_name:
            raise ValueError(f"Method name {name!r} is already registered")
        exact = method.exact
        if type(exact) is not bool:
            raise TypeError(f"Method {name!r} must declare exact as a bool; got {exact!r}")
        priority = _validated_priority(name, method.priority)
        supported_types = method.supported_types()
        self._validate_supported_types(name, supported_types)
        registration = _Registration(
            method, name, exact, priority, supported_types, len(self._registrations)
        )
        self._registrations.append(registration)
        self._by_name[name] = registration
        self._sort_registrations()

    # -- ranking ------------------------------------------------------------

    def _effective_priority(self, registration: _Registration[M]) -> int | None:
        return self._priority_overrides.get(registration.name, registration.priority)

    def _rank(self, registration: _Registration[M]) -> tuple[int, int, int]:
        """Exactness, then opt-in status, then priority: the selection key that needs no call."""
        priority = self._effective_priority(registration)
        return (0 if registration.exact else 1, 1 if priority is None else 0, -(priority or 0))

    def _sort_key(self, registration: _Registration[M]) -> tuple[int, int, int, int]:
        return (*self._rank(registration), registration.index)

    def _sort_registrations(self) -> None:
        self._registrations.sort(key=self._sort_key)
        self._type_cache.clear()

    def set_priorities(
        self,
        priorities: Mapping[str, int | None] | None = None,
        /,
        **kwargs: int | None,
    ) -> None:
        """Override the rank of one or more methods.

        Overrides are recorded on the registry and never mutate a method. An
        override cannot change exactness, so it never lifts an approximate
        method above an exact one. A move between ``None`` and an integer
        emits a ``UserWarning``, since it changes whether the method
        participates in automatic selection. When an error is raised, nothing
        is applied.

        Parameters
        ----------
        priorities : mapping of str to int or None, optional
            New priorities by method name; ``None`` makes a method
            opt-in-only. This form accepts names that are not Python
            identifiers.
        **kwargs : int or None
            New priorities by method name, as keywords.

        Raises
        ------
        ValueError
            If a name appears both in ``priorities`` and as a keyword.
        KeyError
            If a name is not registered.
        TypeError
            If a value is not an ``int`` or ``None``, a ``bool`` included.
        """
        overrides: dict[str, int | None] = dict(priorities or {})
        for name, value in kwargs.items():
            if name in overrides:
                raise ValueError(f"Method {name!r} given both in the mapping and as a keyword")
            overrides[name] = value
        for name, value in overrides.items():
            if name not in self._by_name:
                raise KeyError(f"No method named {name!r}. Available: {self._available()}")
            _validated_priority(name, value)
        for name, new_priority in overrides.items():
            old_priority = self._effective_priority(self._by_name[name])
            if (old_priority is None) != (new_priority is None):
                direction = "out of opt-in-only" if old_priority is None else "into opt-in-only"
                warnings.warn(
                    f"Priority override for {name!r} moves it {direction} "
                    f"({old_priority} -> {new_priority}); auto-dispatch "
                    f"participation changes accordingly.",
                    UserWarning,
                    stacklevel=2,
                )
        self._priority_overrides.update(overrides)
        self._sort_registrations()

    # -- query --------------------------------------------------------------

    def get_method(self, name: str) -> M:
        """Look up a method by name.

        Parameters
        ----------
        name : str
            A registered method name.

        Returns
        -------
        M
            The registered method.

        Raises
        ------
        KeyError
            If no method is registered under ``name``.
        """
        try:
            return self._by_name[name].method
        except KeyError:
            raise KeyError(f"No method named {name!r}. Available: {self._available()}") from None

    def _named(self, name: str) -> _Registration[M]:
        """The registration of the method a caller requested with ``method=``.

        Unlike :meth:`get_method`, a name that is not registered raises
        :class:`ResolutionError`, since the request is a dispatch that cannot
        resolve.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise ResolutionError(
                f"No method named {name!r}. Available: {self._available()}"
            ) from None

    def list_methods(self) -> list[str]:
        """Every registered method name, ranked by exactness, priority, and registration order.

        Returns
        -------
        list of str
            The names in the order automatic selection considers them before
            type specificity, which depends on the arguments; opt-in-only
            methods are included at their rank. The listing does not say what
            a call would run; :meth:`check` does.
        """
        return [registration.name for registration in self._registrations]

    def _is_auto_dispatchable(self, registration: _Registration[M]) -> bool:
        return self._effective_priority(registration) is not None

    @staticmethod
    def _passes_exact_only(registration: _Registration[M], exact_only: bool) -> bool:
        return registration.exact or not exact_only

    def _candidates(self, key: Any, exact_only: bool) -> list[_Registration[M]]:
        admitting = self._find_methods(key)
        return [
            registration
            for registration in admitting
            if self._passes_exact_only(registration, exact_only)
        ]

    def check(
        self,
        *args: Any,
        method: str | None = None,
        exact_only: bool = False,
        **kwargs: Any,
    ) -> MethodInfo:
        """Report which method a call would run, without running anything.

        A ``method=`` name bypasses the type pre-filter, not the arity: the
        positional arguments are validated before the named method's
        ``check`` runs.

        Parameters
        ----------
        *args : Any
            Positional arguments of the call; the dispatch key is computed
            from them.
        method : str or None
            A registered method name to probe instead of auto-selecting.
        exact_only : bool
            If ``True``, approximate methods are excluded.
        **kwargs : Any
            Keyword arguments of the call, passed to each method's ``check``.

        Returns
        -------
        MethodInfo
            Under auto-selection, the report of the first candidate in
            selection order whose ``check`` is feasible or unresolved, so an
            unresolved candidate ranked above a feasible one is the one
            reported. When no candidate is feasible, an infeasible report
            that names no method and whose ``description`` lists every method
            tried. With ``method``, that method's report, or an infeasible
            report when ``exact_only`` excludes it.

        Raises
        ------
        ResolutionError
            If ``method`` is not a registered name.
        TypeError
            If there are fewer positional arguments than the registry's arity
            requires, none included.
        """
        key = self._cache_key(args)
        if method is not None:
            named = self._named(method)
            if not self._passes_exact_only(named, exact_only):
                return MethodInfo(
                    feasible=False,
                    description=f"Method {method!r} is approximate and exact_only was requested",
                    method_name=method,
                    exact=named.exact,
                )
            return self._report(named, named.method.check(*args, **kwargs))
        tried: list[str] = []
        for candidate in self._candidates(key, exact_only):
            info = self._report(candidate, candidate.method.check(*args, **kwargs))
            if info.feasible is not False:
                return info
            tried.append(f"{candidate.name}: {info.description or 'infeasible'}")
        return MethodInfo(
            feasible=False,
            description=self._no_method_message(key, tried, exact_only),
        )

    def execute(
        self,
        *args: Any,
        method: str | None = None,
        exact_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the selected method and return its result.

        Under auto-selection the first feasible candidate in selection order
        runs. A ``method=`` name bypasses the type pre-filter, not the arity.
        An exception raised by a method's ``check`` or ``execute`` propagates
        unchanged, and no other method is tried after it.

        Parameters
        ----------
        *args : Any
            Positional arguments of the call; the dispatch key is computed
            from them, and they are passed to the selected method.
        method : str or None
            A registered method name to run instead of auto-selecting.
        exact_only : bool
            If ``True``, approximate methods are excluded.
        **kwargs : Any
            Keyword arguments of the call, passed to the selected method's
            ``check`` and ``execute``.

        Returns
        -------
        Any
            The result of the selected method's ``execute``.

        Raises
        ------
        ResolutionError
            If no candidate is feasible; if the first candidate that is not
            infeasible is unresolved; or if ``method`` names a method that is
            not registered, is infeasible, or is approximate while
            ``exact_only`` is ``True``.
        TypeError
            If there are fewer positional arguments than the registry's arity
            requires, none included.
        """
        key = self._cache_key(args)
        if method is not None:
            named = self._named(method)
            if not self._passes_exact_only(named, exact_only):
                raise ResolutionError(
                    f"Method {method!r} is approximate and exact_only was requested"
                )
            info = named.method.check(*args, **kwargs)
            if info.feasible is None:
                raise ResolutionError(
                    f"Method {method!r} is unresolved; pending: {', '.join(info.pending)}"
                )
            if not info.feasible:
                raise ResolutionError(f"Method {method!r} is not applicable: {info.description}")
            return named.method.execute(*args, **kwargs)
        tried: list[str] = []
        for candidate in self._candidates(key, exact_only):
            info = candidate.method.check(*args, **kwargs)
            if info.feasible is None:
                raise ResolutionError(
                    f"Method {candidate.name!r} is unresolved; pending: {', '.join(info.pending)}"
                )
            if info.feasible is True:
                return candidate.method.execute(*args, **kwargs)
            tried.append(f"{candidate.name}: {info.description or 'infeasible'}")
        raise ResolutionError(self._no_method_message(key, tried, exact_only))

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _report(registration: _Registration[M], feasibility: Feasibility) -> MethodInfo:
        """The method's feasibility with its registered name and exactness."""
        return MethodInfo(
            feasible=feasibility.feasible,
            description=feasibility.description,
            pending=feasibility.pending,
            method_name=registration.name,
            exact=registration.exact,
        )

    def _no_method_message(self, key: Any, tried: list[str], exact_only: bool) -> str:
        formatted = self._format_key(key)
        restriction = " with exact_only" if exact_only else ""
        if tried:
            return f"No feasible method for {formatted}{restriction}. Tried: " + "; ".join(tried)
        return (
            f"No method registered for {formatted}{restriction}. Available: {self.list_methods()}"
        )

    @abstractmethod
    def _cache_key(self, args: tuple[Any, ...]) -> Any:
        """The dispatch key for the positional arguments.

        A type for a unary registry, a pair of types for a binary one; the
        key is what ``supported_types`` is matched against and what the
        method cache is indexed by. Raises ``TypeError`` when ``args`` has
        fewer entries than the arity needs, none included.
        """
        ...

    @abstractmethod
    def _validate_supported_types(self, name: str, supported_types: Any) -> None:
        """Raise ``TypeError`` unless ``supported_types`` has the arity's shape.

        Python enforces nothing about the type parameter of
        :class:`BaseDispatchMethod`, and a wrong shape is admitted silently
        by ``issubclass``, which accepts a tuple as its second argument, so
        registration checks the runtime value. ``name`` is for the message.
        """
        ...

    def _find_methods(self, key: Any) -> list[_Registration[M]]:
        """The auto-dispatchable registrations admitting ``key``, in selection order.

        Ordered by :meth:`_rank`, then by the distance :meth:`_distance`
        reports for the key, then by registration order, and memoized in
        ``self._type_cache[key]``, which :meth:`_sort_registrations` clears
        whenever ranks change.

        Registering a virtual subclass of an abstract base class changes what
        ``issubclass`` answers without any registry call, so the cache is also
        dropped whenever ``abc.get_cache_token()`` moves, as
        ``functools.singledispatch`` does.
        """
        token = abc.get_cache_token()
        if token != self._cache_token:
            self._type_cache.clear()
            self._cache_token = token
        if key not in self._type_cache:
            ranked: list[tuple[tuple[int, int, int], int, int, _Registration[M]]] = []
            for registration in self._registrations:
                if not self._is_auto_dispatchable(registration):
                    continue
                distance = self._distance(registration.supported_types, key)
                if distance is None:
                    continue
                ranked.append(
                    (self._rank(registration), distance, registration.index, registration)
                )
            ranked.sort(key=lambda entry: entry[:3])
            self._type_cache[key] = [entry[3] for entry in ranked]
        return self._type_cache[key]

    @abstractmethod
    def _distance(self, supported_types: Any, key: Any) -> int | None:
        """How far ``key`` is from the closest declared type that admits it.

        ``None`` when no declared type admits the key. Smaller is more
        specific: the position of the admitting type in the argument class's
        method-resolution order, or the length of that order for a type that
        admits by ``issubclass`` without appearing in it, such as a registered
        virtual subclass of an abstract base class. A binary registry sums
        the two sides.
        """
        ...

    @abstractmethod
    def _format_key(self, key: Any) -> str:
        """``key`` as it appears in error messages, such as ``(Left, Right)``."""
        ...


def _is_tuple_of_classes(value: Any) -> bool:
    return isinstance(value, tuple) and all(isinstance(entry, type) for entry in value)


def _mro_distance(key: type, supported_types: tuple[type, ...]) -> int | None:
    """Distance from ``key`` to the closest of ``supported_types`` that admits it, or ``None``."""
    mro = key.__mro__
    distances = [
        mro.index(supported) if supported in mro else len(mro)
        for supported in supported_types
        if issubclass(key, supported)
    ]
    return min(distances, default=None)


class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]):
    """Dispatches on the type of the first positional argument."""

    def _cache_key(self, args: tuple[Any, ...]) -> type:
        if not args:
            raise TypeError(
                "UnaryDispatchRegistry requires a positional argument to dispatch on; got none"
            )
        return type(args[0])

    def _validate_supported_types(self, name: str, supported_types: Any) -> None:
        if not _is_tuple_of_classes(supported_types):
            raise TypeError(
                f"Method {name!r} must declare supported_types as a tuple of classes; "
                f"got {supported_types!r}"
            )

    def _distance(self, supported_types: tuple[type, ...], key: type) -> int | None:
        return _mro_distance(key, supported_types)

    def _format_key(self, key: type) -> str:
        return key.__name__


class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]):
    """Dispatches on the joint type of the first two positional arguments.

    A method matches when the left type is a subclass of one of its left
    types and the right type of one of its right types.
    """

    def _cache_key(self, args: tuple[Any, ...]) -> tuple[type, type]:
        if len(args) < 2:
            raise TypeError(
                "BinaryDispatchRegistry requires at least two positional "
                f"arguments; got {len(args)}"
            )
        return (type(args[0]), type(args[1]))

    def _validate_supported_types(self, name: str, supported_types: Any) -> None:
        if not (
            isinstance(supported_types, tuple)
            and len(supported_types) == 2
            and all(_is_tuple_of_classes(side) for side in supported_types)
        ):
            raise TypeError(
                f"Method {name!r} must declare supported_types as a (left_types, right_types) "
                f"pair of tuples of classes; got {supported_types!r}"
            )

    def _distance(
        self, supported_types: tuple[tuple[type, ...], tuple[type, ...]], key: tuple[type, type]
    ) -> int | None:
        supported_left, supported_right = supported_types
        left = _mro_distance(key[0], supported_left)
        right = _mro_distance(key[1], supported_right)
        if left is None or right is None:
            return None
        return left + right

    def _format_key(self, key: tuple[type, type]) -> str:
        return f"({key[0].__name__}, {key[1].__name__})"
