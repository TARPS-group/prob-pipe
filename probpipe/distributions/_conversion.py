"""Cross-type conversion: moving a law between representations.

A **converter** is a binary dispatch method declaring the source types it
converts from and the targets it converts to, and whether it is exact. The
**converter registry** is keyed on the source's type and the requested target,
which is a distribution class or a capability protocol. A conversion changes
the representation and nothing else: the result carries the source's event
declaration and realizes the same law up to the converter's fidelity.

The registry tests the source first, so a law that already satisfies the
target is returned as it is, with no converter selected. Otherwise it selects a
converter as every dispatch registry does, checks the converter's promise and
its result against the source's event declaration, and records the conversion
in the result's provenance.

Provides:
  - ``ConversionInfo`` – the report of one conversion, before it executes.
  - ``Converter`` – the base of every converter.
  - ``ConverterRegistry`` and its global instance ``converter_registry``.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..core._dispatch import (
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    Feasibility,
    MethodInfo,
    ResolutionError,
    _mro_distance,
    _Registration,
)
from ..core._record_spec import RecordSpec
from ..core._repr import sequence_repr
from ..core._spec_base import NumericArraySpec, TermSpec, _unify_array_shape
from ..core._specs import OutputSpec
from ..core.provenance import Provenance
from ..core.tracked import TrackedTerm
from ._capabilities import (
    _LAW_CAPABILITIES,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
)
from ._distribution import Distribution, DistributionSpec

__all__ = ["ConversionInfo", "Converter", "ConverterRegistry", "converter_registry"]


#: The method of each capability whose guard a conversion to that capability reads.
#: Each takes no argument other than values, so its guard is decided by the law alone.
_GUARDED_METHODS: dict[type, str] = {
    SupportsSampling: "_sample",
    SupportsUnnormalizedLogProb: "_unnormalized_log_prob",
    SupportsLogProb: "_log_prob",
    SupportsRandomUnnormalizedLogProb: "_random_unnormalized_log_prob",
    SupportsRandomLogProb: "_random_log_prob",
    SupportsMean: "_mean",
    SupportsVariance: "_variance",
    SupportsCovariance: "_cov",
    SupportsQuantile: "_quantile",
    SupportsExpectation: "_expectation",
}


@dataclass(frozen=True, repr=False)
class ConversionInfo(MethodInfo):
    """The report of one conversion, before it executes.

    The registry's ``check`` returns one in every state. A feasible report
    promises the declaration the result carries and states whether the
    conversion is exact and whether executing it draws from the source; an
    infeasible report leaves the promise at its defaults, which promise nothing.
    The exactness is the selected converter's declaration, which the registry
    fills in from the registration, so a converter states only its promise.

    Attributes
    ----------
    target_spec : DistributionSpec or None
        The declaration the result will carry, ``None`` unless feasible.
    target_class : type or None
        The representation class of the result, when known.
    capabilities : tuple of type
        The capability protocols guaranteed on the result.
    samples : bool
        Whether executing the conversion draws from the source, a
        workflow-owned random event.

    Raises
    ------
    TypeError
        If ``samples`` is not a ``bool``, or on a condition
        :class:`~probpipe.core._dispatch.Feasibility` rejects.
    ValueError
        If exactly one of ``method_name`` and ``exact`` is set on a report that
        selects a converter or is infeasible; if a feasible or unresolved report
        names no converter and is not exact; if a feasible report promises no
        ``target_spec``; or on a condition ``Feasibility`` rejects.

    Notes
    -----
    ``method_name`` is ``None`` when no converter is selected. That is either an
    infeasible report, whose ``description`` lists each converter tried, or the
    report of a source that already satisfies the target, which needs no
    conversion and is exact. The pending requirements of an unresolved report
    are those of :class:`~probpipe.core._dispatch.Feasibility`.
    """

    target_spec: DistributionSpec | None = None
    target_class: type | None = None
    capabilities: tuple[type, ...] = ()
    samples: bool = False

    def __post_init__(self) -> None:
        Feasibility.__post_init__(self)
        if type(self.samples) is not bool:
            raise TypeError(f"samples must be a bool; got {self.samples!r}")
        if self.method_name is not None or self.feasible is False:
            if (self.method_name is None) != (self.exact is None):
                raise ValueError("method_name and exact are set together or not at all")
        elif self.exact is not True:
            raise ValueError(
                "a feasible or unresolved ConversionInfo names its method, unless it selects "
                "none because the source already satisfies the target, which is exact"
            )
        if self.feasible is True and self.target_spec is None:
            raise ValueError("a feasible ConversionInfo promises its target_spec")

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The method's arguments, then the promise where one is made."""
        fields = super()._repr_arguments()
        if self.target_spec is not None:
            fields.append(("target_spec", repr(self.target_spec)))
        if self.target_class is not None:
            fields.append(("target_class", self.target_class.__name__))
        if self.capabilities:
            names = (capability.__name__ for capability in self.capabilities)
            fields.append(("capabilities", sequence_repr(names)))
        if self.samples:
            fields.append(("samples", "True"))
        return fields


class Converter(BinaryDispatchMethod):
    """A binary dispatch method that converts a law to a requested representation.

    A subclass declares ``name``, ``exact``, and the ``(source, target)`` types
    it supports, and implements a non-executing :meth:`check` and an
    :meth:`execute`. A target type is a distribution class or a capability
    protocol, and both methods receive the target the caller requested, which
    may be a base of a declared target. The registry passes the caller's
    converter options to both as keywords; a converter reads the options it
    names, and its ``execute`` refuses any other.
    """

    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        """The source types this converter reads and the target types it produces."""

    @abstractmethod
    def check(self, source: Any, target_type: type, **options: Any) -> ConversionInfo:
        """Promise the conversion of *source* to *target_type* without executing it.

        It never samples, fits, or evaluates a density, and it reports as
        pending whatever it cannot settle without the source's values.
        """

    @abstractmethod
    def execute(self, source: Any, target_type: type, **options: Any) -> Distribution:
        """Construct the converted law, carrying *source*'s event declaration."""


# ---------------------------------------------------------------------------
# What a target is, and whether a law satisfies it
# ---------------------------------------------------------------------------


def _is_capability(target: type) -> bool:
    """Whether *target* is a capability protocol rather than a distribution class."""
    return target in _LAW_CAPABILITIES or bool(vars(target).get("_is_protocol", False))


def _target_name(target: type) -> str:
    return getattr(target, "__name__", repr(target))


def _satisfaction(source: Any, target: type) -> Feasibility:
    """Whether the law *source* satisfies *target* as it is, its claim and guard included.

    A class target is satisfied by its instances. A capability is satisfied by a
    law that claims it and whose guard, for a capability whose method takes no
    argument other than values, is established. Only a ``Distribution`` can
    satisfy a target as it is, since only a law carries the declaration a
    conversion preserves.
    """
    if not isinstance(source, Distribution) or not isinstance(source, target):
        return Feasibility(
            False, f"a {type(source).__name__} is not a {_target_name(target)} as it is"
        )
    method = _GUARDED_METHODS.get(target)
    if method is None:
        return Feasibility(True)
    return _capability_guard(source, method)


def _guaranteed(capabilities: tuple[type, ...], target: type) -> bool:
    """Whether one of *capabilities* is *target* or refines it."""
    return any(target in getattr(capability, "__mro__", ()) for capability in capabilities)


# ---------------------------------------------------------------------------
# Event preservation
# ---------------------------------------------------------------------------


def _packaging(declaration: OutputSpec) -> str:
    if declaration.exposes_record:
        return "an exposed record"
    (component,) = declaration.components
    return f"the whole term {component!r}"


#: The names of the two declarations a conversion compares, in its messages.
_CONVERSION_SIDES = ("the source", "the result")


def _term_difference(
    expected: TermSpec | None,
    actual: TermSpec | None,
    path: str,
    sides: tuple[str, str] = _CONVERSION_SIDES,
    dtypes: bool = True,
) -> str | None:
    """How the term *actual* departs from *expected* at *path*, or ``None``.

    The kinds and shapes are compared, and with *dtypes* a set dtype of *actual*
    must cast to one *expected* sets by the same-kind rule, as a law matches a
    ``DistributionSpec``. Supports are not compared, since a family's own
    support replaces the source's. A type hole in *expected*, which a backend
    declaration may leave for the converter to fill, matches any term. *sides*
    names the two declarations in the message.
    """
    first, second = sides
    if expected is None:
        return None
    if isinstance(expected, NumericArraySpec) or isinstance(actual, NumericArraySpec):
        if not (isinstance(expected, NumericArraySpec) and isinstance(actual, NumericArraySpec)):
            return f"{path} is {_kind(expected)} in {first} and {_kind(actual)} in {second}"
        try:
            _unify_array_shape(expected.shape, actual.shape, {}, path)
        except ValueError as error:
            return f"{path} has shape {expected.shape} in {first}: {error}"
        if (
            dtypes
            and expected.dtype is not None
            and actual.dtype is not None
            and not np.can_cast(actual.dtype, expected.dtype, casting="same_kind")
        ):
            return (
                f"{path} has dtype {expected.dtype} in {first} and {actual.dtype} in {second}, "
                f"which does not cast to it"
            )
        return None
    if isinstance(expected, RecordSpec) or isinstance(actual, RecordSpec):
        if not (isinstance(expected, RecordSpec) and isinstance(actual, RecordSpec)):
            return f"{path} is {_kind(expected)} in {first} and {_kind(actual)} in {second}"
        if tuple(expected.children) != tuple(actual.children):
            return (
                f"{path} has the fields {list(expected.children)} in {first} and "
                f"{list(actual.children)} in {second}"
            )
        for name, child in expected.children.items():
            difference = _term_difference(
                child, actual.children[name], f"{path}/{name}", sides, dtypes
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path} is {_kind(expected)} in {first} and {_kind(actual)} in {second}"
    return None


def _kind(spec: TermSpec | None) -> str:
    if isinstance(spec, NumericArraySpec):
        return "an array"
    if isinstance(spec, RecordSpec):
        return "a record"
    return f"a {type(spec).__name__}" if spec is not None else "undeclared"


def _event_difference(
    expected: OutputSpec,
    actual: OutputSpec,
    sides: tuple[str, str] = _CONVERSION_SIDES,
    dtypes: bool = True,
) -> str | None:
    """How the declaration *actual* departs from *expected*, or ``None``.

    A conversion preserves the packaging, the component names, and each
    component's kind and shape, and a dtype it sets casts to the source's (see
    :func:`_term_difference`, which *dtypes* passes to). *sides* names the two
    declarations in the message, the source's and the result's by default.
    """
    first, second = sides
    if expected.exposes_record != actual.exposes_record:
        return f"{first} declares {_packaging(expected)} and {second} {_packaging(actual)}"
    if tuple(expected.components) != tuple(actual.components):
        return (
            f"{first} declares the components {list(expected.components)} and {second} "
            f"{list(actual.components)}"
        )
    for name, spec in expected.components.items():
        difference = _term_difference(spec, actual.components[name], name, sides, dtypes)
        if difference is not None:
            return difference
    return None


def _source_declaration(source: Any, options: dict[str, Any]) -> OutputSpec | None:
    """The event declaration a conversion of *source* preserves, or ``None`` when it has none.

    A law's is its own. A backend object carries none, and its declaration is
    the ``event_spec`` option when the caller gives one.
    """
    if isinstance(source, Distribution):
        return source.event_spec
    declared = options.get("event_spec")
    return declared if isinstance(declared, OutputSpec) else None


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Plan:
    """The selected converter and its checked report, before execution.

    ``awaiting`` names the method of a capability target whose guard can be
    decided only on the converted law, which execution checks once the law
    exists.
    """

    registration: _Registration[Converter] | None
    info: ConversionInfo
    awaiting: str | None = None


class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    """The binary dispatch registry of converters, keyed on the source type and the target.

    The target enters the key as itself, not as its type. A converter admits a
    requested target when a target it declares is that class or protocol or a
    subclass of it, so a converter producing a more specific representation or
    a refined capability serves the request.

    The registry's ``check`` and ``execute`` test the source first: a law that
    already satisfies the target, its claim and guard included, is reported
    feasible and exact with no converter selected, and ``execute`` returns it as
    it is. Otherwise a converter is selected as II.7 states, and the registry
    holds it to the conversion contract:

    1. its ``check`` returns a :class:`ConversionInfo`;
    2. a feasible promise carries the source's event declaration;
    3. for a capability target, the promised capabilities include the target or
       a refinement of it, and a guard of the target that only the converted
       law can decide leaves the report unresolved, pending that guard;
    4. the converted law carries the source's event declaration and satisfies
       the target, which settles a guard left pending.

    The result of a conversion records the source and the converter's name and
    exactness in its provenance.
    """

    def _cache_key(self, args: tuple[Any, ...]) -> tuple[type, type]:
        if len(args) < 2:
            raise TypeError(
                f"ConverterRegistry requires a source and a target type; got {len(args)} "
                f"positional arguments"
            )
        source, target = args[0], args[1]
        if not isinstance(target, type):
            raise TypeError(f"a conversion target is a class or a protocol, got {target!r}")
        return (type(source), target)

    def _validate_supported_types(self, name: str, supported_types: Any) -> None:
        """The pair of class tuples II.7 requires, each class one ``issubclass`` can check.

        Raises
        ------
        TypeError
            If *supported_types* is not a pair of tuples of classes, or it lists a
            protocol with a data member, for which ``issubclass`` raises.
        """
        super()._validate_supported_types(name, supported_types)
        for side in supported_types:
            for declared in side:
                try:
                    issubclass(object, declared)
                except TypeError:
                    raise TypeError(
                        f"Method {name!r} declares {declared.__name__}, which issubclass cannot "
                        f"check, so the registry cannot match it; a protocol with a data member "
                        f"belongs in the converter's check"
                    ) from None

    def _distance(
        self, supported_types: tuple[tuple[type, ...], tuple[type, ...]], key: tuple[type, type]
    ) -> int | None:
        supported_sources, supported_targets = supported_types
        source_distance = _mro_distance(key[0], supported_sources)
        target_distances = [
            distance
            for declared in supported_targets
            if (distance := _mro_distance(declared, (key[1],))) is not None
        ]
        if source_distance is None or not target_distances:
            return None
        return source_distance + min(target_distances)

    # -- the plan -----------------------------------------------------------

    def _promise(
        self,
        registration: _Registration[Converter],
        args: tuple[Any, ...],
        options: dict[str, Any],
    ) -> _Plan:
        """The converter's report under its registration, held to the conversion contract.

        Raises
        ------
        TypeError
            If the converter's ``check`` returns anything but a ``ConversionInfo``.
        ValueError
            If a feasible promise does not carry the source's event declaration.
        """
        report = registration.method.check(*args, **options)
        if not isinstance(report, ConversionInfo):
            raise TypeError(
                f"converter {registration.name!r} returned a {type(report).__name__} from check; "
                f"a converter's check returns a ConversionInfo"
            )
        info = replace(report, method_name=registration.name, exact=registration.exact)
        if info.feasible is not True:
            return _Plan(registration, info)
        source, target = args[0], args[1]
        declared = _source_declaration(source, options)
        if declared is not None and info.target_spec is not None:
            difference = _event_difference(declared, info.target_spec.event_spec)
            if difference is not None:
                raise ValueError(
                    f"converter {registration.name!r} promises a law that does not carry the "
                    f"source's event declaration: {difference}"
                )
        if not _is_capability(target):
            return _Plan(registration, info)
        if not _guaranteed(info.capabilities, target):
            promised = [_target_name(capability) for capability in info.capabilities]
            return _Plan(
                registration,
                replace(
                    info,
                    feasible=False,
                    description=(
                        f"it promises the capabilities {promised}, none of which is "
                        f"{_target_name(target)}"
                    ),
                ),
            )
        method = _GUARDED_METHODS.get(target)
        target_class = info.target_class
        if method is None or target_class is None:
            return _Plan(registration, info)
        guard = getattr(target_class, f"{method}_guard", None)
        if guard is None:
            return _Plan(registration, info)
        pending = f"{target_class.__name__}.{method}_guard of the converted law"
        return _Plan(registration, replace(info, feasible=None, pending=(pending,)), method)

    def _plan(
        self,
        args: tuple[Any, ...],
        method: str | None,
        exact_only: bool,
        options: dict[str, Any],
        *,
        listing: bool = True,
    ) -> _Plan:
        """The source as it is, or the selected converter and its report, executing nothing.

        With *listing*, the report of a key no converter admits lists the
        registered converters, as an execution's error does; a check's report
        leaves them out.
        """
        key = self._cache_key(args)
        source, target = args[0], args[1]
        as_is = _satisfaction(source, target)
        if as_is.feasible is not False:
            return _Plan(
                None,
                ConversionInfo(
                    as_is.feasible,
                    description=as_is.description,
                    pending=as_is.pending,
                    exact=True,
                    target_spec=source.spec,
                    target_class=type(source),
                    capabilities=tuple(
                        capability
                        for capability in _LAW_CAPABILITIES
                        if isinstance(source, capability)
                    ),
                ),
            )
        if method is not None:
            named = self._named(method)
            if not self._passes_exact_only(named, exact_only):
                return _Plan(
                    named,
                    ConversionInfo(
                        False,
                        description=(
                            f"Method {method!r} is approximate and exact_only was requested"
                        ),
                        method_name=method,
                        exact=named.exact,
                    ),
                )
            return self._promise(named, args, options)
        tried: list[str] = []
        for candidate in self._candidates(key, exact_only):
            plan = self._promise(candidate, args, options)
            if plan.info.feasible is not False:
                return plan
            tried.append(f"{candidate.name}: {plan.info.description or 'infeasible'}")
        return _Plan(
            None,
            ConversionInfo(
                False, description=self._no_method_message(key, tried, exact_only, listing=listing)
            ),
        )

    # -- the dispatch interface ---------------------------------------------

    def check(
        self,
        *args: Any,
        method: str | None = None,
        exact_only: bool = False,
        **options: Any,
    ) -> ConversionInfo:
        """Report how a conversion would run, without running anything.

        Parameters
        ----------
        *args : Any
            The source and the target.
        method : str or None
            A registered converter to probe instead of auto-selecting.
        exact_only : bool
            If ``True``, approximate converters are excluded.
        **options : Any
            Converter options, passed to each converter's ``check``.

        Returns
        -------
        ConversionInfo
            The report of the source as it is when it satisfies the target or
            its satisfaction is unresolved; otherwise that of the first
            converter in selection order that is not infeasible, or of the named
            one; otherwise an infeasible report listing each converter tried.

        Raises
        ------
        ResolutionError
            If *method* is not a registered name.
        TypeError
            If there are fewer than two positional arguments, the target is not
            a class, or a converter's ``check`` returns another report type.
        ValueError
            If a converter promises a law that does not carry the source's event
            declaration.
        """
        return self._plan(args, method, exact_only, dict(options), listing=False).info

    def execute(
        self,
        *args: Any,
        method: str | None = None,
        exact_only: bool = False,
        **options: Any,
    ) -> Any:
        """Convert the source to the target and return the converted law.

        A source that already satisfies the target is returned as it is.

        Raises
        ------
        ResolutionError
            If no converter is feasible, the first converter that is not
            infeasible is unresolved, *method* names a converter that is not
            registered, is infeasible, or is approximate while *exact_only* is
            set, or a guard of the target rejects the converted law.
        TypeError
            As :meth:`check` raises it, or if the converted law is not a
            ``Distribution`` that satisfies the target.
        ValueError
            If the promise or the converted law does not carry the source's
            event declaration.
        """
        options = dict(options)
        plan = self._plan(args, method, exact_only, options)
        info = plan.info
        if plan.registration is None and info.feasible is True:
            return args[0]
        if info.feasible is False:
            raise ResolutionError(
                info.description
                if plan.registration is None
                else f"Method {plan.registration.name!r} is not applicable: {info.description}"
            )
        if info.feasible is None and plan.awaiting is None:
            subject = (
                "the source's satisfaction of the target"
                if plan.registration is None
                else f"Method {plan.registration.name!r}"
            )
            raise ResolutionError(f"{subject} is unresolved; pending: {', '.join(info.pending)}")
        registration = plan.registration
        result = registration.method.execute(*args, **options)
        self._check_result(registration, args, options, result, plan.awaiting)
        if isinstance(result, TrackedTerm) and result.provenance is None:
            source = args[0]
            result.with_provenance(
                Provenance.create(
                    "convert",
                    parents=[source] if isinstance(source, TrackedTerm) else [],
                    metadata={"converter": registration.name, "exact": registration.exact},
                )
            )
        return result

    @staticmethod
    def _check_result(
        registration: _Registration[Converter],
        args: tuple[Any, ...],
        options: dict[str, Any],
        result: Any,
        awaiting: str | None,
    ) -> None:
        """Hold the converted law to the promise: the source's declaration, and the target.

        Raises
        ------
        TypeError
            If *result* is not a ``Distribution`` that is an instance of, or
            claims, the target.
        ValueError
            If *result* does not carry the source's event declaration.
        ResolutionError
            If the guard *awaiting* names rejects *result*, or leaves it unresolved.
        """
        source, target = args[0], args[1]
        name = registration.name
        if not isinstance(result, Distribution):
            raise TypeError(
                f"converter {name!r} returned a {type(result).__name__}; a conversion returns a "
                f"Distribution"
            )
        declared = _source_declaration(source, options)
        if declared is not None:
            difference = _event_difference(declared, result.event_spec)
            if difference is not None:
                raise ValueError(
                    f"converter {name!r} returned a law that does not carry the source's event "
                    f"declaration: {difference}"
                )
        if not isinstance(result, target):
            raise TypeError(
                f"converter {name!r} returned a {type(result).__name__}, which is not a "
                f"{_target_name(target)}"
            )
        if awaiting is not None:
            guard = _capability_guard(result, awaiting)
            if guard.feasible is not True:
                reason = guard.description or ", ".join(guard.pending)
                raise ResolutionError(
                    f"converter {name!r} returned a law whose {awaiting} the target "
                    f"{_target_name(target)} needs is not available: {reason}"
                )

    def convert(
        self,
        source: Any,
        target_type: type,
        *,
        method: str | None = None,
        exact_only: bool = False,
        **options: Any,
    ) -> Any:
        """Convert *source* to *target_type*, returning a law that satisfies it.

        Parameters
        ----------
        source : Distribution or a backend distribution
            The law to convert, or a backend object to bring into ProbPipe.
        target_type : type
            A distribution class or a capability protocol.
        method : str or None
            A registered converter to run instead of auto-selecting.
        exact_only : bool
            If ``True``, approximate converters are excluded.
        **options : Any
            Converter options, which the selected converter reads; a backend
            object's event declaration is the option ``event_spec``.

        Returns
        -------
        Distribution
            *source* itself when it already satisfies *target_type*, and
            otherwise the converted law.

        Raises
        ------
        ResolutionError
            If no converter is feasible under the controls, or *method* names one
            that is not registered or not applicable.
        TypeError
            If *target_type* is not a class, or a converter breaks the
            conversion contract in kind.
        ValueError
            If the promise or the converted law does not carry the source's
            event declaration.
        """
        return self.execute(source, target_type, method=method, exact_only=exact_only, **options)

    def is_distribution_type(self, obj: Any) -> bool:
        """Whether *obj* is a ProbPipe law or an object a registered converter reads.

        A backend distribution counts when a converter declares its type among
        its sources, so the function engine brings it into ProbPipe.
        """
        if isinstance(obj, Distribution):
            return True
        return any(
            isinstance(obj, source)
            for registration in self._registrations
            for source in registration.supported_types[0]
        )


converter_registry: ConverterRegistry = ConverterRegistry()
