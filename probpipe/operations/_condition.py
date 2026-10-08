"""The condition_on operation, its two stages, and the inference-method registry.

``condition_on(d, given)`` fixes fields of a distribution or of a conditional
distribution and returns the resulting law, normalized. ``given`` is
field-keyed, a ``Record`` or a mapping of field paths. Binding a given slot
applies the kernel, and binding a produced field conditions the law.

A call resolves in two stages. The **exact stage** computes the conditional: it
curries given slots, calls a conditioning capability, or, where no exact route
conditions a produced field, forms the **unnormalized conditional**, the law of
the unconditioned fields whose unnormalized log-density is the joint's at the
given values. The **normalization stage** returns a normalized result as it is,
and otherwise passes the result as the target to the inference-method registry,
whose selected method returns a normalized law. A law renamed at its boundary
passes the target it holds, so a renamed program keeps its methods, and the
normalized law takes the renamed paths. A kernel result is normalized per
value: it normalizes each law it yields once its last given is bound.

The routes, in selection order:

1. ``curry`` binds given slots of a kernel through its ``_condition_on``, which
   is exact unless the kernel claims ``SupportsApproximateConditioning``, and
   normalizes the result.
2. ``slice`` assembles the conditional from a factored law's factors when the
   conditioned fields are the whole event of factors upstream of the rest, or
   part of a law's that conditions on them exactly, and normalizes the result.
3. ``exact_conditioning`` calls ``_condition_on`` on a law claiming
   ``SupportsExactConditioning``.
4. ``inference_methods`` curries any given slots the given names, forms the
   unnormalized conditional of the produced fields by Bayes' rule, and
   normalizes it through the inference-method registry. A kernel normalized
   per value is curried and conditioned through the kernel whose laws it
   normalizes, so its result is normalized once.
5. ``approximate_conditioning`` calls ``_condition_on`` on a law claiming
   ``SupportsApproximateConditioning``.
6. ``unnormalized`` returns the exact stage's result, normalized or not; a caller
   selects it only by name, as ``method="unnormalized"``.

A route that normalizes through the registry is exact when its exact stage and
the selected method both are, so its exact methods rank with the exact routes
and its approximate methods with the approximate ones. No approximate route
therefore runs while an exact one applies, and ``exact_only`` excludes the
approximate conditioning capability and the approximate methods alike, which
raises ``ResolutionError`` for a call whose exact stage leaves an unnormalized
result. ``check`` reports both stages: its route is the exact stage's, and its
method the normalization's, which is ``None`` for a result that needs none.
Whether a result needs normalization is read from the declarations: a kernel's
conditional capabilities state whether its laws are normalized. ``check``
computes no exact stage, so it reports a curry unresolved when the kernel's
capabilities do not state it, and the call reads the computed law's own.

The inference methods' budgets, such as warmup lengths, are the entries of the
``method_options`` control. The routes that normalize pass them to the selected
method, or, when they curry a kernel normalized per value, to the method that
normalizes the law it yields; ``approximate_conditioning`` passes them to the
law's ``_condition_on`` as keyword options, and the other routes read none. The
method that runs validates them: an inference method refuses an entry it does
not read with ``TypeError``, naming the entries it reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar

from .._messages import unknown_names
from ..core._dispatch import (
    BaseDispatchRegistry,
    Feasibility,
    MethodInfo,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from ..core._record_batch import RecordBatch
from ..core._record_spec import RecordSpec
from ..core._repr import format_names, format_value, grouped_label, public_class_name
from ..core._spec_base import NumericSpec, OpaqueSpec, TermSpec, _full_array_shape_or_none
from ..core._specs import OutputSpec, _components_record
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import TrackedTerm
from ..distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsConditionalUnnormalizedLogProb,
    SupportsExactConditioning,
    SupportsUnnormalizedLogProb,
    _capability_guard,
    _capability_subclass,
    _guard_condition,
    _is_normalized,
    _kernel_is_normalized,
)
from ..distributions._conditional import ConditionalDistribution, ConditionalDistributionSpec
from ..distributions._distribution import Distribution, DistributionSpec, _fixes_every_field
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._factored import (
    FactoredDistribution,
    SupportsFactors,
    _bound_factor,
    _joined_label,
    _law_at_defaults,
)
from ..distributions._views import _RenamedDistribution
from ..functions._call import checking
from ..functions._resolution import PointReport
from ..values import FunctionSpec
from ._convert import convert
from ._operation import (
    BoundCall,
    RouteSource,
    _CheckedRoute,
    _RegistryRoute,
    _workflow_draws,
    operation,
)
from ._sample import _record_batch, _sample_result, _sample_shape, sample

__all__ = ["InferenceMethod", "condition_on", "inference_method_registry"]

_PATH_SEP = "/"

#: The name of the route that returns the exact stage's result.
_UNNORMALIZED = "unnormalized"


# ---------------------------------------------------------------------------
# The inference-method registry
# ---------------------------------------------------------------------------


class InferenceMethod(UnaryDispatchMethod):
    """Base class for registered inference methods; ``exact`` is ``False`` unless a method overrides it.

    A method normalizes the target of ``condition_on``'s normalization stage: it
    takes the target alone, a law whose data are already bound, and returns a
    normalized law over the target's event. A subclass declares ``name``,
    ``supported_types``, ``check``, and ``execute``, and overrides ``priority``
    to take part in automatic selection. Its ``check`` reads the interface it
    requires of the target, such as an unnormalized density, a backend program,
    or the joint and the given values to simulate from.

    A method validates the ``method_options`` entries it receives when it runs:
    a subclass names the entries its ``execute`` reads in ``_method_options``
    and calls :meth:`_check_options` before it computes anything. A built-in
    method reads no seed among them: each of its runs draws its key from a
    workflow-owned random event, so ``workflow_run(seed=...)`` reproduces the
    run.

    Notes
    -----
    A method is exact when its result is the conditional law itself, as the
    reweighted atoms of an empirical prior are, and such a method overrides
    ``exact``. A finite MCMC, SG-MCMC, slice, ABC, or variational output stands
    in for the conditional law, whatever its invariant target or asymptotic
    guarantee, so such a method keeps ``exact = False``; those guarantees are
    the method's own documentation.
    """

    #: The ``method_options`` entries the method reads; ``None`` names none, for a
    #: method that validates its entries itself.
    _method_options: ClassVar[tuple[str, ...] | None] = None

    @property
    def exact(self) -> bool:
        return False

    def _check_options(self, options: Mapping[str, Any]) -> None:
        """Refuse a ``method_options`` entry that the method does not read.

        A method whose ``_method_options`` is ``None`` admits every entry.

        Parameters
        ----------
        options : Mapping of str to Any
            The ``method_options`` entries the method received, by name.

        Raises
        ------
        TypeError
            Naming the method, the entries it does not read, and those it reads.
        """
        reads = self._method_options
        if reads is None:
            return
        unread = sorted(set(options) - set(reads))
        if unread:
            raise TypeError(
                f"inference method {self.name!r}: "
                f"{unknown_names('method option', unread, sorted(reads))}"
            )


#: The builder of the target of a model and its observed data, which
#: ``probpipe.inference`` installs, since the way each model binds its data is
#: that package's.
_observed_target: Callable[[Any, Any], Any] | None = None


def _install_observed_target(builder: Callable[[Any, Any], Any]) -> None:
    """Install the builder of the target of a model and the data it is conditioned on."""
    global _observed_target
    _observed_target = builder


class _InferenceMethodRegistry(UnaryDispatchRegistry[UnaryDispatchMethod]):
    """The inference-method registry, whose methods take the target of the normalization stage.

    A call with a second positional argument passes a model and the data it is
    conditioned on, as ``probpipe.condition_on`` does, and dispatches on the
    target the installed builder forms from them.
    """

    def check(
        self, *args: Any, method: str | None = None, exact_only: bool = False, **kwargs: Any
    ) -> MethodInfo:
        """The report of the method a call would run; see :meth:`UnaryDispatchRegistry.check`."""
        return super().check(*self._targets(args), method=method, exact_only=exact_only, **kwargs)

    def execute(
        self, *args: Any, method: str | None = None, exact_only: bool = False, **kwargs: Any
    ) -> Any:
        """The selected method's result; see :meth:`UnaryDispatchRegistry.execute`.

        The keyword options are the call's ``method_options``, which the
        selected method validates when it runs.

        Parameters
        ----------
        *args : Any
            The target of the normalization stage, or a model and the data it is
            conditioned on, from which the installed builder forms the target.
        method : str or None
            A registered method name to run instead of auto-selecting.
        exact_only : bool
            If ``True``, approximate methods are excluded.
        **kwargs : Any
            The keyword options, passed to the selected method's ``check`` and
            ``execute``.

        Returns
        -------
        Distribution
            The normalized law over the target's event that the selected method
            returns.

        Raises
        ------
        TypeError
            If the selected method refuses an option it does not read.
        """
        return super().execute(*self._targets(args), method=method, exact_only=exact_only, **kwargs)

    @staticmethod
    def _targets(args: tuple[Any, ...]) -> tuple[Any, ...]:
        """*args* with a model and its observed data replaced by their target.

        Parameters
        ----------
        args : tuple of Any
            The positional arguments of a registry call: a target, or a model and
            the data it is conditioned on.

        Returns
        -------
        tuple of Any
            A one-element tuple of the target when *args* holds two arguments, and
            *args* itself otherwise.

        Raises
        ------
        TypeError
            If a model and its data are passed before a builder is installed.
        """
        if len(args) != 2:
            return args
        if _observed_target is None:
            raise TypeError("conditioning a model on data requires probpipe.inference; import it")
        return (_observed_target(*args),)


inference_method_registry: UnaryDispatchRegistry[UnaryDispatchMethod] = _InferenceMethodRegistry()
"""The registry of the normalization stage of ``condition_on``, keyed on the target's type.

The methods of ``probpipe.inference`` register here.
"""


# ---------------------------------------------------------------------------
# The given
# ---------------------------------------------------------------------------


def _given_keys(given: Any) -> tuple[str, ...] | None:
    """The field paths *given* names: a Record's fields or a mapping's keys, or None for neither."""
    if isinstance(given, Record):
        return tuple(given.fields)
    if isinstance(given, Mapping):
        return tuple(given)
    return None


def _given_values(given: Any) -> dict[str, Any]:
    """The values *given* holds, keyed by the paths :func:`_given_keys` names."""
    if isinstance(given, Record):
        return dict(given.children)
    return dict(given)


def _head(path: str) -> str:
    """The first segment of *path*: a given slot or an event component."""
    return path.split(_PATH_SEP, 1)[0]


def _slots_of(d: Any) -> frozenset[str]:
    """The given slots of *d*, none for a law."""
    return frozenset(d.given_spec) if isinstance(d, ConditionalDistribution) else frozenset()


# ---------------------------------------------------------------------------
# The reasons a route gives
# ---------------------------------------------------------------------------


def _named(d: Any) -> str:
    """*d* as a message names it: its public class and its label, such as ``Normal 'mu'``."""
    return f"{public_class_name(type(d))} {d.label!r}"


def _given_kind(call: BoundCall) -> Feasibility | None:
    """Why the given names no field path: it is no Record or mapping, or it is empty."""
    given = call.operands["given"]
    keys = _given_keys(given)
    if keys is None:
        return Feasibility(
            False,
            f"given is not a Record or a mapping keyed by field path; "
            f"got {public_class_name(type(given))}",
        )
    if not keys:
        return Feasibility(False, "given is empty; name at least one field", actionable=True)
    return None


def _unknown_paths(d: Any, keys: Iterable[str]) -> list[str]:
    """The keys among *keys* that name neither a given slot of *d* nor a field it declares."""
    slots = _slots_of(d)
    return [key for key in keys if _head(key) not in slots and _spec_at(d.event_spec, key) is None]


def _unknown(d: Any, unknown: list[str]) -> Feasibility:
    """The actionable report that the keys *unknown* name nothing *d* declares."""
    fields = list(d.event_spec.components)
    slots = sorted(_slots_of(d))
    if not slots:
        return Feasibility(False, unknown_names("field", unknown, fields), actionable=True)
    head = (
        f"unknown given slot or field {unknown[0]!r}"
        if len(unknown) == 1
        else f"unknown given slots or fields {unknown}"
    )
    return Feasibility(False, f"{head}; given slots: {slots}, fields: {fields}", actionable=True)


def _nested_field(key: str) -> Feasibility:
    """The actionable report that conditioning on the nested field *key* is not supported."""
    return Feasibility(
        False,
        f"conditioning on the nested field {key!r} is not supported yet; condition on the "
        f"whole field {_head(key)!r}",
        actionable=True,
    )


def _part_of_a_slot(key: str) -> str:
    """The message that binding *key*, part of a structured given slot, is not supported."""
    return (
        f"binding part of the given slot {_head(key)!r} ({key!r}) is not supported yet; "
        f"pass a value for the whole slot"
    )


def _fixes_every(d: Any) -> Feasibility:
    """The actionable report that the given fixes every field of *d*."""
    return Feasibility(False, _fixes_every_field(d.label), actionable=True)


# ---------------------------------------------------------------------------
# The targets of the normalization stage
# ---------------------------------------------------------------------------


def _curried(kernel: ConditionalDistribution, given: Any, **options: Any) -> Any:
    """The law or kernel that *kernel* yields at *given*, its provenance recording the curry.

    A result that carries a record of its own keeps it.
    """
    result = kernel._condition_on(given, **options)
    if result is not kernel and result.provenance is None:
        result.with_provenance(
            Provenance.create(
                "condition_on", parents=[kernel], metadata={"stage": "exact", "route": "curry"}
            )
        )
    return result


def _unconditioned_event(law: Any, produced: Iterable[str]) -> OutputSpec:
    """The declaration of *law*'s components that *produced* leaves unconditioned.

    It is an exposed record of those components, in the order *law* declares
    them.
    """
    record = _components_record(law.event_spec)
    conditioned = set(produced)
    return OutputSpec(
        RecordSpec(
            {name: spec for name, spec in record.children.items() if name not in conditioned}
        )
    )


def _joint_value(value: Any, given: Record) -> Record:
    """The joint's value at *value*, a draw of the unconditioned fields, and at *given*."""
    fields = value.children if isinstance(value, Record) else value
    return Record("value", {**dict(fields), **dict(given.children)})


def _conditional_density(self: _UnnormalizedConditional, value: Any) -> Any:
    """The joint's unnormalized log-density at *value* and the given values.

    Field-keyed given values join *value* into one record of the joint's fields.
    Data the joint does not declare as fields pair with *value* as
    ``(value, data)``, the form a density over parameters and data takes.
    """
    given = self.given
    joint_value = _joint_value(value, given) if self.keyed else (value, given)
    return self.joint._unnormalized_log_prob(joint_value)


class _UnnormalizedConditional(Distribution):
    """The unnormalized conditional: the law of a joint's unconditioned fields at given values.

    Its unnormalized log-density at a value of the unconditioned fields is the
    joint's at that value and the given values. It carries the joint and the
    given values, so a method that simulates rather than evaluates a density
    reads them. It claims ``SupportsUnnormalizedLogProb`` when the joint claims
    an unnormalized density, and no normalized capability, so it is
    unnormalized, and ``condition_on`` passes it to the inference-method
    registry as the target.

    Parameters
    ----------
    joint : Distribution
        The law conditioned.
    given : Record or Any
        The values of the conditioned fields, keyed by their components; or,
        when *keyed* is false, the data of a joint that does not declare them
        as fields, such as a model conditioned through ``probpipe.condition_on``.
    event_spec : OutputSpec
        The declaration of the unconditioned fields.
    keyed : bool
        Whether *given* is keyed by the joint's fields.
    """

    _capability_table: ClassVar = {
        SupportsUnnormalizedLogProb: {"_unnormalized_log_prob": _conditional_density},
    }

    def __new__(
        cls, joint: Distribution, given: Any, event_spec: OutputSpec, *, keyed: bool = True
    ) -> Any:
        claimed = (
            (SupportsUnnormalizedLogProb,) if isinstance(joint, SupportsUnnormalizedLogProb) else ()
        )
        return object.__new__(_capability_subclass(_UnnormalizedConditional, claimed))

    def __init__(
        self, joint: Distribution, given: Any, event_spec: OutputSpec, *, keyed: bool = True
    ) -> None:
        super().__init__(joint.label, event_spec)
        self._joint = joint
        self._given = given
        self._keyed = keyed
        self.with_provenance(
            Provenance.create(
                "condition_on",
                parents=[joint],
                metadata={"stage": "exact", "route": "inference_methods"},
            )
        )

    @property
    def joint(self) -> Distribution:
        """The law conditioned."""
        return self._joint

    @property
    def given(self) -> Any:
        """The values of the conditioned fields, or the data of a joint without such fields."""
        return self._given

    @property
    def keyed(self) -> bool:
        """Whether the given values are keyed by the joint's fields."""
        return self._keyed

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The law conditioned, and the fields its given values fix where they are keyed."""
        fields = [("joint", repr(self._joint))]
        if self._keyed:
            fields.append(("conditioned", format_names(self._given.keys())))
        return fields


def _conditional_kernel_density(
    self: _UnnormalizedConditionalKernel, given: Record | Mapping[str, Any], value: Any
) -> Any:
    """The kernel's unnormalized log-density at *given*, *value*, and the conditioned values."""
    return self.kernel._conditional_unnormalized_log_prob(given, _joint_value(value, self.given))


class _UnnormalizedConditionalKernel(ConditionalDistribution):
    """The unnormalized conditional within each slice of a kernel's unmet givens.

    Binding the given slots yields the unnormalized conditional of the law the
    kernel yields there, at the values of the conditioned fields, or a kernel
    over the slots left. It claims ``SupportsConditionalUnnormalizedLogProb``
    when the kernel does.

    Parameters
    ----------
    kernel : ConditionalDistribution
        The kernel conditioned.
    given : Record
        The values of the conditioned fields, keyed by their components.
    event_spec : OutputSpec
        The declaration of the unconditioned fields, an exposed record.
    """

    _capability_table: ClassVar = {
        SupportsConditionalUnnormalizedLogProb: {
            "_conditional_unnormalized_log_prob": _conditional_kernel_density
        },
    }

    def __new__(cls, kernel: ConditionalDistribution, given: Record, event_spec: OutputSpec) -> Any:
        claimed = (
            (SupportsConditionalUnnormalizedLogProb,)
            if isinstance(kernel, SupportsConditionalUnnormalizedLogProb)
            else ()
        )
        return object.__new__(_capability_subclass(_UnnormalizedConditionalKernel, claimed))

    def __init__(
        self, kernel: ConditionalDistribution, given: Record, event_spec: OutputSpec
    ) -> None:
        super().__init__(kernel.label, kernel.given_spec, event_spec)
        self._kernel = kernel
        self._given = given

    @property
    def kernel(self) -> ConditionalDistribution:
        """The kernel conditioned."""
        return self._kernel

    @property
    def given(self) -> Record:
        """The values of the conditioned fields."""
        return self._given

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The kernel conditioned, and the fields the given values fix."""
        return [("kernel", repr(self._kernel)), ("conditioned", format_names(self._given.keys()))]

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The unnormalized conditional at a value of every given slot, or a kernel over the rest."""
        return _unnormalized_conditional(_curried(self._kernel, given, **kwargs), self._given)


def _unnormalized_conditional(law: Any, given: Record) -> Any:
    """The unnormalized conditional of *law* at *given*, a kernel's within each slice.

    Conditioning an unnormalized conditional on further fields gives the
    unnormalized conditional of the same joint at every given value, so a
    method reads the joint and all the given values.
    """
    event_spec = _unconditioned_event(law, given.fields)
    if isinstance(law, _UnnormalizedConditionalKernel):
        return _UnnormalizedConditionalKernel(law.kernel, _joined(law.given, given), event_spec)
    if isinstance(law, _UnnormalizedConditional) and law.keyed:
        return _UnnormalizedConditional(law.joint, _joined(law.given, given), event_spec)
    if isinstance(law, ConditionalDistribution):
        return _UnnormalizedConditionalKernel(law, given, event_spec)
    return _UnnormalizedConditional(law, given, event_spec)


def _joined(first: Record, second: Record) -> Record:
    """The given values of *first* and *second* in one record."""
    return Record("given", {**dict(first.children), **dict(second.children)})


# ---------------------------------------------------------------------------
# The normalization stage
# ---------------------------------------------------------------------------


_NO_EXACT_METHOD = "the conditional is unnormalized and no exact inference method applies"
_UNNORMALIZED_FIX = 'use method="unnormalized" to get the unnormalized conditional'


def _packaged_alike(declared: OutputSpec, expected: OutputSpec) -> bool:
    """Whether *declared* packages its event as *expected* does, under the same components."""
    return declared.exposes_record == expected.exposes_record and tuple(
        declared.components
    ) == tuple(expected.components)


def _as_declared(source: Any, law: Any) -> EmpiricalDistribution:
    """The atoms and weights of the normalized *law* under *source*'s event declaration.

    The result is an ``EmpiricalDistribution`` carrying *law*'s provenance and
    annotations, since a posterior over a whole-term event draws a one-field
    record of it.
    """
    empirical = EmpiricalDistribution(
        source.label, law.atoms, law.weights, event_spec=source.event_spec
    )
    empirical._init_annotations(law.annotations)
    return empirical.with_provenance(law.provenance)


@dataclass(frozen=True)
class _Normalization:
    """How the normalization stage normalizes a target: the registry, the method, and its budgets.

    Attributes
    ----------
    registry : BaseDispatchRegistry
        The inference-method registry.
    method : str or None
        The method the caller named, or ``None`` for automatic selection.
    exact_only : bool
        Whether only exact methods may run.
    options : Mapping[str, Any]
        The method parameters the call sets.
    """

    registry: BaseDispatchRegistry[Any]
    method: str | None
    exact_only: bool
    options: Mapping[str, Any]

    def with_budgets(self, budgets: Mapping[str, Any]) -> _Normalization:
        """This normalization with the method parameters *budgets* set as well."""
        if not budgets:
            return self
        return replace(self, options=MappingProxyType({**self.options, **budgets}))

    def admits_an_exact_method(self) -> bool:
        """Whether an exact method may run: the named one, or one automatic selection reaches."""
        if self.method is not None:
            return self.registry.get_method(self.method).exact
        return any(
            method.exact and method.priority is not None
            for method in map(self.registry.get_method, self.registry.list_methods())
        )

    def report(self, target: Any) -> Feasibility:
        """The registry's report for normalizing the law *target*, or the target it holds.

        A report that no method applies names ``method="unnormalized"`` when
        ``exact_only`` excluded the approximate methods.
        """
        held, _ = _held_target(target)
        report = self.registry.check(
            held, method=self.method, exact_only=self.exact_only, **self.options
        )
        if report.feasible is False and self.exact_only:
            return replace(
                report,
                description=f"{_NO_EXACT_METHOD}; {_UNNORMALIZED_FIX} ({report.description})",
            )
        return report

    def normalize(self, law: Any) -> Any:
        """The selected method's normalized law for the target *law*.

        The result carries *law*'s event declaration, as the result rule of
        currying requires. A law renamed at its boundary is normalized through
        the target it holds, as :func:`_held_target` states.
        """
        held, renamed = _held_target(law)
        posterior = renamed(
            self.registry.execute(
                held, method=self.method, exact_only=self.exact_only, **self.options
            )
        )
        if _packaged_alike(posterior.event_spec, law.event_spec):
            return posterior
        return _as_declared(law, posterior)


def _held_target(target: Any) -> tuple[Any, Callable[[Any], Any]]:
    """The target the registry normalizes for *target*, and the map of its result to *target*'s.

    A law renamed at its boundary, or the unnormalized conditional of one at a
    given its parent's nodes can take, holds a target the inference methods
    recognize, such as a program's posterior. The registry normalizes that
    target, at the given translated to the parent's nodes, and the normalized
    law takes the renamed paths, since a renamed law conditions as the law it
    holds does (III.7). Any other target is its own.
    """
    if isinstance(target, _RenamedDistribution):
        held, renamed = _held_target(target._parent)
        return held, lambda law: target._event.law(renamed(law))
    if isinstance(target, _UnnormalizedConditional) and target.keyed:
        joint = target.joint
        if isinstance(joint, _RenamedDistribution):
            given = joint._original_given(target.given)
            if given is not None:
                held, renamed = _held_target(_unnormalized_conditional(joint._parent, given))
                return held, lambda law: joint._event.law(renamed(law))
    return target, _unchanged


def _unchanged(law: Any) -> Any:
    return law


def _per_value_sample(
    self: _PerValueNormalization,
    given: Record | Mapping[str, Any],
    key: Any,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """Draws of the normalized law at a value of every given slot."""
    return self._condition_on(given)._sample(key, sample_shape)


class _PerValueNormalization(ConditionalDistribution):
    """A kernel whose laws are another kernel's, each normalized once its last given is bound.

    It is the normalization stage's result for a kernel whose laws are
    unnormalized: binding every given slot yields the law the kernel yields
    there, normalized by the inference-method registry, and binding some yields
    another such kernel. Its laws sample, so it claims
    ``SupportsConditionalSampling``, and it claims
    ``SupportsApproximateConditioning`` unless only exact methods normalize it,
    since evaluating it then runs an approximate method. The budgets a binding
    passes are the method's, and they update those of the normalization.

    Parameters
    ----------
    kernel : ConditionalDistribution
        The kernel whose laws are normalized.
    normalization : _Normalization
        The registry, method, and budgets that normalize each law.
    """

    _capability_table: ClassVar = {SupportsApproximateConditioning: {}}

    def __new__(cls, kernel: ConditionalDistribution, normalization: _Normalization) -> Any:
        claimed = () if normalization.exact_only else (SupportsApproximateConditioning,)
        return object.__new__(_capability_subclass(_PerValueNormalization, claimed))

    def __init__(self, kernel: ConditionalDistribution, normalization: _Normalization) -> None:
        super().__init__(kernel.label, kernel.given_spec, kernel.event_spec)
        self._kernel = kernel
        self._normalization = normalization

    @property
    def kernel(self) -> ConditionalDistribution:
        """The kernel whose laws are normalized."""
        return self._kernel

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The kernel whose laws are normalized."""
        return [("kernel", repr(self._kernel))]

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **kwargs: Any
    ) -> Distribution | ConditionalDistribution:
        """The normalized law at a value of every given slot, or a kernel over the rest.

        *kwargs* are budgets of the method that normalizes the law.
        """
        result = _curried(self._kernel, given)
        if not _needs_normalization(result):
            return result
        return _normalized(result, self._normalization.with_budgets(kwargs))

    def _normalization_report(
        self, given: Record | Mapping[str, Any], budgets: Mapping[str, Any]
    ) -> Feasibility:
        """The registry's report for normalizing the law at a value of every given slot.

        It computes that law and runs no method.
        """
        law = _curried(self._kernel, given)
        if not _needs_normalization(law):
            return Feasibility(True)
        return self._normalization.with_budgets(budgets).report(law)

    def _condition_on_guard(self, paths: tuple[str, ...]) -> Feasibility:
        """Every path names a given slot, since a produced field is conditioned by Bayes' rule.

        The exact stage conditions a produced field through the kernel whose
        laws this kernel normalizes.
        """
        produced = [path for path in paths if _head(path) not in self.given_spec]
        if produced:
            return Feasibility(
                False,
                f"{self.label!r} conditions exactly only on its given slots "
                f"{sorted(self.given_spec)}, not on {produced}",
            )
        return Feasibility(True)

    _conditional_sample = _per_value_sample


def _laws_are_normalized(kernel: ConditionalDistribution) -> bool | None:
    """Whether *kernel*'s laws are normalized, as its declarations state, or None if they do not.

    A kernel that claims the conditional twin of a normalizing capability has
    normalized laws. One that claims the twin of an unnormalized density and no
    normalizing twin, or that forms unnormalized conditionals, has unnormalized
    laws. A kernel that claims neither states nothing about its laws.
    """
    if _kernel_is_normalized(kernel):
        return True
    if isinstance(kernel, (SupportsConditionalUnnormalizedLogProb, _UnnormalizedConditionalKernel)):
        return False
    return None


def _needs_normalization(result: Any) -> bool:
    """Whether the computed *result* needs the normalization stage, as its own declarations state.

    A law needs it unless it is normalized, and a kernel when its laws are
    unnormalized. A kernel that states nothing about its laws is returned as
    it is, and each law it yields is classified once a call binds it.
    """
    if isinstance(result, ConditionalDistribution):
        return _laws_are_normalized(result) is False
    return not _is_normalized(result)


def _normalized(result: Any, normalization: _Normalization) -> Any:
    """*result* normalized: a law by the registry's method, and a kernel per value."""
    if isinstance(result, ConditionalDistribution):
        return _PerValueNormalization(result, normalization)
    return normalization.normalize(result)


# ---------------------------------------------------------------------------
# The exact stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExactStage:
    """One way the exact stage computes the conditional.

    Attributes
    ----------
    check : callable
        ``check(call)``, whether the stage applies, read from the declarations.
    compute : callable
        ``compute(call)``, the stage's result.
    exact : callable
        ``exact(call)``, whether the result is the conditional itself rather
        than a stand-in for it.
    normalized : callable
        ``normalized(call)``, whether the result is normalized, read from the
        declarations, or ``None`` when they do not state it.
    yields_kernel : callable
        ``yields_kernel(call)``, whether the result is a kernel, since a given
        slot stays unbound.
    """

    check: Callable[[BoundCall], Feasibility]
    compute: Callable[[BoundCall], Any]
    exact: Callable[[BoundCall], bool]
    normalized: Callable[[BoundCall], bool | None]
    yields_kernel: Callable[[BoundCall], bool]


def _can_curry(call: BoundCall) -> Feasibility:
    """Every key of the given names a given slot of a conditional distribution.

    Parameters
    ----------
    call : BoundCall
        The bound call of ``condition_on``, whose operands ``d`` and ``given``
        the check reads.

    Returns
    -------
    Feasibility
        A feasible report, or a declined one whose description names the part
        of the condition that fails.

    Raises
    ------
    NotImplementedError
        If a key names part of a structured slot.
    """
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    if not isinstance(d, ConditionalDistribution):
        return Feasibility(False, f"{_named(d)} is not a ConditionalDistribution")
    kind = _given_kind(call)
    if kind is not None:
        return kind
    unknown = _unknown_paths(d, keys)
    if unknown:
        return _unknown(d, unknown)
    fields = [key for key in keys if _head(key) not in _slots_of(d)]
    if fields:
        return Feasibility(False, f"{fields} are fields of {d.label!r}, not given slots")
    for key in keys:
        if _PATH_SEP in key:
            raise NotImplementedError(f"condition_on: {_part_of_a_slot(key)}")
    return Feasibility(True)


def _curry(call: BoundCall) -> Any:
    """The kernel's ``_condition_on`` at the given slots.

    A kernel whose evaluation runs a method, one normalized per value or one
    that claims ``SupportsApproximateConditioning``, receives the call's
    ``method_options`` as the keyword options of ``_condition_on`` and
    validates them; evaluating any other kernel is exact and reads no budget.
    """
    kernel = call.operands["d"]
    options: dict[str, Any] = {}
    if isinstance(kernel, (_PerValueNormalization, SupportsApproximateConditioning)):
        options = dict(call.controls.get("method_options", {}))
    return _curried(kernel, call.operands["given"], **options)


def _evaluation_is_exact(call: BoundCall) -> bool:
    """Evaluating the kernel is exact unless it claims ``SupportsApproximateConditioning``."""
    return not isinstance(call.operands["d"], SupportsApproximateConditioning)


def _curried_is_normalized(call: BoundCall) -> bool | None:
    """Whether the kernel's laws are normalized, as its conditional capabilities declare."""
    return _laws_are_normalized(call.operands["d"])


def _leaves_a_slot(call: BoundCall) -> bool:
    """The given leaves a given slot of the conditioned object unbound."""
    bound = {_head(key) for key in _given_keys(call.operands["given"]) or ()}
    return bool(_slots_of(call.operands["d"]) - bound)


def _can_form_the_unnormalized_conditional(call: BoundCall) -> Feasibility:
    """The given names produced fields, each a component, and every other key is a given slot.

    At least one component must stay unconditioned, as the law of the
    unconditioned fields.
    """
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    kind = _given_kind(call)
    if kind is not None:
        return kind
    unknown = _unknown_paths(d, keys)
    if unknown:
        return _unknown(d, unknown)
    slots = _slots_of(d)
    produced = [key for key in keys if _head(key) not in slots]
    if not produced:
        return Feasibility(False, f"given binds only given slots of {d.label!r} and no field")
    components = set(d.event_spec.components)
    for key in produced:
        if key not in components:
            return _nested_field(key)
    for key in keys:
        if _head(key) in slots and _PATH_SEP in key:
            return Feasibility(False, _part_of_a_slot(key), actionable=True)
    if components <= set(produced):
        return _fixes_every(d)
    return Feasibility(True)


def _evaluated(d: Any) -> Any:
    """The object the exact stage evaluates for *d*.

    It is the kernel whose laws a per-value normalization normalizes, since
    those laws are the unnormalized form of its own, and *d* otherwise.
    """
    return d.kernel if isinstance(d, _PerValueNormalization) else d


def _bayes(call: BoundCall) -> Any:
    """The unnormalized conditional of the produced fields, the given slots curried first.

    A kernel normalized per value is curried and conditioned through the kernel
    it normalizes, so the result is normalized once.
    """
    d, values = call.operands["d"], _given_values(call.operands["given"])
    evaluated = _evaluated(d)
    slots = _slots_of(d)
    bound = {key: value for key, value in values.items() if _head(key) in slots}
    law = _curried(evaluated, bound) if bound else evaluated
    produced = {key: value for key, value in values.items() if key not in bound}
    return _unnormalized_conditional(law, Record("given", produced))


def _bayes_is_exact(call: BoundCall) -> bool:
    """Forming the unnormalized conditional is exact unless it curries an approximate kernel."""
    d = call.operands["d"]
    binds = any(_head(key) in _slots_of(d) for key in _given_keys(call.operands["given"]) or ())
    return not (binds and isinstance(_evaluated(d), SupportsApproximateConditioning))


def _always(call: BoundCall) -> bool:
    return True


def _never(call: BoundCall) -> bool:
    return False


def _slice_plan(d: Any, keys: tuple[str, ...]) -> tuple[Feasibility, dict[int, frozenset[str]]]:
    """Whether the slice applies to the factored law *d* at *keys*, and what it fixes in each factor.

    Parameters
    ----------
    d : Distribution
        A joint law that claims ``SupportsFactors``, read through its
        ``factors``.
    keys : tuple of str
        The field paths the given names.

    Returns
    -------
    tuple
        The report, and the components the given fixes in each factor it
        touches, by the factor's index.
    """
    factors = d.factors
    producer = {
        component: index
        for index, factor in enumerate(factors)
        for component in factor.event_spec.components
    }
    fixed: dict[int, set[str]] = {}
    unknown = [key for key in keys if key not in producer and _spec_at(d.event_spec, key) is None]
    if unknown:
        return _unknown(d, unknown), {}
    for key in keys:
        if key not in producer:
            return _nested_field(key), {}
        fixed.setdefault(producer[key], set()).add(key)
    whole = {
        index
        for index, components in fixed.items()
        if components == set(factors[index].event_spec.components)
    }
    if len(whole) == len(factors):
        return _fixes_every(d), {}
    for index, components in sorted(fixed.items()):
        factor = factors[index]
        slots = factor.given_spec if isinstance(factor, ConditionalDistribution) else {}
        free = sorted(
            component
            for parent in sorted({producer[slot] for slot in slots if slot in producer} - whole)
            for component in factors[parent].event_spec.components
        )
        if free:
            return Feasibility(
                False,
                f"the factor that defines {sorted(components)} conditions on {free}, which "
                f"given does not fix, so the conditional is not a product of the factors",
            ), {}
        if index in whole:
            continue
        others = sorted(set(factor.event_spec.components) - components)
        if not isinstance(factor, SupportsExactConditioning) or isinstance(
            factor, ConditionalDistribution
        ):
            return Feasibility(
                False,
                f"given fixes {sorted(components)} but not {others} of the same factor, which "
                f"cannot condition on part of its fields exactly",
            ), {}
        guard = _capability_guard(factor, "_condition_on", tuple(sorted(components)))
        if guard.feasible is not True:
            return guard, {}
    return Feasibility(True), {index: frozenset(components) for index, components in fixed.items()}


def _can_slice(call: BoundCall) -> Feasibility:
    """The given fixes whole factors upstream of every factor it touches, or part of a law's.

    Every key names a component of a factored law. A factor the given touches
    either has every component fixed, or is a law whose exact conditioning
    admits the fixed components, and every factor it conditions on has every
    component fixed, so the conditional is the product of the other factors at
    the fixed values. At least one component stays unconditioned.
    """
    d, keys = call.operands["d"], _given_keys(call.operands["given"])
    if isinstance(d, ConditionalDistribution):
        return Feasibility(False, f"{_named(d)} is a ConditionalDistribution, not a Distribution")
    if not isinstance(d, SupportsFactors):
        return Feasibility(False, f"{_named(d)} does not implement SupportsFactors")
    kind = _given_kind(call)
    if kind is not None:
        return kind
    return _slice_plan(d, keys)[0]


def _sliced_factors(call: BoundCall) -> list[tuple[Any, frozenset[str], dict[str, Any]]]:
    """Each factor the slice keeps, with the components it fixes there and the values it binds.

    A factor whose components are all fixed drops out, and a kernel binds the
    fixed components it conditions on.
    """
    d = call.operands["d"]
    values = _given_values(call.operands["given"])
    _, fixed = _slice_plan(d, tuple(values))
    kept = []
    for index, factor in enumerate(d.factors):
        components = fixed.get(index, frozenset())
        if components == set(factor.event_spec.components):
            continue
        slots = factor.given_spec if isinstance(factor, ConditionalDistribution) else {}
        bound = {slot: values[slot] for slot in slots if slot in values}
        kept.append((factor, components, bound))
    return kept


def _slice(call: BoundCall) -> Any:
    """The product of the factors the given leaves unconditioned, at the fixed values.

    A law fixed in part is replaced by its exact conditional, and a kernel is
    curried at the fixed components it conditions on, receiving the call's
    ``method_options`` where evaluating it runs a method, as in currying. One
    factor kept is the result itself.

    Parameters
    ----------
    call : BoundCall
        The bound call of ``condition_on``, whose ``d`` is a factored law and
        whose ``given`` fixes components of its factors.

    Returns
    -------
    Distribution
        A ``FactoredDistribution`` of the kept factors, or the one factor kept.
        A result without a provenance gains one that records the slice.

    Raises
    ------
    ValueError
        If a law's exact conditional does not produce the components the given
        leaves free in it.
    """
    d = call.operands["d"]
    values = _given_values(call.operands["given"])
    factors = []
    for factor, components, bound in _sliced_factors(call):
        if components:
            free = set(factor.event_spec.components) - components
            factor = factor._condition_on({key: values[key] for key in sorted(components)})
            if set(factor.event_spec.components) != free:
                raise ValueError(
                    f"the exact conditional of {d.label!r}'s factor on {sorted(components)} "
                    f"produces {sorted(factor.event_spec.components)}, not {sorted(free)}"
                )
        if bound:
            options: dict[str, Any] = {}
            if isinstance(factor, (_PerValueNormalization, SupportsApproximateConditioning)):
                options = dict(call.controls.get("method_options", {}))
            factor = _bound_factor(factor, bound, options)
        factors.append(factor)
    law = (
        _law_at_defaults(factors[0], ())
        if len(factors) == 1
        else FactoredDistribution(_joined_label(f.label for f in factors), factors)
    )
    if law.provenance is None:
        law.with_provenance(
            Provenance.create(
                "condition_on", parents=[d], metadata={"stage": "exact", "route": "slice"}
            )
        )
    return law


def _slice_is_exact(call: BoundCall) -> bool:
    """The slice is exact unless it curries a kernel that claims SupportsApproximateConditioning."""
    return not any(
        bound and isinstance(factor, SupportsApproximateConditioning)
        for factor, _, bound in _sliced_factors(call)
    )


def _sliced_is_normalized(call: BoundCall) -> bool | None:
    """Whether every factor the slice keeps is normalized, as the factors declare.

    A law's exact conditional is normalized, and a kernel's laws are as its
    conditional capabilities state.
    """
    states = []
    for factor, components, _ in _sliced_factors(call):
        if isinstance(factor, ConditionalDistribution):
            states.append(_laws_are_normalized(factor))
        else:
            states.append(True if components else _is_normalized(factor))
    if False in states:
        return False
    return True if all(states) else None


def _reads_shapes_from_data(law: Any) -> bool:
    """Whether *law*, or a law it views or renames, is a program that reads its shapes from its data."""
    while law is not None:
        if getattr(law, "_shapes_from_data", False):
            return True
        law = getattr(law, "_parent", None)
    return False


def _spec_at(declaration: Any, path: str) -> TermSpec | None:
    """The term spec *declaration* states at *path*, or None where it states none.

    The path's first segment names a component of an output declaration or a
    slot of an input declaration, and the rest addresses a node of its record.
    """
    head, _, rest = path.partition("/")
    parts = declaration.components if isinstance(declaration, OutputSpec) else declaration
    spec = parts.get(head) if isinstance(parts, Mapping) else None
    if spec is None or not rest:
        return spec
    try:
        return spec.at_path(rest)
    except (AttributeError, KeyError, TypeError):
        return None


def _given_conformance(call: BoundCall) -> Feasibility | None:
    """Each numeric given conforms to the numeric declaration at its path, a given slot's or a produced field's.

    A path the law declares nothing at is left to the stage's own check. A
    given that is no numeric array, such as a list of numbers, is checked by
    value when the call converts it. A program family reads its observed
    variables' shapes from the data it receives, so the program checks them.
    """
    d, given = call.operands["d"], call.operands["given"]
    if _reads_shapes_from_data(d) or _given_keys(given) is None:
        return None
    slots = d.given_spec if isinstance(d, ConditionalDistribution) else {}
    for path, value in _given_values(given).items():
        expected = _spec_at(slots, path)
        if expected is None:
            expected = _spec_at(d.event_spec, path)
        if not isinstance(expected, NumericSpec) or _full_array_shape_or_none(value) is None:
            continue
        try:
            expected._bind_dims_from_value(value, {}, repr(path))
        except ValueError as error:
            return Feasibility(
                False,
                f"the value given for {path!r} does not match its declaration: {error}",
                actionable=True,
            )
    return None


def _conforming(check: Callable[[BoundCall], Any]) -> Callable[[BoundCall], Any]:
    """*check*, preceded by :func:`_given_conformance`."""

    def checked(call: BoundCall) -> Any:
        mismatch = _given_conformance(call)
        return mismatch if mismatch is not None else check(call)

    checked.__doc__ = check.__doc__
    return checked


_CURRY = _ExactStage(
    check=_conforming(_can_curry),
    compute=_curry,
    exact=_evaluation_is_exact,
    normalized=_curried_is_normalized,
    yields_kernel=_leaves_a_slot,
)
_SLICE = _ExactStage(
    check=_conforming(_can_slice),
    compute=_slice,
    exact=_slice_is_exact,
    normalized=_sliced_is_normalized,
    yields_kernel=_leaves_a_slot,
)
_BAYES = _ExactStage(
    check=_conforming(_can_form_the_unnormalized_conditional),
    compute=_bayes,
    exact=_bayes_is_exact,
    normalized=_never,
    yields_kernel=_leaves_a_slot,
)


# ---------------------------------------------------------------------------
# The routes that normalize
# ---------------------------------------------------------------------------


class _NormalizingRoute(_RegistryRoute):
    """A route that runs one exact stage and normalizes its result through the registry.

    A result that is normalized is returned as it is, a law that is not is the
    target of the registry's selected method, and a kernel is normalized per
    value. Whether the result is normalized is read from the exact stage's
    declarations, and, where they do not state it, from the computed result's
    own: ``check`` computes nothing, so it reports such a route unresolved,
    while a call computes the result to decide. The route's exactness is that
    of its exact stage combined with the selected method's, so it is ranked
    twice, as every registry route is: its exact candidate applies when the
    result needs no approximate step, and its approximate candidate otherwise.
    The report of a normalized result names no method, and that of a
    normalized target names the method that normalizes it.
    """

    def __init__(
        self,
        name: str,
        *,
        source: RouteSource,
        stage: _ExactStage,
        registry: BaseDispatchRegistry[Any],
    ) -> None:
        super().__init__(name, registry=registry)
        self.source = source
        self._stage = stage

    @property
    def condition(self) -> str:
        """The exact stage's condition, and the normalization's."""
        stage = _guard_condition(self._stage.check)
        return (
            f"{stage}; a result that is not normalized is the target of a method the "
            f"registry selects among {', '.join(self.registry.list_methods()) or 'none'}"
        )

    def _normalization(self, call: BoundCall, method: str | None, exact_only: bool) -> Any:
        return _Normalization(
            self.registry, method, exact_only, MappingProxyType(self.method_options(call))
        )

    def probe(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Feasibility:
        """The exact stage's report, then the normalization's, under the controls.

        A probe runs no method. A check computes the exact stage's result only
        to read the target of the normalization, and only when the stage is
        exact, so it reports the route unresolved where the declarations do not
        state whether the result is normalized, or the target is the law an
        approximate kernel yields.
        """
        report = self._stage.check(call)
        if report.feasible is not True:
            return report
        exact = self._stage.exact(call)
        if exact_only and not exact:
            return Feasibility(
                False,
                "binding the given slots runs approximate conditioning "
                "(SupportsApproximateConditioning), which exact_only excludes",
            )
        result = None
        normalized = self._stage.normalized(call)
        if normalized is None:
            if checking():
                return Feasibility(
                    None,
                    pending=(
                        f"route {self.name!r}: {_named(call.operands['d'])} does not declare "
                        f"whether its laws are normalized, which only a call can tell",
                    ),
                )
            result = self._stage.compute(call)
            normalized = not _needs_normalization(result)
        if normalized:
            if method is not None:
                return Feasibility(
                    False,
                    f"the conditional is already normalized, so inference method {method!r} "
                    f"does not apply; drop method",
                    actionable=True,
                )
            d = call.operands["d"]
            if isinstance(d, _PerValueNormalization) and not self._stage.yields_kernel(call):
                return self._evaluation_report(call, d, exact)
            return PointReport(True, exact=exact)
        normalization = self._normalization(call, method, exact_only)
        if self._stage.yields_kernel(call):
            return self._per_value_report(call, normalization, exact)
        if result is None:
            if checking() and not exact:
                evaluated = _named(_evaluated(call.operands["d"]))
                return Feasibility(
                    None,
                    pending=(
                        f"route {self.name!r}: the law that the approximate {evaluated} gives "
                        f"at given, which only a call computes",
                    ),
                )
            result = self._stage.compute(call)
        info = normalization.report(result)
        if info.feasible is not True or exact or not isinstance(info, MethodInfo):
            return info
        return replace(info, exact=False)

    def _evaluation_report(
        self, call: BoundCall, kernel: _PerValueNormalization, exact: bool
    ) -> Feasibility:
        """The report of the method that binding every slot of *kernel* runs, which runs nothing.

        A check computes the law the inner kernel yields only when that kernel
        is evaluated exactly.
        """
        if checking() and isinstance(kernel.kernel, SupportsApproximateConditioning):
            return Feasibility(
                None,
                pending=(
                    f"route {self.name!r}: the law that the approximate "
                    f"{_named(kernel.kernel)} gives at given, which only a call computes",
                ),
            )
        info = kernel._normalization_report(call.operands["given"], self.method_options(call))
        if not isinstance(info, MethodInfo):
            return info if info.feasible is not True else PointReport(True, exact=exact)
        if info.feasible is not True:
            return info
        return replace(info, exact=exact and info.exact)

    def _per_value_report(
        self, call: BoundCall, normalization: _Normalization, exact: bool
    ) -> Feasibility:
        """The report of normalizing a kernel per value, whose method is selected once bound.

        Its laws are exact only when the caller requires exact methods, so that
        report requires a registered exact method that may normalize them.
        """
        if normalization.exact_only and not call.controls["exact_only"]:
            return Feasibility(
                False,
                "the result is a ConditionalDistribution whose laws an inference method that "
                "may be approximate normalizes",
            )
        if normalization.exact_only and not normalization.admits_an_exact_method():
            return Feasibility(
                False,
                "the result is a ConditionalDistribution whose laws are unnormalized, and no "
                'exact inference method applies; use method="unnormalized" to get it '
                "unnormalized",
            )
        exact = exact and normalization.exact_only
        if normalization.method is not None:
            method = self.registry.get_method(normalization.method)
            return MethodInfo(True, method_name=normalization.method, exact=exact and method.exact)
        return PointReport(True, exact=exact)

    def run(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Any:
        """The exact stage's result, normalized as the declarations, or else its own, require."""
        result = self._stage.compute(call)
        normalized = self._stage.normalized(call)
        if normalized is None:
            normalized = not _needs_normalization(result)
        if normalized:
            return result
        return _normalized(result, self._normalization(call, method, exact_only))


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def _condition_on_result(d: TermSpec, given: TermSpec) -> OutputSpec | None:
    """Binding given slots declares the kernel's law, or a kernel over the slots left.

    The result exposes its event, so its components are the law's. Conditioning
    on a produced field leaves the declaration to the law the selected route
    returns.
    """
    if isinstance(d, ConditionalDistributionSpec) and isinstance(given, RecordSpec):
        keys = tuple(given.children)
        if keys and all(key in d.given_spec for key in keys):
            left = d.given_spec.without(*keys)
            if not left.required:
                return OutputSpec(DistributionSpec(d.event_spec))
            return OutputSpec(ConditionalDistributionSpec(left, d.event_spec))
    return None


def _conditioned_label(d: Any, given: Any) -> str:
    """The label of the law that conditioning *d* on *given* returns (II.4).

    Applying a kernel at given slots keeps the kernel's label. Fixing the whole
    events of factors upstream of the rest leaves the other factors at the
    given values, whose labels are joined, so ``condition_on(model, {"mu": 0.5})``
    for ``model = likelihood * prior`` is labeled ``likelihood``. Any other
    conditioning applies Bayes' rule, and its result is labeled by the
    expression of the law and the conditioned paths, as ``model | y``.
    """
    paths = _conditioned_paths(given)
    if paths:
        if isinstance(d, ConditionalDistribution) and {_head(path) for path in paths} <= _slots_of(
            d
        ):
            return d.label
        kept = _factors_left(d, paths)
        if kept is not None:
            return _joined_label(factor.label for factor in kept)
    else:
        paths = (given.label if isinstance(given, TrackedTerm) else format_value(given),)
    return f"{grouped_label(d.label)} | {', '.join(paths)}"


def _factors_left(d: Any, keys: tuple[str, ...]) -> list[Any] | None:
    """The factors of the joint *d* that fixing the whole events of others at *keys* leaves.

    Returns None unless the slice applies and fixes every factor it touches
    whole, so the result is the product of the other factors.
    """
    if not isinstance(d, SupportsFactors) or isinstance(d, ConditionalDistribution):
        return None
    report, fixed = _slice_plan(d, keys)
    factors = d.factors
    if report.feasible is not True or any(
        components != set(factors[index].event_spec.components)
        for index, components in fixed.items()
    ):
        return None
    return [factor for index, factor in enumerate(factors) if index not in fixed]


def _conditioned_paths(given: Any) -> tuple[str, ...] | None:
    """The paths *given* fixes: a value's, each element's of a batch, or each draw's of a law."""
    keys = _given_keys(given)
    if keys is not None:
        return keys
    if isinstance(given, RecordBatch):
        return tuple(given.element_spec.fields)
    if isinstance(given, Distribution):
        return tuple(given.event_spec.components)
    return None


#: The kinds a given is admitted at whole: every kind but a batch, so a batch of
#: givens is swept, one conditioned law per element (VI.11).
_GIVEN_KINDS: tuple[type[TermSpec], ...] = (
    NumericSpec,
    RecordSpec,
    OpaqueSpec,
    FunctionSpec,
    DistributionSpec,
    ConditionalDistributionSpec,
)


@operation(
    result=_condition_on_result,
    roles={"d": (DistributionSpec, ConditionalDistributionSpec), "given": _GIVEN_KINDS},
    label=_conditioned_label,
)
def condition_on(d: Distribution, given: Record | Mapping[str, Any]):
    """Fix fields of *d* at the values *given* holds, and return the resulting law, normalized.

    The exact stage curries given slots, calls a conditioning capability, or
    forms the unnormalized conditional of the produced fields, and the
    normalization stage passes an unnormalized result to the inference-method
    registry. ``with_options(method="unnormalized")`` returns the exact stage's
    result as it is, and ``with_options(exact_only=True)`` raises rather than
    normalize by an approximate method. A batch of givens is swept: each
    element is conditioned on as one given is, and the results form a batch on
    the givens' levels.

    Parameters
    ----------
    d : Distribution or ConditionalDistribution
        The law or kernel to condition.
    given : Record, Mapping, or RecordBatch
        The values, keyed by field path: given slots of a kernel, or fields the
        law produces; or a batch of such values.

    Returns
    -------
    Distribution, ConditionalDistribution, or DistributionBatch
        The conditional, normalized: an ordinary law once every given slot is
        bound, and otherwise a kernel over the slots left whose laws are
        normalized; for a batch of givens, the batch of the conditionals.

    Raises
    ------
    ApplicabilityError
        If *d* is neither distribution kind.
    ResolutionError
        If no route applies under the controls, including an exact stage whose
        result is unnormalized under ``exact_only``.
    """


def _given_paths(d: Any, given: Any) -> tuple[str, ...]:
    """The paths *given* binds: its keys, or every component when the value is a whole draw."""
    keys = _given_keys(given)
    return keys if keys is not None else tuple(d.event_spec.components)


def _conditioning_guard(call: BoundCall, result: OutputSpec | None) -> Any:
    """The law's conditioning guard admits the given's paths, where its class defines one."""
    d = call.operands["d"]
    return _capability_guard(d, "_condition_on", _given_paths(d, call.operands["given"]))


def _exact_stage_by_name(call: BoundCall, result: OutputSpec | None) -> Any:
    """Selected only by name, as ``method="unnormalized"``; the exact stage alone then runs.

    The exact stage is the first of currying, slicing, exact conditioning, and
    the unnormalized conditional that applies.
    """
    if call.controls["method"] != _UNNORMALIZED:
        return Feasibility(False, 'used only with method="unnormalized"')
    reports = []
    for name, stage in _NAMED_STAGES:
        report = stage.check(call)
        if report.feasible is not False:
            return report if report.feasible is None else PointReport(True, exact=stage.exact(call))
        reports.append((name, report))
    actionable = next((report for _, report in reports if report.actionable), None)
    if actionable is not None:
        return actionable
    return Feasibility(
        False, "; ".join(f"{name}: {report.description}" for name, report in reports)
    )


def _exact_stage_result(call: BoundCall, result: OutputSpec | None) -> Any:
    """The result of the exact stage that applies, normalized or not."""
    for _, stage in _NAMED_STAGES:
        if stage.check(call).feasible is True:
            return stage.compute(call)
    raise AssertionError("the exact stage ran with no stage that applies")


condition_on.register_route(
    _NormalizingRoute(
        "curry", source=RouteSource.STRUCTURAL, stage=_CURRY, registry=inference_method_registry
    )
)
condition_on.register_route(
    _NormalizingRoute(
        "slice", source=RouteSource.STRUCTURAL, stage=_SLICE, registry=inference_method_registry
    )
)
_exact_conditioning_route = condition_on.capability_route(
    "exact_conditioning",
    operand="d",
    protocol=SupportsExactConditioning,
    method="_condition_on",
    exact=True,
    check=_conditioning_guard,
)
condition_on.capability_route(
    "approximate_conditioning",
    operand="d",
    protocol=SupportsApproximateConditioning,
    method="_condition_on",
    exact=False,
    check=_conditioning_guard,
)
condition_on.register_route(
    _NormalizingRoute(
        "inference_methods",
        source=RouteSource.REGISTRY,
        stage=_BAYES,
        registry=inference_method_registry,
    )
)
condition_on.register_route(
    _CheckedRoute(
        _UNNORMALIZED,
        source=RouteSource.STRUCTURAL,
        check=_exact_stage_by_name,
        execute=_exact_stage_result,
        exact=None,
    )
)


def _can_condition_exactly(call: BoundCall) -> Feasibility:
    """The law claims ``SupportsExactConditioning``, and its guard admits the given's paths."""
    return _exact_conditioning_route.check(call, None)


def _condition_exactly(call: BoundCall) -> Any:
    """The law's ``_condition_on`` at the given, which returns the conditional law."""
    return call.operands["d"]._condition_on(call.operands["given"])


#: Exact conditioning as a way the exact stage computes the conditional; its
#: result is the conditional law, which is normalized.
_EXACT_CONDITIONING = _ExactStage(
    check=_can_condition_exactly,
    compute=_condition_exactly,
    exact=_always,
    normalized=_always,
    yields_kernel=_leaves_a_slot,
)

#: The ways the exact stage computes the conditional, in the order it tries them,
#: each under the name of the route that runs it.
_NAMED_STAGES: tuple[tuple[str, _ExactStage], ...] = (
    ("curry", _CURRY),
    ("slice", _SLICE),
    ("exact_conditioning", _EXACT_CONDITIONING),
    ("inference_methods", _BAYES),
)


# ---------------------------------------------------------------------------
# The draws and conversions of an unnormalized law
# ---------------------------------------------------------------------------


class _NormalizingThen(_RegistryRoute):
    """A route of ``sample`` or ``convert`` that normalizes an unnormalized law, then answers.

    The law is the target of a method the inference-method registry selects,
    as in ``condition_on``'s normalization stage, and the call is answered on
    the normalized result. The route's exactness is the selected method's, and
    the selected method reads the call's ``method_options``.
    """

    def __init__(
        self,
        name: str,
        *,
        admits: Callable[[BoundCall], Feasibility],
        answer: Callable[[BoundCall, Any], Any],
    ) -> None:
        super().__init__(name, registry=inference_method_registry)
        self._admits = admits
        self._answer = answer

    @property
    def condition(self) -> str:
        """The law is unnormalized, the call is admitted, and a registered method applies."""
        return (
            f"The law is unnormalized, and a method the registry selects among "
            f"{', '.join(self.registry.list_methods()) or 'none'} normalizes it. "
            f"{_guard_condition(self._admits)}"
        )

    def probe(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Feasibility:
        """Whether the law is unnormalized and admitted, then the registry's report for it.

        The registry reports on the target the law holds, as :func:`_held_target`
        states.
        """
        law = call.operands["d"]
        if not isinstance(law, Distribution):
            return Feasibility(False, f"{_named(law)} is not a Distribution")
        if _is_normalized(law):
            return Feasibility(False, f"{_named(law)} is already normalized")
        admitted = self._admits(call)
        if admitted.feasible is not True:
            return admitted
        held, _ = _held_target(law)
        return self.registry.check(
            held, method=method, exact_only=exact_only, **self.method_options(call)
        )

    def run(self, call: BoundCall, *, method: str | None, exact_only: bool) -> Any:
        """The answer on the law the selected method returns, under the law's own paths."""
        held, renamed = _held_target(call.operands["d"])
        normalized = renamed(
            self.registry.execute(
                held, method=method, exact_only=exact_only, **self.method_options(call)
            )
        )
        return self._answer(call, normalized)


def _every_sample_shape(call: BoundCall) -> Feasibility:
    """Every sample shape is admitted."""
    return Feasibility(True)


def _declared_as_source(call: BoundCall, law: Any) -> Any:
    """The normalized *law* under the source's event declaration.

    A law that declares the source's event is returned as it is; otherwise its
    atoms and weights are declared as the source's event.
    """
    source = call.operands["d"]
    if law.event_spec == source.event_spec:
        return law
    return _as_declared(source, law)


def _draws_of(call: BoundCall, law: Any) -> Any:
    """Draws of the normalized *law*, as ``sample`` draws them from a law that samples."""
    draws = _workflow_draws(
        _declared_as_source(call, law),
        _sample_shape(call.operands["sample_shape"]),
        operation_kind="sample",
        execution_mode="sampled",
    )
    result = _sample_result(call.operands["d"].spec, call.operands["sample_shape"])
    return _record_batch(draws, call, result)


def _an_empirical_target(call: BoundCall) -> Feasibility:
    """The target is an empirical class, or a capability protocol an empirical law claims."""
    target = call.operands["target"]
    if issubclass(EmpiricalDistribution, target):
        return Feasibility(True)
    relation = "does not support" if getattr(target, "_is_protocol", False) else "is not a"
    return Feasibility(
        False, f"normalizing gives an EmpiricalDistribution, which {relation} {target.__name__}"
    )


def _empirical_of(call: BoundCall, law: Any) -> Any:
    """The normalized *law* at the target, carrying the source's event declaration.

    Parameters
    ----------
    call : BoundCall
        The bound call of ``convert``, whose ``d`` is the source law and whose
        ``target`` is the class or protocol requested.
    law : Distribution
        The normalized law that the selected inference method returned, with
        ``atoms`` and ``weights``.

    Returns
    -------
    Distribution
        The first of *law* and *law* under the source's declaration that is an
        instance of the target and declares the source's event. Otherwise, an
        ``EmpiricalDistribution`` of *law*'s atoms and weights, under the
        source's label and with *law*'s annotations.

    Raises
    ------
    TypeError
        If the normalized law under the source's declaration is not the target.
    """
    source, target = call.operands["d"], call.operands["target"]
    for candidate in (law, _declared_as_source(call, law)):
        if isinstance(candidate, target) and candidate.event_spec == source.event_spec:
            return candidate
    empirical = EmpiricalDistribution(
        source.label, law.atoms, law.weights, event_spec=source.event_spec
    )
    empirical._init_annotations(law.annotations)
    if not isinstance(empirical, target):
        raise TypeError(
            f"cannot convert {source.label!r} to {target.__name__}: its normalized law is an "
            f"EmpiricalDistribution"
        )
    return empirical


sample.register_route(_NormalizingThen("normalize", admits=_every_sample_shape, answer=_draws_of))
convert.register_route(
    _NormalizingThen("normalize", admits=_an_empirical_target, answer=_empirical_of)
)
