"""The distribution base class, its term spec, and minimal helpers.

Provides:
  - ``Distribution`` – Abstract base for all ProbPipe distributions.
  - ``NumericDistribution`` – The marker of a law whose event is numeric, with its views.
  - ``DistributionSpec`` – The term spec of the distribution kind.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Self

import jax
import jax.numpy as jnp

from ..core._specs import _has_record_outputs, _root_output_name

if TYPE_CHECKING:
    from ..core.constraints import Constraint
    from ..diagnostics.views import DiagnosticsView
    from ._batches import DistributionBatch
    from ._conditional import ConditionalDistribution
    from ._factored import FactoredConditionalDistribution, FactoredDistribution
    from ._views import _EventRenames

from ..core._execution_claims import Traceability, probe, spec_stand_in
from ..core._expression import _GENERIC_LAW_SYMBOL, Named, Signature
from ..core._immutable import transient_memo
from ..core._record_spec import RecordSpec, _layout_record_mapping
from ..core._repr import call_repr, format_names, public_class_name, term_repr, type_name
from ..core._shapes import PathsLike
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import OutputSpec, _paths_error, _read_paths
from ..core.constraints import _known_equal
from ..core.named_tree import _PATH_SEP
from ..core.provenance import Provenance
from ..core.record import Record
from ..core.tracked import Annotated, TrackedTerm, _as_raw_form, _is_raw_form, _TrackedTermMeta
from ._capabilities import SupportsSampling, SupportsUnnormalizedLogProb, _check_guards

# ---------------------------------------------------------------------------
# The event spec: completion and class membership
# ---------------------------------------------------------------------------


#: The label of a law or a kernel whose constructor is given none and that is not a family.
DEFAULT_LABEL = _GENERIC_LAW_SYMBOL


def _complete_event_spec(event_spec: Any) -> OutputSpec:
    """Complete *event_spec* into the output spec of one draw.

    Parameters
    ----------
    event_spec : OutputSpec or RecordSpec
        The event spec a constructor supplies. The fields of a ``RecordSpec``
        become the outputs, even when it has only one, and an ``OutputSpec`` is
        kept as given.

    Returns
    -------
    OutputSpec
        The complete event spec.

    Raises
    ------
    TypeError
        If *event_spec* is neither an ``OutputSpec`` nor a ``RecordSpec``, since
        an event of any other type is a whole term that an ``OutputSpec`` names.
    ValueError
        If the event spec leaves its type unspecified, since filling it in is
        the constructor's job.
    """
    if isinstance(event_spec, OutputSpec):
        declaration = event_spec
    elif isinstance(event_spec, RecordSpec):
        declaration = OutputSpec(event_spec)
    elif isinstance(event_spec, TermSpec):
        raise TypeError(
            f"an event of type {type(event_spec).__name__} needs an output. Declare it as "
            f"an OutputSpec whose keyword names the output, such as OutputSpec(mu=spec)"
        )
    else:
        raise TypeError(
            f"event_spec must be an OutputSpec or a RecordSpec, got {type(event_spec).__name__}"
        )
    if declaration.term_spec is None:
        raise ValueError(
            "event_spec does not declare a type; pass a full spec, such as NumericArraySpec(())"
        )
    return declaration


def _whole_term_event(
    output_name: Any, term: TermSpec, event_spec: OutputSpec | None, owner: str
) -> OutputSpec:
    """The event spec of a whole-term event *term* under *output_name*.

    Parameters
    ----------
    output_name : str
        The name of the event's output, which the constructor received first.
    term : TermSpec
        The type of one draw, which the constructor derived.
    event_spec : OutputSpec or None
        An event spec of the same output that also declares a type, which
        *term* completes as :meth:`OutputSpec.with_term_spec` does, or ``None``.
    owner : str
        The constructor, as error messages name it, such as ``"Normal"``.

    Returns
    -------
    OutputSpec
        The keyword-form event spec whose one output, named *output_name*,
        holds *term*, or *event_spec* completed with *term*.

    Raises
    ------
    TypeError
        If *output_name* is not a string, *event_spec* is not an ``OutputSpec``,
        or the outputs of *event_spec* are a record's fields.
    ValueError
        If *output_name* is not a valid output name, *event_spec* names
        another output, or *event_spec* declares a type that does not unify
        with *term*.
    """
    _check_output(output_name, owner)
    if event_spec is None:
        return OutputSpec.default(term, output_name=output_name)
    if not isinstance(event_spec, OutputSpec):
        raise TypeError(f"event_spec must be an OutputSpec, got {type(event_spec).__name__}")
    if not _has_record_outputs(event_spec) and tuple(event_spec.outputs) != (output_name,):
        raise ValueError(
            f"{owner} has the output {output_name!r}, but its event_spec names "
            f"{list(event_spec.outputs)}. Name the output once"
        )
    return event_spec.with_term_spec(term)


def _check_output(output_name: Any, owner: str) -> None:
    """Raise ``TypeError`` unless *output_name*, the first argument to a constructor, is a string.

    Parameters
    ----------
    output_name : Any
        The name of the event's output, which the constructor received first.
    owner : str
        The constructor, as the message names it, such as ``"Normal"``.

    Raises
    ------
    TypeError
        If *output_name* is not a string.
    """
    if not isinstance(output_name, str):
        raise TypeError(
            f"{owner} takes the name of the output of its event as its first argument, a "
            f"string such as 'mu', but got {type_name(output_name)}"
        )


def _class_label(term: Any) -> str:
    """The default label of a family's instance: the name of its public class, as ``Normal``."""
    return public_class_name(type(term))


def _given_label(label: Any, default: str | None = None, *, owner: str | None = None) -> str:
    """*label*, or *default* when *label* is ``None`` and the constructor has a default.

    Parameters
    ----------
    label : Any
        The label a constructor received.
    default : str or None, optional
        The constructor's default label, or ``None`` for a constructor that
        requires a label.
    owner : str or None, optional
        The constructor the message names, such as ``"conditional_distribution"``.

    Returns
    -------
    str
        *label*, or *default* for ``None``.

    Raises
    ------
    TypeError
        If *label* is not a non-empty string, and it is not ``None`` with a
        default to replace it.
    """
    if label is None and default is not None:
        return default
    if not isinstance(label, str) or not label:
        prefix = "" if owner is None else f"{owner}: "
        raise TypeError(f"{prefix}label must be a non-empty string, got {label!r}")
    return label


def _constructor_label(term: Any, label: Any, default: str) -> str:
    """*label*, or *default* when it is ``None``, recording *default* as *term*'s default label.

    The repr leaves out a label equal to the recorded default, since that
    label repeats what the class already states, as ``Normal('mu', ...)``
    does (:func:`_law_repr`).

    Parameters
    ----------
    term : Any
        The law or kernel under construction.
    label : Any
        The label its constructor received.
    default : str
        Its constructor's default label.

    Returns
    -------
    str
        *label*, or *default* for ``None``.

    Raises
    ------
    TypeError
        If *label* is neither ``None`` nor a non-empty string.
    """
    given = _given_label(label, default, owner=public_class_name(type(term)))
    object.__setattr__(term, "_default_label", default)
    object.__setattr__(term, "_uses_default_label", label is None)
    return given


def _record_default_expression(term: Any) -> None:
    """Record the expression *term* was just given as its default one, if its label is the default.

    The constructors of a law and of a kernel call this after setting the
    label, and a constructor that derives its expression from its arguments,
    as a mixture does, calls it again after storing that expression.
    :func:`_constructor_label` has marked a term whose constructor received no
    label, and :func:`_labeled_by_default` compares the expression the term
    carries with the one recorded here, so a copy, a pickle, and a rename of
    the term keep its status.
    """
    if getattr(term, "_uses_default_label", False):
        object.__setattr__(term, "_default_expression", term._expression)


def _not_an_event_path(
    event_spec: OutputSpec,
    label: str,
    path: str,
    requested: tuple[str, ...] = (),
    *,
    action: str | None = None,
) -> KeyError | TypeError:
    """The error for *path*, which is not an event path of the law *label* over *event_spec*.

    A law over a record lacks that path, which is a ``KeyError`` listing the
    paths it has; *requested* is the tuple of paths that held *path*, or
    ``()``. A law over any other value has no paths to look up, which is a
    ``TypeError`` whose message completes *action*, by default indexing the
    law. :func:`~probpipe.core._specs._paths_error` words both.
    """
    return _paths_error(
        event_spec,
        path,
        owner=repr(label),
        values_word="draws",
        action=action or f"index {label!r} by",
        requested=requested,
    )


def _read_event_paths(
    event_spec: OutputSpec,
    label: str,
    paths: Any,
    *,
    action: str,
    tuple_given: bool | None = None,
) -> str | tuple[str, ...]:
    """*paths*, read as event paths of the law *label* over *event_spec*.

    A law's paths are those of its event spec, so this is the reader of
    :meth:`OutputSpec.__getitem__ <probpipe.core._specs.OutputSpec.__getitem__>`,
    with messages that name the law and its draws. One path is returned as it
    is, and a list of paths in the event's canonical order, whatever order it
    was given in.

    Parameters
    ----------
    event_spec : OutputSpec
        The law's event spec.
    label : str
        The law's label, which the messages name.
    paths : Any
        One path, or a list or tuple of paths, as the caller gave them.
    action : str
        What the caller was asked to do, which the reader's messages complete.
    tuple_given : bool or None
        Whether the caller was given its list of paths as a tuple, which a
        missing path's message reads; ``None`` reads it from *paths*.

    Returns
    -------
    str or tuple of str
        The one path, or the list of paths in canonical order.

    Raises
    ------
    KeyError
        If the law is over a record and a path is not one of its paths.
    TypeError
        If *paths* is neither a string nor a list or tuple of strings, or the
        law is over a value with no paths, such as an array.
    ValueError
        If a path is empty or has an empty name, or a list of paths is empty,
        names a path twice, or holds a path inside another.
    """
    return _read_paths(
        event_spec,
        paths,
        owner=repr(label),
        values_word="draws",
        action=action,
        tuple_given=tuple_given,
    )


def _no_free_dims(term: Any, unbound: set[str], free: set[str] | frozenset[str]) -> str:
    """The message that *term* has no free dimensions *unbound* to bind, with those it has."""
    return (
        f"{public_class_name(type(term))} {term.label!r} has no free dimensions "
        f"{sorted(unbound)} to bind; its free dimensions: {sorted(free) or 'none'}"
    )


def _fixes_every_field(label: str) -> str:
    """The message that a given fixes every field of the law *label*, leaving none to infer."""
    return (
        f"given fixes every field of {label!r}, which leaves no field to infer; leave at least "
        f"one field out of given"
    )


def _labeled_by_default(term: Any) -> bool:
    """Whether *term* still carries the default label its constructor gave it, alone.

    Its expression is then the default label alone, so it records no label that
    a caller or an operation chose and no path held fixed. The default names
    the class, or ``ℙ``, so a law converted from *term* takes its converter's
    label, which names the target (VI.10). A default that the constructor
    derives from its arguments, as a mixture's ``mixture([a(x), b(x)])``,
    describes the law, so a converted law keeps it.
    """
    default = getattr(term, "_default_expression", None)
    return isinstance(default, Named) and term._expression is default


def _shows_label(term: Any, constructor: Any) -> bool:
    """Whether the repr of *term*, as a call of *constructor*'s class, shows the label.

    The call shows the label unless the constructor, given the arguments the
    call shows and no label, gives *term*'s label again. That label is the
    class's default, as ``Normal``, or one the constructor derives from its
    arguments, as a product joins its factors' labels as ``a·b``. A label that
    an operation carried over, as a converted or a conditioned law keeps its
    source's label, shows where it differs from that label.
    """
    return term.label != constructor._derived_label()


def _law_repr(term: Any, arguments: list[tuple[str, str]], constructor: Any = None) -> str:
    """The repr of the law or kernel *term*: a call of its public constructor, then its fixed paths.

    The call holds, in order:

    1. the name of a whole-term event's output, first where the constructor
       takes it first, as ``Normal('mu', ...)``;
    2. *arguments*, the constructor's other arguments in its order;
    3. that name as ``output_name=`` where the constructor takes it as a
       keyword, as an atoms-based law does;
    4. the label as ``label=`` where it differs from the label the
       constructor gives when it is given none, which the call gives again
       (:func:`_shows_label`);
    5. the event spec as ``event_spec=`` where it differs from the default
       for its outputs.

    The paths *term* holds fixed follow as ``fixed=('y',)``, so a conditioned
    law reads apart from an unconditioned one.

    Parameters
    ----------
    term : Any
        The law or kernel.
    arguments : list of (str, str)
        Each other argument's name and its formatted value.
    constructor : Any, optional
        The law or kernel whose class the call names and whose constructor's
        default label and placement of the output's name the call follows, as a
        law that presents another one names that one's class. It defaults to
        *term*.

    Returns
    -------
    str
        The repr, which an enclosing repr lays out again where its line starts.
    """
    constructor = term if constructor is None else constructor
    output_name, declaration = term._event_repr_parts()
    positional = []
    keywords = list(arguments)
    if output_name is not None and constructor._repr_output_name == "first":
        positional.append(repr(output_name))
    elif output_name is not None and constructor._repr_output_name == "keyword":
        keywords.append(("output_name", repr(output_name)))
    if _shows_label(term, constructor):
        keywords.append(("label", repr(term._displayed_label())))
    if declaration is not None:
        keywords.append(("event_spec", declaration))
    paths = _fixed_paths(term)
    if paths:
        keywords.append(("fixed", format_names(paths)))
    return call_repr(constructor._repr_class_name(), positional, keywords)


def _event_repr_parts(declaration: OutputSpec) -> tuple[str | None, str | None]:
    """The output name of the event spec *declaration*, and its repr where it is not the default.

    The keyword form has the name of its one output, and the positional form
    has none, since its outputs are the fields of its type. The event spec is
    the default when it is ``OutputSpec.default`` of its type under that name,
    or the positional form over its type, and the repr then leaves it out.
    """
    output_name = _root_output_name(declaration)
    spec = declaration.term_spec
    default = (
        OutputSpec(spec)
        if output_name is None
        else OutputSpec.default(spec, output_name=output_name)
    )
    return output_name, (None if declaration == default else repr(declaration))


def _declares_numeric_event(value: Any) -> bool:
    """Whether *value* is a law whose declared event is numeric.

    A law still under construction declares nothing yet, so it is not numeric.
    """
    if not isinstance(value, Distribution):
        return False
    declared = getattr(value, "_spec", None)
    return declared is not None and isinstance(declared.event_spec.term_spec, NumericSpec)


def _array_leaves(declaration: OutputSpec) -> dict[str, NumericArraySpec]:
    """Each array leaf of *declaration*, keyed by its path through the outputs."""
    leaves: dict[str, NumericArraySpec] = {}

    def visit(path: str, spec: TermSpec | None) -> None:
        if isinstance(spec, NumericArraySpec):
            leaves[path] = spec
        elif isinstance(spec, RecordSpec):
            for child, child_spec in spec.children.items():
                visit(f"{path}{_PATH_SEP}{child}", child_spec)

    for output_name, spec in declaration.outputs.items():
        visit(output_name, spec)
    return leaves


#: Each marker whose membership is read from an instance's declaration, with the
#: predicate that decides it and what an instance of a class inheriting the marker
#: must declare. ``NumericDistribution`` registers below, and the markers of the
#: conditional and factored kinds register in their own modules.
_DECLARATION_MARKERS: dict[type, tuple[Callable[[Any], bool], str]] = {}


def _check_marker_claims(instance: Any) -> None:
    """Raise ``TypeError`` if *instance*'s class inherits a marker its declaration fails."""
    claimant = type(instance)
    for marker, (holds, requirement) in _DECLARATION_MARKERS.items():
        if issubclass(claimant, marker) and not holds(instance):
            raise TypeError(
                f"{claimant.__name__} inherits {marker.__name__}, so its instances must "
                f"declare {requirement}"
            )


#: The engine behind ``&``, installed by the merge module at import.
_merge_engine: Callable[[Any, Any], Any] | None = None


def _install_merge(engine: Callable[[Any, Any], Any]) -> None:
    """Install the engine that ``&`` delegates to on both distribution kinds.

    Called once, by the merge module at import, so this module never
    imports the module that imports it.
    """
    global _merge_engine
    _merge_engine = engine


#: The law that ``rename`` returns, installed by the views module at import.
_renamed_law_factory: Callable[[Any, OutputSpec, Mapping[str, str]], Any] | None = None


def _install_renamed_law(factory: Callable[[Any, OutputSpec, Mapping[str, str]], Any]) -> None:
    """Install the factory of the law whose draws carry the names ``rename`` gives.

    Called once, by the views module at import, so this module never imports the
    module that imports it.
    """
    global _renamed_law_factory
    _renamed_law_factory = factory


#: The function that records the law a copy is made from, installed by the views
#: module at import.
_copy_source_recorder: Callable[[Any, Any, Mapping[str, str]], Any] | None = None


def _install_copy_source(recorder: Callable[[Any, Any, Mapping[str, str]], Any]) -> None:
    """Install the function that records the law a copy is made from, with the renames.

    Called once, by the views module at import, so this module never imports the
    module that imports it.
    """
    global _copy_source_recorder
    _copy_source_recorder = recorder


def _recorded_copy(copy: Any, law: Any) -> Any:
    """*copy*, which ``with_label`` or a dimension transform returns, recording *law* as its source."""
    if _copy_source_recorder is None:
        raise RuntimeError("the copy source is not installed; import probpipe")
    return _copy_source_recorder(copy, law, {})


#: The coupled marginal that indexing returns, installed by the views module at import.
_coupled_marginal_factory: Callable[[Any, Any], Any] | None = None


def _install_coupled_marginal(factory: Callable[[Any, Any], Any]) -> None:
    """Install the factory of the coupled marginal that ``d[path]`` returns.

    Called once, by the views module at import, so this module never imports the
    module that imports it.
    """
    global _coupled_marginal_factory
    _coupled_marginal_factory = factory


#: Indexing a ``DistributionBatch`` returns a copy of the stored law under a derived
#: label and sets this attribute of the copy to the stored law. A lift reads it to
#: draw two accesses of one element, or an element and its stored law, together
#: (V.5). Detaching the copy deletes the attribute.
_ELEMENT_SOURCE = "_element_source"

#: ``rename``, ``with_label``, ``with_dim_names``, and ``with_dim_sizes`` set
#: this attribute of each law they return that does not hold the law it is made from
#: as its parent. The value is the pair of that law and the renames from its
#: event spec to the result's, which are the identity for a relabeling and for a
#: dimension transform. A lift reads it to draw the result together with that law
#: (V.5). Detaching the result deletes the attribute.
_COPY_SOURCE = "_copy_source"


def _detached_term(term: Any) -> Any:
    """*term*, a law or a kernel, detached from the workflow under its own label.

    The copy shares the representation and keeps the paths *term* holds fixed,
    and it carries no provenance, no annotations, and no reference to a
    container or a parent, such as a batch it was an element of or a law it is
    a copy of.
    """
    clone = term._shallow_copy()
    object.__setattr__(clone, "_provenance", None)
    for workflow_state in ("_annotations", _ELEMENT_SOURCE, _COPY_SOURCE):
        clone.__dict__.pop(workflow_state, None)
    return clone


def _cached_raw(term: Any, build: Callable[[], Any]) -> Any:
    """*term*'s raw form, built by *build* on the first read and kept in its memo.

    Every read returns the same object. A raw form that is itself a term is
    marked as one (:func:`~probpipe.core.tracked._as_raw_form`), so it refuses
    provenance that one reader would otherwise show to every later reader, and
    the raw form of a raw form is itself. A failed build, such as a
    ``ResolutionError``, is not kept, so a later read tries again. The memo is
    transient, so a copy or an unpickled term rebuilds its raw form.

    Parameters
    ----------
    term : Distribution or ConditionalDistribution
        The law or kernel whose raw form is read.
    build : callable
        Builds the raw form from *term*, called at most once per memo.

    Returns
    -------
    Any
        The raw form, such as a standalone law or a backend object.
    """
    if _is_raw_form(term):
        return term
    memo = transient_memo(term)
    if "raw" not in memo:
        raw = build()
        memo["raw"] = _as_raw_form(raw) if isinstance(raw, TrackedTerm) else raw
    return memo["raw"]


def _fixed_paths(term: Any) -> tuple[str, ...]:
    """The paths the law or kernel *term* holds fixed at given values, in the order they were fixed.

    They are read from the term's expression, where a conditioning records the
    paths it fixes, and the signature lists them after ``;``.
    """
    return term._expression.fixed_paths()


def _keeps_fixed_paths(term: Any, source: Any) -> Any:
    """*term*, a law or kernel just derived from *source*, holding the paths *source* holds fixed.

    *term* keeps the paths it holds itself, followed by those of *source* it
    does not hold, which its expression records as a conditioning on them.
    *term* is set in place only when that adds a path, so a caller passes a
    term it has just built, or one that holds every path of *source* already.
    """
    expression = term._expression
    held = expression.with_fixed(_fixed_paths(source))
    if held is not expression:
        term._store_expression(held)
    return term


def _merge_operands(left: Any, right: Any) -> Any:
    """*left* ``&`` *right* through the installed engine.

    The engine returns ``NotImplemented`` for an operand that is neither
    distribution kind, so Python tries the reflected operation and then raises
    ``TypeError``.
    """
    if _merge_engine is None:
        raise RuntimeError("the merge engine is not installed. Import probpipe")
    return _merge_engine(left, right)


def _refuse_distribution_multiplication(other: object) -> Any:
    """Refuse multiplication by a law or a conditional distribution."""
    from ._conditional import ConditionalDistribution

    if isinstance(other, (Distribution, ConditionalDistribution)):
        raise TypeError("cannot multiply distributions with *. To merge them, write lik & prior")
    return NotImplemented


def _refuse_distribution_truth_value() -> NoReturn:
    """Refuse the truth value shared by laws and conditional distributions."""
    raise TypeError("a distribution has no truth value. To merge distributions, write a & b")


class _DistributionMeta(_TrackedTermMeta):
    """The metaclass of every distribution.

    Construction checks that the instance holds its event spec, as the
    tracked-term metaclass checks its label: a class that bypasses
    ``Distribution.__init__`` calls ``_init_declaration`` itself.

    Membership in a marker registered in ``_DECLARATION_MARKERS`` is read from an
    instance's declaration whatever its class, so ``isinstance(d,
    NumericDistribution)`` holds if and only if ``d`` declares a numeric event,
    and every other class check is the ordinary one. A class may claim a marker by
    inheriting it, and construction checks the claim. Creating a class checks
    each capability guard it defines (:func:`._capabilities._check_guards`).
    """

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        # The check runs once the class is complete, since a class that fails
        # while type.__new__ builds it has no ABC caches of its own and would
        # write into its base's.
        super().__init__(*args, **kwargs)
        _check_guards(cls)

    def __instancecheck__(cls, instance: Any) -> bool:
        marker = _DECLARATION_MARKERS.get(cls)
        if marker is not None:
            return marker[0](instance)
        return super().__instancecheck__(instance)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        # A factory __new__ may construct a subclass, whose declaration and claim
        # are the ones to check.
        claimant = type(instance)
        if not isinstance(getattr(instance, "_spec", None), DistributionSpec):
            raise TypeError(
                f"{claimant.__name__}.__init__ left the event undeclared; pass event_spec to "
                f"Distribution.__init__, or call _init_declaration when bypassing it"
            )
        _check_marker_claims(instance)
        return instance


# ---------------------------------------------------------------------------
# Distribution — the base class
# ---------------------------------------------------------------------------


class Distribution(TrackedTerm, Annotated, ABC, metaclass=_DistributionMeta):
    """
    Abstract base for all ProbPipe distributions.

    Every distribution is a tracked term: it is
    :class:`~probpipe.core.tracked.TrackedTerm` (a :attr:`~TrackedTerm.label` and a write-once
    :attr:`~TrackedTerm.provenance`) and
    :class:`~probpipe.core.tracked.Annotated` (free-form
    :attr:`~Annotated.annotations`).  A family's constructor takes the
    name of its event's output first and an optional ``label=``, which defaults to
    the family's class name, so ``Normal("mu", 0.0, 1.0)`` is labeled
    ``Normal`` over the output ``mu``; any other law's label defaults to
    ``ℙ``, and a joint that ``&`` merges is labeled by its operands' labels.
    Every transform preserves the label; only ``with_label`` replaces it. ``str(d)``
    returns the law's :attr:`notation`, its label followed by its signature, as
    ``prior(mu)``, and the repr reads as a call of the constructor, with
    ``label=`` where a caller gave a label other than the default, as
    ``Normal('mu', loc=0.0, scale=1.0, label='prior')``.

    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.SupportsSampling` protocol.

    **The event spec.** A law stores one ``DistributionSpec``, its
    :attr:`spec`, whose :attr:`event_spec` is the output spec of one
    draw. The fields of a bare ``RecordSpec`` are the outputs, and an
    ``OutputSpec`` names the output of a whole-term event, as
    ``OutputSpec(mu=spec)``. :attr:`event_shape` reads the event spec, and a
    law whose event spec is numeric also has the views of
    :class:`NumericDistribution`; none of them is stored.

    Parameters
    ----------
    event_spec : OutputSpec or RecordSpec
        The output spec of one draw, completed as above.
    label : str, optional
        The law's label, a non-empty string, ``ℙ`` by default. A subclass's
        constructor passes the label its caller gave, or its own default.
    _provenance : Provenance, optional
        The provenance of the law that a reconstruction rebuilds. By default the
        provenance stays unset until ``with_provenance`` attaches one.
    _annotations : Mapping[str, Any], optional
        The annotations of the law that a reconstruction rebuilds, copied into the
        law's own store. By default :attr:`annotations` is ``None``.

    Raises
    ------
    TypeError
        If *label* is not a non-empty string, or *event_spec* is neither an
        ``OutputSpec`` nor a ``RecordSpec``.
    ValueError
        If *event_spec* leaves its type unspecified.
    """

    #: The label the constructor gives when it is given none, which the repr leaves out;
    #: ``None`` for a class whose constructor requires a label.
    _default_label: ClassVar[str | None] = None

    #: The memo of values computed from the law on demand, such as its
    #: fingerprint, which a copy or a pickle leaves out.
    _transient_state = ("_memo",)

    #: Where the constructor takes the name of a whole-term event's output, which
    #: the repr follows: ``"first"`` as its first argument, ``"keyword"`` as
    #: ``output_name=`` after its other arguments, or ``None`` for a constructor
    #: that takes no output name.
    _repr_output_name: ClassVar[str | None] = "first"

    def __init__(
        self,
        event_spec: OutputSpec | TermSpec,
        *,
        label: str | None = None,
        _provenance: Provenance | None = None,
        _annotations: Mapping[str, Any] | None = None,
    ):
        label = _constructor_label(self, label, DEFAULT_LABEL) if label is None else label
        _given_label(label, owner=public_class_name(type(self)))
        # ``_provenance`` and ``_annotations`` carry state a reconstruction
        # already holds and that construction cannot otherwise reach: provenance
        # is write-once, and annotations are written after construction, so a
        # rebuilt distribution would come back without either. Private, and the
        # reconstruction paths are the only callers.
        self._init_tracked(label, provenance=_provenance)
        _record_default_expression(self)
        self._init_annotations(_annotations)
        self._init_declaration(event_spec)

    def _init_declaration(self, event_spec: OutputSpec | TermSpec) -> None:
        """Complete *event_spec* and store it as this law's event spec.

        The constructor calls this after setting the label; a class that bypasses
        the constructor calls it itself.
        """
        object.__setattr__(self, "_spec", DistributionSpec(_complete_event_spec(event_spec)))

    # -- the representation ---------------------------------------------------

    @property
    def raw(self) -> Distribution:
        """This law detached from the workflow, under its label and event spec.

        A law is represented by itself, so its raw form is a copy that shares
        its representation and carries no provenance, no annotations, and no
        reference to a container or a parent, such as a batch it was an element
        of or a law it is a copy of. A coupled marginal returns the raw form of its
        parent's marginal at its path instead, and a backend adapter its
        wrapped backend distribution.

        Every read returns the same copy, which refuses provenance, since every
        reader shares it. ``with_label`` gives a copy of it that can carry
        provenance.
        """
        return _cached_raw(self, lambda: _detached_term(self))

    # -- the event spec ----------------------------------------------

    @property
    def spec(self) -> DistributionSpec:
        """The law's term spec, the one stored source of its event spec."""
        return self._spec

    @property
    def event_spec(self) -> OutputSpec:
        """The output spec of one draw, read from :attr:`spec`."""
        return self.spec.event_spec

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The shape of one draw, defined only when a draw is a single array.

        For any other law the attribute is absent, so ``hasattr(law,
        "event_shape")`` is ``False``.

        Raises
        ------
        AttributeError
            If a draw is not a single array, a one-field record included.
        ValueError
            If the declared shape has unbound dimensions.
        """
        spec = self.event_spec.term_spec
        if not isinstance(spec, NumericArraySpec):
            raise AttributeError(
                f"{type(self).__name__} {self.label!r} does not draw a single array; "
                f"event_shape is defined only for one"
            )
        free = spec.free_dims
        if free:
            raise ValueError(
                f"{type(self).__name__} {self.label!r} has unbound dimensions "
                f"{sorted(free)}; bind them with with_dim_sizes"
            )
        return spec.shape

    def _traceability(self) -> Traceability:
        """Whether JAX can trace this law's density, or its sampler when it has none (V.13).

        It is traceable, embeddable, or host-only, and inference methods read it
        instead of tracing the law. The value is derived from the law's parts
        the first time it is read and kept for the law's life: a family is
        traceable, a law built from other laws or kernels takes the least
        permissive of theirs, and any other law is probed once, as
        :meth:`_derived_traceability` states.

        Returns
        -------
        Traceability
            The law's traceability.
        """
        memo = transient_memo(self)
        if "traceability" not in memo:
            memo["traceability"] = self._derived_traceability()
        return memo["traceability"]

    def _derived_traceability(self) -> Traceability:
        """The traceability of a law that states none: its density probed at one value of its event.

        A dimension not yet bound stands in with size one. A law without a
        density probes its sampler at one key instead, and one whose event has
        no stand-in, such as a value that is not an array, is host-only. A subclass whose methods are JAX code, or are built from
        parts that state their traceability, overrides this.

        Returns
        -------
        Traceability
            ``TRACEABLE`` when the probe traced, and ``HOST_ONLY`` otherwise.
        """
        return _probed_law_traceability(self)

    def __getattr__(self, name: str) -> Any:
        """Resolve a view of :class:`NumericDistribution` for a numeric law of another class.

        Python calls this only when ordinary lookup fails. A law whose event spec
        is numeric has the views of the marker whatever its class, so they resolve
        through the marker, and any other missing attribute raises as usual.

        Parameters
        ----------
        name : str
            The attribute that ordinary lookup did not find, such as ``"dtype"``.

        Returns
        -------
        Any
            The value of the marker's view *name* for this law.

        Raises
        ------
        AttributeError
            If *name* is not an attribute of the law, a view of the marker on a law
            whose event spec is not numeric included.
        """
        view = _NUMERIC_VIEWS.get(name)
        if view is None:
            # Repeat ordinary lookup, so the error it met stands, a property's own
            # message included.
            return object.__getattribute__(self, name)
        if not _declares_numeric_event(self):
            raise AttributeError(
                f"{name} is only available for a distribution with a numeric event, but "
                f"{public_class_name(type(self))} {self.label!r} has a non-numeric event"
            )
        return view.__get__(self, type(self))

    # -- copies ---------------------------------------------------------------

    def with_label(self, label: str) -> Self:
        """Return a copy of this law under a new label.

        The copy shares the representation and records the relabeling in its
        provenance, as :meth:`TrackedTerm.with_label` states. It is this law
        under a new label, so a lift draws it together with this law (V.5).

        Parameters
        ----------
        label : str
            The new label, a non-empty string.

        Returns
        -------
        Self
            A copy of the same class under *label*; the original is unchanged.

        Raises
        ------
        TypeError
            If *label* is not a non-empty string.
        """
        return _recorded_copy(super().with_label(label), self)

    # -- dimension transforms -------------------------------------------------

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Bind named symbolic dimensions of the event spec.

        The result is this law with its dimensions bound, so a lift draws it
        together with this law (V.5).

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of the event spec.

        Returns
        -------
        Self
            A copy of the same class and label whose event spec has the sizes
            substituted; the original is unchanged.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the event spec, one an earlier
            call bound included, or a size is negative.
        TypeError
            If a size is not an integer.
        """
        unbound = set(sizes) - self.event_spec.term_spec.free_dims
        if unbound:
            raise ValueError(_no_free_dims(self, unbound, self.event_spec.term_spec.free_dims))
        copy = self._with_declaration(
            self.event_spec.with_dim_sizes(**sizes), "with_dim_sizes", sizes
        )
        return _recorded_copy(copy, self)

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions of the event spec, simultaneously.

        The result is this law under the new dimension names, so a lift draws it
        together with this law (V.5).

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        Self
            A copy of the same class and label whose event spec has the
            dimensions renamed; the original is unchanged.
        """
        copy = self._with_declaration(
            self.event_spec.with_dim_names(**names), "with_dim_names", names
        )
        return _recorded_copy(copy, self)

    def rename(
        self, renames: Mapping[str, str] | None = None, /, **keyword_renames: str
    ) -> Distribution:
        """Rename or move nodes of the event spec by their paths, ``old -> new``.

        The result is this law with :meth:`OutputSpec.rename` applied to
        its event spec. A path starts with an output's name, and the form is
        kept, so a whole term's output is renamed in place with the term's
        fields under it. The law is unchanged: a draw of the result is a draw of
        this law carrying the new names. Renaming a whole term's output alone
        changes only the event spec, so the result is a copy of the same class.
        A factored law renames through its factors where they can carry the
        rename, and the result is the factored joint of the renamed factors over
        the same graph. A rename that gathers outputs under a new node
        regroups the factors that produce them into a packaged sub-joint, one
        factor of the result whose event is the node. A family whose parameters
        carry the event's paths rebuilds itself under the new paths: an
        empirical law over records returns the empirical law of its atoms with
        their fields at the new paths, under the same weights. Any other rename
        that changes the path of a field of a record draw, including a gathering
        whose groups condition on one another in a cycle, returns a law that
        holds this one and renames values at its boundary: draws, moments, and
        marginals on the way out, and scored values, givens, and paths on the
        way in. A lift draws the result together with this law, as it draws a
        view with its parent (V.5).

        Parameters
        ----------
        renames : Mapping[str, str], optional
            The new exact path of each node, keyed by the node's exact path.
        **keyword_renames : str
            Further renames, keyed by paths that are identifiers.

        Returns
        -------
        Distribution
            The renamed law under the same label; the original is unchanged.

        Raises
        ------
        KeyError
            If a key is not a path of the event spec.
        TypeError
            As :meth:`OutputSpec.rename` raises it.
        ValueError
            As :meth:`OutputSpec.rename` raises it.
        """
        renamed_event_spec = self.event_spec.rename(renames, **keyword_renames)
        all_renames = {**dict(renames or {}), **keyword_renames}
        if _renamed_law_factory is None:
            raise RuntimeError("the renamed law is not installed; import probpipe")
        return _renamed_law_factory(self, renamed_event_spec, all_renames)

    def _renamed_in_family(self, event: _EventRenames) -> Distribution | None:
        """The member of this law's family that holds its values under *event*'s new paths, or None.

        ``rename`` calls this for a rename that changes the path of a
        field of a record draw, and returns a law that renames this one's values
        at its boundary when it gives None. A family whose parameters carry the
        event's paths overrides it to return the member that declares
        ``event.renamed`` and whose draw at a key is this law's draw at that key
        under the new paths, as an empirical law over records does. The base
        class returns None.
        """
        return None

    def _with_declaration(
        self, event_spec: OutputSpec, operation: str, arguments: Mapping[str, Any]
    ) -> Self:
        """A copy of this law holding *event_spec*, with provenance recording *operation*."""
        clone = self._shallow_copy()
        object.__setattr__(clone, "_spec", DistributionSpec(event_spec))
        object.__setattr__(clone, "_provenance", None)
        clone.with_provenance(
            Provenance.create(operation, parents=[self], metadata=dict(arguments))
        )
        return clone

    # -- outputs -----------------------------------------------------------

    # Indexing addresses outputs, so the legacy sequence protocol must not make
    # a law iterable through it.
    __iter__ = None

    def __getitem__(self, paths: PathsLike, /) -> Distribution:
        """The law itself at a whole term's output, or the coupled marginal at a path or a list of paths.

        A whole-term law is itself under its output, so ``d[name]`` returns
        ``d``; the output is fixed at construction, so after ``with_label`` the
        law is still addressed by it. Any other event path, an output that is a
        record's field or a path below a whole record's output, gives the
        ``CoupledMarginal`` of the node there, which holds a reference to this law and
        gives the node whole under the path's last name. A list or a tuple of
        paths gives the view over the record that keeps each node at its own
        path, in this law's order, so ``grouped["a", "g/c"]`` draws records
        ``{a, g: {c}}``. A tuple is a list of paths; a nested path is written
        with ``/``.

        Parameters
        ----------
        paths : str, or list or tuple of str
            An event path, which starts with an output's name, or a list of event
            paths to view jointly.

        Returns
        -------
        Distribution
            This law itself, or a ``CoupledMarginal`` of it.

        Raises
        ------
        KeyError
            If this law is over a record and a path is not one of its paths.
        TypeError
            If *paths* is neither a string nor a list or tuple of strings, or
            this law is over a value with no paths, such as an array.
        ValueError
            If a path is empty or has an empty name, or a list of paths is
            empty, names a path twice, or holds a path inside another.
        """
        read_paths = _read_event_paths(
            self.event_spec, self.label, paths, action=f"index {self.label!r} by"
        )
        if read_paths == _root_output_name(self.event_spec):
            return self
        if _coupled_marginal_factory is None:
            raise RuntimeError("the coupled marginal is not installed. Import probpipe")
        return _coupled_marginal_factory(self, read_paths)

    def __contains__(self, path: object, /) -> bool:
        """Whether *path* is a ``str`` and ``self[path]`` gives a law: an event path, or a whole term's output."""
        return path in self.event_spec

    def __mul__(self, other: object) -> Any:
        """Refuse ``*`` between distributions. Use ``&`` to merge them.

        Parameters
        ----------
        other : object
            The right operand.

        Returns
        -------
        Any
            ``NotImplemented`` for an operand of another kind.

        Raises
        ------
        TypeError
            If *other* is a distribution or a conditional distribution.
        """
        return _refuse_distribution_multiplication(other)

    def __bool__(self) -> NoReturn:
        """Refuse a truth value, so ``a and b`` raises. Use ``a & b`` to merge.

        Raises
        ------
        TypeError
            Always: a distribution has no truth value.
        """
        _refuse_distribution_truth_value()

    # -- merge ------------------------------------------------------------

    def __and__(
        self, other: Distribution | ConditionalDistribution
    ) -> FactoredDistribution | FactoredConditionalDistribution:
        """The joint of this law and *other*, merged conditional-first.

        The left operand may condition on what the right produces, so ``lik &
        prior`` reads as ``p(y | β) · p(β)``. The result is a
        ``FactoredDistribution`` when no given is left unmet and a
        ``FactoredConditionalDistribution`` otherwise, flattened over the
        operands' factors. The joint is a product that no one gave a label, so
        its label joins the operands' labels with ``·``, as ``lik·prior``, and
        its notation joins its factors' notations, as ``lik(y | mu)·prior(mu)``.
        Its provenance records the operation ``&`` with this law and *other*
        as its parents, in that order and as written rather than flattened, so
        ``print(joint.provenance)`` includes the lineage of both operands. It
        records none where ``Provenance.create`` records none, as under the
        provenance mode ``off``.

        Parameters
        ----------
        other : Distribution or ConditionalDistribution
            The right operand, whose outputs this law's factors may condition
            on.

        Returns
        -------
        FactoredDistribution or FactoredConditionalDistribution
            The joint, or ``NotImplemented`` when *other* is neither
            distribution kind, so that Python raises ``TypeError``.

        Raises
        ------
        TypeError
            If an operand's draws are distributions or batches, which have no
            outputs to merge by name, until they are wrapped under a name.
        ValueError
            If the two operands are coupled, as two marginals of one law are,
            an output is produced twice, the right operand consumes an output
            the left produces, or matched specs do not unify.
        """
        return _merge_operands(self, other)

    # -- keyword-form value construction ------------------------------------

    def _pack_value(self, **field_kwargs: Any) -> Any:
        """Build a single draw of this distribution from
        named field kwargs — the adapter behind the keyword form of the
        log_prob-family ops (``log_prob(dist, field=value, ...)``).

        Delegates field validation and ``Record`` construction to the general
        :func:`~probpipe.core.record._pack_fields` and layers this
        distribution's value-type convention on top:

        * **single field** → the bare field value, an array, so a scalar
          distribution's ``_log_prob`` still receives a raw array.
        * **multiple fields** → the :class:`~probpipe.core.record.Record`
          built from the named fields.

        Distributions whose ``_log_prob`` consumes a Record but splits it
        internally (e.g. a factored joint, which scores each factor's fields)
        keep this default and do the split in ``_log_prob``. Override only when the
        value type is neither a bare array nor a flat Record (e.g.
        ``StanModel``'s single ``parameters=`` flat array).

        Builds exactly one draw (``sample_shape == ()``). Batched
        evaluation does not go through kwargs — pass the batch positionally
        and let ``Function`` broadcasting handle it.

        Parameters
        ----------
        **field_kwargs : Any
            One value per named field of a draw, keyed by field name.

        Returns
        -------
        Any
            The draw: the bare value of a single field, or the ``Record`` of
            several.

        Raises
        ------
        TypeError
            If the distribution has no named fields, or the kwargs do not
            match its fields exactly (missing or unexpected names).
        """
        from ..core.record import _pack_fields

        fields = getattr(self, "fields", None)
        if fields is None and _declares_numeric_event(self):
            # A numeric law outside the record laws names its fields by its outputs.
            fields = tuple(self.event_spec.outputs)
        if not fields:
            raise TypeError(
                f"{public_class_name(type(self))} {self.label!r} has no named fields, so it "
                f"cannot take the value by keyword; pass the value positionally"
            )
        rec = _pack_fields(fields, field_kwargs, owner=type(self).__name__)
        return field_kwargs[fields[0]] if len(fields) == 1 else rec

    # -- annotations ---------------------------------------------------------
    #
    # ``annotations`` (the general post-construction metadata store) is
    # provided by the :class:`~probpipe.core.tracked.Annotated` mixin.
    # On a fitted posterior the conventional layout is an ``xarray.DataTree``
    # with ``arviz/`` and ``diagnostics/`` subtrees; see :attr:`diagnostics`.

    @property
    def diagnostics(self) -> DiagnosticsView | None:
        """Structured view over diagnostic results stored in :attr:`annotations`.

        Returns ``None`` if no diagnostics have been computed yet.

        Inference backends and the diagnostics subsystem store posterior
        metadata in an ``xarray.DataTree`` attached to the distribution as
        its :attr:`~probpipe.core.tracked.Annotated.annotations`. The
        expected layout is::

            posterior._annotations
            ├── arviz/          # ArviZ-compatible data and raw inputs
            │   ├── posterior
            │   ├── sample_stats
            │   ├── observed_data
            │   ├── posterior_predictive
            │   └── log_likelihood
            └── diagnostics/    # ProbPipe-computed results and metadata
                ├── mcmc        # rhat, ess_bulk, ess_tail, mcse, ...
                └── runs        # on-demand diagnostics such as ppc, loo, spc
                    ├── ppc
                    ├── loo
                    └── spc

        The ``/arviz/`` subtree is intended to be passed to ArviZ functions
        and to hold raw diagnostic ingredients such as sampler statistics,
        posterior predictive samples, and pointwise log likelihoods. In ArviZ
        1.0+, this is an ArviZ-compatible ``DataTree`` rather than the older
        ``InferenceData`` representation.

        This property returns a structured Python accessor over the
        ``/diagnostics/`` subtree only. It is distinct from the ArviZ-compatible
        data used for plotting or ArviZ computations.

        In other words::

            posterior.diagnostics
                # structured ProbPipe view over posterior.annotations["diagnostics"]

            posterior.annotations["arviz"]
                # the ArviZ-compatible xarray DataTree

        Examples
        --------
        ::

            posterior = condition_on(model, data)

            # MCMC diagnostics mutate posterior._annotations in place and return None.
            add_mcmc_diagnostics(posterior)

            posterior.diagnostics.rhat
            # {"intercept": 1.001, "slope": 1.002}

            posterior.diagnostics.warnings
            # []

            posterior.diagnostics.runs
            # []

            # Posterior predictive checks are stored under diagnostics/runs/ppc.
            add_ppc(
                posterior,
                test_fns=[...],
                observed_data=y,
                kernel=likelihood,
            )

            posterior.diagnostics.ppc.result
            # {"var_mean_ratio": {"p_value": 0.43, "observed": 3.2}}

            posterior.diagnostics.runs[0].result
            # {"p_value": {"var_mean_ratio": 0.43}, ...}

            posterior.diagnostics.runs[0].plot_fn
            # "" unless the run wrote ArviZ-compatible plotting inputs

        Notes
        -----
        The diagnostics accessor is read-only. Diagnostic functions such as
        ``add_mcmc_diagnostics`` and ``add_ppc`` are responsible for writing
        diagnostic results into ``posterior._annotations``.
        """
        aux = self.annotations
        if aux is None:
            return None
        children = aux.children if hasattr(aux, "children") else {}
        if "diagnostics" not in children:
            return None
        from ..diagnostics.views import DiagnosticsView

        return DiagnosticsView(aux["diagnostics"])

    # -- batched construction -----------------------------------------------

    @classmethod
    def from_batched_params(
        cls,
        *,
        label: str,
        batch_shape: tuple[int, ...] | None = None,
        **batched_params: Any,
    ) -> DistributionBatch:
        """The separate laws of this class at batched parameters, one per batch position.

        Each parameter's leading axes are the batch axes, and the law at a
        position is this class at that position's parameters. The form of the
        result is not yet decided, so the method raises.

        Parameters
        ----------
        label : str
            The batch's label.
        batch_shape : tuple of int, optional
            The batch axes, inferred from the parameters when omitted.
        **batched_params
            This class's constructor arguments, with the batch axes leading.

        Returns
        -------
        DistributionBatch
            The batch of these laws under the label *label*, with the batch axes
            as its batch shape.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("from_batched_params is not implemented yet")

    # -- notation -----------------------------------------------------------

    @property
    def notation(self) -> str:
        """The law's label followed by its signature, as ``prior(mu)``, which ``str()`` returns.

        The signature lists the event outputs in declaration order, joined
        by ``", "``, then ``;`` and the paths the law holds fixed at given
        values when it holds any, as ``model(mu; y)``. The label is grouped so
        the call applies to all of it, as in ``(lik·prior)(y)``. A product
        without a label reads factor by factor, as ``lik(y | mu)·prior(mu)``,
        and the law of a function lifted over laws reads as the function at
        draws of its inputs, as ``f(beta ~ model; y)``. The notation is a
        rendering of the law's expression, made each time it is shown, which
        shows at most the current ``config.notation.max_depth`` nested levels.
        No operation reads the notation.
        """
        return self._expression.render_notation(self._own_signature(), warn=True)

    def _own_signature(self) -> Signature:
        """The signature the event spec states: the event outputs, in declaration order."""
        return Signature(tuple(self.event_spec.outputs))

    def __str__(self) -> str:
        """The law's :attr:`notation`, as ``prior(mu)``."""
        return self.notation

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        """A call of the law's public constructor, followed by the paths the law holds fixed.

        The name of a whole-term event's output and the family parameters come
        first, as the constructor takes them, then ``label=`` where the label
        differs from the constructor's default, as
        ``Normal('mu', loc=0.0, scale=1.0, label='prior')``, and
        ``event_spec=`` where the event spec differs from the default. The
        paths the law holds fixed follow as ``fixed=('y',)``.
        """
        return _law_repr(self, self._repr_arguments())

    def _repr_class_name(self) -> str:
        """The first public class in this law's method-resolution order, which the repr names."""
        return public_class_name(type(self))

    def _derived_label(self) -> str | None:
        """The label this law's constructor gives it when it is given no label, from the arguments its repr shows.

        It is the default label the constructor recorded, or ``None`` for a
        constructor that requires a label. A constructor that derives its label
        from its arguments, as a product does from its factors, overrides this.
        """
        return self._default_label

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The constructor's arguments the repr shows, besides output name, label, and event spec.

        Each is a name and a formatted value, in the constructor's order. The
        base class shows none.
        """
        return []

    def _event_repr_parts(self) -> tuple[str | None, str | None]:
        """The output name and the event spec's repr, as :func:`_event_repr_parts` gives them."""
        return _event_repr_parts(self.event_spec)


class NumericDistribution(Distribution):
    """The marker of a law whose event spec is numeric, with its views.

    ``isinstance(d, NumericDistribution)`` holds if and only if
    ``d.event_spec.term_spec`` is a :class:`~probpipe.core._spec_base.NumericSpec`,
    so a draw implements ``Numeric`` and the flat-vector interface applies. A
    class whose every instance is numeric may inherit the marker, and
    construction checks that each instance declares a numeric event.

    Every numeric law has the views below, whatever its class: a class that
    inherits the marker has them directly, and any other numeric law resolves
    them through the marker. A law whose event spec is not numeric has none of
    them. Like ``event_shape``, they read the event spec and are never stored.

    The marker is not a dispatch type. Registering a method for it raises
    ``TypeError``, since selection by class would miss a numeric law whose class
    does not inherit it.
    """

    # Read by dispatch registration, which refuses a class whose membership follows
    # an instance's declaration.
    _membership_follows_declaration = True

    @property
    def dtypes(self) -> dict[str, Any]:
        """The declared dtype of each array leaf of a draw, keyed by leaf path."""
        return {path: spec.dtype for path, spec in _array_leaves(self.event_spec).items()}

    @property
    def supports(self) -> dict[str, Constraint | None]:
        """The declared support of each array leaf of a draw, keyed by leaf path."""
        return {path: spec.support for path, spec in _array_leaves(self.event_spec).items()}

    @property
    def dtype(self) -> Any:
        """The dtype every array leaf shares, or None when they differ or there are none."""
        dtypes = set(self.dtypes.values())
        return dtypes.pop() if len(dtypes) == 1 else None

    @property
    def support(self) -> Constraint | None:
        """The support every array leaf shares, or None when they differ or there are none.

        ``supports`` tells leaves that differ from leaves that are unset. Supports
        whose comparison needs a traced value, as under ``jit``, count as different.
        """
        supports = list(self.supports.values())
        if supports and all(_known_equal(supports[0], s) for s in supports[1:]):
            return supports[0]
        return None


_DECLARATION_MARKERS[NumericDistribution] = (_declares_numeric_event, "a numeric event")


# The views a numeric law has whatever its class, which ``Distribution.__getattr__``
# resolves for a class that does not inherit the marker.
_NUMERIC_VIEWS: dict[str, property] = {
    name: vars(NumericDistribution)[name] for name in ("dtypes", "supports", "dtype", "support")
}


# ---------------------------------------------------------------------------
# DistributionSpec — the term spec of the distribution kind
# ---------------------------------------------------------------------------


def _event_form(output_name: str | None) -> str:
    """The form of an event, as a message names it: a record of fields, or one named field."""
    return "a record of fields" if output_name is None else f"the single field {output_name!r}"


def _unify_declarations(
    expected: OutputSpec, actual: OutputSpec, bindings: dict[str, int], path: str
) -> None:
    """Match *actual* against *expected*: form and outputs, then their specs.

    Parameters
    ----------
    expected : OutputSpec
        The event spec to match against, such as a term spec's ``event_spec``.
    actual : OutputSpec
        The event spec a law or a kernel carries.
    bindings : dict[str, int]
        The size of each symbolic dimension bound so far, keyed by its name, which
        unification extends in place.
    path : str
        The location of the event spec that error messages name, such as
        ``"the event spec"``.

    Raises
    ------
    ValueError
        If the form or the name of a whole term's output differs, or the specs do not
        unify in the shared scope *bindings*.
    """
    wanted, found = _root_output_name(expected), _root_output_name(actual)
    if (wanted is None) != (found is None):
        raise ValueError(
            f"{path} declares {_event_form(wanted)}, but the law declares {_event_form(found)}"
        )
    if wanted != found:
        raise ValueError(f"{path} declares the output {wanted!r}, but the law declares {found!r}")
    # A whole term's spec is bound under its output's path, as a record field's is.
    _unify_specs(
        expected.term_spec,
        actual.term_spec,
        bindings,
        path if wanted is None else f"{path}{_PATH_SEP}{wanted}",
    )


@dataclass(frozen=True, init=False)
class DistributionSpec(TermSpec):
    """The distribution kind's term spec: the output spec of one draw.

    Parameters
    ----------
    event_spec : OutputSpec or RecordSpec
        The output spec of one draw. A bare ``RecordSpec`` completes to the
        positional form over it.

    Raises
    ------
    TypeError
        If *event_spec* is neither, since a bare spec of another kind has no
        output name to complete it with.
    ValueError
        If *event_spec* leaves its type unspecified.

    Notes
    -----
    ``is_valid`` accepts a ``Distribution`` whose own event spec unifies with
    this one: the form and the output names agree, and then the outputs
    unify in one scope. An unset dtype accepts any dtype and a set one
    requires a same-kind cast, sizes agree with symbolic dimensions bound
    consistently, and support is not compared.

    ``bind_dims_from_value`` binds a symbolic event spec from a law's own, and
    ``bind_dims_from_spec`` from another ``DistributionSpec``, by that same rule.
    Repeated symbols share one scope, and conflicting sizes raise ValueError.

    Examples
    --------
    >>> from probpipe import DistributionSpec, OutputSpec, NumericArraySpec
    >>> declared = DistributionSpec(OutputSpec(x=NumericArraySpec(("n",))))
    >>> bound = declared.bind_dims_from_spec(DistributionSpec(OutputSpec(x=NumericArraySpec((3,)))))
    >>> bound.event_spec.term_spec.shape
    (3,)
    """

    event_spec: OutputSpec

    def __init__(self, event_spec: OutputSpec | RecordSpec) -> None:
        if isinstance(event_spec, RecordSpec):
            event_spec = OutputSpec(event_spec)
        elif not isinstance(event_spec, OutputSpec):
            raise TypeError(
                f"DistributionSpec.event_spec must be an OutputSpec or a RecordSpec; got "
                f"{type(event_spec).__name__}. Name the field, such as OutputSpec(x=...)"
            )
        if event_spec.term_spec is None:
            raise ValueError("DistributionSpec.event_spec does not declare a type")
        object.__setattr__(self, "event_spec", event_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of the draw this declares."""
        return self.event_spec.term_spec.free_dims

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> DistributionSpec:
        """This spec around a substituted event spec."""
        declaration = self.event_spec
        return DistributionSpec(
            declaration._with_term_spec(declaration.term_spec._substitute_dims(bindings))
        )

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Bind the declared event against the event spec *value* carries."""
        if not isinstance(value, Distribution):
            raise ValueError(f"{path} does not conform to its field spec ({self!r})")
        _unify_declarations(self.event_spec, value.event_spec, bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind the declared event against *actual*'s own."""
        if not isinstance(actual, DistributionSpec):
            return False
        _unify_declarations(self.event_spec, actual.event_spec, bindings, path)
        return True

    def __repr__(self) -> str:
        """The event spec, as the constructor takes it."""
        return term_repr("DistributionSpec", None, [("event_spec", repr(self.event_spec))])

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a ``Distribution`` whose event spec matches this one."""
        if not isinstance(value, Distribution):
            return False
        try:
            _unify_declarations(self.event_spec, value.event_spec, {}, "the event spec")
        except ValueError:
            return False
        return True


def _value_at_declared_kind(spec: TermSpec, record_label: str, value: Any) -> Any:
    """*value* at the kind *spec* declares, as a function's body receives a draw.

    A value of a record spec is a ``Record``, labeled *record_label* when it
    is built here, and a value of an array spec is an array; a value of
    another kind is passed as it is.
    """
    if isinstance(spec, RecordSpec):
        return (
            value
            if isinstance(value, Record)
            else Record(_layout_record_mapping(spec, value), label=record_label)
        )
    if isinstance(spec, NumericArraySpec):
        return jnp.asarray(value)
    return value


def _probed_law_traceability(law: Distribution) -> Traceability:
    """*law*'s density traced at a stand-in of one value of its event, or its sampler at one key."""
    spec = law.event_spec.term_spec
    stand_in = spec_stand_in(spec, unbound_size=1)
    if isinstance(law, SupportsUnnormalizedLogProb) and stand_in is not None:
        probe_result = probe(
            None,
            lambda value: law._unnormalized_log_prob(
                _value_at_declared_kind(spec, law.label, value)
            ),
            stand_in,
            label=law.label,
        )
    elif isinstance(law, SupportsSampling):
        probe_result = probe(
            None, lambda key: law._sample(key, sample_shape=()), jax.random.key(0), label=law.label
        )
    else:
        return Traceability.HOST_ONLY
    return probe_result.traceability
