"""Factored distributions: joints built from an ordered list of factors.

Provides:
  - ``SupportsFactors`` – the capability of carrying an explicit factorization.
  - ``FactoredDistribution`` – a joint ``Distribution`` of its factors.
  - ``FactoredConditionalDistribution`` – a joint ``ConditionalDistribution``,
    whose factors leave some givens unmet.
  - the numeric markers of the two factored kinds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from ..core._array_backend import _is_numeric_leaf
from ..core._compilation import compiled_or_eager
from ..core._dispatch import Feasibility, ResolutionError
from ..core._expression import (
    Named,
    Product,
    Selected,
    joined_labels,
)
from ..core._immutable import transient_memo
from ..core._object_batch import _is_object_array
from ..core._record_batch import RecordBatch
from ..core._record_spec import RecordSpec
from ..core._repr import sequence_repr
from ..core._spec_base import (
    NumericArraySpec,
    NumericSpec,
    OpaqueSpec,
    TermSpec,
    _known_type,
    _unify_specs,
)
from ..core._specs import InputSpec, OutputSpec, _has_record_outputs, _single_output_name
from ..core.named_tree import _PATH_SEP, _unflatten_paths
from ..core.provenance import Provenance
from ..core.record import Record
from ..linalg import DenseLinOp, to_dense
from ._capabilities import (
    _CONDITIONAL_TWINS,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
    _claimed,
    _claims,
    _conjunction,
    _kernel_claims,
    _marginal_claims,
)
from ._conditional import (
    ConditionalDistribution,
    ConditionalDistributionSpec,
    _event_is_numeric,
    _given_side_is_numeric,
    _missing_inputs,
    _unknown_inputs,
)
from ._distribution import (
    _DECLARATION_MARKERS,
    Distribution,
    DistributionSpec,
    _declares_numeric_event,
    _fixed_paths,
    _keeps_fixed_paths,
    _law_repr,
    _no_free_dims,
    _recorded_copy,
)

if TYPE_CHECKING:
    from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = [
    "FactoredConditionalDistribution",
    "FactoredConditionalNumericDistribution",
    "FactoredDistribution",
    "FactoredFullyNumericConditionalDistribution",
    "FactoredNumericConditionalDistribution",
    "FactoredNumericDistribution",
    "SupportsFactors",
]


def _joined_label(labels: Iterable[str]) -> str:
    """The labels of factors joined with ``·``, each read as one unit, as a product's label joins them.

    A label that is itself a product joins as it is, so labels join
    associatively: ``lik·prior`` joined with ``d`` is ``lik·prior·d``. Any other
    label is grouped as :func:`~probpipe.core._repr.grouped_label` groups an
    operand: a compound label is parenthesized, as a label ``model | y`` is, and a
    label with a space is bracketed.
    """
    return joined_labels(labels)


def _is_named(joint: Any) -> bool:
    """Whether the product *joint* was given a label, so it displays by that label.

    A product without a label carries a product of its factors as its
    expression, possibly conditioned or selected, and a labeled one carries
    its label.
    """
    return not isinstance(joint._expression.core(), Product)


def _product_of(factors: Iterable[Any]) -> Product:
    """The expression of the product of *factors*, which displays factor by factor."""
    return Product(tuple(factor._embedded_expression() for factor in factors))


def _with_named(joint: Any, named: bool) -> Any:
    """*joint*, a product just built, which displays by its label when *named* and factor by factor otherwise.

    A product built by ``*`` or by the ``joint`` operation is unlabeled, so it
    displays factor by factor, and a product given a label displays by it.
    *joint* keeps the paths it holds fixed, and it is set in place.
    """
    if named == _is_named(joint):
        return joint
    core = Named(joint.label) if named else _product_of(joint.factors)
    joint._store_expression(core.with_fixed(_fixed_paths(joint)))
    return joint


def _derived_product(joint: Any, source: Any) -> Any:
    """*joint*, a product just built from the product *source*, displayed as *source* is.

    *joint* displays by its label exactly when *source* does, and it holds the
    paths *source* holds fixed. An unlabeled *joint* reads in one of three
    ways:

    1. factor by factor over its own factors, when their labels join to
       *source*'s label, as a renamed ``lik * prior`` does;
    2. as *source* reads, when its factors carry the labels of *source*'s
       factors, as a dimension transform of ``model * d`` does;
    3. by *source*'s label and its own outputs otherwise, as a joint whose
       conditioning dropped a factor does, as ``(a·b)(b)``.
    """
    _keeps_fixed_paths(joint, source)
    if _is_named(source):
        return _with_named(joint, True)
    if _joined_label(factor.label for factor in joint.factors) == source.label:
        return _with_named(joint, False)
    held = _fixed_paths(joint)
    expression = source._expression
    if [factor.label for factor in joint.factors] != [part.label for part in source.factors]:
        expression = Selected(expression.core(), tuple(joint.event_spec.outputs))
    joint._store_expression(expression.with_fixed(held))
    return joint


def _labeled_product(term: Any) -> Any:
    """*term*, the result of a function, displayed by its label when it is a product.

    A product a function returns takes the function's output label as its
    label, so it displays by that label, as ``predict(y, mu)``. An unlabeled
    product is returned as a labeled copy that shares its factors, and any
    other term is returned as it is.
    """
    if isinstance(term, (FactoredDistribution, FactoredConditionalDistribution)) and not _is_named(
        term
    ):
        return _with_named(term._shallow_copy(), True)
    return term


type Factor = Distribution | ConditionalDistribution


@runtime_checkable
class SupportsFactors(Protocol):
    """A distribution built from an ordered list of sub-distributions, its factors.

    Each factor is a ``Distribution`` or a ``ConditionalDistribution``. The
    dependence graph is derived by matching each factor's inputs against
    the outputs the factors to its right produce, so it is not stored.
    """

    @property
    def factors(self) -> tuple[Distribution | ConditionalDistribution, ...]: ...


# ---------------------------------------------------------------------------
# The factor graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FactorGraph:
    """The validated structure of an ordered list of factors.

    Attributes
    ----------
    factors : tuple
        The factors, with dimensions bound in the joint's one scope.
    producers : Mapping[str, int]
        The index of the factor producing each output.
    edges : tuple of (int, int, str)
        Each dependency as the consuming factor's index, the producing factor's
        index, and the matched name.
    unmet : InputSpec or None
        The givens no factor produces, or ``None`` when every given is met.
    event_spec : OutputSpec
        The positional form over the record of every factor's outputs, in
        factor order.
    scope : Mapping[str, int]
        The joint's dimension scope: each dimension name bound while the factors
        were unified, so a later composition binds the same name to the same
        size, whatever the grouping.
    """

    factors: tuple[Factor, ...]
    producers: Mapping[str, int]
    edges: tuple[tuple[int, int, str], ...]
    unmet: InputSpec | None
    event_spec: OutputSpec
    scope: Mapping[str, int]

    def parents(self, index: int) -> set[int]:
        """The indices of the factors that factor *index* conditions on."""
        return {producer for consumer, producer, _ in self.edges if consumer == index}

    def ancestors(self, indices: set[int]) -> set[int]:
        """*indices* with every factor they condition on, transitively."""
        closure = set(indices)
        frontier = list(indices)
        while frontier:
            for parent in self.parents(frontier.pop()):
                if parent not in closure:
                    closure.add(parent)
                    frontier.append(parent)
        return closure


def _given_of(factor: Factor) -> InputSpec | None:
    return factor.given_spec if isinstance(factor, ConditionalDistribution) else None


def _free_dims(factor: Factor) -> frozenset[str]:
    if isinstance(factor, ConditionalDistribution):
        return factor.spec.free_dims
    return factor.event_spec.term_spec.free_dims


def _flattened(
    factors: Sequence[Factor], scope: Mapping[str, int] | None
) -> tuple[tuple[Factor, ...], dict[str, int]]:
    """*factors* with each factored factor replaced by its own, and the merged scope.

    A packaged joint is one factor, since its form is part of its event spec.

    Parameters
    ----------
    factors : sequence of Factor
        The factors in composition order, any of which may be a factored joint.
    scope : Mapping[str, int] or None
        The size of each dimension bound so far, keyed by its name, or ``None``
        when none is bound.

    Returns
    -------
    factors : tuple of Factor
        The flat factors, in composition order.
    scope : dict[str, int]
        *scope* merged with the scope of each flattened joint.

    Raises
    ------
    ValueError
        If two scopes bind one dimension name to different sizes.
    """
    bindings = dict(scope or {})
    flat: list[Factor] = []
    for factor in factors:
        graph = getattr(factor, "_graph", None)
        if (
            isinstance(factor, SupportsFactors)
            and isinstance(graph, _FactorGraph)
            and _single_output_name(factor.event_spec) is None
        ):
            flat.extend(graph.factors)
            for name, size in graph.scope.items():
                if bindings.setdefault(name, size) != size:
                    raise ValueError(
                        f"the dimension {name!r} is bound to {bindings[name]} and to {size}; "
                        f"rename one apart with with_dim_names"
                    )
        else:
            flat.append(factor)
    return tuple(flat), bindings


def _unify_either_way(
    first: TermSpec, second: TermSpec, bindings: dict[str, int], path: str
) -> None:
    """Unify two specs that meet as equals, whichever direction admits the other.

    Parameters
    ----------
    first : TermSpec
        The spec of one unmet input.
    second : TermSpec
        The spec of another unmet input of the same name.
    bindings : dict[str, int]
        The size of each symbolic dimension bound so far, keyed by its name, which
        the direction that unifies extends in place.
    path : str
        The location that error messages name, such as ``"the input 'x'"``.

    Raises
    ------
    ValueError
        If neither direction unifies, with the error of the first.
    """
    trial = dict(bindings)
    try:
        _unify_specs(first, second, trial, path)
    except ValueError as forward:
        trial = dict(bindings)
        try:
            _unify_specs(second, first, trial, path)
        except ValueError:
            raise forward from None
    bindings.update(trial)


def _factor_graph(
    factors: Sequence[Factor], scope: Mapping[str, int] | None = None
) -> _FactorGraph:
    """Validate the ordered *factors* by the composition rules and derive their graph.

    A factored factor enters as its own factors, with its scope. The order is
    conditional-first: a factor may condition on an output that a factor to
    its right produces, and never on one a factor to its left produces. Every
    output is produced once. A matched output's spec must unify with the
    consuming input's, and same-named unmet inputs unify into one input, all in one
    dimension scope, starting from *scope*, whose bindings are applied to the
    factors and recorded with the graph. An optional input is met as a required
    one is, and unmet, it takes its default. The graph records the unmet inputs
    only when a required one is among them, and an unmet input is optional when
    every factor that names it holds it optional.

    Parameters
    ----------
    factors : sequence of Factor
        The factors in conditional-first order.
    scope : Mapping[str, int] or None, optional
        The size of each dimension already bound, keyed by its name, such as the
        scope of a joint being rebuilt.

    Returns
    -------
    _FactorGraph
        The flat factors, bound in the joint's one scope, with their dependencies
        and their unmet inputs.

    Raises
    ------
    TypeError
        If a factor is neither distribution kind.
    ValueError
        If there are no factors, an output is produced twice, a factor
        consumes an output a factor to its left produces, matched specs do
        not unify, or two scopes bind one dimension to different sizes.
    """
    factors, bindings = _flattened(tuple(factors), scope)
    if not factors:
        raise ValueError("a FactoredDistribution needs at least one factor; got none")
    producers: dict[str, int] = {}
    output_term_specs: dict[str, TermSpec] = {}
    for index, factor in enumerate(factors):
        if not isinstance(factor, (Distribution, ConditionalDistribution)):
            raise TypeError(
                f"each factor must be a Distribution or a ConditionalDistribution; got "
                f"{type(factor).__name__}"
            )
        for output_name, spec in factor.event_spec.outputs.items():
            if output_name in producers:
                raise ValueError(
                    f"the output {output_name!r} is produced by both factor "
                    f"{producers[output_name]} ({factors[producers[output_name]].label!r}) and "
                    f"factor {index} ({factor.label!r}); rename it in one of them with "
                    f"with_path_names({{{output_name!r}: ...}}), or build one of them with "
                    f"another output, as Normal('{output_name}_2', ...)"
                )
            producers[output_name] = index
            output_term_specs[output_name] = spec
    edges: list[tuple[int, int, str]] = []
    unmet: dict[str, TermSpec] = {}
    required: set[str] = set()
    for index, factor in enumerate(factors):
        given = _given_of(factor)
        if given is None:
            continue
        for input_name, input_term_spec in given.items():
            producer = producers.get(input_name)
            if producer is None:
                if input_name not in given.optional:
                    required.add(input_name)
                if input_name in unmet:
                    _unify_either_way(
                        unmet[input_name], input_term_spec, bindings, f"the input {input_name!r}"
                    )
                    if isinstance(unmet[input_name], OpaqueSpec) and isinstance(
                        input_term_spec, OpaqueSpec
                    ):
                        unmet[input_name] = _known_type(unmet[input_name], input_term_spec)
                else:
                    unmet[input_name] = input_term_spec
                continue
            if producer < index:
                raise ValueError(
                    f"{factor.label!r} conditions on {input_name!r}, but {factors[producer].label!r} "
                    f"defines {input_name!r} to its left in the product; put "
                    f"{factors[producer].label!r} to the right of {factor.label!r}"
                )
            _unify_specs(
                input_term_spec,
                output_term_specs[input_name],
                bindings,
                f"the input {input_name!r}",
            )
            edges.append((index, producer, input_name))
    if bindings:
        factors = tuple(
            factor.with_dim_sizes(**subset)
            if (subset := {d: bindings[d] for d in _free_dims(factor) if d in bindings})
            else factor
            for factor in factors
        )
        unmet = {input_name: spec._substitute_dims(bindings) for input_name, spec in unmet.items()}
    event_spec = OutputSpec(
        RecordSpec(
            {
                output_name: spec
                for factor in factors
                for output_name, spec in factor.event_spec.outputs.items()
            }
        )
    )
    return _FactorGraph(
        factors=factors,
        producers=producers,
        edges=tuple(edges),
        unmet=InputSpec(unmet).with_optional(*(set(unmet) - required)) if required else None,
        event_spec=event_spec,
        scope=bindings,
    )


def _joint_declaration(graph: _FactorGraph, output_name: str | None) -> OutputSpec:
    """The event spec of a joint of *graph*, packaged under *output_name* when one is given.

    Without *output_name*, the joint's event spec is the positional form over
    the record of its factors' outputs, and a packaged joint declares that
    record as a whole term under *output_name*. A packaged joint's draw is the
    record of its factors' outputs, in the raw form the draw of a joint in the
    positional form has.
    """
    if output_name is None:
        return graph.event_spec
    return OutputSpec(**{output_name: graph.event_spec.term_spec})


def _form_keywords(joint: Any) -> dict[str, str]:
    """The constructor keyword that keeps *joint*'s form: its root output's name, if any."""
    output_name = _single_output_name(joint.event_spec)
    return {} if output_name is None else {"_output_name": output_name}


# ---------------------------------------------------------------------------
# The raw forms of a joint
# ---------------------------------------------------------------------------
#
# A factor contributes its outputs to a joint's value by its own form: a
# whole term's output is the factor's value itself, and the outputs of the
# positional form over a record are the value's immediate children. Scoring reverses the
# extraction, so every factor receives a value of the kind it declares.


def _raw_record(value: Any) -> Any:
    """*value* in the raw form of a record-valued result: the nested mapping of its raw leaves.

    A ``Record`` gives the nested mapping of its leaves, and a batch of records
    the nested mapping of its columns with the batch axes leading. A mapping is
    converted value by value, and any other value is returned as it is.
    """
    if isinstance(value, RecordBatch):
        return _unflatten_paths(value._raw_columns())
    if isinstance(value, Record):
        return value.to_nested_dict()
    if isinstance(value, Mapping):
        return {key: _raw_record(entry) for key, entry in value.items()}
    return value


def _extract(value: Any, segments: tuple[str, ...]) -> Any:
    """The node at *segments* below the top of *value*, a raw value shaped like a draw.

    *value* is in raw form, so a value with paths is a nested mapping of raw
    leaves, which each segment indexes. With no segments *value* is returned
    whole.

    Parameters
    ----------
    value : Any
        A raw value shaped like a draw, such as a draw or a mean.
    segments : tuple of str
        The segments of the node's path below the top of *value*.

    Returns
    -------
    Any
        The raw node, with any leading batch axes of *value* kept on its leaves.

    Raises
    ------
    TypeError
        If *segments* is not empty and *value* is not a mapping.
    """
    for segment in segments:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"the node at {_PATH_SEP.join(segments)!r} cannot be read from a "
                f"{type(value).__name__}; a raw value with paths is a mapping"
            )
        value = value[segment]
    return value


def _raw_leaves(value: Any, spec: TermSpec) -> list[Array]:
    """The array leaves of *value*, a raw value of the event *spec*, in the declaration's order.

    A record value, a ``Record`` or the nested mapping of its leaves, gives one
    array per field of *spec*, in canonical order, and any other value is its
    own one leaf. Leading batch axes stay on every leaf.
    """
    if not isinstance(spec, RecordSpec):
        return [jnp.asarray(value)]
    raw = _raw_record(value)
    return [jnp.asarray(_extract(raw, tuple(path.split(_PATH_SEP)))) for path in spec]


def _children(value: Any) -> Mapping[str, Any]:
    """The immediate children of *value*, the raw form of a record, keyed by name.

    A raw record is a mapping of its fields. A ``Record`` is read through its
    one-level view, and a batch of records through its raw columns.

    Parameters
    ----------
    value : Any
        A raw record, a ``Record``, or a ``RecordBatch``.

    Returns
    -------
    Mapping[str, Any]
        *value* itself when it is a mapping, and a mapping read from it otherwise.

    Raises
    ------
    TypeError
        If *value* is none of these.
    """
    if isinstance(value, Mapping):
        return value
    if isinstance(value, RecordBatch):
        return _raw_record(value)
    children = getattr(value, "children", None)
    if isinstance(children, Mapping):
        return children
    raise TypeError(
        f"a record value must be a Record or a mapping of its fields; got {type(value).__name__}"
    )


def _outputs_of(declaration: OutputSpec, value: Any) -> dict[str, Any]:
    """The value of each output *declaration* declares, extracted from *value*.

    *value* is a draw of the declared event, or an event-typed result such as a
    mean, and the leading axes of a batch stay on each output.
    """
    if not _has_record_outputs(declaration):
        (output_name,) = declaration.outputs
        return {output_name: value}
    children = _children(value)
    return {output_name: children[output_name] for output_name in declaration.outputs}


def _event_of(declaration: OutputSpec, outputs: Mapping[str, Any]) -> Any:
    """The value *declaration* declares, reconstructed from a joint's *outputs*.

    It inverts :func:`_outputs_of`: a whole term is its output's value, and
    the positional form over a record is the mapping of its outputs.
    """
    if not _has_record_outputs(declaration):
        (output_name,) = declaration.outputs
        return outputs[output_name]
    return {output_name: outputs[output_name] for output_name in declaration.outputs}


def _given_values(joint: Any, given: Record | Mapping[str, Any]) -> dict[str, Any]:
    """The value of each input of the conditional *joint* in *given*, by input name.

    Parameters
    ----------
    joint : FactoredConditionalDistribution
        The conditional joint whose inputs *given* binds.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.

    Returns
    -------
    dict[str, Any]
        A new dict of the top-level entries of *given*.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one,
        since a fused conditional path binds every required input.
    """
    top = given.children if hasattr(given, "children") else given
    values = dict(top.items())
    unknown = set(values) - set(joint.given_spec)
    if unknown:
        raise KeyError(_unknown_inputs(joint.label, sorted(unknown), joint.given_spec))
    missing = [input_name for input_name in joint.given_spec.required if input_name not in values]
    if missing:
        raise KeyError(_missing_inputs(joint.label, missing))
    return values


def _leading_axes(value: Any, spec: TermSpec) -> tuple[int, ...] | None:
    """The axes of *value* before the ones *spec* declares, read at its first array leaf.

    Returns ``None`` when *spec* declares no array leaf to read them at.
    """
    if isinstance(spec, NumericArraySpec):
        shape = tuple(jnp.shape(value))
        return shape[: max(len(shape) - len(spec.shape), 0)]
    if isinstance(spec, RecordSpec):
        children = _children(value)
        for name, child in spec.children.items():
            axes = _leading_axes(children[name], child)
            if axes is not None:
                return axes
    return None


def _flatten_draws(tree: Any, batch_shape: tuple[int, ...]) -> Any:
    """*tree* with the leading *batch_shape* axes of each leaf merged into one axis.

    An object array holds a column of values that are not arrays, and it is
    merged too.
    """
    rank, count = len(batch_shape), math.prod(batch_shape)

    def merged(leaf: Any) -> Any:
        if _is_object_array(leaf):
            return leaf.reshape((count, *leaf.shape[rank:]))
        return jnp.reshape(leaf, (count, *jnp.shape(leaf)[rank:]))

    return jax.tree.map(merged, tree)


def _unflatten_draws(tree: Any, batch_shape: tuple[int, ...]) -> Any:
    """*tree* with the leading axis of each leaf split into *batch_shape*."""
    return jax.tree.map(lambda leaf: jnp.reshape(leaf, (*batch_shape, *jnp.shape(leaf)[1:])), tree)


def _stacked(values: Sequence[Any]) -> Any:
    """The pytrees in *values*, stacked leaf by leaf along a new leading axis."""
    return jax.tree.map(lambda *leaves: jnp.stack([jnp.asarray(leaf) for leaf in leaves]), *values)


def _numeric_givens(factor: ConditionalDistribution, values: Mapping[str, Any]) -> bool:
    """Whether every input of *factor* that *values* binds is numeric, so a call traces."""
    return all(isinstance(factor.given_spec[input_name], NumericSpec) for input_name in values)


def _each_value(function: Callable[..., Any], count: int, *trees: Any, traceable: bool) -> Any:
    """*function* at each of the *count* values along the leading axis of *trees*, stacked.

    A traceable call is vectorized with ``jax.vmap``. Otherwise *function* is
    called on one value at a time, since a value that is not an array cannot be
    traced, and the results are stacked leaf by leaf.
    """
    if traceable:
        return jax.vmap(function)(*trees)
    return _stacked(
        [
            function(*(jax.tree.map(lambda leaf, at=index: leaf[at], tree) for tree in trees))
            for index in range(count)
        ]
    )


def _batch_axes(
    factor: ConditionalDistribution, inner: Mapping[str, Any], event: Any
) -> tuple[int, ...]:
    """The batch axes of *factor*'s part of a joint's value, or ``()`` for one value.

    They are read at the first array leaf of the outputs that *inner* binds
    and, when no bound output has one, at the factor's own event. With no
    bound output the factor scores its part in one call.
    """
    for input_name, value in inner.items():
        axes = _leading_axes(value, factor.given_spec[input_name])
        if axes is not None:
            return axes
    if inner:
        axes = _leading_axes(event, factor.event_spec.term_spec)
        if axes is not None:
            return axes
    return ()


# ---------------------------------------------------------------------------
# Sampling and scoring
# ---------------------------------------------------------------------------


def _ancestral_sample(
    graph: _FactorGraph, fixed: Mapping[str, Any], key: PRNGKey, sample_shape: tuple[int, ...]
) -> dict[str, Any]:
    """The draw of the joint of *graph*, each unmet given set by *fixed*.

    Conditional-first order makes right to left an ancestral order, so every
    output a factor conditions on is drawn before the factor draws. Each
    factor's draw is read in its raw form, so the joint's draw is the nested
    mapping of raw leaves whatever form a factor returns.
    """
    keys = jax.random.split(key, len(graph.factors))
    drawn: dict[str, Any] = {}
    for index in reversed(range(len(graph.factors))):
        factor = graph.factors[index]
        if isinstance(factor, ConditionalDistribution):
            draw = _conditional_draw(factor, drawn, fixed, keys[index], sample_shape)
        else:
            draw = factor._sample(keys[index], sample_shape)
        drawn.update(_outputs_of(factor.event_spec, _raw_record(draw)))
    return {output_name: drawn[output_name] for output_name in graph.event_spec.outputs}


def _law_at_defaults(factor: Factor, produced: Collection[str]) -> Factor:
    """*factor*, or its law at its defaults when it is a kernel that needs no input.

    A kernel needs no input when none of its inputs is required or in
    *produced*. A joint whose factors produce none of such a kernel's inputs
    holds them at their defaults, so the kernel standing alone is that law.
    """
    if (
        isinstance(factor, ConditionalDistribution)
        and not factor.given_spec.required
        and not set(factor.given_spec) & set(produced)
    ):
        return factor._condition_on({})
    return factor


def _factor_given(
    factor: ConditionalDistribution, values: Mapping[str, Any], fixed: Mapping[str, Any]
) -> dict[str, Any]:
    """Each input's value for *factor*: an output's in *values*, else an unmet input's in *fixed*.

    An optional input that neither sets is left out, so the factor takes its default.
    """
    return {
        input_name: values[input_name] if input_name in values else fixed[input_name]
        for input_name in factor.given_spec
        if input_name in values
        or input_name in fixed
        or input_name not in factor.given_spec.optional
    }


#: The kernel method a joint maps over the draws of a factor's givens to draw the factor.
_DRAW = "_conditional_sample"


def _is_constant(value: Any) -> bool:
    """Whether *value* is a string, ``None``, a boolean, or a tuple of them.

    Such a value is hashable and compares equal exactly when it is the same
    value, so a compiled program can take it as a static argument.
    """
    if isinstance(value, tuple):
        return all(_is_constant(item) for item in value)
    return value is None or isinstance(value, (str, bool, np.bool_))


def _split_fixed(
    factor: ConditionalDistribution, fixed: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[tuple[str, Any], ...]] | None:
    """The unmet givens in *fixed* a compiled call takes as arguments, and those it takes as constants.

    Returns ``None`` when an unmet given is neither, and the call then runs
    uncompiled. The three cases:

    1. **Arguments:** an input that *factor* declares numeric, holding numeric
       leaves, as :func:`~probpipe.core._array_backend._is_numeric_leaf`
       decides. A new value of the same shape reuses the program.
    2. **Constants:** a value that :func:`_is_constant` admits. It is part of
       the program's identity, so a new value compiles a new program, which
       the factor keeps.
    3. **Neither:** any other value, such as an object whose state may change
       between calls, since a program would keep the state it had when it was
       traced.
    """
    arguments: dict[str, Any] = {}
    constants: list[tuple[str, Any]] = []
    for input_name, value in fixed.items():
        if isinstance(factor.given_spec[input_name], NumericSpec) and all(
            _is_numeric_leaf(leaf) for leaf in jax.tree.leaves(value)
        ):
            arguments[input_name] = value
        elif _is_constant(value):
            constants.append((input_name, value))
        else:
            return None
    return arguments, tuple(constants)


def _at_each_given(
    factor: ConditionalDistribution,
    method: str,
    fixed: Mapping[str, Any],
    count: int,
    values: Mapping[str, Any],
    points: Any,
) -> Any:
    """*factor*'s *method* at each of the *count* values along the leading axis of *values*, stacked.

    The *i*-th call binds the factor's givens to the *i*-th entry of *values*
    and to the unmet givens in *fixed*, and takes the *i*-th entry of *points*:
    a key when *method* draws, and the factor's part of the joint's value when
    *method* is a density.

    A call whose givens are numeric runs one compiled program: the factor's
    method vectorized with ``jax.vmap`` and compiled with ``jax.jit``. It takes
    the draws, the keys or the events, and the unmet givens as
    :func:`_split_fixed` sorts them. The factor's memo holds the program, with
    one compiled form for each set of constants and input shapes. The program
    calls the factor it was built for, so a copy of the factor, which may hold
    other bound values, builds its own. ``ConditionalDistribution`` states
    what the factor's method may read.

    The call runs uncompiled, under ``jax.vmap``, when :func:`_split_fixed`
    returns ``None`` or when the program does not trace, as for a method that
    converts an unmet given to a Python number.
    :func:`~probpipe.core._compilation.compiled_or_eager` records a program
    that does not trace for the structure of the unmet givens, so later calls
    with that structure skip the attempt. When a given is not numeric, the
    method is called on one value at a time.
    """
    fixed = {
        input_name: fixed[input_name] for input_name in factor.given_spec if input_name in fixed
    }
    tail = ((),) if method == _DRAW else ()

    def one(fixed: Mapping[str, Any], values: Mapping[str, Any], point: Any) -> Any:
        return getattr(factor, method)(_factor_given(factor, values, fixed), point, *tail)

    def uncompiled(*_: Any) -> Any:
        return _each_value(partial(one, fixed), count, values, points, traceable=True)

    if not _numeric_givens(factor, values):
        return _each_value(partial(one, fixed), count, values, points, traceable=False)
    split = _split_fixed(factor, fixed)
    if split is None:
        return uncompiled()
    memo = transient_memo(factor)
    program = memo.get(("mapped", method))
    if program is None:

        def mapped(arguments: dict[str, Any], constants: tuple, values: Any, points: Any) -> Any:
            held = {**arguments, **dict(constants)}
            return jax.vmap(partial(one, held))(values, points)

        program = memo[("mapped", method)] = jax.jit(mapped, static_argnums=1)
    arguments, constants = split
    key = ("untraced", method, jax.tree.structure(arguments), constants)
    return compiled_or_eager(memo, key, program, uncompiled, arguments, constants, values, points)


def _conditional_draw(
    factor: ConditionalDistribution,
    drawn: Mapping[str, Any],
    fixed: Mapping[str, Any],
    key: PRNGKey,
    sample_shape: tuple[int, ...],
) -> Any:
    """A draw of *factor* at the outputs in *drawn* it names and its unmet givens in *fixed*.

    Under a non-empty *sample_shape* the factor is mapped over the draws of
    those outputs, one given value per call, each call with its own key.
    """
    inner = {
        input_name: drawn[input_name] for input_name in factor.given_spec if input_name in drawn
    }
    if not inner or not sample_shape:
        return factor._conditional_sample(_factor_given(factor, inner, fixed), key, sample_shape)
    count = math.prod(sample_shape)
    draws = _at_each_given(
        factor,
        _DRAW,
        fixed,
        count,
        _flatten_draws(inner, sample_shape),
        jax.random.split(key, count),
    )
    return _unflatten_draws(draws, sample_shape)


def _score(graph: _FactorGraph, fixed: Mapping[str, Any], value: Any, method: str) -> Array:
    """The sum of each factor's density *method* at its part of *value*, unmet givens in *fixed*."""
    outputs = _children(value)
    total: Any = None
    for factor in graph.factors:
        event = _event_of(factor.event_spec, outputs)
        if isinstance(factor, ConditionalDistribution):
            score = _conditional_score(
                factor, f"_conditional{method}", graph, outputs, fixed, event
            )
        else:
            score = getattr(factor, method)(event)
        total = score if total is None else total + score
    return total


def _conditional_score(
    factor: ConditionalDistribution,
    method: str,
    graph: _FactorGraph,
    outputs: Mapping[str, Any],
    fixed: Mapping[str, Any],
    event: Any,
) -> Array:
    """*factor*'s density *method* at *event*, at the inputs the joint's outputs supply."""
    inner = {
        input_name: outputs[input_name]
        for input_name in factor.given_spec
        if input_name in graph.producers
    }
    batch = _batch_axes(factor, inner, event)
    if not batch:
        return getattr(factor, method)(_factor_given(factor, inner, fixed), event)
    scores = _at_each_given(
        factor, method, fixed, math.prod(batch), *_flatten_draws((inner, event), batch)
    )
    return _unflatten_draws(scores, batch)


# ---------------------------------------------------------------------------
# The moments of an edge-free joint
# ---------------------------------------------------------------------------


def _factor_results(
    graph: _FactorGraph, fixed: Mapping[str, Any], method: str, *arguments: Any
) -> Iterator[tuple[Factor, Any]]:
    """Each factor with the result of its *method*, a conditional factor's at its givens in *fixed*.

    An edge-free joint calls this, so every given of a conditional factor is unmet.
    """
    for factor in graph.factors:
        if isinstance(factor, ConditionalDistribution):
            given = _factor_given(factor, {}, fixed)
            yield factor, getattr(factor, f"_conditional{method}")(given, *arguments)
        else:
            yield factor, getattr(factor, method)(*arguments)


def _assemble_by_output(
    graph: _FactorGraph, fixed: Mapping[str, Any], method: str, *arguments: Any
) -> dict[str, Any]:
    """Each factor's event-typed *method* result in its raw form, assembled per output."""
    assembled: dict[str, Any] = {}
    for factor, result in _factor_results(graph, fixed, method, *arguments):
        assembled.update(_outputs_of(factor.event_spec, _raw_record(result)))
    return {output_name: assembled[output_name] for output_name in graph.event_spec.outputs}


def _block_diagonal(graph: _FactorGraph, fixed: Mapping[str, Any]) -> DenseLinOp:
    """The factors' covariances as the diagonal blocks of one operator, in factor order.

    Each factor's outputs are contiguous in canonical factor order and keep
    the factor's own order, so factor order is the order of the joint's
    flattened coordinates.
    """
    blocks = [to_dense(cov) for _, cov in _factor_results(graph, fixed, "_cov")]
    return DenseLinOp(block_diag(*blocks))


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------


def _requests(graph: _FactorGraph, paths: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    """The requested paths within each factor that produces one, keyed by index in factor order."""
    requested: dict[int, list[str]] = {}
    for path in paths:
        requested.setdefault(graph.producers[path.split(_PATH_SEP, 1)[0]], []).append(path)
    return {index: tuple(requested[index]) for index in sorted(requested)}


def _kept_whole(factor: Factor, requested: tuple[str, ...]) -> bool:
    """Whether a marginal keeps *factor* whole: every output of it is requested."""
    return set(requested) == set(factor.event_spec.outputs)


def _factor_request(requested: tuple[str, ...], *, list_of_paths: bool) -> str | tuple[str, ...]:
    """The path argument of a factor's marginal: the one path of a projection, or a list of paths."""
    return tuple(requested) if list_of_paths else requested[0]


def _sole_field_projection(method: str) -> Callable[..., Any]:
    """The capability of a :class:`_SoleField` that takes the one field of the law's *method*."""

    def projected(self: _SoleField, *arguments: Any) -> Any:
        return _children(_raw_record(getattr(self._law, method)(*arguments)))[self._output_name]

    projected.__name__ = method
    projected.__qualname__ = f"_SoleField.{method}"
    projected.__doc__ = f"The one field of the record law's ``{method}``, in its raw form."
    return projected


def _sole_field_density(method: str) -> Callable[..., Any]:
    """The density *method* of a :class:`_SoleField`, the law's at the record of the value."""

    def density(self: _SoleField, value: Any) -> Array:
        return getattr(self._law, method)({self._output_name: value})

    density.__name__ = method
    density.__qualname__ = f"_SoleField.{method}"
    density.__doc__ = f"The record law's ``{method}`` at the record whose one field is *value*."
    return density


def _sole_field_guard(method: str) -> Callable[..., Feasibility]:
    """The guard of a :class:`_SoleField`'s *method*, which is the law's guard of *method*."""

    def guard(self: _SoleField, *arguments: Any) -> Feasibility:
        return _capability_guard(self._law, method, *arguments)

    guard.__name__ = f"{method}_guard"
    guard.__qualname__ = f"_SoleField.{method}_guard"
    guard.__doc__ = f"The record law's guard of ``{method}``."
    return guard


def _sole_field_cov(self: _SoleField) -> Any:
    """The record law's covariance, since the field and the record flatten alike."""
    return self._law._cov()


def _sole_field_expectation(self: _SoleField, f: Callable[[Any], Array]) -> Array:
    """The record law's expectation of *f* at the one field of each draw, in its raw form."""
    return self._law._expectation(
        lambda record: f(_children(_raw_record(record))[self._output_name])
    )


def _sole_field_marginal(self: _SoleField, paths: str | tuple[str, ...]) -> Distribution:
    """This law at its output, and the record law's marginal at any other path.

    The field and the record have the same event paths, and the marginal keeps
    this law's label.
    """
    if paths == self._output_name:
        return self
    marginal = self._law._marginal(paths)
    return marginal if marginal.label == self.label else marginal.with_label(self.label)


def _sole_field_marginal_guard(self: _SoleField, paths: str | tuple[str, ...]) -> Feasibility:
    """Exact at the output, and the record law's guard at any other path."""
    if paths == self._output_name:
        return Feasibility(True)
    return _capability_guard(self._law, "_marginal", paths)


def _sole_field_marginal_capabilities(
    self: _SoleField, paths: str | tuple[str, ...]
) -> frozenset[type]:
    """This law's claims at its output, and the record law's report at any other path."""
    if paths == self._output_name:
        return _claims(self)
    return _marginal_claims(self._law, paths)


#: Each capability a :class:`_SoleField` takes from its record law, with its methods.
_SOLE_FIELD_CAPABILITIES: dict[type, Mapping[str, Callable[..., Any]]] = {
    SupportsSampling: {
        "_sample": _sole_field_projection("_sample"),
        "_sample_guard": _sole_field_guard("_sample"),
    },
    SupportsLogProb: {
        "_log_prob": _sole_field_density("_log_prob"),
        "_log_prob_guard": _sole_field_guard("_log_prob"),
    },
    SupportsUnnormalizedLogProb: {
        "_unnormalized_log_prob": _sole_field_density("_unnormalized_log_prob"),
        "_unnormalized_log_prob_guard": _sole_field_guard("_unnormalized_log_prob"),
    },
    SupportsMean: {
        "_mean": _sole_field_projection("_mean"),
        "_mean_guard": _sole_field_guard("_mean"),
    },
    SupportsVariance: {
        "_variance": _sole_field_projection("_variance"),
        "_variance_guard": _sole_field_guard("_variance"),
    },
    SupportsCovariance: {"_cov": _sole_field_cov, "_cov_guard": _sole_field_guard("_cov")},
    SupportsQuantile: {
        "_quantile": _sole_field_projection("_quantile"),
        "_quantile_guard": _sole_field_guard("_quantile"),
    },
    SupportsExpectation: {
        "_expectation": _sole_field_expectation,
        "_expectation_guard": _sole_field_guard("_expectation"),
    },
    SupportsMarginals: {
        "_marginal": _sole_field_marginal,
        "_marginal_guard": _sole_field_marginal_guard,
        "_marginal_capabilities": _sole_field_marginal_capabilities,
    },
}


class _SoleField(Distribution):
    """The law of the one field of a one-field record law, declared as a whole term.

    The field determines the record, so the two are one law in two forms,
    and each capability is the record law's through that correspondence. A
    joint's marginal returns it for a projection onto the one output of a
    factor in the positional form over a one-field record, since a projection
    is a whole term.

    Parameters
    ----------
    law : Distribution
        A law whose event spec is the positional form over a record of one
        field.
    """

    _capability_table = _SOLE_FIELD_CAPABILITIES

    def __new__(cls, law: Distribution) -> _SoleField:
        protocols = _claimed(law, _SOLE_FIELD_CAPABILITIES)
        return object.__new__(_capability_subclass(_SoleField, protocols))

    def __init__(self, law: Distribution) -> None:
        ((output_name, spec),) = law.event_spec.outputs.items()
        super().__init__(
            OutputSpec(**{output_name: spec}),
            label=law.label,
        )
        object.__setattr__(self, "_law", law)
        object.__setattr__(self, "_output_name", output_name)

    def __repr__(self) -> str:
        """A record-law constructor call, with the keyword-form event spec as ``event_spec=``."""
        return _law_repr(self, self._law._repr_arguments(), self._law)

    def _event_repr_parts(self) -> tuple[str | None, str | None]:
        """No output name, and the keyword-form event spec.

        These set this law apart from the record law.
        """
        return None, repr(self.event_spec)


def _requested_paths(joint: Any, paths: str | tuple[str, ...]) -> tuple[str, ...]:
    """The paths of *joint*'s factor record that *paths* requests: one path, or several.

    The event paths of a joint in the positional form are its record's paths. A
    packaged joint's paths start with its output's name, below which they are
    the record's paths.

    Parameters
    ----------
    joint : FactoredDistribution
        The joint whose factor record the paths address.
    paths : str or tuple of str
        An event path of the joint, or a list of event paths to select jointly,
        as :func:`~probpipe.core._shapes._as_paths` reads them.

    Returns
    -------
    tuple of str
        Each requested path within the factor record, in the order requested,
        with a packaged joint's leading output name removed.

    Raises
    ------
    KeyError
        If a path is not an event path of the joint, or a packaged joint's
        path names no node below its output.
    TypeError
        If a path is not a string.
    """
    listed_paths = (paths,) if isinstance(paths, str) else tuple(paths)
    output_name = _single_output_name(joint.event_spec)
    record = joint._graph.event_spec.term_spec
    inner: list[str] = []
    for requested in listed_paths:
        if not isinstance(requested, str):
            raise TypeError(f"a field path must be a string; got {type(requested).__name__}")
        within = requested
        if output_name is not None:
            head, _, within = requested.partition(_PATH_SEP)
            if head != output_name or not within:
                raise KeyError(
                    f"{requested!r} is not an event path of {joint.label!r}; its paths start "
                    f"with {output_name + _PATH_SEP!r}"
                )
        if within not in record:
            raise KeyError(
                f"{requested!r} is not an event path of {joint.label!r}; its fields: "
                f"{list(joint.event_spec.outputs)}"
            )
        inner.append(within)
    return tuple(inner)


def _is_list_of_the_output(joint: Any, paths: str | tuple[str, ...]) -> bool:
    """Whether *paths* is the list of one path that names a packaged joint's output.

    The marginal there is the joint itself, exposed as the record that holds
    its event under the output.
    """
    output_name = _single_output_name(joint.event_spec)
    return output_name is not None and paths == (output_name,)


def _fields_of(graph: _FactorGraph, indices: Sequence[int]) -> list[str]:
    """The fields the factors of *graph* at *indices* produce, in the joint's order."""
    return [
        output_name for index in indices for output_name in graph.factors[index].event_spec.outputs
    ]


def _marginal_guard(self: Any, paths: str | tuple[str, ...]) -> Feasibility:
    """Whether the marginal at *paths* is exact, by the factor graph.

    A path must be an event path of the joint. A packaged joint is exact at its
    output, whose marginal is the joint itself, and at the list of that one
    path. The target's ancestor closure must add no
    factor, since integrating out a field that a kept factor conditions on has
    no closed form here. Within the target, a factor requested whole is kept
    whole, and a factor requested in part delegates to its own marginal guard,
    provided no other requested factor conditions on what that reduction
    integrates out.
    """
    graph: _FactorGraph = self._graph
    if paths == _single_output_name(self.event_spec) or _is_list_of_the_output(self, paths):
        return Feasibility(True)
    try:
        listed_paths = _requested_paths(self, paths)
    except (KeyError, TypeError) as error:
        return Feasibility(False, str(error.args[0]) if error.args else repr(error))
    heads = {p.split(_PATH_SEP, 1)[0] for p in listed_paths}
    targets = {graph.producers[head] for head in heads}
    ancestors = graph.ancestors(targets) - targets
    if ancestors:
        integrated = _fields_of(graph, sorted(ancestors))
        return Feasibility(
            False,
            f"the marginal at {list(listed_paths)} integrates out the fields {integrated}, which the "
            f"requested fields condition on",
        )
    whole = {p for p in listed_paths if _PATH_SEP not in p}
    for consumer, producer, name in graph.edges:
        if consumer in targets and producer in targets and name not in whole:
            return Feasibility(
                False,
                f"the marginal at {list(listed_paths)} reduces the field {name!r}, which the fields "
                f"{_fields_of(graph, [consumer])} condition on",
            )
    for index, requested in _requests(graph, listed_paths).items():
        factor = graph.factors[index]
        if _kept_whole(factor, requested):
            continue
        if not isinstance(factor, SupportsMarginals):
            return Feasibility(
                False,
                f"the factor that produces the fields {_fields_of(graph, [index])} has no "
                f"marginals",
            )
        report = _capability_guard(
            factor,
            "_marginal",
            _factor_request(requested, list_of_paths=not isinstance(paths, str)),
        )
        if report.feasible is not True:
            return report
    return Feasibility(True)


# ---------------------------------------------------------------------------
# The derived capabilities
# ---------------------------------------------------------------------------
#
# Each function below is installed under the method name it realizes, in the
# capability subclass of a joint that claims it.


def _joint_sample(self: Any, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> dict[str, Any]:
    """Draw from the joint by ancestral sampling over its factors.

    The factors draw from right to left, and factor ``i`` draws with the
    ``i``-th key of a split of *key* into one key per factor, so one key
    reproduces the draw. A conditional factor conditions on the outputs the
    factors to its right drew. Under a non-empty *sample_shape* such a factor
    is mapped over the draws, each with its own key from a split of the
    factor's key, so it receives one given value per call.

    Parameters
    ----------
    self : FactoredDistribution
        The joint whose capability subclass installs this function as ``_sample``.
    key : PRNGKey
        The key of the draw.
    sample_shape : tuple of int, optional
        Batch axes prepended to every output; ``()`` draws once.

    Returns
    -------
    dict
        Each output's raw value, keyed by output name in canonical factor order.
    """
    return _ancestral_sample(self._graph, {}, key, tuple(sample_shape))


def _joint_log_prob(self: Any, value: Any) -> Array:
    """The normalized log-density of the joint: the sum of its factors' log-densities.

    Each factor scores its own event, reconstructed from the outputs of
    *value* by the factor's form, and a conditional factor is scored at the
    given the outputs it names supply. A batch of values keeps its leading
    axes, and a conditional factor is mapped over the batch, so it receives one
    given value per call.

    Parameters
    ----------
    self : FactoredDistribution
        The joint whose capability subclass installs this function as
        ``_log_prob``.
    value : Mapping[str, Any] or Record
        A value of the joint's event, or a batch of them, keyed by output name.

    Returns
    -------
    Array
        The log-density, shaped as the batch axes.
    """
    return _score(self._graph, {}, value, "_log_prob")


def _joint_unnormalized_log_prob(self: Any, value: Any) -> Array:
    """The log-density of the joint up to an additive constant.

    It is the sum of the factors' unnormalized log-densities, each factor scored
    as for the normalized log-density.

    Parameters
    ----------
    self : FactoredDistribution
        The joint whose capability subclass installs this function as
        ``_unnormalized_log_prob``.
    value : Mapping[str, Any] or Record
        A value of the joint's event, or a batch of them, keyed by output name.

    Returns
    -------
    Array
        The unnormalized log-density, shaped as the batch axes.
    """
    return _score(self._graph, {}, value, "_unnormalized_log_prob")


def _joint_mean(self: Any) -> dict[str, Any]:
    """The mean of an edge-free joint, each output's from the factor that produces it.

    Parameters
    ----------
    self : FactoredDistribution
        The edge-free joint whose capability subclass installs this function as
        ``_mean``.

    Returns
    -------
    dict
        Each output's mean, keyed by output name in canonical factor order.
    """
    return _assemble_by_output(self._graph, {}, "_mean")


def _joint_variance(self: Any) -> dict[str, Any]:
    """The variance of an edge-free joint, each output's from the factor that produces it.

    Parameters
    ----------
    self : FactoredDistribution
        The edge-free joint whose capability subclass installs this function as
        ``_variance``.

    Returns
    -------
    dict
        Each output's variance, keyed by output name in canonical factor order.
    """
    return _assemble_by_output(self._graph, {}, "_variance")


def _joint_cov(self: Any) -> DenseLinOp:
    """The covariance of an edge-free joint over its flattened draw.

    The factors are independent, so the covariance is block diagonal, with each
    factor's covariance as a block in factor order, the order of the joint's
    flattened coordinates.

    Parameters
    ----------
    self : FactoredDistribution
        The edge-free joint whose capability subclass installs this function as
        ``_cov``.

    Returns
    -------
    DenseLinOp
        The ``(d, d)`` covariance, where ``d`` is the size of the flattened draw.
    """
    return _block_diagonal(self._graph, {})


def _joint_quantile(self: Any, q: ArrayLike) -> dict[str, Any]:
    """The quantiles of an edge-free joint at the levels *q*, assembled per output.

    A whole-term factor's quantiles are its output's, and those of a factor in
    the positional form over a record are the children of the factor's result.
    Each factor's result is its event's raw form with the level axes leading in
    each leaf, so the joint's is the same form of its own event.

    Parameters
    ----------
    self : FactoredDistribution
        The edge-free joint whose capability subclass installs this function as
        ``_quantile``.
    q : ArrayLike
        The levels in ``[0, 1]``, of any shape.

    Returns
    -------
    dict
        Each output's quantiles, keyed by output name in canonical factor order.
    """
    return _assemble_by_output(self._graph, {}, "_quantile", q)


def _joint_conditional_sample(
    self: Any,
    given: Record | Mapping[str, Any],
    key: PRNGKey,
    sample_shape: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Draw from the joint at *given* by ancestral sampling over its factors.

    Each unmet given takes its value from *given*, and the factors draw as for
    the unconditional joint's ``_sample``, keys included.

    Parameters
    ----------
    self : FactoredConditionalDistribution
        The conditional joint whose capability subclass installs this function as
        ``_conditional_sample``.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.
    key : PRNGKey
        The key of the draw.
    sample_shape : tuple of int, optional
        Batch axes prepended to every output; ``()`` draws once.

    Returns
    -------
    dict
        Each output's raw value, keyed by output name in canonical factor order.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one.
    """
    return _ancestral_sample(self._graph, _given_values(self, given), key, tuple(sample_shape))


def _joint_conditional_log_prob(self: Any, given: Record | Mapping[str, Any], value: Any) -> Array:
    """The normalized log-density of the joint at *given*: the sum of its factors'.

    Each unmet given takes its value from *given*, and the factors are scored as
    the unconditional joint's are.

    Parameters
    ----------
    self : FactoredConditionalDistribution
        The conditional joint whose capability subclass installs this function as
        ``_conditional_log_prob``.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.
    value : Mapping[str, Any] or Record
        A value of the joint's event, or a batch of them, keyed by output name.

    Returns
    -------
    Array
        The log-density, shaped as the batch axes.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one.
    """
    return _score(self._graph, _given_values(self, given), value, "_log_prob")


def _joint_conditional_unnormalized_log_prob(
    self: Any, given: Record | Mapping[str, Any], value: Any
) -> Array:
    """The log-density of the joint at *given*, up to an additive constant.

    Parameters
    ----------
    self : FactoredConditionalDistribution
        The conditional joint whose capability subclass installs this function as
        ``_conditional_unnormalized_log_prob``.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.
    value : Mapping[str, Any] or Record
        A value of the joint's event, or a batch of them, keyed by output name.

    Returns
    -------
    Array
        The unnormalized log-density, shaped as the batch axes.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one.
    """
    return _score(self._graph, _given_values(self, given), value, "_unnormalized_log_prob")


def _joint_conditional_mean(self: Any, given: Record | Mapping[str, Any]) -> dict[str, Any]:
    """The mean of an edge-free joint at *given*, each output's from its factor.

    Parameters
    ----------
    self : FactoredConditionalDistribution
        The edge-free conditional joint whose capability subclass installs this
        function as ``_conditional_mean``.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.

    Returns
    -------
    dict
        Each output's mean, keyed by output name in canonical factor order.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one.
    """
    return _assemble_by_output(self._graph, _given_values(self, given), "_mean")


def _joint_conditional_variance(self: Any, given: Record | Mapping[str, Any]) -> dict[str, Any]:
    """The variance of an edge-free joint at *given*, each output's from its factor.

    Parameters
    ----------
    self : FactoredConditionalDistribution
        The edge-free conditional joint whose capability subclass installs this
        function as ``_conditional_variance``.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.

    Returns
    -------
    dict
        Each output's variance, keyed by output name in canonical factor order.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one.
    """
    return _assemble_by_output(self._graph, _given_values(self, given), "_variance")


def _joint_conditional_cov(self: Any, given: Record | Mapping[str, Any]) -> DenseLinOp:
    """The block-diagonal covariance of an edge-free joint at *given*.

    Parameters
    ----------
    self : FactoredConditionalDistribution
        The edge-free conditional joint whose capability subclass installs this
        function as ``_conditional_cov``.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.

    Returns
    -------
    DenseLinOp
        The ``(d, d)`` covariance, where ``d`` is the size of the flattened draw.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one.
    """
    return _block_diagonal(self._graph, _given_values(self, given))


def _joint_conditional_quantile(
    self: Any, given: Record | Mapping[str, Any], q: ArrayLike
) -> dict[str, Any]:
    """The quantiles of an edge-free joint at *given* and the levels *q*, per output.

    Parameters
    ----------
    self : FactoredConditionalDistribution
        The edge-free conditional joint whose capability subclass installs this
        function as ``_conditional_quantile``.
    given : Record or Mapping[str, Any]
        A value of each required input, and of any optional one, keyed by
        input name.
    q : ArrayLike
        The levels in ``[0, 1]``, of any shape.

    Returns
    -------
    dict
        Each output's quantiles, keyed by output name in canonical factor order.

    Raises
    ------
    KeyError
        If *given* names an input the joint does not have, or omits a required one.
    """
    return _assemble_by_output(self._graph, _given_values(self, given), "_quantile", q)


def _joint_marginal(self: Any, paths: str | tuple[str, ...]) -> Distribution:
    """The exact marginal of the joint at *paths*, detached from the joint.

    A factor whose every output is requested is kept as it is, and a factor
    requested in part is reduced to its own marginal at its requested paths. A
    projection onto one path returns the one factor it keeps or reduces, as a
    whole term: a kept factor in the positional form over a record of that one
    output is returned as the law of its field. A list of paths returns a law
    over the record that keeps each node at its own path, in the joint's order:
    the one kept factor itself when it is positional over a record, and
    otherwise the joint of the kept factors in factor order (III.8). A packaged
    joint's record holds the kept nodes under its output.

    The marginal carries the expression the ``marginal`` operation gives it
    (VI.8). A marginal over the whole events of some factors, none of which
    conditions on an output outside them, is those factors: one factor
    keeps its own label, and several form a product without a label, in
    factor order. Any other marginal is the joint selected at *paths*, which
    keeps the joint's label. The marginal holds the paths the joint holds
    fixed.

    Parameters
    ----------
    self : FactoredDistribution
        The joint whose capability subclass installs this function as
        ``_marginal``.
    paths : str or tuple of str
        An event path of the joint, or a list of them, as
        :func:`~probpipe.core._shapes._as_paths` reads them.

    Returns
    -------
    Distribution
        The marginal, which shares the joint's factors.

    Raises
    ------
    KeyError
        If a path is not an event path of the joint.
    TypeError
        If a path is not a string.
    ResolutionError
        If the marginal guard does not accept *paths*, so no exact marginal is
        available there.
    """
    from ._views import _carrying, _exposed_under, _marginal_expression_at

    output_name = _single_output_name(self.event_spec)
    if paths == output_name:
        return self
    if _is_list_of_the_output(self, paths):
        return _carrying(
            _exposed_under(self, cast(str, output_name)), _marginal_expression_at(self, paths)
        )
    listed_paths = _requested_paths(self, paths)
    report = _marginal_guard(self, paths)
    if report.feasible is not True:
        reason = report.description or "; ".join(report.pending)
        raise ResolutionError(f"{self.label!r} has no exact marginal at {paths!r}: {reason}")

    graph: _FactorGraph = self._graph
    projection = isinstance(paths, str)
    closed = _closed_factors(self, (paths,) if projection else tuple(paths))
    label = self.label if closed is None else _joined_label(part.label for part in closed)
    kept: list[Distribution] = []
    for index, requested in _requests(graph, listed_paths).items():
        factor = graph.factors[index]
        if not _kept_whole(factor, requested):
            kept.append(factor._marginal(_factor_request(requested, list_of_paths=not projection)))
        elif projection and _has_record_outputs(factor.event_spec):
            kept.append(_SoleField(factor))
        else:
            kept.append(factor)
    if len(kept) == 1 and (projection or _has_record_outputs(kept[0].event_spec)):
        marginal = _law_at_defaults(kept[0], ())
    elif closed is None:
        marginal = FactoredDistribution(
            kept,
            label=label,
        )
    else:
        marginal = FactoredDistribution(
            closed,
            label=label,
        )
    if not projection and output_name is not None:
        marginal = _exposed_under(marginal, output_name)
    return _carrying(marginal, _marginal_expression_at(self, paths))


def _closed_factors(d: Any, paths: tuple[Any, ...]) -> list[Factor] | None:
    """The factors of *d* producing the outputs *paths* name, if none conditions outside them.

    A marginal over these outputs is the product of the factors (VI.8),
    which lists them in factor order. An optional input that no factor of *d*
    produces takes its default, so it conditions on nothing.

    Parameters
    ----------
    d : Any
        A law; only one with ``factors`` has closed factors.
    paths : tuple
        The requested paths.

    Returns
    -------
    list of Factor or None
        The closed factors in factor order, or None when *d* has no factors, a
        path is not a whole output, the outputs split a factor's event,
        or a factor conditions on an output outside them.
    """
    parts = getattr(d, "factors", None)
    wanted = set(paths)
    if not parts or not all(isinstance(path, str) and _PATH_SEP not in path for path in wanted):
        return None
    selected = [part for part in parts if wanted & set(part.event_spec.outputs)]
    produced = {output_name for part in selected for output_name in part.event_spec.outputs}
    if produced != wanted:
        return None
    outputs_of_d = set(d.event_spec.outputs)
    for part in selected:
        if not isinstance(part, ConditionalDistribution):
            continue
        given = part.given_spec
        conditioned = [
            input_name
            for input_name in given
            if input_name in outputs_of_d or input_name not in given.optional
        ]
        if any(input_name not in wanted for input_name in conditioned):
            return None
    return selected


def _kept_claims(
    factor: Factor, requested: tuple[str, ...], *, list_of_paths: bool
) -> frozenset[type]:
    """The claims of *factor* as a marginal keeps it: whole, or reduced to *requested*.

    A conditional factor's claims are those of the law it yields at a given
    value, and a factor reduced to part of its event reports its own marginal
    there, at the one path of a projection or at the list of paths when
    *list_of_paths*.
    """
    if isinstance(factor, ConditionalDistribution):
        return _kernel_claims(factor)
    if _kept_whole(factor, requested):
        return _claims(factor)
    return _marginal_claims(factor, _factor_request(requested, list_of_paths=list_of_paths))


def _joint_marginal_capabilities(self: Any, paths: str | tuple[str, ...]) -> frozenset[type]:
    """The claims of the factors that the marginal at *paths* keeps, read from their declarations.

    One kept factor reports its own claims, since the marginal is that factor
    or its reduction, and a packaged joint's marginal at its output is the
    joint. Several report what the joint of them claims: sampling
    and each density when every kept factor has it, a moment when no kept
    factor conditions on another and every one has the moment, and marginals.

    Parameters
    ----------
    self : FactoredDistribution
        The joint whose capability subclass installs this function as
        ``_marginal_capabilities``.
    paths : str or tuple of str
        An event path of the joint, or a list of them.

    Returns
    -------
    frozenset of type
        The capability protocols the marginal at *paths* claims.

    Raises
    ------
    KeyError
        If a path is not an event path of the joint.
    TypeError
        If a path is not a string.
    """
    if paths == _single_output_name(self.event_spec) or _is_list_of_the_output(self, paths):
        return _claims(self)
    graph: _FactorGraph = self._graph
    requests = _requests(graph, _requested_paths(self, paths))
    list_of_paths = not isinstance(paths, str)
    reports = [
        _kept_claims(graph.factors[index], requested, list_of_paths=list_of_paths)
        for index, requested in requests.items()
    ]
    if len(reports) == 1:
        return reports[0]

    def every(protocol: type) -> bool:
        return all(protocol in report for report in reports)

    claims = {SupportsMarginals}
    if every(SupportsSampling):
        claims.add(SupportsSampling)
    if every(SupportsLogProb):
        claims |= {SupportsLogProb, SupportsUnnormalizedLogProb}
    elif every(SupportsUnnormalizedLogProb):
        claims.add(SupportsUnnormalizedLogProb)
    if not any(
        consumer in requests and producer in requests for consumer, producer, _ in graph.edges
    ):
        claims.update(
            moment
            for moment in (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)
            if every(moment)
        )
    return frozenset(claims)


def _all_claim(factors: Sequence[Factor], protocol: type) -> bool:
    """Whether every factor claims *protocol*, a conditional factor through its twin."""
    twin = _CONDITIONAL_TWINS[protocol]
    return all(
        isinstance(factor, twin if isinstance(factor, ConditionalDistribution) else protocol)
        for factor in factors
    )


#: The method each unconditional capability of a joint names.
_JOINT_METHODS: dict[type, str] = {
    SupportsSampling: "_sample",
    SupportsLogProb: "_log_prob",
    SupportsUnnormalizedLogProb: "_unnormalized_log_prob",
    SupportsMean: "_mean",
    SupportsVariance: "_variance",
    SupportsCovariance: "_cov",
    SupportsQuantile: "_quantile",
}

#: The function that realizes each method a joint may claim, keyed by method name.
_JOINT_IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "_sample": _joint_sample,
    "_log_prob": _joint_log_prob,
    "_unnormalized_log_prob": _joint_unnormalized_log_prob,
    "_mean": _joint_mean,
    "_variance": _joint_variance,
    "_cov": _joint_cov,
    "_quantile": _joint_quantile,
    "_conditional_sample": _joint_conditional_sample,
    "_conditional_log_prob": _joint_conditional_log_prob,
    "_conditional_unnormalized_log_prob": _joint_conditional_unnormalized_log_prob,
    "_conditional_mean": _joint_conditional_mean,
    "_conditional_variance": _joint_conditional_variance,
    "_conditional_cov": _joint_conditional_cov,
    "_conditional_quantile": _joint_conditional_quantile,
    "_marginal": _joint_marginal,
}


def _factors_guard(owner: str, joint_method: str, method: str) -> Callable[[Any], Feasibility]:
    """The guard of a joint's *joint_method*, which calls each factor's *method*.

    A conditional factor is called through its twin, ``_conditional`` followed by
    *method*, and the joint's call is feasible when every factor's is.
    """

    def guard(self: Any) -> Feasibility:
        return _conjunction(
            _capability_guard(
                factor,
                f"_conditional{method}" if isinstance(factor, ConditionalDistribution) else method,
            )
            for factor in self._graph.factors
        )

    guard.__name__ = f"{joint_method}_guard"
    guard.__qualname__ = f"{owner}.{joint_method}_guard"
    guard.__doc__ = f"Every factor's guard of the ``{method}`` the joint calls on it."
    return guard


def _joint_table(owner: str, *, conditional: bool) -> dict[type, Mapping[str, Callable[..., Any]]]:
    """Each capability a joint of the *owner* kind may claim, with its methods and guards.

    A conditional joint claims each capability's conditional twin, whose method
    is ``_conditional`` followed by the unconditional method's name. Each derived
    method carries the conjunction of the factors' guards.
    """
    table: dict[type, Mapping[str, Callable[..., Any]]] = {}
    for protocol, method in _JOINT_METHODS.items():
        joint_protocol, joint_method = protocol, method
        if conditional:
            joint_protocol, joint_method = _CONDITIONAL_TWINS[protocol], f"_conditional{method}"
        table[joint_protocol] = {
            joint_method: _JOINT_IMPLEMENTATIONS[joint_method],
            f"{joint_method}_guard": _factors_guard(owner, joint_method, method),
        }
    if not conditional:
        table[SupportsMarginals] = {
            "_marginal": _JOINT_IMPLEMENTATIONS["_marginal"],
            "_marginal_guard": _marginal_guard,
            "_marginal_capabilities": _joint_marginal_capabilities,
        }
    return table


def _joint_protocols(graph: _FactorGraph, *, conditional: bool) -> set[type]:
    """The capabilities a joint of *graph* claims, decided at construction.

    Sampling and the densities are the intersection of the factors'. The moments
    are claimed by an edge-free joint whose every factor has them; a dependent
    joint claims no moment, since factor-wise conditional moments do not compose.
    The unconditional joint claims marginals, guarded per path.
    """
    factors = graph.factors
    claimed: set[type] = set()
    if _all_claim(factors, SupportsSampling):
        claimed.add(SupportsSampling)
    if _all_claim(factors, SupportsLogProb):
        claimed.add(SupportsLogProb)
    elif _all_claim(factors, SupportsUnnormalizedLogProb):
        claimed.add(SupportsUnnormalizedLogProb)
    if not graph.edges:
        claimed.update(
            moment
            for moment in (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)
            if _all_claim(factors, moment)
        )
    if conditional:
        return {_CONDITIONAL_TWINS[protocol] for protocol in claimed}
    return claimed | {SupportsMarginals}


def _rebuilt(joint: Any, method: str, mapping: Mapping[str, Any], *, free: Any = None) -> Any:
    """*joint* rebuilt from its factors with *method* applied, recording the transform.

    Parameters
    ----------
    joint : FactoredDistribution or FactoredConditionalDistribution
        The joint to rebuild.
    method : str
        The dimension transform each factor applies, ``"with_dim_sizes"`` or
        ``"with_dim_names"``.
    mapping : Mapping[str, Any]
        The transform's keyword arguments, keyed by dimension name: a size or a
        new name.
    free : frozenset of str, optional
        The joint's free dimensions, which must include every key of *mapping*
        when given.

    Returns
    -------
    FactoredDistribution or FactoredConditionalDistribution
        A joint of the class of *joint* over the transformed factors, packaged as
        *joint* is, whose provenance records *method* and *mapping*.

    Raises
    ------
    ValueError
        If *free* is given and *mapping* names a dimension outside it.
    """
    if free is not None:
        unbound = set(mapping) - set(free)
        if unbound:
            raise ValueError(_no_free_dims(joint, unbound, free))
    base = vars(type(joint)).get("_capability_base", type(joint))
    scope = dict(joint._graph.scope)
    if method == "with_dim_sizes":
        scope.update(mapping)
    rebuilt = base(
        _each_factor(joint.factors, method, mapping),
        label=joint.label,
        _scope=scope,
        **_form_keywords(joint),
    )
    return _derived_product(rebuilt, joint).with_provenance(
        Provenance.create(method, parents=[joint], metadata=dict(mapping))
    )


def _each_factor(
    factors: Sequence[Factor], method: str, mapping: Mapping[str, Any]
) -> list[Factor]:
    """Each factor with *method* applied to the entries of *mapping* it declares."""
    result: list[Factor] = []
    for factor in factors:
        subset = {dim: value for dim, value in mapping.items() if dim in _free_dims(factor)}
        result.append(getattr(factor, method)(**subset) if subset else factor)
    return result


# ---------------------------------------------------------------------------
# Refinements
# ---------------------------------------------------------------------------

#: Each registered refinement of the factored law: a more specific class and the
#: predicate over the flattened factors that admits it, in registration order.
_REFINEMENTS: list[tuple[type, Callable[[tuple[Factor, ...]], bool]]] = []


def _register_refinement(cls: type, predicate: Callable[[tuple[Factor, ...]], bool]) -> None:
    """Register *cls* as the class of an unconditional joint whose factors satisfy *predicate*.

    A family registers its factored class at import. ``*``, a joint rebuilt by a
    transform, and a conditional joint bound at its givens construct the most
    specific registered class whose predicate holds for the flattened factors.

    Parameters
    ----------
    cls : type
        A subclass of :class:`FactoredDistribution`, such as a family's factored
        class.
    predicate : callable
        The test that admits *cls*, called with the tuple of flattened factors.

    Raises
    ------
    TypeError
        If *cls* is not a subclass of :class:`FactoredDistribution`.
    """
    if not (isinstance(cls, type) and issubclass(cls, FactoredDistribution)):
        raise TypeError(f"a refinement must be a subclass of FactoredDistribution; got {cls!r}")
    _REFINEMENTS.append((cls, predicate))


def _refined_class(factors: tuple[Factor, ...]) -> type:
    """The most specific registered class whose predicate holds for *factors*.

    A candidate replaces the class chosen so far when it subclasses it, so of
    two unrelated candidates the earlier registration stands; with none, the
    class is :class:`FactoredDistribution`.
    """
    chosen: type = FactoredDistribution
    for cls, holds in _REFINEMENTS:
        if issubclass(cls, chosen) and holds(factors):
            chosen = cls
    return chosen


# ---------------------------------------------------------------------------
# The factored kinds
# ---------------------------------------------------------------------------


class FactoredDistribution(Distribution, SupportsFactors):
    """A joint distribution built from an ordered list of factors, every given met.

    The factors are ``Distribution``s and ``ConditionalDistribution``s in
    conditional-first order: a factor may condition on outputs that factors
    to its right produce. The joint's event spec is the positional form over
    the record of every factor's outputs in factor order, each factor's own
    order kept, and extraction and reconstruction keep each factor's form. Each output
    is produced by exactly one factor, while factor labels may repeat. A
    packaged joint, which a rename that gathers outputs builds, declares
    that record as a whole term under one output, and it is one factor of a
    joint it enters.

    Sampling and the log-density capabilities are the intersection of the
    factors'. The moment capabilities are decided at construction: an edge-free
    joint has a moment exactly when every factor does, and a dependent joint has
    none. The marginal is resolved per path, and its guard reports whether the
    marginal at a path is exact. Constructing the class itself gives the most
    specific registered refinement whose predicate the factors satisfy, as a
    family registers its factored class.

    A draw is a mapping from each output to its raw value, in canonical
    factor order, and so is each event-typed moment. Sampling is ancestral: the
    factors draw from right to left, each conditional factor at the outputs
    it names. Scoring sums the factors' densities, each at its own event
    reconstructed from the outputs.

    A joint constructed by this class is labeled, so its :attr:`notation` is
    its label followed by its signature, as ``model(y, mu)``. A joint that
    ``*`` or the ``joint`` operation builds is unlabeled, and its notation joins
    its factors' notations with ``·``, as ``lik(y | mu)·prior(mu)``, until
    ``with_label`` gives it a label. A law built from a joint, such as a copy,
    a packaged sub-joint, or a marginal that reduces a factor to part of its
    event, is labeled when the joint is. A marginal over the whole events of
    several factors is a product without a label, as ``a(a)·b(b)``, and a
    product that a function returns takes the function's output label. An
    unlabeled product that holds paths fixed reads as its grouped label
    followed by its signature, as ``(lik·prior)(mu; y)``.

    Parameters
    ----------
    factors : Sequence[Distribution | ConditionalDistribution]
        The factors, in conditional-first order.
    label : str, optional
        The joint's display alias. Defaults to the product of its factors' descriptions.
    _scope : Mapping[str, int], optional
        The size of each dimension bound in the joint that a transform rebuilds,
        keyed by its name, which the factor graph starts from.
    _output_name : str, optional
        The name of the output under which a packaged joint declares its record
        as a whole term. By default the joint's outputs are the record's fields.

    Raises
    ------
    TypeError
        If a factor is neither distribution kind.
    ValueError
        If there are no factors, an output is produced twice, a factor
        conditions on an output a factor to its left produces, matched specs
        do not unify, or a given is left unmet, which makes the joint a
        ``FactoredConditionalDistribution``.
    """

    #: The constructor takes no output name, so the repr shows none.
    _repr_output_name: ClassVar[str | None] = None

    _capability_table = _joint_table("FactoredDistribution", conditional=False)

    def __new__(
        cls,
        factors: Sequence[Factor],
        *,
        label: str | None = None,
        _scope: Mapping[str, int] | None = None,
        _output_name: str | None = None,
    ) -> FactoredDistribution:
        graph = _factor_graph(factors, _scope)
        base = vars(cls).get("_capability_base", cls)
        if base is FactoredDistribution:
            base = _refined_class(graph.factors)
        return object.__new__(
            _capability_subclass(base, _joint_protocols(graph, conditional=False))
        )

    def __init__(
        self,
        factors: Sequence[Factor],
        *,
        label: str | None = None,
        _scope: Mapping[str, int] | None = None,
        _output_name: str | None = None,
    ) -> None:
        expression = _product_of(factors) if label is None else None
        rendering = None
        if expression is not None:
            rendering = expression.label_rendering()
            label = rendering[0]
        graph = _factor_graph(factors, _scope)
        if graph.unmet is not None:
            raise ValueError(
                f"no factor of {label!r} defines {sorted(graph.unmet)}, which its factors "
                f"condition on; add a factor for them, or build a "
                f"FactoredConditionalDistribution"
            )
        super().__init__(
            _joint_declaration(graph, _output_name),
            label=label,
        )
        object.__setattr__(self, "_graph", graph)
        if expression is not None:
            self._store_expression(expression, rendering)

    @property
    def factors(self) -> tuple[Factor, ...]:
        """The factors, in conditional-first order."""
        return self._graph.factors

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The factors, in conditional-first order."""
        return [("factors", sequence_repr(repr(factor) for factor in self.factors))]

    def _derived_label(self) -> str:
        """The label the constructor gives the factors the repr shows: their labels joined with ``·``."""
        return _product_of(self.factors).render_label()

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Bind named symbolic dimensions in every factor that declares them.

        The result is this joint with its dimensions bound, so a lift draws it
        together with this joint (V.5).

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of the joint.

        Returns
        -------
        Self
            The joint of the bound factors, under the same label.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the joint.
        """
        rebuilt = _rebuilt(self, "with_dim_sizes", sizes, free=self.event_spec.term_spec.free_dims)
        return _recorded_copy(rebuilt, self)

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions in every factor, simultaneously.

        The result is this joint under the new dimension names, so a lift draws
        it together with this joint (V.5).

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        Self
            The joint of the renamed factors, under the same label.
        """
        return _recorded_copy(_rebuilt(self, "with_dim_names", names), self)


class FactoredConditionalDistribution(ConditionalDistribution, SupportsFactors):
    """A joint conditional distribution: factors that leave some givens unmet.

    Its ``given_spec`` is exactly the unmet givens, same-named ones unified into
    one input, and its event spec is read as for
    :class:`FactoredDistribution`. Conditioning on every given yields a
    ``FactoredDistribution``, and conditioning on some returns a smaller
    ``FactoredConditionalDistribution``.

    Each conditional capability is its unconditional counterpart's on
    :class:`FactoredDistribution`, with every unmet given set by the ``given``
    of the call, and each factor is called through its own capability at that
    value.

    The joint is labeled or unlabeled as :class:`FactoredDistribution` states,
    so an unlabeled one reads factor by factor, as ``lik(y | mu)·prior(mu | tau)``,
    and a labeled one by its label, as ``model(y, mu | tau)``.

    Parameters
    ----------
    factors : Sequence[Distribution | ConditionalDistribution]
        The factors, in conditional-first order.
    label : str, optional
        The joint's display alias. Defaults to the product of its factors' descriptions.
    _scope : Mapping[str, int], optional
        The size of each dimension bound in the joint that a transform rebuilds,
        keyed by its name, which the factor graph starts from.
    _output_name : str, optional
        The name of the output under which a packaged joint declares its record
        as a whole term. By default the joint's outputs are the record's fields.

    Raises
    ------
    TypeError
        If a factor is neither distribution kind.
    ValueError
        If the factors break a composition rule, as for
        :class:`FactoredDistribution`, or meet every given, which makes the
        joint a ``FactoredDistribution``.
    """

    #: The constructor takes no output name, so the repr shows none.
    _repr_output_name: ClassVar[str | None] = None

    _capability_table = _joint_table("FactoredConditionalDistribution", conditional=True)

    def __new__(
        cls,
        factors: Sequence[Factor],
        *,
        label: str | None = None,
        _scope: Mapping[str, int] | None = None,
        _output_name: str | None = None,
    ) -> FactoredConditionalDistribution:
        protocols = _joint_protocols(_factor_graph(factors, _scope), conditional=True)
        base = vars(cls).get("_capability_base", cls)
        return object.__new__(_capability_subclass(base, protocols))

    def __init__(
        self,
        factors: Sequence[Factor],
        *,
        label: str | None = None,
        _scope: Mapping[str, int] | None = None,
        _output_name: str | None = None,
    ) -> None:
        expression = _product_of(factors) if label is None else None
        rendering = None
        if expression is not None:
            rendering = expression.label_rendering()
            label = rendering[0]
        graph = _factor_graph(factors, _scope)
        if graph.unmet is None:
            raise ValueError(
                f"the factors of {label!r} define every field they condition on; build a "
                f"FactoredDistribution instead"
            )
        super().__init__(
            graph.unmet,
            _joint_declaration(graph, _output_name),
            label=label,
        )
        object.__setattr__(self, "_graph", graph)
        if expression is not None:
            self._store_expression(expression, rendering)

    @property
    def factors(self) -> tuple[Factor, ...]:
        """The factors, in conditional-first order."""
        return self._graph.factors

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The factors, in conditional-first order."""
        return [("factors", sequence_repr(repr(factor) for factor in self.factors))]

    def _derived_label(self) -> str:
        """The label the constructor gives the factors the repr shows: their labels joined with ``·``."""
        return _product_of(self.factors).render_label()

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Bind named symbolic dimensions in every factor that declares them.

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of either side.

        Returns
        -------
        Self
            The joint of the bound factors, under the same label.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the joint.
        """
        return _rebuilt(self, "with_dim_sizes", sizes, free=self.spec.free_dims)

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions in every factor, simultaneously.

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        Self
            The joint of the renamed factors, under the same label.
        """
        return _rebuilt(self, "with_dim_names", names)

    def _condition_on(
        self, given: Record | Mapping[str, Any], /, **options: Any
    ) -> Distribution | ConditionalDistribution:
        """Bind inputs in every factor that names them, and rebuild the joint.

        Parameters
        ----------
        given : Record or Mapping[str, Any]
            Values for some or all of the joint's inputs, by input name;
            every given value arrives here.
        **options : Any
            Options for the primitive of each factor that a value binds.

        Returns
        -------
        FactoredDistribution or FactoredConditionalDistribution
            The joint over the bound factors, conditional while a given remains,
            packaged as this joint is.

        Raises
        ------
        KeyError
            If a bound name is not an input of the joint.
        ValueError
            If a factor's primitive returns a law or kernel whose input and event
            specs do not match the factor's.
        """
        top = given.children if hasattr(given, "children") else given
        values = dict(top.items())
        unknown = set(values) - set(self.given_spec)
        if unknown:
            raise KeyError(_unknown_inputs(self.label, sorted(unknown), self.given_spec))
        factors: list[Factor] = []
        for factor in self.factors:
            if isinstance(factor, ConditionalDistribution):
                bound = {
                    input_name: value
                    for input_name, value in values.items()
                    if input_name in factor.given_spec
                }
                if bound:
                    factor = _bound_factor(factor, bound, options)
            factors.append(factor)
        if set(self.given_spec.required) <= set(values):
            joint = FactoredDistribution(
                factors,
                **_form_keywords(self),
                label=self.label,
            )
        else:
            joint = FactoredConditionalDistribution(
                factors,
                **_form_keywords(self),
                label=self.label,
            )
        return _derived_product(joint, self)


def _bound_factor(
    factor: ConditionalDistribution, bound: Mapping[str, Any], options: Mapping[str, Any]
) -> Factor:
    """*factor* conditioned on *bound* under *options*, checked to keep its input and event specs.

    Parameters
    ----------
    factor : ConditionalDistribution
        A kernel factor of the joint.
    bound : Mapping[str, Any]
        Values of some or all of the factor's inputs, keyed by input name.
    options : Mapping[str, Any]
        The options passed to the factor's ``_condition_on``.

    Returns
    -------
    Distribution or ConditionalDistribution
        The result of the factor's ``_condition_on``: a law when *bound* covers
        every required input, and otherwise a kernel over the other inputs.

    Raises
    ------
    ValueError
        If the primitive returns a law or kernel whose event spec, or whose
        remaining inputs, differ from the factor's. Once the required inputs
        are bound, the result is a law.
    """
    result = factor._condition_on(bound, **options)
    remaining = factor.given_spec.without(*bound)
    expected = (
        ConditionalDistributionSpec(remaining, factor.event_spec)
        if remaining.required
        else DistributionSpec(factor.event_spec)
    )
    if not expected.is_valid(result):
        raise ValueError(
            f"{factor.label!r} conditioned on {sorted(bound)} returned "
            f"{type(result).__name__} {getattr(result, 'label', '')!r}, whose input and event "
            f"specs do not match the factor's"
        )
    return result


# ---------------------------------------------------------------------------
# The numeric markers
# ---------------------------------------------------------------------------


class FactoredNumericDistribution(FactoredDistribution):
    """The marker of an unconditional joint whose event is numeric.

    Membership is read from the event spec, as for ``NumericDistribution``, and
    a class that inherits the marker claims it for every instance, which
    construction checks.
    """

    _membership_follows_declaration = True


class FactoredConditionalNumericDistribution(FactoredConditionalDistribution):
    """The marker of a conditional joint whose event is numeric, read from its event spec."""

    _membership_follows_declaration = True


class FactoredNumericConditionalDistribution(FactoredConditionalDistribution):
    """The marker of a conditional joint whose inputs are numeric, by its input spec."""

    _membership_follows_declaration = True


class FactoredFullyNumericConditionalDistribution(
    FactoredNumericConditionalDistribution, FactoredConditionalNumericDistribution
):
    """The marker of a conditional joint whose given and event sides are both numeric."""

    _membership_follows_declaration = True


def _is_factored_conditional(value: Any) -> bool:
    return isinstance(value, FactoredConditionalDistribution)


_DECLARATION_MARKERS[FactoredNumericDistribution] = (
    lambda value: isinstance(value, FactoredDistribution) and _declares_numeric_event(value),
    "a numeric event",
)
_DECLARATION_MARKERS[FactoredConditionalNumericDistribution] = (
    lambda value: _is_factored_conditional(value) and _event_is_numeric(value),
    "a numeric event",
)
_DECLARATION_MARKERS[FactoredNumericConditionalDistribution] = (
    lambda value: _is_factored_conditional(value) and _given_side_is_numeric(value),
    "numeric inputs",
)
_DECLARATION_MARKERS[FactoredFullyNumericConditionalDistribution] = (
    lambda value: (
        _is_factored_conditional(value)
        and _event_is_numeric(value)
        and _given_side_is_numeric(value)
    ),
    "numeric inputs and a numeric event",
)
