"""The target of an inference method: the law ``condition_on`` asks a method to normalize.

``condition_on`` hands each inference method one :class:`InferenceTarget`, and a
method reads the model, the observed values, and the model's prior and
likelihood through it. ``_UnnormalizedConditional`` is the one implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, ClassVar, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp

from ..core._execution_claims import Traceability
from ..core._numeric_record import _reconstruct_from_vector
from ..core._record_spec import NumericRecordSpec
from ..core._repr import format_names
from ..core._spec_base import NumericArraySpec
from ..core._specs import OutputSpec, _has_record_outputs
from ..core.provenance import Provenance
from ..core.record import Record
from ..custom_types import Array
from ..distributions._capabilities import SupportsUnnormalizedLogProb, _capability_subclass
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..distributions._factored import (
    FactoredConditionalDistribution,
    FactoredDistribution,
    _event_of,
    _factor_graph,
    _is_named,
    _law_at_defaults,
    _with_named,
)

__all__ = ["InferenceTarget", "ModelFactors"]


@runtime_checkable
class InferenceTarget(Protocol):
    """The law an inference method normalizes: a model's parameters at its observed values.

    ``condition_on`` hands every inference method one target. The target is the
    law of the quantities left unobserved, called the parameters below, and its
    unnormalized log-density at a value of the parameters is the model's joint
    log-density at that value and the observed values. A method returns a
    normalized law over the target's event, usually through
    :func:`~probpipe.inference.make_posterior`.

    The observed values take one of three forms, which :attr:`given` holds:

    1. **Values of the model's fields.** ``condition_on(model, {"y": y})``
       gives a target whose ``given`` is a :class:`~probpipe.Record` keyed by
       the fields it fixes, and whose event is the model's other fields.
    2. **Data the model takes but does not declare as fields**, such as the
       data block of a program. ``given`` is that data, as the model takes it.
    3. **Nothing.** A law that is itself unnormalized, such as one built from a
       log-density alone, is the target of conditioning on nothing. ``given``
       is an empty ``Record``, and :attr:`joint` is that law.

    A target is a law: it has the event spec, the label, and the capabilities
    of a :class:`~probpipe.Distribution`. It has an unnormalized density, read
    through :func:`~probpipe.unnormalized_log_prob`, exactly when the model has
    one, so a method states the capabilities it needs in its ``check``: a
    method that evaluates a density requires
    :class:`~probpipe.SupportsUnnormalizedLogProb`, and one that simulates
    reads :meth:`factors`.

    A method that works on vectors uses the ``to_vector`` layout of the
    event's values: ``NumericRecord.from_vector(target.event_spec.term_spec,
    vec, label=...)`` reads a vector as a value of the parameters, and the
    chains :func:`~probpipe.inference.make_posterior` takes hold the
    ``to_vector`` coordinates of draws.

    A target is immutable: a method reads it and never changes it.

    Examples
    --------
    A method that weights draws of the prior by the likelihood reads the
    target's factors, and one that runs a chain evaluates its density at a
    vector::

        factors = target.factors()          # None when the model has no prior and likelihood
        spec = target.event_spec.term_spec  # a NumericRecordSpec, whose vector_size is d
        value = NumericRecord.from_vector(spec, vec, label="theta")
        unnormalized_log_prob(target, value)
    """

    @property
    def label(self) -> str:
        """The target's label, which is the model's."""
        ...

    @property
    def event_spec(self) -> OutputSpec:
        """The output spec of one value of the parameters: the event of the law a method returns."""
        ...

    @property
    def joint(self) -> Distribution:
        """The model: the joint law of the parameters and the observed quantities."""
        ...

    @property
    def given(self) -> Any:
        """The observed values: a ``Record`` of fields, the data a model takes, or an empty ``Record``."""
        ...

    def factors(self) -> ModelFactors | None:
        """The model's prior, likelihood, and observed value, or ``None`` when it has none.

        A model has them when it is a merge of factors, such as
        ``likelihood & prior``, conditioned on every field the likelihood
        factors produce and on no field of the prior factors.

        Returns
        -------
        ModelFactors or None
            The prior, the likelihood, and the observed value of the
            likelihood's event; ``None`` for any other model.
        """
        ...


class ModelFactors(NamedTuple):
    """The prior and the likelihood of a model, and the observed value of the likelihood's event.

    :meth:`InferenceTarget.factors` returns it for a model merged from factors,
    such as ``likelihood & prior``, conditioned on the fields its likelihood
    produces.

    Attributes
    ----------
    prior : Distribution
        The law of the parameters: the merge of the factors that produce no
        observed field.
    likelihood : Distribution or ConditionalDistribution
        The law of the observed fields given the parameters: the merge of the
        factors that produce them, a kernel whose inputs are fields of the
        prior, or a law when it reads none.
    observed : Any
        The observed value of the likelihood's event.
    """

    prior: Distribution
    likelihood: Distribution | ConditionalDistribution
    observed: Any


# ---------------------------------------------------------------------------
# The implementation
# ---------------------------------------------------------------------------


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

    It is the one implementation of :class:`InferenceTarget`, and the target
    every inference method receives. Its unnormalized log-density at a value of
    the unconditioned fields is the joint's at that value and the given values.
    It claims ``SupportsUnnormalizedLogProb`` when the joint claims an
    unnormalized density, and no normalized capability, so it is unnormalized.

    Lifecycle: ``condition_on``'s exact stage builds it when no exact route
    conditions a produced field, and the inference-method registry builds the
    target of conditioning on nothing, :meth:`_conditioned_on_nothing`, for a
    law the registry receives as it is. Conditioning it again on further fields
    gives the conditional of the same joint at every given value.

    Invariants: it never changes after it is built, and its traceability is
    its joint's, since its density reads the joint's.

    Parameters
    ----------
    joint : Distribution
        The law conditioned.
    given : Record or Any
        The values of the conditioned fields, keyed by their outputs, an
        empty ``Record`` for conditioning on nothing, or, when *keyed* is false,
        the data of a joint that does not declare them as fields, such as a
        model conditioned through ``probpipe.condition_on``.
    event_spec : OutputSpec
        The event spec of the unconditioned fields.
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
        super().__init__(
            event_spec,
            label=joint.label,
        )
        self._joint = joint
        self._given = given
        self._keyed = keyed
        # Conditioning on nothing is the identity, so that target is its joint
        # and records no condition_on of its own (_lineage_term).
        if not self.conditions_on_nothing:
            self.with_provenance(
                Provenance.create(
                    "condition_on",
                    parents=[joint],
                    metadata={"stage": "exact", "route": "inference_methods"},
                )
            )

    @classmethod
    def _conditioned_on_nothing(cls, law: Distribution) -> _UnnormalizedConditional:
        """The target of normalizing *law* itself: *law* at no given values."""
        return cls(law, Record({}, label="given"), law.event_spec)

    def _derived_traceability(self) -> Traceability:
        """The joint's traceability, whose density this law reads at the given values."""
        return self._joint._traceability()

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

    @property
    def conditions_on_nothing(self) -> bool:
        """Whether the target is its joint at no given values."""
        return self._keyed and not self._given.fields

    def factors(self) -> ModelFactors | None:
        """The model's prior, likelihood, and observed value; see :meth:`InferenceTarget.factors`.

        The likelihood is the merge of the factors that produce the observed
        fields, and the prior the merge of the others. ``None`` when the target
        is not keyed by fields, when the joint has no factors, when a factor
        produces observed and unobserved fields both, when the prior is not a
        law, or when the likelihood reads anything but the prior's fields.
        """
        joint, given = self._joint, self._given
        factor_laws = getattr(joint, "factors", None)
        if not self._keyed or not factor_laws:
            return None
        observed_fields = set(given.fields)
        prior_factors: list[Any] = []
        likelihood_factors: list[Any] = []
        for factor in factor_laws:
            produced = set(factor.event_spec.outputs)
            if not produced & observed_fields:
                prior_factors.append(factor)
            elif produced <= observed_fields:
                likelihood_factors.append(factor)
            else:
                return None
        if not prior_factors or not likelihood_factors:
            return None
        prior = _law_at_defaults(_merge_of(joint, prior_factors), joint.event_spec.outputs)
        likelihood = _merge_of(joint, likelihood_factors)
        if not isinstance(prior, Distribution):
            return None
        likelihood_inputs = (
            set(likelihood.given_spec.required)
            if isinstance(likelihood, ConditionalDistribution)
            else set()
        )
        if not likelihood_inputs <= set(prior.event_spec.outputs):
            return None
        given_values = dict(given.children)
        observed_value = _event_of(
            likelihood.event_spec,
            {
                output_name: given_values[output_name]
                for output_name in likelihood.event_spec.outputs
            },
        )
        return ModelFactors(prior, likelihood, observed_value)

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The law conditioned, and the fields its given values fix where they are keyed."""
        fields = [("joint", repr(self._joint))]
        if self._keyed and self._given.fields:
            fields.append(("conditioned", format_names(self._given.keys())))
        return fields


def _lineage_term(target: Any) -> Any:
    """The term a result of *target* names in its provenance: the law itself for conditioning on nothing."""
    if isinstance(target, _UnnormalizedConditional) and target.conditions_on_nothing:
        return target.joint
    return target


def _as_inference_target(law: Distribution) -> _UnnormalizedConditional:
    """*law* as an inference target: itself when it is one, else the target of conditioning it on nothing."""
    if isinstance(law, _UnnormalizedConditional):
        return law
    return _UnnormalizedConditional._conditioned_on_nothing(law)


# ---------------------------------------------------------------------------
# The flat layout
# ---------------------------------------------------------------------------


def _flat_record_spec(law: Any) -> NumericRecordSpec | None:
    """The spec of the numeric record a position over *law* lays out, or None when its outputs are no record's fields."""
    output_spec = getattr(law, "event_spec", None)
    if not isinstance(output_spec, OutputSpec) or not _has_record_outputs(output_spec):
        return None
    spec = output_spec.term_spec
    return spec if isinstance(spec, NumericRecordSpec) else None


def _flat_unflatten(law: Any) -> Callable[[Array], Any]:
    """The map from a position in the flat layout to a draw of the numeric *law*.

    A draw of one array is the position reshaped to the event, and a draw of a
    numeric record whose fields are the law's outputs is the record whose
    leaves the position lays out in canonical order. Positions may be stacked
    on leading axes for a record; the last axis holds one position.

    Parameters
    ----------
    law : Distribution
        The law whose draws the positions lay out.

    Returns
    -------
    callable
        The map, whose record draws carry *law*'s label. It raises
        ``ValueError`` for a position whose last axis is not the layout's size.

    Raises
    ------
    TypeError
        If *law* draws neither one numeric array of known shape nor a numeric
        record whose fields are its outputs.
    """
    array_spec = _one_array_spec(law)
    if array_spec is not None:
        reshape_position = _reshape_to(array_spec.shape)
        array_size = prod(array_spec.shape)

        def unflatten_array(position: Array) -> Array:
            _check_position_size(law, position, array_size)
            return reshape_position(position)

        return unflatten_array
    record_spec = _flat_record_spec(law)
    if record_spec is None:
        raise TypeError(_no_flat_layout(law))

    def unflatten_record(position: Array) -> Any:
        _check_position_size(law, position, record_spec.vector_size)
        return _reconstruct_from_vector(law.label, record_spec, position)

    return unflatten_record


def _check_position_size(law: Any, position: Array, size: int) -> None:
    """Refuse *position* unless its last axis holds the *size* values of a position over *law*."""
    shape = jnp.shape(position)
    if not shape or shape[-1] != size:
        raise ValueError(
            f"cannot read a position of shape {tuple(shape)} as a draw of {_described(law)}: "
            f"a position holds its {size} parameter values on one axis. Pass an array of "
            f"shape ({size},)"
        )


def _one_array_spec(law: Any) -> NumericArraySpec | None:
    """The spec of the one numeric array of a concrete shape *law* draws, or None."""
    output_spec = getattr(law, "event_spec", None)
    if not isinstance(law, Distribution) or output_spec is None or _has_record_outputs(output_spec):
        return None
    spec = output_spec.term_spec
    return spec if isinstance(spec, NumericArraySpec) and spec.is_concrete else None


def _reshape_to(shape: tuple[int, ...]) -> Callable[[Array], Array]:
    """The map from a position, in row-major order, to an array of *shape*."""

    def reshape_position(position: Array) -> Array:
        return jnp.reshape(position, shape)

    return reshape_position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _described(value: Any) -> str:
    """*value*'s type name and label for a message, such as ``"Normal 'theta'"``.

    An unnormalized conditional, which the caller never builds, is described
    by the joint it conditions.
    """
    if isinstance(value, _UnnormalizedConditional):
        value = value.joint
    label = getattr(value, "label", None)
    name = type(value).__name__
    return f"{name} {label!r}" if isinstance(label, str) else name


def _no_flat_layout(law: Any) -> str:
    """The message for a law whose draws no flat position lays out."""
    return (
        f"cannot lay out the draws of {_described(law)} as a flat position: they must be "
        f"a numeric array or a record of numeric arrays, each of known shape"
    )


def _joint_value(value: Any, given: Record) -> Any:
    """The joint's value at *value*, a draw of the unconditioned fields, and at *given*.

    With no given values the joint's value is *value* itself.
    """
    if not given.fields:
        return value
    fields = value.children if isinstance(value, Record) else value
    return Record({**dict(fields), **dict(given.children)}, label="value")


def _merge_of(joint: Any, factor_laws: list[Any]) -> Any:
    """The merge of *factor_laws* of *joint* under its label, the one factor itself when there is one.

    A merge of several factors is labeled when *joint* is.
    """
    if len(factor_laws) == 1:
        return factor_laws[0]
    if _factor_graph(tuple(factor_laws)).unmet is None:
        merged = FactoredDistribution(
            factor_laws,
            label=joint.label,
        )
    else:
        merged = FactoredConditionalDistribution(
            factor_laws,
            label=joint.label,
        )
    return _with_named(merged, _is_named(joint))
