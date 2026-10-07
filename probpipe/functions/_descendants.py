"""The root-ancestor capture of lifted arguments (V.5).

A lifted argument is a law, and the law whose draws it reads, transitively, is
its **root**. Each of these laws reads another law's draw:

- an element of a batch of laws: its stored law's draw;
- a field view: its parent's draw, projected onto its node;
- every law that ``with_path_names`` returns: the draw of the law it renames,
  moved to the new paths;
- a law of a registered descendant type: its ancestor's draw, mapped as a
  bijector-transformed law pushes its base's draw through its bijector.

A law that renames at its boundary holds the law it renames, and every other
result of ``with_path_names`` records it. The lift groups the arguments by root,
so each group contributes one root draw per repetition and every member
evaluates on it. Hence sibling views co-sample, two accesses of one batch
element co-sample, a law co-samples with its own transform and its own rename,
and the empirical enumeration enumerates a renamed empirical law's atoms as the
law's.

The capture of an argument records its root, the root's sampler, the event
path a projection reads, a canonical descriptor of the descendant graph between
the root and the argument, and the evaluator that maps one root draw to the
argument's draw. It reads the graph once, so a later change to a view or a
transform does not reach a captured plan.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from ..distributions._batches import _element_source
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..distributions._views import FieldView, _projector, _rename_source, _RenamedDistribution

_DISTRIBUTION_SAMPLING_ABI = "probpipe.distribution_sampling/v1"
_DESCRIPTOR_DOMAIN = b"ProbPipe-descendant-descriptor-v1\0"
_PATH_SEP = "/"


@dataclass(frozen=True)
class CapturedStochasticConsumer:
    """One live root plus a canonical and executable descendant path."""

    root: Distribution = field(compare=False, hash=False, repr=False)
    sample_root: Callable[[Any, tuple[int, ...]], Any] = field(
        compare=False,
        hash=False,
        repr=False,
    )
    record_path: tuple[str, ...]
    descendant_descriptor: tuple[Any, ...] | None
    evaluator: Callable[[Any], Any] = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class _Descent:
    """How a law reads another law's draws: the ancestor, the map, and the map's descriptor.

    Attributes
    ----------
    ancestor : Distribution
        The law whose draws the descendant reads.
    forward : Callable[[Any], Any]
        The map from a batch of the ancestor's draws, the batch axes leading, to
        the descendant's draws at them.
    descriptor : tuple
        The canonical descriptor of the map, built from tuples, strings, and
        integers, which identifies it within the plan.
    """

    ancestor: Distribution = field(compare=False, hash=False, repr=False)
    forward: Callable[[Any], Any] = field(compare=False, hash=False, repr=False)
    descriptor: tuple[Any, ...]


#: Each registered descendant type, with the function that gives an instance's
#: descent. A layer above this one registers its types at import.
_DESCENDANT_TYPES: dict[type, Callable[[Any], _Descent]] = {}


def _register_descendant_type(distribution_type: type, descend: Callable[[Any], _Descent]) -> None:
    """Register *distribution_type*, whose instances read the draws *descend* names.

    An instance of a subclass is a descendant by the nearest registered class
    in its method-resolution order.
    """
    _DESCENDANT_TYPES[distribution_type] = descend


def _descent_rule(value: Distribution) -> tuple[type, Callable[[Any], _Descent]] | None:
    """The registered class nearest *value*'s class, with its rule, or None."""
    for klass in type(value).__mro__:
        rule = _DESCENDANT_TYPES.get(klass)
        if rule is not None:
            return klass, rule
    return None


@dataclass(frozen=True)
class _DescriptorAbiSummary:
    """Sorted unique execution ABIs found in one descendant descriptor."""

    sampling_abis: tuple[str, ...]
    provider_abis: tuple[str, ...]
    descendant_adapter_abis: tuple[str, ...]


_EMPTY_DESCRIPTOR_ABI_SUMMARY = _DescriptorAbiSummary((), (), ())


@dataclass(slots=True)
class _StochasticCaptureSession:
    """Call-local identity memo for one stochastic-plan construction."""

    consumers: dict[int, tuple[Distribution, CapturedStochasticConsumer]] = field(
        default_factory=dict
    )
    active_descendants: set[int] = field(default_factory=set)

    def capture_consumer(self, value: Distribution) -> CapturedStochasticConsumer:
        """Capture one consumer, reusing a completed identical object."""
        identity = id(value)
        cached = self.consumers.get(identity)
        if cached is not None:
            source, captured = cached
            if source is not value:
                raise RuntimeError("stochastic consumer identity cache collision")
            return captured

        captured = _capture_stochastic_consumer(value, session=self)
        self.consumers[identity] = (value, captured)
        return captured


def capture_stochastic_consumer(value: Distribution) -> CapturedStochasticConsumer:
    """Capture a law's root and its descendant path without executing either."""
    return _StochasticCaptureSession().capture_consumer(value)


def capture_stochastic_consumers(
    values: Sequence[Distribution],
) -> tuple[CapturedStochasticConsumer, ...]:
    """Capture ordered consumers through one call-local identity memo."""
    session = _StochasticCaptureSession()
    return tuple(session.capture_consumer(value) for value in values)


def sample_captured_consumer(
    captured: CapturedStochasticConsumer,
    key: Any,
    sample_shape: tuple[int, ...],
) -> Any:
    """Sample a captured root once and evaluate its live descendant path."""
    return captured.evaluator(captured.sample_root(key, sample_shape))


def descriptor_digest(descriptor: tuple[Any, ...]) -> str:
    """Return the versioned SHA-256 digest of a canonical descriptor."""
    return hashlib.sha256(canonical_descriptor_bytes(descriptor)).hexdigest()


def _summarize_descriptor_abis(
    descriptor: tuple[Any, ...] | None,
) -> _DescriptorAbiSummary:
    """Collect execution ABIs once without encoding or digesting the descriptor."""
    if descriptor is None:
        return _EMPTY_DESCRIPTOR_ABI_SUMMARY
    if not isinstance(descriptor, tuple):
        raise TypeError("descendant descriptor must be a tuple or None")

    sampling_abis: set[str] = set()
    provider_abis: set[str] = set()
    descendant_adapter_abis: set[str] = set()
    destinations = {
        "sampling_abi": sampling_abis,
        "provider_abi": provider_abis,
        "descendant_adapter_abi": descendant_adapter_abis,
    }

    def collect(value: Any) -> None:
        if not isinstance(value, tuple):
            return
        if len(value) == 2 and isinstance(value[0], str):
            destination = destinations.get(value[0])
            if destination is not None:
                abi = value[1]
                if not isinstance(abi, str):
                    raise TypeError(f"descriptor {value[0]} must be a string")
                if not abi:
                    raise ValueError(f"descriptor {value[0]} must not be empty")
                destination.add(abi)
        for item in value:
            collect(item)

    collect(descriptor)
    return _DescriptorAbiSummary(
        sampling_abis=tuple(sorted(sampling_abis)),
        provider_abis=tuple(sorted(provider_abis)),
        descendant_adapter_abis=tuple(sorted(descendant_adapter_abis)),
    )


def canonical_descriptor_bytes(descriptor: tuple[Any, ...]) -> bytes:
    """Encode descriptor primitives independently of the workflow RNG ABI."""
    return _DESCRIPTOR_DOMAIN + _encode_descriptor_value(descriptor)


def _capture_stochastic_consumer(
    value: Distribution,
    *,
    session: _StochasticCaptureSession,
) -> CapturedStochasticConsumer:
    """The capture of *value* as a batch element, a field view, a renamed law, or a descendant.

    A descendant is a law of a registered descendant type, and a law that is
    none of these is its own root.
    """
    source = _element_source(value)
    if source is not None:
        return _capture_element(value, source, session=session)
    if isinstance(value, FieldView):
        return _capture_field_view(value, session=session)
    if _rename_source(value) is not None:
        # A result of ``with_path_names`` captures as the law that renames at its
        # boundary, so one rename gives one descriptor and one evaluator whatever
        # class the result has.
        return _capture_descendant(value, _RenamedDistribution, _renamed_descent, session=session)
    rule = _descent_rule(value)
    if rule is not None:
        return _capture_descendant(value, *rule, session=session)
    return CapturedStochasticConsumer(
        root=value,
        sample_root=_sampler(value),
        record_path=(),
        descendant_descriptor=None,
        evaluator=_identity,
    )


def _sampler(root: Distribution) -> Callable[[Any, tuple[int, ...]], Any]:
    """The sampler of *root*, which reads ``root._sample`` when it draws.

    An empirical root draws its atoms by :func:`_lift_indices`. A check reads
    the root of a law that does not sample without failing, and the sampling
    lift then declines the law.
    """
    if isinstance(root, EmpiricalDistribution):

        def sample_atoms(key: Any, sample_shape: tuple[int, ...]) -> Any:
            return root._atoms_at(_lift_indices(key, root, sample_shape))

        return sample_atoms

    def sample(key: Any, sample_shape: tuple[int, ...]) -> Any:
        return root._sample(key, sample_shape)

    return sample


def _lift_indices(key: Any, law: EmpiricalDistribution, sample_shape: tuple[int, ...]) -> Any:
    """The indices of the atoms a lift draws from the empirical *law*, shaped as *sample_shape*.

    Both schemes estimate a mean under the law without bias and with no more
    variance than independent draws. Equally weighted atoms are drawn without
    replacement, so ``m`` draws of ``n`` atoms take each atom ``m // n`` or
    ``m // n + 1`` times. Weighted atoms are drawn by stratified resampling,
    which takes one atom from each of ``m`` equal strata of the cumulative
    weights. The indices are then shuffled, so their order pairs them with the
    other arguments' draws at random.
    """
    n, m = law.num_atoms, math.prod(sample_shape)
    order_key, draw_key = jax.random.split(key)
    weights = law._p
    if weights is None:
        whole, rest = divmod(m, n)
        extra = jax.random.choice(draw_key, n, (rest,), replace=False)
        index = jnp.concatenate([jnp.tile(jnp.arange(n), whole), extra])
    else:
        positions = (jnp.arange(m) + jax.random.uniform(draw_key, (m,))) / m
        index = jnp.minimum(jnp.searchsorted(jnp.cumsum(weights), positions), n - 1)
    return jnp.reshape(jax.random.permutation(order_key, index), sample_shape)


def _capture_element(
    element: Distribution,
    source: Distribution,
    *,
    session: _StochasticCaptureSession,
) -> CapturedStochasticConsumer:
    """A batch element's capture, which is its stored law's, since the element shares its draws.

    Parameters
    ----------
    element : Distribution
        The element of a batch of laws.
    source : Distribution
        The element's stored law, as ``_element_source`` returns it.
    session : _StochasticCaptureSession
        The call-local capture session, which captures *source* and whose
        ``active_descendants`` detect a cycle.

    Returns
    -------
    CapturedStochasticConsumer
        The capture of *source*, which the session caches.

    Raises
    ------
    TypeError
        If the element graph is cyclic.
    """
    identity = id(element)
    if identity in session.active_descendants:
        raise TypeError("Cyclic batch element graph is unsupported")
    session.active_descendants.add(identity)
    try:
        return session.capture_consumer(source)
    finally:
        session.active_descendants.remove(identity)


def _capture_field_view(
    view: FieldView,
    *,
    session: _StochasticCaptureSession,
) -> CapturedStochasticConsumer:
    """A field view's capture: its parent's root, and the projection of the view's nodes.

    A view of one path records the path's segments, and a view of a root needs
    no descriptor beyond them. A selection of several paths records them in its
    descriptor.

    Parameters
    ----------
    view : FieldView
        The view of one path of its parent's event, or of a selection of
        several paths.
    session : _StochasticCaptureSession
        The call-local capture session, which captures the parent and whose
        ``active_descendants`` detect a cycle.

    Returns
    -------
    CapturedStochasticConsumer
        The capture with the parent's root and sampler.

    Raises
    ------
    TypeError
        If the view graph is cyclic.
    """
    identity = id(view)
    if identity in session.active_descendants:
        raise TypeError("Cyclic field view graph is unsupported")
    parent, path = view.parent, view.path
    session.active_descendants.add(identity)
    try:
        captured_parent = session.capture_consumer(parent)
        projection = _projector(parent.event_spec, path)
    finally:
        session.active_descendants.remove(identity)
    base = captured_parent.descendant_descriptor
    if isinstance(path, str):
        record_path = tuple(path.split(_PATH_SEP))
        descriptor = (
            None
            if base is None
            else ("record-projection-after-descendant", ("base", base), ("path", record_path))
        )
    else:
        record_path = ()
        descriptor = ("field-selection", ("base", base or ("root",)), ("paths", tuple(path)))
    return CapturedStochasticConsumer(
        root=captured_parent.root,
        sample_root=captured_parent.sample_root,
        record_path=record_path,
        descendant_descriptor=descriptor,
        evaluator=_compose(captured_parent.evaluator, projection),
    )


def _capture_descendant(
    value: Distribution,
    registered: type,
    descend: Callable[[Any], _Descent],
    *,
    session: _StochasticCaptureSession,
) -> CapturedStochasticConsumer:
    """A registered descendant's capture: its ancestor's root, then its map.

    Parameters
    ----------
    value : Distribution
        The descendant law.
    registered : type
        The registered class nearest *value*'s class, whose qualified name the
        descriptor records.
    descend : callable
        The rule registered for *registered*, which gives *value*'s descent.
    session : _StochasticCaptureSession
        The call-local capture session, which captures the ancestor and whose
        ``active_descendants`` detect a cycle.

    Returns
    -------
    CapturedStochasticConsumer
        The capture with the ancestor's root, sampler, and record path.

    Raises
    ------
    TypeError
        If the descendant graph is cyclic.
    """
    identity = id(value)
    if identity in session.active_descendants:
        raise TypeError(f"Cyclic {registered.__name__} descendant graph is unsupported")
    session.active_descendants.add(identity)
    try:
        descent = descend(value)
        captured_ancestor = session.capture_consumer(descent.ancestor)
    finally:
        session.active_descendants.remove(identity)
    descriptor = (
        "transformed-descendant",
        ("descendant_type", f"{registered.__module__}.{registered.__qualname__}"),
        ("sampling_abi", _DISTRIBUTION_SAMPLING_ABI),
        ("base", captured_ancestor.descendant_descriptor or ("root",)),
        ("map", descent.descriptor),
    )
    return CapturedStochasticConsumer(
        root=captured_ancestor.root,
        sample_root=captured_ancestor.sample_root,
        record_path=captured_ancestor.record_path,
        descendant_descriptor=descriptor,
        evaluator=_compose(captured_ancestor.evaluator, descent.forward),
    )


def _encode_descriptor_value(value: Any) -> bytes:
    match value:
        case None:
            return b"N"
        case bool() as flag:
            return b"B\x01" if flag else b"B\x00"
        case int() as number:
            sign = b"-" if number < 0 else b"+"
            magnitude = abs(number)
            raw = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
            return b"I" + sign + len(raw).to_bytes(4, "big") + raw
        case str() as text:
            raw = text.encode("utf-8")
            return b"S" + len(raw).to_bytes(4, "big") + raw
        case bytes() as raw_bytes:
            return b"Y" + len(raw_bytes).to_bytes(4, "big") + raw_bytes
        case tuple() as items:
            encoded = tuple(_encode_descriptor_value(item) for item in items)
            return b"T" + len(encoded).to_bytes(4, "big") + b"".join(encoded)
        case _:
            raise TypeError(f"Unsupported canonical descriptor value: {type(value).__name__}")


def _compose(
    first: Callable[[Any], Any],
    second: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    def evaluate(value: Any) -> Any:
        return second(first(value))

    return evaluate


def _identity(value: Any) -> Any:
    return value


def _renamed_descent(renamed: Distribution) -> _Descent:
    """A renamed law's descent: the draws of the law it renames, moved to the renamed paths.

    Raises
    ------
    TypeError
        If *renamed* renames no law.
    """
    source = _rename_source(renamed)
    if source is None:  # pragma: no cover - the capture reads the source before it descends
        raise TypeError(f"{type(renamed).__name__} renames no law")
    parent, event = source
    return _Descent(
        ancestor=parent,
        forward=event.draw,
        descriptor=("renamed", tuple(sorted(event.leaves.items()))),
    )
