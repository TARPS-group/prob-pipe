"""Closed capture and validation for stochastic realization descendants."""

from __future__ import annotations

import base64
import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb

from ._distribution_base import Distribution
from ._record_distribution import _RecordDistributionView

_DESCENDANT_ADAPTER_ABI = "probpipe.transformed_descendant/v1"
_DESCENDANT_PROVIDER_ABI = "tensorflow_probability.substrates.jax.bijector.forward/v1"
_DISTRIBUTION_SAMPLING_ABI = "probpipe.distribution_sampling/v1"
_DESCRIPTOR_DOMAIN = b"ProbPipe-descendant-descriptor-v1\0"

_TRANSFORMED_DISTRIBUTION_TYPES: set[type] = set()
_UNSUPPORTED_DESCENDANT_TYPES: dict[type, str] = {}

_APPROVED_BIJECTOR_PARAMETERS: dict[type, tuple[str, ...]] = {
    tfb.Identity: (),
    tfb.Exp: (),
    tfb.Square: (),
    tfb.Shift: ("shift",),
    tfb.Scale: ("scale", "log_scale"),
    tfb.Softplus: ("hinge_softness", "low"),
    tfb.Sigmoid: ("low", "high"),
}
_APPROVED_BIJECTOR_TYPES = (*_APPROVED_BIJECTOR_PARAMETERS, tfb.Chain)
_FORWARD_OVERRIDE_NAMES = (
    "forward",
    "_forward",
    "forward_event_shape",
    "_forward_event_shape",
)


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


def _register_transformed_distribution_type(distribution_type: type) -> None:
    """Register one ProbPipe-owned concrete transformed implementation."""
    _TRANSFORMED_DISTRIBUTION_TYPES.add(distribution_type)


def _register_unsupported_descendant_type(distribution_type: type, label: str) -> None:
    """Register one known ProbPipe descendant form that must fail closed."""
    _UNSUPPORTED_DESCENDANT_TYPES[distribution_type] = label


def capture_stochastic_consumer(value: Distribution) -> CapturedStochasticConsumer:
    """Capture a supported root/projection/transform graph without executing it."""
    return _capture_stochastic_consumer(value, active_descendants=set())


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


def canonical_descriptor_bytes(descriptor: tuple[Any, ...]) -> bytes:
    """Encode descriptor primitives independently of the workflow RNG ABI."""
    return _DESCRIPTOR_DOMAIN + _encode_descriptor_value(descriptor)


def encode_semantic_value(value: Any) -> tuple[Any, ...]:
    """Strongly encode one bijector semantic value into tuple-only data."""
    if value is None:
        return ("none",)
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return ("python-int", value)
    if isinstance(value, float):
        return ("python-float64-le", base64.b64encode(struct.pack("<d", value)).decode("ascii"))
    if isinstance(value, np.generic):
        array = np.asarray(value)
        return (
            "numpy-scalar",
            _little_endian_dtype(array.dtype).str,
            base64.b64encode(_little_endian_array_bytes(array)).decode("ascii"),
        )
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        array = np.asarray(value)
        dtype = _little_endian_dtype(array.dtype)
        return (
            "array",
            ("dtype", dtype.str),
            ("shape", tuple(int(axis) for axis in array.shape)),
            (
                "data_base64",
                base64.b64encode(_little_endian_array_bytes(array)).decode("ascii"),
            ),
        )
    raise TypeError(
        "Unsupported transformed-descendant semantic state of type "
        f"{type(value).__module__}.{type(value).__qualname__}; pass an explicit key "
        "or use a supported immutable numeric parameter."
    )


def _capture_stochastic_consumer(
    value: Distribution,
    *,
    active_descendants: set[int],
) -> CapturedStochasticConsumer:
    value_type = type(value)
    unsupported_label = _UNSUPPORTED_DESCENDANT_TYPES.get(value_type)
    if unsupported_label is not None:
        raise TypeError(
            f"Automatic stochastic lifting does not support {unsupported_label}; "
            "pass an explicit key to its direct sampling API or lift its approved root."
        )

    if value_type in _TRANSFORMED_DISTRIBUTION_TYPES:
        return _capture_transformed_distribution(value, active_descendants=active_descendants)

    if any(isinstance(value, known) for known in _TRANSFORMED_DISTRIBUTION_TYPES):
        raise TypeError(
            "Automatic stochastic lifting rejects TransformedDistribution subclasses; "
            f"got {value_type.__module__}.{value_type.__qualname__}."
        )
    if any(isinstance(value, known) for known in _UNSUPPORTED_DESCENDANT_TYPES):
        raise TypeError(
            "Automatic stochastic lifting rejects subclasses of known unsupported "
            f"descendants; got {value_type.__module__}.{value_type.__qualname__}."
        )

    if isinstance(value, _RecordDistributionView):
        captured_parent = _capture_stochastic_consumer(
            value.parent,
            active_descendants=active_descendants,
        )
        evaluator = _compose(captured_parent.evaluator, value._extract)
        descriptor = captured_parent.descendant_descriptor
        if descriptor is not None:
            descriptor = (
                "record-projection-after-descendant",
                ("base", descriptor),
                ("path", tuple(value._key_path)),
            )
        return CapturedStochasticConsumer(
            root=captured_parent.root,
            sample_root=captured_parent.sample_root,
            record_path=tuple(value._key_path),
            descendant_descriptor=descriptor,
            evaluator=evaluator,
        )

    return CapturedStochasticConsumer(
        root=value,
        sample_root=value._sample,
        record_path=(),
        descendant_descriptor=None,
        evaluator=_identity,
    )


def _capture_transformed_distribution(
    value: Distribution,
    *,
    active_descendants: set[int],
) -> CapturedStochasticConsumer:
    identity = id(value)
    if identity in active_descendants:
        raise TypeError("Cyclic TransformedDistribution descendant graph is unsupported")
    _reject_instance_overrides(value, ("_sample", "base", "bijector"))

    active_descendants.add(identity)
    try:
        captured_base = _capture_stochastic_consumer(
            value.base,
            active_descendants=active_descendants,
        )
        bijector_descriptor, forward = _capture_bijector(value.bijector, active_bijectors=set())
    finally:
        active_descendants.remove(identity)

    descriptor = (
        "transformed-descendant",
        (
            "descendant_type",
            "probpipe.distributions.transformed.TransformedDistribution",
        ),
        ("descendant_adapter_abi", _DESCENDANT_ADAPTER_ABI),
        ("sampling_abi", _DISTRIBUTION_SAMPLING_ABI),
        ("provider_abi", _DESCENDANT_PROVIDER_ABI),
        ("base", captured_base.descendant_descriptor or ("root",)),
        ("bijector", bijector_descriptor),
    )
    return CapturedStochasticConsumer(
        root=captured_base.root,
        sample_root=captured_base.sample_root,
        record_path=captured_base.record_path,
        descendant_descriptor=descriptor,
        evaluator=_compose(captured_base.evaluator, forward),
    )


def _capture_bijector(
    bijector: tfb.Bijector,
    *,
    active_bijectors: set[int],
) -> tuple[tuple[Any, ...], Callable[[Any], Any]]:
    bijector_type = type(bijector)
    if bijector_type not in _APPROVED_BIJECTOR_TYPES:
        if isinstance(bijector, _APPROVED_BIJECTOR_TYPES):
            reason = "subclasses of approved bijectors"
        else:
            reason = "this bijector type"
        raise TypeError(
            f"Automatic transformed-descendant lifting does not support {reason}: "
            f"{bijector_type.__module__}.{bijector_type.__qualname__}."
        )

    identity = id(bijector)
    if identity in active_bijectors:
        raise TypeError("Cyclic TFP Chain descendant graph is unsupported")
    _reject_instance_overrides(bijector, _FORWARD_OVERRIDE_NAMES)

    try:
        event_ndims = int(bijector.forward_min_event_ndims)
    except (TypeError, ValueError) as exc:
        raise TypeError("Bijector forward_min_event_ndims must be a concrete integer") from exc
    if event_ndims != 0:
        raise TypeError(
            "Automatic transformed-descendant lifting requires "
            f"forward_min_event_ndims == 0; got {event_ndims}."
        )

    _validate_parameter_surface(bijector)
    active_bijectors.add(identity)
    try:
        if bijector_type is tfb.Chain:
            child_descriptors = tuple(
                _capture_bijector(child, active_bijectors=active_bijectors)[0]
                for child in tuple(bijector.bijectors)
            )
            settings = (
                ("validate_args", encode_semantic_value(bijector.validate_args)),
                (
                    "validate_event_size",
                    encode_semantic_value(bijector.validate_event_size),
                ),
                (
                    "parameters",
                    encode_semantic_value(bijector.parameters.get("parameters")),
                ),
            )
            semantic_parameters: tuple[tuple[str, tuple[Any, ...]], ...] = ()
        else:
            child_descriptors = ()
            settings = (("validate_args", encode_semantic_value(bijector.validate_args)),)
            semantic_parameters = tuple(
                (name, encode_semantic_value(getattr(bijector, name)))
                for name in _APPROVED_BIJECTOR_PARAMETERS[bijector_type]
            )
    finally:
        active_bijectors.remove(identity)

    descriptor = (
        "tfp-bijector",
        ("type", f"{bijector_type.__module__}.{bijector_type.__qualname__}"),
        ("forward_min_event_ndims", event_ndims),
        ("settings", settings),
        ("semantic_parameters", semantic_parameters),
        ("children_native_order", child_descriptors),
    )
    return descriptor, bijector.forward


def _validate_parameter_surface(bijector: tfb.Bijector) -> None:
    parameter_names = set(bijector.parameters) - {"name"}
    if type(bijector) is tfb.Chain:
        expected = {"bijectors", "validate_args", "validate_event_size", "parameters"}
    else:
        expected = {"validate_args", *_APPROVED_BIJECTOR_PARAMETERS[type(bijector)]}
    if parameter_names != expected:
        raise TypeError(
            "Installed TFP bijector parameter surface is incompatible with "
            f"{_DESCENDANT_ADAPTER_ABI}: expected {sorted(expected)!r}, "
            f"got {sorted(parameter_names)!r}."
        )


def _reject_instance_overrides(value: Any, names: tuple[str, ...]) -> None:
    overridden = tuple(name for name in names if name in vars(value))
    if overridden:
        raise TypeError(
            "Automatic transformed-descendant lifting rejects instance method/property "
            f"overrides {overridden!r} on {type(value).__name__}."
        )


def _little_endian_dtype(dtype: np.dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.kind not in "biufc":
        raise TypeError(f"Unsupported bijector parameter dtype {dtype!s}")
    return dtype.newbyteorder("<")


def _little_endian_array_bytes(array: np.ndarray) -> bytes:
    dtype = _little_endian_dtype(array.dtype)
    normalized = np.ascontiguousarray(array.astype(dtype, copy=False))
    return normalized.tobytes(order="C")


def _encode_descriptor_value(value: Any) -> bytes:
    if value is None:
        return b"N"
    if isinstance(value, bool):
        return b"B\x01" if value else b"B\x00"
    if isinstance(value, int):
        sign = b"-" if value < 0 else b"+"
        magnitude = abs(value)
        raw = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
        return b"I" + sign + len(raw).to_bytes(4, "big") + raw
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + len(raw).to_bytes(4, "big") + raw
    if isinstance(value, bytes):
        return b"Y" + len(value).to_bytes(4, "big") + value
    if isinstance(value, tuple):
        encoded = tuple(_encode_descriptor_value(item) for item in value)
        return b"T" + len(encoded).to_bytes(4, "big") + b"".join(encoded)
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
