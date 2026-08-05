"""Structural random-key derivation for workflow-owned stochastic events."""

from __future__ import annotations

import hashlib
import struct
import threading
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..custom_types import PRNGKey

_MAX_U64 = (1 << 64) - 1
_WORD_MASK = (1 << 32) - 1
_DOMAIN = b"ProbPipe-RNG-v1\0"
_BLAKE2S_PERSON = b"PP-RNGv1"
_BLAKE2S_KEY_WORDS = struct.Struct(">2I")

_INT_TAG = b"\x01"
_STRING_TAG = b"\x02"
_BYTES_TAG = b"\x03"
_TUPLE_TAG = b"\x04"

_OCCURRENCE_FIELD_TAG = b"\x10"
_SOURCE_FIELD_TAG = b"\x11"
_UNIT_FIELD_TAG = b"\x12"

type _RandomEventValue = str | bytes | int | tuple[_RandomEventValue, ...]
type _RandomEventPath = tuple[_RandomEventValue, ...]


class _JAXKeyAdapterState(threading.local):
    """Track whether the current thread has certified the JAX key adapter."""

    def __init__(self) -> None:
        self.certified = False


_JAX_KEY_ADAPTER_STATE = _JAXKeyAdapterState()


@dataclass(frozen=True)
class RandomEventIdentity:
    """Canonical structural identity for one workflow random event."""

    occurrence_path: _RandomEventPath
    stochastic_source_id: _RandomEventValue
    logical_unit_id: _RandomEventValue


def seed_to_root_words(seed: int) -> tuple[int, int]:
    """Encode one public workflow seed as canonical big-endian words."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("workflow seed must be an unsigned 64-bit integer")
    if not 0 <= seed <= _MAX_U64:
        raise ValueError("workflow seed must be in the range [0, 2**64 - 1]")
    return seed >> 32, seed & _WORD_MASK


def encode_random_event(identity: RandomEventIdentity) -> bytes:
    """Encode one random-event identity using the version-1 RNG ABI."""
    if not isinstance(identity, RandomEventIdentity):
        raise TypeError("identity must be a RandomEventIdentity")
    return b"".join(
        (
            _DOMAIN,
            _OCCURRENCE_FIELD_TAG,
            _encode_value(identity.occurrence_path),
            _SOURCE_FIELD_TAG,
            _encode_value(identity.stochastic_source_id),
            _UNIT_FIELD_TAG,
            _encode_value(identity.logical_unit_id),
        )
    )


def derive_event_key_words(
    root_words: tuple[int, int],
    identity: RandomEventIdentity,
) -> tuple[int, int]:
    """Derive raw key words through the fixed keyed-BLAKE2s version-1 ABI."""
    root = _validate_word_pair(root_words, name="root_words")
    return _derive_event_key_words(root, encode_random_event(identity))


def derive_event_key_words_from_encoded(
    root_words: tuple[int, int],
    encoded_identity: bytes,
) -> tuple[int, int]:
    """Derive raw key words from an already encoded version-1 event identity."""
    root = _validate_word_pair(root_words, name="root_words")
    if not isinstance(encoded_identity, bytes):
        raise TypeError("encoded_identity must be bytes")
    return _derive_event_key_words(root, encoded_identity)


def _derive_event_key_words(
    root_words: tuple[int, int],
    encoded_identity: bytes,
) -> tuple[int, int]:
    """Apply the version-1 KDF to validated root words and identity bytes."""
    digest = hashlib.blake2s(
        encoded_identity,
        key=_BLAKE2S_KEY_WORDS.pack(*root_words),
        digest_size=_BLAKE2S_KEY_WORDS.size,
        person=_BLAKE2S_PERSON,
    ).digest()
    return _BLAKE2S_KEY_WORDS.unpack(digest)


def jax_key_from_words(words: tuple[int, int]) -> PRNGKey:
    """Wrap canonical words in a per-thread-certified JAX Threefry key."""
    validated = _validate_word_pair(words, name="words")
    key = jax.random.wrap_key_data(
        jnp.asarray(validated, dtype=jnp.uint32),
        impl="threefry2x32",
    )
    if not _JAX_KEY_ADAPTER_STATE.certified:
        _certify_jax_key_adapter(key, validated)
        _JAX_KEY_ADAPTER_STATE.certified = True
    return key


def _certify_jax_key_adapter(
    key: PRNGKey,
    expected_words: tuple[int, int],
) -> None:
    """Certify JAX's typed-key contract and raw-word round trip."""
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key) or (
        key.dtype != jax.random.key_dtype("threefry2x32")
    ):
        raise RuntimeError("installed JAX key adapter does not support Threefry2x32 words")
    round_trip = jax.random.key_data(key)
    if round_trip.shape != (2,):
        raise RuntimeError("installed JAX key adapter does not support Threefry2x32 words")
    if round_trip.dtype != jnp.dtype(jnp.uint32):
        raise RuntimeError("installed JAX key adapter changed the raw key word dtype")
    if tuple(int(word) for word in round_trip) != expected_words:
        raise RuntimeError("installed JAX key adapter changed the raw key word values")


def _validate_random_event_value(value: object) -> None:
    """Validate one value against the recursive workflow RNG identity ABI."""
    _encode_value(value)


def _encode_value(value: object) -> bytes:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid workflow RNG identity fields")
    if isinstance(value, int):
        if not 0 <= value <= _MAX_U64:
            raise ValueError("workflow RNG identity integers must fit unsigned 64 bits")
        return _INT_TAG + value.to_bytes(8, "big")
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return _STRING_TAG + _encode_length(len(payload)) + payload
    if isinstance(value, bytes):
        return _BYTES_TAG + _encode_length(len(value)) + value
    if isinstance(value, tuple):
        return (
            _TUPLE_TAG
            + _encode_length(len(value))
            + b"".join(_encode_value(item) for item in value)
        )
    raise TypeError(
        "workflow RNG identity fields must contain only str, bytes, int, or tuple values"
    )


def _encode_length(length: int) -> bytes:
    if length > _WORD_MASK:
        raise ValueError("workflow RNG identity field is too large to encode")
    return length.to_bytes(4, "big")


def _validate_word_pair(
    words: tuple[int, int],
    *,
    name: str,
) -> tuple[int, int]:
    if not isinstance(words, tuple) or len(words) != 2:
        raise TypeError(f"{name} must be a pair of unsigned 32-bit integers")
    if any(isinstance(word, bool) or not isinstance(word, int) for word in words):
        raise TypeError(f"{name} must be a pair of unsigned 32-bit integers")
    if any(not 0 <= word <= _WORD_MASK for word in words):
        raise ValueError(f"{name} words must be in the range [0, 2**32 - 1]")
    return words
