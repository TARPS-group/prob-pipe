"""Structural random-key derivation for workflow-owned stochastic events."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..custom_types import PRNGKey

_MAX_U64 = (1 << 64) - 1
_WORD_MASK = (1 << 32) - 1
_THREEFRY_PARITY = 0x1BD11BDA
_THREEFRY_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)
_DOMAIN = b"ProbPipe-RNG-v1\0"

_INT_TAG = b"\x01"
_STRING_TAG = b"\x02"
_BYTES_TAG = b"\x03"
_TUPLE_TAG = b"\x04"

_OCCURRENCE_FIELD_TAG = b"\x10"
_SOURCE_FIELD_TAG = b"\x11"
_UNIT_FIELD_TAG = b"\x12"

type _CanonicalValue = str | bytes | int | tuple[_CanonicalValue, ...]


@dataclass(frozen=True)
class RandomEventIdentity:
    """Canonical structural identity for one workflow random event."""

    occurrence_path: _CanonicalValue
    stochastic_source_id: _CanonicalValue
    logical_unit_id: _CanonicalValue


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


def threefry2x32(
    key: tuple[int, int],
    counter: tuple[int, int],
) -> tuple[int, int]:
    """Apply the fixed 20-round Threefry2x32 permutation."""
    key_0, key_1 = _validate_word_pair(key, name="key")
    counter_0, counter_1 = _validate_word_pair(counter, name="counter")
    key_schedule = (
        key_0,
        key_1,
        _THREEFRY_PARITY ^ key_0 ^ key_1,
    )
    state_0 = (counter_0 + key_schedule[0]) & _WORD_MASK
    state_1 = (counter_1 + key_schedule[1]) & _WORD_MASK

    for round_index in range(20):
        state_0 = (state_0 + state_1) & _WORD_MASK
        rotation = _THREEFRY_ROTATIONS[round_index % len(_THREEFRY_ROTATIONS)]
        state_1 = _rotate_left(state_1, rotation) ^ state_0

        if (round_index + 1) % 4 == 0:
            injection = (round_index + 1) // 4
            state_0 = (state_0 + key_schedule[injection % 3]) & _WORD_MASK
            state_1 = (state_1 + key_schedule[(injection + 1) % 3] + injection) & _WORD_MASK

    return state_0, state_1


def derive_event_key_words(
    root_words: tuple[int, int],
    identity: RandomEventIdentity,
) -> tuple[int, int]:
    """Derive raw key words from one root and canonical event identity."""
    state = _validate_word_pair(root_words, name="root_words")
    digest = hashlib.sha256(encode_random_event(identity)).digest()
    for offset in range(0, len(digest), 4):
        digest_word = int.from_bytes(digest[offset : offset + 4], "big")
        state = threefry2x32(state, (0, digest_word))
    return state


def jax_key_from_words(words: tuple[int, int]) -> PRNGKey:
    """Wrap canonical raw words in JAX's checked Threefry typed-key format."""
    validated = _validate_word_pair(words, name="words")
    with jax.ensure_compile_time_eval():
        key = jax.random.wrap_key_data(
            jnp.asarray(validated, dtype=jnp.uint32),
            impl="threefry2x32",
        )
        round_trip = jax.random.key_data(key)
        if str(key.dtype) != "key<fry>" or round_trip.shape != (2,):
            raise RuntimeError("installed JAX key adapter does not support Threefry2x32 words")
        if round_trip.dtype != jnp.dtype(jnp.uint32):
            raise RuntimeError("installed JAX key adapter changed the raw key word dtype")
        if tuple(int(word) for word in round_trip) != validated:
            raise RuntimeError("installed JAX key adapter changed the raw key word values")
    return key


def _encode_value(value: _CanonicalValue) -> bytes:
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


def _rotate_left(word: int, distance: int) -> int:
    return ((word << distance) | (word >> (32 - distance))) & _WORD_MASK
