"""Contract tests for ProbPipe's structural workflow RNG ABI."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe.core._workflow_rng as workflow_rng
from probpipe.core._workflow_rng import (
    RandomEventIdentity,
    derive_event_key_words,
    derive_event_key_words_from_encoded,
    encode_random_event,
    jax_key_from_words,
    seed_to_root_words,
)


class TestWorkflowSeedEncoding:
    def test_unsigned_64_bit_seed_maps_to_big_endian_words(self):
        assert seed_to_root_words(0) == (0, 0)
        assert seed_to_root_words(0x0123456789ABCDEF) == (0x01234567, 0x89ABCDEF)
        assert seed_to_root_words(2**64 - 1) == (0xFFFFFFFF, 0xFFFFFFFF)

    @pytest.mark.parametrize("seed", [True, False, -1, 2**64, 1.5, "7"])
    def test_seed_rejects_values_outside_unsigned_64_bit_integers(self, seed):
        with pytest.raises((TypeError, ValueError)):
            seed_to_root_words(seed)


class TestCanonicalEventEncoding:
    def test_encoding_is_typed_length_prefixed_and_domain_separated(self):
        identity = RandomEventIdentity(
            occurrence_path=("invocation", 0),
            stochastic_source_id=("source", 0),
            logical_unit_id=("singleton",),
        )

        encoded = encode_random_event(identity)

        assert encoded.hex() == (
            "50726f62506970652d524e472d763100"
            "10"
            "0400000002"
            "020000000a696e766f636174696f6e"
            "010000000000000000"
            "11"
            "0400000002"
            "0200000006736f75726365"
            "010000000000000000"
            "12"
            "0400000001"
            "020000000973696e676c65746f6e"
        )

    def test_encoding_rejects_a_non_identity_value(self):
        with pytest.raises(TypeError, match="identity must be a RandomEventIdentity"):
            encode_random_event(("invocation", 0))

    @pytest.mark.parametrize(
        "invalid",
        [
            True,
            -1,
            2**64,
            1.5,
            pytest.param(np.int64(1), id="numpy-integer"),
            ["scope", 0],
        ],
    )
    def test_encoding_rejects_values_outside_the_canonical_algebra(self, invalid):
        identity = RandomEventIdentity(
            occurrence_path=("invocation", 0),
            stochastic_source_id=invalid,
            logical_unit_id=("singleton",),
        )

        with pytest.raises((TypeError, ValueError)):
            encode_random_event(identity)


class TestEventKeyDerivation:
    @pytest.fixture(autouse=True)
    def _reset_jax_key_adapter_state(self, monkeypatch):
        monkeypatch.setattr(
            workflow_rng,
            "_JAX_KEY_ADAPTER_STATE",
            workflow_rng._JAXKeyAdapterState(),
        )

    @pytest.mark.parametrize(
        ("root", "identity", "expected"),
        [
            (
                (0, 0),
                RandomEventIdentity(
                    occurrence_path=("invocation", 0),
                    stochastic_source_id=("source", 0),
                    logical_unit_id=("singleton",),
                ),
                (0xE6CD50EA, 0x8FF642DF),
            ),
            (
                (0xFFFFFFFF, 0xFFFFFFFF),
                RandomEventIdentity(
                    occurrence_path=("scope", 1, ("invocation", 2)),
                    stochastic_source_id=b"\x00\xff",
                    logical_unit_id=("cell", 3, 4),
                ),
                (0x59D939EE, 0x2A74904E),
            ),
        ],
    )
    def test_matches_keyed_blake2s_v1_golden_vectors(self, root, identity, expected):
        assert derive_event_key_words(root, identity) == expected
        assert derive_event_key_words_from_encoded(root, encode_random_event(identity)) == expected

    @pytest.mark.parametrize(
        ("root_words", "error_type"),
        [
            pytest.param([0, 0], TypeError, id="list"),
            pytest.param((0,), TypeError, id="length"),
            pytest.param((True, 0), TypeError, id="boolean"),
            pytest.param((0, 1.5), TypeError, id="non-integer"),
            pytest.param((-1, 0), ValueError, id="negative"),
            pytest.param((0, 2**32), ValueError, id="overflow"),
        ],
    )
    def test_derivation_rejects_malformed_root_words(self, root_words, error_type):
        identity = RandomEventIdentity(
            occurrence_path=("invocation", 0),
            stochastic_source_id=("source", 0),
            logical_unit_id=("singleton",),
        )

        with pytest.raises(error_type, match="root_words"):
            derive_event_key_words(root_words, identity)
        with pytest.raises(error_type, match="root_words"):
            derive_event_key_words_from_encoded(root_words, b"encoded-identity")

    @pytest.mark.parametrize("encoded", [bytearray(b"identity"), memoryview(b"identity"), "x"])
    def test_encoded_derivation_requires_immutable_bytes(self, encoded):
        with pytest.raises(TypeError, match="encoded_identity must be bytes"):
            derive_event_key_words_from_encoded((0, 0), encoded)

    def test_jax_adapter_key_is_consumed_in_eager_jit_and_vmap(self):
        words = (0xE6CD50EA, 0x8FF642DF)
        event_key = jax_key_from_words(words)
        expected_words = jnp.asarray(words, dtype=jnp.uint32)
        expected_draw = jax.random.normal(event_key, shape=(4,))

        @jax.jit
        def computation(x):
            return jax.random.normal(event_key, shape=x.shape)

        jitted = computation(jnp.ones((4,)))
        vmapped = jax.vmap(computation)(jnp.ones((3, 4)))

        assert jax.dtypes.issubdtype(event_key.dtype, jax.dtypes.prng_key)
        assert event_key.dtype == jax.random.key_dtype("threefry2x32")
        assert jnp.array_equal(jax.random.key_data(event_key), expected_words)
        assert jnp.array_equal(jitted, expected_draw)
        assert jnp.array_equal(vmapped, jnp.broadcast_to(expected_draw, (3, 4)))

    def test_jax_adapter_certifies_only_once_per_thread(self, monkeypatch):
        certifications = []
        original_certify = workflow_rng._certify_jax_key_adapter

        def record_certification(key, expected_words):
            certifications.append(expected_words)
            original_certify(key, expected_words)

        monkeypatch.setattr(
            workflow_rng,
            "_certify_jax_key_adapter",
            record_certification,
        )

        word_pairs = ((1, 2), (3, 4), (5, 6))
        keys = [jax_key_from_words(words) for words in word_pairs]

        assert certifications == [(1, 2)]
        for key, expected_words in zip(keys, word_pairs, strict=True):
            assert tuple(int(word) for word in jax.random.key_data(key)) == expected_words

    def test_jax_adapter_certifies_once_in_each_thread(self, monkeypatch):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        certification_threads = []
        original_certify = workflow_rng._certify_jax_key_adapter

        def record_certification(key, expected_words):
            with lock:
                certification_threads.append(threading.get_ident())
            original_certify(key, expected_words)

        monkeypatch.setattr(
            workflow_rng,
            "_certify_jax_key_adapter",
            record_certification,
        )

        def use_adapter(words):
            barrier.wait(timeout=10)
            first = jax_key_from_words(words)
            second = jax_key_from_words(words)
            assert jnp.array_equal(jax.random.key_data(first), jax.random.key_data(second))
            return threading.get_ident()

        with ThreadPoolExecutor(max_workers=2) as executor:
            thread_ids = tuple(executor.map(use_adapter, ((1, 2), (3, 4))))

        assert len(set(thread_ids)) == 2
        assert sorted(certification_threads) == sorted(thread_ids)

    def test_jax_adapter_retries_after_changed_raw_word_values(self, monkeypatch):
        original_wrap_key_data = jax.random.wrap_key_data
        wrong_key = original_wrap_key_data(
            jnp.asarray((99, 100), dtype=jnp.uint32),
            impl="threefry2x32",
        )
        calls = 0

        def wrong_then_correct(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return wrong_key
            return original_wrap_key_data(*args, **kwargs)

        monkeypatch.setattr(jax.random, "wrap_key_data", wrong_then_correct)

        with pytest.raises(RuntimeError, match="raw key word values"):
            jax_key_from_words((1, 2))

        assert not workflow_rng._JAX_KEY_ADAPTER_STATE.certified
        key = jax_key_from_words((1, 2))
        assert workflow_rng._JAX_KEY_ADAPTER_STATE.certified
        assert tuple(int(word) for word in jax.random.key_data(key)) == (1, 2)

    def test_jax_adapter_rejects_a_non_threefry_typed_key(self, monkeypatch):
        wrong_key = jax.random.wrap_key_data(
            jnp.asarray((1, 2, 3, 4), dtype=jnp.uint32),
            impl="rbg",
        )
        monkeypatch.setattr(jax.random, "wrap_key_data", lambda *args, **kwargs: wrong_key)

        with pytest.raises(RuntimeError, match="does not support Threefry2x32"):
            jax_key_from_words((1, 2))

    @pytest.mark.parametrize(
        ("round_trip", "message"),
        [
            pytest.param(
                jnp.asarray((1, 2, 3), dtype=jnp.uint32),
                "does not support Threefry2x32",
                id="shape",
            ),
            pytest.param(
                jnp.asarray((1, 2), dtype=jnp.int32),
                "changed the raw key word dtype",
                id="dtype",
            ),
        ],
    )
    def test_jax_adapter_rejects_malformed_round_trip_data(
        self,
        monkeypatch,
        round_trip,
        message,
    ):
        monkeypatch.setattr(jax.random, "key_data", lambda key: round_trip)

        with pytest.raises(RuntimeError, match=message):
            jax_key_from_words((1, 2))

        assert not workflow_rng._JAX_KEY_ADAPTER_STATE.certified
