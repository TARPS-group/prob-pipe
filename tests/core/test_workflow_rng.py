"""Contract tests for ProbPipe's structural workflow RNG ABI."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe.core._workflow_rng import (
    RandomEventIdentity,
    derive_event_key_words,
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

    @pytest.mark.parametrize("invalid", [True, -1, 2**64, 1.5, ["scope", 0]])
    def test_encoding_rejects_values_outside_the_canonical_algebra(self, invalid):
        identity = RandomEventIdentity(
            occurrence_path=("invocation", 0),
            stochastic_source_id=invalid,
            logical_unit_id=("singleton",),
        )

        with pytest.raises((TypeError, ValueError)):
            encode_random_event(identity)


class TestEventKeyDerivation:
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

    def test_jax_adapter_preserves_raw_words_in_eager_jit_and_vmap(self):
        words = (0xE6CD50EA, 0x8FF642DF)

        def adapted_words():
            return jax.random.key_data(jax_key_from_words(words))

        expected = jnp.asarray(words, dtype=jnp.uint32)
        eager = adapted_words()
        jitted = jax.jit(adapted_words)()
        vmapped = jax.vmap(lambda _: adapted_words())(jnp.arange(3))

        assert str(jax_key_from_words(words).dtype) == "key<fry>"
        assert jnp.array_equal(eager, expected)
        assert jnp.array_equal(jitted, expected)
        assert jnp.array_equal(vmapped, jnp.broadcast_to(expected, (3, 2)))

    def test_jax_adapter_rejects_changed_raw_word_values(self, monkeypatch):
        wrong_key = jax.random.wrap_key_data(
            jnp.asarray((99, 100), dtype=jnp.uint32),
            impl="threefry2x32",
        )
        monkeypatch.setattr(jax.random, "wrap_key_data", lambda *args, **kwargs: wrong_key)

        with pytest.raises(RuntimeError, match="raw key word values"):
            jax_key_from_words((1, 2))
