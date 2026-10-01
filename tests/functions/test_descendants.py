"""Tests for the closed stochastic realization-descendant adapter."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    MultivariateNormal,
    Normal,
    NumericRecordDistributionView,
    NumericRecordSpec,
    ProductDistribution,
)
from probpipe.functions import _descendants
from probpipe.functions._plan import build_broadcast_plan, build_stochastic_plan
from probpipe.values import _binding


def _stochastic_plan(values, n_broadcast_samples=16):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _binding.make_signature_info_from_signature(signature)
    broadcast_plan = build_broadcast_plan(values=values, signature_info=signature_info)
    return build_stochastic_plan(values, broadcast_plan, n_broadcast_samples)


class _RecordingNormal(Normal):
    def __init__(self, calls, *, name="base"):
        self.calls = calls
        super().__init__(loc=0.0, scale=1.0, name=name)

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _RecordingMultivariateNormal(MultivariateNormal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__(loc=jnp.zeros(2), cov=jnp.eye(2), name="base")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


def test_capture_session_rejects_corrupted_identity_cache_entries():
    session = _descendants._StochasticCaptureSession()
    cached_source = Normal("cached", 0.0, 1.0)
    requested_source = Normal("requested", 0.0, 1.0)
    captured_source = session.capture_consumer(cached_source)
    session.consumers[id(requested_source)] = (cached_source, captured_source)

    with pytest.raises(RuntimeError, match="consumer identity cache collision"):
        session.capture_consumer(requested_source)

    cached_bijector = tfb.Shift(1.0)
    requested_bijector = tfb.Shift(1.0)
    captured_bijector = session.capture_bijector(cached_bijector)
    session.bijectors[id(requested_bijector)] = (cached_bijector, captured_bijector)

    with pytest.raises(RuntimeError, match="bijector capture identity cache collision"):
        session.capture_bijector(requested_bijector)


def test_semantic_value_encoding_distinguishes_null_bool_scalar_and_rank_zero_array():
    encoded = {
        _descendants.encode_semantic_value(None),
        _descendants.encode_semantic_value(False),
        _descendants.encode_semantic_value(0.0),
        _descendants.encode_semantic_value(np.float32(0.0)),
        _descendants.encode_semantic_value(jnp.asarray(0.0, dtype=jnp.float32)),
    }

    assert len(encoded) == 5


def test_array_state_is_c_contiguous_little_endian_and_complete():
    value = np.asarray([[1, 2], [3, 4]], dtype=">i4")[:, ::-1]

    encoded = _descendants.encode_semantic_value(value)

    assert encoded == (
        "array",
        ("dtype", "<i4"),
        ("shape", (2, 2)),
        ("data_base64", "AgAAAAEAAAAEAAAAAwAAAA=="),
    )


def test_captured_record_projection_does_not_reread_the_live_view_path():
    root = ProductDistribution(
        x=Normal("x", 0.0, 1.0),
        y=Normal("y", 1.0, 1.0),
    )
    view = root["x"]
    captured = _descendants.capture_stochastic_consumer(view)
    root_sample = root._sample(jax.random.key(13), ())

    object.__setattr__(view, "_key", "y")
    object.__setattr__(view, "_key_path", ("y",))

    np.testing.assert_array_equal(captured.evaluator(root_sample), root_sample["x"])
    assert captured.record_path == ("x",)


@pytest.mark.parametrize("cycle_kind", ["self", "pair"])
def test_cyclic_record_view_graphs_fail_closed(cycle_kind):
    root = ProductDistribution(
        x=Normal("x", 0.0, 1.0),
        y=Normal("y", 1.0, 1.0),
    )
    first = root["x"]
    if cycle_kind == "self":
        object.__setattr__(first, "_parent", first)
    else:
        second = root["y"]
        object.__setattr__(first, "_parent", second)
        object.__setattr__(second, "_parent", first)

    with pytest.raises(TypeError, match="Cyclic record distribution view"):
        _descendants.capture_stochastic_consumer(first)


def test_known_unapproved_record_wrappers_fail_closed():
    root = ProductDistribution(x=Normal("x", 0.0, 1.0))
    flattened = root.as_flat_distribution()
    lifted = NumericRecordDistributionView(
        MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2), name="theta"),
        NumericRecordSpec(a=(), b=()),
    )

    for value, label in (
        (flattened, "FlattenedDistributionView"),
        (lifted, "NumericRecordDistributionView"),
    ):
        with pytest.raises(TypeError, match=label):
            _stochastic_plan({"value": value})
