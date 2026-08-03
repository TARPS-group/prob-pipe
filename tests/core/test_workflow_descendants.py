"""Tests for the closed stochastic realization-descendant adapter."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    EmpiricalDistribution,
    Function,
    MultivariateNormal,
    Normal,
    NumericEventTemplate,
    NumericRecord,
    NumericRecordArray,
    ProductDistribution,
    Record,
    TransformedDistribution,
    workflow_run,
)
from probpipe.core import _workflow_call, _workflow_descendants
from probpipe.core._workflow_plan import build_broadcast_plan, build_stochastic_plan


def _stochastic_plan(values, n_broadcast_samples=16):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _workflow_call.make_signature_info_from_signature(signature)
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


@pytest.mark.parametrize(
    "bijector",
    [
        pytest.param(tfb.Identity(), id="identity"),
        pytest.param(tfb.Exp(), id="exp"),
        pytest.param(tfb.Square(), id="square"),
        pytest.param(tfb.Shift(1.5), id="shift"),
        pytest.param(tfb.Scale(log_scale=0.25), id="scale"),
        pytest.param(tfb.Softplus(hinge_softness=0.75, low=0.1), id="softplus"),
        pytest.param(tfb.Sigmoid(low=-2.0, high=3.0), id="sigmoid"),
        pytest.param(tfb.Chain([tfb.Exp(), tfb.Shift(1.0)]), id="chain"),
    ],
)
def test_approved_bijectors_capture_root_and_live_forward(bijector):
    base = Normal(loc=0.0, scale=1.0, name="base")
    descendant = TransformedDistribution(base, bijector)

    captured = _workflow_descendants.capture_stochastic_consumer(descendant)
    key = jax.random.PRNGKey(9)
    root_samples = base._sample(key, (11,))

    assert captured.root is base
    assert captured.record_path == ()
    assert captured.descendant_descriptor is not None
    np.testing.assert_allclose(
        captured.evaluator(root_samples),
        descendant._sample(key, (11,)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_golden_shift_descriptor_and_digest_are_hard_coded():
    base = MultivariateNormal(
        loc=jnp.zeros(2),
        cov=jnp.eye(2),
        name="base",
    )
    descendant = TransformedDistribution(
        base,
        tfb.Shift(jnp.asarray([1.0, -2.0], dtype=jnp.float32), name="ignored-name"),
        name="also-ignored",
    )
    expected = (
        "stochastic-descendant",
        ("base_source_slot", 0),
        (
            "graph",
            (
                "transformed-descendant",
                (
                    "descendant_type",
                    "probpipe.distributions.transformed.TransformedDistribution",
                ),
                ("descendant_adapter_abi", "probpipe.transformed_descendant/v1"),
                ("sampling_abi", "probpipe.distribution_sampling/v1"),
                (
                    "provider_abi",
                    "tensorflow_probability.substrates.jax.bijector.forward/v1",
                ),
                ("base", ("root",)),
                (
                    "bijector",
                    (
                        "tfp-bijector",
                        (
                            "type",
                            "tensorflow_probability.substrates.jax.bijectors.shift.Shift",
                        ),
                        ("forward_min_event_ndims", 0),
                        ("settings", (("validate_args", ("bool", False)),)),
                        (
                            "semantic_parameters",
                            (
                                (
                                    "shift",
                                    (
                                        "array",
                                        ("dtype", "<f4"),
                                        ("shape", (2,)),
                                        ("data_base64", "AACAPwAAAMA="),
                                    ),
                                ),
                            ),
                        ),
                        ("children_native_order", ()),
                    ),
                ),
            ),
        ),
    )

    plan = _stochastic_plan({"value": descendant})
    descriptor = plan.source_groups[0].consumers[0].descendant_descriptor

    assert descriptor == expected
    assert (
        _workflow_descendants.descriptor_digest(expected)
        == "e63fc9be717837f2c266b5b1a5bf74fbec6808a45e88f8611b90fbfaffb9be29"
    )


def test_semantic_value_encoding_distinguishes_null_bool_scalar_and_rank_zero_array():
    encoded = {
        _workflow_descendants.encode_semantic_value(None),
        _workflow_descendants.encode_semantic_value(False),
        _workflow_descendants.encode_semantic_value(0.0),
        _workflow_descendants.encode_semantic_value(np.float32(0.0)),
        _workflow_descendants.encode_semantic_value(jnp.asarray(0.0, dtype=jnp.float32)),
    }

    assert len(encoded) == 5


def test_array_state_is_c_contiguous_little_endian_and_complete():
    value = np.asarray([[1, 2], [3, 4]], dtype=">i4")[:, ::-1]

    encoded = _workflow_descendants.encode_semantic_value(value)

    assert encoded == (
        "array",
        ("dtype", "<i4"),
        ("shape", (2, 2)),
        ("data_base64", "AgAAAAEAAAAEAAAAAwAAAA=="),
    )


def test_names_are_excluded_but_semantic_parameters_are_not():
    base = Normal(loc=0.0, scale=1.0, name="base")
    first = TransformedDistribution(base, tfb.Shift(1.0, name="first"), name="first")
    renamed = TransformedDistribution(base, tfb.Shift(1.0, name="second"), name="second")
    changed = TransformedDistribution(base, tfb.Shift(2.0, name="first"), name="first")

    first_descriptor = _workflow_descendants.capture_stochastic_consumer(
        first
    ).descendant_descriptor
    renamed_descriptor = _workflow_descendants.capture_stochastic_consumer(
        renamed
    ).descendant_descriptor
    changed_descriptor = _workflow_descendants.capture_stochastic_consumer(
        changed
    ).descendant_descriptor

    assert first_descriptor == renamed_descriptor
    assert first_descriptor != changed_descriptor


def test_nested_transforms_and_chain_child_order_are_structural():
    base = Normal(loc=0.0, scale=1.0, name="base")
    inner = TransformedDistribution(base, tfb.Exp())
    nested = TransformedDistribution(inner, tfb.Shift(2.0))
    first_chain = TransformedDistribution(base, tfb.Chain([tfb.Exp(), tfb.Shift(2.0)]))
    reversed_chain = TransformedDistribution(base, tfb.Chain([tfb.Shift(2.0), tfb.Exp()]))

    nested_capture = _workflow_descendants.capture_stochastic_consumer(nested)
    first_descriptor = _workflow_descendants.capture_stochastic_consumer(
        first_chain
    ).descendant_descriptor
    reversed_descriptor = _workflow_descendants.capture_stochastic_consumer(
        reversed_chain
    ).descendant_descriptor

    assert nested_capture.root is base
    assert nested_capture.descendant_descriptor[5][1][0] == "transformed-descendant"
    assert first_descriptor != reversed_descriptor


def test_root_projection_and_multiple_descendants_form_one_plan_group():
    root = ProductDistribution(
        x=Normal(loc=0.0, scale=1.0, name="x"),
        y=Normal(loc=2.0, scale=1.0, name="y"),
    )
    x = root["x"]
    exp_x = TransformedDistribution(x, tfb.Exp())
    shifted_x = TransformedDistribution(x, tfb.Shift(3.0))

    plan = _stochastic_plan({"root": root, "x": x, "exp_x": exp_x, "shifted_x": shifted_x})

    assert len(plan.source_groups) == 1
    assert plan.runtime_bindings[0].root is root
    assert tuple(consumer.record_path for consumer in plan.source_groups[0].consumers) == (
        (),
        ("x",),
        ("x",),
        ("x",),
    )
    assert plan.source_groups[0].consumers[0].descendant_descriptor is None
    assert plan.source_groups[0].consumers[1].descendant_descriptor is None
    assert plan.source_groups[0].consumers[2].descendant_descriptor is not None
    assert plan.source_groups[0].consumers[3].descendant_descriptor is not None
    assert len(plan.random_events) == 1


def test_descriptor_records_its_plan_local_base_source_slot():
    root = Normal(0.0, 1.0, name="root")
    descendant = TransformedDistribution(root, tfb.Exp())

    plan = _stochastic_plan(
        {
            "independent": Normal(1.0, 1.0, name="independent"),
            "descendant": descendant,
        }
    )

    assert plan.source_groups[1].consumers[0].descendant_descriptor[1] == (
        "base_source_slot",
        1,
    )


@pytest.mark.parametrize(
    ("make_bad", "message"),
    [
        (
            lambda base: TransformedDistribution(base, tfb.Tanh()),
            "does not support this bijector type",
        ),
        (
            lambda base: TransformedDistribution(base, type("CustomExp", (tfb.Exp,), {})()),
            "subclasses of approved bijectors",
        ),
    ],
)
def test_unsupported_bijector_types_fail_closed(make_bad, message):
    descendant = make_bad(Normal(loc=0.0, scale=1.0, name="base"))

    with pytest.raises(TypeError, match=message):
        _workflow_descendants.capture_stochastic_consumer(descendant)


def test_instance_forward_override_fails_closed():
    bijector = tfb.Exp()
    object.__setattr__(bijector, "_forward", lambda value: value)
    descendant = TransformedDistribution(Normal(0.0, 1.0, name="base"), bijector)

    with pytest.raises(TypeError, match="instance method/property overrides"):
        _workflow_descendants.capture_stochastic_consumer(descendant)


def test_transformed_subclass_and_instance_sampling_override_fail_closed():
    class CustomTransformedDistribution(TransformedDistribution):
        def __new__(cls, base, bijector, **kwargs):
            return object.__new__(cls)

    base = Normal(0.0, 1.0, name="base")
    subclassed = CustomTransformedDistribution(base, tfb.Exp())
    overridden = TransformedDistribution(base, tfb.Exp())
    object.__setattr__(overridden, "_sample", lambda key, sample_shape=(): 0.0)

    with pytest.raises(TypeError, match="rejects TransformedDistribution subclasses"):
        _workflow_descendants.capture_stochastic_consumer(subclassed)
    with pytest.raises(TypeError, match="instance method/property overrides"):
        _workflow_descendants.capture_stochastic_consumer(overridden)


def test_nonzero_forward_event_rank_fails_closed():
    bijector = tfb.Exp()
    object.__setattr__(bijector, "_forward_min_event_ndims", 1)
    descendant = TransformedDistribution(Normal(0.0, 1.0, name="base"), bijector)

    with pytest.raises(TypeError, match="forward_min_event_ndims == 0"):
        _workflow_descendants.capture_stochastic_consumer(descendant)


def test_unencodable_semantic_state_fails_closed():
    bijector = tfb.Shift(1.0)
    object.__setattr__(bijector, "_shift", object())
    descendant = TransformedDistribution(Normal(0.0, 1.0, name="base"), bijector)

    with pytest.raises(TypeError, match="semantic state"):
        _workflow_descendants.capture_stochastic_consumer(descendant)


def test_cyclic_descendant_and_chain_graphs_fail_closed():
    base = Normal(0.0, 1.0, name="base")
    descendant = TransformedDistribution(base, tfb.Exp())
    object.__setattr__(descendant, "_base", descendant)

    with pytest.raises(TypeError, match="Cyclic TransformedDistribution"):
        _workflow_descendants.capture_stochastic_consumer(descendant)

    chain = tfb.Chain([tfb.Exp()])
    cyclic_chain_descendant = TransformedDistribution(base, chain)
    object.__setattr__(chain, "_bijectors", (chain,))
    with pytest.raises(TypeError, match="Cyclic TFP Chain"):
        _workflow_descendants.capture_stochastic_consumer(cyclic_chain_descendant)


def test_known_unapproved_record_wrappers_fail_closed():
    root = ProductDistribution(x=Normal(0.0, 1.0, name="x"))
    flattened = root.as_flat_distribution()
    lifted = MultivariateNormal(
        loc=jnp.zeros(2),
        cov=jnp.eye(2),
        name="theta",
    ).as_record_distribution(template=NumericEventTemplate(a=(), b=()))

    for value, label in (
        (flattened, "FlattenedDistributionView"),
        (lifted, "NumericRecordDistributionView"),
    ):
        with pytest.raises(TypeError, match=label):
            _stochastic_plan({"value": value})


def test_unsupported_preflight_does_not_read_entropy_or_sample():
    base = Normal(0.0, 1.0, name="base")
    descendant = TransformedDistribution(base, tfb.Tanh())
    workflow = Function(
        func=lambda value: value,
        dispatch="sequential",
        n_broadcast_samples=8,
    )

    with (
        patch("probpipe.core._workflow_context._os_urandom") as urandom,
        patch.object(type(base), "_sample", wraps=base._sample) as sample_root,
        workflow_run(),
        pytest.raises(TypeError, match="does not support this bijector type"),
    ):
        workflow(descendant)

    urandom.assert_not_called()
    sample_root.assert_not_called()


@pytest.mark.parametrize("dispatch", ["sequential", "jax"])
def test_function_co_samples_root_and_multiple_descendants(dispatch):
    calls = []
    root = _RecordingNormal(calls)
    exponentiated = TransformedDistribution(root, tfb.Exp())
    shifted = TransformedDistribution(root, tfb.Shift(2.0))
    workflow = Function(
        func=lambda base, exp_base, shifted_base: jnp.stack(
            (
                exp_base - jnp.exp(base),
                shifted_base - (base + 2.0),
            )
        ),
        dispatch=dispatch,
        n_broadcast_samples=16,
    )

    with workflow_run(seed=37):
        result = workflow(root, exponentiated, shifted)

    assert [shape for _key, shape in calls] == [(16,)]
    np.testing.assert_allclose(result.samples, 0.0, atol=1e-6)


def test_function_descendant_only_samples_its_captured_root_once():
    calls = []
    root = _RecordingNormal(calls)
    descendant = TransformedDistribution(root, tfb.Exp())
    workflow = Function(
        func=lambda value: value,
        dispatch="sequential",
        n_broadcast_samples=14,
        include_inputs=True,
    )

    with workflow_run(seed=39):
        result = workflow(descendant)

    assert [shape for _key, shape in calls] == [(14,)]
    root_key = calls[0][0]
    expected = jnp.exp(Normal._sample(root, root_key, (14,)))
    np.testing.assert_allclose(result.input_samples["value"], expected, rtol=1e-6)
    np.testing.assert_allclose(result.samples, expected, rtol=1e-6)


def test_sequential_and_jax_consume_the_same_captured_graph():
    calls = []
    root = _RecordingNormal(calls)
    exponentiated = TransformedDistribution(root, tfb.Exp())

    def difference(base, exp_base):
        return exp_base - jnp.exp(base)

    def run(dispatch):
        workflow = Function(
            func=difference,
            dispatch=dispatch,
            n_broadcast_samples=16,
            include_inputs=True,
        )
        with workflow_run(seed=41):
            return workflow(root, exponentiated)

    sequential = run("sequential")
    jax_result = run("jax")

    assert [shape for _key, shape in calls] == [(16,), (16,)]
    np.testing.assert_array_equal(
        sequential.input_samples["base"],
        jax_result.input_samples["base"],
    )
    np.testing.assert_array_equal(
        sequential.input_samples["exp_base"],
        jax_result.input_samples["exp_base"],
    )
    np.testing.assert_allclose(sequential.samples, 0.0, atol=1e-6)
    np.testing.assert_allclose(jax_result.samples, 0.0, atol=1e-6)


def test_exact_empirical_root_and_descendant_keep_weights_once():
    root = EmpiricalDistribution(
        jnp.asarray([1.0, 4.0]),
        weights=jnp.asarray([0.2, 0.8]),
        name="base",
    )
    exponentiated = TransformedDistribution(root, tfb.Exp())
    workflow = Function(
        func=lambda base, exp_base: exp_base - jnp.exp(base),
        dispatch="sequential",
        n_broadcast_samples=16,
        include_inputs=True,
    )

    with patch.object(type(root), "_sample", side_effect=AssertionError("sampled exact root")):
        result = workflow(root, exponentiated)

    assert result.num_atoms == 2
    np.testing.assert_allclose(result.samples, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.weights, jnp.asarray([0.2, 0.8]))
    np.testing.assert_allclose(
        result.input_samples["exp_base"],
        jnp.exp(result.input_samples["base"]),
        rtol=1e-6,
    )


def test_exact_record_projection_then_transform_stays_diagonal():
    root = EmpiricalDistribution(
        Record(
            "draws",
            x=jnp.asarray([1.0, 4.0]),
            y=jnp.asarray([10.0, 40.0]),
        ),
        weights=jnp.asarray([0.3, 0.7]),
        name="joint",
    )
    x = root["x"]
    exponentiated_x = TransformedDistribution(x, tfb.Exp())
    workflow = Function(
        func=lambda joint, x_value, exp_x: jnp.stack(
            (joint["x"] - x_value, exp_x - jnp.exp(x_value))
        ),
        dispatch="sequential",
        n_broadcast_samples=16,
        include_inputs=True,
    )

    result = workflow(root, x, exponentiated_x)

    assert result.num_atoms == 2
    np.testing.assert_allclose(result.samples, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.weights, jnp.asarray([0.3, 0.7]))


def test_mixed_empirical_descendant_multiplies_root_weight_once():
    exact_root = EmpiricalDistribution(
        jnp.asarray([1.0, 4.0]),
        weights=jnp.asarray([0.2, 0.8]),
        name="exact",
    )
    exponentiated = TransformedDistribution(exact_root, tfb.Exp())
    sampled_calls = []
    sampled = _RecordingNormal(sampled_calls, name="sampled")
    workflow = Function(
        func=lambda exact, exp_exact, noise: jnp.stack((exp_exact - jnp.exp(exact), noise)),
        dispatch="sequential",
        n_broadcast_samples=12,
        include_inputs=True,
    )

    with workflow_run(seed=45):
        result = workflow(exact_root, exponentiated, sampled)

    assert result.num_atoms == 12
    assert [shape for _key, shape in sampled_calls] == [(12,)]
    np.testing.assert_allclose(result.samples["marginal"][:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        result.input_samples["exp_exact"],
        jnp.exp(result.input_samples["exact"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.weights,
        jnp.repeat(jnp.asarray([0.2, 0.8]) / 6.0, 6),
        rtol=1e-6,
    )


def test_elementwise_transform_of_vector_event_matches_direct_sampling():
    calls = []
    root = _RecordingMultivariateNormal(calls)
    descendant = TransformedDistribution(root, tfb.Exp())
    captured = _workflow_descendants.capture_stochastic_consumer(descendant)
    key = jax.random.key(43)

    actual = _workflow_descendants.sample_captured_consumer(captured, key, (9,))
    expected = descendant._sample(key, (9,))

    assert [shape for _key, shape in calls] == [(9,)]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dispatch", ["sequential", "jax"])
def test_nested_sweep_samples_shared_transform_root_once_per_cell(dispatch):
    rows = NumericRecordArray.stack(
        [NumericRecord("row", offset=float(index)) for index in range(3)]
    )
    calls = []
    root = _RecordingNormal(calls)
    exponentiated = TransformedDistribution(root, tfb.Exp())
    workflow = Function(
        func=lambda row, base, exp_base: exp_base - jnp.exp(base),
        dispatch=dispatch,
        n_broadcast_samples=12,
    )

    with workflow_run(seed=47):
        result = workflow(rows, root, exponentiated)

    assert result.batch_shape == (3,)
    assert [shape for _key, shape in calls] == [(12,), (12,), (12,)]
    for component in result.components:
        np.testing.assert_allclose(component.samples, 0.0, atol=1e-6)
