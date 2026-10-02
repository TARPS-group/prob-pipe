"""The root-ancestor capture of lifted arguments.

A field view's root is its parent's, a batch element's is its stored law's, and
a bijector-transformed law's is its base's, so a lift groups each with its root:
the root draws once per repetition, and each member evaluates on that draw.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    BijectorTransformedDistribution,
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    MultivariateNormal,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    NumericRecordSpec,
    iterate,
    workflow_run,
)
from probpipe.distributions import FieldView
from probpipe.functions import _descendants
from probpipe.functions._plan import build_broadcast_plan, build_stochastic_plan
from probpipe.values import _binding
from tests._ops import mean, variance


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
        super().__init__(name, 0.0, 1.0)

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _RecordingMultivariateNormal(MultivariateNormal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__("base", jnp.zeros(2), cov=jnp.eye(2))

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


def _joint():
    return Normal("x", 0.0, 1.0) * Normal("y", 2.0, 1.0)


def _raw_mean(law):
    """The mean of *law* at its one leaf, whatever its packaging."""
    return np.asarray(jax.tree.leaves(mean.with_options(raw=True)(law))[0])


def _raw_variance(law):
    """The variance of *law* at its one leaf, whatever its packaging."""
    return np.asarray(jax.tree.leaves(variance.with_options(raw=True)(law))[0])


# -- Field views ---------------------------------------------------------------


class TestFieldViews:
    def test_a_view_captures_its_parents_root_and_projects_its_node(self):
        root = _joint()
        view = root["x"]
        captured = _descendants.capture_stochastic_consumer(view)
        draw = root._sample(jax.random.key(13), ())

        assert captured.root is root
        assert captured.record_path == ("x",)
        assert captured.descendant_descriptor is None
        np.testing.assert_array_equal(captured.evaluator(draw), draw["x"])

    def test_the_captured_projection_does_not_reread_the_live_view_path(self):
        root = _joint()
        view = root["x"]
        captured = _descendants.capture_stochastic_consumer(view)
        draw = root._sample(jax.random.key(13), ())

        object.__setattr__(view, "_path", "y")

        np.testing.assert_array_equal(captured.evaluator(draw), draw["x"])
        assert captured.record_path == ("x",)

    def test_a_selection_records_its_paths_in_its_descriptor(self):
        root = _joint()
        captured = _descendants.capture_stochastic_consumer(FieldView(root, ("y", "x")))
        draw = root._sample(jax.random.key(3), ())

        assert captured.root is root
        assert captured.record_path == ()
        expected = ("field-selection", ("base", ("root",)), ("paths", ("y", "x")))
        assert captured.descendant_descriptor == expected
        projected = captured.evaluator(draw)
        assert list(projected) == ["y", "x"]
        np.testing.assert_array_equal(projected["x"], draw["x"])

    def test_the_golden_selection_descriptor_and_digest_are_hard_coded(self):
        root = _joint()
        plan = _stochastic_plan({"value": FieldView(root, ("y", "x"))})
        descriptor = plan.source_groups[0].consumers[0].descendant_descriptor
        expected = (
            "stochastic-descendant",
            ("base_source_slot", 0),
            ("graph", ("field-selection", ("base", ("root",)), ("paths", ("y", "x")))),
        )

        assert descriptor == expected
        assert (
            _descendants.descriptor_digest(expected[2][1])
            == "fe5bb6d2c5f7cf14bdf47c0b33291a61adee065d46cf1455f22684b79ad530a2"
        )

    @pytest.mark.parametrize("cycle_kind", ["self", "pair"])
    def test_cyclic_field_view_graphs_fail_closed(self, cycle_kind):
        root = _joint()
        first = root["x"]
        if cycle_kind == "self":
            object.__setattr__(first, "_parent", first)
        else:
            second = root["y"]
            object.__setattr__(first, "_parent", second)
            object.__setattr__(second, "_parent", first)

        with pytest.raises(TypeError, match="Cyclic field view"):
            _descendants.capture_stochastic_consumer(first)

    def test_sibling_views_and_their_parent_form_one_plan_group(self):
        root = _joint()
        plan = _stochastic_plan({"root": root, "x": root["x"], "y": root["y"]})

        assert len(plan.source_groups) == 1
        assert plan.runtime_bindings[0].root is root
        assert tuple(consumer.record_path for consumer in plan.source_groups[0].consumers) == (
            (),
            ("x",),
            ("y",),
        )
        assert len(plan.random_events) == 1

    def test_detached_marginals_of_one_joint_form_separate_groups(self):
        root = _joint()
        plan = _stochastic_plan({"x": root._marginal("x"), "y": root._marginal("y")})
        assert len(plan.source_groups) == 2

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_a_function_of_a_parent_and_its_view_reads_one_parent_draw(self, dispatch):
        root = _joint()
        workflow = Function(
            "difference",
            lambda joint, x: joint["x"] - x,
            dispatch=dispatch,
            n_broadcast_samples=16,
        )

        with workflow_run(seed=31):
            result = workflow(root, root["x"])

        np.testing.assert_allclose(_raw_mean(result), 0.0, atol=1e-6)
        np.testing.assert_allclose(_raw_variance(result), 0.0, atol=1e-6)

    def test_two_views_of_one_path_co_sample(self):
        root = _joint()
        workflow = Function(
            "difference", lambda a, b: a - b, dispatch="sequential", n_broadcast_samples=16
        )

        with workflow_run(seed=33):
            result = workflow(root["x"], FieldView(root, "x"))

        np.testing.assert_allclose(_raw_variance(result), 0.0, atol=1e-6)


# -- Batch elements ------------------------------------------------------------


def _difference():
    return Function("difference", lambda a, b: a - b, dispatch="sequential", n_broadcast_samples=16)


class TestBatchElements:
    def test_an_element_captures_its_stored_law_as_root(self):
        root = _joint()
        batch = DistributionBatch("laws", [root, _joint()], "law")

        captured = _descendants.capture_stochastic_consumer(batch[0])

        assert captured.root is root
        assert captured.record_path == ()
        assert captured.descendant_descriptor is None

    def test_views_of_two_accesses_of_one_element_form_one_plan_group(self):
        root = _joint()
        batch = DistributionBatch("laws", [root, _joint()], "law")

        plan = _stochastic_plan({"root": root, "x": batch[0]["x"], "y": batch[0]["y"]})

        assert len(plan.source_groups) == 1
        assert plan.runtime_bindings[0].root is root
        assert tuple(consumer.record_path for consumer in plan.source_groups[0].consumers) == (
            (),
            ("x",),
            ("y",),
        )

    def test_two_accesses_of_one_element_co_sample(self):
        batch = DistributionBatch("laws", [_joint(), _joint()], "law")

        with workflow_run(seed=35):
            result = _difference()(batch[0]["x"], batch[0]["x"])

        np.testing.assert_allclose(_raw_variance(result), 0.0, atol=1e-6)

    def test_an_element_co_samples_with_its_stored_law(self):
        root = _joint()
        batch = DistributionBatch("laws", [root], "law")

        with workflow_run(seed=36):
            result = _difference()(root["x"], batch[0]["x"])

        np.testing.assert_allclose(_raw_variance(result), 0.0, atol=1e-6)

    def test_the_fields_of_an_iterated_law_co_sample(self):
        laws = iterate(step_fn=lambda law, _input: law, initial=_joint(), inputs=[1, 2])

        with workflow_run(seed=38):
            result = _difference()(laws[-1]["x"], laws[-1]["x"])

        np.testing.assert_allclose(_raw_variance(result), 0.0, atol=1e-6)

    def test_cyclic_element_graphs_fail_closed(self):
        element = DistributionBatch("laws", [_joint()], "law")[0]
        object.__setattr__(element, "_element_source", element)

        with pytest.raises(TypeError, match="Cyclic batch element"):
            _descendants.capture_stochastic_consumer(element)


# -- Bijector-transformed laws -------------------------------------------------


_BIJECTORS = [
    pytest.param(tfb.Identity(), id="identity"),
    pytest.param(tfb.Exp(), id="exp"),
    pytest.param(tfb.Shift(1.5), id="shift"),
    pytest.param(tfb.Scale(log_scale=0.25), id="scale"),
    pytest.param(tfb.Softplus(hinge_softness=0.75, low=0.1), id="softplus"),
    pytest.param(tfb.Sigmoid(low=-2.0, high=3.0), id="sigmoid"),
    pytest.param(tfb.Tanh(), id="tanh"),
    pytest.param(tfb.Chain([tfb.Exp(), tfb.Shift(1.0)]), id="chain"),
]


class TestTransformedLaws:
    @pytest.mark.parametrize("bijector", _BIJECTORS)
    def test_a_transformed_law_captures_its_base_as_root_and_maps_its_draws(self, bijector):
        base = Normal("base", 0.0, 1.0)
        descendant = BijectorTransformedDistribution("descendant", base, bijector)

        captured = _descendants.capture_stochastic_consumer(descendant)
        key = jax.random.PRNGKey(9)

        assert captured.root is base
        assert captured.record_path == ()
        assert captured.descendant_descriptor[0] == "transformed-descendant"
        np.testing.assert_allclose(
            captured.evaluator(base._sample(key, (11,))),
            descendant._sample(key, (11,)),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_the_captured_map_does_not_follow_a_later_change_to_the_law(self):
        base = Normal("base", 0.0, 1.0)
        descendant = BijectorTransformedDistribution("descendant", base, tfb.Shift(1.0))
        other = BijectorTransformedDistribution("other", base, tfb.Shift(9.0))
        captured = _descendants.capture_stochastic_consumer(descendant)
        descriptor = captured.descendant_descriptor

        object.__setattr__(descendant, "_bijector", other.bijector)

        np.testing.assert_allclose(captured.evaluator(jnp.asarray(0.0)), 1.0)
        assert captured.descendant_descriptor == descriptor

    def test_descriptors_read_the_bijector_and_not_the_labels(self):
        base = Normal("base", 0.0, 1.0)
        shift = BijectorTransformedDistribution("first", base, tfb.Shift(1.0)).bijector
        first = BijectorTransformedDistribution("first", base, shift)
        relabeled = BijectorTransformedDistribution("second", base.with_name("other"), shift)
        changed = BijectorTransformedDistribution("first", base, tfb.Shift(2.0))

        def descriptor(law):
            return _descendants.capture_stochastic_consumer(law).descendant_descriptor

        assert descriptor(first) == descriptor(relabeled)
        assert descriptor(first) != descriptor(changed)

    def test_nested_transforms_capture_the_innermost_base(self):
        base = Normal("base", 0.0, 1.0)
        inner = BijectorTransformedDistribution("inner", base, tfb.Exp())
        nested = BijectorTransformedDistribution("nested", inner, tfb.Shift(2.0))

        captured = _descendants.capture_stochastic_consumer(nested)
        key = jax.random.PRNGKey(5)

        assert captured.root is base
        assert dict(captured.descendant_descriptor[1:])["base"][0] == "transformed-descendant"
        np.testing.assert_allclose(
            captured.evaluator(base._sample(key, (4,))),
            jnp.exp(base._sample(key, (4,))) + 2.0,
            rtol=1e-6,
        )

    @pytest.mark.parametrize("cycle_kind", ["self"])
    def test_cyclic_transform_graphs_fail_closed(self, cycle_kind):
        descendant = BijectorTransformedDistribution(
            "descendant", Normal("base", 0.0, 1.0), tfb.Exp()
        )
        object.__setattr__(descendant, "_base", descendant)

        with pytest.raises(TypeError, match="Cyclic BijectorTransformedDistribution"):
            _descendants.capture_stochastic_consumer(descendant)

    def test_a_transform_of_a_view_reads_the_views_parent(self):
        root = _joint()
        x = root["x"]
        exp_x = BijectorTransformedDistribution("exp_x", x, tfb.Exp())
        shifted_x = BijectorTransformedDistribution("shifted_x", x, tfb.Shift(3.0))

        plan = _stochastic_plan({"root": root, "x": x, "exp_x": exp_x, "shifted_x": shifted_x})

        assert len(plan.source_groups) == 1
        assert plan.runtime_bindings[0].root is root
        consumers = plan.source_groups[0].consumers
        assert tuple(consumer.record_path for consumer in consumers) == (
            (),
            ("x",),
            ("x",),
            ("x",),
        )
        assert [consumer.descendant_descriptor is None for consumer in consumers] == [
            True,
            True,
            False,
            False,
        ]
        assert len(plan.random_events) == 1

    def test_the_descriptor_records_its_plan_local_base_source_slot(self):
        descendant = BijectorTransformedDistribution(
            "descendant", Normal("root", 0.0, 1.0), tfb.Exp()
        )
        plan = _stochastic_plan(
            {"independent": Normal("independent", 1.0, 1.0), "descendant": descendant}
        )
        assert plan.source_groups[1].consumers[0].descendant_descriptor[1] == (
            "base_source_slot",
            1,
        )

    def test_elementwise_transform_of_a_vector_event_matches_direct_sampling(self):
        calls = []
        root = _RecordingMultivariateNormal(calls)
        descendant = BijectorTransformedDistribution("descendant", root, tfb.Exp())
        captured = _descendants.capture_stochastic_consumer(descendant)
        key = jax.random.key(43)

        actual = _descendants.sample_captured_consumer(captured, key, (9,))
        expected = descendant._sample(key, (9,))

        assert [shape for _key, shape in calls] == [(9,), (9,)]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# -- The capture memo ----------------------------------------------------------


class TestCaptureMemo:
    def test_a_repeated_descendant_is_captured_once_per_plan(self):
        descendant = BijectorTransformedDistribution(
            "descendant", Normal("base", 0.0, 1.0), tfb.Shift(1.0)
        )
        with patch.object(
            _descendants, "_capture_descendant", wraps=_descendants._capture_descendant
        ) as capture:
            plan = _stochastic_plan({"first": descendant, "second": descendant})

        assert capture.call_count == 1
        assert len(plan.source_groups) == 1
        assert len(plan.source_groups[0].consumers) == 2

    def test_a_shared_transformed_ancestor_is_captured_once(self):
        shared = BijectorTransformedDistribution("shared", Normal("base", 0.0, 1.0), tfb.Exp())
        left = BijectorTransformedDistribution("left", shared, tfb.Shift(1.0))
        right = BijectorTransformedDistribution("right", shared, tfb.Scale(2.0))

        with patch.object(
            _descendants, "_capture_descendant", wraps=_descendants._capture_descendant
        ) as capture:
            plan = _stochastic_plan({"left": left, "right": right})

        captured = [call.args[0] for call in capture.call_args_list]
        assert sum(law is shared for law in captured) == 1
        assert len(plan.source_groups) == 1

    def test_the_memo_is_scoped_to_one_plan_build(self):
        descendant = BijectorTransformedDistribution(
            "descendant", Normal("base", 0.0, 1.0), tfb.Shift(1.0)
        )
        with patch.object(
            _descendants, "_capture_descendant", wraps=_descendants._capture_descendant
        ) as capture:
            _stochastic_plan({"value": descendant})
            _stochastic_plan({"value": descendant})

        assert capture.call_count == 2

    def test_the_session_rejects_a_corrupted_identity_cache_entry(self):
        session = _descendants._StochasticCaptureSession()
        cached_source = Normal("cached", 0.0, 1.0)
        requested_source = Normal("requested", 0.0, 1.0)
        captured_source = session.capture_consumer(cached_source)
        session.consumers[id(requested_source)] = (cached_source, captured_source)

        with pytest.raises(RuntimeError, match="consumer identity cache collision"):
            session.capture_consumer(requested_source)


# -- Lifts that co-sample a root and its descendants ---------------------------


class TestLifts:
    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_a_function_co_samples_a_root_and_its_transforms(self, dispatch):
        calls = []
        root = _RecordingNormal(calls)
        exponentiated = BijectorTransformedDistribution("exponentiated", root, tfb.Exp())
        shifted = BijectorTransformedDistribution("shifted", root, tfb.Shift(2.0))
        workflow = Function(
            "function",
            lambda base, exp_base, shifted_base: jnp.stack(
                (exp_base - jnp.exp(base), shifted_base - (base + 2.0))
            ),
            dispatch=dispatch,
            n_broadcast_samples=16,
        )

        with workflow_run(seed=37):
            result = workflow(root, exponentiated, shifted)

        assert [shape for _key, shape in calls] == [(16,)]
        np.testing.assert_allclose(_raw_mean(result), 0.0, atol=1e-6)
        np.testing.assert_allclose(_raw_variance(result), 0.0, atol=1e-6)

    def test_a_lone_descendant_samples_its_root_once(self):
        calls = []
        root = _RecordingNormal(calls)
        descendant = BijectorTransformedDistribution("descendant", root, tfb.Exp())
        workflow = Function(
            "function", lambda value: value, dispatch="sequential", n_broadcast_samples=14
        )

        with workflow_run(seed=39):
            workflow(descendant)

        assert [shape for _key, shape in calls] == [(14,)]

    def test_sequential_and_jax_dispatch_agree_on_the_captured_graph(self):
        def run(dispatch):
            root = Normal("base", 0.0, 1.0)
            exponentiated = BijectorTransformedDistribution("exponentiated", root, tfb.Exp())
            workflow = Function(
                "difference",
                lambda base, exp_base: exp_base - jnp.exp(base),
                dispatch=dispatch,
                n_broadcast_samples=16,
            )
            with workflow_run(seed=41):
                return workflow(root, exponentiated)

        for dispatch in ("sequential", "jax"):
            np.testing.assert_allclose(_raw_variance(run(dispatch)), 0.0, atol=1e-6)

    def test_an_exact_empirical_root_and_its_transform_enumerate_without_sampling(self):
        root = EmpiricalDistribution(
            "base", jnp.asarray([1.0, 4.0]), weights=jnp.asarray([0.2, 0.8])
        )
        exponentiated = BijectorTransformedDistribution("exponentiated", root, tfb.Exp())
        workflow = Function(
            "function",
            lambda base, exp_base: exp_base - jnp.exp(base),
            dispatch="sequential",
            n_broadcast_samples=16,
        )

        with patch.object(type(root), "_sample", side_effect=AssertionError("sampled exact root")):
            result = workflow(root, exponentiated)

        np.testing.assert_allclose(_raw_mean(result), 0.0, atol=1e-5)

    def test_a_transform_passed_before_its_empirical_root_enumerates(self):
        root = EmpiricalDistribution(
            "base", jnp.asarray([1.0, 4.0]), weights=jnp.asarray([0.2, 0.8])
        )
        exponentiated = BijectorTransformedDistribution("exponentiated", root, tfb.Exp())
        workflow = Function(
            "function",
            lambda exp_base, base: exp_base - jnp.exp(base),
            dispatch="sequential",
            n_broadcast_samples=16,
        )

        with patch.object(type(root), "_sample", side_effect=AssertionError("sampled exact root")):
            result = workflow.with_options(exact_only=True)(exponentiated, root)

        assert result.num_atoms == 2
        np.testing.assert_allclose(_raw_mean(result), 0.0, atol=1e-5)

    def test_an_exact_record_projection_then_transform_stays_diagonal(self):
        root = EmpiricalDistribution(
            "joint",
            NumericRecordBatch(
                "draws",
                {"x": jnp.asarray([1.0, 4.0]), "y": jnp.asarray([10.0, 40.0])},
                "draw",
                element_spec=NumericRecordSpec(x=(), y=()),
            ),
            weights=jnp.asarray([0.3, 0.7]),
        )
        x = root["x"]
        exponentiated_x = BijectorTransformedDistribution("exponentiated_x", x, tfb.Exp())
        workflow = Function(
            "function",
            lambda joint, x_value, exp_x: jnp.stack(
                (joint["x"] - x_value, exp_x - jnp.exp(x_value))
            ),
            dispatch="sequential",
            n_broadcast_samples=16,
        )

        result = workflow(root, x, exponentiated_x)

        np.testing.assert_allclose(_raw_mean(result), 0.0, atol=1e-5)

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_a_nested_sweep_samples_a_shared_root_once_per_cell(self, dispatch):
        rows = NumericRecordBatch.stack(
            [NumericRecord("row", offset=float(index)) for index in range(3)],
            level_name="draw",
        )
        calls = []
        root = _RecordingNormal(calls)
        exponentiated = BijectorTransformedDistribution("exponentiated", root, tfb.Exp())
        workflow = Function(
            "function",
            lambda row, base, exp_base: exp_base - jnp.exp(base),
            dispatch=dispatch,
            n_broadcast_samples=12,
        )

        with workflow_run(seed=47):
            workflow(rows, root, exponentiated)

        assert [shape for _key, shape in calls] == [(12,), (12,), (12,)]


# -- Weights of exact empirical roots ------------------------------------------


class TestEmpiricalRootWeights:
    def test_a_lift_of_a_transform_minus_its_base_has_mean_zero(self):
        """``b = exp(a)`` reads ``a``'s draw, so ``b - exp(a)`` is zero on every repetition."""
        base = Normal("a", 0.0, 1.0)
        transformed = BijectorTransformedDistribution("b", base, tfb.Exp())
        workflow = Function(
            "difference",
            lambda a, b: b - jnp.exp(a),
            dispatch="sequential",
            n_broadcast_samples=64,
        )

        with workflow_run(seed=53):
            result = workflow(base, transformed)

        np.testing.assert_allclose(_raw_mean(result), 0.0, atol=1e-6)

    def test_exact_empirical_root_and_descendant_keep_weights_once(self):
        root = EmpiricalDistribution(
            "base",
            jnp.asarray([1.0, 4.0]),
            weights=jnp.asarray([0.2, 0.8]),
        )
        exponentiated = BijectorTransformedDistribution("exponentiated", root, tfb.Exp())
        workflow = Function(
            name="function",
            fn=lambda base, exp_base: exp_base - jnp.exp(base),
            dispatch="sequential",
            n_broadcast_samples=16,
            include_inputs=True,
        )

        with patch.object(type(root), "_sample", side_effect=AssertionError("sampled exact root")):
            result = workflow(root, exponentiated)

        assert result.num_atoms == 2
        np.testing.assert_allclose(np.asarray(result.atoms["function"]), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.weights, jnp.asarray([0.2, 0.8]))
        np.testing.assert_allclose(
            np.asarray(result.atoms["exp_base"]),
            jnp.exp(np.asarray(result.atoms["base"])),
            rtol=1e-6,
        )

    def test_exact_record_projection_then_transform_keeps_the_root_weights(self):
        root = EmpiricalDistribution(
            "joint",
            NumericRecordBatch(
                "draws",
                {"x": jnp.asarray([1.0, 4.0]), "y": jnp.asarray([10.0, 40.0])},
                "draw",
                element_spec=NumericRecordSpec(x=(), y=()),
            ),
            weights=jnp.asarray([0.3, 0.7]),
        )
        x = root["x"]
        exponentiated_x = BijectorTransformedDistribution("exponentiated_x", x, tfb.Exp())
        workflow = Function(
            name="function",
            fn=lambda joint, x_value, exp_x: jnp.stack(
                (joint["x"] - x_value, exp_x - jnp.exp(x_value))
            ),
            dispatch="sequential",
            n_broadcast_samples=16,
            include_inputs=True,
        )

        result = workflow(root, x, exponentiated_x)

        assert result.num_atoms == 2
        np.testing.assert_allclose(np.asarray(result.atoms["function"]), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.weights, jnp.asarray([0.3, 0.7]))

    def test_mixed_empirical_descendant_multiplies_root_weight_once(self):
        exact_root = EmpiricalDistribution(
            "exact",
            jnp.asarray([1.0, 4.0]),
            weights=jnp.asarray([0.2, 0.8]),
        )
        exponentiated = BijectorTransformedDistribution("exponentiated", exact_root, tfb.Exp())
        sampled_calls = []
        sampled = _RecordingNormal(sampled_calls, name="sampled")
        workflow = Function(
            name="function",
            fn=lambda exact, exp_exact, noise: jnp.stack((exp_exact - jnp.exp(exact), noise)),
            dispatch="sequential",
            n_broadcast_samples=12,
            include_inputs=True,
        )

        with workflow_run(seed=45):
            result = workflow(exact_root, exponentiated, sampled)

        assert result.num_atoms == 12
        assert [shape for _key, shape in sampled_calls] == [(12,)]
        np.testing.assert_allclose(np.asarray(result.atoms["function"])[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(result.atoms["exp_exact"]),
            jnp.exp(np.asarray(result.atoms["exact"])),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            result.weights,
            jnp.repeat(jnp.asarray([0.2, 0.8]) / 6.0, 6),
            rtol=1e-6,
        )
