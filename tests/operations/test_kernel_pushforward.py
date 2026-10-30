"""Pushforward and mixture kernels compose through binding and conditional sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    RecordBatch,
    RecordSpec,
    condition_on,
    conditional_distribution,
    log_prob,
    mean,
    mixture,
    pushforward,
    sample,
    variance,
    workflow_run,
)
from probpipe.core._dispatch import ResolutionError
from probpipe.core._specs import OutputSpec
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.families import MixtureDistribution
from probpipe.functions._call import ApplicabilityError

from ._laws import REAL


def kernel():
    return conditional_distribution(
        lambda theta, x, z: Normal("y", theta + x + z, 1.0),
        given_spec={"theta": REAL, "x": REAL, "z": REAL},
    )


@pytest.mark.parametrize("finite", [True, False])
def test_remaining_inputs_and_eager_binding(finite):
    mixing = (
        EmpiricalDistribution(jnp.array([-1.0, 3.0]), jnp.array([0.25, 0.75]), output_name="theta")
        if finite
        else Normal("theta", 2.0, 1.0)
    )
    q = mixture.using(max_evaluations=17)(kernel(), mixing)
    assert isinstance(q, ConditionalDistribution)
    assert set(q.given_spec) == {"x", "z"}
    assert q.event_spec == kernel().event_spec
    qx = condition_on.using(exact_only=True)(q, {"x": 2.0})
    assert set(qx.given_spec) == {"z"}
    with workflow_run(seed=0):
        law = condition_on(qx, {"z": 1.0})
    assert isinstance(law, MixtureDistribution if finite else EmpiricalDistribution)
    if finite:
        np.testing.assert_allclose(mean(law), 5.0, atol=1e-6)
        np.testing.assert_allclose(variance(law), 4.0, atol=5e-6)
        assert np.isfinite(log_prob(law, 5.0))
    else:
        assert law.num_atoms == 17
        assert law.provenance.metadata["exact"] is False
        before = np.asarray(law.atoms).copy()
        sample(law, (8,))
        np.testing.assert_array_equal(before, law.atoms)


def test_inference_and_static_check():
    k = conditional_distribution(lambda x: Normal("y", x, 1.0), given_spec={"x": REAL})
    calls = []

    def double(y):
        calls.append(True)
        return 2 * y

    assert pushforward.check(k, double).feasible is None
    assert calls == []
    q = pushforward(k, double)
    assert calls
    assert q.event_spec.outputs.keys() == {"double"}
    key = jax.random.key(4)
    np.testing.assert_array_equal(
        q._conditional_sample({"x": 1.0}, key, (5,)),
        2 * k._conditional_sample({"x": 1.0}, key, (5,)),
    )


def test_opaque_map_requires_declaration_without_running():
    calls = []
    f = Function(lambda y: calls.append(y) or float(y), traceable=False)
    k = conditional_distribution(lambda x: Normal("y", x, 1.0), given_spec={"x": REAL})
    with pytest.raises(ApplicabilityError, match="output_spec"):
        pushforward(k, f)
    assert calls == []


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
def test_conditional_sampler_matches_source_projection(shape):
    q = mixture(kernel(), Normal("theta", 0.0, 1.0))
    key = jax.random.key(8)
    given = {"x": 2.0, "z": -1.0}
    joint = kernel() & Normal("theta", 0.0, 1.0)
    expected = joint._conditional_sample(given, key, shape)["y"]
    actual = q._conditional_sample(given, key, shape)
    np.testing.assert_array_equal(actual, expected)
    assert np.shape(actual) == shape


def test_finite_binding_exact_only_and_generic_sampling_exact_only():
    k = condition_on(kernel(), {"z": 0.0})
    finite = mixture(k, EmpiricalDistribution(jnp.array([0.0, 1.0]), output_name="theta"))
    assert isinstance(condition_on.using(exact_only=True)(finite, {"x": 0.0}), MixtureDistribution)
    generic = mixture(k, Normal("theta", 0.0, 1.0))
    with pytest.raises(ResolutionError):
        condition_on.using(exact_only=True)(generic, {"x": 0.0})
    assert sample.using(exact_only=True).check(generic, given={"x": 0.0}).feasible is True


def test_joint_preserves_upstream_draw():
    k = conditional_distribution(
        lambda theta, x: EmpiricalDistribution(jnp.asarray([x + theta]), output_name="y"),
        given_spec={"theta": REAL, "x": REAL},
    )
    q = mixture(k, EmpiricalDistribution(jnp.array([2.0]), output_name="theta"))
    with workflow_run(seed=2):
        draws = sample(q & Normal("x", 0.0, 1.0), (11,))
    np.testing.assert_allclose(np.asarray(draws["y"]) - np.asarray(draws["x"]), 2.0, atol=5e-7)


def test_opaque_execution_counts_and_eager_atoms():
    calls = []

    def simulator(y):
        calls.append(float(y))
        return np.asarray(float(y) + 3.0, dtype=np.float32)

    f = Function(
        simulator, traceable=False, output_spec=OutputSpec(out=REAL), executor="sequential"
    )
    k = conditional_distribution(lambda x: Normal("y", x, 1.0), given_spec={"x": REAL})
    q = pushforward.using(max_evaluations=7)(k, f)
    assert calls == []
    sample(q, given={"x": 0.0}, sample_shape=(3,))
    assert len(calls) == 3
    law = condition_on(q, {"x": 0.0})
    assert len(calls) == 10
    sample(law, (4,))
    assert len(calls) == 10


def test_record_output_and_fixed_arguments():
    k = conditional_distribution(lambda x: Normal("y", x, 1.0), given_spec={"x": REAL})
    f = Function(
        lambda y, offset: {"out": y + offset}, output_spec=OutputSpec(RecordSpec(out=REAL))
    )
    q = pushforward(k, f, {"offset": 2.0})
    key = jax.random.key(1)
    draws = q._conditional_sample({"x": 0.0}, key, (3,))
    assert isinstance(draws, dict)
    np.testing.assert_allclose(
        draws["out"], k._conditional_sample({"x": 0.0}, key, (3,)) + 2.0, atol=1e-6
    )


def test_optional_and_nested_input_binding():
    k = conditional_distribution(
        lambda theta, group, offset=3.0: Normal("y", theta + group["a"] + group["b"] + offset, 1.0),
        given_spec={"theta": REAL, "group": RecordSpec(a=REAL, b=REAL)},
    )
    q = mixture(k, EmpiricalDistribution(jnp.array([2.0]), output_name="theta"))
    partial = condition_on(q, {"group/a": 4.0})
    law = condition_on(partial, {"group/b": 5.0})
    np.testing.assert_allclose(mean(law), 14.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("finite", [True, False])
def test_mixing_overrides_optional_input_after_binding(finite):
    k = conditional_distribution(
        lambda group, theta=0.0: EmpiricalDistribution(
            jnp.array([theta + group["a"] + group["b"]]), output_name="y"
        ),
        given_spec={"group": RecordSpec(a=REAL, b=REAL)},
    )
    mixing = EmpiricalDistribution(jnp.array([3.0]), output_name="theta")
    if not finite:
        mixing = Normal("theta", 3.0, 0.01)
    q = mixture.using(max_evaluations=500)(k, mixing)
    partial = condition_on(q, {"group/a": 1.0})
    with workflow_run(seed=3):
        law = condition_on(partial, {"group/b": 2.0})
    # Even a single mixing draw is within .05 of 3 with probability > .999999.
    np.testing.assert_allclose(mean(law), 6.0, rtol=0, atol=0.05)


def test_symbolic_dimensions_bind_through_pushforward():
    from probpipe import InputSpec, NumericArraySpec

    class IdentityKernel(ConditionalDistribution):
        def __init__(self):
            spec = NumericArraySpec(("n",), jnp.float32)
            super().__init__(
                InputSpec(x=spec, theta=REAL).with_optional("theta"), OutputSpec(y=spec)
            )

        def _condition_on(self, given, /, **options):
            return EmpiricalDistribution(
                jnp.stack([given["x"] + given.get("theta", 0.0)]), output_name="y"
            )

        def _conditional_sample(self, given, key, sample_shape=()):
            return jnp.broadcast_to(
                given["x"] + given.get("theta", 0.0), (*sample_shape, *given["x"].shape)
            )

    k = IdentityKernel()
    f = Function(
        lambda y: y + 1.0, output_spec=OutputSpec(out=NumericArraySpec(("n",), jnp.float32))
    )
    q = pushforward(k, f)
    given = {"x": jnp.arange(3.0, dtype=jnp.float32)}
    np.testing.assert_array_equal(sample(q, (2,), given=given), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    law = condition_on.using(exact_only=True)(q, given)
    np.testing.assert_array_equal(mean(law), [1.0, 2.0, 3.0])
    mixed = mixture(k, EmpiricalDistribution(jnp.array([2.0]), output_name="theta"))
    law = condition_on.using(exact_only=True)(mixed, given)
    np.testing.assert_array_equal(mean(law), [2.0, 3.0, 4.0])


def test_binding_without_conditional_sampling_and_without_provenance():
    from probpipe import ProvenanceMode, config

    class BindOnly(ConditionalDistribution):
        def __init__(self):
            super().__init__({"x": REAL}, OutputSpec(y=REAL))

        def _condition_on(self, given, /, **options):
            return EmpiricalDistribution(jnp.array([given["x"], given["x"] + 2.0]), output_name="y")

    config.provenance.mode = ProvenanceMode.OFF
    q = pushforward(BindOnly(), Function(lambda y: y * 2.0, output_spec=OutputSpec(out=REAL)))
    law = condition_on.using(exact_only=True)(q, {"x": 1.0})
    np.testing.assert_allclose(mean(law), 4.0, rtol=1e-6, atol=1e-6)
    draws = sample.using(exact_only=True)(q, (5,), given={"x": 1.0})
    assert set(np.asarray(draws)) <= {2.0, 6.0}


def test_binding_constructs_source_law_once():
    calls = []

    def build(x):
        calls.append(x)
        return Normal("y", x, 1.0)

    k = conditional_distribution(build, given_spec={"x": REAL})
    calls.clear()
    q = pushforward.using(max_evaluations=3)(
        k, Function(lambda y: y + 1.0, output_spec=OutputSpec(out=REAL))
    )
    assert calls == []
    assert condition_on.check(q, {"x": 0.0}).feasible is None
    assert calls == []
    condition_on(q, {"x": 0.0})
    assert len(calls) == 1


def test_batch_valued_events_keep_event_levels():
    from probpipe import NumericArrayBatch
    from probpipe.core._batch import BatchSpec

    class BatchKernel(ConditionalDistribution):
        def __init__(self):
            super().__init__({"x": REAL}, OutputSpec(trajectory=BatchSpec(REAL, time=2)))

        def _condition_on(self, given, /, **options):
            atoms = NumericArrayBatch(
                jnp.stack([jnp.array([given["x"], given["x"] + 1.0])]),
                ("sample", "time"),
                element=REAL,
            )
            return EmpiricalDistribution(atoms, event_spec=self.event_spec)

        def _conditional_sample(self, given, key, sample_shape=()):
            return jnp.broadcast_to(jnp.array([given["x"], given["x"] + 1.0]), (*sample_shape, 2))

    k = BatchKernel()
    f = Function(
        lambda trajectory: jnp.sum(jnp.asarray(trajectory)), output_spec=OutputSpec(total=REAL)
    )
    q = pushforward(k, f)
    np.testing.assert_array_equal(sample(q, (2, 3), given={"x": 4.0}), np.full((2, 3), 9.0))


def test_opaque_map_in_joint_evaluates_once_per_draw():
    calls = []

    def simulator(y):
        calls.append(float(y))
        return np.float32(float(y) + 1.0)

    f = Function(
        simulator, traceable=False, output_spec=OutputSpec(out=REAL), executor="sequential"
    )
    k = conditional_distribution(
        lambda x: EmpiricalDistribution(jnp.array([x]), output_name="y"), given_spec={"x": REAL}
    )
    q = pushforward.using(max_evaluations=100)(k, f)
    draws = sample(q & Normal("x", 0.0, 1.0), (5,))
    assert len(calls) == 5
    np.testing.assert_allclose(
        np.asarray(draws["out"]), np.asarray(draws["x"]) + 1.0, rtol=1e-6, atol=1e-6
    )


def test_construction_method_and_budget_survive_binding():
    k = conditional_distribution(
        lambda x: EmpiricalDistribution(jnp.array([x, x + 2.0]), output_name="y"),
        given_spec={"x": REAL},
    )
    f = Function(lambda y: y + 1.0, output_spec=OutputSpec(out=REAL))
    q = pushforward.using(method="sampling_lift", max_evaluations=9)(k, f)
    law = condition_on(q, {"x": 1.0})
    assert law.num_atoms == 9
    assert law.provenance.metadata["exact"] is False


def test_generic_joint_has_the_analytic_moments_and_covariance():
    k = conditional_distribution(
        lambda theta, x: Normal("y", theta + x, 1.0), given_spec={"theta": REAL, "x": REAL}
    )
    q = mixture(k, Normal("theta", 0.0, 1.0))
    with workflow_run(seed=2):
        draws = sample(q & Normal("x", 0.0, 1.0), (10000,))
    y, x = np.asarray(draws["y"]), np.asarray(draws["x"])
    # Seeds 0–3: |mean| <= .035, |variance-3| <= .084, |E[XY]-1| <= .054.
    np.testing.assert_allclose(np.mean(y), 0.0, rtol=0, atol=0.1)
    np.testing.assert_allclose(np.var(y), 3.0, rtol=0, atol=0.25)
    np.testing.assert_allclose(np.mean(x * y), 1.0, rtol=0, atol=0.16)


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("form", ["array", "record", "nested"])
def test_mixture_kernel_preserves_event_form(finite, form):
    def build(theta, x):
        law = Normal("y", theta + x, 1.0)
        if form == "array":
            return law
        if form == "record":
            return EmpiricalDistribution(
                RecordBatch({"y": jnp.stack([theta + x, theta + x + 2.0])}, "atom"),
                event_spec=OutputSpec(RecordSpec(y=REAL)),
            )
        return (law & Normal("z", 0.0, 1.0)).rename({"y": "group/y", "z": "group/z"})

    k = conditional_distribution(build, given_spec={"theta": REAL, "x": REAL})
    mu = (
        EmpiricalDistribution(jnp.array([0.0, 2.0]), output_name="theta")
        if finite
        else Normal("theta", 0.0, 1.0)
    )
    q = mixture.using(max_evaluations=5)(k, mu)
    assert q.event_spec == k.event_spec
    one = sample(q, given={"x": 1.0})
    assert one.spec == k.event_spec.term_spec
    law = condition_on(q, {"x": 1.0})
    assert law.event_spec == k.event_spec


def test_nested_kernel_pushforwards_keep_exact_binding():
    k = conditional_distribution(
        lambda x: EmpiricalDistribution(jnp.array([x, x + 2.0]), output_name="y"),
        given_spec={"x": REAL},
    )
    f = Function(lambda y: 2 * y, output_spec=OutputSpec(out=REAL))
    q = pushforward(pushforward(k, f), f)
    law = condition_on.using(exact_only=True)(q, {"x": 1.0})
    np.testing.assert_allclose(mean(law), 8.0, atol=1e-6, rtol=1e-6)
    assert law.provenance.metadata["exact"] is True


def test_finite_kernel_remains_exact_when_bound_inside_joint():
    k = condition_on(kernel(), {"z": 0.0})
    q = mixture(k, EmpiricalDistribution(jnp.array([0.0, 2.0]), output_name="theta"))
    law = condition_on.using(exact_only=True)(q & Normal("x", 0.0, 1.0), {"x": 3.0})
    np.testing.assert_allclose(mean(law), 4.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(variance(law), 2.0, rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize(
    "boundary",
    ["rename", "joint", "renamed_joint", "conditional_joint", "renamed_conditional_joint"],
)
@pytest.mark.parametrize("finite", [True, False])
def test_binding_fidelity_survives_composition(boundary, finite):
    calls = []

    def build(x):
        calls.append(x)
        return (
            EmpiricalDistribution(jnp.array([x, x + 2.0]), output_name="y")
            if finite
            else Normal("y", x, 1.0)
        )

    k = conditional_distribution(build, given_spec={"x": REAL})
    q = pushforward.using(max_evaluations=7)(
        k, Function(lambda y: y + 1.0, output_spec=OutputSpec(z=REAL))
    )
    if boundary in ("rename", "renamed_joint"):
        q = q.rename({"x": "w"})
    name = "w" if boundary in ("rename", "renamed_joint") else "x"
    if boundary in ("conditional_joint", "renamed_conditional_joint"):
        q = q & Normal("independent", 0.0, 1.0)
        if boundary == "renamed_conditional_joint":
            q = q.rename({"x": "w"})
            name = "w"
    elif boundary != "rename":
        q = q & Normal(name, 0.0, 1.0)
    calls.clear()
    if finite:
        law = condition_on.using(exact_only=True)(q, {name: 1.0})
        assert len(calls) == 1
        actual = mean(law)["z"] if "conditional_joint" in boundary else mean(law)
        np.testing.assert_allclose(actual, 3.0, rtol=1e-6)
    else:
        with pytest.raises(ResolutionError):
            condition_on.using(exact_only=True)(q, {name: 1.0})
        law = condition_on(q, {name: 1.0})
        if "conditional_joint" in boundary:
            assert law.factors[0].num_atoms == 7
        else:
            assert law.num_atoms == 7
        assert law.provenance.metadata["exact"] is False


def test_given_pushforward_keeps_exact_atoms_and_weights():
    from probpipe import Weights

    weights = Weights(n=2, weights=jnp.array([0.25, 0.75]))
    k = conditional_distribution(
        lambda x: EmpiricalDistribution(jnp.array([x, x + 2.0]), weights, output_name="y"),
        given_spec={"x": REAL},
    )
    f = Function(lambda y: y + 1.0, output_spec=OutputSpec(z=REAL))
    law = pushforward.using(max_evaluations=5)(k, f, given={"x": 1.0})
    assert law.num_atoms == 2
    np.testing.assert_array_equal(law.atoms, [2.0, 4.0])
    np.testing.assert_allclose(law.weights, [0.25, 0.75], rtol=1e-6)


def test_mixture_sampling_guard_receives_known_inputs():
    from probpipe.core._dispatch import Feasibility

    class Guarded(ConditionalDistribution):
        def __init__(self):
            super().__init__({"x": REAL, "theta": REAL}, OutputSpec(y=REAL))

        def _condition_on(self, given, /, **options):
            raise NotImplementedError("this kernel only supplies conditional sampling")

        def _conditional_sample(self, given, key, sample_shape=()):
            return jnp.full(sample_shape, given["x"] + given["theta"])

        def _conditional_sample_guard(self, given):
            return Feasibility(given["x"] >= 0.0, "x must be nonnegative")

    q = mixture(Guarded(), EmpiricalDistribution(jnp.array([2.0]), output_name="theta"))
    np.testing.assert_array_equal(
        sample.using(exact_only=True)(q, (3,), given={"x": 1.0}), [3.0] * 3
    )
    assert q._conditional_sample_guard({"x": -1.0}).feasible is False
    assert q._conditional_sample_guard().feasible is None
    renamed = q.rename({"x": "w"})
    np.testing.assert_array_equal(sample(renamed, (3,), given={"w": 1.0}), [3.0] * 3)
    assert renamed._conditional_sample_guard({"w": -1.0}).feasible is False


def test_renamed_kernel_binds_nested_inputs_without_evaluating_early():
    calls = []

    def build(x, offset):
        calls.append(True)
        return EmpiricalDistribution(jnp.array([x + offset]), output_name="y")

    k = conditional_distribution(build, given_spec={"x": REAL, "offset": REAL})
    q = pushforward(k, Function(lambda y: y + 1.0, output_spec=OutputSpec(z=REAL)))
    q = q.rename({"x": "group/x", "offset": "group/offset"})
    calls.clear()
    assert condition_on.check(q, {"group/x": 2.0}).feasible is not False
    partial = condition_on.using(exact_only=True)(q, {"group/x": 2.0})
    assert calls == []
    law = condition_on.using(exact_only=True)(partial, {"group/offset": 3.0})
    assert len(calls) == 1
    np.testing.assert_allclose(mean(law), 6.0, rtol=1e-6)
