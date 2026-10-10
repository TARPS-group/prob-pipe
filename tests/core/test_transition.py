"""Tests for probpipe.core.transition — iterate, with_conversion, with_resampling."""

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Distribution,
    EmpiricalDistribution,
    Function,
    MultivariateNormal,
    Normal,
    Provenance,
    Weights,
    converter_registry,
    iterate,
    with_conversion,
    with_resampling,
    workflow_run,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def initial():
    """A simple 2-D EmpiricalDistribution centered at zero."""
    return EmpiricalDistribution(jnp.zeros((50, 2)), component="initial", label="initial")


def _component(dist):
    """The one component of *dist*'s whole-term event, which the steps keep."""
    (component,) = dist.event_spec.components
    return component


def shift_step(dist, offset):
    """Shift every atom by a scalar. Returns a bare Distribution under *dist*'s label."""
    return EmpiricalDistribution(
        dist.atoms.values + offset, component=_component(dist), label=dist.label
    )


def provenance_step(dist, value):
    """A step that sets its own provenance."""
    new_dist = EmpiricalDistribution(dist.atoms.values + value, component=_component(dist))
    new_dist.with_provenance(Provenance("custom_step", parents=(dist,), metadata={"value": value}))
    return new_dist


def _produced(element):
    """The provenance of the law a batch element views, which the fold produced."""
    return element.provenance.parents[1].provenance


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------


class TestIterate:
    def test_basic(self, initial):
        """iterate returns a DistributionBatch including the initial.

        Indexing, iteration, and len all work, and an element is a view of the
        law the fold produced at that step.
        """
        from probpipe import DistributionBatch

        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0])
        assert isinstance(dists, DistributionBatch)
        assert len(dists) == 3  # initial + 2 steps
        assert dists[0].atoms is initial.atoms
        assert all(isinstance(d, Distribution) for d in dists)

    def test_values(self, initial):
        """Step results have correct sample values."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0])
        assert jnp.allclose(dists[1].atoms.values, jnp.ones((50, 2)))
        assert jnp.allclose(dists[2].atoms.values, jnp.full((50, 2), 3.0))

    def test_provenance_auto_attach(self, initial):
        """Provenance is auto-attached when step function doesn't set it."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0])
        produced = _produced(dists[1])
        assert produced is not None
        assert produced.operation == "iterate"
        assert produced.metadata["step"] == 0
        assert len(produced.parents) == 1
        assert produced.parents[0].label == initial.label

    def test_provenance_preserved(self, initial):
        """Provenance set by step function is not overwritten."""
        dists = iterate(step_fn=provenance_step, initial=initial, inputs=[1.0])
        produced = _produced(dists[1])
        assert produced.operation == "custom_step"
        assert produced.metadata["value"] == 1.0

    def test_provenance_chain(self, initial):
        """Each step's provenance points to the previous distribution."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0, 3.0])
        for step in (1, 2, 3):
            previous = _produced(dists[step]).parents[0]
            assert previous.label == "initial"
            assert previous.provenance == _produced(dists[step - 1])

    def test_callback(self, initial):
        """Callback receives correct (index, dist) pairs."""
        recorded = []

        def cb(i, dist):
            recorded.append((i, float(dist.atoms.values[0, 0])))

        iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0, 3.0], callback=cb)
        assert len(recorded) == 3
        assert recorded[0] == (0, 1.0)
        assert recorded[1] == (1, 3.0)
        assert recorded[2] == (2, 6.0)

    def test_callback_early_stop(self, initial):
        """Callback returning False truncates iteration."""

        def stop_after_one(i, dist):
            if i >= 1:
                return False

        dists = iterate(
            step_fn=shift_step,
            initial=initial,
            inputs=[1.0, 2.0, 3.0, 4.0],
            callback=stop_after_one,
        )
        # initial + steps 0 and 1 (stops after callback for step 1)
        assert len(dists) == 3

    def test_empty_inputs(self, initial):
        """Empty inputs returns list with only the initial distribution."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[])
        assert len(dists) == 1
        assert dists[0].atoms is initial.atoms

    def test_bad_return_type(self, initial):
        """Non-Distribution return raises TypeError."""

        def bad_step(dist, inp):
            return "not a distribution"

        with pytest.raises(TypeError, match="must return a Distribution, got str"):
            iterate(step_fn=bad_step, initial=initial, inputs=[1])

    def test_final_is_last(self, initial):
        """dists[-1] is the final distribution."""
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0])
        assert jnp.allclose(dists[-1].atoms.values, jnp.full((50, 2), 3.0))

    def test_the_laws_are_on_one_level_named_iterate(self, initial):
        dists = iterate(step_fn=shift_step, initial=initial, inputs=[1.0, 2.0])
        assert dists.level_names == ("iterate",)
        assert dists.event_spec == initial.event_spec

    def test_a_step_that_changes_the_event_declaration_raises(self, initial):
        """The visited laws share one event declaration, as a batch's elements do."""

        def widen(dist, inp):
            return EmpiricalDistribution(jnp.zeros((50, 3)), component=_component(dist))

        with pytest.raises(TypeError):
            iterate(step_fn=widen, initial=initial, inputs=[1.0])


# ---------------------------------------------------------------------------
# with_conversion
# ---------------------------------------------------------------------------


class TestWithConversion:
    def test_returns_function(self):
        """with_conversion returns a Function."""
        step = with_conversion(shift_step, MultivariateNormal)
        assert isinstance(step, Function)
        assert "with_conversion" in step._label
        assert "shift_step" in step._label
        assert "MultivariateNormal" in step._label

    def test_converts_output(self, initial):
        """Output is converted to target type."""
        step = with_conversion(shift_step, MultivariateNormal)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert isinstance(dists[-1], MultivariateNormal)

    def test_pre_conversion_in_provenance_parents(self, initial):
        """Pre-conversion distribution is accessible via provenance parents."""
        step = with_conversion(shift_step, MultivariateNormal)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        converted = dists[-1]
        # The converter sets provenance with the source dist as parent
        assert converted.provenance is not None
        assert len(converted.provenance.parents) > 0

    def test_multi_step_stays_parametric(self):
        """Each step produces a parametric distribution usable as next prior."""
        from probpipe import sample as pp_sample

        def parametric_step(dist, shift):
            samples = jnp.asarray(pp_sample(dist, sample_shape=(50,))) + shift
            return EmpiricalDistribution(samples, component="x")

        initial = MultivariateNormal("x", loc=jnp.zeros(2), cov=jnp.eye(2))
        step = with_conversion(parametric_step, MultivariateNormal)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0, 2.0, 3.0])
        for d in dists[1:]:
            assert isinstance(d, MultivariateNormal)


# ---------------------------------------------------------------------------
# with_resampling
# ---------------------------------------------------------------------------

#: The particle count of the resampling-randomness tests.
_PARTICLES = 100


def ten_heavy_particles(dist, inp):
    """The particles 0 to 99, of which the first ten hold the weight equally, so ESS / N is 0.1."""
    log_w = jnp.where(jnp.arange(_PARTICLES) < 10, 0.0, -100.0)
    atoms = jnp.arange(_PARTICLES, dtype=jnp.float32).reshape(_PARTICLES, 1)
    return EmpiricalDistribution(atoms, Weights(log_weights=log_w), component="x")


def drawn_heavy_particles(dist, inp):
    """Draws of *dist* by the converter registry, of which the first ten hold the weight equally."""
    drawn = converter_registry.convert(dist, EmpiricalDistribution, num_samples=_PARTICLES)
    log_w = jnp.where(jnp.arange(_PARTICLES) < 10, 0.0, -100.0)
    return EmpiricalDistribution(drawn.atoms.values, Weights(log_weights=log_w), component="x")


def _resampled_atoms(step, seed=None):
    """The atoms of one step of *step*, in ``workflow_run(seed=seed)`` when *seed* is given."""
    initial = EmpiricalDistribution(jnp.zeros((_PARTICLES, 1)), component="x")
    with contextlib.nullcontext() if seed is None else workflow_run(seed=seed):
        dists = iterate(step_fn=step, initial=initial, inputs=[0.0])
    return np.asarray(dists[-1].atoms.values)


class TestWithResampling:
    def test_returns_function(self):
        """with_resampling returns a Function."""
        step = with_resampling(shift_step, ess_threshold=0.5)
        assert isinstance(step, Function)
        assert "with_resampling" in step._label
        assert "shift_step" in step._label

    def test_no_resample_uniform(self):
        """Uniform weights -> no resampling (ESS = N)."""
        initial = EmpiricalDistribution(jnp.zeros((100, 2)), component="x")
        step = with_resampling(shift_step, ess_threshold=0.5)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert _produced(dists[-1]).operation == "workflow.with_resampling(shift_step)"

    def test_resample_degenerate(self):
        """Highly non-uniform weights -> resampling triggered."""
        n = 100
        log_w = jnp.full(n, -100.0).at[0].set(0.0)
        samples = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2)

        def weighted_step(dist, inp):
            return EmpiricalDistribution(samples, Weights(log_weights=log_w), component="x")

        initial = EmpiricalDistribution(jnp.zeros((n, 2)), component="x")
        step = with_resampling(weighted_step, ess_threshold=0.5)
        dists = iterate(step_fn=step, initial=initial, inputs=[0.0])
        resampled = dists[-1]
        np.testing.assert_allclose(resampled.weights, 1.0 / n)
        assert _produced(resampled).operation == "workflow.with_resampling(weighted_step)"

    def test_resample_stores_ess_in_metadata(self):
        """Pre-resampling ESS is stored in provenance metadata."""
        n = 50
        log_w = jnp.full(n, -100.0).at[0].set(0.0)

        def weighted_step(dist, inp):
            return EmpiricalDistribution(
                jnp.zeros((n, 2)), Weights(log_weights=log_w), component="x"
            )

        initial = EmpiricalDistribution(jnp.zeros((n, 2)), component="x")
        step = with_resampling(weighted_step, ess_threshold=0.5)
        raw = step.apply(initial, 0.0)
        wrapped = iterate(step_fn=step, initial=initial, inputs=[0.0])[-1]
        assert raw.provenance.operation == "resample"
        assert "ess" in raw.provenance.metadata
        assert _produced(wrapped).operation == "workflow.with_resampling(weighted_step)"
        assert "ess_ratio" in raw.provenance.metadata
        assert raw.provenance.metadata["ess_ratio"] < 0.5

    def test_non_empirical_passthrough(self):
        """Non-EmpiricalDistribution passes through unchanged."""
        initial = MultivariateNormal("z", loc=jnp.zeros(2), cov=jnp.eye(2))

        def mvn_step(dist, inp):
            return MultivariateNormal("z", loc=jnp.ones(2) * inp, cov=jnp.eye(2))

        step = with_resampling(mvn_step, ess_threshold=0.5)
        dists = iterate(step_fn=step, initial=initial, inputs=[1.0])
        assert isinstance(dists[-1], MultivariateNormal)


class TestResamplingRandomness:
    """Each resampling is a workflow-owned random event of the enclosing scope."""

    def test_a_seeded_scope_reproduces_the_resampling(self):
        """One wrapper called in two scopes of one seed resamples the same particles."""
        step = with_resampling(ten_heavy_particles)
        np.testing.assert_array_equal(
            _resampled_atoms(step, seed=0), _resampled_atoms(step, seed=0)
        )

    def test_another_seed_resamples_other_particles(self):
        first = _resampled_atoms(with_resampling(ten_heavy_particles), seed=0)
        second = _resampled_atoms(with_resampling(ten_heavy_particles), seed=1)
        assert not np.array_equal(first, second)

    def test_two_resamplings_in_one_scope_differ(self):
        initial = EmpiricalDistribution(jnp.zeros((_PARTICLES, 1)), component="x")
        step = with_resampling(ten_heavy_particles)
        with workflow_run(seed=0):
            dists = iterate(step_fn=step, initial=initial, inputs=[0.0, 0.0])
        assert not np.array_equal(dists[1].atoms.values, dists[2].atoms.values)

    def test_a_resampling_outside_any_scope_is_fresh(self):
        first = _resampled_atoms(with_resampling(ten_heavy_particles))
        second = _resampled_atoms(with_resampling(ten_heavy_particles))
        assert not np.array_equal(first, second)

    def test_apply_resamples_after_a_step_that_draws(self):
        """The step's draw and the resampling are two events of one ``apply`` evaluation."""
        step = with_resampling(drawn_heavy_particles)
        initial = Normal("x", loc=0.0, scale=1.0)
        with workflow_run(seed=0):
            first = step.apply(initial, 0.0)
        with workflow_run(seed=0):
            second = step.apply(initial, 0.0)

        assert first.provenance.operation == "resample"
        np.testing.assert_allclose(first.weights, 1.0 / _PARTICLES)
        np.testing.assert_array_equal(first.atoms.values, second.atoms.values)


# ---------------------------------------------------------------------------
# Nestability
# ---------------------------------------------------------------------------


class TestNestability:
    def test_nested_iterate(self, initial):
        """A step function can call iterate internally."""

        def inner_step(dist, value):
            return EmpiricalDistribution(dist.atoms.values + value, component=_component(dist))

        def outer_step(dist, batch):
            """Each outer step runs an inner iterate loop."""
            inner_dists = iterate(inner_step, dist, batch)
            return inner_dists[-1]

        outer_inputs = [[0.1, 0.2], [0.3, 0.4, 0.5]]
        dists = iterate(step_fn=outer_step, initial=initial, inputs=outer_inputs)
        assert len(dists) == 3  # initial + 2 outer steps
        # Total shift: (0.1+0.2) + (0.3+0.4+0.5) = 1.5
        assert jnp.allclose(dists[-1].atoms.values, jnp.full((50, 2), 1.5))
