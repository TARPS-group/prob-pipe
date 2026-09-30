"""The empirical law: a finite, possibly weighted set of atoms of any event type.

An ``EmpiricalDistribution`` takes its atoms in the event's batch form, or as an
array whose leading axis indexes array atoms, with weights that default to
uniform and are normalized. Without a declaration, record atoms expose their
fields and any other atoms form a whole-term event whose component defaults to
the law's name; an ``event_spec`` names the components, and a type hole is
filled from the atoms. The law samples by weighted resampling, integrates any
function exactly over its atoms, and has exact marginals, which are the
empirical laws of the projected atoms under the same weights. A numeric event
also has the weighted mean, variance, covariance, and per-coordinate quantiles,
and no event has a density.
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    FunctionBatch,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    OpaqueBatch,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordBatch,
    RecordSpec,
    Weights,
    sample,
)
from probpipe.core._dispatch import Feasibility
from probpipe.core.constraints import positive
from probpipe.distributions import DistributionBatch, FieldView, NumericDistribution
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
)
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.linalg import DenseLinOp
from probpipe.operations import expectation, expectation_method_registry

_MOMENTS = (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile)
_ALWAYS = (SupportsSampling, SupportsExpectation, SupportsMarginals)

#: Four scalar atoms and unnormalized weights whose normalized form is (0.1, 0.2, 0.3, 0.4).
_VALUES = jnp.array([1.0, 2.0, 4.0, 7.0])
_WEIGHTS = jnp.array([1.0, 2.0, 3.0, 4.0])
_NORMALIZED = np.array([0.1, 0.2, 0.3, 0.4])

#: Three record atoms over ``b`` of shape (2,) and ``a``, declared in that order.
_B = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
_A = jnp.array([1.0, 2.0, 3.0])
_RECORD_WEIGHTS = np.array([0.5, 0.25, 0.25])
_RECORD_SPEC = RecordSpec(b=(2,), a=())

#: A mixed record: an opaque label and a numeric group.
_MIXED_SPEC = RecordSpec(label=None, g=RecordSpec(u=(), v=(2,)))
_LABELS = np.array(["north", "south", "east"], dtype=object)
_U = jnp.array([1.0, 2.0, 3.0])
_V = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def _array_law(weights=_WEIGHTS) -> EmpiricalDistribution:
    return EmpiricalDistribution("theta", _VALUES, weights)


def _record_atoms() -> NumericRecordBatch:
    return NumericRecordBatch("rows", {"b": _B, "a": _A}, "row", element_spec=_RECORD_SPEC)


def _record_law() -> EmpiricalDistribution:
    return EmpiricalDistribution("post", _record_atoms(), jnp.asarray(_RECORD_WEIGHTS))


def _mixed_atoms() -> RecordBatch:
    columns = {"label": _LABELS, "g/u": _U, "g/v": _V}
    return RecordBatch("sites", columns, "site", element_spec=_MIXED_SPEC)


def _opaque_law() -> EmpiricalDistribution:
    atoms = OpaqueBatch("labels", ["north", "south", "east"], "site")
    return EmpiricalDistribution("where", atoms, jnp.array([0.2, 0.3, 0.5]))


def _weighted_sum(weights, values):
    """The sum ``Σᵢ wᵢ vᵢ``, computed term by term."""
    return sum(float(w) * np.asarray(v, dtype=float) for w, v in zip(weights, values))


def _flat_record_atoms() -> np.ndarray:
    """The flat coordinates of each record atom: ``b`` raveled, then ``a``."""
    return np.array([[*np.asarray(b), float(a)] for b, a in zip(_B, _A)])


# -- Event completion ---------------------------------------------------------


class TestEventCompletion:
    def test_record_atoms_expose_their_fields(self):
        law = _record_law()
        assert law.event_spec == OutputSpec(_RECORD_SPEC)
        assert law.event_spec.exposes_record
        assert list(law.event_spec.components) == ["b", "a"]

    def test_array_atoms_form_a_whole_term_under_the_law_name(self):
        law = EmpiricalDistribution("theta", jnp.zeros((5, 2)))
        assert law.event_spec == OutputSpec(theta=NumericArraySpec((2,), jnp.float32))
        assert law.event_shape == (2,)

    def test_opaque_atoms_form_a_whole_term_under_the_law_name(self):
        assert _opaque_law().event_spec == OutputSpec(where=OpaqueSpec())

    def test_laws_as_atoms_form_a_random_measure(self):
        laws = DistributionBatch("laws", [Normal("x", 0.0, 1.0), Normal("x", 1.0, 2.0)], "law")
        law = EmpiricalDistribution("measure", laws)
        assert law.event_spec == OutputSpec(measure=laws.element_spec)

    def test_the_component_is_captured_once(self):
        renamed = EmpiricalDistribution("theta", jnp.zeros((5, 2))).with_name("other")
        assert list(renamed.event_spec.components) == ["theta"]

    def test_a_batch_keeps_its_element_declaration(self):
        declared = NumericArraySpec((2,), jnp.float32, positive)
        atoms = NumericArrayBatch("draws", jnp.ones((4, 2)), "draw", element_spec=declared)
        assert EmpiricalDistribution("x", atoms).event_spec == OutputSpec(x=declared)

    @pytest.mark.parametrize(
        ("atoms", "spec"),
        [
            pytest.param(jnp.zeros((3, 2)), NumericArraySpec((2,), jnp.float32), id="array"),
            pytest.param(_record_atoms(), _RECORD_SPEC, id="record"),
            pytest.param(OpaqueBatch("labels", ["a", "b"], "site"), OpaqueSpec(), id="opaque"),
        ],
    )
    def test_a_type_hole_is_filled_from_the_atoms(self, atoms, spec):
        law = EmpiricalDistribution("posterior", atoms, event_spec=OutputSpec(beta=None))
        assert law.name == "posterior"
        assert law.event_spec == OutputSpec(beta=spec)

    def test_a_declared_type_binds_its_dimensions_from_the_atoms(self):
        law = EmpiricalDistribution(
            "posterior", jnp.zeros((4, 3)), event_spec=OutputSpec(beta=NumericArraySpec(("k",)))
        )
        assert law.event_spec == OutputSpec(beta=NumericArraySpec((3,), jnp.float32))

    def test_a_declared_record_is_completed_with_the_atoms_record(self):
        law = EmpiricalDistribution("post", _record_atoms(), event_spec=OutputSpec(_RECORD_SPEC))
        assert law.event_spec == OutputSpec(_RECORD_SPEC)

    def test_a_declared_type_that_does_not_unify_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            EmpiricalDistribution(
                "posterior", jnp.zeros((4, 3)), event_spec=OutputSpec(beta=NumericArraySpec((2,)))
            )

    def test_an_exposed_record_declaration_needs_record_atoms(self):
        with pytest.raises(TypeError, match="RecordSpec"):
            EmpiricalDistribution("x", jnp.zeros((4, 3)), event_spec=OutputSpec(RecordSpec(a=(3,))))

    def test_an_event_spec_is_an_output_spec(self):
        with pytest.raises(TypeError, match="OutputSpec"):
            EmpiricalDistribution("x", jnp.zeros(4), event_spec=NumericArraySpec(()))

    def test_every_batch_axis_indexes_atoms(self):
        atoms = NumericArrayBatch(
            "draws",
            jnp.arange(12.0).reshape(2, 3, 2),
            ("chain", "draw"),
            element_spec=NumericArraySpec((2,)),
        )
        law = EmpiricalDistribution("x", atoms)
        assert law.num_atoms == 6
        assert law.atoms is atoms
        assert law.event_spec == OutputSpec(x=NumericArraySpec((2,)))
        assert jnp.allclose(law._mean(), jnp.array([5.0, 6.0]))

    def test_a_bare_array_is_stored_as_a_batch_of_atoms(self):
        law = _array_law()
        assert isinstance(law.atoms, NumericArrayBatch)
        assert law.atoms.level_names == ("atom",)
        assert jnp.array_equal(law.atoms.values, _VALUES)

    @pytest.mark.parametrize(
        ("atoms", "numeric"),
        [
            pytest.param(_VALUES, True, id="array"),
            pytest.param(_record_atoms(), True, id="numeric-record"),
            pytest.param(_mixed_atoms(), False, id="mixed-record"),
            pytest.param(OpaqueBatch("labels", ["a", "b"], "site"), False, id="opaque"),
        ],
    )
    def test_numeric_membership_follows_the_atoms(self, atoms, numeric):
        assert isinstance(EmpiricalDistribution("x", atoms), NumericDistribution) is numeric


class TestConstructionErrors:
    @pytest.mark.parametrize(
        "atoms",
        [
            pytest.param(["a", "b"], id="list"),
            pytest.param(Record("r", a=jnp.zeros(3)), id="one-record"),
            pytest.param(np.array(["a", "b"], dtype=object), id="object-array"),
        ],
    )
    def test_atoms_are_a_batch_or_a_numeric_array(self, atoms):
        with pytest.raises(TypeError, match="batch form"):
            EmpiricalDistribution("x", atoms)

    def test_an_array_of_atoms_has_a_leading_axis(self):
        with pytest.raises(ValueError, match="leading axis"):
            EmpiricalDistribution("x", jnp.asarray(1.0))

    @pytest.mark.parametrize(
        "atoms",
        [
            pytest.param(jnp.zeros((0, 2)), id="array"),
            pytest.param(OpaqueBatch("labels", [], "site"), id="batch"),
        ],
    )
    def test_a_law_has_at_least_one_atom(self, atoms):
        with pytest.raises(ValueError, match="at least one atom"):
            EmpiricalDistribution("x", atoms)

    def test_the_name_is_required(self):
        with pytest.raises(TypeError, match="non-empty name"):
            EmpiricalDistribution("", _VALUES)


# -- Weights --------------------------------------------------------------------


class TestWeights:
    def test_weights_default_to_uniform(self):
        law = EmpiricalDistribution("theta", _VALUES)
        assert law.num_atoms == 4
        assert np.allclose(law.weights, 0.25)

    def test_weights_are_normalized(self):
        assert np.allclose(_array_law().weights, _NORMALIZED)

    def test_a_weights_object_built_from_log_weights_is_adopted(self):
        log_weights = Weights(log_weights=jnp.log(_WEIGHTS))
        assert np.allclose(_array_law(log_weights).weights, _NORMALIZED)

    def test_weights_may_be_shaped_like_the_batch_axes(self):
        atoms = NumericArrayBatch(
            "draws",
            jnp.arange(6.0).reshape(2, 3),
            ("chain", "draw"),
            element_spec=NumericArraySpec(()),
        )
        # The weights of the atoms 0, ..., 5 in row-major order are 1, 2, 0, 0, 0, 3;
        # in column-major order they would be 1, 0, 2, 0, 0, 3, with mean 19/6.
        law = EmpiricalDistribution("x", atoms, jnp.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))
        assert np.allclose(law.weights, np.array([1.0, 2.0, 0.0, 0.0, 0.0, 3.0]) / 6.0)
        assert jnp.allclose(law._mean(), (0.0 * 1.0 + 1.0 * 2.0 + 5.0 * 3.0) / 6.0)

    @pytest.mark.parametrize(
        ("weights", "match"),
        [
            pytest.param(jnp.array([1.0, -1.0, 1.0, 1.0]), "non-negative", id="negative"),
            pytest.param(jnp.array([1.0, 1.0]), "does not match", id="count"),
            pytest.param(jnp.zeros(4), "positive", id="zero-sum"),
        ],
    )
    def test_invalid_weights_raise(self, weights, match):
        with pytest.raises(ValueError, match=match):
            _array_law(weights)


# -- Sampling ---------------------------------------------------------------------


class TestSampling:
    @pytest.mark.parametrize("sample_shape", [(), (7,), (2, 3)])
    def test_one_key_reproduces_the_draws(self, sample_shape):
        law, key = _array_law(), jax.random.PRNGKey(3)
        assert jnp.array_equal(law._sample(key, sample_shape), law._sample(key, sample_shape))

    def test_sampling_traces_under_jit(self):
        key = jax.random.PRNGKey(3)
        for law in (_array_law(), _record_law()):
            traced = jax.jit(lambda k, law=law: law._sample(k, (5,)))(key)
            assert jax.tree.all(jax.tree.map(jnp.array_equal, traced, law._sample(key, (5,))))

    def test_different_keys_give_different_draws(self):
        law = _array_law()
        first = law._sample(jax.random.PRNGKey(0), (50,))
        second = law._sample(jax.random.PRNGKey(1), (50,))
        assert not jnp.array_equal(first, second)

    def test_the_resampling_frequencies_are_the_weights(self):
        draws = np.asarray(_array_law()._sample(jax.random.PRNGKey(0), (20_000,)))
        frequencies = [np.mean(draws == value) for value in np.asarray(_VALUES)]
        assert np.allclose(frequencies, _NORMALIZED, atol=0.015)

    def test_an_atom_of_zero_weight_is_never_drawn(self):
        law = _array_law(jnp.array([1.0, 0.0, 1.0, 1.0]))
        assert not np.any(np.asarray(law._sample(jax.random.PRNGKey(0), (2_000,))) == 2.0)

    def test_array_draws_prepend_the_sample_axes(self):
        law = EmpiricalDistribution("x", jnp.arange(12.0).reshape(4, 3))
        draws = law._sample(jax.random.PRNGKey(0), (5, 2))
        assert draws.shape == (5, 2, 3)
        atoms = {tuple(row) for row in np.asarray(law.atoms.values)}
        assert {tuple(row) for row in np.asarray(draws).reshape(-1, 3)} <= atoms

    def test_a_record_draw_is_one_atom_whole(self):
        draw = _record_law()._sample(jax.random.PRNGKey(0))
        assert isinstance(draw, dict)
        assert _RECORD_SPEC.is_valid(draw)
        rows = {(*np.asarray(b), float(a)) for b, a in zip(_B, _A)}
        assert (*np.asarray(draw["b"]), float(draw["a"])) in rows

    @pytest.mark.parametrize("sample_shape", [(), (5,)])
    def test_a_record_draw_is_the_nested_mapping_of_its_raw_leaves(self, sample_shape):
        draws = EmpiricalDistribution("m", _mixed_atoms())._sample(
            jax.random.PRNGKey(0), sample_shape
        )
        assert isinstance(draws, dict) and isinstance(draws["g"], dict)
        assert list(draws) == ["label", "g"] and list(draws["g"]) == ["u", "v"]
        assert jnp.shape(draws["g"]["v"]) == (*sample_shape, 2)
        if sample_shape:
            assert draws["label"].dtype == object and draws["label"].shape == sample_shape
            rows = {(label, float(u)) for label, u in zip(_LABELS, _U)}
            assert {(label, float(u)) for label, u in zip(draws["label"], draws["g"]["u"])} <= rows
        else:
            assert draws["label"] in set(_LABELS)

    def test_record_draws_resample_whole_rows(self):
        draws = _record_law()._sample(jax.random.PRNGKey(0), (200,))
        assert draws["b"].shape == (200, 2)
        assert draws["a"].shape == (200,)
        rows = {(*np.asarray(b), float(a)) for b, a in zip(_B, _A)}
        drawn = {(*np.asarray(b), float(a)) for b, a in zip(draws["b"], draws["a"])}
        assert drawn <= rows

    def test_opaque_draws_are_the_stored_objects(self):
        law = _opaque_law()
        assert law._sample(jax.random.PRNGKey(0)) in {"north", "south", "east"}
        draws = law._sample(jax.random.PRNGKey(0), (6,))
        assert draws.shape == (6,)
        assert set(draws) <= {"north", "south", "east"}

    def test_the_sample_operation_returns_the_declared_kind(self):
        key = jax.random.PRNGKey(0)
        assert isinstance(sample(_array_law(), key=key, sample_shape=(4,)), NumericArrayBatch)
        assert isinstance(sample(_record_law(), key=key, sample_shape=(4,)), NumericRecordBatch)
        for law in (_array_law(), _record_law(), _opaque_law()):
            assert law.event_spec.spec.is_valid(sample(law, key=key))

    def test_the_sample_operation_draws_a_batch_of_mixed_records(self):
        law = EmpiricalDistribution("m", _mixed_atoms())
        draws = sample(law, key=jax.random.PRNGKey(0), sample_shape=(4,))
        assert type(draws) is RecordBatch
        assert (draws.batch_shape, draws.level_names) == ((4,), ("sample",))
        assert draws.element_spec == law.event_spec.spec


# -- Moments --------------------------------------------------------------------


class TestMoments:
    def test_the_mean_of_array_atoms(self):
        # 0.1 * 1 + 0.2 * 2 + 0.3 * 4 + 0.4 * 7
        assert jnp.allclose(_array_law()._mean(), 4.5)

    def test_the_variance_of_array_atoms(self):
        # 0.1 * 3.5**2 + 0.2 * 2.5**2 + 0.3 * 0.5**2 + 0.4 * 2.5**2, about the mean 4.5,
        # with the normalized weights and no correction for bias
        assert jnp.allclose(_array_law()._variance(), 5.05)

    def test_the_moments_of_vector_atoms_are_per_coordinate(self):
        atoms = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]])
        law = EmpiricalDistribution("x", jnp.asarray(atoms))
        mean = atoms.mean(axis=0)
        assert jnp.allclose(law._mean(), mean)
        assert jnp.allclose(law._variance(), ((atoms - mean) ** 2).mean(axis=0))

    def test_the_covariance_is_a_dense_operator_over_the_flat_coordinates(self):
        atoms = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]])
        law = EmpiricalDistribution("x", jnp.asarray(atoms))
        mean = atoms.mean(axis=0)
        expected = sum(np.outer(atom - mean, atom - mean) for atom in atoms) / 3
        cov = law._cov()
        assert isinstance(cov, DenseLinOp)
        assert jnp.allclose(cov.to_dense(), expected)

    def test_the_covariance_of_scalar_atoms_is_one_by_one(self):
        cov = _array_law()._cov()
        assert cov.shape == (1, 1)
        assert jnp.allclose(cov.to_dense(), 5.05)

    def test_the_moments_of_record_atoms_are_shaped_like_a_draw(self):
        law = _record_law()
        mean = law._mean()
        assert isinstance(mean, dict) and list(mean) == ["b", "a"]
        assert _RECORD_SPEC.is_valid(mean)
        assert jnp.allclose(mean["a"], 1.75)
        assert jnp.allclose(mean["b"], jnp.array([0.75, 1.0]))
        variance = law._variance()
        expected_a = _weighted_sum(_RECORD_WEIGHTS, [(a - 1.75) ** 2 for a in np.asarray(_A)])
        assert jnp.allclose(variance["a"], expected_a)

    def test_the_record_covariance_follows_the_canonical_field_order(self):
        coordinates = _flat_record_atoms()
        mean = _weighted_sum(_RECORD_WEIGHTS, coordinates)
        expected = _weighted_sum(
            _RECORD_WEIGHTS, [np.outer(row - mean, row - mean) for row in coordinates]
        )
        assert jnp.allclose(_record_law()._cov().to_dense(), expected, atol=1e-6)

    def test_the_quantile_is_the_generalized_inverse_of_the_weighted_cdf(self):
        # The CDF of the atoms 1, 2, 4, and 7 reaches 0.1, 0.3, 0.6, and 1 at them,
        # and the quantile at q is the smallest atom where it reaches q.
        law = _array_law()
        levels = jnp.array([0.05, 0.2, 0.45, 0.5, 0.61, 0.99])
        assert jnp.array_equal(law._quantile(levels), jnp.array([1.0, 2.0, 4.0, 4.0, 7.0, 7.0]))
        assert jnp.array_equal(law._quantile(jnp.array([0.0, 1.0])), jnp.array([1.0, 7.0]))

    def test_the_median_of_four_equally_weighted_atoms_is_the_second(self):
        law = EmpiricalDistribution("x", jnp.array([3.0, 1.0, 4.0, 2.0]))
        assert float(law._quantile(0.5)) == 2.0
        assert jnp.array_equal(law._quantile(jnp.array([0.25, 0.75])), jnp.array([1.0, 3.0]))

    def test_an_atom_of_zero_weight_has_no_effect_on_the_quantiles(self):
        levels = jnp.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        with_zero = EmpiricalDistribution("x", jnp.array([1.0, 2.9, 3.0]), jnp.array([0.5, 0, 0.5]))
        without = EmpiricalDistribution("x", jnp.array([1.0, 3.0]), jnp.array([0.5, 0.5]))
        assert jnp.array_equal(with_zero._quantile(levels), without._quantile(levels))

    def test_every_quantile_is_an_atom(self):
        atoms = jax.random.normal(jax.random.PRNGKey(0), (9, 2))
        weights = jax.random.uniform(jax.random.PRNGKey(1), (9,))
        quantiles = EmpiricalDistribution("x", atoms, weights)._quantile(jnp.linspace(0, 1, 11))
        for coordinate in range(2):
            assert set(np.asarray(quantiles[:, coordinate]).tolist()) <= set(
                np.asarray(atoms[:, coordinate]).tolist()
            )

    def test_the_quantile_of_array_atoms_puts_the_levels_before_the_event_shape(self):
        law = EmpiricalDistribution("x", jnp.arange(12.0).reshape(4, 3))
        assert law._quantile(0.5).shape == (3,)
        assert law._quantile(jnp.array([0.25, 0.75])).shape == (2, 3)
        assert law._quantile(jnp.full((2, 2), 0.5)).shape == (2, 2, 3)

    def test_the_quantile_of_record_atoms_is_the_mapping_of_each_leaf_s_quantiles(self):
        levels = jnp.array([0.25, 0.5])
        quantiles = _record_law()._quantile(levels)
        assert isinstance(quantiles, dict) and list(quantiles) == ["b", "a"]
        assert quantiles["b"].shape == (2, 2) and quantiles["a"].shape == (2,)
        for coordinate in range(2):
            law = EmpiricalDistribution("c", _B[:, coordinate], _RECORD_WEIGHTS)
            assert jnp.allclose(quantiles["b"][:, coordinate], law._quantile(levels))
        a_law = EmpiricalDistribution("c", _A, _RECORD_WEIGHTS)
        assert jnp.allclose(quantiles["a"], a_law._quantile(levels))

    def test_one_level_of_record_atoms_is_shaped_like_a_draw(self):
        quantiles = _record_law()._quantile(0.5)
        assert _RECORD_SPEC.is_valid(quantiles)
        assert (jnp.shape(quantiles["b"]), jnp.shape(quantiles["a"])) == ((2,), ())

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(_opaque_law, id="opaque"),
            pytest.param(lambda: EmpiricalDistribution("m", _mixed_atoms()), id="mixed-record"),
        ],
    )
    def test_a_non_numeric_event_claims_no_moment(self, make):
        law = make()
        assert not any(isinstance(law, moment) for moment in _MOMENTS)
        assert all(isinstance(law, capability) for capability in _ALWAYS)
        assert type(law) is EmpiricalDistribution


# -- Expectation ------------------------------------------------------------------


class TestExpectation:
    def test_the_expectation_is_the_weighted_sum_over_the_atoms(self):
        expected = _weighted_sum(_NORMALIZED, np.asarray(_VALUES) ** 2)
        assert jnp.allclose(_array_law()._expectation(lambda x: x**2), expected)

    def test_a_record_integrand_reads_each_atom_as_its_raw_mapping(self):
        expected = _weighted_sum(
            _RECORD_WEIGHTS, [float(a) * float(np.sum(b)) for b, a in zip(_B, _A)]
        )
        received: list[type] = []

        def integrand(atom: dict) -> jax.Array:
            received.append(type(atom))
            return atom["a"] * jnp.sum(atom["b"])

        assert jnp.allclose(_record_law()._expectation(integrand), expected)
        assert received and all(kind is dict for kind in received)

    def test_the_identity_integrates_to_the_mean(self):
        law = _record_law()
        integrated = law._expectation(lambda record: record)
        mean = law._mean()
        assert jnp.allclose(integrated["a"], mean["a"])
        assert jnp.allclose(integrated["b"], mean["b"])

    def test_a_pytree_integrand_integrates_leaf_by_leaf(self):
        integrated = _array_law()._expectation(lambda x: {"first": x, "second": x**2})
        assert jnp.allclose(integrated["first"], _weighted_sum(_NORMALIZED, _VALUES))
        assert jnp.allclose(integrated["second"], _weighted_sum(_NORMALIZED, _VALUES**2))

    def test_opaque_atoms_are_integrated_one_by_one(self):
        integrated = _opaque_law()._expectation(lambda label: jnp.asarray(float(len(label))))
        assert jnp.allclose(integrated, 0.2 * 5 + 0.3 * 5 + 0.5 * 4)

    def test_a_mixed_record_is_integrated_one_atom_at_a_time(self):
        law = EmpiricalDistribution("m", _mixed_atoms())
        integrated = law._expectation(lambda atom: len(atom["label"]) * atom["g"]["u"])
        assert jnp.allclose(integrated, (5 * 1.0 + 5 * 2.0 + 4 * 3.0) / 3)

    def test_callable_atoms_are_integrated_at_a_point(self):
        law = EmpiricalDistribution("f", FunctionBatch("fs", [jnp.sin, jnp.cos], "f"))
        assert jnp.allclose(law._expectation(lambda f: f(0.0)), 0.5)

    def test_the_expectation_operation_takes_the_exact_method(self):
        law, integrand = _array_law(), (lambda x: x**2)
        assert expectation_method_registry.check(law, integrand).method_name == "exact"
        result = expectation(law, integrand)
        assert jnp.allclose(jnp.asarray(result), _weighted_sum(_NORMALIZED, _VALUES**2))


# -- Marginals --------------------------------------------------------------------


class TestMarginals:
    def test_the_marginal_of_a_field_is_the_empirical_law_of_its_column(self):
        law = _record_law()
        marginal = law._marginal("b")
        assert isinstance(marginal, EmpiricalDistribution)
        assert not isinstance(marginal, FieldView)
        assert marginal.name == "b"
        assert marginal.event_spec == OutputSpec(b=_RECORD_SPEC["b"])
        assert jnp.array_equal(marginal.atoms.values, _B)
        assert np.allclose(marginal.weights, _RECORD_WEIGHTS)
        assert jnp.allclose(marginal._mean(), law._mean()["b"])

    def test_the_marginal_of_a_group_is_its_subtree_whole(self):
        law = EmpiricalDistribution("m", _mixed_atoms())
        marginal = law._marginal("g")
        assert marginal.event_spec == OutputSpec(g=_MIXED_SPEC.at_path("g"))
        assert isinstance(marginal, SupportsMean)
        assert jnp.allclose(marginal._mean()["u"], jnp.mean(_U))
        draw = marginal._sample(jax.random.PRNGKey(0))
        assert isinstance(draw, dict)
        assert list(draw.keys()) == ["u", "v"]

    def test_the_marginal_of_a_nested_leaf_takes_its_final_segment(self):
        marginal = EmpiricalDistribution("m", _mixed_atoms())._marginal("g/v")
        assert marginal.name == "g/v"
        assert marginal.event_spec == OutputSpec(v=NumericArraySpec((2,)))
        assert jnp.allclose(marginal._mean(), jnp.mean(_V, axis=0))

    def test_the_marginal_of_an_opaque_field_samples_its_labels(self):
        marginal = EmpiricalDistribution("m", _mixed_atoms())._marginal("label")
        assert marginal.event_spec == OutputSpec(label=OpaqueSpec())
        assert marginal._sample(jax.random.PRNGKey(0)) in set(_LABELS)

    def test_a_selection_of_paths_is_an_exposed_record_that_keeps_the_rows(self):
        law = EmpiricalDistribution("m", _mixed_atoms())
        marginal = law._marginal(("label", "g/v"))
        assert marginal.name == "label, g/v"
        assert marginal.event_spec == OutputSpec(RecordSpec(label=None, v=(2,)))
        rows = {(label, *np.asarray(v)) for label, v in zip(_LABELS, _V)}
        for key in jax.random.split(jax.random.PRNGKey(0), 10):
            draw = marginal._sample(key)
            assert (draw["label"], *np.asarray(draw["v"])) in rows

    def test_a_selection_whose_paths_share_a_final_segment_raises(self):
        spec = RecordSpec(p=RecordSpec(x=()), q=RecordSpec(x=()))
        atoms = NumericRecordBatch("rows", {"p/x": _A, "q/x": _U}, "row", element_spec=spec)
        law = EmpiricalDistribution("m", atoms)
        with pytest.raises(ValueError, match="final segments"):
            law._marginal(("p/x", "q/x"))
        assert _capability_guard(law, "_marginal", ("p/x", "q/x")).feasible is False

    def test_a_path_that_is_not_an_event_path_raises_and_is_declined(self):
        law = _record_law()
        with pytest.raises(KeyError, match="not an event path"):
            law._marginal("c")
        report = _capability_guard(law, "_marginal", "c")
        assert report.feasible is False
        assert "not an event path" in report.description

    def test_the_marginal_is_exact_at_every_event_path(self):
        law = EmpiricalDistribution("m", _mixed_atoms())
        for path in ("label", "g", "g/u", ("label", "g/u")):
            assert _capability_guard(law, "_marginal", path) == Feasibility(True)

    def test_the_paths_of_a_whole_record_term_start_with_its_component(self):
        law = EmpiricalDistribution("post", _record_atoms(), event_spec=OutputSpec(params=None))
        assert law._marginal("params").event_spec == law.event_spec
        assert law._marginal("params/a").event_spec == OutputSpec(a=NumericArraySpec(()))
        with pytest.raises(KeyError):
            law._marginal("a")

    def test_the_marginal_keeps_the_atoms_levels(self):
        atoms = NumericRecordBatch(
            "draws",
            {"b": jnp.zeros((2, 3, 2)), "a": jnp.zeros((2, 3))},
            ("chain", "draw"),
            element_spec=_RECORD_SPEC,
        )
        law = EmpiricalDistribution("post", atoms)
        assert law._marginal("a").atoms.level_names == ("chain", "draw")
        assert law._marginal(("a", "b")).atoms.level_names == ("chain", "draw")


# -- Densities and capability classes -----------------------------------------------


class TestCapabilities:
    @pytest.mark.parametrize("make", [_array_law, _record_law, _opaque_law])
    def test_the_law_claims_no_density(self, make):
        law = make()
        assert not isinstance(law, SupportsLogProb)
        assert not isinstance(law, SupportsUnnormalizedLogProb)
        with pytest.raises(AttributeError):
            _capability_guard(law, "_log_prob")

    def test_a_numeric_instance_claims_the_moments_through_its_class(self):
        law = _array_law()
        assert type(law) is _capability_subclass(EmpiricalDistribution, _MOMENTS)
        assert all(isinstance(law, capability) for capability in _MOMENTS + _ALWAYS)
        assert type(law).__name__ == "EmpiricalDistribution"

    def test_a_numeric_instance_round_trips_through_pickle(self):
        law = _record_law().with_name("renamed")
        restored = pickle.loads(pickle.dumps(law))
        assert type(restored) is type(law)
        assert (restored.name, restored.spec) == (law.name, law.spec)
        assert jnp.allclose(restored._mean()["b"], law._mean()["b"])
        assert np.allclose(restored.weights, law.weights)
