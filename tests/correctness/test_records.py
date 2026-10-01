"""Nested and multi-field events through every layer: empirical laws, joints, views, and renames.

The record schemas and their atoms come from :mod:`tests.correctness._records`,
so each contract is checked on a flat record, a record of two groups, and a
record three levels deep, with atoms on one, two, and three levels. Every
expected value is computed by hand from the atoms in NumPy: a weighted average,
a generalized inverse CDF, a covariance in the canonical leaf order, or a
correlation bounded by four standard errors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    Normal,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    OutputSpec,
    RecordSpec,
    function,
    workflow_run,
)
from probpipe.core.constraints import real
from probpipe.distributions._factored import _raw_record
from tests._ops import (
    EmpiricalDistribution,
    FieldView,
    cov,
    log_prob,
    marginal,
    mean,
    quantile,
    sample,
    variance,
)
from tests.correctness import _records
from tests.correctness._laws import Population, RecordObservationKernel
from tests.correctness._records import LAYOUTS, SCHEMAS, columns, group_paths, leaf_paths

#: The four-standard-error band of every Monte Carlo comparison.
Z = 4.0

#: The number of draws behind every Monte Carlo comparison.
DRAWS = 4000

_CASES = [
    pytest.param(schema, layout, id=f"{schema}-{layout}") for schema, layout in _records.cases()
]


def _law(schema: str, layout: str, *, weighted: bool = False) -> EmpiricalDistribution:
    atoms = _records.atoms(SCHEMAS[schema], LAYOUTS[layout])
    weights = None
    if weighted:
        weights = jax.random.uniform(jax.random.PRNGKey(1), atoms.batch_shape, minval=0.2, maxval=2)
    return EmpiricalDistribution("law", atoms, weights)


def _at(raw, path):
    for segment in path.split("/"):
        raw = raw[segment]
    return np.asarray(raw, dtype=np.float64)


def _weights(law) -> np.ndarray:
    return np.asarray(law.weights, dtype=np.float64)


def _inverse_cdf(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """``inf{x : F(x) >= q}`` of each coordinate's weighted CDF, by sorting in NumPy."""
    flat = values.reshape(values.shape[0], -1)
    result = np.empty(flat.shape[1])
    for j in range(flat.shape[1]):
        order = np.argsort(flat[:, j], kind="stable")
        cumulative = np.cumsum(weights[order])
        index = np.searchsorted(cumulative, q * cumulative[-1], side="left")
        result[j] = flat[order[min(index, len(order) - 1)], j]
    return result.reshape(values.shape[1:])


# ---------------------------------------------------------------------------
# Empirical laws over record batches with several levels
# ---------------------------------------------------------------------------


class TestEmpiricalRecords:
    @pytest.mark.parametrize(("schema", "layout"), _CASES)
    def test_the_law_exposes_the_schema_and_counts_every_atom(self, schema, layout):
        law = _law(schema, layout)
        assert law.event_spec == OutputSpec(SCHEMAS[schema])
        assert law.num_atoms == int(np.prod(LAYOUTS[layout][1]))
        assert law.atoms.level_names == LAYOUTS[layout][0]
        np.testing.assert_allclose(_weights(law), 1.0 / law.num_atoms)

    @pytest.mark.parametrize(("schema", "layout"), _CASES)
    def test_the_mean_is_a_record_of_the_schema_with_each_leafs_average(self, schema, layout):
        law = _law(schema, layout)
        result = mean(law)
        assert isinstance(result, NumericRecord)
        assert result.spec == SCHEMAS[schema]
        raw = mean.with_options(raw=True)(law)
        for path, values in columns(law.atoms, SCHEMAS[schema]).items():
            np.testing.assert_allclose(_at(raw, path), values.mean(axis=0), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(("schema", "layout"), _CASES)
    def test_weighted_moments_follow_the_weights_in_row_major_order(self, schema, layout):
        """Weights shaped like the batch axes weigh the atoms in row-major order."""
        law = _law(schema, layout, weighted=True)
        weights = _weights(law)
        means = mean.with_options(raw=True)(law)
        variances = variance.with_options(raw=True)(law)
        for path, values in columns(law.atoms, SCHEMAS[schema]).items():
            expected_mean = np.tensordot(weights, values, axes=1)
            expected_variance = np.tensordot(weights, (values - expected_mean) ** 2, axes=1)
            np.testing.assert_allclose(_at(means, path), expected_mean, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(
                _at(variances, path), expected_variance, rtol=1e-4, atol=1e-6
            )

    @pytest.mark.parametrize("weighted", [False, True], ids=["uniform", "weighted"])
    @pytest.mark.parametrize(("schema", "layout"), _CASES)
    def test_the_quantiles_are_the_generalized_inverse_cdf_per_leaf(self, schema, layout, weighted):
        """Each leaf's quantiles are ``inf{x : F(x) >= q}`` with the level axes leading."""
        law = _law(schema, layout, weighted=weighted)
        levels = np.array([0.1, 0.5, 0.9])
        raw = quantile.with_options(raw=True)(law, jnp.asarray(levels))
        weights = _weights(law)
        for path, values in columns(law.atoms, SCHEMAS[schema]).items():
            node = _at(raw, path)
            assert node.shape == (3, *values.shape[1:])
            for index, q in enumerate(levels):
                np.testing.assert_allclose(node[index], _inverse_cdf(values, weights, q))

    def test_one_level_gives_a_record_and_several_give_a_batch_on_a_quantile_level(self):
        law = _law("two-groups", "two-levels")
        one = quantile(law, 0.5)
        assert isinstance(one, NumericRecord) and one.spec == SCHEMAS["two-groups"]
        several = quantile(law, jnp.array([0.1, 0.9]))
        assert isinstance(several, NumericRecordBatch)
        assert (several.batch_shape, several.level_names) == ((2,), ("quantile",))
        assert several.element_spec == SCHEMAS["two-groups"]

    @pytest.mark.parametrize(("schema", "layout"), _CASES)
    def test_the_covariance_follows_the_canonical_leaf_order(self, schema, layout):
        law = _law(schema, layout)
        flat = np.concatenate(
            [
                values.reshape(law.num_atoms, -1)
                for values in columns(law.atoms, SCHEMAS[schema]).values()
            ],
            axis=1,
        )
        result = cov.with_options(raw=True)(law)
        dense = np.asarray(result.to_dense() if hasattr(result, "to_dense") else result)
        np.testing.assert_allclose(
            dense, np.cov(flat, rowvar=False, bias=True), rtol=1e-4, atol=1e-5
        )

    @pytest.mark.parametrize(("schema", "layout"), _CASES)
    def test_every_draw_is_one_whole_atom(self, schema, layout):
        """A draw takes every leaf from one atom, so the leaves are co-drawn."""
        law = _law(schema, layout)
        with workflow_run(seed=2):
            draws = sample(law, sample_shape=(32,))
        assert (draws.batch_shape, draws.level_names) == ((32,), ("sample",))
        assert draws.element_spec == SCHEMAS[schema]

        def rows(batch):
            return np.concatenate(
                [v.reshape(v.shape[0], -1) for v in columns(batch, SCHEMAS[schema]).values()], 1
            )

        atoms, drawn = rows(law.atoms), rows(draws)
        for row in drawn:
            assert np.any(np.all(np.isclose(atoms, row), axis=1))

    @pytest.mark.parametrize(("schema", "layout"), _CASES)
    def test_the_marginal_of_a_group_keeps_the_levels_and_the_weights(self, schema, layout):
        law = _law(schema, layout, weighted=True)
        parent_mean = mean.with_options(raw=True)(law)
        for path in group_paths(SCHEMAS[schema]):
            group = marginal(law, path)
            node = _records.node_of(SCHEMAS[schema], path)
            assert isinstance(group, EmpiricalDistribution)
            assert group.event_spec == OutputSpec(**{path.rsplit("/", 1)[-1]: node})
            assert group.atoms.level_names == law.atoms.level_names
            np.testing.assert_allclose(_weights(group), _weights(law))
            group_mean = mean.with_options(raw=True)(group)
            for leaf in leaf_paths(node):
                np.testing.assert_allclose(
                    _at(group_mean, leaf), _at(parent_mean, f"{path}/{leaf}"), rtol=1e-6
                )

    def test_the_marginal_of_a_selection_exposes_a_record_of_its_final_segments(self):
        law = _law("deep", "two-levels")
        selection = marginal(law, ("model/theta/mu", "y"))
        assert selection.event_spec == OutputSpec(
            RecordSpec(mu=SCHEMAS["deep"].at_path("model", "theta", "mu"), y=SCHEMAS["deep"]["y"])
        )
        parent_mean = mean.with_options(raw=True)(law)
        selected_mean = mean.with_options(raw=True)(selection)
        np.testing.assert_allclose(_at(selected_mean, "mu"), _at(parent_mean, "model/theta/mu"))
        np.testing.assert_allclose(_at(selected_mean, "y"), _at(parent_mean, "y"))


# ---------------------------------------------------------------------------
# Factored joints of record-valued factors
# ---------------------------------------------------------------------------

#: The population record the record-valued factor draws, with a positive scale.
_POPULATION = RecordSpec(
    mu=NumericArraySpec((), jnp.float32), tau=NumericArraySpec((), jnp.float32)
)


def _population_atoms() -> EmpiricalDistribution:
    """An empirical law over the record ``population = {mu, tau}``, with ``tau`` positive."""
    k_mu, k_tau = jax.random.split(jax.random.PRNGKey(5))
    mu = (1.0 + jax.random.normal(k_mu, (200,))).astype(jnp.float32)
    tau = jnp.exp(0.3 * jax.random.normal(k_tau, (200,))).astype(jnp.float32)
    atoms = NumericRecordBatch(
        "atoms",
        {"population": {"mu": mu, "tau": tau}},
        "atom",
        element_spec=RecordSpec(population=_POPULATION),
    )
    return EmpiricalDistribution("hyper", atoms)


def _theta_given_population():
    """The kernel ``theta | population ~ N(mu, tau)``, whose given slot is a record."""
    return RecordObservationKernel(
        "theta",
        {"population": _POPULATION},
        NumericArraySpec((), jnp.float32, real),
        lambda population: tfd.Normal(population["mu"], population["tau"]),
    )


class TestRecordValuedFactors:
    def test_the_joint_exposes_each_factors_components_with_their_packaging(self):
        joint = _population_atoms() * Normal("x", 0.0, 1.0)
        spec = joint.event_spec.spec
        assert list(spec.children) == ["population", "x"]
        assert spec.children["population"] == _POPULATION

    def test_a_joint_draw_takes_the_record_component_from_one_atom(self):
        hyper = _population_atoms()
        joint = hyper * Normal("x", 0.0, 1.0)
        with workflow_run(seed=3):
            draws = _raw_record(sample(joint, sample_shape=(64,)))
        atoms = _raw_record(hyper.atoms)["population"]
        pairs = np.stack([np.asarray(atoms["mu"]), np.asarray(atoms["tau"])], axis=1)
        drawn = np.stack(
            [np.asarray(draws["population"]["mu"]), np.asarray(draws["population"]["tau"])], 1
        )
        for row in drawn:
            assert np.any(np.all(np.isclose(pairs, row), axis=1))

    def test_the_edge_free_joint_has_each_factors_mean(self):
        hyper = _population_atoms()
        means = mean.with_options(raw=True)(hyper * Normal("x", 0.5, 1.0))
        hyper_mean = mean.with_options(raw=True)(hyper)
        np.testing.assert_allclose(_at(means, "population/mu"), _at(hyper_mean, "population/mu"))
        np.testing.assert_allclose(_at(means, "population/tau"), _at(hyper_mean, "population/tau"))
        assert float(means["x"]) == pytest.approx(0.5)

    def test_the_marginal_of_the_record_component_is_its_factors_law(self):
        hyper = _population_atoms()
        group = marginal(hyper * Normal("x", 0.0, 1.0), "population")
        assert group.event_spec == OutputSpec(population=_POPULATION)
        np.testing.assert_allclose(
            _at(mean.with_options(raw=True)(group), "mu"),
            _at(mean.with_options(raw=True)(hyper), "population/mu"),
        )

    def test_the_marginal_at_a_nested_path_reduces_its_factor(self):
        hyper = _population_atoms()
        leaf = marginal(hyper * Normal("x", 0.0, 1.0), "population/tau")
        assert leaf.event_spec == OutputSpec(tau=_POPULATION["tau"])
        np.testing.assert_allclose(
            np.asarray(mean.with_options(raw=True)(leaf)),
            _at(mean.with_options(raw=True)(hyper), "population/tau"),
        )

    def test_a_kernel_given_a_record_draws_with_the_law_of_total_variance(self):
        """``theta | population ~ N(mu, tau)`` gives ``E[theta] = E[mu]`` and ``Var = E[tau²] + Var mu``."""
        hyper = _population_atoms()
        joint = _theta_given_population() * hyper
        with workflow_run(seed=4):
            draws = _raw_record(sample(joint, sample_shape=(DRAWS,)))
        atoms = _raw_record(hyper.atoms)["population"]
        mu, tau = np.asarray(atoms["mu"], np.float64), np.asarray(atoms["tau"], np.float64)
        expected_mean, expected_variance = mu.mean(), (tau**2).mean() + mu.var()
        theta = np.asarray(draws["theta"], np.float64)
        assert abs(theta.mean() - expected_mean) <= Z * np.sqrt(expected_variance / DRAWS)
        squares = (theta - theta.mean()) ** 2
        assert abs(squares.mean() - expected_variance) <= Z * squares.std() / np.sqrt(DRAWS)

    def test_the_joint_density_scores_each_factor_at_its_record(self):
        """The density of ``population * x`` is the population's record density plus the normal's."""
        joint = Population() * Normal("x", 0.0, 1.0)
        value = {"population": {"mu": 0.3, "tau": 2.0}, "x": 0.5}
        expected = (
            scipy.stats.norm.logpdf(0.3, 0.0, 5.0)
            + scipy.stats.halfcauchy.logpdf(2.0, scale=5.0)
            + scipy.stats.norm.logpdf(0.5)
        )
        assert float(log_prob.with_options(raw=True)(joint, value)) == pytest.approx(
            expected, rel=1e-5
        )


# ---------------------------------------------------------------------------
# Field views at nested paths, and selections
# ---------------------------------------------------------------------------


def _correlated() -> EmpiricalDistribution:
    """An empirical law over the two-group schema whose coordinates have correlation 0.8."""
    return EmpiricalDistribution(
        "law", _records.atoms(SCHEMAS["two-groups"], (("atom",), (2000,)), seed=6)
    )


def _atom_correlation(law, first: str, second: str) -> float:
    values = columns(law.atoms, SCHEMAS["two-groups"])
    a, b = (
        values[first].reshape(law.num_atoms, -1)[:, 0],
        values[second].reshape(law.num_atoms, -1)[:, 0],
    )
    return float(np.corrcoef(a, b)[0, 1])


def _assert_correlation(a, b, expected, draws=DRAWS):
    """The sample correlation of *a* and *b* within four standard errors of *expected*, on Fisher's scale."""
    observed = np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]
    assert abs(np.arctanh(observed) - np.arctanh(expected)) <= Z / np.sqrt(draws - 3), (
        observed,
        expected,
    )


class TestNestedViews:
    def test_sibling_views_drawn_with_one_key_project_one_parent_draw(self):
        law = _correlated()
        key = jax.random.PRNGKey(7)
        parent = law._sample(key, (16,))
        mu = FieldView(law, "population/mu")._sample(key, (16,))
        theta = FieldView(law, "groups/theta")._sample(key, (16,))
        np.testing.assert_array_equal(np.asarray(mu), np.asarray(parent["population"]["mu"]))
        np.testing.assert_array_equal(np.asarray(theta), np.asarray(parent["groups"]["theta"]))

    def test_the_correlation_of_co_sampled_siblings_is_the_parents(self):
        law = _correlated()
        key = jax.random.PRNGKey(8)
        mu = FieldView(law, "population/mu")._sample(key, (DRAWS,))
        theta = FieldView(law, "groups/theta")._sample(key, (DRAWS,))
        expected = _atom_correlation(law, "population/mu", "groups/theta")
        _assert_correlation(mu, np.asarray(theta)[:, 0], expected)

    def test_siblings_drawn_with_independent_keys_are_uncorrelated(self):
        law = _correlated()
        mu = FieldView(law, "population/mu")._sample(jax.random.PRNGKey(9), (DRAWS,))
        theta = FieldView(law, "groups/theta")._sample(jax.random.PRNGKey(10), (DRAWS,))
        _assert_correlation(mu, np.asarray(theta)[:, 0], 0.0)

    @pytest.mark.pending(
        reason=(
            "bug: the sampling lift groups only the earlier record views by parent, so sibling "
            "FieldViews of one law are drawn independently"
        ),
        raises=AssertionError,
    )
    def test_a_function_of_sibling_views_keeps_their_correlation(self):
        """The lift of ``a * b`` over two sibling views co-samples them, so ``E[ab]`` keeps the covariance.

        Over the parent's atoms ``E[ab]`` differs from ``E[a] E[b]`` by the
        covariance, about 1.0, which is many standard errors at 4000 draws.
        """
        law = _correlated()

        @function
        def product(a: float, b: float) -> float:
            return a * b

        values = columns(law.atoms, SCHEMAS["two-groups"])
        products = values["population/mu"] * values["population/tau"]
        with workflow_run(seed=11):
            law_of_product = product.with_options(n_broadcast_samples=DRAWS)(
                FieldView(law, "population/mu"), FieldView(law, "population/tau")
            )
        estimate = float(jax.tree.leaves(mean.with_options(raw=True)(law_of_product))[0])
        assert abs(estimate - products.mean()) <= Z * products.std() / np.sqrt(DRAWS)

    def test_views_of_a_dependent_joint_keep_its_dependence(self):
        """Views of ``theta`` and ``population/mu`` in ``p(theta | population) p(population)`` co-sample.

        ``E[theta | population] = mu``, so ``Cov(theta, mu) = Var mu`` and the
        correlation is ``sqrt(Var mu / Var theta)``, with ``Var theta = E[tau²]
        + Var mu`` over the population's atoms.
        """
        hyper = _population_atoms()
        joint = _theta_given_population() * hyper
        key = jax.random.PRNGKey(16)
        theta = FieldView(joint, "theta")._sample(key, (DRAWS,))
        mu = FieldView(joint, "population/mu")._sample(key, (DRAWS,))
        atoms = _raw_record(hyper.atoms)["population"]
        mu_atoms, tau_atoms = (np.asarray(atoms[k], np.float64) for k in ("mu", "tau"))
        expected = np.sqrt(mu_atoms.var() / ((tau_atoms**2).mean() + mu_atoms.var()))
        _assert_correlation(theta, mu, expected)

    @pytest.mark.parametrize("path", ["population/mu", "groups/theta", "population"])
    def test_a_view_reads_the_parents_moments_and_quantiles_at_its_node(self, path):
        law = _correlated()
        view = FieldView(law, path)
        component = path.rsplit("/", 1)[-1]
        for operation in (mean, variance):
            parent_raw = operation.with_options(raw=True)(law)
            view_raw = operation.with_options(raw=True)(view)
            node = parent_raw
            for segment in path.split("/"):
                node = node[segment]
            np.testing.assert_allclose(
                np.asarray(jax.tree.leaves(view_raw)), np.asarray(jax.tree.leaves(node))
            )
        levels = jnp.array([0.25, 0.75])
        parent_q = quantile.with_options(raw=True)(law, levels)
        view_q = quantile.with_options(raw=True)(view, levels)
        node = parent_q
        for segment in path.split("/"):
            node = node[segment]
        assert list(view.event_spec.components) == [component]
        np.testing.assert_allclose(
            np.asarray(jax.tree.leaves(view_q)), np.asarray(jax.tree.leaves(node))
        )

    def test_a_group_view_draws_the_nested_mapping_of_its_node(self):
        law = _correlated()
        with workflow_run(seed=12):
            draws = sample(FieldView(law, "population"), sample_shape=(8,))
        assert isinstance(draws, NumericRecordBatch)
        assert draws.element_spec == _records.node_of(SCHEMAS["two-groups"], "population")
        assert set(_raw_record(draws)) == {"mu", "tau"}


class TestSelections:
    def test_a_selection_exposes_a_record_of_its_final_segments_in_order(self):
        view = FieldView(_correlated(), ("groups/theta", "population/mu"))
        assert view.event_spec == OutputSpec(
            RecordSpec(
                theta=SCHEMAS["two-groups"].at_path("groups", "theta"),
                mu=SCHEMAS["two-groups"].at_path("population", "mu"),
            )
        )

    def test_a_selection_co_samples_its_nodes(self):
        law = _correlated()
        view = FieldView(law, ("population/mu", "groups/theta"))
        with workflow_run(seed=13):
            draws = _raw_record(sample(view, sample_shape=(DRAWS,)))
        expected = _atom_correlation(law, "population/mu", "groups/theta")
        _assert_correlation(draws["mu"], np.asarray(draws["theta"])[:, 0], expected)

    def test_the_moments_of_a_selection_are_the_parents_at_its_nodes(self):
        law = _correlated()
        view = FieldView(law, ("population/mu", "groups/theta"))
        parent = mean.with_options(raw=True)(law)
        selected = mean.with_options(raw=True)(view)
        np.testing.assert_allclose(_at(selected, "mu"), _at(parent, "population/mu"))
        np.testing.assert_allclose(_at(selected, "theta"), _at(parent, "groups/theta"))

    def test_the_covariance_of_a_selection_is_the_parents_block_in_selection_order(self):
        law = _correlated()
        view = FieldView(law, ("groups/theta", "population/mu"))
        parent = cov.with_options(raw=True)(law)
        parent = np.asarray(parent.to_dense() if hasattr(parent, "to_dense") else parent)
        # The parent's flat layout is (mu, tau, theta[0:4]); the selection's is (theta, mu).
        order = [2, 3, 4, 5, 0]
        selected = cov.with_options(raw=True)(view)
        selected = np.asarray(selected.to_dense() if hasattr(selected, "to_dense") else selected)
        np.testing.assert_allclose(selected, parent[np.ix_(order, order)], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Renames that move nodes into and out of groups
# ---------------------------------------------------------------------------


def _two_groups() -> EmpiricalDistribution:
    return _law("two-groups", "two-levels", weighted=True)


class TestRenames:
    @pytest.mark.parametrize(
        ("renames", "moved"),
        [
            pytest.param({"population/mu": "mu"}, {"mu": "population/mu"}, id="out-of-a-group"),
            pytest.param(
                {"groups/theta": "population/theta"},
                {"population/theta": "groups/theta"},
                id="into-a-group",
            ),
            pytest.param(
                {"population": "hyper"},
                {"hyper/mu": "population/mu", "hyper/tau": "population/tau"},
                id="a-whole-group",
            ),
        ],
    )
    def test_a_rename_moves_the_node_and_translates_its_values(self, renames, moved):
        """The renamed law's draws, moments, and quantiles at a new path are the original's at the old."""
        law = _two_groups()
        renamed = law.with_path_names(renames)
        new_paths = set(leaf_paths(renamed.event_spec.spec))
        assert set(moved) <= new_paths
        assert not set(renames) & new_paths
        key = jax.random.PRNGKey(14)
        original_draw, renamed_draw = law._sample(key, (8,)), renamed._sample(key, (8,))
        levels = jnp.array([0.2, 0.8])
        for new, old in moved.items():
            np.testing.assert_array_equal(
                _at(_raw_record(renamed_draw), new), _at(_raw_record(original_draw), old)
            )
            for operation in (mean, variance):
                np.testing.assert_allclose(
                    _at(operation.with_options(raw=True)(renamed), new),
                    _at(operation.with_options(raw=True)(law), old),
                )
            np.testing.assert_allclose(
                _at(quantile.with_options(raw=True)(renamed, levels), new),
                _at(quantile.with_options(raw=True)(law, levels), old),
            )

    def test_the_marginal_at_a_new_path_is_the_original_marginal_at_the_old(self):
        law = _two_groups()
        renamed = law.with_path_names({"population/mu": "mu"})
        np.testing.assert_allclose(
            np.asarray(mean.with_options(raw=True)(marginal(renamed, "mu"))),
            np.asarray(mean.with_options(raw=True)(marginal(law, "population/mu"))),
        )

    def test_the_inverse_rename_restores_every_path_and_its_values(self):
        """Moving a leaf out of its group and back restores its path and its values.

        The moved leaf returns at the end of its group, so the comparison is by
        path rather than by the order of the group's fields.
        """
        law = _two_groups()
        round_trip = law.with_path_names({"population/mu": "mu"}).with_path_names(
            {"mu": "population/mu"}
        )
        spec = law.event_spec.spec
        assert set(leaf_paths(round_trip.event_spec.spec)) == set(leaf_paths(spec))
        for path in leaf_paths(spec):
            assert round_trip.event_spec.spec.at_path(*path.split("/")) == spec.at_path(
                *path.split("/")
            )
            np.testing.assert_allclose(
                _at(mean.with_options(raw=True)(round_trip), path),
                _at(mean.with_options(raw=True)(law), path),
            )

    def test_a_factored_joint_moves_a_component_into_a_group(self):
        joint = Normal("a", 0.0, 1.0) * Normal("b", 2.0, 1.0)
        renamed = joint.with_path_names({"a": "g/a"})
        assert set(leaf_paths(renamed.event_spec.spec)) == {"g/a", "b"}
        assert float(_at(mean.with_options(raw=True)(renamed), "g/a")) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Raw forms and per-leaf quantiles of record results
# ---------------------------------------------------------------------------


def _is_nested_mapping_of_arrays(value) -> bool:
    if isinstance(value, dict):
        return all(_is_nested_mapping_of_arrays(child) for child in value.values())
    return isinstance(value, jax.Array | np.ndarray)


class TestRawForms:
    @pytest.mark.parametrize("operation", [mean, variance], ids=["mean", "variance"])
    def test_a_record_valued_raw_result_is_a_nested_mapping_of_raw_leaves(self, operation):
        raw = operation.with_options(raw=True)(_law("deep", "three-levels"))
        assert _is_nested_mapping_of_arrays(raw)
        assert set(raw) == {"model", "y"} and set(raw["model"]) == {"theta", "noise"}

    def test_a_raw_batch_of_draws_is_the_nested_mapping_of_columns_with_the_batch_axes_leading(
        self,
    ):
        with workflow_run(seed=15):
            raw = sample.with_options(raw=True)(_law("deep", "two-levels"), sample_shape=(5, 2))
        assert _is_nested_mapping_of_arrays(raw)
        assert np.shape(raw["model"]["theta"]["sd"]) == (5, 2, 2)
        assert np.shape(raw["y"]) == (5, 2, 3)

    def test_per_leaf_quantiles_put_the_level_axes_first(self):
        levels = jnp.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]])
        raw = quantile.with_options(raw=True)(_law("deep", "one-level"), levels)
        assert np.shape(raw["model"]["theta"]["mu"]) == (2, 3)
        assert np.shape(raw["model"]["theta"]["sd"]) == (2, 3, 2)
        assert np.shape(raw["y"]) == (2, 3, 3)

    @pytest.mark.pending(
        reason="a record detaches to the nested mapping of its raw leaves", raises=AttributeError
    )
    def test_a_record_result_detaches_to_its_nested_mapping(self):
        law = _law("two-groups", "one-level")
        detached = mean(law).raw()
        assert _is_nested_mapping_of_arrays(detached)
        np.testing.assert_allclose(
            _at(detached, "population/mu"), _at(mean.with_options(raw=True)(law), "population/mu")
        )


def test_the_schemas_and_layouts_cover_nesting_and_levels():
    """The generators span a flat, a two-group, and a three-deep schema, on one to three levels."""
    depths = [max(path.count("/") for path in leaf_paths(spec)) for spec in SCHEMAS.values()]
    assert depths == [0, 1, 2]
    assert [len(names) for names, _ in LAYOUTS.values()] == [1, 2, 3]
