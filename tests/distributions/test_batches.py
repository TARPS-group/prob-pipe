"""The batch forms of the two distribution kinds.

A ``DistributionBatch`` holds separate laws sharing one event declaration, and a
``ConditionalDistributionBatch`` separate kernels sharing one given declaration
and one event declaration. Construction checks every element against the
element spec and names the position that fails; the shared declarations are
views on the batch's spec; indexing and levels follow ``Batch``; and the kind
table presents a law-valued or kernel-valued record field as the matching batch.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Batch,
    BatchSpec,
    EmpiricalDistribution,
    Laplace,
    MultivariateNormal,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordBatch,
    RecordSpec,
    StudentT,
    log_prob,
    mean,
    sample,
    variance,
    workflow_run,
)
from probpipe.core._kinds import batch_class_for_spec, term_class_for_spec
from probpipe.distributions import (
    ConditionalDistribution,
    ConditionalDistributionBatch,
    ConditionalDistributionSpec,
    Distribution,
    DistributionBatch,
    DistributionSpec,
)

SCALAR = NumericArraySpec(())


# -- Elements -----------------------------------------------------------------


class _Law(Distribution):
    """A law that declares its event and implements no capability."""


class Kernel(ConditionalDistribution):
    """A kernel over the slots and event it is given, giving a law over that event."""

    def _condition_on(self, given, /, **kwargs):
        return _Law(self.name, self.event_spec)


def _laws(count: int, name: str = "x") -> list[Normal]:
    """``count`` normal laws over one declaration, the law at position ``i`` with mean ``i``."""
    return [Normal(name, float(i), 1.0) for i in range(count)]


def _kernels(count: int, given=None, event=None) -> list[Kernel]:
    return [
        Kernel(
            "lik",
            {"mu": SCALAR} if given is None else given,
            OutputSpec(y=SCALAR) if event is None else event,
        )
        for _ in range(count)
    ]


def _objects(values: list, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """*values* as an object array, without unpacking any of them."""
    store = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        store[position] = value
    return store if shape is None else store.reshape(shape)


def _mean_of(law: Distribution) -> float:
    return float(np.asarray(mean(law)))


def _record_law(location: float) -> EmpiricalDistribution:
    """A law over records ``{x, y}`` whose atoms are all ``x == location`` and ``y == -location``."""
    x = jnp.full((8,), float(location))
    spec = NumericRecordSpec(x=NumericArraySpec((), x.dtype), y=NumericArraySpec((), x.dtype))
    return EmpiricalDistribution(
        "xy", NumericRecordBatch("atoms", {"x": x, "y": -x}, "atom", element_spec=spec)
    )


# -- Tests --------------------------------------------------------------------


class TestDistributionBatchConstruction:
    def test_a_distribution_batch_is_a_batch_specified_by_a_batch_spec(self):
        laws = _laws(2)
        batch = DistributionBatch("laws", laws, "law")
        assert isinstance(batch, Batch)
        assert isinstance(batch.spec, BatchSpec)
        assert batch.spec.element_spec == laws[0].spec
        assert batch.spec.is_valid(batch)

    def test_element_spec_defaults_to_the_first_element_spec(self):
        laws = _laws(3)
        batch = DistributionBatch("laws", laws, "law")
        assert batch.element_spec == laws[0].spec
        assert isinstance(batch.element_spec, DistributionSpec)

    def test_laws_of_different_families_share_one_declaration(self):
        laws = [Normal("x", 0.0, 1.0), Laplace("x", 0.0, 1.0), StudentT("x", 3.0, 0.0, 1.0)]
        batch = DistributionBatch("laws", laws, "law")
        assert batch.batch_shape == (3,)
        assert batch.event_spec == laws[0].event_spec

    def test_by_default_every_element_shares_the_first_declaration(self):
        laws = [
            MultivariateNormal("x", jnp.zeros(2), cov=jnp.eye(2)),
            MultivariateNormal("x", jnp.zeros(3), cov=jnp.eye(3)),
        ]
        with pytest.raises(TypeError, match="at 1"):
            DistributionBatch("laws", laws, "law")

    def test_a_polymorphic_element_spec_admits_elements_of_different_sizes(self):
        laws = [
            MultivariateNormal("x", jnp.zeros(2), cov=jnp.eye(2)),
            MultivariateNormal("x", jnp.zeros(3), cov=jnp.eye(3)),
        ]
        declared = DistributionSpec(OutputSpec(x=NumericArraySpec(("n",))))
        batch = DistributionBatch("laws", laws, "law", element_spec=declared)
        assert batch.element_spec == declared
        assert batch.event_spec == declared.event_spec

    @pytest.mark.parametrize(
        "stranger",
        [
            pytest.param(Normal("other", 0.0, 1.0), id="other-component"),
            pytest.param(MultivariateNormal("x", jnp.zeros(2), cov=jnp.eye(2)), id="other-shape"),
            pytest.param(_Law("x", OutputSpec(x=OpaqueSpec())), id="other-kind"),
            pytest.param(3.0, id="not-a-law"),
            pytest.param(_kernels(1, event=OutputSpec(x=SCALAR))[0], id="a-kernel"),
        ],
    )
    def test_an_element_that_does_not_match_raises_naming_its_position(self, stranger):
        laws = _laws(3)
        laws[1] = stranger
        with pytest.raises(TypeError, match="at 1"):
            DistributionBatch("laws", laws, "law")

    def test_a_law_of_other_components_raises_naming_the_components(self):
        with pytest.raises(TypeError, match=r"at 1 .*components \['x'\] and the law \['other'\]"):
            DistributionBatch("laws", [Normal("x", 0.0, 1.0), Normal("other", 0.0, 1.0)], "law")

    def test_a_mismatch_in_a_batch_of_several_axes_names_its_index(self):
        laws = _laws(6)
        laws[5] = Normal("other", 0.0, 1.0)
        with pytest.raises(TypeError, match=r"at \(1, 2\)"):
            DistributionBatch("grid", _objects(laws, shape=(2, 3)), ("row", "col"))

    def test_a_first_element_that_is_not_a_law_raises_naming_its_position(self):
        with pytest.raises(TypeError, match=r"\b0\b"):
            DistributionBatch("laws", [3.0, *_laws(2)], "law")

    def test_an_empty_batch_needs_an_element_spec(self):
        with pytest.raises(ValueError, match="element_spec"):
            DistributionBatch("laws", [], "law")

    def test_an_empty_batch_with_an_element_spec_constructs(self):
        declared = _laws(1)[0].spec
        batch = DistributionBatch("laws", [], "law", element_spec=declared)
        assert batch.batch_shape == (0,)
        assert batch.event_spec == declared.event_spec

    @pytest.mark.parametrize(
        "element_spec",
        [
            pytest.param(OutputSpec(x=SCALAR), id="output-spec"),
            pytest.param(SCALAR, id="array-spec"),
            pytest.param(
                ConditionalDistributionSpec({"mu": SCALAR}, OutputSpec(x=SCALAR)), id="kernel-spec"
            ),
        ],
    )
    def test_an_element_spec_that_is_not_a_distribution_spec_raises(self, element_spec):
        with pytest.raises(TypeError, match="DistributionSpec"):
            DistributionBatch("laws", _laws(2), "law", element_spec=element_spec)


class TestDistributionBatchDeclarations:
    def test_event_spec_is_the_shared_declaration_read_from_spec(self):
        laws = _laws(2)
        batch = DistributionBatch("laws", laws, "law")
        assert batch.event_spec is batch.element_spec.event_spec
        assert batch.event_spec == laws[0].event_spec

    def test_a_batch_of_random_measures_declares_law_valued_draws(self):
        drawn = DistributionSpec(OutputSpec(beta=SCALAR))
        measures = [_Law("measure", OutputSpec(posterior=drawn)) for _ in range(2)]
        batch = DistributionBatch("measures", measures, "measure")
        assert batch.event_spec == OutputSpec(posterior=drawn)
        assert batch.event_spec.spec == drawn

    @pytest.mark.pending(
        reason="a batch exposes the components of its element declaration",
        raises=AttributeError,
    )
    def test_the_component_of_a_batch_of_laws_defaults_to_the_batch_label(self):
        laws = _laws(2)
        batch = DistributionBatch("laws", laws, "law")
        assert dict(batch.components) == {"laws": laws[0].spec}

    @pytest.mark.pending(
        reason="a name key addresses the component of every element", raises=TypeError
    )
    def test_the_component_of_a_batch_of_laws_addresses_the_batch_itself(self):
        batch = DistributionBatch("laws", _laws(2), "law")
        assert batch["laws"] is batch


class TestIndexing:
    def test_an_element_carries_the_law_stored_at_its_position(self):
        laws = _laws(3)
        batch = DistributionBatch("laws", laws, "law")
        for position, law in enumerate(laws):
            element = batch[position]
            assert type(element) is Normal
            assert element.spec == law.spec
            assert _mean_of(element) == position
        assert _mean_of(batch[-1]) == 2.0

    def test_an_element_is_a_view_named_by_its_position(self):
        batch = DistributionBatch("laws", _laws(3), "law")
        assert batch[1].label == "laws[law=1]"
        assert batch.at_levels(law=2).label == "laws[law=2]"

    def test_an_element_shares_the_stored_law_and_leaves_it_untouched(self):
        laws = _laws(3)
        batch = DistributionBatch("laws", laws, "law")
        element = batch[1]
        assert element is not laws[1]
        assert element._tfp_dist is laws[1]._tfp_dist
        assert laws[1].label == "x" and laws[1].provenance is None

    def test_an_element_records_the_batch_and_the_stored_law(self):
        laws = _laws(3)
        batch = DistributionBatch("laws", laws, "law")
        provenance = batch[2].provenance
        assert provenance.operation == "__getitem__"
        assert [parent.label for parent in provenance.parents] == ["laws", "x"]
        assert provenance.metadata == {"position": [2]}

    def test_iteration_visits_the_laws_along_the_leading_axis(self):
        batch = DistributionBatch("laws", _laws(3), "law")
        assert len(batch) == 3
        assert [_mean_of(element) for element in batch] == [0.0, 1.0, 2.0]

    def test_a_sub_batch_is_a_distribution_batch_named_by_its_selection(self):
        batch = DistributionBatch("laws", _laws(4), "law")
        for sub in (batch[1:3], batch.at_levels(law=slice(1, 3))):
            assert isinstance(sub, DistributionBatch)
            assert sub.label == "laws[law=1:3]"
            assert sub.batch_shape == (2,)
            assert sub.level_names == ("law",)
            assert sub.event_spec == batch.event_spec
            assert [_mean_of(element) for element in sub] == [1.0, 2.0]

    def test_one_position_per_axis_selects_one_law(self):
        batch = DistributionBatch("grid", _objects(_laws(6), shape=(2, 3)), ("row", "col"))
        assert _mean_of(batch[1, 2]) == 5.0
        assert _mean_of(batch.at_levels(row=1, col=2)) == 5.0
        row = batch[1]
        assert isinstance(row, DistributionBatch)
        assert row.label == "grid[row=1]"
        assert [_mean_of(element) for element in row] == [3.0, 4.0, 5.0]


class TestLevels:
    def test_each_axis_is_its_own_level_by_default(self):
        batch = DistributionBatch("grid", _objects(_laws(6), shape=(2, 3)), ("row", "col"))
        assert batch.batch_shape == (2, 3)
        assert batch.batch_size == 6
        assert batch.axis_groups == ((2,), (3,))
        assert batch.level_names == ("row", "col")

    def test_axes_per_level_groups_axes_into_one_level(self):
        batch = DistributionBatch(
            "grid", _objects(_laws(6), shape=(2, 3)), "cell", axes_per_level=(2,)
        )
        assert batch.axis_groups == ((2, 3),)
        assert batch.level_names == ("cell",)

    @pytest.mark.parametrize(
        "level_names",
        [
            pytest.param(("row", "row"), id="duplicate"),
            pytest.param(("the row", "col"), id="not-an-identifier"),
            pytest.param(("row",), id="too-few"),
        ],
    )
    def test_level_names_are_unique_identifiers_one_per_level(self, level_names):
        with pytest.raises(ValueError):
            DistributionBatch("grid", _objects(_laws(6), shape=(2, 3)), level_names)

    def test_with_level_names_renames_the_levels_later_views_are_named_by(self):
        batch = DistributionBatch("laws", _laws(3), "law")
        renamed = batch.with_level_names(law="model")
        assert isinstance(renamed, DistributionBatch)
        assert renamed.label == "laws"
        assert renamed.level_names == ("model",)
        assert renamed[0:2].label == "laws[model=0:2]"
        assert batch.level_names == ("law",)


class TestConditionalDistributionBatch:
    def test_element_spec_defaults_to_the_first_kernel_spec(self):
        kernels = _kernels(2)
        batch = ConditionalDistributionBatch("kernels", kernels, "kernel")
        assert isinstance(batch, Batch)
        assert batch.element_spec == kernels[0].spec
        assert isinstance(batch.element_spec, ConditionalDistributionSpec)

    def test_given_spec_and_event_spec_are_the_shared_declarations_read_from_spec(self):
        kernels = _kernels(2, given={"mu": SCALAR, "x": NumericArraySpec((3,))})
        batch = ConditionalDistributionBatch("kernels", kernels, "kernel")
        assert batch.given_spec is batch.element_spec.given_spec
        assert batch.event_spec is batch.element_spec.event_spec
        assert batch.given_spec == kernels[0].given_spec
        assert batch.event_spec == kernels[0].event_spec

    @pytest.mark.parametrize(
        "stranger",
        [
            pytest.param(_kernels(1, given={"sigma": SCALAR})[0], id="other-given-slot"),
            pytest.param(_kernels(1, event=OutputSpec(z=SCALAR))[0], id="other-component"),
            pytest.param(Normal("y", 0.0, 1.0), id="a-law"),
            pytest.param("kernel", id="not-a-kernel"),
        ],
    )
    def test_an_element_that_does_not_match_raises_naming_its_position(self, stranger):
        kernels = _kernels(3)
        kernels[1] = stranger
        with pytest.raises(TypeError, match="at 1"):
            ConditionalDistributionBatch("kernels", kernels, "kernel")

    def test_a_polymorphic_element_spec_admits_kernels_of_different_sizes(self):
        kernels = [
            Kernel("lik", {"x": NumericArraySpec((size,))}, OutputSpec(y=NumericArraySpec((size,))))
            for size in (2, 3)
        ]
        declared = ConditionalDistributionSpec(
            {"x": NumericArraySpec(("n",))}, OutputSpec(y=NumericArraySpec(("n",)))
        )
        batch = ConditionalDistributionBatch("kernels", kernels, "kernel", element_spec=declared)
        assert batch.given_spec == declared.given_spec

    def test_an_empty_batch_needs_an_element_spec(self):
        with pytest.raises(ValueError, match="element_spec"):
            ConditionalDistributionBatch("kernels", [], "kernel")
        declared = _kernels(1)[0].spec
        batch = ConditionalDistributionBatch("kernels", [], "kernel", element_spec=declared)
        assert batch.batch_shape == (0,)
        assert batch.given_spec == declared.given_spec

    @pytest.mark.parametrize(
        "element_spec",
        [
            pytest.param(DistributionSpec(OutputSpec(y=SCALAR)), id="distribution-spec"),
            pytest.param(OutputSpec(y=SCALAR), id="output-spec"),
        ],
    )
    def test_an_element_spec_that_is_not_a_conditional_distribution_spec_raises(self, element_spec):
        with pytest.raises(TypeError, match="ConditionalDistributionSpec"):
            ConditionalDistributionBatch(
                "kernels", _kernels(2), "kernel", element_spec=element_spec
            )

    def test_an_element_carries_the_kernel_stored_at_its_position(self):
        kernels = _kernels(2)
        batch = ConditionalDistributionBatch("kernels", kernels, "kernel")
        assert type(batch[1]) is Kernel
        assert batch[1].spec == kernels[1].spec

    def test_an_element_is_a_view_of_the_stored_kernel(self):
        kernels = _kernels(2)
        batch = ConditionalDistributionBatch("kernels", kernels, "kernel")
        element = batch[1]
        assert element.label == "kernels[kernel=1]"
        assert [parent.label for parent in element.provenance.parents] == ["kernels", "lik"]
        assert kernels[1].label == "lik"

    def test_a_distribution_batch_refuses_kernels(self):
        with pytest.raises(TypeError, match="Distribution"):
            DistributionBatch("laws", _kernels(2), "law")


class TestKindRegistration:
    """Each distribution kind registers its batch form, which presents a record field."""

    @pytest.mark.parametrize(
        ("spec", "batch_class"),
        [
            pytest.param(DistributionSpec(OutputSpec(x=SCALAR)), DistributionBatch, id="law"),
            pytest.param(
                ConditionalDistributionSpec({"mu": SCALAR}, OutputSpec(y=SCALAR)),
                ConditionalDistributionBatch,
                id="kernel",
            ),
        ],
    )
    def test_each_spec_registers_its_batch_form(self, spec, batch_class):
        assert batch_class_for_spec(spec) is batch_class

    @pytest.mark.parametrize(
        ("spec", "term_class"),
        [
            pytest.param(DistributionSpec(OutputSpec(x=SCALAR)), Distribution, id="law"),
            pytest.param(
                ConditionalDistributionSpec({"mu": SCALAR}, OutputSpec(y=SCALAR)),
                ConditionalDistribution,
                id="kernel",
            ),
        ],
    )
    def test_each_spec_registers_its_tracked_class(self, spec, term_class):
        assert term_class_for_spec(spec) is term_class

    def test_a_record_batch_admits_a_field_declared_as_a_law(self):
        laws = _laws(3)
        batch = RecordBatch(
            "design",
            {"prior": _objects(laws), "x": jnp.zeros(3)},
            "row",
            element_spec=RecordSpec({"prior": laws[0].spec, "x": ()}),
        )
        assert batch.batch_shape == (3,)
        assert batch.element_spec["prior"] == laws[0].spec

    def test_reading_a_law_field_gives_a_distribution_batch_on_the_record_levels(self):
        laws = _laws(4)
        batch = RecordBatch(
            "design",
            {"prior": _objects(laws, shape=(2, 2)), "x": jnp.zeros((2, 2))},
            ("chain", "draw"),
            element_spec=RecordSpec({"prior": laws[0].spec, "x": ()}),
        )
        column = batch["prior"]
        assert isinstance(column, DistributionBatch)
        assert column.axis_groups == ((2,), (2,))
        assert column.level_names == ("chain", "draw")
        assert column.element_spec == laws[0].spec
        assert column.event_spec == laws[0].event_spec
        assert _mean_of(column[1, 0]) == 2.0

    def test_reading_a_kernel_field_gives_a_conditional_distribution_batch(self):
        kernels = _kernels(2)
        batch = RecordBatch(
            "design",
            {"likelihood": _objects(kernels)},
            "row",
            element_spec=RecordSpec({"likelihood": kernels[0].spec}),
        )
        column = batch["likelihood"]
        assert isinstance(column, ConditionalDistributionBatch)
        assert column.level_names == ("row",)
        assert column.given_spec == kernels[0].given_spec
        assert column.event_spec == kernels[0].event_spec

    def test_a_field_entry_that_does_not_match_raises_naming_its_position(self):
        laws = _laws(3)
        declared = laws[0].spec
        laws[1] = Normal("other", 0.0, 1.0)
        with pytest.raises(TypeError, match="at 1"):
            RecordBatch(
                "design",
                {"prior": _objects(laws), "x": jnp.zeros(3)},
                "row",
                element_spec=RecordSpec({"prior": declared, "x": ()}),
            )

    def test_stacked_records_holding_laws_give_a_distribution_column(self):
        laws = _laws(3)
        records = [Record("draw", prior=law, x=float(i)) for i, law in enumerate(laws)]
        column = RecordBatch.stack(records, level_name="row")["prior"]
        assert isinstance(column, DistributionBatch)
        assert column.level_names == ("row",)
        assert column.element_spec == laws[0].spec
        assert [_mean_of(element) for element in column] == [0.0, 1.0, 2.0]


class TestOperationsSweepTheLaws:
    """An operation maps over the laws of a batch, keeping the batch's levels (VI.11)."""

    def test_a_draw_of_each_law_is_a_batch_on_the_laws_level(self):
        batch = DistributionBatch("laws", _laws(4), "law")
        with workflow_run(seed=0):
            drawn = sample(batch)
        assert isinstance(drawn, NumericArrayBatch)
        assert (drawn.batch_shape, drawn.level_names) == ((4,), ("law",))

    def test_draws_of_each_law_nest_the_sample_level_inside_the_laws(self):
        batch = DistributionBatch("laws", _laws(4), "law")
        with workflow_run(seed=0):
            drawn = sample(batch, sample_shape=(7,))
        assert isinstance(drawn, NumericArrayBatch)
        assert (drawn.batch_shape, drawn.level_names) == ((4, 7), ("law", "sample"))
        assert tuple(drawn.element_spec.shape) == ()

    def test_each_law_draws_its_own_values(self):
        laws = [Normal("x", 100.0 * i, 1e-3) for i in range(3)]
        with workflow_run(seed=0):
            drawn = sample(DistributionBatch("laws", laws, "law"), sample_shape=(200,))
        np.testing.assert_allclose(drawn.values.mean(axis=-1), [0.0, 100.0, 200.0], atol=0.2)

    def test_several_levels_stay_in_front_of_the_sample_level(self):
        batch = DistributionBatch("grid", _objects(_laws(6), shape=(2, 3)), ("row", "col"))
        with workflow_run(seed=0):
            drawn = sample(batch, sample_shape=(5,))
        assert (drawn.batch_shape, drawn.level_names) == ((2, 3, 5), ("row", "col", "sample"))

    def test_record_laws_draw_a_record_batch(self):
        batch = DistributionBatch("laws", [_record_law(i) for i in range(3)], "law")
        with workflow_run(seed=0):
            one = sample(batch)
            several = sample(batch, sample_shape=(5,))
        assert isinstance(one, NumericRecordBatch)
        np.testing.assert_allclose(one["x"], [0.0, 1.0, 2.0])
        assert several.level_names == ("law", "sample")
        assert several["x"].shape == several["y"].shape == (3, 5)

    def test_the_moments_of_each_law_form_a_batch(self):
        batch = DistributionBatch("laws", [Normal("x", float(i), i + 1.0) for i in range(3)], "law")
        means, variances = mean(batch), variance(batch)
        assert isinstance(means, NumericArrayBatch) and means.batch_shape == (3,)
        np.testing.assert_allclose(means.values, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(variances.values, [1.0, 4.0, 9.0])

    def test_the_mean_of_each_record_law_is_a_record_batch(self):
        means = mean(DistributionBatch("laws", [_record_law(i) for i in range(3)], "law"))
        assert isinstance(means, NumericRecordBatch)
        np.testing.assert_allclose(means["x"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(means["y"], [0.0, -1.0, -2.0])

    def test_one_value_is_scored_under_each_law(self):
        laws = _laws(3)
        scores = log_prob(DistributionBatch("laws", laws, "law"), jnp.asarray(0.0))
        assert isinstance(scores, NumericArrayBatch) and scores.batch_shape == (3,)
        expected = [float(law._log_prob(0.0)) for law in laws]
        np.testing.assert_allclose(scores.values, expected, rtol=1e-5)
