"""Labeling across the tracked terms, the batches, and the operations.

Every tracked term receives its label at construction and preserves it through
structural transforms. Only ``with_label`` replaces it. New operation results
and accessed views receive their labels when constructed, across every kind.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    FunctionBatch,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecord,
    NumericRecordBatch,
    Opaque,
    OpaqueBatch,
    OutputSpec,
    Record,
    RecordBatch,
    RecordSpec,
    condition_on,
    conditional_distribution,
    convert,
    cov,
    expectation,
    factor,
    function,
    log_prob,
    marginal,
    mean,
    prob,
    quantile,
    sample,
    unnormalized_log_prob,
    unnormalized_prob,
    variance,
    workflow_run,
)
from probpipe.core._expression import Signature
from probpipe.core._specs import NumericRecordSpec
from probpipe.distributions import FactoredDistribution
from probpipe.distributions._batches import DistributionBatch
from probpipe.functions._call import ApplicabilityError

KEY = jax.random.PRNGKey(0)
ELEMENT = NumericRecordSpec(a=())
COLUMNS = {"a": jnp.arange(4.0)}


def _labeled(kind):
    """One instance of *kind* built with an explicit label."""
    return {
        "Record": lambda: Record(
            {"a": 1.0},
            label="given",
        ),
        "NumericRecord": lambda: NumericRecord(
            {"a": 1.0},
            label="given",
        ),
        "NumericArray": lambda: NumericArray(
            jnp.arange(3.0),
            label="given",
        ),
        "Opaque": lambda: Opaque(
            object(),
            label="given",
        ),
        "Function": lambda: Function(lambda: 1, label="given"),
        "Normal": lambda: Normal("x", 0.0, 1.0, label="given"),
        "RecordBatch": lambda: RecordBatch(
            COLUMNS,
            "lvl",
            element_spec=ELEMENT,
            label="given",
        ),
        "NumericRecordBatch": lambda: NumericRecordBatch(
            COLUMNS,
            "lvl",
            element_spec=ELEMENT,
            label="given",
        ),
        "NumericArrayBatch": lambda: NumericArrayBatch(
            jnp.arange(4.0),
            "lvl",
            element_spec=NumericArraySpec(shape=()),
            label="given",
        ),
        "OpaqueBatch": lambda: OpaqueBatch(
            [1, 2],
            "lvl",
            label="given",
        ),
        "FunctionBatch": lambda: FunctionBatch(
            [lambda: 1],
            "lvl",
            label="given",
        ),
    }[kind]()


EVERY_KIND = [
    "Record",
    "NumericRecord",
    "NumericArray",
    "Opaque",
    "Function",
    "Normal",
    "RecordBatch",
    "NumericRecordBatch",
    "NumericArrayBatch",
    "OpaqueBatch",
    "FunctionBatch",
]


class TestLabelsAreKept:
    """The rule every kind shares, and the one an operation reads before relabeling."""

    @pytest.mark.parametrize("kind", EVERY_KIND)
    def test_a_given_label_is_kept_verbatim(self, kind):
        assert _labeled(kind).label == "given"

    @pytest.mark.parametrize("kind", EVERY_KIND)
    def test_label_origin_is_not_part_of_the_public_term(self, kind):
        term = _labeled(kind)
        assert not hasattr(term, "name_is_auto")
        assert not hasattr(term, "_name_is_auto")
        renamed = term.with_label("replacement")
        assert renamed.label == "replacement"
        assert term.label == "given"
        assert not hasattr(renamed, "name_is_auto")

    @pytest.mark.parametrize("kind", [Record, NumericRecord])
    def test_former_metadata_keyword_is_an_ordinary_record_field(self, kind):
        record = kind({"name_is_auto": True}, label="flags")
        assert tuple(record) == ("name_is_auto",)
        assert bool(record["name_is_auto"])
        assert record.label == "flags"


class TestWhichKindsRequireALabel:
    """A label is required where nothing else identifies the value.

    A record derives ``record(field,...)`` from its fields, a batch of records
    from its element spec's fields, and a batch of functions or laws from its
    members, so each of those requires a label only when it has nothing to
    derive one from. An array and an opaque value carry nothing to describe
    them, so they always require one.
    """

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            pytest.param(
                lambda: Record({}),
                r"^cannot derive a default label for a Record with no fields",
                id="Record",
            ),
            pytest.param(
                lambda: NumericRecord({}),
                r"^cannot derive a default label for a Record with no fields",
                id="NumericRecord",
            ),
            pytest.param(
                lambda: NumericArray(jnp.ones(3)),
                r"missing 1 required keyword-only argument: 'label'",
                id="NumericArray",
            ),
            pytest.param(
                lambda: Opaque(object()),
                r"missing 1 required keyword-only argument: 'label'",
                id="Opaque",
            ),
            pytest.param(
                lambda: NumericArrayBatch(jnp.arange(4.0), "lvl"),
                r"missing 1 required keyword-only argument: 'label'",
                id="NumericArrayBatch",
            ),
            pytest.param(
                lambda: OpaqueBatch(["north"], "site"),
                r"missing 1 required keyword-only argument: 'label'",
                id="OpaqueBatch",
            ),
            pytest.param(
                lambda: FunctionBatch([], "variant"),
                r"^cannot derive a default label for an empty FunctionBatch",
                id="FunctionBatch",
            ),
        ],
    )
    def test_a_label_is_required(self, build, match):
        with pytest.raises(TypeError, match=match):
            build()

    @pytest.mark.parametrize(
        ("build", "label"),
        [
            pytest.param(lambda: Record({"a": 1.0, "b": 2.0}), "record(a,b)", id="Record"),
            pytest.param(
                lambda: RecordBatch({"a": jnp.zeros(3)}, "draw"), "record(a)", id="RecordBatch"
            ),
            pytest.param(lambda: FunctionBatch([abs], "variant"), "[abs(x)]", id="FunctionBatch"),
        ],
    )
    def test_a_structured_value_derives_its_label(self, build, label):
        assert build().label == label

    def test_a_function_takes_its_callables_name(self):
        def predict():
            return 1.0

        assert Function(predict).label == "predict"


class TestADerivedLabelSaysSo:
    """A view labels itself after the position it selected, and marks it auto."""

    @staticmethod
    def _batch():
        return NumericArrayBatch(
            jnp.arange(12.0).reshape(4, 3),
            "draw",
            element_spec=NumericArraySpec(shape=(3,)),
            label="posterior",
        )

    def test_an_element_is_labeled_for_its_position(self):
        element = self._batch()[1]

        assert element.label == "posterior[draw=1]"

    def test_a_sub_batch_is_labeled_for_its_slice(self):
        sub = self._batch()[1:3]

        assert sub.label == "posterior[draw=1:3]"

    def test_a_derived_label_builds_on_the_given_one(self):
        """So the lineage reads back to the batch a caller actually labeled."""
        assert self._batch()[1].label.startswith("posterior")


class TestAnOperationLabelsItsResultByItsLaw:
    """A value computed from a law is labeled by that value over the law (II.4).

    The law here is a prior over ``height``, so a draw reads ``height ~ prior``.
    """

    LAW = Normal("height", 0.0, 1.0, label="prior")

    @pytest.mark.parametrize(
        ("compute", "label"),
        [
            (lambda d: mean(d), "E[height ~ prior]"),
            (lambda d: variance(d), "Var[height ~ prior]"),
            (lambda d: log_prob(d, jnp.asarray(0.0)), "log prior(height)"),
        ],
        ids=["mean", "variance", "log_prob"],
    )
    def test_a_scalar_law_result_is_labeled_over_the_law(self, compute, label):
        assert compute(self.LAW).label == label

    def test_a_record_law_draw_is_labeled_by_its_components_and_the_law(self):
        joint = FactoredDistribution(
            [Normal("a", 0.0, 1.0)],
            label="joint",
        )

        assert sample(joint).label == "a ~ joint"

    @pytest.mark.parametrize("sample_shape", [(), (4,)], ids=["single", "batch"])
    def test_draws_are_labeled_as_one_draw(self, sample_shape):
        """Both a single draw and a batch cross the same result boundary."""
        given = sample(Normal("height", 0.0, 1.0, label="prior"), sample_shape=sample_shape)

        assert given.label == "height ~ prior"

    @staticmethod
    def _params():
        prior_x = Normal("x", 0.0, 1.0, label="prior_x")
        return (prior_x * Normal("y", 2.0, 3.0, label="prior_y")).with_label("params")

    def test_a_record_mean_is_labeled_over_the_law_and_names_its_components(
        self, full_provenance_mode
    ):
        law = self._params()
        result = mean(law)
        assert result.label == "E[(x, y) ~ params]"
        assert list(result.keys()) == ["mean(x)", "mean(y)"]
        assert float(result["mean(x)"]) == 0.0
        assert float(result["mean(y)"]) == 2.0
        assert result.provenance.parents[0].parent is mean
        assert result.provenance.parents[1].parent is law

    def test_conditioning_on_a_factor_takes_the_label_of_the_factor_it_leaves(
        self, full_provenance_mode
    ):
        """Fixing the whole event of a factor leaves the other factor (VI.6)."""
        law = self._params()
        result = condition_on(law, {"x": 1.0})
        assert result.label == "prior_y"
        assert tuple(result.event_spec.components) == ("y",)
        assert tuple(law.event_spec.components) == ("x", "y")
        assert float(mean(result)) == 2.0
        assert float(variance(result)) == 9.0
        assert result.provenance.parents[0].parent is condition_on
        assert result.provenance.parents[1].parent is law

    def test_a_converted_law_keeps_its_label(self, full_provenance_mode):
        law = Normal("theta", 2.0, 3.0, label="prior")
        result = convert(law, Normal)
        assert result is not law
        assert result.label == "prior"
        assert tuple(result.event_spec.components) == ("theta",)
        assert float(mean(result)) == 2.0
        assert float(variance(result)) == 9.0
        assert result.provenance.parents[0].parent is convert
        assert result.provenance.parents[1].parent is law


class TestTheOutputBoundaryLabelsEveryKindAlike:
    """Whatever kind a body returns, the result is labeled by the function's call."""

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("numeric", lambda: jnp.arange(3.0)),
            ("mapping", lambda: {"a": 1.0}),
            ("opaque", lambda: "a string"),
            ("callable", lambda: lambda: 1),
            ("sequence", lambda: [1.0, 2.0]),
            ("empty mapping", lambda: {}),
            ("empty sequence", lambda: []),
        ],
    )
    def test_the_result_is_labeled_by_the_call(self, label, body):
        result = Function(body, label="myfunc")()

        assert result.label == "myfunc()"


class TestLevelsAreNamedForWhatMintsThem:
    """An operation names the level it mints after itself (design V.9)."""

    def test_sample_mints_a_sample_level(self):
        drawn = sample(Normal("height", 0.0, 1.0), sample_shape=(5,))

        assert drawn.level_names == ("sample",)

    def test_a_record_drawing_law_mints_the_same_level(self):
        joint = FactoredDistribution(
            [Normal("a", 0.0, 1.0)],
            label="joint",
        )

        drawn = sample(joint, sample_shape=(5,))

        assert drawn.level_names == ("sample",)

    @pytest.mark.parametrize(
        ("atoms", "component", "expected"),
        [
            pytest.param(
                jnp.linspace(0.0, 1.0, 5), "atoms", "NumericArrayBatch", id="numeric-atoms"
            ),
            pytest.param(
                NumericRecordBatch(
                    {"u": jnp.arange(4.0)},
                    "row",
                    element_spec=RecordSpec(u=()),
                    label="rows",
                ),
                None,
                "NumericRecordBatch",
                id="record-atoms",
            ),
            pytest.param(
                OpaqueBatch(
                    [object() for _ in range(3)],
                    "atom",
                    label="objects",
                ),
                "atoms",
                "OpaqueBatch",
                id="opaque-atoms",
            ),
        ],
    )
    def test_a_law_that_assembles_its_own_draws_still_gets_the_level(
        self, atoms, component, expected
    ):
        """The boundary mints the level for every kind of draw.

        These laws lay the draws out themselves, in the batch form of their atoms,
        and name no level.
        """
        from probpipe import EmpiricalDistribution

        drawn = sample(EmpiricalDistribution(atoms, component=component), sample_shape=(3,))

        assert type(drawn).__name__ == expected
        assert (drawn.batch_shape, drawn.level_names) == ((3,), ("sample",))

    def test_a_single_draw_from_such_a_law_is_not_a_batch(self):
        """No sample_shape, no level to mint."""
        from probpipe import EmpiricalDistribution

        drawn = sample(EmpiricalDistribution(jnp.linspace(0.0, 1.0, 5), component="atoms"))

        assert not isinstance(drawn, NumericArrayBatch)

    def test_the_draws_take_the_label_of_one_draw(self):
        from probpipe import EmpiricalDistribution

        drawn = sample(
            EmpiricalDistribution(
                OpaqueBatch(
                    [object() for _ in range(3)],
                    "atom",
                    label="objects",
                ),
                component="atoms",
                label="empirical",
            ),
            sample_shape=(3,),
        )

        assert drawn.label == "atoms ~ empirical"


class TestABatchOperandKeepsItsLevelsThroughAnOperation:
    """Design V.9: a density op maps elementwise "with the batch axes preserved".

    An operation whose value parameter is `Any`-hinted takes the batch whole and
    evaluates it in one vectorized call — the fused implementation V.9 allows.
    That is not licence to hand back a bare array: the axes the operand accounted
    for are levels, and a result that drops them says the draws were one value.

    Every op that scores a value is covered, since which of them restates the
    levels is not something a caller should have to know. `prob` and the two
    unnormalized ops used to drop them, so the same draws scored as a batch under
    one op and as one wide value under another.
    """

    LAW = Normal("height", 0.0, 1.0, label="prior")

    @pytest.fixture(
        params=[log_prob, prob, unnormalized_log_prob, unnormalized_prob], ids=lambda op: op.label
    )
    def density_op(self, request):
        return request.param

    def test_scoring_a_batch_of_draws_keeps_the_sample_level(self, density_op):
        drawn = sample(self.LAW, sample_shape=(3,))

        scored = density_op(self.LAW, drawn)

        assert (scored.batch_shape, scored.level_names) == ((3,), ("sample",))

    def test_the_result_is_labeled_over_the_law(self, density_op):
        """A score reads ``log`` and the law's notation, and a density the notation."""
        drawn = sample(self.LAW, sample_shape=(3,))

        expected = (
            "prior(height)" if density_op in (prob, unnormalized_prob) else "log prior(height)"
        )
        assert density_op(self.LAW, drawn).label == expected

    def test_several_levels_are_all_restated(self, density_op):
        """The operand's own tiling, not one flat axis."""
        drawn = NumericArrayBatch(
            jnp.zeros((2, 3)),
            ("chain", "draw"),
            element_spec=NumericArraySpec(()),
            axes_per_level=(1, 1),
            label="draws",
        )

        scored = density_op(self.LAW, drawn)

        assert (scored.batch_shape, scored.level_names) == ((2, 3), ("chain", "draw"))

    def test_a_single_draw_is_still_a_single_value(self, density_op):
        """No operand levels to restate, so nothing is invented."""
        scored = density_op(self.LAW, sample(self.LAW))

        assert not isinstance(scored, NumericArrayBatch)

    def test_a_raw_array_of_several_values_does_not_conform(self, density_op):
        """A bare array states no levels, so it is one value of the wrong shape."""
        with pytest.raises(ApplicabilityError, match="does not conform"):
            density_op(self.LAW, jnp.zeros(3))


class TestRawDrawLabeling:
    @pytest.mark.parametrize(
        ("value", "declaration", "completed"),
        [
            (2.0, OutputSpec(x=NumericArraySpec(())), NumericArraySpec((), dtype="float32")),
            ({"x": 2.0}, OutputSpec(RecordSpec(x=NumericArraySpec(()))), RecordSpec(x=())),
        ],
        ids=["scalar", "mapping"],
    )
    def test_a_declared_function_result_takes_the_requested_label(
        self, value, declaration, completed
    ):
        wrapped = Function(
            lambda: value,
            output_spec=declaration,
            output_label="law",
            label="producer",
        )
        result = wrapped()
        assert wrapped.apply() is value
        assert result.label == "law"
        # The declaration leaves the dtype open, so the result takes the returned one (II.2).
        assert result.spec == completed
        assert float(result["x"] if isinstance(result, Record) else result) == 2.0


class TestEveryAggregateIsLabeledForItsFunction:
    """The labeling table, widened across the axes that had diverged.

    A sweep's aggregate is built by the boundary, not by a caller, so its label is
    the producing function's call on the swept batch, as ``double(rows)``. Three
    paths disagreed: the
    undeclared record aggregate took `stack`'s class-name default, and the scalar,
    opaque, and declared paths marked a derived label as user-given — which would
    stop a later operation relabeling it.
    """

    @staticmethod
    def _rows(n: int = 3):
        from probpipe.core._specs import NumericRecordSpec

        return NumericRecordBatch(
            {"x": jnp.arange(float(n))},
            "row",
            element_spec=NumericRecordSpec(x=()),
            label="rows",
        )

    def _swept(self, body, **controls):
        return Function(body, label="double", dispatch="sequential", **controls)(v=self._rows())

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("numeric", lambda v: jnp.asarray(v["x"]) * 2),
            ("mapping", lambda v: {"y": jnp.asarray(v["x"])}),
            ("opaque", lambda v: "tag"),
            ("callable", lambda v: lambda: 1),
            ("sequence", lambda v: [jnp.asarray(v["x"]), jnp.asarray(v["x"])]),
        ],
    )
    def test_an_undeclared_aggregate_is_labeled_by_the_call(self, label, body):
        result = self._swept(body)

        assert result.label == "double(rows)"

    def test_a_declared_aggregate_is_labeled_the_same_way(self):
        from probpipe import RecordSpec

        result = self._swept(lambda v: {"y": jnp.asarray(v["x"])}, output_spec=RecordSpec(y=()))

        assert result.label == "double(rows)"

    def test_a_multi_axis_sweep_is_labeled_the_same_way(self):
        """The re-cut to the sweep's own geometry is a separate construction, and
        it had its own labeling."""
        from probpipe.core._specs import NumericRecordSpec

        grid = NumericRecordBatch(
            {"x": jnp.arange(6.0).reshape(2, 3)},
            ("a", "b"),
            element_spec=NumericRecordSpec(x=()),
            label="grid",
        )

        result = Function(
            lambda v: {"y": jnp.asarray(v["x"])}, label="double", dispatch="sequential"
        )(v=grid)

        assert result.label == "double(grid)"
        assert result.level_names == ("a", "b")


class TestNoKindInventsALabel:
    """The rule the whole layer now shares: a label is given, or derived from
    something that carries meaning. A class name carries none.

    Every batch defaulted to its own lowercased class name, so a pipeline full
    of them read `recordbatch`, `opaquebatch`, `numericrecordbatch` — labels that
    say what the object *is*, which its type already says, and nothing about
    which one it is.

    That every constructor takes the label first, positional-only and with no
    default behind it, is asserted from the signatures themselves in
    `test_batch.py`'s `TestTheConstructorSignatureContract`. What is left here is
    the other half of the rule: where a *derived* label comes from.
    """

    def test_stack_derives_its_label_from_what_it_stacks(self):
        """Derived from real content, so no call site has to invent one: a batch
        of `draw` records is about `draw`."""
        rows = [
            NumericRecord(
                {"a": float(i)},
                label="draw",
            )
            for i in range(3)
        ]

        batch = NumericRecordBatch.stack(rows, level_name="row")

        assert batch.label == "draw"

    def test_stack_takes_a_better_label_when_offered(self):
        rows = [
            NumericRecord(
                {"a": float(i)},
                label="draw",
            )
            for i in range(3)
        ]

        batch = NumericRecordBatch.stack(rows, level_name="row", label="posterior")

        assert batch.label == "posterior"

    def test_a_structural_transform_preserves_the_label(self):
        """There is no class-name default to re-derive from, and an auto name is
        something derived rather than a placeholder."""
        batch = NumericRecordBatch(
            {"a": jnp.zeros(3), "b": jnp.zeros(3)},
            "lvl",
            element_spec=NumericRecordSpec(a=(), b=()),
            label="derived",
        )

        edited = batch.without("b")

        assert edited.label == "derived"


# ---------------------------------------------------------------------------
# The labels of results and of values computed from a law (design II.4)
# ---------------------------------------------------------------------------


def _prior() -> Normal:
    """A law ``prior`` over ``mu``."""
    return Normal("mu", 0.0, 1.0, label="prior")


def _likelihood() -> Any:
    """A kernel ``lik`` over ``y`` given ``mu``."""
    return conditional_distribution(
        lambda mu: Normal("y", mu, 1.0), given_spec={"mu": NumericArraySpec(())}, label="lik"
    )


def _model() -> Any:
    """The joint ``model`` over ``y`` and ``mu``."""
    return (_likelihood() * _prior()).with_label("model")


def _empirical_model() -> Any:
    """``model`` over ``y`` and ``mu`` whose prior is empirical, so conditioning on ``y`` is exact."""
    atoms = jnp.linspace(-2.0, 2.0, 41)
    prior = EmpiricalDistribution(atoms, event_spec=OutputSpec(mu=None), label="prior")
    return (_likelihood() * prior).with_label("model")


def _glm(*slots: str) -> Any:
    """A kernel ``glm`` over ``y`` given *slots*."""

    def location(**given: Any) -> Normal:
        return Normal("y", sum(given.values()), 1.0)

    def body(beta, sigma):
        return location(beta=beta, sigma=sigma)

    spec = {slot: NumericArraySpec(()) for slot in slots}
    if slots == ("beta",):
        return conditional_distribution(
            lambda beta: location(beta=beta), given_spec=spec, label="glm"
        )
    return conditional_distribution(body, given_spec=spec, label="glm")


class TestTheLabelsOfResults:
    """A law derived from a law keeps its label, and its signature is its own (II.4)."""

    @pytest.mark.parametrize(
        ("compute", "notation"),
        [
            pytest.param(lambda: _model()["y"], "model(y)", id="view"),
            pytest.param(lambda: marginal(_model(), "mu"), "prior(mu)", id="marginal-at-a-factor"),
            pytest.param(lambda: _model()["mu"], "prior(mu)", id="view-at-a-factor"),
            pytest.param(lambda: factor(_model(), "mu"), "prior(mu)", id="factor"),
            pytest.param(
                lambda: condition_on(_empirical_model(), {"y": 0.5}), "model(mu; y)", id="posterior"
            ),
            pytest.param(
                lambda: condition_on(_model(), {"mu": 0.5}), "lik(y; mu)", id="factor-left"
            ),
            pytest.param(
                lambda: condition_on(_glm("beta"), {"beta": 1.0}), "glm(y; beta)", id="kernel"
            ),
            pytest.param(
                lambda: condition_on(_glm("beta", "sigma"), {"beta": 1.0}),
                "glm(y | sigma; beta)",
                id="kernel-at-some-slots",
            ),
        ],
    )
    def test_a_derived_law_displays_by_the_issue_table(self, compute, notation):
        with workflow_run(seed=0):
            assert compute().notation == notation

    def test_the_prior_predictive_keeps_the_models_label(self):
        """``marginal(model, "y")`` integrates ``mu`` out, which no route of this model does."""
        expression = marginal._derived_expression({"d": _model(), "field": "y"})
        assert expression.render_notation(Signature(("y",))) == "model(y)"


#: A default that is not a scalar, which a signature shows as ``…``.
_OFFSETS = jnp.zeros(3)


def _counts() -> Any:
    """A kernel over ``y`` given ``K`` and ``r``, with the slot ``n0`` defaulting to 50.0."""

    def counts(K: jax.Array, r: jax.Array, n0: float = 50.0) -> Normal:
        return Normal("y", K * r * n0, 1.0)

    return conditional_distribution(
        counts, given_spec={"K": NumericArraySpec(()), "r": NumericArraySpec(())}
    )


class TestADefaultedSlotShowsItsDefault:
    """A given slot or a parameter with a default reads ``name=value`` until a given binds it."""

    def test_a_kernel_shows_its_defaulted_slot(self):
        assert str(_counts()) == "counts(y | K, r, n0=50.0)"

    def test_a_law_left_at_a_default_keeps_the_slot_as_given(self):
        at = condition_on(_counts(), {"K": 300.0, "r": 0.4})
        assert at.notation == "counts(y | n0=50.0; K, r)"
        assert not hasattr(at, "given_spec")

    def test_a_curried_kernel_keeps_the_defaulted_slot_after_the_free_slots(self):
        assert str(condition_on(_counts(), {"K": 300.0})) == "counts(y | r, n0=50.0; K)"

    def test_binding_the_defaulted_slot_moves_it_to_the_fixed_paths(self):
        bound = condition_on(_counts(), {"K": 300.0, "r": 0.4, "n0": 10.0})
        assert bound.notation == "counts(y; K, r, n0)"
        assert str(condition_on(_counts(), {"n0": 10.0})) == "counts(y | K, r; n0)"

    def test_a_factor_left_by_conditioning_a_joint_keeps_the_defaulted_slot(self):
        prior = (Normal("K", 300.0, 10.0) * Normal("r", 0.4, 0.1)).with_label("prior")
        model = _counts() * prior
        assert str(model) == "counts(y | K, r, n0=50.0)·prior(K, r)"
        assert condition_on(model, {"K": 300.0, "r": 0.4}).notation == ("counts(y | n0=50.0; K, r)")

    def test_a_relabeled_law_keeps_the_defaulted_slot(self):
        at = condition_on(_counts(), {"K": 300.0, "r": 0.4}).with_label("at_values")
        assert at.notation == "at_values(y | n0=50.0; K, r)"

    def test_a_relabeled_law_keeps_the_defaulted_slot_inside_another_term(self):
        at = condition_on(_counts(), {"K": 300.0, "r": 0.4}).with_label("at_values")
        assert log_prob(at, 1.0).label == "log at_values(y | n0=50.0; K, r)"
        assert str(at * Normal("z", 0.0, 1.0)) == "at_values(y | n0=50.0; K, r)·Normal(z)"

    def test_a_value_computed_from_the_law_reads_its_notation(self):
        at = condition_on(_counts(), {"K": 300.0, "r": 0.4})
        assert log_prob(at, 1.0).label == "log counts(y | n0=50.0; K, r)"
        with workflow_run(seed=0):
            assert sample(at).label == "y ~ counts; K, r"

    def test_a_non_scalar_default_shows_an_ellipsis(self):
        def shifted(mu: jax.Array, offsets: jax.Array = _OFFSETS) -> Normal:
            return Normal("y", mu + offsets, 1.0)

        kernel = conditional_distribution(shifted, given_spec={"mu": NumericArraySpec(())})
        assert str(kernel) == "shifted(y | mu, offsets=…)"

    def test_a_function_shows_its_parameters_defaults(self):
        @function
        def predict(x: jax.Array, scale: float = 1.0) -> jax.Array:
            return x * scale

        assert str(predict) == "predict(x, scale=1.0)"


class TestTheLabelsOfValuesComputedFromALaw:
    """A value computed from a law is labeled by the value over the law, in probability notation."""

    @pytest.mark.parametrize(
        ("operation", "arguments", "label", "components"),
        [
            pytest.param(sample, lambda: (_prior(),), "mu ~ prior", ("mu",), id="draw"),
            pytest.param(sample, lambda: (_prior(), (3,)), "mu ~ prior", ("mu",), id="draws"),
            pytest.param(sample, lambda: (_model(),), "(y, mu) ~ model", ("y", "mu"), id="joint"),
            pytest.param(
                log_prob, lambda: (_prior(), 0.3), "log prior(mu)", ("log_prob(mu)",), id="score"
            ),
            pytest.param(
                log_prob,
                lambda: (_model(), {"y": 0.1, "mu": 0.3}),
                "log model(y, mu)",
                ("log_prob(y, mu)",),
                id="joint-score",
            ),
            pytest.param(
                unnormalized_log_prob,
                lambda: (_prior(), 0.3),
                "log prior(mu)",
                ("unnormalized_log_prob(mu)",),
                id="unnormalized-score",
            ),
            pytest.param(prob, lambda: (_prior(), 0.3), "prior(mu)", ("prob(mu)",), id="density"),
            pytest.param(
                mean, lambda: (_model(),), "E[(y, mu) ~ model]", ("mean(y)", "mean(mu)"), id="mean"
            ),
            pytest.param(
                variance,
                lambda: (_model(),),
                "Var[(y, mu) ~ model]",
                ("variance(y)", "variance(mu)"),
                id="variance",
            ),
            pytest.param(
                cov, lambda: (_model(),), "Cov[(y, mu) ~ model]", ("cov(y, mu)",), id="cov"
            ),
            pytest.param(
                quantile, lambda: (_prior(), 0.5), "Q[mu ~ prior]", ("quantile(mu)",), id="quantile"
            ),
            pytest.param(
                mean, lambda: (_model()["y"],), "E[y ~ model]", ("mean(y)",), id="mean-of-a-view"
            ),
        ],
    )
    def test_a_value_is_labeled_by_the_issue_table(self, operation, arguments, label, components):
        with workflow_run(seed=0):
            result = operation(*arguments())
        assert result.label == label
        assert tuple(operation.check(*arguments()).result.components) == components

    def test_an_expectation_is_labeled_by_its_integrand_at_a_draw(self):
        def square(x: jax.Array) -> jax.Array:
            return x**2

        with workflow_run(seed=0):
            assert expectation(_prior(), square).label == "E[square(mu ~ prior)]"
            assert expectation(_prior(), lambda x: x).label == "E[f(mu ~ prior)]"

    def test_an_expectation_names_its_integrand_as_a_lifted_call_does(self):
        """A relabeled Function keeps its output label, which both labels read."""

        @function(output_spec=OutputSpec(squared=None), output_label="sq")
        def square(x: jax.Array) -> jax.Array:
            return x**2

        renamed = square.with_label("g")
        with workflow_run(seed=0):
            lifted = mean(renamed.with_options(n_broadcast_samples=8)(_prior()))
            assert expectation(_prior(), renamed).label == lifted.label == "E[sq(mu ~ prior)]"

    def test_a_draw_and_a_score_of_a_posterior_list_its_fixed_paths(self):
        with workflow_run(seed=0):
            posterior = condition_on(_empirical_model(), {"y": 0.5})
            assert sample(posterior).label == "mu ~ model; y"
            assert log_prob(_prior()._with_expression(posterior._expression), 0.1).label == (
                "log model(mu; y)"
            )

    def test_an_element_of_a_batch_of_draws_groups_the_draw(self):
        with workflow_run(seed=0):
            draws = sample(_prior(), sample_shape=(3,))
            scores = log_prob(_prior(), draws)
        assert draws[0].label == "(mu ~ prior)[sample=0]"
        assert scores.label == "log prior(mu)"
        assert scores[0].label == "(log prior(mu))[sample=0]"

    def test_a_batch_of_laws_reads_as_one_law_under_its_label(self):
        laws = DistributionBatch(
            [Normal("effect", float(i), 1.0) for i in range(3)],
            "school",
            label="schools",
        )
        with workflow_run(seed=0):
            assert sample(laws).label == "effect ~ schools"
            assert mean(laws).label == "E[effect ~ schools]"
            scores = log_prob(laws, 0.0)
        assert scores.label == "log schools(effect)"
        assert scores[1].label == "(log schools(effect))[school=1]"
        assert laws[1].notation == "schools[school=1](effect)"
        assert str(laws) == "schools(effect) over school"

    def test_a_field_of_a_draw_takes_its_key(self):
        with workflow_run(seed=0):
            assert sample(_model())["y"].label == "y"

    def test_an_operator_on_values_is_labeled_by_its_expression(self):
        effect = NumericArray(
            jnp.asarray(1.0),
            label="effect",
        )
        assert (2 * effect).label == "2 * effect"
        assert (-(effect + 1.0)).label == "-(effect + 1.0)"
        with workflow_run(seed=0):
            assert (2 * mean(_prior())).label == "2 * E[mu ~ prior]"


class TestTheLabelsOfALiftedFunction:
    """A function lifted over laws is the function applied to draws of its inputs (II.4)."""

    def test_its_law_reads_as_the_function_at_a_draw_of_a_posterior(self):
        @function(output_spec=OutputSpec(damage_probability=None))
        def challenger_damage_probability(beta: jax.Array) -> jax.Array:
            return jax.nn.sigmoid(beta * 31.0)

        with workflow_run(seed=2):
            posterior = condition_on(
                _empirical_model().with_label("oring_model").with_path_names(y="damage", mu="beta"),
                {"damage": 0.5},
            )
            damage_prob = challenger_damage_probability(posterior["beta"])
            summary = mean(damage_prob)
        notation = "challenger_damage_probability(beta ~ oring_model; damage)"
        assert (damage_prob.label, damage_prob.notation) == (
            "challenger_damage_probability",
            notation,
        )
        assert summary.label == f"E[{notation}]"

    def test_inputs_drawn_together_share_one_draw(self):
        @function(output_spec=OutputSpec(f=None))
        def f(a: jax.Array, b: jax.Array) -> jax.Array:
            return a + b

        model = (Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)).with_label("model")
        with workflow_run(seed=0):
            assert f(model["a"], model["b"]).notation == "f((a, b) ~ model)"
            assert f(model["a"], 2.0).notation == "f(a ~ Normal, 2.0)"
            prior = Normal("x", 0.0, 1.0, label="prior")
            assert (
                f(
                    prior,
                    NumericArray(
                        1.0,
                        label="c",
                    ),
                ).notation
                == "f(x ~ prior, c)"
            )

    def test_an_array_argument_appears_by_its_parameters_name(self):
        @function(output_spec=OutputSpec(g=None))
        def g(a: jax.Array, X: jax.Array) -> jax.Array:
            return a

        with workflow_run(seed=0):
            prior = Normal("a", 0.0, 1.0, label="prior")
            assert g(prior, jnp.ones((5, 2))).notation == "g(a ~ prior, X)"

    def test_a_sweep_of_broadcasts_carries_the_lifted_call(self):
        @function(output_spec=OutputSpec(shifted=None))
        def shifted(mu: jax.Array, tau: jax.Array) -> jax.Array:
            return mu + tau

        taus = NumericArrayBatch(
            jnp.array([0.0, 1.0]),
            "tau",
            label="tau",
        )
        with workflow_run(seed=0):
            laws = shifted.with_options(n_broadcast_samples=8)(
                Normal("mu", 0.0, 1.0, label="prior"), taus
            )
            means = mean(laws)
        assert laws.label == "shifted"
        assert laws[1].label == "shifted[tau=1]"
        assert means.label == "E[shifted(mu ~ prior, tau)]"
        assert means[1].label == "E[shifted(mu ~ prior, tau)][tau=1]"

    def test_a_lifted_batch_displays_its_call_and_its_element_its_rows_call(self):
        @function(output_spec=OutputSpec(effect_of=None))
        def effect_of(mu: jax.Array, tau: jax.Array) -> jax.Array:
            return mu + tau

        taus = NumericArrayBatch(
            jnp.arange(1.0, 6.0),
            "tau",
            label="tau",
        )
        with workflow_run(seed=0):
            laws = effect_of.with_options(n_broadcast_samples=8)(
                Normal("mu", 0.0, 1.0, label="prior"), taus
            )
            element = laws[3]
            assert mean(element).label == "E[effect_of(mu ~ prior, 4.0)]"
            assert sample(element).label == "effect_of(mu ~ prior, 4.0)"
        assert str(laws) == "effect_of(mu ~ prior, tau) over tau"
        # The element keeps the label of its position, and displays as its row's call.
        assert (element.label, str(element)) == ("effect_of[tau=3]", "effect_of(mu ~ prior, 4.0)")
        selection = laws[1:3]
        assert selection.label == "effect_of[tau=1:3]"
        assert str(selection) == "effect_of(mu ~ prior, tau)[tau=1:3] over tau"

    def test_a_law_passed_to_a_lifted_call_displays_by_its_notation(self):
        with workflow_run(seed=0):
            lifted = log_prob(
                Normal("g", 0.0, 1.0, label="prior"), Normal("q", 0.0, 1.0, label="proposal")
            )
        assert (lifted.label, lifted.notation) == ("log_prob", "log_prob(prior(g), q ~ proposal)")

    def test_a_lifted_score_keeps_the_component_its_check_declares(self):
        prior = Normal("g", 0.0, 1.0, label="prior")
        proposal = Normal("q", 0.0, 1.0, label="proposal")
        declared = log_prob.check(prior, proposal).result
        with workflow_run(seed=0):
            lifted = log_prob(prior, proposal)
        assert tuple(lifted.event_spec.components) == tuple(declared.components)
        assert tuple(lifted.event_spec.components) == ("log_prob(g)",)

    def test_a_parameter_the_caller_omits_is_not_shown(self):
        @function(output_spec=OutputSpec(scaled_value=None))
        def scaled(x: jax.Array, scale: float = 2.0) -> jax.Array:
            return x * scale

        lift = scaled.with_options(n_broadcast_samples=4)
        with workflow_run(seed=0):
            prior = Normal("mu", 0.0, 1.0, label="prior")
            assert lift(prior).notation == "scaled(mu ~ prior)"

    @pytest.mark.parametrize(
        "scale", [2.0, jnp.float32(2.0)], ids=["the-default-literal", "an-equal-array"]
    )
    def test_a_parameter_the_caller_passes_is_shown_even_at_its_default(self, scale):
        @function(output_spec=OutputSpec(scaled_value=None))
        def scaled(x: jax.Array, scale: float = 2.0) -> jax.Array:
            return x * scale

        lift = scaled.with_options(n_broadcast_samples=4)
        with workflow_run(seed=0):
            prior = Normal("mu", 0.0, 1.0, label="prior")
            assert lift(prior, scale=scale).notation == "scaled(mu ~ prior, 2.0)"
            assert lift(prior, scale).notation == "scaled(mu ~ prior, 2.0)"

    def test_a_function_called_on_values_is_labeled_by_the_call(self):
        @function
        def f(a: jax.Array) -> jax.Array:
            return a + 1

        assert (
            f(
                NumericArray(
                    jnp.asarray(1.0),
                    label="x",
                )
            ).label
            == "f(x)"
        )
