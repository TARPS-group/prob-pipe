"""Naming across the tracked terms, the batches, and the operations.

Every tracked term receives its name at construction and preserves it through
structural transforms. Only ``with_name`` replaces it. New operation results
and accessed views receive their names when constructed, across every kind.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    DistributionArray,
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
    ProductDistribution,
    Record,
    RecordBatch,
    RecordSpec,
)
from probpipe.core._specs import NumericRecordSpec
from probpipe.core._workflow_result import _wrap_as_term
from probpipe.core.ops import (
    log_prob,
    mean,
    prob,
    sample,
    unnormalized_log_prob,
    unnormalized_prob,
    variance,
)

KEY = jax.random.PRNGKey(0)
ELEMENT = NumericRecordSpec(a=())
COLUMNS = {"a": jnp.arange(4.0)}


def _named(kind):
    """One instance of *kind* built with an explicit name."""
    return {
        "Record": lambda: Record("given", a=1.0),
        "NumericRecord": lambda: NumericRecord("given", a=1.0),
        "NumericArray": lambda: NumericArray(
            "given",
            jnp.arange(3.0),
        ),
        "Opaque": lambda: Opaque("given", object()),
        "Function": lambda: Function(func=lambda: 1, name="given"),
        "Normal": lambda: Normal("given", 0.0, 1.0),
        "RecordBatch": lambda: RecordBatch(
            "given",
            COLUMNS,
            "lvl",
            element_spec=ELEMENT,
        ),
        "NumericRecordBatch": lambda: NumericRecordBatch(
            "given",
            COLUMNS,
            "lvl",
            element_spec=ELEMENT,
        ),
        "NumericArrayBatch": lambda: NumericArrayBatch(
            "given",
            jnp.arange(4.0),
            "lvl",
            element_spec=NumericArraySpec(shape=()),
        ),
        "OpaqueBatch": lambda: OpaqueBatch(
            "given",
            [1, 2],
            "lvl",
        ),
        "FunctionBatch": lambda: FunctionBatch(
            "given",
            [lambda: 1],
            "lvl",
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


class TestNamesAreKept:
    """The rule every kind shares, and the one an operation reads before renaming."""

    @pytest.mark.parametrize("kind", EVERY_KIND)
    def test_a_given_name_is_kept_verbatim(self, kind):
        assert _named(kind).name == "given"

    @pytest.mark.parametrize("kind", EVERY_KIND)
    def test_name_origin_is_not_part_of_the_public_term(self, kind):
        term = _named(kind)
        assert not hasattr(term, "name_is_auto")
        assert not hasattr(term, "_name_is_auto")
        renamed = term.with_name("replacement")
        assert renamed.name == "replacement"
        assert term.name == "given"
        assert not hasattr(renamed, "name_is_auto")

    @pytest.mark.parametrize("kind", [Record, NumericRecord])
    def test_former_metadata_keyword_is_an_ordinary_record_field(self, kind):
        record = kind("flags", name_is_auto=True)
        assert tuple(record) == ("name_is_auto",)
        assert bool(record["name_is_auto"])
        assert record.name == "flags"


class TestWhichKindsRequireAName:
    """A name is required where nothing else identifies the value.

    A record has fields and a batch has levels, but neither says *which* record
    or batch this is. Where the class can derive something meaningful — a
    callable's own ``__name__`` — it does at construction.
    """

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda: Record(), id="Record"),
            pytest.param(lambda: NumericRecord(), id="NumericRecord"),
            pytest.param(lambda: Opaque(object()), id="Opaque"),
            pytest.param(
                lambda: NumericArrayBatch(
                    jnp.arange(4.0), "lvl", element_spec=NumericArraySpec(shape=())
                ),
                id="NumericArrayBatch",
            ),
        ],
    )
    def test_a_name_is_required(self, build):
        with pytest.raises(TypeError):
            build()

    def test_a_numeric_array_requires_a_name(self):
        """It carries no fields to describe it, so a class-name default would
        name every array in a pipeline alike."""
        with pytest.raises(TypeError, match="name"):
            NumericArray()

    def test_a_lone_value_is_not_enough_for_a_numeric_array(self):
        """The name comes first, so a single argument is the name and the value
        is what the refusal asks for."""
        with pytest.raises(TypeError, match="value"):
            NumericArray(jnp.arange(3.0))

    def test_a_function_takes_its_callables_name(self):
        def predict():
            return 1.0

        assert Function(func=predict).name == "predict"


class TestADerivedNameSaysSo:
    """A view names itself after the position it selected, and marks it auto."""

    @staticmethod
    def _batch():
        return NumericArrayBatch(
            "posterior",
            jnp.arange(12.0).reshape(4, 3),
            "draw",
            element_spec=NumericArraySpec(shape=(3,)),
        )

    def test_an_element_is_named_for_its_position(self):
        element = self._batch()[1]

        assert element.name == "posterior[draw=1]"

    def test_a_sub_batch_is_named_for_its_slice(self):
        sub = self._batch()[1:3]

        assert sub.name == "posterior[draw=1:3]"

    def test_a_derived_name_builds_on_the_given_one(self):
        """So the lineage reads back to the batch a caller actually named."""
        assert self._batch()[1].name.startswith("posterior")


class TestAnOperationNamesItsResult:
    """Sampling retains supplied names; summaries and densities derive theirs."""

    LAW = Normal("height", 0.0, 1.0)

    @pytest.mark.parametrize(
        ("label", "compute"),
        [
            ("mean", lambda d: mean(d)),
            ("variance", lambda d: variance(d)),
            ("log_prob", lambda d: log_prob(d, value=jnp.asarray(0.0))),
        ],
    )
    def test_a_scalar_law_result_is_named_for_the_operation(self, label, compute):
        result = compute(self.LAW)

        assert result.name == label

    def test_a_record_law_result_is_named_for_the_law(self):
        """An already tracked draw retains the name its producer set."""
        joint = ProductDistribution(a=Normal("a", 0.0, 1.0), name="joint")

        drawn = sample(joint, key=KEY)

        assert drawn.name == "joint"

    @pytest.mark.parametrize("sample_shape", [(), (4,)], ids=["single", "batch"])
    def test_draws_take_the_laws_name(self, sample_shape):
        """Raw draws are named for the law, so it is a caller's statement
        exactly when the caller's name for the law was one."""
        given = sample(Normal("height", 0.0, 1.0), sample_shape=sample_shape, key=KEY)

        assert given.name == "height"


class TestTheOutputBoundaryNamesEveryKindAlike:
    """Whatever kind a body returns, the result takes the function's name."""

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
    def test_the_result_takes_the_functions_name(self, label, body):
        result = Function(func=body, name="myfunc")()

        assert result.name == "myfunc"


class TestLevelsAreNamedForWhatMintsThem:
    """An operation names the level it mints after itself (design V.9)."""

    def test_sample_mints_a_sample_level(self):
        drawn = sample(Normal("height", 0.0, 1.0), sample_shape=(5,), key=KEY)

        assert drawn.level_names == ("sample",)

    def test_a_record_drawing_law_mints_the_same_level(self):
        joint = ProductDistribution(a=Normal("a", 0.0, 1.0), name="joint")

        drawn = sample(joint, sample_shape=(5,), key=KEY)

        assert drawn.level_names == ("sample",)

    def test_a_sweep_mints_the_level_it_swept(self):
        """A returned sequence ranges over nothing the call named, so the level
        takes the function's own name."""
        result = Function(func=lambda: [1.0, 2.0], name="myfunc")()

        assert result.level_names == ("myfunc",)

    @pytest.mark.parametrize(
        ("atoms", "expected"),
        [
            pytest.param(jnp.linspace(0.0, 1.0, 5), "NumericRecordBatch", id="numeric-atoms"),
            pytest.param(
                [Record("a", {"u": jnp.asarray(float(i))}) for i in range(4)],
                "NumericRecordBatch",
                id="record-atoms",
            ),
            pytest.param([object() for _ in range(3)], "OpaqueBatch", id="opaque-atoms"),
        ],
    )
    def test_a_law_that_assembles_its_own_draws_still_gets_the_level(self, atoms, expected):
        """The boundary mints the level for every kind of draw.

        These laws lay the draws out themselves — as record columns, or as an array
        of stored objects — and named nothing. The draws came back as one value:
        a record whose fields had grown an axis, or a single opaque object holding
        the whole array.
        """
        from probpipe import EmpiricalDistribution

        drawn = sample(EmpiricalDistribution("atoms", atoms), sample_shape=(3,), key=KEY)

        assert type(drawn).__name__ == expected
        assert (drawn.batch_shape, drawn.level_names) == ((3,), ("sample",))

    def test_a_single_draw_from_such_a_law_is_not_a_batch(self):
        """No sample_shape, no level to mint."""
        from probpipe import EmpiricalDistribution

        drawn = sample(EmpiricalDistribution("atoms", jnp.linspace(0.0, 1.0, 5)), key=KEY)

        assert not isinstance(drawn, NumericRecordBatch)

    def test_the_draws_keep_the_law_s_name_and_whether_it_was_given(self):
        from probpipe import EmpiricalDistribution

        drawn = sample(
            EmpiricalDistribution("atoms", [object() for _ in range(3)]),
            sample_shape=(3,),
            key=KEY,
        )

        assert drawn.name == "atoms"


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

    LAW = Normal("height", 0.0, 1.0)

    @pytest.fixture(
        params=[log_prob, prob, unnormalized_log_prob, unnormalized_prob], ids=lambda op: op.name
    )
    def density_op(self, request):
        return request.param

    def test_scoring_a_batch_of_draws_keeps_the_sample_level(self, density_op):
        drawn = sample(self.LAW, sample_shape=(3,), key=KEY)

        scored = density_op(self.LAW, drawn)

        assert (scored.batch_shape, scored.level_names) == ((3,), ("sample",))

    def test_the_result_is_named_for_the_operand_it_scored(self, density_op):
        drawn = sample(self.LAW, sample_shape=(3,), key=KEY)

        assert density_op(self.LAW, drawn).name == drawn.name

    def test_several_levels_are_all_restated(self, density_op):
        """The operand's own tiling, not one flat axis."""
        drawn = NumericArrayBatch(
            "draws",
            jnp.zeros((2, 3)),
            ("chain", "draw"),
            element_spec=NumericArraySpec(()),
            axes_per_level=(1, 1),
        )

        scored = density_op(self.LAW, drawn)

        assert (scored.batch_shape, scored.level_names) == ((2, 3), ("chain", "draw"))

    def test_a_single_draw_is_still_a_single_value(self, density_op):
        """No operand levels to restate, so nothing is invented."""
        scored = density_op(self.LAW, sample(self.LAW, key=KEY))

        assert not isinstance(scored, NumericArrayBatch)

    def test_a_raw_array_operand_is_left_alone(self, density_op):
        """A bare array states no levels, so the result carries none."""
        scored = density_op(self.LAW, jnp.zeros(3))

        assert not isinstance(scored, NumericArrayBatch)


class TestRawDrawNaming:
    @pytest.mark.parametrize(
        "make, kind, levels",
        [
            pytest.param(lambda: 2.0, NumericArray, None, id="numeric"),
            pytest.param(lambda: {"x": 2.0}, Record, None, id="mapping"),
            pytest.param(lambda: "tag", Opaque, None, id="opaque"),
            pytest.param(lambda: lambda: 2.0, Function, None, id="callable"),
            pytest.param(lambda: [], OpaqueBatch, ("sample",), id="empty-list"),
            pytest.param(lambda: (), OpaqueBatch, ("sample",), id="empty-tuple"),
            pytest.param(lambda: [1.0, 2.0], NumericArrayBatch, ("sample",), id="numeric-list"),
            pytest.param(lambda: ("a", "b"), OpaqueBatch, ("sample",), id="opaque-tuple"),
            pytest.param(lambda: [lambda: 1.0], FunctionBatch, ("sample",), id="callable-list"),
            pytest.param(
                lambda: [{"x": 1.0}, {"x": 2.0}],
                NumericRecordBatch,
                ("sample",),
                id="numeric-record-list",
            ),
            pytest.param(
                lambda: [{"x": "a"}, {"x": "b"}], RecordBatch, ("sample",), id="record-list"
            ),
            pytest.param(
                lambda: [_named("NumericArrayBatch")],
                NumericArrayBatch,
                ("sample", "lvl"),
                id="numeric-batch-list",
            ),
            pytest.param(
                lambda: [_named("OpaqueBatch")],
                OpaqueBatch,
                ("sample", "lvl"),
                id="opaque-batch-list",
            ),
            pytest.param(
                lambda: [_named("NumericRecordBatch")],
                NumericRecordBatch,
                ("sample", "lvl"),
                id="record-batch-list",
            ),
            pytest.param(
                lambda: [Normal("component", 0.0, 1.0)],
                DistributionArray,
                None,
                id="distribution-list",
            ),
        ],
    )
    def test_a_raw_draw_takes_the_laws_name_without_renaming_levels(self, make, kind, levels):
        class Sampler:
            name = "law"
            _sampling_cost = "low"
            _preferred_orchestration = None

            def _sample(self, key, sample_shape=()):
                return make()

        law = Sampler()
        result = sample(law, key=KEY)

        assert isinstance(result, kind)
        assert result.name == "law"
        assert result.provenance is not None
        if levels is not None:
            assert result.level_names == levels

    @pytest.mark.parametrize("value", [2.0, {"x": 2.0}], ids=["scalar", "mapping"])
    def test_a_declared_raw_result_takes_the_requested_name(self, value):
        template = RecordSpec(x=NumericArraySpec(()))
        result = _wrap_as_term(value, "sample", template, name="law")

        assert isinstance(result, Record)
        assert result.name == "law"
        assert result.event_template == template
        assert float(result["x"]) == 2.0

    def test_a_raw_draws_name_is_validated_by_its_constructor(self):
        class Sampler:
            name = ""
            _sampling_cost = "low"
            _preferred_orchestration = None

            def _sample(self, key, sample_shape=()):
                return 2.0

        with pytest.raises(TypeError, match=r"NumericArray\.__init__ must set a non-empty name"):
            sample(Sampler(), key=KEY)


class TestEveryAggregateIsNamedForItsFunction:
    """The naming table, widened across the axes that had diverged.

    A sweep's aggregate is built by the boundary, not by a caller, so its name is
    the producing function's. Three paths disagreed: the
    undeclared record aggregate took `stack`'s class-name default, and the scalar,
    opaque, and declared paths marked a derived name as user-given — which would
    stop a later operation renaming it.
    """

    @staticmethod
    def _rows(n: int = 3):
        from probpipe.core._specs import NumericRecordSpec

        return NumericRecordBatch(
            "rows",
            {"x": jnp.arange(float(n))},
            "row",
            element_spec=NumericRecordSpec(x=()),
        )

    def _swept(self, body, **controls):
        return Function(func=body, name="double", dispatch="sequential", **controls)(v=self._rows())

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
    def test_an_undeclared_aggregate_is_named_for_the_function(self, label, body):
        result = self._swept(body)

        assert result.name == "double"

    def test_a_declared_aggregate_is_named_the_same_way(self):
        from probpipe import RecordSpec

        result = self._swept(lambda v: {"y": jnp.asarray(v["x"])}, output_template=RecordSpec(y=()))

        assert result.name == "double"

    def test_a_multi_axis_sweep_is_named_the_same_way(self):
        """The re-cut to the sweep's own geometry is a separate construction, and
        it had its own naming."""
        from probpipe.core._specs import NumericRecordSpec

        grid = NumericRecordBatch(
            "grid",
            {"x": jnp.arange(6.0).reshape(2, 3)},
            ("a", "b"),
            element_spec=NumericRecordSpec(x=()),
        )

        result = Function(
            func=lambda v: {"y": jnp.asarray(v["x"])}, name="double", dispatch="sequential"
        )(v=grid)

        assert result.name == "double"
        assert result.level_names == ("a", "b")


class TestNoKindInventsAName:
    """The rule the whole layer now shares: a name is given, or derived from
    something that carries meaning. A class name carries none.

    Every batch defaulted to its own lowercased class name, so a pipeline full
    of them read `recordbatch`, `opaquebatch`, `numericrecordbatch` — names that
    say what the object *is*, which its type already says, and nothing about
    which one it is.

    That every constructor takes the name first, positional-only and with no
    default behind it, is asserted from the signatures themselves in
    `test_batch.py`'s `TestTheConstructorSignatureContract`. What is left here is
    the other half of the rule: where a *derived* name comes from.
    """

    def test_stack_derives_its_name_from_what_it_stacks(self):
        """Derived from real content, so no call site has to invent one: a batch
        of `draw` records is about `draw`."""
        rows = [NumericRecord("draw", a=float(i)) for i in range(3)]

        batch = NumericRecordBatch.stack(rows, level_name="row")

        assert batch.name == "draw"

    def test_stack_takes_a_better_name_when_offered(self):
        rows = [NumericRecord("draw", a=float(i)) for i in range(3)]

        batch = NumericRecordBatch.stack(rows, level_name="row", name="posterior")

        assert batch.name == "posterior"

    def test_a_structural_transform_preserves_the_name(self):
        """There is no class-name default to re-derive from, and an auto name is
        something derived rather than a placeholder."""
        batch = NumericRecordBatch(
            "derived",
            {"a": jnp.zeros(3), "b": jnp.zeros(3)},
            "lvl",
            element_spec=NumericRecordSpec(a=(), b=()),
        )

        edited = batch.without("b")

        assert edited.name == "derived"
