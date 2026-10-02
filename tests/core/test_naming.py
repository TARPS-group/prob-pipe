"""Naming across the tracked terms, the batches, and the operations.

Every tracked term receives its name at construction and preserves it through
structural transforms. Only ``with_label`` replaces it. New operation results
and accessed views receive their names when constructed, across every kind.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
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
    Record,
    RecordBatch,
    RecordSpec,
    log_prob,
    mean,
    prob,
    sample,
    unnormalized_log_prob,
    unnormalized_prob,
    variance,
)
from probpipe.core._specs import NumericRecordSpec
from probpipe.distributions import FactoredDistribution
from probpipe.functions._call import ApplicabilityError
from probpipe.functions._result import _wrap_as_term

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
        "Function": lambda: Function(fn=lambda: 1, name="given"),
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
        assert _named(kind).label == "given"

    @pytest.mark.parametrize("kind", EVERY_KIND)
    def test_name_origin_is_not_part_of_the_public_term(self, kind):
        term = _named(kind)
        assert not hasattr(term, "name_is_auto")
        assert not hasattr(term, "_name_is_auto")
        renamed = term.with_label("replacement")
        assert renamed.label == "replacement"
        assert term.label == "given"
        assert not hasattr(renamed, "name_is_auto")

    @pytest.mark.parametrize("kind", [Record, NumericRecord])
    def test_former_metadata_keyword_is_an_ordinary_record_field(self, kind):
        record = kind("flags", name_is_auto=True)
        assert tuple(record) == ("name_is_auto",)
        assert bool(record["name_is_auto"])
        assert record.label == "flags"


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

        assert Function(name="predict", fn=predict).label == "predict"


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

        assert element.label == "posterior[draw=1]"

    def test_a_sub_batch_is_named_for_its_slice(self):
        sub = self._batch()[1:3]

        assert sub.label == "posterior[draw=1:3]"

    def test_a_derived_name_builds_on_the_given_one(self):
        """So the lineage reads back to the batch a caller actually named."""
        assert self._batch()[1].label.startswith("posterior")


class TestAnOperationLabelsItsResultByItsLaw:
    """An operation's result takes the label of its primary operand, the law."""

    LAW = Normal("height", 0.0, 1.0)

    @pytest.mark.parametrize(
        "compute",
        [
            lambda d: mean(d),
            lambda d: variance(d),
            lambda d: log_prob(d, jnp.asarray(0.0)),
        ],
        ids=["mean", "variance", "log_prob"],
    )
    def test_a_scalar_law_result_takes_the_laws_label(self, compute):
        assert compute(self.LAW).label == "height"

    def test_a_record_law_draw_takes_the_laws_label(self):
        joint = FactoredDistribution("joint", [Normal("a", 0.0, 1.0)])

        assert sample(joint).label == "joint"

    @pytest.mark.parametrize("sample_shape", [(), (4,)], ids=["single", "batch"])
    def test_draws_take_the_laws_label(self, sample_shape):
        """Both a single draw and a batch cross the same result boundary."""
        given = sample(Normal("height", 0.0, 1.0), sample_shape=sample_shape)

        assert given.label == "height"


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
        result = Function(fn=body, name="myfunc")()

        assert result.label == "myfunc"


class TestLevelsAreNamedForWhatMintsThem:
    """An operation names the level it mints after itself (design V.9)."""

    def test_sample_mints_a_sample_level(self):
        drawn = sample(Normal("height", 0.0, 1.0), sample_shape=(5,))

        assert drawn.level_names == ("sample",)

    def test_a_record_drawing_law_mints_the_same_level(self):
        joint = FactoredDistribution("joint", [Normal("a", 0.0, 1.0)])

        drawn = sample(joint, sample_shape=(5,))

        assert drawn.level_names == ("sample",)

    @pytest.mark.parametrize(
        ("atoms", "expected"),
        [
            pytest.param(jnp.linspace(0.0, 1.0, 5), "NumericArrayBatch", id="numeric-atoms"),
            pytest.param(
                NumericRecordBatch(
                    "rows", {"u": jnp.arange(4.0)}, "row", element_spec=RecordSpec(u=())
                ),
                "NumericRecordBatch",
                id="record-atoms",
            ),
            pytest.param(
                OpaqueBatch("objects", [object() for _ in range(3)], "atom"),
                "OpaqueBatch",
                id="opaque-atoms",
            ),
        ],
    )
    def test_a_law_that_assembles_its_own_draws_still_gets_the_level(self, atoms, expected):
        """The boundary mints the level for every kind of draw.

        These laws lay the draws out themselves, in the batch form of their atoms,
        and name no level.
        """
        from probpipe import EmpiricalDistribution

        drawn = sample(EmpiricalDistribution("atoms", atoms), sample_shape=(3,))

        assert type(drawn).__name__ == expected
        assert (drawn.batch_shape, drawn.level_names) == ((3,), ("sample",))

    def test_a_single_draw_from_such_a_law_is_not_a_batch(self):
        """No sample_shape, no level to mint."""
        from probpipe import EmpiricalDistribution

        drawn = sample(EmpiricalDistribution("atoms", jnp.linspace(0.0, 1.0, 5)))

        assert not isinstance(drawn, NumericArrayBatch)

    def test_the_draws_take_the_laws_label(self):
        from probpipe import EmpiricalDistribution

        drawn = sample(
            EmpiricalDistribution(
                "atoms", OpaqueBatch("objects", [object() for _ in range(3)], "atom")
            ),
            sample_shape=(3,),
        )

        assert drawn.label == "atoms"


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
        params=[log_prob, prob, unnormalized_log_prob, unnormalized_prob], ids=lambda op: op.label
    )
    def density_op(self, request):
        return request.param

    def test_scoring_a_batch_of_draws_keeps_the_sample_level(self, density_op):
        drawn = sample(self.LAW, sample_shape=(3,))

        scored = density_op(self.LAW, drawn)

        assert (scored.batch_shape, scored.level_names) == ((3,), ("sample",))

    def test_the_result_takes_the_laws_label(self, density_op):
        drawn = sample(self.LAW, sample_shape=(3,))

        assert density_op(self.LAW, drawn).label == "height"

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
        scored = density_op(self.LAW, sample(self.LAW))

        assert not isinstance(scored, NumericArrayBatch)

    def test_a_raw_array_of_several_values_does_not_conform(self, density_op):
        """A bare array states no levels, so it is one value of the wrong shape."""
        with pytest.raises(ApplicabilityError, match="does not conform"):
            density_op(self.LAW, jnp.zeros(3))


class TestRawDrawNaming:
    @pytest.mark.parametrize("value", [2.0, {"x": 2.0}], ids=["scalar", "mapping"])
    def test_a_declared_raw_result_takes_the_requested_name(self, value):
        from probpipe import OutputSpec

        template = RecordSpec(x=NumericArraySpec(()))
        declaration = (
            OutputSpec(template) if isinstance(value, dict) else OutputSpec(x=template["x"])
        )
        result = _wrap_as_term(value, "sample", declaration, name="law")
        assert result.label == "law"
        assert result.spec == declaration.spec
        assert float(result["x"] if isinstance(result, Record) else result) == 2.0


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
        return Function(fn=body, name="double", dispatch="sequential", **controls)(v=self._rows())

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

        assert result.label == "double"

    def test_a_declared_aggregate_is_named_the_same_way(self):
        from probpipe import RecordSpec

        result = self._swept(lambda v: {"y": jnp.asarray(v["x"])}, output_spec=RecordSpec(y=()))

        assert result.label == "double"

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
            fn=lambda v: {"y": jnp.asarray(v["x"])}, name="double", dispatch="sequential"
        )(v=grid)

        assert result.label == "double"
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

        assert batch.label == "draw"

    def test_stack_takes_a_better_name_when_offered(self):
        rows = [NumericRecord("draw", a=float(i)) for i in range(3)]

        batch = NumericRecordBatch.stack(rows, level_name="row", name="posterior")

        assert batch.label == "posterior"

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

        assert edited.label == "derived"
