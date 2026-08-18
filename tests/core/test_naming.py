"""Naming across the tracked terms, the batches, and the operations.

Every tracked term carries a name and a flag saying whether the name was given
by a caller or derived for it. The flag is what tells a later operation whether
it may rename: a derived name is a placeholder, a given one is a statement. The
rules are spread over the classes that implement them, so they are pinned here
in one place, where a divergence between two kinds is visible as two rows of the
same table rather than as two files that never meet.

The invariant that ties them together: **a name a caller did not give is marked
auto, and a name a caller gave is never overwritten.**
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
    ProductDistribution,
    Record,
    RecordBatch,
)
from probpipe.core.event_template import NumericEventTemplate
from probpipe.core.ops import log_prob, mean, sample, variance

KEY = jax.random.PRNGKey(0)
ELEMENT = NumericEventTemplate(a=())
COLUMNS = {"a": jnp.arange(4.0)}


def _named(kind):
    """One instance of *kind* built with an explicit name."""
    return {
        "Record": lambda: Record("given", a=1.0),
        "NumericRecord": lambda: NumericRecord("given", a=1.0),
        "NumericArray": lambda: NumericArray(jnp.arange(3.0), name="given"),
        "Opaque": lambda: Opaque("given", object()),
        "Function": lambda: Function(func=lambda: 1, name="given"),
        "Normal": lambda: Normal(0.0, 1.0, name="given"),
        "RecordBatch": lambda: RecordBatch(COLUMNS, "lvl", element_spec=ELEMENT, name="given"),
        "NumericRecordBatch": lambda: NumericRecordBatch(
            COLUMNS, "lvl", element_spec=ELEMENT, name="given"
        ),
        "NumericArrayBatch": lambda: NumericArrayBatch(
            jnp.arange(4.0), "lvl", element_spec=NumericArraySpec(shape=()), name="given"
        ),
        "OpaqueBatch": lambda: OpaqueBatch([1, 2], "lvl", name="given"),
        "FunctionBatch": lambda: FunctionBatch([lambda: 1], "lvl", name="given"),
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


class TestAGivenNameIsKeptAndMarkedGiven:
    """The rule every kind shares, and the one an operation reads before renaming."""

    @pytest.mark.parametrize("kind", EVERY_KIND)
    def test_a_given_name_is_kept_verbatim(self, kind):
        assert _named(kind).name == "given"

    @pytest.mark.parametrize("kind", EVERY_KIND)
    def test_a_given_name_is_not_marked_auto(self, kind):
        assert _named(kind).name_is_auto is False


class TestWhichKindsRequireAName:
    """A name is required where nothing else identifies the value.

    A record has fields and a batch has levels, but neither says *which* record
    or batch this is. Where the class can derive something meaningful — a
    callable's own ``__name__`` — it does, and marks it auto.
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
            NumericArray(jnp.arange(3.0))

    def test_a_function_takes_its_callables_name(self):
        def predict():
            return 1.0

        assert (Function(func=predict).name, Function(func=predict).name_is_auto) == (
            "predict",
            True,
        )

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(
                lambda: RecordBatch(COLUMNS, "lvl", element_spec=ELEMENT),
                "recordbatch",
                id="RecordBatch",
            ),
            pytest.param(
                lambda: NumericRecordBatch(COLUMNS, "lvl", element_spec=ELEMENT),
                "numericrecordbatch",
                id="NumericRecordBatch",
            ),
            pytest.param(lambda: OpaqueBatch([1, 2], "lvl"), "opaquebatch", id="OpaqueBatch"),
            pytest.param(
                lambda: FunctionBatch([lambda: 1], "lvl"), "functionbatch", id="FunctionBatch"
            ),
        ],
    )
    def test_the_other_batches_fall_back_to_their_class_name(self, build, expected):
        """Recorded rather than endorsed: `NumericArrayBatch` requires a name, and
        these do not. Aligning them is tracked separately."""
        batch = build()

        assert (batch.name, batch.name_is_auto) == (expected, True)


class TestADerivedNameSaysSo:
    """A view names itself after the position it selected, and marks it auto."""

    @staticmethod
    def _batch():
        return NumericArrayBatch(
            jnp.arange(12.0).reshape(4, 3),
            "draw",
            element_spec=NumericArraySpec(shape=(3,)),
            name="posterior",
        )

    def test_an_element_is_named_for_its_position(self):
        element = self._batch()[1]

        assert (element.name, element.name_is_auto) == ("posterior[draw=1]", True)

    def test_a_sub_batch_is_named_for_its_slice(self):
        sub = self._batch()[1:3]

        assert (sub.name, sub.name_is_auto) == ("posterior[draw=1:3]", True)

    def test_a_derived_name_builds_on_the_given_one(self):
        """So the lineage reads back to the batch a caller actually named."""
        assert self._batch()[1].name.startswith("posterior")


class TestAnOperationNamesItsResult:
    """What an operation hands back is named, and marked auto — a caller named
    the *inputs*, not this."""

    LAW = Normal(0.0, 1.0, name="height")

    @pytest.mark.parametrize(
        ("label", "compute"),
        [
            ("sample", lambda d: sample(d, key=KEY)),
            ("mean", lambda d: mean(d)),
            ("variance", lambda d: variance(d)),
            ("log_prob", lambda d: log_prob(d, value=jnp.asarray(0.0))),
        ],
    )
    def test_a_scalar_law_result_is_named_for_the_operation(self, label, compute):
        result = compute(self.LAW)

        assert (result.name, result.name_is_auto) == (label, True)

    def test_a_record_law_result_is_named_for_the_law(self):
        """Recorded rather than endorsed.

        A record-drawing law builds its own value, already named for itself, and
        the output boundary keeps a tracked term as it is. So the same operation
        names its result for the operation over a scalar law and for the law over
        a record-valued one.
        """
        joint = ProductDistribution(a=Normal(0.0, 1.0, name="a"), name="joint")

        drawn = sample(joint, key=KEY)

        assert (drawn.name, drawn.name_is_auto) == ("joint", True)

    def test_a_name_the_operation_invented_is_marked_auto(self):
        """The load-bearing half: an invented name is a placeholder, so a later
        operation may replace it without discarding a caller's statement."""
        for compute in (
            lambda d: sample(d, key=KEY),
            lambda d: mean(d),
            lambda d: variance(d),
        ):
            assert compute(self.LAW).name_is_auto is True

    def test_a_name_taken_from_the_law_carries_the_laws_flag(self):
        """A batch of draws is named for the law, so it is a caller's statement
        exactly when the caller's name for the law was one."""
        given = sample(Normal(0.0, 1.0, name="height"), sample_shape=(4,), key=KEY)

        assert (given.name, given.name_is_auto) == ("height", False)


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

        assert (result.name, result.name_is_auto) == ("myfunc", True)


class TestLevelsAreNamedForWhatMintsThem:
    """An operation names the level it mints after itself (design V.9)."""

    def test_sample_mints_a_sample_level(self):
        drawn = sample(Normal(0.0, 1.0, name="height"), sample_shape=(5,), key=KEY)

        assert drawn.level_names == ("sample",)

    def test_a_record_drawing_law_mints_the_same_level(self):
        joint = ProductDistribution(a=Normal(0.0, 1.0, name="a"), name="joint")

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

        drawn = sample(EmpiricalDistribution(atoms, name="atoms"), sample_shape=(3,), key=KEY)

        assert type(drawn).__name__ == expected
        assert (drawn.batch_shape, drawn.level_names) == ((3,), ("sample",))

    def test_a_single_draw_from_such_a_law_is_not_a_batch(self):
        """No sample_shape, no level to mint."""
        from probpipe import EmpiricalDistribution

        drawn = sample(EmpiricalDistribution(jnp.linspace(0.0, 1.0, 5), name="atoms"), key=KEY)

        assert not isinstance(drawn, NumericRecordBatch)

    def test_the_draws_keep_the_law_s_name_and_whether_it_was_given(self):
        from probpipe import EmpiricalDistribution

        drawn = sample(
            EmpiricalDistribution([object() for _ in range(3)], name="atoms"),
            sample_shape=(3,),
            key=KEY,
        )

        assert (drawn.name, drawn.name_is_auto) == ("atoms", False)

class TestABatchOperandKeepsItsLevelsThroughAnOperation:
    """Design V.9: `log_prob` maps elementwise "with the batch axes preserved".

    An operation whose value parameter is `Any`-hinted takes the batch whole and
    evaluates it in one vectorized call — the fused implementation V.9 allows.
    That is not licence to hand back a bare array: the axes the operand accounted
    for are levels, and a result that drops them says the draws were one value.
    """

    LAW = Normal(0.0, 1.0, name="height")

    def test_log_prob_of_a_batch_of_draws_keeps_the_sample_level(self):
        drawn = sample(self.LAW, sample_shape=(3,), key=KEY)

        scored = log_prob(self.LAW, drawn)

        assert (scored.batch_shape, scored.level_names) == ((3,), ("sample",))

    def test_the_result_is_named_for_the_operand_it_scored(self):
        drawn = sample(self.LAW, sample_shape=(3,), key=KEY)

        assert log_prob(self.LAW, drawn).name == drawn.name

    def test_a_single_draw_is_still_a_single_value(self):
        """No operand levels to restate, so nothing is invented."""
        scored = log_prob(self.LAW, sample(self.LAW, key=KEY))

        assert not isinstance(scored, NumericArrayBatch)

    def test_a_raw_array_operand_is_left_alone(self):
        """A bare array states no levels, so the result carries none."""
        scored = log_prob(self.LAW, jnp.zeros(3))

        assert not isinstance(scored, NumericArrayBatch)
