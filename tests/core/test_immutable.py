"""The ``Immutable`` mixin: the assignment guard and the state round-trip.

The classes that mix it in are covered by their own suites; these tests pin the
mixin's contract directly, including the storage forms a hand-written round-trip
got wrong before it was shared.
"""

import copy
import pickle

import jax.numpy as jnp
import pytest

from probpipe import NumericRecord, NumericRecordBatch, Record, RecordBatch, function
from probpipe.core._immutable import Immutable
from probpipe.core.event_template import EventTemplate

# ---------------------------------------------------------------------------
# Hosts declared here rather than reused: the point is the storage forms, and
# each is a shape a real host takes. They are module-level so pickle can find
# them by name.
# ---------------------------------------------------------------------------


class Slotted(Immutable):
    __slots__ = ("left", "right")

    def __init__(self, left, right):
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


class OneBareStringSlot(Immutable):
    # A bare string, not a tuple: iterating it yields characters, so a walk over
    # ``__slots__`` clones this host without its storage and raises nothing.
    __slots__ = "store"

    def __init__(self, store):
        object.__setattr__(self, "store", store)


class DictCarrying(Immutable):
    # No ``__slots__``, so the state lives in an instance dictionary that no walk
    # over ``__slots__`` would find.
    def __init__(self, value):
        object.__setattr__(self, "value", value)


class SlottedWithDictSubclass(Slotted):
    def __init__(self, left, right, extra):
        super().__init__(left, right)
        object.__setattr__(self, "extra", extra)


class WithMemo(Immutable):
    __slots__ = ("_memo", "value")
    _transient_state = ("_memo",)

    def __init__(self, value):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "_memo", {"expensive": True})


class WithStore(Immutable):
    __slots__ = ("store",)
    _decoupled_state = ("store",)

    def __init__(self, store):
        object.__setattr__(self, "store", store)


ROUND_TRIPS = pytest.mark.parametrize(
    "operation",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda o: pickle.loads(pickle.dumps(o)), id="pickle"),
    ],
)


class TestTheGuard:
    def test_assignment_names_the_class_the_caller_touched(self):
        with pytest.raises(AttributeError, match="Slotted is immutable"):
            Slotted(1, 2).left = 3

    def test_a_subclass_reports_itself_not_its_base(self):
        # The message a hardcoded class name got wrong for every subclass.
        with pytest.raises(AttributeError, match="SlottedWithDictSubclass is immutable"):
            SlottedWithDictSubclass(1, 2, 3).left = 4

    def test_deletion_raises_too(self):
        with pytest.raises(AttributeError, match="Slotted is immutable"):
            del Slotted(1, 2).left

    def test_a_new_attribute_is_refused_as_well(self):
        with pytest.raises(AttributeError, match="DictCarrying is immutable"):
            DictCarrying(1).added = 2


class TestEveryStorageFormRoundTrips:
    """The three ways a hand-written round-trip lost state in silence."""

    @ROUND_TRIPS
    def test_slots(self, operation):
        restored = operation(Slotted(1, [2]))
        assert (restored.left, restored.right) == (1, [2])

    @ROUND_TRIPS
    def test_a_bare_string_slots_declaration(self, operation):
        assert operation(OneBareStringSlot([1, 2])).store == [1, 2]

    @ROUND_TRIPS
    def test_an_instance_dictionary(self, operation):
        assert operation(DictCarrying("v")).value == "v"

    @ROUND_TRIPS
    def test_a_subclass_that_adds_a_dictionary_to_slots(self, operation):
        restored = operation(SlottedWithDictSubclass(1, 2, "e"))
        assert (restored.left, restored.right, restored.extra) == (1, 2, "e")

    @ROUND_TRIPS
    def test_the_reconstruction_keeps_its_class(self, operation):
        assert type(operation(SlottedWithDictSubclass(1, 2, 3))) is SlottedWithDictSubclass

    def test_an_unassigned_slot_stays_unassigned(self):
        partial = object.__new__(Slotted)
        object.__setattr__(partial, "left", 1)
        restored = pickle.loads(pickle.dumps(partial))
        assert restored.left == 1
        assert not hasattr(restored, "right")


class TestTransientState:
    @ROUND_TRIPS
    def test_a_memo_is_not_carried(self, operation):
        assert not hasattr(operation(WithMemo(1)), "_memo")

    @ROUND_TRIPS
    def test_the_rest_of_the_state_still_is(self, operation):
        assert operation(WithMemo(7)).value == 7

    def test_a_subclass_inherits_the_declaration(self):
        class Sub(WithMemo):
            __slots__ = ()

        # Copied rather than pickled: a class defined in a function body has no
        # importable name, which is a limit of ``pickle`` and not of the mixin.
        assert not hasattr(copy.copy(Sub(1)), "_memo")


class TestDecoupledState:
    @ROUND_TRIPS
    def test_the_store_survives(self, operation):
        assert operation(WithStore({"k": 1})).store == {"k": 1}

    def test_a_write_on_the_copy_does_not_reach_the_original(self):
        original = WithStore({"k": 1})
        clone = copy.copy(original)
        clone.store["added"] = 2
        assert "added" not in original.store


class TestTheHostsInTheTree:
    """The mixin's real hosts: each reports itself, and round-trips."""

    @pytest.fixture(
        params=[
            pytest.param(lambda: Record("r", {"x": jnp.ones(2), "tag": "m"}), id="record"),
            pytest.param(lambda: NumericRecord("nr", {"x": jnp.ones(2)}), id="numeric-record"),
            pytest.param(lambda: EventTemplate(x=(2,), tag=None), id="event-template"),
            pytest.param(
                lambda: RecordBatch.stack(
                    [Record("r", {"x": jnp.ones(2), "tag": "m"}, name_is_auto=True)] * 2,
                    level_name="draw",
                ),
                id="record-batch",
            ),
            pytest.param(
                lambda: NumericRecordBatch.stack(
                    [NumericRecord("nr", {"x": jnp.ones(2)}, name_is_auto=True)] * 2,
                    level_name="draw",
                ),
                id="numeric-record-batch",
            ),
        ]
    )
    def term(self, request):
        return request.param()

    def test_it_is_immutable(self, term):
        with pytest.raises(AttributeError, match=f"{type(term).__name__} is immutable"):
            term.attribute = 1

    def test_it_round_trips_through_pickle(self, term):
        restored = pickle.loads(pickle.dumps(term))
        assert type(restored) is type(term)
        assert restored == term

    def test_it_round_trips_through_copy(self, term):
        assert copy.copy(term) == term
        assert copy.deepcopy(term) == term

    def test_a_function_is_immutable_and_names_itself(self):
        @function
        def double(x: float) -> float:
            return x * 2

        with pytest.raises(AttributeError, match="Function is immutable"):
            double.attribute = 1

    def test_a_function_still_constructs_through_its_own_window(self):
        # ``Function``'s constructor assigns normally, so it opens a window over
        # itself rather than writing through ``object.__setattr__``. The window
        # must be shut by the time the caller has the object.
        @function
        def double(x: float) -> float:
            return x * 2

        assert double(2.0) is not None
        assert not getattr(double, "_initializing", False)
