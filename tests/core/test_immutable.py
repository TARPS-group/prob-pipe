"""The ``Immutable`` mixin: the assignment guard and the state round-trip.

The classes that mix it in are covered by their own suites; these tests pin the
mixin's contract directly, including the storage forms a hand-written round-trip
got wrong before it was shared.
"""

import copy
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import NumericRecord, NumericRecordBatch, OpaqueSpec, Record, RecordBatch, function
from probpipe.core._immutable import Immutable
from probpipe.core._specs import RecordSpec

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


class TestAMemoIsRebuiltAfterARoundTrip:
    """A transient memo is left unset, so whatever reads it must rebuild it.

    ``NumericRecord``'s conversion cache is the one in the tree. Only a *native*
    leaf reaches it — a ``jax`` leaf is returned without conversion — so these
    use numpy and xarray leaves.
    """

    @ROUND_TRIPS
    def test_a_numpy_leaf_still_converts(self, operation):
        record = NumericRecord(
            {"x": np.array([1.0, 2.0])},
            label="nr",
        )
        assert operation(record).to_vector().tolist() == [1.0, 2.0]

    @ROUND_TRIPS
    def test_a_converted_leaf_reconverts(self, operation):
        # Converting first populates the memo on the original, which the copy
        # must not depend on.
        record = NumericRecord(
            {"x": np.array([1.0, 2.0])},
            label="nr",
        )
        record.to_vector()
        assert operation(record).to_vector().tolist() == [1.0, 2.0]

    @ROUND_TRIPS
    def test_a_native_container_leaf_still_converts(self, operation):
        xr = pytest.importorskip("xarray")
        leaf = xr.DataArray([1.0, 2.0], dims=["t"], coords={"t": [10, 20]})
        record = NumericRecord(
            {"x": leaf},
            label="nr",
        )
        assert operation(record).to_vector().tolist() == [1.0, 2.0]
        # The native leaf itself survives; only the converted form is rebuilt.
        assert operation(record).raw("x").dims == ("t",)


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
            pytest.param(
                lambda: Record(
                    {"x": jnp.ones(2), "tag": "m"},
                    label="r",
                ),
                id="record",
            ),
            pytest.param(
                lambda: NumericRecord(
                    {"x": jnp.ones(2)},
                    label="nr",
                ),
                id="numeric-record",
            ),
            pytest.param(lambda: RecordSpec(x=(2,), tag=OpaqueSpec()), id="event-template"),
            pytest.param(
                lambda: RecordBatch.stack(
                    [
                        Record(
                            {"x": jnp.ones(2), "tag": "m"},
                            label="r",
                        )
                    ]
                    * 2,
                    level_name="draw",
                ),
                id="record-batch",
            ),
            pytest.param(
                lambda: NumericRecordBatch.stack(
                    [
                        NumericRecord(
                            {"x": jnp.ones(2)},
                            label="nr",
                        )
                    ]
                    * 2,
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


class TestEveryTrackedTermIsImmutable:
    """The guarantee, over every tracked term the package defines.

    Discovered rather than listed: a term added later is covered without anyone
    remembering to add it here, which is what keeps the rule from decaying back
    into the per-class opt-in it replaced.
    """

    @staticmethod
    def every_tracked_class() -> list[type]:
        import importlib
        import pkgutil

        import probpipe
        from probpipe.core.tracked import TrackedTerm

        for module in pkgutil.walk_packages(probpipe.__path__, "probpipe."):
            try:
                importlib.import_module(module.name)
            except ImportError as exc:
                # An optional backend that is not installed is skipped; a
                # failure inside probpipe is a bug the sweep must report.
                if (exc.name or "").partition(".")[0] == "probpipe":
                    raise

        seen: set[type] = set()

        def collect(cls: type) -> None:
            for subclass in cls.__subclasses__():
                if subclass not in seen:
                    seen.add(subclass)
                    collect(subclass)

        collect(TrackedTerm)
        return sorted(seen, key=lambda c: c.__name__)

    def test_the_mixin_is_in_every_tracked_term_s_mro(self):
        offenders = [c.__name__ for c in self.every_tracked_class() if not issubclass(c, Immutable)]
        assert offenders == []

    def test_no_tracked_class_overrides_the_guard(self):
        # A class defining its own ``__setattr__`` or ``__delattr__`` is back to
        # the per-class rule, and free to disagree with the message or the
        # exception. This is what stops one from appearing quietly.
        overriding = [
            c.__name__
            for c in self.every_tracked_class()
            if "__setattr__" in c.__dict__ or "__delattr__" in c.__dict__
        ]
        assert overriding == []

    def test_a_distribution_refuses_assignment_and_deletion(self):
        from probpipe import Normal

        term = Normal("x", 0.0, 1.0)
        with pytest.raises(AttributeError, match="Normal is immutable"):
            term.fitted = True
        with pytest.raises(AttributeError, match="Normal is immutable"):
            del term._label

    def test_a_record_refuses_assignment_and_deletion(self):
        term = Record(
            {"x": jnp.ones(2)},
            label="r",
        )
        with pytest.raises(AttributeError, match="Record is immutable"):
            term.attribute = 1
        with pytest.raises(AttributeError, match="Record is immutable"):
            del term._label

    def test_a_record_batch_refuses_assignment_and_names_itself(self):
        term = RecordBatch.stack(
            [
                Record(
                    {"x": jnp.ones(2)},
                    label="r",
                )
            ]
            * 2,
            level_name="draw",
        )
        with pytest.raises(AttributeError, match="RecordBatch is immutable"):
            term.attribute = 1


class TestTheConstructionWindow:
    def test_it_closes_when_the_constructor_returns(self):
        term = Record(
            {"x": jnp.ones(2)},
            label="r",
        )
        with pytest.raises(AttributeError):
            term.attribute = 1

    def test_an_init_that_returns_a_value_is_refused(self):
        # ``type.__call__`` raises on this, and splitting it must not lose that.
        from probpipe.core.tracked import TrackedTerm

        class Returning(TrackedTerm):
            def __init__(self):
                self._init_tracked("t")
                return "oops"

            def raw(self):
                return None

        with pytest.raises(TypeError, match="should return None"):
            Returning()

    def test_it_closes_when_a_constructor_raises(self):
        # Through the metaclass rather than by opening the window by hand: what
        # is under test is that ``_TrackedTermMeta.__call__`` closes the window
        # on the way out of a failing ``__init__``. The half-built object is
        # kept from ``__new__``, since the failed call returns nothing.
        from probpipe.core._immutable import _constructing_now
        from probpipe.core.tracked import TrackedTerm

        built = []

        class Failing(TrackedTerm):
            def __new__(cls):
                instance = super().__new__(cls)
                built.append(instance)
                return instance

            def __init__(self):
                self._init_tracked("failing")
                self.partial = 1
                raise ValueError("no")

            def raw(self):
                return None

        with pytest.raises(ValueError, match="no"):
            Failing()

        (half_built,) = built
        assert half_built.partial == 1  # the constructor got that far
        assert _constructing_now() == {}
        with pytest.raises(AttributeError, match="Failing is immutable"):
            half_built.partial = 2

    def test_it_covers_one_instance_and_not_another(self):
        from probpipe.core._immutable import constructing

        inner = object.__new__(Slotted)
        outer = object.__new__(Slotted)
        with constructing(outer):
            outer.left = 1  # the window is open on this one
            with pytest.raises(AttributeError, match="Slotted is immutable"):
                inner.left = 1  # and on this one it is not

    def test_a_window_on_one_instance_nests(self):
        # An inner block must leave the outer one open: what closes a window is
        # the last exit, not the first. Reached whenever a constructor that
        # already runs in a window opens one on itself — through a helper that
        # allocates and initializes, say.
        from probpipe.core._immutable import _constructing_now, constructing

        instance = object.__new__(Slotted)
        with constructing(instance):
            with constructing(instance):
                instance.left = 1
            instance.right = 2  # still inside the outer window
        assert (instance.left, instance.right) == (1, 2)
        assert _constructing_now() == {}
        with pytest.raises(AttributeError, match="Slotted is immutable"):
            instance.left = 3

    def test_constructing_a_term_inside_another_leaves_both_correct(self):
        from probpipe import Normal
        from probpipe.distributions import FactoredDistribution

        # Different instances rather than one nested in itself: the factors
        # are built first, and the joint's own window is unaffected by theirs.
        joint = FactoredDistribution(
            [Normal("x", 0.0, 1.0, label="a")],
            label="j",
        )
        assert joint.label == "j"
        assert joint.factors[0].label == "a"
        with pytest.raises(AttributeError, match="is immutable"):
            joint.attribute = 1
        with pytest.raises(AttributeError, match="Normal is immutable"):
            joint.factors[0].attribute = 1


class TestAClassBuiltAtRuntime:
    """What the round-trip does for a class that has no importable name.

    Some distribution families build a subclass per capability set, so the class
    an instance reports exists only in memory. ``pickle`` stores a class by name,
    so these families reconstruct through a module-level factory instead. These
    pin the behavior so a change to it is deliberate.
    """

    @staticmethod
    def _joint():
        from probpipe import Gamma, Normal

        return Normal("a", 0.0, 1.0) * Gamma("b", 2.0, 1.0)

    @staticmethod
    def _field_view():
        from probpipe import Gamma, Normal

        return (Normal("a", 0.0, 1.0) * Gamma("b", 2.0, 1.0))["a"]

    @pytest.fixture(
        params=[
            pytest.param("_joint", id="joint"),
            pytest.param("_field_view", id="field-view"),
        ]
    )
    def runtime_classed(self, request):
        return getattr(self, request.param)()

    def test_its_class_is_not_importable_by_name(self, runtime_classed):
        import importlib

        cls = type(runtime_classed)
        module = importlib.import_module(cls.__module__)
        assert getattr(module, cls.__qualname__, None) is not cls

    def test_copy_and_deepcopy_still_work(self, runtime_classed):
        # They hold the class object rather than its name.
        assert type(copy.copy(runtime_classed)) is type(runtime_classed)
        assert type(copy.deepcopy(runtime_classed)) is type(runtime_classed)

    def test_cloudpickle_handles_it(self, runtime_classed):
        # It serializes the class by value, which is what the Ray and Prefect
        # paths rely on.
        cloudpickle = pytest.importorskip("cloudpickle")
        restored = pickle.loads(cloudpickle.dumps(runtime_classed))
        assert type(restored).__name__ == type(runtime_classed).__name__

    def test_standard_pickle_rebuilds_it_through_the_factory(self, runtime_classed):
        # The pickle names a module-level rebuild, never the runtime class.
        assert type(pickle.loads(pickle.dumps(runtime_classed))) is type(runtime_classed)
