"""Round-trip tests for the terms that reconstruct through their own ``__reduce__``.

These tests ensure that Record, EventTemplate, NumericRecord, RecordBatch,
NumericRecordBatch, and ProductDistribution can survive pickle serialization,
which is required for Ray task distribution (Ray uses cloudpickle to ship
arguments to workers), and that a copy or an unpickle preserves everything the
term was carrying.

The core issue was that Record.__setattr__ raises "Record is immutable", so
pickle's default restore mechanism (create empty instance + __setattr__) failed.
The fix adds __reduce__ to each class, delegating reconstruction to the normal
constructor.
"""

import copy
import pickle

import jax.numpy as jnp
import pytest

from probpipe import (
    Normal,
    NumericRecord,
    NumericRecordBatch,
    ProductDistribution,
    RecordBatch,
)
from probpipe.core._empirical import BootstrapReplicateDistribution, EmpiricalDistribution
from probpipe.core.event_template import (
    ArraySpec,
    EventTemplate,
    NumericEventTemplate,
    OpaqueSpec,
)
from probpipe.core.record import Record

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def roundtrip(obj):
    return pickle.loads(pickle.dumps(obj))


def cloudpickle_roundtrip(obj):
    cloudpickle = pytest.importorskip("cloudpickle")
    return pickle.loads(cloudpickle.dumps(obj))


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def test_record_pickle_roundtrip():
    r = Record("myrecord", x=jnp.array(1.0), y=jnp.array([2.0, 3.0]))
    r2 = roundtrip(r)
    assert r2.name == "myrecord"
    assert r2.fields == ("x", "y")
    assert float(r2["x"]) == pytest.approx(1.0)
    assert list(r2["y"]) == pytest.approx([2.0, 3.0])


def test_record_pickle_auto_name():
    r = Record("r", {"a": jnp.array(1.0), "b": jnp.array(2.0)}, name_is_auto=True)
    r2 = roundtrip(r)
    assert r2.name == r.name
    assert r2.name_is_auto is True
    assert r2.fields == ("a", "b")


def test_record_immutability_after_unpickle():
    r = Record("r", x=jnp.array(1.0))
    r2 = roundtrip(r)
    with pytest.raises(AttributeError, match="immutable"):
        r2.x = jnp.array(99.0)


def test_record_provenance_preserved():
    from probpipe.core.provenance import Provenance

    r = Record("r", x=jnp.array(1.0))
    r.with_provenance(Provenance(operation="test_op", metadata={"k": "v"}))
    assert r._provenance is not None

    r2 = roundtrip(r)
    assert r2._provenance is not None
    assert r2._provenance.operation == "test_op"
    assert r2._provenance.metadata == {"k": "v"}


def test_record_no_provenance_roundtrip():
    r = Record("r", x=jnp.array(1.0))
    assert r._provenance is None
    r2 = roundtrip(r)
    assert r2._provenance is None


# ---------------------------------------------------------------------------
# EventTemplate
# ---------------------------------------------------------------------------


def test_event_template_pickle_roundtrip():
    t = EventTemplate(label=None, x=())
    t2 = roundtrip(t)
    assert type(t2) is EventTemplate
    assert t2.fields == ("label", "x")
    assert t2["label"] == OpaqueSpec()
    assert t2["x"] == ArraySpec(())


def test_numeric_event_template_pickle_roundtrip():
    t = EventTemplate(x=(), y=(3,))
    assert type(t) is NumericEventTemplate
    t2 = roundtrip(t)
    assert type(t2) is NumericEventTemplate
    assert t2.fields == ("x", "y")
    assert t2.vector_size == 4  # () + (3,)


# ---------------------------------------------------------------------------
# NumericRecord
# ---------------------------------------------------------------------------


def test_numeric_record_pickle_roundtrip():
    nr = NumericRecord("nr", r=jnp.array(1.8), K=jnp.array(70.0), phi=jnp.array(10.0))
    nr2 = roundtrip(nr)
    assert nr2.fields == ("r", "K", "phi")
    assert float(nr2["r"]) == pytest.approx(1.8)
    assert nr2.vector_size == 3


def test_numeric_record_vector_size_recomputed():
    nr = NumericRecord("nr", a=jnp.ones((2, 3)))
    assert nr.vector_size == 6
    nr2 = roundtrip(nr)
    assert nr2.vector_size == 6


def test_numeric_record_cloudpickle_roundtrip():
    nr = NumericRecord("nr", x=jnp.array(1.0), y=jnp.array([2.0, 3.0]))
    nr2 = cloudpickle_roundtrip(nr)
    assert nr2.fields == ("x", "y")
    assert float(nr2["x"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RecordBatch
# ---------------------------------------------------------------------------


def test_record_batch_pickle_roundtrip():
    template = EventTemplate(x=(), y=(3,))
    ra = RecordBatch(
        {"x": jnp.array([1.0, 2.0]), "y": jnp.ones((2, 3))},
        level_names="draw",
        axis_groups=((2,),),
        element_spec=template,
    )
    ra2 = roundtrip(ra)
    assert ra2.batch_shape == (2,)
    assert ra2.event_template.fields == ("x", "y")
    assert list(ra2["x"]) == pytest.approx([1.0, 2.0])


def test_record_batch_template_preserved():
    template = EventTemplate(x=(), y=(3,))
    ra = RecordBatch(
        {"x": jnp.array([1.0]), "y": jnp.ones((1, 3))},
        level_names="draw",
        axis_groups=((1,),),
        element_spec=template,
    )
    ra2 = roundtrip(ra)
    assert ra2.event_template == template


# ---------------------------------------------------------------------------
# NumericRecordBatch
# ---------------------------------------------------------------------------


def test_numeric_record_batch_pickle_roundtrip():
    template = EventTemplate(x=(), y=(2,))
    nrb = NumericRecordBatch(
        {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.ones((3, 2))},
        level_names="draw",
        axis_groups=((3,),),
        element_spec=template,
    )
    nra2 = roundtrip(nrb)
    assert type(nra2) is NumericRecordBatch
    assert nra2.batch_shape == (3,)
    assert list(nra2["x"]) == pytest.approx([1.0, 2.0, 3.0])


def test_numeric_record_batch_cloudpickle_roundtrip():
    template = EventTemplate(x=())
    nrb = NumericRecordBatch(
        {"x": jnp.array([1.0, 2.0])},
        level_names="draw",
        axis_groups=((2,),),
        element_spec=template,
    )
    nra2 = cloudpickle_roundtrip(nrb)
    assert type(nra2) is NumericRecordBatch
    assert nra2.batch_shape == (2,)


# ---------------------------------------------------------------------------
# EmpiricalDistribution and BootstrapReplicateDistribution
# ---------------------------------------------------------------------------


def test_empirical_distribution_pickle():
    dist = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0, 4.0]), name="x")
    dist2 = roundtrip(dist)
    assert dist2.num_atoms == 4


def test_bootstrap_replicate_pickle():
    base = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="x")
    brd = BootstrapReplicateDistribution(base, name="x")
    brd2 = roundtrip(brd)
    # Verify it round-tripped as the right type and is callable
    assert type(brd2).__name__ == "RecordBootstrapReplicateDistribution"
    assert "replicate_size=3" in repr(brd2)


# ---------------------------------------------------------------------------
# Backend aux survives a pickle round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def xr_da():
    xr = pytest.importorskip("xarray")
    return xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=["t"],
        coords={"t": [10, 20, 30]},
        attrs={"units": "meters"},
        name="temps",
    )


def _coord_ints(leaf):
    return [int(v) for v in leaf.coords["t"].values]


class TestNumericRecordNativePickle:
    """A NumericRecord's native leaves round-trip through pickle (and
    cloudpickle, for Ray transport) verbatim: the backend objects pickle
    themselves, so metadata survives at every nesting level with no capture
    or restore step.
    """

    def test_pickle_preserves_top_level_xarray_native(self, xr_da):
        nr = NumericRecord("nr", temps=xr_da, extra=jnp.array(1.0))
        restored = roundtrip(nr)
        # Native leaves pickle themselves: the restored field IS a DataArray.
        assert restored["temps"].dims == ("t",)
        assert _coord_ints(restored["temps"]) == [10, 20, 30]
        assert restored["temps"].attrs == {"units": "meters"}
        assert type(restored["temps"]).__name__ == "DataArray"

    def test_pickle_preserves_nested_xarray_native(self, xr_da):
        # A nested native leaf pickles through the nested record verbatim.
        outer = NumericRecord("outer", grp=NumericRecord("grp", temps=xr_da))
        back = roundtrip(outer)
        assert back.at_path("grp/temps").dims == ("t",)
        assert _coord_ints(back.at_path("grp/temps")) == [10, 20, 30]

    def test_cloudpickle_preserves_xarray_native(self, xr_da):
        # Ray ships task arguments via cloudpickle.
        back = cloudpickle_roundtrip(NumericRecord("nr", temps=xr_da))
        assert back["temps"].dims == ("t",)
        assert _coord_ints(back["temps"]) == [10, 20, 30]

    def test_pickle_preserves_pandas_series_native(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"], name="obs")
        back = roundtrip(NumericRecord("nr", vals=s))
        restored = back["vals"]
        assert isinstance(restored, pd.Series)
        assert list(restored.index) == ["a", "b", "c"]
        assert restored.name == "obs"
        assert restored.dtype == s.dtype
        assert [float(v) for v in restored] == [1.0, 2.0, 3.0]

    def test_pickle_preserves_pandas_dataframe_native(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=["r0", "r1"])
        back = roundtrip(NumericRecord("nr", table=df))
        restored = back["table"]
        assert isinstance(restored, pd.DataFrame)
        assert list(restored.columns) == ["x", "y"]
        assert list(restored.index) == ["r0", "r1"]
        assert [float(v) for v in restored["x"]] == [1.0, 2.0]

    def test_cloudpickle_preserves_pandas_series_native(self):
        pd = pytest.importorskip("pandas")
        s = pd.Series([4.0, 5.0], index=["p", "q"], name="w")
        back = cloudpickle_roundtrip(NumericRecord("nr", vals=s))
        assert list(back["vals"].index) == ["p", "q"]
        assert back["vals"].name == "w"


# ---------------------------------------------------------------------------
# Pickle preserves an explicit (non-inferred) event_template
# ---------------------------------------------------------------------------


class TestPicklePreservesTemplate:
    """An explicit template that ``infer_from`` cannot reconstruct — a leaf
    ``support`` or an ``OpaqueSpec.meta`` — must survive pickling. Otherwise the
    transported record would silently re-infer a weaker schema and compare
    unequal to its origin (a Ray / cloudpickle schema-drift hazard).
    """

    def test_plain_record_template_survives(self):
        from probpipe.core.constraints import positive

        tpl = EventTemplate(x=ArraySpec(shape=(3,), support=positive), tag=OpaqueSpec(meta="units"))
        r = Record("r", {"x": jnp.ones(3), "tag": "meters"}, event_template=tpl)
        assert not isinstance(r, NumericRecord)  # opaque leaf keeps it a plain Record
        back = roundtrip(r)
        assert back.event_template == r.event_template
        assert back == r

    def test_numeric_record_template_survives(self):
        from probpipe.core.constraints import positive

        tpl = EventTemplate(x=ArraySpec(shape=(3,), support=positive))
        nr = NumericRecord("nr", {"x": jnp.ones(3)}, event_template=tpl)
        back = roundtrip(nr)
        assert back.event_template == nr.event_template
        assert back == nr

    def test_cloudpickle_preserves_template(self):
        from probpipe.core.constraints import positive

        tpl = EventTemplate(x=ArraySpec(shape=(3,), support=positive))
        nr = NumericRecord("nr", {"x": jnp.ones(3)}, event_template=tpl)
        assert cloudpickle_roundtrip(nr).event_template == nr.event_template

    def test_aux_native_path_template_survives(self):
        xr = pytest.importorskip("xarray")
        from probpipe.core.constraints import positive

        da = xr.DataArray([1.0, 2.0, 3.0], dims=["t"], coords={"t": [10, 20, 30]})
        tpl = EventTemplate(x=ArraySpec(shape=(3,), support=positive))
        nr = NumericRecord("nr", {"x": da}, event_template=tpl)
        back = roundtrip(nr)
        assert back.event_template == nr.event_template  # explicit template survived
        assert back["x"].dims == ("t",)  # the native leaf survived verbatim

    def test_pickle_bare_array_record(self):
        # Bare jax leaves are their own native form; the single pickle path
        # round-trips them directly.
        nr = NumericRecord("nr", x=jnp.array([1.0, 2.0]), y=jnp.array(3.0))
        back = roundtrip(nr)
        assert [float(v) for v in back["x"]] == [1.0, 2.0]
        assert float(back["y"]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Pickle preserves annotations
# ---------------------------------------------------------------------------


class TestRoundTripPreservesAnnotations:
    """Annotations are written *after* construction — the documented exception
    to immutability — so no constructor argument carries them and a state list
    assembled from ``__init__`` misses them. They must survive every
    reconstruction path (#409), which ``__reduce__`` governs for ``copy`` as
    well as for ``pickle``.
    """

    @pytest.fixture(
        params=[
            # An opaque leaf keeps this one a plain ``Record`` rather than
            # promoting it, so both classes in the family are covered.
            pytest.param(lambda: Record("r", {"x": jnp.ones(3), "tag": "meters"}), id="record"),
            pytest.param(lambda: NumericRecord("nr", {"x": jnp.ones(3)}), id="numeric-record"),
            pytest.param(
                lambda: ProductDistribution(value=Normal(0.0, 1.0, name="value"), name="joint"),
                id="product-distribution",
            ),
        ]
    )
    def term(self, request):
        term = request.param()
        # The store is written through ``object.__setattr__``, as the library's
        # own writers do: an immutability guard refuses plain assignment.
        object.__setattr__(term, "_annotations", {"diagnostics": {"n_eff": 42}})
        return term

    def test_pickle_preserves_annotations(self, term):
        assert roundtrip(term).annotations == {"diagnostics": {"n_eff": 42}}

    def test_cloudpickle_preserves_annotations(self, term):
        assert cloudpickle_roundtrip(term).annotations == {"diagnostics": {"n_eff": 42}}

    def test_copy_preserves_annotations(self, term):
        assert copy.copy(term).annotations == {"diagnostics": {"n_eff": 42}}
        assert copy.deepcopy(term).annotations == {"diagnostics": {"n_eff": 42}}

    def test_copy_decouples_the_container(self, term):
        # Writers add entries in place, so a shared container would let a write
        # on the copy show through on the original.
        clone = copy.copy(term)
        clone.annotations["added"] = 1
        assert "added" not in term.annotations

    def test_unannotated_term_stays_unannotated(self):
        assert roundtrip(Record("r", {"x": jnp.ones(3)})).annotations is None
        assert roundtrip(ProductDistribution(v=Normal(0.0, 1.0, name="v"))).annotations is None

    def test_no_state_is_lost(self, term):
        # The general form of the bug: compare every attribute the object
        # reports, so a field added later cannot go missing silently. The
        # conversion cache is the one documented exclusion — a memo, rebuilt on
        # demand.
        def assigned(obj):
            state = object.__getstate__(obj)
            instance_dict, slots = state if isinstance(state, tuple) else (state, {})
            return (set(instance_dict or {}) | set(slots or {})) - {"_jax_cache"}

        assert assigned(term) - assigned(roundtrip(term)) == set()

    def test_pickle_without_annotations_still_loads(self):
        # Reconstruction stays callable with the shorter argument list a pickle
        # written before annotations were serialized carries.
        from probpipe.core.record import _unpickle_record

        r = _unpickle_record({"x": jnp.ones(3)}, "r", False, None)
        assert r.name == "r"
        assert r.annotations is None
