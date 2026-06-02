"""Pickle / cloudpickle round-trip tests for the Record family.

These tests ensure that Record, RecordTemplate, NumericRecord, RecordArray,
and NumericRecordArray can survive pickle serialization, which is required for
Ray task distribution (Ray uses cloudpickle to ship arguments to workers).

The core issue was that Record.__setattr__ raises "Record is immutable", so
pickle's default restore mechanism (create empty instance + __setattr__) failed.
The fix adds __reduce__ to each class, delegating reconstruction to the normal
constructor.
"""

import pickle

import jax.numpy as jnp
import pytest

from probpipe import (
    NumericRecord,
    NumericRecordArray,
    RecordArray,
)
from probpipe.core._empirical import BootstrapReplicateDistribution, EmpiricalDistribution
from probpipe.core.record import NumericRecordTemplate, Record, RecordTemplate


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
    r = Record(x=jnp.array(1.0), y=jnp.array([2.0, 3.0]), name="myrecord")
    r2 = roundtrip(r)
    assert r2.name == "myrecord"
    assert r2.fields == ("x", "y")
    assert float(r2["x"]) == pytest.approx(1.0)
    assert list(r2["y"]) == pytest.approx([2.0, 3.0])


def test_record_pickle_auto_name():
    r = Record(a=jnp.array(1.0), b=jnp.array(2.0))
    r2 = roundtrip(r)
    assert r2.name == r.name
    assert r2.fields == ("a", "b")


def test_record_immutability_after_unpickle():
    r = Record(x=jnp.array(1.0))
    r2 = roundtrip(r)
    with pytest.raises(AttributeError, match="immutable"):
        r2.x = jnp.array(99.0)


def test_record_provenance_preserved():
    from probpipe.core.provenance import Provenance
    r = Record(x=jnp.array(1.0))
    r.with_source(Provenance(operation="test_op", metadata={"k": "v"}))
    assert r._source is not None

    r2 = roundtrip(r)
    assert r2._source is not None
    assert r2._source.operation == "test_op"
    assert r2._source.metadata == {"k": "v"}


def test_record_no_provenance_roundtrip():
    r = Record(x=jnp.array(1.0))
    assert r._source is None
    r2 = roundtrip(r)
    assert r2._source is None


# ---------------------------------------------------------------------------
# RecordTemplate
# ---------------------------------------------------------------------------


def test_record_template_pickle_roundtrip():
    t = RecordTemplate(label=None, x=())
    t2 = roundtrip(t)
    assert type(t2) is RecordTemplate
    assert t2.fields == ("label", "x")
    assert t2["label"] is None
    assert t2["x"] == ()


def test_numeric_record_template_pickle_roundtrip():
    t = RecordTemplate(x=(), y=(3,))
    assert type(t) is NumericRecordTemplate
    t2 = roundtrip(t)
    assert type(t2) is NumericRecordTemplate
    assert t2.fields == ("x", "y")
    assert t2.flat_size == 4  # () + (3,)


# ---------------------------------------------------------------------------
# NumericRecord
# ---------------------------------------------------------------------------


def test_numeric_record_pickle_roundtrip():
    nr = NumericRecord(r=jnp.array(1.8), K=jnp.array(70.0), phi=jnp.array(10.0))
    nr2 = roundtrip(nr)
    assert nr2.fields == ("r", "K", "phi")
    assert float(nr2["r"]) == pytest.approx(1.8)
    assert nr2.flat_size == 3


def test_numeric_record_flat_size_recomputed():
    nr = NumericRecord(a=jnp.ones((2, 3)))
    assert nr.flat_size == 6
    nr2 = roundtrip(nr)
    assert nr2.flat_size == 6


def test_numeric_record_cloudpickle_roundtrip():
    nr = NumericRecord(x=jnp.array(1.0), y=jnp.array([2.0, 3.0]))
    nr2 = cloudpickle_roundtrip(nr)
    assert nr2.fields == ("x", "y")
    assert float(nr2["x"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RecordArray
# ---------------------------------------------------------------------------


def test_record_array_pickle_roundtrip():
    template = RecordTemplate(x=(), y=(3,))
    ra = RecordArray(
        {"x": jnp.array([1.0, 2.0]), "y": jnp.ones((2, 3))},
        batch_shape=(2,),
        template=template,
    )
    ra2 = roundtrip(ra)
    assert ra2.batch_shape == (2,)
    assert ra2.fields == ("x", "y")
    assert list(ra2["x"]) == pytest.approx([1.0, 2.0])


def test_record_array_template_preserved():
    template = RecordTemplate(x=(), y=(3,))
    ra = RecordArray(
        {"x": jnp.array([1.0]), "y": jnp.ones((1, 3))},
        batch_shape=(1,),
        template=template,
    )
    ra2 = roundtrip(ra)
    assert ra2.template == template


# ---------------------------------------------------------------------------
# NumericRecordArray
# ---------------------------------------------------------------------------


def test_numeric_record_array_pickle_roundtrip():
    template = RecordTemplate(x=(), y=(2,))
    nra = NumericRecordArray(
        {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.ones((3, 2))},
        batch_shape=(3,),
        template=template,
    )
    nra2 = roundtrip(nra)
    assert type(nra2) is NumericRecordArray
    assert nra2.batch_shape == (3,)
    assert list(nra2["x"]) == pytest.approx([1.0, 2.0, 3.0])


def test_numeric_record_array_cloudpickle_roundtrip():
    template = RecordTemplate(x=())
    nra = NumericRecordArray(
        {"x": jnp.array([1.0, 2.0])},
        batch_shape=(2,),
        template=template,
    )
    nra2 = cloudpickle_roundtrip(nra)
    assert type(nra2) is NumericRecordArray
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
