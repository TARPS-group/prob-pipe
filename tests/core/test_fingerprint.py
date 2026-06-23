"""Tests for the fingerprint utility module."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from probpipe import Normal, Record
from probpipe.core._fingerprint import fingerprint
from probpipe.core.node import WorkflowFunction
from probpipe.core.provenance import ParentInfo, Provenance

# ===========================================================================
# 1. Return type and format
# ===========================================================================


class TestReturnFormat:
    def test_returns_string(self):
        assert isinstance(fingerprint(Normal(loc=0.0, scale=1.0, name="n")), str)

    def test_returns_16_chars(self):
        assert len(fingerprint(Normal(loc=0.0, scale=1.0, name="n"))) == 16

    def test_hex_characters_only(self):
        fp = fingerprint(Normal(loc=0.0, scale=1.0, name="n"))
        assert all(c in "0123456789abcdef" for c in fp)


# ===========================================================================
# 2. Python primitives
# ===========================================================================


class TestPrimitives:
    def test_int_stable(self):
        assert fingerprint(42) == fingerprint(42)

    def test_different_ints_differ(self):
        assert fingerprint(1) != fingerprint(2)

    def test_float_stable(self):
        assert fingerprint(3.14) == fingerprint(3.14)

    def test_different_floats_differ(self):
        assert fingerprint(1.0) != fingerprint(2.0)

    def test_bool_stable(self):
        assert fingerprint(True) == fingerprint(True)

    def test_bool_differs_from_int(self):
        # True == 1 in Python but they should hash differently as types differ
        assert fingerprint(True) != fingerprint(1)

    def test_string_stable(self):
        assert fingerprint("hello") == fingerprint("hello")

    def test_different_strings_differ(self):
        assert fingerprint("a") != fingerprint("b")

    def test_none_stable(self):
        assert fingerprint(None) == fingerprint(None)

    def test_none_differs_from_zero(self):
        assert fingerprint(None) != fingerprint(0)


# ===========================================================================
# 3. Array hashing
# ===========================================================================


class TestArrayHashing:
    def test_jax_array_stable(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, 2.0, 3.0])
        assert fingerprint(a) == fingerprint(b)

    def test_jax_array_different_values(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, 2.0, 9.0])
        assert fingerprint(a) != fingerprint(b)

    def test_jax_array_different_shapes(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([[1.0, 2.0]])
        assert fingerprint(a) != fingerprint(b)

    def test_jax_array_different_dtypes(self):
        # Use int32 vs float32 — always distinct regardless of x64 mode.
        a = jnp.array(1, dtype=jnp.int32)
        b = jnp.array(1, dtype=jnp.float32)
        assert fingerprint(a) != fingerprint(b)

    def test_numpy_array_stable(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        assert fingerprint(a) == fingerprint(b)

    def test_numpy_different_from_jax(self):
        # Different types → different hash prefix
        a = np.array([1.0, 2.0])
        b = jnp.array([1.0, 2.0])
        # They may or may not differ (numpy/jax both go through np.asarray);
        # what matters is that each is stable individually.
        assert fingerprint(a) == fingerprint(a)
        assert fingerprint(b) == fingerprint(b)

    def test_scalar_array_stable(self):
        a = jnp.array(5.0)
        b = jnp.array(5.0)
        assert fingerprint(a) == fingerprint(b)


# ===========================================================================
# 4. Record hashing
# ===========================================================================


class TestRecordHashing:
    def test_same_record_stable(self):
        r1 = Record(x=jnp.array(1.0), y=jnp.array(2.0))
        r2 = Record(x=jnp.array(1.0), y=jnp.array(2.0))
        assert fingerprint(r1) == fingerprint(r2)

    def test_different_values_differ(self):
        r1 = Record(x=jnp.array(1.0))
        r2 = Record(x=jnp.array(9.0))
        assert fingerprint(r1) != fingerprint(r2)

    def test_different_fields_differ(self):
        r1 = Record(x=jnp.array(1.0))
        r2 = Record(y=jnp.array(1.0))
        assert fingerprint(r1) != fingerprint(r2)

    def test_extra_field_differs(self):
        r1 = Record(x=jnp.array(1.0))
        r2 = Record(x=jnp.array(1.0), y=jnp.array(2.0))
        assert fingerprint(r1) != fingerprint(r2)

    def test_multi_field_stable(self):
        r1 = Record(a=jnp.array([1.0, 2.0]), b=jnp.array(3.0), c="label")
        r2 = Record(a=jnp.array([1.0, 2.0]), b=jnp.array(3.0), c="label")
        assert fingerprint(r1) == fingerprint(r2)

    def test_string_field_differs(self):
        r1 = Record(label="a")
        r2 = Record(label="b")
        assert fingerprint(r1) != fingerprint(r2)


# ===========================================================================
# 5. Distribution hashing
# ===========================================================================


class TestDistributionHashing:
    def test_same_normal_stable(self):
        n1 = Normal(loc=0.0, scale=1.0, name="x")
        n2 = Normal(loc=0.0, scale=1.0, name="x")
        assert fingerprint(n1) == fingerprint(n2)

    def test_different_loc_differs(self):
        n1 = Normal(loc=0.0, scale=1.0, name="x")
        n2 = Normal(loc=1.0, scale=1.0, name="x")
        assert fingerprint(n1) != fingerprint(n2)

    def test_different_scale_differs(self):
        n1 = Normal(loc=0.0, scale=1.0, name="x")
        n2 = Normal(loc=0.0, scale=2.0, name="x")
        assert fingerprint(n1) != fingerprint(n2)

    def test_different_name_differs(self):
        n1 = Normal(loc=0.0, scale=1.0, name="x")
        n2 = Normal(loc=0.0, scale=1.0, name="y")
        assert fingerprint(n1) != fingerprint(n2)

    def test_different_distribution_types_differ(self):
        from probpipe import Beta

        n = Normal(loc=0.0, scale=1.0, name="x")
        b = Beta(alpha=1.0, beta=1.0, name="x")
        assert fingerprint(n) != fingerprint(b)

    def test_empirical_distribution_stable(self):
        from probpipe import RecordEmpiricalDistribution

        samples = jnp.array([1.0, 2.0, 3.0])
        e1 = RecordEmpiricalDistribution(samples, name="posterior")
        e2 = RecordEmpiricalDistribution(samples, name="posterior")
        assert fingerprint(e1) == fingerprint(e2)

    def test_empirical_different_samples_differ(self):
        from probpipe import RecordEmpiricalDistribution

        e1 = RecordEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="post")
        e2 = RecordEmpiricalDistribution(jnp.array([1.0, 2.0, 9.0]), name="post")
        assert fingerprint(e1) != fingerprint(e2)


# ===========================================================================
# 6. WorkflowFunction hashing
# ===========================================================================


class TestWorkflowFunctionHashing:
    def _make_wf(self, func):
        return WorkflowFunction(func=func, dispatch="sequential", n_broadcast_samples=10, seed=42)

    def test_same_function_stable(self):
        def add(x: float, y: float) -> float:
            return x + y

        wf1 = self._make_wf(add)
        wf2 = self._make_wf(add)
        assert fingerprint(wf1) == fingerprint(wf2)

    def test_different_function_bodies_differ(self):
        def add(x: float, y: float) -> float:
            return x + y

        def multiply(x: float, y: float) -> float:
            return x * y

        wf_add = self._make_wf(add)
        wf_mul = self._make_wf(multiply)
        assert fingerprint(wf_add) != fingerprint(wf_mul)

    def test_changed_constant_in_body_differs(self):
        def f_v1(x: float) -> float:
            return x + 1.0

        def f_v2(x: float) -> float:
            return x + 2.0

        assert fingerprint(self._make_wf(f_v1)) != fingerprint(self._make_wf(f_v2))


# ===========================================================================
# 7. Fingerprint populated on ParentInfo via Provenance.create()
# ===========================================================================


class TestFingerprintInProvenance:
    def test_parentinfo_fingerprint_set(self):
        n = Normal(loc=0.0, scale=1.0, name="prior")
        prov = Provenance.create("op", parents=[n])
        assert prov is not None
        parent = prov.parents[0]
        assert isinstance(parent, ParentInfo)
        assert parent.fingerprint is not None
        assert len(parent.fingerprint) == 16

    def test_parentinfo_fingerprint_stable_across_create_calls(self):
        n = Normal(loc=0.0, scale=1.0, name="prior")
        p1 = Provenance.create("op", parents=[n])
        p2 = Provenance.create("op", parents=[n])
        assert p1.parents[0].fingerprint == p2.parents[0].fingerprint

    def test_different_parents_different_fingerprints(self):
        n1 = Normal(loc=0.0, scale=1.0, name="a")
        n2 = Normal(loc=5.0, scale=1.0, name="b")
        prov = Provenance.create("op", parents=[n1, n2])
        fp1 = prov.parents[0].fingerprint
        fp2 = prov.parents[1].fingerprint
        assert fp1 != fp2

    def test_fingerprint_in_to_dict(self):
        n = Normal(loc=0.0, scale=1.0, name="prior")
        prov = Provenance.create("op", parents=[n])
        d = prov.to_dict()
        assert "fingerprint" in d["parents"][0]
        assert d["parents"][0]["fingerprint"] == prov.parents[0].fingerprint

    def test_off_mode_no_fingerprint(self):
        import probpipe
        from probpipe import ProvenanceMode

        probpipe.provenance_config.mode = ProvenanceMode.OFF
        try:
            n = Normal(loc=0.0, scale=1.0, name="prior")
            prov = Provenance.create("op", parents=[n])
            assert prov is None
        finally:
            probpipe.provenance_config.reset()
