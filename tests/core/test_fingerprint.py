"""Tests for the fingerprint utility module."""

from __future__ import annotations

import subprocess
import sys
import textwrap

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
        # Two separately-constructed equal values must produce the same digest.
        assert fingerprint(int("42")) == fingerprint(int("42"))

    def test_different_ints_differ(self):
        assert fingerprint(1) != fingerprint(2)

    def test_big_int_does_not_crash(self):
        # Integers outside the signed-64-bit range must not raise.
        assert isinstance(fingerprint(2**100), str)
        assert isinstance(fingerprint([1, 2**100]), str)

    def test_float_stable(self):
        assert fingerprint(float("3.14")) == fingerprint(float("3.14"))

    def test_different_floats_differ(self):
        assert fingerprint(1.0) != fingerprint(2.0)

    def test_nan_and_inf_do_not_crash(self):
        assert isinstance(fingerprint(float("nan")), str)
        assert isinstance(fingerprint(float("inf")), str)
        assert isinstance(fingerprint(float("-inf")), str)

    def test_bool_stable(self):
        assert fingerprint(bool(1)) == fingerprint(bool(1))

    def test_bool_differs_from_int(self):
        # True == 1 in Python but they must hash differently — different types.
        assert fingerprint(True) != fingerprint(1)

    def test_false_differs_from_zero(self):
        assert fingerprint(False) != fingerprint(0)

    def test_string_stable(self):
        assert fingerprint("hello") == fingerprint("hello")

    def test_different_strings_differ(self):
        assert fingerprint("a") != fingerprint("b")

    def test_none_stable(self):
        assert fingerprint(type(None)()) == fingerprint(type(None)())

    def test_none_differs_from_zero(self):
        assert fingerprint(None) != fingerprint(0)

    def test_empty_list_stable(self):
        assert fingerprint([]) == fingerprint([])

    def test_empty_dict_stable(self):
        assert fingerprint({}) == fingerprint({})

    def test_list_stable(self):
        assert fingerprint([1, 2, 3]) == fingerprint([1, 2, 3])

    def test_tuple_stable(self):
        assert fingerprint((1, 2)) == fingerprint((1, 2))

    def test_list_differs_from_tuple(self):
        assert fingerprint([1, 2]) != fingerprint((1, 2))

    def test_dict_stable(self):
        assert fingerprint({"a": 1, "b": 2}) == fingerprint({"a": 1, "b": 2})

    def test_dict_different_values_differ(self):
        assert fingerprint({"a": 1}) != fingerprint({"a": 2})

    def test_unknown_type_does_not_crash(self):
        class _Opaque:
            def __repr__(self):
                return "opaque"

        assert isinstance(fingerprint(_Opaque()), str)


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

    def test_scalar_array_stable(self):
        a = jnp.array(5.0)
        b = jnp.array(5.0)
        assert fingerprint(a) == fingerprint(b)

    def test_max_array_bytes_none_hashes_fully(self):
        # With no cap, two arrays that differ only in their last element must
        # produce different digests even when they are large.
        a = np.zeros(1000)
        b = np.zeros(1000)
        b[-1] = 1.0
        assert fingerprint(a, max_array_bytes=None) != fingerprint(b, max_array_bytes=None)


# ===========================================================================
# 4. Depth guard
# ===========================================================================


class TestDepthGuard:
    def test_deeply_nested_list_does_not_crash(self):
        # Build a list nested 40 levels deep — beyond the depth=32 guard.
        obj: list = []
        for _ in range(40):
            obj = [obj]
        assert isinstance(fingerprint(obj), str)


# ===========================================================================
# 5. Record hashing
# ===========================================================================


class TestRecordHashing:
    def test_same_record_stable(self):
        r1 = Record("r", x=jnp.array(1.0), y=jnp.array(2.0))
        r2 = Record("r", x=jnp.array(1.0), y=jnp.array(2.0))
        assert fingerprint(r1) == fingerprint(r2)

    def test_different_values_differ(self):
        r1 = Record("r", x=jnp.array(1.0))
        r2 = Record("r", x=jnp.array(9.0))
        assert fingerprint(r1) != fingerprint(r2)

    def test_different_fields_differ(self):
        r1 = Record("r", x=jnp.array(1.0))
        r2 = Record("r", y=jnp.array(1.0))
        assert fingerprint(r1) != fingerprint(r2)

    def test_extra_field_differs(self):
        r1 = Record("r", x=jnp.array(1.0))
        r2 = Record("r", x=jnp.array(1.0), y=jnp.array(2.0))
        assert fingerprint(r1) != fingerprint(r2)

    def test_multi_field_stable(self):
        r1 = Record("r", a=jnp.array([1.0, 2.0]), b=jnp.array(3.0), c="label")
        r2 = Record("r", a=jnp.array([1.0, 2.0]), b=jnp.array(3.0), c="label")
        assert fingerprint(r1) == fingerprint(r2)

    def test_string_field_differs(self):
        r1 = Record("r", label="a")
        r2 = Record("r", label="b")
        assert fingerprint(r1) != fingerprint(r2)


# ===========================================================================
# 6. Distribution hashing
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

    def test_empirical_non_uniform_weights_differ(self):
        """IS/SMC reweighting must produce a different fingerprint."""
        from probpipe import RecordEmpiricalDistribution

        samples = jnp.array([1.0, 2.0, 3.0])
        uniform = RecordEmpiricalDistribution(samples, name="post")
        reweighted = RecordEmpiricalDistribution(
            samples, weights=jnp.array([0.7, 0.2, 0.1]), name="post"
        )
        assert fingerprint(uniform) != fingerprint(reweighted)

    def test_kde_distribution_stable(self):
        """KDE (composite TFP distribution) must be stable."""
        from probpipe.distributions.kde import KDEDistribution

        pts = jnp.array([0.0, 1.0, 2.0])
        k1 = KDEDistribution(pts, name="kde")
        k2 = KDEDistribution(pts, name="kde")
        assert fingerprint(k1) == fingerprint(k2)

    def test_kde_different_points_differ(self):
        """Two KDE distributions with different data must have different fingerprints."""
        from probpipe.distributions.kde import KDEDistribution

        k1 = KDEDistribution(jnp.array([0.0, 1.0, 2.0]), name="kde")
        k2 = KDEDistribution(jnp.array([0.0, 1.0, 99.0]), name="kde")
        assert fingerprint(k1) != fingerprint(k2)


class TestBootstrapSourceFingerprint:
    """A sampleable-source bootstrap's fingerprint covers the live source
    distribution, so replicates of different sources fingerprint distinctly."""

    def test_different_sources_differ(self):
        from probpipe import BootstrapReplicateDistribution

        b1 = BootstrapReplicateDistribution(Normal(loc=0.0, scale=1.0, name="x"), replicate_size=10)
        b2 = BootstrapReplicateDistribution(Normal(loc=5.0, scale=1.0, name="x"), replicate_size=10)
        assert fingerprint(b1) != fingerprint(b2)

    def test_same_source_matches(self):
        from probpipe import BootstrapReplicateDistribution

        b1 = BootstrapReplicateDistribution(Normal(loc=0.0, scale=1.0, name="x"), replicate_size=10)
        b2 = BootstrapReplicateDistribution(Normal(loc=0.0, scale=1.0, name="x"), replicate_size=10)
        assert fingerprint(b1) == fingerprint(b2)


# ===========================================================================
# 7. WorkflowFunction hashing
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

        assert fingerprint(self._make_wf(add)) != fingerprint(self._make_wf(multiply))

    def test_changed_constant_in_body_differs(self):
        def f_v1(x: float) -> float:
            return x + 1.0

        def f_v2(x: float) -> float:
            return x + 2.0

        assert fingerprint(self._make_wf(f_v1)) != fingerprint(self._make_wf(f_v2))

    def test_nested_lambda_stable_across_processes(self):
        """A function with a nested lambda must produce the same digest in two
        separate processes — the old repr(co_consts) path embedded memory
        addresses and broke this guarantee."""
        script = textwrap.dedent("""
            import sys
            sys.path.insert(0, sys.argv[1])
            from probpipe.core._fingerprint import fingerprint
            from probpipe.core.node import WorkflowFunction

            def f(x: float) -> float:
                transform = lambda v: v * 2.0  # noqa: E731
                return transform(x)

            wf = WorkflowFunction(func=f, dispatch="sequential", n_broadcast_samples=10, seed=42)
            print(fingerprint(wf))
        """)
        site = str(next(p for p in sys.path if "site-packages" in p))
        run = lambda: subprocess.check_output(  # noqa: E731
            [sys.executable, "-c", script, site], text=True
        ).strip()
        assert run() == run()


# ===========================================================================
# 8. Fingerprint populated on ParentInfo via Provenance.create()
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

    def test_fingerprint_error_logs_warning(self, monkeypatch, caplog):
        """A fingerprint failure emits a warning and sets fingerprint=None."""
        import logging

        from probpipe.core import provenance as prov_mod

        def _bad_fp(obj, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(prov_mod, "_fingerprint_func", _bad_fp, raising=False)

        # Patch the lazy import inside create() to use our broken function.
        import probpipe.core._fingerprint as fp_mod

        monkeypatch.setattr(fp_mod, "fingerprint", _bad_fp)

        n = Normal(loc=0.0, scale=1.0, name="prior")
        with caplog.at_level(logging.WARNING, logger="probpipe.core.provenance"):
            prov = Provenance.create("op", parents=[n])

        assert prov is not None
        assert prov.parents[0].fingerprint is None
        assert any(
            "fingerprint" in r.message.lower() or "boom" in r.message for r in caplog.records
        )


# ===========================================================================
# 9. Review-fix regressions — determinism, collisions, leaf-keyed records
# ===========================================================================


class TestWorkflowFunctionCapture:
    """Bytecode alone is not enough: referenced names, closures, and defaults."""

    def _wf(self, func):
        return WorkflowFunction(func=func, dispatch="sequential", n_broadcast_samples=10, seed=42)

    def test_called_name_differs(self):
        # ``jnp.sin`` vs ``jnp.cos``: identical co_code + co_consts, differing
        # only in co_names — a collision before the fix.
        assert fingerprint(self._wf(lambda x: jnp.sin(x))) != fingerprint(
            self._wf(lambda x: jnp.cos(x))
        )

    def test_closure_capture_differs(self):
        def make(k):
            def g(x):
                return x + k

            return g

        assert fingerprint(self._wf(make(1.0))) != fingerprint(self._wf(make(2.0)))

    def test_default_arg_differs(self):
        def d1(x, k=1.0):
            return x + k

        def d2(x, k=2.0):
            return x + k

        assert fingerprint(self._wf(d1)) != fingerprint(self._wf(d2))


class TestSetHashing:
    def test_order_independent(self):
        assert fingerprint({1, 2, 3}) == fingerprint({3, 1, 2})

    def test_content_differs(self):
        assert fingerprint({1, 2, 3}) != fingerprint({1, 2, 4})

    def test_frozenset_order_independent(self):
        assert fingerprint(frozenset({"a", "b"})) == fingerprint(frozenset({"b", "a"}))

    def test_stable_across_processes(self):
        """Set iteration order is PYTHONHASHSEED-randomized; the digest is not."""
        import os

        script = textwrap.dedent("""
            import sys
            sys.path.insert(0, sys.argv[1])
            from probpipe.core._fingerprint import fingerprint
            print(fingerprint({"alpha", "beta", "gamma", "delta"}))
        """)
        site = str(next(p for p in sys.path if "site-packages" in p))

        def run(seed):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            return subprocess.check_output(
                [sys.executable, "-c", script, site], text=True, env=env
            ).strip()

        assert run("1") == run("2")


class TestNumpyScalarHashing:
    def test_int64_stable_and_differs(self):
        # np.int64 is not an int subclass — it hit the repr fallback before.
        assert fingerprint(np.int64(5)) == fingerprint(np.int64(5))
        assert fingerprint(np.int64(5)) != fingerprint(np.int64(6))

    def test_float32_stable(self):
        assert fingerprint(np.float32(1.5)) == fingerprint(np.float32(1.5))


class TestFloatCanonicalization:
    def test_negative_zero_equals_zero(self):
        assert fingerprint(-0.0) == fingerprint(0.0)

    def test_nan_payloads_collapse(self):
        import struct

        alt_nan = struct.unpack(">d", struct.pack(">Q", 0x7FF8000000000001))[0]
        assert alt_nan != alt_nan  # a NaN with a non-canonical payload
        assert fingerprint(float("nan")) == fingerprint(alt_nan)


class TestNestedRecordHashing:
    def test_nested_record_stable(self):
        r1 = Record("r", outer=Record("r", a=1.0, b=2.0), m=3.0)
        r2 = Record("r", outer=Record("r", a=1.0, b=2.0), m=3.0)
        assert fingerprint(r1) == fingerprint(r2)

    def test_nested_leaf_change_differs(self):
        r1 = Record("r", outer=Record("r", a=1.0, b=2.0), m=3.0)
        r2 = Record("r", outer=Record("r", a=1.0, b=9.0), m=3.0)
        assert fingerprint(r1) != fingerprint(r2)


class TestEmpiricalReweighting:
    """The Record-backed empirical class stores no ``_samples`` — reweighted
    posteriors must still be distinguished (was a silent collision)."""

    def _emp(self, weights):
        from probpipe.core._empirical import EmpiricalDistribution

        s = jnp.array([1.0, 2.0, 3.0])
        return EmpiricalDistribution(s, log_weights=jnp.log(jnp.array(weights)), name="p")

    def test_reweighted_differs(self):
        assert fingerprint(self._emp([0.7, 0.2, 0.1])) != fingerprint(self._emp([0.1, 0.2, 0.7]))

    def test_same_weights_stable(self):
        assert fingerprint(self._emp([0.7, 0.2, 0.1])) == fingerprint(self._emp([0.7, 0.2, 0.1]))


class TestArrayEdgeCases:
    def test_over_cap_tail_change_detected(self):
        # Above the cap: a change in the LAST element must be seen — the old
        # sampling never covered the buffer tail.
        base = np.arange(100_000, dtype=np.float64)
        changed = base.copy()
        changed[-1] = -1.0
        assert fingerprint(jnp.asarray(base), max_array_bytes=1000) != fingerprint(
            jnp.asarray(changed), max_array_bytes=1000
        )

    def test_object_dtype_array_stable_by_content(self):
        a = np.array(["x", "y", "z"], dtype=object)
        b = np.array(["x", "y", "z"], dtype=object)
        assert fingerprint(a) == fingerprint(b)


class TestParentInfoIdentity:
    def test_fingerprint_excluded_from_equality(self):
        # Two descriptors for the same ancestor compare/hash equal regardless of
        # the content digest, so a fingerprint can't perturb ancestor-set dedup.
        a = ParentInfo(type_name="X", name="n", provenance=None, fingerprint="aaaaaaaaaaaaaaaa")
        b = ParentInfo(type_name="X", name="n", provenance=None, fingerprint="bbbbbbbbbbbbbbbb")
        assert a == b
        assert hash(a) == hash(b)
