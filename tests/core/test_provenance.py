"""Comprehensive tests for provenance tracking across all distribution operations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

import probpipe
from probpipe import (
    Beta,
    JointGaussian,
    Normal,
    NumericRecord,
    NumericRecordArray,
    ProductDistribution,
    Provenance,
    ProvenanceMode,
    RecordEmpiricalDistribution,
    SequentialJointDistribution,
    TransformedDistribution,
    condition_on,
    from_distribution,
    provenance_ancestors,
    provenance_dag,
)
from probpipe.core.node import Function
from probpipe.core.provenance import ParentInfo

# ===========================================================================
# 1. Provenance basics (dataclass, with_provenance, write-once)
# ===========================================================================


class TestProvenanceBasics:
    def test_provenance_fields(self):
        p = Provenance("test_op", metadata={"key": "val"})
        assert p.operation == "test_op"
        assert p.parents == ()
        assert p.metadata == {"key": "val"}
        assert p.inputs == {}

    def test_inputs_follow_metadata_for_positional_compatibility(self):
        p = Provenance("test_op", (), {"key": "val"})

        assert p.metadata == {"key": "val"}
        assert p.inputs == {}

    def test_plain_input_fingerprints_are_content_sensitive(self):
        first = Provenance.create(
            "op",
            inputs={"scalar": 1.0, "array": np.asarray([1.0, 2.0])},
        )
        same = Provenance.create(
            "op",
            inputs={"scalar": 1.0, "array": np.asarray([1.0, 2.0])},
        )
        different = Provenance.create(
            "op",
            inputs={"scalar": 5.0, "array": np.asarray([1.0, 3.0])},
        )

        assert first is not None
        assert same is not None
        assert different is not None
        assert first.inputs["scalar"].fingerprint == same.inputs["scalar"].fingerprint
        assert first.inputs["array"].fingerprint == same.inputs["array"].fingerprint
        assert first.inputs["scalar"].fingerprint != different.inputs["scalar"].fingerprint
        assert first.inputs["array"].fingerprint != different.inputs["array"].fingerprint

    def test_provenance_with_parents(self):
        n = Normal(loc=0.0, scale=1.0, name="n")
        p = Provenance("op", parents=(n,))
        assert len(p.parents) == 1
        assert p.parents[0] is n

    def test_repr(self):
        n = Normal(loc=0.0, scale=1.0, name="n")
        p = Provenance("op", parents=(n,))
        assert "op" in repr(p)
        assert "n" in repr(p)

    def test_write_once(self):
        n = Normal(loc=0.0, scale=1.0, name="n")
        n.with_provenance(Provenance("first"))
        with pytest.raises(RuntimeError, match="already set"):
            n.with_provenance(Provenance("second"))

    def test_with_provenance_none_is_noop_distribution(self):
        """with_provenance(None) leaves a Distribution unchanged."""
        n = Normal(loc=0.0, scale=1.0, name="n")
        result = n.with_provenance(None)
        assert result is n
        assert n.provenance is None

    def test_with_provenance_none_is_noop_record(self):
        """with_provenance(None) leaves a Record unchanged."""
        from probpipe import Record

        r = Record("r", {"x": jnp.array(1.0)})
        result = r.with_provenance(None)
        assert result is r
        assert r.provenance is None


# ===========================================================================
# 2. ParentInfo hash and equality
# ===========================================================================


class TestParentInfoHashEq:
    def test_hashable_without_provenance(self):
        """Root ParentInfo (provenance=None) is hashable."""
        p = ParentInfo(type_name="Normal", name="prior")
        assert isinstance(hash(p), int)

    def test_hashable_with_provenance(self):
        """Non-root ParentInfo (source set) is hashable despite Provenance.metadata."""
        src = Provenance("op", metadata={"key": "val"})
        p = ParentInfo(type_name="Gamma", name="mid", provenance=src)
        assert isinstance(hash(p), int)

    def test_usable_in_set(self):
        """set(provenance_ancestors(d)) does not raise TypeError."""
        X = Normal(loc=0.0, scale=1.0, name="X")
        Y = Normal(loc=0.0, scale=1.0, name="Y")
        Y.with_provenance(Provenance.create("op", parents=[X]))
        ancestors = provenance_ancestors(Y)
        s = set(ancestors)
        assert len(s) == 1

    def test_equal_same_fields(self):
        """Two ParentInfo with identical fields compare equal."""
        src = Provenance("op")
        a = ParentInfo(type_name="Normal", name="x", provenance=src)
        b = ParentInfo(type_name="Normal", name="x", provenance=src)
        assert a == b

    def test_not_equal_different_name(self):
        a = ParentInfo(type_name="Normal", name="x")
        b = ParentInfo(type_name="Normal", name="y")
        assert a != b

    def test_not_equal_different_type(self):
        a = ParentInfo(type_name="Normal", name="x")
        b = ParentInfo(type_name="Gamma", name="x")
        assert a != b

    def test_obj_excluded_from_eq(self):
        """obj field does not affect equality — same ancestor, different live ref."""
        n = Normal(loc=0.0, scale=1.0, name="x")
        a = ParentInfo(type_name="Normal", name="x", parent=n)
        b = ParentInfo(type_name="Normal", name="x", parent=None)
        assert a == b

    def test_hash_contract(self):
        """Equal ParentInfo must have equal hashes."""
        src = Provenance("op", metadata={"k": 1})
        a = ParentInfo(type_name="Normal", name="x", provenance=src)
        b = ParentInfo(type_name="Normal", name="x", provenance=src)
        assert a == b
        assert hash(a) == hash(b)


# ===========================================================================
# 3. from_distribution provenance
# ===========================================================================


class TestFromDistributionProvenance:
    def test_normal_from_distribution(self):
        src = Beta(alpha=2.0, beta=5.0, name="beta_src")
        converted = from_distribution(src, Normal)
        assert converted.provenance is not None
        assert converted.provenance.operation == "workflow.from_distribution"
        assert len(converted.provenance.parents) == 2
        assert isinstance(converted.provenance.parents[0], ParentInfo)
        assert converted.provenance.parents[0].name == "from_distribution"
        assert converted.provenance.parents[1].name == "beta_src"

    def test_empirical_from_distribution(self):
        src = Normal(loc=0.0, scale=1.0, name="norm_src")
        ed = from_distribution(src, RecordEmpiricalDistribution, n_samples=100)
        assert ed.provenance is not None
        assert ed.provenance.operation == "workflow.from_distribution"
        assert len(ed.provenance.parents) == 2
        assert isinstance(ed.provenance.parents[0], ParentInfo)
        assert ed.provenance.parents[0].name == "from_distribution"
        assert ed.provenance.parents[1].name == "norm_src"


# ===========================================================================
# 3. TransformedDistribution provenance
# ===========================================================================


class TestTransformedDistributionProvenance:
    def test_transform_provenance_attached(self):
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp())
        assert td.provenance is not None
        assert td.provenance.operation == "transform"
        assert len(td.provenance.parents) == 1
        assert isinstance(td.provenance.parents[0], ParentInfo)
        assert td.provenance.parents[0].name == "base"
        assert td.provenance.metadata["bijector"] == "Exp"

    def test_transform_chain_provenance(self):
        base = Normal(loc=0.0, scale=1.0, name="base")
        bij = tfb.Chain([tfb.Exp(), tfb.Shift(1.0)])
        td = TransformedDistribution(base, bij)
        assert td.provenance.metadata["bijector"] == "Chain"

    def test_transform_with_empirical_base(self):
        ed = RecordEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="x")
        td = TransformedDistribution(ed, tfb.Exp())
        assert td.provenance is not None
        assert td.provenance.operation == "transform"
        assert len(td.provenance.parents) == 1
        assert isinstance(td.provenance.parents[0], ParentInfo)
        assert td.provenance.parents[0].name == "x"


# ===========================================================================
# 4. Conditioning provenance
# ===========================================================================


class TestConditioningProvenance:
    def test_product_condition_on(self):
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=1.0, scale=2.0, name="y"),
        )
        raw = condition_on.apply(joint, x=jnp.array(0.0))
        cond = condition_on(joint, x=jnp.array(0.0))
        assert raw.provenance.operation == "condition_on"
        assert "x" in raw.provenance.metadata["conditioned"]
        assert cond.provenance is not None
        assert cond.provenance.operation == "workflow.condition_on"
        assert len(cond.provenance.parents) == 2
        assert isinstance(cond.provenance.parents[0], ParentInfo)
        assert [parent.name for parent in cond.provenance.parents] == [
            "condition_on",
            joint.name,
        ]

    def test_condition_on_records_plain_observation_by_parameter(self):
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=1.0, scale=2.0, name="y"),
        )

        at_zero = condition_on(joint, x=jnp.array(0.0))
        at_five = condition_on(joint, x=jnp.array(5.0))

        assert at_zero.provenance is not None
        assert at_five.provenance is not None
        assert [parent.fingerprint for parent in at_zero.provenance.parents] == [
            parent.fingerprint for parent in at_five.provenance.parents
        ]
        assert at_zero.provenance.inputs["**kwargs['x']"].fingerprint != (
            at_five.provenance.inputs["**kwargs['x']"].fingerprint
        )

    def test_sequential_condition_on(self):
        seq = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
        )
        cond = condition_on(seq, z=jnp.array(1.0))
        assert cond.provenance.operation == "workflow.condition_on"
        assert len(cond.provenance.parents) == 2
        assert isinstance(cond.provenance.parents[0], ParentInfo)

    def test_gaussian_condition_on(self):
        jg = JointGaussian(
            mean=jnp.zeros(2),
            cov=jnp.eye(2),
            x=1,
            y=1,
        )
        cond = condition_on(jg, x=jnp.array([0.0]))
        assert cond.provenance.operation == "workflow.condition_on"
        assert [parent.name for parent in cond.provenance.parents] == [
            "condition_on",
            jg.name,
        ]


# ===========================================================================
# 5. Broadcasting provenance
# ===========================================================================


class TestBroadcastingProvenance:
    def test_broadcast_loop_provenance(self, full_provenance_mode):
        n = Normal(loc=0.0, scale=1.0, name="input_normal")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(x=n)
        assert hasattr(result, "samples")
        assert result.provenance is not None
        assert result.provenance.operation == "broadcast"
        assert len(result.provenance.parents) == 2
        assert isinstance(result.provenance.parents[0], ParentInfo)
        assert result.provenance.parents[0].parent is wf
        assert result.provenance.parents[1].parent is n
        assert result.provenance.metadata["dispatch"] == "sequential"
        assert result.provenance.metadata["n_samples"] == 20
        assert result.provenance.metadata["func"] == "identity"
        assert result.provenance.metadata["broadcast_args"] == ["x"]

    def test_broadcast_jax_provenance(self):
        n = Normal(loc=0.0, scale=1.0, name="jax_input")

        def double(x: float) -> float:
            return 2.0 * x

        wf = Function(func=double, dispatch="jax", n_broadcast_samples=20, seed=42)
        result = wf(x=n)
        assert hasattr(result, "samples")
        assert result.provenance is not None
        assert result.provenance.operation == "broadcast"
        assert result.provenance.metadata["dispatch"] == "jax"

    @pytest.mark.parametrize("dispatch", ["sequential", "jax"])
    def test_broadcast_records_static_plain_inputs(self, dispatch):
        n = Normal(loc=0.0, scale=1.0, name="input_normal")

        def shift(x: float, offset: float = 2.0) -> float:
            return x + offset

        wf = Function(func=shift, dispatch=dispatch, n_broadcast_samples=5, seed=42)

        result = wf(n)

        assert result.provenance is not None
        assert tuple(result.provenance.inputs) == ("offset",)
        assert result.provenance.inputs["offset"].fingerprint is not None

    def test_broadcast_multiple_parents(self):
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = Normal(loc=1.0, scale=0.5, name="b")

        def add(x: float, y: float) -> float:
            return x + y

        wf = Function(func=add, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(x=a, y=b)
        assert result.provenance is not None
        assert len(result.provenance.parents) == 3
        assert [parent.name for parent in result.provenance.parents] == ["add", "a", "b"]

    def test_broadcast_enumerate_provenance(self):
        """Enumeration path should also get provenance."""
        ed = RecordEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="x")
        n = Normal(loc=0.0, scale=1.0, name="n")

        def add(a: float, b: float) -> float:
            return a + b

        wf = Function(func=add, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(a=ed, b=n)
        assert hasattr(result, "samples")
        assert result.provenance is not None
        assert result.provenance.operation == "broadcast"

    def test_sweep_records_static_plain_inputs(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=float(value)) for value in range(3)]
        )

        def shift(row, offset: float = 2.0) -> float:
            return row["value"] + offset

        result = Function(func=shift)(rows)

        assert result.provenance is not None
        assert tuple(result.provenance.inputs) == ("offset",)
        assert result.provenance.inputs["offset"].fingerprint is not None

    def test_nested_broadcast_records_static_plain_inputs(self):
        rows = NumericRecordArray.stack(
            [NumericRecord("row", value=float(value)) for value in range(2)]
        )
        noise = Normal(loc=0.0, scale=1.0, name="noise")

        def add_noise(row, random_value: float, offset: float = 2.0) -> float:
            return row["value"] + random_value + offset

        wf = Function(func=add_noise, dispatch="sequential", n_broadcast_samples=5, seed=42)

        result = wf(rows, noise)

        assert result.provenance is not None
        assert result.components[0].provenance is not None
        assert tuple(result.provenance.inputs) == ("offset",)
        assert tuple(result.components[0].provenance.inputs) == ("offset",)


# ===========================================================================
# 6. Provenance chains (multi-step lineage)
# ===========================================================================


class TestProvenanceChains:
    def test_two_step_chain(self):
        """from_distribution → condition_on creates a 2-step chain."""
        src = Beta(alpha=2.0, beta=5.0, name="prior")
        converted = from_distribution(src, Normal, name="x")
        joint = ProductDistribution(x=converted, y=Normal(loc=0.0, scale=1.0, name="y"))
        cond = condition_on(joint, x=jnp.array(0.0))

        # cond's provenance points to joint
        assert cond.provenance.operation == "workflow.condition_on"
        # joint has no provenance (constructed directly)
        assert joint.provenance is None
        # but converted has provenance pointing to src
        assert converted.provenance.operation == "workflow.from_distribution"
        assert isinstance(converted.provenance.parents[1], ParentInfo)
        assert converted.provenance.parents[1].name == "prior"

    def test_transform_then_broadcast(self, full_provenance_mode):
        """transform → broadcast creates a 2-step chain."""
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def log_val(x: float) -> float:
            return jnp.log(x)

        wf = Function(func=log_val, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(x=td)
        # result → broadcast → td → transform → base
        assert result.provenance.operation == "broadcast"
        assert result.provenance.parents[0].parent is wf
        parent_td = result.provenance.parents[1]
        assert isinstance(parent_td, ParentInfo)
        assert parent_td.parent is td
        assert parent_td.provenance.operation == "transform"
        assert isinstance(parent_td.provenance.parents[0], ParentInfo)
        assert parent_td.provenance.parents[0].parent is base


# ===========================================================================
# 7. Serialization
# ===========================================================================


class TestSerialization:
    def test_to_dict_basic(self):
        p = Provenance("test_op", metadata={"key": "val"})
        d = p.to_dict()
        assert d["operation"] == "test_op"
        assert d["parents"] == []
        assert d["metadata"] == {"key": "val"}

    def test_to_dict_with_parents(self):
        n = Normal(loc=0.0, scale=1.0, name="my_normal")
        p = Provenance.create("op", parents=[n], metadata={"x": 1})
        d = p.to_dict()
        assert len(d["parents"]) == 1
        assert d["parents"][0]["type"] == "Normal"
        assert d["parents"][0]["name"] == "my_normal"

    def test_to_dict_with_plain_inputs(self):
        value = jnp.array([1.0, 2.0])
        p = Provenance.create("op", inputs={"x": value})

        assert p is not None
        d = p.to_dict()

        assert d["inputs"]["x"]["type"] == type(value).__name__
        assert d["inputs"]["x"]["name"] is None
        assert d["inputs"]["x"]["fingerprint"] == p.inputs["x"].fingerprint

    def test_to_dict_recursive(self):
        """Recursive serialization follows provenance chains."""
        src = Normal(loc=0.0, scale=1.0, name="src")
        td = TransformedDistribution(src, tfb.Exp(), name="transformed")
        p = Provenance.create("broadcast", parents=[td])
        d = p.to_dict(recurse=True)
        # td has source (transform), which should be serialized via ParentInfo.provenance
        assert "provenance" in d["parents"][0]
        assert d["parents"][0]["provenance"]["operation"] == "transform"

    def test_to_dict_non_recursive(self):
        src = Normal(loc=0.0, scale=1.0, name="src")
        td = TransformedDistribution(src, tfb.Exp(), name="transformed")
        p = Provenance.create("broadcast", parents=[td])
        d = p.to_dict(recurse=False)
        assert "provenance" not in d["parents"][0]

    def test_from_dict_roundtrip(self):
        p = Provenance("my_op", metadata={"a": 1, "b": "two"})
        d = p.to_dict()
        restored = Provenance.from_dict(d)
        assert restored.operation == "my_op"
        assert restored.metadata["a"] == 1
        assert restored.metadata["b"] == "two"
        # Parents can't be reconstructed, so they're empty
        assert restored.parents == ()

    def test_from_dict_preserves_parent_info(self):
        n = Normal(loc=0.0, scale=1.0, name="n")
        p = Provenance.create("op", parents=[n])
        d = p.to_dict()
        restored = Provenance.from_dict(d)
        # Parent info preserved in metadata
        assert restored.metadata["_parents_info"][0]["type"] == "Normal"
        assert restored.metadata["_parents_info"][0]["name"] == "n"

    def test_from_dict_preserves_plain_input_info(self):
        p = Provenance.create("op", inputs={"x": jnp.array([1.0, 2.0])})

        assert p is not None
        restored = Provenance.from_dict(p.to_dict())

        assert restored.inputs == {}
        assert restored.metadata["_inputs_info"]["x"]["fingerprint"] == (p.inputs["x"].fingerprint)

    def test_from_dict_accepts_legacy_dict_without_inputs(self):
        restored = Provenance.from_dict({"operation": "legacy", "parents": [], "metadata": {}})

        assert restored.inputs == {}
        assert restored.metadata["_inputs_info"] == {}

    def test_to_dict_fingerprint_included(self):
        """fingerprint is serialized when set on a ParentInfo."""
        pi = ParentInfo(type_name="Normal", name="x", fingerprint="abc123")
        p = Provenance("op", parents=(pi,))
        d = p.to_dict()
        assert d["parents"][0]["fingerprint"] == "abc123"

    def test_to_dict_fingerprint_omitted_when_none(self):
        """fingerprint key is absent when fingerprint is None."""
        pi = ParentInfo(type_name="Normal", name="x")
        p = Provenance("op", parents=(pi,))
        d = p.to_dict()
        assert "fingerprint" not in d["parents"][0]

    def test_to_dict_filters_non_serializable(self):
        """Non-JSON-serializable metadata values are stringified."""
        p = Provenance("op", metadata={"array": jnp.array([1, 2, 3])})
        d = p.to_dict()
        assert isinstance(d["metadata"]["array"], str)


# ===========================================================================
# 8. provenance_ancestors
# ===========================================================================


class TestProvenanceAncestors:
    def test_no_provenance_returns_empty(self):
        n = Normal(loc=0.0, scale=1.0, name="n")
        assert provenance_ancestors(n) == []

    def test_single_parent(self):
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp())
        ancestors = provenance_ancestors(td)
        assert len(ancestors) == 1
        assert isinstance(ancestors[0], ParentInfo)
        assert ancestors[0].name == "base"

    def test_chain_of_ancestors(self, full_provenance_mode):
        """Function plus base → transform are all broadcast ancestors."""
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=td)
        ancestors = provenance_ancestors(result)
        # result → Function and result → td → base.
        assert len(ancestors) == 3
        assert isinstance(ancestors[0], ParentInfo)
        assert ancestors[0].parent is wf
        assert isinstance(ancestors[1], ParentInfo)
        assert ancestors[1].parent is td
        assert isinstance(ancestors[2], ParentInfo)
        assert ancestors[2].parent is base

    def test_no_duplicates(self, full_provenance_mode):
        """Same parent appearing in multiple roles doesn't duplicate."""
        n = Normal(loc=0.0, scale=1.0, name="shared")

        def add(x: float, y: float) -> float:
            return x + y

        wf = Function(func=add, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n, y=n)
        ancestors = provenance_ancestors(result)
        # The Function comes first; n appears as both args but is deduplicated.
        assert len(ancestors) == 2
        assert isinstance(ancestors[0], ParentInfo)
        assert ancestors[0].parent is wf
        assert ancestors[1].parent is n

    def test_diamond_no_duplicates(self):
        """Shared ancestor in a diamond DAG appears exactly once.

        Diamond shape:
              X
             / \\
            A   B
             \\ /
              C
        A and B are both derived from X. C is derived from both A and B.
        provenance_ancestors(C) should contain X exactly once.
        """
        X = Normal(loc=0.0, scale=1.0, name="X")
        A = Normal(loc=0.0, scale=1.0, name="A")
        B = Normal(loc=0.0, scale=1.0, name="B")
        C = Normal(loc=0.0, scale=1.0, name="C")

        A.with_provenance(Provenance.create("op", parents=[X]))
        B.with_provenance(Provenance.create("op", parents=[X]))
        C.with_provenance(Provenance.create("op", parents=[A, B]))

        ancestors = provenance_ancestors(C)
        names = [a.name for a in ancestors]
        assert names.count("X") == 1, f"X appeared {names.count('X')} times: {names}"
        assert set(names) == {"A", "B", "X"}


# ===========================================================================
# 9. provenance_dag
# ===========================================================================


def _count_dag_entries(dot) -> tuple[int, int]:
    """Count (nodes, edges) in a graphviz Digraph via its body list."""
    nodes = sum(1 for line in dot.body if "->" not in line and " [" in line)
    edges = sum(1 for line in dot.body if "->" in line)
    return nodes, edges


class TestProvenanceDag:
    @pytest.fixture(autouse=True)
    def _require_graphviz(self):
        # provenance_dag() builds a graphviz.Digraph; skip this class's
        # tests when the optional graphviz package is not installed.
        pytest.importorskip("graphviz")

    def test_basic_dag_has_correct_node_and_edge_count(self):
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")
        dag = provenance_dag(td)
        # Two distributions -> 2 nodes, 1 transform edge.
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 2
        assert num_edges == 1
        # Cross-check: structural ancestor set must agree.
        ancestors = provenance_ancestors(td)
        assert len(ancestors) == 1
        assert isinstance(ancestors[0], ParentInfo)
        assert ancestors[0].name == "base"

    def test_no_provenance_single_node(self):
        n = Normal(loc=0.0, scale=1.0, name="alone")
        dag = provenance_dag(n)
        # Single distribution with no parents -> 1 node, 0 edges.
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 1
        assert num_edges == 0
        assert provenance_ancestors(n) == []

    def test_multi_step_dag_structure(self, full_provenance_mode):
        base = Normal(loc=0.0, scale=1.0, name="prior")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=td)
        dag = provenance_dag(result)
        # result <- Function and result <- td <- base: 4 nodes, 3 edges.
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 4
        assert num_edges == 3
        # Ancestor chain must contain base (root); all nodes are ParentInfo in FULL mode.
        ancestors = provenance_ancestors(result)
        assert any(a.parent is wf for a in ancestors)
        assert any(a.parent is td for a in ancestors)
        assert any(a.parent is base for a in ancestors)

    def test_plain_inputs_are_not_dag_ancestors(self):
        wf = Function(func=lambda x: x + 1)

        result = wf(jnp.asarray(2.0))

        ancestors = provenance_ancestors(result)
        assert [ancestor.name for ancestor in ancestors] == [wf.name]
        dag = provenance_dag(result)
        assert _count_dag_entries(dag) == (2, 1)

    def test_diamond_dag_no_duplicate_nodes(self):
        """Shared ancestor in a diamond renders as a single node, not two."""
        X = Normal(loc=0.0, scale=1.0, name="X")
        A = Normal(loc=0.0, scale=1.0, name="A")
        B = Normal(loc=0.0, scale=1.0, name="B")

        A.with_provenance(Provenance.create("op", parents=[X]))
        B.with_provenance(Provenance.create("op", parents=[X]))

        # Build a joint whose provenance has both A and B as parents.
        # We test on A or B themselves since provenance_dag only accepts
        # a single root; verify via ancestors that X is deduplicated.
        # For the dag test, we use A since it's the simpler traversal.
        # The real diamond requires a root C; approximate by checking
        # ancestors of a manually-sourced C.
        C = Normal(loc=0.0, scale=1.0, name="C")
        C.with_provenance(Provenance.create("op", parents=[A, B]))

        dag = provenance_dag(C)
        # C + A + B + X = 4 nodes. X must appear only once despite
        # being reachable via both A and B.
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 4, f"Expected 4 nodes, got {num_nodes}"
        # Edges: C←A, C←B, A←X, B←X = 4 edges
        assert num_edges == 4, f"Expected 4 edges, got {num_edges}"


# ===========================================================================
# 10. ProvenanceMode behaviour
# ===========================================================================


class TestProvenanceModes:
    def test_lightweight_is_default(self):
        assert probpipe.provenance_config.mode is ProvenanceMode.LIGHTWEIGHT

    def test_lightweight_stores_parent_info(self):
        """LIGHTWEIGHT mode stores ParentInfo descriptors, not live refs."""
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        assert result.provenance is not None
        assert len(result.provenance.parents) == 2
        function_parent, parent = result.provenance.parents
        assert function_parent.type_name == "Function"
        assert function_parent.name == "identity"
        assert isinstance(parent, ParentInfo)
        assert parent.type_name == "Normal"
        assert parent.name == "input"

    def test_lightweight_parent_info_not_live_ref(self):
        """In LIGHTWEIGHT mode, parents are not live Distribution references."""
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        parent = result.provenance.parents[1]
        assert parent is not n

    def test_lightweight_plain_input_has_no_live_ref(self):
        value = jnp.asarray([1.0, 2.0])

        provenance = Provenance.create("op", inputs={"value": value})

        assert provenance is not None
        info = provenance.inputs["value"]
        assert info.name is None
        assert info.parent is None
        assert info.fingerprint is not None

    def test_lightweight_ancestors_returns_parentinfo(self):
        """In LIGHTWEIGHT mode provenance_ancestors returns ParentInfo descriptors.

        The DAG is preserved via ParentInfo.provenance, but live objects are not
        held — ParentInfo.parent is None.
        """
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        ancestors = provenance_ancestors(result)
        assert len(ancestors) == 2
        assert isinstance(ancestors[0], ParentInfo)
        assert [ancestor.name for ancestor in ancestors] == ["identity", "input"]
        assert ancestors[0].parent is None

    def test_lightweight_dag_includes_function_and_input(self):
        """provenance_dag() shows leaf, Function, and input in LIGHTWEIGHT mode.

        Parent nodes are descriptors (no live objects), so the DAG retains both
        direct parents without keeping either live reference.
        """
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        dag = provenance_dag(result)
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 3
        assert num_edges == 2

    def test_off_mode_no_provenance(self):
        """OFF mode attaches no provenance to workflow results."""
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        assert result.provenance is None

    def test_mode_setter_rejects_non_enum(self):
        with pytest.raises(TypeError, match="ProvenanceMode"):
            probpipe.provenance_config.mode = "full"

    def test_reset_restores_lightweight(self):
        probpipe.provenance_config.mode = ProvenanceMode.FULL
        probpipe.provenance_config.reset()
        assert probpipe.provenance_config.mode is ProvenanceMode.LIGHTWEIGHT

    def test_lightweight_parent_is_garbage_collected(self):
        """LIGHTWEIGHT mode does not prevent the parent from being GC'd."""
        import gc
        import weakref

        parent = Normal(loc=0.0, scale=1.0, name="parent")
        child = Normal(loc=0.0, scale=1.0, name="child")
        child.with_provenance(Provenance.create("op", parents=[parent]))

        ref = weakref.ref(parent)
        del parent
        gc.collect()

        assert ref() is None, "LIGHTWEIGHT provenance should not hold a strong ref to the parent"

    def test_full_mode_parent_is_retained(self, full_provenance_mode):
        """FULL mode keeps the parent alive via ParentInfo.parent."""
        import gc
        import weakref

        parent = Normal(loc=0.0, scale=1.0, name="parent")
        child = Normal(loc=0.0, scale=1.0, name="child")
        child.with_provenance(Provenance.create("op", parents=[parent]))

        ref = weakref.ref(parent)
        del parent
        gc.collect()

        assert ref() is not None, "FULL mode should retain the parent via ParentInfo.parent"

    def test_full_mode_plain_input_is_retained(self, full_provenance_mode):
        value = jnp.asarray([1.0, 2.0])

        provenance = Provenance.create("op", inputs={"value": value})

        assert provenance is not None
        assert provenance.inputs["value"].parent is value

    def test_off_mode_ancestors_empty(self):
        """In OFF mode provenance_ancestors returns an empty list."""
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        assert provenance_ancestors(result) == []

    def test_off_mode_dag_single_node(self):
        """In OFF mode provenance_dag renders only the root node with no edges."""
        pytest.importorskip("graphviz")
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = Function(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        dag = provenance_dag(result)
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 1
        assert num_edges == 0
