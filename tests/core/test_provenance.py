"""Comprehensive tests for provenance tracking across all distribution operations."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

import probpipe
from probpipe import (
    Beta,
    JointGaussian,
    Normal,
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
from probpipe.core.node import WorkflowFunction
from probpipe.core.provenance import ParentInfo

# ===========================================================================
# 1. Provenance basics (dataclass, with_source, write-once)
# ===========================================================================


class TestProvenanceBasics:
    def test_provenance_fields(self):
        p = Provenance("test_op", metadata={"key": "val"})
        assert p.operation == "test_op"
        assert p.parents == ()
        assert p.metadata == {"key": "val"}

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
        n.with_source(Provenance("first"))
        with pytest.raises(RuntimeError, match="already set"):
            n.with_source(Provenance("second"))

    def test_with_source_none_is_noop_distribution(self):
        """with_source(None) leaves a Distribution unchanged."""
        n = Normal(loc=0.0, scale=1.0, name="n")
        result = n.with_source(None)
        assert result is n
        assert n.source is None

    def test_with_source_none_is_noop_record(self):
        """with_source(None) leaves a Record unchanged."""
        from probpipe import Record

        r = Record({"x": jnp.array(1.0)})
        result = r.with_source(None)
        assert result is r
        assert r.source is None


# ===========================================================================
# 2. ParentInfo hash and equality
# ===========================================================================


class TestParentInfoHashEq:
    def test_hashable_without_source(self):
        """Root ParentInfo (source=None) is hashable."""
        p = ParentInfo(type_name="Normal", name="prior")
        assert isinstance(hash(p), int)

    def test_hashable_with_source(self):
        """Non-root ParentInfo (source set) is hashable despite Provenance.metadata."""
        src = Provenance("op", metadata={"key": "val"})
        p = ParentInfo(type_name="Gamma", name="mid", source=src)
        assert isinstance(hash(p), int)

    def test_usable_in_set(self):
        """set(provenance_ancestors(d)) does not raise TypeError."""
        X = Normal(loc=0.0, scale=1.0, name="X")
        Y = Normal(loc=0.0, scale=1.0, name="Y")
        Y.with_source(Provenance.create("op", parents=[X]))
        ancestors = provenance_ancestors(Y)
        s = set(ancestors)
        assert len(s) == 1

    def test_equal_same_fields(self):
        """Two ParentInfo with identical fields compare equal."""
        src = Provenance("op")
        a = ParentInfo(type_name="Normal", name="x", source=src)
        b = ParentInfo(type_name="Normal", name="x", source=src)
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
        a = ParentInfo(type_name="Normal", name="x", obj=n)
        b = ParentInfo(type_name="Normal", name="x", obj=None)
        assert a == b

    def test_hash_contract(self):
        """Equal ParentInfo must have equal hashes."""
        src = Provenance("op", metadata={"k": 1})
        a = ParentInfo(type_name="Normal", name="x", source=src)
        b = ParentInfo(type_name="Normal", name="x", source=src)
        assert a == b
        assert hash(a) == hash(b)


# ===========================================================================
# 3. from_distribution provenance
# ===========================================================================


class TestFromDistributionProvenance:
    def test_normal_from_distribution(self):
        src = Beta(alpha=2.0, beta=5.0, name="beta_src")
        converted = from_distribution(src, Normal)
        assert converted.source is not None
        assert converted.source.operation == "from_distribution"
        assert len(converted.source.parents) == 1
        assert isinstance(converted.source.parents[0], ParentInfo)
        assert converted.source.parents[0].name == "beta_src"

    def test_empirical_from_distribution(self):
        src = Normal(loc=0.0, scale=1.0, name="norm_src")
        ed = from_distribution(src, RecordEmpiricalDistribution, n_samples=100)
        assert ed.source is not None
        assert ed.source.operation == "from_distribution"
        assert len(ed.source.parents) == 1
        assert isinstance(ed.source.parents[0], ParentInfo)
        assert ed.source.parents[0].name == "norm_src"


# ===========================================================================
# 3. TransformedDistribution provenance
# ===========================================================================


class TestTransformedDistributionProvenance:
    def test_transform_provenance_attached(self):
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp())
        assert td.source is not None
        assert td.source.operation == "transform"
        assert len(td.source.parents) == 1
        assert isinstance(td.source.parents[0], ParentInfo)
        assert td.source.parents[0].name == "base"
        assert td.source.metadata["bijector"] == "Exp"

    def test_transform_chain_provenance(self):
        base = Normal(loc=0.0, scale=1.0, name="base")
        bij = tfb.Chain([tfb.Exp(), tfb.Shift(1.0)])
        td = TransformedDistribution(base, bij)
        assert td.source.metadata["bijector"] == "Chain"

    def test_transform_with_empirical_base(self):
        ed = RecordEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="x")
        td = TransformedDistribution(ed, tfb.Exp())
        assert td.source is not None
        assert td.source.operation == "transform"
        assert len(td.source.parents) == 1
        assert isinstance(td.source.parents[0], ParentInfo)
        assert td.source.parents[0].name == "x"


# ===========================================================================
# 4. Conditioning provenance
# ===========================================================================


class TestConditioningProvenance:
    def test_product_condition_on(self):
        joint = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=1.0, scale=2.0, name="y"),
        )
        cond = condition_on(joint, x=jnp.array(0.0))
        assert cond.source is not None
        assert cond.source.operation == "condition_on"
        assert len(cond.source.parents) == 1
        assert isinstance(cond.source.parents[0], ParentInfo)
        assert "x" in cond.source.metadata["conditioned"]

    def test_sequential_condition_on(self):
        seq = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
        )
        cond = condition_on(seq, z=jnp.array(1.0))
        assert cond.source.operation == "condition_on"
        assert len(cond.source.parents) == 1
        assert isinstance(cond.source.parents[0], ParentInfo)

    def test_gaussian_condition_on(self):
        jg = JointGaussian(
            mean=jnp.zeros(2),
            cov=jnp.eye(2),
            x=1,
            y=1,
        )
        cond = condition_on(jg, x=jnp.array([0.0]))
        assert cond.source.operation == "condition_on"
        assert "x" in cond.source.metadata["conditioned"]


# ===========================================================================
# 5. Broadcasting provenance
# ===========================================================================


class TestBroadcastingProvenance:
    def test_broadcast_loop_provenance(self, full_provenance_mode):
        n = Normal(loc=0.0, scale=1.0, name="input_normal")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(x=n)
        assert hasattr(result, "samples")
        assert result.source is not None
        assert result.source.operation == "broadcast"
        assert len(result.source.parents) == 1
        assert isinstance(result.source.parents[0], ParentInfo)
        assert result.source.parents[0].obj is n
        assert result.source.metadata["dispatch"] == "sequential"
        assert result.source.metadata["n_samples"] == 20
        assert result.source.metadata["func"] == "identity"
        assert result.source.metadata["broadcast_args"] == ["x"]

    def test_broadcast_jax_provenance(self):
        n = Normal(loc=0.0, scale=1.0, name="jax_input")

        def double(x: float) -> float:
            return 2.0 * x

        wf = WorkflowFunction(func=double, dispatch="jax", n_broadcast_samples=20, seed=42)
        result = wf(x=n)
        assert hasattr(result, "samples")
        assert result.source is not None
        assert result.source.operation == "broadcast"
        assert result.source.metadata["dispatch"] == "jax"

    def test_broadcast_multiple_parents(self):
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = Normal(loc=1.0, scale=0.5, name="b")

        def add(x: float, y: float) -> float:
            return x + y

        wf = WorkflowFunction(func=add, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(x=a, y=b)
        assert result.source is not None
        assert len(result.source.parents) == 2

    def test_broadcast_enumerate_provenance(self):
        """Enumeration path should also get provenance."""
        ed = RecordEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="x")
        n = Normal(loc=0.0, scale=1.0, name="n")

        def add(a: float, b: float) -> float:
            return a + b

        wf = WorkflowFunction(func=add, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(a=ed, b=n)
        assert hasattr(result, "samples")
        assert result.source is not None
        assert result.source.operation == "broadcast"


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
        assert cond.source.operation == "condition_on"
        # joint has no provenance (constructed directly)
        assert joint.source is None
        # but converted has provenance pointing to src
        assert converted.source.operation == "from_distribution"
        assert isinstance(converted.source.parents[0], ParentInfo)
        assert converted.source.parents[0].name == "prior"

    def test_transform_then_broadcast(self, full_provenance_mode):
        """transform → broadcast creates a 2-step chain."""
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def log_val(x: float) -> float:
            return jnp.log(x)

        wf = WorkflowFunction(func=log_val, dispatch="sequential", n_broadcast_samples=20, seed=42)
        result = wf(x=td)
        # result → broadcast → td → transform → base
        assert result.source.operation == "broadcast"
        parent_td = result.source.parents[0]
        assert isinstance(parent_td, ParentInfo)
        assert parent_td.obj is td
        assert parent_td.source.operation == "transform"
        assert isinstance(parent_td.source.parents[0], ParentInfo)
        assert parent_td.source.parents[0].obj is base


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

    def test_to_dict_recursive(self):
        """Recursive serialization follows provenance chains."""
        src = Normal(loc=0.0, scale=1.0, name="src")
        td = TransformedDistribution(src, tfb.Exp(), name="transformed")
        p = Provenance.create("broadcast", parents=[td])
        d = p.to_dict(recurse=True)
        # td has source (transform), which should be serialized via ParentInfo.source
        assert "source" in d["parents"][0]
        assert d["parents"][0]["source"]["operation"] == "transform"

    def test_to_dict_non_recursive(self):
        src = Normal(loc=0.0, scale=1.0, name="src")
        td = TransformedDistribution(src, tfb.Exp(), name="transformed")
        p = Provenance.create("broadcast", parents=[td])
        d = p.to_dict(recurse=False)
        assert "source" not in d["parents"][0]

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
        """base → transform → broadcast gives 2 ancestors for broadcast result."""
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=td)
        ancestors = provenance_ancestors(result)
        # result → td → base; both steps now produce ParentInfo in FULL mode.
        assert len(ancestors) == 2
        assert isinstance(ancestors[0], ParentInfo)
        assert ancestors[0].obj is td
        assert isinstance(ancestors[1], ParentInfo)
        assert ancestors[1].obj is base

    def test_no_duplicates(self, full_provenance_mode):
        """Same parent appearing in multiple roles doesn't duplicate."""
        n = Normal(loc=0.0, scale=1.0, name="shared")

        def add(x: float, y: float) -> float:
            return x + y

        wf = WorkflowFunction(func=add, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n, y=n)
        ancestors = provenance_ancestors(result)
        # n appears as both args but dedup keeps only one ParentInfo for it
        assert len(ancestors) == 1
        assert isinstance(ancestors[0], ParentInfo)
        assert ancestors[0].obj is n

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

        A.with_source(Provenance.create("op", parents=[X]))
        B.with_source(Provenance.create("op", parents=[X]))
        C.with_source(Provenance.create("op", parents=[A, B]))

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

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=td)
        dag = provenance_dag(result)
        # result <- td <- base : 3 nodes, 2 edges.
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 3
        assert num_edges == 2
        # Ancestor chain must contain base (root); all nodes are ParentInfo in FULL mode.
        ancestors = provenance_ancestors(result)
        assert any(a.obj is td for a in ancestors)
        assert any(a.obj is base for a in ancestors)

    def test_diamond_dag_no_duplicate_nodes(self):
        """Shared ancestor in a diamond renders as a single node, not two."""
        X = Normal(loc=0.0, scale=1.0, name="X")
        A = Normal(loc=0.0, scale=1.0, name="A")
        B = Normal(loc=0.0, scale=1.0, name="B")

        A.with_source(Provenance.create("op", parents=[X]))
        B.with_source(Provenance.create("op", parents=[X]))

        # Build a joint whose provenance has both A and B as parents.
        # We test on A or B themselves since provenance_dag only accepts
        # a single root; verify via ancestors that X is deduplicated.
        # For the dag test, we use A since it's the simpler traversal.
        # The real diamond requires a root C; approximate by checking
        # ancestors of a manually-sourced C.
        C = Normal(loc=0.0, scale=1.0, name="C")
        C.with_source(Provenance.create("op", parents=[A, B]))

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

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        assert result.source is not None
        assert len(result.source.parents) == 1
        parent = result.source.parents[0]
        assert isinstance(parent, ParentInfo)
        assert parent.type_name == "Normal"
        assert parent.name == "input"

    def test_lightweight_parent_info_not_live_ref(self):
        """In LIGHTWEIGHT mode, parents are not live Distribution references."""
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        parent = result.source.parents[0]
        assert parent is not n

    def test_lightweight_ancestors_returns_parentinfo(self):
        """In LIGHTWEIGHT mode provenance_ancestors returns ParentInfo descriptors.

        The DAG is preserved via ParentInfo.source, but live objects are not
        held — ParentInfo.obj is None.
        """
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        ancestors = provenance_ancestors(result)
        assert len(ancestors) == 1
        assert isinstance(ancestors[0], ParentInfo)
        assert ancestors[0].name == "input"
        assert ancestors[0].obj is None

    def test_lightweight_dag_two_nodes(self):
        """provenance_dag() shows leaf + parent node in LIGHTWEIGHT mode.

        The parent node is a ParentInfo descriptor (no live object), so the
        DAG has 2 nodes and 1 edge even though no live Distribution ref is held.
        """
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        dag = provenance_dag(result)
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 2
        assert num_edges == 1

    def test_off_mode_no_provenance(self):
        """OFF mode attaches no provenance to workflow results."""
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        assert result.source is None

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
        child.with_source(Provenance.create("op", parents=[parent]))

        ref = weakref.ref(parent)
        del parent
        gc.collect()

        assert ref() is None, "LIGHTWEIGHT provenance should not hold a strong ref to the parent"

    def test_full_mode_parent_is_retained(self, full_provenance_mode):
        """FULL mode keeps the parent alive via ParentInfo.obj."""
        import gc
        import weakref

        parent = Normal(loc=0.0, scale=1.0, name="parent")
        child = Normal(loc=0.0, scale=1.0, name="child")
        child.with_source(Provenance.create("op", parents=[parent]))

        ref = weakref.ref(parent)
        del parent
        gc.collect()

        assert ref() is not None, "FULL mode should retain the parent via ParentInfo.obj"

    def test_off_mode_ancestors_empty(self):
        """In OFF mode provenance_ancestors returns an empty list."""
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        assert provenance_ancestors(result) == []

    def test_off_mode_dag_single_node(self):
        """In OFF mode provenance_dag renders only the root node with no edges."""
        pytest.importorskip("graphviz")
        probpipe.provenance_config.mode = ProvenanceMode.OFF
        n = Normal(loc=0.0, scale=1.0, name="input")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, dispatch="sequential", n_broadcast_samples=10, seed=42)
        result = wf(x=n)
        dag = provenance_dag(result)
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 1
        assert num_edges == 0
