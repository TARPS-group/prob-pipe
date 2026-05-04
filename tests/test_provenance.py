"""Comprehensive tests for provenance tracking across all distribution operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    Normal,
    Beta,
    MultivariateNormal,
    RecordEmpiricalDistribution,
    TransformedDistribution,
    ProductDistribution,
    SequentialJointDistribution,
    JointGaussian,
    Provenance,
    provenance_ancestors,
    provenance_dag,
    condition_on,
    from_distribution,
)
from probpipe.core.node import WorkflowFunction


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


# ===========================================================================
# 2. from_distribution provenance
# ===========================================================================

class TestFromDistributionProvenance:

    def test_normal_from_distribution(self):
        src = Beta(alpha=2.0, beta=5.0, name="beta_src")
        converted = from_distribution(src, Normal)
        assert converted.source is not None
        assert converted.source.operation == "from_distribution"
        assert converted.source.parents == (src,)

    def test_empirical_from_distribution(self):
        src = Normal(loc=0.0, scale=1.0, name="norm_src")
        ed = from_distribution(src, RecordEmpiricalDistribution, n_samples=100)
        assert ed.source is not None
        assert ed.source.operation == "from_distribution"
        assert ed.source.parents == (src,)


# ===========================================================================
# 3. TransformedDistribution provenance
# ===========================================================================

class TestTransformedDistributionProvenance:

    def test_transform_provenance_attached(self):
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp())
        assert td.source is not None
        assert td.source.operation == "transform"
        assert td.source.parents == (base,)
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
        assert td.source.parents == (ed,)


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
        assert cond.source.parents == (joint,)
        assert "x" in cond.source.metadata["conditioned"]

    def test_sequential_condition_on(self):
        seq = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
        )
        cond = condition_on(seq, z=jnp.array(1.0))
        assert cond.source.operation == "condition_on"
        assert cond.source.parents == (seq,)

    def test_gaussian_condition_on(self):
        jg = JointGaussian(
            mean=jnp.zeros(2), cov=jnp.eye(2), x=1, y=1,
        )
        cond = condition_on(jg, x=jnp.array([0.0]))
        assert cond.source.operation == "condition_on"
        assert "x" in cond.source.metadata["conditioned"]


# ===========================================================================
# 5. Broadcasting provenance
# ===========================================================================

class TestBroadcastingProvenance:

    def test_broadcast_loop_provenance(self):
        n = Normal(loc=0.0, scale=1.0, name="input_normal")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, vectorize="loop",
                      n_broadcast_samples=20, seed=42)
        result = wf(x=n)
        assert hasattr(result, "samples")
        assert result.source is not None
        assert result.source.operation == "broadcast"
        assert result.source.parents == (n,)
        assert result.source.metadata["vectorize"] == "loop"
        assert result.source.metadata["n_samples"] == 20
        assert result.source.metadata["func"] == "identity"
        assert result.source.metadata["broadcast_args"] == ["x"]

    def test_broadcast_jax_provenance(self):
        n = Normal(loc=0.0, scale=1.0, name="jax_input")

        def double(x: float) -> float:
            return 2.0 * x

        wf = WorkflowFunction(func=double, vectorize="jax",
                      n_broadcast_samples=20, seed=42)
        result = wf(x=n)
        assert hasattr(result, "samples")
        assert result.source is not None
        assert result.source.operation == "broadcast"
        assert result.source.metadata["vectorize"] == "jax"

    def test_broadcast_multiple_parents(self):
        a = Normal(loc=0.0, scale=1.0, name="a")
        b = Normal(loc=1.0, scale=0.5, name="b")

        def add(x: float, y: float) -> float:
            return x + y

        wf = WorkflowFunction(func=add, vectorize="loop",
                      n_broadcast_samples=20, seed=42)
        result = wf(x=a, y=b)
        assert result.source is not None
        assert len(result.source.parents) == 2

    def test_broadcast_enumerate_provenance(self):
        """Enumeration path should also get provenance."""
        ed = RecordEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]), name="x")
        n = Normal(loc=0.0, scale=1.0, name="n")

        def add(a: float, b: float) -> float:
            return a + b

        wf = WorkflowFunction(func=add, vectorize="loop",
                      n_broadcast_samples=20, seed=42)
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
        assert converted.source.parents[0] is src

    def test_transform_then_broadcast(self):
        """transform → broadcast creates a 2-step chain."""
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def log_val(x: float) -> float:
            return jnp.log(x)

        wf = WorkflowFunction(func=log_val, vectorize="loop",
                      n_broadcast_samples=20, seed=42)
        result = wf(x=td)
        # result → broadcast → td → transform → base
        assert result.source.operation == "broadcast"
        parent_td = result.source.parents[0]
        assert parent_td is td
        assert parent_td.source.operation == "transform"
        assert parent_td.source.parents[0] is base


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
        p = Provenance("op", parents=(n,), metadata={"x": 1})
        d = p.to_dict()
        assert len(d["parents"]) == 1
        assert d["parents"][0]["type"] == "Normal"
        assert d["parents"][0]["name"] == "my_normal"

    def test_to_dict_recursive(self):
        """Recursive serialization follows provenance chains."""
        src = Normal(loc=0.0, scale=1.0, name="src")
        td = TransformedDistribution(src, tfb.Exp(), name="transformed")
        p = Provenance("broadcast", parents=(td,))
        d = p.to_dict(recurse=True)
        # td has source (transform), which should be serialized
        assert "source" in d["parents"][0]
        assert d["parents"][0]["source"]["operation"] == "transform"

    def test_to_dict_non_recursive(self):
        src = Normal(loc=0.0, scale=1.0, name="src")
        td = TransformedDistribution(src, tfb.Exp(), name="transformed")
        p = Provenance("broadcast", parents=(td,))
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
        p = Provenance("op", parents=(n,))
        d = p.to_dict()
        restored = Provenance.from_dict(d)
        # Parent info preserved in metadata
        assert restored.metadata["_parents_info"][0]["type"] == "Normal"
        assert restored.metadata["_parents_info"][0]["name"] == "n"

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
        assert ancestors[0] is base

    def test_chain_of_ancestors(self):
        """base → transform → broadcast gives 2 ancestors for broadcast result."""
        base = Normal(loc=0.0, scale=1.0, name="base")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, vectorize="loop",
                      n_broadcast_samples=10, seed=42)
        result = wf(x=td)
        ancestors = provenance_ancestors(result)
        # result → td → base
        assert len(ancestors) == 2
        assert td in ancestors
        assert base in ancestors

    def test_no_duplicates(self):
        """Same parent appearing in multiple roles doesn't duplicate."""
        n = Normal(loc=0.0, scale=1.0, name="shared")

        def add(x: float, y: float) -> float:
            return x + y

        wf = WorkflowFunction(func=add, vectorize="loop",
                      n_broadcast_samples=10, seed=42)
        result = wf(x=n, y=n)
        ancestors = provenance_ancestors(result)
        # n appears as both parents but should only be listed once
        assert ancestors.count(n) == 1


# ===========================================================================
# 9. provenance_dag
# ===========================================================================

def _count_dag_entries(dot) -> tuple[int, int]:
    """Count (nodes, edges) in a graphviz Digraph via its body list."""
    nodes = sum(1 for line in dot.body if "->" not in line and " [" in line)
    edges = sum(1 for line in dot.body if "->" in line)
    return nodes, edges


class TestProvenanceDag:

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
        assert ancestors[0] is base

    def test_no_provenance_single_node(self):
        n = Normal(loc=0.0, scale=1.0, name="alone")
        dag = provenance_dag(n)
        # Single distribution with no parents -> 1 node, 0 edges.
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 1
        assert num_edges == 0
        assert provenance_ancestors(n) == []

    def test_multi_step_dag_structure(self):
        base = Normal(loc=0.0, scale=1.0, name="prior")
        td = TransformedDistribution(base, tfb.Exp(), name="positive")

        def identity(x: float) -> float:
            return x

        wf = WorkflowFunction(func=identity, vectorize="loop",
                      n_broadcast_samples=10, seed=42)
        result = wf(x=td)
        dag = provenance_dag(result)
        # result <- td <- base : 3 nodes, 2 edges.
        num_nodes, num_edges = _count_dag_entries(dag)
        assert num_nodes == 3
        assert num_edges == 2
        # Ancestor chain must contain base (root).
        ancestors = provenance_ancestors(result)
        assert base in ancestors
        assert td in ancestors
