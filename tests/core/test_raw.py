"""Every kind defines ``raw()``, its representation detached from the workflow (design II.4)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import (
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    FunctionBatch,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    Opaque,
    OpaqueBatch,
    Record,
    RecordBatch,
    RecordSpec,
)
from probpipe.core.provenance import Provenance
from probpipe.core.tracked import TrackedTerm
from probpipe.distributions._batches import _element_source
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._distribution import Distribution

from ..operations._laws import Gaussian, Kernel


class TestTheContract:
    def test_raw_is_abstract_on_tracked_term(self):
        assert "raw" in TrackedTerm.__abstractmethods__

    def test_a_kind_that_does_not_define_raw_cannot_be_constructed(self):
        class Bare(TrackedTerm):
            def __init__(self):
                self._init_tracked("bare")

        with pytest.raises(TypeError, match="raw"):
            Bare()


class TestValues:
    def test_an_array_is_its_stored_array(self):
        stored = jnp.arange(3.0)
        assert NumericArray("x", stored).raw() is stored

    def test_an_opaque_value_is_the_wrapped_value(self):
        wrapped = ("Rubin (1981)", "SAT points")
        assert Opaque("source", wrapped).raw() is wrapped

    def test_a_record_is_the_nested_mapping_of_its_raw_leaves(self):
        effect = jnp.asarray(28.0)
        school = Record("school", {"data": {"effect": effect, "se": 15.0}, "label": "A"})
        raw = school.raw()
        assert type(raw) is dict and list(raw) == ["data", "label"]
        assert raw["data"]["effect"] is effect and raw["label"] == "A"

    def test_a_record_path_is_one_node(self):
        effect = jnp.asarray(28.0)
        school = Record("school", {"data": {"effect": effect, "se": 15.0}, "label": "A"})
        assert school.raw("data/effect") is effect
        assert school.raw(("data", "effect")) is effect
        assert list(school.raw("data")) == ["effect", "se"]

    def test_a_record_path_that_is_not_one_raises_key_error(self):
        with pytest.raises(KeyError):
            Record("r", x=1.0).raw("y")

    def test_a_law_held_as_a_leaf_is_detached(self):
        law = Normal("theta", 0.0, 1.0)
        record = Record("r", law=law, x=1.0)
        assert isinstance(record.raw("law"), tfd.Normal)


class TestBatches:
    def test_an_array_batch_is_its_stored_array(self):
        stored = jnp.zeros((4, 2))
        batch = NumericArrayBatch("x", stored, "draw", element_spec=NumericArraySpec((2,)))
        assert batch.raw() is stored

    def test_a_record_batch_is_the_nested_mapping_of_its_columns(self):
        effects = jnp.arange(3.0)
        batch = RecordBatch(
            "schools",
            {"data/effect": effects, "label": np.array(["A", "B", "C"], dtype=object)},
            "school",
            element_spec=RecordSpec({"data/effect": (), "label": Opaque("l", "A").spec}),
        )
        raw = batch.raw()
        assert list(raw) == ["data", "label"]
        assert raw["data"]["effect"] is effects
        assert raw["label"].dtype == object and list(raw["label"]) == ["A", "B", "C"]

    def test_an_object_batch_is_its_frozen_object_array(self):
        batch = OpaqueBatch("labels", ["north", "south"], "site")
        raw = batch.raw()
        assert raw.dtype == object and list(raw) == ["north", "south"]
        assert not raw.flags.writeable

    def test_a_function_batch_is_the_object_array_of_its_callables(self):
        def double(x):
            return 2 * x

        assert FunctionBatch("f", [double], "variant").raw()[0] is double

    def test_a_batch_of_laws_is_the_object_array_of_the_stored_laws(self):
        law = Gaussian("g", 1.0)
        raw = DistributionBatch("laws", [law, Gaussian("g", 2.0)], "law").raw()
        assert raw.dtype == object and raw[0] is law

    def test_a_sub_batch_is_a_view_of_the_same_store(self):
        stored = jnp.arange(6.0)
        batch = NumericArrayBatch("x", stored, "draw", element_spec=NumericArraySpec(()))
        np.testing.assert_array_equal(batch[2:4].raw(), stored[2:4])


class TestFunctions:
    def test_a_function_is_its_wrapped_callable(self):
        def add(x, y):
            return x + y

        assert Function("add", add).raw() is add


class TestDistributions:
    def test_a_law_is_itself_detached(self):
        law = Gaussian("g", 1.0).with_provenance(Provenance.create("made", parents=[]))
        object.__setattr__(law, "_annotations", {"diagnostics": {"n_eff": 42}})
        detached = law.raw()
        assert isinstance(detached, Distribution) and detached is not law
        assert (detached.label, detached.spec) == (law.label, law.spec)
        assert detached.provenance is None and detached.annotations is None
        assert detached.loc == 1.0

    def test_a_batch_element_drops_its_container(self):
        element = DistributionBatch("laws", [Gaussian("g"), Gaussian("g", 2.0)], "law")[1]
        assert _element_source(element) is not None
        assert _element_source(element.raw()) is None

    def test_an_empirical_law_keeps_its_atoms(self):
        law = EmpiricalDistribution("e", jnp.arange(4.0))
        detached = law.raw()
        assert isinstance(detached, EmpiricalDistribution)
        np.testing.assert_array_equal(detached.atoms.values, law.atoms.values)

    def test_the_backend_adapter_is_its_backend_distribution(self):
        assert isinstance(Normal("x", 0.0, 1.0).raw(), tfd.Normal)

    def test_a_kernel_is_itself_detached(self):
        kernel = Kernel("k")
        detached = kernel.raw()
        assert isinstance(detached, ConditionalDistribution) and detached is not kernel
        assert (detached.label, detached.spec) == (kernel.label, kernel.spec)
        assert detached.provenance is None
