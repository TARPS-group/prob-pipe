"""Contract tests of joint: composition after realigning the right factor's field names."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from probpipe import ApplicabilityError
from probpipe.distributions._batches import DistributionBatch
from probpipe.distributions._conditional import ConditionalDistribution
from probpipe.distributions._factored import FactoredDistribution, _is_named
from probpipe.operations import RouteSource
from probpipe.operations._joint import joint

from ._laws import Gaussian, Kernel


class TestJoint:
    def test_without_renames_joint_is_composition(self):
        likelihood, prior = Kernel("y", ("slope",)), Gaussian("slope")
        result = joint(likelihood, prior)
        assert isinstance(result, FactoredDistribution)
        assert result.event_spec == (likelihood * prior).event_spec

    def test_the_composed_law_carries_the_declaration_composition_derives(self):
        likelihood, prior = Kernel("y", ("slope",)), Gaussian("slope")
        assert joint.check(likelihood, prior).result is None

    def test_a_rename_connects_a_producer_to_the_slot_its_consumer_names(self):
        likelihood, prior = Kernel("y", ("slope",)), Gaussian("beta")
        assert isinstance(joint(likelihood, prior), ConditionalDistribution)
        realigned = joint(likelihood, prior, beta="slope")
        assert isinstance(realigned, FactoredDistribution)
        assert set(realigned.event_spec.components) == {"y", "slope"}

    def test_the_joint_is_labeled_as_composition_labels_it(self):
        likelihood, prior = Kernel("y", ("slope",)), Gaussian("beta")
        assert joint(likelihood, prior, beta="slope").label == (likelihood * prior).label

    def test_the_joint_is_unlabeled_so_it_reads_factor_by_factor(self):
        likelihood, prior = Kernel("y", ("slope",)), Gaussian("beta")
        result = joint(likelihood, prior, beta="slope")
        assert _is_named(result) is False
        assert str(result) == result.notation == "y(y | slope)·beta(slope)"

    def test_joint_equals_composition_with_the_renamed_right_factor(self):
        likelihood, prior = Kernel("y", ("slope",)), Gaussian("beta")
        assert (
            joint(likelihood, prior, beta="slope").event_spec
            == (likelihood * prior.with_path_names(beta="slope")).event_spec
        )

    def test_a_factor_that_is_not_a_distribution_raises_applicability_error(self):
        with pytest.raises(ApplicabilityError, match="'B' accepts"):
            joint(Kernel(), jnp.zeros(2))

    def test_a_batch_of_laws_is_consumed_as_an_object_rather_than_swept(self):
        laws = DistributionBatch("laws", [Gaussian("mu"), Gaussian("mu", 1.0)], "laws")
        with pytest.raises(ApplicabilityError, match=r"'B' accepts.*DistributionBatch"):
            joint(Kernel(), laws)

    def test_the_route_is_structural_and_exact(self):
        (route,) = joint.summary().routes
        assert (route.name, route.source, route.exact) == ("compose", RouteSource.STRUCTURAL, True)

    def test_a_kernel_on_the_right_is_realigned_too(self):
        upstream = Kernel("mu", ("alpha",))
        realigned = joint(Kernel("y", ("m",)), upstream, mu="m")
        assert set(realigned.given_spec) == {"alpha"}
