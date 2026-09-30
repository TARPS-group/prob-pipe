"""Contract tests of mixture, a derived operation over composition and evaluation."""

from __future__ import annotations

import pytest

from probpipe.core._specs import OutputSpec
from probpipe.distributions._distribution import DistributionSpec
from probpipe.operations._mixture import mixture

from ._laws import Gaussian, Kernel


def test_mixture_is_derived_from_its_identity():
    assert mixture.is_derived
    assert mixture.identity == "evaluate(_kernel_output_projection(K), K * mixing)"


def test_the_result_carries_the_kernel_event_declaration():
    kernel = Kernel("y", ("mu",))
    assert mixture.check(kernel, Gaussian("mu")).result == OutputSpec(
        mixture=DistributionSpec(kernel.event_spec)
    )


@pytest.mark.pending(reason="the kernel output projection and evaluate realize the identity")
def test_the_mixture_is_the_law_of_the_kernel_output():
    assert mixture(Kernel("y", ("mu",)), Gaussian("mu")).event_spec == Kernel().event_spec
