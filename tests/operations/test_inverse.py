"""Contract tests of inverse and log_det_jacobian, whose capabilities the value layer declares."""

from __future__ import annotations

import pytest

from probpipe.operations import RouteSource
from probpipe.operations._inverse import inverse, log_det_jacobian
from probpipe.values import Function


def test_each_carries_one_capability_route():
    for op in (inverse, log_det_jacobian):
        (route,) = op.summary().routes
        assert (route.name, route.source, route.exact) == ("exact", RouteSource.CAPABILITY, True)


@pytest.mark.pending(reason="the value layer defines SupportsInverse and is_invertible")
def test_the_inverse_of_a_map_is_a_function():
    assert isinstance(
        inverse(
            Function(
                lambda x: 2.0 * x,
                label="double",
            )
        ),
        Function,
    )


@pytest.mark.pending(reason="the value layer defines SupportsLogDetJacobian")
def test_the_log_determinant_of_a_linear_map_is_constant():
    assert (
        float(
            log_det_jacobian(
                Function(
                    lambda x: 2.0 * x,
                    label="double",
                ),
                1.0,
            )
        )
        > 0.0
    )
