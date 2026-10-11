"""Execution, step 7 of the stack (design V.9).

The selected route runs under one of the four dispatch modes and optionally
under orchestration, which is off by default. Because keys attach to structure,
the modes agree up to the effects of evaluation order. An unsupported mode is
refused before sampling, and a route's own error propagates without another
route being tried.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import Function, OutputSpec, WorkflowKind, function, workflow_run

from ._design_helpers import atom_leaves, record_law, standard_normal


def _affine(z):
    return 2.0 * z + 1.0


class TestDispatchModes:
    def test_the_modes_agree_on_a_lifted_call(self):
        law = standard_normal()
        results = {}
        for dispatch in ("jax", "sequential", "thread"):
            wrapped = Function(
                _affine,
                n_broadcast_samples=8,
                dispatch=dispatch,
                label="affine",
                output_spec=OutputSpec(affine=None),
            )
            with workflow_run(seed=5):
                results[dispatch] = atom_leaves(wrapped(law))[0]

        np.testing.assert_allclose(results["jax"], results["sequential"], rtol=1e-6)
        np.testing.assert_allclose(results["thread"], results["sequential"], rtol=1e-6)

    def test_auto_runs_a_body_that_does_not_trace(self):
        @function(n_broadcast_samples=6, dispatch="auto", output_spec=OutputSpec(untraceable=None))
        def untraceable(z):
            return jnp.asarray(float(z) ** 2)

        with workflow_run(seed=5):
            result = untraceable(standard_normal())

        assert result.num_atoms == 6

    def test_auto_agrees_with_the_mode_it_selects(self):
        law = standard_normal()
        with workflow_run(seed=5):
            auto = Function(
                _affine,
                n_broadcast_samples=8,
                dispatch="auto",
                label="affine",
                output_spec=OutputSpec(affine=None),
            )(law)
        with workflow_run(seed=5):
            jax_mode = Function(
                _affine,
                n_broadcast_samples=8,
                dispatch="jax",
                label="affine",
                output_spec=OutputSpec(affine=None),
            )(law)

        np.testing.assert_allclose(atom_leaves(auto)[0], atom_leaves(jax_mode)[0], rtol=1e-6)

    def test_jax_maps_an_enumeration_as_sequential_dispatch_evaluates_it(self):
        mapped = Function(
            _affine,
            n_broadcast_samples=64,
            dispatch="jax",
            label="affine",
            output_spec=OutputSpec(affine=None),
        )
        sequential = Function(
            _affine,
            n_broadcast_samples=64,
            dispatch="sequential",
            label="affine",
            output_spec=OutputSpec(affine=None),
        )

        result = mapped(record_law()["a"])

        assert result.provenance.metadata["dispatch"] == "jax"
        np.testing.assert_allclose(
            atom_leaves(result)[0], atom_leaves(sequential(record_law()["a"]))[0], rtol=1e-6
        )

    def test_an_unsupported_mode_is_refused_before_sampling(self):
        @function(n_broadcast_samples=64, dispatch="jax")
        def untraceable(z):
            return jnp.asarray(float(z) ** 2)

        with workflow_run(seed=5), pytest.raises(ValueError, match="dispatch='jax'"):
            untraceable(record_law()["a"])


class TestOrchestration:
    def test_orchestration_is_off_by_default(self):
        assert (
            Function(
                _affine,
                label="affine",
            ).effective_workflow_kind
            is WorkflowKind.OFF
        )


class TestFailures:
    @pytest.mark.parametrize("dispatch", ["sequential", "thread", "auto"])
    def test_the_routes_own_error_propagates(self, dispatch):
        class Diverged(RuntimeError):
            pass

        def diverge(z):
            raise Diverged("the iteration diverged")

        wrapped = Function(
            diverge,
            n_broadcast_samples=6,
            dispatch=dispatch,
            label="diverge",
        )

        with workflow_run(seed=5), pytest.raises(Diverged):
            wrapped(standard_normal())
