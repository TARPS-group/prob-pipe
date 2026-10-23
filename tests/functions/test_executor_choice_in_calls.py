"""Execution, step 7 of the stack (design V.9).

The selected route runs its evaluation points on an executor and optionally
under orchestration, which is off by default. A function JAX can trace is
mapped by ``jax``, and a function that claims ``traceable=False`` runs one
evaluation point at a time. Because keys attach to structure, the executors
agree up to the effects of evaluation order. Naming ``jax`` for a function JAX
cannot trace is refused, and a route's own error ends the call without another
route being tried.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EvaluationError,
    Function,
    OutputSpec,
    ResolutionError,
    WorkflowKind,
    function,
    workflow_run,
)

from ._design_helpers import atom_leaves, record_law, standard_normal


def _affine(z):
    return 2.0 * z + 1.0


class TestExecutorChoice:
    def test_the_executors_agree_on_a_lifted_call(self):
        law = standard_normal()
        results = {}
        executors = {}
        for executor, traceable in (("jax", None), ("sequential", False), ("thread", False)):
            wrapped = Function(
                _affine,
                max_evaluations=8,
                traceable=traceable,
                label="affine",
                output_spec=OutputSpec(affine=None),
            ).using(executor=executor)
            with workflow_run(seed=5):
                result = wrapped(law)
            results[executor] = atom_leaves(result)[0]
            executors[executor] = result.provenance.metadata["evaluations"][0]["executor"]

        assert executors == {"jax": "jax", "sequential": "sequential", "thread": "thread"}
        np.testing.assert_allclose(results["jax"], results["sequential"], rtol=1e-6)
        np.testing.assert_allclose(results["thread"], results["sequential"], rtol=1e-6)

    def test_a_body_that_does_not_trace_runs_one_draw_at_a_time(self):
        @function(max_evaluations=6, output_spec=OutputSpec(untraceable=None))
        def untraceable(z):
            return jnp.asarray(float(z) ** 2)

        with workflow_run(seed=5):
            result = untraceable(standard_normal())

        assert result.num_atoms == 6
        assert result.provenance.metadata["evaluations"][0]["executor"] == "sequential"

    def test_the_default_executor_agrees_with_jax(self):
        law = standard_normal()
        with workflow_run(seed=5):
            default = Function(
                _affine,
                max_evaluations=8,
                label="affine",
                output_spec=OutputSpec(affine=None),
            )(law)
        with workflow_run(seed=5):
            jax_executor = Function(
                _affine,
                max_evaluations=8,
                label="affine",
                output_spec=OutputSpec(affine=None),
            ).using(executor="jax")(law)

        assert default.provenance.metadata["evaluations"][0]["executor"] == "jax"
        np.testing.assert_allclose(atom_leaves(default)[0], atom_leaves(jax_executor)[0], rtol=1e-6)

    def test_jax_maps_an_enumeration_as_the_one_at_a_time_path_evaluates_it(self):
        mapped = Function(
            _affine,
            max_evaluations=64,
            label="affine",
            output_spec=OutputSpec(affine=None),
        ).using(executor="jax")
        sequential = Function(
            _affine,
            max_evaluations=64,
            traceable=False,
            label="affine",
            output_spec=OutputSpec(affine=None),
        ).using(executor="sequential")

        result = mapped(record_law()["a"])

        assert result.provenance.metadata["evaluations"][0]["executor"] == "jax"
        np.testing.assert_allclose(
            atom_leaves(result)[0], atom_leaves(sequential(record_law()["a"]))[0], rtol=1e-6
        )

    def test_jax_is_refused_for_a_body_jax_cannot_trace(self):
        @function(max_evaluations=64)
        def untraceable(z):
            return jnp.asarray(float(z) ** 2)

        with (
            workflow_run(seed=5),
            pytest.raises(ResolutionError, match="executor 'jax' cannot run"),
        ):
            untraceable.using(executor="jax")(record_law()["a"])


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
    @pytest.mark.parametrize(
        ("executor", "traceable"),
        [("sequential", False), ("thread", False), (None, None)],
        ids=["sequential", "thread", "default"],
    )
    def test_the_routes_own_error_ends_the_call(self, executor, traceable):
        class Diverged(RuntimeError):
            pass

        def diverge(z):
            raise Diverged("the iteration diverged")

        wrapped = Function(
            diverge,
            max_evaluations=6,
            traceable=traceable,
            label="diverge",
        )
        if executor is not None:
            wrapped = wrapped.using(executor=executor)

        with (
            workflow_run(seed=5),
            pytest.raises(EvaluationError, match="the iteration diverged") as excinfo,
        ):
            wrapped(standard_normal())

        assert isinstance(excinfo.value.__cause__, Diverged)
