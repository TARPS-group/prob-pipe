"""Exact conditioning of a joint whose prior is an empirical law, by reweighting its atoms.

An empirical law puts its weight on finitely many atoms, so Bayes' rule
applies to it exactly: the posterior has the same atoms, and each atom's
weight is its prior weight times the likelihood of the given values at the
atom, renormalized. A particle filter's update step is this computation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .._weights import Weights
from ..core._dispatch import Feasibility
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._specs import OutputSpec
from ..distributions._capabilities import SupportsConditionalLogProb
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..operations._condition import InferenceMethod
from ._approximate_distribution import _record_run
from ._inference_utils import (
    flat_record,
    flat_unflatten,
    is_jax_traceable,
    model_factors,
    parameter_given,
)

__all__: list[str] = []


def _flat_atoms(prior: EmpiricalDistribution) -> jnp.ndarray:
    """The atoms of *prior* as one array ``(atoms, d)`` in the layout :func:`flat_unflatten` reads.

    Raises
    ------
    TypeError
        If an atom has a leaf that is not numeric.
    """
    count = prior.num_atoms
    rows = prior._rows
    record = flat_record(prior)
    if record is None:
        columns = [rows] if not isinstance(rows, dict) else list(rows.values())
    else:
        columns = [rows[path] for path in record]
    return jnp.concatenate(
        [jnp.reshape(jnp.asarray(column), (count, -1)) for column in columns], -1
    )


def _target_atoms(prior: EmpiricalDistribution, declaration: OutputSpec) -> Any:
    """The atoms of *prior* packaged as the target's *declaration* packages its event.

    A prior over a record keeps its atoms, whose fields are the declaration's
    components. A prior over one array, whose component the declaration exposes
    as a field, becomes a one-field record batch on one level.
    """
    expected = prior.event_spec
    if declaration.exposes_record == expected.exposes_record and tuple(
        declaration.components
    ) == tuple(expected.components):
        return prior.atoms
    (component,) = declaration.components
    levels = prior.atoms.level_names
    level = levels[0] if len(levels) == 1 else "atom"
    return NumericRecordBatch(prior.label, {component: prior._rows}, (level,), axes_per_level=(1,))


class EmpiricalReweightingMethod(InferenceMethod):
    """Exact Bayes' rule for a joint whose prior is an empirical law, registered as ``empirical_reweighting``.

    The target is the unnormalized conditional of a factored joint at observed
    values, whose prior, the joint of the factors that produce no observed
    field, is an :class:`~probpipe.EmpiricalDistribution` over numeric atoms,
    and whose likelihood claims a normalized conditional log-density. The
    posterior is the empirical law on the prior's atoms whose weights are the
    prior's multiplied by the likelihood of the observed values at each atom,
    so the method is exact. The likelihood is evaluated at every atom in one
    ``jax.vmap`` when it traces, and one atom at a time otherwise.
    """

    _method_options = ()

    @property
    def name(self) -> str:
        return "empirical_reweighting"

    @property
    def exact(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return 100

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    def check(self, target: Any, /, **kwargs: Any) -> Feasibility:
        """Whether the target's prior is an empirical law over numeric atoms and its likelihood has a density."""
        factors = model_factors(target)
        if factors is None:
            return Feasibility(
                False,
                "Requires a factored joint at observed values of the fields a likelihood produces",
            )
        if not isinstance(factors.prior, EmpiricalDistribution):
            return Feasibility(
                False, f"Requires an empirical prior; got {type(factors.prior).__name__}"
            )
        likelihood = factors.likelihood
        if not isinstance(likelihood, ConditionalDistribution) or not isinstance(
            likelihood, SupportsConditionalLogProb
        ):
            return Feasibility(
                False, "Requires a likelihood that claims SupportsConditionalLogProb"
            )
        try:
            _flat_atoms(factors.prior)
        except (TypeError, ValueError) as error:
            return Feasibility(False, f"Requires numeric atoms: {error}")
        return Feasibility(True)

    def execute(self, target: Any, /, **kwargs: Any) -> EmpiricalDistribution:
        """The prior's atoms, each weighted by its prior weight times the likelihood there.

        Raises
        ------
        ValueError
            If the observed values have zero likelihood at every atom.
        """
        self._check_options(kwargs)
        factors = model_factors(target)
        prior, likelihood = factors.prior, factors.likelihood
        unflatten = flat_unflatten(prior)

        def log_likelihood(theta: jnp.ndarray) -> jnp.ndarray:
            given = parameter_given(factors, unflatten(theta))
            return likelihood._conditional_log_prob(given, factors.observed)

        atoms = _flat_atoms(prior)
        if is_jax_traceable(log_likelihood, atoms[0]):
            values = jax.vmap(log_likelihood)(atoms)
        else:
            values = jnp.stack([jnp.asarray(log_likelihood(theta)) for theta in atoms])
        log_weights = jnp.log(jnp.asarray(prior.weights)) + jnp.reshape(values, (-1,))
        if not bool(jnp.any(jnp.isfinite(log_weights))):
            raise ValueError(
                f"the observed values have zero likelihood at every atom of {prior.label!r}, "
                f"so the posterior is undefined"
            )
        weights = Weights(log_weights=log_weights)
        result = EmpiricalDistribution(
            "posterior",
            _target_atoms(prior, target.event_spec),
            weights,
            event_spec=target.event_spec,
        )
        return _record_run(
            result,
            (target,),
            self.name,
            num_atoms=prior.num_atoms,
            effective_sample_size=float(weights.effective_sample_size),
        )
