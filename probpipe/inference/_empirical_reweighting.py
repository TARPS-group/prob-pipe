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
from ..core._repr import format_outputs
from ..core._specs import OutputSpec, _has_record_outputs
from ..custom_types import PRNGKey
from ..distributions._capabilities import SupportsConditionalLogProb
from ..distributions._conditional import ConditionalDistribution
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution
from ..operations._condition import InferenceMethod
from ..operations._inference_target import (
    InferenceTarget,
    _described,
    _flat_record_spec,
    _flat_unflatten,
)
from ._approximate_distribution import _record_run
from ._inference_utils import (
    model_factors,
    parameter_given,
    traces_in_jax,
    unfactored_model_reason,
)

__all__: list[str] = []


def _flat_atoms(prior: EmpiricalDistribution) -> jnp.ndarray:
    """The atoms of *prior* as one array ``(atoms, d)`` in the flat layout of the prior's draws.

    Parameters
    ----------
    prior : EmpiricalDistribution
        The empirical prior.

    Returns
    -------
    jax.Array
        One row per atom, which concatenates the atom's flattened leaves.

    Raises
    ------
    TypeError
        If an atom has a leaf that is not numeric.
    """
    count = prior.num_atoms
    rows = prior._rows
    record = _flat_record_spec(prior)
    if record is None:
        columns = [rows] if not isinstance(rows, dict) else list(rows.values())
    else:
        columns = [rows[path] for path in record]
    return jnp.concatenate(
        [jnp.reshape(jnp.asarray(column), (count, -1)) for column in columns], -1
    )


def _target_atoms(prior: EmpiricalDistribution, declaration: OutputSpec) -> Any:
    """The atoms of *prior* in the form the target's output spec *declaration* gives its event.

    A prior over a record keeps its atoms, whose fields are the outputs of
    *declaration*. A prior over one array, whose output *declaration* names
    as a field, becomes a one-field record batch on one level.
    """
    expected = prior.event_spec
    if _has_record_outputs(declaration) == _has_record_outputs(expected) and tuple(
        declaration.outputs
    ) == tuple(expected.outputs):
        return prior.atoms
    (output_name,) = declaration.outputs
    levels = prior.atoms.level_names
    level = levels[0] if len(levels) == 1 else "atom"
    return NumericRecordBatch(
        {output_name: prior._rows},
        (level,),
        axes_per_level=(1,),
        label=prior.label,
    )


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
    def description(self) -> str:
        return "Bayes' rule on an empirical prior, which reweights its atoms by the likelihood."

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
            return Feasibility(False, unfactored_model_reason(target))
        if not isinstance(factors.prior, EmpiricalDistribution):
            return Feasibility(
                False,
                f"the prior must be an EmpiricalDistribution; got {_described(factors.prior)}",
            )
        likelihood = factors.likelihood
        if not isinstance(likelihood, ConditionalDistribution) or not isinstance(
            likelihood, SupportsConditionalLogProb
        ):
            return Feasibility(
                False,
                "the likelihood must be a ConditionalDistribution with a conditional "
                f"log-density (SupportsConditionalLogProb); got {_described(likelihood)}",
            )
        try:
            _flat_atoms(factors.prior)
        except (TypeError, ValueError) as error:
            return Feasibility(False, f"the prior's atoms must be numeric: {error}")
        return Feasibility(True)

    def execute(
        self, target: InferenceTarget, /, *, key: PRNGKey, **options: Any
    ) -> EmpiricalDistribution:
        """The prior's atoms, each weighted by its prior weight times the likelihood there.

        Parameters
        ----------
        target : InferenceTarget
            The unnormalized conditional of a factored joint at observed values,
            which ``check`` admits.
        key : PRNGKey
            The run's random key, which the method does not read, since it draws nothing.
        **options : Any
            The call's ``method_options``, of which the method reads none.

        Returns
        -------
        EmpiricalDistribution
            The posterior labeled ``posterior``, whose provenance records the
            atom count and the effective sample size.

        Raises
        ------
        ValueError
            If the observed values have zero likelihood at every atom.
        """
        self._check_options(options)
        factors = model_factors(target)
        prior, likelihood = factors.prior, factors.likelihood
        unflatten = _flat_unflatten(prior)

        def log_likelihood(theta: jnp.ndarray) -> jnp.ndarray:
            given = parameter_given(factors, unflatten(theta))
            return likelihood._conditional_log_prob(given, factors.observed)

        atoms = _flat_atoms(prior)
        if traces_in_jax(likelihood):
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
        atoms = _target_atoms(prior, target.event_spec)
        # The posterior's atoms are labeled by its outputs, as an inference
        # method's are.
        atoms_label = format_outputs(target.event_spec.outputs)
        if atoms.label != atoms_label:
            atoms = atoms.with_label(atoms_label)
        result = EmpiricalDistribution(
            atoms, weights, label="posterior", event_spec=target.event_spec
        )
        return _record_run(
            result,
            (target,),
            self.name,
            num_atoms=prior.num_atoms,
            effective_sample_size=float(weights.effective_sample_size),
        )
