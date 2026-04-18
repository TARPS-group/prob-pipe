"""SequentialJointDistribution --- autoregressive joint distribution.

Components can be :class:`Distribution` instances (roots) or callables
that receive previously-sampled values and return a ``Distribution``
(conditionals).
"""

from __future__ import annotations

import inspect
from types import MappingProxyType

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from ..core.distribution import (
    NumericRecordDistribution,
    _mc_expectation,
)
from ..core._record_distribution import RecordDistribution, _build_record_template
from ..core.record import Record
from ..core.provenance import Provenance
from ..core.protocols import (
    SupportsConditioning,
    SupportsLogProb,
    SupportsSampling,
)
from ._joint_utils import (
    KeyPath,
    _parse_condition_args,
)


class SequentialJointDistribution(RecordDistribution, SupportsSampling, SupportsLogProb, SupportsConditioning):
    """
    Joint distribution with autoregressive (sequential) dependence.

    Components can be :class:`Distribution` instances (roots) or callables
    that receive previously-sampled values and return a ``Distribution``
    (conditionals).

    Example::

        joint = SequentialJointDistribution(
            z=Normal(loc=0.0, scale=1.0, name="z"),
            x=lambda z: Normal(loc=z, scale=0.5, name="x"),
            y=lambda z, x: Normal(loc=z + x, scale=0.1, name="y"),
        )

    Callable signatures are inspected: parameter names must match earlier
    component names.

    Parameters
    ----------
    name : str, optional
        Distribution name.
    **components : Distribution or Callable[..., Distribution]
        Named components in topological (dependency) order.
    """

    _sampling_cost = "medium"
    _preferred_orchestration = None

    def __init__(
        self,
        *,
        name: str | None = None,
        **components: NumericRecordDistribution | callable,
    ):
        if not components:
            raise ValueError("SequentialJointDistribution requires at least one component.")

        self._raw_components: dict[str, NumericRecordDistribution | callable] = dict(components)
        if name is None:
            name = "sequential(" + ",".join(components.keys()) + ")"
        super().__init__(name=name)
        self._conditioned_names: frozenset[str] = frozenset()
        self._conditioned_values: dict[str, Array] = {}
        self._sampleable_error: str | None = None
        # Map callable component names to their dependency parameter names
        self._callable_parents: dict[str, tuple[str, ...]] = {}

        # Validate ordering: callable args must reference earlier names
        seen: list[str] = []
        for cname, comp in self._raw_components.items():
            if callable(comp) and not isinstance(comp, NumericRecordDistribution):
                params = list(inspect.signature(comp).parameters.keys())
                for p in params:
                    if p not in seen:
                        raise ValueError(
                            f"Component '{cname}' depends on '{p}', which "
                            f"is not defined before it. "
                            f"Available: {seen}"
                        )
                self._callable_parents[cname] = tuple(params)
            seen.append(cname)

        # Do a prototype forward pass to determine component distributions
        # and compute event shapes / slices
        proto_key = jax.random.PRNGKey(0)
        proto_structured = self._sample_sequential(proto_key, ())
        self._proto_components: dict[str, NumericRecordDistribution] = {}

        resolved: dict[str, NumericRecordDistribution] = {}
        for cname, comp in self._raw_components.items():
            if isinstance(comp, NumericRecordDistribution):
                resolved[cname] = comp
            else:
                # Resolve the callable with zero-valued parents to get shape info
                parent_vals = {}
                for prev_name in list(self._raw_components.keys()):
                    if prev_name == cname:
                        break
                    parent_vals[prev_name] = proto_structured[prev_name]
                sig = inspect.signature(comp)
                call_kw = {p: parent_vals[p] for p in sig.parameters if p in parent_vals}
                resolved[cname] = comp(**call_kw)
        self._proto_components = resolved

        # Build _components dict from resolved prototypes (for shape introspection)
        self._components = resolved
        self._record_template = _build_record_template(self._components)

    @staticmethod
    def _compute_sampleable_error(
        conditioned_names: frozenset[str],
        callable_parents: dict[str, tuple[str, ...]],
    ) -> str | None:
        """Return an error message if sampling is impossible, else None.

        A conditioned non-root component is sampleable only if all of its
        parents are also conditioned.  Root components have no parents, so
        conditioning on them is always sampleable.
        """
        bad = []
        for cname in conditioned_names:
            parents = callable_parents.get(cname, ())
            unconditioned = [p for p in parents if p not in conditioned_names]
            if unconditioned:
                bad.append((cname, unconditioned))
        if not bad:
            return None
        details = "; ".join(
            f"'{c}' has unconditioned parent(s) {ps}" for c, ps in bad
        )
        return (
            f"Cannot sample from this SequentialJointDistribution: "
            f"{details}. Forward sampling would draw these ancestors "
            f"from the prior, not the posterior. Condition on the "
            f"parent(s) as well, or use log_prob() to evaluate the "
            f"density."
        )

    def _check_sampleable(self) -> None:
        """Raise if sampling is not possible due to non-root conditioning."""
        if self._sampleable_error is not None:
            raise NotImplementedError(self._sampleable_error)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in topological (insertion) order."""
        return tuple(self._components.keys())

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-component event shapes from component distributions."""
        return {k: v.event_shape for k, v in self._components.items()}

    # flatten_value / unflatten_value inherited from RecordDistribution

    @property
    def components(self):
        """Read-only view of the component distributions."""
        return MappingProxyType(self._components)

    def _sample_sequential(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...],
    ) -> dict[str, Array]:
        """Sample components sequentially, feeding earlier samples to later callables.

        Returns a dict of *all* components (including conditioned ones).
        Callers that expose results externally should filter to unconditioned.
        """
        self._check_sampleable()
        keys = jax.random.split(key, len(self._raw_components))
        sampled: dict[str, Array] = {}

        for subkey, (cname, comp) in zip(keys, self._raw_components.items()):
            if cname in self._conditioned_values:
                # Conditioned component: broadcast fixed value to sample_shape
                val = self._conditioned_values[cname]
                sampled[cname] = jnp.broadcast_to(val, sample_shape + val.shape)
            elif isinstance(comp, NumericRecordDistribution):
                # Root distribution: sample with sample_shape
                sampled[cname] = comp._sample(subkey, sample_shape)
            else:
                # Conditional: callable receives batched parent samples,
                # returning a batched distribution.  Sample with () since
                # the batch is already in the distribution's batch_shape.
                sig = inspect.signature(comp)
                call_kw = {p: sampled[p] for p in sig.parameters if p in sampled}
                dist = comp(**call_kw)
                sampled[cname] = dist._sample(subkey)

        return sampled

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ):
        from ..core._record_array import NumericRecordArray
        full = self._sample_sequential(key, sample_shape)
        fields = {k: v for k, v in full.items() if k not in self._conditioned_names}
        if sample_shape:
            return NumericRecordArray(
                fields, batch_shape=sample_shape,
                template=self.record_template,
            )
        return Record(fields)

    def _eval_log_prob(self, value, *, components: str) -> Array:
        """Evaluate log-density over selected components.

        Parameters
        ----------
        value : dict[str, ArrayLike] or Record
            Dict of unconditioned component values.
        components : ``"all"`` or ``"unconditioned"``
            Which components to include in the sum.  ``"all"`` sums over
            every component (conditioned ones evaluated at their observed
            values), giving the unnormalized conditional.
            ``"unconditioned"`` sums only over unconditioned components
            (with conditioned values plugged in as parents), giving the
            normalized conditional when the Markov structure permits it.
        """
        if isinstance(value, Record):
            value = value.to_dict()
        structured = {k: jnp.asarray(v) for k, v in value.items()}

        # Add conditioned values so callables can receive them
        for cname, val in self._conditioned_values.items():
            structured[cname] = val

        total = None
        for cname, comp in self._raw_components.items():
            if components == "unconditioned" and cname in self._conditioned_names:
                continue
            val = structured[cname]
            if isinstance(comp, NumericRecordDistribution):
                lp = comp._log_prob(val)
            else:
                sig = inspect.signature(comp)
                call_kw = {p: structured[p] for p in sig.parameters if p in structured}
                cond_dist = comp(**call_kw)
                lp = cond_dist._log_prob(val)
            total = lp if total is None else total + lp

        return total

    def _log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Evaluate the normalized log-density.

        For an unconditioned joint, this is the full joint log p(x).

        After conditioning on a set of components whose parents are all
        also conditioned (i.e., the conditioned set forms a root
        sub-graph), the Markov structure makes the normalized conditional
        log p(unconditioned | conditioned) computable by summing only
        the unconditioned components' log-densities with conditioned
        values substituted for their parents.

        Raises ``NotImplementedError`` when conditioning makes the
        normalizing constant intractable (e.g., conditioning on a leaf
        whose parents are unconditioned).
        """
        if self._sampleable_error is not None:
            raise NotImplementedError(
                "log_prob is not available for this conditioned "
                "SequentialJointDistribution because the normalizing "
                "constant is intractable.  Use unnormalized_log_prob "
                "instead."
            )
        return self._eval_log_prob(value, components="unconditioned")

    def _unnormalized_log_prob(self, value: dict[str, ArrayLike]) -> Array:
        """Evaluate the (possibly unnormalized) log-density.

        For an unconditioned joint, this equals the full joint log p(x).
        After conditioning, this sums log-densities of *all* components
        (conditioned ones evaluated at their observed values), giving
        log p(unconditioned, conditioned=values), which is proportional
        to log p(unconditioned | conditioned=values).
        """
        return self._eval_log_prob(value, components="all")

    def _mean(self) -> Record:
        """Per-component means (approximate --- uses prototype components).

        For sequential joints, the true marginal mean is not simply the
        per-component mean because later components depend on earlier
        samples.  This returns the prototype (prior-evaluated) means
        as an approximation.
        """
        return Record({k: v._mean() for k, v in self._proto_components.items()})

    def _variance(self) -> Record:
        """Per-component variances (approximate --- uses prototype components)."""
        return Record({k: v._variance() for k, v in self._proto_components.items()})

    def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
        return _mc_expectation(self, f, key=key, num_evaluations=num_evaluations, return_dist=return_dist)

    def _condition_on(self, observed=None, /, **kwargs):
        observed_leaves = _parse_condition_args(self, observed, kwargs)
        return self._condition_on_impl(observed_leaves)

    def _condition_on_impl(
        self, observed_leaves: dict[KeyPath, ArrayLike],
    ) -> "SequentialJointDistribution":
        """Condition on observed component values.

        The resulting distribution is sampleable as long as every conditioned
        non-root component has all of its parents also conditioned (so that no
        unconditioned ancestor would be drawn from the prior instead of the
        posterior).  Root components have no parents, so conditioning on them
        is always sampleable.  If the sampleability condition is violated, a
        :class:`NotImplementedError` is raised at sample time.
        ``log_prob()`` always works regardless (returns the unnormalized
        conditional log-density).

        .. note::

            ``SequentialJointDistribution`` only supports **flat dicts**
            (no nesting).  All key paths must be length-1 (top-level
            component names).
        """
        # Enforce flat-only: all key paths must be length 1
        for path in observed_leaves:
            if len(path) != 1:
                raise TypeError(
                    f"SequentialJointDistribution only supports flat "
                    f"(non-nested) components.  Cannot condition on "
                    f"nested key path {path!r}."
                )
        # Convert {("x",): val} -> {"x": val}
        observed = {path[0]: val for path, val in observed_leaves.items()}

        all_conditioned = self._conditioned_names | frozenset(observed)
        if len(all_conditioned) >= len(self._raw_components):
            raise ValueError(
                "Cannot condition on all component distributions --- "
                "at least one must remain unconditioned."
            )

        result = SequentialJointDistribution.__new__(SequentialJointDistribution)
        result._raw_components = dict(self._raw_components)  # originals unchanged
        result._name = self._name
        result._proto_components = dict(self._proto_components)
        result._callable_parents = self._callable_parents
        result._conditioned_names = all_conditioned
        result._conditioned_values = {
            **self._conditioned_values,
            **{k: jnp.asarray(v, dtype=jnp.float32) for k, v in observed.items()},
        }
        result._sampleable_error = self._compute_sampleable_error(
            result._conditioned_names, result._callable_parents,
        )

        # Expose only unconditioned components
        unconditioned = {
            k: v for k, v in self._proto_components.items()
            if k not in result._conditioned_names
        }
        result._components = unconditioned
        result._record_template = _build_record_template(unconditioned)

        result.with_source(Provenance(
            "condition_on", parents=(self,),
            metadata={"conditioned": list(observed)},
        ))
        return result

    def __repr__(self) -> str:
        parts = []
        for k, v in self._raw_components.items():
            if isinstance(v, NumericRecordDistribution):
                parts.append(f"{k}={type(v).__name__}")
            else:
                parts.append(f"{k}=<callable>")
        comp_str = ", ".join(parts)
        name_str = f", name='{self._name}'" if self._name else ""
        return f"SequentialJointDistribution({comp_str}{name_str})"
