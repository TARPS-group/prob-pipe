"""PyMCModel: wraps PyMC models as ProbPipe distributions.

Inference is handled by registered methods in ``probpipe.inference``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import jax.numpy as jnp

from ..core.distribution import Distribution
from ..core.record import NumericRecordTemplate
from ..custom_types import Array
from ..inference._approximate_distribution import ApproximateDistribution
from ._base import ProbabilisticModel

logger = logging.getLogger(__name__)

__all__ = ["PyMCModel"]


def _to_numpy(value: Any) -> Any:
    """Coerce JAX / Record-leaf arrays to numpy for PyMC compatibility.

    PyMC's PyTensor backend doesn't multiply tensor variables against
    raw JAX arrays; numpy arrays work transparently. Pass-through for
    values that don't expose ``__array__`` (e.g. plain Python ints).
    """
    import numpy as _np
    if hasattr(value, "__array__"):
        return _np.asarray(value)
    return value


class PyMCModel(ProbabilisticModel):
    """PyMC model wrapper.

    Wraps a PyMC model-building function as a ProbPipe
    :class:`ProbabilisticModel`.

    Parameters
    ----------
    model_fn : callable
        Function that takes ``**observed`` keyword arguments and
        returns a ``pymc.Model`` context.  Example::

            def my_model(y=None):
                with pm.Model() as m:
                    mu = pm.Normal("mu", 0, 1)
                    sigma = pm.HalfNormal("sigma", 1)
                    pm.Normal("y", mu, sigma, observed=y)
                return m
    name : str or None
        Model name for provenance.

    Raises
    ------
    ImportError
        If ``pymc`` is not installed.
    """

    def __init__(
        self,
        model_fn: Callable[..., Any],
        *,
        name: str | None = None,
    ):
        try:
            import pymc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pymc is required for PyMCModel. "
                "Install it with: pip install pymc"
            ) from e

        self._model_fn = model_fn
        # ``Distribution`` metaclass requires a non-empty name; default
        # to the class name when the caller doesn't supply one.
        self._name = name if name else "PyMCModel"

        # Discover observed variable names from the model function signature.
        # Parameters with default value None are treated as observed variables
        # that will receive data at conditioning time.
        import inspect

        sig = inspect.signature(model_fn)
        self._observed_names = tuple(
            name
            for name, p in sig.parameters.items()
            if p.default is None
        )

        # Build the model once without data to discover free parameters.
        # When called with no args, observed vars become free RVs.
        self._unconditioned_model = model_fn()
        all_free = {rv.name for rv in self._unconditioned_model.free_RVs}
        observed_set = set(self._observed_names)
        self._param_names = tuple(
            rv.name
            for rv in self._unconditioned_model.free_RVs
            if rv.name not in observed_set
        )

    # -- Distribution interface ---------------------------------------------

    def _param_rvs(self, model: Any) -> Iterator[tuple[str, Any]]:
        """Yield ``(name, rv)`` for each free parameter, in
        ``_param_names`` order, looked up in *model*.

        Parameters
        ----------
        model : pymc.Model
            A build of this model (conditioned or not). Its free
            (non-observed) random variables must be exactly the set
            frozen as ``_param_names`` at construction.

        Raises
        ------
        ValueError
            If *model*'s set of free (non-observed) random variables
            differs from ``_param_names`` (frozen from the no-data build
            at construction) in either direction — a canonical parameter
            missing from this build, or a new free RV introduced by it.
            ProbPipe does not support models whose random-variable set
            changes with the data (dynamic random variables); see
            https://github.com/TARPS-group/prob-pipe/issues/232.
        """
        free_rvs = {rv.name: rv for rv in model.free_RVs}
        # Reject additive dynamic RVs: a free RV in this build that was
        # not a free parameter at construction (and isn't an observed
        # name). Without this it would be silently dropped from both the
        # template and the chain (extraction filters to _param_names),
        # so the user's parameter would vanish from the posterior.
        extra = set(free_rvs) - set(self._param_names) - set(self._observed_names)
        if extra:
            raise ValueError(
                f"PyMC build introduced free random variable(s) "
                f"{sorted(extra)} not present in the model built at "
                f"construction. ProbPipe does not support models whose "
                f"set of free random variables changes with the data "
                f"(dynamic random variables); the parameter set must be "
                f"fixed across builds, with only per-variable shapes "
                f"allowed to depend on data size. See "
                f"https://github.com/TARPS-group/prob-pipe/issues/232."
            )
        for name in self._param_names:
            # Reject subtractive dynamic RVs: a canonical parameter that
            # this build dropped.
            if name not in free_rvs:
                raise ValueError(
                    f"PyMC random variable {name!r} is present in the "
                    f"model built at construction (no data) but absent "
                    f"from this build. ProbPipe does not support models "
                    f"whose set of free random variables changes with the "
                    f"data (dynamic random variables); the parameter set "
                    f"must be fixed across builds, with only per-variable "
                    f"shapes allowed to depend on data size. See "
                    f"https://github.com/TARPS-group/prob-pipe/issues/232."
                )
            yield name, free_rvs[name]

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Total number of scalar free parameters (observed excluded).

        Derived from :attr:`record_template` so the flat size and the
        per-field template can never disagree; consequently this raises
        on a non-concrete free-RV shape, just as the template does.
        """
        return (self.record_template.flat_size,)

    def _record_template_for(self, model: Any) -> NumericRecordTemplate:
        """Parameter template built from a specific PyMC model build.

        Inference paths call this with the data-conditioned model they
        construct and pass the result through to :func:`make_posterior`,
        so the posterior carries Record structure (``mean(post)`` returns
        a ``NumericRecord`` keyed by the PyMC RV names) and the template
        matches the chain even for RVs whose shape depends on data size
        (e.g. per-observation random effects, ``pm.Normal('alpha', 0, 1,
        shape=X.shape[0])``). The public :attr:`record_template` property
        uses the declared (no-data) build instead.

        Scalar PyMC RVs (e.g. ``pm.Normal('intercept', 0, 1)``) become
        fields with event shape ``()``; shape-:math:`k` RVs (e.g.
        ``pm.Normal('beta', 0, 1, shape=k)``) become fields with event
        shape ``(k,)``.

        Parameters
        ----------
        model : pymc.Model
            A build of this model (typically the data-conditioned build
            returned by :meth:`_pymc_model`) to introspect for free-RV
            shapes.

        Returns
        -------
        NumericRecordTemplate
            One field per free parameter (observed variables excluded),
            keyed by RV name and carrying its event shape.

        Raises
        ------
        ValueError
            If any free RV has a non-concrete (``None``) dimension in
            its ``type.shape``. The record-template machinery requires
            concrete shapes — silently dropping a ``None`` dim would
            produce an under-shaped template and confusing downstream
            errors. Also raised if *model*'s free-RV set differs from the
            construction-time parameter set in either direction — a
            canonical parameter missing, or a new RV introduced (dynamic
            random variables are unsupported; see :meth:`_param_rvs`).
        """
        fields: dict[str, tuple[int, ...]] = {}
        for name, rv in self._param_rvs(model):
            raw_shape = tuple(rv.type.shape)
            if any(s is None for s in raw_shape):
                raise ValueError(
                    f"PyMC RV {name!r} has a non-concrete shape "
                    f"{raw_shape}; PyMCModel templates require every free "
                    f"RV to have a fully concrete event shape. Specify the "
                    f"shape explicitly when declaring the RV (e.g. "
                    f"`pm.Normal({name!r}, 0, 1, shape=k)`)."
                )
            fields[name] = tuple(int(s) for s in raw_shape)
        return NumericRecordTemplate(**fields)

    @property
    def record_template(self) -> NumericRecordTemplate:
        """Declared parameter template from the no-data build done at
        construction (see :meth:`_record_template_for`).

        For RVs whose shape depends on data size, the conditioned shapes
        are resolved at inference time via :meth:`_record_template_for`;
        this property cannot know them without data.
        """
        return self._record_template_for(self._unconditioned_model)

    # -- Named components interface ------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        return self._param_names + self._observed_names

    def __getitem__(self, key: str) -> Any:
        if key in self._param_names or key in self._observed_names:
            return key  # placeholder — PyMC doesn't expose sub-distributions easily
        raise KeyError(f"Unknown component: {key!r}")

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._param_names

    # -- Sampling (prior predictive) ----------------------------------------

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        """Prior predictive sampling via PyMC."""
        import pymc as pm

        n = 1
        for s in sample_shape:
            n *= s

        model = self._model_fn()
        with model:
            prior = pm.sample_prior_predictive(samples=max(n, 1))

        # Concatenate parameter values into a single array
        arrays = []
        for name in self._param_names:
            vals = prior.prior[name].values.reshape(n, -1)
            arrays.append(jnp.asarray(vals))
        samples = jnp.concatenate(arrays, axis=-1)

        if sample_shape == ():
            return samples[0]
        return samples.reshape(*sample_shape, -1)

    # -- PyMC model access (for nutpie integration) -------------------------

    def _pymc_model(self, data: Any = None) -> Any:
        """Build a PyMC model, optionally with data.

        Three accepted input forms for ``data``:

        * ``None`` — build the unconditioned model (used at
          construction time to discover free RVs).
        * ``dict`` or ``Record`` (incl. ``RecordArray``) — unpack the
          fields named by ``_observed_names`` and pass them as keyword
          arguments to the model function. This is the canonical
          multi-observed-variable path: provenance tracks every named
          input rather than reading covariates from a closure.
        * Bare array — pass as the first observed variable. Only
          unambiguous when the model has exactly one observed name.

        Array-typed values are coerced to numpy before being passed to
        the user's model function — PyMC's tensor backend doesn't
        multiply with raw JAX arrays.

        The returned model is stateless and not retained on ``self``;
        callers that need the conditioned shapes (e.g. for
        :meth:`_record_template_for`) pass this build back explicitly.
        """
        if data is None:
            return self._model_fn()
        if isinstance(data, dict):
            return self._model_fn(**{k: _to_numpy(v) for k, v in data.items()})
        # Local import to avoid a modeling→core cycle at module load.
        from ..core.record import Record
        if isinstance(data, Record):
            return self._model_fn(**{
                name: _to_numpy(data[name])
                for name in self._observed_names
                if name in data.fields
            })
        return self._model_fn(**{self._observed_names[0]: _to_numpy(data)})

    def __repr__(self) -> str:
        params = ", ".join(self._param_names)
        return f"PyMCModel(params=[{params}])"
