"""PyMCModel: wraps PyMC models as ProbPipe distributions.

Inference is handled by registered methods in ``probpipe.inference``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import jax.numpy as jnp

from ..core.distribution import Distribution
from ..core.record import NumericEventTemplate
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
                "pymc is required for PyMCModel. Install it with: pip install pymc"
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
            name for name, p in sig.parameters.items() if p.default is None
        )

        # Build the model once without data to discover free parameters.
        # When called with no args, observed vars become free RVs.
        self._unconditioned_model = model_fn()
        observed_set = set(self._observed_names)
        self._param_names = tuple(
            rv.name for rv in self._unconditioned_model.free_RVs if rv.name not in observed_set
        )

    # -- Distribution interface ---------------------------------------------

    def _param_rvs(
        self,
        model: Any,
        names: tuple[str, ...] | list[str],
    ) -> Iterator[tuple[str, Any]]:
        """Yield ``(name, rv)`` for each name in *names*, looked up in
        *model*'s free RVs, in the given order.

        Parameters
        ----------
        model : pymc.Model
            A build of this model.
        names : sequence of str
            Free-RV names to yield, in order. Every name must be a free
            RV of *model*.
        """
        free_rvs = {rv.name: rv for rv in model.free_RVs}
        for name in names:
            yield name, free_rvs[name]

    def _conditioned_param_names(self, model: Any) -> tuple[str, ...]:
        """Free-parameter names to infer for a data-conditioned *model*.

        Returns the canonical parameters (``_param_names``, frozen at
        construction) plus any observed names left free in this build —
        observed variables the caller did not supply data for, which are
        then inferred (**partial conditioning**). Supplied observed
        variables are observed (not free) and excluded.

        Raises
        ------
        ValueError
            If *model*'s free-RV set differs from the construction-time
            parameter set by a **non-observed** RV — a canonical parameter
            the build dropped, or a new non-observed free RV it
            introduced. ProbPipe does not support models whose
            non-observed random-variable set changes with the data
            (dynamic random variables); see
            https://github.com/TARPS-group/prob-pipe/issues/232.
        """
        free = {rv.name for rv in model.free_RVs}
        missing = [n for n in self._param_names if n not in free]
        if missing:
            raise ValueError(
                f"PyMC random variable(s) {sorted(missing)} present in the "
                f"model built at construction (no data) but absent from "
                f"this build. ProbPipe does not support models whose set "
                f"of free random variables changes with the data (dynamic "
                f"random variables); the parameter set must be fixed across "
                f"builds, with only per-variable shapes allowed to depend "
                f"on data size. See "
                f"https://github.com/TARPS-group/prob-pipe/issues/232."
            )
        extra = free - set(self._param_names) - set(self._observed_names)
        if extra:
            raise ValueError(
                f"PyMC build introduced free random variable(s) "
                f"{sorted(extra)} not present in the model built at "
                f"construction. ProbPipe does not support models whose set "
                f"of free random variables changes with the data (dynamic "
                f"random variables); the parameter set must be fixed across "
                f"builds, with only per-variable shapes allowed to depend "
                f"on data size. See "
                f"https://github.com/TARPS-group/prob-pipe/issues/232."
            )
        # Partial conditioning: include observed names left free.
        omitted_observed = tuple(n for n in self._observed_names if n in free)
        return tuple(self._param_names) + omitted_observed

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Total number of scalar free parameters (observed excluded).

        Derived from :attr:`event_template`; raises on a non-concrete
        free-RV shape, as the template does.
        """
        return (self.event_template.flat_size,)

    def _event_template_for(
        self,
        model: Any,
        names: tuple[str, ...] | list[str],
    ) -> NumericEventTemplate:
        """Parameter template over *names*, read from a PyMC *model* build.

        Inference passes the data-conditioned build and the names from
        :meth:`_conditioned_param_names`; the :attr:`event_template`
        property passes the no-data build and ``_param_names``. Scalar
        PyMC RVs become fields with event shape ``()``; shape-:math:`k`
        RVs become fields with event shape ``(k,)``.

        Parameters
        ----------
        model : pymc.Model
            A build of this model to introspect for free-RV shapes.
        names : sequence of str
            Free-RV names to include, in field order.

        Returns
        -------
        NumericEventTemplate
            One field per name, carrying its event shape.

        Raises
        ------
        ValueError
            If any named free RV has a non-concrete (``None``) dimension
            in its ``type.shape``.
        """
        fields: dict[str, tuple[int, ...]] = {}
        for name, rv in self._param_rvs(model, names):
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
        return NumericEventTemplate(**fields)

    @property
    def event_template(self) -> NumericEventTemplate:
        """Declared parameter template from the no-data build (canonical
        parameters, observed variables excluded).

        Data-dependent shapes, and any observed variable left free under
        partial conditioning, are resolved at inference time via
        :meth:`_event_template_for`; this property reflects neither.
        """
        return self._event_template_for(
            self._unconditioned_model,
            self._param_names,
        )

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
            prior = pm.sample_prior_predictive(draws=max(n, 1))

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

        The returned model is not retained on ``self``; callers that need
        the conditioned shapes pass it back explicitly.
        """
        if data is None:
            return self._model_fn()
        if isinstance(data, dict):
            return self._model_fn(**{k: _to_numpy(v) for k, v in data.items()})
        # Local import to avoid a modeling→core cycle at module load.
        from ..core.record import Record

        if isinstance(data, Record):
            return self._model_fn(
                **{
                    name: _to_numpy(data[name])
                    for name in self._observed_names
                    if name in data.fields
                }
            )
        return self._model_fn(**{self._observed_names[0]: _to_numpy(data)})

    def __repr__(self) -> str:
        params = ", ".join(self._param_names)
        return f"PyMCModel(params=[{params}])"
