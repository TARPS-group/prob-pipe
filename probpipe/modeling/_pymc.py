"""PyMCModel: wraps PyMC models as ProbPipe distributions.

Inference is handled by registered methods in ``probpipe.inference``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Callable

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
        # Cached by ``_pymc_model`` when given data, so shape queries can
        # see data-dependent RV shapes rather than the no-data build's.
        self._last_conditioned_model: Any | None = None

    # -- Distribution interface ---------------------------------------------

    def _introspect_model(self) -> Any:
        """Model whose RV shapes drive ``event_shape`` /
        ``record_template``: the data-conditioned build when available,
        else the unconditioned one from construction.
        """
        return self._last_conditioned_model or self._unconditioned_model

    def _param_rvs(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(name, rv)`` for each free parameter, in
        ``_param_names`` order, looked up in the introspected model.
        """
        free_rvs = {rv.name: rv for rv in self._introspect_model().free_RVs}
        for name in self._param_names:
            rv = free_rvs.get(name)
            if rv is not None:
                yield name, rv

    @property
    def event_shape(self) -> tuple[int, ...]:
        # Total number of scalar parameters (excluding observed)
        total = 0
        for _name, rv in self._param_rvs():
            size = 1
            for s in rv.type.shape:
                if s is not None:
                    size *= s
            total += size
        return (total,)

    @property
    def record_template(self) -> NumericRecordTemplate:
        """Template that pairs each free PyMC parameter with its shape.

        Inference methods pass this through to :func:`make_posterior` so
        the resulting :class:`ApproximateDistribution` carries Record
        structure: ``mean(post)`` returns a ``NumericRecord`` keyed by
        the PyMC RV names, matching the field layout of any other
        ProbPipe posterior.

        Scalar PyMC RVs (e.g. ``pm.Normal('intercept', 0, 1)``) become
        fields with event shape ``()``; shape-:math:`k` RVs (e.g.
        ``pm.Normal('beta', 0, 1, shape=k)``) become fields with event
        shape ``(k,)``. Shapes that depend on the conditioning data
        (e.g. per-observation random effects, ``pm.Normal('alpha', 0,
        1, shape=X.shape[0])``) are reported correctly once the model
        has been conditioned on data.

        Raises
        ------
        ValueError
            If any free RV has a non-concrete (``None``) dimension in
            its ``type.shape``. The record-template machinery requires
            concrete shapes — silently dropping a ``None`` dim would
            produce an under-shaped template and confusing downstream
            errors.
        """
        fields: dict[str, tuple[int, ...]] = {}
        for name, rv in self._param_rvs():
            raw_shape = tuple(rv.type.shape)
            if any(s is None for s in raw_shape):
                raise ValueError(
                    f"PyMC RV {name!r} has a non-concrete shape "
                    f"{raw_shape}; PyMCModel.record_template requires "
                    f"every free RV to have a fully concrete event "
                    f"shape. Specify the shape explicitly when "
                    f"declaring the RV (e.g. "
                    f"`pm.Normal({name!r}, 0, 1, shape=k)`)."
                )
            fields[name] = tuple(int(s) for s in raw_shape)
        return NumericRecordTemplate(**fields)

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

        When ``data`` is given, the built model is cached on
        ``self._last_conditioned_model`` so later shape queries see the
        data-conditioned RV shapes.
        """
        if data is None:
            return self._model_fn()
        if isinstance(data, dict):
            model = self._model_fn(**{k: _to_numpy(v) for k, v in data.items()})
        else:
            # Local import to avoid a modeling→core cycle at module load.
            from ..core.record import Record
            if isinstance(data, Record):
                model = self._model_fn(**{
                    name: _to_numpy(data[name])
                    for name in self._observed_names
                    if name in data.fields
                })
            else:
                model = self._model_fn(
                    **{self._observed_names[0]: _to_numpy(data)}
                )
        self._last_conditioned_model = model
        return model

    def __repr__(self) -> str:
        params = ", ".join(self._param_names)
        return f"PyMCModel(params=[{params}])"
