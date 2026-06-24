"""StanModel: wraps Stan models as ProbPipe distributions via BridgeStan.

Uses BridgeStan for log-probability evaluation and gradients.
Inference is handled by registered methods in ``probpipe.inference``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import cached_property
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from ..core.distribution import Distribution
from ..core.protocols import SupportsLogProb
from ..core.record import NumericEventTemplate
from ..custom_types import Array, ArrayLike
from ._base import ProbabilisticModel

logger = logging.getLogger(__name__)

__all__ = ["StanModel"]


def _to_f64(x: ArrayLike) -> np.ndarray:
    """Coerce *x* to the contiguous ``float64`` ndarray BridgeStan requires.

    BridgeStan's ctypes interface rejects anything that is not a NumPy
    ``float64`` array — a JAX array fails the ``ndarray`` check, and a
    ``float32`` array fails the dtype check — so every value crossing into
    ``param_constrain`` / ``param_unconstrain`` / ``log_density`` passes
    through here first.
    """
    return np.asarray(x, dtype=np.float64)


class _StanBlock(NamedTuple):
    """One Stan parameter block recovered from BridgeStan's flattened names."""

    name: str
    shape: tuple[int, ...]  # () for a scalar parameter
    # advanced-index arrays mapping a ``shape``-shaped array onto the block's
    # flat slice in BridgeStan's order; () for a scalar.
    gather: tuple[Array, ...]


def _param_blocks(flat_names: Sequence[str]) -> tuple[_StanBlock, ...]:
    """Group BridgeStan's dot-flattened parameter names into typed blocks.

    BridgeStan reports one flattened name per scalar (``mu``, ``theta.1``,
    ``L.1.1`` — dot-separated, 1-indexed), with each block's scalars emitted
    consecutively. This recovers each block's name, its reconstructed
    multidimensional ``shape`` (``()`` for a scalar), and the advanced-index
    ``gather`` that maps a ``shape``-shaped array onto the block's slice of the
    flat parameter vector in BridgeStan's exact order. Placing each scalar by
    its parsed index makes the round-trip correct for matrices (column-major)
    and arrays alike, with no assumption about the flattening convention.
    """
    blocks: list[_StanBlock] = []
    i, n = 0, len(flat_names)
    while i < n:
        block = flat_names[i].split(".", 1)[0]
        idx_tuples: list[tuple[int, ...]] = []
        while i < n and flat_names[i].split(".", 1)[0] == block:
            idx_tuples.append(tuple(int(x) - 1 for x in flat_names[i].split(".")[1:]))
            i += 1
        if idx_tuples == [()]:
            blocks.append(_StanBlock(block, (), ()))
            continue
        ndim = len(idx_tuples[0])
        shape = tuple(max(t[d] for t in idx_tuples) + 1 for d in range(ndim))
        gather = tuple(jnp.asarray([t[d] for t in idx_tuples]) for d in range(ndim))
        blocks.append(_StanBlock(block, shape, gather))
    return tuple(blocks)


def _pack_block_params(
    owner: str,
    blocks: tuple[_StanBlock, ...],
    field_kwargs: dict[str, Any],
) -> Array:
    """Assemble the flat Stan parameter vector from per-block keyword values.

    Each value must match its block's declared shape; values are scattered into
    BridgeStan's flat order via each block's ``gather`` and concatenated. The
    flat result is exactly what ``_log_prob`` consumes; the flat array can also
    be passed positionally instead.
    """
    expected = [b.name for b in blocks]
    expected_set = set(expected)
    missing = [name for name in expected if name not in field_kwargs]
    extra = [k for k in field_kwargs if k not in expected_set]
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unexpected {extra}")
        raise TypeError(
            f"{owner}: the keyword form expects the Stan parameter blocks "
            f"{tuple(expected)} — {'; '.join(detail)}."
        )
    out: list[Array] = []
    for b in blocks:
        arr = jnp.asarray(field_kwargs[b.name])
        if tuple(arr.shape) != b.shape:
            raise TypeError(
                f"{owner}: parameter {b.name!r} expects shape {b.shape}, got {tuple(arr.shape)}."
            )
        out.append(jnp.reshape(arr, (1,)) if b.shape == () else arr[b.gather])
    return jnp.concatenate(out) if out else jnp.zeros((0,))


class StanModel(ProbabilisticModel, SupportsLogProb):
    """Stan model via BridgeStan and CmdStanPy.

    Uses BridgeStan for log-probability evaluation and JAX-compatible
    gradients.  Posterior sampling (via ``condition_on``) uses
    CmdStanPy's interface to Stan's native NUTS sampler.

    Parameters are in the constrained space by default; use
    :meth:`as_unconstrained_distribution` for the unconstrained
    parameterization.

    Parameters
    ----------
    stan_file : str
        Path to a ``.stan`` file.
    data : dict or None
        Stan data dictionary.  Can also be provided at conditioning
        time.
    name : str or None
        Model name for provenance.

    Raises
    ------
    ImportError
        If ``bridgestan`` is not installed.
    """

    def __init__(
        self,
        stan_file: str,
        *,
        data: dict | None = None,
        name: str | None = None,
    ):
        try:
            import bridgestan
        except ImportError as e:
            raise ImportError(
                "bridgestan is required for StanModel. Install it with: pip install bridgestan"
            ) from e

        self._stan_file = stan_file
        self._stan_data = data
        # ``Distribution`` metaclass requires a non-empty name; default
        # to the class name when the caller doesn't supply one.
        self._name = name if name else "StanModel"

        # Compile and instantiate. BridgeStan's constructor takes the
        # ``.stan`` path directly (compiling on demand) and serializes a
        # data dict via stanio; ``data=None`` means no data.
        self._bs_model = bridgestan.StanModel(stan_file, data=data)
        self._num_params = self._bs_model.param_unc_num()

    # -- Distribution interface ---------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._num_params,)

    # -- Named components interface ------------------------------------------

    @cached_property
    def _blocks(self) -> tuple[_StanBlock, ...]:
        """Constrained-space parameter blocks, from BridgeStan's ``param_names()``."""
        return _param_blocks(self._bs_model.param_names())

    @cached_property
    def event_template(self) -> NumericEventTemplate:
        """One field per Stan parameter block, shaped from BridgeStan's names."""
        return NumericEventTemplate({b.name: b.shape for b in self._blocks})

    @property
    def fields(self) -> tuple[str, ...]:
        return self.event_template.fields

    def __getitem__(self, key: str) -> Any:
        if key in self.fields:
            return key  # placeholder — Stan doesn't expose sub-distributions
        raise KeyError(f"Unknown component: {key!r}")

    # -- ProbabilisticModel interface ---------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        # Per-scalar flattened names (the Stan / CmdStan convention) — distinct
        # from ``fields``, which are the per-block names used by the keyword form.
        return tuple(self._bs_model.param_names())

    # -- SupportsLogProb interface ------------------------------------------

    def _pack_value(self, **field_kwargs: Any) -> Array:
        """Keyword form: one array per Stan parameter block in the constrained
        space, assembled into the flat array ``_log_prob`` consumes (see
        :func:`_pack_block_params`). A flat array may still be passed
        positionally."""
        return _pack_block_params(type(self).__name__, self._blocks, field_kwargs)

    def _log_prob(self, value: Any) -> Array:
        """Log-density in the constrained parameter space.

        Internally unconstrains the parameters before calling Stan's
        log_density.
        """
        params_unc = self._bs_model.param_unconstrain(_to_f64(value))
        return jnp.asarray(self._bs_model.log_density(params_unc))

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """Delegates to _log_prob (Stan provides the full joint)."""
        return self._log_prob(value)

    def _unnormalized_prob(self, value: Any) -> Array:
        return jnp.exp(self._unnormalized_log_prob(value))

    def _prob(self, value: Any) -> Array:
        return jnp.exp(self._log_prob(value))

    # -- Constrained / unconstrained transformations ------------------------

    def param_constrain(self, params_unc: ArrayLike) -> Array:
        """Transform unconstrained parameters to constrained space."""
        return jnp.asarray(self._bs_model.param_constrain(_to_f64(params_unc)))

    def param_unconstrain(self, params: ArrayLike) -> Array:
        """Transform constrained parameters to unconstrained space."""
        return jnp.asarray(self._bs_model.param_unconstrain(_to_f64(params)))

    def as_unconstrained_distribution(self) -> _UnconstrainedStanView:
        """Return a view of this model in the unconstrained parameter space."""
        return _UnconstrainedStanView(self)

    # -- BridgeStan model access (for nutpie integration) -------------------

    def _bridgestan_model(self, data: dict | None = None) -> Any:
        """Return the BridgeStan model, optionally with new data."""
        if data is not None:
            import bridgestan

            return bridgestan.StanModel(self._stan_file, data=data)
        return self._bs_model

    def __repr__(self) -> str:
        return f"StanModel(stan_file={self._stan_file!r}, num_params={self._num_params})"


class _UnconstrainedStanView(Distribution[Any], SupportsLogProb):
    """View of a StanModel in the unconstrained parameter space."""

    def __init__(self, model: StanModel):
        self._model = model
        # ``base.name`` is guaranteed non-empty by the Distribution
        # metaclass — wrap with an ``_unconstrained`` suffix.
        self._name = f"{model.name}_unconstrained"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._model.event_shape

    @cached_property
    def _blocks(self) -> tuple[_StanBlock, ...]:
        """Unconstrained-space parameter blocks, from ``param_unc_names()``."""
        return _param_blocks(self._model._bs_model.param_unc_names())

    @cached_property
    def event_template(self) -> NumericEventTemplate:
        """One field per unconstrained Stan parameter block."""
        return NumericEventTemplate({b.name: b.shape for b in self._blocks})

    @property
    def fields(self) -> tuple[str, ...]:
        return self.event_template.fields

    def _pack_value(self, **field_kwargs: Any) -> Array:
        """Keyword form: one array per Stan parameter block in the unconstrained
        space, assembled into the flat array ``_log_prob`` consumes. A flat
        array may still be passed positionally."""
        return _pack_block_params(type(self).__name__, self._blocks, field_kwargs)

    def _log_prob(self, value: Any) -> Array:
        """Log-density directly in unconstrained space."""
        return jnp.asarray(self._model._bs_model.log_density(_to_f64(value)))

    def _unnormalized_log_prob(self, value: Any) -> Array:
        return self._log_prob(value)

    def _unnormalized_prob(self, value: Any) -> Array:
        return jnp.exp(self._unnormalized_log_prob(value))

    def _prob(self, value: Any) -> Array:
        return jnp.exp(self._log_prob(value))

    def __repr__(self) -> str:
        return f"UnconstrainedStanView({self._model!r})"
