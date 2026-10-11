"""The two prototype fixes of issue #531, applied as monkeypatches.

Modes:
  baseline  the code as it is (nothing is patched)
  sampler   jit TFPDistribution._sample through one module-level jitted function
  kernel    the sampler patch, and a jit of the vmapped kernel in _conditional_draw

These are prototypes for measurement. The kernel patch keys its cache on
``id(factor)``, which is unsafe in real code because an id is reused once its
object is freed.
"""

import math
from functools import partial

import jax

MODES = ("baseline", "sampler", "kernel")


def apply(mode):
    """Patch ProbPipe for *mode*, one of :data:`MODES`."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode in ("sampler", "kernel"):
        from probpipe.families._backend import TFPDistribution

        @partial(jax.jit, static_argnames="sample_shape")
        def _backend_sample(backend, key, sample_shape):
            return backend.sample(seed=key, sample_shape=sample_shape)

        def _sample(self, key, sample_shape=()):
            return _backend_sample(self._tfp_dist, key, tuple(sample_shape))

        TFPDistribution._sample = _sample

    if mode == "kernel":
        import probpipe.distributions._factored as factored

        original_draw = factored._conditional_draw
        jitted = {}

        def _conditional_draw(factor, drawn, fixed, key, sample_shape):
            inner = {slot: drawn[slot] for slot in factor.given_spec if slot in drawn}
            if not inner or not sample_shape or not factored._numeric_givens(factor, inner):
                return original_draw(factor, drawn, fixed, key, sample_shape)
            draw = jitted.get(id(factor))
            if draw is None:

                def one(values, fixed, subkey):
                    given = factored._factor_given(factor, values, fixed)
                    return factor._conditional_sample(given, subkey, ())

                draw = jitted[id(factor)] = jax.jit(jax.vmap(one, in_axes=(0, None, 0)))
            count = math.prod(sample_shape)
            draws = draw(
                factored._flatten_draws(inner, sample_shape),
                dict(fixed),
                jax.random.split(key, count),
            )
            return factored._unflatten_draws(draws, sample_shape)

        factored._conditional_draw = _conditional_draw
