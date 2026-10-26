"""Fixed workloads for measuring ProbPipe execution, without statistical scoring."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from time import perf_counter
from typing import Any
from unittest.mock import patch

GROUPS = ("minibatch", "family", "joint", "function", "inference")
SCENARIOS = ("same_object", "new_object", "structure_change")


@dataclass(frozen=True)
class Case:
    """A versioned workload, independent of its measurement process."""

    group: str
    operation: str
    size: int
    n: int = 0
    variant: str = "current"
    chains: int = 1
    version: int = 1

    @property
    def name(self) -> str:
        return f"{self.group}/{self.operation}/{self.variant}/{self.n}/{self.size}/{self.chains}"

    @property
    def scenarios(self) -> tuple[str, ...]:
        """Situations that measure workload behavior rather than wrapper creation."""
        if self.group == "minibatch" and self.operation in ("indices", "vmap64"):
            return ("same_object", "structure_change")
        return SCENARIOS


def measurement_context(case: Case) -> AbstractContextManager[Any]:
    """Install the gradient reference before tracing, outside all timed calls."""
    if case.group == "minibatch" and case.operation in ("gradient", "gradient_scan"):
        from probpipe.inference._minibatch import _draw_indices

        sampler = _draw_indices if case.variant == "current" else permutation
        return patch("probpipe.inference._minibatch._draw_indices", sampler)
    return nullcontext()


def catalog(suite: str = "quick") -> list[Case]:
    """List workloads without importing JAX or initializing a device."""
    sizes = (32,) if suite == "quick" else (1, 32, 1024)
    cases = []
    for size in sizes:
        for family in ("normal", "gamma"):
            cases.extend(Case("family", op, size, variant=family) for op in ("sample", "density"))
        for joint in ("independent", "conditional"):
            cases.extend(Case("joint", op, size, variant=joint) for op in ("sample", "density"))
        cases.append(Case("function", "direct", size))
        cases.extend(Case("function", "lift", size, variant=e) for e in ("sequential", "jax"))
    matrix = [(1000, 32)]
    if suite == "full":
        matrix = [(n, b) for n in (1000, 10_000, 100_000, 1_000_000) for b in (32, 256)]
        matrix += [
            (65536, 256),
            (66049, 257),
            (1_000_000, 4096),
            (10_000, 5000),
            (10_000, 5001),
            (1_000_000, 900_000),
            (1_000_000, 999_968),
            (1024, 1024),
        ]
    for variant in ("current", "permutation"):
        cases.extend(Case("minibatch", "indices", b, n, variant) for n, b in matrix)
        mapped = [(1000, 32)] if suite == "quick" else [(10_000, 256), (65536, 256)]
        cases.extend(Case("minibatch", "vmap64", b, n, variant) for n, b in mapped)
        for size in (32,) if suite == "quick" else (32, 256):
            for op in ("gradient", "gradient_scan"):
                cases.append(
                    Case("minibatch", op, size, 1000 if suite == "quick" else 1_000_000, variant)
                )
    cases.append(Case("inference", "rwmh", 100, n=32))
    if suite == "full":
        cases.append(Case("inference", "rwmh", 1000, n=32, chains=2))
        present = {c.name for c in cases}
        cases.extend(c for c in catalog("quick") if c.name not in present)
    return [replace(c, version=2) if c.group == "minibatch" else c for c in cases]


def permutation(key: Any, n: int, batch_size: int, *, with_replacement: bool) -> Any:
    """The original full-population reference for index generation."""
    import jax

    if with_replacement:
        return jax.random.randint(key, (batch_size,), 0, n)
    return jax.random.permutation(key, n)[:batch_size]


def prepare(
    case: Case, count: int, scenario: str, seed: int
) -> tuple[list[tuple[Callable[[], Any], int]], list[float]]:
    """Prepare calls without invoking any measured operation.

    The first call always establishes the original structure. Subsequent calls
    reuse it, construct another owner, or alternate between two static sizes.
    Construction and input preparation belong to the separately timed setup.
    """
    import jax
    import jax.numpy as jnp

    keys = list(jax.random.split(jax.random.key(seed), count))
    if case.operation == "vmap64":
        keys = [jax.random.split(key, 64) for key in keys]
    jax.block_until_ready(keys)
    construction_ms = []

    def construct(size: int, shift: float) -> Callable[..., Any]:
        start = perf_counter()
        owner = build(case, size, shift)
        construction_ms.append((perf_counter() - start) * 1000)
        return owner

    first = construct(case.size, 0.0)
    structures = {case.size: first}
    calls = []
    for i, key in enumerate(keys):
        size = case.size + (i % 2) if scenario == "structure_change" else case.size
        # A full minibatch cannot grow beyond its population.
        if case.group == "minibatch":
            size = min(size, case.n)
            if size == case.size and scenario == "structure_change" and i % 2:
                size = max(1, size - 1)
        shift = i * 0.001
        if i == 0 or scenario == "same_object":
            owner = first
        elif scenario == "new_object":
            owner = construct(size, shift)
        else:
            if size not in structures:
                structures[size] = construct(size, 0.0)
            owner = structures[size]
        value = jnp.asarray(shift, dtype=jnp.float32)
        jax.block_until_ready(value)
        calls.append((partial(owner, key, seed + i, value), size))
    return calls, construction_ms


def build(case: Case, size: int, shift: float) -> Callable[..., Any]:
    """Construct one owner and its inputs; return a callable that executes it."""
    import jax
    import jax.numpy as jnp

    from probpipe import (
        Function,
        Gamma,
        MultivariateNormal,
        Normal,
        NumericArrayBatch,
        NumericRecordBatch,
        condition_on,
        log_prob,
        sample,
        workflow_run,
    )
    from probpipe.families import BernoulliFamily, GaussianFamily, glm_likelihood

    if case.group == "minibatch":
        from probpipe.inference._blackjax_sgmcmc import _build_grad_estimator
        from probpipe.inference._minibatch import MinibatchedDistribution, _draw_indices

        sampler = _draw_indices if case.variant == "current" else permutation
        indices = partial(sampler, n=case.n, batch_size=size, with_replacement=False)
        if case.operation in ("indices", "vmap64"):
            draw = jax.jit(jax.vmap(indices) if case.operation == "vmap64" else indices)
            return lambda key, seed, value: draw(key)
        x = jnp.linspace(-2, 2, case.n)
        design = jnp.stack([jnp.ones(case.n), x], axis=1)
        response = (jnp.arange(case.n) % 3 == 0).astype(jnp.float32)
        prior = MultivariateNormal("beta", loc=jnp.zeros(2), cov=jnp.eye(2))
        likelihood = glm_likelihood("y", BernoulliFamily(), X=design)
        estimator = _build_grad_estimator(
            MinibatchedDistribution(prior, likelihood, response, size)
        )
        position = jnp.array([0.3 + shift, -0.2])
        jax.block_until_ready((design, response, position))

        def scan(position: Any, key: Any) -> Any:
            def step(state: Any, unused: Any) -> tuple[Any, None]:
                key, total = state
                key, subkey = jax.random.split(key)
                return (key, total + estimator(position, subkey)), None

            return jax.lax.scan(step, (key, jnp.zeros_like(position)), None, length=25)[0][1]

        operation = jax.jit(scan if case.operation == "gradient_scan" else estimator)

        def gradient(key: Any, seed: int, value: Any) -> Any:
            return operation(position + value, key)

        return gradient

    if case.group == "family":
        law = Normal("x", shift, 1.0) if case.variant == "normal" else Gamma("x", 2.0 + shift, 1.0)
        values = jnp.ones(size)
    elif case.group == "joint":
        if case.variant == "independent":
            law = Normal("x", shift, 1.0) & Normal("y", 0.0, 1.0)
            values = {"x": jnp.ones(size), "y": jnp.ones(size)}
        else:
            # Keep declaration and dictionary order aligned until mapped records
            # preserve it: https://github.com/TARPS-group/prob-pipe/issues/642
            law = glm_likelihood(
                "a", GaussianFamily(), X=jnp.ones((1, 1)), dispersion=1.0
            ) & MultivariateNormal("beta", loc=jnp.array([shift]), cov=jnp.eye(1))
            values = {"a": jnp.ones((size, 1)), "beta": jnp.ones((size, 1))}
    elif case.group == "function":
        fun = Function(
            _transform, label="benchmark_transform", traceable=True, max_evaluations=size
        )
        if case.operation == "lift":
            fun = fun.using(executor=case.variant)
            argument = Normal("x", shift, 1.0)
        else:
            argument = jnp.ones(size)
    else:
        design = jnp.stack([jnp.ones(case.n), jnp.linspace(-1, 1, case.n)], axis=1)
        law = glm_likelihood("y", GaussianFamily(), X=design, dispersion=1.0) & MultivariateNormal(
            "beta", loc=jnp.zeros(2), cov=jnp.eye(2)
        )
        values = jnp.linspace(-0.2, 0.2, case.n)
        fit = condition_on.using(
            method="blackjax_rwmh",
            method_options={
                "adapt": False,
                "num_warmup": 0,
                "step_size": 0.1,
                "num_results": size,
                "num_chains": case.chains,
                "init": jnp.zeros(2),
            },
        )
    # Synchronize prepared arrays, without sampling or scoring the model.
    if case.group != "function":
        jax.block_until_ready(values)
    elif case.operation == "direct":
        jax.block_until_ready(argument)

    def execute(key: Any, seed: int, value: Any) -> Any:
        with workflow_run(seed=seed):
            if case.group == "function":
                return fun(x=argument + value) if case.operation == "direct" else fun(x=argument)
            if case.group == "inference":
                return fit(law, {"y": values + value})
            if case.operation == "sample":
                return sample(law, (size,))
            data = (
                {name: column + value for name, column in values.items()}
                if isinstance(values, dict)
                else values + value
            )
            batch = (
                NumericRecordBatch(data, "point", label="values")
                if isinstance(data, dict)
                else NumericArrayBatch(data, "point", label="values")
            )
            return log_prob(law, batch)

    return execute


# Function infers its lifting declaration from this unannotated callable.
# Annotating x as Any would accept distributions directly and disable lifting.
def _transform(x):
    return x * x + 1.0


def arrays(result: Any) -> list[Any]:
    """Extract completed numerical outputs without a second public operation."""
    import jax

    if hasattr(result, "atoms"):
        result = result.atoms
    return [leaf for leaf in jax.tree.leaves(result) if hasattr(leaf, "dtype")]


def validate(case: Case, result: Any, size: int) -> None:
    """Check execution outputs, never statistical quality."""
    import numpy as np

    leaves = arrays(result)
    if not leaves or not all(np.isfinite(np.asarray(x)).all() for x in leaves):
        raise ValueError("Measured operation returned no numerical outputs or non-finite values")
    if case.group == "minibatch" and case.operation in ("indices", "vmap64"):
        data = np.asarray(leaves[0])
        shape = (64, size) if case.operation == "vmap64" else (size,)
        if data.shape != shape or np.any(data < 0) or np.any(data >= case.n):
            raise ValueError("Invalid minibatch index shape or range")
        rows = data.reshape(-1, size)
        if any(len(np.unique(row)) != size for row in rows):
            raise ValueError("Duplicate minibatch index")
    elif case.group == "inference":
        if tuple(result.atoms.batch_shape) != (case.chains, size):
            raise ValueError("Unexpected inference chain/draw shape")
    elif case.operation in ("sample", "density", "direct", "lift"):
        if any(x.shape[0] != size for x in leaves):
            raise ValueError("Unexpected batch shape")
