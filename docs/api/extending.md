> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Registries for extensions

An extension implements a base class or a protocol of ProbPipe and registers an instance with a registry, and this page states, for each kind of extension, what to implement and where to register it.
It also documents the dispatch registries, the operation registry, and the base classes these extensions implement, and its last section documents the registry catalog.
The registries that users query are documented on the pages of their topics: `inference_method_registry` on [Inference methods](inference.md) and `converter_registry` on [Conversion](conversion.md).

## A new family

A family subclasses `Distribution`, or `ConditionalDistribution` for a kernel, and passes its label and the event spec of one draw to the base constructor as `label` and `event_spec`.
A draw that is one whole term is declared by an `OutputSpec` whose one keyword names its output, such as `OutputSpec(y=NumericArraySpec((22,)))`.
A family's constructor takes the output's name as its first argument and the label as the optional keyword `label=`, which defaults to the class name.
So `Normal("mu", 0.0, 1.0)` is labeled `Normal`, and `Normal("mu", 0.0, 1.0, label="prior")` is labeled `prior`.
A family over a TensorFlow Probability distribution subclasses `TFPDistribution`, which samples and scores through the backend distribution.
A family claims a capability by defining its implementation method, such as `_sample` for `SupportsSampling` or `_mean` for `SupportsMean`, and claims a conditioning capability by inheriting `SupportsExactConditioning` or `SupportsApproximateConditioning`.
A kernel's conditional methods, such as `_conditional_sample`, must be pure, since a joint compiles them when it maps the kernel over draws, as `ConditionalDistribution` states.
[Distributions and families](distributions.md) documents the base classes and each capability.

A family registers with no registry, since an operation selects a capability route by the protocols the family claims.
A family that stores its laws at batched parameters in one backend object implements `SupportsArrayBackend`.

::: probpipe.SupportsArrayBackend

## A new inference method

An inference method normalizes the target that `condition_on` forms when no exact route conditions a law, and it returns a normalized law over the target's event.
It subclasses `InferenceMethod`, declares `name` and `supported_types`, implements `check` and `execute`, and registers with `inference_method_registry.register`:

```python
from typing import Any

from probpipe import Distribution, SupportsUnnormalizedLogProb, inference_method_registry
from probpipe.inference import Feasibility, InferenceMethod


class ImportanceSampling(InferenceMethod):
    _method_options = ("num_draws",)  # the method_options entries execute reads

    @property
    def name(self) -> str:
        return "my_importance_sampling"

    @property
    def description(self) -> str:  # one line, which the registry catalog shows
        return "Self-normalized importance sampling from the prior."

    def supported_types(self) -> tuple[type, ...]:
        return (Distribution,)

    def check(self, target: Any, /, **options: Any) -> Feasibility:
        if not isinstance(target, SupportsUnnormalizedLogProb):
            return Feasibility(False, "the target has no unnormalized density")
        return Feasibility(True)

    def execute(self, target: Any, /, **options: Any) -> Distribution:
        self._check_options(options)
        ...  # draw, weight, and return a normalized law over the target's event


inference_method_registry.register(ImportanceSampling())
```

`condition_on.using(method="my_importance_sampling")` runs the method by name.
The method takes part in automatic selection once it overrides `priority`.

### Exactness and priority

A method declares its exactness and its priority, and the two are independent:

1. `exact`: whether the result denotes the conditional law itself. `InferenceMethod` declares `exact = False`, the declaration of a finite MCMC, variational, or ABC output, which stands in for the conditional law. A method whose result is the conditional law overrides it, as `empirical_reweighting` does: the posterior of an empirical prior is its atoms reweighted by the likelihood. The `exact_only` control excludes the approximate methods.
2. `priority`: the rank, which [Dispatch registries](#dispatch-registries) defines with its default, `None`.

A new method takes its rank relative to the nearest of the ranks of the built-in methods:

| Method | Priority |
|---|---|
| `empirical_reweighting` (exact) | 100 |
| `nutpie_nuts` | 88 |
| `blackjax_nuts` | 85 |
| `cmdstan_nuts`, `pymc_nuts` | 82 |
| `blackjax_elliptical_slice` | 75 |
| `blackjax_rwmh` | 55 |
| `blackjax_sgld` | 45 |
| `pyabc_smcabc` | 6 |

Five criteria, in decreasing weight, place a new approximate method among these ranks:

1. Robustness: how often the method gives a usable answer without tuning for the model, once its `check` passes.
2. Cost: the computation per effective draw or per converged result, whether a method saves it by exploiting the model's structure or by a faster backend.
3. Approximation quality: an approximation with a controlled error ranks above an asymptotically exact MCMC method, which ranks above an approximation whose error does not vanish with more computation.
4. Diagnostics: a method that reports its failures ranks above one that fails without a signal.
5. Breadth: the range of models the method applies to, which breaks ties only, since `check` decides which methods apply.

::: probpipe.inference.InferenceMethod
    options:
      show_root_full_path: true

## A new converter

A converter moves a law to another representation.
It subclasses `Converter` and defines five members:

1. `name`: the name by which a call selects the converter;
2. `exact`: whether the converted law is the source law in another representation;
3. `supported_types`: the pair of the source types and the target types;
4. `check`: the promise of a conversion as a `ConversionInfo`, computed without converting, which leaves `method_name` and `exact` for the registry to set from the registration;
5. `execute`: the converted law, which carries the source's event spec.

`check` and `execute` receive a call's converter options as keywords, such as `num_samples`, and a converter's `execute` must raise `TypeError` for an option it does not read.
`converter_registry.register` registers an instance, and `convert` and `with_conversion` then select among the registered converters.
The shipped converters have priorities from 10 to 20, and a new converter sets its `priority` to take part in automatic selection.

::: probpipe.Converter

## A new operation or route

`@operation` declares an operation from its signature and its result rule, and it registers the operation with `operation_registry`.
An operation registers its routes after construction, with `register_route` for any object that has the members of `OperationRoute`, or with one of four helpers:

1. `structural_route`: an implementation that reads the operands' declared structure;
2. `capability_route`: a call of a capability that one operand claims, such as `_mean`;
3. `registry_route`: a delegation to a dispatch registry, whose selected method realizes the call;
4. `fallback_route`: a generic scheme for a stated domain, which ranks below every other route of its exactness.

`operation_registry.describe("mean")` prints the operands and the routes of `mean` in selection order.

::: probpipe.operations.operation
    options:
      show_root_full_path: true

::: probpipe.operation_registry

::: probpipe.operations.OperationRegistry
    options:
      show_root_full_path: true

::: probpipe.operations.OperationRoute
    options:
      show_root_full_path: true

::: probpipe.operations.RouteSource
    options:
      show_root_full_path: true

::: probpipe.operations.BoundCall
    options:
      show_root_full_path: true

::: probpipe.operations.OperationSummary
    options:
      show_root_full_path: true

::: probpipe.operations.OperandSummary
    options:
      show_root_full_path: true

::: probpipe.operations.RouteSummary
    options:
      show_root_full_path: true

## A new evaluation rule

A call `f(d)` of a function at a distribution or a batch resolves among the evaluation rules, and a rule realizes the call for a pair of a map type and an operand type, as a closed form or a batched routine does.
A rule is a `BinaryDispatchMethod` whose `supported_types` is a pair of the map types and the operand types.
Its `check` and `execute` take the map and the operand positionally and the keywords `parameter`, `fixed_args`, and `controls`, and `evaluation_rule_registry.register` registers an instance.
The registry ships three rules: the sampling lift and the elementwise sweep, which are floors and so rank below every other rule, and the exact enumeration of empirical laws.

::: probpipe.evaluation_rule_registry

## A bijector for a constraint

`register_bijector` registers the factory of the bijector that `bijector_for` returns for a constraint, keyed on a `Constraint` subclass or on one constraint.

::: probpipe.register_bijector

## A new array backend

`register_array_backend` makes the instances of an array-container type numeric leaves, and its `ArrayBackend` argument holds the functions that read their shapes and dtypes and convert them to arrays.

::: probpipe.register_array_backend

::: probpipe.ArrayBackend

::: probpipe.array_backend_for

## A new executor

An executor decides how the evaluations a call asks for run: one after another, on threads, in worker processes, compiled by JAX, or on a cluster.
It never decides what they compute, so changing the executor changes no result beyond the last bit of rounding.
The evaluations of one callable that one request asks for, such as 200 runs of a simulator at 200 parameter values, form one `EvaluationPoints`, and an executor runs it by its chunks.
A user names an executor on a `Function`, as `simulate.using(executor="thread", executor_options={"max_workers": 8})`.
For now a lifted call and a sweep still run by their `dispatch` control, and a call only checks the executor's name and options; they move onto executors in a coming change.

ProbPipe ships four executors in `executor_registry`:

- `jax`, which maps a callable JAX can trace over every evaluation point at once, and is selected automatically for one;
- `sequential`, which runs the evaluation points one after another, and is selected automatically for every other callable;
- `thread`, which runs chunks on a pool of threads, with the options `max_workers` and `chunk_size`;
- `process`, which runs chunks in worker processes started under the calling process's JAX settings, with the options `max_workers`, `chunk_size`, and `start_method`.

`thread` and `process` run only when named, since user code may not be safe to run in several threads or to send to another process.

A new executor subclasses `EvaluationExecutor`, sets its `name` and `description`, and implements `check` and `execute`.
Its options are the parameters of `execute` after the set of evaluation points, which `check` takes too.
`execute` runs every chunk, passing each the set's shared payload, and returns one outcome per evaluation point in order; `EvaluationOutcomes.merge` restores that order from chunks that finished in any order.
A chunk pickles, so an executor may run it in another process or on another machine, and `shipping_check()` reports whether the set can be sent.
Pickling a chunk again, to retry it after its first run was lost, gives the retry the same random keys.
An executor that runs chunks elsewhere returns `EvaluationOutcomes.lost(chunk, reason)` for a chunk that never finished.
An executor with no `priority` runs only when named.
`check_executor`, in `probpipe.testing`, checks an executor against the contract before it is registered:

<!-- docs-test: skip, a Slurm cluster and submitit -->
```python
import submitit

from probpipe import EvaluationExecutor, EvaluationOutcomes, executor_registry
from probpipe.testing import check_executor


def run_chunk(chunk, shared_payload):
    return chunk.run(shared_payload)


class SlurmExecutor(EvaluationExecutor):
    name = "slurm"
    description = "Run chunks of evaluation points as Slurm array jobs."

    def check(self, evaluation_points, /, *, n_jobs=1, partition="short", retries=0):
        return evaluation_points.shipping_check()

    def execute(self, evaluation_points, /, *, n_jobs=1, partition="short", retries=0):
        cluster = submitit.AutoExecutor(folder="slurm_logs")
        cluster.update_parameters(slurm_partition=partition)
        shared_payload = evaluation_points.shared_payload()
        pending_chunks = evaluation_points.chunks(num_chunks=n_jobs)
        chunk_outcomes, failed_jobs = [], []
        for _ in range(retries + 1):
            jobs = cluster.map_array(run_chunk, pending_chunks, [shared_payload] * len(pending_chunks))
            failed_jobs = []
            for chunk, job in zip(pending_chunks, jobs):
                try:
                    chunk_outcomes.append(job.result())   # errors of user code are in the outcomes
                except Exception as job_error:            # the job itself died
                    failed_jobs.append((chunk, job_error))
            pending_chunks = [chunk for chunk, _ in failed_jobs]
            if not pending_chunks:
                break
        chunk_outcomes += [EvaluationOutcomes.lost(chunk, str(job_error)) for chunk, job_error in failed_jobs]
        return EvaluationOutcomes.merge(chunk_outcomes)


check_executor(SlurmExecutor())
executor_registry.register(SlurmExecutor())
```

::: probpipe.EvaluationExecutor
    options:
      members:
        - check
        - execute

::: probpipe.executor_registry

::: probpipe.EvaluationPoints

::: probpipe.EvaluationChunk

::: probpipe.SharedPayload

::: probpipe.EvaluationOutcomes

::: probpipe.EvaluationFailure

::: probpipe.testing.check_executor

## Dispatch registries

A dispatch registry holds named methods and selects one for a call by the types of its arguments.
It ranks the methods that admit a call by five criteria, in decreasing precedence:

1. Floor status: a method that declares `is_floor` is a fallback, and it ranks after every method that does not, whatever their exactness and priority.
2. Exactness: exact methods rank before approximate ones.
3. Priority: a higher priority ranks first among methods of the same floor status and exactness, and a method whose priority is `None` runs only when a call names it.
4. Specificity: the method whose declared types are closest to the arguments' classes ranks first.
5. Registration order: the method registered first ranks first.

The registry runs the first method in this order whose `check` reports the call feasible.
`set_priorities` re-ranks the methods of a registry at runtime and keeps each method's exactness and floor status.

::: probpipe.inference.BaseDispatchRegistry
    options:
      show_root_full_path: true

::: probpipe.inference.UnaryDispatchRegistry
    options:
      show_root_full_path: true

::: probpipe.inference.BinaryDispatchRegistry
    options:
      show_root_full_path: true

::: probpipe.inference.BaseDispatchMethod
    options:
      show_root_full_path: true

::: probpipe.inference.UnaryDispatchMethod
    options:
      show_root_full_path: true

::: probpipe.inference.BinaryDispatchMethod
    options:
      show_root_full_path: true

::: probpipe.inference.Feasibility
    options:
      show_root_full_path: true

::: probpipe.inference.MethodInfo
    options:
      show_root_full_path: true

::: probpipe.inference.UnarySupportedTypes
    options:
      show_root_full_path: true

::: probpipe.inference.BinarySupportedTypes
    options:
      show_root_full_path: true

## The registry catalog

`registry_catalog` lists every cataloged registry and its entries, where an entry is one registered item of a registry, such as an inference method or a converter.
It answers three queries:

- `print(registry_catalog)`: one row per registry, with its kind, its entry count, and its description;
- `registry_catalog.describe("inference")`: one registry's entries in its own order, each with its priority where the registry ranks its entries and its exactness where the registry declares it, with a dispatch registry's floors and its opt-in-only methods each in a section of their own;
- `registry_catalog["inference"].describe_entry("blackjax_nuts")`: one entry, as an `EntrySummary`.

The catalog holds six registries:

- `"bijectors"`: the bijector factories that `bijector_for` selects from;
- `"converters"`: `converter_registry`;
- `"evaluation_rules"`: `evaluation_rule_registry`;
- `"executors"`: `executor_registry`;
- `"inference"`: `inference_method_registry`;
- `"operations"`: `operation_registry`.

A registry joins the catalog by implementing `SupportsRegistryCataloging` and by an explicit `registry_catalog.register(registry)`, which rejects an empty or duplicate name.
Constructing a registry does not catalog it, so a registry that a test builds stays out of the catalog.
The catalog has no removal, so reloading a module that registers a registry, as `importlib.reload` or IPython's `%autoreload` does, raises `ValueError`; restart the process instead.
A dispatch registry implements the protocol already, and it takes its catalog name and description at construction:

```python
from probpipe import registry_catalog
from probpipe.inference import BinaryDispatchRegistry

kl_registry = BinaryDispatchRegistry(name="kl", description="KL divergence between two laws.")
registry_catalog.register(kl_registry)
```

A dispatch method may declare a one-line `description` property, as it declares its `name`, which the registry reads at registration and the catalog shows.
`register_bijector` takes a `description` keyword for the same purpose.

::: probpipe.registry_catalog

::: probpipe.RegistryCatalog
    options:
      show_root_full_path: true

::: probpipe.SupportsRegistryCataloging
    options:
      show_root_full_path: true

::: probpipe.EntrySummary
    options:
      show_root_full_path: true

::: probpipe.RegistryInfo
    options:
      show_root_full_path: true
