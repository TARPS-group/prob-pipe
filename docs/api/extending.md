> **AI-generated.** An AI assistant drafted this page, and no maintainer has reviewed it yet. Please report errors on the issue tracker.

# Registries for extensions

An extension implements a base class or a protocol of ProbPipe and registers an instance with a registry, and this page states, for each kind of extension, what to implement and where to register it.
It also documents the dispatch registries, the operation registry, and the base classes these extensions implement.
The registries that users query are documented on the pages of their topics: `inference_method_registry` on [Inference methods](inference.md) and `converter_registry` on [Conversion](conversion.md).

## A new family

A family subclasses `Distribution`, or `ConditionalDistribution` for a kernel, and passes the declaration of one draw to the constructor as `event_spec`.
A family over a TensorFlow Probability distribution subclasses `TFPDistribution`, which samples and scores through the backend distribution.
A family claims a capability by defining its implementation method, such as `_sample` for `SupportsSampling` or `_mean` for `SupportsMean`, and claims a conditioning capability by inheriting `SupportsExactConditioning` or `SupportsApproximateConditioning`.
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

`condition_on.with_options(method="my_importance_sampling")` runs the method by name.
The method takes part in automatic selection once it overrides `priority`.

### Exactness and priority

A method declares its exactness and its priority, and the two are independent:

1. `exact`: whether the result denotes the conditional law itself. `InferenceMethod` declares `exact = False`, since a finite MCMC, variational, or ABC output stands in for the conditional law, and every built-in method keeps that declaration. The `exact_only` control excludes the approximate methods.
2. `priority`: the rank among methods of the same exactness, which [Dispatch registries](#dispatch-registries) defines with its default, `None`.

A new method takes its rank relative to the nearest of the ranks of the built-in methods:

| Method | Priority |
|---|---|
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
4. `check`: the promise of a conversion as a `ConversionInfo`, computed without converting;
5. `execute`: the converted law, which carries the source's event declaration.

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
The registry ships three rules: the sampling lift and the elementwise sweep, which rank below every other rule, and the exact enumeration of empirical laws.

::: probpipe.evaluation_rule_registry

## A bijector for a constraint

`register_bijector` registers the factory of the bijector that `bijector_for` returns for a constraint, keyed on a `Constraint` subclass or on one constraint.

::: probpipe.register_bijector

## A new array backend

`register_array_backend` makes the instances of an array-container type numeric leaves, and its `ArrayBackend` argument holds the functions that read their shapes and dtypes and convert them to arrays.

::: probpipe.register_array_backend

::: probpipe.ArrayBackend

::: probpipe.array_backend_for

## Dispatch registries

A dispatch registry holds named methods and selects one for a call by the types of its arguments.
It ranks the methods that admit a call by four criteria, in decreasing precedence:

1. Exactness: exact methods rank before approximate ones.
2. Priority: a higher priority ranks first among methods of the same exactness, and a method whose priority is `None` runs only when a call names it.
3. Specificity: the method whose declared types are closest to the arguments' classes ranks first.
4. Registration order: the method registered first ranks first.

`set_priorities` re-ranks the methods of a registry at runtime and keeps each method's exactness.

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
