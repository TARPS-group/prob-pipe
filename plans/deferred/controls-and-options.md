# Plan: one contract for `using`: controls and options

> **Status: proposed; not approved.**
> Written on 2026-10-28 against `main` at `a45cd20f`.
> Review it, and answer its questions, before implementing anything from it.
> `plans/deferred.md` lists it under "Controls and options".
> Every operation takes controls, so § 6 lists the edits the other deferred plans need, the inference-interface plan's first.
> Once approved, its rules move into `design/` (II.7, V.2, V.4, V.9, VI.0, VI.6, VI.10, and the glossary) and its steps into the plans of the work after the workshop.

**Decisions this plan asks for, all open** (each explained where it arises, and gathered in § 8):

1. A given control or option that the call does not read warns rather than raises (§ 4.4).
2. Controls set by the decorator do not warn; controls set by `using` do (§ 4.4).
3. Options given without `method` are checked against every method the call could select (§ 4.3).
4. Entries declare their options with explicit `Option` objects rather than through signatures (§ 4.2).
5. The inference options are renamed to one vocabulary, so `num_results` becomes `num_draws` (§ 5.3).
6. A method may take `backend_options`, forwarded unchanged to its backend (§ 5.2).
7. The converter option `num_samples` goes, and the control alone sets a conversion's draws (§ 5.4).
8. The name `describe_using` (§ 4.7).

---

## 1. Why this plan exists

`using` sets how a call runs, apart from its arguments:

```python
posterior = condition_on.using(method="blackjax_nuts", method_options={"num_warmup": 2000})(model, data)
```

It takes two kinds of setting.
**Controls**, such as `num_samples` and `method`, are defined by ProbPipe for every call.
**Options**, such as NUTS's `num_warmup`, are defined by one registered method and passed in `method_options`.
Neither kind has one contract today.
A setting can do nothing without a word, one configured copy raises on one model and runs on another, and a user has no way to list what a call accepts:

```python
import jax.numpy as jnp
from probpipe import EmpiricalDistribution, function, workflow_run

@function
def square(theta):
    return theta ** 2

posterior = EmpiricalDistribution(jnp.arange(300.0), output_name="theta")

with workflow_run(seed=0):
    square.using(num_samples=1000)(posterior)             # 256 atoms: num_samples is ignored, silently
    square.using(method_options={"bogus": 1})(posterior)  # runs: a misspelled option is ignored, silently
```

The goal is one contract for what `using` takes, one declaration of every control and option that every reader uses, one rule for a setting a call does not read, and one way to find out what a call accepts.

## 2. The design today

### 2.1 What `design/` states

| Section | What it states |
|---|---|
| V.2 | A `Function` keeps its arguments apart from the framework's **controls**: `max_evaluations`, `num_samples`, `include_inputs`, `method`, `exact_only`, `conversions`, `method_options`, `raw`, `executor`, `executor_options`, and `workflow_kind`. |
| V.2 | A control takes its value from the call's own setting, then from a value an enclosing call passes down, then from a `config.using` block, the session's value, and the default. |
| V.2 | `using` raises `TypeError` or `ValueError` for an unknown control or an inadmissible value. "The selected method rejects a `method_options` entry it does not read when the call runs." |
| V.2 | `method_options` is "one mapping of the numerical budgets the selected method reads". |
| V.2 | `max_evaluations` bounds the evaluations of a lifted call, and `num_samples` is the number of Monte Carlo draws of a summary or of a converter that draws. |
| V.4 | A parameter's conversion is set by `conversions={"d": {"method": "tfp", "exact_only": True, **options}}`, one flat mapping per parameter. |
| V.9 | `executor_options` belong to the executor named with them, which validates them; an option the executor does not read raises `ValueError` at the call; a named executor that ran nothing warns. The `jax` executor's option is `batch_size`. |
| VI.0 | A derived route passes `exact_only` and the two counts to the calls its formula makes, and "the other controls stay each call's own". |
| VI.6 | Method parameters such as warmup counts go through `method_options`, which the selected method validates. |
| VI.10 | A converter that draws takes the call's `num_samples` "unless the converter's own option of that name is given". |
| II.7 | The registry catalog shows each entry's name, exactness, floor status, priority, and description. |
| glossary | Canonical names for `max_evaluations`, `num_samples`, `executor_options`, and `chunk_size`, and for two method options, `target_accept_prob` and `max_tree_depth`. |

`plans/decisions.md` adds one rule, from 2026-10-20: `using` checks only what a control shows alone, its name and the form of its value, and the call raises the errors of executors and methods.

### 2.2 What the design leaves open

1. **A setting the call does not read.** No rule covers it, and #675 asks for one.
2. **How an option is declared.** Each method chooses its names, defaults, and checks, and II.7 gives an entry no place to state them, so the catalog cannot list them.
3. **Options that are not budgets.** V.2 says "budgets", but methods also take a starting point (`init`), callables (`summary_fn`), and display flags (`progress_bar`).
4. **What a shared name means.** Only two method options have canonical names, so `num_warmup` means one thing in every MCMC method by convention alone.
5. **Which options a call checks before it selects a method.** "The selected method rejects" an entry, so whether a call raises depends on the method it selects.
6. **The error type.** V.9 raises `ValueError` for an unknown executor option, and V.2 states no type for an unknown method option.
7. **The `jax` executor's `batch_size`.** It counts the evaluation points in one chunk, which the glossary calls `chunk_size`, while SG-MCMC's `batch_size` is a minibatch size.

## 3. The code today

### 3.1 Where the code agrees with the design

1. **Control names and values.** `Function.using` refuses an unknown control when `using` is called, listing the controls, and `_validate_options` checks each value (`values/_function_base.py:1412`, `:1821`).
2. **Resetting.** A control given as `None` returns to its default, and `executor` and `executor_options` reset together.
3. **Executor options.** An executor refuses an option it does not read, at the call, and a named executor that ran nothing warns.
4. **Passed controls.** A derived route passes `exact_only` and the two counts (`_passing_controls_to_formula_calls`).
5. **A selected method's options.** An inference method refuses an option outside its `_method_options` (`operations/_condition.py:227`).

### 3.2 Where the code departs from the design or from itself

**A. Settings that do nothing.**
Every call below runs to completion with no warning, on `main` at `a45cd20f`, with `n = Normal("x", 0.0, 1.0)` and `square` and `posterior` as in § 1:

| Call | What is ignored | Why |
|---|---|---|
| `mean.using(num_samples=10)(n)` | `num_samples` | the exact route `closed_form` reads no count |
| `mean.using(method_options={"bogus": 1})(n)` | an option no route of `mean` takes | an exact capability route drops `method_options` (`operations/_operation.py:614`) |
| `mean.using(include_inputs=True)(n)` | `include_inputs` | only a lift reads it |
| `square.using(max_evaluations=3)(2.0)` | `max_evaluations` | a call on a value lifts nothing |
| `square.using(num_samples=1000)(posterior)` | `num_samples` | a lift reads `max_evaluations` (#675) |
| `square.using(executor_options={"bogus": 1})(posterior)` | an option no executor takes | `jax` compiles the traceable `square` and receives none of the options |
| `condition_on.using(num_samples=50, method="blackjax_rwmh", method_options={"num_results": 20, "num_chains": 1, ...})` | `num_samples` | no inference method reads it, so the posterior has 20 atoms |
| `condition_on.using(method="pyabc_smcabc", method_options={"eps": schedule, "eps_alpha": 0.2})` | `eps_alpha` | the method's docstring says `eps_alpha` is "ignored if `eps` is given" |

**B. One configured copy, two outcomes.**
Whether an option is refused depends on the method the call selects:

```python
likelihood = conditional_distribution(
    lambda mu: Normal("y", mu, 1.0), given_spec={"mu": NumericArraySpec(())}, label="lik"
)
long_warmup = condition_on.using(method_options={"num_warmup": 2000})

long_warmup(likelihood & Normal("mu", 0.0, 1.0), {"y": 0.5})   # runs blackjax_nuts, 2000 warmup steps

empirical_prior = EmpiricalDistribution(jnp.linspace(-2.0, 2.0, 9), output_name="mu")
long_warmup(likelihood & empirical_prior, {"y": 0.5})
# TypeError: inference method 'empirical_reweighting': unknown method option 'num_warmup';
#   there are no method options
```

`num_warmup` is a real option, and the reweighted posterior is correct without it.
Yet this call raises, while `mean` in row 2 of A accepts an option that no route of `mean` takes.

**C. Two spellings for one count.**
`convert` takes its number of draws from the control and from a converter option, and the option wins without a word:

```python
convert.using(num_samples=7)(n, EmpiricalDistribution)                                    # 7 atoms
convert.using(num_samples=7, method_options={"num_samples": 9})(n, EmpiricalDistribution)  # 9 atoms
```

**D. Five ways to declare a setting.**
Each kind of setting keeps its names, defaults, checks, and documentation in different places:

| Setting | Names | Defaults | Checks | Documentation |
|---|---|---|---|---|
| `config` setting | a `_Setting` per name (`core/config.py:67`) | the section's attributes | `_Setting.check` | the property's docstring |
| control | `_CONTROL_DEFAULTS` (`values/_function_base.py:734`) | the same mapping, and `_DEFAULTED_BY_NONE` | `_validate_options` | the decorator's docstring |
| inference option | the `_method_options` tuple, in ten modules | `options.get(name, default)` inside `execute` | inline, and `nuts_controls` | the factory's docstring |
| executor option | the parameters of `execute`, read by `_executor_option_names` | the parameters' defaults | inline, such as `_checked_worker_count` | the class docstring |
| converter option | a literal tuple passed to `_refuse_unread` by each converter | `options.get(...)` | inline, such as `_sample_count` | `docs/api/extending.md` |

An unknown name raises `TypeError` from the inference methods and the converters, and `ValueError` from the executors.

**E. Names and defaults drift.**
The inference methods share names by convention, and they disagree:

| Option | Today |
|---|---|
| `num_warmup` default | 500 in BlackJAX, TFP, PyMC, and nutpie; 1000 in `cmdstan_nuts`; 0 in SG-MCMC |
| `target_accept_prob` default | 0.8 in `blackjax_nuts`; 0.75 in `tfp_nuts` |
| draws per chain | `num_results`, TFP's word, while the posterior names its level `draw` |
| counts | `num_chains` beside `n_particles` (`pyabc_smcabc`) and `n_windows` (`blackjax_rwmh`) |
| bounds | `min_acceptance_rate` beside `minimum_epsilon` and `max_total_nr_simulations`, all in `pyabc_smcabc` |

Different algorithms may need different defaults, as SG-MCMC's step size does, but nothing records which differences are intended.

**F. Nothing lists the options.**
`registry_catalog.describe("inference")` lists each method's priority, exactness, and description, and no option.
The defaults of `blackjax_nuts` are written only in its docstring.
`mean.options` returns the eleven controls, so "options" names the controls on a `Function` and the entries of `method_options` everywhere else.

**G. Hand-kept lists of control names.**
Seven module constants each list some controls for one purpose:

| Constant | Controls |
|---|---|
| `functions/_rules.py:71` `_FORWARDED_CONTROLS` | `max_evaluations`, `num_samples`, `include_inputs`, `workflow_kind`, `exact_only`, `method_options` |
| `operations/_pushforward.py:408` `_FORWARDED_CONTROLS` | `max_evaluations`, `workflow_kind`, `method_options` |
| `operations/_pushforward.py` `_MAP_CONTROLS` | `max_evaluations`, `executor`, `executor_options`, `workflow_kind` |
| `values/_function_base.py` `_PASSED_COUNT_CONTROLS` | the two counts |
| `values/_function_base.py` `_EXECUTION_CHOICE_CONTROLS` | `executor`, `executor_options` |
| `values/_function_base.py` `_DEFAULTED_BY_NONE` | seven controls |
| `core/_compilation.py` `_STATIC_CONTROL_NAMES` | the two counts |

The two lists named `_FORWARDED_CONTROLS` disagree on which controls a call passes to its map.
Both pass `method_options`, which VI.0's "the other controls stay each call's own" forbids.

**H. `conversions` mixes two levels.**
One flat mapping holds the selection keys `method` and `exact_only` beside the converter's options, and `functions/_normalization.py:229` pops them apart.

**I. Reads are recorded for one control only.**
`_note_num_samples_used` records each read of `num_samples`, for provenance (`core/_call_context.py:1053`).
No other control's reads are recorded.

## 4. The proposed design: the core contract

### 4.1 Controls, options, and settings

A call is configured at three levels, which differ in who defines the name.
ProbPipe defines the controls, a registered entry defines its options, and `config` holds the session's defaults of some controls.

> [!IMPORTANT]
> **Definition (control).** A *control* is a name that ProbPipe defines for every `Function` call, set by the decorator or by `using`, with one meaning in every call. The controls form a closed set: `max_evaluations`, `num_samples`, `include_inputs`, `method`, `exact_only`, `conversions`, `method_options`, `executor`, `executor_options`, `raw`, and `workflow_kind`.

> [!IMPORTANT]
> **Definition (option).** An *option* is a name that one registered entry defines, such as an inference method, an executor, a converter, or an approximate capability. It is passed in the options mapping of the control that selects such entries: `method_options` for a route's method, `executor_options` for an executor, and the `method_options` of a `conversions` entry for a converter.

> [!IMPORTANT]
> **Definition (setting).** A *setting* is a field of a `config` section. A setting of `config.sampling` is the session's default of the control of the same name.

Consequences:

1. **No option has a control's name.** Each name has one owner, so `method_options={"num_samples": 9}` is refused (§ 5.4).
2. **`Function.options` becomes `Function.controls`.** "Options" then means only the entries of an options mapping.
3. **`method_options` holds every option of a method,** whether or not it is a budget. V.2's "numerical budgets" becomes "the options of the selected method".

### 4.2 One declaration for every control, option, and setting

Every control, option, and setting is declared once, by one `Option` object.
Every reader reads that declaration: validation, defaults, the catalog, `describe_using`, provenance, and the documentation.

```python
@dataclass(frozen=True)
class Option:
    name: str
    default: Any                                  # REQUIRED for an option the caller must give
    check: Callable[[Any], Any]                   # returns the admitted value, or raises TypeError or ValueError
    description: str                              # one line, shown by the catalog and describe_using
    kind: Literal["static", "inert"] = "static"   # static: it can change what a compiled JAX program computes
    passes_as: str | None = None                  # the backend argument it sets, such as "iter_sampling"

    def with_default(self, default: Any) -> Option: ...

class Options(Mapping[str, Option]):              # ordered and immutable
    def __init__(self, *options: Option, exclusive: Iterable[tuple[str, ...]] = ()) -> None: ...
    def resolve(self, given: Mapping[str, Any], *, owner: str) -> Mapping[str, Any]: ...
        # every name declared, every value checked, no two exclusive names given, every default filled in
    @classmethod
    def from_signature(cls, callable_: Callable[..., Any]) -> Options: ...   # keyword parameters as options
```

Each owner holds one `Options`:

| Owner | Its declaration | It replaces |
|---|---|---|
| `Function` | `CONTROLS` | `_CONTROL_DEFAULTS`, `_DEFAULTED_BY_NONE`, `_validate_options` |
| a `config` section | its `settings` | `_Setting` |
| a dispatch method | `options`, a class attribute of `BaseDispatchMethod` | `_method_options`, `_check_options`, `_refuse_unread`, `_executor_option_names`, `nuts_controls`, `_sample_count` |
| an approximate capability | `Options.from_signature` of its method, such as `_condition_on` | the free-form `**options` |

A registry resolves the selected method's options before it calls `check` and `execute`.
A method therefore receives every option it declares, checked, with its default filled in:

```python
class BlackJAXNuts(InferenceMethod):
    options = Options(
        common_options["num_draws"],
        common_options["num_warmup"],
        common_options["num_chains"],
        common_options["init"],
        common_options["step_size"].with_default(0.1),
        common_options["target_accept_prob"],
        common_options["max_tree_depth"],
    )

    def execute(self, target, /, *, key, num_draws, num_warmup, num_chains, init, step_size,
                target_accept_prob, max_tree_depth): ...
```

Consequences:

1. **No method writes a default inline.** `options.get("num_warmup", 500)` goes, so each default is stated once.
2. **Provenance records the resolved options,** defaults included, so a recorded run still says what it ran after a default changes.
3. **Exclusive options are declared.** pyabc's `epsilon` and `epsilon_quantile` (today `eps` and `eps_alpha`) are one exclusive pair, so giving both raises `ValueError` instead of ignoring one.

**Decision 4: how an entry declares its options.**
- **Recommendation.** Explicit `Options`, as above. One type serves controls, settings, and options, and it carries checks and descriptions that a signature cannot.
- **Alternative.** Keyword parameters of `execute`, as the executors use today. A signature gives names and defaults, and a docstring gives descriptions, but the checks stay in each method. Under the recommendation, capabilities still use signatures through `Options.from_signature`, because a capability is a method of a user's class.

### 4.3 When each name and value is checked

A control's owner is known when `using` is called.
An option's owner is the entry the call selects, which only the call knows.
The checks therefore split by what they need:

| Check | At `using` or the decorator | At the call and at `check` |
|---|---|---|
| a control's name | `TypeError`, listing the controls | — |
| a control's value | the control's `check` | — |
| an options mapping's keys: strings, and none a control's name | `TypeError` | — |
| the value of an option with a common name (§ 4.5) | the common option's `check` | — |
| an option's name | — | `TypeError` unless some candidate declares it |
| an option's value | — | the selected entry's `check` |
| an option that the selected entry does not declare | — | `UnusedControlWarning` (§ 4.4) |

> [!IMPORTANT]
> **Definition (candidates).** The *candidates* of a call for an options mapping are the entries the call may select with it.
> 1. For `method_options`: the entry that `method` names, when the call names one; otherwise every method of the operation's registry routes and every approximate capability of its capability routes, opt-in-only methods excluded.
> 2. For `executor_options`: the executor that `executor` names, or otherwise every executor that automatic selection may pick.

Consequences:

1. **A misspelled option always raises.** `mean.using(method_options={"bogus": 1})(n)` raises `TypeError`, because no route of `mean` takes `bogus`.
2. **Another candidate's option warns.** `long_warmup` of § 3.2 B warns on the empirical prior, because `blackjax_nuts`, a candidate, takes `num_warmup`.
3. **A named method is strict.** `condition_on.using(method="empirical_reweighting", method_options={"num_warmup": 2000})` raises, because the one candidate does not take `num_warmup`.
4. **An unknown name raises `TypeError` everywhere,** as Python does for an unexpected keyword argument, and an inadmissible value raises `ValueError`. The executors' `ValueError` for an unknown option becomes `TypeError`.
5. **The decision of 2026-10-20 holds.** Nothing `using` checks depends on the registries, since a common option's check needs no entry.

**Decision 3: options given without `method`.**
- **Recommendation.** Check the names against the candidates, as above. A configured copy such as `condition_on.using(method_options={"num_chains": 8})` serves every model, and a misspelling still raises.
- **Alternative.** Require `method` whenever `method_options` is given, as V.9 requires `executor` with `executor_options` on an operation. It is simpler, and it forbids the common case of setting `num_chains` while automatic selection picks the method.

### 4.4 A setting the call does not read warns

A user who sets a control or an option expects it to change what the call does.
When it changes nothing, the call says so.

> [!IMPORTANT]
> **Definition (given and read).**
> 1. A control is *given* to a call when the `using` view the call runs on sets it. An option is *given* when the call's options mapping holds it.
> 2. A control is *read* by a call when code that runs for the call looks up its value, such as a route's `check` or `execute`, the lift's plan, a converter, or an executor. A read by a call that a derived route makes, of a control the derived route passed down, is also a read by the derived call.
> 3. An option is *read* when the entry that declares it runs for the call.

> [!IMPORTANT]
> **Rule (unread settings).** A call that succeeds warns once with `UnusedControlWarning`, a subclass of `UserWarning`, for each control and option it was given and did not read. The message names the setting and the route or entry that ran, and it names the setting that governs that work where one does.

The call context keeps a **read ledger**, which generalizes today's `_NumSamplesUsed`.
A route looks up its controls through `call.controls`, a mapping that notes each lookup in the ledger.
Code that only reports, such as provenance, uses `call.controls.peek`, which notes nothing.
When the call closes, the given settings that the ledger lacks are the unread ones.
The executor warning of V.9, rule 7 ("condition_on named executor='thread', but … 'thread' ran nothing"), becomes the `executor` case of this rule, and its message is kept.

The messages follow `STYLE_GUIDE.md` § 9.3:

```text
square did not use num_samples=1000, because a call on a distribution evaluates square
    max_evaluations=256 times. Set max_evaluations=1000 to evaluate it more often
mean did not use num_samples=500, because its route closed_form computes the mean exactly
mean did not use include_inputs=True, because the call lifted no argument
condition_on did not use the method option num_warmup=2000, because the method it ran,
    empirical_reweighting, takes no options
condition_on did not use num_samples=50, because blackjax_rwmh keeps num_chains × num_draws
    = 1 × 20 draws. Set the draws per chain with method_options={"num_draws": ...}
```

Routes and entries also declare the controls they can read, as `reads`.
`check` reports from `reads` which given settings the selected route cannot read (§ 4.7), and a test asserts that every call's ledger is within its selected route's `reads`.

Consequences:

1. **#675 is resolved,** with its option A, a warning per call, and its rule for counts passed down.
2. **A configured copy warns only on the calls it does not fit,** so `mean.using(num_samples=500)` is silent on a law that has no closed-form mean.
3. **`config` blocks and session values never warn,** because they are defaults for every call in their scope.
4. **The test suite turns the warning into an error** with `filterwarnings = error::probpipe.UnusedControlWarning`, so ProbPipe's own code never passes a setting that does nothing.

**Decision 1: warn or raise.**
- **Recommendation.** Warn. The result is correct for the route that ran, and a reusable configured copy should not fail on the calls it does not fit. A user who wants strictness writes `warnings.simplefilter("error", UnusedControlWarning)`.
- **Alternative.** Raise. It is stricter, and it makes each configured copy usable only on the routes its first users meet.

**Decision 2: controls set by the decorator.**
- **Recommendation.** They do not warn. `@function(max_evaluations=500)` states the function's policy for every call that lifts, so its calls on values leave it unread by design. `using` asks something of the calls at hand, so only its controls warn.
- **Alternative.** Both warn. The rule is shorter, and a function whose author sets a lift budget warns on every call on values.

### 4.5 One name, one meaning: the common options

A name that several entries read means one thing in all of them.
`probpipe.options.common_options` declares each such name once, with its meaning, its check, and a default.
An entry that takes the name uses that declaration and may change only its default.

**The naming rule.**

1. `num_<things>` is the number of things an entry produces or performs.
2. `max_<things>` is an upper bound, which the entry may stop short of.
3. `min_<quantity>` is a lower bound.
4. `chunk_size` is the number of evaluation points in one chunk, the glossary's name, and `batch_size` is the number of data points in one minibatch.

The two counts already follow it: `num_samples` draws that many, and `max_evaluations` may enumerate fewer (§ 5.1).

The first common options:

| Name | Meaning | Check | Default |
|---|---|---|---|
| `num_draws` | the draws in each chain of the result, the size of its `draw` level; a method without chains has one | positive integer | 1000 |
| `num_warmup` | the steps each chain takes and discards before it keeps draws | nonnegative integer | 500 |
| `num_chains` | the independent chains, the size of the result's `chain` level | positive integer | 4 |
| `init` | where each chain starts, in the four forms that `blackjax_nuts`'s docstring states today | one of the forms | `None` |
| `step_size` | the integrator's step size, or its initial value when warmup adapts it | positive real | per method |
| `target_accept_prob` | the acceptance rate that step-size adaptation targets | real in (0, 1) | 0.8 |
| `max_tree_depth` | the most doublings of a NUTS trajectory | positive integer | 10 |
| `batch_size` | the data points in one minibatch | positive integer | required |
| `progress_bar` | whether the backend prints its progress; `inert` | bool | per method |
| `chunk_size` | the evaluation points in one chunk | positive integer or `None` | `None` |
| `max_workers` | the most workers that run at once | positive integer or `None` | `None` |
| `retries`, `retry_errors`, `bisect`, `cancel_on_failure` | V.9's common executor options, deferred by the execution plan | as V.9 states | as V.9 states |

Consequences:

1. **Registration checks the vocabulary.** An entry that declares a common name with a different `check` is refused at registration with `TypeError`.
2. **A different default is visible.** `cmdstan_nuts`'s `num_warmup=1000` is a `with_default` in its declaration, and the catalog shows it beside the common default.
3. **The glossary's table of canonical names lists every common option,** and the code's declaration is the source both read.

### 4.6 How a control passes to other calls

Each control declares how it passes to the calls that a call makes, in one field of its declaration, `passes`.
The seven constants of § 3.2 G become reads of that field:

| `passes` | Meaning | Controls |
|---|---|---|
| `"own"` | belongs to the call alone | `method`, `method_options`, `conversions`, `include_inputs`, `raw` |
| `"to_formula"` | a derived route passes it to the calls its formula makes, unless the called `Function` sets its own (VI.0) | `max_evaluations`, `num_samples` |
| `"restricts"` | holds for every call in the call's scope, at any depth | `exact_only` |
| `"inherited"` | applies to every nested call that sets none, by V.9's rules | `executor` and `executor_options`, as one unit; `workflow_kind` |

`pushforward`, the lift's rules, and the derived routes read `passes`, and none keeps a list of its own.
A route that runs a map, such as `pushforward`'s, passes controls as a derived route does.
It therefore passes the counts and keeps `method_options` to itself, which ends the departure of § 3.2 G.

### 4.7 Finding out what a call accepts

A user asks three questions, and each answer is read from the declarations:

1. **"What can I set on this operation?"** `f.describe_using()` lists the controls, each with its value in effect and the source of that value, and the options of each candidate, grouped by route. `f.describe_using(method="blackjax_nuts")` shows one entry's options in full.
2. **"What does this method take?"** The catalog answers: `EntrySummary` gains the field `options`, which holds one `OptionInfo` per option, and `registry_catalog.describe("inference")` lists each method's option names.
3. **"What will this call use?"** `f.check(...)` returns a `CallReport` with two new fields. `options` holds the selected entry's resolved options, and `unused` names the given settings the call would warn about.

```python
>>> print(condition_on.describe_using(method="blackjax_nuts"))
condition_on.using(...)
  controls                 value    from
    method                 None     default     the route or method to run, by name
    exact_only             False    default     run exact routes only
    num_samples            1024     session     not read by blackjax_nuts
    ...
  method_options of blackjax_nuts (registry 'inference')
    num_draws              1000                 draws in each chain, the size of the 'draw' level
    num_warmup             500                  steps each chain discards before it keeps draws
    num_chains             4                    chains, the size of the 'chain' level
    init                   None                 where each chain starts
    step_size              0.1      (method)    initial step size of the adaptation
    target_accept_prob     0.8                  acceptance rate the adaptation targets
    max_tree_depth         10                   most doublings of a trajectory

>>> condition_on.using(num_samples=50).check(model, data).unused
('num_samples',)
```

The API reference of each method takes its list of options from the declaration.
A docs test asserts that each entry's docstring names every option it declares, so the two cannot drift.

**Decision 8: the name `describe_using`.**
- **Recommendation.** `describe_using`, which pairs with `using` and with the catalog's `describe`.
- **Alternative.** `using_help`, or a `describe` method on an object that `using` would become.

### 4.8 Where the contract is implemented

| Module | What it holds |
|---|---|
| `probpipe/options.py` (new, public) | `Option`, `Options`, `OptionInfo`, `REQUIRED`, `common_options`, `UnusedControlWarning` |
| `values/_function_base.py` | `CONTROLS`, which `using` and the decorator resolve through |
| `core/config.py` | each section's `settings`, as `Options` |
| `core/_dispatch.py`, `core/_catalog.py` | `BaseDispatchMethod.options`, resolution in `check` and `execute`, `EntrySummary.options` |
| `core/_call_context.py` | the read ledger, which replaces `_NumSamplesUsed` |
| `operations/_operation.py`, `functions/_call.py` | a call's candidates, route `reads`, `describe_using`, `CallReport.options` and `CallReport.unused` |

The contract deletes `_CONTROL_DEFAULTS`, `_DEFAULTED_BY_NONE`, `_validate_options`, `_method_options`, `_check_options`, `_refuse_unread`, `_sample_count`, `_executor_option_names`, `_check_executor_reads_options`, `nuts_controls`, the seven constants of § 3.2 G, and every `options.get(name, default)` in `probpipe/inference/`.

## 5. Particular cases

### 5.1 The two counts

The two counts count different work, so each has one definition and its own readers.

> [!IMPORTANT]
> **Definition (the two counts).**
> 1. `max_evaluations` is an upper bound on the number of times a call evaluates a function lifted over laws. The lift enumerates every combination of atoms when they number at most `max_evaluations`, which is exact, and otherwise it draws `max_evaluations` times.
> 2. `num_samples` is the number of draws of a law that a Monte Carlo estimate or a drawing conversion takes.

| Route or entry | Reads |
|---|---|
| a lift (`empirical_enumeration`, `sampling_lift`), and `pushforward`'s map | `max_evaluations` |
| the Monte Carlo routes of `mean`, `variance`, `cov`, `quantile`, and `expectation`; a converter that draws | `num_samples` |
| `expectation(d, f)` and `pushforward_mean` | `num_samples`, spent as the map's `max_evaluations` (VI.0) |
| an exact route; an inference method | neither |

An inference method's draws are set by its options, as `num_chains × num_draws`, because the cost of an MCMC run and its diagnostics depend on how the draws split across chains.
A `num_samples` given to `condition_on` therefore warns, naming `num_draws` (§ 4.4).

### 5.2 Method options: declared, or passed to the backend?

A method that wraps a backend, such as CmdStan or pyabc, receives two kinds of argument.
Some have a meaning across methods, such as `num_draws`.
Others belong to the backend alone, such as CmdStan's `adapt_init_phase` or pyabc's `transitions`.
The proposal takes both, in two layers:

1. **Declared options.** Each option with a meaning in ProbPipe's vocabulary is declared, and the method translates it. `passes_as` records the backend argument it sets, so `describe_using` shows `num_draws → iter_sampling` for `cmdstan_nuts`.
2. **`backend_options`.** A method may declare one more option, `backend_options`, a mapping forwarded unchanged to the backend call. ProbPipe checks that it is a mapping and that none of its keys is the `passes_as` of a declared option, so each backend argument has one spelling.

```python
condition_on.using(
    method="cmdstan_nuts",
    method_options={"num_draws": 2000, "backend_options": {"adapt_init_phase": 100}},
)(model, data)
```

A method declares `backend_options` only when its backend refuses an argument it does not know.
The rule of § 4.4 then holds through the backend's own error, which the method raises again under its own name.
Provenance records `backend_options` with the other resolved options.

**Decision 6: `backend_options`.**
- **Recommendation.** Offer it, under the conditions above. ProbPipe then declares the options users share across methods, and a user can still set every argument of the backend.
- **Alternative.** Declare every backend argument a method supports, and forward nothing. It gives the strongest checks and the longest declarations, and each declaration must follow its backend's releases.

### 5.3 The inference methods

Under § 4.5, the inference options are renamed to the common vocabulary:

| Method | Today | Proposed |
|---|---|---|
| every MCMC method, SG-MCMC, `pymc_advi` | `num_results` | `num_draws` |
| `pyabc_smcabc` | `n_particles`, `max_total_nr_simulations`, `minimum_epsilon` | `num_particles`, `max_simulations`, `min_epsilon` |
| `pyabc_smcabc` | `eps`, `eps_alpha` | `epsilon`, `epsilon_quantile`, declared exclusive (§ 4.2) |
| `pyabc_smcabc` | `sampler`, `transitions` | entries of `backend_options` |
| `blackjax_rwmh` | `n_windows` | `num_windows` |
| `pymc_nuts` | `cores` | `max_workers` |
| `tfp_nuts` | `target_accept_prob` defaults to 0.75 | the common 0.8, or a `with_default` whose reason its docstring states |

The renames break every script that sets `num_results`.
The old names are accepted for one release, with a `DeprecationWarning` that names the new one.

**Decision 5: the renames.**
- **Recommendation.** Rename, so the option names follow § 4.5 and match the result's levels `chain` and `draw`.
- **Alternative.** Keep `num_results` and the backend names, and declare them as they are. The declarations still bring checks and discovery, and the vocabulary keeps two words for draws.

### 5.4 Converters and the `conversions` control

A conversion is a nested call, which selects a converter, possibly an exact one, with options.
Its settings therefore take the shape of a call's own:

```python
predict.using(conversions={"d": {"method": "tfp", "exact_only": True, "num_samples": 500,
                                 "method_options": {"check_support": False}}})
```

1. **A `conversions` entry takes four keys:** the controls `method`, `exact_only`, and `num_samples`, and `method_options`, which holds the converter's options. Each control means what it means on a call, for that parameter's conversion alone.
2. **The converter option `num_samples` goes.** A converter that draws reads the `num_samples` in effect for its conversion: the entry's own key, and otherwise the call's control. `convert.using(method_options={"num_samples": 9})` raises `TypeError`, with the fix `convert.using(num_samples=9)`.
3. **`check_support` and `event_spec`** are declared converter options.
4. **A `conversions` entry for a parameter that does not convert** warns by § 4.4.

**Decision 7: the converter's `num_samples`.**
- **Recommendation.** Remove it, as above, and drop VI.10's clause "unless the converter's own option of that name is given".
- **Alternative.** Keep it as an option under a name of its own, such as `num_conversion_draws`. Two names then count draws of one conversion.

### 5.5 Executors

V.9's seven rules for choosing an executor stand, and the executors change only in how they declare:

1. Each executor declares `options`, which include the common executor options of § 4.5, and `_executor_option_names` goes.
2. `Option.kind` is the distinction between static and run-time options of `execution-layer.md` § 6.10: a static executor option enters a JAX program's cache key, and an inert one does not.
3. V.9's table names the `jax` executor's option `chunk_size`, which leaves `batch_size` to minibatches.
4. Options for an executor that did not run, such as `chunk_size` when `jax` compiled every evaluation point, warn by § 4.4. An option that no candidate executor declares raises by § 4.3.

### 5.6 Derived operations, maps, and requesters

1. **Derived operations** pass the controls whose `passes` is `"to_formula"` or `"restricts"` (§ 4.6). A count given to `standard_deviation` is read when the `variance` its formula calls reads it.
2. **Maps.** `pushforward` and the lift's rules pass the same controls, and `method_options` stays with the operation's own call.
3. **Requesters,** which run another operation many times, take that operation's settings as one argument, `using={...}`, checked as `using` checks its keywords. `simulation_based_calibration(using={"method": "blackjax_nuts", "method_options": {...}})` replaces its separate `method` and `method_options` arguments.

### 5.7 The decorator, a configured copy, and a `config` block

The precedence of V.2 is unchanged.
Each layer differs only in whether an unread value warns:

| Layer | Example | Warns when unread |
|---|---|---|
| decorator | `@function(max_evaluations=500)` | no (decision 2) |
| `using` view | `mean.using(num_samples=500)` | yes |
| `config` block | `with config.sampling.using(num_samples=500):` | no |
| session value | `config.sampling.num_samples = 500` | no |

Options have no `config` section, so a session cannot set `num_chains` for every `condition_on`; question 9 of § 8 asks whether it should.

### 5.8 Optimizers

The `optimize` operation of `differentiability.md` § 6.2 has an optimizer registry, whose entries follow this contract with no new rule.
It adds the common options of an iterative solver:

| Name | Meaning |
|---|---|
| `max_iterations` | an upper bound on the solver's iterations |
| `tolerance` | the convergence tolerance, in the norm the solver states |
| `learning_rate` | the step size of a first-order method |
| `init` | the starting point, in the forms of § 4.5 |

```python
optimize.using(method="optax_adam", method_options={"learning_rate": 1e-2, "max_iterations": 5000})(loss)
```

`pymc_advi` keeps `num_iterations`, since it always runs that many.

### 5.9 Entries written outside ProbPipe

A method written outside ProbPipe declares `options` from `common_options` and its own `Option`s, and its `execute` receives resolved values.
`docs/api/extending.md` shows that form in place of `_method_options` and `options.get`.
`probpipe.testing.check_executor` and a new `check_inference_method` assert the declaration rules: a run reads only declared options, no option has a control's name, and each common name keeps its check.

## 6. Edits to the other plans and to `design/`

| Plan or section | Edit |
|---|---|
| `inference-interface.md` § 5, § 7 | Each adapter of step P2 receives resolved options, so `execute(target, /, *, key, **options)` takes declared keyword parameters. `initial_vectors(target, key, num_chains)` reads the `num_chains` option, and `init`'s forms are the common option's. P2 follows step C2 below, or makes the renames of § 5.3 itself. |
| `execution-layer.md` § 6.10, and its deferred rows | Its static options are the options whose `kind` is `"static"`. The common executor options and the deferred `memo` control are declared as § 4.2 states. |
| `execution-introspection.md` | The executor warning of V.9, rule 7, is the `executor` case of § 4.4, with its message kept. `CallReport.describe()` shows the new fields `options` and `unused`. |
| `differentiability.md` § 6.2 | The optimizer registry declares its options as § 5.8 states. |
| `numeric-contract.md`, `vector-interface.md`, `batch-contract.md` | None. |
| II.7 | `BaseDispatchMethod.options`; resolution before `check` and `execute`; `EntrySummary.options`. |
| V.2 | Controls, options, and settings (§ 4.1); the checks of § 4.3; the rule of § 4.4; `Function.controls`; `describe_using`. |
| V.4 | The shape of a `conversions` entry (§ 5.4). |
| V.9 | Executor declarations; `TypeError` for an unknown option; `chunk_size` for `jax` (§ 5.5). |
| VI.0, VI.6, VI.10 | Passing by `passes` (§ 4.6); `method_options` holds every option; the converter's `num_samples` goes. |
| glossary | "control", "option", "setting", and "common option"; the common options in the table of canonical names. |

This plan answers #675 in § 4.4.
#585 is its converse, a budget that was read and ruled out an exact route, and it keeps its own proposal.

## 7. Steps

Each step is one PR.

1. **C1, declarations:** `probpipe/options.py`; the controls and the `config` settings declared as `Options`; `passes` in place of the seven constants of § 3.2 G; `Function.controls`, with `Function.options` deprecated.
2. **C2, entries:** `BaseDispatchMethod.options` and resolution in the registries; every inference method, executor, converter, and approximate capability declared; `common_options`; the renames of § 5.3 with deprecated aliases; the helpers of § 4.8 deleted; `EntrySummary.options`.
3. **C3, candidates:** option names checked against a call's candidates (§ 4.3); `TypeError` for every unknown name; the `conversions` shape of § 5.4; `simulation_based_calibration(using=...)`.
4. **C4, reads:** the read ledger, `call.controls.peek`, route `reads`, and `UnusedControlWarning`; V.9's executor warning folded in; the test suite's warning filter.
5. **C5, discovery:** `describe_using`, `CallReport.options` and `CallReport.unused`, the catalog's options, and the docs test of § 4.7.

`design/` changes with each step, as § 6 lists.

**Tests.**

- One conformance test, parametrized over every registered entry, asserts that its declaration keeps § 4.5's vocabulary and that a run reads only what it declares.
- One test, parametrized over the rows of § 3.2 A, asserts each warning and its message.
- Each of consequences 1 to 3 of § 4.3 has a test.

**What users see.**

- **Breaking:** the renamed inference options and `Function.options`, each with an alias for one release; the converter option `num_samples`; the shape of `conversions`; `TypeError` in place of `ValueError` for an unknown executor option; `simulation_based_calibration`'s `method` and `method_options`.
- **Fixed:** every row of § 3.2 A warns; the configured copy of § 3.2 B warns where it raised; a misspelled option raises on every route.
- **Added:** `describe_using`; `CallReport.options` and `CallReport.unused`; options in the catalog; `probpipe.options`; `backend_options`.

## 8. Questions for the maintainer

1. **Warn or raise for an unread setting.** Recommended: warn (§ 4.4).
2. **Controls set by the decorator.** Recommended: they do not warn (§ 4.4).
3. **Options without `method`.** Recommended: their names are checked against every candidate (§ 4.3).
4. **How entries declare.** Recommended: explicit `Options`, with signatures for capabilities only (§ 4.2).
5. **The inference renames.** Recommended: rename to the common vocabulary, with aliases for one release (§ 5.3).
6. **`backend_options`.** Recommended: offer it, for backends that refuse unknown arguments (§ 5.2).
7. **The converter's `num_samples`.** Recommended: remove it (§ 5.4).
8. **The name `describe_using`.** Recommended: keep it (§ 4.7).
9. **Session defaults for options.** Should `config` gain a section, such as `config.inference.using(num_chains=8)`? Recommended: not now, since a configured copy covers the case and no user has asked.
10. **`tfp_nuts`'s acceptance target.** Recommended: the common 0.8, unless TFP's 0.75 has a reason its docstring can state (§ 5.3).

## Appendix A. Where settings are declared and read today

| Kind | Declared in | Read with |
|---|---|---|
| controls | `values/_function_base.py`: `_CONTROL_DEFAULTS`, `_DEFAULTED_BY_NONE`, `_validate_options`, `_checked_count` | `call.controls[...]`, `f.options[...]`, `resolved_controls[...]`, `_num_samples_in_effect` |
| `config` settings | `core/config.py`: a `_Setting` per name | the section's properties |
| inference options | `_method_options` in ten modules of `probpipe/inference/` | `options.get(...)` in each `execute`; `nuts_controls` and `chain_initial_states` in `_inference_utils.py` |
| executor options | the parameters of `execute`, read by `_executor_option_names` (`core/_executors.py:1062`) | the parameters |
| converter options | `_refuse_unread` calls in `families/_converters.py` | `options.get(...)`, `_sample_count`, `_declared` |
| capability options | none; `_condition_on(**options)` takes any | each implementation |

## Appendix B. Readings this plan adopts

1. **"Budgets" in V.2** reads as "options": every option a method takes goes through `method_options`, as the code already does for `init` and `summary_fn`.
2. **VI.0's "the other controls stay each call's own"** decides against both `_FORWARDED_CONTROLS` lists, which pass `method_options` to a map's call.
3. **V.9's `jax` option `batch_size`** reads as the glossary's `chunk_size`, because it counts evaluation points per chunk; the `jax` executor's code takes no option yet.
4. **An exact capability route that drops `method_options`** (`operations/_operation.py:614`) is right to ignore them, and only its silence departs from this plan, which § 4.4 ends.
