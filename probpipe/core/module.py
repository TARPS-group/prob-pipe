from typing import Callable, Optional, Dict, Any, get_type_hints, Type, ClassVar, FrozenSet, Iterable
import functools
import inspect
from prefect import flow, task
from probpipe.core.distributions import Distribution, EmpiricalDistribution, BootstrapDistribution
from probpipe.core.multivariate import Normal1D, Multivariate
from dataclasses import dataclass, field

_MISSING = object()
_DISTR_BASE = (Distribution, Multivariate)
_DISTR_INST = (Distribution, Multivariate, EmpiricalDistribution, BootstrapDistribution)

class InputSpec:
    type: Optional[Type] = None
    required: bool = False
    default: Any = _MISSING #_MISSING means "no default"


class module:
    # Contract-level default (subclasses can override this)
    REQUIRED_DEPS: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, required_deps: Optional[Iterable[str]] = None,  conversion_by_KDE: bool = False,\
                 conversion_num_samples: int = 1024, conversion_fit_kwargs: dict | None = None, **dependencies):
        
        self.dependencies: Dict[str, 'module'] = {}
        self.inputs: Dict[str, Dict[str, Any]] = {}
        self._run_funcs: Dict[str, Callable] = {}
        self._conv_num_samples = conversion_num_samples
        self._conv_by_kde = conversion_by_KDE
        self._conv_fit_kwargs = conversion_fit_kwargs or {}

        # Instance-level, editable copy of required deps
        self.required_deps: set[str] = set(required_deps) if required_deps is not None else set(self.REQUIRED_DEPS)

        for name, dep in dependencies.items():
            if name in self.dependencies:
                raise ValueError(f"Dependency '{name}' already registered")
            self.dependencies[name] = dep

        self._prefect_task: Optional[Callable] = None
        self._prefect_flow: Optional[Callable] = None

    def require_deps(self, *names: str) -> None:
        """Optional runtime configuration hook."""
        self.required_deps.update(names)

    def set_input(self, **input_defaults):
        for key, spec in input_defaults.items():
            if isinstance(spec, InputSpec):
                self.inputs[key] = {
                    'type': spec.type,
                    'required': spec.required,
                    'default': spec.default,
                }
            else:
                # old-style default only
                self.inputs[key] = {
                    'type': None,
                    'required': False,
                    'default': spec,
                }


    def run_func(self, f: Callable, *, name: Optional[str] = None, as_task: bool = False):
        """
        Register a function as a run function for this module.
        Internalized: automatically wraps with type_check and Distribution conversion.
        """
        run_name = name or f.__name__
        if run_name in self._run_funcs:
            raise RuntimeError(f"Run function '{run_name}' already registered")
        
        def _ensure_inputs_satisfied(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(kwargs)

            # Fill defaults
            for k, meta in self.inputs.items():
                if k not in merged and meta.get('default', _MISSING) is not _MISSING:
                    merged[k] = meta['default']

            # Missing required?
            missing = [k for k, meta in self.inputs.items()
                    if meta.get('required') and k not in merged and meta.get('default', _MISSING) is _MISSING]
            if missing:
                raise TypeError(f"Missing required inputs: {missing}")

            # Unknown keys?
            unknown = [k for k in merged.keys() if k not in self.inputs]
            if unknown:
                raise TypeError(f"Unknown inputs provided: {unknown}. "
                                f"Declared inputs are: {list(self.inputs.keys())}")

            return merged
        
        def _ensure_dependencies_available():
            missing = [k for k in self.required_deps if k not in self.dependencies]
            if missing:
                raise RuntimeError(
                    f"Missing required dependencies: {missing}. "
                    f"Have: {list(self.dependencies.keys())}"
                )
                    
        def _is_distribution_instance(v) -> bool:
            # value is an instance
            return isinstance(v, _DISTR_INST)

        def _is_distribution_type(tp) -> bool:
            # annotation is a type (base or subclass)
            return isinstance(tp, type) and (tp in _DISTR_BASE or issubclass(tp, _DISTR_BASE))
            
        def type_check(args, kwargs, *, f):
            """
            - If annotation expects a distribution:
                * If base type (Distribution/Multivariate): value must be distribution-like.
                * If a specific subclass: require that subclass; else convert via .from_distribution.
            - Else if annotation is a plain class: isinstance(value, that class).
            - Skips Union/Optional/generics by design.
            """
            hints = get_type_hints(f)
            bound = inspect.signature(f).bind_partial(*args, **kwargs)
            bound.apply_defaults()

            for name, value in bound.arguments.items():
                if name == "self":
                    continue

                expected = hints.get(name)
                if expected is None:
                    continue

                # ---- Distribution-typed parameter ----
                if _is_distribution_type(expected):
                    # Specific subclass (not the base)
                    if expected not in _DISTR_BASE and issubclass(expected, _DISTR_BASE):
                        if not isinstance(value, expected):
                            # === CONVERSION HERE ===
                            converted = expected.from_distribution(
                                value,
                                num_samples=self._conv_num_samples,
                                conversion_by_KDE=self._conv_by_kde,
                                **self._conv_fit_kwargs,
                            )
                            if not isinstance(converted, expected):
                                raise TypeError(
                                    f"Conversion for '{name}' did not produce {expected.__name__}; "
                                    f"got {type(converted).__name__}"
                                )
                            # write back so the wrapped function receives the right type
                            kwargs[name] = converted
 
                    else:
                        # Base type expected → accept any distribution-like instance
                        if not _is_distribution_instance(value):
                            raise TypeError(
                                f"Argument '{name}' must be a distribution-like "
                                f"(Distribution/Multivariate/Empirical/Bootstrap); got {type(value).__name__}"
                            )
                    continue

                # ---- Plain class annotation (e.g., np.ndarray, int, float, ...) ----
                if isinstance(expected, type):
                    if not isinstance(value, expected):
                        raise TypeError(
                            f"Argument '{name}' expected {expected.__name__}; got {type(value).__name__}"
                        )
                    continue


        # --- Wrapper ---
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _ensure_dependencies_available()
            kwargs = _ensure_inputs_satisfied(kwargs)
            args, kwargs= type_check(args, kwargs, f=f)
            return f(*args, **kwargs)

        # Wrap as Prefect flow or task
        if as_task:
            pf = task(wrapper)
            self._prefect_task = pf
        else:
            pf = flow(wrapper)
            self._prefect_flow = pf

        # Register
        self._run_funcs[run_name] = pf
        return pf

    def run(self, name: Optional[str] = None, **inputs):
        if not self._run_funcs:
            raise RuntimeError("No run functions registered")
        if name is None:
            if len(self._run_funcs) == 1:
                name = next(iter(self._run_funcs))
            else:
                raise ValueError("Multiple run functions registered; specify which to run")
        run_func = self._run_funcs.get(name)
        if run_func is None:
            raise ValueError(f"Run function '{name}' not found")
        return run_func(**inputs)

    def __repr__(self):
        return f"<module deps={list(self.dependencies.keys())} inputs={list(self.inputs.keys())} run_funcs={list(self._run_funcs.keys())}>"

    def __str__(self):
        deps = ", ".join(self.dependencies.keys()) or "None"
        inputs = ", ".join(self.inputs.keys()) or "None"
        run_funcs = ", ".join(self._run_funcs.keys()) or "None"
        return f"module:\n  Dependencies: {deps}\n  Inputs: {inputs}\n  Run Functions: {run_funcs}"