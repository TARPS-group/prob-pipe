from typing import Callable, Optional, Dict, Any, get_type_hints, Type, ClassVar, FrozenSet, Iterable
import functools
from prefect import flow, task
from core.distributions import Distribution, EmpiricalDistribution, BootstrapDistribution
from core.multivariate import Normal1D, Multivariate
from makefun import with_signature
import inspect

_MISSING = object()
_DISTR_BASE = (Distribution, Multivariate)
_DISTR_INST = (Distribution, Multivariate, EmpiricalDistribution, BootstrapDistribution)

class module:
    # Contract-level default (subclasses can override this)
    REQUIRED_DEPS: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, required_deps: Optional[Iterable[str]] = None,  conversion_by_KDE: bool = False,
                 conversion_num_samples: int = 1024, conversion_fit_kwargs: Optional[dict] = None, **dependencies):
        
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
            if isinstance(spec, dict):
                self.inputs[key] = {
                    'type': spec.get('type'),
                    'required': spec.get('required', False),
                    'default': spec.get('default', _MISSING),
                }
            else:
                # Interpret plain value as default only
                self.inputs[key] = {
                    'type': None,
                    'required': False,
                    'default': spec,
                }

    def run_func(self, f: Callable, *, name: Optional[str] = None, as_task: bool = False, return_type: Optional[type] = None):
        run_name = name or f.__name__
        sig = inspect.signature(f)


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
                raise TypeError(f"Unknown inputs provided: {unknown}. Declared inputs are: {list(self.inputs.keys())}")

            return merged
        
        def _ensure_dependencies_available():
            missing = [k for k in self.required_deps if k not in self.dependencies]
            if missing:
                raise RuntimeError(
                    f"Missing required dependencies: {missing}. Have: {list(self.dependencies.keys())}"
                )
                    
        def _is_distribution_instance(v) -> bool:
            return isinstance(v, _DISTR_INST)

        def _is_distribution_type(tp) -> bool:
            return isinstance(tp, type) and (tp in _DISTR_BASE or issubclass(tp, _DISTR_BASE))
            
        def type_check(args, kwargs, *, f):
            hints = get_type_hints(f)
            for name, value in kwargs.items():
                if name == "self":
                    continue

                expected = hints.get(name)
                if expected is None:
                    continue

                if _is_distribution_type(expected):
                    # Specific subclass (not base)
                    if expected not in _DISTR_BASE and issubclass(expected, _DISTR_BASE):
                        if not isinstance(value, expected):
                            converted = expected.from_distribution(
                                value,
                                num_samples=self._conv_num_samples,
                                conversion_by_KDE=self._conv_by_kde,
                                **self._conv_fit_kwargs,
                            )
                            if not isinstance(converted, expected):
                                raise TypeError(
                                    f"Conversion for '{name}' did not produce {expected.__name__}; got {type(converted).__name__}"
                                )
                            kwargs[name] = converted
                    else:
                        if not _is_distribution_instance(value):
                            raise TypeError(
                                f"Argument '{name}' must be distribution-like (Distribution/Multivariate/Empirical/Bootstrap); "
                                f"got {type(value).__name__}"
                            )
                    continue

                if isinstance(expected, type):
                    if not isinstance(value, expected):
                        raise TypeError(
                            f"Argument '{name}' expected {expected.__name__}; got {type(value).__name__}"
                        )
            return args, kwargs

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _ensure_dependencies_available()
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            bound_args = bound.arguments
            bound_args = _ensure_inputs_satisfied(bound_args)
            _, bound_args = type_check((), bound_args, f=f)
            result = f(**bound_args)

            if return_type is not None:
                if not isinstance(result, return_type):
                    if hasattr(return_type, 'from_distribution'):
                        result = return_type.from_distribution(
                            result,
                            num_samples=self._conv_num_samples,
                            conversion_by_KDE=self._conv_by_kde,
                            **self._conv_fit_kwargs,
                        )
                    else:
                        raise TypeError(
                            f"Return value is not instance of {return_type}, "
                            "and 'from_distribution' method is not available for conversion."
                        )
            return result


        # Wrap as Prefect flow or task
        if as_task:
            pf = task(wrapper)
            self._prefect_task = pf
        else:
            pf = flow(wrapper)
            self._prefect_flow = pf

        # Register and assign as attribute for direct call
        self._run_funcs[run_name] = pf
        setattr(self, run_name, pf)
        return pf

    def __repr__(self):
        return f"<module deps={list(self.dependencies.keys())} inputs={list(self.inputs.keys())} run_funcs={list(self._run_funcs.keys())}>"

    def __str__(self):
        deps = ", ".join(self.dependencies.keys()) or "None"
        inputs = ", ".join(self.inputs.keys()) or "None"
        run_funcs = ", ".join(self._run_funcs.keys()) or "None"
        return f"module:\n  Dependencies: {deps}\n  Inputs: {inputs}\n  Run Functions: {run_funcs}"