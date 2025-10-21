from typing import Callable, Optional, Dict, Any, get_type_hints, Type, ClassVar, FrozenSet, Iterable, Set
import functools
from dataclasses import dataclass
import inspect

from prefect import flow, task

from .distributions import Distribution, EmpiricalDistribution, BootstrapDistribution
from .multivariate import Multivariate


__all__ = [
    "Module",
]

_MISSING = object()
_DISTR_BASE = (Distribution, Multivariate)
_DISTR_INST = (Distribution, Multivariate, EmpiricalDistribution, BootstrapDistribution)

@dataclass
class InputSpec:
    type: Optional[Type] = None
    required: bool = False
    default: Any = _MISSING #_MISSING means "no default"


class Module(object):
    """
    Base class for modules with dependency injection, input specification,
    type checking, and Prefect task/flow integration.
    """

    
    # REQUIRED_DEPS: ClassVar[FrozenSet[str]] = frozenset()
    # DEPENDENCIES: ClassVar[Dict[str, Type['Module']]] = {}
    DEPENDENCIES: ClassVar[Set[str]] = set()



    def __init__(
        self,
        required_deps: Optional[Iterable[str]] = None,
        conversion_by_KDE: bool = False,
        conversion_num_samples: int = 1024,
        conversion_fit_kwargs: Optional[dict] = None,
        **dependencies: 'Module',
    ):
        """
        Initialize the module.

        Args:
            required_deps: Optional iterable of required dependency names.
            conversion_by_KDE: Use KDE for distribution conversion if True.
            conversion_num_samples: Number of samples for distribution conversion.
            conversion_fit_kwargs: Additional kwargs for distribution fit.
            dependencies: Named module dependencies.
        """

        # Enforce that all declared dependencies are provided
        missing = [name for name in self.DEPENDENCIES if name not in dependencies]
        if missing:
            raise RuntimeError(f"Missing required dependencies: {missing}")

        # Enforce no unexpected dependencies
        unexpected = [name for name in dependencies if name not in self.DEPENDENCIES]
        if unexpected:
            raise RuntimeError(f"Unexpected dependencies provided: {unexpected}")

        # Enforce that each dependency is an instance of the declared type and a Module subclass
        for name, dep_instance in dependencies.items():
            if not isinstance(dep_instance, Module):
                raise TypeError(f"Dependency '{name}' must be Module subclass instance; got {type(dep_instance)}")
                    
                

        # Store dependencies in an internal dict
        # self.dependencies: Dict[str, Module] = dependencies.copy()
        self.dependencies = dict(dependencies)

        self.inputs: Dict[str, Dict[str, Any]] = {}  
        self._run_funcs: Dict[str, Callable] = {}   # Registered run functions

        self._conv_num_samples = conversion_num_samples
        self._conv_by_kde = conversion_by_KDE
        self._conv_fit_kwargs = conversion_fit_kwargs or {}

        # Per-run-function input specification (built from signature or overridden)
        self._inputs_for_run: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Instance-level set of required dependency names
        # self.required_deps: set[str] = set(required_deps) if required_deps is not None else set(self.REQUIRED_DEPS)

        # Prefect decorated flow or task placeholders for last registered run func
        self._prefect_task: Optional[Callable] = None
        self._prefect_flow: Optional[Callable] = None

    # def require_deps(self, *names: str) -> None:
    #     """
    #     Add required dependencies at runtime.

    #     Args:
    #         names: Names of dependencies to require.
    #     """
    #     self.required_deps.update(names)

    def set_input(self, **input_defaults):
        """
        Set input specifications or defaults.

        Args:
            input_defaults: Mapping from input name to InputSpec or default value.
        """
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

    def run_func(
            self, 
            f: Callable, 
            *, 
            name: Optional[str] = None, 
            as_task: bool = True, 
            return_type: Optional[type] = None,
            ) -> Callable:
        """
        Register a function as a run function decorated with Prefect task or flow.

        Args:
            f: The user function to register.
            name: Optional name for the run function; defaults to f.__name__.
            as_task: If True, decorate as Prefect task; else decorate as flow.
            return_type: Expected return type; if provided, enforces or attempts conversion.

        Returns:
            The decorated Prefect callable (task or flow).
        """
        run_name = name or f.__name__
        sig = inspect.signature(f)

        if run_name in self._run_funcs:
            raise RuntimeError(f"Run function '{run_name}' already registered")
        
        # infering inputs from signature at registration time 
        def _autofill_inputs_from_signature(func: Callable, run_name: str) -> None:
            """
            Autofill input specs for a run function based on its signature and annotations.
            """
            hints = get_type_hints(func)
            specs: Dict[str, Dict[str, Any]] = {}
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue

                # classifying required vs optional by default
                has_default = (param.default is not inspect._empty)
                default_val = param.default if has_default else _MISSING
                required = not has_default

                ann = hints.get(pname)
                ptype = ann if isinstance(ann, type) else None

                # not treating the dependency params as inputs. they’ll be injected by name
                if pname in self.dependencies:
                    continue

                specs[pname] = {'type': ptype, 'required': required, 'default': default_val}

            # merging any staged defaults from set_input() done before run registration
            if "_default" in self._inputs_for_run:
                specs = {**specs, **self._inputs_for_run["_default"]}
                del self._inputs_for_run["_default"]

            self._inputs_for_run[run_name] = specs
        
        
        def _ensure_inputs_satisfied(kwargs: Dict[str, Any], *, run_name: str) -> Dict[str, Any]:
            # using the per-run input schema
            input_specs = self._inputs_for_run.get(run_name, {})
            merged = dict(kwargs)

            # Filling defaults
            for k, meta in input_specs.items():
                if k not in merged and meta.get('default', _MISSING) is not _MISSING:
                    merged[k] = meta['default']

            # Missing required?
            missing = [k for k, meta in input_specs.items()
                       if meta.get('required') and k not in merged and meta.get('default', _MISSING) is _MISSING]
            if missing:
                raise TypeError(f"Missing required inputs: {missing}")

            # Unknown keys? (ignore names that are parameters of the function but not inputs, e.g., deps)
            allowed = set(input_specs.keys())
            fn_params = set(sig.parameters.keys()) - {"self"}
            unknown = [k for k in merged.keys() if k not in allowed and k not in fn_params]
            if unknown:
                raise TypeError(f"Unknown inputs provided: {unknown}. Declared inputs are: {sorted(allowed)}")

            return merged
        
        # def _ensure_dependencies_available():
        #     missing = [k for k in self.required_deps if k not in self.dependencies]
        #     if missing:
        #         raise RuntimeError(
        #             f"Missing required dependencies: {missing}. Have: {list(self.dependencies.keys())}"
        #         )
        def _ensure_dependencies_available():
            missing = [k for k in self.DEPENDENCIES if k not in self.dependencies]
            if missing:
                raise RuntimeError(
                    f"Missing required dependencies at runtime: {missing}. Available: {list(self.dependencies.keys())}"
                )
                    
        def _is_distribution_instance(v) -> bool:
            return isinstance(v, _DISTR_INST)

        def _is_distribution_type(tp) -> bool:
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
                    #print("It is a distribution")
                    #print(f"value is {value}")
                    #print(f"expected is {expected}")

                    # Specific subclass (not the base)
                    if expected not in _DISTR_BASE and issubclass(expected, _DISTR_BASE):
                        if not isinstance(value, expected):
                            print("Conversion Happening")
                            print(f"converting from {value} to {expected}")
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

            return args, kwargs
        
        
        #AUTO FILLING THE INPUTS HERE
        _autofill_inputs_from_signature(f, run_name)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            _ensure_dependencies_available()  

            # 1) validating inputs against the per-run schema
            user_kwargs = _ensure_inputs_satisfied(kwargs, run_name=run_name)

            # 2) type checking + conversions on inputs only
            _, user_kwargs = type_check((), user_kwargs, f=f)

            # 3) inject dependencies by name (only the ones the function actually accepts)
            #Injecting dependencies after validation/type_check so they aren’t treated as “unknown inputs”
            sig_params = set(sig.parameters.keys())
            dep_kwargs = {k: v for k, v in self.dependencies.items() if k in sig_params and k not in user_kwargs}

            call_kwargs = {**dep_kwargs, **user_kwargs}

            # 4) call and (optionally) post-convert return type
            result = f(**call_kwargs)

            return result

            
        pr_annot = task if as_task else flow
        pf = pr_annot(wrapper)


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