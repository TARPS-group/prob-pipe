from typing import Callable, Optional, Dict, Any, get_type_hints, Type, ClassVar, FrozenSet, Iterable, Set
from types import MappingProxyType
import functools
from dataclasses import dataclass
import inspect

from prefect import flow, task

from .distributions import Distribution, EmpiricalDistribution
from .multivariate import Multivariate
from numpy.typing import NDArray


__all__ = [
    "Module",
]

_MISSING = object()
_DISTR_BASE = (Distribution, Multivariate)
_DISTR_INST = (Distribution, Multivariate, EmpiricalDistribution)

@dataclass
class InputSpec:
    """Specification for a module input.

    Describes the expected type, requirement status, and default value for
    an input to a :class:`Module`. Used internally by :meth:`Module.set_input`
    to enforce validation and provide defaults.

    Attributes:
        type: Expected Python type for this input.
        required: Whether this input must be explicitly provided.
        default: Default value; `_MISSING` indicates no default provided.
    """
    type: Optional[Type] = None
    required: bool = False
    default: Any = _MISSING #_MISSING means "no default"


class Module(object):
    """Base class for all ProbPipe modules.

    Provides dependency injection, input specification, type validation, and
    seamless integration with Prefect tasks and flows. Each subclass defines
    a probabilistic computation that can depend on other modules and be
    composed into larger pipelines.

    Typical usage:
        1. Subclass :class:`Module` and declare required dependencies via
           the class variable :pyattr:`DEPENDENCIES`.
        2. Define inputs using :meth:`set_input`.
        3. Register computational functions using :meth:`run_func`.
        4. Call registered functions as Prefect tasks or flows.

    Notes:
        - Distribution-type inputs are automatically converted when compatible.
        - Functions registered via :meth:`run_func` are exposed as Prefect tasks
          or flows depending on the `as_task` flag.
        - Dependencies are validated at both initialization and runtime.

    Attributes:
        DEPENDENCIES: Names of required dependency modules.
        dependencies: Injected dependency instances.
        inputs: Declared input specifications.
        _run_funcs: Registered computational functions.
        _conv_num_samples: Default number of samples for distribution conversion.
        _conv_by_kde: Whether to use KDE-based conversion.
        _conv_fit_kwargs: Additional keyword arguments for fitting.
    """

    
    # REQUIRED_DEPS: ClassVar[FrozenSet[str]] = frozenset()
    # DEPENDENCIES: ClassVar[Dict[str, Type['Module']]] = {}
    # DEPENDENCIES: ClassVar[Set[str]] = set()
    DEPENDENCIES: ClassVar[Dict[str, Type['Module']]] = MappingProxyType({})



    def __init__(
        self,
        required_deps: Optional[Iterable[str]] = None,
        conversion_by_KDE: bool = False,
        conversion_num_samples: int = 1024,
        conversion_fit_kwargs: Optional[dict] = None,
        **dependencies: 'Module',
    ):
        """Initializes the module and validates dependencies.

        Args:
            required_deps: Explicitly required dependency
                names to check during initialization.
            conversion_by_KDE: Whether to use kernel density estimation (KDE)
                for automatic distribution conversion. Defaults to ``False``.
            conversion_num_samples: Number of samples to draw for empirical-to-
                parametric conversions. Defaults to ``1024``.
            conversion_fit_kwargs: Extra keyword arguments forwarded
                to distribution fitting methods. Defaults to ``None``.
            **dependencies: Dependency modules to inject into this module.

        Raises:
            RuntimeError: If required dependencies are missing or unexpected ones are provided.
            TypeError: If a dependency is not a subclass of :class:`Module`.
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
        """Defines input specifications for the module.

        Each keyword argument specifies either an :class:`InputSpec`
        (for type and requirement) or a simple default value.

        Example:
            >>> self.set_input(x=InputSpec(type=NDArray, required=True), sigma=1.0)

        Args:
            **input_defaults: Key–value pairs of input names and their specifications.
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
        """Registers a computational function as a Prefect-executable task or flow.

        This decorator performs automatic input validation, dependency injection,
        and Prefect task wrapping. Registered functions become callable attributes
        of the module (e.g., ``module.my_function()``).

        Steps performed:
            1. Infers input schema and type hints.
            2. Validates and injects dependencies.
            3. Applies type checking and optional conversions.
            4. Wraps as a Prefect task or flow depending on ``as_task``.

        Args:
            f: User-defined function implementing the computation.
            name: Custom function name override.
                Defaults to the original function name.
            as_task: Whether to register the function as a Prefect task
                (``True``) or flow (``False``). Defaults to ``True``.
            return_type: Expected return type for validation.

        Returns:
            Callable: The decorated Prefect task or flow.

        Raises:
            RuntimeError: If a run function with the same name is already registered.
            TypeError: If inputs or dependencies fail validation.
        """
        
        run_name = name or f.__name__
        sig = inspect.signature(f)

        if run_name in self._run_funcs:
            raise RuntimeError(f"Run function '{run_name}' already registered")
        
        # infering inputs from signature at registration time 
        def _autofill_inputs_from_signature(func: Callable, run_name: str) -> None:
            """Infers input specifications from a function’s signature and type hints.

            Parses parameter annotations and default values to automatically populate
            input schemas used for validation at runtime. Dependencies are excluded
            from input registration, as they are injected automatically.
    
            Args:
                func: Function whose signature is analyzed.
                run_name: Name under which the run function will be registered.
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
            """Validates provided inputs and fills missing defaults.

            Ensures that all required inputs are present, fills in defaults where
            applicable, and checks for unknown input keys.
    
            Args:
                kwargs: Input keyword arguments provided at runtime.
                run_name: Name of the registered run function being validated.
    
            Returns:
                Validated and completed input mapping.
    
            Raises:
                TypeError: If required inputs are missing or unexpected inputs are given.
            """
            
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
            """Ensures all declared dependencies are available at runtime.

            Raises:
                RuntimeError: If any required dependencies are missing.
            """
            
            missing = [k for k in self.DEPENDENCIES if k not in self.dependencies]
            if missing:
                raise RuntimeError(
                    f"Missing required dependencies at runtime: {missing}. Available: {list(self.dependencies.keys())}"
                )
                    
        def _is_distribution_instance(v) -> bool:
            """Checks whether a value is an instance of a supported Distribution class.

            Args:
                v: Value to inspect.
    
            Returns:
                ``True`` if ``v`` is a distribution instance, ``False`` otherwise.
            """
            
            return isinstance(v, _DISTR_INST)

        def _is_distribution_type(tp) -> bool:
            """Checks whether a type hint refers to a known Distribution type.

            Args:
                tp: Type annotation to inspect.
    
            Returns:
                ``True`` if the type is a subclass or base of a Distribution type.
            """
            
            return isinstance(tp, type) and (tp in _DISTR_BASE or issubclass(tp, _DISTR_BASE))
            
        def type_check(args, kwargs, *, f):
            """Performs type checking and automatic distribution conversion.

            Validates argument types against type hints and, when necessary,
            automatically converts compatible distribution instances to the
            expected subclass using their ``from_distribution`` method.
    
            Args:
                args: Positional arguments passed to the function.
                kwargs: Keyword arguments passed to the function.
                f: Function being type-checked.
    
            Returns:
                The possibly updated ``(args, kwargs)`` pair.
    
            Raises:
                TypeError: If an argument fails type validation or conversion fails.
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
    
                # ---- Plain class annotation (e.g., NDArray, int, float, ...) ----
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
            """Executes the registered run function with validation and dependency injection.

            This wrapper enforces runtime validation of inputs and dependencies,
            performs type checking and conversion, injects required dependencies,
            and finally executes the underlying user function.
    
            Args:
                *args: Positional arguments for the function.
                **kwargs: Keyword arguments for the function.
    
            Returns:
                Result returned by the wrapped function.
            """
            _ensure_dependencies_available()  

            bound_args = sig.bind(*args, **kwargs)  # bind positional and keyword args
            bound_args.apply_defaults()
            all_args = bound_args.arguments  # OrderedDict of all parameters with their values


            # 1) validating inputs against the per-run schema
            # user_kwargs = _ensure_inputs_satisfied(kwargs, run_name=run_name)
            user_kwargs = _ensure_inputs_satisfied(all_args, run_name=run_name)

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
        # if as_task:
        #     pf = task(wrapper)  # no validate_parameters option for task
        # else:
        #     pf = flow(validate_parameters=False)(wrapper)
        pf = pr_annot(wrapper)  # now it works without validate_parameters option


        # Register and assign as attribute for direct call
        self._run_funcs[run_name] = pf
        setattr(self, run_name, pf)
        return pf

    def __repr__(self):
        """Return a compact summary representation of the module."""
        return f"<module deps={list(self.dependencies.keys())} inputs={list(self.inputs.keys())} run_funcs={list(self._run_funcs.keys())}>"

    def __str__(self):
        """Return a human-readable, multi-line summary of the module configuration."""
        deps = ", ".join(self.dependencies.keys()) or "None"
        inputs = ", ".join(self.inputs.keys()) or "None"
        run_funcs = ", ".join(self._run_funcs.keys()) or "None"
        return f"module:\n  Dependencies: {deps}\n  Inputs: {inputs}\n  Run Functions: {run_funcs}"
