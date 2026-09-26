"""The distribution base class, its term spec, and minimal helpers.

Provides:
  - ``Distribution`` – Abstract base for all ProbPipe distributions.
  - ``NumericDistribution`` – The marker of a law whose event is numeric, with its views.
  - ``DistributionSpec`` – The term spec of the distribution kind.
  - Global defaults for expectation sampling.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from ..core._distribution_array import DistributionArray
    from ..core.constraints import Constraint
    from ..diagnostics.views import DiagnosticsView

from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import OutputSpec
from ..core.provenance import Provenance
from ..core.tracked import Annotated, TrackedTerm, _TrackedTermMeta

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_EVALUATIONS: int = 1024
"""Default number of function evaluations for sample-based expectations."""

RETURN_APPROX_DIST: bool = True
"""When True, approximate expectations return a BootstrapDistribution
capturing MC error instead of a plain array."""


def set_default_num_evaluations(n: int) -> None:
    """Set the global default for ``expectation()`` on infinite-support distributions."""
    global DEFAULT_NUM_EVALUATIONS
    if n < 1:
        raise ValueError("num_evaluations must be at least 1")
    DEFAULT_NUM_EVALUATIONS = n


def set_return_approx_dist(value: bool) -> None:
    """Set whether approximate expectations return error-tracking distributions."""
    global RETURN_APPROX_DIST
    RETURN_APPROX_DIST = bool(value)


# ---------------------------------------------------------------------------
# The event declaration: completion and class membership
# ---------------------------------------------------------------------------


def _complete_event_spec(event_spec: Any, name: str) -> OutputSpec:
    """Complete *event_spec* into the output declaration of one draw (II.2, III.7).

    Parameters
    ----------
    event_spec : OutputSpec or TermSpec
        The declaration a constructor supplies. A ``RecordSpec`` exposes its
        fields, even when it has one; any other term spec is a whole term whose
        component is *name*; an ``OutputSpec`` is kept as given.
    name : str
        The law's name, captured as the component of a whole-term event.

    Returns
    -------
    OutputSpec
        The complete declaration.

    Raises
    ------
    TypeError
        If *event_spec* is neither an ``OutputSpec`` nor a ``TermSpec``.
    ValueError
        If the declaration has a type hole, since filling one is the
        constructor's job, or *name* is not a valid component name.
    """
    if isinstance(event_spec, OutputSpec):
        declaration = event_spec
    elif isinstance(event_spec, RecordSpec):
        declaration = OutputSpec(event_spec)
    elif isinstance(event_spec, TermSpec):
        declaration = OutputSpec(**{name: event_spec})
    else:
        raise TypeError(
            f"event_spec must be an OutputSpec or a TermSpec, got {type(event_spec).__name__}"
        )
    if declaration.spec is None:
        raise ValueError(
            f"the event declaration of {name!r} has a type hole; a distribution stores "
            f"only a complete declaration"
        )
    return declaration


def _whole_term_component(declaration: OutputSpec) -> str | None:
    """The component of a whole-term declaration, or None for an exposed record."""
    return declaration._component_name


def _declares_numeric_event(value: Any) -> bool:
    """Whether *value* is a law whose declared event is numeric (II.3).

    A law still under construction declares nothing yet, so it is not numeric.
    """
    if not isinstance(value, Distribution):
        return False
    declared = getattr(value, "_spec", None)
    return declared is not None and isinstance(declared.event_spec.spec, NumericSpec)


def _array_leaves(declaration: OutputSpec) -> dict[str, NumericArraySpec]:
    """Each array leaf of *declaration*, keyed by its path through the components."""
    leaves: dict[str, NumericArraySpec] = {}

    def visit(path: str, spec: TermSpec | None) -> None:
        if isinstance(spec, NumericArraySpec):
            leaves[path] = spec
        elif isinstance(spec, RecordSpec):
            for child, child_spec in spec.children.items():
                visit(f"{path}/{child}", child_spec)

    for component, spec in declaration.components.items():
        visit(component, spec)
    return leaves


class _DistributionMeta(_TrackedTermMeta):
    """The metaclass of every distribution.

    Construction checks that the instance holds its event declaration, as the
    tracked-term metaclass checks its name: a class that bypasses
    ``Distribution.__init__`` calls ``_init_declaration`` itself.

    ``isinstance(d, NumericDistribution)`` holds if and only if ``d`` declares a
    numeric event, whatever its class, and every other class check is the ordinary
    one. A class whose every instance is numeric may claim the marker by
    inheriting it, and construction checks the claim.
    """

    def __instancecheck__(cls, instance: Any) -> bool:
        if cls is NumericDistribution:
            return _declares_numeric_event(instance)
        return super().__instancecheck__(instance)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        if not isinstance(getattr(instance, "_spec", None), DistributionSpec):
            raise TypeError(
                f"{cls.__name__}.__init__ left the event undeclared; pass event_spec to "
                f"Distribution.__init__, or call _init_declaration when bypassing it"
            )
        if issubclass(cls, NumericDistribution) and not _declares_numeric_event(instance):
            raise TypeError(
                f"{cls.__name__} inherits NumericDistribution, so its instances must "
                f"declare a numeric event"
            )
        return instance


# ---------------------------------------------------------------------------
# Distribution — the base class
# ---------------------------------------------------------------------------


class Distribution(TrackedTerm, Annotated, ABC, metaclass=_DistributionMeta):
    """
    Abstract base for all ProbPipe distributions.

    Every distribution is a tracked term: it is
    :class:`~probpipe.core.tracked.TrackedTerm` (a :attr:`~TrackedTerm.name` and a write-once
    :attr:`~TrackedTerm.provenance`) and
    :class:`~probpipe.core.tracked.Annotated` (free-form
    :attr:`~Annotated.annotations`).  A distribution's constructor takes
    its name as the required first argument, as ``Normal("x", 0.0, 1.0)``
    does. The classes the design retires, such as ``ProductDistribution``
    and ``DistributionArray``, still take it as a keyword and some derive one
    when it is omitted, an interim implementation detail. Every transform
    preserves the name; only ``with_name`` replaces it.

    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.core.protocols.SupportsSampling` protocol.

    **The event declaration.** A law stores one ``DistributionSpec``, its
    :attr:`spec`, whose :attr:`event_spec` is the output declaration of one draw
    (II.2). A bare ``RecordSpec`` exposes its fields; any other term spec is a
    whole-term event whose component is the law's ``name``, captured once, so
    ``with_name`` never moves it. :attr:`event_shape` reads the declaration, and
    a law whose declaration is numeric also has the views of
    :class:`NumericDistribution`; none of them is stored.

    Parameters
    ----------
    name : str
        Non-empty name for this distribution.
    event_spec : OutputSpec or TermSpec
        The declaration of one draw, completed as above.

    Raises
    ------
    TypeError
        If *name* is not a non-empty string, or *event_spec* is not a spec.
    ValueError
        If *event_spec* has a type hole, or a whole-term event's component, the
        name, is not a valid component name.
    """

    # -- Immutability: deferred for this layer ------------------------------

    def __delattr__(self, name: str) -> None:
        """Permit deletion, for the reason :meth:`__setattr__` gives.

        Goes with that method: removing the exemption means removing both.
        """
        object.__delattr__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Permit assignment, which :class:`TrackedTerm` otherwise refuses.

        Interim, and the only exemption from the rule that a tracked term is
        immutable. It stands because the contract for a *fitted* mapping is not
        settled: the documented way to build an emulator is to subclass a random
        function and train it in place, and until fitting has a contract that
        produces a new term instead, enforcing immutability here would break that
        pattern without offering a replacement.

        Deleting this method **and** :meth:`__delattr__` turns the guard on for
        the whole distribution layer. Both, or the layer keeps half an
        exemption: a trainer that clears what it fitted would still raise.
        """
        object.__setattr__(self, name, value)

    def __init__(
        self,
        name: str,
        event_spec: OutputSpec | TermSpec,
        *,
        _provenance: Provenance | None = None,
        _annotations: Mapping[str, Any] | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise TypeError(f"{type(self).__name__} requires a non-empty name= argument")
        # ``_provenance`` and ``_annotations`` carry state a reconstruction
        # already holds and that construction cannot otherwise reach: provenance
        # is write-once, and annotations are written after construction, so a
        # rebuilt distribution would come back without either. Private, and the
        # reconstruction paths are the only callers.
        self._init_tracked(name, provenance=_provenance)
        self._init_annotations(_annotations)
        self._init_declaration(event_spec)

    def _init_declaration(self, event_spec: OutputSpec | TermSpec) -> None:
        """Complete *event_spec* and store it as this law's declaration.

        The constructor calls this after setting the name; a class that bypasses
        the constructor calls it itself.
        """
        object.__setattr__(
            self, "_spec", DistributionSpec(_complete_event_spec(event_spec, self._name))
        )

    # -- the event declaration ----------------------------------------------

    @property
    def spec(self) -> DistributionSpec:
        """The law's term spec, the one stored source of its event declaration."""
        return self._spec

    @property
    def event_spec(self) -> OutputSpec:
        """The output declaration of one draw (II.2), a view on :attr:`spec`."""
        return self.spec.event_spec

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The shape of one draw, defined only when a draw is a single array.

        Raises
        ------
        TypeError
            If a draw is not a single array, a one-field record included.
        ValueError
            If the declared shape has unbound dimensions.
        """
        spec = self.event_spec.spec
        if not isinstance(spec, NumericArraySpec):
            raise TypeError(
                f"{type(self).__name__} {self.name!r} does not draw a single array; "
                f"event_shape is defined only for one"
            )
        free = spec.free_dims
        if free:
            raise ValueError(
                f"{type(self).__name__} {self.name!r} has unbound dimensions "
                f"{sorted(free)}; bind them with with_dim_sizes"
            )
        return spec.shape

    def __getattr__(self, name: str) -> Any:
        """Resolve a view of :class:`NumericDistribution` for a numeric law of another class.

        Python calls this only when ordinary lookup fails. A law whose declaration
        is numeric has the views of the marker whatever its class, so they resolve
        through the marker, and any other missing attribute raises as usual.

        Raises
        ------
        AttributeError
            If *name* is not an attribute of the law, a view of the marker on a law
            whose declaration is not numeric included.
        """
        view = _NUMERIC_VIEWS.get(name)
        if view is None:
            # Repeat ordinary lookup, so the error it met stands, a property's own
            # message included.
            return object.__getattribute__(self, name)
        if not _declares_numeric_event(self):
            raise AttributeError(
                f"{type(self).__name__} declares a non-numeric event, and {name} belongs to "
                f"NumericDistribution"
            )
        return view.__get__(self, type(self))

    # -- dimension transforms -------------------------------------------------

    def with_dim_sizes(self, **sizes: int) -> Self:
        """Bind named symbolic dimensions of the declaration (II.1).

        Parameters
        ----------
        **sizes : int
            Sizes for free dimensions of the declaration.

        Returns
        -------
        Self
            A copy of the same class and name whose declaration has the sizes
            substituted; the original is unchanged.

        Raises
        ------
        ValueError
            If a name is not a free dimension of the declaration, one an earlier
            call bound included, or a size is negative.
        TypeError
            If a size is not an integer.
        """
        unbound = set(sizes) - self.event_spec.spec.free_dims
        if unbound:
            raise ValueError(
                f"{type(self).__name__} {self.name!r} has no free dimensions "
                f"{sorted(unbound)} to bind"
            )
        return self._with_declaration(
            self.event_spec.with_dim_sizes(**sizes), "with_dim_sizes", sizes
        )

    def with_dim_names(self, **names: str) -> Self:
        """Rename symbolic dimensions of the declaration, simultaneously (II.1).

        Parameters
        ----------
        **names : str
            New names keyed by old; names that are not free are ignored.

        Returns
        -------
        Self
            A copy of the same class and name whose declaration has the
            dimensions renamed; the original is unchanged.
        """
        return self._with_declaration(
            self.event_spec.with_dim_names(**names), "with_dim_names", names
        )

    def _with_declaration(
        self, event_spec: OutputSpec, operation: str, arguments: Mapping[str, Any]
    ) -> Self:
        """A copy of this law holding *event_spec*, with provenance recording *operation*."""
        clone = self._shallow_copy()
        object.__setattr__(clone, "_spec", DistributionSpec(event_spec))
        object.__setattr__(clone, "_provenance", None)
        clone.with_provenance(
            Provenance.create(operation, parents=[self], metadata=dict(arguments))
        )
        return clone

    # -- components -----------------------------------------------------------

    def __getitem__(self, key: str | tuple[str, ...]) -> Distribution:
        """The law of the component or field at *key* (III.7).

        A whole-term law is itself under its component, so ``d[name]`` returns
        ``d``. For an exposed record, the result is today's field view, an interim
        implementation detail.

        Raises
        ------
        KeyError
            If a whole-term law's component is not *key*.
        """
        component = _whole_term_component(self.event_spec)
        if component is not None:
            if key == component:
                return self
            raise KeyError(key)
        from ..core._record_distribution import _RecordDistributionView

        return _RecordDistributionView(self, key)

    # -- keyword-form value construction ------------------------------------

    def _pack_value(self, **field_kwargs: Any) -> Any:
        """Build a single draw of this distribution from
        named field kwargs — the adapter behind the keyword form of the
        log_prob-family ops (``log_prob(dist, field=value, ...)``).

        Delegates field validation and ``Record`` construction to the general
        :func:`~probpipe.core.record._pack_fields` and layers this
        distribution's value-type convention on top:

        * **single field** → the bare field value, an array, so a scalar
          distribution's ``_log_prob`` still receives a raw array.
        * **multiple fields** → the :class:`~probpipe.core.record.Record`
          built from the named fields.

        Distributions whose ``_log_prob`` consumes a Record but splits it
        internally (e.g. ``SimpleModel`` → ``(params, data)``) keep this
        default and do the split in ``_log_prob``. Override only when the
        value type is neither a bare array nor a flat Record (e.g.
        ``StanModel``'s single ``parameters=`` flat array).

        Builds exactly one draw (``sample_shape == ()``). Batched
        evaluation does not go through kwargs — pass the batch positionally
        and let ``Function`` broadcasting handle it.

        Raises
        ------
        TypeError
            If the distribution has no named fields, or the kwargs do not
            match its fields exactly (missing or unexpected names).
        """
        from ..core.record import _pack_fields

        fields = getattr(self, "fields", None)
        if not fields:
            raise TypeError(
                f"{type(self).__name__} does not support the keyword form of "
                f"the log_prob-family ops (it has no named fields); pass a "
                f"positional value."
            )
        rec = _pack_fields(fields, field_kwargs, owner=type(self).__name__)
        return field_kwargs[fields[0]] if len(fields) == 1 else rec

    # -- approximation tracking ---------------------------------------------

    @property
    def is_approximate(self) -> bool:
        """Whether this distribution is an approximation.

        Approximate distributions are typically produced by sampling,
        variational inference, MCMC, bootstrap procedures, or other numerical
        approximations.
        """
        return getattr(self, "_approximate", False)

    # -- annotations ---------------------------------------------------------
    #
    # ``annotations`` (the general post-construction metadata store) is
    # provided by the :class:`~probpipe.core.tracked.Annotated` mixin.
    # On a fitted posterior the conventional layout is an ``xarray.DataTree``
    # with ``arviz/`` and ``diagnostics/`` subtrees; see :attr:`diagnostics`.

    @property
    def diagnostics(self) -> DiagnosticsView | None:
        """Structured view over diagnostic results stored in :attr:`annotations`.

        Returns ``None`` if no diagnostics have been computed yet.

        Inference backends and the diagnostics subsystem store posterior
        metadata in an ``xarray.DataTree`` attached to the distribution as
        its :attr:`~probpipe.core.tracked.Annotated.annotations`. The
        expected layout is::

            posterior._annotations
            ├── arviz/          # ArviZ-compatible data and raw inputs
            │   ├── posterior
            │   ├── sample_stats
            │   ├── observed_data
            │   ├── posterior_predictive
            │   └── log_likelihood
            └── diagnostics/    # ProbPipe-computed results and metadata
                ├── mcmc        # rhat, ess_bulk, ess_tail, mcse, ...
                └── runs        # on-demand diagnostics such as ppc, loo, spc
                    ├── ppc
                    ├── loo
                    └── spc

        The ``/arviz/`` subtree is intended to be passed to ArviZ functions
        and to hold raw diagnostic ingredients such as sampler statistics,
        posterior predictive samples, and pointwise log likelihoods. In ArviZ
        1.0+, this is an ArviZ-compatible ``DataTree`` rather than the older
        ``InferenceData`` representation.

        This property returns a structured Python accessor over the
        ``/diagnostics/`` subtree only. It is distinct from the ArviZ-compatible
        data used for plotting or ArviZ computations.

        In other words::

            posterior.diagnostics
                # structured ProbPipe view over posterior.annotations["diagnostics"]

            posterior.arviz_data
                # ArviZ-compatible xarray DataTree subtree, typically
                # posterior.annotations["arviz"]

            posterior.inference_data
                # backward-compatible alias for posterior.arviz_data

        Examples
        --------
        ::

            posterior = condition_on(model, data)

            # MCMC diagnostics mutate posterior._annotations in place and return None.
            add_mcmc_diagnostics(posterior)

            posterior.diagnostics.rhat
            # {"intercept": 1.001, "slope": 1.002}

            posterior.diagnostics.warnings
            # []

            posterior.diagnostics.runs
            # []

            # Posterior predictive checks are stored under diagnostics/runs/ppc.
            add_ppc(
                posterior,
                test_fns=[...],
                observed_data=y,
                generative_likelihood=lik,
            )

            posterior.diagnostics.ppc.result
            # {"var_mean_ratio": {"p_value": 0.43, "observed": 3.2}}

            posterior.diagnostics.runs[0].result
            # {"p_value": {"var_mean_ratio": 0.43}, ...}

            posterior.diagnostics.runs[0].plot_fn
            # "" unless the run wrote ArviZ-compatible plotting inputs

        Notes
        -----
        The diagnostics accessor is read-only. Diagnostic functions such as
        ``add_mcmc_diagnostics`` and ``add_ppc`` are responsible for writing
        diagnostic results into ``posterior._annotations``.
        """
        aux = self.annotations
        if aux is None:
            return None
        children = aux.children if hasattr(aux, "children") else {}
        if "diagnostics" not in children:
            return None
        from ..diagnostics.views import DiagnosticsView

        return DiagnosticsView(aux["diagnostics"])

    # -- batched-construction alias ----------------------------------------

    @classmethod
    def from_batched_params(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...] | None = None,
        **batched_params,
    ) -> DistributionArray:
        """Class-method alias for :meth:`DistributionArray.from_batched_params`.

        Lets users write the ergonomic per-class form::

            Normal.from_batched_params(loc=jnp.zeros(5), scale=1.0, name="x")

        instead of the universal entry point::

            DistributionArray.from_batched_params(
                Normal, loc=jnp.zeros(5), scale=1.0, name="x",
            )

        Both produce the same ``DistributionArray`` — the alias is a
        thin classmethod that calls the universal factory with
        ``cls`` bound. Subclasses inherit the alias automatically;
        no per-family override is needed.

        See :meth:`DistributionArray.from_batched_params` for the full
        contract (dispatch on
        :class:`~probpipe.core.protocols.SupportsArrayBackend`,
        ``batch_shape`` inference, per-cell name suffixing).
        """
        # Local import: ``DistributionArray`` inherits from ``Distribution``,
        # so importing it at module top would create a cycle.
        from ..core._distribution_array import DistributionArray

        return DistributionArray.from_batched_params(
            cls,
            name=name,
            batch_shape=batch_shape,
            **batched_params,
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        return f"{parts[0]}({', '.join(parts[1:])})"


class NumericDistribution(Distribution):
    """The marker of a law whose event declaration is numeric (II.3), with its views.

    ``isinstance(d, NumericDistribution)`` holds if and only if
    ``d.event_spec.spec`` is a :class:`~probpipe.core._spec_base.NumericSpec`,
    so a draw implements ``Numeric`` and the flat-vector interface applies. A
    class whose every instance is numeric may inherit the marker, and
    construction checks that each instance declares a numeric event.

    Every numeric law has the views below, whatever its class: a class that
    inherits the marker has them directly, and any other numeric law resolves
    them through the marker. A law whose declaration is not numeric has none of
    them. Like ``event_shape``, they read the declaration and are never stored.
    """

    @property
    def dtypes(self) -> dict[str, Any]:
        """The declared dtype of each array leaf of a draw, keyed by leaf path."""
        return {path: spec.dtype for path, spec in _array_leaves(self.event_spec).items()}

    @property
    def supports(self) -> dict[str, Constraint | None]:
        """The declared support of each array leaf of a draw, keyed by leaf path."""
        return {path: spec.support for path, spec in _array_leaves(self.event_spec).items()}

    @property
    def dtype(self) -> Any:
        """The dtype every array leaf shares, or None when they differ or there are none."""
        dtypes = set(self.dtypes.values())
        return dtypes.pop() if len(dtypes) == 1 else None

    @property
    def support(self) -> Constraint | None:
        """The support every array leaf shares, or None when they differ or there are none."""
        supports = set(self.supports.values())
        return supports.pop() if len(supports) == 1 else None


# The views a numeric law has whatever its class, which ``Distribution.__getattr__``
# resolves for a class that does not inherit the marker.
_NUMERIC_VIEWS: dict[str, property] = {
    name: vars(NumericDistribution)[name] for name in ("dtypes", "supports", "dtype", "support")
}


# ---------------------------------------------------------------------------
# DistributionSpec — the term spec of the distribution kind
# ---------------------------------------------------------------------------


def _unify_declarations(
    expected: OutputSpec, actual: OutputSpec, bindings: dict[str, int], path: str
) -> None:
    """Match *actual* against *expected*: packaging and components, then their specs.

    Raises
    ------
    ValueError
        If the packaging or the whole-term component differs, or the specs do not
        unify in the shared scope *bindings*.
    """
    wanted, found = _whole_term_component(expected), _whole_term_component(actual)
    if (wanted is None) != (found is None):
        raise ValueError(
            f"{path} declares {'an exposed record' if wanted is None else f'the whole term {wanted!r}'}, "
            f"but the law declares {'an exposed record' if found is None else f'the whole term {found!r}'}"
        )
    if wanted != found:
        raise ValueError(
            f"{path} declares the component {wanted!r}, but the law declares {found!r}"
        )
    # A whole term's spec is bound under its component's path, as a record field's is.
    _unify_specs(
        expected.spec, actual.spec, bindings, path if wanted is None else f"{path}/{wanted}"
    )


@dataclass(frozen=True, init=False)
class DistributionSpec(TermSpec):
    """The distribution kind's term spec: the output declaration of one draw.

    Parameters
    ----------
    event_spec : OutputSpec or RecordSpec
        The declaration of one draw (II.2). A bare ``RecordSpec`` completes to the
        exposed form.

    Raises
    ------
    TypeError
        If *event_spec* is neither, since a bare spec of another kind has no
        component name to complete it with.
    ValueError
        If the declaration has a type hole.

    Notes
    -----
    ``is_valid`` accepts a ``Distribution`` whose own declaration unifies with
    this one: the packaging and the component names agree, and then the
    components unify in one scope. An unset dtype accepts any dtype and a set one
    requires a same-kind cast, sizes agree with symbolic dimensions bound
    consistently, and support is not compared.

    ``bind_dims_from_value`` binds a symbolic declaration from a law's own, and
    ``bind_dims_from_spec`` from another ``DistributionSpec``, by that same rule.
    Repeated symbols share one scope, and conflicting sizes raise ValueError.

    Examples
    --------
    >>> from probpipe import DistributionSpec, OutputSpec, NumericArraySpec
    >>> declared = DistributionSpec(OutputSpec(x=NumericArraySpec(("n",))))
    >>> bound = declared.bind_dims_from_spec(DistributionSpec(OutputSpec(x=NumericArraySpec((3,)))))
    >>> bound.event_spec.spec.shape
    (3,)
    """

    event_spec: OutputSpec

    def __init__(self, event_spec: OutputSpec | RecordSpec) -> None:
        if isinstance(event_spec, RecordSpec):
            event_spec = OutputSpec(event_spec)
        elif not isinstance(event_spec, OutputSpec):
            raise TypeError(
                f"DistributionSpec.event_spec must be an OutputSpec or a RecordSpec, got "
                f"{type(event_spec).__name__}, which has no component name to complete it with"
            )
        if event_spec.spec is None:
            raise ValueError("DistributionSpec.event_spec has a type hole")
        object.__setattr__(self, "event_spec", event_spec)

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of the draw this declares."""
        return self.event_spec.spec.free_dims

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> DistributionSpec:
        """This spec around a substituted event declaration."""
        declaration = self.event_spec
        return DistributionSpec(declaration._with_spec(declaration.spec._substitute_dims(bindings)))

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Bind the declared event against the declaration *value* carries."""
        if not isinstance(value, Distribution):
            raise ValueError(f"{path} does not conform to its field spec ({self!r})")
        _unify_declarations(self.event_spec, value.event_spec, bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind the declared event against *actual*'s own."""
        if not isinstance(actual, DistributionSpec):
            return False
        _unify_declarations(self.event_spec, actual.event_spec, bindings, path)
        return True

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a ``Distribution`` whose declaration matches this one."""
        if not isinstance(value, Distribution):
            return False
        try:
            _unify_declarations(self.event_spec, value.event_spec, {}, "the declaration")
        except ValueError:
            return False
        return True
