"""The distribution base class, its term spec, and minimal helpers.

Provides:
  - ``Distribution`` – Abstract base for all ProbPipe distributions.
  - ``NumericDistribution`` – The marker of a law whose event is numeric, with its views.
  - ``DistributionSpec`` – The term spec of the distribution kind.
  - Global defaults for expectation sampling.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from ..core._distribution_array import DistributionArray
    from ..core.constraints import Constraint
    from ..diagnostics.views import DiagnosticsView
    from ._conditional import ConditionalDistribution
    from ._factored import FactoredConditionalDistribution, FactoredDistribution

from ..core._record_spec import RecordSpec
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import OutputSpec
from ..core.constraints import _known_equal
from ..core.provenance import Provenance
from ..core.tracked import Annotated, TrackedTerm, _TrackedTermMeta
from ._capabilities import _check_guards

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_EVALUATIONS: int = 1024
"""Default number of function evaluations for sample-based expectations."""


def set_default_num_evaluations(n: int) -> None:
    """Set the global default for ``expectation()`` on infinite-support distributions."""
    global DEFAULT_NUM_EVALUATIONS
    if n < 1:
        raise ValueError("num_evaluations must be at least 1")
    DEFAULT_NUM_EVALUATIONS = n


# ---------------------------------------------------------------------------
# The event declaration: completion and class membership
# ---------------------------------------------------------------------------


def _complete_event_spec(event_spec: Any, name: str) -> OutputSpec:
    """Complete *event_spec* into the output declaration of one draw.

    Parameters
    ----------
    event_spec : OutputSpec or TermSpec
        The declaration a constructor supplies. A ``RecordSpec`` exposes its
        fields, even when it has one; any other term spec is a whole term whose
        component defaults to *name*; an ``OutputSpec`` is kept as given.
    name : str
        The law's name, the default component of a whole-term event.

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
    elif isinstance(event_spec, TermSpec):
        declaration = OutputSpec.default(event_spec, component=name)
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
    if declaration.exposes_record:
        return None
    (component,) = declaration.components
    return component


def _declares_numeric_event(value: Any) -> bool:
    """Whether *value* is a law whose declared event is numeric.

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


#: Each marker whose membership is read from an instance's declaration, with the
#: predicate that decides it and what an instance of a class inheriting the marker
#: must declare. ``NumericDistribution`` registers below, and the markers of the
#: conditional and factored kinds register in their own modules.
_DECLARATION_MARKERS: dict[type, tuple[Callable[[Any], bool], str]] = {}


def _check_marker_claims(instance: Any) -> None:
    """Raise ``TypeError`` if *instance*'s class inherits a marker its declaration fails."""
    claimant = type(instance)
    for marker, (holds, requirement) in _DECLARATION_MARKERS.items():
        if issubclass(claimant, marker) and not holds(instance):
            raise TypeError(
                f"{claimant.__name__} inherits {marker.__name__}, so its instances must "
                f"declare {requirement}"
            )


#: The engine behind ``*``, installed by the composition module at import.
_composition_engine: Callable[[Any, Any], Any] | None = None


def _install_composition(engine: Callable[[Any, Any], Any]) -> None:
    """Install the engine that ``*`` delegates to on both distribution kinds.

    Called once, by the composition module at import, so this module never
    imports the module that imports it.
    """
    global _composition_engine
    _composition_engine = engine


#: The law that ``with_path_names`` returns for a rename that reaches a field of a
#: record draw, installed by the views module at import.
_renamed_law_factory: Callable[[Any, OutputSpec, Mapping[str, str]], Any] | None = None


def _install_renamed_law(factory: Callable[[Any, OutputSpec, Mapping[str, str]], Any]) -> None:
    """Install the factory of the law whose draws carry the names ``with_path_names`` gives.

    Called once, by the views module at import, so this module never imports the
    module that imports it.
    """
    global _renamed_law_factory
    _renamed_law_factory = factory


def _compose_operands(left: Any, right: Any) -> Any:
    """*left* ``*`` *right* through the installed engine.

    The engine returns ``NotImplemented`` for an operand that is neither
    distribution kind, so Python tries the reflected operation and a scalar
    operand can scale.
    """
    if _composition_engine is None:
        raise RuntimeError("the composition engine is not installed; import probpipe")
    return _composition_engine(left, right)


class _DistributionMeta(_TrackedTermMeta):
    """The metaclass of every distribution.

    Construction checks that the instance holds its event declaration, as the
    tracked-term metaclass checks its name: a class that bypasses
    ``Distribution.__init__`` calls ``_init_declaration`` itself.

    Membership in a marker registered in ``_DECLARATION_MARKERS`` is read from an
    instance's declaration whatever its class, so ``isinstance(d,
    NumericDistribution)`` holds if and only if ``d`` declares a numeric event,
    and every other class check is the ordinary one. A class may claim a marker by
    inheriting it, and construction checks the claim. Creating a class checks
    each capability guard it defines (:func:`._capabilities._check_guards`).
    """

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        # The check runs once the class is complete, since a class that fails
        # while type.__new__ builds it has no ABC caches of its own and would
        # write into its base's.
        super().__init__(*args, **kwargs)
        _check_guards(cls)

    def __instancecheck__(cls, instance: Any) -> bool:
        marker = _DECLARATION_MARKERS.get(cls)
        if marker is not None:
            return marker[0](instance)
        return super().__instancecheck__(instance)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        # A factory __new__ may construct a subclass, whose declaration and claim
        # are the ones to check.
        claimant = type(instance)
        if not isinstance(getattr(instance, "_spec", None), DistributionSpec):
            raise TypeError(
                f"{claimant.__name__}.__init__ left the event undeclared; pass event_spec to "
                f"Distribution.__init__, or call _init_declaration when bypassing it"
            )
        _check_marker_claims(instance)
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
    does. A few classes, such as ``ProductDistribution`` and
    ``DistributionArray``, take it as a keyword instead and derive one when it
    is omitted. Every transform preserves the name; only ``with_name``
    replaces it.

    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.SupportsSampling` protocol.

    **The event declaration.** A law stores one ``DistributionSpec``, its
    :attr:`spec`, whose :attr:`event_spec` is the output declaration of one
    draw. A bare ``RecordSpec`` exposes its fields; any other term spec is a
    whole-term event whose component defaults to the law's ``name``, captured
    once at construction. :attr:`event_shape` reads the declaration, and
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
        If *event_spec* has a type hole, or it is a bare term spec other than a
        record and *name* is not a valid component name.
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
            raise TypeError(
                f"{type(self).__name__} requires a non-empty name as its first argument"
            )
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
        """The output declaration of one draw, read from :attr:`spec`."""
        return self.spec.event_spec

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The shape of one draw, defined only when a draw is a single array.

        For any other law the attribute is absent, so ``hasattr(law,
        "event_shape")`` is ``False``.

        Raises
        ------
        AttributeError
            If a draw is not a single array, a one-field record included.
        ValueError
            If the declared shape has unbound dimensions.
        """
        spec = self.event_spec.spec
        if not isinstance(spec, NumericArraySpec):
            raise AttributeError(
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
        """Bind named symbolic dimensions of the declaration.

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
        """Rename symbolic dimensions of the declaration, simultaneously.

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

    def with_path_names(
        self, mapping: Mapping[str, str] | None = None, /, **kwargs: str
    ) -> Distribution:
        """Rename or move nodes of the event declaration by their paths, ``old -> new``.

        The result is this law with :meth:`OutputSpec.with_path_names` applied to
        its declaration. A path starts with a component, and the packaging is
        kept, so a whole term's component is renamed in place with the term's
        fields under it. The law is unchanged: a draw of the result is a draw of
        this law carrying the new names. Renaming a whole term's component alone
        changes only the declaration, so the result is a copy of the same class.
        A rename that reaches a field of a record draw returns a law that holds
        this one and renames values at its boundary: draws, moments, and
        marginals on the way out, and scored values, givens, and paths on the way
        in.

        Parameters
        ----------
        mapping : Mapping[str, str], optional
            The new exact path of each node, keyed by the node's exact path.
        **kwargs : str
            Further renames, keyed by paths that are identifiers.

        Returns
        -------
        Distribution
            The renamed law under the same name; the original is unchanged.

        Raises
        ------
        KeyError
            If a key is not a path of the declaration.
        ValueError
            As :meth:`OutputSpec.with_path_names` raises it.
        NotImplementedError
            If this law is factored and a rename reaches a field of its record
            draw, since a joint renames through its factors.
        """
        renamed = self.event_spec.with_path_names(mapping, **kwargs)
        renames = {**dict(mapping or {}), **kwargs}
        if renamed.spec == self.event_spec.spec:
            return self._with_declaration(renamed, "with_path_names", renames)
        if _renamed_law_factory is None:
            raise RuntimeError("the renamed law is not installed; import probpipe")
        return _renamed_law_factory(self, renamed, renames)

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

    # Indexing addresses components, so the legacy sequence protocol must not make
    # a law iterable through it.
    __iter__ = None

    def __getitem__(self, key: str | tuple[str, ...]) -> Distribution:
        """The law of the component or field at *key*.

        A whole-term law is itself under its component, given as a string or a
        one-element tuple, so ``d[name]`` returns ``d``. The component is fixed at
        construction, so after ``with_name`` the law is still addressed by it. For
        an exposed record, the result is today's field view, an interim
        implementation detail.

        Raises
        ------
        KeyError
            If a whole-term law's component is not *key*.
        """
        component = _whole_term_component(self.event_spec)
        if component is not None:
            if key == component or key == (component,):
                return self
            raise KeyError(key)
        from ..core._record_distribution import _RecordDistributionView

        return _RecordDistributionView(self, key)

    # -- composition ------------------------------------------------------------

    def __mul__(
        self, other: Distribution | ConditionalDistribution
    ) -> FactoredDistribution | FactoredConditionalDistribution:
        """The joint of this law and *other*, composed conditional-first.

        The left operand may condition on what the right produces, so ``lik *
        prior`` reads as ``p(y | β) · p(β)``. The result is a
        ``FactoredDistribution`` when no given is left unmet and a
        ``FactoredConditionalDistribution`` otherwise, flattened over the
        operands' factors and labeled by their labels joined with ``·``.

        Returns
        -------
        FactoredDistribution or FactoredConditionalDistribution
            The joint, or ``NotImplemented`` when *other* is neither
            distribution kind, so that a scalar operand can scale instead.

        Raises
        ------
        ValueError
            If a component is produced twice, the right operand consumes a
            component the left produces, or matched specs do not unify.
        """
        return _compose_operands(self, other)

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
    """The marker of a law whose event declaration is numeric, with its views.

    ``isinstance(d, NumericDistribution)`` holds if and only if
    ``d.event_spec.spec`` is a :class:`~probpipe.core._spec_base.NumericSpec`,
    so a draw implements ``Numeric`` and the flat-vector interface applies. A
    class whose every instance is numeric may inherit the marker, and
    construction checks that each instance declares a numeric event.

    Every numeric law has the views below, whatever its class: a class that
    inherits the marker has them directly, and any other numeric law resolves
    them through the marker. A law whose declaration is not numeric has none of
    them. Like ``event_shape``, they read the declaration and are never stored.

    The marker is not a dispatch type. Registering a method for it raises
    ``TypeError``, since selection by class would miss a numeric law whose class
    does not inherit it.
    """

    # Read by dispatch registration, which refuses a class whose membership follows
    # an instance's declaration.
    _membership_follows_declaration = True

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
        """The support every array leaf shares, or None when they differ or there are none.

        ``supports`` tells leaves that differ from leaves that are unset. Supports
        whose comparison needs a traced value, as under ``jit``, count as different.
        """
        supports = list(self.supports.values())
        if supports and all(_known_equal(supports[0], s) for s in supports[1:]):
            return supports[0]
        return None


_DECLARATION_MARKERS[NumericDistribution] = (_declares_numeric_event, "a numeric event")


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
        The declaration of one draw. A bare ``RecordSpec`` completes to the
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
