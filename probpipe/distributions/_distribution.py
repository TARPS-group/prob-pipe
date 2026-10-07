"""The distribution base class, its term spec, and minimal helpers.

Provides:
  - ``Distribution`` – Abstract base for all ProbPipe distributions.
  - ``NumericDistribution`` – The marker of a law whose event is numeric, with its views.
  - ``DistributionSpec`` – The term spec of the distribution kind.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from ..core.constraints import Constraint
    from ..diagnostics.views import DiagnosticsView
    from ._batches import DistributionBatch
    from ._conditional import ConditionalDistribution
    from ._factored import FactoredConditionalDistribution, FactoredDistribution
    from ._views import _EventRenames

from ..core._record_spec import RecordSpec
from ..core._repr import public_class_name, term_repr
from ..core._spec_base import NumericArraySpec, NumericSpec, TermSpec, _unify_specs
from ..core._specs import OutputSpec
from ..core.constraints import _known_equal
from ..core.provenance import Provenance
from ..core.tracked import Annotated, TrackedTerm, _TrackedTermMeta
from ._capabilities import _check_guards

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
        The law's label, which is the default component of a whole-term event.

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


def _is_default_declaration(declaration: OutputSpec, name: str) -> bool:
    """Whether *declaration* is the one a bare spec completes to under the label *name* (III.7)."""
    try:
        return declaration == OutputSpec.default(declaration.spec, component=name)
    except ValueError:
        # Only a label that is a valid component has a default whole-term declaration.
        return False


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


#: The law that ``with_path_names`` returns, installed by the views module at import.
_renamed_law_factory: Callable[[Any, OutputSpec, Mapping[str, str]], Any] | None = None


def _install_renamed_law(factory: Callable[[Any, OutputSpec, Mapping[str, str]], Any]) -> None:
    """Install the factory of the law whose draws carry the names ``with_path_names`` gives.

    Called once, by the views module at import, so this module never imports the
    module that imports it.
    """
    global _renamed_law_factory
    _renamed_law_factory = factory


#: The field view that indexing returns, installed by the views module at import.
_field_view_factory: Callable[[Any, Any], Any] | None = None


def _install_field_view(factory: Callable[[Any, Any], Any]) -> None:
    """Install the factory of the field view that ``d[path]`` returns.

    Called once, by the views module at import, so this module never imports the
    module that imports it.
    """
    global _field_view_factory
    _field_view_factory = factory


#: Indexing a ``DistributionBatch`` returns a copy of the stored law under a derived
#: label and sets this attribute of the copy to the stored law. A lift reads it to
#: draw two accesses of one element, or an element and its stored law, together
#: (V.5). Detaching the copy deletes the attribute.
_ELEMENT_SOURCE = "_element_source"

#: ``with_path_names`` sets this attribute of each law it returns that does not hold
#: the law it renames as its parent. The value is the pair of that law and the
#: renames from its declaration to the result's. A lift reads it to draw the result
#: together with the law it renames (V.5). Detaching the result deletes the
#: attribute.
_RENAME_SOURCE = "_rename_source"


def _detached_term(term: Any) -> Any:
    """*term*, a law or a kernel, detached from the workflow under its own label.

    The copy shares the representation, and it carries no provenance, no
    annotations, and no reference to a container or a parent, such as a batch
    it was an element of or a law it renames.
    """
    clone = term._shallow_copy()
    object.__setattr__(clone, "_provenance", None)
    for workflow_state in ("_annotations", _ELEMENT_SOURCE, _RENAME_SOURCE):
        clone.__dict__.pop(workflow_state, None)
    return clone


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
    tracked-term metaclass checks its label: a class that bypasses
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
    :class:`~probpipe.core.tracked.TrackedTerm` (a :attr:`~TrackedTerm.label` and a write-once
    :attr:`~TrackedTerm.provenance`) and
    :class:`~probpipe.core.tracked.Annotated` (free-form
    :attr:`~Annotated.annotations`).  A distribution's constructor takes
    its label as the required first argument, as ``Normal("x", 0.0, 1.0)``
    does; a joint that ``*`` composes is labeled by its operands' labels. Every
    transform preserves the label; only ``with_label`` replaces it.

    Sampling and expectation capabilities are provided by the
    :class:`~probpipe.SupportsSampling` protocol.

    **The event declaration.** A law stores one ``DistributionSpec``, its
    :attr:`spec`, whose :attr:`event_spec` is the output declaration of one
    draw. A bare ``RecordSpec`` exposes its fields; any other term spec is a
    whole-term event whose component defaults to the law's label, captured
    once at construction. :attr:`event_shape` reads the declaration, and
    a law whose declaration is numeric also has the views of
    :class:`NumericDistribution`; none of them is stored.

    Parameters
    ----------
    label : str
        The law's label, which must be a non-empty string.
    event_spec : OutputSpec or TermSpec
        The declaration of one draw, completed as above.
    _provenance : Provenance, optional
        The provenance of the law that a reconstruction rebuilds. By default the
        provenance stays unset until ``with_provenance`` attaches one.
    _annotations : Mapping[str, Any], optional
        The annotations of the law that a reconstruction rebuilds, copied into the
        law's own store. By default :attr:`annotations` is ``None``.

    Raises
    ------
    TypeError
        If *label* is not a non-empty string, or *event_spec* is not a spec.
    ValueError
        If *event_spec* has a type hole, or it is a bare term spec other than a
        record and *label* is not a valid component name.
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
        label: str,
        event_spec: OutputSpec | TermSpec,
        *,
        _provenance: Provenance | None = None,
        _annotations: Mapping[str, Any] | None = None,
    ):
        if not isinstance(label, str) or not label:
            raise TypeError(
                f"{type(self).__name__} requires a non-empty label as its first argument"
            )
        # ``_provenance`` and ``_annotations`` carry state a reconstruction
        # already holds and that construction cannot otherwise reach: provenance
        # is write-once, and annotations are written after construction, so a
        # rebuilt distribution would come back without either. Private, and the
        # reconstruction paths are the only callers.
        self._init_tracked(label, provenance=_provenance)
        self._init_annotations(_annotations)
        self._init_declaration(event_spec)

    def _init_declaration(self, event_spec: OutputSpec | TermSpec) -> None:
        """Complete *event_spec* and store it as this law's declaration.

        The constructor calls this after setting the label; a class that bypasses
        the constructor calls it itself.
        """
        object.__setattr__(
            self, "_spec", DistributionSpec(_complete_event_spec(event_spec, self._label))
        )

    # -- the representation ---------------------------------------------------

    def raw(self) -> Distribution:
        """This law detached from the workflow, under its label and declaration.

        A law is represented by itself, so its raw form is a copy that shares
        its representation and carries no provenance, no annotations, and no
        reference to a container or a parent, such as a batch it was an element
        of or a law it renames. A field view returns its detached marginal
        instead, and a backend adapter its wrapped backend distribution.
        """
        return _detached_term(self)

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
                f"{type(self).__name__} {self.label!r} does not draw a single array; "
                f"event_shape is defined only for one"
            )
        free = spec.free_dims
        if free:
            raise ValueError(
                f"{type(self).__name__} {self.label!r} has unbound dimensions "
                f"{sorted(free)}; bind them with with_dim_sizes"
            )
        return spec.shape

    def __getattr__(self, name: str) -> Any:
        """Resolve a view of :class:`NumericDistribution` for a numeric law of another class.

        Python calls this only when ordinary lookup fails. A law whose declaration
        is numeric has the views of the marker whatever its class, so they resolve
        through the marker, and any other missing attribute raises as usual.

        Parameters
        ----------
        name : str
            The attribute that ordinary lookup did not find, such as ``"dtype"``.

        Returns
        -------
        Any
            The value of the marker's view *name* for this law.

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
            A copy of the same class and label whose declaration has the sizes
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
                f"{type(self).__name__} {self.label!r} has no free dimensions "
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
            A copy of the same class and label whose declaration has the
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
        A factored law renames through its factors where they can carry the
        rename, and the result is the factored joint of the renamed factors over
        the same graph. A rename that gathers components under a new node
        regroups the factors that produce them into a packaged sub-joint, one
        factor of the result whose event is the node. A family whose parameters
        carry the event's paths rebuilds itself under the new paths: an
        empirical law over records returns the empirical law of its atoms with
        their fields at the new paths, under the same weights. Any other rename
        that changes the path of a field of a record draw, including a gathering
        whose groups condition on one another in a cycle, returns a law that
        holds this one and renames values at its boundary: draws, moments, and
        marginals on the way out, and scored values, givens, and paths on the
        way in. A lift draws the result together with this law, as it draws a
        view with its parent (V.5).

        Parameters
        ----------
        mapping : Mapping[str, str], optional
            The new exact path of each node, keyed by the node's exact path.
        **kwargs : str
            Further renames, keyed by paths that are identifiers.

        Returns
        -------
        Distribution
            The renamed law under the same label; the original is unchanged.

        Raises
        ------
        KeyError
            If a key is not a path of the declaration.
        ValueError
            As :meth:`OutputSpec.with_path_names` raises it.
        """
        renamed = self.event_spec.with_path_names(mapping, **kwargs)
        renames = {**dict(mapping or {}), **kwargs}
        if _renamed_law_factory is None:
            raise RuntimeError("the renamed law is not installed; import probpipe")
        return _renamed_law_factory(self, renamed, renames)

    def _renamed_in_family(self, event: _EventRenames) -> Distribution | None:
        """The member of this law's family that holds its values under *event*'s new paths, or None.

        ``with_path_names`` calls this for a rename that changes the path of a
        field of a record draw, and returns a law that renames this one's values
        at its boundary when it gives None. A family whose parameters carry the
        event's paths overrides it to return the member that declares
        ``event.renamed`` and whose draw at a key is this law's draw at that key
        under the new paths, as an empirical law over records does. The base
        class returns None.
        """
        return None

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
        """The law itself at a whole term's component, or the field view at another event path.

        A whole-term law is itself under its component, so ``d[name]`` returns
        ``d``; the component is fixed at construction, so after ``with_label`` the
        law is still addressed by it. Any other event path, a field of an exposed
        record or a path below a whole record's component, gives the
        ``FieldView`` of the node there, which holds a reference to this law, and
        a tuple of paths gives the view of their selection.

        Parameters
        ----------
        key : str or tuple of str
            An event path, which starts with a component, or a tuple of event
            paths to view jointly.

        Returns
        -------
        Distribution
            This law itself, or a ``FieldView`` of it.

        Raises
        ------
        KeyError
            If *key* is not an event path of this law, or names one that is not.
        TypeError
            If *key* is neither a string nor a tuple of strings.
        ValueError
            If *key* is an empty tuple, or two selected paths share their final
            segment.
        """
        if isinstance(key, str) and key == _whole_term_component(self.event_spec):
            return self
        if _field_view_factory is None:
            raise RuntimeError("the field view is not installed; import probpipe")
        return _field_view_factory(self, key)

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

        Parameters
        ----------
        other : Distribution or ConditionalDistribution
            The right operand, whose components this law's factors may condition
            on.

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
        internally (e.g. a factored joint, which scores each factor's fields)
        keep this default and do the split in ``_log_prob``. Override only when the
        value type is neither a bare array nor a flat Record (e.g.
        ``StanModel``'s single ``parameters=`` flat array).

        Builds exactly one draw (``sample_shape == ()``). Batched
        evaluation does not go through kwargs — pass the batch positionally
        and let ``Function`` broadcasting handle it.

        Parameters
        ----------
        **field_kwargs : Any
            One value per named field of a draw, keyed by field name.

        Returns
        -------
        Any
            The draw: the bare value of a single field, or the ``Record`` of
            several.

        Raises
        ------
        TypeError
            If the distribution has no named fields, or the kwargs do not
            match its fields exactly (missing or unexpected names).
        """
        from ..core.record import _pack_fields

        fields = getattr(self, "fields", None)
        if fields is None and _declares_numeric_event(self):
            # A numeric law outside the record laws names its fields by its components.
            fields = tuple(self.event_spec.components)
        if not fields:
            raise TypeError(
                f"{type(self).__name__} does not support the keyword form of "
                f"the log_prob-family ops (it has no named fields); pass a "
                f"positional value."
            )
        rec = _pack_fields(fields, field_kwargs, owner=type(self).__name__)
        return field_kwargs[fields[0]] if len(fields) == 1 else rec

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

            posterior.annotations["arviz"]
                # the ArviZ-compatible xarray DataTree

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
                kernel=likelihood,
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

    # -- batched construction -----------------------------------------------

    @classmethod
    def from_batched_params(
        cls,
        *,
        label: str,
        batch_shape: tuple[int, ...] | None = None,
        **batched_params: Any,
    ) -> DistributionBatch:
        """The separate laws of this class at batched parameters, one per batch position.

        Each parameter's leading axes are the batch axes, and the law at a
        position is this class at that position's parameters. The form of the
        result is not yet decided, so the method raises.

        Parameters
        ----------
        label : str
            The batch's label.
        batch_shape : tuple of int, optional
            The batch axes, inferred from the parameters when omitted.
        **batched_params
            This class's constructor arguments, with the batch axes leading.

        Returns
        -------
        DistributionBatch
            The batch of these laws under the label *label*, with the batch axes
            as its batch shape.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("Distribution.from_batched_params")

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        """The public class, the label, the family parameters, and a declaration that is not the default.

        The event declaration is shown when it differs from the one a bare spec
        completes to under the law's label (III.7), as after ``with_label`` or
        for a declared component.
        """
        fields = [*self._repr_arguments(), *self._event_repr_arguments()]
        return term_repr(self._repr_class_name(), self.label, fields)

    def _repr_class_name(self) -> str:
        """The first public class in this law's method-resolution order, which the repr names."""
        return public_class_name(type(self))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The family parameters the repr shows, each by name and formatted value; none here."""
        return []

    def _event_repr_arguments(self) -> list[tuple[str, str]]:
        """The event declaration, unless it is the default for this law's label."""
        if _is_default_declaration(self.event_spec, self.label):
            return []
        return [("event_spec", repr(self.event_spec))]


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

    Parameters
    ----------
    expected : OutputSpec
        The declaration to match against, such as a term spec's ``event_spec``.
    actual : OutputSpec
        The declaration a law or a kernel carries.
    bindings : dict[str, int]
        The size of each symbolic dimension bound so far, keyed by its name, which
        unification extends in place.
    path : str
        The location of the declaration that error messages name, such as
        ``"the declaration"``.

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

    def __repr__(self) -> str:
        """The event declaration, as the constructor takes it."""
        return term_repr("DistributionSpec", None, [("event_spec", repr(self.event_spec))])

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a ``Distribution`` whose declaration matches this one."""
        if not isinstance(value, Distribution):
            return False
        try:
            _unify_declarations(self.event_spec, value.event_spec, {}, "the declaration")
        except ValueError:
            return False
        return True
