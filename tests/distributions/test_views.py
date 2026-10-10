"""A field view is the law of one event path of its parent.

``FieldView(parent, path)`` declares the node at ``path`` whole, under a
component named by the path's final segment, is labeled by the path, and holds a
reference to its parent. A tuple of paths selects several nodes as an exposed
record of them. Indexing a view joins the view's path to the key after the view's
component. A view claims each capability its parent's can derive: a projection
when the parent has it, covariance and quantiles only at a numeric node, and the
densities and marginals when the parent has marginals. Each derived capability
carries the parent's guard for the call it makes, and computes from the parent's
answer: a draw, a moment, or an expectation is projected onto the view's nodes,
a covariance or quantiles are restricted to the view's coordinates of the
parent's flat vector, and the density, marginals, and conditioning read the
parent at the parent's paths.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import probpipe
from probpipe import (
    KDEDistribution,
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    NumericSpec,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordSpec,
)
from probpipe.core._dispatch import Feasibility, MathematicalDomainError
from probpipe.distributions import (
    ConditionalDistribution,
    Distribution,
    DistributionSpec,
    FactoredDistribution,
    FieldView,
    NumericDistribution,
    SupportsConditionalLogProb,
    SupportsMarginals,
)
from probpipe.distributions._capabilities import (
    SupportsApproximateConditioning,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
    _capability_guard,
    _capability_subclass,
    _marginal_claims,
)
from probpipe.distributions._distribution import _fixed_paths
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.families import Cauchy, HalfCauchy, StudentT
from probpipe.linalg import DenseLinOp, LinOp
from probpipe.operations._marginal import marginal
from probpipe.operations._moments import mean
from tests._posterior import posterior_of

# -- Declarations -------------------------------------------------------------

#: The declared spec of a scalar normal draw, so that a scalar node of a double
#: matches the declaration of a normal marginal.
_REAL = Normal("x", 0.0, 1.0).event_spec.spec

#: A numeric exposed record: a group two levels deep, and a vector beside it.
_EVENT = OutputSpec(RecordSpec(model=RecordSpec(theta=RecordSpec(mu=_REAL, tau=(2,))), y=(3,)))

#: A mixed exposed record: an opaque leaf, a numeric leaf, and a mixed group.
_MIXED = OutputSpec(
    RecordSpec(label=OpaqueSpec(), x=(2,), group=RecordSpec(tag=OpaqueSpec(), w=()))
)

#: A whole record term, whose paths start with its component.
_WHOLE = OutputSpec(parameters=RecordSpec(beta=(2,), sigma=()))

_MEAN = jnp.array([1.0, 2.0, 3.0])
_COV = jnp.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])


# -- Test doubles -------------------------------------------------------------


class _Law(Distribution):
    """A law over a declared event that claims no capability."""

    def __init__(self, label: str, event_spec: OutputSpec) -> None:
        super().__init__(
            event_spec,
            label=label,
        )


def _unreachable(self: Any, *args: Any, **kwargs: Any) -> Any:
    raise AssertionError("deriving the capabilities of a view never calls its parent")


#: The methods a parent double defines to claim each capability.
_PARENT_METHODS: dict[type, tuple[str, ...]] = {
    SupportsSampling: ("_sample",),
    SupportsMean: ("_mean",),
    SupportsVariance: ("_variance",),
    SupportsCovariance: ("_cov",),
    SupportsQuantile: ("_quantile",),
    SupportsExpectation: ("_expectation",),
    SupportsLogProb: ("_log_prob",),
    SupportsUnnormalizedLogProb: ("_unnormalized_log_prob",),
    SupportsRandomLogProb: ("_random_log_prob",),
    SupportsRandomUnnormalizedLogProb: ("_random_unnormalized_log_prob",),
    SupportsMarginals: ("_marginal",),
    SupportsExactConditioning: ("_condition_on",),
    SupportsApproximateConditioning: ("_condition_on",),
}


def _parent(*protocols: type, event_spec: OutputSpec = _EVENT) -> Distribution:
    """A law over *event_spec* that claims exactly *protocols*, each by inheritance."""
    namespace = {
        method: _unreachable for protocol in protocols for method in _PARENT_METHODS[protocol]
    }
    parent_class = type(_Law)("_Parent", (_Law, *protocols), namespace)
    return parent_class("parent", event_spec)


class _UnguardedMarginalLaw(_Law, SupportsMarginals):
    """A law whose marginal at any path is a standard normal and has no guard.

    The marginal at a tuple of paths is the product of standard normals named by
    their final segments. Each path whose marginal is requested is recorded in
    ``marginal_calls``. With ``scores=False`` the marginal at a path is a law
    with no density instead. The law reports that its marginals claim the
    density when they score, or as ``reports`` states; each path whose report
    is read is recorded in ``report_calls``.
    """

    def __init__(
        self,
        label: str,
        event_spec: OutputSpec,
        *,
        scores: bool = True,
        reports: bool | None = None,
    ) -> None:
        super().__init__(label, event_spec)
        self.scores = scores
        self.reports = scores if reports is None else reports
        self.marginal_calls: list[Any] = []
        self.report_calls: list[Any] = []

    def _marginal_capabilities(self, path: str | tuple[str, ...]) -> frozenset[type]:
        self.report_calls.append(path)
        return frozenset({SupportsLogProb, SupportsUnnormalizedLogProb} if self.reports else ())

    def _marginal(self, path: str | tuple[str, ...]) -> Distribution:
        self.marginal_calls.append(path)
        if not isinstance(path, str):
            components = [each.rsplit("/", 1)[-1] for each in path]
            return FactoredDistribution(
                [Normal(c, 0.0, 1.0) for c in components],
                label="marginal",
            )
        component = path.rsplit("/", 1)[-1]
        if self.scores:
            return Normal(component, 0.0, 1.0)
        return _Law(component, OutputSpec(**{component: _REAL}))


class _MarginalLaw(_UnguardedMarginalLaw):
    """A law with marginals whose guard returns the report stored for a path.

    A path with no stored report is feasible.
    """

    def __init__(
        self,
        label: str,
        event_spec: OutputSpec,
        reports: dict[Any, Feasibility] | None = None,
    ) -> None:
        super().__init__(label, event_spec)
        self.reports = dict(reports or {})

    def _marginal_guard(self, path: str | tuple[str, ...]) -> Feasibility:
        return self.reports.get(path, Feasibility(True))


class _MomentMarginalLaw(_MarginalLaw):
    """A law with guarded marginals that reports a normal marginal's moments, and claims none.

    A view of it answers each moment from the marginal at its path.
    """

    def _marginal_capabilities(self, path: str | tuple[str, ...]) -> frozenset[type]:
        self.report_calls.append(path)
        return frozenset({SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile})


class _GuardedMeanLaw(_Law, SupportsMean):
    """A law whose mean is rejected by its guard."""

    REJECTED = Feasibility(False, "the mean does not exist")

    def _mean(self) -> Any:
        raise AssertionError("the guard rejects the mean")

    def _mean_guard(self) -> Feasibility:
        return self.REJECTED


class _ConditioningLaw(_Law, SupportsExactConditioning):
    """A law over the default event that conditions exactly on ``model/theta/tau`` alone.

    Each given it receives is recorded in ``given_calls``, and its guard rejects
    any other set of given paths.
    """

    def __init__(self, label: str, event_spec: OutputSpec) -> None:
        super().__init__(label, event_spec)
        self.given_calls: list[dict[str, Any]] = []

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution:
        self.given_calls.append(dict(given))
        remaining = RecordSpec(model=RecordSpec(theta=RecordSpec(mu=_REAL)), y=(3,))
        return _Law("conditioned", OutputSpec(remaining))

    def _condition_on_guard(self, paths: tuple[str, ...]) -> Feasibility:
        if tuple(paths) == ("model/theta/tau",):
            return Feasibility(True)
        return Feasibility(False, f"no exact conditioning on {sorted(paths)}")


class _CovarianceLaw(_Law, SupportsCovariance):
    """A numeric law over ``x`` of shape (1,) and ``y`` of shape (2,) with a dense covariance."""

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(RecordSpec(x=(1,), y=(2,))))

    def _cov(self) -> LinOp:
        return DenseLinOp(_COV)


class _QuantileLaw(_Law, SupportsQuantile):
    """A numeric law over ``x`` of shape (1,) and ``y`` of shape (2,).

    The quantile at a level is the level plus the coordinate's index in the flat
    event, returned as the mapping of each leaf's quantiles with the level axes
    leading.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(RecordSpec(x=(1,), y=(2,))))

    def _quantile(self, q: Any) -> Any:
        flat = jnp.asarray(q)[..., None] + jnp.arange(3.0)
        return {"x": flat[..., :1], "y": flat[..., 1:]}


class _FiniteLaw(_Law, SupportsExpectation):
    """A law over ``a`` and ``b`` with the atoms (0, 1) of weight 1/4 and (1, 3) of weight 3/4."""

    ATOMS = ((0.0, 1.0, 0.25), (1.0, 3.0, 0.75))

    def __init__(self, label: str) -> None:
        super().__init__(label, OutputSpec(RecordSpec(a=_REAL, b=_REAL)))

    def _expectation(self, f: Any) -> Any:
        return sum(
            weight
            * f(
                Record(
                    {"a": jnp.asarray(a), "b": jnp.asarray(b)},
                    label="atom",
                )
            )
            for a, b, weight in self.ATOMS
        )


#: The covariance of the flat vector ``(mu, tau, y)`` of a draw of the default event.
_FLAT_COV = jnp.eye(6) + 0.1 * jnp.outer(jnp.arange(6.0), jnp.arange(6.0))


class _NumericLaw(
    _Law, SupportsSampling, SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile
):
    """A numeric law over the default event whose flat vector is ``(mu, tau, y)``, of size 6.

    A draw is a standard normal flat vector split into the leaves, a batch of
    draws is a record batch on the level ``sample``, and the mean is ``(0, ..., 5)``.
    The quantile at a level is the level plus the coordinate's index, returned as
    the nested mapping of each leaf's quantiles with the level axes leading.
    """

    def __init__(self, label: str) -> None:
        super().__init__(label, _EVENT)

    @staticmethod
    def _fields(flat: Any) -> dict[str, Any]:
        return {
            "model": {"theta": {"mu": flat[..., 0], "tau": flat[..., 1:3]}},
            "y": flat[..., 3:],
        }

    def _record(self, flat: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        fields = self._fields(flat)
        if sample_shape:
            return NumericRecordBatch(
                fields,
                "sample",
                element_spec=self.event_spec.spec,
                axes_per_level=(len(sample_shape),),
                label=self.label,
            )
        return Record(
            fields,
            event_template=self.event_spec.spec,
            label=self.label,
        )

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self._record(jax.random.normal(key, (*sample_shape, 6)), sample_shape)

    def _mean(self) -> Any:
        return self._record(jnp.arange(6.0))

    def _variance(self) -> Any:
        return self._record(jnp.diag(_FLAT_COV))

    def _cov(self) -> LinOp:
        return DenseLinOp(_FLAT_COV)

    def _quantile(self, q: Any) -> Any:
        return self._fields(jnp.asarray(q)[..., None] + jnp.arange(6.0))


class _WholeLaw(_Law, SupportsSampling, SupportsCovariance):
    """A law over the whole record ``parameters``, whose flat vector is ``(beta, sigma)``."""

    def __init__(self, label: str) -> None:
        super().__init__(label, _WHOLE)

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        flat = jax.random.normal(key, (*sample_shape, 3))
        return Record(
            {"beta": flat[..., :2], "sigma": flat[..., 2]},
            label="parameters",
        )

    def _cov(self) -> LinOp:
        return DenseLinOp(_COV)


class _LevelGuardedQuantileLaw(_QuantileLaw):
    """A quantile law whose guard rejects levels outside the unit interval."""

    def _quantile_guard(self, q: Any) -> bool:
        """Every level lies in the unit interval."""
        levels = jnp.asarray(q)
        return bool(jnp.all((levels >= 0) & (levels <= 1)))


class _Kernel(ConditionalDistribution):
    """A kernel that produces ``y`` given ``beta`` and claims no capability but binding."""

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution:
        return Normal("y", given["beta"], 1.0)


class _ScoringKernel(_Kernel, SupportsConditionalLogProb):
    """The kernel of :class:`_Kernel` with the normal density of ``y`` given ``beta``."""

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        return Normal("y", given["beta"], 1.0)._log_prob(value)


def _product() -> Distribution:
    return Normal("a", 0.0, 1.0) * Normal("b", 2.0, 3.0)


def _joint_gaussian() -> Distribution:
    """``x`` and ``y`` of the means ``_MEAN`` and the diagonal blocks of ``_COV``, independent."""
    return MultivariateNormal("x", _MEAN[:1], cov=_COV[:1, :1]) * MultivariateNormal(
        "y", _MEAN[1:], cov=_COV[1:, 1:]
    )


def _dependent_joint() -> Distribution:
    """The joint ``p(y | beta) p(beta)``, whose marginal at ``y`` has no closed form."""
    return _Kernel(
        {"beta": _REAL},
        OutputSpec(y=_REAL),
        label="likelihood",
    ) * Normal("beta", 0.0, 1.0, label="beta")


# -- The derivation table -----------------------------------------------------

#: Every capability whose claim the tests read. The random densities are included,
#: although no row gives them.
_CAPABILITIES = (
    SupportsSampling,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
    SupportsQuantile,
    SupportsExpectation,
    SupportsLogProb,
    SupportsUnnormalizedLogProb,
    SupportsMarginals,
    SupportsExactConditioning,
    SupportsApproximateConditioning,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
)

#: Each row of the derivation table: a parent capability and the capabilities it
#: gives a view.
_ROWS: dict[type, set[type]] = {
    SupportsSampling: {SupportsSampling},
    SupportsMean: {SupportsMean},
    SupportsVariance: {SupportsVariance},
    SupportsCovariance: {SupportsCovariance},
    SupportsQuantile: {SupportsQuantile},
    SupportsExpectation: {SupportsExpectation},
    SupportsMarginals: {SupportsMarginals},
    SupportsExactConditioning: {SupportsExactConditioning},
    SupportsApproximateConditioning: {SupportsApproximateConditioning},
}

#: The rows that also require the view's node to be numeric.
_NUMERIC_ROWS = {SupportsCovariance, SupportsQuantile}


def _claimed(term: Any) -> set[type]:
    """The capabilities *term* claims."""
    return {protocol for protocol in _CAPABILITIES if isinstance(term, protocol)}


#: The density rows, which a view claims from its parent's report of its marginal.
_DENSITIES = {SupportsLogProb, SupportsUnnormalizedLogProb}

#: The moment rows, which a view also claims from its parent's report of its marginal.
_MOMENTS = {SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile}


def _derived(parent_claims: set[type], report: frozenset[type], *, numeric: bool) -> set[type]:
    """What the derivation table gives a view of a parent claiming *parent_claims*.

    *report* is the parent's report of its marginal at the view's path, from
    which a parent with marginals gives the view the normalized density, or
    else the unnormalized one, and each moment the report includes.
    """
    derived: set[type] = set()
    for capability, gives in _ROWS.items():
        if capability in parent_claims and (numeric or capability not in _NUMERIC_ROWS):
            derived |= gives
    if SupportsMarginals in parent_claims:
        if SupportsLogProb in report:
            derived |= _DENSITIES
        elif SupportsUnnormalizedLogProb in report:
            derived.add(SupportsUnnormalizedLogProb)
        derived |= {
            moment
            for moment in _MOMENTS
            if moment in report and (numeric or moment not in _NUMERIC_ROWS)
        }
    return derived


def _name(protocol: type) -> str:
    return protocol.__name__


# -- Tests --------------------------------------------------------------------

_DECLARATIONS = [
    pytest.param(_EVENT, "y", OutputSpec(y=NumericArraySpec((3,))), id="leaf"),
    pytest.param(
        _EVENT,
        "model",
        OutputSpec(model=RecordSpec(theta=RecordSpec(mu=_REAL, tau=(2,)))),
        id="group",
    ),
    pytest.param(
        _EVENT, "model/theta", OutputSpec(theta=RecordSpec(mu=_REAL, tau=(2,))), id="nested-group"
    ),
    pytest.param(_EVENT, "model/theta/mu", OutputSpec(mu=_REAL), id="nested-leaf"),
    pytest.param(
        _WHOLE,
        "parameters",
        OutputSpec(parameters=RecordSpec(beta=(2,), sigma=())),
        id="whole-term",
    ),
    pytest.param(
        _WHOLE, "parameters/sigma", OutputSpec(sigma=NumericArraySpec(())), id="whole-term-field"
    ),
    pytest.param(_MIXED, "label", OutputSpec(label=OpaqueSpec()), id="opaque-leaf"),
    pytest.param(
        _MIXED, "group", OutputSpec(group=RecordSpec(tag=OpaqueSpec(), w=())), id="mixed-group"
    ),
]

_PATHS = [pytest.param(*case.values[:2], id=case.id) for case in _DECLARATIONS]


class TestDeclaration:
    @pytest.mark.parametrize(("event_spec", "path", "declaration"), _DECLARATIONS)
    def test_the_declaration_is_the_node_whole_under_its_final_segment(
        self, event_spec, path, declaration
    ):
        view = FieldView(_Law("parent", event_spec), path)
        assert view.event_spec == declaration
        assert not view.event_spec.exposes_record
        assert view.spec == DistributionSpec(declaration)

    @pytest.mark.parametrize(("event_spec", "path"), _PATHS)
    def test_a_view_keeps_its_parent_label_and_reads_its_parent_there(self, event_spec, path):
        parent = _Law("parent", event_spec)
        view = FieldView(parent, path)
        assert view.label == "parent"
        assert view.parent is parent
        assert view.path == path

    @pytest.mark.parametrize(
        ("path", "numeric"), [("x", True), ("group/w", True), ("label", False), ("group", False)]
    )
    def test_numeric_membership_follows_the_node(self, path, numeric):
        view = FieldView(_Law("parent", _MIXED), path)
        assert isinstance(view, NumericDistribution) is numeric

    def test_a_leaf_view_has_the_leaf_shape_and_a_group_view_has_none(self):
        parent = _Law("parent", _EVENT)
        assert FieldView(parent, "model/theta/tau").event_shape == (2,)
        assert not hasattr(FieldView(parent, "model/theta"), "event_shape")

    def test_the_numeric_views_are_keyed_by_the_paths_of_the_view(self):
        view = FieldView(_Law("parent", _WHOLE), "parameters")
        assert list(view.dtypes) == ["parameters/beta", "parameters/sigma"]
        assert list(view.supports) == ["parameters/beta", "parameters/sigma"]

    def test_a_renamed_view_keeps_its_parent_path_and_declaration(self):
        parent = _Law("parent", _EVENT)
        view = FieldView(parent, "model/theta")
        renamed = view.with_label("coefficients")
        assert renamed.label == "coefficients"
        assert renamed.parent is parent
        assert renamed.path == "model/theta"
        assert renamed.event_spec == view.event_spec

    def test_the_provenance_of_a_view_records_its_parent(self):
        parent = _product()
        view = FieldView(parent, "a")
        assert view.provenance is not None
        assert [info.label for info in view.provenance.parents] == [parent.label]

    def test_binding_a_dimension_of_a_view_binds_it_in_the_parent(self):
        parent = _Law("parent", OutputSpec(RecordSpec(a=("n",), b=("n",))))
        bound = FieldView(parent, "a").with_dim_sizes(n=3)
        assert bound.path == "a"
        assert bound.event_spec == OutputSpec(a=NumericArraySpec((3,)))
        assert bound.parent.event_spec == parent.event_spec.with_dim_sizes(n=3)


_NOT_EVENT_PATHS = [
    pytest.param(_EVENT, "", id="empty"),
    pytest.param(_EVENT, "z", id="unknown-component"),
    pytest.param(_EVENT, "theta", id="field-without-its-component"),
    pytest.param(_EVENT, "model/phi", id="unknown-field"),
    pytest.param(_EVENT, "y/0", id="below-a-leaf"),
    pytest.param(_EVENT, "/model", id="leading-separator"),
    pytest.param(_EVENT, "model/", id="trailing-separator"),
    pytest.param(_EVENT, "model//theta", id="empty-segment"),
    pytest.param(_WHOLE, "beta", id="whole-term-field-without-its-component"),
    pytest.param(_WHOLE, "parameters/beta/0", id="below-a-whole-term-leaf"),
    pytest.param(OutputSpec(x=NumericArraySpec((3,))), "x/0", id="below-a-whole-array"),
]


class TestPaths:
    @pytest.mark.parametrize(("event_spec", "path"), _NOT_EVENT_PATHS)
    def test_a_path_that_is_not_an_event_path_raises_key_error(self, event_spec, path):
        with pytest.raises(KeyError):
            FieldView(_Law("parent", event_spec), path)

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(object, id="object"),
            pytest.param(lambda: {"y": 0.0}, id="mapping"),
            pytest.param(
                lambda: Record(
                    {"y": 0.0},
                    label="r",
                ),
                id="record",
            ),
            pytest.param(
                lambda: _Kernel(
                    {"beta": _REAL},
                    OutputSpec(y=_REAL),
                    label="likelihood",
                ),
                id="kernel",
            ),
        ],
    )
    def test_a_parent_that_is_not_a_distribution_raises_type_error(self, make):
        with pytest.raises(TypeError, match="Distribution"):
            FieldView(make(), "y")


class TestIndexing:
    @pytest.mark.parametrize("path", ["model", "model/theta", "y"])
    def test_a_view_at_its_own_component_is_itself(self, path):
        view = FieldView(_Law("parent", _EVENT), path)
        assert view[path.rsplit("/", 1)[-1]] is view

    @pytest.mark.parametrize(
        ("path", "key", "joined"),
        [
            ("model", "model/theta", "model/theta"),
            ("model", "model/theta/tau", "model/theta/tau"),
            ("model/theta", "theta/mu", "model/theta/mu"),
        ],
    )
    def test_a_path_within_the_view_joins_the_path_of_the_view(self, path, key, joined):
        parent = _Law("parent", _EVENT)
        view = FieldView(parent, path)[key]
        assert isinstance(view, FieldView)
        assert view.parent is parent
        assert view.path == joined
        assert view.event_spec == FieldView(parent, joined).event_spec

    def test_indexing_a_view_twice_joins_both_keys(self):
        parent = _Law("parent", _EVENT)
        view = FieldView(parent, "model")["model/theta"]["theta/mu"]
        assert view.parent is parent
        assert view.path == "model/theta/mu"

    def test_a_joined_view_claims_what_a_view_at_the_joined_path_claims(self):
        parent = _parent(SupportsMean, SupportsMarginals)
        joined = FieldView(parent, "model")["model/theta/mu"]
        assert type(joined) is type(FieldView(parent, "model/theta/mu"))

    @pytest.mark.parametrize(
        "key", ["theta", "model/", "", "model/phi", "y", "model//theta", "model/theta/mu/0"]
    )
    def test_a_key_that_is_not_an_event_path_of_the_view_raises_key_error(self, key):
        view = FieldView(_Law("parent", _EVENT), "model")
        with pytest.raises(KeyError):
            view[key]

    def test_a_view_renamed_by_label_is_still_addressed_by_its_component(self):
        view = FieldView(_Law("parent", _EVENT), "model/theta").with_label("coefficients")
        assert view["theta"] is view

    def test_a_view_renamed_by_path_is_addressed_by_its_new_component(self):
        view = FieldView(_Law("parent", _EVENT), "y").with_path_names(y="obs")
        assert view["obs"] is view
        with pytest.raises(KeyError):
            view["y"]

    def test_a_selection_of_several_paths_exposes_a_record_of_them_in_order(self):
        view = FieldView(_Law("parent", _EVENT), "model/theta")
        selection = view[("theta/tau", "theta/mu")]
        assert selection.event_spec.exposes_record
        assert list(selection.event_spec.components) == ["tau", "mu"]
        assert selection.event_spec.spec == RecordSpec(tau=(2,), mu=_REAL)

    def test_a_selection_whose_final_segments_collide_raises_value_error(self):
        twins = RecordSpec(g=RecordSpec(a=RecordSpec(x=()), b=RecordSpec(x=())))
        view = FieldView(_Law("parent", OutputSpec(twins)), "g")
        with pytest.raises(ValueError):
            view[("g/a/x", "g/b/x")]

    def test_a_selection_keeps_its_parent_label_and_reads_its_parent_at_them(self):
        parent = _Law("parent", _EVENT)
        selection = FieldView(parent, "model/theta")[("theta/tau", "theta/mu")]
        assert selection.parent is parent
        assert selection.path == ("model/theta/tau", "model/theta/mu")
        assert selection.label == "parent"

    def test_a_selection_of_one_path_exposes_a_record_of_one_field(self):
        selection = FieldView(_Law("parent", _EVENT), ("model/theta",))
        assert selection.event_spec == OutputSpec(RecordSpec(theta=RecordSpec(mu=_REAL, tau=(2,))))

    @pytest.mark.parametrize(
        ("key", "path"),
        [
            ("theta/mu", "model/theta/mu"),
            ("theta", "model/theta"),
            (("theta/tau", "y"), ("model/theta/tau", "y")),
        ],
    )
    def test_a_key_of_a_selection_is_the_parent_view_at_the_node_path(self, key, path):
        parent = _Law("parent", _EVENT)
        view = FieldView(parent, ("model/theta", "y"))[key]
        assert view.parent is parent
        assert view.path == path

    @pytest.mark.parametrize("key", [("theta/mu", 3), ("theta/phi",), 3])
    def test_a_key_that_is_not_a_selection_of_view_paths_raises_key_error(self, key):
        view = FieldView(_Law("parent", _EVENT), "model/theta")
        with pytest.raises(KeyError):
            view[key]

    def test_an_empty_selection_raises_value_error(self):
        """An empty tuple is malformed rather than an unknown path."""
        parent = _Law("parent", _EVENT)
        with pytest.raises(ValueError, match="select at least one field path"):
            FieldView(parent, ())
        with pytest.raises(ValueError, match="select at least one field path"):
            parent[()]
        with pytest.raises(ValueError, match="select at least one field path"):
            FieldView(parent, "model/theta")[()]

    @pytest.mark.parametrize("path", [["y"], ("y", 0), 0])
    def test_a_path_that_is_neither_a_string_nor_a_tuple_of_them_raises_type_error(self, path):
        with pytest.raises((TypeError, AttributeError)):
            FieldView(_Law("parent", _EVENT), path)

    def test_indexing_an_exposed_record_returns_a_field_view(self):
        parent = _product()
        view = parent["a"]
        assert isinstance(view, FieldView)
        assert view.parent is parent
        assert view.path == "a"

    def test_indexing_a_whole_record_below_its_component_returns_a_field_view(self):
        parent = _Law("parent", _WHOLE)
        view = parent["parameters/beta"]
        assert isinstance(view, FieldView)
        assert view.parent is parent
        assert view.path == "parameters/beta"


class TestCapabilityDerivation:
    @pytest.mark.parametrize("path", ["model", "model/theta/mu", "y"])
    @pytest.mark.parametrize("capability", list(_ROWS), ids=_name)
    def test_a_parent_capability_gives_the_view_exactly_its_row(self, capability, path):
        view = FieldView(_parent(capability), path)
        assert _claimed(view) == _ROWS[capability]

    def test_a_parent_claiming_every_row_gives_the_view_every_row(self):
        capabilities = set(_ROWS) - {SupportsApproximateConditioning}
        view = FieldView(_parent(*capabilities, SupportsLogProb), "model")
        expected = set().union(*(_ROWS[capability] for capability in capabilities))
        assert _claimed(view) == expected | _DENSITIES

    def test_a_parent_with_marginals_reports_its_own_density_by_default(self):
        assert _claimed(FieldView(_parent(SupportsMarginals, SupportsLogProb), "y")) == {
            SupportsMarginals,
            *_DENSITIES,
        }
        unnormalized = _parent(SupportsMarginals, SupportsUnnormalizedLogProb)
        assert _claimed(FieldView(unnormalized, "y")) == {
            SupportsMarginals,
            SupportsUnnormalizedLogProb,
        }

    def test_a_view_claims_the_density_its_parent_reports_for_the_marginal(self):
        assert _claimed(FieldView(_UnguardedMarginalLaw("parent", _EVENT), "y")) >= _DENSITIES
        silent = _UnguardedMarginalLaw("parent", _EVENT, scores=False)
        assert not _DENSITIES & _claimed(FieldView(silent, "y"))

    def test_a_view_claims_each_moment_its_parent_reports_for_the_marginal(self):
        parent = _MomentMarginalLaw("parent", _EVENT)
        assert _claimed(FieldView(parent, "model/theta/mu")) >= _MOMENTS
        assert parent.report_calls == ["model/theta/mu"]

    def test_a_view_of_a_root_factor_of_a_dependent_joint_claims_its_moments(self):
        """The dependent joint claims no moment, and its root factor's marginal has each one."""
        joint = _dependent_joint()
        assert not _MOMENTS & _claimed(joint)
        assert _claimed(FieldView(joint, "beta")) >= _MOMENTS
        assert not _MOMENTS & _claimed(FieldView(joint, "y"))

    def test_a_moment_from_the_marginal_needs_a_numeric_node_for_covariance(self):
        parent = _MomentMarginalLaw("parent", _MIXED)
        assert _claimed(FieldView(parent, "label")) & _MOMENTS == {SupportsMean, SupportsVariance}
        assert _claimed(FieldView(parent, "x")) >= _MOMENTS

    def test_a_view_reads_the_report_once_at_construction(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        view = FieldView(parent, "model/theta/mu")
        view._log_prob(0.5)
        view._log_prob(1.5)
        assert parent.report_calls == ["model/theta/mu"]

    def test_a_joint_reports_the_claims_of_the_factors_a_marginal_keeps(self):
        joint = Normal("a", 0.0, 1.0) * EmpiricalDistribution(
            jnp.array([0.0, 1.0, 3.0]), component="b"
        )
        assert _claimed(joint["a"]) >= _DENSITIES
        assert not _DENSITIES & _claimed(joint["b"])
        assert SupportsSampling in _claimed(joint["b"])

    def test_a_parent_claiming_no_capability_gives_a_view_of_the_base_class(self):
        view = FieldView(_Law("parent", _EVENT), "y")
        assert type(view) is FieldView
        assert _claimed(view) == set()

    def test_a_parent_density_without_marginals_gives_the_view_no_density(self):
        assert _claimed(FieldView(_parent(SupportsLogProb), "y")) == set()

    def test_a_parent_that_does_not_sample_still_gives_the_view_its_moments(self):
        view = FieldView(_parent(SupportsMean, SupportsVariance, SupportsCovariance), "y")
        assert _claimed(view) == {SupportsMean, SupportsVariance, SupportsCovariance}

    @pytest.mark.parametrize(
        ("path", "numeric"), [("x", True), ("group/w", True), ("label", False), ("group", False)]
    )
    def test_covariance_and_quantiles_need_a_numeric_node(self, path, numeric):
        parent = _parent(SupportsMean, SupportsCovariance, SupportsQuantile, event_spec=_MIXED)
        expected = {SupportsMean} | ({SupportsCovariance, SupportsQuantile} if numeric else set())
        assert _claimed(FieldView(parent, path)) == expected

    def test_a_view_derives_no_random_density(self):
        parent = _parent(
            SupportsMarginals, SupportsRandomLogProb, SupportsRandomUnnormalizedLogProb
        )
        assert _claimed(FieldView(parent, "y")) == _ROWS[SupportsMarginals]
        assert _marginal_claims(parent, "y") >= {SupportsRandomLogProb}

    @pytest.mark.parametrize(
        ("make", "path"),
        [
            pytest.param(_product, "a", id="product"),
            pytest.param(
                lambda: Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0), "b", id="independent-joint"
            ),
            pytest.param(_dependent_joint, "y", id="dependent-joint"),
            pytest.param(_dependent_joint, "beta", id="dependent-joint-root"),
            pytest.param(_joint_gaussian, "y", id="joint-gaussian"),
        ],
    )
    def test_a_view_of_a_library_law_claims_what_the_table_derives(self, make, path):
        parent = make()
        view = FieldView(parent, path)
        numeric = isinstance(view.event_spec.spec, NumericSpec)
        report = _marginal_claims(parent, path)
        assert _claimed(view) == _derived(_claimed(parent), report, numeric=numeric)

    @pytest.mark.parametrize(
        ("paths", "numeric"), [(("x", "group/w"), True), (("x", "label"), False)]
    )
    def test_a_selection_claims_covariance_and_quantiles_when_every_node_is_numeric(
        self, paths, numeric
    ):
        parent = _parent(SupportsMean, SupportsCovariance, SupportsQuantile, event_spec=_MIXED)
        expected = {SupportsMean} | ({SupportsCovariance, SupportsQuantile} if numeric else set())
        assert _claimed(FieldView(parent, paths)) == expected


class TestGuards:
    @pytest.mark.parametrize("method", ["_log_prob", "_unnormalized_log_prob"])
    def test_the_density_guard_is_the_parent_marginal_guard_at_the_path(self, method):
        rejected = Feasibility(False, "no closed form at model/theta/mu")
        parent = _MarginalLaw("parent", _EVENT, {"model/theta/mu": rejected})
        assert _capability_guard(FieldView(parent, "model/theta/mu"), method) == rejected
        assert _capability_guard(FieldView(parent, "y"), method) == Feasibility(True)

    @pytest.mark.parametrize("method", ["_log_prob", "_unnormalized_log_prob"])
    def test_an_unresolved_marginal_guard_leaves_the_density_unresolved(self, method):
        unresolved = Feasibility(None, pending=("the size of n",))
        parent = _MarginalLaw("parent", _EVENT, {"model": unresolved})
        assert _capability_guard(FieldView(parent, "model"), method) == unresolved

    def test_a_parent_marginal_with_no_guard_gives_a_feasible_density(self):
        view = FieldView(_UnguardedMarginalLaw("parent", _EVENT), "model/theta/mu")
        assert _capability_guard(view, "_log_prob") == Feasibility(True)

    @pytest.mark.parametrize("path", ["y", "beta"])
    def test_the_density_guard_of_a_joint_view_is_the_joint_marginal_guard(self, path):
        joint = _ScoringKernel(
            {"beta": _REAL},
            OutputSpec(y=_REAL),
            label="likelihood",
        ) * Normal("beta", 0.0, 1.0)
        report = _capability_guard(FieldView(joint, path), "_log_prob")
        assert report == _capability_guard(joint, "_marginal", path)
        # The root factor's marginal is exact, and the child's integrates out its parent.
        assert report.feasible is (path == "beta")

    def test_the_density_refusal_names_the_fields_rather_than_the_factors(self):
        joint = _ScoringKernel(
            {"beta": _REAL},
            OutputSpec(y=_REAL),
            label="likelihood",
        ) * Normal("beta", 0.0, 1.0).with_label("prior")
        report = _capability_guard(FieldView(joint, "y"), "_log_prob")
        assert report.feasible is False
        assert "['y']" in report.description and "['beta']" in report.description
        assert "prior" not in report.description

    def test_a_joint_view_of_a_factor_without_a_density_claims_none(self):
        joint = _dependent_joint()
        assert not _DENSITIES & _claimed(FieldView(joint, "y"))
        assert _claimed(FieldView(joint, "beta")) >= _DENSITIES

    @pytest.mark.parametrize("method", ["_log_prob", "_unnormalized_log_prob", "_marginal"])
    def test_a_view_of_a_parent_without_marginals_has_no_density_or_marginal_to_guard(self, method):
        view = FieldView(_parent(SupportsSampling, SupportsLogProb), "y")
        with pytest.raises(AttributeError):
            _capability_guard(view, method)

    def test_a_marginal_reported_without_a_density_gives_no_density_to_guard(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT, scores=False)
        with pytest.raises(AttributeError):
            _capability_guard(FieldView(parent, "model/theta/mu"), "_log_prob")

    def test_the_marginal_guard_at_a_view_path_is_the_parent_guard_at_the_joined_path(self):
        rejected = Feasibility(False, "no closed form at model/theta/mu")
        parent = _MarginalLaw("parent", _EVENT, {"model/theta/mu": rejected})
        view = FieldView(parent, "model/theta")
        assert _capability_guard(view, "_marginal", "theta/mu") == rejected
        assert _capability_guard(view, "_marginal", "theta") == Feasibility(True)

    def test_the_marginal_guard_of_several_view_paths_is_the_parent_guard_at_theirs(self):
        rejected = Feasibility(False, "the pair has no closed form")
        parent = _MarginalLaw("parent", _EVENT, {("model/theta/mu", "model/theta/tau"): rejected})
        view = FieldView(parent, "model/theta")
        assert _capability_guard(view, "_marginal", ("theta/mu", "theta/tau")) == rejected

    @pytest.mark.parametrize("method", ["_mean", "_variance", "_cov"])
    def test_a_moment_from_the_marginal_takes_the_parent_marginal_guard(self, method):
        rejected = Feasibility(False, "no closed form at model/theta/mu")
        parent = _MomentMarginalLaw("parent", _EVENT, {"model/theta/mu": rejected})
        assert _capability_guard(FieldView(parent, "model/theta/mu"), method) == rejected
        assert _capability_guard(FieldView(parent, "y"), method) == Feasibility(True)

    def test_a_moment_from_the_marginal_takes_the_marginal_guard_of_the_moment(self):
        """A traced tail parameter leaves the root factor's mean unresolved, and the view's."""
        reports = []

        def probe(df):
            joint = _Kernel(
                {"beta": _REAL},
                OutputSpec(y=_REAL),
                label="likelihood",
            ) * StudentT("beta", df, 0.0, 1.0)
            reports.append(_capability_guard(FieldView(joint, "beta"), "_mean"))
            reports.append(_capability_guard(marginal(joint, "beta"), "_mean"))
            return df

        jax.eval_shape(probe, 3.0)
        view_report, marginal_report = reports
        assert view_report.feasible is None
        assert view_report == marginal_report

    def test_a_projected_capability_carries_the_parent_guard(self):
        view = FieldView(_GuardedMeanLaw("parent", _EVENT), "y")
        assert _capability_guard(view, "_mean") == _GuardedMeanLaw.REJECTED

    def test_the_conditioning_guard_is_the_parent_guard_at_the_given_paths(self):
        parent = _ConditioningLaw("parent", _EVENT)
        view = FieldView(parent, "model/theta")
        rejected = parent._condition_on_guard(("model/theta/mu",))
        assert _capability_guard(view, "_condition_on", ("theta/mu",)) == rejected
        assert _capability_guard(view, "_condition_on", ("theta/tau",)) == Feasibility(True)

    @pytest.mark.parametrize("path", ["theta/phi", "model/theta", ("theta/mu", "phi")])
    def test_the_marginal_guard_rejects_a_path_that_is_not_a_view_path(self, path):
        view = FieldView(_MarginalLaw("parent", _EVENT), "model/theta")
        report = _capability_guard(view, "_marginal", path)
        assert report.feasible is False
        assert "not an event path of the view" in report.description

    def test_the_marginal_guard_rejects_paths_that_share_a_final_segment(self):
        twins = RecordSpec(g=RecordSpec(a=RecordSpec(x=()), b=RecordSpec(x=())))
        view = FieldView(_MarginalLaw("parent", OutputSpec(twins)), "g")
        assert _capability_guard(view, "_marginal", ("g/a/x", "g/b/x")).feasible is False
        assert _capability_guard(view, "_marginal", ()).feasible is False

    def test_the_density_guard_of_a_selection_is_the_parent_marginal_guard_at_its_paths(self):
        rejected = Feasibility(False, "the pair has no closed form")
        parent = _MarginalLaw("parent", _EVENT, {("model/theta/mu", "y"): rejected})
        assert (
            _capability_guard(FieldView(parent, ("model/theta/mu", "y")), "_log_prob") == rejected
        )

    @pytest.mark.parametrize("paths", [("theta",), ("theta/mu", "theta/tau")])
    def test_the_conditioning_guard_rejects_a_given_that_covers_the_view(self, paths):
        view = FieldView(_ConditioningLaw("parent", _EVENT), "model/theta")
        report = _capability_guard(view, "_condition_on", paths)
        assert report.feasible is False
        assert "fixes every field" in report.description

    def test_the_conditioning_guard_rejects_a_path_that_is_not_a_view_path(self):
        view = FieldView(_ConditioningLaw("parent", _EVENT), "model/theta")
        assert _capability_guard(view, "_condition_on", ("y",)).feasible is False

    def test_the_quantile_guard_is_the_parent_guard_at_the_levels(self):
        view = FieldView(_LevelGuardedQuantileLaw("parent"), "y")
        assert _capability_guard(view, "_quantile", jnp.array([0.25, 0.5])) == Feasibility(True)
        assert _capability_guard(view, "_quantile", jnp.array([1.5])).feasible is False


class TestDerivedBehavior:
    def test_sibling_views_drawn_with_one_key_project_one_parent_draw(self, key):
        parent = _product()
        draw = parent._sample(key)
        assert jnp.array_equal(FieldView(parent, "a")._sample(key), draw["a"])
        assert jnp.array_equal(FieldView(parent, "b")._sample(key), draw["b"])

    def test_a_batched_draw_prepends_the_sample_shape(self, key):
        parent = _joint_gaussian()
        draws = FieldView(parent, "y")._sample(key, (4,))
        assert draws.shape == (4, 2)
        assert jnp.array_equal(draws, parent._sample(key, (4,))["y"])

    def test_the_view_mean_is_the_parent_mean_at_the_path(self):
        assert jnp.allclose(FieldView(_joint_gaussian(), "y")._mean(), _MEAN[1:])

    def test_the_view_variance_is_the_parent_variance_at_its_coordinates(self):
        assert jnp.allclose(FieldView(_joint_gaussian(), "y")._variance(), jnp.diag(_COV)[1:])

    def test_the_view_covariance_is_the_sub_block_of_the_parent_covariance(self):
        cov = FieldView(_CovarianceLaw("parent"), "y")._cov()
        assert isinstance(cov, LinOp)
        assert cov.shape == (2, 2)
        assert jnp.allclose(cov.to_dense(), _COV[1:, 1:])

    def test_the_view_quantiles_are_the_parent_quantiles_at_its_node(self):
        levels = jnp.array([0.25, 0.5])
        quantiles = FieldView(_QuantileLaw("parent"), "y")._quantile(levels)
        assert quantiles.shape == (2, 2)
        assert jnp.allclose(quantiles, levels[:, None] + jnp.array([1.0, 2.0]))

    @pytest.mark.parametrize(("q", "shape"), [(0.5, ()), (jnp.array([0.25, 0.75]), (2,))])
    def test_the_quantiles_of_a_scalar_leaf_of_an_empirical_record(self, q, shape):
        atoms = NumericRecordBatch(
            {"b": jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]), "a": jnp.array([3.0, 1.0, 2.0])},
            "row",
            element_spec=RecordSpec(b=(2,), a=()),
            label="rows",
        )
        parent = EmpiricalDistribution(atoms, label="post")
        quantiles = FieldView(parent, "a")._quantile(q)
        assert jnp.shape(quantiles) == shape
        expected = EmpiricalDistribution(jnp.array([3.0, 1.0, 2.0]), component="a")._quantile(q)
        assert jnp.allclose(quantiles, expected)

    def test_a_selection_of_a_whole_array_keeps_every_level(self):
        parent = EmpiricalDistribution(jnp.array([4.0, 1.0, 3.0, 2.0]), component="theta")
        levels = jnp.array([0.25, 0.5, 1.0])
        quantiles = FieldView(parent, ("theta",))._quantile(levels)
        assert list(quantiles) == ["theta"]
        assert jnp.allclose(quantiles["theta"], parent._quantile(levels))

    def test_the_view_quantiles_of_a_parent_with_record_quantiles(self):
        values = jnp.array([[1.0, 2.0], [3.0, 5.0], [2.0, 4.0]])
        atoms = NumericRecordBatch(
            {"u": values[:, 0], "v": values[:, 1]},
            "row",
            element_spec=RecordSpec(u=(), v=()),
            label="rows",
        )
        parent = EmpiricalDistribution(atoms, label="e")
        levels = jnp.array([0.25, 0.5])
        quantiles = FieldView(parent, "v")._quantile(levels)
        assert quantiles.shape == (2,)
        assert jnp.allclose(quantiles, parent._quantile(levels)["v"])

    def test_a_moment_from_the_marginal_is_the_marginal_moment(self):
        joint = _Kernel(
            {"beta": _REAL},
            OutputSpec(y=_REAL),
            label="likelihood",
        ) * Normal("beta", 1.5, 2.0)
        view = FieldView(joint, "beta")
        assert jnp.allclose(view._mean(), 1.5)
        assert jnp.allclose(view._variance(), 4.0)
        assert jnp.allclose(view._cov().to_dense(), Normal("beta", 1.5, 2.0)._cov().to_dense())
        assert jnp.allclose(view._quantile(0.5), 1.5)

    def test_the_mean_of_a_root_factor_view_is_exact(self):
        joint = _Kernel(
            {"beta": _REAL},
            OutputSpec(y=_REAL),
            label="likelihood",
        ) * Normal("beta", 1.5, 2.0)
        report = mean.check(joint["beta"])
        assert (report.route, report.exact) == ("closed_form", True)
        assert jnp.allclose(jnp.asarray(mean(joint["beta"])), 1.5)

    def test_a_view_of_a_half_cauchy_root_factor_has_an_infinite_mean(self):
        joint = _Kernel(
            {"beta": _REAL},
            OutputSpec(y=_REAL),
            label="likelihood",
        ) * HalfCauchy("beta", 0.0, 1.0)
        assert float(jnp.asarray(mean(joint["beta"]))) == jnp.inf
        assert float(jnp.asarray(mean(marginal(joint, "beta")))) == jnp.inf

    def test_a_view_raises_as_its_marginal_does_for_a_moment_that_does_not_exist(self):
        joint = _Kernel(
            {"beta": _REAL},
            OutputSpec(y=_REAL),
            label="likelihood",
        ) * Cauchy("beta", 0.0, 1.0)
        with pytest.raises(MathematicalDomainError, match="does not exist"):
            mean(marginal(joint, "beta"))
        with pytest.raises(MathematicalDomainError, match="does not exist"):
            mean(joint["beta"])

    def test_the_view_expectation_composes_with_the_projection(self):
        expectation = FieldView(_FiniteLaw("parent"), "b")._expectation(lambda b: b**2)
        assert jnp.allclose(expectation, 0.25 * 1.0 + 0.75 * 9.0)

    @pytest.mark.parametrize("method", ["_log_prob", "_unnormalized_log_prob"])
    def test_the_view_density_is_the_density_of_the_parent_marginal(self, method):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        density = getattr(FieldView(parent, "model/theta/mu"), method)(0.5)
        assert jnp.allclose(density, Normal("mu", 0.0, 1.0)._log_prob(0.5))
        assert parent.marginal_calls == ["model/theta/mu"]

    def test_the_view_marginal_is_the_parent_marginal_at_the_joined_path(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        marginal = FieldView(parent, "model/theta")._marginal("theta/mu")
        assert parent.marginal_calls == ["model/theta/mu"]
        assert not isinstance(marginal, FieldView)
        assert list(marginal.event_spec.components) == ["mu"]

    def test_the_view_marginal_keeps_the_view_label(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        assert FieldView(parent, "model/theta")._marginal("theta/mu").label == "parent"
        relabeled = FieldView(parent, "model/theta").with_label("theta")
        assert relabeled._marginal(("theta/mu", "theta/tau")).label == "theta"

    def test_conditioning_a_view_conditions_its_parent_at_the_given_paths(self):
        parent = _ConditioningLaw("parent", _EVENT)
        conditioned = FieldView(parent, "model/theta")._condition_on({"theta/tau": jnp.zeros(2)})
        assert [set(given) for given in parent.given_calls] == [{"model/theta/tau"}]
        assert conditioned.event_spec == OutputSpec(theta=RecordSpec(mu=_REAL))

    def test_a_group_view_draws_the_nested_mapping_of_the_parent_draw_at_its_node(self, key):
        parent = _NumericLaw("parent")
        draw = FieldView(parent, "model/theta")._sample(key)
        assert isinstance(draw, dict)
        assert list(draw.keys()) == ["mu", "tau"]
        assert jnp.array_equal(draw["tau"], parent._sample(key)["model/theta/tau"])

    def test_a_batched_group_draw_is_the_mapping_of_the_parent_columns_at_its_node(self, key):
        parent = _NumericLaw("parent")
        draws = FieldView(parent, "model/theta")._sample(key, (4,))
        assert isinstance(draws, dict)
        assert jnp.shape(draws["mu"]) == (4,)
        assert jnp.array_equal(draws["tau"], parent._sample(key, (4,))["model/theta/tau"])

    def test_a_view_of_a_mapping_parent_draws_its_node_of_the_mapping(self, key):
        parent = Normal("a", 0.0, 1.0) * Normal("b", 2.0, 3.0) * Normal("c", 0.0, 1.0)
        draws = FieldView(parent, ("c", "b"))._sample(key, (3,))
        parent_draws = parent._sample(key, (3,))
        assert isinstance(draws, dict) and list(draws) == ["c", "b"]
        assert jnp.array_equal(draws["c"], parent_draws["c"])
        assert jnp.array_equal(draws["b"], parent_draws["b"])

    def test_a_view_of_a_whole_record_at_its_component_draws_the_term(self, key):
        parent = _WholeLaw("parent")
        draw = FieldView(parent, "parameters")._sample(key)
        assert isinstance(draw, dict) and list(draw) == ["beta", "sigma"]
        term = parent._sample(key)
        assert all(jnp.array_equal(draw[name], term[name]) for name in draw)

    def test_a_view_of_a_whole_record_field_draws_the_field_of_the_term(self, key):
        parent = _WholeLaw("parent")
        draw = FieldView(parent, "parameters/beta")._sample(key)
        assert jnp.array_equal(draw, parent._sample(key)["beta"])

    def test_the_mean_of_a_group_view_is_the_sub_record_of_the_parent_mean(self):
        mean = FieldView(_NumericLaw("parent"), "model/theta")._mean()
        assert jnp.allclose(mean["mu"], 0.0)
        assert jnp.allclose(mean["tau"], jnp.array([1.0, 2.0]))

    @pytest.mark.parametrize(
        ("path", "coordinates"),
        [
            ("model/theta/mu", [0]),
            ("model/theta/tau", [1, 2]),
            ("model/theta", [0, 1, 2]),
            ("model", [0, 1, 2]),
            ("y", [3, 4, 5]),
        ],
    )
    def test_the_covariance_of_a_node_is_the_parent_block_at_its_coordinates(
        self, path, coordinates
    ):
        cov = FieldView(_NumericLaw("parent"), path)._cov()
        index = jnp.asarray(coordinates)
        assert isinstance(cov, LinOp)
        assert jnp.allclose(cov.to_dense(), _FLAT_COV[index][:, index])

    @pytest.mark.parametrize(
        ("path", "coordinates"), [("parameters", [0, 1, 2]), ("parameters/sigma", [2])]
    )
    def test_the_covariance_of_a_whole_record_node_follows_the_term_coordinates(
        self, path, coordinates
    ):
        index = jnp.asarray(coordinates)
        cov = FieldView(_WholeLaw("parent"), path)._cov()
        assert jnp.allclose(cov.to_dense(), _COV[index][:, index])

    def test_the_quantiles_of_a_nested_node_are_the_parent_quantiles_at_its_coordinates(self):
        quantiles = FieldView(_NumericLaw("parent"), "model/theta/tau")._quantile(0.5)
        assert jnp.allclose(quantiles, jnp.array([1.5, 2.5]))

    def test_the_density_raises_when_the_reported_marginal_does_not_score(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT, scores=False, reports=True)
        view = FieldView(parent, "model/theta/mu")
        with pytest.raises(TypeError, match="no normalized density"):
            view._log_prob(0.5)

    def test_the_marginal_at_a_renamed_component_is_named_by_it(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        view = FieldView(parent, "model/theta/mu").with_path_names(mu="location")
        marginal = view._marginal("location")
        assert parent.marginal_calls == ["model/theta/mu"]
        assert list(marginal.event_spec.components) == ["location"]

    def test_the_marginal_at_a_path_that_is_not_a_view_path_raises_key_error(self):
        view = FieldView(_UnguardedMarginalLaw("parent", _EVENT), "model/theta")
        with pytest.raises(KeyError):
            view._marginal("theta/phi")

    def test_conditioning_on_a_path_that_is_not_a_view_path_raises_key_error(self):
        parent = _ConditioningLaw("parent", _EVENT)
        with pytest.raises(KeyError):
            FieldView(parent, "model/theta")._condition_on({"y": jnp.zeros(3)})
        assert parent.given_calls == []

    def test_conditioning_on_every_field_of_the_view_raises_value_error(self):
        parent = _ConditioningLaw("parent", _EVENT)
        with pytest.raises(ValueError, match="fixes every field"):
            FieldView(parent, "model/theta/tau")._condition_on({"tau": jnp.zeros(2)})
        assert parent.given_calls == []

    def test_the_raw_form_of_a_view_is_the_backend_object_of_the_marginal(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        raw = FieldView(parent, "model/theta/mu").raw()
        assert parent.marginal_calls == ["model/theta/mu"]
        assert isinstance(raw, tfd.Normal)

    def test_the_raw_form_of_a_view_is_the_detached_marginal_without_a_backend(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT, scores=False)
        view = FieldView(parent, "model/theta/mu")
        raw = view.raw()
        assert parent.marginal_calls == ["model/theta/mu"]
        assert not isinstance(raw, FieldView)
        # The detached marginal carries the view's expression, whatever label the
        # double gives its marginal.
        assert (raw.label, raw.spec, raw.provenance) == ("parent", view.spec, None)
        assert raw.notation == view.notation == "parent(mu)"


class TestTheViewOfAWeightedLaw:
    """A field view of a weighted law takes the weighted moments at its path.

    The atoms of ``a`` are 0, 1, 2, and 3 with weights 0.7, 0.1, 0.1, and 0.1,
    so the weighted mean is 0.6 and the weighted variance 1.04, where the
    unweighted ones are 1.5 and 1.25.
    """

    _WEIGHTS = jnp.array([0.7, 0.1, 0.1, 0.1])
    _A = jnp.array([0.0, 1.0, 2.0, 3.0])
    _B = jnp.array([10.0, 20.0, 30.0, 40.0])

    def _record_atoms(self):
        return NumericRecordBatch(
            {"a": self._A, "b": self._B},
            "obs",
            element_spec=NumericRecordSpec(a=(), b=()),
            label="atoms",
        )

    def test_a_weighted_posterior(self):
        prior = Normal("a", 0.0, 1.0) * Normal("b", 0.0, 1.0)
        chain = jnp.stack([self._A, self._B], axis=1)
        posterior = posterior_of([chain], weights=self._WEIGHTS, event_spec=prior.event_spec)
        view = posterior["a"]
        assert float(probpipe.mean(view)) == pytest.approx(0.6)
        assert float(probpipe.variance(view)) == pytest.approx(1.04)

    def test_a_weighted_record_empirical_law(self):
        view = EmpiricalDistribution(self._record_atoms(), self._WEIGHTS, label="d")["a"]
        assert float(probpipe.mean(view)) == pytest.approx(0.6)
        assert float(probpipe.variance(view)) == pytest.approx(1.04)

    def test_a_weighted_record_kde(self):
        # A KDE's mean is its atoms' weighted mean, whatever the bandwidth.
        law = KDEDistribution(self._record_atoms(), weights=self._WEIGHTS, label="kde")
        assert float(probpipe.mean(law["a"])) == pytest.approx(0.6)


def _with_fixed_paths(term: Any, *paths: str) -> Any:
    """*term* holding *paths* fixed, as conditioning on them records."""
    term._store_expression(term._expression.with_fixed(paths))
    return term


class TestNotation:
    """A view reads as the detached marginal at its path: its label followed by its components."""

    def test_a_view_reads_by_its_parents_label_and_its_component(self):
        model = _dependent_joint().with_label("model")
        assert str(model["y"]) == model["y"].notation == "model(y)"

    def test_a_view_of_a_nested_path_reads_by_the_final_segment(self):
        assert FieldView(_Law("parent", _EVENT), "model/theta").notation == "parent(theta)"

    def test_a_selection_reads_by_its_components_in_order(self):
        view = FieldView(_Law("parent", _EVENT), ("y", "model/theta"))
        assert view.notation == "parent(y, theta)"

    def test_a_view_of_an_unlabeled_product_groups_its_label(self):
        assert _dependent_joint()["y"].notation == "(likelihood·beta)(y)"

    def test_a_view_keeps_the_paths_its_parent_holds_fixed(self):
        parent = _with_fixed_paths(_Law("model", _EVENT), "obs")
        view = parent["y"]
        assert _fixed_paths(view) == ("obs",)
        assert view.notation == "model(y; obs)"

    @pytest.mark.parametrize(
        "derive",
        [
            pytest.param(lambda view: view["model/theta/mu"], id="sub-view"),
            pytest.param(lambda view: view.with_label("other"), id="with_label"),
            pytest.param(lambda view: view.with_dim_names(n="m"), id="with_dim_names"),
        ],
    )
    def test_a_view_derived_from_a_view_keeps_the_fixed_paths(self, derive):
        parent = _with_fixed_paths(_Law("model", _EVENT), "obs")
        assert _fixed_paths(derive(parent["model"])) == ("obs",)

    def test_the_detached_marginal_keeps_the_fixed_paths(self):
        joint = Normal("a", 0.0, 1.0, label="a") * EmpiricalDistribution(
            jnp.array([0.0, 1.0, 3.0]), component="b", label="b"
        )
        parent = _with_fixed_paths(joint.with_label("model"), "obs")
        detached = parent["b"].raw()
        assert _fixed_paths(detached) == ("obs",)
        assert detached.notation == "b(b; obs)"

    def test_a_view_at_a_whole_factor_displays_as_the_factor(self):
        model = (
            _Kernel(
                {"mu": _REAL},
                OutputSpec(y=_REAL),
                label="lik",
            )
            * _prior()
        ).with_label("model")
        view, detached = model["mu"], marginal(model, "mu")
        assert (view.label, view.notation) == (detached.label, detached.notation)
        assert (view.label, view.notation) == ("prior", "prior(mu)")

    @pytest.mark.parametrize(
        ("paths", "label", "notation"),
        [
            pytest.param(("a", "b"), "a·b", "a(a)·b(b)", id="declaration-order"),
            pytest.param(("b", "a"), "b·a", "b(b)·a(a)", id="selection-order"),
        ],
    )
    def test_a_view_of_several_whole_factors_displays_factor_by_factor(
        self, paths, label, notation
    ):
        model = (
            Normal("a", 0.0, 1.0, label="a")
            * Normal("b", 0.0, 1.0, label="b")
            * Normal("c", 0.0, 1.0, label="c")
        ).with_label("model")
        view, detached = model[paths], marginal(model, paths)
        assert (view.label, view.notation) == (detached.label, detached.notation)
        assert (view.label, view.notation) == (label, notation)

    def test_a_selection_no_product_can_order_displays_by_its_joined_label(self):
        """``lik`` conditions on ``mu``, so no product lists ``prior`` first."""
        model = (
            _Kernel(
                {"mu": _REAL},
                OutputSpec(y=_REAL),
                label="lik",
            )
            * _prior()
        ).with_label("model")
        view, detached = model[("mu", "y")], marginal(model, ("mu", "y"))
        assert (view.label, view.notation) == (detached.label, detached.notation)
        assert view.notation == "(lik·prior)(mu, y)"

    def test_a_view_at_part_of_a_factor_keeps_its_parents_label(self):
        atoms = NumericRecordBatch(
            {"u": jnp.array([0.0, 1.0]), "v": jnp.array([2.0, 3.0])},
            "obs",
            element_spec=NumericRecordSpec(u=(), v=()),
            label="atoms",
        )
        model = (
            EmpiricalDistribution(atoms, label="pair") * Normal("a", 0.0, 1.0, label="a")
        ).with_label("model")
        view, detached = model["u"], marginal(model, "u")
        assert (view.label, view.notation) == (detached.label, detached.notation)
        assert view.notation == "model(u)"

    def test_a_view_of_a_view_displays_as_the_marginal_of_the_view(self):
        model = (
            Normal("a", 0.0, 1.0, label="a")
            * Normal("b", 0.0, 1.0, label="b")
            * Normal("c", 0.0, 1.0, label="c")
        ).with_label("model")
        selection = model[("b", "a")]
        for key, notation in [("a", "a(a)"), (("a", "b"), "a(a)·b(b)")]:
            view, detached = selection[key], marginal(selection, key)
            assert (view.label, view.notation) == (detached.label, detached.notation)
            assert view.notation == notation == model[key].notation

    def test_a_sub_view_off_the_factors_keeps_the_label_of_the_view(self):
        parent = _Law("parent", _EVENT)
        view = parent["model"].with_label("theta")
        assert view["model/theta/mu"].notation == "theta(mu)"

    def test_a_relabeled_view_of_several_factors_displays_by_its_label(self):
        model = Normal("a", 0.0, 1.0, label="a") * Normal("b", 0.0, 1.0, label="b")
        view = model[("a", "b")].with_label("pair")
        assert view.notation == "pair(a, b)"
        assert marginal(view, ("a", "b")).notation == "pair(a, b)"

    def test_a_view_of_several_factors_holding_fixed_paths_lists_them(self):
        model = _with_fixed_paths(
            (Normal("a", 0.0, 1.0, label="a") * Normal("b", 0.0, 1.0, label="b")), "obs"
        )
        view, detached = model[("a", "b")], marginal(model, ("a", "b"))
        assert view.notation == detached.notation == "(a·b)(a, b; obs)"

    def test_a_view_at_a_factor_holding_fixed_paths_holds_them_first(self):
        factor = _with_fixed_paths(_prior(), "obs")
        model = _with_fixed_paths(
            (
                _Kernel(
                    {"mu": _REAL},
                    OutputSpec(y=_REAL),
                    label="lik",
                )
                * factor
            ).with_label("model"),
            "z",
        )
        view, detached = model["mu"], marginal(model, "mu")
        assert _fixed_paths(view) == _fixed_paths(detached) == ("obs", "z")
        assert view.notation == detached.notation == "prior(mu; obs, z)"

    def test_a_view_and_the_marginal_of_a_law_without_factors_share_a_notation(self):
        atoms = NumericRecordBatch(
            {"y": jnp.array([0.0, 1.0]), "mu": jnp.array([2.0, 3.0])},
            "obs",
            element_spec=NumericRecordSpec(y=(), mu=()),
            label="atoms",
        )
        model = EmpiricalDistribution(atoms, label="model")
        assert model["y"].notation == marginal(model, "y").notation == "model(y)"


def _prior() -> Distribution:
    """``prior``, a standard normal law over ``mu``."""
    return Normal("mu", 0.0, 1.0, label="prior")


class TestTheRawFormOfAView:
    """A view's ``raw()`` is the raw form of its parent's marginal at the path."""

    def test_a_view_at_a_tfp_factor_gives_the_tfp_distribution(self):
        model = (
            _Kernel(
                {"mu": _REAL},
                OutputSpec(y=_REAL),
                label="lik",
            )
            * _prior()
        ).with_label("model")
        raw = model["mu"].raw()
        assert isinstance(raw, tfd.Normal)
        assert (float(raw.loc), float(raw.scale)) == (0.0, 1.0)

    def test_a_view_of_a_gaussian_joint_gives_the_tfp_marginal_of_the_factor(self):
        raw = _joint_gaussian()["y"].raw()
        assert isinstance(raw, tfd.MultivariateNormalLinearOperator)
        assert jnp.allclose(raw.mean(), _MEAN[1:])

    def test_a_view_of_a_law_without_a_backend_gives_the_detached_marginal(self):
        joint = Normal("a", 0.0, 1.0, label="a") * EmpiricalDistribution(
            jnp.array([0.0, 1.0, 3.0]), component="b", label="b"
        )
        view = joint.with_label("model")["b"]
        raw = view.raw()
        assert isinstance(raw, EmpiricalDistribution) and not isinstance(raw, FieldView)
        assert (raw.notation, raw.provenance) == ("b(b)", None)
        assert raw.notation == marginal(joint.with_label("model"), "b").notation


class TestSelections:
    """A selection's capabilities read the parent at every selected node, in order."""

    _PATHS = ("y", "model/theta")

    def test_a_selection_co_samples_the_mapping_of_its_nodes(self, key):
        parent = _NumericLaw("parent")
        draw = parent._sample(key)
        selected = FieldView(parent, self._PATHS)._sample(key)
        assert isinstance(selected, dict) and isinstance(selected["theta"], dict)
        assert list(selected) == ["y", "theta"]
        assert jnp.array_equal(selected["y"], draw["y"])
        assert jnp.array_equal(selected["theta"]["tau"], draw["model/theta/tau"])

    def test_the_declaration_admits_a_selection_draw(self, key):
        selection = FieldView(_NumericLaw("parent"), self._PATHS)
        assert selection.event_spec.spec.is_valid(selection._sample(key))

    def test_a_batched_selection_draw_is_the_mapping_of_the_parent_columns(self, key):
        parent = _NumericLaw("parent")
        draws = FieldView(parent, self._PATHS)._sample(key, (4,))
        parent_draws = parent._sample(key, (4,))
        assert isinstance(draws, dict)
        assert list(draws) == ["y", "theta"] and list(draws["theta"]) == ["mu", "tau"]
        assert jnp.array_equal(draws["theta"]["mu"], parent_draws["model/theta/mu"])
        assert jnp.array_equal(draws["y"], parent_draws["y"])

    def test_the_moments_of_a_selection_are_mappings_of_the_parent_moments(self):
        selection = FieldView(_NumericLaw("parent"), self._PATHS)
        mean, variance = selection._mean(), selection._variance()
        assert isinstance(mean, dict) and isinstance(variance, dict)
        assert jnp.allclose(mean["y"], jnp.array([3.0, 4.0, 5.0]))
        assert jnp.allclose(mean["theta"]["tau"], jnp.array([1.0, 2.0]))
        assert jnp.allclose(variance["theta"]["mu"], _FLAT_COV[0, 0])

    def test_the_covariance_of_a_selection_keeps_the_selection_order(self):
        cov = FieldView(_NumericLaw("parent"), ("y", "model/theta/mu"))._cov()
        index = jnp.array([3, 4, 5, 0])
        assert cov.shape == (4, 4)
        assert jnp.allclose(cov.to_dense(), _FLAT_COV[index][:, index])

    def test_the_quantiles_of_a_selection_are_the_mapping_of_its_nodes_in_order(self):
        levels = jnp.array([0.25, 0.5])
        quantiles = FieldView(_NumericLaw("parent"), ("y", "model/theta/mu"))._quantile(levels)
        assert list(quantiles) == ["y", "mu"]
        assert jnp.allclose(quantiles["y"], levels[:, None] + jnp.array([3.0, 4.0, 5.0]))
        assert jnp.allclose(quantiles["mu"], levels)

    def test_the_expectation_of_a_selection_integrates_the_record_of_its_nodes(self):
        selection = FieldView(_FiniteLaw("parent"), ("b", "a"))
        expectation = selection._expectation(lambda record: record["b"] - record["a"])
        assert jnp.allclose(expectation, 0.25 * (1.0 - 0.0) + 0.75 * (3.0 - 1.0))

    def test_the_density_of_a_selection_is_the_parent_marginal_density_at_its_paths(self):
        parent = _UnguardedMarginalLaw("parent", OutputSpec(RecordSpec(a=_REAL, b=_REAL)))
        density = FieldView(parent, ("b", "a"))._log_prob({"b": 0.5, "a": -1.0})
        standard = Normal("x", 0.0, 1.0)
        assert parent.marginal_calls == [("b", "a")]
        assert jnp.allclose(density, standard._log_prob(0.5) + standard._log_prob(-1.0))

    def test_the_marginal_at_a_selection_path_is_the_parent_marginal_at_the_node_path(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        marginal = FieldView(parent, self._PATHS)._marginal("theta/mu")
        assert parent.marginal_calls == ["model/theta/mu"]
        assert list(marginal.event_spec.components) == ["mu"]

    def test_the_marginal_at_several_selection_paths_is_the_parent_marginal_at_theirs(self):
        parent = _UnguardedMarginalLaw("parent", _EVENT)
        marginal = FieldView(parent, self._PATHS)._marginal(("theta/mu", "y"))
        assert parent.marginal_calls == [("model/theta/mu", "y")]
        assert list(marginal.event_spec.components) == ["mu", "y"]

    def test_conditioning_a_selection_drops_a_node_the_given_covers(self):
        parent = _ConditioningLaw("parent", _EVENT)
        conditioned = FieldView(parent, ("model/theta/tau", "y"))._condition_on(
            {"tau": jnp.zeros(2)}
        )
        assert [set(given) for given in parent.given_calls] == [{"model/theta/tau"}]
        assert isinstance(conditioned, FieldView)
        assert conditioned.path == ("y",)
        assert conditioned.event_spec == OutputSpec(RecordSpec(y=(3,)))

    def test_a_selection_round_trips_through_pickle(self):
        selection = FieldView(_product(), ("b", "a"))
        restored = pickle.loads(pickle.dumps(selection))
        assert type(restored) is type(selection)
        assert (restored.label, restored.path, restored.spec) == (
            selection.label,
            selection.path,
            selection.spec,
        )


_ROUND_TRIP_PARENTS = [
    pytest.param(_product, "a", id="product"),
    pytest.param(lambda: _MarginalLaw("parent", _EVENT), "model/theta", id="marginals"),
    pytest.param(lambda: _Law("parent", _EVENT), "y", id="no-capability"),
]


class TestCapabilityClasses:
    def test_a_view_class_is_the_field_view_subclass_for_its_capabilities(self):
        view = FieldView(_parent(SupportsMean, SupportsVariance), "y")
        assert type(view) is _capability_subclass(FieldView, [SupportsVariance, SupportsMean])

    def test_views_with_one_capability_set_share_one_class(self):
        first = FieldView(_parent(SupportsMean, SupportsMarginals), "y")
        second = FieldView(
            _parent(SupportsMean, SupportsMarginals, event_spec=_WHOLE), "parameters/beta"
        )
        assert type(first) is type(second)

    def test_views_with_different_capability_sets_have_different_classes(self):
        classes = {type(FieldView(_parent(capability), "y")) for capability in _ROWS}
        assert len(classes) == len(_ROWS)

    def test_a_view_class_keeps_the_name_of_field_view(self):
        view_class = type(FieldView(_parent(SupportsSampling), "y"))
        assert view_class is not FieldView
        assert issubclass(view_class, FieldView)
        assert (view_class.__name__, view_class.__qualname__, view_class.__module__) == (
            "FieldView",
            "FieldView",
            FieldView.__module__,
        )

    @pytest.mark.parametrize(("make", "path"), _ROUND_TRIP_PARENTS)
    def test_pickle_restores_the_view_in_its_capability_class(self, make, path):
        view = FieldView(make(), path)
        restored = pickle.loads(pickle.dumps(view))
        assert type(restored) is type(view)
        assert (restored.label, restored.path, restored.spec) == (view.label, view.path, view.spec)
        assert restored.parent.spec == view.parent.spec
        assert _claimed(restored) == _claimed(view)

    @pytest.mark.parametrize(("make", "path"), _ROUND_TRIP_PARENTS)
    def test_copy_restores_the_view_sharing_its_parent(self, make, path):
        view = FieldView(make(), path)
        restored = copy.copy(view)
        assert type(restored) is type(view)
        assert restored.parent is view.parent
        assert (restored.label, restored.path, restored.spec) == (view.label, view.path, view.spec)

    @pytest.mark.parametrize(("make", "path"), _ROUND_TRIP_PARENTS)
    def test_deepcopy_restores_the_view_with_a_copy_of_its_parent(self, make, path):
        view = FieldView(make(), path)
        restored = copy.deepcopy(view)
        assert type(restored) is type(view)
        assert restored.parent is not view.parent
        assert restored.parent.spec == view.parent.spec
        assert (restored.label, restored.path, restored.spec) == (view.label, view.path, view.spec)

    @pytest.mark.parametrize(("make", "path"), _ROUND_TRIP_PARENTS)
    def test_a_rename_keeps_the_view_in_its_capability_class(self, make, path):
        view = FieldView(make(), path)
        renamed = view.with_label("renamed")
        assert type(renamed) is type(view)
        assert renamed.parent is view.parent

    def test_pickle_round_trips_in_a_fresh_process(self, tmp_path):
        view = FieldView(_product(), "a")
        pickled = tmp_path / "view.pkl"
        pickled.write_bytes(pickle.dumps(view))
        script = textwrap.dedent(
            """
            import json
            import pickle
            import sys

            from probpipe.distributions import FieldView, _capabilities

            with open(sys.argv[1], "rb") as source:
                view = pickle.load(source)
            claimed = sorted(
                name for name in json.loads(sys.argv[2])
                if isinstance(view, getattr(_capabilities, name))
            )
            print(json.dumps(
                [type(view).__name__, type(view) is FieldView, view.label, view.path, claimed]
            ))
            """
        )
        root = str(Path(probpipe.__file__).resolve().parents[1])
        search_path = os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
        names = json.dumps([protocol.__name__ for protocol in _CAPABILITIES])
        result = subprocess.run(
            [sys.executable, "-c", script, str(pickled), names],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": search_path},
        )
        assert result.returncode == 0, result.stderr
        restored = json.loads(result.stdout.strip().splitlines()[-1])
        claimed = sorted(protocol.__name__ for protocol in _claimed(view))
        assert restored == ["FieldView", False, view.label, view.path, claimed]
