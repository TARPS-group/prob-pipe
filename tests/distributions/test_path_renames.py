"""Renaming event paths and given slots keeps the law and renames its values.

``with_path_names`` on a law whose rename reaches a field of a record draw
returns a law that holds the original and renames at its boundary: its draws,
moments, and marginals carry the new names, and a scored value, a given, or a
path under the new names reaches the original under the old ones. Renaming a
whole term's component alone changes only the declaration. On a kernel, a
given-side target is the node's new path, so a rename may group slots or split
a field out of a structured slot; binding the renamed kernel translates the
given to the original slots and renames what the binding returns.
"""

from __future__ import annotations

import pickle
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from probpipe import (
    MultivariateNormal,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    OutputSpec,
    Record,
    RecordBatch,
    RecordSpec,
)
from probpipe.core._dispatch import Feasibility
from probpipe.distributions import (
    ConditionalDistribution,
    Distribution,
    FactoredConditionalDistribution,
)
from probpipe.distributions._capabilities import (
    SupportsConditionalLogProb,
    SupportsConditionalMean,
    SupportsConditionalSampling,
    SupportsCovariance,
    SupportsExactConditioning,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMarginals,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
    _capability_guard,
)
from probpipe.distributions._empirical import EmpiricalDistribution
from probpipe.linalg import DenseLinOp, LinOp

_REAL = Normal("x", 0.0, 1.0).event_spec.spec
_SCALAR = NumericArraySpec(())
_MEAN = jnp.array([1.0, 2.0, 3.0])
_COV = jnp.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])

#: A nested exposed record: a group two levels deep and a vector beside it.
_NESTED = OutputSpec(RecordSpec(model=RecordSpec(theta=RecordSpec(mu=_REAL, tau=(2,))), y=(3,)))


# -- Test doubles -------------------------------------------------------------


class _Law(Distribution):
    """A law over a declared event that claims no capability."""

    def __init__(self, name: str, event_spec: OutputSpec) -> None:
        super().__init__(name, event_spec)


class _NestedLaw(
    _Law, SupportsSampling, SupportsMean, SupportsMarginals, SupportsExactConditioning
):
    """A law over the nested event that records its marginal paths and givens.

    A draw is a standard normal flat vector split into ``(mu, tau, y)``, the
    mean is ``(0, ..., 5)`` so split, the marginal at a path is a standard
    normal named by the path's final segment, and conditioning returns the law
    of ``model/theta/mu`` and ``y``.
    """

    def __init__(self, name: str = "parent") -> None:
        super().__init__(name, _NESTED)
        self.marginal_calls: list[Any] = []
        self.given_calls: list[dict[str, Any]] = []

    def _record(self, flat: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        fields = {
            "model": {"theta": {"mu": flat[..., 0], "tau": flat[..., 1:3]}},
            "y": flat[..., 3:],
        }
        if sample_shape:
            return RecordBatch(
                self.name,
                fields,
                "sample",
                element_spec=self.event_spec.spec,
                axes_per_level=(len(sample_shape),),
            )
        return Record(self.name, fields, event_template=self.event_spec.spec)

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self._record(jax.random.normal(key, (*sample_shape, 6)), sample_shape)

    def _mean(self) -> Any:
        return self._record(jnp.arange(6.0))

    def _marginal(self, path: str | tuple[str, ...]) -> Distribution:
        self.marginal_calls.append(path)
        component = path.rsplit("/", 1)[-1]
        if component == "theta":
            return _Law("theta", OutputSpec(theta=RecordSpec(mu=_REAL, tau=(2,))))
        return Normal(component, 0.0, 1.0)

    def _marginal_guard(self, path: str | tuple[str, ...]) -> Feasibility:
        if path == "y":
            return Feasibility(False, "no closed form at y")
        return Feasibility(True)

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Distribution:
        self.given_calls.append(dict(given))
        return _Law(
            "conditioned",
            OutputSpec(RecordSpec(model=RecordSpec(theta=RecordSpec(mu=_REAL)), y=(3,))),
        )


class _FiniteLaw(_Law, SupportsExpectation):
    """A law over ``a`` and ``b`` with the atoms (0, 1) of weight 1/4 and (1, 3) of weight 3/4."""

    def __init__(self, name: str = "parent") -> None:
        super().__init__(name, OutputSpec(RecordSpec(a=_REAL, b=_REAL)))

    def _expectation(self, f: Any) -> Any:
        atoms = ((0.0, 1.0, 0.25), (1.0, 3.0, 0.75))
        return sum(
            weight * f(Record("atom", a=jnp.asarray(a), b=jnp.asarray(b))) for a, b, weight in atoms
        )


class _GuardedMeanLaw(_Law, SupportsMean):
    """A law over the nested event whose mean guard rejects."""

    REJECTED = Feasibility(False, "the mean does not exist")

    def __init__(self, name: str = "parent") -> None:
        super().__init__(name, _NESTED)

    def _mean(self) -> Any:
        raise AssertionError("the guard rejects the mean")

    def _mean_guard(self) -> Feasibility:
        return self.REJECTED


class _WholeRecordLaw(_Law, SupportsSampling):
    """The whole record ``parameters`` of a parent's draws."""

    def __init__(self, parent: Distribution) -> None:
        super().__init__("whole", OutputSpec(parameters=parent.event_spec.spec))
        self.parent = parent

    def _sample(self, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        return self.parent._sample(key, sample_shape)


class _CovLaw(_Law, SupportsCovariance):
    """A law over ``a`` and ``b`` whose covariance operator is one stored object."""

    def __init__(self) -> None:
        super().__init__("parent", OutputSpec(RecordSpec(a=(1,), b=(2,))))
        self.cov = DenseLinOp(_COV)

    def _cov(self) -> LinOp:
        return self.cov


class _QuantileLaw(_Law, SupportsQuantile):
    """A law over ``a`` and ``b`` whose quantile at ``q`` is ``q`` plus each coordinate's index.

    The quantiles take the event's raw form, a mapping of per-field arrays with
    the level axes first, or with *flat* the flat coordinates after the levels.
    """

    def __init__(self, *, flat: bool = False) -> None:
        super().__init__("parent", OutputSpec(RecordSpec(a=(1,), b=(2,))))
        self.flat = flat

    def _quantile(self, q: Any) -> Any:
        coordinates = jnp.asarray(q)[..., None] + jnp.arange(3.0)
        if self.flat:
            return coordinates
        return {"a": coordinates[..., :1], "b": coordinates[..., 1:]}


class _NamedDensityLaw(_Law, SupportsLogProb):
    """A law over ``u`` and ``y`` whose density reads each field by its name."""

    def __init__(self) -> None:
        super().__init__("parent", OutputSpec(RecordSpec(u=(1,), y=(2,))))

    def _log_prob(self, value: Any) -> Any:
        u, y = jnp.asarray(value["u"]), jnp.asarray(value["y"])
        return -jnp.sum((u - 1.0) ** 2) - 2.0 * jnp.sum((y - jnp.array([3.0, 4.0])) ** 2)


class _RecordingKernel(ConditionalDistribution):
    """A kernel that records each given it binds and curries over the slots left.

    Binding every slot returns a law over the kernel's event whose name is the
    bound values, in slot order; binding some returns the kernel over the rest,
    which keeps the values bound so far.
    """

    def __init__(
        self,
        name: str,
        given_spec: Any,
        event_spec: OutputSpec,
        *,
        calls: list[dict[str, Any]] | None = None,
        bound: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, given_spec, event_spec)
        self.calls = [] if calls is None else calls
        self.bound = dict(bound or {})

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        given = dict(given)
        self.calls.append(given)
        bound = {**self.bound, **given}
        remaining = {slot: spec for slot, spec in self.given_spec.items() if slot not in given}
        if remaining:
            return _RecordingKernel(
                self.name, remaining, self.event_spec, calls=self.calls, bound=bound
            )
        return _Law(self.name, self.event_spec)


class _MeanKernel(ConditionalDistribution, SupportsConditionalSampling, SupportsConditionalMean):
    """``y | mu``: a record ``(y, z)`` whose mean is ``(mu, 0)``, and a draw ``(mu + e, e)``."""

    def __init__(self, name: str = "k") -> None:
        super().__init__(name, {"mu": _SCALAR}, OutputSpec(RecordSpec(y=_SCALAR, z=_SCALAR)))

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        return _Law(self.name, self.event_spec)

    def _conditional_mean(self, given: Any) -> Any:
        return Record("mean", y=jnp.asarray(given["mu"]), z=jnp.asarray(0.0))

    def _conditional_mean_guard(self) -> Feasibility:
        return Feasibility(True)

    def _conditional_sample(self, given: Any, key: Any, sample_shape: tuple[int, ...] = ()) -> Any:
        noise = jax.random.normal(key, sample_shape)
        return Record("draw", y=given["mu"] + noise, z=noise)


class _ScoreKernel(ConditionalDistribution, SupportsConditionalLogProb):
    """``(y, z) | mu`` whose density reads the given and each field by name."""

    def __init__(self, name: str = "k") -> None:
        super().__init__(name, {"mu": _SCALAR}, OutputSpec(RecordSpec(y=_SCALAR, z=_SCALAR)))

    def _condition_on(self, given: Any, /, **kwargs: Any) -> Any:
        return _Law(self.name, self.event_spec)

    def _conditional_log_prob(self, given: Any, value: Any) -> Any:
        return -((value["y"] - given["mu"]) ** 2) - 2.0 * value["z"] ** 2


def _product() -> Distribution:
    return Normal("a", 0.0, 1.0) * Normal("b", 2.0, 3.0)


#: Three record atoms over ``a/x`` and ``b/y``.
_GROUPED = RecordSpec(a=RecordSpec(x=_SCALAR), b=RecordSpec(y=_SCALAR))
_XS, _YS = jnp.array([1.0, 2.0, 4.0]), jnp.array([10.0, 20.0, 40.0])


def _grouped_law() -> EmpiricalDistribution:
    atoms = NumericRecordBatch("rows", {"a/x": _XS, "b/y": _YS}, "row", element_spec=_GROUPED)
    return EmpiricalDistribution("grouped", atoms)


def _whole_record_law() -> EmpiricalDistribution:
    """The atoms of ``beta`` and ``sigma`` as a whole record under the component ``parameters``."""
    spec = RecordSpec(beta=_SCALAR, sigma=_SCALAR)
    atoms = NumericRecordBatch("rows", {"beta": _XS, "sigma": _YS}, "row", element_spec=spec)
    return EmpiricalDistribution("p", atoms, event_spec=OutputSpec(parameters=None))


# -- The renamed law ------------------------------------------------------------


class TestRenamedLawDeclaration:
    def test_renaming_a_whole_term_component_keeps_the_class(self):
        law = Normal("x", 0.0, 1.0)
        renamed = law.with_path_names(x="location")
        assert type(renamed) is type(law)
        assert list(renamed.event_spec.components) == ["location"]

    def test_renaming_a_field_keeps_the_name_and_renames_the_declaration(self):
        parent = _product()
        renamed = parent.with_path_names(a="x")
        assert renamed.name == parent.name
        assert renamed.event_spec == parent.event_spec.with_path_names(a="x")
        assert renamed.provenance is not None
        assert renamed.provenance.operation == "with_path_names"
        assert [info.name for info in renamed.provenance.parents] == [parent.name]

    def test_a_renamed_law_claims_the_capabilities_of_its_parent(self):
        parent = MultivariateNormal("x", _MEAN[:1], cov=_COV[:1, :1]) * MultivariateNormal(
            "y", _MEAN[1:], cov=_COV[1:, 1:]
        )
        renamed = parent.with_path_names(y="w")
        claims = (
            SupportsSampling,
            SupportsLogProb,
            SupportsMean,
            SupportsVariance,
            SupportsCovariance,
        )
        assert all(isinstance(renamed, protocol) for protocol in claims)
        assert not isinstance(_Law("p", _NESTED).with_path_names(y="w"), SupportsSampling)

    def test_a_renamed_law_round_trips_through_pickle(self):
        renamed = _product().with_path_names(a="x")
        restored = pickle.loads(pickle.dumps(renamed))
        assert type(restored) is type(renamed)
        assert (restored.name, restored.spec) == (renamed.name, renamed.spec)


class TestRenamedLawValues:
    def test_a_draw_carries_the_new_names(self, key):
        parent = _NestedLaw()
        renamed = parent.with_path_names({"model/theta/mu": "model/theta/m", "y": "obs"})
        draw = renamed._sample(key)
        assert list(draw.keys()) == ["model/theta/m", "model/theta/tau", "obs"]
        assert jnp.array_equal(draw["model/theta/m"], parent._sample(key)["model/theta/mu"])
        assert renamed.event_spec.spec.is_valid(draw)

    def test_a_batch_of_draws_carries_the_new_names(self, key):
        parent = _NestedLaw()
        draws = parent.with_path_names(y="obs")._sample(key, (4,))
        assert isinstance(draws, RecordBatch)
        assert jnp.array_equal(draws["obs"], parent._sample(key, (4,))["y"])

    def test_a_field_of_a_whole_record_term_is_renamed_in_its_draws(self, key):
        parent = Normal("beta", 0.0, 1.0) * Normal("sigma", 1.0, 1.0)
        whole = _WholeRecordLaw(parent)
        renamed = whole.with_path_names({"parameters/beta": "parameters/b"}).with_path_names(
            parameters="theta"
        )
        assert renamed.event_spec == OutputSpec(theta=RecordSpec(b=_REAL, sigma=_REAL))
        draw = renamed._sample(key)
        assert list(draw.keys()) == ["b", "sigma"]

    def test_the_mean_carries_the_new_names(self):
        mean = _NestedLaw().with_path_names({"model/theta": "model/coefficients"})._mean()
        assert jnp.allclose(mean["model/coefficients/tau"], jnp.array([1.0, 2.0]))

    def test_the_covariance_is_the_parent_covariance(self):
        parent = _CovLaw()
        assert parent.with_path_names(a="x")._cov() is parent.cov

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param({"x": jnp.array([0.5]), "y": jnp.array([1.0, 2.0])}, id="mapping"),
            pytest.param(Record("v", x=jnp.array([0.5]), y=jnp.array([1.0, 2.0])), id="record"),
        ],
    )
    def test_the_density_scores_the_value_under_the_original_names(self, value):
        parent = _NamedDensityLaw()
        renamed = parent.with_path_names(u="x")
        original = {"u": jnp.array([0.5]), "y": jnp.array([1.0, 2.0])}
        assert jnp.allclose(renamed._log_prob(value), parent._log_prob(original))
        assert jnp.allclose(renamed._unnormalized_log_prob(value), parent._log_prob(original))

    def test_the_expectation_integrates_the_renamed_value(self):
        renamed = _FiniteLaw().with_path_names(a="x")
        expectation = renamed._expectation(lambda record: record["b"] - record["x"])
        assert jnp.allclose(expectation, 0.25 * (1.0 - 0.0) + 0.75 * (3.0 - 1.0))


class TestRenamedLawPaths:
    def test_the_marginal_at_a_renamed_path_is_the_parent_marginal_at_the_original(self):
        parent = _NestedLaw()
        renamed = parent.with_path_names({"model/theta/mu": "model/theta/m"})
        marginal = renamed._marginal("model/theta/m")
        assert parent.marginal_calls == ["model/theta/mu"]
        assert list(marginal.event_spec.components) == ["m"]

    def test_the_marginal_of_a_group_takes_the_renames_below_it(self):
        parent = _NestedLaw()
        renamed = parent.with_path_names({"model/theta/mu": "model/theta/m"}).with_path_names(
            {"model/theta": "model/coefficients"}
        )
        marginal = renamed._marginal("model/coefficients")
        assert parent.marginal_calls == ["model/theta"]
        assert marginal.event_spec == OutputSpec(coefficients=RecordSpec(m=_REAL, tau=(2,)))

    def test_the_marginal_guard_is_the_parent_guard_at_the_original_path(self):
        renamed = _NestedLaw().with_path_names(y="obs")
        assert _capability_guard(renamed, "_marginal", "obs") == Feasibility(
            False, "no closed form at y"
        )
        assert _capability_guard(renamed, "_marginal", "model") == Feasibility(True)
        assert _capability_guard(renamed, "_marginal", "y").feasible is False

    def test_the_marginal_at_a_path_that_is_not_an_event_path_raises_key_error(self):
        with pytest.raises(KeyError):
            _NestedLaw().with_path_names(y="obs")._marginal("y")

    def test_conditioning_reaches_the_parent_under_the_original_names(self):
        parent = _NestedLaw()
        renamed = parent.with_path_names({"model/theta/tau": "model/theta/t", "y": "obs"})
        conditioned = renamed._condition_on({"model/theta/t": jnp.zeros(2)})
        assert [list(given) for given in parent.given_calls] == [["model/theta/tau"]]
        assert conditioned.event_spec == OutputSpec(
            RecordSpec(model=RecordSpec(theta=RecordSpec(mu=_REAL)), obs=(3,))
        )

    def test_a_given_record_is_renamed_below_its_path(self):
        parent = _NestedLaw()
        renamed = parent.with_path_names({"model/theta/tau": "model/theta/t"})
        renamed._condition_on({"model/theta": {"t": jnp.zeros(2), "mu": jnp.asarray(0.0)}})
        assert list(parent.given_calls[0]["model/theta"]) == ["mu", "tau"]

    def test_the_conditioning_guard_translates_the_paths(self):
        renamed = _NestedLaw().with_path_names(y="obs")
        assert _capability_guard(renamed, "_condition_on", ("obs",)) == Feasibility(True)
        assert _capability_guard(renamed, "_condition_on", ("y",)).feasible is False

    def test_a_projected_capability_carries_the_parent_guard(self):
        renamed = _GuardedMeanLaw().with_path_names(y="obs")
        assert _capability_guard(renamed, "_mean") == _GuardedMeanLaw.REJECTED

    def test_the_marginal_guard_rejects_paths_whose_final_segments_collide(self):
        renamed = _grouped_law().with_path_names({"b/y": "b/x"})
        report = _capability_guard(renamed, "_marginal", ("a/x", "b/x"))
        assert report.feasible is False
        assert "final segment" in report.description
        with pytest.raises(ValueError, match="final segment"):
            renamed._marginal(("a/x", "b/x"))


class TestRenamedLawMoves:
    """A target is the node's new exact path, so a rename may move fields of the draws."""

    def test_a_moved_field_is_appended_to_its_new_parent_in_the_draws(self, key):
        parent = _NestedLaw()
        renamed = parent.with_path_names({"model/theta/mu": "mu", "y": "model/y"})
        draw = renamed._sample(key)
        assert list(draw.keys()) == ["model/theta/tau", "model/y", "mu"]
        original = parent._sample(key)
        assert jnp.array_equal(draw["mu"], original["model/theta/mu"])
        assert jnp.array_equal(draw["model/y"], original["y"])
        assert renamed.event_spec.spec.is_valid(draw)

    def test_a_batch_of_draws_moves_its_columns(self, key):
        parent = _NestedLaw()
        draws = parent.with_path_names({"y": "model/y"})._sample(key, (4,))
        assert tuple(draws.element_spec.keys()) == ("model/theta/mu", "model/theta/tau", "model/y")
        assert jnp.array_equal(draws["model/y"], parent._sample(key, (4,))["y"])

    def test_the_mean_moves_with_its_fields(self):
        mean = _NestedLaw().with_path_names({"y": "model/y"})._mean()
        assert list(mean.keys()) == ["model/theta/mu", "model/theta/tau", "model/y"]
        assert jnp.allclose(mean["model/y"], jnp.array([3.0, 4.0, 5.0]))

    def test_the_covariance_follows_the_moved_coordinates(self):
        # ``a`` moves behind ``b``, so the coordinates of a draw are ``(b, a)``.
        cov = _CovLaw().with_path_names({"a": "g/a"})._cov()
        order = jnp.array([1, 2, 0])
        assert jnp.allclose(cov.to_dense(), _COV[order][:, order])

    @pytest.mark.parametrize("flat", [False, True], ids=["raw-form", "flat-coordinates"])
    def test_the_quantiles_follow_the_moved_fields(self, flat):
        q = jnp.array([0.25, 0.75])
        quantiles = _QuantileLaw(flat=flat).with_path_names({"a": "g/a"})._quantile(q)
        of_a, of_b = q[:, None] + 0.0, q[:, None] + jnp.array([1.0, 2.0])
        if flat:
            assert jnp.allclose(quantiles, jnp.concatenate([of_b, of_a], axis=-1))
        else:
            assert list(quantiles) == ["b", "g"]
            assert jnp.allclose(quantiles["b"], of_b)
            assert jnp.allclose(quantiles["g"]["a"], of_a)

    def test_the_quantiles_carry_the_new_names(self):
        quantiles = _QuantileLaw().with_path_names(a="x")._quantile(jnp.array(0.5))
        assert list(quantiles) == ["x", "b"]

    def test_the_density_scores_a_moved_value_under_the_original_paths(self):
        parent = _NamedDensityLaw()
        renamed = parent.with_path_names({"u": "g/u"})
        value = {"y": jnp.array([1.0, 2.0]), "g": {"u": jnp.array([0.5])}}
        original = {"u": jnp.array([0.5]), "y": jnp.array([1.0, 2.0])}
        assert jnp.allclose(renamed._log_prob(value), parent._log_prob(original))

    def test_a_moved_empirical_law_permutes_its_covariance_and_moves_its_quantiles(self):
        parent = _grouped_law()
        renamed = parent.with_path_names({"a/x": "b/x"})
        assert renamed.event_spec == OutputSpec(RecordSpec(b=RecordSpec(y=_SCALAR, x=_SCALAR)))
        order = jnp.array([1, 0])
        assert jnp.allclose(renamed._cov().to_dense(), parent._cov().to_dense()[order][:, order])
        q = jnp.array([0.5])
        moved, original = renamed._quantile(q), parent._quantile(q)
        assert list(moved) == ["b"]
        assert list(moved["b"]) == ["y", "x"]
        assert jnp.allclose(moved["b"]["x"], original["a"]["x"])
        assert jnp.allclose(moved["b"]["y"], original["b"]["y"])

    def test_the_marginal_at_a_moved_field_is_the_parent_marginal_at_its_origin(self):
        parent = _grouped_law()
        renamed = parent.with_path_names({"a/x": "x"})
        marginal = renamed._marginal("x")
        assert list(marginal.event_spec.components) == ["x"]
        assert jnp.allclose(marginal._mean(), jnp.mean(_XS))
        assert renamed._marginal("b").event_spec == OutputSpec(b=RecordSpec(y=_SCALAR))

    def test_the_marginal_at_a_group_that_gathers_several_nodes_is_rejected(self):
        renamed = _grouped_law().with_path_names({"a/x": "g/x", "b/y": "g/y"})
        assert _capability_guard(renamed, "_marginal", "g").feasible is False
        with pytest.raises(ValueError, match="no single node"):
            renamed._marginal("g")
        assert _capability_guard(renamed, "_marginal", "g/x") == Feasibility(True)

    def test_a_view_at_a_regrouping_node_claims_no_density(self, key):
        renamed = (Normal("a", 0.0, 1.0) * Normal("b", 1.0, 1.0)).with_path_names({"a": "g/a"})
        view = renamed["g"]
        assert isinstance(view, SupportsSampling)
        assert not isinstance(view, SupportsLogProb)
        assert jnp.array_equal(view._sample(key)["a"], renamed._sample(key)["g"]["a"])

    def test_a_given_at_a_moved_field_reaches_the_parent_at_its_origin(self):
        parent = _NestedLaw()
        parent.with_path_names({"model/theta/tau": "tau"})._condition_on({"tau": jnp.zeros(2)})
        assert [list(given) for given in parent.given_calls] == [["model/theta/tau"]]

    def test_the_conditioned_law_keeps_the_moves_of_the_fields_that_remain(self):
        parent = _NestedLaw()
        renamed = parent.with_path_names({"y": "model/y"})
        conditioned = renamed._condition_on({"model/theta/tau": jnp.zeros(2)})
        assert conditioned.event_spec == OutputSpec(
            RecordSpec(model=RecordSpec(theta=RecordSpec(mu=_REAL), y=(3,)))
        )


class TestRenamingARenamedLaw:
    """A renamed law applies a further rename to its parent, composed with its own."""

    def test_a_component_rename_after_a_field_rename_keeps_the_field_rename(self, key):
        law = _whole_record_law()
        renamed = law.with_path_names({"parameters/beta": "parameters/b"}).with_path_names(
            parameters="theta"
        )
        assert renamed._parent is law
        assert renamed.event_spec == OutputSpec(theta=RecordSpec(b=_SCALAR, sigma=_SCALAR))
        assert _capability_guard(renamed, "_marginal", "theta/b") == Feasibility(True)
        marginal = renamed._marginal("theta/b")
        assert list(marginal.event_spec.components) == ["b"]
        assert jnp.allclose(marginal._mean(), jnp.mean(_XS))
        assert list(renamed._sample(key).keys()) == ["b", "sigma"]

    def test_a_move_after_a_rename_composes_both(self, key):
        parent = _NestedLaw()
        renamed = parent.with_path_names(y="obs").with_path_names({"obs": "model/obs"})
        assert renamed._parent is parent
        draw = renamed._sample(key)
        assert list(draw.keys()) == ["model/theta/mu", "model/theta/tau", "model/obs"]
        assert jnp.array_equal(draw["model/obs"], parent._sample(key)["y"])
        renamed._marginal("model/obs")
        assert parent.marginal_calls == ["y"]


# -- The renamed kernel ---------------------------------------------------------


def _vector(size: int) -> NumericArraySpec:
    return NumericArraySpec((size,))


class TestRenamedKernelBinding:
    def test_a_grouped_slot_binds_the_original_slots(self):
        kernel = _RecordingKernel("k", {"a": _SCALAR, "b": _vector(2)}, OutputSpec(y=_SCALAR))
        grouped = kernel.with_path_names({"a": "theta/a", "b": "theta/b"})
        law = grouped._condition_on({"theta": {"a": 1.0, "b": jnp.zeros(2)}})
        assert isinstance(law, Distribution)
        assert list(kernel.calls[0]) == ["a", "b"]
        assert kernel.calls[0]["a"] == 1.0

    def test_split_slots_bind_the_original_structured_slot(self):
        kernel = _RecordingKernel(
            "k", {"theta": RecordSpec(a=_SCALAR, b=_vector(2))}, OutputSpec(y=_SCALAR)
        )
        split = kernel.with_path_names({"theta/a": "a"})
        split._condition_on({"a": 1.0, "theta": {"b": jnp.zeros(2)}})
        (given,) = kernel.calls
        assert list(given) == ["theta"]
        assert given["theta"]["a"] == 1.0
        assert jnp.array_equal(given["theta"]["b"], jnp.zeros(2))

    def test_binding_one_split_slot_holds_its_field_until_the_slot_completes(self):
        kernel = _RecordingKernel(
            "k", {"theta": RecordSpec(a=_SCALAR, b=_vector(2))}, OutputSpec(y=_SCALAR)
        )
        curried = kernel.with_path_names({"theta/a": "a"})._condition_on({"a": 1.0})
        assert isinstance(curried, ConditionalDistribution)
        assert list(curried.given_spec) == ["theta"]
        assert curried.given_spec["theta"] == RecordSpec(b=_vector(2))
        assert kernel.calls == []
        law = curried._condition_on({"theta": {"b": jnp.zeros(2)}})
        assert isinstance(law, Distribution)
        assert kernel.calls[0]["theta"]["a"] == 1.0

    def test_binding_some_slots_curries_to_the_renamed_kernel_over_the_rest(self):
        kernel = _RecordingKernel("k", {"mu": _SCALAR, "sigma": _SCALAR}, OutputSpec(y=_SCALAR))
        renamed = kernel.with_path_names(mu="loc", sigma="scale", y="obs")
        curried = renamed._condition_on({"loc": 1.0})
        assert isinstance(curried, ConditionalDistribution)
        assert list(curried.given_spec) == ["scale"]
        assert list(curried.event_spec.components) == ["obs"]
        law = curried._condition_on({"scale": 2.0})
        assert [dict(given) for given in kernel.calls] == [{"mu": 1.0}, {"sigma": 2.0}]
        assert list(law.event_spec.components) == ["obs"]

    def test_the_returned_law_carries_the_renamed_fields(self):
        kernel = _RecordingKernel(
            "k", {"mu": _SCALAR}, OutputSpec(RecordSpec(y=_SCALAR, z=_SCALAR))
        )
        law = kernel.with_path_names(y="obs")._condition_on({"mu": 1.0})
        assert law.event_spec == OutputSpec(RecordSpec(obs=_SCALAR, z=_SCALAR))

    def test_binding_part_of_a_structured_slot_raises_value_error(self):
        kernel = _RecordingKernel("k", {"a": _SCALAR, "b": _SCALAR}, OutputSpec(y=_SCALAR))
        grouped = kernel.with_path_names({"a": "theta/a", "b": "theta/b"})
        with pytest.raises(ValueError, match="part of a slot"):
            grouped._condition_on({"theta": {"a": 1.0}})

    def test_a_key_that_is_not_a_slot_raises_key_error(self):
        kernel = _RecordingKernel("k", {"mu": _SCALAR}, OutputSpec(y=_SCALAR))
        with pytest.raises(KeyError):
            kernel.with_path_names(mu="loc")._condition_on({"mu": 1.0})

    def test_a_renamed_kernel_round_trips_through_pickle(self):
        kernel = _RecordingKernel("k", {"mu": _SCALAR}, OutputSpec(y=_SCALAR))
        renamed = kernel.with_path_names(mu="loc")
        restored = pickle.loads(pickle.dumps(renamed))
        assert (restored.name, restored.spec) == (renamed.name, renamed.spec)


class TestRenamedKernelMoves:
    @pytest.mark.parametrize(
        ("moves", "error"),
        [
            pytest.param({"a": "a/x"}, ValueError, id="into-its-own-node"),
            pytest.param({"a": "g", "b": "g/b"}, ValueError, id="overlapping-targets"),
            pytest.param({"a": "b"}, ValueError, id="onto-a-slot"),
            pytest.param({"a": "g//a"}, ValueError, id="empty-segment"),
            pytest.param({"a": "not a name"}, ValueError, id="not-an-identifier"),
            pytest.param({"a/x": "x"}, KeyError, id="below-a-leaf"),
        ],
    )
    def test_a_move_the_given_side_cannot_take_raises(self, moves, error):
        kernel = _RecordingKernel("k", {"a": _SCALAR, "b": _SCALAR}, OutputSpec(y=_SCALAR))
        with pytest.raises(error):
            kernel.with_path_names(moves)

    def test_a_field_renamed_within_its_slot_keeps_its_position(self):
        kernel = _RecordingKernel(
            "k", {"theta": RecordSpec(a=_SCALAR, b=_SCALAR)}, OutputSpec(y=_SCALAR)
        )
        renamed = kernel.with_path_names({"theta/a": "theta/alpha"})
        assert renamed.given_spec["theta"] == RecordSpec(alpha=_SCALAR, b=_SCALAR)

    def test_swapped_slots_exchange_their_names(self):
        kernel = _RecordingKernel("k", {"a": _SCALAR, "b": _vector(2)}, OutputSpec(y=_SCALAR))
        swapped = kernel.with_path_names(a="b", b="a")
        assert list(swapped.given_spec) == ["b", "a"]
        assert swapped.given_spec["a"] == _vector(2)
        swapped._condition_on({"a": jnp.zeros(2), "b": 1.0})
        assert kernel.calls[0]["a"] == 1.0

    def test_one_call_reads_the_targets_of_both_sides_as_exact_paths(self):
        kernel = _RecordingKernel(
            "k",
            {"theta": RecordSpec(a=_SCALAR, b=_SCALAR)},
            OutputSpec(RecordSpec(g=RecordSpec(mu=_SCALAR, sigma=_SCALAR), y=_SCALAR)),
        )
        renamed = kernel.with_path_names({"theta/a": "alpha", "g/mu": "m"})
        assert list(renamed.given_spec) == ["theta", "alpha"]
        assert renamed.given_spec["theta"] == RecordSpec(b=_SCALAR)
        assert renamed.event_spec == OutputSpec(
            RecordSpec(g=RecordSpec(sigma=_SCALAR), y=_SCALAR, m=_SCALAR)
        )
        law = renamed._condition_on({"theta": {"b": 1.0}, "alpha": 2.0})
        assert kernel.calls[0]["theta"]["a"] == 2.0
        assert law.event_spec == renamed.event_spec

    def test_a_factored_kernel_renames_through_its_factors(self):
        first = _RecordingKernel("k1", {"x": _SCALAR}, OutputSpec(a=_SCALAR))
        joint = first * _RecordingKernel("k2", {"w": _SCALAR}, OutputSpec(b=_SCALAR))
        renamed = joint.with_path_names(x="u", b="c")
        assert isinstance(renamed, FactoredConditionalDistribution)
        assert set(renamed.given_spec) == {"u", "w"}
        assert list(renamed.event_spec.components) == ["a", "c"]
        assert list(renamed.factors[0].given_spec) == ["u"]
        assert list(renamed.factors[1].event_spec.components) == ["c"]
        renamed._condition_on({"u": 1.0, "w": 2.0})
        assert first.calls == [{"x": 1.0}]

    def test_a_rename_the_factors_cannot_carry_renames_at_the_joint_boundary(self):
        """Moving a whole term's component into a group changes no factor, so the joint holds it."""
        joint = _RecordingKernel("k1", {"x": _SCALAR}, OutputSpec(a=_SCALAR)) * _RecordingKernel(
            "k2", {"w": _SCALAR}, OutputSpec(b=_SCALAR)
        )
        renamed = joint.with_path_names({"a": "g/a"})
        assert not isinstance(renamed, FactoredConditionalDistribution)
        assert renamed.event_spec == OutputSpec(RecordSpec(b=_SCALAR, g=RecordSpec(a=_SCALAR)))
        assert set(renamed.given_spec) == {"x", "w"}


class TestRenamedKernelCapabilities:
    def test_a_renamed_kernel_claims_the_conditional_capabilities_of_its_parent(self):
        renamed = _MeanKernel().with_path_names(mu="loc")
        assert isinstance(renamed, SupportsConditionalMean)
        assert isinstance(renamed, SupportsConditionalSampling)
        assert not isinstance(
            _RecordingKernel("k", {"mu": _SCALAR}, OutputSpec(y=_SCALAR)).with_path_names(mu="loc"),
            SupportsConditionalMean,
        )

    def test_the_conditional_mean_translates_the_given_and_renames_the_mean(self):
        renamed = _MeanKernel().with_path_names(mu="loc", y="obs")
        mean = renamed._conditional_mean({"loc": 2.0})
        assert list(mean.keys()) == ["obs", "z"]
        assert jnp.allclose(mean["obs"], 2.0)

    def test_a_conditional_draw_carries_the_new_names(self, key):
        renamed = _MeanKernel().with_path_names(y="obs")
        draw = renamed._conditional_sample({"mu": 1.0}, key)
        assert list(draw.keys()) == ["obs", "z"]
        assert jnp.allclose(draw["obs"] - draw["z"], 1.0)

    def test_a_conditional_capability_needs_every_slot(self):
        kernel = _RecordingKernel("k", {"mu": _SCALAR, "sigma": _SCALAR}, OutputSpec(y=_SCALAR))
        renamed = kernel.with_path_names(mu="loc")
        with pytest.raises(ValueError, match="every slot"):
            renamed._parent_given({"loc": 1.0})

    def test_a_conditional_capability_carries_the_parent_guard(self):
        renamed = _MeanKernel().with_path_names(mu="loc")
        assert _capability_guard(renamed, "_conditional_mean") == Feasibility(True)

    def test_a_moved_event_field_moves_in_the_conditional_draws_and_mean(self, key):
        renamed = _MeanKernel().with_path_names({"y": "g/y"})
        assert list(renamed._conditional_mean({"mu": 2.0}).keys()) == ["z", "g/y"]
        draw = renamed._conditional_sample({"mu": 1.0}, key)
        assert list(draw.keys()) == ["z", "g/y"]
        assert jnp.allclose(draw["g/y"] - draw["z"], 1.0)

    def test_the_conditional_density_scores_under_the_original_names(self):
        parent = _ScoreKernel()
        renamed = parent.with_path_names(mu="loc", y="g/obs")
        value = {"z": jnp.asarray(0.5), "g": {"obs": jnp.asarray(2.0)}}
        expected = parent._conditional_log_prob(
            {"mu": 1.0}, {"y": jnp.asarray(2.0), "z": jnp.asarray(0.5)}
        )
        assert jnp.allclose(renamed._conditional_log_prob({"loc": 1.0}, value), expected)
