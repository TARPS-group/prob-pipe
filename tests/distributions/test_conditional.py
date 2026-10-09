"""The conditional distribution: a probability kernel, its term spec, and its markers.

A ``ConditionalDistribution`` stores one ``ConditionalDistributionSpec``: a
non-empty ``InputSpec`` of given slots and an event declaration completed as a
``Distribution``'s is. ``given_spec`` and ``event_spec`` are views on it, the
given-slot and produced-component names are disjoint, and symbolic dimensions
are scoped over both sides jointly. The numeric markers read membership from
the declarations, and ``_condition_on`` is the one method a kernel must
implement.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    InputSpec,
    Normal,
    NumericArraySpec,
    OpaqueSpec,
    OutputSpec,
    RecordSpec,
    TrackedTerm,
    condition_on,
    mean,
)
from probpipe.core._expression import expression_of, with_fixed
from probpipe.distributions import (
    ConditionalDistribution,
    ConditionalDistributionSpec,
    ConditionalNumericDistribution,
    Distribution,
    DistributionSpec,
    FullyNumericConditionalDistribution,
    NumericConditionalDistribution,
    NumericDistribution,
)
from probpipe.distributions._distribution import _detached_term, _fixed_paths

SCALAR = NumericArraySpec(())
LABEL = OpaqueSpec()
MARKERS = (
    ConditionalNumericDistribution,
    NumericConditionalDistribution,
    FullyNumericConditionalDistribution,
)


def _array(*shape: int | str) -> NumericArraySpec:
    return NumericArraySpec(shape)


# -- Kernels ------------------------------------------------------------------


class _Law(Distribution):
    """A law that declares its event and implements no capability."""


class _GivesDeclaredLaw:
    """Implements the primitive with a law over the kernel's own event."""

    def _condition_on(self, given, /, **kwargs):
        return _Law(self.name, self.event_spec)


class Kernel(_GivesDeclaredLaw, ConditionalDistribution):
    """A kernel over the slots and event it is given."""


class NumericEventKernel(_GivesDeclaredLaw, ConditionalNumericDistribution):
    """A class that claims a numeric event for every instance."""


class NumericGivenKernel(_GivesDeclaredLaw, NumericConditionalDistribution):
    """A class that claims numeric given slots for every instance."""


class FullyNumericKernel(_GivesDeclaredLaw, FullyNumericConditionalDistribution):
    """A class that claims both sides numeric for every instance."""


class LocationKernel(ConditionalDistribution):
    """``y | mu ~ Normal(mu, 1)``: a normal law over the kernel's event at each ``mu``."""

    def __init__(self, label: str = "lik") -> None:
        super().__init__(label, {"mu": SCALAR}, OutputSpec(y=SCALAR))

    def _condition_on(self, given, /, **kwargs):
        return Normal(self.label, given["mu"], 1.0, event_spec=self.event_spec)


def _kernel(given=None, event=None, label: str = "k") -> Kernel:
    return Kernel(
        label,
        {"mu": SCALAR} if given is None else given,
        OutputSpec(y=SCALAR) if event is None else event,
    )


def _mean_of(law: Distribution) -> np.ndarray:
    return np.asarray(mean(law))


# -- Tests --------------------------------------------------------------------


class TestKinds:
    """A kernel and a law are distinct tracked kinds; neither inherits from the other."""

    def test_neither_class_inherits_from_the_other(self):
        assert not issubclass(ConditionalDistribution, Distribution)
        assert not issubclass(Distribution, ConditionalDistribution)
        assert issubclass(ConditionalDistribution, TrackedTerm)

    def test_a_kernel_is_not_a_law_and_a_law_is_not_a_kernel(self):
        kernel = _kernel()
        law = Normal("y", 0.0, 1.0)
        assert not isinstance(kernel, Distribution)
        assert not isinstance(kernel, NumericDistribution)
        assert not isinstance(law, ConditionalDistribution)
        assert not any(isinstance(law, marker) for marker in MARKERS)

    def test_a_kernel_is_immutable(self):
        kernel = _kernel()
        with pytest.raises(AttributeError, match="immutable"):
            kernel.label = "other"
        with pytest.raises(AttributeError, match="immutable"):
            del kernel._name


class TestConstruction:
    def test_a_mapping_given_becomes_an_input_spec_in_its_order(self):
        kernel = _kernel(given={"mu": SCALAR, "x": _array(3)})
        assert isinstance(kernel.given_spec, InputSpec)
        assert list(kernel.given_spec) == ["mu", "x"]
        assert dict(kernel.given_spec) == {"mu": SCALAR, "x": _array(3)}

    def test_an_input_spec_given_is_kept(self):
        given = InputSpec(mu=SCALAR, theta=RecordSpec(a=SCALAR, b=_array(2)))
        assert _kernel(given=given).given_spec == given

    def test_a_bare_term_spec_completes_to_a_whole_term_under_the_kernel_name(self):
        kernel = _kernel(event=_array(3), label="lik")
        assert kernel.event_spec == OutputSpec(lik=_array(3))
        assert not kernel.event_spec.exposes_record

    def test_the_default_component_is_captured_once(self):
        kernel = _kernel(event=SCALAR, label="lik")
        renamed = kernel.with_label("other")
        assert renamed.label == "other"
        assert kernel.label == "lik"
        assert renamed.spec == kernel.spec
        assert list(renamed.event_spec.components) == ["lik"]

    @pytest.mark.parametrize(
        "record",
        [
            pytest.param(RecordSpec(a=SCALAR, b=_array(2)), id="two-fields"),
            pytest.param(RecordSpec(a=SCALAR), id="one-field"),
        ],
    )
    def test_a_record_spec_exposes_its_fields(self, record):
        kernel = _kernel(event=record)
        assert kernel.event_spec.exposes_record
        assert kernel.event_spec == OutputSpec(record)
        assert dict(kernel.event_spec.components) == dict(record.children)

    @pytest.mark.parametrize(
        "declaration",
        [
            pytest.param(OutputSpec(y=_array(2)), id="whole-array"),
            pytest.param(OutputSpec(parameters=RecordSpec(a=SCALAR, b=SCALAR)), id="whole-record"),
            pytest.param(OutputSpec(RecordSpec(a=SCALAR, b=SCALAR)), id="exposed-record"),
        ],
    )
    def test_an_output_spec_is_kept_as_declared(self, declaration):
        assert _kernel(event=declaration).event_spec == declaration


class TestConstructionErrors:
    @pytest.mark.parametrize(
        "given", [pytest.param({}, id="mapping"), pytest.param(InputSpec(), id="input-spec")]
    )
    def test_an_empty_given_raises(self, given):
        with pytest.raises(ValueError, match="at least one given slot"):
            _kernel(given=given)

    @pytest.mark.parametrize(
        ("given", "event", "name"),
        [
            pytest.param({"y": SCALAR}, OutputSpec(y=SCALAR), "k", id="whole-term-component"),
            pytest.param({"a": SCALAR}, RecordSpec(a=SCALAR, b=SCALAR), "k", id="record-field"),
            pytest.param({"lik": SCALAR}, SCALAR, "lik", id="default-component"),
        ],
    )
    def test_a_given_slot_named_like_a_produced_component_raises(self, given, event, name):
        with pytest.raises(ValueError, match="both as a given slot and as an output field"):
            _kernel(given=given, event=event, label=name)

    def test_a_missing_label_raises(self):
        with pytest.raises(TypeError, match="label"):
            Kernel(given_spec={"mu": SCALAR}, event_spec=SCALAR)

    @pytest.mark.parametrize("name", ["", None, 3])
    def test_a_name_that_is_not_a_non_empty_string_raises(self, name):
        with pytest.raises(TypeError, match="non-empty label"):
            _kernel(label=name)

    @pytest.mark.parametrize("event", [3.0, (3,), "y", None])
    def test_an_event_that_is_not_a_spec_raises(self, event):
        with pytest.raises(TypeError, match="event_spec"):
            Kernel("k", {"mu": SCALAR}, event)

    def test_an_event_with_a_type_hole_raises(self):
        with pytest.raises(ValueError, match="does not declare a type"):
            _kernel(event=OutputSpec(y=None))

    @pytest.mark.parametrize("given", [["mu"], SCALAR, 3])
    def test_a_given_that_is_not_a_mapping_raises(self, given):
        with pytest.raises(TypeError, match="given_spec"):
            _kernel(given=given)

    @pytest.mark.parametrize("slot", ["m u", "lambda", "theta/a", ""])
    def test_a_given_slot_that_is_not_an_identifier_raises(self, slot):
        with pytest.raises(ValueError, match="identifiers"):
            _kernel(given={slot: SCALAR})

    @pytest.mark.parametrize("slot_spec", [None, (3,), 3.0])
    def test_a_given_slot_that_is_not_a_term_spec_raises(self, slot_spec):
        with pytest.raises(TypeError, match="TermSpec"):
            _kernel(given={"mu": slot_spec})

    def test_a_bare_event_needs_a_name_that_is_a_valid_component(self):
        with pytest.raises(ValueError, match="component names"):
            _kernel(event=SCALAR, label="a/b")

    def test_a_kernel_that_leaves_its_declaration_unset_raises(self):
        class Undeclared(ConditionalDistribution):
            def __init__(self, label):
                self._init_tracked(label)

            def _condition_on(self, given, /, **kwargs):
                return _Law(self.name, SCALAR)

        with pytest.raises(TypeError, match="undeclared"):
            Undeclared("k")


class TestViews:
    """Both declarations are views on the one stored ``ConditionalDistributionSpec``."""

    def test_the_kernel_stores_one_conditional_distribution_spec(self):
        kernel = _kernel(given={"mu": SCALAR}, event=OutputSpec(y=_array(2)))
        assert kernel.spec == ConditionalDistributionSpec({"mu": SCALAR}, OutputSpec(y=_array(2)))

    def test_given_spec_and_event_spec_are_read_from_spec(self):
        kernel = _kernel()
        assert kernel.given_spec is kernel.spec.given_spec
        assert kernel.event_spec is kernel.spec.event_spec

    def test_a_kernel_satisfies_its_own_spec(self):
        kernel = _kernel(given={"mu": SCALAR, "x": _array(3)}, event=RecordSpec(a=SCALAR))
        assert kernel.spec.is_valid(kernel)


class TestConditionalDistributionSpec:
    def test_a_record_event_completes_to_the_exposed_form(self):
        spec = ConditionalDistributionSpec({"mu": SCALAR}, RecordSpec(y=SCALAR))
        assert spec.event_spec == OutputSpec(RecordSpec(y=SCALAR))
        assert isinstance(spec.given_spec, InputSpec)

    @pytest.mark.parametrize("event", [SCALAR, LABEL, 3.0])
    def test_an_event_that_is_neither_a_declaration_nor_a_record_raises(self, event):
        with pytest.raises(TypeError, match="event_spec"):
            ConditionalDistributionSpec({"mu": SCALAR}, event)

    @pytest.mark.parametrize(
        ("given", "event", "error", "match"),
        [
            pytest.param({}, OutputSpec(y=SCALAR), ValueError, "at least one", id="empty-given"),
            pytest.param(
                {"mu": SCALAR},
                OutputSpec(y=None),
                ValueError,
                "does not declare a type",
                id="type-hole",
            ),
            pytest.param(
                {"y": SCALAR}, OutputSpec(y=SCALAR), ValueError, "both as a given", id="shared"
            ),
            pytest.param(["mu"], OutputSpec(y=SCALAR), TypeError, "given_spec", id="not-mapping"),
        ],
    )
    def test_a_declaration_a_kernel_cannot_carry_raises(self, given, event, error, match):
        with pytest.raises(error, match=match):
            ConditionalDistributionSpec(given, event)

    def test_is_valid_accepts_a_matching_kernel(self):
        spec = ConditionalDistributionSpec({"mu": SCALAR, "x": _array(3)}, OutputSpec(y=SCALAR))
        assert spec.is_valid(_kernel(given={"mu": SCALAR, "x": _array(3)}))

    def test_given_slots_match_by_name_in_any_order(self):
        spec = ConditionalDistributionSpec({"a": SCALAR, "b": _array(2)}, OutputSpec(y=SCALAR))
        kernel = _kernel(given={"b": _array(2), "a": SCALAR})
        assert spec.is_valid(kernel)
        assert kernel.spec == spec
        assert hash(kernel.spec) == hash(spec)

    @pytest.mark.parametrize(
        ("given", "event"),
        [
            pytest.param({"x": _array(3), "z": SCALAR}, OutputSpec(y=_array(3)), id="extra-slot"),
            pytest.param({"w": _array(3)}, OutputSpec(y=_array(3)), id="renamed-slot"),
            pytest.param({"x": _array(3, 2)}, OutputSpec(y=_array(3)), id="slot-rank"),
            pytest.param({"x": _array(3)}, OutputSpec(z=_array(3)), id="other-component"),
            pytest.param({"x": _array(3)}, RecordSpec(y=_array(3)), id="other-packaging"),
            pytest.param({"x": _array(3)}, OutputSpec(y=_array(4)), id="other-size"),
        ],
    )
    def test_is_valid_refuses_a_kernel_whose_declarations_differ(self, given, event):
        spec = ConditionalDistributionSpec({"x": _array(3)}, OutputSpec(y=_array(3)))
        assert not spec.is_valid(_kernel(given=given, event=event))

    def test_is_valid_refuses_a_value_that_is_not_a_kernel(self):
        spec = ConditionalDistributionSpec({"mu": SCALAR}, OutputSpec(y=SCALAR))
        assert not spec.is_valid(_Law("y", OutputSpec(y=SCALAR)))
        assert not spec.is_valid(3.0)
        assert not spec.is_valid({"mu": SCALAR})

    def test_the_free_dimensions_span_both_sides(self):
        spec = ConditionalDistributionSpec({"x": _array("n")}, OutputSpec(y=_array("n", "m")))
        assert spec.free_dims == {"n", "m"}
        assert not spec.is_concrete

    @pytest.mark.parametrize(("event_size", "valid"), [(3, True), (4, False)])
    def test_a_dimension_shared_by_both_sides_binds_once(self, event_size, valid):
        spec = ConditionalDistributionSpec({"x": _array("n")}, OutputSpec(y=_array("n")))
        kernel = _kernel(given={"x": _array(3)}, event=OutputSpec(y=_array(event_size)))
        assert spec.is_valid(kernel) is valid

    def test_binding_from_a_kernel_binds_both_sides(self):
        spec = ConditionalDistributionSpec({"x": _array("n")}, OutputSpec(y=_array("n")))
        kernel = _kernel(given={"x": _array(3)}, event=OutputSpec(y=_array(3)))
        expected = ConditionalDistributionSpec({"x": _array(3)}, OutputSpec(y=_array(3)))
        assert spec.bind_dims_from_value(kernel) == expected
        assert spec.bind_dims_from_spec(kernel.spec) == expected

    def test_binding_from_a_kernel_whose_sides_disagree_raises(self):
        spec = ConditionalDistributionSpec({"x": _array("n")}, OutputSpec(y=_array("n")))
        kernel = _kernel(given={"x": _array(3)}, event=OutputSpec(y=_array(4)))
        with pytest.raises(ValueError, match="'n'"):
            spec.bind_dims_from_value(kernel)
        with pytest.raises(ValueError, match="'n'"):
            spec.bind_dims_from_spec(kernel.spec)

    def test_the_spec_substitutes_and_renames_dimensions_on_both_sides(self):
        spec = ConditionalDistributionSpec({"x": _array("n")}, OutputSpec(y=_array("n")))
        assert spec.with_dim_sizes(n=5) == ConditionalDistributionSpec(
            {"x": _array(5)}, OutputSpec(y=_array(5))
        )
        assert spec.with_dim_names(n="m") == ConditionalDistributionSpec(
            {"x": _array("m")}, OutputSpec(y=_array("m"))
        )

    @staticmethod
    def _schema_with_a_kernel_field() -> tuple[RecordSpec, ConditionalDistributionSpec]:
        declared = ConditionalDistributionSpec({"x": _array("n")}, OutputSpec(y=_array("n")))
        return RecordSpec({"likelihood": declared, "z": _array("n")}), declared

    def test_as_a_record_field_it_binds_in_one_scope_with_its_siblings(self):
        schema, declared = self._schema_with_a_kernel_field()
        kernel = _kernel(given={"x": _array(3)}, event=OutputSpec(y=_array(3)))
        bound = schema.bind_dims_from_value({"likelihood": kernel, "z": jnp.zeros(3)})
        assert bound["z"].shape == (3,)
        assert bound["likelihood"] == declared.with_dim_sizes(n=3)

    def test_as_a_record_field_a_sibling_of_another_size_raises(self):
        schema, _ = self._schema_with_a_kernel_field()
        kernel = _kernel(given={"x": _array(3)}, event=OutputSpec(y=_array(3)))
        with pytest.raises(ValueError, match="'n'"):
            schema.bind_dims_from_value({"likelihood": kernel, "z": jnp.zeros(4)})


class TestDimensionTransforms:
    """``with_dim_sizes`` and ``with_dim_names`` act on both sides of the kernel."""

    @staticmethod
    def _polymorphic() -> Kernel:
        return _kernel(
            given={"x": _array("n"), "theta": RecordSpec(w=_array("m"))},
            event=OutputSpec(y=_array("n", "m")),
        )

    def test_with_dim_sizes_binds_a_dimension_on_both_sides(self):
        bound = self._polymorphic().with_dim_sizes(n=3)
        assert bound.given_spec["x"] == _array(3)
        assert bound.given_spec["theta"] == RecordSpec(w=_array("m"))
        assert bound.event_spec == OutputSpec(y=_array(3, "m"))
        assert bound.spec.free_dims == {"m"}

    def test_with_dim_sizes_reaches_a_structured_given_slot(self):
        bound = self._polymorphic().with_dim_sizes(m=2)
        assert bound.given_spec["theta"] == RecordSpec(w=_array(2))
        assert bound.event_spec == OutputSpec(y=_array("n", 2))

    def test_with_dim_sizes_returns_a_new_kernel_of_the_same_class_and_name(self):
        kernel = self._polymorphic()
        bound = kernel.with_dim_sizes(n=3, m=2)
        assert bound is not kernel
        assert type(bound) is Kernel
        assert bound.label == kernel.label
        assert bound.spec.is_concrete
        assert kernel.spec.free_dims == {"n", "m"}

    @pytest.mark.parametrize(
        "rebind",
        [
            pytest.param(lambda k: k.with_dim_sizes(q=3), id="unknown"),
            pytest.param(lambda k: k.with_dim_sizes(n=3).with_dim_sizes(n=3), id="already-bound"),
        ],
    )
    def test_with_dim_sizes_refuses_a_name_that_is_not_free(self, rebind):
        with pytest.raises(ValueError, match="no free dimensions"):
            rebind(self._polymorphic())

    def test_with_dim_sizes_refuses_a_negative_size(self):
        with pytest.raises(ValueError, match="non-negative"):
            self._polymorphic().with_dim_sizes(n=-1)

    def test_with_dim_sizes_refuses_a_size_that_is_not_an_integer(self):
        with pytest.raises(TypeError, match="integer"):
            self._polymorphic().with_dim_sizes(n=2.5)

    def test_with_dim_names_renames_on_both_sides_simultaneously(self):
        swapped = self._polymorphic().with_dim_names(n="m", m="n")
        assert swapped.given_spec["x"] == _array("m")
        assert swapped.given_spec["theta"] == RecordSpec(w=_array("n"))
        assert swapped.event_spec == OutputSpec(y=_array("m", "n"))

    def test_with_dim_names_returns_a_new_kernel_of_the_same_class_and_name(self):
        kernel = self._polymorphic()
        renamed = kernel.with_dim_names(n="rows")
        assert renamed is not kernel
        assert type(renamed) is Kernel
        assert renamed.label == kernel.label
        assert renamed.spec.free_dims == {"rows", "m"}
        assert kernel.spec.free_dims == {"n", "m"}


# Declarations whose event side is numeric, or not.
_NUMERIC_EVENTS = [
    pytest.param(OutputSpec(y=SCALAR), id="array"),
    pytest.param(RecordSpec(a=SCALAR, b=_array(2)), id="exposed-record"),
    pytest.param(OutputSpec(parameters=RecordSpec(a=SCALAR)), id="whole-record"),
]
_OTHER_EVENTS = [
    pytest.param(OutputSpec(y=LABEL), id="opaque"),
    pytest.param(RecordSpec(a=SCALAR, label=LABEL), id="record-with-opaque-field"),
    pytest.param(OutputSpec(law=DistributionSpec(OutputSpec(b=SCALAR))), id="law-valued"),
]
# Given sides whose every slot is numeric, or not.
_NUMERIC_GIVENS = [
    pytest.param({"mu": SCALAR}, id="array"),
    pytest.param({"mu": SCALAR, "x": _array(3)}, id="two-arrays"),
    pytest.param({"theta": RecordSpec(a=SCALAR, b=_array(2))}, id="numeric-record-slot"),
]
_OTHER_GIVENS = [
    pytest.param({"mu": SCALAR, "label": LABEL}, id="one-opaque-slot"),
    pytest.param({"theta": RecordSpec(a=SCALAR, label=LABEL)}, id="record-slot-with-opaque"),
    pytest.param({"prior": DistributionSpec(OutputSpec(b=SCALAR))}, id="law-valued-slot"),
]


class TestNumericMarkers:
    """Membership in each marker is read from the declarations, whatever the class."""

    @pytest.mark.parametrize("event", _NUMERIC_EVENTS)
    def test_a_numeric_event_is_a_member_of_the_event_marker(self, event):
        assert isinstance(_kernel(event=event), ConditionalNumericDistribution)

    @pytest.mark.parametrize("event", _OTHER_EVENTS)
    def test_a_non_numeric_event_is_not_a_member_of_the_event_marker(self, event):
        assert not isinstance(_kernel(event=event), ConditionalNumericDistribution)

    @pytest.mark.parametrize("given", _NUMERIC_GIVENS)
    def test_numeric_given_slots_are_members_of_the_given_marker(self, given):
        assert isinstance(_kernel(given=given), NumericConditionalDistribution)

    @pytest.mark.parametrize("given", _OTHER_GIVENS)
    def test_a_non_numeric_given_slot_is_not_a_member_of_the_given_marker(self, given):
        assert not isinstance(_kernel(given=given), NumericConditionalDistribution)

    @pytest.mark.parametrize(
        ("given", "event", "member"),
        [
            pytest.param({"mu": SCALAR}, OutputSpec(y=SCALAR), True, id="both"),
            pytest.param({"mu": LABEL}, OutputSpec(y=SCALAR), False, id="event-only"),
            pytest.param({"mu": SCALAR}, OutputSpec(y=LABEL), False, id="given-only"),
            pytest.param({"mu": LABEL}, OutputSpec(y=LABEL), False, id="neither"),
        ],
    )
    def test_the_fully_numeric_marker_needs_both_sides(self, given, event, member):
        kernel = _kernel(given=given, event=event)
        assert isinstance(kernel, FullyNumericConditionalDistribution) is member

    def test_the_fully_numeric_marker_refines_both_markers(self):
        assert issubclass(FullyNumericConditionalDistribution, NumericConditionalDistribution)
        assert issubclass(FullyNumericConditionalDistribution, ConditionalNumericDistribution)

    @pytest.mark.parametrize("marker", MARKERS, ids=lambda marker: marker.__name__)
    def test_a_marker_adds_no_operations(self, marker):
        assert {name for name in vars(marker) if not name.startswith("_")} == set()

    @pytest.mark.parametrize(
        ("cls", "marker", "given", "event"),
        [
            pytest.param(
                NumericEventKernel,
                ConditionalNumericDistribution,
                {"mu": LABEL},
                OutputSpec(y=SCALAR),
                id="event",
            ),
            pytest.param(
                NumericGivenKernel,
                NumericConditionalDistribution,
                {"mu": SCALAR},
                OutputSpec(y=LABEL),
                id="given",
            ),
            pytest.param(
                FullyNumericKernel,
                FullyNumericConditionalDistribution,
                {"mu": SCALAR},
                OutputSpec(y=SCALAR),
                id="fully",
            ),
        ],
    )
    def test_a_class_that_inherits_a_marker_constructs_when_the_claim_holds(
        self, cls, marker, given, event
    ):
        assert isinstance(cls("k", given, event), marker)

    @pytest.mark.parametrize(
        ("cls", "given", "event"),
        [
            pytest.param(NumericEventKernel, {"mu": SCALAR}, OutputSpec(y=LABEL), id="event"),
            pytest.param(
                NumericGivenKernel, {"mu": SCALAR, "label": LABEL}, OutputSpec(y=SCALAR), id="given"
            ),
            pytest.param(FullyNumericKernel, {"mu": SCALAR}, OutputSpec(y=LABEL), id="fully-event"),
            pytest.param(FullyNumericKernel, {"mu": LABEL}, OutputSpec(y=SCALAR), id="fully-given"),
        ],
    )
    def test_a_class_that_inherits_a_marker_refuses_a_declaration_that_fails_it(
        self, cls, given, event
    ):
        with pytest.raises(TypeError, match="inherits"):
            cls("k", given, event)


class TestPrimitive:
    """``_condition_on`` is the one method a kernel must implement."""

    def test_condition_on_is_the_only_abstract_method(self):
        assert ConditionalDistribution.__abstractmethods__ == frozenset({"_condition_on"})

    def test_the_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="_condition_on"):
            ConditionalDistribution("k", {"mu": SCALAR}, OutputSpec(y=SCALAR))

    def test_a_subclass_without_the_primitive_cannot_be_instantiated(self):
        class WithoutPrimitive(ConditionalDistribution):
            pass

        with pytest.raises(TypeError, match="_condition_on"):
            WithoutPrimitive("k", {"mu": SCALAR}, OutputSpec(y=SCALAR))

    def test_a_subclass_with_the_primitive_gives_a_law_over_its_event(self):
        kernel = LocationKernel()
        law = kernel._condition_on({"mu": 2.0})
        assert isinstance(law, Distribution)
        assert DistributionSpec(kernel.event_spec).is_valid(law)
        np.testing.assert_allclose(_mean_of(law), 2.0)


# Renames a kernel over the slot ``mu`` and the whole-term event ``y`` cannot take.
_REFUSED_RENAMES = [
    pytest.param(lambda k: k.with_path_names(sigma="s"), KeyError, id="not-a-path"),
    pytest.param(lambda k: k.with_path_names(mu="y"), ValueError, id="slot-onto-a-component"),
    pytest.param(lambda k: k.with_path_names(y="mu"), ValueError, id="component-onto-a-slot"),
    pytest.param(lambda k: k.with_path_names(mu="y/mu"), ValueError, id="slot-into-a-component"),
    pytest.param(lambda k: k.with_path_names(mu=""), ValueError, id="empty-name"),
    pytest.param(lambda k: k.with_path_names({"mu": "a"}, mu="b"), ValueError, id="twice"),
    pytest.param(lambda k: k.with_path_names(), ValueError, id="no-renames"),
]


class TestWithPathNames:
    """``with_path_names`` renames or moves names on either side and keeps the kernel."""

    def test_a_renamed_given_slot_keeps_its_position(self):
        kernel = _kernel(given={"a": SCALAR, "b": _array(2), "c": SCALAR})
        renamed = kernel.with_path_names(b="beta")
        assert list(renamed.given_spec) == ["a", "beta", "c"]
        assert renamed.given_spec["beta"] == _array(2)
        assert renamed.event_spec == kernel.event_spec
        assert renamed.label == kernel.label

    def test_a_renamed_event_component_renames_the_draws(self):
        kernel = _kernel(event=OutputSpec(y=_array(2)))
        renamed = kernel.with_path_names(y="obs")
        assert renamed.event_spec == OutputSpec(obs=_array(2))
        assert renamed.given_spec == kernel.given_spec

    def test_renames_on_both_sides_apply_together(self):
        renamed = _kernel().with_path_names(mu="loc", y="obs")
        assert list(renamed.given_spec) == ["loc"]
        assert list(renamed.event_spec.components) == ["obs"]

    def test_the_renamed_kernel_conditions_on_the_renamed_slot(self):
        renamed = LocationKernel().with_path_names(mu="loc", y="obs")
        law = renamed._condition_on({"loc": 2.0})
        assert list(law.event_spec.components) == ["obs"]
        np.testing.assert_allclose(_mean_of(law), 2.0)

    def test_path_targets_group_given_slots_into_a_structured_slot(self):
        kernel = _kernel(given={"a": SCALAR, "b": _array(2)})
        grouped = kernel.with_path_names({"a": "theta/a", "b": "theta/b"})
        assert list(grouped.given_spec) == ["theta"]
        assert grouped.given_spec["theta"] == RecordSpec(a=SCALAR, b=_array(2))
        assert grouped.event_spec == kernel.event_spec

    def test_a_path_key_splits_a_field_out_of_a_structured_slot(self):
        kernel = _kernel(given={"theta": RecordSpec(a=SCALAR, b=_array(2))})
        split = kernel.with_path_names({"theta/a": "a"})
        assert list(split.given_spec) == ["theta", "a"]
        assert split.given_spec["theta"] == RecordSpec(b=_array(2))
        assert split.given_spec["a"] == SCALAR

    @pytest.mark.parametrize(("rename", "error"), _REFUSED_RENAMES)
    def test_a_rename_the_kernel_cannot_take_raises(self, rename, error):
        with pytest.raises(error):
            rename(_kernel())


def _with_fixed_paths(term, *paths: str):
    """*term* holding *paths* fixed, as applying a kernel at given values records."""
    term._store_expression(with_fixed(expression_of(term), paths))
    return term


class TestNotation:
    """A kernel reads as its label, its components, ``|``, and its given slots."""

    def test_a_kernel_reads_by_its_components_and_its_given_slots(self):
        glm = _kernel(given={"beta": SCALAR}, label="glm")
        assert glm.notation == "glm(y | beta)"

    def test_every_given_slot_is_listed_in_declaration_order(self):
        glm = _kernel(given={"beta": SCALAR, "sigma": SCALAR}, label="glm")
        assert glm.notation == "glm(y | beta, sigma)"

    def test_every_component_is_listed_in_declaration_order(self):
        kernel = _kernel(event=OutputSpec(RecordSpec(y=(), z=())), label="k")
        assert kernel.notation == "k(y, z | mu)"

    def test_str_returns_the_notation_and_the_repr_keeps_the_label_first(self):
        glm = _kernel(given={"beta": SCALAR}, label="glm")
        assert str(glm) == "glm(y | beta)"
        assert repr(glm).startswith("Kernel('glm', given=('beta',)")

    def test_fixed_paths_follow_the_given_slots(self):
        glm = _with_fixed_paths(_kernel(given={"sigma": SCALAR}, label="glm"), "beta")
        assert glm.notation == "glm(y | sigma; beta)"

    def test_a_kernel_holds_no_path_fixed_by_default(self):
        assert _fixed_paths(_kernel()) == ()

    @pytest.mark.parametrize(
        "copy",
        [
            pytest.param(lambda k: k.raw(), id="raw"),
            pytest.param(lambda k: _detached_term(k), id="detached"),
            pytest.param(lambda k: k.with_label("glm2"), id="with_label"),
            pytest.param(lambda k: k.with_dim_sizes(n=3), id="with_dim_sizes"),
            pytest.param(lambda k: k.with_dim_names(n="m"), id="with_dim_names"),
            pytest.param(lambda k: k.with_path_names(sigma="s"), id="rename-a-slot"),
            pytest.param(lambda k: k.with_path_names(y="obs"), id="rename-a-component"),
        ],
    )
    def test_a_copy_keeps_the_fixed_paths(self, copy):
        kernel = _kernel(given={"sigma": _array("n")}, event=OutputSpec(y=_array("n")))
        assert _fixed_paths(copy(_with_fixed_paths(kernel, "beta"))) == ("beta",)


class TestConditionOnOperation:
    """``condition_on`` binds a kernel's given slots by applying the kernel."""

    def test_binding_every_given_slot_returns_the_law_the_kernel_gives(self):
        kernel = LocationKernel()
        law = condition_on(kernel, {"mu": 2.0})
        assert isinstance(law, Distribution)
        assert DistributionSpec(kernel.event_spec).is_valid(law)
        np.testing.assert_allclose(_mean_of(law), 2.0)
