"""Executable input and output spec examples from design II.2 and shared dimension contracts."""

from __future__ import annotations

import copy
import pickle
import re
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BatchSpec,
    Distribution,
    DistributionBatch,
    DistributionSpec,
    EmpiricalDistribution,
    Function,
    FunctionSpec,
    InputSpec,
    Normal,
    NumericArraySpec,
    NumericRecordSpec,
    NumericSpec,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordSpec,
    TermSpec,
    distribution,
    positive,
    sample,
    workflow_run,
)
from probpipe.core._specs import (
    _has_record_outputs,
    _output_record,
    _root_output_name,
)


@pytest.fixture
def declared_law():
    class DeclaredLaw(Distribution):
        def __init__(self, template):
            super().__init__(
                template,
                label="law",
            )

    return DeclaredLaw


_REAL = NumericArraySpec(())


def _grouped_spec():
    """The output spec of a law over ``{a, g: {b, c}}``."""
    return OutputSpec(RecordSpec(a=_REAL, g=RecordSpec(b=_REAL, c=_REAL)))


def _law_over_laws():
    """A law over laws, declared in the positional form over a law over an array ``x``."""
    laws = DistributionBatch([Normal("x", 0.0, 1.0), Normal("x", 1.0, 2.0)], "law")
    return EmpiricalDistribution(laws, event_spec=OutputSpec(laws.element_spec.term_spec))


class TestOutputSpec:
    def test_ii2_output_forms(self):
        beta = NumericArraySpec((3,))
        sigma = NumericArraySpec(())
        whole = OutputSpec(beta=beta)
        assert whole.term_spec is beta
        assert dict(whole.outputs) == {"beta": beta}

        hole = OutputSpec(beta=None)
        assert hole.term_spec is None
        assert dict(hole.outputs) == {"beta": None}
        assert not hole.is_concrete

        exposed = OutputSpec(RecordSpec(beta=beta))
        assert isinstance(exposed.term_spec, RecordSpec)
        assert exposed.term_spec != whole.term_spec
        assert dict(exposed.outputs) == dict(whole.outputs)
        assert exposed != whole

        # Two keywords would make the form depend on how many names there are.
        with pytest.raises(TypeError, match="takes one keyword"):
            OutputSpec(beta=beta, sigma=sigma)
        exposed_two = OutputSpec(RecordSpec(beta=beta, sigma=sigma))
        assert list(exposed_two.outputs) == ["beta", "sigma"]
        assert _has_record_outputs(exposed_two) and not _has_record_outputs(whole)

        parameters = RecordSpec(beta=beta, sigma=sigma)
        wrapped = OutputSpec(parameters=parameters)
        assert wrapped.term_spec is parameters
        assert dict(wrapped.outputs) == {"parameters": parameters}
        assert wrapped != OutputSpec(parameters)

    def test_output_exposure_is_immediate_and_keeps_higher_order_kinds(self):
        nested = RecordSpec(beta=(3,), sigma=())
        law = DistributionSpec(nested)
        exposed = OutputSpec(RecordSpec(parameters=nested, law=law))
        assert list(exposed.outputs) == ["parameters", "law"]
        assert exposed.outputs["parameters"] is nested
        assert exposed.outputs["law"] is law
        assert list(OutputSpec(posterior=law).outputs) == ["posterior"]

    def test_a_law_has_no_outputs_and_a_batch_has_its_elements(self):
        # A law as a value has no outputs: the laws of its draws' fields do not determine it.
        record = RecordSpec(mu=(), tau=())
        law = OutputSpec(DistributionSpec(OutputSpec(record)))
        assert (list(law.outputs), _has_record_outputs(law)) == ([], False)
        assert law.rename({"mu": "center"}).term_spec == DistributionSpec(
            OutputSpec(record.rename({"mu": "center"}))
        )
        batch = OutputSpec(BatchSpec(record, sample=3))
        assert (list(batch.outputs), _has_record_outputs(batch)) == (["mu", "tau"], False)
        assert batch.rename({"mu": "center"}).term_spec == BatchSpec(
            record.rename({"mu": "center"}), sample=3
        )
        with pytest.raises(TypeError, match=r"with_term_spec\(\) needs a DistributionSpec"):
            law.with_term_spec(record)

    def test_a_batch_of_arrays_exposes_no_outputs(self):
        with pytest.raises(TypeError):
            OutputSpec(BatchSpec(NumericArraySpec(()), sample=3))

    def test_a_record_field_given_none_is_refused_naming_the_opaque_spec(self):
        """None is a pending type, so a record field takes OpaqueSpec() instead."""
        with pytest.raises(TypeError, match=r"OpaqueSpec\(\)"):
            RecordSpec(payload=None)
        output = OutputSpec(RecordSpec(payload=OpaqueSpec()))
        assert isinstance(output.outputs["payload"], OpaqueSpec)
        assert output.is_concrete

    def test_output_empty_record_is_explicit(self):
        output = OutputSpec(RecordSpec())
        assert isinstance(output.term_spec, RecordSpec)
        assert dict(output.outputs) == {}
        with pytest.raises(ValueError):
            OutputSpec()

    def test_output_rejects_ambiguous_or_untyped_forms(self):
        spec = NumericArraySpec(())
        for args, kwargs in [
            ((spec,), {}),
            ((RecordSpec(x=()), RecordSpec(y=())), {}),
            ((RecordSpec(x=()),), {"y": spec}),
            ((), {"x": None, "y": spec}),
        ]:
            with pytest.raises(TypeError):
                OutputSpec(*args, **kwargs)

    def test_output_dimension_transforms_preserve_the_form_and_unspecified_types(self):
        spec = RecordSpec(x=("n",))
        for output in (OutputSpec(parameters=spec), OutputSpec(spec)):
            renamed = output.with_dim_names(n="m").with_dim_sizes(m=3)
            assert list(renamed.outputs) == list(output.outputs)
            assert renamed.term_spec == RecordSpec(x=(3,))
        hole = OutputSpec(beta=None)
        assert hole.with_dim_sizes(n=3).with_dim_names(n="m") == hole

    def test_the_record_of_the_outputs_of_each_form(self):
        assert _output_record(OutputSpec(theta=NumericArraySpec(()))) == RecordSpec(theta=())
        assert _output_record(OutputSpec(RecordSpec(a=()))) == RecordSpec(a=())


class TestTheFinalNames:
    """``term_spec``, ``outputs``, ``with_term_spec``, and ``output_name=`` replace the old names."""

    def test_the_term_spec_and_the_outputs_of_each_form(self):
        record = RecordSpec(mu=(), tau=())
        positional = OutputSpec(record)
        keyword = OutputSpec(x=NumericArraySpec(()))
        assert positional.term_spec is record
        assert list(positional.outputs) == ["mu", "tau"]
        assert list(positional.outputs.values()) == list(record.children.values())
        assert keyword.term_spec == NumericArraySpec(())
        assert list(keyword.outputs.items()) == [("x", NumericArraySpec(()))]

    def test_with_term_spec_completes_an_unspecified_type_given_positionally(self):
        completed = OutputSpec(mean=None).with_term_spec(NumericArraySpec((3,)))
        assert completed == OutputSpec(mean=NumericArraySpec((3,)))
        with pytest.raises(TypeError, match="positional"):
            OutputSpec(mean=None).with_term_spec(term_spec=NumericArraySpec(()))

    def test_default_takes_the_output_name(self):
        array = NumericArraySpec(())
        assert OutputSpec.default(array, output_name="x") == OutputSpec(x=array)
        assert OutputSpec.default(term_spec=array, output_name="x") == OutputSpec(x=array)
        with pytest.raises(TypeError, match="component"):
            OutputSpec.default(array, component="x")

    @pytest.mark.parametrize("old_name", ["spec", "components", "exposes_record", "with_spec"])
    @pytest.mark.parametrize(
        "output_spec",
        [
            pytest.param(OutputSpec(x=NumericArraySpec(())), id="keyword"),
            pytest.param(OutputSpec(RecordSpec(mu=(), tau=())), id="positional"),
            pytest.param(Normal("mu", 0.0, 1.0).event_spec, id="event-spec"),
        ],
    )
    def test_the_old_names_are_gone(self, output_spec, old_name):
        with pytest.raises(AttributeError, match=old_name):
            getattr(output_spec, old_name)

    def test_a_family_takes_its_output_name(self):
        assert Normal(output_name="x", loc=0.0, scale=1.0).event_spec == OutputSpec(
            x=Normal("x", 0.0, 1.0).event_spec.term_spec
        )
        with pytest.raises(TypeError, match="component"):
            Normal(component="x", loc=0.0, scale=1.0)


class TestTheReadersOfAnOutputSpec:
    """The private helpers through which code outside ``core/_specs.py`` reads an output spec."""

    def test_each_form_has_its_root_name_and_its_record_form(self):
        record = OutputSpec(RecordSpec(a=()))
        keyword = OutputSpec(x=NumericArraySpec(()))
        law_positional = OutputSpec(Normal("mu", 0.0, 1.0).spec)
        assert (_root_output_name(record), _has_record_outputs(record)) == (None, True)
        assert (_root_output_name(keyword), _has_record_outputs(keyword)) == ("x", False)
        assert (_root_output_name(law_positional), _has_record_outputs(law_positional)) == (
            None,
            False,
        )

    def test_the_record_of_the_outputs_of_a_record_and_of_a_named_value(self):
        assert _output_record(OutputSpec(RecordSpec(a=(), b=(2,)))) == RecordSpec(a=(), b=(2,))
        assert _output_record(OutputSpec(theta=NumericArraySpec((3,)))) == RecordSpec(
            theta=NumericArraySpec((3,))
        )

    @pytest.mark.parametrize(
        ("output_spec", "described"),
        [
            pytest.param(OutputSpec(Normal("mu", 0.0, 1.0).spec), "distributions", id="law"),
            pytest.param(
                OutputSpec(BatchSpec(RecordSpec(mu=(), tau=()), sample=3)), "batches", id="batch"
            ),
        ],
    )
    def test_the_record_of_the_outputs_is_refused_where_outputs_do_not_read_the_values(
        self, output_spec, described
    ):
        # A batch of records has the record of its marginal batches' types once batch outputs
        # are typed by their marginal batches; until then it is refused, as before.
        with pytest.raises(TypeError, match=f"its values are {described}.*Wrap them under a name"):
            _output_record(output_spec)

    def test_an_output_spec_over_a_law_has_no_outputs(self):
        # The distributions of a law's draws' fields do not determine it, so a law as a value
        # has no outputs, and a batch of laws has none either.
        law_spec = Normal("mu", 0.0, 1.0).spec
        assert OutputSpec(law_spec).outputs == {}
        assert OutputSpec(_law_over_laws().spec).outputs == {}

    def test_a_law_over_laws_has_no_paths(self):
        # x names the output at the root of the laws it is over, not a path of the law over them.
        law_over_laws = _law_over_laws()
        assert law_over_laws.event_spec.outputs == {}
        assert "x" not in law_over_laws
        with pytest.raises(
            TypeError, match=r"cannot index .* by 'x', since its draws are distributions"
        ):
            law_over_laws["x"]
        with pytest.raises(TypeError, match="Wrap them under a name"):
            _output_record(law_over_laws.event_spec)


class TestOutputSpecSelection:
    """``OutputSpec[]`` and ``in``: selecting from an output spec by path."""

    def test_one_path_to_a_single_value_names_it_by_the_path_s_last_name(self):
        assert _grouped_spec()["g/b"] == OutputSpec(b=_REAL)
        assert _grouped_spec()["a"] == OutputSpec(a=_REAL)

    def test_one_path_to_a_group_has_the_group_s_children_as_outputs(self):
        assert _grouped_spec()["g"] == OutputSpec(RecordSpec(b=_REAL, c=_REAL))
        assert list(_grouped_spec()["g"].outputs) == ["b", "c"]

    def test_a_list_keeps_each_node_at_its_path_in_the_spec_s_order(self):
        assert _grouped_spec()[["g"]] == OutputSpec(RecordSpec(g=RecordSpec(b=_REAL, c=_REAL)))
        assert _grouped_spec()[["g/c", "a"]] == OutputSpec(
            RecordSpec(a=_REAL, g=RecordSpec(c=_REAL))
        )
        # A tuple is a list of paths.
        assert _grouped_spec()["g/c", "a"] == _grouped_spec()[["g/c", "a"]]

    def test_a_field_holding_a_law_gives_the_law_s_output_spec(self):
        law_spec = (Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)).spec
        wrapped = OutputSpec(RecordSpec(params=law_spec))
        assert wrapped["params"] == OutputSpec(law_spec)
        assert wrapped["params"].outputs == {}

    def test_a_function_returning_a_wrapped_law_selects_the_law_s_output_spec(self):
        law = Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)
        fit = Function(
            lambda data: {"params": law},
            input_spec={"data": NumericArraySpec((3,))},
            output_spec=OutputSpec(RecordSpec(params=law.spec)),
        )
        assert fit.output_spec["params"] == OutputSpec(law.spec)

    def test_the_root_name_of_the_keyword_form_gives_the_spec_itself(self):
        # An interim reading, as a distribution over a single value is indexed by its name.
        spec = OutputSpec(x=_REAL)
        assert spec["x"] is spec
        assert "x" in spec
        assert spec[["x"]] == OutputSpec(RecordSpec(x=_REAL))

    def test_the_keyword_form_over_a_record_selects_below_its_name(self):
        # The keyword form over a record: its name gives itself, and its paths start with it.
        spec = OutputSpec(params=RecordSpec(mu=_REAL, sigma=_REAL))
        assert spec["params"] is spec
        assert spec["params/mu"] == OutputSpec(mu=_REAL)
        assert spec[["params/mu"]] == OutputSpec(RecordSpec(params=RecordSpec(mu=_REAL)))

    def test_in_tests_paths_and_never_raises(self):
        spec = _grouped_spec()
        assert "g/b" in spec and "g" in spec and "a" in spec
        assert "z" not in spec and "g/z" not in spec
        assert ["a"] not in spec and 0 not in spec and None not in spec
        assert "y" not in OutputSpec(x=_REAL)
        assert "mu" not in OutputSpec(Normal("mu", 0.0, 1.0).spec)

    def test_a_path_the_spec_lacks_raises_key_error_listing_its_paths(self):
        with pytest.raises(
            KeyError,
            match=re.escape(
                "unknown path 'z' in the output spec; available paths: ['a', 'g', 'g/b', 'g/c']"
            ),
        ):
            _grouped_spec()["z"]
        with pytest.raises(KeyError, match="To select the nested path, write 'g/b'"):
            _grouped_spec()["g", "b"]

    def test_a_spec_with_no_paths_raises_type_error(self):
        with pytest.raises(
            TypeError,
            match="cannot index the output spec by 'y', since its values are single values with "
            "no paths",
        ):
            OutputSpec(x=_REAL)["y"]
        with pytest.raises(TypeError, match="since its values are distributions"):
            OutputSpec(Normal("mu", 0.0, 1.0).spec)["mu"]
        with pytest.raises(TypeError, match="the type of its output 'x' is not known yet"):
            OutputSpec(x=None)["y"]
        with pytest.raises(TypeError, match="the type of its output 'x' is not known yet"):
            OutputSpec(x=None)[["x"]]

    @pytest.mark.parametrize(
        ("paths", "error"),
        [
            ("", ValueError),
            ("g//b", ValueError),
            ([], ValueError),
            (("a", "a"), ValueError),
            (("g", "g/b"), ValueError),
            ({"a"}, TypeError),
            (("a", 0), TypeError),
            (None, TypeError),
        ],
        ids=repr,
    )
    def test_a_malformed_path_is_refused_by_the_path_reader(self, paths, error):
        with pytest.raises(error):
            _grouped_spec()[paths]

    def test_an_output_spec_is_not_iterable(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "cannot iterate over an OutputSpec. Iterate over its outputs instead, as "
                "list(spec.outputs)"
            ),
        ):
            list(_grouped_spec())


class TestOutputSpecRemoval:
    """``OutputSpec.without``."""

    def test_removal_keeps_every_other_node_at_its_path(self):
        assert _grouped_spec().without("g/c") == OutputSpec(
            RecordSpec(a=_REAL, g=RecordSpec(b=_REAL))
        )
        assert _grouped_spec().without("g/b", "g/c") == OutputSpec(RecordSpec(a=_REAL))

    def test_selection_and_removal_partition_the_fields(self):
        spec = _grouped_spec()
        kept, removed = spec[["a", "g/b"]], spec.without("a", "g/b")
        assert set(_output_record(kept).keys()) | set(_output_record(removed).keys()) == {
            "a",
            "g/b",
            "g/c",
        }
        assert not set(_output_record(kept).keys()) & set(_output_record(removed).keys())

    def test_the_keyword_form_over_a_record_keeps_its_name(self):
        spec = OutputSpec(params=RecordSpec(mu=_REAL, sigma=_REAL))
        assert spec.without("params/sigma") == OutputSpec(params=RecordSpec(mu=_REAL))

    def test_removing_everything_is_refused(self):
        with pytest.raises(ValueError):
            _grouped_spec().without("a", "g")
        with pytest.raises(ValueError, match="cannot remove 'x' from the output spec"):
            OutputSpec(x=_REAL).without("x")

    def test_a_path_the_spec_lacks_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown path 'z' in the output spec"):
            _grouped_spec().without("z")

    def test_a_spec_with_no_paths_raises_type_error(self):
        with pytest.raises(
            TypeError, match="cannot remove 'y' from the output spec, since its values are single"
        ):
            OutputSpec(x=_REAL).without("y")

    def test_paths_are_given_one_per_argument(self):
        with pytest.raises(TypeError):
            _grouped_spec().without(("a", "g/c"))
        with pytest.raises(ValueError):
            _grouped_spec().without("g", "g/b")


class TestOutputSpecMerge:
    """``&`` on output specs."""

    def test_the_outputs_of_both_the_left_s_first(self):
        merged = OutputSpec(x=_REAL) & _grouped_spec()
        assert merged == OutputSpec(RecordSpec(x=_REAL, a=_REAL, g=RecordSpec(b=_REAL, c=_REAL)))
        assert list((_grouped_spec() & OutputSpec(x=_REAL)).outputs) == ["a", "g", "x"]

    def test_selections_merge_back_into_their_spec(self):
        spec = _grouped_spec()
        assert spec[["a"]] & spec[["g"]] == spec

    def test_an_output_in_both_is_refused(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "cannot merge the left output spec and the right output spec, since both have "
                "the output 'a'. Rename it in one of them, as rename({'a': 'a_2'})"
            ),
        ):
            _grouped_spec() & OutputSpec(a=_REAL)

    def test_a_spec_over_distributions_is_refused_with_the_advice_to_wrap(self):
        law_spec = Normal("mu", 0.0, 1.0).spec
        with pytest.raises(
            TypeError,
            match=re.escape(
                "cannot merge the left output spec, since its values are distributions, which "
                "have no outputs to read by name. Wrap them under a name, as "
                "OutputSpec(RecordSpec(params=...))"
            ),
        ):
            OutputSpec(law_spec) & OutputSpec(x=_REAL)
        assert OutputSpec(RecordSpec(params=law_spec)) & OutputSpec(x=_REAL) == OutputSpec(
            RecordSpec(params=law_spec, x=_REAL)
        )

    def test_a_spec_over_batches_is_refused_until_their_outputs_are_one_record(self):
        batch_spec = BatchSpec(RecordSpec(a=_REAL, b=_REAL), n=2)
        assert list(OutputSpec(batch_spec).outputs) == ["a", "b"]
        with pytest.raises(
            TypeError,
            match=re.escape(
                "cannot merge the left output spec, since its values are batches, whose outputs "
                "are not read as one record yet. Wrap them under a name"
            ),
        ):
            OutputSpec(batch_spec) & OutputSpec(x=_REAL)

    def test_a_spec_whose_type_is_not_known_is_refused(self):
        with pytest.raises(TypeError, match="the type of its output 'x' is not known yet"):
            OutputSpec(x=_REAL) & OutputSpec(x=None)

    def test_another_operand_is_not_merged(self):
        with pytest.raises(TypeError):
            _grouped_spec() & RecordSpec(z=_REAL)


class TestOutputSpecEquality:
    def test_equal_specs_are_equal_and_hash_alike(self):
        assert _grouped_spec() == _grouped_spec()
        assert hash(_grouped_spec()) == hash(_grouped_spec())
        assert OutputSpec(x=_REAL) == OutputSpec(x=_REAL)

    def test_the_root_name_and_the_field_order_count(self):
        assert OutputSpec(x=_REAL) != OutputSpec(y=_REAL)
        assert OutputSpec(x=_REAL) != OutputSpec(RecordSpec(x=_REAL))
        assert OutputSpec(RecordSpec(a=_REAL, b=_REAL)) != OutputSpec(RecordSpec(b=_REAL, a=_REAL))

    def test_an_output_spec_equals_only_an_output_spec(self):
        assert _grouped_spec() != _grouped_spec().term_spec
        assert OutputSpec(x=_REAL) != _REAL


class TestALawOverLaws:
    """A law whose event is positional over a law: it has no outputs, and a merge refuses it."""

    def test_it_builds_with_no_outputs(self):
        assert _law_over_laws().event_spec.outputs == {}

    def test_a_merge_refuses_it_with_the_advice_to_wrap(self):
        with pytest.raises(
            TypeError,
            match=r"cannot merge factor 0 \('ℙ'\), since its draws are distributions, which have "
            r"no outputs to read by name\. Wrap them under a name",
        ):
            (_law_over_laws() & Normal("z", 0.0, 1.0))

    def test_wrapped_under_a_name_it_merges(self):
        law_spec = Normal("x", 0.0, 1.0).spec
        wrapped = distribution(
            sample=lambda key: {"params": Normal("x", 0.0, 1.0)},
            event_spec=OutputSpec(RecordSpec(params=law_spec)),
        )
        joint = wrapped & Normal("z", 0.0, 1.0)
        assert list(joint.event_spec.outputs) == ["params", "z"]
        with workflow_run(seed=0):
            draw = sample(joint)
        assert isinstance(draw["params"], Distribution)
        assert draw.keys() == ("params", "z")


class TestOutputSpecCompletion:
    def test_default_exposes_a_record_and_names_any_other_term(self):
        record = RecordSpec(beta=NumericArraySpec(()))
        assert OutputSpec.default(record, output_name="posterior") == OutputSpec(record)
        array = NumericArraySpec((3,))
        assert OutputSpec.default(array, output_name="beta") == OutputSpec(beta=array)

    def test_with_term_spec_fills_an_unspecified_type_under_the_declared_name(self):
        array = NumericArraySpec((3,), jnp.float32)
        assert OutputSpec(beta=None).with_term_spec(array) == OutputSpec(beta=array)

    def test_with_term_spec_keeps_the_produced_spec_when_the_declared_one_unifies(self):
        # The output spec sets only the shape; the producer's dtype stands.
        produced = NumericArraySpec((3,), jnp.float32)
        assert OutputSpec(beta=NumericArraySpec((3,))).with_term_spec(produced) == OutputSpec(
            beta=produced
        )

    def test_with_term_spec_keeps_the_declared_dtype_and_support(self):
        declared = NumericArraySpec(("k",), jnp.float32, positive)
        assert OutputSpec(beta=declared).with_term_spec(NumericArraySpec((3,))) == OutputSpec(
            beta=NumericArraySpec((3,), jnp.float32, positive)
        )
        # A produced dtype that casts to the declared one does not replace it.
        produced = NumericArraySpec((3,), jnp.float64)
        assert OutputSpec(beta=declared).with_term_spec(produced) == OutputSpec(
            beta=NumericArraySpec((3,), jnp.float32, positive)
        )

    def test_with_term_spec_unifies_a_record_field_by_field(self):
        declared = RecordSpec(a=NumericArraySpec((), jnp.float32), b=NumericArraySpec(("k",)))
        produced = RecordSpec(a=NumericArraySpec(()), b=NumericArraySpec((2,), jnp.float32))
        assert OutputSpec(declared).with_term_spec(produced) == OutputSpec(
            RecordSpec(a=NumericArraySpec((), jnp.float32), b=NumericArraySpec((2,), jnp.float32))
        )

    def test_with_term_spec_rejects_a_spec_that_does_not_unify(self):
        with pytest.raises(ValueError, match="has dimension 3, expected 2"):
            OutputSpec(beta=NumericArraySpec((2,))).with_term_spec(NumericArraySpec((3,)))

    def test_with_term_spec_names_the_node_that_does_not_unify_by_its_path(self):
        with pytest.raises(ValueError, match=r"^beta has dimension 3"):
            OutputSpec(beta=NumericArraySpec((2,))).with_term_spec(NumericArraySpec((3,)))
        whole = OutputSpec(params=RecordSpec(a=NumericArraySpec((2,))))
        with pytest.raises(ValueError, match=r"^params/a has dimension 3"):
            whole.with_term_spec(RecordSpec(a=NumericArraySpec((3,))))
        # In the positional form the record's fields are the outputs, so a path starts at a field.
        exposed = OutputSpec(RecordSpec(a=NumericArraySpec((2,))))
        with pytest.raises(ValueError, match=r"^a has dimension 3"):
            exposed.with_term_spec(RecordSpec(a=NumericArraySpec((3,))))
        with pytest.raises(
            ValueError,
            match=r"^fields do not match the declared fields: missing \['a'\], unexpected \['b'\]",
        ):
            exposed.with_term_spec(RecordSpec(b=NumericArraySpec((2,))))

    def test_with_term_spec_keeps_an_exposed_record_exposed(self):
        produced = RecordSpec(a=NumericArraySpec(()), b=NumericArraySpec((2,)))
        declared = OutputSpec(RecordSpec(a=NumericArraySpec(()), b=NumericArraySpec((2,))))
        assert declared.with_term_spec(produced) == OutputSpec(produced)
        with pytest.raises(TypeError, match="needs a RecordSpec"):
            declared.with_term_spec(NumericArraySpec(()))

    def test_a_whole_term_is_renamed_by_its_output(self):
        array = NumericArraySpec(())
        assert OutputSpec(prior=array).rename(prior="beta") == OutputSpec(beta=array)
        assert OutputSpec(prior=None).rename(prior="beta") == OutputSpec(beta=None)
        with pytest.raises(KeyError):
            OutputSpec(prior=array).rename(sigma="s")
        with pytest.raises(KeyError):
            OutputSpec(prior=array).rename({"prior/x": "y"})
        # The output is renamed in place: moving it into a group would
        # change the form, so a path target raises.
        with pytest.raises(ValueError, match="cannot move it into a group"):
            OutputSpec(prior=array).rename(prior="group/beta")

    def test_a_whole_record_renames_its_fields_through_its_output(self):
        record = RecordSpec(beta=NumericArraySpec(()), sigma=NumericArraySpec(()))
        renamed = OutputSpec(parameters=record).rename({"parameters/beta": "parameters/b"})
        assert renamed == OutputSpec(parameters=record.rename(beta="b"))
        assert renamed.rename(parameters="theta") == OutputSpec(theta=record.rename(beta="b"))
        with pytest.raises(KeyError):
            OutputSpec(parameters=record).rename(beta="b")

    def test_an_output_and_a_field_below_it_are_renamed_in_two_calls(self):
        # Each target is a new exact path, so renaming the output to theta
        # and its field to theta/b in one call gives one target inside another.
        record = RecordSpec(beta=NumericArraySpec(()), sigma=NumericArraySpec(()))
        with pytest.raises(ValueError, match="overlap"):
            OutputSpec(parameters=record).rename(
                {"parameters": "theta", "parameters/beta": "theta/b"}
            )

    def test_both_forms_of_one_interface_take_the_same_paths(self):
        inner = RecordSpec(beta=NumericArraySpec(()))
        renames = {"parameters/beta": "parameters/b"}
        renamed_inner = RecordSpec(b=NumericArraySpec(()))
        whole = OutputSpec(parameters=inner).rename(renames)
        exposed = OutputSpec(RecordSpec(parameters=inner)).rename(renames)
        assert whole.rename(parameters="theta") == OutputSpec(theta=renamed_inner)
        assert exposed.rename(parameters="theta") == OutputSpec(RecordSpec(theta=renamed_inner))

    def test_an_output_and_a_field_of_one_name_have_distinct_paths(self):
        record = RecordSpec(beta=NumericArraySpec(()), sigma=NumericArraySpec(()))
        declaration = OutputSpec(beta=record)
        assert declaration.rename(beta="b") == OutputSpec(b=record)
        assert declaration.rename({"beta/beta": "beta/b"}) == OutputSpec(
            beta=record.rename(beta="b")
        )

    def test_a_whole_record_moves_a_field_within_its_output(self):
        record = RecordSpec(a=NumericArraySpec(()), g=RecordSpec(b=NumericArraySpec(())))
        moved = OutputSpec(p=record).rename({"p/a": "p/g/a"})
        assert moved == OutputSpec(
            p=RecordSpec(g=RecordSpec(b=NumericArraySpec(()), a=NumericArraySpec(())))
        )

    @pytest.mark.parametrize(
        "renames",
        [
            pytest.param({"p/a": "a"}, id="a-field-out-of-the-output"),
            pytest.param({"p/a": "p"}, id="a-field-onto-the-output"),
            pytest.param({"p": "q", "p/a": "p/a2"}, id="a-field-under-the-old-output"),
        ],
    )
    def test_a_whole_record_keeps_its_fields_under_its_output(self, renames):
        record = RecordSpec(a=NumericArraySpec(()), b=NumericArraySpec(()))
        with pytest.raises(ValueError, match="must stay under"):
            OutputSpec(p=record).rename(renames)

    def test_a_whole_term_refuses_a_repeated_or_an_empty_rename(self):
        record = RecordSpec(beta=NumericArraySpec(()))
        with pytest.raises(ValueError, match="more than once"):
            OutputSpec(p=record).rename({"p": "q"}, p="r")
        with pytest.raises(ValueError, match="more than once"):
            OutputSpec(p=record).rename({"p/beta": "b"}, **{"p/beta": "c"})
        with pytest.raises(ValueError, match="at least one rename"):
            OutputSpec(p=record).rename()

    def test_an_exposed_record_renames_its_fields(self):
        record = RecordSpec(beta=NumericArraySpec(()), sigma=NumericArraySpec(()))
        renamed = OutputSpec(record).rename(beta="b")
        assert renamed == OutputSpec(record.rename(beta="b"))
        assert _has_record_outputs(renamed)
        nested = OutputSpec(RecordSpec(g=RecordSpec(mu=NumericArraySpec(()))))
        assert tuple(nested.rename({"g/mu": "g/m"}).outputs) == ("g",)
        with pytest.raises(KeyError):
            nested.rename(mu="m")

    def test_an_exposed_record_moves_fields_across_its_outputs(self):
        scalar = NumericArraySpec(())
        declaration = OutputSpec(RecordSpec(g=RecordSpec(mu=scalar, sigma=scalar), y=scalar))
        out = declaration.rename({"g/mu": "mu"})
        assert out == OutputSpec(RecordSpec(g=RecordSpec(sigma=scalar), y=scalar, mu=scalar))
        into = declaration.rename({"y": "g/y"})
        assert into == OutputSpec(RecordSpec(g=RecordSpec(mu=scalar, sigma=scalar, y=scalar)))
        assert _has_record_outputs(into)
        # An output that the moves leave empty is removed.
        emptied = OutputSpec(RecordSpec(g=RecordSpec(mu=scalar))).rename({"g/mu": "m"})
        assert tuple(emptied.outputs) == ("m",)


class TestDeclarationConstruction:
    @pytest.mark.parametrize(
        ("decl", "field"),
        [(InputSpec(x=NumericArraySpec(())), "_inputs"), (OutputSpec(x=None), "_term_spec")],
    )
    def test_declarations_reject_attribute_mutation(self, decl, field):
        original = getattr(decl, field)
        with pytest.raises(FrozenInstanceError):
            setattr(decl, field, original)
        with pytest.raises(FrozenInstanceError):
            delattr(decl, field)
        with pytest.raises(AttributeError):
            _ = decl.unknown_attribute
        with pytest.raises(FrozenInstanceError):
            decl.unknown_attribute = None
        with pytest.raises(FrozenInstanceError):
            del decl.unknown_attribute
        assert getattr(decl, field) is original
        assert not hasattr(decl, "unknown_attribute")

    @pytest.mark.parametrize("name", ["name", "term_spec", "outputs", "record_spec", "inputs"])
    def test_constructor_keywords_are_output_names(self, name):
        spec = NumericArraySpec(())
        assert dict(OutputSpec(**{name: spec}).outputs) == {name: spec}
        assert InputSpec(**{name: spec})[name] is spec

    @pytest.mark.parametrize("name", ["a/b", "", "two words", "1x", "class"])
    def test_input_names_are_python_identifiers(self, name):
        with pytest.raises(ValueError, match="input names must be Python identifiers"):
            InputSpec({name: NumericArraySpec(())})

    @pytest.mark.parametrize("name", ["two words", "1x", "class", "post-1", "*args[0]"])
    def test_an_output_name_follows_the_field_rule(self, name):
        # An output that binds a Python parameter must be an identifier, which
        # binding checks; the output spec accepts any field name.
        spec = NumericArraySpec(())
        assert list(OutputSpec(**{name: spec}).outputs) == [name]
        assert list(OutputSpec(RecordSpec({name: spec})).outputs) == [name]

    @pytest.mark.parametrize("name", ["a/b", ""])
    def test_an_output_name_is_non_empty_without_a_slash(self, name):
        with pytest.raises(ValueError, match="output names must be non-empty"):
            OutputSpec(**{name: NumericArraySpec(())})

    @pytest.mark.parametrize("value", [None, {}, 1, "spec"])
    def test_inputs_require_specs(self, value):
        with pytest.raises(TypeError):
            InputSpec(x=value)

    def test_declarations_do_not_alias_mutable_constructor_mappings(self):
        source = {"x": NumericArraySpec(())}
        inputs = InputSpec(source)
        source["y"] = OpaqueSpec()
        assert list(inputs) == ["x"]
        with pytest.raises(TypeError):
            inputs["x"] = source["x"]
        with pytest.raises(TypeError):
            InputSpec(source, y=OpaqueSpec())
        for output in (OutputSpec(x=source["x"]), OutputSpec(RecordSpec(x=()))):
            with pytest.raises(TypeError):
                output.outputs["x"] = OpaqueSpec()
            with pytest.raises(FrozenInstanceError):
                output._term_spec = OpaqueSpec()


class TestDeclarationRoundTrips:
    @pytest.mark.parametrize(
        "decl",
        [
            InputSpec(x=NumericArraySpec(("n",))),
            OutputSpec(x=None),
            OutputSpec(x=NumericArraySpec(("n",))),
            OutputSpec(RecordSpec(x=("n",))),
            RecordSpec(group=RecordSpec(x=("n",))),
            DistributionSpec(RecordSpec(x=("n",))),
        ],
    )
    def test_declaration_copy_and_pickle(self, decl):
        for restored in (copy.copy(decl), copy.deepcopy(decl), pickle.loads(pickle.dumps(decl))):
            assert restored == decl
            assert hash(restored) == hash(decl)
            assert type(restored) is type(decl)

    def test_declarations_can_be_static_jit_arguments(self):
        @jax.jit(static_argnames="spec")
        def total(x, spec):
            assert spec.term_spec.is_valid(x)
            return jnp.sum(x)

        spec = OutputSpec(x=NumericArraySpec((3,)))
        assert total(jnp.arange(3.0), spec) == 3


class TestDimensionBinding:
    @pytest.mark.parametrize("actual", [None, {"x": NumericArraySpec((3,))}])
    def test_binding_requires_an_actual_term_spec(self, actual):
        with pytest.raises(TypeError, match=r"bind_dims_from_spec\(\) expects a TermSpec, got"):
            RecordSpec(x=(3,)).bind_dims_from_spec(actual)

    @pytest.mark.parametrize("kind", ["array", "batch"])
    def test_a_name_both_sides_declare_is_one_free_dimension(self, kind):
        if kind == "array":
            declared, actual = NumericArraySpec(("n",)), NumericArraySpec(("n",))
        else:
            declared = BatchSpec(OpaqueSpec(), draw="n")
            actual = BatchSpec(OpaqueSpec(), draw="n")
        assert declared.bind_dims_from_spec(actual).free_dims == {"n"}

    @pytest.mark.parametrize("kind", ["array", "batch"])
    def test_two_different_names_at_one_axis_raise(self, kind):
        if kind == "array":
            declared, actual = NumericArraySpec(("n",)), NumericArraySpec(("m",))
        else:
            declared = BatchSpec(OpaqueSpec(), draw="n")
            actual = BatchSpec(OpaqueSpec(), draw="m")
        with pytest.raises(ValueError, match="symbolic dimension 'm' where 'n' is expected"):
            declared.bind_dims_from_spec(actual)

    def test_a_symbol_on_the_other_side_binds_to_this_side_size(self):
        declared = RecordSpec(a=(3,), b=("m",))
        actual = RecordSpec(a=("m",), b=("m",))
        assert declared.bind_dims_from_spec(actual) == RecordSpec(a=(3,), b=(3,))

    def test_a_symbol_bound_on_one_side_constrains_the_other(self):
        with pytest.raises(ValueError, match="already bound to 3"):
            RecordSpec(a=(3,), b=(4,)).bind_dims_from_spec(RecordSpec(a=("m",), b=("m",)))

    @pytest.mark.parametrize(
        ("expected", "actual_spec", "actual_value", "value_error"),
        [
            pytest.param(
                NumericArraySpec((3,)),
                RecordSpec(x=(3,)),
                Record(
                    {"x": np.zeros(3)},
                    label="r",
                ),
                "does not conform",
                id="array-receives-record",
            ),
            pytest.param(
                RecordSpec(x=(3,)),
                NumericArraySpec((3,)),
                np.zeros(3),
                r"must be a record or a mapping of fields, got an array of shape \(3,\)",
                id="record-receives-array",
            ),
        ],
    )
    @pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
    def test_spec_binding_never_unwraps_a_single_field_record(
        self, expected, actual_spec, actual_value, value_error, nested
    ):
        if nested:
            expected = RecordSpec(field=expected)
            actual_spec = RecordSpec(field=actual_spec)
            actual_value = {"field": actual_value}
        with pytest.raises(ValueError, match="does not conform"):
            expected.bind_dims_from_spec(actual_spec)
        with pytest.raises(ValueError, match=value_error):
            expected.bind_dims_from_value(actual_value)

    def test_shared_symbol_scope_across_inputs_and_nested_batch(self):
        inputs = InputSpec(
            data=NumericArraySpec(("n",)),
            batch=BatchSpec(NumericArraySpec(("n",)), draw="n"),
        )
        actual = InputSpec(
            data=NumericArraySpec((3,)),
            batch=BatchSpec(NumericArraySpec((3,)), draw=3),
        )
        assert inputs.bind_dims_from_spec(actual) == actual
        conflict = InputSpec(dict(actual) | {"data": NumericArraySpec((4,))})
        with pytest.raises(ValueError, match="already bound"):
            inputs.bind_dims_from_spec(conflict)
        assert inputs.free_dims == {"n"}

    def test_bind_from_value_and_spec_preserves_declared_metadata(self):
        expected = RecordSpec(x=NumericArraySpec(("n",), dtype="float64"), y=("n",))
        values = {"x": np.zeros(3, dtype="float32"), "y": np.ones(3)}
        bound = expected.bind_dims_from_value(values)
        assert bound == expected.with_dim_sizes(n=3)
        assert expected.bind_dims_from_spec(RecordSpec(x=(3,), y=(3,))) == bound
        assert expected.is_valid(values)
        assert not expected.is_valid(values | {"y": np.ones(4)})
        assert not expected.is_valid({"x": np.zeros(3)})
        assert expected.free_dims == {"n"}

    def test_dimension_renaming_is_simultaneous_and_crosses_all_containers(self):
        spec = RecordSpec(
            data=NumericArraySpec(("n", "m"), dtype="float32"),
            batch=BatchSpec(NumericArraySpec(("n",)), draw="m"),
            law=DistributionSpec(RecordSpec(x=("n",))),
            function=FunctionSpec(
                InputSpec(RecordSpec(x=("m",)).children),
                OutputSpec(result=NumericArraySpec(("n",))),
            ),
        )
        renamed = spec.with_dim_names(n="m", m="n")
        assert renamed["data"].shape == ("m", "n")
        assert renamed["data"].dtype == np.dtype("float32")
        assert renamed["batch"].axis_groups == (("n",),)
        assert renamed["batch"].element_spec.term_spec.shape == ("m",)
        assert renamed["batch"].level_names == ("draw",)
        assert renamed["law"].event_spec.outputs["x"].shape == ("m",)
        assert renamed["function"].input_spec["x"].shape == ("n",)
        assert renamed["function"].output_spec.term_spec.shape == ("m",)
        assert spec["data"].shape == ("n", "m")
        for value in (None, 2):
            with pytest.raises(TypeError):
                spec.with_dim_names(n=value)
        for value in ("", "my dim"):
            with pytest.raises(ValueError, match="must be a Python identifier"):
                spec.with_dim_names(n=value)
        with pytest.raises(ValueError):
            spec.with_dim_sizes(n=-1)

    def test_a_declaration_checks_renames_and_sizes_with_no_spec_to_pass_them_to(self):
        with pytest.raises(ValueError, match="must be a Python identifier"):
            InputSpec({}).with_dim_names(n="a b")
        with pytest.raises(TypeError, match="must be a non-negative int, got bool True"):
            OutputSpec(x=None).with_dim_sizes(n=True)

    def test_binding_array_values_checks_dtype_and_record_kind(self):
        spec = NumericArraySpec(("n",), dtype="int32")
        with pytest.raises(
            ValueError, match="has dtype float32, which cannot be cast to the declared"
        ):
            spec.bind_dims_from_value(np.zeros(3, dtype="float32"))
        record = Record(
            {"x": np.zeros(3)},
            label="record",
        )
        assert not spec.is_valid(record)
        with pytest.raises(ValueError):
            spec.bind_dims_from_value(record)


class TestSpecKinds:
    def test_all_specs_use_one_kind_protocol(self):
        for spec in (NumericArraySpec(()), OpaqueSpec(), RecordSpec(x=()), FunctionSpec()):
            assert isinstance(spec, TermSpec)
        numeric = RecordSpec(group=RecordSpec(x=(2, 3)), y=())
        assert isinstance(numeric, NumericRecordSpec)
        assert isinstance(numeric, NumericSpec)
        assert numeric.vector_size == 7
        assert numeric.leaf_shapes == {"group/x": (2, 3), "y": ()}
        assert NumericArraySpec(()).vector_size == 1
        assert NumericArraySpec((0, 3)).vector_size == 0
        with pytest.raises(ValueError):
            _ = NumericArraySpec(("n",)).vector_size

    def test_record_spec_inference_keeps_stored_term_kinds(self):
        from probpipe import NumericArray, Opaque

        array = NumericArray(
            np.zeros(2),
            spec=NumericArraySpec((2,), dtype="float64"),
            label="array",
        )
        opaque = Opaque(
            object(),
            spec=OpaqueSpec(meta="metadata"),
            label="opaque",
        )
        record = Record(
            {"x": 1.0},
            label="record",
        )
        inferred = RecordSpec.infer_from({"array": array, "opaque": opaque, "record": record})
        assert inferred["array"] is array.spec
        assert inferred["opaque"] is opaque.spec
        assert inferred.children["record"] is record.spec

    def test_record_schema_infers_distribution_and_callable_kinds(self):
        from probpipe import Function, Normal

        law = Normal("x", 0.0, 1.0)
        function = Function(
            lambda x: x,
            label="function",
            input_spec=InputSpec(RecordSpec(x=()).children),
            output_spec=RecordSpec(y=()),
        )
        schema = RecordSpec.infer_from(
            {"law": law, "function": function, "raw_callable": lambda x: x}
        )
        assert schema["law"] == law.spec
        assert schema["function"] == FunctionSpec(
            InputSpec(RecordSpec(x=()).children), OutputSpec(RecordSpec(y=()))
        )
        assert schema["raw_callable"] == FunctionSpec()

    def test_an_empirical_over_opaque_atoms_is_a_distribution_field(self):
        from probpipe import EmpiricalDistribution, OpaqueBatch

        law = EmpiricalDistribution(
            OpaqueBatch(
                ["a", "b"],
                "law",
                label="labels",
            ),
            output_name="law",
        )
        schema = RecordSpec.infer_from({"law": law})
        assert schema["law"] == law.spec
        assert law.event_spec == OutputSpec(law=OpaqueSpec(type=str))

        record = Record(
            {"law": law},
            label="r",
        )
        assert type(record["law"]) is type(law) and record["law"].spec == law.spec
        assert record.spec == schema
        assert schema.is_valid(record)

    def test_numeric_batch_field_keeps_its_batch_kind(self):
        from probpipe import NumericArrayBatch

        batch = NumericArrayBatch(
            np.zeros((2, 3)),
            "draw",
            element=NumericArraySpec((3,)),
            label="draws",
        )
        schema = RecordSpec.infer_from({"batch": batch})
        assert schema["batch"] is batch.spec
        assert type(schema) is RecordSpec
        record = Record(
            {"batch": batch},
            label="container",
        )
        assert record.spec == schema
        assert not NumericArraySpec((2, 3)).is_valid(batch)

    def test_numeric_record_layout_reads_each_numeric_specs_own_layout(self):
        from dataclasses import dataclass

        from probpipe import NumericRecord, NumericRecordBatch

        @dataclass(frozen=True)
        class Coordinates(NumericSpec):
            size: int

            def _vector_size(self):
                return self.size

            def is_valid(self, value):
                return np.shape(value) == (self.size,)

        schema = RecordSpec(x=Coordinates(4), nested=RecordSpec(y=(2, 3)), label=OpaqueSpec())
        numeric = schema.numeric_subset()
        assert isinstance(numeric, NumericRecordSpec)
        assert numeric.vector_size == 10
        assert numeric.leaf_shapes == {"x": (4,), "nested/y": (2, 3)}
        with pytest.raises(TypeError, match=r"field 'x' must have a NumericArraySpec"):
            NumericRecord.from_vector(numeric, np.zeros(10), label="value")
        with pytest.raises(TypeError, match=r"field 'x' must have a NumericArraySpec"):
            NumericRecordBatch.from_vector(
                numeric, np.zeros((2, 10)), level_names="row", label="values"
            )


class TestDistributionSchemaAvailability:
    def test_inference_preserves_an_available_distribution_schema(self, declared_law):
        template = RecordSpec(x=NumericArraySpec((3,), dtype="float64", support=positive))
        law = declared_law(template)
        inferred = RecordSpec.infer_from({"law": law})
        assert inferred["law"] == DistributionSpec(template)
        assert inferred["law"].event_spec.term_spec is template
        record = Record(
            {"law": law},
            label="r",
        )
        assert type(record["law"]) is type(law) and record["law"].spec == law.spec
        assert record.spec == inferred


class TestInputSpec:
    def test_optional_inputs_are_part_of_the_declaration(self):
        array = NumericArraySpec(())
        inputs = InputSpec(a=array, b=array).with_optional("b")
        assert inputs.required == ("a",)
        assert inputs.optional == {"b"}
        assert inputs != InputSpec(a=array, b=array)
        assert hash(inputs) != hash(InputSpec(a=array, b=array))
        assert repr(inputs).endswith(".with_optional('b')")

    def test_an_optional_input_stays_optional(self):
        inputs = InputSpec(a=NumericArraySpec(("n",)), b=NumericArraySpec(())).with_optional("b")
        assert inputs.with_dim_sizes(n=2).optional == {"b"}
        assert inputs.with_dim_names(n="m").optional == {"b"}
        assert inputs.bind_dims_from_value({"a": np.ones(2), "b": 1.0}).optional == {"b"}
        assert inputs.bind_dims_from_spec(inputs.with_dim_sizes(n=2)).optional == {"b"}
        assert inputs.without("a").optional == {"b"}
        assert inputs.without("b").optional == frozenset()

    def test_marking_inputs_optional_accumulates(self):
        array = NumericArraySpec(())
        inputs = InputSpec(a=array, b=array, c=array).with_optional("a").with_optional("c")
        assert inputs.optional == {"a", "c"}
        assert inputs.required == ("b",)

    def test_an_input_spec_with_optional_inputs_pickles(self):
        import pickle

        inputs = InputSpec(a=NumericArraySpec(()), b=NumericArraySpec(())).with_optional("b")
        assert pickle.loads(pickle.dumps(inputs)) == inputs

    def test_marking_an_unknown_input_optional_raises(self):
        with pytest.raises(KeyError, match="'c'"):
            InputSpec(a=NumericArraySpec(())).with_optional("c")

    def test_binding_values_requires_a_mapping(self):
        with pytest.raises(
            TypeError, match=r"InputSpec\.bind_dims_from_value\(\) expects a mapping, got NoneType"
        ):
            InputSpec(x=OpaqueSpec()).bind_dims_from_value(None)

    def test_binding_specs_requires_an_input_spec(self):
        with pytest.raises(
            TypeError, match=r"InputSpec\.bind_dims_from_spec\(\) expects an InputSpec, got"
        ):
            InputSpec(x=OpaqueSpec()).bind_dims_from_spec({"x": OpaqueSpec()})

    def test_inputs_follow_mapping_equality_while_preserving_input_order(self):
        array = NumericArraySpec(())
        left = InputSpec(x=array, y=OpaqueSpec())
        right = InputSpec(y=OpaqueSpec(), x=array)
        assert left == dict(left)
        assert left == right
        assert hash(left) == hash(right)
        assert list(left) == ["x", "y"]
        assert list(right) == ["y", "x"]

    def test_binding_values_preserves_inputs_metadata_and_the_declaration(self):
        from probpipe.core.constraints import positive

        array = NumericArraySpec(("n",), dtype="float64", support=positive)
        inputs = InputSpec(data=array, nested=RecordSpec(x=("n",)))
        bound = inputs.bind_dims_from_value(
            {
                "nested": Record(
                    {"x": np.ones(3)},
                    label="r",
                ),
                "data": np.ones(3, dtype="float32"),
            }
        )
        assert bound == inputs.with_dim_sizes(n=3)
        assert list(bound) == ["data", "nested"]
        assert bound["data"].support is positive
        assert bound["data"].dtype == np.dtype("float64")
        assert inputs.free_dims == {"n"}

    @pytest.mark.parametrize("values", [{}, {"x": np.ones(3), "extra": 1}])
    def test_binding_requires_exact_input_names(self, values):
        inputs = InputSpec(x=NumericArraySpec(("n",)))
        with pytest.raises(ValueError, match="inputs"):
            inputs.bind_dims_from_value(values)
        actual = InputSpec({name: NumericArraySpec(()) for name in values})
        with pytest.raises(ValueError, match="inputs"):
            inputs.bind_dims_from_spec(actual)

    def test_empty_mapping_and_dimension_transforms(self):
        empty = InputSpec()
        assert dict(empty) == {}
        assert empty.bind_dims_from_value({}) == empty
        assert empty.bind_dims_from_spec(InputSpec()) == empty
        assert empty.is_concrete
        inputs = InputSpec(x=NumericArraySpec(("n",)), y=NumericArraySpec(("m",)))
        renamed = inputs.with_dim_names(n="m", m="n")
        assert renamed["x"].shape == ("m",)
        assert renamed["y"].shape == ("n",)
        assert renamed.with_dim_sizes(n=2, m=3) == InputSpec(
            x=NumericArraySpec((3,)), y=NumericArraySpec((2,))
        )


class TestRecordValueValidation:
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("as_record", [False, True])
    @pytest.mark.parametrize("dtype, valid", [("int32", False), ("float64", True)])
    def test_dtype_checks_actual_fields(self, nested, as_record, dtype, valid):
        spec = RecordSpec(x=NumericArraySpec(("n",), dtype=dtype))
        data = {"x": np.zeros(3, dtype="float32")}
        if nested:
            spec = RecordSpec(group=spec)
            data = {"group": data}
        value = (
            Record(
                data,
                label="value",
            )
            if as_record
            else data
        )

        assert spec.is_valid(value) is valid
        if valid:
            assert spec.bind_dims_from_value(value) == spec.with_dim_sizes(n=3)
        else:
            with pytest.raises(ValueError, match="has dtype float32, which cannot be cast"):
                spec.bind_dims_from_value(value)
        assert spec.free_dims == {"n"}

    def test_a_spec_in_a_value_field_does_not_stand_in_for_an_array(self):
        spec = RecordSpec(x=(3,))
        values = {"x": NumericArraySpec((3,))}
        assert not spec.is_valid(values)
        with pytest.raises(ValueError, match="does not conform"):
            spec.bind_dims_from_value(values)

    def test_spec_binding_uses_only_declared_dtype_information(self):
        spec = RecordSpec(x=NumericArraySpec(("n",), dtype="int32"))
        assert spec.bind_dims_from_spec(RecordSpec(x=(3,))) == spec.with_dim_sizes(n=3)

    def test_validation_and_binding_work_under_jit_and_vmap(self):
        spec = RecordSpec(x=NumericArraySpec(("n",), dtype="float32"))

        @jax.jit
        def total(x):
            record = Record(
                {"x": x},
                label="row",
            )
            assert spec.is_valid(record)
            assert spec.bind_dims_from_value(record) == spec.with_dim_sizes(n=3)
            return jnp.sum(record["x"])

        rows = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
        np.testing.assert_allclose(jax.vmap(total)(rows), [3.0, 12.0], rtol=0, atol=0)


class TestNestedValueBinding:
    @pytest.fixture(params=["direct", "record", "nested_record", "input"])
    def wrap_binding(self, request):
        def wrap(spec, value):
            match request.param:
                case "direct":
                    return spec, value
                case "record":
                    return RecordSpec(value=spec), {"value": value}
                case "nested_record":
                    return RecordSpec(group=RecordSpec(value=spec)), {"group": {"value": value}}
                case "input":
                    return InputSpec(value=spec), {"value": value}

        return wrap

    @pytest.mark.parametrize("kind", ["function", "distribution"])
    @pytest.mark.parametrize("size", [3, 4])
    def test_fixed_and_concretized_specs_follow_the_same_binding_rules(
        self, wrap_binding, kind, size
    ):
        from probpipe import EmpiricalDistribution, Function, NumericRecordBatch

        if kind == "function":

            def spec_type(spec):
                return FunctionSpec(InputSpec(spec.children))

            reference = Function(
                lambda x: x, label="function", input_spec=InputSpec(RecordSpec(x=(3,)).children)
            )
            actual = Function(
                lambda x: x,
                label="function",
                input_spec=InputSpec(RecordSpec(x=(size,)).children),
            )
        else:
            spec_type = DistributionSpec

            def law(width):
                atoms = NumericRecordBatch(
                    {"x": np.zeros((2, width))},
                    "row",
                    element=RecordSpec(x=(width,)),
                    label="rows",
                )
                return EmpiricalDistribution(atoms, label="x")

            reference = law(3)
            actual = law(size)

        # Both kinds bind by unification; function binding reads the available
        # input and output specs without checking callable compatibility.
        dtype = "float64" if kind == "function" else None
        symbolic = spec_type(RecordSpec(x=NumericArraySpec(("n",), dtype=dtype)))
        fixed = spec_type(RecordSpec(x=NumericArraySpec((3,), dtype=dtype)))
        declared, value = wrap_binding(symbolic, actual)
        expected, _ = wrap_binding(symbolic.with_dim_sizes(n=size), actual)
        bound = declared.bind_dims_from_value(value)
        assert bound == expected
        assert bound.bind_dims_from_value(value) == bound

        for spec in (fixed, symbolic.with_dim_sizes(n=3), symbolic.bind_dims_from_value(reference)):
            assert spec == fixed
            declared, value = wrap_binding(spec, actual)
            if size == 3:
                assert declared.bind_dims_from_value(value) == declared
            else:
                with pytest.raises(ValueError, match="dimension 4, expected 3"):
                    declared.bind_dims_from_value(value)
        assert symbolic.free_dims == {"n"}

    @pytest.mark.parametrize(
        ("declared", "actual", "matches"),
        [
            pytest.param(
                NumericArraySpec((3,)),
                NumericArraySpec((3,), dtype="int32"),
                True,
                id="unset-dtype",
            ),
            pytest.param(
                NumericArraySpec((3,), dtype="float64"),
                NumericArraySpec((3,), dtype="float32"),
                True,
                id="same-kind-float",
            ),
            pytest.param(
                NumericArraySpec((3,), dtype="int32"),
                NumericArraySpec((3,), dtype="float32"),
                False,
                id="float-against-int",
            ),
            pytest.param(
                NumericArraySpec((3,), dtype="float32"),
                NumericArraySpec((3,), dtype="int32"),
                True,
                id="same-kind-int-to-float",
            ),
            pytest.param(
                NumericArraySpec((3,), support=positive), NumericArraySpec((3,)), True, id="support"
            ),
            pytest.param(
                NumericArraySpec((3,)),
                NumericArraySpec((3,), support=positive),
                True,
                id="support-on-the-law",
            ),
            pytest.param(NumericArraySpec((3,)), NumericArraySpec((4,)), False, id="size"),
        ],
    )
    def test_concrete_distribution_values_match_by_unification(
        self, wrap_binding, declared_law, declared, actual, matches
    ):
        """A law matches when its event spec unifies with the declared one.

        An unset dtype accepts any dtype and a set one a same-kind cast, sizes must
        agree, and support is not compared.
        """
        law = declared_law(RecordSpec(x=actual))
        spec = DistributionSpec(RecordSpec(x=declared))
        schema = RecordSpec(law=spec)
        declared_spec, value = wrap_binding(spec, law)
        assert spec.is_valid(law) is matches
        assert schema.is_valid({"law": law}) is matches
        if matches:
            assert declared_spec.bind_dims_from_value(value) == declared_spec
            assert schema.is_valid(
                Record(
                    {"law": law},
                    event_template=schema,
                    label="value",
                )
            )
        else:
            with pytest.raises(ValueError):
                declared_spec.bind_dims_from_value(value)
            with pytest.raises(ValueError):
                Record(
                    {"law": law},
                    event_template=schema,
                    label="value",
                )

    def test_a_law_with_other_field_order_does_not_match(self, wrap_binding, declared_law):
        law = declared_law(RecordSpec(y=(), x=(3,)))
        spec = DistributionSpec(RecordSpec(x=(3,), y=()))
        schema = RecordSpec(law=spec)
        assert not spec.is_valid(law)
        declared, value = wrap_binding(spec, law)
        with pytest.raises(ValueError, match="not in the declared order"):
            declared.bind_dims_from_value(value)
        assert not schema.is_valid({"law": law})

    def test_fields_sharing_a_dimension_bind_it_once(self, declared_law):
        spec = DistributionSpec(RecordSpec(x=("n",), y=("n",)))
        agreeing = declared_law(RecordSpec(x=(3,), y=(3,)))
        assert spec.bind_dims_from_value(agreeing) == DistributionSpec(RecordSpec(x=(3,), y=(3,)))
        with pytest.raises(ValueError, match="already bound"):
            spec.bind_dims_from_value(declared_law(RecordSpec(x=(3,), y=(4,))))

    def test_concretized_distribution_binds_like_a_fixed_one(self, wrap_binding, declared_law):
        symbolic = DistributionSpec(RecordSpec(x=NumericArraySpec(("n",), dtype="float64")))
        law = declared_law(RecordSpec(x=NumericArraySpec((3,), dtype="float32")))
        bound = symbolic.bind_dims_from_value(law)
        fixed = DistributionSpec(RecordSpec(x=NumericArraySpec((3,), dtype="float64")))
        for spec in (bound, symbolic.with_dim_sizes(n=3), fixed):
            assert spec == fixed
            declared, value = wrap_binding(spec, law)
            assert declared.bind_dims_from_value(value) == declared

    def test_missing_callable_declarations_remain_unspecified(self, wrap_binding):
        symbolic = FunctionSpec(InputSpec(RecordSpec(x=("n",)).children))
        for spec in (symbolic, symbolic.with_dim_sizes(n=3)):
            declared, value = wrap_binding(spec, lambda x: x)
            assert declared.bind_dims_from_value(value) == declared
        assert symbolic.free_dims == {"n"}


class TestNestedSpecBinding:
    @pytest.fixture(params=["direct", "record", "nested_record", "input", "batch"])
    def wrap_spec(self, request):
        def wrap(spec):
            match request.param:
                case "direct":
                    return spec
                case "record":
                    return RecordSpec(value=spec)
                case "nested_record":
                    return RecordSpec(group=RecordSpec(value=spec))
                case "input":
                    return InputSpec(value=spec)
                case "batch":
                    return BatchSpec(spec, draw=2)

        return wrap

    @pytest.mark.parametrize(
        "expected, actual, result",
        [
            (
                FunctionSpec(),
                FunctionSpec(output_spec=OutputSpec(result=NumericArraySpec((3,)))),
                FunctionSpec(),
            ),
            (
                FunctionSpec(InputSpec(RecordSpec(x=("n",)).children)),
                FunctionSpec(),
                FunctionSpec(InputSpec(RecordSpec(x=("n",)).children)),
            ),
            (
                FunctionSpec(output_spec=OutputSpec(result=NumericArraySpec(("n",)))),
                FunctionSpec(output_spec=OutputSpec(result=NumericArraySpec((3,)))),
                FunctionSpec(output_spec=OutputSpec(result=NumericArraySpec((3,)))),
            ),
            (
                NumericArraySpec((3,)),
                NumericArraySpec((3,), dtype="float32"),
                NumericArraySpec((3,)),
            ),
        ],
    )
    def test_binding_is_independent_of_nesting(self, wrap_spec, expected, actual, result):
        declared = wrap_spec(expected)
        assert declared.bind_dims_from_spec(wrap_spec(actual)) == wrap_spec(result)
        assert declared == wrap_spec(expected)

    @pytest.mark.parametrize(
        "expected, actual",
        [
            (FunctionSpec(), OpaqueSpec()),
            (
                FunctionSpec(output_spec=OutputSpec(result=NumericArraySpec((2,)))),
                FunctionSpec(output_spec=OutputSpec(result=NumericArraySpec((3,)))),
            ),
            (NumericArraySpec((3,), dtype="int32"), NumericArraySpec((3,), dtype="float32")),
            (NumericArraySpec((3,)), RecordSpec(x=(3,))),
            (DistributionSpec(RecordSpec(x=(3,))), OpaqueSpec()),
            (DistributionSpec(RecordSpec(x=(3,))), RecordSpec(x=(3,))),
            (DistributionSpec(RecordSpec(x=(3,))), DistributionSpec(RecordSpec(x=(4,)))),
            (
                DistributionSpec(RecordSpec(x=NumericArraySpec((3,), dtype="int32"))),
                DistributionSpec(RecordSpec(x=NumericArraySpec((3,), dtype="float32"))),
            ),
        ],
    )
    def test_mismatches_are_rejected_at_every_nesting_depth(self, wrap_spec, expected, actual):
        with pytest.raises(ValueError):
            wrap_spec(expected).bind_dims_from_spec(wrap_spec(actual))

    @pytest.mark.parametrize("size", [3, "n"])
    @pytest.mark.parametrize("dtype", [None, "float32"])
    def test_distribution_spec_binding_preserves_declared_order_and_metadata(
        self, wrap_spec, size, dtype
    ):
        expected = DistributionSpec(
            RecordSpec(x=NumericArraySpec((size,), dtype="float64", support=positive), y=())
        )
        actual = DistributionSpec(RecordSpec(x=NumericArraySpec((3,), dtype=dtype), y=()))
        declared = wrap_spec(expected)
        bound = declared.bind_dims_from_spec(wrap_spec(actual))
        assert bound == wrap_spec(expected.with_dim_sizes(n=3))
        assert bound.bind_dims_from_spec(wrap_spec(actual)) == bound
        assert declared == wrap_spec(expected)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_distribution_spec_binding_shares_dimensions(self, wrap_spec, reverse):
        expected = DistributionSpec(RecordSpec(x=("n",), nested=RecordSpec(y=("n",))))
        fields = [("x", NumericArraySpec((3,))), ("nested", RecordSpec(y=(4,)))]
        if reverse:
            fields.reverse()
        message = "not in the declared order" if reverse else "already bound"
        with pytest.raises(ValueError, match=message):
            wrap_spec(expected).bind_dims_from_spec(
                wrap_spec(DistributionSpec(RecordSpec(dict(fields))))
            )

    @pytest.mark.parametrize("reverse", [False, True])
    def test_input_value_binding_shares_dimensions_in_either_order(self, reverse):
        fields = [("x", NumericArraySpec(("n",))), ("group", RecordSpec(y=("n",)))]
        if reverse:
            fields.reverse()
        inputs = InputSpec(dict(fields))
        with pytest.raises(ValueError, match="already bound"):
            inputs.bind_dims_from_value({"x": np.zeros(3), "group": {"y": np.zeros(4)}})


class TestDeclarationMessages:
    def test_a_positional_array_spec_is_told_to_take_a_keyword(self):
        with pytest.raises(TypeError, match=r"cannot take a NumericArraySpec positionally; name"):
            OutputSpec(NumericArraySpec(()))

    def test_a_shape_tuple_input_is_an_array_spec(self):
        assert InputSpec(x=(3,))["x"] == NumericArraySpec((3,))

    def test_an_unknown_optional_input_lists_the_inputs(self):
        with pytest.raises(KeyError, match=r"unknown input 'y'; available inputs: \['x'\]"):
            InputSpec(x=NumericArraySpec(())).with_optional("y")

    def test_binding_values_names_the_missing_and_unexpected_inputs(self):
        spec = InputSpec(x=OpaqueSpec(), y=OpaqueSpec())
        with pytest.raises(ValueError, match=r"missing \['y'\], unexpected \['z'\]$"):
            spec.bind_dims_from_value({"x": 1.0, "z": 2.0})
