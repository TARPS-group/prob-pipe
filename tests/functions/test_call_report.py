"""The report ``check`` returns: its repr, its registry listing, and its notebook table."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import jax.numpy as jnp
import pytest

from probpipe import MultivariateNormal, Normal, function
from probpipe.core._dispatch import MethodInfo
from probpipe.core._repr import WIDTH
from probpipe.functions._call import CallReport
from probpipe.operations._condition import condition_on
from probpipe.operations._evaluate import evaluate
from probpipe.operations._moments import expectation, mean


@function
def _shifted(mu, tau, z):
    return mu + tau * z


def _evaluate_report() -> CallReport:
    return evaluate.check(_shifted, Normal("mu", 0.0, 5.0), fixed_args={"tau": 3.0, "z": 0.5})


def _rows(report: CallReport) -> list[list[str]]:
    """The cells of each body row of the report's table."""
    root = ElementTree.fromstring(report._repr_html_())
    return [
        [cell.text or "" for cell in row] for row in root.iter("tr") if row.find("td") is not None
    ]


class TestTheRepr:
    def test_it_reads_as_a_constructor_call_of_the_fields_it_sets(self):
        report = mean.check(MultivariateNormal("z", jnp.zeros(2), jnp.eye(2)))

        text = repr(report)

        assert text.startswith(
            "CallReport(\n    routes=(MethodInfo(True, method_name='closed_form'"
        )
        assert "deferred=" not in text and "lifted=" not in text

    def test_no_line_passes_the_width_but_a_long_reason(self):
        lines = repr(condition_on.check(Normal("a", 0.0, 1.0) * Normal("y", 0.0, 1.0), {"y": 1.0}))

        assert all(len(line) <= WIDTH or "description=" in line for line in lines.splitlines())


class TestTheRegistryMethods:
    def test_routes_that_share_a_registry_list_its_methods_once(self):
        report = condition_on.check(Normal("a", 0.0, 1.0) * Normal("y", 0.0, 1.0), {"y": 1.0})

        (routes,) = report.methods
        assert routes == "curry, slice, bayes"
        assert "nutpie_nuts" in report.methods[routes]
        assert not any("Available" in info.description for info in report.routes)

    def test_a_route_entry_names_the_rule_its_registry_selects(self):
        report = _evaluate_report()

        assert [info.method_name for info in report.routes] == [
            "evaluation_rules (exact methods)",
            "evaluation_rules/sampling_lift",
        ]
        assert report.selected.method_name == "evaluation_rules/sampling_lift"
        assert report.methods["evaluation_rules"][-1] == "sampling_lift"

    def test_an_entry_that_selects_nothing_reads_at_the_exactness_it_covers(self):
        report = expectation.check(Normal("mu", 0.0, 5.0), lambda x: x**2)

        exact_methods = report.routes[1]
        assert exact_methods.method_name == "evaluation_rules (exact methods)"
        assert exact_methods.feasible is False and exact_methods.exact is True


class TestTheNotebookTable:
    def test_the_html_is_well_formed_with_one_row_per_route(self):
        report = _evaluate_report()

        rows = _rows(report)

        assert [row[1:3] for row in rows[:2]] == [
            ["evaluation_rules", "exact methods"],
            ["evaluation_rules", "sampling_lift"],
        ]

    def test_the_selected_route_is_marked(self):
        rows = _rows(_evaluate_report())

        assert [row[0] for row in rows[:2]] == ["", "selected"]
        assert rows[1][3:5] == ["approximate", "feasible"]

    def test_a_reason_is_escaped(self):
        report = CallReport(
            routes=(MethodInfo(False, "a < b & c", method_name="r", exact=True),),
            selected=MethodInfo(False, "no route applies"),
        )

        markup = report._repr_html_()

        assert "a &lt; b &amp; c" in markup
        assert _rows(report)[0][5] == "a < b & c"

    def test_pending_and_deferred_entries_are_rows_with_their_state(self):
        report = CallReport(
            routes=(MethodInfo(None, pending=("the event shape",), method_name="r", exact=False),),
            deferred=("the result's type is completed from the returned value",),
        )

        rows = _rows(report)

        assert rows[0][4:] == ["unresolved", "pending: the event shape"]
        assert rows[1][4:] == ["deferred", "the result's type is completed from the returned value"]

    @pytest.mark.parametrize("term", ["lifted", "methods of evaluation_rules"])
    def test_the_section_below_lists_the_lifted_arguments_and_the_methods(self, term):
        report = _shifted.check(Normal("mu", 0.0, 5.0), 3.0, 0.5)
        evaluated = _evaluate_report()

        markup = report._repr_html_() + evaluated._repr_html_()

        assert f"<b>{term}</b>" in markup
