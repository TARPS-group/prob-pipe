"""A record of summaries displays as a table (II.4)."""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe
from probpipe import (
    DistributionBatch,
    Gamma,
    Normal,
    NumericRecordBatch,
    Poisson,
    Record,
    conditional_distribution,
    mean,
    quantile,
    standard_deviation,
    summarize,
    workflow_run,
)
from probpipe.core._table import COMPUTATIONS
from probpipe.core.config import ProvenanceMode
from probpipe.core.provenance import Provenance


@pytest.fixture
def model():
    return (Normal("mu", 0.0, 2.0) & Gamma("tau", 2.0, 1.0)).with_label("model")


@pytest.fixture
def moments(model):
    return Record({"mean": mean(model), "sd": standard_deviation(model)}, label="moments")


class TestARecordOfSummariesIsATable:
    def test_a_record_built_by_hand_shows_a_row_per_path_and_a_column_per_field(self, moments):
        assert repr(moments) == (
            "moments\n"
            "       mean         sd\n"
            "  mu     0.  2.\n"
            "  tau    2.  1.4142135\n"
            "mean, sd: exact."
        )

    def test_the_tables_lines_are_indented_as_a_records_lines_are(self, model):
        table = Record({"mean": mean(model)}, label="moments")
        lines = Record({"mu": mean(model["mu"])}, label="lines")

        assert all(line.startswith("  ") for line in repr(table).splitlines()[1:-1])
        assert repr(lines).splitlines()[1].startswith("  mu")

    def test_str_is_the_table_without_its_label_and_footer(self, moments):
        assert str(moments) == "     mean         sd\nmu     0.  2.\ntau    2.  1.4142135"

    def test_numpys_print_options_format_each_column(self, model):
        record = Record({"sd": standard_deviation(model)}, label="spread")

        with np.printoptions(precision=2):
            assert str(record) == "       sd\nmu   2.\ntau  1.41"

    def test_an_array_path_takes_a_row_per_element(self):
        law = Normal("y", jnp.array([1.0, 2.0]), 1.0) & Normal("a", 0.5, 2.0)

        summary = summarize(law, mean, {"sd": standard_deviation})

        assert str(summary) == "      mean  sd\ny[0]   1.   1.\ny[1]   2.   1.\na      0.5  2."

    def test_a_matrix_path_names_each_element_by_its_indices(self):
        summary = summarize(Normal("x", jnp.eye(2), 1.0), mean)

        assert [line.split("  ")[0] for line in str(summary).splitlines()[1:]] == [
            "x[0, 0]",
            "x[0, 1]",
            "x[1, 0]",
            "x[1, 1]",
        ]

    def test_a_long_array_path_is_shortened_like_an_array(self):
        summary = summarize(
            Normal("y", jnp.arange(10.0), 1.0), {"m": mean, "s": standard_deviation}
        )

        with np.printoptions(threshold=5, edgeitems=2):
            assert str(summary) == (
                "        m    s\n"
                "y[0]   0.   1.\n"
                "y[1]   1.   1.\n"
                "...   ...  ...\n"
                "y[8]   8.   1.\n"
                "y[9]   9.   1."
            )

    def test_a_column_of_plain_values_sits_beside_the_summaries(self, model):
        record = Record({"mean": mean(model), "truth": {"mu": 0.1, "tau": 2.1}}, label="check")

        assert repr(record) == (
            "check\n       mean  truth\n  mu     0.    0.1\n  tau    2.    2.1\nmean: exact."
        )

    def test_the_table_is_an_html_table_in_a_notebook(self, model):
        record = Record({"mean": mean(model), "sd": standard_deviation(model)}, label="a<b")

        html = record._repr_html_()

        assert html.startswith("<div><b>a&lt;b</b><table>")
        assert "<thead><tr><th></th><th>mean</th><th>sd</th></tr></thead>" in html
        assert '<tr><th style="text-align: left">tau</th><td>2.</td>' in html
        assert html.endswith("<p>mean, sd: exact.</p></div>")

    def test_inside_a_trace_the_repr_is_the_summary(self, moments):
        shown = []

        def show(record):
            shown.append((repr(record), record._repr_html_()))
            return record

        jax.jit(show)(moments)

        assert shown == [
            (
                "NumericRecord('<no description>', "
                "fields=('mean/mu', 'mean/tau', 'sd/mu', 'sd/tau'))",
                None,
            )
        ]


class TestARecordThatIsNotOfSummariesShowsALinePerPath:
    def test_records_of_plain_values_show_a_line_per_path(self):
        record = Record({"a": {"x": 1.0}, "b": {"x": 2.0}}, label="r")

        assert repr(record) == "NumericRecord('r')\n  a/x   1.\n  b/x   2."

    def test_a_record_of_one_group_shows_a_line_per_path(self):
        record = Record({"population": {"mu": 0.0, "tau": 1.0}}, label="draw")

        assert repr(record) == (
            "NumericRecord('draw')\n  population/mu    0.\n  population/tau   1."
        )

    def test_the_mean_of_a_law_whose_groups_share_field_names_shows_a_line_per_path(self):
        group_a = (Normal("mu", 0.0, 1.0) & Normal("sigma", 1.0, 1.0)).rename(
            {"mu": "a/mu", "sigma": "a/sigma"}
        )
        group_b = (Normal("mu", 2.0, 1.0) & Normal("sigma", 3.0, 1.0)).rename(
            {"mu": "b/mu", "sigma": "b/sigma"}
        )

        lines = repr(mean((group_a & group_b).with_label("groups"))).splitlines()

        assert [line.split()[0] for line in lines[1:]] == ["a/mu", "a/sigma", "b/mu", "b/sigma"]

    def test_records_over_different_paths_show_a_line_per_path(self, model):
        record = Record({"a": mean(model), "b": {"y": 2.0, "z": 1.0}}, label="r")

        assert repr(record).startswith("NumericRecord('r')\n  a/mu")

    def test_records_with_different_shapes_at_a_path_show_a_line_per_path(self, model):
        record = Record({"a": mean(model), "b": {"mu": jnp.ones(2), "tau": 1.0}}, label="r")

        assert repr(record).startswith("NumericRecord('r')\n  a/mu")

    def test_a_record_with_a_field_that_is_not_a_record_shows_a_line_per_path(self, model):
        record = Record({"a": mean(model), "b": 2.0}, label="r")

        assert repr(record).startswith("NumericRecord('r')\n  a/mu")

    def test_without_provenance_a_record_of_summaries_shows_a_line_per_path(
        self, model, monkeypatch
    ):
        monkeypatch.setattr(probpipe.config.provenance, "mode", ProvenanceMode.OFF)

        summary = summarize(model, mean)

        assert repr(summary) == (
            "NumericRecord('summary of model(mu, tau)')\n  mean/mu    0.\n  mean/tau   2."
        )


class TestTheFooterStatesHowEachColumnWasComputed:
    def test_a_monte_carlo_column_states_its_draws(self):
        law = Normal("mu", 0.0, 2.0) & Poisson("count", 4.0)
        draws = quantile.options["num_samples"]

        with workflow_run(seed=0):
            record = Record({"mean": mean(law), "median": quantile(law, 0.5)}, label="r")

        assert repr(record).splitlines()[-1] == (
            f"mean: exact. median: Monte Carlo, {draws:,} draws."
        )

    def test_an_approximate_column_that_records_no_draws_reads_approximate(self):
        # The standard deviation of a law with a kernel takes the variance in a
        # nested call, whose draws its own provenance does not record.
        def y_given_mu(mu: jax.Array) -> Normal:
            return Normal("y", jnp.full(2, mu), 1.0)

        prior = Normal("mu", 0.0, 2.0)
        kernel = conditional_distribution(y_given_mu, given_spec=prior.event_spec.outputs)
        with workflow_run(seed=0):
            record = Record({"sd": standard_deviation(kernel & prior)}, label="r")

        assert repr(record).splitlines()[-1] == "sd: approximate."

    def test_a_record_whose_provenance_describes_its_columns_reads_that(self, model):
        record = Record({"mean": mean(model)}, label="r").with_provenance(
            Provenance.create("check", metadata={COMPUTATIONS: {"mean": "from the draws"}})
        )

        assert repr(record).splitlines()[-1] == "mean: from the draws."


def _batch_of_laws(*locs: float) -> DistributionBatch:
    return DistributionBatch([Normal("x", float(loc), 1.0) for loc in locs], "law", label="b")


def _titles(display: str) -> list[str]:
    """The element titles of a batch's repr, which are the lines indented by two spaces."""
    return [line[2:] for line in display.splitlines() if re.match(r"  \S", line)]


class TestABatchOfSummariesShowsATablePerElement:
    def test_each_element_is_a_table_titled_by_its_levels(self):
        summary = summarize(_batch_of_laws(0.0, 1.0), {"m": mean, "s": standard_deviation})

        assert repr(summary) == (
            "NumericRecordBatch('summary of b(x) over law', levels={'law': 2})\n"
            "  law=0\n"
            "        m   s\n"
            "    x  0.  1.\n"
            "  law=1\n"
            "        m   s\n"
            "    x  1.  1.\n"
            "m, s: exact."
        )

    def test_elements_computed_differently_each_state_how(self):
        laws = DistributionBatch([Normal("x", 0.0, 1.0), Poisson("x", 4.0)], "law", label="b")
        draws = quantile.options["num_samples"]

        with workflow_run(seed=0):
            lines = repr(summarize(laws, {"median": lambda d: quantile(d, 0.5)})).splitlines()

        assert lines[4] == "    median: exact."
        assert lines[-1] == f"    median: Monte Carlo, {draws:,} draws."

    def test_str_is_the_tables_without_the_head_and_footer(self):
        summary = summarize(_batch_of_laws(0.0, 1.0), {"m": mean})

        assert str(summary) == "law=0\n    m\nx  0.\nlaw=1\n    m\nx  1."

    def test_a_long_batch_shows_its_first_and_last_elements(self):
        summary = summarize(_batch_of_laws(*range(6)), {"m": mean})

        with np.printoptions(threshold=4, edgeitems=1):
            display = repr(summary)

        assert _titles(display) == ["law=0", "...", "law=5"]

    def test_a_batch_is_shortened_when_its_rows_number_more_than_the_threshold(self):
        # Three elements of two rows each are six rows, more than the threshold of five.
        laws = DistributionBatch(
            [Normal("y", jnp.full(2, float(loc)), 1.0) for loc in range(3)], "law", label="b"
        )

        with np.printoptions(threshold=5, edgeitems=1):
            display = repr(summarize(laws, {"m": mean}))

        assert _titles(display) == ["law=0", "...", "law=2"]

    def test_an_element_of_several_levels_is_titled_by_each(self):
        described = {COMPUTATIONS: {"m": ["exact"] * 4}}
        batch = NumericRecordBatch(
            {"m/x": jnp.arange(4.0).reshape(2, 2)}, ("chain", "draw"), label="b"
        ).with_provenance(Provenance.create("check", metadata=described))

        assert _titles(repr(batch)) == [
            "chain=0, draw=0",
            "chain=0, draw=1",
            "chain=1, draw=0",
            "chain=1, draw=1",
        ]

    def test_the_tables_are_html_tables_in_a_notebook(self):
        summary = summarize(_batch_of_laws(0.0, 1.0), {"m": mean, "s": standard_deviation})

        html = summary._repr_html_()

        assert html.count("<table>") == 2
        assert "<p><b>law=1</b></p>" in html
        assert html.count("<p>m, s: exact.</p>") == 2

    def test_a_batch_of_plain_records_shows_its_columns(self):
        batch = NumericRecordBatch(
            {"m/x": jnp.array([0.0, 1.0]), "s/x": jnp.array([2.0, 3.0])}, "law", label="b"
        )

        assert repr(batch).splitlines()[1:] == ["  m/x   [0. 1.]", "  s/x   [2. 3.]"]
