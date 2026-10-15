"""A record whose top-level fields are records over the same paths displays as a table (II.4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Gamma,
    Normal,
    NumericRecordBatch,
    Poisson,
    Record,
    conditional_distribution,
    mean,
    quantile,
    standard_deviation,
    workflow_run,
)


@pytest.fixture
def model():
    return (Normal("mu", 0.0, 2.0) * Gamma("tau", 2.0, 1.0)).with_label("model")


class TestARecordOfRecordsIsATable:
    def test_a_record_built_by_hand_shows_a_row_per_path_and_a_column_per_field(self, model):
        record = Record({"mean": mean(model), "sd": standard_deviation(model)}, label="moments")

        assert repr(record) == (
            "moments\n     mean         sd\nmu     0.  2.\ntau    2.  1.4142135\nmean, sd: exact."
        )

    def test_str_is_the_table_without_its_label_and_footer(self, model):
        record = Record({"mean": mean(model), "sd": standard_deviation(model)}, label="moments")

        assert str(record) == ("     mean         sd\nmu     0.  2.\ntau    2.  1.4142135")

    def test_numpys_print_options_format_each_column(self, model):
        record = Record({"sd": standard_deviation(model)}, label="spread")

        with np.printoptions(precision=2):
            assert str(record) == "       sd\nmu   2.\ntau  1.41"

    def test_an_array_path_takes_a_row_per_element(self):
        record = Record(
            {"mean": {"y": jnp.array([1.0, 2.0]), "a": 0.5}, "sd": {"y": jnp.ones(2), "a": 2.0}},
            label="hand",
        )

        assert str(record) == "      mean  sd\ny[0]   1.   1.\ny[1]   2.   1.\na      0.5  2."

    def test_a_matrix_path_names_each_element_by_its_indices(self):
        record = Record({"m": {"x": jnp.eye(2)}}, label="hand")

        assert [line.split("  ")[0] for line in str(record).splitlines()[1:]] == [
            "x[0, 0]",
            "x[0, 1]",
            "x[1, 0]",
            "x[1, 1]",
        ]

    def test_a_long_array_path_is_shortened_like_an_array(self):
        record = Record({"m": {"y": jnp.arange(10.0)}, "s": {"y": jnp.ones(10)}}, label="hand")

        with np.printoptions(threshold=5, edgeitems=2):
            assert str(record) == (
                "        m    s\n"
                "y[0]   0.   1.\n"
                "y[1]   1.   1.\n"
                "...   ...  ...\n"
                "y[8]   8.   1.\n"
                "y[9]   9.   1."
            )

    def test_records_over_different_paths_show_one_line_per_path(self):
        record = Record({"a": {"x": 1.0}, "b": {"y": 2.0}}, label="r")

        assert repr(record) == "NumericRecord('r')\n  a/x   1.\n  b/y   2."

    def test_records_with_different_shapes_at_a_path_show_one_line_per_path(self):
        record = Record({"a": {"x": jnp.ones(2)}, "b": {"x": 2.0}}, label="r")

        assert repr(record).startswith("NumericRecord('r')\n  a/x")

    def test_a_record_with_a_field_that_is_not_a_record_shows_one_line_per_path(self):
        record = Record({"a": {"x": 1.0}, "b": 2.0}, label="r")

        assert repr(record) == "NumericRecord('r')\n  a/x   1.\n  b     2."

    def test_the_table_is_an_html_table_in_a_notebook(self, model):
        record = Record({"mean": mean(model), "sd": standard_deviation(model)}, label="a<b")

        html = record._repr_html_()

        assert html.startswith("<div><b>a&lt;b</b><table>")
        assert "<thead><tr><th></th><th>mean</th><th>sd</th></tr></thead>" in html
        assert '<tr><th style="text-align: left">tau</th><td>2.</td>' in html
        assert html.endswith("<p>mean, sd: exact.</p></div>")

    def test_inside_a_trace_the_repr_is_the_summary(self):
        shown = []

        def show(record):
            shown.append((repr(record), record._repr_html_()))
            return record

        jax.jit(show)(Record({"a": {"x": 1.0}, "b": {"x": 2.0}}, label="r"))

        assert shown == [("NumericRecord('<no description>', fields=('a/x', 'b/x'))", None)]


class TestTheFooterReadsEachColumnsProvenance:
    def test_columns_without_provenance_give_no_footer(self):
        record = Record({"a": {"x": 1.0}, "b": {"x": 2.0}}, label="r")

        assert repr(record) == "r\n    a   b\nx  1.  2."

    def test_a_monte_carlo_column_states_its_draws(self):
        law = Normal("mu", 0.0, 2.0) * Poisson("count", 4.0)
        draws = quantile.options["n_broadcast_samples"]

        with workflow_run(seed=0):
            record = Record({"mean": mean(law), "median": quantile(law, 0.5)}, label="r")

        assert repr(record).splitlines()[-1] == (
            f"mean: exact. median: Monte Carlo, {draws:,} draws."
        )

    def test_an_approximate_column_that_records_no_draws_reads_approximate(self):
        # The standard deviation of a law with a kernel takes the variance in a
        # nested call, whose draws its own provenance does not record.
        def counts_given_mu(mu: jax.Array) -> Normal:
            return Normal("y", jnp.full(2, mu), 1.0)

        prior = Normal("mu", 0.0, 2.0)
        kernel = conditional_distribution(counts_given_mu, given_spec=prior.event_spec.components)
        with workflow_run(seed=0):
            record = Record({"sd": standard_deviation(kernel * prior)}, label="r")

        assert repr(record).splitlines()[-1] == "sd: approximate."


class TestABatchOfTablesShowsATablePerElement:
    def test_each_element_is_a_table_titled_by_its_levels(self):
        batch = NumericRecordBatch(
            {"m/x": jnp.array([0.0, 1.0]), "s/x": jnp.array([2.0, 3.0])}, "law", label="b"
        )

        assert repr(batch) == (
            "NumericRecordBatch('b', levels={'law': 2})\n"
            "law=0\n"
            "    m   s\n"
            "x  0.  2.\n"
            "law=1\n"
            "    m   s\n"
            "x  1.  3."
        )

    def test_str_is_the_tables_without_the_head(self):
        batch = NumericRecordBatch({"m/x": jnp.array([0.0, 1.0])}, "law", label="b")

        assert str(batch) == "law=0\n    m\nx  0.\nlaw=1\n    m\nx  1."

    def test_a_long_batch_shows_its_first_and_last_elements(self):
        batch = NumericRecordBatch(
            {"m/x": jnp.arange(6.0).reshape(2, 3), "s/x": jnp.ones((2, 3))},
            ("chain", "draw"),
            label="b",
        )

        with np.printoptions(threshold=4, edgeitems=1):
            lines = repr(batch).splitlines()
        titles = [line for line in lines if line.startswith("chain=") or line == "..."]

        assert titles == ["chain=0, draw=0", "...", "chain=1, draw=2"]

    def test_the_tables_are_html_tables_in_a_notebook(self):
        batch = NumericRecordBatch(
            {"m/x": jnp.array([0.0, 1.0]), "s/x": jnp.array([2.0, 3.0])}, "law", label="b"
        )

        html = batch._repr_html_()

        assert html.count("<table>") == 2
        assert "<p><b>law=1</b></p>" in html
