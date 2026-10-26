"""The prose checker's candidates for the writing rules and the document battery."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "design"))

import prose


def _kinds(tmp_path: Path, text: str) -> list[tuple[int, str]]:
    path = tmp_path / "page.md"
    path.write_text(text)
    return [(candidate.line, candidate.kind) for candidate in prose.candidates([path])]


class TestFlags:
    def test_plain_prose_has_no_candidate(self, tmp_path):
        text = "The registry selects one method, so the call returns its result.\n"
        assert _kinds(tmp_path, text) == []

    def test_a_vocabulary_word_and_an_intensifier_are_candidates(self, tmp_path):
        text = "The public surface is precisely two methods.\n"
        assert _kinds(tmp_path, text) == [(1, "vocabulary"), (1, "intensifier")]

    def test_exactly_in_its_mathematical_sense_is_no_candidate(self, tmp_path):
        text = "Each kind has exactly one batch form, exactly when it is a value.\n"
        assert _kinds(tmp_path, text) == []

    def test_a_quoted_word_and_a_code_span_are_no_candidates(self, tmp_path):
        text = 'Avoid "surface" as a figure of speech, and call `touch()` by its name.\n'
        assert _kinds(tmp_path, text) == []

    def test_an_appositive_after_a_code_span_is_a_candidate(self, tmp_path):
        text = "The term exposes `raw()`, the one access point.\n"
        assert _kinds(tmp_path, text) == [(1, "appositive")]

    def test_a_parenthetical_of_five_words_is_a_candidate(self, tmp_path):
        text = "A law draws (as the sampler in the engine does) one value.\n"
        assert _kinds(tmp_path, text) == [(1, "parenthetical")]

    def test_a_comma_list_of_four_items_is_a_candidate(self, tmp_path):
        assert _kinds(tmp_path, "It reads specs, records, batches, and laws.\n") == [
            (1, "comma list")
        ]

    def test_a_list_wrapped_across_lines_is_read_as_one_paragraph(self, tmp_path):
        text = "It reads specs,\nrecords, batches,\nand laws.\n"
        assert _kinds(tmp_path, text) == [(1, "comma list")]

    def test_a_fenced_code_block_is_skipped(self, tmp_path):
        assert _kinds(tmp_path, "```python\nsurface = 1  # precisely\n```\n") == []


class TestBattery:
    def test_an_unknown_section_and_an_unbalanced_fence_are_candidates(self, tmp_path):
        kinds = _kinds(tmp_path, "See VIII.4 for details.\n\n```python\nx = 1\n")
        assert kinds == [(1, "unknown section"), (0, "unbalanced fence")]

    def test_a_section_of_the_design_is_no_candidate(self, tmp_path):
        assert _kinds(tmp_path, "See II.4 for identity.\n") == []

    def test_a_table_row_with_another_column_count_is_a_candidate(self, tmp_path):
        text = "| a | b |\n|---|---|\n| 1 | 2 | 3 |\n"
        assert _kinds(tmp_path, text) == [(3, "table columns")]

    def test_trailing_whitespace_and_a_double_space_are_candidates(self, tmp_path):
        assert _kinds(tmp_path, "One  claim. \n") == [
            (1, "trailing whitespace"),
            (1, "double space"),
        ]


def test_main_exits_zero_with_candidates_unless_strict(tmp_path, capsys):
    path = tmp_path / "page.md"
    path.write_text("The surface is small.\n")
    assert prose.main([str(path)]) == 0
    assert prose.main([str(path), "--strict"]) == 1
    assert "vocabulary: surface" in capsys.readouterr().out
