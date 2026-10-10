"""The PR hygiene check of a title, a branch, and a body."""

from __future__ import annotations

import pr_hygiene
import pytest

TEMPLATE = """## Summary

<!-- One paragraph. -->

## Test plan

## Checklist

- [ ] PR title follows `<type>(<scope>): <subject>`
- [ ] Linked to an issue above
"""


class TestTitle:
    @pytest.mark.parametrize(
        "title",
        [
            "docs(contributing): document PR conventions",
            "feat(functions)!: the default sample count is 256",
        ],
    )
    def test_a_conventional_title_has_no_finding(self, title):
        assert pr_hygiene.title_findings(title) == []

    @pytest.mark.parametrize(
        "title",
        ["Fix the sampler", "A1.6: rename the label", "fix: guard empty record", "fix(core):"],
    )
    def test_any_other_title_is_a_finding(self, title):
        (finding,) = pr_hygiene.title_findings(title)
        assert "<type>(<scope>): <subject>" in finding


class TestBranch:
    def test_a_dev_branch_with_a_kebab_case_description_has_no_finding(self):
        assert pr_hygiene.branch_findings("dev/record-array-views") == []

    @pytest.mark.parametrize("branch", ["claude/brave-bose-477e3c", "dev/A1.6", "main"])
    def test_any_other_branch_is_a_finding(self, branch):
        (finding,) = pr_hygiene.branch_findings(branch)
        assert branch in finding


class TestBody:
    def test_a_body_with_every_section_and_item_has_no_finding(self):
        body = TEMPLATE.replace("- [ ] Linked", "- [x] Linked") + "\nMore text.\n"
        assert pr_hygiene.body_findings(body, TEMPLATE) == []

    def test_a_missing_section_and_a_missing_item_are_findings(self):
        body = "## Summary\n\nText.\n\n## Checklist\n\n- [x] Linked to an issue above\n"
        assert pr_hygiene.body_findings(body, TEMPLATE) == [
            "the body lacks the section '## Test plan'",
            "the body lacks the checklist item 'PR title follows `<type>(<scope>): <subject>`'",
        ]

    def test_an_empty_body_lacks_every_section(self):
        findings = pr_hygiene.body_findings("", TEMPLATE)
        assert len(findings) == 5


def test_the_repository_template_has_sections_and_items():
    template = pr_hygiene.TEMPLATE.read_text()
    assert pr_hygiene.body_findings(template, template) == []
    assert len(pr_hygiene.body_findings("", template)) > 5


def test_main_reports_each_finding_and_exits_one(monkeypatch, tmp_path, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("PR_TITLE", "Fix things")
    monkeypatch.setenv("PR_BRANCH", "dev/fix-things")
    monkeypatch.setenv("PR_BODY", pr_hygiene.TEMPLATE.read_text())
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert pr_hygiene.main() == 1
    assert capsys.readouterr().out.startswith("::warning title=PR hygiene::the title")
    assert "the title 'Fix things'" in summary.read_text()
