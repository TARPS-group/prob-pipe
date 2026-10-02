# ProbPipe agent guide

This file maps each task to the document that owns its rules.
It owns the everyday commands and the verification steps, and every other rule is stated in the document it points to.

## Project

ProbPipe is a Python framework for probabilistic pipelines with automated uncertainty quantification.
The design reference in `design/` describes the target state, and the documents below describe the code as it stands.

## Read first

| Task | Document that owns its rules |
|---|---|
| Any change to `probpipe/` | `CONTRACTS.md`: the contract directives every PR follows |
| Naming, imports, types, tests, and module layout | `STYLE_GUIDE.md` |
| The contract of an abstraction or an operation | its section of `design/`, listed in `design/README.md` |
| A term of art, or the canonical name of a recurring concept | `design/glossary.md` |
| The package a module belongs in | `design/package-structure.md` |
| A new family, inference method, converter, or array backend | `docs/api/extending.md` |
| Test tolerances and baselines | `STYLE_GUIDE.md` §8.6 |
| A docs page or a notebook | `CONTRIBUTING.md` § Documentation |
| Installation | `CONTRIBUTING.md` § Installation |
| The branch, the PR title and body, the CHANGELOG entry, and labels | `CONTRIBUTING.md` § PR Workflow and `.github/PULL_REQUEST_TEMPLATE.md` |

Where `design/` disagrees with the code or with a contributor document, `CONTRACTS.md` directive 5 decides.

## Commands

Run each command from the repository root.
`uv run` uses the `.venv` that `uv sync` creates, as `CONTRIBUTING.md` § Installation describes.

```bash
uv run pytest tests/core/test_record.py -x              # one file, stopping at the first failure
uv run pytest -p no:xdist -o "addopts="                 # serially, for a debugger
uv run pytest                                           # the full suite, in parallel
uv run pytest --cov=probpipe --cov-report=term-missing  # the full suite with coverage
uv run ruff check .                                     # lint the tree
uv run ruff format .                                    # format the tree; --check only reports
pre-commit run --all-files                              # every pre-commit hook over the tree
uv run --with 'pyright[nodejs]' pyright                 # type check
uv run mkdocs build --strict                            # build the docs, failing on a warning
uv run mkdocs serve                                     # preview the docs
uv run python scripts/design/design_blocks.py check III.7 --module probpipe.distributions
uv run python scripts/design/ledger.py --no-tests       # the stubs and the stale docs
```

The `design_blocks.py` line compares the code blocks of design section III.7 with the code.
CI pins the pyright version in `.github/workflows/ci.yml`.

## Verify

1. **Select the tests.** `python3 scripts/ci/import_graph.py test-targets <changed .py files>` prints the targets that CI runs for changed source files, and each changed test file runs too.
2. **Compare counts with the base branch.** Run the targets on the branch and on its base, and compare the pass, skip, and xfail counts. A test that skips on the branch and passes on the base hides a failure, as a missing optional backend does.
3. **Run the design and documentation tests.** A change to `design/` needs `tests/docs/` and the four `test_design_conformance.py` files under `tests/`, and a change to a rule document or an agent file needs `tests/docs/`.
4. **Run the ruff gate.** `uv run ruff check .` and `uv run ruff format --check .` pass.
5. **Run the full suite once before opening a PR.** It takes about seven minutes.

## Skills and checks

The skills in `.claude/skills/` run the multi-step procedures, and `.claude/skills/README.md` describes each one.
An agent that does not load skills can read a skill's `SKILL.md` and follow its steps.

- `check-pr`: the pre-PR checks of a branch, and a PR body drafted from the template;
- `design-check`: the design sections a change touches, checked against the code and the design's conventions;
- `review-pr`: a review of a PR against the rule documents;
- `review-all`: four independent reviews of a PR, merged into one report;
- `audit-tests`: an audit of the test suite;
- `criticize-design` and `criticize-with-docs`: an interview that stress-tests a design.

These checks run without being asked:

- the pre-commit hooks: ruff lint and format, file hygiene, and `no-issue-numbers`, which rejects an issue or PR number in `probpipe/`;
- CI: the blocking ruff gate, the tests a change selects, the notebooks, the docs build, the advisory type check, the design ledger report, and the advisory PR hygiene job;
- `tests/docs/`: each path, name, and section pointer that the rule documents and the agent files cite exists, this file stays within 120 lines, and each CHANGELOG release has one heading per change type;
- `tests/test_version.py`: the two `pyproject.toml` files share one version.
