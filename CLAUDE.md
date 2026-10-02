@AGENTS.md

## Claude Code

- Run a skill with its slash command, such as `/review-pr 468`.
- `.claude/settings.json` formats each Python file that an edit writes, through the `ruff-format` hook of `.pre-commit-config.yaml`.
- Keep permissions and other personal settings in `.claude/settings.local.json`, which `.gitignore` lists.
