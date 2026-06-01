---
name: commit-push
description: Run code-review, the pytest suite, and pre-commit hooks; update docs if drifted; write a Conventional Commits message; commit and push to main on GitHub (origin mgrts/skewed-sequences); optionally bump the [tool.poetry] version and push a release tag. Stops at every gate (failed review, failed tests, failed hooks, conflicting rebase) and requires explicit confirmation before committing and pushing.
---

# Commit & push for skewed-sequences

Analyze pending changes, review them, run the test suite + pre-commit hooks, update
docs if needed, write a Conventional Commits message, and push to `main`. Optionally
bump the package version and push a release tag.

The default branch is **`main`**; origin is **`git@github.com:mgrts/skewed-sequences.git`**
(GitHub, owner `mgrts`). This is a solo research repo, so the default flow pushes
directly to `main` after gates pass and the user confirms.

## Arguments

`$ARGUMENTS` — optional. A free-form commit message (used verbatim as the subject after
type inference) and/or flags: `--no-push` (commit only), `--release` (also bump version
and offer a tag). There is **no issue tracker** — never invent ticket references.

## Important

- **Conventional Commits**: `type(scope): subject`. Types: `feat`, `fix`, `refactor`,
  `perf`, `test`, `docs`, `chore`, `build`, `ci`. Scope is optional but encouraged
  (e.g. `loss`, `models`, `train`, `data`, `cli`, `config`, `experiments`, `viz`).
- **NEVER** list Claude among commit authors. Do not add a `Co-Authored-By` trailer,
  set `--author` to Claude/Anthropic, use an `@anthropic.com` address, or add a
  "Generated with Claude" line — to the commit message OR a PR body. This is a hard
  project rule, not a default: the `guard_git` PreToolUse hook **blocks** any `git
  commit` carrying such attribution, so a slip is denied rather than committed.
- Do **NOT** use `--force`, `--no-verify`, or any destructive git flag. The repo's
  guard-git hook will block these anyway. If a step fails, stop and ask the user.

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
git branch --show-current
```

If there are no changes, stop: "Nothing to commit." If the current branch is not `main`,
note it and ask the user whether to proceed on this branch or switch.

### Step 2: Run the code-review skill

Invoke the `code-review` skill on the pending diff.

- **Critical / High** findings: stop. Show them and ask whether to proceed anyway, fix
  automatically, or cancel. Do not move on without explicit acknowledgement.
- **Medium / Low** findings: print as a heads-up and continue.

### Step 3: Run tests

```bash
make test        # poetry run pytest  (currently 98 tests)
```

If tests fail: show failures, try to fix obvious causes from the diff (e.g. an
import-path drift after a rename, or a pinned count in `test_config.py` that needs to
move with a config change), re-run. If still failing, stop and ask.

### Step 4: Run pre-commit hooks

```bash
make pre-commit    # poetry run pre-commit run --all-files
```

If hooks fail: black/isort/flake8/end-of-file/trailing-whitespace auto-fix on a re-run —
re-run once. If they still fail after one auto-fix pass, stop and ask. Never bypass with
`--no-verify`. If `check-added-large-files` or `detect-private-key` trips, do NOT try to
force it through — surface the offending file to the user.

### Step 5: Update documentation

Read `README.md` and `CLAUDE.md`; update only sections that drifted from reality:

- **New CLI command / sub-app** → README CLI reference table + `CLAUDE.md` package map.
- **New `config.py` constant or changed default** → README Configuration table.
- **New module under `skewed_sequences/`** → `CLAUDE.md` package map (+ README structure).
- **Test count changed** → README "N tests" claim and `CLAUDE.md` (the count is read from
  `pytest --collect-only -q`).
- **A CRITICAL invariant changed** (lazy imports, OUTPUT_LENGTH, MLflow keys, loss
  convention) → update the relevant `CLAUDE.md` section.

If nothing drifted, skip this step. Do not rewrite docs that are already correct.

### Step 6: Optional version bump + release (only if `--release` or the user asks)

By default, do NOT bump the version on every commit. If a release is requested:

- Patch-bump `version` under **`[tool.poetry]`** in `pyproject.toml` (e.g.
  `1.2.3 → 1.2.4`). Never add a PEP 621 `[project]` table.
- Form the tag `v<version>` (e.g. `v1.2.4`) — created in Step 10 after the push.

### Step 7: Generate the Conventional Commits message

**Subject** (≤ 72 chars): `type(scope): summary`. Infer the type from the diff:

- new capability (loss, dataset, model, CLI command) → `feat`
- bug fix → `fix`
- behaviour-preserving restructure → `refactor`
- speed/memory → `perf`
- tests only → `test`
- docs/CLAUDE.md only → `docs`
- tooling/deps/version → `chore` / `build`

If `$ARGUMENTS` supplied a message, use it verbatim as the subject (after the type).

**Body** (after a blank line): one line per significant change. If config grids, MLflow
keys, model shapes, or experiment naming changed, explicitly note the synchronized
test/consumer updates so the contract reads as kept-whole. Add the version line only if
Step 6 bumped it:

```
Version: 1.2.3 -> 1.2.4
```

No AI-attribution trailer.

### Step 8: Show summary and confirm

Print: code-review result, test result, pre-commit result, doc updates (or "none"),
version bump (or "none"), files to be committed (`git status --short`), and the full
commit message. Then ask with `AskUserQuestion`:

```
question: "Commit and push to origin/main?"
header: "Commit & Push"
options:
  - "Yes" — stage all changes, commit, rebase onto origin/main, push.
  - "No"  — cancel, leave the working tree as-is.
```

Do NOT proceed without an explicit "Yes". If `--no-push` was passed, the option is
"Commit only (no push)".

### Step 9: Commit and push

```bash
git add -A
git commit -m "<subject>

<body>"
git fetch origin main
git rebase origin/main
```

If the rebase conflicts, **abort** (`git rebase --abort`) and tell the user to resolve
manually — do not auto-resolve. Then (unless `--no-push`):

```bash
git push origin main
```

If the push fails (branch protection, auth, network), do NOT retry and do NOT force.
Show the error and suggest pushing a feature branch + opening a PR
(`git switch -c <branch> && git push -u origin <branch> && gh pr create`).

### Step 10: Optional release tag

Only if Step 6 bumped the version. Read `version` from `pyproject.toml`, form
`v<version>`. Check it does not already exist:

```bash
git rev-parse "v<version>" 2>/dev/null
```

If it exists, surface that and skip. Otherwise confirm with `AskUserQuestion`, then:

```bash
git tag -a "v<version>" -m "Release v<version>"
git push origin "v<version>"
```

If the tag push fails, do NOT retry or delete the local tag; report that it exists
locally and can be pushed manually.

### Step 11: Final report

```
Pushed to origin/main.
Review: passed (or: N findings)   Tests: 98 passed   Pre-commit: passed
Doc updates: <files or "none">
Version: <bump or "no bump">      Tag: <v.. pushed | skipped>
```

Or, if the push was blocked, show the error and the feature-branch + PR suggestion.
