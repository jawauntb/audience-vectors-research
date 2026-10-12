# Handoff: pytest CI via testonce

Date: 2026-10-11

## What changed

- New `.github/workflows/tests.yml` (the repo had no CI): ubuntu-latest, Python 3.12,
  `uv sync --frozen --group dev --extra modal --compile-bytecode`, then every
  `tests/test_*.py` runs through `testonce each` (`jawauntb/testonce@main`).
- New `testonce.json`: ignores tool caches and the dedicated temp dir.
- `.gitignore`: `.testonce/`.

No test was removed, skipped or weakened. A failing test still fails the job
(`testonce each` exits 1; checked locally with a deliberately failing probe test).

## Things a future agent must know

- The `modal` extra is required: `tests/test_*_modal*.py` import `modal` and fail
  without it.
- `PYTHONDONTWRITEBYTECODE=1` is load-bearing, not a speed tweak. With a warm `.pyc`
  Python only `stat`s the `.py`, and testonce records a stat as "exists", so an edit
  to a repo module would not rerun the tests that import it (reproduced locally).
  Do not add `**/__pycache__/**` to `ignore` and do not drop that variable.
  Site-packages are precompiled by `--compile-bytecode`, so this costs little.
- `TMPDIR=/tmp/testonce-pytest-tmp` keeps Python's startup temp-dir probe from
  making every test "modify" `/tmp` (uncacheable). It is outside the checkout on
  purpose: `test_attention_capture_manifest_builder.py` expects absolute tmp paths and
  fails if `tmp_path` is under the cwd.
- Running testonce locally on a machine with warm `src/**/__pycache__` is unsound for
  the reason above. Delete them or set `PYTHONDONTWRITEBYTECODE=1` and clear first.

## Not done

- [ ] Ruff and pyright are not run in CI (`ruff check src tests` reports 9 existing
      errors locally); not added here to keep this PR to the test step.
- [ ] Confirm CI wall time and disk on the first runner: `uv.lock` pins CUDA torch.
- [ ] Add `noCache` entries if the audit run ever quarantines a clock-dependent test.
