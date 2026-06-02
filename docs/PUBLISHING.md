# Repository Publishing Policy

This repository should publish source code, reviewable protocol documents, and
small synthetic fixtures only. Raw datasets, participant responses, model
weights, caches, generated media, and generated report bundles should stay out
of git unless a maintainer explicitly approves a narrow exception.

## Artifact Policy

Keep these out of git:

- Raw or licensed data under `data/`, including BOLD Moments stimuli, fMRI
  files, VideoMem, Memento10k, VIDEM, PEEK, and VideoLectures-derived assets.
- Human-subject or platform exports, including Prolific responses, participant
  IDs, survey exports, attention-check rows, free-text answers, and raw
  randomization logs that can be linked to a participant.
- Model weights and learned artifacts, including `.pt`, `.pth`, `.ckpt`,
  `.safetensors`, `.npy`, `.npz`, `.zarr`, and `.parquet` outputs.
- Generated media and bundles, including videos, hosted-site packages, paper
  bundles, `.zip` archives, and generated demo galleries.
- Local environments, package caches, Modal caches, notebook checkpoints, and
  tool caches.

Allowed by default:

- Source code in `src/`, `scripts/`, and `tests/`.
- Small hand-written docs, runbooks, protocols, and templates that do not
  include private participant data or licensed third-party assets.
- `.env.example` with placeholder values only.
- Tiny test fixtures only when they are synthetic, redistributable, and needed
  for automated tests.

If a binary artifact is required for reproducibility, prefer one of these
patterns instead of committing it:

- Record the generation command, source dataset, model revision, and checksum in
  a Markdown manifest.
- Store the artifact in an external bucket, registry, or release asset with
  access controls appropriate to the license.
- Add a small synthetic fixture that exercises the same code path.

## Secret Handling

Never commit `.env` or real API credentials. Before publishing, scan tracked
changes for keys and tokens:

```bash
rg -n --hidden --glob '!.git/**' --glob '!data/**' --glob '!.venv/**' \
  '(api[_-]?key|secret|token|password|BEGIN (RSA|OPENSSH|PRIVATE) KEY|sk-|hf_|ghp_|xox[baprs]-)'
```

Treat Modal-forwarded environment variables and Hugging Face tokens as private.
Use `.env.example` only for names, defaults, and comments.

## PR Workflow

Before opening a publish or hygiene PR:

1. Start from a fresh `main` worktree and avoid mixing research/code changes
   with metadata-only hygiene changes.
2. Check repository size and large files:

   ```bash
   du -sh . .git
   find . -path ./.git -prune -o -path ./.venv -prune -o -type f -size +25M -print
   git ls-files -z | xargs -0 ls -lh | sort -k5 -hr | head -40
   ```

3. Check status, ignored files, and untracked files:

   ```bash
   git status --short --ignored
   git ls-files --others --exclude-standard
   ```

4. Run the standard quality gates for code changes:

   ```bash
   uv run ruff check .
   uv run pyright
   uv run pytest -q
   ```

   For docs-only changes, run at least the status, ignored-file, and secret
   checks above.

5. Review every file staged for commit with `git diff --staged --stat` and
   `git diff --staged`.

## Visibility Recommendation

Default to private repository visibility while this project references gated
models, non-commercial TRIBE terms, licensed datasets, unpublished human-study
materials, and generated media. Consider public visibility only after removing
or externally gating private artifacts, confirming dataset and model license
compatibility, and publishing a clear artifact-access statement.
