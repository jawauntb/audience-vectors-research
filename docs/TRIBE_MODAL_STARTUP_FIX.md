# TRIBE Modal Startup Fix

Use this when a repo runs TRIBE/tribev2 in Modal and the app fails at
startup/import time.

## Symptom

The Modal image builds or deploys, but the TRIBE predictor crashes before
serving. A common root cause is an unpinned `exca` transitive dependency:
`neuralset==0.0.2`, pulled by TRIBE, still imports
`exca.steps.base.NoValue`. Floating `exca` can resolve to a version without
that compatibility surface.

## Audit

Search the repo:

```bash
rg -n "facebookresearch/tribev2|tribev2|TribeModel|exca" .
```

For a quick audit across neighboring repos:

```bash
cd ~/path-containing-repos
rg -n "facebookresearch/tribev2|from tribev2 import TribeModel|exca.steps.base|uv_pip_install\\(|pip_install\\(" \
  --glob '!**/.git/**' --glob '!**/node_modules/**' --glob '!**/.venv/**'
```

## Fix

If a Modal image installs TRIBE like this:

```python
.uv_pip_install(
    "git+https://github.com/facebookresearch/tribev2.git@..."
)
```

or:

```python
.pip_install("git+https://github.com/facebookresearch/tribev2.git@...")
```

pin `exca==0.5.25` in the same install layer:

```python
_TRIBE_EXCA_VERSION = "0.5.25"

.uv_pip_install(
    "git+https://github.com/facebookresearch/tribev2.git@<pinned-commit>",
    f"exca=={_TRIBE_EXCA_VERSION}",
)
```

Also add a build-time preflight so Modal catches the failure during image build
instead of production container startup:

```python
_TRIBE_IMPORT_RUNTIME_PREFLIGHT_COMMAND = (
    'python -c "import exca.steps.base as exca_base; '
    "exca_base.NoValue(); "
    "from tribev2 import TribeModel; "
    'print(TribeModel.__name__)"'
)

.run_commands(
    _TRIBE_IMPORT_RUNTIME_PREFLIGHT_COMMAND,
    ...,
)
```

## After Patching

1. Run the repo's lint, typecheck, and targeted tests.
2. Deploy or rebuild every Modal environment using that TRIBE image.
3. You should not need to repopulate TRIBE/HuggingFace weights unless the repo
   changed its volume or cache setup.
