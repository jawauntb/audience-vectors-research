"""Unified Modal app. All GPU/CPU jobs register on this single app.

Import this module to deploy or run any function — function modules
import `app` from here, so they register on import.

`.env` is loaded explicitly here because Modal captures host env as a
Secret at app-import time, and `modal run` / `modal deploy` don't load
dotenv on their own. Without this, gated-repo tokens (HF_TOKEN, etc.)
silently miss the Modal worker.
"""

from __future__ import annotations

import importlib
import os
from copy import deepcopy
from pathlib import Path

import modal

# Load .env from the repo root before reading os.environ for the secret.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _REPO_ROOT / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(_ENV_PATH, override=False)
    except ImportError:
        # Fallback: parse the file ourselves so Modal still gets the env.
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)

APP_BASE_NAME = "audience-vectors"
MODAL_REGION = os.environ.get("MODAL_REGION") or None


def get_app_name() -> str:
    env = os.environ.get("DEPLOYMENT_ENV", "dev")
    return f"{APP_BASE_NAME}-{env}"


# Forward host env vars to Modal as secrets. Strip the few that break
# Dockerfile parsing or aren't safe to forward.
_env_dict: dict[str, str | None] = deepcopy(dict(os.environ))
for _drop in ("PATH",):
    _env_dict.pop(_drop, None)
_env_dict = {k: v for k, v in _env_dict.items() if not k.startswith("BASH_FUNC_")}

env_secrets = [modal.Secret.from_dict(_env_dict)] if _env_dict else []

app = modal.App(get_app_name())


# Function modules to register at import time. Keep this list ordered;
# each module imports `app` from here and decorates @app.function /
# @app.cls at import time.
_FUNCTION_MODULES: tuple[str, ...] = (
    "tribe_predictor",
    "vjepa_predictor",
    "cogvideox_generator",
    "svd_generator",
    "wan22_generator",
    "wan22_lora_trainer",
    "debug_volume",
    "video_analyzer_site",
)

for _mod in _FUNCTION_MODULES:
    importlib.import_module(f"audience_vectors.modal_app.functions.{_mod}")


__all__ = ["app", "env_secrets", "MODAL_REGION", "get_app_name"]
