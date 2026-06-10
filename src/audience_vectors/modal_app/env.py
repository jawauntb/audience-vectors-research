"""Environment helpers for Modal app startup."""

from __future__ import annotations

from collections.abc import MutableMapping

LLAMA_READ_TOKEN_ENV = "LLAMA_READ_TOKEN"
HUGGINGFACE_TOKEN_ENVS = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def normalize_huggingface_token_env(environ: MutableMapping[str, str]) -> None:
    """Mirror LLAMA_READ_TOKEN into standard Hugging Face token env names.

    Hugging Face Hub and Transformers integrations look for names such as
    HF_TOKEN. The project-specific LLAMA_READ_TOKEN keeps the secret's purpose
    obvious while this aliasing makes Modal workers compatible with the
    underlying libraries.
    """
    token = environ.get(LLAMA_READ_TOKEN_ENV, "").strip()
    if not token:
        return
    for env_name in HUGGINGFACE_TOKEN_ENVS:
        environ.setdefault(env_name, token)
