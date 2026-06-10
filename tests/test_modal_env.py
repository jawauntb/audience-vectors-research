from audience_vectors.modal_app.env import normalize_huggingface_token_env


def test_normalize_huggingface_token_env_aliases_llama_read_token() -> None:
    env = {"LLAMA_READ_TOKEN": "hf_secret"}

    normalize_huggingface_token_env(env)

    assert env["HF_TOKEN"] == "hf_secret"
    assert env["HUGGINGFACE_HUB_TOKEN"] == "hf_secret"
    assert env["HUGGINGFACE_TOKEN"] == "hf_secret"


def test_normalize_huggingface_token_env_preserves_explicit_standard_tokens() -> None:
    env = {
        "LLAMA_READ_TOKEN": "hf_llama",
        "HF_TOKEN": "hf_existing",
    }

    normalize_huggingface_token_env(env)

    assert env["HF_TOKEN"] == "hf_existing"
    assert env["HUGGINGFACE_HUB_TOKEN"] == "hf_llama"
