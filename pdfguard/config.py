"""Settings, read from environment variables or secrets.env (or .env).

The API key is never stored in the code. Paste it into `secrets.env` in the
project folder (that file is in .gitignore, so it is never committed):

    PDFGUARD_PROVIDER=local       # free: open model from Hugging Face on your machine
    CEREBRAS_API_KEY=csk-...      # or Cerebras
    HF_TOKEN=hf_...               # or Hugging Face Inference Providers (also used to download gated models)

A file called `.env` works too.
"""
from __future__ import annotations

import os
from pathlib import Path

PROVIDERS = {
    # name: (chat completions URL, models URL, env var for the key, default model)
    "cerebras": ("https://api.cerebras.ai/v1/chat/completions", "https://api.cerebras.ai/v1/models",
                 "CEREBRAS_API_KEY", "gpt-oss-120b"),
    "huggingface": ("https://router.huggingface.co/v1/chat/completions", "https://router.huggingface.co/v1/models",
                    "HF_TOKEN", "openai/gpt-oss-120b"),
}


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from secrets.env or .env (current folder, then project root).

    Values already set in the environment win.
    """
    for env in (Path.cwd() / "secrets.env", Path.cwd() / ".env",
                Path(__file__).resolve().parents[1] / "secrets.env", Path(__file__).resolve().parents[1] / ".env"):
        if not env.is_file():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
        return


_load_dotenv()


LOCAL = "local"


def provider() -> str:
    """PDFGUARD_PROVIDER if set (local, cerebras, huggingface), else whichever key is present, else local."""
    chosen = os.environ.get("PDFGUARD_PROVIDER", "").strip().lower()
    if chosen in ("hf", "hugging_face", "hugging-face"):
        chosen = "huggingface"
    if chosen in ("local", "offline", "transformers", "mlx"):
        return LOCAL
    if chosen in PROVIDERS:
        return chosen
    for name, (_, _, env, _) in PROVIDERS.items():
        if os.environ.get(env, "").strip():
            return name
    # no provider chosen and no API key: use the free local Hugging Face model if it can run here
    import importlib.util
    if importlib.util.find_spec("mlx_lm") or importlib.util.find_spec("transformers"):
        return LOCAL
    return "cerebras"


def is_local() -> bool:
    return provider() == LOCAL


def local_runtime_installed() -> bool:
    import importlib.util
    return any(importlib.util.find_spec(m) is not None for m in ("mlx_lm", "transformers"))


def url() -> str:
    """Chat completions URL. CEREBRAS_BASE_URL overrides it, for testing only."""
    return os.environ.get("CEREBRAS_BASE_URL") or PROVIDERS[provider()][0]


def models_url() -> str:
    return PROVIDERS[provider()][1]


def key_name() -> str:
    return "none" if is_local() else PROVIDERS[provider()][2]


def api_key() -> str:
    return "" if is_local() else os.environ.get(key_name(), "").strip()


def model() -> str:
    chosen = os.environ.get("PDFGUARD_MODEL", "").strip()
    if chosen:
        return chosen
    if is_local():
        from .local import default_model
        return default_model()
    return PROVIDERS[provider()][3]


def llm_available() -> bool:
    return local_runtime_installed() if is_local() else bool(api_key())


def masked_key() -> str:
    if is_local():
        return "not needed (local model)"
    k = api_key()
    return f"{k[:4]}...{k[-4:]}" if len(k) > 10 else ("set" if k else "not set")
