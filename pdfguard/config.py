"""Settings, read from environment variables or secrets.env (or .env).

The API key is never stored in the code. Paste it into `secrets.env` in the
project folder (that file is in .gitignore, so it is never committed):

    CEREBRAS_API_KEY=csk-...      # Cerebras
    HF_TOKEN=hf_...               # or Hugging Face Inference Providers

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


def provider() -> str:
    """PDFGUARD_PROVIDER if set, otherwise whichever key is present (Cerebras first)."""
    chosen = os.environ.get("PDFGUARD_PROVIDER", "").strip().lower()
    if chosen in ("hf", "hugging_face", "hugging-face"):
        chosen = "huggingface"
    if chosen in PROVIDERS:
        return chosen
    for name, (_, _, env, _) in PROVIDERS.items():
        if os.environ.get(env, "").strip():
            return name
    return "cerebras"


def url() -> str:
    """Chat completions URL. CEREBRAS_BASE_URL overrides it, for testing only."""
    return os.environ.get("CEREBRAS_BASE_URL") or PROVIDERS[provider()][0]


def models_url() -> str:
    return PROVIDERS[provider()][1]


def key_name() -> str:
    return PROVIDERS[provider()][2]


def api_key() -> str:
    return os.environ.get(key_name(), "").strip()


def model() -> str:
    return os.environ.get("PDFGUARD_MODEL", "").strip() or PROVIDERS[provider()][3]


def llm_available() -> bool:
    return bool(api_key())


def masked_key() -> str:
    k = api_key()
    return f"{k[:4]}...{k[-4:]}" if len(k) > 10 else ("set" if k else "not set")
