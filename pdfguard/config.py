"""Settings, read from environment variables or secrets.env (or .env).

The API key is never stored in the code. Paste it into `secrets.env` in the
project folder (that file is in .gitignore, so it is never committed):

    CEREBRAS_API_KEY=csk-...

A file called `.env` works too.
"""
from __future__ import annotations

import os
from pathlib import Path

CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_MODEL = "gpt-oss-120b"


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


def url() -> str:
    """Override only for testing (e.g. a mock server)."""
    return os.environ.get("CEREBRAS_BASE_URL", CEREBRAS_URL)


def api_key() -> str:
    return os.environ.get("CEREBRAS_API_KEY", "").strip()


def model() -> str:
    return os.environ.get("PDFGUARD_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def llm_available() -> bool:
    return bool(api_key())


def masked_key() -> str:
    k = api_key()
    return f"{k[:4]}...{k[-4:]}" if len(k) > 10 else ("set" if k else "not set")
