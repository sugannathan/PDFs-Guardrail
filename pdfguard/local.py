"""Run an open model from Hugging Face on your own machine. Free, no API key needed.

Backends:
  - MLX (Apple Silicon Macs, fast, 4-bit): pip install mlx-lm
  - transformers (any machine; uses Apple GPU "mps" or CUDA when available): pip install transformers torch

The model is downloaded from Hugging Face on first use and cached
(~/.cache/huggingface). It is loaded once per process and reused for
every finding.
"""
from __future__ import annotations

import os
import platform
import sys

# Defaults: Qwen3 4B Instruct (Apache 2.0, not gated, modern, good at following JSON instructions).
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

_cache: dict = {}


def backend() -> str:
    chosen = os.environ.get("PDFGUARD_LOCAL_BACKEND", "").strip().lower()
    if chosen in ("mlx", "transformers"):
        return chosen
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_lm  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "transformers"


def default_model() -> str:
    return DEFAULT_MLX_MODEL if backend() == "mlx" else DEFAULT_HF_MODEL


def _messages(system: str, user: str, model_id: str) -> list[dict]:
    # Gemma chat templates have no system role: put the instructions into the user turn.
    if "gemma" in model_id.lower():
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prompt(tokenizer, messages: list[dict]) -> str:
    try:
        # enable_thinking=False switches off the reasoning mode of Qwen3 hybrid models; others ignore it.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_transformers(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    token = os.environ.get("HF_TOKEN") or None          # only needed for gated models such as Gemma
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token, dtype=dtype).to(device)
    model.eval()
    return tok, model, device


def _load_mlx(model_id: str):
    from mlx_lm import load
    model, tok = load(model_id)
    return tok, model, "mlx"


def load(model_id: str):
    key = (backend(), model_id)
    if key not in _cache:
        print(f"[pdfguard] loading {model_id} ({backend()}) - first run downloads it from Hugging Face ...",
              file=sys.stderr)
        _cache[key] = _load_mlx(model_id) if backend() == "mlx" else _load_transformers(model_id)
    return _cache[key]


def generate(system: str, user: str, model_id: str, max_new_tokens: int = 200) -> str:
    tok, model, device = load(model_id)
    prompt = _prompt(tok, _messages(system, user, model_id))

    if device == "mlx":
        from mlx_lm import generate as mlx_generate
        try:
            from mlx_lm.sample_utils import make_sampler
            return mlx_generate(model, tok, prompt=prompt, max_tokens=max_new_tokens,
                                sampler=make_sampler(temp=0.0), verbose=False)
        except ImportError:
            return mlx_generate(model, tok, prompt=prompt, max_tokens=max_new_tokens, verbose=False)

    import torch
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
