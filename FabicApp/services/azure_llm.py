"""Azure OpenAI completion client for the Fabric data source.

Fabric has no in-warehouse LLM functions, so this module is the Fabric
counterpart of Snowflake's `SNOWFLAKE.CORTEX.COMPLETE(...)` and of the LLM
behind the Cortex Agents. Settings (.env):
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
  AZURE_OPENAI_DEPLOYMENT (default gpt-4.1-mini)
"""
from __future__ import annotations

import os
import threading

from openai import AzureOpenAI

_lock = threading.Lock()
_client: AzureOpenAI | None = None


def _setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"')


def deployment() -> str:
    return _setting("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini") or "gpt-4.1-mini"


def _get_client() -> AzureOpenAI:
    global _client
    with _lock:
        if _client is None:
            endpoint = _setting("AZURE_OPENAI_ENDPOINT")
            api_key = _setting("AZURE_OPENAI_API_KEY")
            if not endpoint or not api_key:
                raise RuntimeError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY missing from .env")
            _client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=_setting("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
        return _client


def complete(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1800,
    temperature: float = 0.2,
    timeout: float = 120.0,
    json_mode: bool = False,
) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(
        model=deployment(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Azure OpenAI returned empty text")
    return text
