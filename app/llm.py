"""Provider selection for optional OpenAI-compatible chat completions."""
from __future__ import annotations

import os
from typing import Any


def llm_config() -> tuple[str, str, str] | None:
    """Return (api_key, base_url, model), or None when no provider key is configured."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        return (
            api_key,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            os.getenv("LLM_MODEL", "gemini-3.8-flash"),
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        return (api_key, "", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    raise ValueError("LLM_PROVIDER must be 'openai' or 'gemini'")


def llm_enabled() -> bool:
    return llm_config() is not None


def create_llm_client() -> tuple[Any, str]:
    config = llm_config()
    if config is None:
        raise RuntimeError("No LLM provider API key is configured")
    api_key, base_url, model = config
    from openai import OpenAI

    client_options: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    return OpenAI(**client_options), model
