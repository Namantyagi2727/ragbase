"""
ui/providers.py — Which inference backend is configured, and what that means for privacy.

Direct local Ollama is the default. Prism is offered only when it is configured
(PRISM_API_KEY set); its guardrail, cache and provider-verification behaviour
is described from docs/PRISM.md. The public showcase offers no provider choice.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import config

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def prism_configured() -> bool:
    return bool(os.environ.get("PRISM_API_KEY", "").strip())


def available_backends() -> list[str]:
    return ["ollama"] + (["prism"] if prism_configured() else [])


def _host(url: str) -> str:
    return urlparse(url).hostname or url


def notice(backend: str, ollama_model: str) -> dict:
    """{title, lines[]} describing where question and document text are sent."""
    if backend == "prism":
        allowed = ", ".join(sorted(config.PRISM_ALLOWED_PROVIDERS)) or "none"
        return {"title": f"Prism gateway · virtual model “{config.PRISM_MODEL}”", "lines": [
            f"Requests go to Prism at {_host(config.PRISM_BASE_URL)}, which routes them to a provider.",
            f"RAGBase accepts answers only from allowed providers ({allowed}); any other provider, "
            "including a cache hit that does not report its origin, is rejected (fail closed).",
            "Prism keeps each response in its Redis cache for one hour, so document-derived text is held there; "
            "an identical repeated request may be rejected as a cache hit.",
            "Prism's guardrails scan the full prompt and can block requests (for example, injection-like text).",
            "The provider actually used is shown under each answer.",
        ]}
    host = _host(config.OLLAMA_BASE_URL)
    where = "on this machine" if host in _LOCAL_HOSTS else f"at {host}"
    return {"title": f"Direct local inference · Ollama · {ollama_model}", "lines": [
        f"Questions and retrieved passages are sent only to the Ollama server {where}.",
        "No hosted inference service is used.",
    ]}


def notice_html(backend: str, ollama_model: str) -> str:
    import html

    n = notice(backend, ollama_model)
    body = "<br>".join(html.escape(x) for x in n["lines"])
    return f'<div class="rb-notice"><b>{html.escape(n["title"])}</b><br>{body}</div>'
