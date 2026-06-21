"""Real LLM backends for the tool-calling router (PLAN.md step 4 / Block 45).

The step-4 :class:`~pattern_brain.routing.LLMRouter` already runs the tool-calling
loop against ONE injected ``complete(system, user, tools)`` callable; this module
provides real implementations of that callable, plus auto-detection, so the
router uses an actual LLM when one is configured and falls back to the offline
heuristic otherwise.

OPTIONAL, DEFENSIVE (same discipline as the torch nodes, §0b): nothing here is a
hard dependency. Two backends:

* **Ollama** (local, default-preferred — keeps Rule-1 separation, no cloud/cost):
  reachable at ``$OLLAMA_HOST`` or ``http://localhost:11434``. Uses only the
  stdlib (``urllib``), so it needs no extra package.
* **Anthropic** (cloud): needs the ``anthropic`` SDK installed AND
  ``$ANTHROPIC_API_KEY`` set.

If neither is available, :func:`auto_completer` returns ``None`` and
:func:`default_llm_router` returns ``None`` — the caller then keeps the offline
:class:`~pattern_brain.routing.HeuristicRouter`. The response PARSERS are pure
functions so they're unit-testable without any live backend.

Domain-agnostic (Rule 23): only generic tool schemas + a chosen tool name.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .routing import LLMRouter, Router

Completer = Callable[[str, str, List[Dict[str, Any]]], Dict[str, Any]]
# A conversational callable: a list of {role, content} messages -> assistant text.
TextCompleter = Callable[..., str]

DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"  # small/fast is ideal for routing


# ----------------------------------------------------------------- parsers
def parse_anthropic_response(resp: Any) -> Dict[str, Any]:
    """Pure parser: extract the chosen tool from an Anthropic Messages response
    (an object with ``.content`` blocks, or the equivalent dict)."""
    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content", [])
    for block in content or []:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "tool_use":
            name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
            if name:
                return {"name": name, "reason": "anthropic tool_use"}
    return {"name": "stop", "reason": "anthropic returned no tool_use"}


def parse_ollama_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Pure parser: extract the chosen tool from an Ollama /api/chat response."""
    calls = (resp.get("message", {}) or {}).get("tool_calls", []) or []
    if calls:
        fn = calls[0].get("function", {}) or {}
        name = fn.get("name")
        if name:
            return {"name": name, "reason": "ollama tool_call"}
    return {"name": "stop", "reason": "ollama returned no tool_call"}


def _to_anthropic_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t.get("input_schema", {"type": "object", "properties": {}})}
            for t in tools]


def _to_ollama_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"],
        "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
        for t in tools]


# OpenAI-compatible /chat/completions uses the same tool shape as Ollama.
_to_openai_tools = _to_ollama_tools


def parse_openai_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Pure parser: extract the chosen tool from an OpenAI-compatible
    /chat/completions response (``choices[0].message.tool_calls[0].function``)."""
    choices = resp.get("choices") or []
    if choices:
        msg = choices[0].get("message", {}) or {}
        calls = msg.get("tool_calls") or []
        if calls:
            fn = calls[0].get("function", {}) or {}
            name = fn.get("name")
            if name:
                return {"name": name, "reason": "openai tool_call"}
    return {"name": "stop", "reason": "openai returned no tool_call"}


def parse_openai_text(resp: Dict[str, Any]) -> str:
    """Pure parser: extract plain assistant text from an OpenAI-compatible
    /chat/completions response (used by the conversational chat path)."""
    choices = resp.get("choices") or []
    if choices:
        return (choices[0].get("message", {}) or {}).get("content", "") or ""
    return ""


# ----------------------------------------------- free cloud provider registry
# Every entry is OpenAI-`/chat/completions`-compatible, so ONE completer covers
# all of them (Block 56 sweep). Order = the decided priority chain: free Chinese
# clouds first (GLM-Flash is free/unlimited; DeepSeek the cheap reasoner; Qwen
# the coding specialist), then the large free-tier US-hosted endpoints, then
# local Ollama, then cloud Anthropic, then the offline heuristic.
#
# A provider is "available" iff its API-key env var is set. No key set anywhere
# -> the chain is empty and the caller degrades to local/heuristic exactly as
# before (graceful degradation, same discipline as Block 54 / the torch nodes).
CLOUD_PROVIDERS: List[Dict[str, str]] = [
    {"name": "zai",       "env": "ZAI_API_KEY",
     "base_url": "https://api.z.ai/api/paas/v4", "model": "glm-4.5-flash"},
    {"name": "deepseek",  "env": "DEEPSEEK_API_KEY",
     "base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    {"name": "qwen",      "env": "DASHSCOPE_API_KEY",
     "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    {"name": "groq",      "env": "GROQ_API_KEY",
     "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    {"name": "cerebras",  "env": "CEREBRAS_API_KEY",
     "base_url": "https://api.cerebras.ai/v1", "model": "gpt-oss-120b"},
    {"name": "nvidia",    "env": "NVIDIA_API_KEY",
     "base_url": "https://integrate.api.nvidia.com/v1", "model": "deepseek-ai/deepseek-v4-flash"},
    {"name": "mistral",   "env": "MISTRAL_API_KEY",
     "base_url": "https://api.mistral.ai/v1", "model": "mistral-large-latest"},
    {"name": "openrouter", "env": "OPENROUTER_API_KEY",
     "base_url": "https://openrouter.ai/api/v1", "model": "deepseek/deepseek-chat"},
]

# Per-provider in-process cooldown (seconds since epoch until which to skip it).
# Set when a provider returns 429/402/403 so the chain stops hammering it. The
# loop is one process, so an in-memory dict suffices (no Redis, Rule 1 / §0b).
import time as _time
_PROVIDER_COOLDOWN: Dict[str, float] = {}
_COOLDOWN_SECONDS = 90.0


def _cooling_down(name: str) -> bool:
    return _PROVIDER_COOLDOWN.get(name, 0.0) > _time.time()


def _trip_cooldown(name: str) -> None:
    _PROVIDER_COOLDOWN[name] = _time.time() + _COOLDOWN_SECONDS


def _openai_post(base_url: str, api_key: str, payload: Dict[str, Any],
                 timeout: float, name: str = "") -> Dict[str, Any]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        # A browser-like User-Agent: some providers (Groq, Cerebras) sit behind
        # Cloudflare, which 403s the default "Python-urllib" signature (CF error
        # 1010). This is a client-fingerprint block, not an auth problem.
        headers={"Content-Type": "application/json",
                 "Accept": "application/json",
                 "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) pattern-brain/1.0",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:           # noqa: F841
        if name and e.code in (429, 402, 403):
            _trip_cooldown(name)
        raise


def openai_compatible_completer(base_url: str, model: str, api_key: str,
                                timeout: float = 30.0, name: str = "") -> Completer:
    """A tool-picking ``complete`` callable for ANY OpenAI-compatible endpoint
    (Z.AI/DeepSeek/Qwen/Groq/Cerebras/NVIDIA/Mistral/OpenRouter). Same contract
    as :func:`ollama_completer` so it drops straight into :class:`LLMRouter`."""
    def complete(system: str, user: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "tools": _to_openai_tools(tools),
            "tool_choice": "required",
        }
        return parse_openai_response(_openai_post(base_url, api_key, payload, timeout, name))
    return complete


def openai_compatible_chat(base_url: str, model: str, api_key: str,
                           timeout: float = 60.0, name: str = "") -> "TextCompleter":
    """A free-form conversational callable for an OpenAI-compatible endpoint —
    the foundation the agent + dashboard chat (ENG-3/ENG-4) talk through."""
    def chat(messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        return parse_openai_text(_openai_post(base_url, api_key, payload, timeout, name))
    return chat


# -------------------------------------------------------------- availability
def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def ollama_available(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(ollama_host() + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def anthropic_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- completers
def ollama_completer(model: str = DEFAULT_OLLAMA_MODEL, timeout: float = 30.0) -> Completer:
    """A ``complete`` callable backed by a local Ollama server (stdlib only)."""
    host = ollama_host()

    def complete(system: str, user: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "tools": _to_ollama_tools(tools),
        }
        req = urllib.request.Request(
            host + "/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return parse_ollama_response(json.loads(r.read().decode()))
    return complete


def anthropic_completer(model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = 512) -> Completer:
    """A ``complete`` callable backed by the Anthropic Messages API (tool use)."""
    import anthropic
    client = anthropic.Anthropic()

    def complete(system: str, user: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            tools=_to_anthropic_tools(tools),
            tool_choice={"type": "any"},          # force a tool selection
            messages=[{"role": "user", "content": user}])
        return parse_anthropic_response(resp)
    return complete


# --------------------------------------------------- cloud chain availability
def available_cloud_providers() -> List[Dict[str, str]]:
    """The configured cloud providers whose API key is set and which aren't in
    cooldown, in the decided priority order (Block 56 / §9)."""
    return [p for p in CLOUD_PROVIDERS
            if os.environ.get(p["env"]) and not _cooling_down(p["name"])]


# ----------------------------------------------------------- auto + default
# The decided chain (PLAN.md §9 / Block 56): cloud-primary for reasoning, local
# Ollama as the guaranteed offline fallback, cloud Anthropic last, then the
# offline heuristic (when auto_completer returns None). Cloud providers come
# first because algorithm invention is the most reasoning-heavy call and the
# free Chinese clouds outclass a no-GPU local model on this box.
ALL_BACKENDS: Tuple[str, ...] = tuple(p["name"] for p in CLOUD_PROVIDERS) + ("ollama", "anthropic")
DEFAULT_PREFER: Tuple[str, ...] = ALL_BACKENDS


def _make_tool_completer(name: str) -> Optional[Completer]:
    """A tool-picking completer for one named backend if it's available now."""
    if name == "ollama":
        return ollama_completer() if ollama_available() else None
    if name == "anthropic":
        return anthropic_completer() if anthropic_available() else None
    for p in available_cloud_providers():
        if p["name"] == name:
            return openai_compatible_completer(p["base_url"], p["model"],
                                               os.environ[p["env"]], name=name)
    return None


def auto_completer(prefer: Tuple[str, ...] = DEFAULT_PREFER,
                   ) -> Tuple[Optional[Completer], Optional[str]]:
    """Return ``(completer, backend_name)`` for the first available backend in
    ``prefer`` order (cloud-primary, then local Ollama, then Anthropic), or
    ``(None, None)`` if no backend is reachable — the caller then keeps the
    offline heuristic."""
    for backend in prefer:
        c = _make_tool_completer(backend)
        if c is not None:
            return c, backend
    return None, None


def auto_text_completer(prefer: Tuple[str, ...] = DEFAULT_PREFER,
                        ) -> Tuple[Optional[TextCompleter], Optional[str]]:
    """Like :func:`auto_completer` but returns a free-form conversational
    callable (the agent's reasoning + the dashboard chat talk through this)."""
    for backend in prefer:
        if backend == "ollama" and ollama_available():
            return ollama_chat(), "ollama"
        if backend == "anthropic" and anthropic_available():
            return anthropic_chat(), "anthropic"
        for p in available_cloud_providers():
            if p["name"] == backend:
                return (openai_compatible_chat(p["base_url"], p["model"],
                                               os.environ[p["env"]], name=backend), backend)
    return None, None


def default_llm_router() -> Optional[Router]:
    """An :class:`LLMRouter` wired to the highest-priority available backend, or
    ``None`` if none is — in which case the caller keeps the heuristic router."""
    completer, name = auto_completer()
    if completer is None:
        return None
    return LLMRouter(completer, name=f"llm-{name}")


def llm_backend_status() -> Dict[str, Any]:
    """What the router/agent would use right now (for the dashboard / diagnostics)."""
    _, name = auto_completer()
    return {
        "ollama_available": ollama_available(),
        "anthropic_available": anthropic_available(),
        "cloud_available": [p["name"] for p in available_cloud_providers()],
        "active_backend": name,                       # None -> heuristic fallback
        "router": f"llm-{name}" if name else "heuristic (no LLM backend configured)",
        "chain": list(DEFAULT_PREFER),
    }


# -------------------------------------------------- conversational completers
def ollama_chat(model: str = DEFAULT_OLLAMA_MODEL, timeout: float = 60.0) -> TextCompleter:
    """Free-form chat over a local Ollama server (stdlib only)."""
    host = ollama_host()

    def chat(messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
        payload = {"model": model, "stream": False, "messages": messages,
                   "options": {"temperature": temperature}}
        req = urllib.request.Request(
            host + "/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (json.loads(r.read().decode()).get("message", {}) or {}).get("content", "") or ""
    return chat


def anthropic_chat(model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = 1024) -> TextCompleter:
    """Free-form chat over the Anthropic Messages API."""
    import anthropic
    client = anthropic.Anthropic()

    def chat(messages: List[Dict[str, str]], temperature: float = 0.4) -> str:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        resp = client.messages.create(model=model, max_tokens=max_tokens,
                                       system=system or None, temperature=temperature,
                                       messages=turns)
        parts = [getattr(b, "text", "") for b in (getattr(resp, "content", None) or [])]
        return "".join(parts)
    return chat
