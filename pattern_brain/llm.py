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
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .routing import LLMRouter, Router

Completer = Callable[[str, str, List[Dict[str, Any]]], Dict[str, Any]]

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


# ----------------------------------------------------------- auto + default
def auto_completer(prefer: Tuple[str, ...] = ("ollama", "anthropic"),
                   ) -> Tuple[Optional[Completer], Optional[str]]:
    """Return ``(completer, backend_name)`` for the first available backend in
    ``prefer`` order (local Ollama first, then cloud Anthropic), or ``(None, None)``
    if no LLM backend is reachable."""
    for backend in prefer:
        if backend == "ollama" and ollama_available():
            return ollama_completer(), "ollama"
        if backend == "anthropic" and anthropic_available():
            return anthropic_completer(), "anthropic"
    return None, None


def default_llm_router() -> Optional[Router]:
    """An :class:`LLMRouter` wired to whatever real backend is available, or
    ``None`` if none is — in which case the caller keeps the heuristic router."""
    completer, name = auto_completer()
    if completer is None:
        return None
    return LLMRouter(completer, name=f"llm-{name}")


def llm_backend_status() -> Dict[str, Any]:
    """What the router would use right now (for the dashboard / diagnostics)."""
    completer, name = auto_completer()
    return {
        "ollama_available": ollama_available(),
        "anthropic_available": anthropic_available(),
        "active_backend": name,                       # None -> heuristic fallback
        "router": f"llm-{name}" if name else "heuristic (no LLM backend configured)",
    }
