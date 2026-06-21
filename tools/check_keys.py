"""Live health-check for the configured LLM provider keys (Block 56 / ENG-0).

Reads keys from the project .env (Rule 1, gitignored), then sends a tiny real
chat request to EACH provider through pattern_brain.llm's own OpenAI-compatible
path — so this doubles as live proof that ENG-0 works end-to-end. Keys are masked
in all output; nothing is printed in full.

Run:  python3 tools/check_keys.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pattern_brain as pb
from pattern_brain import llm


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def mask(key: str) -> str:
    return f"{key[:6]}…{key[-4:]} (len {len(key)})" if key else "—"


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(here, ".env"))
    print("=" * 78)
    print("Pattern Brain — LLM provider key health-check (live)")
    print("=" * 78)
    results = []
    for p in pb.CLOUD_PROVIDERS:
        name, env, base, model = p["name"], p["env"], p["base_url"], p["model"]
        key = os.environ.get(env, "")
        if not key:
            print(f"  {name:11s} [{env}]  SKIP — no key set")
            results.append((name, "skip"))
            continue
        chat = llm.openai_compatible_chat(base, model, key, timeout=30.0, name=name)
        t0 = time.time()
        try:
            reply = chat([{"role": "user", "content": "Reply with exactly: OK"}])
            dt = time.time() - t0
            ok = bool(reply and reply.strip())
            status = "OK " if ok else "EMPTY"
            print(f"  {name:11s} {status} {dt:5.1f}s  model={model}")
            print(f"               key={mask(key)}  reply={reply.strip()[:60]!r}")
            results.append((name, "ok" if ok else "empty"))
        except Exception as e:
            dt = time.time() - t0
            msg = str(e).split("\n")[0][:140]
            print(f"  {name:11s} FAIL {dt:5.1f}s  model={model}")
            print(f"               key={mask(key)}  error={msg}")
            results.append((name, "fail"))
    print("-" * 78)
    oks = [n for n, s in results if s == "ok"]
    print(f"WORKING: {oks}")
    print(f"NOT WORKING: {[n for n, s in results if s in ('fail', 'empty')]}")
    print(f"NO KEY: {[n for n, s in results if s == 'skip']}")


if __name__ == "__main__":
    main()
