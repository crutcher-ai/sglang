#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _think_split_qwen(text: str) -> Tuple[Optional[str], str]:
    """Split Qwen thinking by the last </think>.
    Returns (reasoning_or_None, content). Handles closing-tag-only quirk.
    """
    if not text:
        return None, ""
    idx = text.rfind("</think>")
    if idx == -1:
        return None, text
    # Include the closing tag in the reasoning side for clarity
    reasoning = text[: idx + len("</think>")]
    content = text[idx + len("</think>") :].lstrip("\n\r ")
    return reasoning if reasoning.strip() else None, content


def _load_manifest_paths() -> Tuple[str, str, str]:
    host_root = os.environ.get(
        "HOST_OBS_ROOT", os.path.expanduser("~/sglang-observability")
    )
    run_meta = os.path.join(host_root, "telemetry", "container_run_meta.env")
    if not os.path.isfile(run_meta):
        raise RuntimeError(f"manifest pointer not found: {run_meta}")
    manifest_host = ""
    with open(run_meta, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CONTAINER_RUN_META_JSON_HOST="):
                manifest_host = line.strip().split("=", 1)[1]
                break
    if not manifest_host or not os.path.isfile(manifest_host):
        raise RuntimeError("host manifest path missing or not found")
    with open(manifest_host, "r", encoding="utf-8") as f:
        m = json.load(f)
    run_id = (m.get("run") or {}).get("container_run_id") or m.get("container_run_id")
    if not run_id:
        raise RuntimeError("container_run_id missing in manifest")
    log_file = (m.get("storage") or {}).get("log_file")
    if not log_file:
        raise RuntimeError("log_file missing in manifest")
    return run_id, manifest_host, log_file


def _build_messages(
    system: Optional[str],
    context: Optional[str],
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if context:
        msgs.append({"role": "user", "content": context})
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _one_shot(args: argparse.Namespace) -> int:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        print("This tool requires the 'openai' Python package.", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    run_id, manifest_host, log_file = _load_manifest_paths()
    test_id = args.test_id
    started = _iso_now()
    t0 = time.perf_counter()

    messages = _build_messages(args.system, args.context, args.prompt)

    try:
        resp = client.chat.completions.create(
            model=args.model_id,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except Exception as e:
        finished = _iso_now()
        total_ms = int((time.perf_counter() - t0) * 1000)
        out = {
            "schema_version": 1,
            "test_id": test_id,
            "status": "http_error",
            "http_status": None,
            "stop_reason": None,
            "usage": None,
            "timings": {
                "start_ts": started,
                "end_ts": finished,
                "total_latency_ms": total_ms,
            },
            "request_snapshot": {
                "system": args.system,
                "context": args.context,
                "prompt": args.prompt,
                "sampling": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                },
                "thinking_hint": args.thinking_hint,
            },
            "response_snapshot": None,
            "prom_bookmark": {
                "container_run_id": run_id,
                "window": {"start_ts": started, "end_ts": finished},
            },
            "container_log_anchor": {
                "path": log_file,
                "window": {"start_ts": started, "end_ts": finished},
            },
            "error": str(e),
        }
        print(json.dumps(out, ensure_ascii=False))
        return 1

    finished = _iso_now()
    total_ms = int((time.perf_counter() - t0) * 1000)

    choice = resp.choices[0]
    # message.content may be str per OpenAI spec
    raw = getattr(choice.message, "content", None)
    if isinstance(raw, list):
        # Concatenate text parts if list
        raw = "".join([p.get("text") or "" for p in raw if isinstance(p, dict)])
    if raw is None:
        raw = ""
    reasoning, content = (None, raw)
    if args.thinking_hint == "qwen-thinking":
        # Prefer server-provided split if present
        rc = getattr(choice.message, "reasoning_content", None)
        if isinstance(rc, str) and rc.strip():
            reasoning, content = rc, (raw or "")
        else:
            r, c = _think_split_qwen(raw)
            reasoning, content = r, c

    usage = getattr(resp, "usage", None)
    usage_obj = None
    if usage is not None:
        usage_obj = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    out = {
        "schema_version": 1,
        "test_id": test_id,
        "status": "ok",
        "http_status": 200,
        "stop_reason": getattr(choice, "finish_reason", None),
        "usage": usage_obj,
        "timings": {
            "start_ts": started,
            "end_ts": finished,
            "total_latency_ms": total_ms,
        },
        "request_snapshot": {
            "base_url": base_url,
            "model_id": args.model_id,
            "system": args.system,
            "context": args.context,
            "prompt": args.prompt,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
            "thinking_hint": args.thinking_hint,
        },
        "response_snapshot": {
            "assistant_text_raw": raw,
            "assistant_reasoning_text": reasoning,
            "assistant_content_text": content,
        },
        "prom_bookmark": {
            "container_run_id": run_id,
            "window": {"start_ts": started, "end_ts": finished},
        },
        "container_log_anchor": {
            "path": log_file,
            "window": {"start_ts": started, "end_ts": finished},
        },
    }

    # Write artifacts
    # manifest_host already resides under .../container_runs/<RUN_ID>/
    run_dir = Path(manifest_host).parent / "inference" / test_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "transcript.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "messages": messages,
                "assistant_text_raw": raw,
                "assistant_reasoning_text": reasoning,
                "assistant_content_text": content,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "usage": usage_obj,
                "timings": out["timings"],
                "status": out["status"],
                "stop_reason": out["stop_reason"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(out, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    one = sp.add_parser("one-shot", help="Run a single chat completion")
    one.add_argument("--test-id", required=True)
    one.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    one.add_argument("--api-key", default="dummy")
    one.add_argument("--model-id", default="local")
    one.add_argument("--system")
    one.add_argument("--context")
    one.add_argument("--prompt", required=True)
    one.add_argument("--temperature", type=float, default=0.6)
    one.add_argument("--top-p", type=float, default=0.95)
    one.add_argument("--max-tokens", type=int, default=1024)
    one.add_argument(
        "--thinking-hint", choices=["qwen-thinking", "none"], default="qwen-thinking"
    )

    args = ap.parse_args()
    if args.cmd == "one-shot":
        return _one_shot(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
