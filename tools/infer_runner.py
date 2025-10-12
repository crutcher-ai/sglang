#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from infer_client import _think_split_qwen, _load_manifest_paths  # reuse helpers


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_scenario(path: str) -> Dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(txt)
    except Exception:
        # Try JSON
        return json.loads(txt)


def _build_messages(system: Optional[str], context: Optional[str], history: List[Dict[str, str]], prompt: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if context:
        msgs.append({"role": "user", "content": context})
    msgs.extend(history)
    msgs.append({"role": "user", "content": prompt})
    return msgs


def run_scenario(args: argparse.Namespace) -> int:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        print("This tool requires the 'openai' Python package.", file=sys.stderr)
        return 2

    sc = _load_scenario(args.scenario)
    session = sc.get("session") or {}
    base_url = (session.get("base_url") or "http://127.0.0.1:30000/v1").rstrip("/")
    model_id = session.get("model_id") or "local"
    system = session.get("system")
    defaults = session.get("defaults") or {}
    temperature = float(defaults.get("temperature", 0.6))
    top_p = float(defaults.get("top_p", 0.95))
    max_tokens = int(defaults.get("max_tokens", 1024))
    default_hint = "qwen-thinking" if defaults.get("qwen_enable_thinking", True) else "none"

    client = OpenAI(base_url=base_url, api_key=args.api_key)
    run_id, manifest_host, log_file = _load_manifest_paths()

    # NDJSON stream of results
    def emit(obj: Dict[str, Any]):
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    # One-shot tests
    for test in sc.get("one_shot", []) or []:
        tid = test.get("id") or "os"
        context = test.get("context")
        prompt = test.get("prompt") or ""
        ov = test.get("overrides") or {}
        hint = (ov.get("thinking_hint") or default_hint)
        t = float(ov.get("temperature", temperature))
        p = float(ov.get("top_p", top_p))
        mt = int(ov.get("max_tokens", max_tokens))
        started = _iso_now()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=_build_messages(system, context, [], prompt),
                max_tokens=mt,
                temperature=t,
                top_p=p,
            )
            choice = resp.choices[0]
            raw = getattr(choice.message, "content", None)
            if isinstance(raw, list):
                raw = "".join([x.get("text") or "" for x in raw if isinstance(x, dict)])
            if raw is None:
                raw = ""
            r, c = (None, raw)
            if hint == "qwen-thinking":
                rc = getattr(choice.message, "reasoning_content", None)
                if isinstance(rc, str) and rc.strip():
                    r, c = rc, raw
                else:
                    r, c = _think_split_qwen(raw)
            usage = getattr(resp, "usage", None)
            usage_obj = None
            if usage is not None:
                usage_obj = {"prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None)}
            emit({
                "test_id": tid,
                "status": "ok",
                "http_status": 200,
                "stop_reason": getattr(choice, "finish_reason", None),
                "usage": usage_obj,
                "timings": {"start_ts": started, "end_ts": _iso_now()},
                "request_snapshot": {"system": system, "context": context, "prompt": prompt, "sampling": {"temperature": t, "top_p": p, "max_tokens": mt}},
                "response_snapshot": {"assistant_text_raw": raw, "assistant_reasoning_text": r, "assistant_content_text": c},
                "prom_bookmark": {"container_run_id": run_id, "window": {"start_ts": started, "end_ts": _iso_now()}},
                "container_log_anchor": {"path": log_file, "window": {"start_ts": started, "end_ts": _iso_now()}},
            })
        except Exception as e:
            emit({"test_id": tid, "status": "http_error", "error": str(e)})

    # Multi-turn conversations
    for conv in sc.get("multi_turn", []) or []:
        cid = conv.get("id") or "conv"
        context = conv.get("context")
        ov = conv.get("overrides") or {}
        hint = (ov.get("thinking_hint") or default_hint)
        t = float(ov.get("temperature", temperature))
        p = float(ov.get("top_p", top_p))
        mt = int(ov.get("max_tokens", max_tokens))
        history: List[Dict[str, str]] = []
        started_conv = _iso_now()
        ok = True
        for i, turn in enumerate(conv.get("turns") or []):
            prompt = turn.get("user") or ""
            started = _iso_now()
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=_build_messages(system, context, history, prompt),
                    max_tokens=mt,
                    temperature=t,
                    top_p=p,
                )
                choice = resp.choices[0]
                raw = getattr(choice.message, "content", None)
                if isinstance(raw, list):
                    raw = "".join([x.get("text") or "" for x in raw if isinstance(x, dict)])
                if raw is None:
                    raw = ""
                r, c = (None, raw)
                if hint == "qwen-thinking":
                    rc = getattr(choice.message, "reasoning_content", None)
                    if isinstance(rc, str) and rc.strip():
                        r, c = rc, raw
                    else:
                        r, c = _think_split_qwen(raw)
                # Record full round in history: user then assistant (content-only)
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": c})
                usage = getattr(resp, "usage", None)
                usage_obj = None
                if usage is not None:
                    usage_obj = {"prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None)}
                emit({
                    "test_id": f"{cid}:{i+1}",
                    "status": "ok",
                    "http_status": 200,
                    "stop_reason": getattr(choice, "finish_reason", None),
                    "usage": usage_obj,
                    "timings": {"start_ts": started, "end_ts": _iso_now()},
                    "request_snapshot": {"system": system, "context": context, "prompt": prompt, "sampling": {"temperature": t, "top_p": p, "max_tokens": mt}},
                    "response_snapshot": {"assistant_text_raw": raw, "assistant_reasoning_text": r, "assistant_content_text": c},
                    "prom_bookmark": {"container_run_id": run_id, "window": {"start_ts": started, "end_ts": _iso_now()}},
                    "container_log_anchor": {"path": log_file, "window": {"start_ts": started, "end_ts": _iso_now()}},
                })
            except Exception as e:
                emit({"test_id": f"{cid}:{i+1}", "status": "http_error", "error": str(e)})
                ok = False
                break
        emit({
            "test_id": cid,
            "status": "ok" if ok else "partial",
            "policy_applied": "content-only continuation",
            "prom_bookmark": {"container_run_id": run_id, "window": {"start_ts": started_conv, "end_ts": _iso_now()}},
        })

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="Path to scenario YAML or JSON")
    ap.add_argument("--api-key", default="dummy")
    args = ap.parse_args()
    return run_scenario(args)


if __name__ == "__main__":
    sys.exit(main())
