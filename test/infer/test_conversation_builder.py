import pytest

from tools.infer_client import _think_split_qwen


def build_messages(system, context, history, prompt):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if context:
        msgs.append({"role": "user", "content": context})
    msgs.extend(history)
    msgs.append({"role": "user", "content": prompt})
    return msgs


def test_content_only_continuation():
    system = "You are helpful."
    context = "Context A."
    history = []

    # Turn 1
    prompt1 = "Prove 2+2=4."
    msgs1 = build_messages(system, context, history, prompt1)
    assert msgs1[0]["role"] == "system"
    assert msgs1[-1]["content"] == prompt1

    # Model returns thinking then content
    raw = "<think>reasoning...</think>Final answer."
    r, c = _think_split_qwen(raw)
    assert r is not None and c == "Final answer."
    history.append({"role": "assistant", "content": c})

    # Turn 2 uses only content from prior turn
    prompt2 = "Generalize."
    msgs2 = build_messages(system, context, history, prompt2)
    assert any(m["role"] == "assistant" and m["content"] == "Final answer." for m in msgs2)
    assert not any("<think>" in (m.get("content") or "") for m in msgs2)

