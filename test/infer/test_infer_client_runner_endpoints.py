import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

infer_client = importlib.import_module("infer_client")
infer_runner = importlib.import_module("infer_runner")


def test_build_messages_ordering():
    msgs = infer_client._build_messages(
        "sys", "ctx", "prompt", history=[{"role": "assistant", "content": "hi"}]
    )
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "ctx"}
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-1] == {"role": "user", "content": "prompt"}


def test_load_manifest_paths_reads_pointer(temp_obs_root):
    telemetry = temp_obs_root / "telemetry"
    manifest = telemetry / "container_runs" / "container-run-001.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "container_run_id": "container-run-001",
                "storage": {"log_file": str(telemetry / "logs" / "run.log")},
            }
        )
    )
    pointer = telemetry / "container_run_meta.env"
    pointer.write_text(f"CONTAINER_RUN_META_JSON_HOST={manifest}\n")

    run_id, manifest_host, log_file = infer_client._load_manifest_paths()
    assert run_id == "container-run-001"
    assert Path(manifest_host) == manifest
    assert log_file == str(telemetry / "logs" / "run.log")


class _DummyUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _DummyChoice:
    finish_reason = "stop"

    def __init__(self, text):
        self.message = type("M", (), {"content": text, "reasoning_content": None})()


class _DummyResponse:
    def __init__(self, text):
        self.choices = [_DummyChoice(text)]
        self.usage = _DummyUsage()


class DummyOpenAI:
    def __init__(self, base_url=None, api_key=None):  # noqa: D401
        self.base_url = base_url
        self.api_key = api_key
        self.calls = []
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Comp", (), {})()

        def create(**kwargs):
            self.calls.append(kwargs)
            return _DummyResponse("<think>calc</think>Answer")

        self.chat.completions.create = create


def test_one_shot_happy_path(monkeypatch, tmp_path, capsys):
    manifest_host = tmp_path / "container_runs" / "container-run-xyz.json"
    manifest_host.parent.mkdir(parents=True, exist_ok=True)
    manifest_host.write_text("{}")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    log_file_container = "/telemetry/logs/run.log"

    monkeypatch.setattr(
        infer_client,
        "_load_manifest_paths",
        lambda: ("container-run-xyz", str(manifest_host), log_file_container),
    )
    monkeypatch.setitem(sys.modules, "openai", type("Mod", (), {"OpenAI": DummyOpenAI}))

    args = type(
        "Args",
        (),
        {
            "test_id": "test-1",
            "base_url": "http://127.0.0.1:30000/v1",
            "api_key": "dummy",
            "model_id": "local",
            "system": "sys",
            "context": "ctx",
            "prompt": "hello",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 64,
            "thinking_hint": "qwen-thinking",
        },
    )()

    result = infer_client._one_shot(args)
    assert result == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert out_lines, "expected JSON output"
    out = json.loads(out_lines[-1])
    assert isinstance(out["schema_version"], int)
    assert out["schema_version"] == 1
    assert out["status"] == "ok"
    assert out["container_log_anchor"]["path"] == log_file_container
    assert out["response_snapshot"]["assistant_content_text"] == "Answer"
    assert out["response_snapshot"]["assistant_reasoning_text"] == "<think>calc</think>"
    transcript = manifest_host.parent / "inference" / "test-1" / "transcript.json"
    assert transcript.exists()
    data = json.loads(transcript.read_text())
    assert data["assistant_content_text"] == "Answer"


class DeterministicOpenAI(DummyOpenAI):
    def __init__(self, *args, **kwargs):  # noqa: D401
        super().__init__(*args, **kwargs)

        def create(model, messages, max_tokens, temperature, top_p):
            self.calls.append(
                {
                    "model": model,
                    "messages": list(messages),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
            return _DummyResponse("<think>path</think>Done")

        self.chat.completions.create = create


def test_run_scenario(monkeypatch, tmp_path, capsys):
    manifest_host = tmp_path / "container_runs" / "container-run-abc.json"
    manifest_host.parent.mkdir(parents=True, exist_ok=True)
    manifest_host.write_text("{}")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    log_file_container = "/telemetry/logs/run.log"

    scenario_path = tmp_path / "scenario.json"
    scenario = {
        "session": {
            "system": "sys",
            "defaults": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 32,
                "qwen_enable_thinking": True,
            },
        },
        "one_shot": [
            {"id": "os1", "prompt": "Hello"},
            {
                "id": "os-no-think",
                "prompt": "Hi",
                "overrides": {"thinking_hint": "none"},
            },
        ],
        "multi_turn": [
            {
                "id": "conv1",
                "turns": [{"user": "Hi"}],
            }
        ],
    }
    scenario_path.write_text(json.dumps(scenario))

    monkeypatch.setitem(
        sys.modules, "openai", type("Mod", (), {"OpenAI": DeterministicOpenAI})
    )
    import importlib

    importlib.reload(infer_runner)
    monkeypatch.setattr(
        infer_runner,
        "_load_manifest_paths",
        lambda: ("container-run-abc", str(manifest_host), log_file_container),
    )

    infer_runner.run_scenario(
        type("Args", (), {"scenario": str(scenario_path), "api_key": "dummy"})()
    )

    output = [
        json.loads(line)
        for line in capsys.readouterr().out.strip().splitlines()
        if line.strip()
    ]
    assert all(
        isinstance(entry.get("schema_version"), int)
        and entry.get("schema_version") == 1
        for entry in output
    )
    assert any(
        entry["test_id"] == "os1" and entry["status"] == "ok" for entry in output
    )
    assert any(
        entry.get("test_id") == "conv1" and entry.get("status") in {"ok", "partial"}
        for entry in output
    )
    no_think_records = [
        entry for entry in output if entry.get("test_id") == "os-no-think"
    ]
    assert no_think_records, "missing os-no-think record"
    for record in no_think_records:
        snapshot = record.get("response_snapshot") or {}
        assert snapshot.get("assistant_reasoning_text") is None
        assert "<think>" in (snapshot.get("assistant_content_text") or "")
    for record in output:
        anchor = record.get("container_log_anchor") or {}
        if anchor.get("path"):
            assert anchor["path"] == log_file_container
    # Ensure reasoning stripped from assistant content in multi-turn history
    conv_steps = [entry for entry in output if entry["test_id"].startswith("conv1:")]
    assert conv_steps
    for step in conv_steps:
        assert "<think>" not in step["response_snapshot"]["assistant_content_text"]
