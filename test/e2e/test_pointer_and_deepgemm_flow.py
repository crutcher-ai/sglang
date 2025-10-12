import json
import os
import subprocess
import time
from pathlib import Path

import pytest


def sh(cmd: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, env=env)


@pytest.mark.skipif(
    os.environ.get("SGL_E2E_CONTAINER_TESTS") != "1",
    reason="E2E container+GPU smoke; enable with SGL_E2E_CONTAINER_TESTS=1",
)
def test_observable_inference_smoke():

    env = os.environ.copy()
    env.setdefault("READY_TIMEOUT", os.environ.get("SGL_TEST_READY_TIMEOUT", "240"))
    env.setdefault(
        "MEM_FRACTION_STATIC", os.environ.get("SGL_TEST_MEM_FRACTION", "0.98")
    )
    context_override = os.environ.get("SGL_TEST_CONTEXT_LENGTH", "8192")
    env.setdefault("CONTEXT_LENGTH", context_override)
    env.setdefault("MAX_PREFILL_TOKENS", context_override)
    env.setdefault("MAX_TOTAL_TOKENS", context_override)
    env.setdefault(
        "CHUNKED_PREFILL_SIZE", os.environ.get("SGL_TEST_CHUNKED_PREFILL", "2048")
    )
    host_root = Path(
        env.get("HOST_OBS_ROOT", os.path.expanduser("~/sglang-observability"))
    )
    telemetry_root = host_root / "telemetry"
    run_meta = telemetry_root / "container_run_meta.env"

    model_container_path = env.get(
        "SGL_TEST_MODEL_CONTAINER_PATH",
        "/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    )
    model_host_path = (
        host_root / "models" / Path(model_container_path).relative_to("/models")
    )
    if not model_host_path.exists():
        pytest.skip(f"model snapshot missing at {model_host_path}")

    def run_checked(cmd: str) -> subprocess.CompletedProcess:
        res = sh(cmd, env=env)
        assert res.returncode == 0, res.stderr
        return res

    start_info = None
    try:
        run_checked("./scripts/start_observable_container.sh")

        t0 = time.time()
        while time.time() - t0 < 120:
            if run_meta.exists() and run_meta.stat().st_size > 0:
                break
            time.sleep(1)
        assert run_meta.exists(), "manifest pointer not created"
        pointer = run_meta.read_text()
        assert "CONTAINER_RUN_META_JSON_HOST=" in pointer
        manifest_path = Path(
            pointer.split("CONTAINER_RUN_META_JSON_HOST=", 1)[1].splitlines()[0]
        )
        assert manifest_path.exists(), "manifest JSON missing"
        manifest = json.loads(manifest_path.read_text())
        log_file_container = (manifest.get("storage") or {}).get("log_file")
        paths = manifest.get("paths") or {}
        host_paths = paths.get("host") or {}
        container_paths = paths.get("container") or {}
        assert host_paths.get("log_file"), "paths.host.log_file missing"
        assert container_paths.get("log_file"), "paths.container.log_file missing"
        assert log_file_container, "storage.log_file missing"
        log_file_host = Path(host_paths["log_file"])
        assert log_file_host.exists(), "host log file missing after container start"

        # Cache ensure pass
        ensure_cmd = (
            f"./scripts/cache/populate_caches.sh --model {model_container_path} "
            "--tp 1 --flashinfer ensure --inductor ensure --deep-gemm ensure --moe skip"
        )
        res = run_checked(ensure_cmd)
        assert "RESULT_STATUS" in res.stdout

        # Second ensure should be mostly noops
        res = run_checked(ensure_cmd)
        assert "noop" in res.stdout

        # Manifest should now reference the prep_result path with status ok
        manifest = json.loads(manifest_path.read_text())
        prep_host_path = manifest.get("paths", {}).get("host", {}).get("prep_result")
        assert prep_host_path, "paths.host.prep_result missing"
        prep_host_path = Path(prep_host_path)
        assert prep_host_path.exists(), "prep_result.json missing on host"
        prep_content = json.loads(prep_host_path.read_text())
        assert (
            prep_content.get("status") == "ok"
        ), f"prep_result status not ok: {prep_content.get('status')}"

        # Start server
        start_cmd = "./scripts/infer/start_server.sh"
        start_res = sh(start_cmd, env=env)
        if start_res.returncode != 0:
            if log_file_host.exists():
                content = log_file_host.read_text()
                if "Not enough memory" in content:
                    pytest.skip(
                        "SGLang server could not start: insufficient GPU memory even after lowering context length. "
                        "Adjust TP or offload settings."
                    )
            assert False, start_res.stderr
        start_info = json.loads(start_res.stdout.strip().splitlines()[-1])
        assert (
            isinstance(start_info.get("schema_version"), int)
            and start_info["schema_version"] == 1
        )
        assert start_info["health"] == "ready"
        # Double-check helper health endpoint immediately afterwards
        status_res = sh("./scripts/infer/status.sh", env=env)
        assert status_res.stdout.strip() == "ready"

        # Inference one-shot
        client_cmd = (
            "python tools/infer_client.py one-shot "
            "--test-id pytest-smoke "
            "--prompt 'Say hello' "
            "--thinking-hint qwen-thinking "
            "--max-tokens 32"
        )
        client_res = run_checked(client_cmd)
        payload = json.loads(client_res.stdout.strip().splitlines()[-1])
        assert payload["status"] == "ok"
        assert payload["response_snapshot"]["assistant_content_text"].strip()
        assert payload["container_log_anchor"]["path"] == log_file_container

        scenario_path = host_root / "sglang_e2e_scenario.json"
        scenario_path.write_text(
            json.dumps(
                {
                    "session": {
                        "system": "You are helpful.",
                        "defaults": {
                            "temperature": 0.2,
                            "top_p": 0.8,
                            "max_tokens": 128,
                            "qwen_enable_thinking": True,
                        },
                    },
                    "one_shot": [
                        {"id": "os-smoke", "prompt": "Reply with a short greeting."}
                    ],
                    "multi_turn": [
                        {
                            "id": "conv-smoke",
                            "turns": [
                                {"user": "Describe the colour of grass."},
                                {"user": "Answer in one word."},
                            ],
                        }
                    ],
                }
            )
        )
        try:
            runner_res = run_checked(
                f"python tools/infer_runner.py --scenario {scenario_path}"
            )
            ndjson_lines = [
                json.loads(line)
                for line in runner_res.stdout.splitlines()
                if line.strip()
            ]
            assert ndjson_lines, "infer_runner emitted no records"
            assert all(
                isinstance(record.get("schema_version"), int)
                and record.get("schema_version") == 1
                for record in ndjson_lines
            )
            assert any(record.get("test_id") == "os-smoke" for record in ndjson_lines)
            assert any(record.get("test_id") == "conv-smoke" for record in ndjson_lines)
            for record in ndjson_lines:
                anchor = record.get("container_log_anchor") or {}
                if anchor.get("path"):
                    assert anchor["path"] == log_file_container
        finally:
            scenario_path.unlink(missing_ok=True)

        # Stop server cleanly
        stop_res = sh("./scripts/infer/stop_server.sh", env=env)
        assert stop_res.returncode in (0, 1)
        status_after_stop = sh("./scripts/infer/status.sh", env=env)
        assert status_after_stop.stdout.strip() == "down"
    finally:
        sh("./scripts/infer/stop_server.sh", env=env)
        sh("./scripts/stop_observable_container.sh", env=env)
        assert (
            not run_meta.exists()
        ), "pointer file should be removed after container stop"

    assert start_info is not None
