import json
import os
from pathlib import Path
from subprocess import PIPE, run

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = REPO_ROOT / "scripts" / "infer" / "start_server.sh"
STATUS_SCRIPT = REPO_ROOT / "scripts" / "infer" / "status.sh"


def _write_manifest(root: Path, run_id: str) -> Path:
    manifest_dir = root / "telemetry" / "container_runs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "telemetry" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_id}.log"
    log_path.write_text("")
    manifest = {
        "container_run_id": run_id,
        "storage": {"log_file": f"/telemetry/logs/{run_id}.log"},
    }
    manifest_path = manifest_dir / f"{run_id}.json"
    manifest_path.write_text(json.dumps(manifest))
    pointer = root / "telemetry" / "container_run_meta.env"
    pointer.write_text(
        f"CONTAINER_RUN_META_JSON=/telemetry/container_runs/{run_id}.json\n"
        f"CONTAINER_RUN_META_JSON_HOST={manifest_path}\n"
    )
    return manifest_path


@pytest.mark.usefixtures("fake_exec_bin", "temp_obs_root")
def test_status_script_behaviour(temp_obs_root):
    def _run(env):
        proc = run(["bash", str(STATUS_SCRIPT)], stdout=PIPE, text=True, env=env)
        return proc.stdout.strip()

    env = os.environ.copy()
    env["TMP_CURL_BEHAVIOR"] = "ready"
    assert _run(env) == "ready"

    env["TMP_CURL_BEHAVIOR"] = "starting"
    assert _run(env) == "starting"

    env.pop("TMP_CURL_BEHAVIOR")
    assert _run(env) == "down"


@pytest.mark.usefixtures("fake_exec_bin")
def test_start_server_short_circuits_when_ready(temp_obs_root, monkeypatch):
    run_id = "container-run-test"
    _write_manifest(temp_obs_root, run_id)

    docker_log = temp_obs_root / "docker.log"
    stub = temp_obs_root / "docker_stub.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == ps ]]; then exit 0; fi\n"
        'if [[ $1 == exec ]]; then echo exec >> "${TMP_DOCKER_LOG}"; exit 0; fi\n'
    )
    stub.chmod(0o755)
    monkeypatch.setenv("TMP_DOCKER_LOG", str(docker_log))
    docker_path = Path(os.environ["PATH"].split(":")[0]) / "docker"
    docker_path.write_text(stub.read_text())
    docker_path.chmod(0o755)

    env = os.environ.copy()
    env["TMP_CURL_BEHAVIOR"] = "ready"
    proc = run(["bash", str(START_SCRIPT)], text=True, capture_output=True, env=env)
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip())
    assert isinstance(data["schema_version"], int)
    assert data["schema_version"] == 1
    assert data["health"] == "ready"
    assert not docker_log.exists()


@pytest.mark.usefixtures("fake_exec_bin")
def test_start_server_invokes_docker_with_overrides(
    temp_obs_root, monkeypatch, tmp_path
):
    run_id = "container-run-test"
    _write_manifest(temp_obs_root, run_id)

    state_file = tmp_path / "docker_state"
    docker_log = tmp_path / "docker_exec.log"

    handler = tmp_path / "handler.sh"
    handler.write_text(
        "#!/usr/bin/env bash\n"
        "state=$1\n"
        "log=$2\n"
        "shift 2\n"
        "if [[ ! -f $state ]]; then\n"
        "  cat <<'JSON'\n"
        '{"model": "/models/test", "kv": "fp8_e4m3", "mem": "0.93", "chunk": 1024, "ctx": 8192, "maxp": 8192, "maxt": 8192, "mamba": 7, "trace": 1, "otlp": "localhost:4317"}\n'
        "JSON\n"
        "  touch $state\n"
        "else\n"
        '  echo "$@" >> $log\n'
        "fi\n"
    )
    handler.chmod(0o755)

    stub = temp_obs_root / "docker_handler.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == ps ]]; then exit 0; fi\n"
        "if [[ $1 == exec ]]; then\n"
        f'  {handler} {state_file} {docker_log} "$@"\n'
        "  exit 0\n"
        "fi\n"
    )
    stub.chmod(0o755)
    docker_path = Path(os.environ["PATH"].split(":")[0]) / "docker"
    docker_path.write_text(stub.read_text())
    docker_path.chmod(0o755)

    env = os.environ.copy()
    env.setdefault("READY_TIMEOUT", "0")
    # Ensure new overrides are forwarded as flags
    env["MAX_MAMBA_CACHE_SIZE"] = "7"
    env["ENABLE_TRACE"] = "1"
    env["OLTP_TRACES_ENDPOINT"] = "localhost:4317"
    proc = run(["bash", str(START_SCRIPT)], text=True, capture_output=True, env=env)
    assert proc.returncode != 0
    assert docker_log.exists()
    recorded = docker_log.read_text().strip()
    assert "--model-path" in recorded
    assert "--mem-fraction-static" in recorded
    assert "--max-mamba-cache-size" in recorded
    assert "--enable-trace" in recorded
    assert "--oltp-traces-endpoint" in recorded
