import os
from pathlib import Path
from typing import List

import pytest


@pytest.fixture()
def temp_obs_root(tmp_path, monkeypatch):
    """Provide HOST_OBS_ROOT rooted in a pytest temp dir and ensure structure."""
    root = tmp_path / "obs"
    telemetry = root / "telemetry"
    for sub in ("logs", "prometheus", "jaeger", "container_runs"):
        (telemetry / sub).mkdir(parents=True, exist_ok=True)
    profiles = root / "profiles"
    for sub in (
        "triton",
        "torchinductor",
        "flashinfer",
        "deep_gemm",
        "moe_configs/configs",
        ".locks",
        ".in_progress",
    ):
        (profiles / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOST_OBS_ROOT", str(root))
    monkeypatch.setenv("HOST_TELEMETRY_ROOT", str(telemetry))
    monkeypatch.setenv("HOST_PROFILES_ROOT", str(profiles))
    monkeypatch.setenv("SGLANG_TEST_LOG_DIR", str(telemetry / "logs"))
    return root


@pytest.fixture()
def fake_exec_bin(tmp_path, monkeypatch):
    """Inject a shim directory into PATH with fake docker/curl binaries."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def write_exe(name: str, lines: List[str]):
        payload = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", *lines]) + "\n"
        path = bin_dir / name
        path.write_text(payload)
        path.chmod(0o755)
        return path

    docker_lines = [
        'echo docker "$@" >> "${TMP_DOCKER_LOG:-/tmp/pytest_docker.log}"',
        "if [[ $1 == ps ]]; then exit 0; fi",
        "if [[ $1 == exec ]]; then",
        '  if [[ -n "${TMP_DOCKER_EXEC_SCRIPT:-}" ]]; then',
        '    exec bash -lc "${TMP_DOCKER_EXEC_SCRIPT}"',
        "  fi",
        "  exit 0",
        "fi",
        "exit 0",
    ]
    write_exe("docker", docker_lines)

    curl_lines = [
        'behavior="${TMP_CURL_BEHAVIOR:-}"',
        "if [[ -n $behavior ]]; then",
        "  case $behavior in",
        "    ready)",
        "      exit 0",
        "      ;;",
        "    starting)",
        "      if [[ $1 == -fsS ]]; then exit 22; fi",
        "      # consume optional flags (--max-time, -o, -w) without affecting output",
        '      args=("$@")',
        "      for ((i=0; i<${#args[@]}; ++i)); do",
        '        case "${args[i]}" in',
        "          -w|--write-out|--max-time|-o)",
        "            ((i++))",
        "            ;;",
        "        esac",
        "      done",
        '      printf "503\\n"',
        "      exit 0",
        "      ;;",
        "    *) ;;",
        "  esac",
        "fi",
        "exit 1",
    ]
    write_exe("curl", curl_lines)

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return bin_dir
