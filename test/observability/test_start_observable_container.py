import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/start_observable_container.sh").resolve()


def run_start(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_overrides)
    # Force validate-only to avoid docker usage during tests
    env.setdefault("VALIDATE_ONLY", "1")
    # Use a unique container name just in case
    env.setdefault("CONTAINER_NAME", f"sglang-dev-test-{os.getpid()}")
    return subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env)


def expected_dirs(root: Path) -> list[Path]:
    return [
        root / "telemetry" / "logs",
        root / "telemetry" / "prometheus",
        root / "telemetry" / "jaeger",
        root / "telemetry" / "container_runs",
        root / "profiles",
        root / "profiles" / "triton",
        root / "profiles" / "torchinductor",
        root / "profiles" / "flashinfer",
        root / "profiles" / "deep_gemm",
        root / "profiles" / "moe_configs" / "configs",
        root / "profiles" / ".locks",
        root / "profiles" / ".in_progress",
        root / "models",
    ]


def test_blank_slate_creates_dirs(tmp_path: Path):
    host_root = tmp_path / "obs"
    res = run_start({"HOST_OBS_ROOT": str(host_root)})
    assert res.returncode == 0, res.stderr
    # Directories created
    for d in expected_dirs(host_root):
        assert d.is_dir(), f"missing {d}"
        st = d.stat()
        assert st.st_uid == os.getuid() and st.st_gid == os.getgid(), f"ownership mismatch for {d}"
    # In VALIDATE_ONLY mode, no container is launched and no pointer file is created
    run_meta = host_root / "telemetry" / "container_run_meta.env"
    assert not run_meta.exists(), "pointer file should not be created in validate-only"


def test_existing_dirs_pass_preflight(tmp_path: Path):
    host_root = tmp_path / "obs"
    # Pre-create directories
    for d in expected_dirs(host_root):
        d.mkdir(parents=True, exist_ok=True)
    res = run_start({"HOST_OBS_ROOT": str(host_root)})
    assert res.returncode == 0, res.stderr


def test_dir_owner_mismatch_bails(tmp_path: Path):
    host_root = tmp_path / "obs"
    # Pre-create directories
    for d in expected_dirs(host_root):
        d.mkdir(parents=True, exist_ok=True)
    # Simulate mismatch by expecting root ownership (0:0)
    res = run_start({
        "HOST_OBS_ROOT": str(host_root),
        "EXPECT_OWNER_UID": "0",
        "EXPECT_OWNER_GID": "0",
    })
    assert res.returncode == 3
    assert "ERROR:" in res.stderr
    assert "not 0:0" in res.stderr


def test_pointer_file_mismatch_bails(tmp_path: Path):
    host_root = tmp_path / "obs"
    # Pre-create directories and an existing pointer file
    for d in expected_dirs(host_root):
        d.mkdir(parents=True, exist_ok=True)
    run_meta = host_root / "telemetry" / "container_run_meta.env"
    run_meta.write_text("")
    # Expect a different owner only for pointer file
    res = run_start({
        "HOST_OBS_ROOT": str(host_root),
        "EXPECT_POINTER_OWNER_UID": "0",
        "EXPECT_POINTER_OWNER_GID": "0",
    })
    assert res.returncode == 3
    assert "ERROR:" in res.stderr
    assert str(run_meta) in res.stderr
    assert "not 0:0" in res.stderr
