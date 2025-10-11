import os
import time
import subprocess
from pathlib import Path

import pytest


def sh(cmd: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, env=env)


@pytest.mark.skip(reason="E2E container+GPU smoke; enable with SGL_E2E_CONTAINER_TESTS=1")
def test_pointer_created_and_deepgemm_ensure_smoke(tmp_path: Path):
    if os.environ.get("SGL_E2E_CONTAINER_TESTS") != "1":
        pytest.skip("set SGL_E2E_CONTAINER_TESTS=1 to run")

    host_root = tmp_path / "obs"
    env = os.environ.copy()
    env.update({"HOST_OBS_ROOT": str(host_root)})

    # Start container
    r = sh("./scripts/start_observable_container.sh", env=env)
    assert r.returncode == 0, r.stderr

    # Wait for pointer atomically created
    run_meta = host_root / "telemetry" / "container_run_meta.env"
    t0 = time.time()
    while time.time() - t0 < 90:
        if run_meta.exists():
            break
        time.sleep(1)
    assert run_meta.exists(), "pointer not created in time"
    content = run_meta.read_text()
    assert "CONTAINER_RUN_META_JSON_HOST=" in content
    manifest = Path(content.split("CONTAINER_RUN_META_JSON_HOST=",1)[1].splitlines()[0])
    assert manifest.exists(), "manifest not present"

    # Ensure DeepGEMM caching (should start compile)
    r = sh("./scripts/cache/populate_caches.sh --model /models/<YOUR_MODEL> --tp 1 --deep-gemm ensure --moe skip --flashinfer skip --inductor skip", env=env)
    assert r.returncode in (0, ), r.stderr

    # Stop and restart container
    sh("./scripts/stop_observable_container.sh", env=env)
    r = sh("./scripts/start_observable_container.sh", env=env)
    assert r.returncode == 0, r.stderr

    # Ensure again; should NOOP quickly
    r = sh("./scripts/cache/populate_caches.sh --model /models/<YOUR_MODEL> --tp 1 --deep-gemm ensure --moe skip --flashinfer skip --inductor skip", env=env)
    assert r.returncode in (0, ), r.stderr

