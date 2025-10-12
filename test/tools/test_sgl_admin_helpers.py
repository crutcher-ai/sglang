import importlib.util
import json
import sys
from pathlib import Path

import pytest


class _DummyTyperApp:
    def __init__(self, *args, **kwargs):
        pass

    def command(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def add_typer(self, *args, **kwargs):
        return None


class _DummyTyperModule:
    BadParameter = ValueError

    @staticmethod
    def Option(*args, default=None, **_kwargs):  # noqa:ARG002
        return default

    @staticmethod
    def Typer(*args, **kwargs):
        return _DummyTyperApp()


sys.modules.setdefault("typer", _DummyTyperModule())

REPO_ROOT = Path(__file__).resolve().parents[2]
SGL_ADMIN_PATH = REPO_ROOT / ".devcontainer" / "tools" / "sgl_admin.py"

spec = importlib.util.spec_from_file_location("sgl_admin", SGL_ADMIN_PATH)
sgl_admin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sgl_admin)  # type: ignore


def _fresh_env(tmp_path: Path):
    profiles = tmp_path / "profiles"
    jit = profiles / "flashinfer" / "90a"
    home = tmp_path / "home"
    (home / ".cache/flashinfer").mkdir(parents=True, exist_ok=True)
    env = {
        "HOME": str(home),
        "TRITON_CACHE_DIR": str(profiles / "triton"),
        "FLASHINFER_WORKSPACE_DIR": str(profiles / "flashinfer"),
        "TORCHINDUCTOR_CACHE_DIR": str(profiles / "torchinductor"),
        "SGLANG_DG_CACHE_DIR": str(profiles / "deep_gemm"),
        "SGLANG_MOE_CONFIG_DIR": str(profiles / "moe_configs"),
        "FLASHINFER_JIT_LOG_DIR": str(jit),
    }
    return env


def test_prepare_env_creates_directories_and_symlink(tmp_path, monkeypatch):
    env = _fresh_env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    prepared = sgl_admin._prepare_env()

    assert prepared["PYTHONPATH"] == "/workspaces/sglang/python"

    for key in (
        "TRITON_CACHE_DIR",
        "FLASHINFER_WORKSPACE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "SGLANG_DG_CACHE_DIR",
        "SGLANG_MOE_CONFIG_DIR",
    ):
        assert Path(prepared[key]).is_dir()

    home_jit = Path(prepared["HOME"]) / ".cache/flashinfer/90a"
    assert home_jit.is_symlink()
    assert home_jit.resolve() == Path(prepared["FLASHINFER_JIT_LOG_DIR"]).resolve()


def test_prepare_env_reuses_existing_symlink(tmp_path, monkeypatch):
    env = _fresh_env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    target = Path(env["FLASHINFER_JIT_LOG_DIR"])
    target.mkdir(parents=True, exist_ok=True)
    home_jit = Path(env["HOME"]) / ".cache/flashinfer/90a"
    home_jit.symlink_to(target)

    prepared = sgl_admin._prepare_env()
    assert home_jit.is_symlink()
    assert home_jit.resolve() == target


def test_merge_moe_configs_sorts_keys(tmp_path):
    target = tmp_path / "config.json"
    target.write_text(json.dumps({"512": {"a": 1}}))
    merged = sgl_admin._merge_moe_configs(target, {256: {"a": 2}, 1024: {"a": 3}})
    assert list(merged.keys()) == ["256", "512", "1024"]
    data = json.loads(target.read_text())
    assert list(data.keys()) == ["256", "512", "1024"]


@pytest.mark.parametrize(
    "spec,mode,values",
    [
        (None, "default", None),
        ("", "default", None),
        ("all", "all", None),
        ("2,4, 8", "list", [2, 4, 8]),
    ],
)
def test_normalize_moe_batch_spec_valid(spec, mode, values):
    resolved_mode, resolved_values = sgl_admin._normalize_moe_batch_spec(spec)
    assert resolved_mode == mode
    if values is None:
        assert resolved_values is None
    else:
        assert resolved_values == values


@pytest.mark.parametrize("spec", ["zero", "-1", "2,x"])
def test_normalize_moe_batch_spec_invalid(spec):
    with pytest.raises(Exception):
        sgl_admin._normalize_moe_batch_spec(spec)


def test_prom_probe_sets_ok_when_positive(monkeypatch):
    class DummyResp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    payload = {
        "status": "success",
        "data": {
            "result": [
                {"value": "1.0"},
            ]
        },
    }

    def fake_get(url, params=None, timeout=None):  # pylint: disable=unused-argument
        return DummyResp(payload)

    monkeypatch.setitem(
        sys.modules, "requests", type("R", (), {"get": staticmethod(fake_get)})
    )

    probe = sgl_admin._prom_probe("run-1")
    assert probe["ok"] is True
    assert probe["sample_count"] == 1


def test_prom_probe_handles_failure(monkeypatch):
    def fake_get(url, params=None, timeout=None):  # pylint: disable=unused-argument
        raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules, "requests", type("R", (), {"get": staticmethod(fake_get)})
    )
    probe = sgl_admin._prom_probe("run-1")
    assert probe["ok"] is False
    assert probe["sample_count"] == 0


def test_caches_ensure_reports_port_busy(tmp_path, monkeypatch, capfd):
    run_id = "container-run-port-busy"
    telemetry = tmp_path / "telemetry"
    container_runs = telemetry / "container_runs"
    logs_dir = telemetry / "logs"
    container_runs.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = container_runs / f"{run_id}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "container_run_id": run_id,
                "storage": {"log_file": f"/telemetry/logs/{run_id}.log"},
                "paths": {
                    "container": {
                        "telemetry_root": "/telemetry",
                        "log_file": f"/telemetry/logs/{run_id}.log",
                    },
                    "host": {
                        "telemetry_root": str(telemetry),
                        "log_file": str(logs_dir / f"{run_id}.log"),
                    },
                },
            }
        )
    )

    pointer = telemetry / "container_run_meta.env"
    pointer.write_text(
        f"CONTAINER_RUN_META_JSON={manifest_path}\n"
        f"CONTAINER_RUN_META_JSON_HOST={manifest_path}\n"
    )

    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RUN_META_FILE", str(pointer))
    monkeypatch.setenv("HOST_TELEMETRY_ROOT", str(telemetry))
    monkeypatch.setenv("HOST_PROFILES_ROOT", str(profiles_root))

    env_dirs = {
        "HOME": tmp_path / "home",
        "TRITON_CACHE_DIR": tmp_path / "cache" / "triton",
        "FLASHINFER_WORKSPACE_DIR": tmp_path / "cache" / "flashinfer",
        "TORCHINDUCTOR_CACHE_DIR": tmp_path / "cache" / "inductor",
        "SGLANG_DG_CACHE_DIR": tmp_path / "cache" / "deep_gemm",
        "SGLANG_MOE_CONFIG_DIR": tmp_path / "cache" / "moe_configs",
        "FLASHINFER_JIT_LOG_DIR": tmp_path / "cache" / "flashinfer" / "jit",
    }
    for path in env_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    def fake_prepare_env():
        env = {key: str(value) for key, value in env_dirs.items()}
        env.setdefault("PYTHONPATH", "/workspaces/sglang/python")
        return env

    monkeypatch.setattr(sgl_admin, "_prepare_env", fake_prepare_env)
    monkeypatch.setattr(sgl_admin, "_permissions_preflight", lambda *a, **k: None)
    monkeypatch.setattr(sgl_admin, "_acquire_lock", lambda *a, **k: True)
    monkeypatch.setattr(sgl_admin, "_release_lock", lambda *a, **k: None)
    monkeypatch.setattr(
        sgl_admin,
        "_device_info",
        lambda: {
            "device_name": "dummy",
            "torch_version": "0",
            "triton_version": "0",
            "flashinfer_version": "0",
            "compute_capability": "sm_00",
            "cuda": "0",
            "driver_version": "0",
        },
    )
    monkeypatch.setattr(
        sgl_admin, "_prom_probe", lambda _run_id: {"ok": True, "sample_count": 0}
    )
    monkeypatch.setattr(sgl_admin, "_pick_warmup_port_require_30000", lambda: None)

    sgl_admin.caches_ensure(
        model="/models/test",
        tp=1,
        deep_gemm="skip",
        moe="skip",
        flashinfer="ensure",
        inductor="ensure",
    )

    capfd.readouterr()

    prep_result = json.loads((container_runs / run_id / "prep_result.json").read_text())
    assert isinstance(prep_result["schema_version"], int)
    assert prep_result["schema_version"] == 1
    assert prep_result["status"] == "partial"
    assert prep_result["stages"]["flashinfer"]["error_type"] == "port_busy"
    assert prep_result["stages"]["inductor"]["error_type"] == "port_busy"
