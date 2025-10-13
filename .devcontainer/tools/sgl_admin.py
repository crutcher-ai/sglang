#!/usr/bin/env python3
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import typer  # type: ignore
except Exception:
    print("This CLI requires 'typer' (pip install typer).", file=sys.stderr)
    raise

__VERSION__ = "0.1.0"

app = typer.Typer(help="SGLang admin utilities (caches inspect/ensure)")
caches_app = typer.Typer(help="Cache management commands")
app.add_typer(caches_app, name="caches")

logger = logging.getLogger(__name__)


def _default_tools_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_json_config(env_var: str, default_name: str) -> Dict[str, Any]:
    path = os.environ.get(env_var)
    if not path:
        path = str(_default_tools_dir() / default_name)
    p = Path(path)
    if not p.exists():
        raise RuntimeError(
            f"Required config file not found: {path}. This file is required for cache operations."
        )
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse config JSON at {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read config file at {path}: {e}") from e


def _load_caching_config() -> Dict[str, Any]:
    # Fail-fast if caching-config.json is missing or invalid
    return _load_json_config("SGL_CACHING_CONFIG", "caching-config.json")


def _load_sglang_config() -> Dict[str, Any]:
    # Fail-fast if sglang-config.json is missing or invalid
    return _load_json_config("SGL_CONFIG", "sglang-config.json")


@app.command("version")
def version():
    """Print sgl-admin version."""
    print(__VERSION__)


def _read_env_ptr() -> Path:
    env_path = Path(
        os.environ.get("RUN_META_FILE", "/telemetry/container_run_meta.env")
    )
    return env_path


def _load_manifest_path() -> Path:
    ptr = _read_env_ptr()
    if not ptr.exists() or ptr.stat().st_size == 0:
        raise RuntimeError(f"manifest pointer not ready: {ptr}")
    manifest = None
    with ptr.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CONTAINER_RUN_META_JSON="):
                manifest = line.strip().split("=", 1)[1]
                break
    if not manifest:
        raise RuntimeError("CONTAINER_RUN_META_JSON not found in pointer file")
    p = Path(manifest)
    if not p.exists():
        raise RuntimeError(f"manifest missing: {p}")
    return p


def _load_manifest_paths() -> tuple[Path, Optional[Path]]:
    ptr = _read_env_ptr()
    if not ptr.exists() or ptr.stat().st_size == 0:
        raise RuntimeError(f"manifest pointer not ready: {ptr}")
    cont = None
    host = None
    with ptr.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CONTAINER_RUN_META_JSON_HOST="):
                host = Path(line.strip().split("=", 1)[1])
            if line.startswith("CONTAINER_RUN_META_JSON="):
                cont = Path(line.strip().split("=", 1)[1])
    if cont is None:
        raise RuntimeError("CONTAINER_RUN_META_JSON not found in pointer file")
    return cont, host


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _version_safe(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "unknown"


def _device_info() -> Dict[str, str]:
    torch_v = _version_safe("torch")
    triton_v = _version_safe("triton")
    flashinfer_v = _version_safe("flashinfer")
    device_name = "unknown"
    compute_capability = "unknown"
    try:
        import torch  # type: ignore

        device_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        compute_capability = f"sm_{cc[0]}{cc[1]}"
    except Exception:
        pass
    driver_version = "unknown"
    cuda = "unknown"
    try:
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                "nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1",
            ]
        )  # noqa: E501
        cuda = out.decode().strip() or "unknown"
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1",
            ]
        )  # noqa: E501
        driver_version = out.decode().strip() or "unknown"
    except Exception:
        pass
    return {
        "torch_version": torch_v,
        "triton_version": triton_v,
        "flashinfer_version": flashinfer_v,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "driver_version": driver_version,
        "cuda": cuda,
    }


def _lookup_uid_gid(user: str) -> tuple[int, int]:
    try:
        import pwd  # noqa: WPS433

        pw = pwd.getpwnam(user)
        return pw.pw_uid, pw.pw_gid
    except Exception:
        return (os.getuid(), os.getgid())


def _prepare_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HOME", "/home/devuser")
    env.setdefault("XDG_CACHE_HOME", "/profiles")
    env.setdefault("TRITON_CACHE_DIR", "/profiles/triton")
    env.setdefault("FLASHINFER_WORKSPACE_DIR", "/profiles/flashinfer")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/profiles/torchinductor")
    # Canonical DeepGEMM cache directory (persisted on host mounts)
    env.setdefault("SGL_DG_CACHE_DIR", "/profiles/deep_gemm")
    env.setdefault("SGLANG_MOE_CONFIG_DIR", "/profiles/moe_configs")
    env.setdefault("FLASHINFER_JIT_LOG_DIR", "/profiles/flashinfer/90a")
    env.setdefault("PYTHONPATH", "/workspaces/sglang/python")
    # Precreate dirs - fail fast if not writable
    for p in (
        Path(env["TRITON_CACHE_DIR"]),
        Path(env["FLASHINFER_WORKSPACE_DIR"]),
        Path(env["TORCHINDUCTOR_CACHE_DIR"]),
        Path(env["SGL_DG_CACHE_DIR"]),
        Path(env["SGLANG_MOE_CONFIG_DIR"]),
        Path(env["FLASHINFER_JIT_LOG_DIR"]),
        Path(env["HOME"]) / ".cache/flashinfer/90a",
    ):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create cache directory {p}: {e}. Ensure the directory is writable by devuser."
            ) from e

    # Enforce single FlashInfer JIT log path via symlink (fail-fast)
    jit_target = Path(env["FLASHINFER_JIT_LOG_DIR"]).resolve()
    home_jit = Path(env["HOME"]) / ".cache/flashinfer/90a"
    link_ready = False
    if home_jit.exists() or home_jit.is_symlink():
        try:
            if os.path.samefile(str(home_jit), str(jit_target)):
                link_ready = True
        except (FileNotFoundError, OSError):
            link_ready = False

        if not link_ready:
            if home_jit.is_symlink():
                try:
                    home_jit.unlink()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to remove stale FlashInfer symlink at {home_jit}: {e}"
                    ) from e
            else:
                # Preserve previous contents before replacing with symlink
                backup_suffix = time.strftime("%Y%m%dT%H%M%S")
                backup_path = (
                    home_jit.parent / f"{home_jit.name}.previous-{backup_suffix}"
                )
                counter = 0
                while backup_path.exists():
                    counter += 1
                    backup_path = (
                        home_jit.parent
                        / f"{home_jit.name}.previous-{backup_suffix}-{counter}"
                    )
                try:
                    home_jit.rename(backup_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to archive existing FlashInfer directory {home_jit}: {e}"
                    ) from e
    if not link_ready:
        try:
            home_jit.parent.mkdir(parents=True, exist_ok=True)
            home_jit.symlink_to(jit_target)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create FlashInfer JIT symlink at {home_jit}: {e}"
            ) from e

    # Ensure JIT log file is writable - fail if not
    jit_log = home_jit / "flashinfer_jit.log"
    if jit_log.exists() and not os.access(str(jit_log), os.W_OK):
        raise RuntimeError(
            f"FlashInfer JIT log {jit_log} exists but is not writable. Check ownership."
        )
    if not jit_log.exists():
        try:
            with open(jit_log, "a", encoding="utf-8"):
                pass
        except Exception as e:
            raise RuntimeError(
                f"Failed to create FlashInfer JIT log {jit_log}: {e}"
            ) from e
    return env


def _extract_server_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        model_cfg = config["model"]
        server_cfg = config["server"]
    except KeyError as e:
        raise RuntimeError(f"Missing required key in sglang-config.json: {e}") from e

    def _require(section: Dict[str, Any], key: str) -> Any:
        if key not in section:
            raise RuntimeError(f"Missing '{key}' in sglang-config.json section")
        return section[key]

    defaults = {
        "kv_cache_dtype": _require(model_cfg, "kv_cache_dtype"),
        "moe_dtype": _require(model_cfg, "moe_dtype"),
        "mem_fraction_static": float(_require(server_cfg, "mem_fraction_static")),
        "chunked_prefill_size": int(_require(server_cfg, "chunked_prefill_size")),
        "context_length": int(_require(server_cfg, "context_length")),
        "max_prefill_tokens": int(_require(server_cfg, "max_prefill_tokens")),
        "max_total_tokens": int(_require(server_cfg, "max_total_tokens")),
        "max_mamba_cache_size": int(server_cfg.get("max_mamba_cache_size", 1)),
    }
    defaults["default_model_path"] = (
        model_cfg.get("default_model_path") or ""
    ).strip() or None
    defaults["trust_remote_code"] = bool(server_cfg.get("trust_remote_code", True))
    return defaults


@dataclass
class StageResult:
    ran: bool
    status: str
    code: int
    dur: float
    artifacts: Dict[str, Any]
    error_type: Optional[str] = None
    warnings: Optional[list] = None
    errors: Optional[list] = None


def _normalize_moe_batch_spec(spec: Optional[str]) -> tuple[str, Optional[List[int]]]:
    """Return (mode, values) where mode is one of {'default','all','list'}."""
    if spec is None:
        return ("default", None)
    spec = spec.strip()
    if not spec:
        return ("default", None)
    if spec.lower() == "all":
        return ("all", None)
    batches: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise typer.BadParameter(
                f"Invalid batch size '{part}'. Use integers or 'all'."
            )
        value = int(part)
        if value <= 0:
            raise typer.BadParameter("Batch sizes must be positive integers.")
        batches.append(value)
    if not batches:
        return ("default", None)
    return ("list", batches)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_moe_configs(target: Path, new_entries: Dict[str, Any]) -> Dict[str, Any]:
    merged = _load_json(target)
    for key, value in new_entries.items():
        merged[str(key)] = value

    def _sort_key(key: str) -> Any:
        try:
            return int(key)
        except ValueError:
            return key

    ordered = {k: merged[k] for k in sorted(merged, key=_sort_key)}
    _atomic_write_json(target, ordered)
    return ordered


def _resolve_moe_dtype(flag: Optional[str], *, default: Optional[str] = None) -> str:
    """Resolve MOE dtype from CLI flag or provided default/config."""
    dtype = (flag or "").strip()
    if dtype:
        return dtype
    if default is not None:
        return default
    # Read from sglang-config.json
    config = _load_sglang_config()
    return config["model"]["moe_dtype"]  # Fail-fast if key missing


def _devuser_write_test(dir_path: Path, env: Optional[Dict[str, str]] = None) -> bool:
    """Attempt to create and delete a file in dir_path as devuser.
    Returns True if successful, False otherwise.
    """
    d = str(dir_path)
    # ensure directory exists first (may still fail if parent isn't writable)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    snippet = (
        f'set -e; d={json.dumps(d)}; mkdir -p "$d"; '
        f't="$d/.sgl_write_test_$$"; echo ok > "$t"; rm -f "$t"'
    )
    try:
        run_cmd = (
            f"sudo -u devuser -E bash -lc {json.dumps(snippet)}"
            if Path("/home/devuser").exists()
            else f"bash -lc {json.dumps(snippet)}"
        )
        rc = subprocess.call(["bash", "-lc", run_cmd], env=env)
        return rc == 0
    except Exception:
        return False


def _maybe_fix_perms(dir_path: Path, env: Optional[Dict[str, str]] = None) -> bool:
    """Optionally attempt recursive chown to devuser when SGL_FIX_PERMS is set.
    Returns True if a subsequent write test passes, False otherwise.
    """
    fix_flag = os.environ.get("SGL_FIX_PERMS", "").lower() in {"1", "true", "yes"}
    if not fix_flag:
        return False
    uid, gid = _lookup_uid_gid("devuser")
    try:
        subprocess.check_call(
            [
                "bash",
                "-lc",
                f"chown -R {uid}:{gid} {json.dumps(str(dir_path))} || true",
            ]
        )
    except Exception:
        pass
    return _devuser_write_test(dir_path, env)


def _permissions_preflight(
    stage: str, code: int, paths: list[Path], env: Dict[str, str]
) -> Optional[StageResult]:
    """Preflight temporarily disabled: unified devuser writes + symlinked JIT path avoid drift."""
    return None


def _verify_moe_consumption_in_log(
    log_file: Optional[Path], stages: Dict[str, StageResult]
) -> None:
    """Set verified_in_log true for moe_tune if warm-ups succeeded and log contains the config consumption line."""
    if not log_file or not isinstance(log_file, Path) or not log_file.exists():
        return
    moe = stages.get("moe_tune")
    if not moe or moe.status != "ok":
        return
    # Only verify when at least one warm-up stage succeeded (server would have run)
    warm_ok = any(
        (stages.get(n) and stages[n].status == "ok") for n in ("flashinfer", "inductor")
    )
    if not warm_ok:
        return
    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    m = re.search(r"Using MoE kernel config from\s+(.+?)\.?\s*$", text, re.MULTILINE)
    if not m:
        return
    path_str = m.group(1).strip()
    consuming_base = os.path.basename(path_str)
    # Ensure artifacts dict is mutable
    moe.artifacts = moe.artifacts or {}
    moe.artifacts["verified_in_log"] = True
    moe.artifacts["consuming_config_basename"] = consuming_base
    moe.artifacts.setdefault(
        "verify_log", {"container_path": str(log_file), "host_path": None}
    )


def _dir_stats(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "file_count": 0,
            "latest_mtime_iso": None,
        }
    size = 0
    count = 0
    latest = 0.0
    for root, _, files in os.walk(p):
        for fn in files:
            fp = Path(root) / fn
            try:
                st = fp.stat()
            except FileNotFoundError:
                continue
            size += st.st_size
            count += 1
            if st.st_mtime > latest:
                latest = st.st_mtime
    iso = (
        None
        if latest == 0
        else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(latest))
    )
    return {
        "exists": True,
        "size_bytes": size,
        "file_count": count,
        "latest_mtime_iso": iso,
    }


def _flashinfer_signature(
    info: Dict[str, Any],
    server_defaults: Dict[str, Any],
    model_slug: Optional[str],
    tp: int,
) -> Dict[str, Any]:
    return {
        "model_slug": model_slug,
        "tp": tp,
        "device_name": info.get("device_name"),
        "compute_capability": info.get("compute_capability"),
        "cuda": info.get("cuda"),
        "driver_version": info.get("driver_version"),
        "torch_version": info.get("torch_version"),
        "triton_version": info.get("triton_version"),
        "flashinfer_version": info.get("flashinfer_version"),
        "kv_cache_dtype": server_defaults.get("kv_cache_dtype"),
        "mem_fraction_static": server_defaults.get("mem_fraction_static"),
        "chunked_prefill_size": server_defaults.get("chunked_prefill_size"),
        "context_length": server_defaults.get("context_length"),
        "max_prefill_tokens": server_defaults.get("max_prefill_tokens"),
        "max_total_tokens": server_defaults.get("max_total_tokens"),
    }


@app.command("inspect")
@caches_app.command("inspect")
def caches_inspect():
    """Emit a live snapshot of caches under /profiles (container paths) as JSON."""
    profiles = Path("/profiles")
    envsig = _device_info()
    envsig.update({"sglang_commit": "unknown", "tp_size": None, "model_slug": None})
    snapshot: Dict[str, Any] = {"schema_version": "1"}
    for key in ("triton", "torchinductor", "flashinfer", "deep_gemm"):
        sub = profiles / key
        s = _dir_stats(sub)
        s.update({"valid": bool(s["exists"]), "path": str(sub), "signature": envsig})
        # partial markers
        mroot = profiles / ".in_progress"
        partial = False
        pinfo = None
        if mroot.exists():
            for m in mroot.rglob("*"):
                if m.is_file():
                    try:
                        pinfo = json.loads(m.read_text())
                    except Exception:
                        pinfo = {"owner_pid": None, "started_at": None}
                    partial = True
                    break
        s["partial"] = partial
        if partial:
            s["valid"] = False
            s["reason"] = "in_progress_or_aborted"
            s["partial_info"] = pinfo
        snapshot[key] = s
    print(json.dumps(snapshot, indent=2))


def _run_subprocess(
    cmd: str,
    log_file: Optional[Path],
    heartbeat_name: str,
    env: Optional[Dict[str, str]] = None,
) -> int:
    start = time.time()
    # Prefer to run as devuser to avoid root-owned artefacts on host mounts
    run_cmd = f"bash -lc {json.dumps(cmd)}"
    proc = subprocess.Popen(
        ["bash", "-lc", run_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    hb_last = start
    with (
        log_file.open("a", encoding="utf-8") if log_file else open(os.devnull, "w")
    ) as lf:
        for line in proc.stdout:  # type: ignore[attr-defined]
            lf.write(line)
            now = time.time()
            if now - hb_last >= 60:
                hb_last = now
                print(
                    json.dumps(
                        {
                            "name": heartbeat_name,
                            "phase": "progress",
                            "elapsed_s": int(now - start),
                        }
                    )
                )
    return proc.wait()


def _port_is_free(port: int) -> bool:
    try:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
        return True
    except Exception:
        return False


def _pick_warmup_port_require_30000() -> Optional[int]:
    return 30000 if _port_is_free(30000) else None


def _start_warmup_server(
    model: str,
    tp: int,
    port: int,
    server_args: Dict[str, Any],
    log_file: Optional[Path],
    enable_compile: bool,
    env: Optional[Dict[str, str]],
    trust_remote_code: bool,
) -> subprocess.Popen:
    args = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--enable-metrics",
        "--tp-size",
        str(tp),
        "--mem-fraction-static",
        str(server_args["mem_fraction_static"]),
        "--kv-cache-dtype",
        str(server_args["kv_cache_dtype"]),
        "--chunked-prefill-size",
        str(server_args["chunked_prefill_size"]),
        "--max-mamba-cache-size",
        str(server_args["max_mamba_cache_size"]),
        "--context-length",
        str(server_args["context_length"]),
        "--max-prefill-tokens",
        str(server_args["max_prefill_tokens"]),
        "--max-total-tokens",
        str(server_args["max_total_tokens"]),
        # memory-safe warm-up defaults for request concurrency
        "--max-running-requests",
        "1",
        "--max-queued-requests",
        "1",
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    else:
        args.append("--no-trust-remote-code")
    if enable_compile:
        args.extend(["--enable-torch-compile", "--torch-compile-max-bs", "1"])
    stdout = log_file.open("a", encoding="utf-8") if log_file else subprocess.DEVNULL
    cmd = " ".join(map(lambda s: json.dumps(s), args))
    proc = subprocess.Popen(
        ["bash", "-lc", cmd], stdout=stdout, stderr=subprocess.STDOUT, env=env
    )
    return proc


def _wait_http_ready(port: int, timeout: int = 90) -> bool:
    import requests

    url = f"http://127.0.0.1:{port}/get_model_info"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _send_warmup_requests(port: int, dp_size: int = 1) -> bool:
    import requests

    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": "The capital city of France is",
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _stop_server(proc: subprocess.Popen, timeout_s: int = 10) -> Dict[str, Any]:
    cleanup = {"kill_sent": True, "killed": False, "timeout_s": timeout_s}
    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
            cleanup["killed"] = True
        except subprocess.TimeoutExpired:
            proc.kill()
            cleanup["killed"] = True
    except Exception:
        cleanup["killed"] = False
    return cleanup


def _prom_probe(run_id: str) -> Dict[str, Any]:
    # Best-effort: query local Prometheus for recent token increases under this run label
    import requests

    probe = {
        "prometheus_query": "increase(sglang:prompt_tokens_total[1m])",
        "with_run_filter": f'increase(sglang:prompt_tokens_total{{container_run="{run_id}"}}[1m])',
        "sample_count": 0,
        "ok": False,
    }
    try:
        q = {
            "query": probe["with_run_filter"],
        }
        r = requests.get("http://127.0.0.1:9090/api/v1/query", params=q, timeout=3)
        data = r.json()
        if data.get("status") == "success":
            res = data.get("data", {}).get("result", [])
            probe["sample_count"] = len(res)
            # consider ok if any vector result has a positive value
            ok = False
            for it in res:
                try:
                    val = float(it.get("value", [0, "0"][1]))
                    if val > 0:
                        ok = True
                        break
                except Exception:
                    continue
            probe["ok"] = ok
    except Exception:
        pass
    return probe


def _acquire_lock(slug: str, timeout: int = 600) -> bool:
    locks = Path("/profiles/.locks")
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / f"{slug}.lock"
    start = time.time()
    while lock.exists():
        try:
            st = lock.stat()
            held_for = int(time.time() - st.st_mtime)
        except FileNotFoundError:
            continue
        print(json.dumps({"phase": "wait_lock", "slug": slug, "held_for_s": held_for}))
        if held_for > timeout:
            try:
                lock.unlink(missing_ok=True)
                print(
                    json.dumps(
                        {
                            "phase": "lock_reclaimed",
                            "slug": slug,
                            "held_for_s": held_for,
                        }
                    )
                )
                break
            except Exception:
                pass
        if time.time() - start > timeout:
            return False
        time.sleep(30)
    try:
        lock.write_text(str(os.getpid()))
        return True
    except Exception:
        return False


def _release_lock(slug: str) -> None:
    lock = Path("/profiles/.locks") / f"{slug}.lock"
    try:
        lock.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def _mark_in_progress(stage: str, started_at: str) -> Path:
    mroot = Path("/profiles/.in_progress")
    mroot.mkdir(parents=True, exist_ok=True)
    mark = mroot / f"{stage}.json"
    try:
        _atomic_write_json(mark, {"owner_pid": os.getpid(), "started_at": started_at})
    except Exception:
        pass
    return mark


def _clear_in_progress(mark: Path) -> None:
    try:
        mark.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


@app.command("ensure")
@caches_app.command("ensure")
def caches_ensure(
    model: Optional[str] = typer.Option(None, "--model", help="Model ID/path"),
    tp: int = typer.Option(
        1, "--tp", help="Tensor parallel size (default 1 on single GH200)"
    ),
    deep_gemm: str = typer.Option("ensure", "--deep-gemm", help="ensure|rebuild|skip"),
    moe: str = typer.Option("ensure", "--moe", help="ensure|rebuild|skip"),
    flashinfer: str = typer.Option(
        "ensure", "--flashinfer", help="ensure|rebuild|skip"
    ),
    inductor: str = typer.Option("ensure", "--inductor", help="ensure|rebuild|skip"),
    moe_batch_sizes: Optional[str] = typer.Option(
        None,
        "--moe-batch-sizes",
        help="Comma separated list of batch sizes (e.g. '512,4096') or 'all' (default).",
    ),
    moe_dtype: Optional[str] = typer.Option(
        None,
        "--moe-dtype",
        help="Optional dtype override passed to the Triton tuner (e.g. fp8_w8a8).",
    ),
    json_out: bool = typer.Option(
        True,
        "--json/--no-json",
        help="Deprecated; inspect always prints JSON. Ignored here.",
    ),
    prom_ping: bool = typer.Option(
        False,
        "--prom-ping/--no-prom-ping",
        help="After prep, attempt a trivial /generate request to the local server to ensure Prometheus records activity.",
    ),
):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    manifest_path, manifest_host_path = _load_manifest_paths()
    data = json.loads(manifest_path.read_text())
    run_id = data.get("container_run_id")
    run_dir = manifest_path.parent
    log_file = (
        Path(data.get("storage", {}).get("log_file", ""))
        if data.get("storage")
        else None
    )

    # Write prep_result.json under a run-scoped directory for stability
    try:
        run_out_dir = run_dir / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        run_out_dir = run_dir
    prep_path = run_out_dir / "prep_result.json"
    info = _device_info()
    # spec-file support removed
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Fallback commit if manifest didn't include it
    git_commit = data.get("git_revision", "unknown")
    if git_commit == "unknown":
        try:
            out = subprocess.check_output(
                ["bash", "-lc", "git -C /workspaces/sglang rev-parse HEAD"],
                stderr=subprocess.DEVNULL,
            )
            git_commit = out.decode().strip() or "unknown"
        except Exception:
            pass

    ignored_spec_keys: list[str] = []

    # Load configs
    sglang_config = _load_sglang_config()
    server_defaults = _extract_server_defaults(sglang_config)
    caching_config = _load_caching_config()

    if model is None or not str(model).strip():
        model_from_config = server_defaults.get("default_model_path")
        if not model_from_config:
            raise RuntimeError(
                "No model provided via --model and default_model_path is missing in sglang-config.json."
                " Update the config or pass --model explicitly."
            )
        model = model_from_config

    # Determine model slug (basename if path-like)
    model_slug = None
    if model:
        try:
            model_slug = os.path.basename(str(model).rstrip("/")) or model
        except Exception:
            model_slug = model

    server_args_snapshot = {
        "mem_fraction_static": server_defaults["mem_fraction_static"],
        "kv_cache_dtype": server_defaults["kv_cache_dtype"],
        "chunked_prefill_size": server_defaults["chunked_prefill_size"],
        "context_length": server_defaults["context_length"],
        "max_prefill_tokens": server_defaults["max_prefill_tokens"],
        "max_total_tokens": server_defaults["max_total_tokens"],
        "max_mamba_cache_size": server_defaults["max_mamba_cache_size"],
        "trust_remote_code": server_defaults.get("trust_remote_code", True),
    }
    run_obj: Dict[str, Any] = {
        "schema_version": 1,
        "status": "partial",
        "run": {
            "run_id": run_id,
            "model_slug": model_slug,
            "tp": tp,
            "device_name": info["device_name"],
            "compute_capability": info["compute_capability"],
            "cuda": info["cuda"],
            "driver_version": info["driver_version"],
            "torch_version": info["torch_version"],
            "triton_version": info["triton_version"],
            "flashinfer_version": info["flashinfer_version"],
            "sglang_commit": git_commit,
            "started_at": started_at,
            "finished_at": None,
            "duration_s": None,
            "settings": {
                "source": "flags",
                "spec_version": "1",
                "tp": tp,
                "blocks": [],
                "warmup_port": 30000,
                "mem_fraction": server_args_snapshot["mem_fraction_static"],
                "write_mode": "atomic",
                "env": {
                    "TRITON_CACHE_DIR": "/profiles/triton",
                    "TORCHINDUCTOR_CACHE_DIR": "/profiles/torchinductor",
                    "FLASHINFER_WORKSPACE_DIR": "/profiles/flashinfer",
                    "SGL_DG_CACHE_DIR": "/profiles/deep_gemm",
                    "SGLANG_MOE_CONFIG_DIR": "/profiles/moe_configs",
                    "FLASHINFER_JIT_LOG_DIR": "/profiles/flashinfer/90a",
                },
                "ignored_spec_keys": ignored_spec_keys,
                "server_defaults": server_args_snapshot,
            },
        },
        "stages": {},
        "telemetry_probe": {
            "prometheus_query": "increase(sglang:prompt_tokens_total[1m])",
            "with_run_filter": f'increase(sglang:prompt_tokens_total{{container_run="{run_id}"}}[1m])',
            "sample_count": 0,
            "ok": False,
        },
        "errors": [],
        "warnings": [],
    }

    flashinfer_timeout = int(
        caching_config.get("flashinfer", {}).get("warmup_timeout_s", 300)
    )
    inductor_timeout = int(
        caching_config.get("inductor", {}).get("warmup_timeout_s", 300)
    )
    moe_lock_timeout = int(caching_config.get("moe", {}).get("lock_timeout_s", 600))

    # MOE batch sizes: CLI flag overrides config
    if moe_batch_sizes is not None:
        batch_spec = moe_batch_sizes
    else:
        # Read from caching-config.json
        moe_config = caching_config.get("moe", {})
        batch_list = moe_config.get("batch_sizes", [])
        batch_spec = ",".join(str(b) for b in batch_list) if batch_list else None

    batch_mode, batch_values = _normalize_moe_batch_spec(batch_spec)
    resolved_moe_dtype = _resolve_moe_dtype(
        moe_dtype, default=server_defaults["moe_dtype"]
    )

    if batch_mode == "list" and batch_values:
        recorded_spec = ",".join(str(v) for v in batch_values)
    elif batch_mode in {"default", "all"}:
        recorded_spec = batch_spec or "all"
    else:
        recorded_spec = batch_spec

    run_obj["run"]["settings"].setdefault("moe", {})
    run_obj["run"]["settings"]["moe"].update(
        {
            "mode": moe,
            "batch_spec": recorded_spec,
            "dtype": resolved_moe_dtype,
        }
    )

    t0 = time.time()
    env = _prepare_env()
    stages: Dict[str, StageResult] = {}

    # DeepGEMM
    if deep_gemm != "skip":
        # Fail-fast permissions and disk preflight
        dg_dir = Path(env["SGL_DG_CACHE_DIR"])
        try:
            dg_dir.mkdir(parents=True, exist_ok=True)
            testf = dg_dir / ".write_test"
            with open(testf, "w", encoding="utf-8") as f:
                f.write("ok")
            testf.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            stages["deep_gemm"] = StageResult(
                True,
                "error",
                10,
                0.0,
                {},
                "permissions_unexpected",
                errors=[f"not writable: {dg_dir}"],
            )
        else:
            # Signature/noop check
            sig = {
                "device_name": info["device_name"],
                "compute_capability": info["compute_capability"],
                "cuda": info["cuda"],
                "driver_version": info["driver_version"],
                "torch_version": info["torch_version"],
                "triton_version": info["triton_version"],
                "sglang_commit": git_commit,
                "tp": tp,
                "model_slug": run_obj["run"]["model_slug"],
                "mem_fraction_static": server_args_snapshot["mem_fraction_static"],
                "kv_cache_dtype": server_args_snapshot["kv_cache_dtype"],
                "chunked_prefill_size": server_args_snapshot["chunked_prefill_size"],
                "context_length": server_args_snapshot["context_length"],
            }
            sig_dir = dg_dir / (run_obj["run"]["model_slug"] or "default")
            sig_file = sig_dir / "signature.json"
            if deep_gemm == "ensure" and sig_file.exists():
                try:
                    prior = json.loads(sig_file.read_text())
                except Exception:
                    prior = None
                if prior == sig:
                    stages["deep_gemm"] = StageResult(
                        False, "noop", 0, 0.0, {"signature": sig}
                    )
            if "deep_gemm" not in stages:
                # Proceed to compile
                dg_def = caching_config["deep_gemm"]  # Fail-fast if key missing
                # Allow env override, otherwise use config value (no fallback)
                lock_to = int(
                    os.environ.get("SGL_LOCK_TIMEOUT_DEEPGEMM")
                    or dg_def["lock_timeout_s"]
                )
                if not _acquire_lock("deep_gemm", timeout=lock_to):
                    stages["deep_gemm"] = StageResult(
                        True,
                        "error",
                        10,
                        0.0,
                        {},
                        "lock_timeout",
                        errors=["lock held too long"],
                    )  # noqa: E501
                else:
                    mark = _mark_in_progress("deep_gemm", started_at)
                    try:
                        sig_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    compile_log = sig_dir / "compile.log"
                    # Disable metrics/tracing for compile to avoid collisions
                    env2 = env.copy()
                    memf = str(
                        dg_def["mem_fraction_static"]
                    )  # Fail-fast if key missing
                    env2["SGL_COMPILE_MEM_FRACTION"] = memf
                    env2["SGLANG_COMPILE_MEM_FRACTION"] = memf
                    # Ensure DeepGEMM sees the canonical cache dir
                    if "SGL_DG_CACHE_DIR" in env:
                        env2["SGL_DG_CACHE_DIR"] = env["SGL_DG_CACHE_DIR"]
                    compile_to = int(
                        dg_def["compile_timeout_s"]
                    )  # Fail-fast if key missing
                    compile_cmd_parts = [
                        "python",
                        "-m",
                        "sglang.compile_deep_gemm",
                        "--model-path",
                        model,
                        "--tp",
                        str(tp),
                        "--timeout",
                        str(compile_to),
                        "--kv-cache-dtype",
                        str(server_args_snapshot["kv_cache_dtype"]),
                        "--mem-fraction-static",
                        memf,
                        "--chunked-prefill-size",
                        str(server_args_snapshot["chunked_prefill_size"]),
                        "--context-length",
                        str(server_args_snapshot["context_length"]),
                        "--max-prefill-tokens",
                        str(server_args_snapshot["max_prefill_tokens"]),
                        "--max-total-tokens",
                        str(server_args_snapshot["max_total_tokens"]),
                        "--max-mamba-cache-size",
                        str(server_args_snapshot["max_mamba_cache_size"]),
                    ]
                    if server_defaults.get("trust_remote_code", True):
                        compile_cmd_parts.append("--trust-remote-code")
                    else:
                        compile_cmd_parts.append("--no-trust-remote-code")
                    compile_cmd = " ".join(json.dumps(p) for p in compile_cmd_parts)
                    cmd = f"{compile_cmd} 2>&1 | tee -a {json.dumps(str(compile_log))}"
                    t = time.time()
                    rc = _run_subprocess(cmd, log_file, "deep_gemm", env=env2)
                    dur = time.time() - t
                    if rc == 0:
                        try:
                            _atomic_write_json(sig_file, sig)
                        except Exception:
                            pass
                        # Validate compile.log contains expected config
                        warnings = []
                        try:
                            log_text = compile_log.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                            # Check for expected kv_cache_dtype (use configured value from signature)
                            expected_dtype = sig["kv_cache_dtype"]
                            if expected_dtype.lower() not in log_text.lower():
                                warnings.append(
                                    f"compile.log does not contain '{expected_dtype}' - kv_cache_dtype may not have been applied"
                                )
                            # Check for mem_fraction errors
                            if (
                                "not enough memory" in log_text.lower()
                                and "mem-fraction-static" in log_text.lower()
                            ):
                                warnings.append(
                                    "compile.log contains memory errors - mem_fraction_static may be incorrect"
                                )
                        except Exception as e:
                            warnings.append(f"Failed to validate compile.log: {e}")

                        artifacts = {
                            "cache_dir": {
                                "container_path": env["SGL_DG_CACHE_DIR"],
                                "host_path": None,
                            },
                            "compile_log": {
                                "container_path": str(compile_log),
                                "host_path": None,
                            },
                            "signature": sig,
                        }
                        stages["deep_gemm"] = StageResult(
                            True,
                            "ok",
                            0,
                            dur,
                            artifacts,
                            warnings=warnings if warnings else None,
                        )
                        _clear_in_progress(mark)
                    else:
                        err_type = "nvrtc_compile_failed"
                        try:
                            tail = []
                            with open(
                                compile_log, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                lines = f.readlines()
                                tail = [l.rstrip() for l in lines[-20:]]
                            if any("libcuda" in l.lower() for l in tail):
                                err_type = "libcuda_missing"
                            elif any("nvrtc" in l.lower() for l in tail):
                                err_type = "nvrtc_not_found"
                        except Exception:
                            tail = []
                        artifacts = {
                            "compile_log": {
                                "container_path": str(compile_log),
                                "host_path": None,
                            },
                            "compile_log_tail": tail,
                        }
                        stages["deep_gemm"] = StageResult(
                            True,
                            "error",
                            10,
                            dur,
                            artifacts,
                            err_type,
                            errors=["compile_deep_gemm failed"],
                        )  # noqa: E501
                    _release_lock("deep_gemm")
    else:
        stages["deep_gemm"] = StageResult(False, "skipped", 10, 0.0, {})

    # MoE tuning (best-effort; requires ray)
    if moe != "skip":
        # permissions preflight for moe configs
        pf = _permissions_preflight(
            "moe_tune", 11, [Path(env["SGLANG_MOE_CONFIG_DIR"])], env
        )
        if pf:
            stages["moe_tune"] = pf
        else:
            # check ray presence
            try:
                __import__("ray")
                triton_u = info["triton_version"].replace(".", "_")
                out_dir = Path(f"/profiles/moe_configs/configs/triton_{triton_u}")
                out_dir.mkdir(parents=True, exist_ok=True)
                dtype_arg = (
                    f" --dtype {resolved_moe_dtype}" if resolved_moe_dtype else ""
                )
                batch_targets: List[Optional[int]]
                if batch_mode == "list" and batch_values:
                    batch_targets = [int(v) for v in batch_values]
                else:
                    batch_targets = [None]

                if not _acquire_lock("moe_tune", timeout=moe_lock_timeout):
                    stages["moe_tune"] = StageResult(
                        True,
                        "error",
                        11,
                        0.0,
                        {},
                        "lock_timeout",
                        errors=["moe tuner lock held too long"],
                    )
                else:
                    mark = _mark_in_progress("moe_tune", started_at)
                    total_dur = 0.0
                    completed_batches: List[int] = []
                    skipped_batches: List[int] = []
                    artifacts: Dict[str, Any] = {}
                    error_result: Optional[StageResult] = None
                    aggregated_entries: Dict[str, Any] = {}
                    pattern = (
                        "*.json"
                        if not resolved_moe_dtype
                        else f"*dtype={resolved_moe_dtype}*.json"
                    )
                    existing_configs = sorted(
                        out_dir.glob(pattern),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if existing_configs:
                        try:
                            aggregated_entries = _load_json(existing_configs[0])
                        except Exception as exc:
                            logger.warning(
                                "Failed to load existing MoE config %s: %s",
                                existing_configs[0],
                                exc,
                            )

                    for batch in batch_targets:
                        batch_arg = "" if batch is None else f" --batch-size {batch}"
                        if batch is not None and str(batch) in aggregated_entries:
                            skipped_batches.append(int(batch))
                            continue
                        cmd = (
                            f"cd {out_dir} && python /workspaces/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py "
                            f"--model '{model}' --tp {tp} --tune{dtype_arg}{batch_arg}"
                        )
                        t = time.time()
                        rc = _run_subprocess(cmd, log_file, "moe_tune", env=env)
                        run_dur = time.time() - t
                        total_dur += run_dur
                        if rc != 0:
                            error_result = StageResult(
                                True,
                                "error",
                                11,
                                total_dur,
                                {},
                                "tuner_execution_error",
                                errors=["moe tuner failed"],
                            )  # noqa: E501
                            break

                        cfgs = sorted(
                            out_dir.glob("*.json"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        if not cfgs:
                            error_result = StageResult(
                                True,
                                "error",
                                11,
                                total_dur,
                                {},
                                "config_missing",
                                errors=["moe tuner produced no configs"],
                            )  # noqa: E501
                            break
                        cfg = cfgs[0]
                        try:
                            new_entries = _load_json(cfg)
                        except Exception as exc:
                            error_result = StageResult(
                                True,
                                "error",
                                11,
                                total_dur,
                                {},
                                "config_parse_error",
                                errors=[f"{exc}"],
                            )
                            break
                        for key, value in new_entries.items():
                            aggregated_entries[str(key)] = value
                        merged = _merge_moe_configs(cfg, aggregated_entries)
                        completed_batches.extend(int(k) for k in new_entries.keys())

                        aggregated_entries = merged
                        artifacts = {
                            "config_file": {
                                "container_path": str(cfg),
                                "host_path": None,
                            },
                            "config_hash": None,
                            "triton_version": info["triton_version"],
                            "verified_in_log": False,
                            "available_batch_sizes": sorted(
                                int(k) for k in merged.keys()
                            ),
                            "new_batch_sizes": sorted(set(completed_batches)),
                            "skipped_batch_sizes": sorted(set(skipped_batches)),
                            "dtype": resolved_moe_dtype,
                        }
                    if error_result:
                        stages["moe_tune"] = error_result
                    else:
                        if completed_batches:
                            stages["moe_tune"] = StageResult(
                                True, "ok", 0, total_dur, artifacts
                            )
                        else:
                            if not artifacts:
                                artifacts = {
                                    "available_batch_sizes": sorted(
                                        int(k) for k in aggregated_entries.keys()
                                    ),
                                    "new_batch_sizes": [],
                                    "skipped_batch_sizes": sorted(set(skipped_batches)),
                                    "dtype": resolved_moe_dtype,
                                }
                            stages["moe_tune"] = StageResult(
                                False, "noop", 0, 0.0, artifacts
                            )
                    _clear_in_progress(mark)
                    _release_lock("moe_tune")
            except Exception:
                # If explicitly requested and ray missing, return error instead of noop
                stages["moe_tune"] = StageResult(
                    True,
                    "error",
                    11,
                    0.0,
                    {},
                    "ray_missing",
                    errors=["ray not available; cannot run tuner"],
                )  # noqa: E501
    else:
        stages["moe_tune"] = StageResult(False, "skipped", 11, 0.0, {})

    # FlashInfer + TorchInductor warm-up via short-lived server
    # Only if a model is provided
    if model:
        # Pick warm-up port preferring Prom-scraped 30000
        port = _pick_warmup_port_require_30000()
        if port is None:
            stages["flashinfer"] = StageResult(
                True, "error", 12, 0.0, {}, "port_busy", errors=["port 30000 is busy"]
            )  # noqa: E501
            stages["inductor"] = StageResult(
                True, "error", 13, 0.0, {}, "port_busy", errors=["port 30000 is busy"]
            )  # noqa: E501
            # finalize early
            run_obj["status"] = "partial"
            run_obj["errors"] = [
                name for name, result in stages.items() if result.status == "error"
            ]
            for name, result in stages.items():
                run_obj["stages"][name] = {
                    "ran": result.ran,
                    "status": result.status,
                    "status_code": result.code,
                    "duration_s": round(result.dur, 3) if result.dur else 0,
                    "artifacts": result.artifacts,
                    "error_type": result.error_type,
                    "warnings": result.warnings or [],
                    "errors": result.errors or [],
                }
            _atomic_write_json(prep_path, run_obj)
            print(f"RESULT_JSON {prep_path}")
            summary = " ".join(
                [
                    f"{k}:{stages[k].status}"
                    for k in ("deep_gemm", "moe_tune", "flashinfer", "inductor")
                    if k in stages
                ]
            )
            print(f"RESULT_STATUS partial {summary}")
            return
        run_obj["run"]["settings"]["warmup_port"] = port

        # FLASHINFER stage
        if flashinfer != "skip":
            # permissions preflight for flashinfer workspace and jit log dirs
            pf = _permissions_preflight(
                "flashinfer",
                12,
                [
                    Path(env["FLASHINFER_WORKSPACE_DIR"]),
                    Path(env["FLASHINFER_JIT_LOG_DIR"]),
                ],
                env,
            )
            if pf:
                stages["flashinfer"] = pf
            else:
                flashinfer_dir = Path(env["FLASHINFER_WORKSPACE_DIR"])
                flashinfer_sig_file = flashinfer_dir / "signature.json"
                flashinfer_signature = _flashinfer_signature(
                    info,
                    server_args_snapshot,
                    run_obj["run"].get("model_slug"),
                    tp,
                )

                warmup_needed = True
                stats_before = _dir_stats(flashinfer_dir)
                if (
                    flashinfer == "ensure"
                    and flashinfer_sig_file.exists()
                    and stats_before.get("file_count", 0) > 0
                ):
                    try:
                        prior_sig = json.loads(flashinfer_sig_file.read_text())
                    except Exception:
                        prior_sig = None
                    if prior_sig == flashinfer_signature:
                        artifacts = {
                            "workspace_dir": {
                                "container_path": str(flashinfer_dir),
                                "host_path": None,
                            },
                            "files": {
                                "count": stats_before.get("file_count", 0),
                                "bytes": stats_before.get("size_bytes", 0),
                                "latest_mtime_iso": stats_before.get(
                                    "latest_mtime_iso"
                                ),
                            },
                        }
                        stages["flashinfer"] = StageResult(
                            False, "noop", 0, 0.0, artifacts
                        )
                        warmup_needed = False

                if warmup_needed:
                    # Start server with compile disabled (focus on flashinfer workspace)
                    t = time.time()
                    mark = _mark_in_progress("flashinfer", started_at)
                    memf = run_obj["run"]["settings"]["mem_fraction"]
                    proc = _start_warmup_server(
                        model,
                        tp,
                        port,
                        server_args_snapshot,
                        log_file,
                        enable_compile=False,
                        env=env,
                        trust_remote_code=server_defaults.get(
                            "trust_remote_code", True
                        ),
                    )
                    ok_ready = _wait_http_ready(port, timeout=flashinfer_timeout)
                    # Do not adjust mem_fraction on the fly; single attempt per policy
                    ok_req = ok_ready and _send_warmup_requests(port)
                    cleanup = _stop_server(proc)
                    dur = time.time() - t
                    if ok_ready and ok_req:
                        stats_after = _dir_stats(flashinfer_dir)
                        artifacts = {
                            "workspace_dir": {
                                "container_path": str(flashinfer_dir),
                                "host_path": None,
                            },
                            "files": {
                                "count": stats_after.get("file_count", 0),
                                "bytes": stats_after.get("size_bytes", 0),
                                "latest_mtime_iso": stats_after.get("latest_mtime_iso"),
                            },
                            "cleanup": cleanup,
                        }
                        stages["flashinfer"] = StageResult(
                            True, "ok", 0, dur, artifacts
                        )
                        try:
                            _atomic_write_json(
                                flashinfer_sig_file, flashinfer_signature
                            )
                        except Exception:
                            pass
                        _clear_in_progress(mark)
                    else:
                        _clear_in_progress(mark)
                        stages["flashinfer"] = StageResult(
                            True,
                            "error",
                            12,
                            dur,
                            {},
                            "oom_during_warmup" if not ok_ready else "request_failed",
                            errors=["warm-up failed"],
                        )  # noqa: E501

        else:
            stages["flashinfer"] = StageResult(False, "skipped", 12, 0.0, {})

        # INDUCTOR stage
        if inductor != "skip":
            # permissions preflight for inductor cache dir
            pf = _permissions_preflight(
                "inductor", 13, [Path(env["TORCHINDUCTOR_CACHE_DIR"])], env
            )
            if pf:
                stages["inductor"] = pf
            else:
                # Start server with torch.compile enabled to prime Inductor cache
                t = time.time()
                mark = _mark_in_progress("inductor", started_at)
                memf = run_obj["run"]["settings"]["mem_fraction"]
                proc = _start_warmup_server(
                    model,
                    tp,
                    port,
                    server_args_snapshot,
                    log_file,
                    enable_compile=True,
                    env=env,
                    trust_remote_code=server_defaults.get("trust_remote_code", True),
                )
                ok_ready = _wait_http_ready(port, timeout=inductor_timeout)
                # Do not adjust mem_fraction on the fly; single attempt per policy
                ok_req = ok_ready and _send_warmup_requests(port)
                cleanup = _stop_server(proc)
                dur = time.time() - t
                if ok_ready and ok_req:
                    artifacts = {
                        "cache_dir": {
                            "container_path": "/profiles/torchinductor",
                            "host_path": None,
                        },
                        "cleanup": cleanup,
                    }
                    stages["inductor"] = StageResult(True, "ok", 0, dur, artifacts)
                    _clear_in_progress(mark)
                else:
                    _clear_in_progress(mark)
                    stages["inductor"] = StageResult(
                        True,
                        "error",
                        13,
                        dur,
                        {},
                        "oom_during_warmup" if not ok_ready else "request_failed",
                        errors=["warm-up failed"],
                    )  # noqa: E501
        else:
            stages["inductor"] = StageResult(False, "skipped", 13, 0.0, {})
    else:
        stages["flashinfer"] = StageResult(False, "noop", 0, 0.0, {}, warnings=["no model provided"])  # type: ignore[arg-type]
        stages["inductor"] = StageResult(False, "noop", 0, 0.0, {}, warnings=["no model provided"])  # type: ignore[arg-type]

    # Build stage JSON files then aggregate prep_result.json
    for name, r in stages.items():
        run_obj["stages"][name] = {
            "ran": r.ran,
            "status": r.status,
            "status_code": r.code,
            "duration_s": round(r.dur, 3) if r.dur else 0,
            "artifacts": r.artifacts,
            "error_type": r.error_type,
            "warnings": r.warnings or [],
            "errors": r.errors or [],
        }
        try:
            stage_dir = prep_path.parent / "stages"
            stage_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(stage_dir / f"{name}.json", run_obj["stages"][name])
        except Exception:
            pass

    # Post-warmup: attempt to verify MoE config consumption in the session log
    try:
        _verify_moe_consumption_in_log(log_file, stages)
    except Exception:
        pass

    # Optional: Try a single prompt to ensure Prometheus sees some activity (hello-world)
    if prom_ping:
        try:
            port = (
                int(run_obj["run"]["settings"]["warmup_port"])
                if run_obj.get("run")
                else 30000
            )
        except Exception:
            port = 30000
        try:
            cmd = (
                f"bash /workspaces/sglang/scripts/infer/hello_world.sh 127.0.0.1 {port}"
            )
            subprocess.run(["bash", "-lc", cmd], check=False)
        except Exception:
            pass

    # Aggregate from stage files present (fail-fast on critical errors)
    try:
        stage_dir = prep_path.parent / "stages"
        stages_all = {}
        for p in stage_dir.glob("*.json"):
            try:
                stages_all[p.stem] = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                # Log but continue - corrupt stage file shouldn't break aggregation
                logger.warning(f"Failed to parse stage JSON {p}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error reading stage file {p}: {e}")
                raise
        run_obj["stages"] = stages_all or run_obj["stages"]
    except FileNotFoundError:
        # Stage dir doesn't exist yet - this is expected if we're writing stages inline
        pass
    except Exception as e:
        # Critical error in aggregation - surface it
        run_obj["warnings"].append(f"Stage aggregation failed: {e}")
        logger.error(f"Failed to aggregate stage files: {e}")
    status = "ok"
    failures = [n for n, r in run_obj["stages"].items() if r.get("status") == "error"]
    if failures:
        status = "partial"
        run_obj["errors"] = failures
    run_obj["status"] = status
    run_obj["run"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_obj["run"]["duration_s"] = round(time.time() - t0, 3)

    # Telemetry probe (after warm-ups and optional ping)
    try:
        run_obj["telemetry_probe"] = _prom_probe(run_id)
    except Exception:
        pass
    _atomic_write_json(prep_path, run_obj)
    # Emit provider_prep_finished event (best-effort)
    try:
        status_event = run_obj.get("status") or "unknown"
        model_slug = run_obj.get("run", {}).get("model_slug") or "unknown"
        subprocess.run(
            [
                "bash",
                "-lc",
                f'bash /workspaces/sglang/.devcontainer/observability/eventlog.sh event provider_prep_finished run_id=\\"{run_id}\\" status=\\"{status_event}\\" model_slug=\\"{model_slug}\\"',
            ],
            check=False,
        )
    except Exception:
        pass
    # Convenience: record prep_result path into manifest (container + host rebased when available)
    try:
        m = json.loads(manifest_path.read_text())
        m.setdefault("paths", {}).setdefault("container", {})["prep_result"] = str(
            prep_path
        )
        if manifest_host_path:
            # mirror run-scoped layout under host path
            host_prep = manifest_host_path.parent / run_id / "prep_result.json"
            m.setdefault("paths", {}).setdefault("host", {})["prep_result"] = str(
                host_prep
            )
        _atomic_write_json(manifest_path, m)
    except Exception as e:
        # Non-critical: prep_result was written, just couldn't update manifest
        logger.warning(f"Failed to update manifest with prep_result path: {e}")
    # human-friendly result lines
    print(f"RESULT_JSON {prep_path}")
    summary = " ".join(
        [
            f"{k}:{stages[k].status}"
            for k in ("deep_gemm", "moe_tune", "flashinfer", "inductor")
        ]
    )
    print(f"RESULT_STATUS {status} {summary}")


if __name__ == "__main__":
    app()
