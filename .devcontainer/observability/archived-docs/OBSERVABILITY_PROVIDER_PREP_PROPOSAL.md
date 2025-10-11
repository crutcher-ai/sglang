# SGLang GH200 Devcontainer Observability + Provider‑Prep Proposal (Draft)

Status: Draft for review with SLICE‑Bench
Date: 2025‑10‑09

This document scopes the near‑term changes to the GH200 devcontainer‑based
observability and kernel‑cache preparation (“provider prep”) used during
benchmarking. It is intentionally focused on the devcontainer workflow and host
wrappers, not on compose or upstream changes.

## Goals

- Enable predictable, host‑readable persistence for telemetry and kernel caches
  outside the repo tree.
- Expose explicit, controllable “provider prep” entrypoints to warm caches for:
  DeepGEMM, Triton/MoE fusion, FlashInfer workspace, and TorchInductor.
- Emit visible, structured progress in the per‑run session log for long
  warmups, and write a cache snapshot at container startup to support
  orchestration decisions.
- Keep server launch ergonomics stable for benchmarks running on the host
  (localhost ports via host networking).

## Non‑Goals (for this iteration)

- No end‑of‑run cache snapshot (startup snapshot only).
- No source changes inside `python/sglang/…` (wrappers translate flags as
  needed). No compose/Helm changes.

## Proposed Changes (Summary)

1) Host storage defaults (overrideable) and container mounts

- Telemetry root on host (logs, Prom TSDB, Jaeger, manifests):
  - `HOST_TELEMETRY_ROOT=${XDG_DATA_HOME:-$HOME/.local/share}/sglang-dev/telemetry`
  - Mounted into container at `/telemetry`
- Kernel caches root on host (persist across container runs):
  - `HOST_PROFILES_ROOT=${XDG_CACHE_HOME:-$HOME/.cache}/sglang/profiles`
  - Mounted into container at `/profiles`
- Start script (`scripts/start_observable_container.sh`) will:
  - Create the above directories if missing (host side).
  - Mount subpaths as bind volumes:
    - `/telemetry/logs`, `/telemetry/prometheus`, `/telemetry/jaeger`,
      `/telemetry/container_runs`, and `/telemetry/container_run_meta.env`.
    - `/profiles` (single root for all caches).
  - Pass host identity for ownership:
    - `-e RUN_FILE_UID=$(id -u) -e RUN_FILE_GID=$(id -g)`
  - Preserve existing flags (`--network host`, `--gpus all`, `--cap-add SYS_ADMIN`).

2) Environment variables inside the container (unchanged usage)

- Set explicitly (via `devcontainer.json` or `docker run -e`) so caches always
  land under `/profiles`:
  - `TRITON_CACHE_DIR=/profiles/triton`
  - `TORCHINDUCTOR_CACHE_DIR=/profiles/torchinductor`
  - `FLASHINFER_WORKSPACE_DIR=/profiles/flashinfer`
  - `SGLANG_DG_CACHE_DIR=/profiles/deep_gemm`
  - `SGLANG_MOE_CONFIG_DIR=/profiles/moe_configs`  (MoE tuner outputs live here)

2.1) Operational defaults and permissions

- Entry/launcher sets `umask 002` so group read/write is preserved for produced
  artifacts. After provider‑prep succeeds, wrappers run
  `chmod -R g+rwX /profiles` to avoid permission surprises across host users.
- All timestamps written to JSON are UTC ISO‑8601 (`YYYY‑MM‑DDTHH:MM:SSZ`).

3) Startup cache snapshot (host‑side) + validity/signature

- After the container emits the manifest pointer,
  `scripts/start_observable_container.sh` computes a **startup cache snapshot**
  and merges it into the manifest under `profile_cache.start`.
- For each cache: fields now include
  - `exists` (bool), `path` (host‑relative when possible), `size_bytes`,
    `file_count`, `latest_mtime_iso`.
  - `valid` (bool) and a `signature` block with environment hints for reuse
    decisions: `{ device_name, cuda, torch_version, triton_version,
    sglang_commit, model_slug }`. If invalid, include `reason` string.
- Add `schema_version: "1"` inside `profile_cache.start` and set `partial: true`
  when any in‑progress markers are present (see §5.3). Always include
  `signature.model_slug` if known.
- Purpose: let orchestration decide whether to run provider‑prep before the
  benchmark. No end‑of‑run snapshot in this iteration.

4) Provider‑prep entrypoints (host scripts that run inside the container)

- `scripts/cache/inspect_caches.sh`
  - Reports per‑cache stats: path, exists, size, file_count, newest file age,
    and a short “newest files” list.
  - Flags: `--triton --inductor --flashinfer --deep-gemm` (default: all),
    `--json` for machine‑readable output.

- `scripts/cache/populate_caches.sh`
  - Drives long‑running warmups with structured progress and appends to the
    per‑run session log (manifest provides the log file path).
  - Select caches with `--triton --inductor --flashinfer --deep-gemm` (or
    `--all`). Common flags include `--model`, `--tp`, `--block-size`,
    `--max-seq-len`, `--timeout`, `--heartbeat-sec 60`.
  - Implementation notes (inside container via `docker exec`):
    - DeepGEMM: call `python -m sglang.compile_deep_gemm` with provided model,
      TP, and block size(s). Emit JSON heartbeats while waiting for NVRTC.
    - Triton/MoE tuning (treated as a real tuning step, not just JIT warmup):
      run the official Triton MoE tuner
      (`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`) to
      produce a config JSON under
      `${SGLANG_MOE_CONFIG_DIR}/configs/triton_<ver>/E=...N=...dtype=...block=[...].json`.
      Then launch a short‑lived server and verify consumption by grepping the
      server log for a “Using MoE kernel config” line.
    - FlashInfer workspace: prefer a real server path on a warm‑up port
      (e.g., `--mem-fraction 0.96`) and issue 2–3 scripted requests
      (prefill 8k, decode 128–256) to allocate workspaces and populate kernels;
      stop afterward.
    - TorchInductor: same server warm‑up path; ensure a representative decode
      graph compiles via `torch.compile` during the scripted requests.
  - Outputs: in addition to log heartbeats, write a single JSON summary file
    (see §6) at `/telemetry/container_runs/<run_id>/prep_result.json` that
    includes both container and host‑rebased artifact paths and a top‑level
    `run` context block.

4.3) Effective environment snapshot

- Record, in machine‑readable form, the effective environment that influences
  artifacts. In `prep_result.json.run.settings.env` include at least:
  `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_DIR`,
  `SGLANG_DG_CACHE_DIR`, `SGLANG_MOE_CONFIG_DIR`, and the chosen
  `warmup_port` and `mem_fraction`. If a stage overrides any of these, repeat a
  per‑stage `env` map inside that stage.

4.4) Warm‑up port selection and collision handling

- Default warm‑up server port: `30999`. If occupied, try a short fallback list
  in descending order (e.g., `30998, 30997, 30996, 30995`). Record the chosen
  port in `prep_result.json.run.settings.warmup_port`. If all ports fail,
  return a clear error with `error_type:"port_all_busy"`.

4.1) Optional JSON spec input

- `populate_caches.sh` keeps flag‑based CLI but may also accept `--spec-file`
  pointing to a JSON spec like:
  ```json
  {
    "model_path": "Qwen/Qwen2.5-7B",
    "tp": 8,
    "ops": {
      "deep_gemm": "ensure",
      "flashinfer": "ensure",
      "moe": {"mode":"tune","policy":{"strategy":"triton","dtype":"fp8","blocks":[128,256]}},
      "inductor": "ensure"
    }
  }
  ```
  Flags override spec fields when both are present.

4.2) Warm‑up mem‑fraction policy (heuristic)

- Default server warm‑up launches with `--mem-fraction 0.96`.
- If `nvidia-smi` indicates < 4 GiB free a few seconds after launch, retry once
  at `--mem-fraction 0.93`; if still OOM, abort with a clear message in JSON and
  session log. This policy is container‑internal and opaque to SLICE‑Bench.

4.2.1) Warm‑up lifecycle guarantees and cleanup

- Record `warmup_pid` for any auxiliary server processes. On stage end (success
  or error), attempt cleanup: send SIGTERM, wait up to 10s, then SIGKILL if
  needed. In `prep_result.json` add:
  ```json
  "cleanup": { "kill_sent": true, "killed": true, "timeout_s": 10 }
  ```
  If cleanup cannot confirm termination, set `error_type:"cleanup_failed"` on
  that stage.

5) Logging conventions (visible + parseable)

- All provider‑prep output goes into the devcontainer **session log** (same
  per‑run log file that `init-run.sh` tees to). Host wrappers append to it by
  redirecting `docker exec` stdout/stderr.
- Human lines for eyes:
  - `[OPTIMIZE deep_gemm] starting {model=..., tp=..., block=...}`
- JSON lines for tools (single‑line objects prefixed for easy grepping):
  - `OPTIMIZE_JSON {"name":"deep_gemm","phase":"start","ts":"...","settings":{...}}`
  - progress: `OPTIMIZE_JSON {"name":"deep_gemm","phase":"progress","elapsed_s":300,"hint":"nvrtc ..."}`
  - end: `OPTIMIZE_JSON {"name":"deep_gemm","phase":"end","status":"ok","duration_s":1243}`
- Optionally, at the end of each subtask, the wrapper may write a small status
  block into the manifest under `optimizations.<name>`.

5.1) In‑progress markers and partial caches

- During each prep stage create a marker file under
  `/profiles/.in_progress/<slug>/<stage>`; remove it only on success. Startup
  snapshot treats presence as `valid: false` and sets `partial: true` with
  `reason: "in_progress_or_aborted"`. Marker content includes `owner_pid` and
  `started_at` for diagnostics; these are mirrored under
  `profile_cache.start.<cache>.partial_info`.

5.2) Standardized stage codes

- Per stage, use `status` ∈ {`ok`,`error`,`skipped`,`noop`} and `status_code`:
  - `0` ok, `10` deep_gemm, `11` moe_tune, `12` flashinfer, `13` inductor.
  Exit codes match the same numbers.

5.3) Error classification

- Per stage, include `error_type` drawn from a small enum, e.g.,
  `nvrtc_compile_failed`, `oom_during_warmup`, `tuner_execution_error`,
  `verify_log_missing`, `lock_timeout`, `port_all_busy`, `permissions_unexpected`.
  Keep `status_code` as defined above.

5.4) No‑op clarity

- When a stage is skipped due to a valid cache, set `status:"noop"` and add
  `reason:"valid_signature"` along with an echo of the `signature` block used
  for the decision.

5.5) CLI result pointers (human convenience)

- After writing `prep_result.json`, print two summary lines to stdout:
  - `RESULT_JSON /telemetry/container_runs/<run_id>/prep_result.json`
  - `RESULT_STATUS ok deep_gemm:noop moe_tune:ok flashinfer:ok inductor:ok`

6) Manifest schema additions (minimal)

- Existing manifest fields remain unchanged. Add:
  - `profile_cache.start`: per‑cache snapshot taken at container startup.
  - Optional: `optimizations` object with per‑provider summaries when a warmup
    completes (status, duration_s, size_delta_bytes if measured).
  - Optional (future‑proof): explicit service URLs so consumers need not assume
    localhost, e.g., `services.prometheus.url` and `services.jaeger.url`.
- Example excerpt:

```json
{
  "profile_cache": {
    "start": {
      "triton": {
        "exists": true,
        "path": ".cache/sglang/profiles/triton",
        "size_bytes": 734003200,
        "file_count": 182,
        "latest_mtime_iso": "2025-10-09T02:44:31Z"
      },
      "torchinductor": { "exists": false },
      "flashinfer": { "exists": true, "size_bytes": 1048576 },
      "deep_gemm": { "exists": true, "file_count": 96 }
    }
  }
}
```

6.1) Telemetry surface status (clarification)

- The manifest already contains a `telemetry_surfaces` block (Prom exporter
  statuses are live; `sglang_metrics`/`tracing` are `expected` until the server
  is launched by the benchmark). SLICE‑Bench can rely on this to gate runs.
  - When capability checks degrade (e.g., SYS_ADMIN for DCGM), set the affected
    surface `status:"degraded"` and include a concise warning string, in
    addition to listing the warning in the top‑level `warnings` array.

## Slice‑Bench Integration Points

- Discovery:
  - Read `.env` pointer at `HOST_TELEMETRY_ROOT/container_run_meta.env`.
  - Prefer `CONTAINER_RUN_META_JSON_HOST` if present; fallback to
    `CONTAINER_RUN_META_JSON` (container path).
- Preflight:
  - Inspect `profile_cache.start` to decide which provider‑prep steps to run.
  - Optionally call `scripts/cache/inspect_caches.sh --json` for fresh data
    before prep.
- Launch:
  - When starting SGLang, wrappers may accept `--otlp-endpoint` and translate
    to the server’s current flag name. Tracing enablement stays optional.
- Progress:
  - Prefer reading `/telemetry/container_runs/<run_id>/prep_result.json` for a
    single machine‑readable outcome across all prep stages. Logs remain a
    secondary, human‑friendly channel.

## Networking & Ports (unchanged)

- Container runs with `--network host` so all endpoints are reachable via
  localhost from the host‑side benchmark:
  - SGLang: 30000, Router: 29000
  - Prometheus: 9090, Jaeger UI: 16686, OTLP gRPC: 4317, OTLP HTTP: 4318
  - node_exporter: 9100, dcgm‑exporter: 9400
- Optional future: support `PORT_BASE` to shift all listening ports if running
  multiple stacks on one host (not required for this iteration).

## Permissions & Ownership

- Start script passes `RUN_FILE_UID`/`RUN_FILE_GID` to the container. The
  entrypoint already uses these to `chown` run artefacts (log, TSDB, traces,
  manifest). Cache directories are host‑owned and writable, so their files are
  readable by the host user.
- DCGM exporter requires `--cap-add SYS_ADMIN` and GPUs; `init-run.sh` already
  attempts `setcap` and records warnings in the manifest if capability setup is
  degraded.

## Acceptance Criteria

- Starting the helper container produces a manifest with `profile_cache.start`
  covering the four caches, each with `schema_version: "1"`, validity,
  signature (including model_slug), and `partial` when applicable.
- Caches persist across container restarts under
  `${XDG_CACHE_HOME:-$HOME/.cache}/sglang/profiles`.
- Provider‑prep commands stream progress to the session log with human and
  `OPTIMIZE_JSON` lines and emit `/telemetry/.../prep_result.json`; failures are
  obvious in the JSON status and via non‑zero exit codes.

## JSON Structures (draft)

### prep_result.json (written at /telemetry/container_runs/<run_id>/)

```json
{
  "schema_version": "1",
  "status": "ok|error|partial",
  "run": {
    "run_id": "container-run-...",
    "model_slug": "qwen2.5-7b",
    "tp": 8,
    "device_name": "NVIDIA GH200",
    "cuda": "12.5",
    "torch_version": "2.4.1",
    "triton_version": "3.0.0",
    "sglang_commit": "abc1234",
    "started_at": "2025-10-09T01:12:00Z",
    "finished_at": "2025-10-09T01:32:43Z",
    "duration_s": 1243,
    "settings": {
      "source": "flags|spec",
      "spec_version": "1",
      "tp": 8,
      "blocks": [128,256],
      "warmup_port": 30999,
      "mem_fraction": 0.96,
      "write_mode": "atomic",
      "env": {
        "TRITON_CACHE_DIR": "/profiles/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/profiles/torchinductor",
        "FLASHINFER_WORKSPACE_DIR": "/profiles/flashinfer",
        "SGLANG_DG_CACHE_DIR": "/profiles/deep_gemm",
        "SGLANG_MOE_CONFIG_DIR": "/profiles/moe_configs"
      },
      "ignored_spec_keys": ["typo_key"]
    }
  },
  "stages": {
    "deep_gemm": {
      "ran": true,
      "status": "ok|error|skipped|noop",
      "status_code": 0,
      "duration_s": 600,
      "artifacts": {
        "cache_dir": {"container_path": "/profiles/deep_gemm", "host_path": "~/.cache/sglang/profiles/deep_gemm"},
        "cubins": {"count": 96, "bytes": 123456789},
        "log_file": {"container_path": "/telemetry/logs/....log", "host_path": "..."}
      },
      "error_type": null,
      "warnings": [],
      "errors": []
    },
    "moe_tune": {
      "ran": true,
      "status": "ok",
      "status_code": 0,
      "duration_s": 85,
      "artifacts": {
        "config_file": {"container_path": "/profiles/moe_configs/configs/triton_3.0.0/E=...json", "host_path": "..."},
        "config_hash": "sha256:...",
        "triton_version": "3.0.0",
        "verified_in_log": true,
        "verify_log": {"container_path": "/telemetry/logs/....log", "host_path": "..."},
        "active_config_link": {"container_path": "/profiles/moe_configs/configs/active/qwen2.5-7b.json", "host_path": "..."}
      }
    },
    "flashinfer": {
      "ran": true,
      "status": "ok",
      "status_code": 0,
      "duration_s": 45,
      "artifacts": {
        "workspace_dir": {"container_path": "/profiles/flashinfer", "host_path": "..."}
      }
    },
    "inductor": {
      "ran": true,
      "status": "ok",
      "status_code": 0,
      "duration_s": 120,
      "artifacts": {
        "cache_dir": {"container_path": "/profiles/torchinductor", "host_path": "..."},
        "files": {"count": 182, "bytes": 734003200}
      }
    }
  },
  "telemetry_probe": {
    "prometheus_query": "increase(sglang:prompt_tokens_total[30s])",
    "with_run_filter": "increase(sglang:prompt_tokens_total{container_run=\"<run_id>\"}[30s])",
    "sample_count": 12,
    "ok": true
  },
  "errors": [],
  "warnings": []
}
```

### Startup cache snapshot shape (profile_cache.start)

```json
{
  "schema_version": "1",
  "triton": {
    "exists": true,
    "partial": false,
    "partial_info": {"owner_pid": 0, "started_at": "2025-10-09T02:40:00Z"},
    "valid": true,
    "path": ".cache/sglang/profiles/triton",
    "size_bytes": 734003200,
    "file_count": 182,
    "latest_mtime_iso": "2025-10-09T02:44:31Z",
    "signature": {
      "model_slug": "qwen2.5-7b",
      "device_name": "NVIDIA GH200",
      "compute_capability": "sm_90",
      "cuda": "12.5",
      "driver_version": "555.85",
      "torch_version": "2.4.1",
      "triton_version": "3.0.0",
      "flashinfer_version": "1.1.0",
      "sglang_commit": "abc1234",
      "tp_size": 8
    }
  }
}
```

### Spec‑file for populate_caches.sh (optional input)

```json
{
  "schema_version": "1",
  "model_path": "Qwen/Qwen2.5-7B",
  "tp": 8,
  "ops": {
    "deep_gemm": "ensure|rebuild|skip",
    "flashinfer": "ensure|rebuild|skip",
    "moe": {
      "mode": "ensure|rebuild|skip",
      "policy": { "strategy": "auto|fixed|sweep", "dtype": "fp8_w8a8", "batch_sizes": [128,256] }
    },
    "inductor": "ensure|rebuild|skip"
  },
  "timeout_s": 3600
}
```

Notes:
- Unknown keys are ignored with a warning in the session log and JSON output.
- Flags always override spec values; the resolved settings are echoed in
  `prep_result.json.run.settings` with `settings.source` indicating the winner.

## Locking and backoff

- A slug‑scoped lock under `/profiles/.locks/<slug>.lock` prevents concurrent
  prep. If a lock exists, emit a JSON heartbeat every 30s:
  `{"phase":"wait_lock","slug":"<slug>","held_for_s":123}`. Abort with
  `error_type:"lock_timeout"` when exceeding the stage timeout.

## Disk usage delta

- For each stage, report `disk_delta_bytes` and `files_delta` where feasible by
  sampling directory usage before and after the stage. Include in the stage
  artifacts or as top‑level `stage_deltas` if cleaner.

## Reset helper (optional)

- Provide `scripts/cache/reset_caches.sh` with filters:
  - `--older-than DAYS`, `--match-slug PREFIX`, and per‑cache selectors.
  - Emit a JSON report of removed entries and log victims to the session log.


## MoE Tuning Details

- Filename pattern (example):
  `configs/triton_<triton_version>/E=<experts>__N=<hidden>__dtype=<dtype>__block=<bsize>__dev=<device_name>__tp=<tp>.json`
- Selection criteria include: E, N, dtype, block shape, device name, Triton
  version, and TP size if relevant.
- Verification: when launching the warm‑up server, ensure the exact log line
  appears: `Using MoE kernel config: <basename>`. Record `<basename>` and
  `config_hash` (SHA‑256) in `prep_result.json`.
  - Determinism: record tuner inputs (grid, dtype, seed if any) and a
    `tuner_version`/commit under `stages.moe_tune.artifacts`.

## Tiny CLI (`sgl-admin`) Plan

- `sgl-admin caches inspect --json` → prints a live snapshot with
  `schema_version: "1"`, including `valid`, `signature`, and `partial`.
- `sgl-admin caches ensure --json [--spec-file file.json | flags...]` → streams
  heartbeats and writes `prep_result.json`, then prints one line:
  `RESULT_JSON /telemetry/container_runs/<run_id>/prep_result.json`.
  - Also print `RESULT_STATUS ...` summarizing per‑stage outcomes for quick
    human review.

## Quick Acceptance Tests

- Cold start: manifest `profile_cache.start` shows `exists=false` or
  `valid=false`. Run `sgl-admin caches ensure --json --all --model ...` → exit 0,
  `prep_result.json.status=ok`, and live inspect reports `valid=true`.
- Abort mid‑prep: kill DeepGEMM step → `.in_progress` marker persists;
  inspect reports `valid=false`, `partial=true`, `reason="in_progress_or_aborted"`.
- MoE verify: after tuning, warm‑up server log contains
  `Using MoE kernel config: <basename>` and `prep_result.json.stages.moe_tune.verified_in_log=true`.

## Atomic JSON writes

- All JSON artifacts (`prep_result.json`, any `cache_snapshot.json`) are written
  atomically: write to a temporary file in the target directory, `fsync()` the
  file, `rename()` to the final path, then `fsync()` the directory. The
  `prep_result.json.run.settings.write_mode` is set to `"atomic"` to signal this
  contract to consumers.

## Disk headroom check

- Before heavy stages (DeepGEMM, TorchInductor), check free disk space under the
  relevant cache directory. If insufficient, fail early with
  `error_type:"insufficient_disk"` and include `required_bytes` and
  `free_bytes` in the stage errors payload.

## Exit status precedence

- If multiple stages fail, the process returns the status code of the first
  failing stage in execution order. All failures are listed in
  `prep_result.json.errors` with their stage names and error summaries.

- The external benchmark can continue to hit `localhost:*` endpoints with no
  additional port flags.

## Open Questions

- TorchInductor warmup target(s): which graph(s) do we compile by default for
  your benchmark suite? (A small set of representative shapes vs.
  model‑specific profiles.)
- Triton/MoE warmup coverage: which block sizes are priority (e.g., 128/256)?
  Should we scope to MoE kernels only or also include common attention ops?
- Do we want a simple `scripts/cache/reset_caches.sh` with safe pruning options
  (e.g., `--older-than DAYS`), or keep it manual for now?

- Prep concurrency control: we plan a simple lockfile under
  `/profiles/.locks/<slug>.lock` to avoid concurrent prep clobbering; OK?

- Return codes: propose standardized stage‑specific non‑zero codes
  (10=deep_gemm, 11=moe_tune, 12=flashinfer, 13=inductor) for CI diagnostics.

## Migration Plan (minimal steps)

1) Update `scripts/start_observable_container.sh` to:
   - Honor `HOST_TELEMETRY_ROOT` and `HOST_PROFILES_ROOT` with XDG defaults.
   - Mount the new host paths into `/telemetry` and `/profiles`.
   - Pass `RUN_FILE_UID/GID` envs.
   - Compute and merge `profile_cache.start` into the manifest.
   - Include validity/signature fields; derive a stable `model_slug` when a
     model is specified to provider‑prep.

2) Add host utilities:
   - `scripts/cache/inspect_caches.sh` (supports `--json`).
   - `scripts/cache/populate_caches.sh` (per‑provider, with progress heartbeats,
     `prep_result.json` and optional manifest updates). Accept optional
     `--spec-file` JSON.

3) (Optional) Tiny container CLI emitting JSON

- Provide `sgl-admin` (Typer‑based) inside the container with subcommands:
  - `sgl-admin caches inspect --json`
  - `sgl-admin caches ensure --json [flags|--spec-file <path>]`
- This thin wrapper consolidates outputs and avoids log parsing for consumers.

3) (Optional) Document the JSON log convention and manifest fields in
   `docs/advanced_features/observability.md` once finalized with SLICE‑Bench.

---

Appendix A — Example Session Log Lines

```
[OPTIMIZE deep_gemm] starting {model=DeepSeek-V3, tp=8, block=128}
OPTIMIZE_JSON {"name":"deep_gemm","phase":"start","ts":"2025-10-09T01:12:00Z","settings":{"model":"DeepSeek-V3","tp":8,"block":128}}
OPTIMIZE_JSON {"name":"deep_gemm","phase":"progress","elapsed_s":300,"hint":"nvrtc compiling GEMM variants (3/9)"}
OPTIMIZE_JSON {"name":"deep_gemm","phase":"end","status":"ok","duration_s":1243}

[OPTIMIZE triton] warming blocks=[128,256] shapes=(...)
OPTIMIZE_JSON {"name":"triton","phase":"progress","block":128,"elapsed_s":42}
OPTIMIZE_JSON {"name":"triton","phase":"end","status":"ok","duration_s":85,"blocks":[128,256]}
```

Appendix B — Example Host Paths

- Telemetry root: `~/.local/share/sglang-dev/telemetry/`
  - `logs/`, `prometheus/<container-run-*>/`, `jaeger/<container-run-*>/`,
    `container_runs/`, `container_run_meta.env`
- Caches root: `~/.cache/sglang/profiles/`
  - `triton/`, `torchinductor/`, `flashinfer/`, `deep_gemm/`,
    `moe_configs/configs/`

---

## Change Log

- Incorporated SLICE‑Bench feedback:
  - Added `SGLANG_MOE_CONFIG_DIR` and turned MoE into a real tuning step that
    persists config JSON and verifies server consumption.
  - Added cache `valid`/`signature` fields to the startup snapshot.
  - Introduced a single prep result JSON
    (`/telemetry/container_runs/<run_id>/prep_result.json`) so consumers don’t
    need to tail logs.
  - Documented an optional JSON spec for provider‑prep and a thin `sgl-admin`
    CLI that emits JSON.
  - Clarified telemetry surface status and added optional explicit service URLs
    in the manifest for future decoupling from localhost.
  - Proposed lock/lease and standardized return codes for robust CI behavior.
  - Added schema_version fields, run context block, explicit container/host
    artifact paths, in‑progress markers → `partial`, a mem‑fraction retry
    heuristic, exact MoE verification strings, and acceptance tests.
