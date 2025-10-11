Provider‑Prep Engineer’s Notebook (Excruciating Detail)

Status: Live primitives; DeepGEMM compile path refactor in progress
Date: 2025‑10‑10
Audience: The next agent/engineer responsible for provider‑prep + observability + cache priming in the SGLang devcontainer.

Read This First (Non‑Negotiables)
- Never write to bind mounts (/telemetry, /profiles) as root. All writes must be authored by the unprivileged devuser.
- The pointer file (/telemetry/container_run_meta.env) is created atomically by the container only after the manifest exists. The host must never pre‑create a blank pointer.
- There is exactly one per‑run session log; init and sgl-admin append to the same file for a given run id.
- For any server that allocates memory or affects kernel caches, pass explicit CLI flags. Do not rely on Python object handoffs across processes.
- Cache priming defaults are hard policy for all stages: mem_fraction_static=0.94, kv_cache_dtype=fp8_e4m3, chunked_prefill_size=4096, max_mamba_cache_size=1, context_length=272000, max_prefill_tokens=272000, disable_cuda_graph=false. If logs show otherwise, the run is invalid.
- Stage writes are append-only: write stage JSON first, then aggregate. Never lose prior success because the aggregator failed.

Table of Contents
1) Core Principles (Do/Don’t) and Invariants
2) Environment & Mounts: Host ⇄ Container Contracts
3) Pointer, Manifest, and Logs: Single‑Source‑of‑Truth Semantics
4) Permissions: Ownership Map and Allowed Writers
Ownership Map (who can write where)
- Host user (e.g., ubuntu:ubuntu): owns $HOME/sglang‑observability and all subdirectories. The start script verifies ownership of telemetry/, profiles/, models/, huggingface/ and fails fast if mismatched (unless explicitly disabled).
- devuser inside container: mapped to host UID:GID; authors all bind‑mount files created by the container (manifest, pointer, session log, TSDB/traces, cache artifacts, stage JSONs).
- root inside container: allowed to start privileged services only; root never writes into bind mounts.

Allowed writers and examples
- /telemetry/logs: devuser only (init‑run tee, sgl-admin subprocess logs).
- /telemetry/container_runs: devuser only (manifest, stage JSONs, prep_result.json).
- /telemetry/container_run_meta.env: devuser only; atomic write via .tmp → rename.
- /profiles/*: devuser only (triton, torchinductor, flashinfer, deep_gemm, moe_configs, .locks, .in_progress).
- /tmp (container): root can write (nv‑hostengine pid/log), but this path is not a bind mount and will never affect host‑visible files.

Anti‑patterns (must not reintroduce)
- chmod/umask footguns (e.g., chmod 600 everywhere, umask 002 masking intent). We rely on default umask and correct author identity.
- “Fix” ownership with runtime chown. If a path is not writable by devuser, that is an error we surface, not something we mutate.

5) Entrypoints and Flows (Host → Container → CLI → Subprocess)
6) sgl_admin.py Internals (Orchestration, Locks, Stage JSON, Aggregation)
7) Stage Details
   7.1 DeepGEMM compile (canonical, with CLI flags)
   7.2 FlashInfer warm‑up (metrics on, strict port 30000)
   7.3 TorchInductor warm‑up (metrics on, strict port 30000)
   7.4 MoE configs (triton tuner; per‑size runs; verification)
8) Defaults for All Caching Tasks (why these)
We use these values across all caching tasks to match known‑good production runs and avoid JIT/OOM flakiness:
- mem_fraction_static = 0.94
  Rationale: Balances model weights + KV cache pool against activations/graphs for large context (272k) without starving the runtime. Lower computed values (~0.86) have consistently failed.
- kv_cache_dtype = fp8_e4m3
  Rationale: Required for intended memory footprint and kernel selection; bfloat16 here will blow up KV memory and defeat the purpose of priming.
- chunked_prefill_size = 4096
  Rationale: Matches working prefill chunking used in numerous runs; aligns with priming coverage for deep kernels.
- max_mamba_cache_size = 1
  Rationale: Keep Mamba cache minimal during priming to reduce memory footprint.
- context_length = 272000; max_prefill_tokens = 272000
  Rationale: Match the target serving envelope so the right shapes are compiled.
- disable_cuda_graph = false
  Rationale: Keep CUDA graph path enabled unless explicitly testing otherwise; avoids leaving graph-only codepaths unprimed.

Compile‑only exception
- For DeepGEMM compile, we set --enable-metrics=false and --enable-trace=false to decouple from Prometheus and avoid reserving :30000 for a non‑scraped step. Warm‑ups retain metrics to enable telemetry validation.

9) Failure Modes and Fast Triage (symptom → cause → fix)
1. Root‑owned files under telemetry/profiles
   - Cause: older init wrote binds as root; wrapper/scripts wrote logs or pointer as root.
   - Fix: ensure all writes happen as devuser (init and sgl-admin). Delete any root‑owned artifacts and rerun start.

2. Two logs “starting at the same time”; sgl-admin appears stuck
   - Cause: stale pointer; host latched an old manifest path, so init wrote to a new log while sgl-admin appended to an old log.
   - Fix: start deletes stale pointer before container launch; wait for new pointer existence; ensure both init and sgl-admin append to the same per‑run log.

3. Warm-ups fail with port 30000 busy
   - Cause: stray server holding 30000 or earlier failure left a server alive.
   - Fix: stop container; verify no sglang.launch_server remains; restart and rerun. Do not add fallback ports.

4. DeepGEMM compile fails immediately with “Not enough memory… mem_fraction_static=0.863” and kv dtype bfloat16
   - Cause: compile server launched without explicit CLI flags; runtime recomputed mem_fraction and selected default KV dtype.
   - Fix: launch via CLI with exact flags; verify compile.log prints fp8_e4m3 dtype and no mem fraction error.

5. prep_result.json “lost” earlier success after a later failure
   - Cause: single aggregate file being rewritten.
   - Fix: write per‑stage JSONs first; aggregate later; on aggregation failure, stage files remain intact.

6. Permissions “fixed” by runtime chown/chmod
   - Cause: scripts attempted to patch ownership at runtime.
   - Fix: remove all such code; if a dir isn’t writable by devuser, error out with a precise message; operator fixes ownership on host.

7. Inspect shows caches under ~/.cache/… instead of /profiles/…
   - Cause: default XDG fallback.
   - Fix: remove XDG fallback; inspect reads canonical paths from manifest only; fail if manifest/pointer missing.

10) Tests: Unit (no GPU), Smoke (GPU), CI Hooks
Unit
- test/args/test_server_args_flow.py: Verifies that explicit CLI flags parse into ServerArgs as given (mem_fraction_static=0.94, kv_cache_dtype=fp8_e4m3, chunked_prefill_size=4096, max_mamba_cache_size=1, max_prefill_tokens/context_length=272k, disable_cuda_graph=false). Uses a minimal local config.json with model_type+architectures to avoid touching HF and sets --load-format dummy and --skip-tokenizer-init.
- test/args/test_compile_cli_args.py: Verifies that the compile CLI args builder includes the canonical flags; this guards against accidental changes to defaults.

Smoke (GPU‑gated; SGL_E2E_CONTAINER_TESTS=1)
- test/e2e/test_pointer_and_deepgemm_flow.py: Start container, assert pointer appears, run DeepGEMM ensure (first compile), stop/restart, ensure again (noop). Replace /models/<YOUR_MODEL> in the test or pass SGL_E2E_MODEL.

CI Hooks
- Add a job to run unit tests and a “validate‑only” start that checks pointer/manifest ownership. Gate GPU smoke behind an env flag.

11) Minimal Refactor Plan (StageRunner extraction)
12) Operational Playbooks (Fresh host/container, Rerun, Cleanup)
13) Appendices
    A. Paths & Directory Map
    B. JSON Schemas (pointer, manifest, stage, prep_result)
    C. Port/Lifecycle Contracts
    D. SYS_ADMIN / Capabilities Rationale
    E. DeepGEMM Compile CLI (canonical list)
    F. Grep Cookbook
    G. Glossary

1) Core Principles (Do/Don’t) and Invariants
Do
- Write all bind‑mount files as devuser. Root must never author files under /telemetry or /profiles. Root is only used to start privileged exporters (nv‑hostengine, capability-granted binaries) that touch non‑bind paths.
- Make the container responsible for pointer and manifest. init‑run.sh writes /telemetry/container_run_meta.env.tmp then renames to /telemetry/container_run_meta.env only after the manifest is fully written.
- Keep one per‑run log: /telemetry/logs/container‑run‑<run_id>.log. Both init and sgl-admin append to this file. No separate “human log” files per stage.
- Pass explicit CLI flags for any SGLang server used for priming. Do not rely on Python ServerArgs objects across processes; they can get recomputed/overridden.
- Fail fast on ownership and permissions; never “fix” by chown/chmod at runtime.
- Stage JSON first, then aggregate. If aggregation fails, prior stage JSONs remain the source of truth.

Don’t
- Don’t pre‑create the pointer file in host start scripts. It creates stale run races.
- Don’t “enrich” the manifest from the host after boot. The container writes the final manifest.
- Don’t install /usr/local/bin/sgl-admin clones. Execute the repo file (python /workspaces/sglang/tools/sgl_admin.py) so bind‑mounted edits apply immediately.
- Don’t add XDG fallbacks for cache roots; use the canonical paths provided in the manifest and fail if missing.
- Don’t add fallback ports. 30000 is strict for warm‑ups. If it’s busy, fail fast and fix the lifecycle.
- Don’t “adjust” mem_fraction_static; it must be 0.94 for cache priming. KV cache dtype must be fp8_e4m3.

2) Environment & Mounts: Host ⇄ Container Contracts
Host roots (defaults)
- Parent: $HOME/sglang‑observability
  - telemetry/: TSDB, traces, per‑run manifests, logs, pointer file
  - profiles/: kernel caches (triton, torchinductor, flashinfer, deep_gemm, moe_configs)
  - models/: model storage (mounted to /models)
  - huggingface/: HF cache for devuser in container

Container mounts
- /telemetry ← $HOME/sglang‑observability/telemetry
- /profiles  ← $HOME/sglang‑observability/profiles
- /models    ← $HOME/sglang‑observability/models
- /home/devuser/.cache/huggingface ← $HOME/sglang‑observability/huggingface
- /workspaces/sglang ← repo working tree

Services (init‑run.sh)
- Prometheus :9090, Jaeger (OTLP :4317/:4318, admin :14269, UI :16686), node_exporter :9100, dcgm_exporter :9400, and nv‑hostengine.
- Telemetry exports are recorded in the manifest under services and telemetry_surfaces.

Strict port policy for SGLang server processes
- 30000: reserved for warm‑ups (FlashInfer/Inductor) that verify Prom scrape. If 30000 is busy, we error; we do not “fall back.”
- DeepGEMM compile: independence from Prometheus; compile disables metrics/trace to avoid coupling to 30000. It still binds an HTTP port for readiness/generate, but scrape is off.

3) Pointer, Manifest, and Logs: Single‑Source‑of‑Truth Semantics
Pointer (/telemetry/container_run_meta.env)
- Written only by the container as devuser, atomically. Example:
  CONTAINER_RUN_META_JSON=/telemetry/container_runs/container‑run‑20251010T183737Z‑e37a089d.json
  CONTAINER_RUN_META_JSON_HOST=/home/ubuntu/sglang‑observability/telemetry/container_runs/container‑run‑20251010T183737Z‑e37a089d.json
- Host start deletes any stale pointer before container launch, then waits for existence (not contents).

Manifest (/telemetry/container_runs/<run_id>.json)
- Base: container_run_id, started_at, image, git_revision, service ports, storage.
- Enrichments (container‑side):
  - paths.container: relative paths (storage_root, log_file, etc.).
  - paths.host: telemetry_root, profiles_root, models_root (supplied by start via env).
  - profile_cache.start: per‑cache exists/size/count/latest_mtime, partial markers, host_path hints.
- Never mutated by host scripts after creation.

Logs (/telemetry/logs/container‑run‑<run_id>.log)
- init‑run: tee’s background services to this file as devuser.
- sgl-admin: appends subprocess streams to the same file. RESULT_* lines go to stdout but do not create new log files.
ASCII flow (happy path)

Host                                                    Container (PID 1)                              Container (devuser)
-----                                                   ------------------                              -------------------
./scripts/start_observable_container.sh  ─────docker run────▶ /opt/observability/init-run.sh ─▶ start services, write manifest
       │                                                 │                                        │
       │                                                 └───▶ write pointer .tmp → rename ◀─────┘
       ▼
wait for pointer
       ▼
docker exec -u devuser sgl_admin.py caches ensure──────▶ stage subprocesses (devuser) ───────────▶ append to session log; write stage JSON
                                                                                                  └──▶ aggregate prep_result.json

Host start script (scripts/start_observable_container.sh)
- Creates host telemetry/profiles/models/huggingface directories.
- Deletes stale pointer if exists.
- docker run with: -v $HOST_TELEMETRY_ROOT:/telemetry, -v $HOST_PROFILES_ROOT:/profiles, -v $HOST_MODELS_ROOT:/models, -v $HOST_HF_ROOT:/home/devuser/.cache/huggingface; passes HOST_* paths via -e.
- Wait loop: tests existence of /telemetry/container_run_meta.env; then checks that the manifest referenced exists on host and prints both host and container paths.

Container init (/.devcontainer/observability/init-run.sh)
- Creates per‑run id, session log path; ensures telemetry subdirs exist as devuser.
- Starts Prometheus, Jaeger, node_exporter, dcgm_exporter; captures any warnings.
- Writes manifest (base), then enriches with paths.host/container and profile_cache.start snapshot.
- Publishes pointer atomically (.tmp → rename) as devuser.

Populate caches (scripts/cache/populate_caches.sh)
- docker exec -u devuser bash -lc "python /workspaces/sglang/tools/sgl_admin.py caches ensure …"
- No wrapper binaries. Repo file is the only entrypoint, so code changes in the repo always apply.

6) sgl_admin.py Internals (Orchestration, Locks, Stage JSON, Aggregation)

Entry points
- tools/sgl_admin.py caches inspect → prints JSON snapshot of /profiles caches. Does not touch logs.
- tools/sgl_admin.py caches ensure → orchestrates the selected stages and writes per‑stage JSON + aggregate prep_result.json; prints RESULT_* lines to stdout.

Environment preparation (_prepare_env)
- Sets HOME=/home/devuser and canonical cache roots:
  - TRITON_CACHE_DIR=/profiles/triton
  - TORCHINDUCTOR_CACHE_DIR=/profiles/torchinductor
  - FLASHINFER_WORKSPACE_DIR=/profiles/flashinfer
  - SGLANG_DG_CACHE_DIR=/profiles/deep_gemm
  - SGLANG_MOE_CONFIG_DIR=/profiles/moe_configs
  - FLASHINFER_JIT_LOG_DIR=/profiles/flashinfer/90a and symlink from ~/.cache/flashinfer/90a

Locks and in‑progress markers
- Lock path: /profiles/.locks/<slug>.lock. Simple file lock; if present, we emit wait heartbeats and fail after a configured timeout.
- Marker: /profiles/.in_progress/<stage>.json { owner_pid, started_at }. Presence marks caches as partial in both inspect and profile_cache.start.

Stage JSON then aggregate
- For each stage, build a dict { ran, status, status_code, duration_s, artifacts, error_type, warnings[], errors[] }, write atomically to /telemetry/container_runs/<run_id>/stages/<stage>.json.
- Then rebuild /telemetry/container_runs/<run_id>/prep_result.json by reading all stage JSONs present. If aggregation fails, prior stage JSONs remain intact.

Telemetry probe (optional)
- After warm-ups, we query Prometheus for token increases under this run label. This is best‑effort and recorded under telemetry_probe (ok flag + sample count).
7.1 DeepGEMM compile (canonical)

Goal
- Precompile DeepGEMM kernels so that serving (or subsequent warm-ups) do not JIT during request handling.

Artifacts
- /profiles/deep_gemm/<model_slug>/compile.log (append-only; contains server prints and exceptions)
- /profiles/deep_gemm/<model_slug>/signature.json (on success; used for strict noop)
- /telemetry/container_runs/<run_id>/stages/deep_gemm.json (per-run stage output)
- /telemetry/container_runs/<run_id>/prep_result.json (aggregate)

Signature (keys)
- device_name, compute_capability, cuda, driver_version, torch_version, triton_version, sglang_commit, tp, model_slug

Ensure vs Rebuild
- ensure: if signature.json exists and matches the above tuple, return noop (ran=false, status=noop) and do not launch a server.
- rebuild: ignore signature.json and run the compile path; overwrite signature.json on success.

Permissions preflight (fail-fast, no mutations)
- Verify /profiles/deep_gemm is writable by devuser (create+delete .write_test). If not, status=error with error_type=permissions_unexpected.
- Verify disk headroom (default 1 GiB, configurable): if insufficient, status=error with error_type=insufficient_disk and include required/free bytes.

Lock/Marker
- Lock: /profiles/.locks/deep_gemm.lock; if held beyond timeout (default 600s), error with lock_timeout.
- Marker: /profiles/.in_progress/deep_gemm.json; create at start, remove on success, leave on error.

Server launch (compile) — CLI flags only (no Python object handoffs)
- We launch sglang.launch_server with the required flags (no “auto compute,” no “helpful overrides”). Canonical CLI:

  python -m sglang.launch_server \
    --model-path /models/<MODEL> \
    --host 127.0.0.1 --port 30000 \
    --tp-size <TP> \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.94 \
    --chunked-prefill-size 4096 \
    --max-mamba-cache-size 1 \
    --max-prefill-tokens 272000 \
    --context-length 272000 \
    --enable-metrics false --enable-trace false

Notes:
- disable_cuda_graph flag defaults false; we do not pass it to keep CUDA graph enabled (your policy).
- We intentionally set metrics/trace false during compile to avoid port 30000 coupling to Prometheus; warm-ups retain metrics (see §7.2/7.3).

Readiness/Trigger/Teardown
- Readiness: poll /v1/models (200 OK) up to compile timeout (default 3600s).
- Trigger: POST /generate { input_ids:[0,1,2,3], sampling_params:{ max_new_tokens: 8, temperature: 0 } } (timeout 60s). Non‑200 → error with body.
- Teardown: SIGTERM → wait 10s → SIGKILL; ensure process exit.

Validation in compile.log
- kv cache dtype print must be fp8_e4m3 (torch.float8_e4m3fn or fnuz on HIP): “Using KV cache dtype: …”.
- There must be no “Not enough memory. Please try to increase --mem-fraction-static.” at mem_fraction_static=0.94.
- If a failure occurs, include the trailing 20 lines of compile.log in stage JSON under compile_log_tail.

Stage JSON (deep_gemm.json)
- Example fields: { ran:true, status:"ok", status_code:0, duration_s:…, artifacts:{ cache_dir, compile_log, signature }, error_type:null, warnings:[], errors:[] }

Common failure modes
- permissions_unexpected: /profiles/deep_gemm not writable → fix host dir ownership; never chown from inside container.
- insufficient_disk: fail early; increase disk space.
- lock_timeout: concurrent run; resolve contention or increase timeout if truly necessary.
- nvrtc_not_found / libcuda_missing: system‑level; surfaced via compile_log_tail.
- port_busy (if metrics/trace were enabled during compile): another server bound to 30000; stop servers or change policy; for compile we keep metrics off to avoid this.

7.2 FlashInfer warm‑up (metrics on, strict port 30000)

Goal
- Build/validate FlashInfer workspace under /profiles/flashinfer and verify end‑to‑end with a short generate.

CLI flags (short‑lived server)
- Same defaults as compile, but metrics enabled:
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.94 --chunked-prefill-size 4096 \
  --max-mamba-cache-size 1 --max-prefill-tokens 272000 --context-length 272000 \
  --enable-metrics true --enable-trace false --port 30000

Lifecycle
- Error immediately if 30000 is busy (no fallback), readiness (GET /get_model_info), generate tiny request, teardown. Record file counts and bytes under /profiles/flashinfer.

7.3 TorchInductor warm‑up (metrics on, strict port 30000)

Goal
- Prime TorchInductor cache under /profiles/torchinductor with a small compile workload.

CLI flags (short‑lived server)
- Same defaults + --enable-torch-compile true; metrics enabled; strict 30000 policy. Readiness, trigger, teardown identical to FlashInfer.

7.4 Fused MoE configs (triton tuner)

Goal
- Persist batch‑keyed JSON configs under /profiles/moe_configs/configs/triton_<ver>/ and verify runtime consumption.

Execution
- Run benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py with explicit batch size(s) and dtype. Simplify to “one size per invocation” to reduce complexity; persist per‑size results.
- Verification: scan session log for the consumption string (e.g., “Using MoE kernel config: <basename>”). Record verified_in_log true/false in stage JSON.
Goal: Extract shared orchestration concerns from sgl_admin.py into a minimal StageRunner interface, keeping sgl_admin focused on IO, locks, and aggregation. Avoid large rewrites; this is a surgical extraction.

StageRunner (per stage)
- preflight() → returns error StageResult or None
- build_command() → returns command/args for the stage server (or tuner)
- run() → executes command with readiness/trigger/teardown, returns artifacts dict
- describe_artifacts() → canonicalize paths for stage JSON
- cleanup() → ensure servers are dead, remove markers/locks
- error_map(log_tail) → classifies errors into error_type values

sgl_admin orchestrates
- permission/disk preflight → lock → marker → call StageRunner → stage JSON write → aggregate → release lock → clear marker.

12) Operational Playbooks

Fresh Host, Fresh Container
- ./scripts/stop_observable_container.sh
- rm -rf $HOME/sglang‑observability (optional)
- ./scripts/start_observable_container.sh
- Verify: pointer exists; manifest contains paths.host.*; single session log created.

DeepGEMM ensure (first run)
- ./scripts/cache/populate_caches.sh --model /models/<MODEL> --tp 1 --deep-gemm ensure --moe skip --flashinfer skip --inductor skip
- Watch /profiles/deep_gemm/<slug>/compile.log for fp8_e4m3 dtype and absence of mem fraction error; expect stage JSON ok.

DeepGEMM ensure (second run)
- ./scripts/stop_observable_container.sh && ./scripts/start_observable_container.sh
- Repeat ensure; expect status noop.

Cleanup port 30000 busy
- docker exec sglang-dev pgrep -a python | grep sglang.launch_server; kill if present; restart container.
A) Paths & Directory Map

Host (default parent: $HOME/sglang‑observability)
- telemetry/
  - logs/
    - container‑run‑*.log
  - prometheus/
    - <run_id>/ (TSDB)
    - prometheus.yml (generated)
  - jaeger/
    - <run_id>/badger/{key,data}
  - container_runs/
    - <run_id>.json (manifest)
    - <run_id>/stages/*.json (per‑stage)
    - <run_id>/prep_result.json
  - container_run_meta.env (pointer; atomic writes)
- profiles/
  - triton/
  - torchinductor/
  - flashinfer/
  - deep_gemm/
    - <model_slug>/compile.log
    - <model_slug>/signature.json
  - moe_configs/
    - configs/triton_<ver>/*.json
  - .locks/
  - .in_progress/
- models/
- huggingface/

Container (bind mounts)
- /telemetry ↔ host telemetry
- /profiles  ↔ host profiles
- /models    ↔ host models
- /home/devuser/.cache/huggingface ↔ host hf cache
- /workspaces/sglang ↔ repo

B) JSON Schemas (shapes)

Pointer file (2 lines)
- CONTAINER_RUN_META_JSON=<container path to manifest>
- CONTAINER_RUN_META_JSON_HOST=<host path to manifest>

Stage JSON (example)
{
  "ran": true,
  "status": "ok|error|noop|skipped",
  "status_code": 0,
  "duration_s": 123.456,
  "artifacts": { "cache_dir": {"container_path": "/profiles/deep_gemm", "host_path": null} },
  "error_type": null,
  "warnings": [],
  "errors": []
}

prep_result.json (aggregate)
{
  "schema_version": "1",
  "status": "ok|partial",
  "run": {
    "run_id": "container‑run‑…",
    "model_slug": "…",
    "tp": 1,
    "device_name": "…", "compute_capability": "sm_90",
    "cuda": "12.8", "driver_version": "570.148.08",
    "torch_version": "…", "triton_version": "…", "sglang_commit": "…",
    "started_at": "…Z", "finished_at": "…Z", "duration_s": 123.456,
    "settings": { "source": "flags", "mem_fraction": 0.94, … }
  },
  "stages": { "deep_gemm": { … }, "flashinfer": { … }, "inductor": { … }, "moe_tune": { … } },
  "telemetry_probe": { "ok": true, "sample_count": 12, … },
  "errors": [], "warnings": []
}

C) Port/Lifecycle Contracts
- 30000 reserved for warm‑ups; fail fast if busy; no fallback ports.
- Compile step disables metrics/trace and thus does not need 30000; readiness still via HTTP.
- Teardown must terminate all short‑lived servers; leaving 30000 occupied is a bug.

D) SYS_ADMIN / Capabilities Rationale
- dcgm-exporter requires CAP_SYS_ADMIN to expose profiling metrics via NVML/DCGM. We set filecap at build and include --cap-add SYS_ADMIN in docker run so the capability is in the container’s bounding set. This has nothing to do with file writes or ownership.

E) DeepGEMM Compile CLI (canonical list)
python -m sglang.launch_server \
  --model-path /models/<MODEL> \
  --host 127.0.0.1 --port 30000 \
  --tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-mamba-cache-size 1 \
  --max-prefill-tokens 272000 \
  --context-length 272000 \
  --enable-metrics false --enable-trace false

F) Grep Cookbook
- Pointer appears: grep -n "CONTAINER_RUN_META_JSON_HOST" $HOME/sglang‑observability/telemetry/container_run_meta.env
- New session log path: jq -r '.storage.log_file' <manifest.json>
- kv dtype in compile: rg -n "Using KV cache dtype" /profiles/deep_gemm/*/compile.log
- mem fraction errors: rg -n "Not enough memory.*mem-fraction-static" /profiles/deep_gemm/*/compile.log
- port busy: rg -n "Address already in use|port 30000 is busy" /telemetry/logs/container‑run-*.log

G) Glossary
- Pointer: the 2‑line file that points to the current run’s manifest (container and host paths).
- Manifest: per‑run JSON with services, storage, enriched paths, and per‑cache snapshot.
- Stage JSON: per‑stage outcome file; stage‑first write policy prevents data loss.
- Prep result: aggregate of stage JSONs for convenient consumption.
- Ensure: only act if cache not valid; noop if signature matches.
- Rebuild: force rerun regardless of signature.

14) Q&A and Deep Dives (Answering the Open Questions)

14.1 Are we using the DeepGEMM script?
- Yes. The DeepGEMM priming path is driven by python/sglang/compile_deep_gemm.py.
- What it did before: launched the server via Python multiprocessing with a ServerArgs object, refined settings in‑process, and then waited for readiness/trigger. This led to config drift because the child process’s runtime could recompute/override parameters (mem_fraction_static, kv dtype) in __post_init__ / model‑specific handlers.
- What it does now (refactor in progress, and recommended): build a CLI for python -m sglang.launch_server with the exact flags and start the server as a true subprocess. That guarantees the runtime sees precisely our settings. Readiness → generate → teardown remains the same. See §7.1 and Appendix E for the canonical CLI.
- Compile‑only policy: metrics/tracing disabled to decouple from Prometheus. Warm‑ups still require 30000 and keep metrics on for verification.

14.2 What tests exist? What do they do? Are they actually doing anything?
- Upstream test suites (high level; not exhaustive):
  - python/sglang/test/*: runtime and API tests (server lifecycle utilities, generate calls, attention backends, LoRA, quantization paths, etc.).
  - sgl-router/tests and sgl-router/py_test: router component tests (Rust + Python integration).
  - sgl-kernel/tests: kernel unit tests.
  - test/srt/*: many scenario tests around SRT behavior.
- Our added tests (provider‑prep specific):
  - test/args/test_server_args_flow.py: Unit test that asserts CLI flags for mem_fraction_static, kv_cache_dtype, chunked_prefill_size, max_mamba_cache_size, max_prefill_tokens, context_length, and disable_cuda_graph are preserved by prepare_server_args. Purpose: pin the interface where arguments actually enter the server; prevent unexpected recompute.
  - test/args/test_compile_cli_args.py: Unit test that the compile CLI builder emits the exact DeepGEMM compile flags (0.94, fp8_e4m3, etc.). Purpose: guard against accidental changes to canonical compile defaults.
  - test/e2e/test_pointer_and_deepgemm_flow.py (GPU‑gated; skipped by default): Smoke flow to start container, wait for pointer, ensure DeepGEMM (first run), stop/restart, ensure again (noop). Purpose: validate pointer/manifest/logging contracts and ensure/noop semantics end‑to‑end. Requires SGL_E2E_CONTAINER_TESTS=1 and a real model under /models.
- Are upstream tests “doing anything” for provider‑prep? Not directly; they validate runtime and APIs, not the provider‑prep orchestration. Our new tests are surgical guardrails around the failure modes we hit: pointer races, CLI flag propagation, and compile CLI composition.

14.3 What code did we abandon mid‑edit? What’s next?
- compile_deep_gemm changeover:
  - Before: multiprocessing + ServerArgs handoff. After: pure CLI launch via _build_compile_cli_args + subprocess.Popen.
  - Status: Implemented; we still need to remove any leftover code paths that assume multiprocessing.
- sgl_admin.py DeepGEMM: switched to stage JSON first then aggregate; added signature/noop and explicit permission/disk preflight. Next: call a shared helper that builds the CLI and parse durations more explicitly; add a short‑circuit “readiness failed early” error classification.
- spec‑file support: removed from sgl_admin and scripts. Next: delete any dead branches referencing spec input; ensure docs/scripts reflect the flag‑only interface.
- Host enrichments: removed. Next: double‑check nothing else tries to mutate the manifest on the host; all enrichments live in init‑run.
- Inspect fallback removal: XDG fallback removed. Next: ensure downstream tools read paths exclusively via manifest.
- StageRunner refactor: not done yet; recommended as a follow‑up to isolate per‑stage complexity.

14.4 How do SGLang arguments get passed to the server?
- CLI → prepare_server_args → ServerArgs.__post_init__ → runtime (ModelRunner, Scheduler). This is the only safe contract for priming runs.
- Hazards inside ServerArgs:
  - _handle_gpu_memory_settings: computes mem_fraction_static only if it is None. If you pass --mem-fraction-static 0.94, it should not recompute.
  - _handle_model_specific_adjustments: some branches (e.g., DeepSeek NSA) can force mem_fraction_static to 0.8 and change kv dtype. We do not expect to hit those, but they are hard overrides to watch for.
  - adjust_mem_fraction_for_vlm: can reduce mem fraction for VLM models.
- Practical rule: pass every required caching knob explicitly with CLI flags and verify logs; if any “auto” or hard override path changes them, treat it as a bug and remove the override for provider‑prep.

14.5 What configurations are we using for the server? Where are they defined?
- Defaults for caching (hard policy; §8):
  mem_fraction_static=0.94, kv_cache_dtype=fp8_e4m3, chunked_prefill_size=4096, max_mamba_cache_size=1, context_length=272000, max_prefill_tokens=272000, disable_cuda_graph=false.
- Where set:
  - DeepGEMM compile: python/sglang/compile_deep_gemm.py builds a CLI with these flags and invokes sglang.launch_server as a subprocess; metrics/trace disabled here.
  - Warm-ups (FlashInfer/Inductor): tools/sgl_admin.py::_start_warmup_server passes the same required flags when launching short‑lived servers; metrics enabled and strict port 30000 check.
  - tools/cache_defaults.json declares compile_timeout_s, lock_timeout_s, and mem fraction policy; sgl_admin uses these for timeouts and mem fraction env mirrors.

14.6 What do the provider‑prep steps do? In what order should they run?
- Recommended order (single‑GPU case):
  1) DeepGEMM compile (ensure first): produce DeepGEMM kernels under /profiles/deep_gemm and write signature.json. Rationale: avoids long JITs later and reduces end‑to‑end compile time in warm-ups.
  2) MoE tuning (sizes one‑by‑one): produce batch‑keyed Triton configs under /profiles/moe_configs; later servers will consume these if available.
  3) FlashInfer warm‑up: populate /profiles/flashinfer and verify via /generate; port 30000 must be free; metrics on.
  4) TorchInductor warm‑up: compile representative graphs; port 30000 must be free; metrics on.
- Reruns: DeepGEMM ensure should be noop via signature. MoE can be run per size as needed. Warm-ups should be repeatable and idempotent (they may recompile but that is acceptable).

14.7 Should we use uv or pip? What mess was corrected?
- The container image already installs uv for devuser. We should use uv run for unit tests and tooling inside the container to avoid PATH / venv issues. For invoking sgl_admin and server modules, calling python -m is acceptable because we’re not managing multiple venvs in provider‑prep scripts; but for consistency and reproducibility, uv run pytest (or uvx) is preferred for tests.
- Mess corrected:
  - Removed /usr/local sgl-admin copies; using repo path eliminated drift.
  - Removed host manifest enrichment; pointer/manifest is now container‑owned and atomic.
  - Deleted XDG fallback for inspect; no more silent divergence to ~/.cache.
  - Fixed stale pointer race by deleting pointer before container start.
- No “pip hacking” is required for provider‑prep flows; we rely on the environment shipped in the container.

14.8 What caused the disastrous behavior earlier? What context was missing?
- Wrong approach: starting the compile server via a Python object handoff. The child runtime recomputed mem_fraction and kv dtype; we never saw our intended values. Root cause: not respecting the true CLI boundary for runtime configuration.
- Pointer race: not clearing stale pointer caused start to latch an old manifest/log; sgl-admin wrote to an old file and appeared “stuck.”
- Port occupancy: a long‑lived sglang::scheduler was using ~79 GiB, shrinking headroom and compounding failures. We didn’t stop stray servers before priming.
- Overbroad “safety” code: chmod 600, umask changes, runtime setcap, XDG fallbacks — all introduced brittle behavior and masked the real problem.
- Missing explicit lifecycle policies: no hard rule for 30000, no atomic pointer semantics, no stage JSON before aggregate.

14.9 What needs to be done next?
- Finish the compile_deep_gemm CLI changeover and delete any leftover multiprocessing code paths. Ensure sgl_admin calls the new CLI builder.
- Add unit tests: lock timeout (shortened), permission fail‑fast, stage JSON write on error, and readiness failure classification.
- Add smoke coverage for FlashInfer/Inductor warm‑ups with port 30000 checks and teardown verification.
- Consider adding a “caching mode” for container start to disable Prom scraping if we want to run all priming first and then enable metrics for serving runs.
- Migrate environment variable usage to SGLANG_* only (remove SGL_* warnings) once callers are updated.

