# Handoff to SLICE‑Bench: Ready for Smokes

Status: Ready
Date: 2025‑10‑09
Owner: sglang devcontainer + observability stack

## TL;DR
- Helper container is stopped now; you can start it or call your provider‑prep (which starts it if needed).
- Model is staged and mounted:
  - Host: `$HOME/sglang-observability/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`
  - Container: `/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`
- Use `--tp 1` for this single GH200. Do not use `--tp 2` (will OOM).
- sgl-admin is baked into the image; prep_result.json is written atomically and discoverable via the run manifest.
- Storage moved under `$HOME/sglang-observability` with clean ownership; services run as the mapped host uid:gid.

## Changes Since Last Coordination

1) Parent storage root moved (host)
- New parent: `$HOME/sglang-observability`
  - Telemetry (TSDB, traces, logs, run manifests): `telemetry/`
  - Caches (Triton/Inductor/FlashInfer/DeepGEMM/MoE configs): `profiles/`
  - Models: `models/`
  - HuggingFace cache: `huggingface/`
- Start script creates these on startup and mounts them into the container.

2) Permissions and uid:gid alignment
- At container start, devuser inside the container is remapped to the host uid:gid. All observability services (Prom/Jaeger/node_exporter/dcgm‑exporter) and provider‑prep subprocesses run as that user.
- umask is set to 002; new files on binds are ubuntu:ubuntu owned on the host.
- Start script preflight fails if the new host roots are owned by a different uid:gid; bypass with `HOST_DIR_OWNERSHIP_IGNORE=1` if needed.

3) sgl-admin baked into the image
- No manual copy. CLI is available at `/usr/local/bin/sgl-admin`.
- Provider‑prep JSON contract unchanged:
  - `prep_result.json` (schema_version:"1") with run context, per‑stage status, artifacts, telemetry_probe.
  - `RESULT_JSON <container_path>` and `RESULT_STATUS ...` on stdout.
  - Manifest gains `paths.container.prep_result` (and host‑rebased path when available).

4) Image alignment
- Dockerfile and init script updated so a fresh rebuild "just works" with no post‑steps.
- Confirmed multi‑arch base images resolve to `linux/arm64`.

## What You Should Use (Bench Agent)

- Manifest pointer (host): `$HOME/sglang-observability/telemetry/container_run_meta.env`
- Read `prep_result.json` via `paths.container.prep_result` from the manifest; use `telemetry_probe.ok` for METRICS_READY.
- Use `--tp 1` on this single GH200 host.

### Example: provider‑prep

- With your CLI (delegates to sgl-admin):

```
uv run python -m slice_bench.cli provider-prep custom/sglang/qwen3-next-80b-fp8-baseline   --tune-moe --skip-moe-fusion-cache   --skip-gemm-cache   --skip-warmup-cache   --tp 1
```

- Or inside the container (raw):

```
docker exec sglang-dev sgl-admin caches ensure --json   --model /models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8   --tp 1   --flashinfer ensure   --inductor ensure
```

Expected: `RESULT_JSON ...`, `RESULT_STATUS ok ...`, and `telemetry_probe.ok: true` in JSON when warm‑up is scraped by Prometheus.

### Example: inference smoke

No change to your existing commands. Summaries read `prep_metrics_ready` from the latest `prep_result.json` (using manifest paths or container fallback).

## Ready State Checklist (we ran these)
- Model staged at the new models root; container mount verified.
- Helper start writes enriched manifest (profile_cache.start + paths.*).
- Prometheus ready at `http://127.0.0.1:9090/-/ready`.
- sgl-admin baked; CLI help works inside the container.
- Ownership of TSDB/logs/caches: ubuntu:ubuntu on host.

## Container Lifecycle for This Handoff
- Helper is currently stopped (per request). Start it when you begin smokes:

```
./scripts/start_observable_container.sh
# or simply run provider-prep (your CLI will start the helper if needed)
```

- If you need to stop it:

```
./scripts/stop_observable_container.sh
```

## Notes & Edge Cases
- If any `$HOME/sglang-observability/{telemetry,profiles,models,huggingface}` path exists but is owned by a different uid:gid, start script exits with a clear message. Fix by `chown -R $(id -u):$(id -g) ...` or set `HOST_DIR_OWNERSHIP_IGNORE=1` to proceed.
- Do not use `--tp 2` on this single GH200; use `--tp 1`.

## Versions & Architecture
- Host arch: linux/arm64 (GH200).
- Base images are multi‑arch; arm64 manifests confirmed:
  - `lmsysorg/sglang:dev` (arm64)
  - `nvidia/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04` (arm64)
- Observability versions:
  - Prometheus 3.5.0, Jaeger 1.73.0, node_exporter 1.9.1, dcgm‑exporter 4.4.1-4.5.2-ubuntu22.04.

All clear from our side. Ping if you want us to run provider‑prep first to pre‑flip METRICS_READY, otherwise you can start smokes immediately.
