# Host → Container Workflow Verification Plan

This plan walks through the four host-driven operations we need to support: start
the container, build caches, run inference, and stop the container. Each section
lists the steps and concrete checks so we catch failures early.

## 0. Host Prep

- Ensure `~/sglang-observability` exists with `models/` populated (e.g.,
  `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`).
- Rebuild or restart the devcontainer image if the Dockerfile changed.

**Verify**
- `~/sglang-observability/models`, `profiles`, `telemetry/logs`,
  `telemetry/prometheus`, `telemetry/jaeger` exist and are owned by the invoking
  user.
- `/profiles` is empty aside from scaffolding; only the expected model files
  exist.

## 1. Container Lifecycle

1. (Optional) `scripts/start_observable_container.sh VALIDATE_ONLY=1` to run
   directory/ownership preflight.
2. `scripts/start_observable_container.sh` to launch.
3. `scripts/stop_observable_container.sh` to stop.

**Verify after start**
- Manifest pointer `~/sglang-observability/telemetry/container_run_meta.env`
  points to the current run (file mode `600`, owned by the host user);
  older manifests remain under `telemetry/container_runs/` for history.
- `/profiles` scaffolding exists and is writable by devuser.
- `/telemetry/logs/container-run-*/.log` exists; Prometheus and Jaeger
  directories are created. Node exporter / DCGM exporter write logs to the
  container log.
- `docker logs sglang-dev` shows background services starting with no errors.

- Pointer file is removed (`container_run_meta.env` no longer present).
- No cache artifacts were generated; cache directories stay empty.
- Host directories retain `ubuntu:ubuntu` ownership and are ready for a restart.

## 2. Cache Stages (fresh container per stage)

For each stage, run inside the container with only that stage enabled. Reset
`/profiles` between runs.

1. FlashInfer: `sgl_admin caches ensure --model <path> --flashinfer ensure --deep-gemm skip --inductor skip --moe skip`
   - Expect success in `prep_result.json`; `/profiles/flashinfer` writable and
     populated.
2. TorchInductor: same command with `--inductor ensure`.
3. DeepGEMM: `--deep-gemm ensure`; `compile.log` should end cleanly, signature
   written.
4. MoE tuner (optional last stage) ensuring lock acquisition/release.

**Verify** (per stage)
- Post-run `prep_result.json` marks the stage `ok`.
- Cache directories contain artifacts owned by devuser.
- No lingering `.lock` files; lock cleanup works.

## 3. Inference Smoke Test

- Start the container.
- Launch SGLang server using the JSON defaults.
- Send a simple prompt from the host.

**Verify**
- Server starts without CLI/context-length errors.
- Response returns; `/telemetry/logs` captures the request.
- Prometheus scrape directory updates.

## 4. Teardown & Persistence Check

- Stop the container.
- Ensure built caches remain under `/profiles` for reuse.
- Confirm telemetry directories still contain data.
- Restart container to verify it tolerates pre-existing host caches/logs.
