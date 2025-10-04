# Devcontainer Observability Stack

This directory holds the observability orchestration for the SGLang devcontainer.
It defines the Docker Compose topology, OpenTelemetry collector configuration, and
Prometheus scrape rules used during local benchmarking and qualification.

## Compose Topology

The compose file is `.devcontainer/docker-compose-observability.yml`. It defines
these services (all optional via the `observability` profile unless noted):

| Service | Purpose | Notes |
| --- | --- | --- |
| `sglang-dev` | Primary development container | Builds from `.devcontainer/Dockerfile.gh200`, initializes a per-run context via `init-run.sh`, exposes `/profiles` for caches/logs |
| `otel-collector` | Receives OTLP traces/metrics and forwards traces to Jaeger | Configured by `otel-collector.yml` |
| `jaeger` | Stores and serves traces | Badger DB persisted under `.devcontainer/storage/jaeger/` (`BADGER_EPHEMERAL=false`) |
| `prometheus` | Scrapes metrics from SGLang, exporters, and benchmarks | TSDB persisted under `.devcontainer/storage/prometheus/` |
| `dcgm-exporter` | Exposes detailed GPU telemetry | Runs NVIDIA's DCGM exporter (`nvidia/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04`), scraped at port 9400 |
| `node-exporter` | Exposes host CPU/memory/disk metrics | Scraped at port 9100 (requires Linux host mounts) |

## Container Manifest

The base image (`lmsysorg/sglang:dev-arm64`) is augmented in
`.devcontainer/Dockerfile.gh200` to:

- Create non-root `devuser` mirroring host UID/GID.
- Install `uv` and Rust toolchains for Python/Rust development.
- Copy shell/editor dotfiles for convenience.
- Pre-create `/home/devuser/.cache` and `/home/devuser/.local` to avoid
  permission issues with uv, FlashInfer, and DeepGEMM caches.

When the compose stack starts, `init-run.sh` generates a container run
identifier (`container-run-<timestamp>-<id>`), records the metadata in
`/telemetry/container_run_meta.env` (configurable via `RUN_META_FILE`), and
tees the container's stdout/stderr into a single log file at
`/telemetry/logs/container-run-<timestamp>-<id>.log`.

## Storage Layout

`./.devcontainer/setup-storage.sh` prepares host-side directories with the
following structure:

```
.devcontainer/storage/
  models/                # model checkpoints (durable)
  huggingface/           # HF caches (durable)
  profiles/
    deep_gemm/           # DeepGEMM compiled kernels
    flashinfer/          # FlashInfer workspaces
    moe_configs/configs/ # Triton MoE tuning outputs
  logs/
    container-run-*.log  # One container-lifetime log per stack invocation
  prometheus/            # Prometheus TSDB (durable)
  jaeger/
    badger/              # Jaeger primary badger store (durable)
    badger2/             # Jaeger secondary badger store (durable)
```

Everything under `.devcontainer/storage/` persists across container lifetimes.
Prometheus and Jaeger mount their respective directories; caches live inside
`/profiles`, while the run metadata and log file are written directly under
`/telemetry` (mirrored to `.devcontainer/storage`).

`setup-storage.sh` also relaxes permissions on the Prometheus and Jaeger
directories so the upstream containers (which run as unprivileged users) can
write their data without additional configuration.

Run this script before launching the compose stack; if you run it after the
containers have already written data you may need elevated privileges to adjust
ownership of the Prometheus/Jaeger stores.

Ephemeral data (e.g., transient OTEL processor buffers) stay in-container.

## Metrics & Traces

- Prometheus (`observability/prometheus.yml`) scrapes:
  - SGLang runtime at `sglang-dev:30000`.
  - Router metrics at `sglang-dev:29000` when present.
  - NVIDIA DCGM exporter at `dcgm-exporter:9400`.
  - Node exporter at `node-exporter:9100`.
  - Placeholder `benchmark` job for harness-provided metrics.
- OpenTelemetry collector (`observability/otel-collector.yml`) accepts OTLP
  gRPC/HTTP traffic and forwards to Jaeger and the debug exporter. Both the
  SGLang server and benchmarks should emit spans to the collector.

## Logging

`init-run.sh` tees all container stdout/stderr into the run-scoped log file
reported in `container_run_meta.env`. To keep SGLang output with the rest of the
container log, start it like this:

```bash
docker exec -d sglang-dev bash -lc '
  source /telemetry/container_run_meta.env
  python -m sglang.launch_server \
    --model-path /models/Qwen2.5-7B-Instruct-1M \
    --context-length 32768 \
    --max-running-requests 1 \
    --max-total-tokens 32768 \
    --host 0.0.0.0 --port 30000 \
    --enable-metrics \
    --tensor-parallel-size 1 \
    --disable-cuda-graph >> "${CONTAINER_LOG_FILE}" 2>&1 &
'
```

After `docker compose down`, the corresponding
`.devcontainer/storage/logs/container-run-*.log` remains for auditing.

## Workflow Summary

1. Run `./.devcontainer/setup-storage.sh` on the host once to prepare the storage tree.
2. Launch the devcontainer via VS Code or `docker compose -f .devcontainer/docker-compose-observability.yml up -d`.
3. (Optional) enable the `observability` profile to start Prometheus, Jaeger,
   DCGM exporter, and node exporter.
4. Benchmarks and SGLang can read `CONTAINER_RUN_ID` and
   `CONTAINER_LOG_FILE` from `/telemetry/container_run_meta.env` to tag metrics
   or to append to the container-lifetime log.

This README should be updated when services, mounts, or tooling versions change.
