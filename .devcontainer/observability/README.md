# Devcontainer Observability Stack

The devcontainer image bundles Prometheus, Jaeger all-in-one, node_exporter, and
dcgm-exporter alongside the SGLang runtime so that every container lifetime
produces a self-contained observability bundle.

## Components

All services run inside the `sglang-dev` container and are started by
`.devcontainer/observability/init-run.sh`. For DCGM profiling gauges (SM/DRAM/
tensor activity, NVML utilisation, etc.) the container must launch with
`--cap-add SYS_ADMIN`; the init script will attempt to apply `cap_sys_admin` to
`dcgm-exporter` and logs a warning if the capability is missing. (Grace↔Hopper
C2C/host-memory counters are not exposed on this VM; see
`metrics_catalog.final.md` for the supported set.)

| Component        | Listen Ports | Data Directory                                   |
|------------------|--------------|--------------------------------------------------|
| Prometheus       | 9090         | `/telemetry/prometheus/container-run-*/`         |
| Jaeger UI/OTLP   | 16686 / 4317 | `/telemetry/jaeger/container-run-*/badger/{key,value}` |
| node_exporter    | 9100         | none (metrics only)                              |
| dcgm-exporter    | 9400         | none (metrics only)                              |
| SGLang server    | 30000 / 29000| user-launched via `docker exec`                  |

## Container Lifecycle

On start, `init-run.sh`:

1. Generates a run identifier `container-run-<timestamp>-<id>`.
2. Writes metadata to `/telemetry/container_run_meta.env` (run ID, start time,
   log path, Prometheus/Jaeger storage paths).
3. Creates one log file per run at `/telemetry/logs/container-run-<timestamp>-<id>.log`
   and tees all container stdout/stderr into it.
4. Launches Prometheus, Jaeger all-in-one, node_exporter, and dcgm-exporter as
   background processes after starting `nv-hostengine`.
5. Executes the devcontainer command (`sleep infinity`), leaving the container
   ready for interactive work.

Each run therefore produces a single log file, a Prometheus TSDB directory, and
Jaeger badger data rooted under `.devcontainer/storage/` on the host.

## Storage Layout

Running `./.devcontainer/setup-storage.sh` prepares the host-side directory tree:

```
.devcontainer/storage/
  models/                # model checkpoints (durable)
  huggingface/           # HF caches (durable)
  profiles/
    deep_gemm/
    flashinfer/
    moe_configs/configs/
    torchinductor/
    triton/
  logs/
    container-run-*.log  # One container-lifetime log per run
  container_run_meta.env # Updated each time the container starts
  prometheus/            # Per-run TSDB directories (created at runtime)
  jaeger/                # Per-run badger directories (created at runtime)
```

Prometheus and Jaeger are configured to write into subdirectories named after
`CONTAINER_RUN_ID`. Older runs can be inspected by pointing Prometheus/Jaeger at
those persisted directories.

## Launching SGLang

Because `init-run.sh` already captures stdout/stderr, simply start SGLang from
inside the container and append to the log noted in `container_run_meta.env`:

```bash
# On the host
RUN_META=.devcontainer/storage/container_run_meta.env
source "$RUN_META"

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

The log survives `docker compose down` / `docker stop`; each subsequent container
lifetime produces a new file.

## Metrics & Traces

- Prometheus scrapes exporters on localhost (node_exporter at `:9100`,
  dcgm-exporter at `:9400`, and SGLang at `:30000` once a server instance is
  running). The global `external_labels` block injects `container_run` with the
  current run ID.
- The Jaeger all-in-one binary exposes OTLP gRPC/HTTP on `:4317/:4318` so SGLang
  can emit traces directly without a separate collector.
- A complete list of currently emitted metrics is maintained in
  `.devcontainer/observability/metrics_catalog.final.md`.

## Workflow Summary

1. Run `./.devcontainer/setup-storage.sh` once to prepare the host directories and
   seed `.devcontainer/storage/container_run_meta.env` for run metadata.
2. Open the devcontainer (VS Code or `devcontainer open`)—Prometheus, Jaeger, and
   the exporters start automatically.
3. Start SGLang via `docker exec ...` as shown above; issue requests and collect
   metrics/traces.
4. When finished, stop the container; inspect
   `.devcontainer/storage/logs/container-run-*.log` and the matching Prometheus/
   Jaeger directories for the full observability artifact.

Update this README whenever binaries, ports, or storage paths change.
