# Devcontainer Guide (Observability + Slice-Bench Integration)

This guide describes the `sglang-dev` observability helper container, its
bundled services, storage layout, lifecycle scripts, and the manifest contract
consumed by Slice-Bench during development. The container runs on the host
network so every exposed port is reachable without additional `docker` flags.

## Components & Ports

| Component        | Listen Ports | Host Storage (relative)                       |
|------------------|--------------|----------------------------------------------|
| Prometheus       | 9090         | `.devcontainer/storage/prometheus/<run>/`     |
| Jaeger UI/OTLP   | 16686 / 4317 / 4318 | `.devcontainer/storage/jaeger/<run>/badger/{key,data}` |
| node_exporter    | 9100         | n/a (metrics only)                            |
| dcgm-exporter    | 9400         | n/a (metrics only)                            |
| SGLang server    | 30000 (router 29000) | user-launched; logs under `.devcontainer/storage/logs/` |

## Container Startup Flow

When the devcontainer starts, `.devcontainer/observability/init-run.sh`:

1. Generates a run identifier `container-run-<timestamp>-<id>`.
2. Writes `/telemetry/container_run_meta.env` containing only pointers to the
   manifest (`CONTAINER_RUN_META_JSON` and, when available, `CONTAINER_RUN_META_JSON_HOST`).
3. Creates a per-run log file in `/telemetry/logs/` and tees all stdout/stderr
   into it.
4. Launches Prometheus, Jaeger, node_exporter, dcgm-exporter, and `nv-hostengine`.
5. Prints the manifest locations to stdout for consumers.
6. Executes the devcontainer command (`sleep infinity`), keeping the container
   ready for interactive work.

## Storage Layout

Running `./.devcontainer/setup-storage.sh` creates the host-side directory
structure that is bind-mounted into the container:

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
    container-run-*.log  # one container-lifetime log per run
  container_run_meta.env # pointer to latest manifest (start script also echoes it)
  container_runs/        # JSON manifest per run (written by init-run.sh)
  prometheus/            # per-run TSDB directories
  jaeger/                # per-run badger directories
```

Prometheus and Jaeger write to subdirectories named after `CONTAINER_RUN_ID`,
allowing historical runs to be inspected later.

## Lifecycle Scripts

- `scripts/start_observable_container.sh`
  - Launches (or replaces) the `sglang-dev` container.
  - Blocks until `/telemetry/container_run_meta.env` points at a manifest JSON
    and prints:
    - `CONTAINER_RUN_META_JSON_HOST=<absolute host path>`
    - `CONTAINER_RUN_META_JSON=<container path>`
  - Emits heartbeat lines (`Container starting...`) every ~5 seconds while the
    container comes up.
  - Runs the container in host-network mode so ports (30000 for SGLang, 29000 for router,
    9090/9400/9100/16686/4317/4318) are immediately reachable from the host.
  - Truncates the env pointer before launch so stale run IDs cannot leak
    through.
  - Rewrites the manifest to add host-relative paths, cache metadata, and
    telemetry-surface status before printing readiness.
  - If the manifest contains warnings, they are printed before the “ready” line.
  - Exits non-zero if Docker reports the container stopped or failed; the tail of
    the container log (last 50 lines) is written to stderr for debugging.
  - `--help` prints usage; no other CLI flags are accepted. Set `HOST_MANIFEST_ROOT`
    in the environment to override the default host manifest directory
    (`.devcontainer/storage/container_runs`).

- `scripts/stop_observable_container.sh`
  - Removes the `sglang-dev` container if it exists. Exits 0 if nothing was
    running. Supports `--help` for usage text.

Both scripts assume Slice-Bench (or the user) runs them from the repository root
so relative paths resolve correctly.

## Run Metadata

Every container lifetime produces:

- Log file: `/telemetry/logs/<CONTAINER_RUN_ID>.log`
- Prometheus TSDB: `/telemetry/prometheus/<CONTAINER_RUN_ID>/`
- Jaeger storage: `/telemetry/jaeger/<CONTAINER_RUN_ID>/badger/{key,data}`
- Manifest: `/telemetry/container_runs/<CONTAINER_RUN_ID>.json`

`/telemetry/container_run_meta.env` (mounted at
`.devcontainer/storage/container_run_meta.env`) now only contains the manifest
pointers:

```
CONTAINER_RUN_META_JSON=/telemetry/container_runs/container-run-...
CONTAINER_RUN_META_JSON_HOST=/absolute/host/path/.devcontainer/storage/container_runs/container-run-....json
```

Slice-Bench should prefer the host path when present and fall back to the
container path otherwise.

### Manifest Highlights

- `telemetry_surfaces` reports the status of each data surface (`sglang_metrics`,
  `tracing`, `node_metrics`, `dcgm_metrics`). Exporter-backed surfaces mirror
  their runtime state; the others report `expected` because Slice-Bench controls
  server launch.
- `paths.host` lists repo-relative paths for logs, Prometheus TSDB, Jaeger
  storage, and cache roots. `paths.container` provides the matching paths
  relative to `/telemetry`.
- `profile_storage` enumerates each profile directory with its host-relative
  path and whether cached data is present at startup.
- `storage_roots` maps the broader cache directories (`models`, `huggingface`,
  `profiles`) to repo-relative paths. Cache presence is reported per-profile via
  `profile_storage`.
- `warnings` retains the authoritative list of degradations; your automation
  should continue to gate on an empty list unless a test requires otherwise.

## Launching SGLang

Because the init script already captures stdout/stderr, launch SGLang after
reading the manifest pointer at `.devcontainer/storage/container_run_meta.env`:

```bash
# On the host
RUN_META=.devcontainer/storage/container_run_meta.env

# Prefer absolute host path when provided, fall back to the container path.
MANIFEST_PATH=$(awk -F= '/CONTAINER_RUN_META_JSON_HOST/{print $2}' "$RUN_META")
if [ -z "$MANIFEST_PATH" ]; then
  MANIFEST_PATH=$(awk -F= '/CONTAINER_RUN_META_JSON=/{print $2}' "$RUN_META")
fi

LOG_FILE=$(jq -r '.storage.log_file' "$MANIFEST_PATH")

docker exec -d sglang-dev bash -lc "
  python -m sglang.launch_server \\
    --model-path /models/Qwen2.5-7B-Instruct-1M \\
    --context-length 32768 \\
    --max-running-requests 1 \\
    --max-total-tokens 32768 \\
    --host 0.0.0.0 --port 30000 \\
    --enable-metrics \\
    --tensor-parallel-size 1 \\
    --disable-cuda-graph >> '$LOG_FILE' 2>&1 &"
```

Each container lifetime emits a fresh log file in `.devcontainer/storage/logs/`
even across `docker stop` / `docker compose down`.

## Warning Taxonomy

`init-run.sh` writes the following warning strings when applicable. They appear
both in the container stdout and in the manifest’s `warnings` array. Slice-Bench
can gate on these values to determine whether a run is valid for a given suite.

1. `dcgm-exporter binary not found at <path>; skipping capability grant`
2. `setcap not available; ensure dcgm-exporter has cap_sys_admin`
3. `dcgm-exporter capability self-check failed; profiling metrics may be unavailable`
4. `unable to set cap_sys_admin on <path>; start container with --cap-add SYS_ADMIN`
5. `nv-hostengine already running (pids: <list>)`
6. `nv-hostengine failed to start; dcgm metrics may be degraded`
7. `node_exporter failed to start`
8. `dcgm-exporter failed to start`

Any new warning should be added to this list and documented before Slice-Bench
relies on it.

### Expected Exporter Log Messages

You will see a few informational/warning lines in the container log that do not
map to `warnings` in the manifest:

- node_exporter reports “running as root” and a udev permissions warning because
  init-run.sh starts it as root inside the container; collectors remain active.
- dcgm-exporter logs “Not collecting CPU/NvSwitch metrics …” when the GH200 DCGM
  module lacks those entities. The metrics listed are simply skipped—no blank
  series are emitted.

These lines are purely informational and do not indicate exporter failure. If an
exporter actually fails to start, the manifest will contain a `warnings` entry
and `telemetry_surfaces.*.status` will reflect `failed`.

## Typical Workflow (Host)

1. `./scripts/start_observable_container.sh`
2. Capture `CONTAINER_RUN_META_JSON_HOST=` from stdout and load the manifest.
3. Launch the SGLang server (and Slice-Bench workload) using paths from the
   manifest.
4. After workloads finish, call `./scripts/stop_observable_container.sh`.
5. Inspect logs / metrics / traces under `.devcontainer/storage/*` as needed.

## Notes

- Multiple benchmarks may reuse the same container run; record the run ID in
  benchmark artefacts for traceability.
- The start script does not enforce warning policies—it simply surfaces them.
  Benchmark automation should decide whether to proceed.
- Existing changes in `.devcontainer/Dockerfile.gh200` and
  `.devcontainer/setup-storage.sh` are part of ongoing development and remain
  uncommitted upstream.
