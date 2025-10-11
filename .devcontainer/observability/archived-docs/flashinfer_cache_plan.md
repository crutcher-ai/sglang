# FlashInfer Cache Execution Plan (2025-10-11)

This note outlines the end-to-end plan for building, validating, and persisting the FlashInfer cache inside the SGLang
observability devcontainer. The goal is to establish why the workflow should succeed, enumerate the checks for each phase,
and highlight failure risks before executing anything.

## 1. Preconditions & Environment

- **Model availability** – `~/sglang-observability/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` is fully downloaded (77 GB) and
  mounted into the container at `/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`. `.devcontainer/tools/sglang-config.json`
  now sets this path as `model.default_model_path` so `sgl_admin` resolves it without extra flags.
- **Host mounts** – `scripts/start_observable_container.sh` creates `/profiles`, `/telemetry`, and subdirectories owned by the
  host user (`ubuntu`). Because the container’s `devuser` UID/GID is remapped to match the host, any writes made by
  `devuser` land with the correct ownership on the host mount.
- **Runtime entrypoint** – `.devcontainer/observability/init-run.sh` launches exporters as root, then runs the container payload
  (e.g., `sleep infinity` or our exec shells) via `as_devuser`. When we `docker exec -u devuser`, `sgl_admin` and the warm-up
  server processes stay under the host UID, so we expect FlashInfer artifacts to be created as `ubuntu:ubuntu`.
- **Config sanity** – Prior to running FlashInfer we will clamp the effective context parameters to a supported size (32 K for
  Qwen 80B) to avoid out-of-memory failures. The current JSON still lists 272 K, so we must update
  `server.{context_length,max_prefill_tokens,max_total_tokens}` and adjust downstream signatures before running the stage.

## 2. Theory of Operation

1. `sgl_admin caches ensure --flashinfer ensure --inductor skip --deep-gemm skip --moe skip` loads
   `.devcontainer/tools/sglang-config.json`, resolves the default model path, and prepares an environment via `_prepare_env()`:
   - Enforces `FLASHINFER_WORKSPACE_DIR=/profiles/flashinfer` and `FLASHINFER_JIT_LOG_DIR=/profiles/flashinfer/90a`.
   - Creates the workspace directory and a devuser-owned symlink from `~/.cache/flashinfer/90a` → `/profiles/flashinfer/90a`.
2. `_start_warmup_server()` launches `python -m sglang.launch_server` with `--enable-metrics`, port `30000`, and the cache
   directories injected via environment variables. Because the parent process is `devuser`, the server’s PID inherits
   devuser ownership; FlashInfer kernels loaded through the symlink should therefore write into `/profiles/flashinfer/...` on
   the host with user ownership intact.
3. The warm-up server waits for `/get_model_info`, then sends a deterministic generation request. This drives FlashInfer to
   populate JIT caches (kernels and logs) under `/profiles/flashinfer`. After a successful build, `sgl_admin` now writes
   `/profiles/flashinfer/signature.json` capturing the model slug, kv-cache dtype, context length, and GPU/runtime metadata.
   On subsequent `--flashinfer ensure` runs, the CLI compares this signature to the requested settings and returns `noop`
   without relaunching the server whenever the cache matches.
4. `_stop_server()` terminates the warm-up process and records cleanup metadata. `sgl_admin` writes `prep_result.json` and the
   per-stage JSON under `/telemetry/container_runs/<run>/stages/flashinfer.json`.

## 3. Validation Steps (before, during, after)

**Before running**
- Start a fresh container (`start_observable_container.sh`) and verify pointer + manifest as documented in
  `docs/verification_run_2025-10-11.md`.
- Inside the container (as devuser) confirm `/profiles/flashinfer` is empty apart from scaffolding and `.in_progress`.
- Update the server context parameters in `sglang-config.json` to 32 K (context/max_prefill/max_total) and commit the same
  values to the DeepGEMM signature logic when we reach that stage.

**During FlashInfer ensure**
- Run `sgl_admin caches ensure --flashinfer ensure --deep-gemm skip --inductor skip --moe skip` with `--tp 1` (GH200 single
  device). Expect console heartbeat lines plus `RESULT_STATUS ok flashinfer:ok ...`.
- Tail `/telemetry/logs/<run>.log` for:
  - FlashInfer JIT compilation lines (ensures kernels were emitted).
  - SGLang server startup, HTTP readiness, and warm-up request completion.
- Query Prometheus targets (`curl http://127.0.0.1:9090/api/v1/targets`) while the warm-up server is running to confirm the `sglang` scrape endpoint flips to `UP` (`up{job="sglang",instance="localhost:30000"} == 1`). Once the warm-up process exits, Prometheus reports the target as down, so capture the evidence before shutdown.

**After the run**
- Inspect `/profiles/flashinfer` for new files and ensure `.locks/flashinfer.lock` and `.in_progress/flashinfer.json` are gone.
- Read `prep_result.json` to verify `stages.flashinfer.status == "ok"`, artifacts include the cache directory, and note the `telemetry_probe` outcome (best-effort because the warm-up server terminates immediately after the metrics pulse).
- Persist evidence: `sgl_admin caches inspect` snapshot, Prometheus sample query output, and log excerpts.

**Persistence check**
- Stop the container (`scripts/stop_observable_container.sh`), leaving `/profiles/flashinfer` populated.
- Restart the container and run either `sgl_admin caches ensure --flashinfer ensure ...` or a smoke inference; Stage should
  detect the existing cache (status `noop`) or the server should show no JIT recompilation in the logs.
- Issue a simple prompt (`curl localhost:30000/generate`) and confirm Prometheus metrics still increment when using the cache.

## 4. Risk Assessment & Mitigations

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Context length 272 K exceeds Qwen 80B limits | Warm-up OOM or FlashInfer refusing to build | Update config to 32 K before running; ensure DeepGEMM signature uses same value later |
| FlashInfer still writes under `/root/.cache` | Host receives root-owned artifacts, breaking persistence | `_prepare_env()` symlinks `~/.cache/flashinfer/90a` to `/profiles/flashinfer/90a` and runs everything as devuser, so the only way this fails is if `sgl_admin` is invoked as root. We will run via `docker exec -u devuser` and verify ownership after warm-up |
| Warm-up port 30000 busy | `sgl_admin` aborts with `port_busy` | Ensure no lingering SGLang server before running; stop container or kill processes if needed |
| Prometheus scrape fails | `telemetry_probe.ok=false` | Validate exporters up before warm-up; inspect `/telemetry/prometheus/prometheus.yml` to confirm job entries; collect Prom log if needed |

## 5. Exit Criteria

1. `prep_result.json` records `flashinfer.status = ok`, `telemetry_probe.ok = true`, and the run `status = ok`.
2. `/profiles/flashinfer` contains the new cache artifacts owned by `ubuntu:ubuntu`.
3. Prometheus TSDB shows samples for `sglang:prompt_tokens_total` bounded to the run ID.
4. Restarted container reuses the cache without creating root-owned files or forcing recompilation.
5. Documentation updated with findings and any deviations from expectations.

Once these criteria are satisfied we will proceed to the TorchInductor stage, keeping the same methodology for cache resets,
telemetry verification, and persistence checks.
