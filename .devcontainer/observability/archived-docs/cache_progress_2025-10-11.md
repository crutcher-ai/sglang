# Cache & Telemetry Progress – 2025-10-11

## Repository & Config
- Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 downloaded to `~/sglang-observability/models/Qwen/...` and set as the devcontainer default (`.devcontainer/tools/sglang-config.json`).
- Server limits reduced to 32K tokens (context, prefill, total) to keep warm-ups within model constraints.
- FlashInfer warm-up timeout raised to 600 s in `.devcontainer/tools/caching-config.json`.

## FlashInfer Cache
- Initial ensure run (`container-run-20251011T031450Z-761abc61`) built 51 FlashInfer artifacts (~112 MB) under `/profiles/flashinfer` with correct host ownership.
- Added signature support in `.devcontainer/tools/sgl_admin.py`:
  - Successful builds now write `/profiles/flashinfer/signature.json` describing model slug, kv cache dtype, context length, TP, and GPU runtime.
  - Subsequent `--flashinfer ensure` calls compare against the signature and return `noop` when caches match (confirmed with run `container-run-20251011T035413Z-5303cb9c`).
- Rebuild run (`container-run-20251011T040545Z-00668570`) captured Prometheus evidence that `up{job="sglang"}` flipped to `1` while the warm-up server was live (`query_range` samples at UNIX timestamps 1760156002–1760156006). Raw response saved from `curl -g 'http://127.0.0.1:9090/api/v1/query_range?...'`.
  - `curl -g "http://127.0.0.1:9090/api/v1/query_range?query=up{job=\"sglang\"}&start=2025-10-11T04:12:40Z&end=2025-10-11T04:13:35Z&step=1s"` returned a 1 for `localhost:30000` during `2025-10-11T04:13:22Z` ±3 s, confirming Prometheus observed the warm-up server before it shut down.

## Telemetry & Ownership Checks
- `/home/ubuntu/sglang-observability/profiles/flashinfer` stays `ubuntu:ubuntu`; server writes as `devuser` thanks to UID/GID remapping and the symlinked home cache.
- Prep manifests under `~/sglang-observability/telemetry/container_runs/…/prep_result.json` record FlashInfer status (`ok` or `noop`) and artifact counts. Telemetry probe currently remains best-effort because the warm-up server exits immediately after the metrics pulse.

## TorchInductor Cache
- Increased `inductor.warmup_timeout_s` to 900 s to allow Triton compilation to finish for the 80 B model (`.devcontainer/tools/caching-config.json`).
- Fresh container run (`container-run-20251011T042650Z-c2c3fa6a`) with `--inductor ensure` completed in ~274 s, producing TorchInductor artifacts under `/profiles/torchinductor` (host ownership `ubuntu:ubuntu`). Stage JSON confirms `status: "ok"` with cleanup metadata.

## DeepGEMM Cache
- `--deep-gemm rebuild` on run `container-run-20251011T140321Z-6694089e` produced `/profiles/deep_gemm/Qwen3-Next-80B-A3B-Thinking-FP8/compile.log` and a matching `signature.json` (context length 32 K, kv cache dtype fp8_e4m3). Run duration ~418 s.
- A follow-up `--deep-gemm ensure` in the same container returned `status: "noop"`, confirming the signature check short-circuits rebuilds without touching the cache.

## MoE Cache
- Editable install/hooks fixed the earlier helper mismatch; recent runs inside `container-run-20251011T172136Z-db5c775d` succeeded with `moe_tune.status == "ok"`.
- Sequential ensures added the requested batch sizes into `/profiles/moe_configs/configs/triton_3_4_0/E=512,N=512,device_name=NVIDIA_GH200_480GB,dtype=fp8_w8a8,block_shape=[128, 128].json`, which now contains `[2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 4096]`.
- `stages/moe_tune.json` captures each pass, including the new batch additions and duration estimates (~16–45 minutes depending on size).

## Outstanding Work
1. Run the inference smoke test against the pre-built caches, confirm telemetry probe success, and document Prometheus scrape behaviour for a long-lived server.
