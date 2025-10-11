# SGLang Cache Admin Tools

This directory houses the cache admin CLI and configuration used during
development to populate and inspect Triton/FlashInfer/TorchInductor/MoE caches
inside the helper container.

Contents
- `sgl_admin.py` — Typer-based CLI that orchestrates cache warm-ups and MoE
  tuning. It writes per-stage JSON under the current run and aggregates a
  `prep_result.json` for convenience.
- `sglang-config.json` — Defaults for server/model parameters (e.g.
  `kv_cache_dtype`, `context_length`, `mem_fraction_static`, and default
  model path). The CLI reads this at runtime.
- `caching-config.json` — Timeouts and policy knobs for warm-ups and locks.

How to run (inside the container)
```
docker exec -u devuser sglang-dev bash -lc "python /workspaces/sglang/.devcontainer/tools/sgl_admin.py --help"
```

Examples
- Inspect caches:
```
docker exec -u devuser sglang-dev bash -lc \
  "python /workspaces/sglang/.devcontainer/tools/sgl_admin.py caches inspect"
```

- Prepare only MoE configs for a single batch (Qwen3‑Next‑80B‑A3B FP8, TP=1):
```
docker exec -u devuser sglang-dev bash -lc \
  "python /workspaces/sglang/.devcontainer/tools/sgl_admin.py \
     caches ensure --tp 1 --moe ensure --moe-batch-sizes 512 \
     --flashinfer skip --inductor skip --deep-gemm skip"
```

- Prepare multiple batches in one run:
```
docker exec -u devuser sglang-dev bash -lc \
  "python /workspaces/sglang/.devcontainer/tools/sgl_admin.py \
     caches ensure --tp 1 --moe ensure --moe-batch-sizes 8,16,32,512,4096 \
     --flashinfer skip --inductor skip --deep-gemm skip"
```

Outputs
- Stage JSON files: `<manifest_dir>/<CONTAINER_RUN_ID>/stages/*.json`
- Aggregate: `<manifest_dir>/<CONTAINER_RUN_ID>/prep_result.json`
- MoE configs: `/profiles/moe_configs/configs/triton_<ver>/E=...,N=...,dtype=...[,block_shape=...][,per_channel_quant=true].json`

Notes
- The CLI sets `PYTHONPATH=/workspaces/sglang/python` for subprocesses and
  ensures cache directories exist and are writable as `devuser`.
- MoE helper was updated to accept `per_channel_quant` and will look for that
  filename variant when enabled; it falls back to the legacy name when absent.
- The default model path is read from `sglang-config.json`. Override with
  `--model <path>` if needed.

