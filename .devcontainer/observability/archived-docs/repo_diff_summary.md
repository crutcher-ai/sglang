# Repository Delta vs. upstream `main`

This note captures every tracked change we currently carry relative to upstream `origin/main`. It is intended as the reset point for re‑explaining the architecture and deciding what to keep, rework, or drop.

## Devcontainer & Observability Stack

- **.devcontainer/Dockerfile (removed)** – upstream’s generic container image is deleted.
- **.devcontainer/Dockerfile.gh200 (new)** – builds a GH200‑specific devcontainer image. Key additions:
  - Creates a `devuser` account with sudo, installs tooling (oh-my-zsh, uv, rustup, typer, ray).
  - Installs observability binaries (Prometheus 3.5.0, node_exporter 1.9.1, Jaeger 1.73.0) and NVIDIA DCGM exporters.
  - Sets up `/profiles` and `/telemetry` mounts, symlinks FlashInfer cache into `$HOME`.
- **.devcontainer/devcontainer.json (modified)** – switches to the new Dockerfile, binds host storage for models/caches/telemetry, runs as `devuser`, and wires post-create hooks.
- **.devcontainer/README.md (new)** – documents the devcontainer layout and observability mounts.
- **.devcontainer/post-create.sh (new)** – bootstraps the workspace after VS Code creates the container (permissions, uv sync, etc.).
- **.devcontainer/setup-storage.sh (new)** – helper to pre-create host-side storage trees for models, telemetry, and caches.
- **.devcontainer/observability/** (new directory)
  - `init-run.sh` – container entrypoint that provisions per-run telemetry directories, launches Prometheus/Jaeger/node_exporter/dcgm-exporter, writes run manifests, and manages pointer files.
  - `metrics_catalog.final.md` – curated list of metrics exposed by the observability stack.
  - `prometheus.yml.tmpl` – templated scrape config consumed by `init-run.sh`.
- **.gitignore** – adds `.devcontainer/storage/` to keep bind-mount directories out of git.

## Caching & Admin Tooling

- **python/sglang/compile_deep_gemm.py (rewritten)**
  - Replaces the old `ServerArgs`/multiprocessing flow with a direct CLI invocation of `sglang.launch_server`.
  - Accepts explicit CLI flags for cache-related parameters (`--kv-cache-dtype`, `--mem-fraction-static`, etc.) and forwards them to the server.
  - Forwards `--trust-remote-code/--no-trust-remote-code` based on arguments.
  - Adds readiness polling that exits when the subprocess dies and ensures the child process is terminated in `finally`.
  - Prints success once the warm-up request returns but otherwise streams server stdout/stderr to the parent (no longer piped to `/dev/null`).

  > **Known gaps:** still uses sglang defaults for context length (272k) unless callers override; readiness loop only exits early if `proc.poll()` is non-`None` (added in our version but needs upstream verification).

- **.devcontainer/tools/** – currently untracked in git but contains the new `sgl_admin.py`, `sglang-config.json`, `caching-config.json` we’ve been editing. (Not part of the official diff yet but worth flagging if we plan to keep them.)

## Kernel / MoE Config Support

- **python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py**
  - `get_config_file_name` now accepts `block_shape: List[int]` and a `per_channel_quant` flag, appending `,per_channel_quant=1` to generated filenames when appropriate. This allows tuner outputs to differentiate per-channel quant configs on disk.

## Container Control Scripts

- **scripts/start_observable_container.sh (new)** – host-side launcher that:
  - Validates / prepares host directories (`~/sglang-observability/…`).
  - Runs `docker run` with GPU, telemetry, and storage mounts.
  - Waits for `/telemetry/container_run_meta.env` to be populated and prints the resolved manifest paths.
- **scripts/stop_observable_container.sh (new)** – simple helper to stop/remove the running `sglang-dev` container.

## Documentation Additions

- **.devcontainer/README.md** – overview of the devcontainer and observability expectations.
- **.devcontainer/observability/metrics_catalog.final.md** – enumerates metric names we intend to scrape (Prometheus + DCGM).
- **docs/provider_caching_refactor_review.md** (new, untracked) – prior findings from the caching refactor review.
- **docs/repo_diff_summary.md** – this document.

## Removed / Replaced Assets

- Original `.devcontainer/Dockerfile` (generic image) – deleted in favor of the GH200-specific build.

## Summary Table

| Area | Files | Purpose |
| --- | --- | --- |
| Devcontainer image | `.devcontainer/Dockerfile.gh200`, `.devcontainer/devcontainer.json`, `.devcontainer/post-create.sh`, `.devcontainer/setup-storage.sh`, `.devcontainer/observability/*` | Custom GH200 devcontainer with baked-in observability stack and storage mounts |
| Observability docs | `.devcontainer/README.md`, `.devcontainer/observability/metrics_catalog.final.md`, `.devcontainer/observability/prometheus.yml.tmpl` | Document and template telemetry stack |
| Caching helper | `python/sglang/compile_deep_gemm.py` | New subprocess-based DeepGEMM compile script |
| MoE config naming | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` | Include block shape list + per-channel quant marker in filenames |
| Container lifecycle | `scripts/start_observable_container.sh`, `scripts/stop_observable_container.sh` | Host wrappers to manage the devcontainer |
| Git hygiene | `.gitignore` | Ignore persistent storage mounts |

## Untracked but Present (consider committing or discarding)

- `.devcontainer/tools/` – houses the rewritten `sgl_admin.py`, `sglang-config.json`, `caching-config.json`. Currently excluded from git; decide whether they become part of the official diff.
- `docs/provider_caching_refactor_review.md`, `docs/provider_prep_engineers_notebook.md`, `scripts/cache/`, `test/args|e2e|observability|prep/` – all untracked directories introduced during recent work.

---

This reflects the repository state as of `git status` today. Let me know if you want a deeper diff for any file or a trimmed set to keep when you rebuild from scratch.
