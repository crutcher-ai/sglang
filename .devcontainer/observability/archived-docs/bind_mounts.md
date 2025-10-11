# Bind Mount Layout

| Host path (default root: `$HOME/sglang-observability`) | Container path | Contents |
| --- | --- | --- |
| `.../telemetry` | `/telemetry` | Prometheus TSDB, Jaeger data, run manifests, logs |
| `.../profiles` | `/profiles` | Kernel caches: DeepGEMM, FlashInfer, TorchInductor, Triton, MoE configs, lock & in-progress markers |
| `.../models` | `/models` | Downloaded model snapshots used by SGLang |
| `.../huggingface` *(optional)* | `/home/devuser/.cache/huggingface` | HF cache (can be removed if unused) |
| repo root | `/workspaces/sglang` | Source tree |

- `start_observable_container.sh` ensures all required host directories exist before launching the container. It can optionally skip the ownership checks via `HOST_DIR_OWNERSHIP_IGNORE=1`.
- The devcontainer variant in `.devcontainer/devcontainer.json` mirrors the same mounts when opened via VS Code.
- Removing the Hugging Face cache mount is safe if the repo doesn’t need a shared cache between runs.
