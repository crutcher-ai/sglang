# Single-Step Cutover: One Mount, One Venv, No PYTHONPATH

Status: draft-for-implementation (no phases, no shims)

Scope: Only runtime/helper code used by the observability helper and server launch. Tests/docs/bench and upstream Dockerfiles are unchanged in this pass.

Outcome in one commit:
- Bind repo to `/sgl-workspace/sglang` (upstream path).
- Use a single, isolated venv at `/sgl-workspace/sglang/.venv` for all Python.
- Remove all PYTHONPATH/.pth/sitecustomize tricks; rely on editable install into `.venv`.
- Preinstall CUDA torch wheels (cu129) into `.venv`; then install sglang and explicit pins.

## Files To Edit (exact and minimal)

1) `scripts/start_observable_container.sh`
- Change three strings:
  - `INIT_HOOK="/sgl-workspace/sglang/.devcontainer/post-create.sh"`
  - Repo bind mount: `-v "${ROOT_DIR}:/sgl-workspace/sglang"`
  - Working dir: `-w /sgl-workspace/sglang`

2) `.devcontainer/tools/run_dev_setup.sh`
- Set venv path: `VENV_PATH="/sgl-workspace/sglang/.venv"`
- Create venv without system site packages:
  - `uv venv --python python3 --allow-existing "$VENV_PATH"`
- Remove:
  - Any `export PYTHONPATH=...`
  - Any code writing `.pth` or `sitecustomize.py`
- Detect CUDA tag from container system torch and preinstall wheels into `.venv`:
  - `CUDA_TAG=$(/usr/bin/python3 - <<'PY'\nimport torch;print((torch.version.cuda or '').replace('.',''))\nPY
)`
  - `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu${CUDA_TAG}` (expect `129`)
  - `uv pip install --python "$VENV_PYTHON" --index-url "$TORCH_INDEX_URL" --extra-index-url https://pypi.org/simple "torch==2.8.0" "torchvision==0.19.0" "torchaudio==2.8.0"`
- Install sglang editably and explicit pins:
  - `TARGET="/sgl-workspace/sglang/python"` (append extras if needed)
  - `uv pip install --python "$VENV_PYTHON" --upgrade --editable "$TARGET"`
  - `uv pip install --python "$VENV_PYTHON" --quiet --upgrade "transformers==${TRANSFORMERS_VERSION:-4.57.0}"`

3) `.devcontainer/observability/init-run.sh`
- Import validation must use `.venv` interpreter:
  - `as_devuser /sgl-workspace/sglang/.venv/bin/python -c 'import sglang'`

4) `scripts/infer/start_server.sh`
- Default interpreter:
  - `PYTHON_BIN="${PYTHON_BIN:-/sgl-workspace/sglang/.venv/bin/python}"`
- Remove all `-e PYTHONPATH=...` envs in docker exec calls.
- Replace `/workspaces/sglang` path literals with `/sgl-workspace/sglang`.
- Add a preflight before launch (inside container, using `$PYTHON_BIN`) that asserts:
  - `torch.cuda.is_available()` is True
  - `getattr(torch.version, 'cuda', None) == '12.9'`
  - `import sgl_kernel, sgl_kernel.common_ops` succeeds
  - On failure: print one-line remediation (the `uv pip install --index-url ... cu129` command) and exit non-zero.
- On readiness timeout: tail the last ~200 lines of the current run log (path is in the manifest) and exit non-zero.

That’s it. No other files or defaults are changed.

## Cutover Checklist (run in order)
1) Stop old helper: `docker rm -f sglang-dev jaeger-v2 || true`
2) Apply the edits above.
3) Start helper: `./scripts/start_observable_container.sh`
4) Verify `.venv`:
   - `docker exec -u devuser sglang-dev /sgl-workspace/sglang/.venv/bin/python -c 'import torch;print(torch.__version__, torch.version.cuda, __import__("torch").cuda.is_available())'` → `2.8.0+cu129 12.9 True`
   - `docker exec -u devuser sglang-dev /sgl-workspace/sglang/.venv/bin/python -c 'import sglang, sgl_kernel.common_ops; print("ok")'` → `ok`
5) Launch server: `READY_TIMEOUT=600 ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on SGLANG_DEBUG=1 ./scripts/infer/start_server.sh`
6) Confirm ready:
   - `./scripts/infer/status.sh` → `ready`
   - `curl -s http://127.0.0.1:30000/trace_probe` shows `tracing_enabled: true`
   - `curl -s http://127.0.0.1:16686/api/v3/services` lists `sglang` after you send one request

## Rollback (single step)
- Revert this commit. The helper returns to `/workspaces/sglang` + old venv behavior.

## Notes
- We intentionally do not change tests/docs/bench or VS Code settings here. The runtime helper is the only target of this cutover.
