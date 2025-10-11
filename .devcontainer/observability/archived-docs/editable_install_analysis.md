# Editable Install Regression Analysis (2025-10-11)

## Context

We manage the GH200 devcontainer via `scripts/start_observable_container.sh`, which launches the custom image `sglang-dev:gh200` by calling `.devcontainer/observability/init-run.sh` directly. This flow skips VS Code’s post-create hook (`.devcontainer/post-create.sh`), so the base image’s preinstalled SGLang wheel under `/sgl-workspace` remains active. Without an editable re-install, Python inside the container continues to import from that wheel, ignoring any changes made in the bind-mounted repository.

## Symptoms

- MoE tuner exits with `TypeError: get_config_file_name() takes from 3 to 4 positional arguments but 5 were given`. Our repo version of `get_config_file_name` accepts `(…, per_channel_quant=False)`; the wheel version does not. The TypeError indicates we are still importing the wheel.
- `python -c "import sglang; print(sglang.__file__)"` inside the container points to `/sgl-workspace/…`, confirming imports come from the packaged copy.

## Root Cause

Editable install never happens when launching via the helper scripts: the base image contains `pip install sglang` from build time, and because `.devcontainer/post-create.sh` is not executed, the wheel is never overridden by an editable install pointing at `/workspaces/sglang/python`.

## Constraints

1. Ensure the container always imports from the bind-mounted repo, not the baked wheel.
2. Avoid mutating `/sgl-workspace` directly; keep the workspace mount as the single source of truth.
3. Keep ownership consistent (devuser) and avoid root-owned artifacts under `/profiles` or site-packages.
4. Changing SGLang (and later Triton) should require no manual reinstall steps.

## Remediation Plan (Revised)

1. **Run the post-create logic on every container start, as devuser, with an editable install that includes extras.**
   - In the host launcher, pass `INIT_RUN_HOOK=/workspaces/sglang/.devcontainer/post-create.sh` when invoking `docker run`. `init-run.sh` now executes the hook via `as_devuser bash -lc`, so the script runs as devuser without nested `sudo` calls.
   - Update `.devcontainer/post-create.sh` so that the editable branch handles extras explicitly, e.g.:
     ```bash
     ${PYTHON_BIN} -m pip install -e /workspaces/sglang/python
     if [[ -n "${EXTRAS}" ]]; then
         ${PYTHON_BIN} -m pip install "/workspaces/sglang/python[${EXTRAS}]"
     fi
     ```
     This replaces the previous non-editable `pip install "./python[${EXTRAS}]"` path and guarantees the egg-link points to the repo.

2. **Set import precedence explicitly.**
   - Export `PYTHONPATH=/workspaces/sglang/python` (not `/workspaces/sglang`) both globally in `init-run.sh` and inside `_prepare_env()` in `.devcontainer/tools/sgl_admin.py`. This ensures any subprocess or ad-hoc shell resolves modules from the workspace even if the editable install is skipped.

3. **Validation after restart.**
   - After starting the container with the hook (inside the container or via `docker exec -i -u devuser sglang-dev python - <<'PY' ...`):
     ```bash
     python - <<'PY'
     from inspect import signature
     import sglang
     from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import get_config_file_name
     print("sglang file:", sglang.__file__)
     print("get_config_file_name signature:", signature(get_config_file_name))
     PY
     ```
     Expect `sglang file` under `/workspaces/sglang/python/...` and signature `(E: int, N: int, dtype: Optional[str], block_shape: Optional[List[int]] = None, per_channel_quant: bool = False) -> str`.

4. **Re-run the MoE tuner (512, 4096, batch lists) only after validation.**
   - The previous TypeError should disappear; the tuner will write configs under `~/sglang-observability/profiles/moe_configs/...`.

## Alternatives Considered (Rejected)

- Manual `pip install -e` after each start: error-prone and easy to forget.
- Copying repo files into `/sgl-workspace`: hides the problem, causes drift, and complicates future rebuilds.
- Rebuilding the base image without a wheel: heavier change; we still need an editable install for live development regardless.

## Next Steps (once plan approved)

1. Patch `.devcontainer/post-create.sh` to ensure the editable install always occurs with extras.
2. Modify `scripts/start_observable_container.sh` to set `INIT_RUN_HOOK=/workspaces/sglang/.devcontainer/post-create.sh` before `docker run` so the entrypoint executes the hook as devuser.
3. Adjust `_prepare_env()` (and any other references) to export `PYTHONPATH=/workspaces/sglang/python`.
4. Restart the container, run the validation snippet, and only then retry MoE tuning.

## Notes

- Running `pip install -e` on every start is typically quick; if start time becomes an issue, we can add a guard that skips install when the existing egg-link already points to `/workspaces/sglang/python`.
- If future images remove the preinstalled wheel, the editable install still provides the desired behavior.
- Keep the documentation (`docs/cache_progress_2025-10-11.md`) updated so others know the launcher relies on `INIT_RUN_HOOK` to maintain the editable install.
