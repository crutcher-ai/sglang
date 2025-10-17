#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/sgl-workspace/sglang}
PYTHON_SRC="${REPO_ROOT}/python"
HTTP_SERVER_FILE="${PYTHON_SRC}/sglang/srt/entrypoints/http_server.py"
ARCH=$(uname -m)
PY_EXTRAS=${PY_EXTRAS:-tracing}
TMS_REF=${TMS_REF:-master}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-4.57.0}
MARKER="${RUN_ROOT:-}/.init_ok"
VENV_PATH="${REPO_ROOT}/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not found in PATH" >&2
  exit 1
fi

mkdir -p "$HOME/.cache/pip" "$HOME/.local/bin"
find "$PYTHON_SRC" -maxdepth 1 -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true

# Create a venv that can see system site packages (leverages container's CUDA torch)
uv venv --python python3 --system-site-packages --allow-existing "$VENV_PATH" >/dev/null
VENV_PYTHON="${VENV_PATH}/bin/python"

if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  uv pip install --python "$VENV_PYTHON" --quiet --upgrade "torch-memory-saver @ git+https://github.com/fzyzcjy/torch_memory_saver@${TMS_REF}" --no-deps
fi

# Ensure venv does not shadow system CUDA torch with CPU builds
uv pip uninstall --python "$VENV_PYTHON" -y torch torchvision torchaudio >/dev/null 2>&1 || true

# Do NOT install torch/torchvision/torchaudio into the venv: use the CUDA builds from system site-packages.

TARGET="$PYTHON_SRC"
if [[ -n "$PY_EXTRAS" ]]; then
  TARGET="${PYTHON_SRC}[${PY_EXTRAS}]"
fi

uv pip install --python "$VENV_PYTHON" --upgrade --no-deps --editable "$TARGET"
uv pip install --python "$VENV_PYTHON" --quiet --upgrade "transformers==${TRANSFORMERS_VERSION}"

ln -sf "${VENV_PATH}/bin/python" "$HOME/.local/bin/python"
ln -sf "${VENV_PATH}/bin/python" "$HOME/.local/bin/python3"
ln -sf "${VENV_PATH}/bin/pip" "$HOME/.local/bin/pip"

if ! grep -q '/trace_probe' "$HTTP_SERVER_FILE"; then
  echo "ERROR: /trace_probe route not present in $HTTP_SERVER_FILE" >&2
  exit 1
fi

if [[ -n "${MARKER}" ]]; then
  mkdir -p "$(dirname "${MARKER}")"
  echo "ok" > "${MARKER}"
fi
