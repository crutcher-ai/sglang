#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/sglang

PYTHON_BIN="python"
ARCH="$(uname -m)"
EXTRAS="${PY_EXTRAS:-tracing}"
TMS_REF="${TMS_REF:-master}"

# Pre-create pip cache for devuser and ensure ownership to avoid permission warnings
mkdir -p /home/devuser/.cache/pip || true
chown -R devuser:devuser /home/devuser/.cache || true

# Upgrade pip tooling as devuser to avoid root-owned artifacts
sudo -u devuser -E ${PYTHON_BIN} -m pip install -U pip setuptools wheel

# On ARM64 the 0.0.9rc1 sdist published to PyPI is missing headers.
# Proactively install from Git so the build pulls in the complete sources.
if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    sudo -u devuser -E ${PYTHON_BIN} -m pip install "torch-memory-saver @ git+https://github.com/fzyzcjy/torch_memory_saver@${TMS_REF}" --upgrade --no-deps
fi

# Install the requested optional dependencies (defaults to [tracing]).
if [[ -n "${EXTRAS}" ]]; then
    sudo -u devuser -E ${PYTHON_BIN} -m pip install -e "/workspaces/sglang/python[${EXTRAS}]"
else
    sudo -u devuser -E ${PYTHON_BIN} -m pip install -e /workspaces/sglang/python
fi

# Quick sanity check so failures surface during container creation.
sudo -u devuser -E ${PYTHON_BIN} - <<'PY'
import sys

try:
    import torch_memory_saver as tms
except Exception as exc:  # pragma: no cover
    print("torch_memory_saver import failed:", exc)
    sys.exit(1)

print("torch_memory_saver version:", getattr(tms, "__version__", "unknown"))
print("has configure_subprocess:", hasattr(tms, "configure_subprocess"))
PY
