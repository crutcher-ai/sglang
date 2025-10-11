#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/sglang

PYTHON_BIN="python"
ARCH="$(uname -m)"
EXTRAS="${PY_EXTRAS:-tracing}"
TMS_REF="${TMS_REF:-master}"

${PYTHON_BIN} -m pip install -U pip setuptools wheel

# On ARM64 the 0.0.9rc1 sdist published to PyPI is missing headers.
# Proactively install from Git so the build pulls in the complete sources.
if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    ${PYTHON_BIN} -m pip install "torch-memory-saver @ git+https://github.com/fzyzcjy/torch_memory_saver@${TMS_REF}" --upgrade --no-deps
fi

# Install the requested optional dependencies (defaults to [tracing]).
if [[ -n "${EXTRAS}" ]]; then
    ${PYTHON_BIN} -m pip install -e "/workspaces/sglang/python[${EXTRAS}]"
else
    ${PYTHON_BIN} -m pip install -e /workspaces/sglang/python
fi

# Quick sanity check so failures surface during container creation.
${PYTHON_BIN} - <<'PY'
import sys

try:
    import torch_memory_saver as tms
except Exception as exc:  # pragma: no cover
    print("torch_memory_saver import failed:", exc)
    sys.exit(1)

print("torch_memory_saver version:", getattr(tms, "__version__", "unknown"))
print("has configure_subprocess:", hasattr(tms, "configure_subprocess"))
PY
