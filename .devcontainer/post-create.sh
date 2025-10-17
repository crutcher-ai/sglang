#!/usr/bin/env bash
set -euo pipefail

cd /sgl-workspace/sglang

export PY_EXTRAS="${PY_EXTRAS:-tracing}"
export TMS_REF="${TMS_REF:-master}"
export TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.0}"

exec /sgl-workspace/sglang/.devcontainer/tools/run_dev_setup.sh
