#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/sglang

# Install tracing extras if they are not already present
if ! python -m pip show opentelemetry-sdk >/dev/null 2>&1; then
    python -m pip install ./python[tracing]
fi
