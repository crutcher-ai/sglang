#!/usr/bin/env bash
set -euo pipefail

# This script works from any directory; resolve paths relative to repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."
storage_root="./.devcontainer/storage"

mkdir -p \
    "$storage_root/models" \
    "$storage_root/huggingface" \
    "$storage_root/profiles" \
    "$storage_root/profiles/deep_gemm" \
    "$storage_root/profiles/moe_configs/configs" \
    "$storage_root/profiles/triton" \
    "$storage_root/profiles/torchinductor" \
    "$storage_root/profiles/flashinfer" \
    "$storage_root/logs/server" \
    "$storage_root/logs/benchmark" \
    "$storage_root/runs" \
    "$storage_root/prometheus" \
    "$storage_root/jaeger/badger" \
    "$storage_root/jaeger/badger2"

# Ensure host user owns the directories so they map cleanly into the container.
if command -v id >/dev/null 2>&1; then
    uid=$(id -u)
    gid=$(id -g)
    chown -R "$uid":"$gid" "$storage_root"
fi

# Prometheus/Jaeger containers run as unprivileged users; relax perms for their stores.
chmod 777 "$storage_root/prometheus" "$storage_root/jaeger/badger" "$storage_root/jaeger/badger2"

echo "Devcontainer storage prepared under $storage_root"
