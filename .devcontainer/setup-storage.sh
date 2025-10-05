#!/usr/bin/env bash
set -euo pipefail

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
    "$storage_root/logs" \
    "$storage_root/prometheus" \
    "$storage_root/jaeger" \
    "$storage_root/container_runs"

meta_file="$storage_root/container_run_meta.env"
if [ ! -f "$meta_file" ]; then
    touch "$meta_file"
fi

if command -v id >/dev/null 2>&1; then
    uid=$(id -u)
    gid=$(id -g)
    if ! chown -R "$uid":"$gid" "$storage_root" 2>/dev/null; then
        echo "Warn: unable to update ownership for some entries under $storage_root (try sudo?)" >&2
    fi
fi

chmod 600 "$meta_file"

echo "Devcontainer storage prepared under $storage_root"
