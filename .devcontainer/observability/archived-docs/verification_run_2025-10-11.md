# Container lifecycle verification (2025-10-11)

## Mount roots on host (after `scripts/start_observable_container.sh`)

```
/home/ubuntu/sglang-observability           ubuntu:ubuntu  drwxrwxr-x
├── models                                   ubuntu:ubuntu  drwxrwxr-x
│   └── Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 (downloaded snapshot)
├── profiles                                 ubuntu:ubuntu  drwxrwxr-x
│   ├── .in_progress                         ubuntu:ubuntu  drwxrwxr-x
│   ├── .locks                               ubuntu:ubuntu  drwxrwxr-x
│   ├── deep_gemm                            ubuntu:ubuntu  drwxrwxr-x
│   ├── flashinfer                           ubuntu:ubuntu  drwxrwxr-x
│   ├── moe_configs/                         ubuntu:ubuntu  drwxrwxr-x
│   ├── torchinductor                        ubuntu:ubuntu  drwxrwxr-x
│   └── triton                               ubuntu:ubuntu  drwxrwxr-x
└── telemetry                                ubuntu:ubuntu  drwxr-xr-x
    ├── container_run_meta.env               ubuntu:ubuntu  -rw-------
    ├── container_runs/                      ubuntu:ubuntu  drwxrwxr-x
    │   └── container-run-20251011T023409Z-541fb327.json (ubuntu:ubuntu)
    ├── jaeger/                              ubuntu:ubuntu  drwxrwxr-x
    │   └── container-run-20251011T023409Z-541fb327
    ├── logs/                                ubuntu:ubuntu  drwxrwxr-x
    │   └── container-run-20251011T023409Z-541fb327.log (devuser:devuser)
    └── prometheus/                          ubuntu:ubuntu  drwxrwxr-x
        ├── container-run-20251011T023409Z-541fb327
        └── prometheus.yml (devuser:devuser)
```

All directories were created by the start script and owned by the invoking user; the manifest JSON is now written as `ubuntu:ubuntu` and the pointer file is mode `600`.

## In-container view (`docker exec -u devuser sglang-dev ...`)

- `/models` → bind mount of host models tree (`devuser:devuser`).
- `/profiles` → bind mount with stage subdirectories pre-created (`devuser:devuser`).
- `/telemetry` → bind mount containing manifest, logs, Prometheus and Jaeger roots.

`/telemetry/container_run_meta.env` points to the container manifest (owned by devuser). On stop the pointer is removed.

## Run manifest excerpt

```
{
  "container_run_id": "container-run-20251011T013326Z-a3cd640b",
  "started_at": "2025-10-11T01:33:26Z",
  "image": "sglang-dev:gh200",
  "services": {"prometheus": {"port": 9090}, ...},
  "storage": {
    "log_file": "/telemetry/logs/container-run-20251011T013326Z-a3cd640b.log",
    "prometheus_dir": "/telemetry/prometheus/container-run-20251011T013326Z-a3cd640b",
    "jaeger_dir": "/telemetry/jaeger/container-run-20251011T013326Z-a3cd640b"
  }
}
```

## Notes

- Startup time (container to manifest) ≈ 38 s.
- Helper start script recreates directories as the invoking user; no manual `chown` required.
- Pointer file is removed during `scripts/stop_observable_container.sh` to avoid stale-run races.
- No cache artifacts yet; `/profiles` remains empty aside from scaffolding.
