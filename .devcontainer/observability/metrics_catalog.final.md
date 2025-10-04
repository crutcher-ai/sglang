# Observability Metrics Catalog — GH200 Cloud Baseline (Finalized)

This document describes the telemetry that is actually available in the current
SGLang devcontainer for GH200 cloud VMs. It is derived from the v3 catalog for
SGLang/node_exporter, the cleaned v4 DCGM set, and the practical limitations we
encountered (e.g., Hopper C2C PMUs unavailable, Nsight tooling not bundled).
Use this as the reference when building dashboards, alerts, or benchmark
reporting.

## 1. SGLang Server Metrics

Emitted by `python/sglang/srt/metrics/collector.py`. These metrics exist today
and require no extra flags beyond `--enable-metrics` (already used by the
server launcher).

### Queueing & Scheduler Gauges
| Metric | Type | Description |
|--------|------|-------------|
| `sglang:num_queue_reqs` | gauge | Requests in the waiting queue. |
| `sglang:num_running_reqs` | gauge | Active requests. |
| `sglang:num_running_reqs_offline_batch` | gauge | Low-priority offline requests. |
| `sglang:num_prefill_prealloc_queue_reqs` | gauge | Prefill prealloc queue depth. |
| `sglang:num_prefill_inflight_queue_reqs` | gauge | Prefill inflight queue depth. |
| `sglang:num_decode_prealloc_queue_reqs` | gauge | Decode prealloc queue depth. |
| `sglang:num_decode_transfer_queue_reqs` | gauge | Decode transfer queue depth. |
| `sglang:num_paused_reqs` | gauge | Requests paused by async weight sync. |
| `sglang:num_retracted_reqs` | gauge | Requests retracted due to resource limits. |
| `sglang:num_grammar_queue_reqs` | gauge | Grammar compilation queue depth. |
| `sglang:num_used_tokens` | gauge | Tokens currently held (KV usage proxy). |
| `sglang:token_usage` | gauge | Scheduler-computed token usage ratio. |
| `sglang:swa_token_usage` | gauge | SWA token usage (hybrid cache). |
| `sglang:cache_hit_rate` | gauge | Prefix cache hit rate (gauge form). |
| `sglang:kv_transfer_speed_gb_s` | gauge | KV transfer speed (GB/s). |
| `sglang:kv_transfer_latency_ms` | gauge | KV transfer latency (ms). |
| `sglang:max_running_requests_under_SLO` | gauge | Headroom under SLO (if provided). |

### Token & Request Counters / Histograms
| Metric | Type | Description |
|--------|------|-------------|
| `sglang:generation_tokens_total` | counter | Generated tokens. |
| `sglang:prompt_tokens_total` | counter | Prefill tokens. |
| `sglang:cached_tokens_total` | counter | Prompt tokens served from cache. |
| `sglang:num_requests_total` | counter | Completed requests. |
| `sglang:num_aborted_requests_total` | counter | Aborted requests. |
| `sglang:num_bootstrap_failed_reqs_total` | counter | Bootstrap failures. |
| `sglang:num_transfer_failed_reqs_total` | counter | Transfer failures. |
| `sglang:time_to_first_token_seconds` | histogram | TTFT distribution. |
| `sglang:e2e_request_latency_seconds` | histogram | End-to-end latency. |
| `sglang:request_latency_seconds` | histogram | Latency per stage. |
| `sglang:per_stage_req_latency_seconds{stage}` | histogram | Stage-level latency (prefill/decode/etc.). |
| `sglang:queue_time_seconds` | histogram | Queue wait duration. |
| `sglang:inter_token_latency_seconds` | histogram | Inter-token latency. |
| `sglang:grammar_compilation_time_seconds` | histogram | Grammar compilation duration. |
| `sglang:grammar_tree_traversal_time_avg/max` | histogram | Grammar tree traversal. |
| `sglang:grammar_schema_count` | histogram | Grammar schema count. |
| `sglang:grammar_ebnf_size` | histogram | Grammar EBNF size. |
| `sglang:num_grammar_cache_hit_total` | counter | Grammar cache hits. |
| `sglang:num_grammar_aborted_total` | counter | Grammar aborts. |
| `sglang:num_grammar_total` | counter | Grammar requests. |

### Engine & Utilisation Gauges
| Metric | Type | Description |
|--------|------|-------------|
| `sglang:gen_throughput` | gauge | Tokens/sec. |
| `sglang:utilization` | gauge | Scheduler utilisation signal. |
| `sglang:engine_startup_time` | gauge | Engine startup time. |
| `sglang:engine_load_weights_time` | gauge | Weight loading time. |
| `sglang:spec_accept_length` | gauge | Speculative decoding acceptance length. |
| `sglang:swa_token_usage` | gauge | SWA-specialised token usage. |

> **Note:** Proposed counters such as `sglang:h2d_bytes_total` or Radix cache hit
> counters do **not** exist yet. Dashboards should treat them as future work.

## 2. DCGM Metrics (GPU)

The exporter runs DCGM 4.4.1-4.5.2 with only supported fields:

| Metric | Type | Description |
|--------|------|-------------|
| `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_MEM_CLOCK` | gauge | SM / HBM clock MHz. |
| `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_MEMORY_TEMP` | gauge | GPU / HBM temperature °C. |
| `DCGM_FI_DEV_POWER_USAGE` | gauge | Power draw (W). |
| `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | counter | Energy since boot (mJ). |
| `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE` | gauge | Framebuffer usage (MiB). |
| `DCGM_FI_DEV_GPU_UTIL` | gauge | GPU utilisation (%). |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | gauge | Copy-engine utilisation (%). |
| `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | counter | PCIe replay events. |
| `DCGM_FI_DEV_XID_ERRORS` | gauge | Last XID error. |
| `DCGM_FI_PROF_SM_ACTIVE` | gauge | Active SM cycles (%). |
| `DCGM_FI_PROF_SM_OCCUPANCY` | gauge | Warp occupancy (%). |
| `DCGM_FI_PROF_DRAM_ACTIVE` | gauge | DRAM active fraction (%). |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | gauge | Tensor pipe active fraction (%). |
| `DCGM_FI_PROF_PCIE_TX_BYTES`, `DCGM_FI_PROF_PCIE_RX_BYTES` | gauge | PCIe TX/RX rate (bytes/sec). |
| `DCGM_FI_PROF_NVLINK_TX_BYTES`, `DCGM_FI_PROF_NVLINK_RX_BYTES` | counter | NVLink TX/RX bytes (on systems exposing them). |

> **Known Gap:** There are no Grace↔Hopper C2C or host-memory counters exposed via
> DCGM today. The GH200 perf PMUs are not accessible on this VM image.

## 3. node_exporter Metrics (Host)

All standard collectors remain enabled (`cpu`, `meminfo`, `pressure`, `diskstats`,
`filesystem`, `netdev`, etc.). These provide the signals referenced in the v2
catalog (PSI, major faults, swap activity, IO time, load averages, network
throughput, etc.). No additional textfile collectors are configured by default.

## 4. Jaeger Tracing

Jaeger all-in-one listens on:
- gRPC (`otel`): `:4317`
- HTTP (`otel`): `:4318`
- UI: `:16686`

The SGLang server emits spans when `--enable-trace` is set (already in the
example launcher). Traces are stored in `/telemetry/jaeger/<CONTAINER_RUN_ID>/`.

## 5. Logs & Metadata

- Run log: `/telemetry/logs/container-run-*.log` (stdout/stderr tee). The example
  run in this session is `container-run-20251004T203819Z-df21c950.log`.
- `container_run_meta.env` retains the currently active run metadata (ID, start
  time, log path, Prometheus/Jaeger storage directories).
- Prometheus TSDB: `/telemetry/prometheus/<CONTAINER_RUN_ID>/`
- Jaeger Badger DB: `/telemetry/jaeger/<CONTAINER_RUN_ID>/`

## 6. Limitations & Known Gaps

- **Grace↔Hopper C2C metrics:** The expected perf PMUs (`nvidia_nvlink_c2c*`,
  `nvidia_scf_pmu*`) are not exposed on this VM (no device nodes in
  `/sys/bus/event_source/devices`). perf stat commands fail with “Cannot find
  PMU `nvidia_nvlink_c2c0_pmu_0`.” Capturing host↔HBM bytes requires either
  enabling those PMUs at the kernel/driver level or using Nsight Compute CLI
  outside the container.
- **Application-level byte counters:** Proposed SGLang counters for H2D/D2H bytes,
  KV residency, MoE imbalance, etc. are not implemented yet. Current dashboards
  must rely on existing metrics.
- **Textfile collectors:** No helpers are shipping to scrape `/proc/<pid>` or perf
  outputs. If we want those, they must be added explicitly later.

## 7. Suggested Future Enhancements (Optional)

These are not implemented but remain worthwhile once priorities allow:

1. **SGLang byte counters** – instrument the key CUDA copy / KV spill / offload
   paths to emit H2D/D2H totals and residency gauges. This would give a direct
   host↔HBM view even without Perf PMUs.
2. **Radix cache counters** – convert `sglang:cache_hit_rate` gauge into
   counter pairs (`hits`, `queries`) so PromQL hit-rate calculations are
   robust.
3. **MoE per-expert metrics** – expose skew/entropy counters for MoE deployments
   (bounded label cardinality).
4. **Textfile collector for `/proc`** – for long-running benchmarks, capturing
   per-process IO / RSS / faults may help correlate host pressure events.
5. **Nsight Compute integration** – scripted `ncu` runs for targeted profiling
   if GH200 PMUs remain unavailable.

---

**Summary:** The devcontainer collects the full SGLang scheduler surface, the
sanctioned DCGM metrics, and node_exporter’s host stats. Correlate energy, power,
queueing, and latency using these sources. Grace↔Hopper C2C bytes are currently
out of scope until NVIDIA exposes the PMUs (or Nsight tooling is adopted).
