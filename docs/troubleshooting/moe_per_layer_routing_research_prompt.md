# Troubleshooting Log: MoE Per-Layer Routing Telemetry (Research Prompt)

> **Audience**: External reviewers (LLMs or humans) who only have access to upstream `sglang` v0.5.3 source. This document supplies the additional context from our fork so reviewers can reason about why our per-layer Mixture-of-Experts (MoE) routing telemetry still lacks the detail we need.
>
> **Status**: Troubleshooting / research prompt. Delete once the issue is resolved.

---

## 1. Executive Summary

We forked `sglang` at tag `v0.5.3` and added a telemetry pipeline intended to capture per-token, per-layer MoE routing decisions (top-k expert IDs) for the Qwen3-Next-80B-A3B-Thinking-FP8 model. The goal is to flush structured JSONL so we can post-process routing skew and load balance. Despite wiring new recorder hooks and an `ExpertTraceWriter`, the generated JSONL lacks the expected per-layer content (either empty `tokens` arrays or missing flushes entirely depending on runs).

We need a second set of eyes to determine what we are still missing relative to upstream. This prompt summarizes:

- The runtime environment and how we launch the helper/server.
- Exact environment variables and HTTP calls used to enable the recorder and trace writer.
- The current observable symptoms.
- All fork-specific code changes that could influence MoE routing telemetry (with inline diffs versus upstream `v0.5.3`).
- A list of focused questions for reviewers.

Upstream-only reviewers should be able to reproduce the reasoning path using the snippets provided here.

---

## 2. Environment Overview

- **Hardware**: NVIDIA GH200 (96 GB HBM, 480 GB host DDR) per `docs/perf_baseline_gh200.md`.
- **Helper topology**: We run `./scripts/start_observable_container.sh` to launch a host-network helper container (`sglang-dev`) that starts Prometheus (9090), node_exporter (9100), dcgm-exporter (9400), Jaeger v2 (UI 16686, OTLP 4317/4318). Pointer file: `~/sglang-observability/telemetry/container_run_meta.env`.
- **Server launch**: `./scripts/infer/start_server.sh` (custom script in our fork) starts `sglang.launch_server` inside the helper, blocks on readiness (`/get_model_info` + `/trace_probe`), and prints a single JSON record when ready.
- **Model**: `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` with tensor-parallel size 1.
- **Recorder settings** (set in environment prior to launch):
  ```bash
  EXPERT_DISTRIBUTION_RECORDER_MODE=per_token
  SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace
  SGLANG_MOE_TRACE_PHASE=decode
  SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5
  SGLANG_FORCE_STANDARD_TOPK=1
  SGLANG_EXTRA_ARGS="--moe-runner-backend triton"
  ```
  Additional core envs: `MEM_FRACTION_STATIC=0.98`, `CONTEXT_LENGTH=16384`, `MAX_PREFILL_TOKENS=16384`, `MAX_TOTAL_TOKENS=16384`, `MAX_MAMBA_CACHE_SIZE=160`, `KV_CACHE_DTYPE=fp8_e4m3`, `ENABLE_TRACE=1`, `OTEL_TRACES_SAMPLER=always_on`.
- **MoE dump**: After driving traffic with `scripts/bench/run_window_bench.sh`, we call `curl -sf http://127.0.0.1:30000/dump_expert_distribution_record`.

---

## 3. Observed Behaviour

1. The HTTP dump returns HTTP 200 with a JSON payload confirming `recording: false` (expected) and sometimes listing `expert_trace_files`, e.g.:
   ```json
   {
     "metadata": {
       "expert_trace_files": [
         "/home/devuser/sglang-observability/telemetry/expert-trace/expert_trace_container-run-20241018T232215Z-a4591_srv-9115f50c6fbb47bf915943a7dd7077e1_rank00000_00001.jsonl"
       ]
     }
   }
   ```
2. The referenced JSONL exists but contains records with empty or near-empty `tokens` arrays. Example record (sanitized):
   ```json
   {"run_id":"container-run-20241018T232215Z-a4591","session_id":"srv-9115f50c6fbb47bf915943a7dd7077e1","rank":0,"phase":"decode","step":142,"layer":17,"timestamp_ns":1729299345123456789,"tokens":[],"flush_reason":"interval"}
   ```
   Occasionally a handful of tokens appear, but counts per layer are far lower than the number of decode tokens emitted.
3. When we disable `SGLANG_FORCE_STANDARD_TOPK`, the JSONL is empty (0 bytes). With the flag enabled, files flush but are missing content as above.
4. Prometheus counters (`sglang:prompt_tokens_total`, `sglang:generation_tokens_total`) and analyzer output confirm decode workload succeeded (e.g. 163 840 decode tokens over ~60 s window).
5. The on-disk recorder `.pt` statistics (inside `/telemetry/logs/...`) do reflect aggregate counts per layer, so the high-level recorder is capturing something.

**Working hypothesis**: even though we call `handle_on_select_experts`, the `topk_ids` tensor either contains sentinel `-1` entries (which we filter), or the hook fires for only a subset of tokens/layers. We want reviewers to confirm whether our modifications are sufficient or if we missed a necessary upstream call (e.g., normalization path, DeepEP dispatch relation, or hybrid layers).

---

## 4. Steps Already Taken

1. **Forcing STANDARD TopK**: Added `SGLANG_FORCE_STANDARD_TOPK=1` env; verified via server logs that `/trace_probe` reports tracing enabled and writer logs the env combination.
2. **Switching MoE runner backend**: Set `SGLANG_EXTRA_ARGS="--moe-runner-backend triton"` to avoid FlashInfer/BYPASSED fast paths. Behaviour unchanged (empty `tokens`).
3. **Sampling phases**: Tried `SGLANG_MOE_TRACE_PHASE=decode`, `prefill`, and `both`; decode-only still empty, both yields identical emptiness for decode and zero entries for prefill.
4. **Flush cadence**: Lowered `SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC` to 1 s to ensure we flush frequently—no difference besides more empty files.
5. **Validation**: Confirmed `ExpertDistributionRecorder.dump_record(output_mode="file")` returns metadata with `expert_trace_files`, so writer is active.
6. **Checked aggregator `_TOP_K_NUM`**: We set `_DetailSinglePassGatherer._TOP_K_NUM = 10` to match Qwen3-Next’s `num_experts_per_tok`; still empty tokens in JSONL.

---

## 5. Fork-Specific Code Changes (vs `v0.5.3`)

Below are the edits that touch MoE routing, tracing, or recorder logic. Reviewers should factor these differences in their analysis.

### 5.1 New `ExpertTraceWriter` module

_File_: `python/sglang/srt/trace/expert_trace.py`

Purpose: collects per-step routing info when `SGLANG_MOE_TRACE_DIR` is set. It records slots, positions, seq_lens when a forward pass starts, and appends per-token expert IDs inside `handle_on_select_experts`.

```python
class ExpertTraceWriter:
    ENV_TRACE_DIR = "SGLANG_MOE_TRACE_DIR"
    ENV_TRACE_PHASE = "SGLANG_MOE_TRACE_PHASE"
    ENV_TRACE_FLUSH_INTERVAL = "SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC"
    ENV_TRACE_SAMPLE_RATE = "SGLANG_MOE_TRACE_SAMPLE_RATE"
    ...
    def handle_on_select_experts(self, step_id: int, layer_id: int, topk_ids: torch.Tensor):
        ...
        for idx, experts in enumerate(experts_list):
            filtered = [int(ex) for ex in experts if ex >= 0]
            if not filtered:
                continue
            token_entry = {
                "slot": slots[idx] if idx < len(slots) else idx,
                "position": positions[idx] if idx < len(positions) else None,
                "seq_len": seq_lens[idx] if idx < len(seq_lens) else None,
                "experts": filtered,
            }
            tokens.append(token_entry)
        ...
```

### 5.2 Recorder integration hooks

_File_: `python/sglang/srt/eplb/expert_distribution.py`

Key differences (excerpt):

```diff
@@
-from sglang.srt.server_args import ServerArgs
+from sglang.srt.server_args import ServerArgs
+from sglang.srt.trace import ExpertTraceWriter
+from sglang.srt.trace.expert_trace import get_expert_trace_writer
@@
-        self._accumulator = _Accumulator.init_new(server_args, expert_location_metadata, rank)
+        self._accumulator = _Accumulator.init_new(server_args, expert_location_metadata, rank)
+        self._expert_trace_writer: Optional[ExpertTraceWriter] = get_expert_trace_writer()
@@
-        if not self._recording:
-            return
+        if not self._recording:
+            if self._expert_trace_writer:
+                step_id = self._current_forward_pass_id.value
+                if step_id is not None:
+                    self._expert_trace_writer.on_forward_pass_start(step_id, forward_batch)
+            return
@@
-        getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)
+        if should_run_gatherer:
+            gatherer = ...
+            getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)
+
+        if (
+            self._expert_trace_writer
+            and hook_name == "on_select_experts"
+            and "topk_ids" in kwargs
+        ):
+            step_id = self._current_forward_pass_id.value
+            layer_idx = self._current_layer_idx.value
+            if step_id is not None and layer_idx is not None:
+                self._expert_trace_writer.handle_on_select_experts(step_id, layer_idx, kwargs["topk_ids"])
@@
-        self._accumulator.reset()
+        self._accumulator.reset()
+        if self._expert_trace_writer:
+            self._expert_trace_writer.reset()
@@
-        self._recording = True
+        self._recording = True
+        if self._expert_trace_writer:
+            self._expert_trace_writer.start_record()
@@
-        self._recording = False
+        self._recording = False
+        if self._expert_trace_writer:
+            self._expert_trace_writer.stop_record()
@@
-        output = self._accumulator.dump(output_mode=output_mode)
-        self._reset()
-        return output
+        output = self._accumulator.dump(output_mode=output_mode)
+        trace_files: List[str] = []
+        if self._expert_trace_writer:
+            trace_files = self._expert_trace_writer.flush(reason="dump_record")
+        self._reset()
+        if output_mode == "file" and trace_files:
+            output = output or {}
+            output.setdefault("metadata", {})
+            output["metadata"]["expert_trace_files"] = trace_files
+        return output
@@
-class _DetailSinglePassGatherer(_SinglePassGatherer):
-    _TOP_K_NUM = 8
+class _DetailSinglePassGatherer(_SinglePassGatherer):
+    _TOP_K_NUM = 10
```

### 5.3 TopK output override

_File_: `python/sglang/srt/layers/moe/topk.py`

```diff
@@ class TopK(CustomOp):
-        if self.topk_config.output_format is not None:
+        force_standard = get_bool_env_var("SGLANG_FORCE_STANDARD_TOPK")
+        if force_standard:
+            output_format = TopKOutputFormat.STANDARD
+        elif self.topk_config.output_format is not None:
             output_format = self.topk_config.output_format
```

### 5.4 Request struct tweak

_File_: `python/sglang/srt/managers/io_struct.py`

We allow expert distribution RPCs to carry optional request IDs so we can tie dumps back to runs.

```diff
 class ExpertDistributionReq(BaseReq):
-    action: ExpertDistributionReqType
+    action: ExpertDistributionReqType
+
+    def __init__(self, *, action: ExpertDistributionReqType, rid: Optional[Union[str, List[str]]] = None):
+        self.rid = rid
+        self.action = action
```

### 5.5 Launch script (new in fork)

_File_: `scripts/infer/start_server.sh`

Highlights relevant to reviewers:

- Reads `.devcontainer/tools/sglang-config.json` for defaults, then overrides with env variables listed earlier.
- Forwards recorder/writer envs directly into the containerized server process:
  ```bash
  docker exec -u devuser \
    -e SGLANG_MOE_TRACE_DIR="$SGLANG_MOE_TRACE_DIR" \
    -e SGLANG_MOE_TRACE_PHASE="$SGLANG_MOE_TRACE_PHASE" \
    -e SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC="$SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC" \
    -e EXPERT_DISTRIBUTION_RECORDER_MODE="$EXPERT_DISTRIBUTION_RECORDER_MODE" \
    -e SGLANG_FORCE_STANDARD_TOPK="$SGLANG_FORCE_STANDARD_TOPK" \
    ...
  ```
- After readiness, it calls `/trace_probe` to ensure tracing has registered threads before emitting the ready JSON.

Full script is 400+ lines; happy to provide more context if needed.

---

## 6. Questions for Reviewers

1. **Hook coverage**: In upstream `v0.5.3`, are there execution paths where `select_experts` (STANDARD) still generates `-1` placeholders for tokens (e.g., padding, fused shared experts) that our filter removes? Should we capture additional metadata (e.g., `token_index`) to reconcile token counts?
2. **DeepEP interplay**: With Qwen3-Next on GH200, do DeepEP `dispatch_a/dispatch_b` paths emit additional top-k remapping we should hook? We currently rely on the TopK output and the recorder’s gatherers. Are there situations where DeepEP rewrites expert IDs after `on_select_experts` fires?
3. **Forward batch metadata**: We pull `forward_batch.req_pool_indices`, `.positions`, `.seq_lens_cpu`. Could these be `None` for decode tokens in certain schedulers (two-batch overlap, hybrid GDN)? If so, how should we map tokens back to slots to avoid dropping them?
4. **_TOP_K_NUM alignment**: We hard-coded `_TOP_K_NUM = 10`. Should we instead source this from `expert_location_metadata` to avoid mismatches when fewer experts are emitted (e.g., shared experts)? Could truncated tensors explain empty tokens arrays?
5. **Sampling windows**: Does the recorder only cover tokens emitted while `recording=True`? We start recording implicitly when the server launches with `enable_expert_distribution_metrics`. Are there code paths where `recording` is false but the writer still needs to capture events?
6. **Alternative hooks**: Is there a better place upstream to intercept routing decisions (e.g., inside `ExpertLocationDispatcher`, DeepEP buffers) that guarantees coverage even when the Standard path is bypassed?
7. **Expected output shape**: For Qwen3-Next 80B, what should a ground-truth per-token record look like? If upstream has tests or tooling (e.g., `tools/analyze_expert_trace.py`), what inputs should yield non-empty per-layer data?

Any insight on these topics—especially confirmation that our hooks should already capture all decode tokens—would help focus the next debugging steps.

---

## 7. Reproduction Checklist (for reviewers with upstream only)

Even without our fork, reviewers can reason about the issue by following these steps:

1. Inspect upstream `python/sglang/srt/eplb/expert_distribution.py` to understand how `_DetailSinglePassGatherer` and `on_select_experts` behave in `per_token` mode.
2. Check `python/sglang/srt/layers/moe/topk.py` to confirm when `select_experts` is invoked vs Triton/BYPASSED outputs.
3. Review DeepEP dispatch logic (`python/sglang/srt/layers/moe/token_dispatcher/deepep.py`) to see how expert IDs are rearranged after routing.
4. Compare this document’s diffs with upstream to assess whether our integration points are sufficient.
5. Evaluate whether tokens might be dropped because of pad tokens, shared experts, or gating thresholds.

---

## 8. Appendix A — Sample Ready JSON

```json
{
  "schema_version": 1,
  "run_id": "container-run-20241018T232215Z-a4591",
  "server_session_id": "srv-9115f50c6fbb47bf915943a7dd7077e1",
  "health": "ready",
  "port": 30000,
  "manifest_host_path": "/home/ubuntu/sglang-observability/telemetry/container_runs/container-run-20241018T232215Z-a4591/manifest.json",
  "log_file": "/telemetry/logs/container-run-20241018T232215Z-a4591.log",
  "started_at_iso": "2024-10-18T23:22:25Z",
  "session_record_host_path": "/home/ubuntu/sglang-observability/telemetry/container_runs/container-run-20241018T232215Z-a4591/logs/provider_sessions/20241018T232245Z_srv-9115f50c6fbb47bf915943a7dd7077e1/start.json",
  "sizing": {
    "tp_size": 1,
    "mem_fraction_static": 0.98,
    "chunked_prefill_size": 16384,
    "context_length": 16384,
    "max_prefill_tokens": 16384,
    "max_total_tokens": 16384,
    "max_mamba_cache_size": 160
  }
}
```

---

## 9. Appendix B — Commands

1. Start helper: `./scripts/start_observable_container.sh`
2. Start server (env as above):
   ```bash
   EXPERT_DISTRIBUTION_RECORDER_MODE=per_token \
   SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace \
   SGLANG_MOE_TRACE_PHASE=decode \
   SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5 \
   SGLANG_FORCE_STANDARD_TOPK=1 \
   SGLANG_EXTRA_ARGS="--moe-runner-backend triton" \
   MEM_FRACTION_STATIC=0.98 \
   CONTEXT_LENGTH=16384 \
   MAX_PREFILL_TOKENS=16384 \
   MAX_TOTAL_TOKENS=16384 \
   MAX_MAMBA_CACHE_SIZE=160 \
   KV_CACHE_DTYPE=fp8_e4m3 \
   ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on \
   ./scripts/infer/start_server.sh
   ```
3. Drive workload: `CONCURRENCY=96 TOTAL=640 scripts/bench/run_window_bench.sh`
4. Dump recorder: `curl -sf http://127.0.0.1:30000/dump_expert_distribution_record`
5. Inspect trace: `ls -l $HOME/sglang-observability/telemetry/expert-trace/expert_trace_*`

---

## 10. Appendix C — Pointers for Future Updates

- If reviewers identify additional code paths (e.g., post-routing remap) we should hook, document them here before implementing.
- Capture any sample outputs that do contain per-token data for comparison.
- Once fixed, delete this file to avoid stale troubleshooting notes lingering in the repo.
