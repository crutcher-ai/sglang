# Troubleshooting Log (Decode Coverage & Expert Affinity)

> **Audience**: External reviewers (LLMs or humans) with access only to upstream `sglang` v0.5.3 and the public Qwen3-Next FP8 checkpoints. This document explains the issue we’re seeing on our fork so they can reason about it without seeing our internal repository.
>
> **Goal**: Understand why per-token MoE routing traces contain only prefill records (no decode rows) and determine how to capture fuller “stack-ranked” expert choices (beyond the routed top-k) so we can study expert affinity across consecutive selections.
>
> **Status**: Active troubleshooting. Delete this file after the decode gap is resolved.

---

## 1. TL;DR

- We forked `sglang` at `v0.5.3` and added telemetry that logs per-token, per-layer routed experts via an `ExpertTraceWriter` hooked into `ExpertDistributionRecorder`.
- With the current pipeline we capture rich prefill traces (e.g., tensor of shape `[num_tokens, 10]` per layer) but **decode tokens never appear in the JSONL**, even when we generate 64 completion tokens.
- We need help pinpointing why decode is missing and how to optionally record more than the routed top-10 weights so downstream scripts can compute expert affinity (e.g., per-layer Jaccard overlap, sequential transitions).
- **New experiment (2025-10-20 03:45 UTC):** we re-ran with CUDA graphs explicitly disabled (`--disable-cuda-graph --disable-cuda-graph-padding`) and started recording via `POST /start_expert_distribution_record` before issuing the request. The resulting trace (`container-run-20251020T034230Z-bb457d36`) still contains only `phase: "prefill"` rows—decode remains absent. This suggests the decode path is bypassing our hook for reasons beyond CUDA graph replay (e.g., alternate routing code path, additional recorder gating, or decode-specific workers).

---

## 2. Environment & Workflow Recap

- **Hardware**: GH200 (96 GB HBM, 480 GB DDR). Helper container launched via `scripts/start_observable_container.sh` (host networking, Jaeger v2 on 16686/4317, Prometheus 9090, node 9100, dcgm 9400).
- **Server launch**: custom `scripts/infer/start_server.sh` (contract described in the prior troubleshooting doc). Key env overrides:
  ```bash
  EXPERT_DISTRIBUTION_RECORDER_MODE=per_token
  SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace
  SGLANG_MOE_TRACE_PHASE=all
  SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5
  SGLANG_FORCE_STANDARD_TOPK=1
  SGLANG_EXTRA_ARGS="--moe-runner-backend triton"
  MEM_FRACTION_STATIC=0.98
  CONTEXT_LENGTH=16384
  MAX_PREFILL_TOKENS=16384
  MAX_TOTAL_TOKENS=16384
  MAX_MAMBA_CACHE_SIZE=160
  KV_CACHE_DTYPE=fp8_e4m3
  ENABLE_TRACE=1
  OTEL_TRACES_SAMPLER=always_on
  ```
- **Test workload**: single request to `/v1/chat/completions` with `max_tokens=64`, `temperature=0.2`. Prefill length ~6 tokens, decode 64 tokens.
- **Dump**: `curl -sf http://127.0.0.1:30000/dump_expert_distribution_record` immediately after the request.
- **Trace files**: latest session `container-run-20251020T030833Z-3d7d0ab2` produced:
  - `expert_trace_container-run-20251020T030833Z-3d7d0ab2_srv-f173ee0ff6724920a826790c77678e57_rank0_0000{1,2,3}.jsonl`

---

## 3. Observations (Decode Gap)

1. Running our analyzer (`python3 tools/analyze_expert_trace.py --dir …/expert-trace`) shows:
   ```
   files=8 records=192 token_expert_events=22560
   phase summary:
      prefill: records=192 token_events=22560
   ```
   → No `phase="decode"` records anywhere, despite 64 completion tokens.

2. Inspecting the newest JSONL (snipped for brevity):
   ```json
   {"run_id": "…", "phase": "prefill", "step": 10, "layer": 43,
    "tokens": [{"slot": 1, "position": 0, "seq_len": 16, "experts": [482, 273, …]}, …] }
   ```
   All entries are `phase: prefill` with a full 15-token spread (expected for a batch-size-one prefill). No decode tokens (which should appear as `phase: decode`, typically one token per active sequence per step).

3. We confirmed the trace writer is called every time `_on_hook("on_select_experts",…)` fires. Prefill clearly triggers the hook. Decode either never fires the hook or the writer discards the events.

4. There is no sampling. `SGLANG_MOE_TRACE_SAMPLE_RATE` defaults to 1.0 (all events). The writer logs every routed token that reaches `_on_select_experts`.

5. Hypothesis: decode runs through a different path (e.g., disaggregated decode worker, speculative decoding, or DeepEP low-latency mode) that never hits our hook or hits it under a different signature. Capturing decode may require hooking additional locations (e.g., `on_deepep_dispatch_*` or the decode-specific worker) or ensuring the recorder stays in `per_token` mode during decode.

---

## 4. Current Trace Files (for reviewers)

Each line in `expert_trace_container-run-20251020T030833Z-3d7d0ab2_srv-f173…_rank0_0000X.jsonl` is:
```json
{
  "run_id": "container-run-20251020T030833Z-3d7d0ab2",
  "session_id": "srv-f173…",
  "rank": 0,
  "phase": "prefill",
  "step": <forward pass id>,
  "layer": <int>,
  "timestamp_ns": <int>,
  "ids_kind": "topk_ids",  # indicates we captured routed expert IDs (not logical idx)
  "topk_dim": 10,            # width of expert list per token (top_k)
  "tokens": [
    {"slot": <seq slot>, "position": <token pos>, "seq_len": <int or null>, "experts": [<expert ids…>]},
    …
  ],
  "flush_reason": "dump_record" | "interval"
}
```
- `tokens` array length equals the number of active sequences at that step (prefill step had ~16 entries). `experts` contains the 10 routed experts per token (shared expert logging is handled elsewhere in the recorder). All values are ints, safe for JSON consumption.
- Currently, every record has `phase="prefill"`; decode data is absent.

This dataset includes older sessions from prior experiments. For decode debugging focus on the newest files above.

---

## 5. Desired Analysis (“Expert Affinity”)

We want to characterise **expert affinity** — e.g., which experts are repeatedly selected across consecutive tokens or layers, whether there’s a “hot set” of experts that dominates certain sequences, and ideally compute similarity metrics (Jaccard, conditional probabilities) across layers and time.

To get there we need:
1. Decode-phase data (per token) alongside prefill.
2. Ideally, more insight than just the top-10 experts per token. For affinity analysis we may want the entire ranked list or at least a deeper cutoff (e.g., top-20). The current `TopK` router emits only top-10. We’d like suggestions for tapping into the router logits (perhaps before top-k truncation) or otherwise configuring a higher top-k if supported by the model/runtime.
3. Better reporting — e.g., scripts that compute, per layer:
   - Frequency of each expert during prefill vs. decode.
   - Overlap between consecutive tokens’ expert sets (same layer) to feed Jaccard.
   - Cross-layer transitions (e.g., probability an expert in layer L routes to expert Y in layer L+1).

Current analyzer is intentionally simple and needs to be extended. We’ll do that once decode data is available.

---

## 6. Fork-Specific Changes (relevant to decode)

Differences from upstream `v0.5.3` relevant to this investigation:

1. `python/sglang/srt/eplb/expert_distribution.py`
   ```diff
   +        if self._expert_trace_writer:
   +            step_id = self._current_forward_pass_id.value
   +            if step_id is not None:
   +                self._expert_trace_writer.on_forward_pass_start(step_id, forward_batch)

   +        if self._expert_trace_writer and hook_name == "on_select_experts":
   +            ids_tensor = kwargs.get("topk_ids") or kwargs.get("topk_idx")
   +            ids_kind = "topk_ids" if "topk_ids" in kwargs else "topk_idx" if "topk_idx" in kwargs else "unknown"
   +            if step_id is not None and layer_idx is not None and ids_tensor is not None:
   +                self._expert_trace_writer.handle_on_select_experts(step_id, layer_idx, ids_tensor, ids_kind)

   +class _DetailSinglePassGatherer(_SinglePassGatherer):
   +    _TOP_K_NUM = 10  # adjusted from upstream’s 8 for Qwen3-Next
   ```
   We call our writer even when `recording` is true; upstream gatherers only run when recording is enabled, so we mirrored the lifecycle.

2. `python/sglang/srt/trace/expert_trace.py`
   - New `ExpertTraceWriter` (see prior doc for full listing) writes JSONL when `SGLANG_MOE_TRACE_DIR` is set. It records slots/positions/seq_len plus the routed experts (top 10). JSON encoding ensures all values are ints.

3. `python/sglang/srt/layers/moe/topk.py`
   ```diff
   +        force_standard = get_bool_env_var("SGLANG_FORCE_STANDARD_TOPK")
   +        if force_standard:
   +            output_format = TopKOutputFormat.STANDARD
   ```
   Forces the STANDARD path to ensure `_on_select_experts` is called even when Triton/BYPASSED would skip it.

No other deliberate changes in decode pipeline.

---

## 7. Questions for Reviewers

1. **Decode Hook Coverage**: In upstream `v0.5.3`, does decode routing go through the same `TopK` / `ExpertDistributionRecorder.on_select_experts` path as prefill? If not, which code path handles decode (e.g., disaggregated workers, speculative loops) and where should we hook to capture those tokens?

2. **Disaggregation / Two-Phase Decode**: Does SGLang split prefill vs. decode into different components (e.g., separate scheduler loops or reentrant gatherers) that might leave `recording` false or skip the trace writer? We auto-start recording when `enable_expert_distribution_metrics=True`, but maybe decode operates with a distinct recorder mode (`stat`, `stat_approx`) unless we manually call `/start_expert_distribution_record`.

3. **Full Expert Stack**:
   - Is there a supported way (config / env) to increase `top_k` beyond 10 for Qwen3-Next, so we can see a deeper ranked list?
   - Alternatively, can we tap into the router logits before truncation (e.g., capture `router_logits` or `routing_data` per token) without expensive instrumentation?
   - Do we need to modify the model’s config (`num_experts_per_tok`) to a higher value, or is that fixed when the checkpoint was trained?

4. **Decode vs Prefill Data Size**: The current experiment shows `prefill` records ~200 tokens per layer; decode has 64 tokens but nothing logged. What in the code prevents decode tokens from hitting `on_select_experts`? Are decode tokens bypassing the recorder due to caching, or does the decode worker skip the global recorder when `per_token` mode is set?

5. **Downstream Reporting**: Any suggestions from upstream on best practices for affinity analysis? e.g., existing scripts or tests that compute expert skew or correlation we can draw from.

---

## 8. Artifacts for Reviewers

Please refer to the latest trace JSONL (prefill-only) to understand the format:
- `expert_trace_container-run-20251020T030833Z-3d7d0ab2_srv-f173ee0ff6724920a826790c77678e57_rank0_00003.jsonl`
  (each line is a single layer step; please note the `phase` field).

If decode instrumentation is identified, we will regenerate the trace and share updated files.

---

## 9. Next Steps

- Pending expert review, likely instrument additional decode-specific hooks (if needed) or adjust recorder initialization for decode.
- Once decode data is flowing, extend reporting scripts to compute per-layer/phase expert affinity metrics (Jaccard, transition probabilities, etc.).
- Investigate capturing more than top-10 experts if feasible.

---

## 10. Round‑2 Instrumentation Summary (2025‑10‑20)

We added env‑gated taps for decode under fused Triton paths (EP=1): runner emit (`SGLANG_MOE_TRACE_DECODE_FROM_RUNNER=1`), standard dispatcher emit (`SGLANG_MOE_TRACE_FROM_DISPATCH=1`), a layer‑context guard around fused decode dispatch/apply, server argv logging to confirm CLI flags, and an optional backend assert (`SGLANG_ASSERT_MOE_RUNNER`).

STANDARD proof and TRITON+taps still show `phase=decode: records=0` in our Qwen3‑Next FP8 runs (prefill records healthy). Next step: verify flags via `[trace_debug] argv=...`, and if honored, add a tiny, env‑gated kernel scribe to store [T,K] indices during decode for emission.
