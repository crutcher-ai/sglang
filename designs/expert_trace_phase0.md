# Expert Trace Phase 0 – Design Doc

## Goal
Capture per-layer, per-decode-step expert selections in Qwen3-Next without disturbing the existing execution path, so we can analyze how many unique experts are activated in each step under varying concurrency. Phase 0 focuses on instrumentation and data capture; data analysis comes later.

## Scope
- Decode steps only (prefill may be added later).
- Prototype-level logging: offline use, JSONL output acceptable (performance secondary).
- No behavior change when tracing disabled (default).
- Integrate with existing expert distribution recorder to reuse context (layer/step metadata).

## Sources of truth
- `python/sglang/srt/models/qwen2_moe.py` – MoE block and existing calls to `ExpertDistributionRecorder`.
- `python/sglang/srt/eplb/expert_distribution.py` – per-layer hooks with forward-pass context.
- `python/sglang/srt/model_executor/forward_batch_info.py` – per-token metadata in `ForwardBatch`.

## Requirements
1. Trace enabling controlled via env vars (e.g., `SGLANG_MOE_TRACE_DIR`, `SGLANG_MOE_TRACE_PHASE`).
2. Captured record must include:
   - `run_id`, `server_session_id`, `rank`
   - `step` (forward pass id), `phase` (`decode`), `layer`
   - per-token entries: request slot (`req_pool_indices`), token position, seq len, top-k expert ids
   - timestamp for alignment
3. Buffering: accumulate records in memory and flush to JSONL in trace dir at intervals and on demand (`/dump_expert_distribution_record`).
4. Integration: piggy-back on `ExpertDistributionRecorder` hooks (no modification to MoE layers).
5. Safety: No tracing work when disabled; minimal additional allocations; non-blocking copies to host.

## Architecture
### ExpertTraceWriter (new module `python/sglang/srt/trace/expert_trace.py`)
- Initialized at startup if `SGLANG_MOE_TRACE_DIR` set.
- Stores configuration (phase filter, flush interval, optional sample rate).
- Maintains rank-local buffers:
  - `current_step_tokens`: maps `step -> {slot, position, seq_len}` captured at forward-pass start.
  - `records`: list of per-layer records to be flushed.
- Methods:
  - `on_forward_pass_start(step, forward_batch)` – cache per-token metadata (CPU tensors) when entering a forward pass. Should erase prior cached data for that step.
  - `handle_on_select_experts(step, layer, topk_ids)` – build per-layer record using cached metadata and append to `records`.
  - `on_forward_pass_end(step)` – optional cleanup.
  - `flush(reason)` – write current `records` to JSONL and reset buffer; return file path list.
  - `shutdown()` – flush remaining records on process exit.

### Hooks in `ExpertDistributionRecorder`
Modify `_ExpertDistributionRecorderReal` to:
- create `self._trace_writer` if enabled;
- call writer methods in `_on_forward_pass_start`, `_on_hook` (specifically when `hook_name` is `on_select_experts`), `_on_forward_pass_end`, `_reset`, `start_record`, `stop_record`, `dump_record` (collecting file paths in return payload).

Phase filter: only record when forward mode matches configuration (initial default `decode`).

### File naming and metadata
- Use `container_run_id` and `server_session_id` from env (`SGLANG_CONTAINER_RUN_ID`, `SGLANG_SERVER_SESSION_ID`, fallback to manifest if unset).
- Output file pattern: `expert_trace_{run}_{session}_rank{rank}_{flush_idx}.jsonl` under trace dir.
- Each JSON object corresponds to `(step, layer)` pair.

### JSON schema (per record)
```
{
  "run_id": "container-run-…",
  "session_id": "srv-…",
  "rank": 0,
  "phase": "decode",
  "step": 123,
  "layer": 17,
  "timestamp_ns": 1700000000000,
  "tokens": [
    {"slot": 5, "position": 413, "seq_len": 420, "experts": [17, 232, …]},
    …
  ]
}
```
Shared expert is omitted (always active). If needed later, include `"shared": true` per token.

## Flush strategy
- Check `time.monotonic_ns()` on each append; if interval exceeded, flush.
- Flush when `dump_record` called and during shutdown / recorder reset.
- Include file paths in HTTP response to `/dump_expert_distribution_record` (extend payload with `"expert_trace_files": [...]`).

## Error handling
- If `forward_batch` metadata missing, skip recording for that step.
- Handle empty token arrays (e.g., idle rank) gracefully.

## Testing plan
1. Run a small decode (single token) with tracing enabled; confirm JSONL exists and matches schema.
2. Run with tracing disabled; confirm behaviour unchanged (no files, no extra work).
3. Invoke `/dump_expert_distribution_record`; verify response contains aggregate `.pt` and trace file list.

## Future work (beyond Phase 0)
- Switch to binary output (`.pt`, `.npz`) for efficiency.
- Prefill tracing, sampling rate controls.
- Analysis tooling and visualization.
- Optional streaming to metrics/logging surfaces.
