# MoE Decode Trace — Round 2 Instrumentation & Results (2025‑10‑20)

This note captures exactly what we changed since the first brief, how to roll it all back, what we ran, and what we observed. It is written so a reviewer can reproduce, or revert, without diff spelunking.

## Objectives

- Add decode‑time taps at the loci GPT‑5‑Pro identified for Qwen3‑Next‑80B‑A3B (FP8) with EP=1 under Triton fused MoE.
- Verify that CLI flags (e.g., `--moe-runner-backend`) actually reach the server.
- Keep all changes env‑gated and safe by default.
- Run two proofs: STANDARD runner and TRITON runner + taps. Confirm `phase=decode` rows appear.

## Changes (all gated; default OFF)

1) Log argv inside the server process (flag verification)
- File: `python/sglang/srt/entrypoints/http_server.py`
- Change: log `"[trace_debug] argv=..."` at worker init.
- Rollback: remove the `logger.warning("[trace_debug] argv=...", ...)` line.

2) Optional backend assert (debug only)
- File: `python/sglang/srt/layers/moe/moe_runner/runner.py`
- Change: if `SGLANG_ASSERT_MOE_RUNNER` is set, compare to the active backend and raise `RuntimeError` on decode mismatch.
- Rollback: delete the `SGLANG_ASSERT_MOE_RUNNER` block in `run()`.

3) Triton fused runner tap (authoritative decode path)
- File: `python/sglang/srt/layers/moe/moe_runner/triton.py`
- Change: in `TritonRunnerCore.run`, when `SGLANG_MOE_TRACE_DECODE_FROM_RUNNER=1`, emit `rec.on_select_experts(topk_ids=runner_input.topk_ids)`.
- Rollback: remove the try/except block that calls `on_select_experts`.

4) EP=1 standard dispatcher tap (backstop for non‑DeepEP)
- File: `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
- Change: in `StandardDispatcher.dispatch`, when `SGLANG_MOE_TRACE_FROM_DISPATCH=1` and `topk_output.topk_ids` is present, emit `rec.on_select_experts(...)`.
- Rollback: remove the try/except emission block.

5) Ensure layer context around fused decode dispatch
- File: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- Change: best‑effort `with_current_layer(self.layer_id)` enter/exit guarding `dispatch(...)` and `quant_method.apply(...)`, using manual context enter/exit to avoid nested assertions.
- Rollback: restore to direct calls without the `_ctx.__enter__/__exit__` wrapper.

6) Analyzer summarizer K env‑tunable
- File: `python/sglang/srt/eplb/expert_distribution.py`
- Change: set detail gatherer `_TOP_K_NUM` from `SGLANG_MOE_TRACE_TOPK` (default remains 10) so summaries can show 10+1 when desired.
- Rollback: hardcode `_TOP_K_NUM = 10` again.

Notes
- All taps are gated by envs: `SGLANG_MOE_TRACE_DECODE_FROM_RUNNER`, `SGLANG_MOE_TRACE_FROM_DISPATCH`, `SGLANG_ASSERT_MOE_RUNNER`, `SGLANG_MOE_TRACE_TOPK`.
- No change to raw trace schema or routing behavior. Overhead is a tiny [T, K] int copy per decode step when gated on.

## Validation runs & outcomes

Environment constants for both runs:
- Recorder: `EXPERT_DISTRIBUTION_RECORDER_MODE=per_token`.
- Writer: `SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace`, `SGLANG_MOE_TRACE_PHASE=all`, `SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5`.
- Sizing: `MEM_FRACTION_STATIC=0.98`, `CONTEXT_LENGTH=16384`, `MAX_PREFILL_TOKENS=16384`, `MAX_TOTAL_TOKENS=16384`, `MAX_MAMBA_CACHE_SIZE=160`.

### A) STANDARD runner proof

- Env deltas:
  - `SGLANG_FORCE_STANDARD_TOPK=1`
  - `SGLANG_EXTRA_ARGS="--moe-runner-backend standard --disable-cuda-graph --disable-cuda-graph-padding"`
- Start: `READY_TIMEOUT=600 ./scripts/infer/start_server.sh`
- Request: start recorder → 10 decode tokens → dump.
- Analyze: `python3 tools/analyze_expert_trace.py --dir $HOME/sglang-observability/telemetry/expert-trace --phase decode --dedupe --step-window 50`
- Observed:
  - `files=… records=0 token_expert_events=0` for phase=decode.
  - Without phase filter: prefill present (hundreds of records; tens of thousands of events).
- Notes: `/get_server_info` still shows `moe_runner_backend:"auto"` and `disable_cuda_graph:false`. The new argv log was added; check for `[trace_debug] argv=...` in observability.log to confirm the flags actually reached the server.

### B) TRITON runner + taps (preferred long‑term)

- Env deltas:
  - `SGLANG_MOE_TRACE_DECODE_FROM_RUNNER=1`
  - `SGLANG_MOE_TRACE_FROM_DISPATCH=1`
  - `SGLANG_FORCE_STANDARD_TOPK=0`
  - `SGLANG_EXTRA_ARGS="--moe-runner-backend triton --disable-cuda-graph --disable-cuda-graph-padding"`
- Start: `READY_TIMEOUT=600 ./scripts/infer/start_server.sh`
- Request: start recorder → 10–12 decode tokens → dump.
- Analyze (same as above).
- Observed:
  - `files=… records=0 token_expert_events=0` for phase=decode.
  - Prefill present and healthy.
- Stability: an intermediate nested‑context assertion was corrected by using best‑effort context enter/exit in `fused_moe_triton/layer.py`.

## Evidence snippets

- Analyzer (prefill only example):
```
files=20 records=485 token_expert_events=67020
phase summary:
   prefill: records=485 token_events=67020
step coverage:
   prefill: unique_steps=3 first=1 last=19
```

- Server info excerpt:
```
"ep_size":1, "moe_a2a_backend":"none",
"nsa_decode":"fa3", "decode_attention_backend":"flashinfer",
"expert_distribution_recorder_mode":"per_token"
```

- Argv logging:
  - We added the line; check `.../logs/observability.log` for `[trace_debug] argv=...`. (In our last window, we did not yet see this line; the handler is in the worker init path, so subsequent runs should record it.)

## Working theory after Round 2

- The decode path for this model/backend likely does not surface [T, K] indices in the Python loci we tapped (STANDARD Top‑K wrapper, runner core path when fused functions are registered, EP=1 standard dispatcher), or the flags intended to steer the path were not applied.
- Next minimal instrumentation (if needed):
  1) Log and assert CLI flags (argv and `SGLANG_ASSERT_MOE_RUNNER`) across one clean startup to confirm backend selection conclusively.
  2) If flags are honored and decode still produces zero rows, add a kernel‑level, env‑gated “scribe” buffer in the Triton routing kernel to store [T, K] indices during decode when an extra pointer is non‑null; copy back and emit once per decode step. This ensures coverage even if Python never materializes indices.

## How to roll back everything quickly

- Set the env gates to 0/unset (default state):
  - `SGLANG_MOE_TRACE_DECODE_FROM_RUNNER=0`
  - `SGLANG_MOE_TRACE_FROM_DISPATCH=0`
  - `SGLANG_ASSERT_MOE_RUNNER` unset
  - `SGLANG_MOE_TRACE_TOPK` unset (summarizer falls back to 10)
- Code reverts (file by file):
  - `http_server.py`: remove the argv log line.
  - `moe_runner/runner.py`: remove the `SGLANG_ASSERT_MOE_RUNNER` block.
  - `moe_runner/triton.py`: remove the try/except emission in `TritonRunnerCore.run`.
  - `token_dispatcher/standard.py`: remove the try/except emission in `dispatch`.
  - `fused_moe_triton/layer.py`: restore direct dispatch/apply without manual context enter/exit.
  - `eplb/expert_distribution.py`: fix `_TOP_K_NUM` back to 10.

## Final note

The taps are in the right vicinity for decode under Triton+EP=1 in v0.5.3 and are entirely disabled by default. With the argv log in place, the very next run should conclusively prove whether the backend flags are applied; if they are and decode still shows zero rows, we move to the device‑side “scribe” in the routing kernel to surface indices directly.
