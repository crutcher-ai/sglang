# MoE Decode Trace — Consolidated Troubleshooting Brief (for GPT‑5‑Pro)

Audience: upstream reviewers with a clean SGLang v0.5.3 tree who can apply a patch bundle, reproduce our runs, and help pinpoint why decode‑time per‑token MoE routing is not being recorded.

Status (2025‑10‑20): Prefill routing is captured reliably; decode remains absent despite disabling CUDA graphs, forcing STANDARD Top‑K, and adding conservative debug taps. This brief consolidates all context, diffs, and exact commands to reproduce before proposing where to instrument next.

## TL;DR

- We added a minimal ExpertTraceWriter + analyzer and wired it to ExpertDistributionRecorder so that per‑step, per‑layer routed experts are flushed to JSONL (token→[experts]).
- Prefill shows up with healthy counts and distributions. Decode shows zero rows across multiple runs, even when CUDA graphs are disabled and STANDARD Top‑K is forced.
- Model: Qwen/Qwen3‑Next‑80B‑A3B‑Thinking‑FP8; TP=1; EP=1 (moe_a2a_backend=none); attention backend flashinfer; decode nsa=fa3.
- We suspect the decode path for this model/attention combo bypasses the Python on_select_experts hook entirely (and possibly also our model‑level emit), or reuse of router decisions differs from prefill.

## What we changed (high‑level)

1) Writer + Recorder glue
   - New `python/sglang/srt/trace/expert_trace.py` and export in `trace/__init__.py`.
   - Recorder (`eplb/expert_distribution.py`) forwards STANDARD/NPU top‑k ids to the writer, includes trace files in HTTP dump.
   - Optional writer de‑dup per step (env `SGLANG_MOE_TRACE_DEDUP`, default on).

2) Top‑K hook resilience
   - `layers/moe/topk.py`: add debug env `SGLANG_FORCE_STANDARD_TOPK=1` that forces STANDARD output format so `get_global_expert_distribution_recorder().on_select_experts(topk_ids=...)` is always called.

3) Decode‑safe fallback taps (gated, conservative)
   - `layers/moe/token_dispatcher/deepep.py`: in `dispatch_b()` (normal & low‑latency), when `SGLANG_MOE_TRACE_FROM_DISPATCH=1`, emit `on_select_experts(topk_ids=topk_idx)`. No‑op when EP=1/deepep=none.
   - `models/qwen3_moe.py`: in `op_select_experts`, when `SGLANG_MOE_TRACE_FORCE_EMIT=1`, emit `on_select_experts(topk_ids=state.topk_idx_local)` right after computing indices.

4) Ops / UX
   - Launcher (`scripts/infer/start_server.sh`): forwards relevant envs; logs a single ready JSON line; validates `/trace_probe`.
   - Helper (`scripts/start_observable_container.sh`): launches Jaeger v2 + exporters; writes per‑run manifest; not essential for reproducing decode issue but is part of our environment.
   - Analyzer (`tools/analyze_expert_trace.py`): `--phase`, `--dedupe`, `--step-window` to quickly verify decode coverage.

## Exact environment

- Hardware: NVIDIA GH200 (96GB HBM); CUDA 12.8+; torch 2.8; single GPU.
- Model path: `/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`.
- Server config typical overrides: `MEM_FRACTION_STATIC≈0.98`, `CONTEXT_LENGTH=16384`, `MAX_*_TOKENS=16384`, `MAX_MAMBA_CACHE_SIZE=160`, `KV_CACHE_DTYPE=fp8_e4m3`.
- Recorder mode: `EXPERT_DISTRIBUTION_RECORDER_MODE=per_token`.
- Writer envs: `SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace`, `SGLANG_MOE_TRACE_PHASE=all`, `SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5`.
- Attention backend per `/get_server_info`: `prefill_attention_backend=flashinfer`, `decode_attention_backend=flashinfer`, `nsa_decode=fa3`.
- For DeepEP: EP size is 1 in our tests (moe_a2a_backend=none); dispatch taps are inert in this configuration.

## Reproduction (clean sequence)

1) Apply patch bundle to a clean v0.5.3 tree (see below for patch file).
2) Start the model server with a small context to keep HBM comfortable:

```
# Standard path (preferred A)
export EXPERT_DISTRIBUTION_RECORDER_MODE=per_token
export SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace
export SGLANG_MOE_TRACE_PHASE=all
export SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5
export SGLANG_FORCE_STANDARD_TOPK=1
export ENABLE_TRACE=1
export OTEL_TRACES_SAMPLER=always_on
export MEM_FRACTION_STATIC=0.98
export CONTEXT_LENGTH=16384
export MAX_PREFILL_TOKENS=16384
export MAX_TOTAL_TOKENS=16384
export MAX_MAMBA_CACHE_SIZE=160
export SGLANG_EXTRA_ARGS="--moe-runner-backend triton"
READY_TIMEOUT=600 ./scripts/infer/start_server.sh
```

3) Verify health & tracing threads:
```
curl -s http://127.0.0.1:30000/trace_probe
```

4) Start recording, send one short request (8–12 decode tokens), dump:
```
curl -sSf -X POST http://127.0.0.1:30000/start_expert_distribution_record
curl -sSf -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"List three prime numbers."}],"temperature":0.2,"max_tokens":10}' > /dev/null
curl -sSf http://127.0.0.1:30000/dump_expert_distribution_record > /dev/null
```

5) Analyze:
```
python3 tools/analyze_expert_trace.py --dir $HOME/sglang-observability/telemetry/expert-trace \
  --phase decode --dedupe --step-window 50
```

Expected (buggy): `files>0 records=0 token_expert_events=0` for phase=decode. Without `--phase`, you should see prefill records and per‑layer distributions.

6) Try eliminating graph influence (B):
```
./scripts/infer/stop_server.sh
export SGLANG_FORCE_STANDARD_TOPK=0
export SGLANG_MOE_TRACE_FROM_DISPATCH=0
export SGLANG_MOE_TRACE_FORCE_EMIT=1
export SGLANG_EXTRA_ARGS="--disable-cuda-graph --disable-cuda-graph-padding"
READY_TIMEOUT=600 ./scripts/infer/start_server.sh
<repeat step 4 & 5>
```

Expected (still buggy): decode absent.

## Evidence (summaries from our runs)

Analyzer (without phase filter):

```
files=12 records=289 token_expert_events=37500
phase summary:
   prefill: records=289 token_events=37500
step coverage:
   prefill: unique_steps=3 first=1 last=19
```

Newest JSONL tail (prefill only):

```
{"phase":"prefill","step":10,"layer":46, ...
  "tokens":[{"slot":1,"position":0,"seq_len":15,"experts":[118,168,93,88,427,415,106,150,364,403]}, ...]}
```

Server info (abbrev):

```
ep_size=1, moe_a2a_backend="none", attention_backend=flashinfer, nsa_decode="fa3",
mem_fraction_static=0.98, context_length=16384, kv_cache_dtype="fp8_e4m3",
expert_distribution_recorder_mode="per_token"
```

## Code map (where we hooked)

- Recorder/Writer glue: `python/sglang/srt/eplb/expert_distribution.py` → `python/sglang/srt/trace/expert_trace.py`
  - on forward pass start/end and on_select_experts events.
- STANDARD path (Top‑K): `python/sglang/srt/layers/moe/topk.py`
  - Calls `get_global_expert_distribution_recorder().on_select_experts(topk_ids)`; `SGLANG_FORCE_STANDARD_TOPK` bypasses TRITON/BYPASSED formats.
- Dispatch taps (optional, DeepEP only): `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
  - Emits `on_select_experts(topk_idx)` in `dispatch_b()` when `SGLANG_MOE_TRACE_FROM_DISPATCH=1`.
- Model‑level force emit (optional): `python/sglang/srt/models/qwen3_moe.py`
  - In `op_select_experts`, when `SGLANG_MOE_TRACE_FORCE_EMIT=1` emit `on_select_experts(topk_idx_local)`.

## Why this might still miss decode (our current hypotheses)

1) For the Qwen3‑Next‑80B‑A3B FP8 + flashinfer decode path (`nsa_decode=fa3`, EP=1), single‑token decode may not call `op_select_experts` at all (e.g., reusing prefill routing or flowing through a fused kernel path that does not revisit the Python Top‑K).

2) If decode reuses cached router decisions, the authoritative mapping from token→experts might live in a compacted buffer that the MLP consumes directly; our current tap points (STANDARD `TopK` or model `op_select_experts`) would be too early.

3) Our phase labeling is not the issue: writer accepts `phase=all`, and analyzer `--phase decode` still finds zero rows; when not filtered, only `prefill` is present, suggesting writer was simply never called for decode.

## What we’re asking you to do

1) Apply the patch bundle (below) onto v0.5.3 and confirm you reproduce the absence of decode rows with the runs above.

2) Identify the decode‑time locus for per‑token expert routing in this model configuration (TP=1, EP=1, nsa_decode=fa3):
   - Does decode call `Qwen3MoeSparseMoeBlock.op_select_experts`? If not, where is the routing computed/consumed?
   - If routing is reused from prefill, where is the per‑token mapping stored and can we serialize it per step?
   - If decode uses a fused path, is there a practical place (device or Python) to read out the top‑k indices per token without a large overhead?

3) Propose the minimal robust hook for decode that works for this attention/backend/model mix.

## Patch bundle

We collected a consolidated patch of all relevant changes versus v0.5.3 at:

- `contrib/moe_decode_trace_consolidated.patch`

Apply on top of a clean v0.5.3 tree:

```
git apply contrib/moe_decode_trace_consolidated.patch
```

If you prefer a branch: cherry‑pick the diffs touching these paths:

- `python/sglang/srt/trace/**`
- `python/sglang/srt/eplb/expert_distribution.py`
- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/tokenizer_communicator_mixin.py`
- `scripts/infer/start_server.sh`
- `tools/analyze_expert_trace.py`

## Acceptance criteria (what “fixed” looks like)

- After one 8–12 token completion, analyzer shows non‑zero `phase=decode` rows:
  - `files>=1`, `records>0`, `token_expert_events>0`
  - `step coverage` lists at least one decode step
  - JSONLs contain `{ "phase":"decode", "tokens":[{"experts":[...]}] }` rows for the active layers.

## Appendix — Snippets & Logs

- Ready JSON example:

```
{"schema_version":1,"run_id":"...","server_session_id":"...","health":"ready","port":30000,
 "manifest_host_path":".../manifest.json","log_file":"/telemetry/.../observability.log",
 "sizing":{"tp_size":1,"mem_fraction_static":0.98,"context_length":16384,
            "max_prefill_tokens":16384,"max_total_tokens":16384,"max_mamba_cache_size":160}}
```

- /get_server_info keys confirming environment:

```
"kv_cache_dtype":"fp8_e4m3","attention_backend":"flashinfer",
"nsa_decode":"fa3","ep_size":1,"moe_a2a_backend":"none",
"expert_distribution_recorder_mode":"per_token"
```
