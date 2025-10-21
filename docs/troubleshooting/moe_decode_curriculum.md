# Quick Onboarding: MoE Decode Trace Investigation

## Objective
Bring a fresh reviewer up to speed (in one short session) so they can help us capture and analyse per-token MoE routing during **decode**, and propose ways to log deeper expert rankings for affinity work.

## 1. Minimal Background Reading (≈15 min)
1. **Telemetry pipeline** – skim `docs/troubleshooting/moe_per_layer_routing_research_prompt.md` §0–5 to understand how the helper, server launcher, and `ExpertTraceWriter` integrate.
2. **Decode gap recap** – read `docs/troubleshooting/moe_decode_gap_research_prompt.md` (esp. TL;DR + latest experiment note). This logs the failed attempts (graphs off, recorder started) so the reviewer knows decode is still missing.
3. **Raw data sample** – open `scratch_expert_tokens.json` (generated from the latest prefill trace) to see the JSON schema we use for downstream affinity calculations (token → layer → expert IDs).

## 2. Hands-on Verification (≈20 min)
1. Start helper: `./scripts/start_observable_container.sh`.
2. Launch server with tracing: `./scripts/infer/start_server.sh` (envs already wired for tracing; new reviewers should confirm `SGLANG_EXTRA_ARGS` includes the current knobs).
3. Start recording via API: `curl -sSf -X POST http://127.0.0.1:30000/start_expert_distribution_record`.
4. Issue a short request (e.g., 8 decode tokens); dump recorder; inspect new `expert_trace_*` files.
   - Run `python3 tools/analyze_expert_trace.py --dir … --run-id … --session-id … --top 5`.
   - Confirm phase summary (expect `prefill` only at present) and store results for later comparison once decode is fixed.

## 3. Debug Focus (≈30 min)
1. Compare our fork with upstream to locate decode routing paths (TopK variants, DeepEP dispatch, CUDA graph capture sites).
2. Identify where to hook decode without relying on Python callbacks that graphs may skip – possibilities include:
   - device-side scribe buffers in Triton TopK kernels,
   - alternative recorder hooks (`on_deepep_dispatch_*`),
   - ensuring decode worker actually runs in `per_token` mode.
3. Explore “shadow top-k” logging (capture ranked experts beyond the dispatched top‑10) per GPT‑5 guidance; document feasibility.

## 4. Deliverables for Follow-up Discussion
- Confirmed hypothesis (or new evidence) about decode bypassing the recorder.
- Concrete proposal (code locations, signatures) for capturing decode per-token exports.
- Plan for logging deeper rankings (top‑K′ or router logits) gated by debug envs.

Keep notes concise; all findings feed into the troubleshooting docs above before making code changes.
