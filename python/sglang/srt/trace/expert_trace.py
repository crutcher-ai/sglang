import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass
class _StepState:
    phase: str
    slots: Optional[List[int]]
    positions: Optional[List[int]]
    seq_lens: Optional[List[int]]


def _forward_mode_to_phase(mode: ForwardMode) -> str:
    if mode.is_decode():
        return "decode"
    if mode.is_prefill():
        return "prefill"
    if mode.is_mixed():
        return "mixed"
    if mode.is_target_verify():
        return "target_verify"
    if mode.is_draft_extend():
        return "draft_extend"
    if mode.is_split_prefill():
        return "split_prefill"
    return "other"


class ExpertTraceWriter:
    """Collects per-step MoE routing decisions for offline analysis."""

    ENV_TRACE_DIR = "SGLANG_MOE_TRACE_DIR"
    ENV_TRACE_PHASE = "SGLANG_MOE_TRACE_PHASE"
    ENV_TRACE_FLUSH_INTERVAL = "SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC"
    ENV_TRACE_DEDUP = "SGLANG_MOE_TRACE_DEDUP"
    ENV_TRACE_SAMPLE_RATE = "SGLANG_MOE_TRACE_SAMPLE_RATE"

    def __init__(
        self,
        trace_dir: Path,
        phase_filter: str = "decode",
        flush_interval_sec: float = 10.0,
        sample_rate: float = 1.0,
    ):
        self._trace_dir = trace_dir
        self._trace_dir.mkdir(parents=True, exist_ok=True)

        phase_filter = phase_filter.lower()
        self._phase_filter = phase_filter
        self._accept_decode = phase_filter in {"decode", "both", "all"}
        self._accept_prefill = phase_filter in {"prefill", "both", "all"}
        self._flush_interval_ns = int(flush_interval_sec * 1e9)
        self._sample_rate = max(0.0, min(sample_rate, 1.0))

        self._run_id = os.getenv("SGL_CONTAINER_RUN_ID", "unknown_run")
        self._session_id = os.getenv("SGL_SERVER_SESSION_ID", "unknown_session")

        self._rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                self._rank = torch.distributed.get_rank()
            except RuntimeError:
                self._rank = 0

        self._records: List[Dict] = []
        self._step_state: Dict[int, _StepState] = {}
        self._lock = threading.Lock()
        self._flush_index = 0
        self._last_flush_ns = time.monotonic_ns()
        self._dedup_enabled = os.getenv(self.ENV_TRACE_DEDUP, "1").lower() not in (
            "0",
            "false",
            "no",
        )
        # Per-step seen set for lightweight de-duplication
        self._seen_per_step: Dict[int, set] = {}

    # ---------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> Optional["ExpertTraceWriter"]:
        trace_dir = os.getenv(cls.ENV_TRACE_DIR)
        if not trace_dir:
            return None

        phase = os.getenv(cls.ENV_TRACE_PHASE, "decode")
        try:
            flush_interval = float(os.getenv(cls.ENV_TRACE_FLUSH_INTERVAL, "10"))
        except ValueError:
            flush_interval = 10.0
        try:
            sample_rate = float(os.getenv(cls.ENV_TRACE_SAMPLE_RATE, "1.0"))
        except ValueError:
            sample_rate = 1.0

        return cls(
            Path(trace_dir),
            phase_filter=phase,
            flush_interval_sec=flush_interval,
            sample_rate=sample_rate,
        )

    @property
    def enabled(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle hooks wired from ExpertDistributionRecorder
    # ------------------------------------------------------------------
    def on_forward_pass_start(self, step_id: int, forward_batch: ForwardBatch):
        phase = _forward_mode_to_phase(forward_batch.forward_mode)

        if not self._phase_is_enabled(phase):
            # Still cache minimal state so that on_select_experts can skip quickly.
            state = _StepState(phase=phase, slots=None, positions=None, seq_lens=None)
        else:
            slots = None
            if forward_batch.req_pool_indices is not None:
                try:
                    slots = forward_batch.req_pool_indices.cpu().tolist()
                except AttributeError:
                    slots = list(forward_batch.req_pool_indices)
                slots = [int(x) for x in slots]
            positions = None
            if forward_batch.positions is not None:
                try:
                    positions = forward_batch.positions.cpu().tolist()
                except AttributeError:
                    positions = list(forward_batch.positions)
                positions = [int(x) for x in positions]
            seq_lens = None
            if forward_batch.seq_lens_cpu is not None:
                seq_lens = [int(x) for x in forward_batch.seq_lens_cpu]
            elif forward_batch.seq_lens is not None:
                try:
                    seq_lens = forward_batch.seq_lens.cpu().tolist()
                except AttributeError:
                    seq_lens = list(forward_batch.seq_lens)
                seq_lens = [int(x) for x in seq_lens]

            state = _StepState(
                phase=phase, slots=slots, positions=positions, seq_lens=seq_lens
            )

        with self._lock:
            self._step_state[step_id] = state
            if self._dedup_enabled:
                self._seen_per_step.setdefault(step_id, set())

    def handle_on_select_experts(
        self,
        step_id: int,
        layer_id: int,
        ids_tensor: torch.Tensor,
        ids_kind: str = "topk_ids",
    ):
        with self._lock:
            state = self._step_state.get(step_id)
            if state is None:
                return

        if not self._phase_is_enabled(state.phase):
            return

        if self._sample_rate < 1.0 and random.random() > self._sample_rate:
            return

        experts_tensor = ids_tensor
        if experts_tensor.is_cuda:
            experts_tensor = experts_tensor.detach().cpu()
        else:
            experts_tensor = experts_tensor.detach()

        if experts_tensor.ndim == 1:
            experts_tensor = experts_tensor.unsqueeze(1)

        experts_list = experts_tensor.tolist()
        tokens = []
        slots = state.slots or []
        positions = state.positions or []
        seq_lens = state.seq_lens or []

        for idx, experts in enumerate(experts_list):
            filtered = [int(ex) for ex in experts if float(ex) >= 0]
            if not filtered:
                continue
            slot_val = slots[idx] if idx < len(slots) else idx
            pos_val = positions[idx] if idx < len(positions) else None
            seq_val = seq_lens[idx] if idx < len(seq_lens) else None

            token_entry = {
                "slot": int(slot_val) if slot_val is not None else None,
                "position": int(pos_val) if pos_val is not None else None,
                "seq_len": int(seq_val) if seq_val is not None else None,
                "experts": filtered,
            }

            if self._dedup_enabled:
                # Deduplicate within the same step by (layer, slot, position, experts)
                key = (
                    int(layer_id),
                    token_entry["slot"],
                    token_entry.get("position"),
                    tuple(filtered),
                )
                seen = self._seen_per_step.get(step_id)
                if seen is not None and key in seen:
                    continue
                if seen is not None:
                    seen.add(key)

            tokens.append(token_entry)

        record = {
            "run_id": self._run_id,
            "session_id": self._session_id,
            "rank": self._rank,
            "phase": state.phase,
            "step": int(step_id),
            "layer": int(layer_id),
            "timestamp_ns": time.time_ns(),
            "ids_kind": ids_kind,
            "topk_dim": int(experts_tensor.shape[1]) if experts_tensor.ndim > 1 else 0,
            "tokens": tokens,
        }

        with self._lock:
            self._records.append(record)
            self._maybe_flush_locked()

    def on_forward_pass_end(self, step_id: int):
        with self._lock:
            self._step_state.pop(step_id, None)
            self._seen_per_step.pop(step_id, None)

    def start_record(self):
        # no-op; state is already empty
        pass

    def stop_record(self):
        self.flush(reason="stop_record")

    def reset(self):
        with self._lock:
            self._step_state.clear()
            self._records.clear()
            self._last_flush_ns = time.monotonic_ns()
            self._seen_per_step.clear()

    def flush(self, reason: str = "manual") -> List[str]:
        with self._lock:
            return self._flush_locked(reason)

    def shutdown(self):
        self.flush(reason="shutdown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _phase_is_enabled(self, phase: str) -> bool:
        if phase == "decode":
            return self._accept_decode
        if phase == "prefill":
            return self._accept_prefill
        if self._phase_filter in {"all", "both"}:
            return True
        return phase == self._phase_filter

    def _maybe_flush_locked(self):
        now = time.monotonic_ns()
        if self._flush_interval_ns <= 0:
            return
        if now - self._last_flush_ns >= self._flush_interval_ns:
            self._flush_locked("interval")
            self._last_flush_ns = now

    def _flush_locked(self, reason: str) -> List[str]:
        if not self._records:
            return []
        self._flush_index += 1
        filename = (
            f"expert_trace_{self._run_id}_{self._session_id}_rank{self._rank}"
            f"_{self._flush_index:05d}.jsonl"
        )
        path = self._trace_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for record in self._records:
                record_with_reason = dict(record)
                record_with_reason["flush_reason"] = reason
                f.write(json.dumps(record_with_reason))
                f.write("\n")

        self._records.clear()
        self._last_flush_ns = time.monotonic_ns()
        return [str(path)]


# Convenience helper for recorder integration


def get_expert_trace_writer() -> Optional[ExpertTraceWriter]:
    return ExpertTraceWriter.from_env()
