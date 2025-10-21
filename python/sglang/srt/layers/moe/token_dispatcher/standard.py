from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    hidden_states: torch.Tensor
    topk_output: TopKOutput

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """Standard combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)


class StandardDispatcher(BaseDispatcher):

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        # Optional: emit per-token expert ids when EP=1/standard path, gated by env
        try:
            if get_bool_env_var("SGLANG_MOE_TRACE_FROM_DISPATCH"):
                rec = get_global_expert_distribution_recorder()
                rec.on_select_experts(topk_ids=topk_output.topk_ids)
        except Exception:
            pass

        return StandardDispatchOutput(
            hidden_states=hidden_states, topk_output=topk_output
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if isinstance(combine_input, StandardCombineInput):
            return combine_input.hidden_states
        else:
            # TODO: this branch should be removed in the future
            assert isinstance(combine_input, torch.Tensor)
            return combine_input
