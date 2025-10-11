import pytest

from sglang.srt.server_args import prepare_server_args
import os
from pathlib import Path


def test_flags_propagate_exactly_no_overrides():
    # Use a dummy model config to avoid touching HF or disk
    model = "/tmp/sgl_test_dummy_model"
    Path(model).mkdir(parents=True, exist_ok=True)
    (Path(model) / "config.json").write_text('{"model_type":"llama","architectures":["LlamaForCausalLM"]}')
    argv = [
        "--model-path", model,
        "--kv-cache-dtype", "fp8_e4m3",
        "--mem-fraction-static", "0.94",
        "--chunked-prefill-size", "4096",
        "--max-mamba-cache-size", "1",
        "--max-prefill-tokens", "272000",
        "--context-length", "272000",
        "--load-format", "dummy",
        "--skip-tokenizer-init",
        # DO NOT set --disable-cuda-graph (defaults to False)
    ]

    sa = prepare_server_args(argv)

    # Assert the intended settings flow through unchanged
    assert sa.mem_fraction_static == 0.94
    assert sa.kv_cache_dtype == "fp8_e4m3"
    assert sa.chunked_prefill_size == 4096
    assert sa.max_mamba_cache_size == 1
    assert sa.max_prefill_tokens == 272000
    assert sa.context_length == 272000
    assert sa.disable_cuda_graph is False
