import argparse
from sglang.srt.server_args import ServerArgs
from sglang.compile_deep_gemm import CompileArgs, refine_server_args


def test_compile_refine_preserves_server_defaults_and_sets_compile_policy():
    # Build baseline ServerArgs from CLI-like inputs (policy is set by caller)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw = parser.parse_args([
        "--model-path", "/models/placeholder",
        "--host", "127.0.0.1",
        "--port", "30000",
        "--tp", "1",
        "--kv-cache-dtype", "fp8_e4m3",
        "--mem-fraction-static", "0.94",
        "--chunked-prefill-size", "4096",
        "--max-mamba-cache-size", "1",
        "--max-prefill-tokens", "272000",
        "--context-length", "272000",
        "--max-total-tokens", "272000",
    ])
    sa = ServerArgs.from_cli_args(raw)

    # Refine for compile stage and assert compile-specific policy applied
    ca = CompileArgs(timeout=3600)
    refine_server_args(sa, ca)

    assert sa.kv_cache_dtype == "fp8_e4m3"
    assert sa.mem_fraction_static == 0.94
    assert sa.chunked_prefill_size == 4096
    assert sa.max_mamba_cache_size == 1
    assert sa.max_prefill_tokens == 272000
    assert sa.context_length == 272000
    assert sa.max_total_tokens == 272000

    # Compile policy knobs
    assert sa.disable_cuda_graph is True
    assert sa.enable_torch_compile is False
    assert sa.watchdog_timeout == 3600
