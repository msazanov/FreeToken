from __future__ import annotations


def test_prefill_sweep_straddles_mmvq_boundary_and_reaches_1024():
    from benchmarks.tq3_4s_prefill_bench import PREFILL_TOKEN_SWEEP

    assert PREFILL_TOKEN_SWEEP == (1, 6, 7, 16, 64, 128, 256, 512, 1024)


def test_prefill_benchmark_labels_dense_and_moe_dispatch_honestly():
    from benchmarks.tq3_4s_prefill_bench import dense_dispatch, moe_dispatch

    assert dense_dispatch(1) == "packed_mmvq"
    assert dense_dispatch(6) == "packed_mmvq"
    assert dense_dispatch(7) == "exact_materialized_dequant_plus_gemm"
    assert dense_dispatch(1024) == "exact_materialized_dequant_plus_gemm"
    assert moe_dispatch(1) == "packed_selected_expert_mmvq"
    assert moe_dispatch(1024) == "packed_selected_expert_mmvq"


def test_prefill_source_provenance_covers_both_dispatch_paths():
    from benchmarks.tq3_4s_prefill_bench import source_provenance

    sources = source_provenance()
    assert "benchmarks/tq3_4s_prefill_bench.py" in sources
    assert "python/freetoken/layers/gguf.py" in sources
    assert "python/freetoken/moe/fused_q4_0.py" in sources
    assert "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu" in sources
    assert "python/freetoken/kernel/csrc/gguf/moe_vec.cuh" in sources
    assert all(len(digest) == 64 for digest in sources.values())
