from __future__ import annotations


def test_moe_benchmark_uses_real_ornith_top8_geometry():
    from benchmarks.tq3_4s_moe_bench import ORNITH_MOE_GEOMETRY

    assert ORNITH_MOE_GEOMETRY == {
        "hidden": 2048,
        "intermediate": 512,
        "top_k": 8,
        "resident_slots": 8,
        "tokens": 1,
    }


def test_moe_source_provenance_covers_dispatch_and_routing_sources():
    from benchmarks.tq3_4s_moe_bench import source_provenance

    sources = source_provenance()

    assert "python/freetoken/kernel/csrc/gguf/moe_vec.cuh" in sources
    assert "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu" in sources
    assert "python/freetoken/kernel/gguf.py" in sources
    assert "python/freetoken/layers/activation.py" in sources
    assert "python/freetoken/moe/fused_q4_0.py" in sources
    assert "benchmarks/tq3_4s_moe_bench.py" in sources
    assert all(len(digest) == 64 for digest in sources.values())
