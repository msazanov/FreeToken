from __future__ import annotations


def test_summarize_samples_keeps_raw_order_and_interpolates_percentiles():
    from benchmarks.tq3_4s_kernel_bench import summarize_samples

    samples = [4.0, 1.0, 3.0, 2.0]

    assert summarize_samples(samples) == {
        "count": 4,
        "min_ms": 1.0,
        "p50_ms": 2.5,
        "mean_ms": 2.5,
        "p95_ms": 3.8499999999999996,
        "max_ms": 4.0,
        "raw_ms": samples,
    }


def test_real_ornith_expert_shapes_cover_gate_up_and_down():
    from benchmarks.tq3_4s_kernel_bench import ORNITH_EXPERT_SHAPES

    assert ORNITH_EXPERT_SHAPES == {
        "expert_gate_up": (512, 2048),
        "expert_down": (2048, 512),
    }


def test_source_provenance_hashes_benchmark_and_kernel_inputs():
    from benchmarks.tq3_4s_kernel_bench import source_provenance

    provenance = source_provenance()

    assert set(provenance) == {
        "benchmarks/tq3_4s_kernel_bench.py",
        "python/freetoken/kernel/csrc/gguf/ggml-common.h",
        "python/freetoken/kernel/csrc/gguf/gguf_kernel.cu",
        "python/freetoken/kernel/csrc/gguf/mmvq.cuh",
        "python/freetoken/kernel/csrc/gguf/vecdotq.cuh",
        "python/freetoken/models/gguf/dequant.py",
    }
    assert all(len(digest) == 64 for digest in provenance.values())


def test_benchmark_alternates_fast_and_fallback_order():
    from benchmarks.tq3_4s_kernel_bench import operation_order

    assert operation_order(0) == ("packed_mmvq", "exact_cuda_dequant_plus_torch_mm")
    assert operation_order(1) == ("exact_cuda_dequant_plus_torch_mm", "packed_mmvq")


def test_quality_acceptance_is_explicit_and_machine_readable():
    from benchmarks.tq3_4s_kernel_bench import quality_acceptance

    good = quality_acceptance(
        {"cosine": 0.99995, "relative_l2": 0.006},
        min_cosine=0.9999,
        max_relative_l2=0.01,
    )
    bad = quality_acceptance(
        {"cosine": 0.99995, "relative_l2": 0.02},
        min_cosine=0.9999,
        max_relative_l2=0.01,
    )

    assert good == {
        "passed": True,
        "min_cosine": 0.9999,
        "max_relative_l2": 0.01,
    }
    assert bad["passed"] is False
