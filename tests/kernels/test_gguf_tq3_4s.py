"""SM75 parity gate for materialized TurboQuant ``TQ3_4S`` dequantization.

This is deliberately separate from MMVQ/MMQ/MoE coverage.  A correct materialized
dequantizer is the numeric authority those faster kernels must match; advertising a
matrix kernel before this gate passes would let a wrong WHT or approximate centroid
table produce fluent-looking corruption.
"""

from __future__ import annotations

import math

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason="needs the target SM75 CUDA device",
)


def _pack_same_index(index: int) -> bytes:
    packed = sum(index << (3 * lane) for lane in range(8))
    return packed.to_bytes(3, "little")


def _adversarial_and_random_rows() -> torch.Tensor:
    """Three rows of three blocks with deterministic, legal type-46 bytes."""
    # Centroid[4] is +0.230106 in the authoritative codebook but +14/59.012 in the
    # fork's approximate DP4A path.  A constant transformed block exposes accidental
    # reuse of that faster codebook before the general random cases run.
    discriminator = bytes([0x60] * 4) + _pack_same_index(4) * 4

    generator = torch.Generator().manual_seed(20260830)
    random_blocks = torch.randint(0, 256, (7, 16), dtype=torch.uint8, generator=generator)
    # E3M5 byte zero is valid, but force a broad, finite scale range in every group.
    scale_bank = torch.tensor([0x20, 0x5F, 0x80, 0xFF], dtype=torch.uint8)
    random_blocks[:, :4] = torch.stack([scale_bank.roll(i) for i in range(7)])

    # One transformed subgroup is exactly zero while the other three remain live.
    zero_scale = bytes([0x00, 0x20, 0x80, 0xFF]) + _pack_same_index(7) * 4

    blocks = torch.cat(
        (
            torch.tensor(list(discriminator + zero_scale), dtype=torch.uint8).view(2, 16),
            random_blocks,
        ),
        dim=0,
    )
    return blocks.reshape(3, 3 * 16).contiguous()


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 1e-6), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)],
)
def test_tq3_4s_cuda_materialized_dequant_matches_cpu_oracle_on_sm75(dtype, atol):
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import GGML_TQ3_4S, dequant_tq3_4s

    assert torch.cuda.get_device_capability() == (7, 5)
    packed = _adversarial_and_random_rows()
    expected = dequant_tq3_4s(packed.flatten(), dtype).reshape(3, 96).cuda()

    actual = ggml_dequantize(
        packed.cuda(), GGML_TQ3_4S, 3, 96, dtype
    )
    torch.cuda.synchronize()

    assert actual.shape == (3, 96)
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)

    # Independent literal discriminator: a constant transformed block maps to
    # one non-zero coefficient after inverse WHT. This does not call the CPU oracle.
    literal = torch.zeros(32, dtype=dtype, device="cuda")
    literal[0] = math.sqrt(32.0) * 0.230106 / 64.0
    torch.testing.assert_close(actual[0, :32], literal, rtol=0, atol=atol)


def _approximate_transformed_weight(packed: torch.Tensor) -> torch.Tensor:
    """Decode the donor's DP4A levels but deliberately leave weights WHT-rotated."""
    from freetoken.models.gguf.dequant import decode_tq3_e3m5

    blocks = packed.reshape(-1, 16)
    scales = decode_tq3_e3m5(blocks[:, :4])
    groups = blocks[:, 4:].reshape(-1, 4, 3).to(torch.int32)
    words = groups[..., 0] | (groups[..., 1] << 8) | (groups[..., 2] << 16)
    shifts = (torch.arange(8, dtype=torch.int32) * 3).view(1, 1, 8)
    indices = (words.unsqueeze(-1) >> shifts) & 7
    levels = torch.tensor([-113, -73, -42, -14, 13, 41, 72, 112], dtype=torch.float32)
    codebook = levels * 0.017704291602768495
    return (codebook[indices.long()] * scales.unsqueeze(-1)).reshape(packed.shape[0], -1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tq3_4s_mmvq_rotates_activation_and_matches_materialized_reference(dtype):
    """The fast approximate DP4A path must preserve the original-domain matvec."""
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
    from freetoken.models.gguf.dequant import GGML_TQ3_4S, dequant_tq3_4s

    out_features, in_features, vectors = 64, 512, 2
    generator = torch.Generator().manual_seed(20260831)
    packed = torch.randint(
        0,
        256,
        (out_features, in_features // 32 * 16),
        dtype=torch.uint8,
        generator=generator,
    )
    # Include an exactly-zero subgroup so MMVQ, not just materialized dequant,
    # proves that the E3M5 zero byte cannot leak stale values into DP4A.
    scale_bank = torch.tensor([0x00, 0x5F, 0x80, 0xFF], dtype=torch.uint8)
    packed.reshape(-1, 16)[:, :4] = torch.stack(
        [scale_bank.roll(i) for i in range(packed.numel() // 16)]
    )
    x = (torch.randn(vectors, in_features, generator=generator) * 0.5).to(dtype)

    dense = dequant_tq3_4s(packed.flatten(), torch.float32).reshape(out_features, in_features)
    reference = x.float() @ dense.t()
    actual = ggml_mul_mat_vec_a8(
        packed.cuda(), x.cuda(), GGML_TQ3_4S, out_features
    ).float().cpu()

    relative_l2 = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )

    # Negative control: the packed values live in the transformed domain. Dotting
    # them with the raw activation is mathematically wrong and must be much worse.
    wrong = x.float() @ _approximate_transformed_weight(packed).t()
    wrong_relative_l2 = torch.linalg.vector_norm(wrong - reference) / torch.linalg.vector_norm(reference)

    assert cosine.item() > 0.9999
    assert relative_l2.item() < 0.01
    assert wrong_relative_l2.item() > relative_l2.item() * 100


def test_tq3_4s_mmvq_rejects_partial_blocks_and_unsafe_views():
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
    from freetoken.models.gguf.dequant import GGML_TQ3_4S

    rows, columns = 2, 32
    packed_bytes = columns // 2
    weight = torch.zeros(rows, packed_bytes, dtype=torch.uint8, device="cuda")
    activation = torch.zeros(1, columns, dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="multiple of 32"):
        ggml_mul_mat_vec_a8(weight, torch.zeros(1, 33, device="cuda"), GGML_TQ3_4S, rows)

    noncontiguous_weight = torch.zeros(
        rows, packed_bytes * 2, dtype=torch.uint8, device="cuda"
    )[:, ::2]
    assert not noncontiguous_weight.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        ggml_mul_mat_vec_a8(noncontiguous_weight, activation, GGML_TQ3_4S, rows)

    noncontiguous_activation = torch.zeros(1, columns * 2, device="cuda")[:, ::2]
    assert not noncontiguous_activation.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        ggml_mul_mat_vec_a8(weight, noncontiguous_activation, GGML_TQ3_4S, rows)

    misaligned_storage = torch.zeros(
        rows * packed_bytes + 1, dtype=torch.uint8, device="cuda"
    )
    misaligned_weight = misaligned_storage[1:].view(rows, packed_bytes)
    assert misaligned_weight.is_contiguous()
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        ggml_mul_mat_vec_a8(misaligned_weight, activation, GGML_TQ3_4S, rows)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tq3_4s_large_batch_prefill_fallback_matches_exact_reference(dtype):
    """Batch > 6 must use exact materialization, never an unsupported MMQ case."""
    from freetoken.layers.gguf import _MMVQ_SAFE, fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_TQ3_4S, dequant_tq3_4s

    rows, columns, tokens = 64, 512, _MMVQ_SAFE + 1
    generator = torch.Generator().manual_seed(20260907)
    packed = torch.randint(
        0,
        256,
        (rows, columns // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    scale_bank = torch.tensor([0x00, 0x5F, 0x80, 0xFF], dtype=torch.uint8)
    packed.reshape(-1, 16)[:, :4] = torch.stack(
        [scale_bank.roll(i) for i in range(packed.numel() // 16)]
    )
    x = (torch.randn(tokens, columns, generator=generator) * 0.25).to(dtype)

    dense = dequant_tq3_4s(packed.flatten(), torch.float32).reshape(rows, columns)
    expected = x.float() @ dense.t()
    actual = fused_mul_mat_gguf(
        x.cuda(), packed.cuda(), GGML_TQ3_4S
    ).float().cpu()

    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine.item() > 0.999
    assert relative_l2.item() < 0.02
