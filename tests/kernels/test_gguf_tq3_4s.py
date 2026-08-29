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
