"""Reference math for the RTX 2070-proven TQ4-NC KV cache."""

from __future__ import annotations

import functools

import torch

from freetoken.kernel.triton.dsv4.hadamard import hadamard_transform


TQ4_CENTROIDS = (
    -2.7325895710, -2.0690172265, -1.6180463860, -1.2562311973,
    -0.9423404565, -0.6567591185, -0.3880482995, -0.1283950299,
    0.1283950299, 0.3880482995, 0.6567591185, 0.9423404565,
    1.2562311973, 1.6180463860, 2.0690172265, 2.7325895710,
)
TQ4_BOUNDARIES = tuple((TQ4_CENTROIDS[i] + TQ4_CENTROIDS[i + 1]) * 0.5 for i in range(15))


@functools.lru_cache(maxsize=None)
def _cpu_signs(layer_id: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Deterministic Rademacher signs, independent of process RNG state."""
    mask = (1 << 64) - 1
    rows: list[list[float]] = []
    for head in range(num_kv_heads):
        row: list[float] = []
        seed = ((layer_id + 1) * 0x9E3779B97F4A7C15 + (head + 1) * 0xD1B54A32D192ED03) & mask
        for dim in range(head_dim):
            z = (seed + dim * 0x9E3779B97F4A7C15) & mask
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z ^= z >> 31
            row.append(1.0 if z & 1 else -1.0)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


@functools.lru_cache(maxsize=None)
def _device_signs(
    layer_id: int, num_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return _cpu_signs(layer_id, num_kv_heads, head_dim).to(device=device, dtype=dtype)


def tq4_signs(
    layer_id: int, num_kv_heads: int, head_dim: int, *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError(f"tq4-nc Hadamard dimension must be a power of two, got {head_dim}")
    return _device_signs(layer_id, num_kv_heads, head_dim, torch.device(device), dtype)


def _mapped_signs(x: torch.Tensor, layer_id: int, num_kv_heads: int) -> torch.Tensor:
    heads = x.shape[-2]
    if heads % num_kv_heads:
        raise ValueError(f"{heads} heads cannot map evenly to {num_kv_heads} KV heads")
    return tq4_signs(
        layer_id, num_kv_heads, x.shape[-1], device=x.device, dtype=x.dtype
    ).repeat_interleave(heads // num_kv_heads, dim=0)


def randomized_hadamard(
    x: torch.Tensor, *, layer_id: int, num_kv_heads: int, inverse: bool = False
) -> torch.Tensor:
    """Apply the deterministic TQ4 transform ``H D`` (or its inverse ``D H``)."""
    if x.ndim < 2:
        raise ValueError("tq4-nc transform expects a head and feature dimension")
    signs = _mapped_signs(x, layer_id, num_kv_heads)
    signs = signs.view((1,) * (x.ndim - 2) + signs.shape)
    return hadamard_transform(x) * signs if inverse else hadamard_transform(x * signs)


def encode_tq4(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS-normalize, Lloyd-Max quantize, and pack two codes per byte."""
    if values.shape[-1] % 2:
        raise ValueError("tq4-nc requires an even head dimension")
    values_f32 = values.float()
    scales_f32 = values_f32.square().mean(dim=-1).sqrt()
    normalized = values_f32 / scales_f32.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    boundaries = torch.tensor(TQ4_BOUNDARIES, device=values.device, dtype=torch.float32)
    codes = torch.bucketize(normalized, boundaries).to(torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous(), scales_f32.to(torch.bfloat16)


def decode_tq4(packed: torch.Tensor, scales: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    if packed.dtype is not torch.uint8:
        raise TypeError(f"tq4-nc packed data must be uint8, got {packed.dtype}")
    if head_dim % 2 or packed.shape[-1] != head_dim // 2:
        raise ValueError("packed width does not match the requested tq4-nc head dimension")
    codes = torch.empty((*packed.shape[:-1], head_dim), device=packed.device, dtype=torch.long)
    codes[..., 0::2], codes[..., 1::2] = (packed & 0x0F).long(), (packed >> 4).long()
    centroids = torch.tensor(TQ4_CENTROIDS, device=packed.device, dtype=torch.float32)
    return (centroids[codes] * scales.float().unsqueeze(-1)).to(torch.bfloat16)


__all__ = ["TQ4_BOUNDARIES", "TQ4_CENTROIDS", "decode_tq4", "encode_tq4", "randomized_hadamard", "tq4_signs"]
