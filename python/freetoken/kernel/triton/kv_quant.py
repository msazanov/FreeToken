from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kvcache.tq4 import TQ4_BOUNDARIES, encode_tq4, randomized_hadamard


_FP8_E5M2_MAX = 57344.0
_INT8_MAX = 127.0
_SCALE_EPS = 1.0e-8


@functools.lru_cache(maxsize=None)
def _tq4_boundaries(device_index: int) -> torch.Tensor:
    return torch.tensor(
        (*TQ4_BOUNDARIES, float("inf")), device=torch.device("cuda", device_index),
        dtype=torch.float32,
    )


@triton.jit
def _tq4_code(values, boundaries):
    return tl.sum((values[:, None] > boundaries[None, :]).to(tl.int32), axis=1)


@triton.jit
def _store_tq4_nc_kv_kernel(
    k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, indices_ptr,
    boundaries_ptr, stride_kt, stride_vt, stride_kcs, stride_kch, stride_vcs,
    stride_vch, stride_kss, stride_vss, D: tl.constexpr, PACKED_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token, kv_head = tl.program_id(0), tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    source_offset = kv_head * D + offs_d
    slot = tl.load(indices_ptr + token)
    boundaries = tl.load(boundaries_ptr + tl.arange(0, 16))
    k = tl.load(k_ptr + token * stride_kt + source_offset, mask=mask_d, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + token * stride_vt + source_offset, mask=mask_d, other=0.0).to(tl.float32)
    k_scale, v_scale = tl.sqrt(tl.sum(k * k, axis=0) / D), tl.sqrt(tl.sum(v * v, axis=0) / D)
    k_safe, v_safe = tl.maximum(k_scale, 1.0e-8), tl.maximum(v_scale, 1.0e-8)
    offs_p = tl.arange(0, PACKED_D)
    even, odd = 2 * offs_p, 2 * offs_p + 1
    k_even = tl.load(k_ptr + token * stride_kt + kv_head * D + even, mask=even < D, other=0.0).to(tl.float32) / k_safe
    k_odd = tl.load(k_ptr + token * stride_kt + kv_head * D + odd, mask=odd < D, other=0.0).to(tl.float32) / k_safe
    v_even = tl.load(v_ptr + token * stride_vt + kv_head * D + even, mask=even < D, other=0.0).to(tl.float32) / v_safe
    v_odd = tl.load(v_ptr + token * stride_vt + kv_head * D + odd, mask=odd < D, other=0.0).to(tl.float32) / v_safe
    tl.store(k_cache_ptr + slot * stride_kcs + kv_head * stride_kch + offs_p, _tq4_code(k_even, boundaries) | (_tq4_code(k_odd, boundaries) << 4))
    tl.store(v_cache_ptr + slot * stride_vcs + kv_head * stride_vch + offs_p, _tq4_code(v_even, boundaries) | (_tq4_code(v_odd, boundaries) << 4))
    tl.store(k_scale_ptr + slot * stride_kss + kv_head, k_scale.to(tl.bfloat16))
    tl.store(v_scale_ptr + slot * stride_vss + kv_head, v_scale.to(tl.bfloat16))


@triton.jit
def _store_quantized_kv_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    stride_kt,
    stride_vt,
    stride_kcs,
    stride_kch,
    stride_vcs,
    stride_vch,
    stride_kss,
    stride_vss,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_INT8: tl.constexpr,
):
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    source_offset = kv_head * D + offs_d
    slot = tl.load(indices_ptr + token)

    k = tl.load(k_ptr + token * stride_kt + source_offset, mask=mask_d, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + token * stride_vt + source_offset, mask=mask_d, other=0.0).to(tl.float32)

    limit = 127.0 if IS_INT8 else 57344.0
    k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) / limit, 1.0e-8)
    v_scale = tl.maximum(tl.max(tl.abs(v), axis=0) / limit, 1.0e-8)
    k_quant = tl.maximum(tl.minimum(k / k_scale, limit), -limit)
    v_quant = tl.maximum(tl.minimum(v / v_scale, limit), -limit)

    tl.store(
        k_cache_ptr + slot * stride_kcs + kv_head * stride_kch + offs_d,
        k_quant,
        mask=mask_d,
    )
    tl.store(
        v_cache_ptr + slot * stride_vcs + kv_head * stride_vch + offs_d,
        v_quant,
        mask=mask_d,
    )
    tl.store(k_scale_ptr + slot * stride_kss + kv_head, k_scale.to(tl.bfloat16))
    tl.store(v_scale_ptr + slot * stride_vss + kv_head, v_scale.to(tl.bfloat16))


def _store_quantized_kv_torch(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Reference path for CPU tests; CUDA uses the fused Triton kernel below."""
    num_kv_heads, head_dim = k_cache.shape[1:]
    k_vectors = k.view(k.shape[0], num_kv_heads, head_dim)
    v_vectors = v.view(v.shape[0], num_kv_heads, head_dim)
    limit = _INT8_MAX if k_cache.dtype is torch.int8 else _FP8_E5M2_MAX
    k_f32 = k_vectors.float()
    v_f32 = v_vectors.float()
    k_factors = (k_f32.abs().amax(dim=-1) / limit).clamp_min(_SCALE_EPS)
    v_factors = (v_f32.abs().amax(dim=-1) / limit).clamp_min(_SCALE_EPS)
    rows = indices.to(device=k_cache.device, dtype=torch.long)
    k_cache.index_copy_(0, rows, (k_f32 / k_factors.unsqueeze(-1)).to(k_cache.dtype))
    v_cache.index_copy_(0, rows, (v_f32 / v_factors.unsqueeze(-1)).to(v_cache.dtype))
    k_scale.index_copy_(0, rows, k_factors.to(k_scale.dtype))
    v_scale.index_copy_(0, rows, v_factors.to(v_scale.dtype))


def store_quantized_kv(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Quantize and scatter BF16 K/V vectors into FP8 E5M2 or INT8 cache rows."""
    assert k_cache.dtype in (torch.float8_e5m2, torch.int8)
    assert k_cache.shape == v_cache.shape
    assert k_cache.dim() == 3
    assert k_scale.shape == k_cache.shape[:2]
    assert v_scale.shape == k_cache.shape[:2]
    assert k.shape[0] == v.shape[0] == indices.numel()
    assert k.numel() == v.numel() == indices.numel() * k_cache.shape[1] * k_cache.shape[2]

    if not k_cache.is_cuda:
        _store_quantized_kv_torch(
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            indices=indices,
            k=k,
            v=v,
        )
        return

    k = k.contiguous().view(k.shape[0], -1)
    v = v.contiguous().view(v.shape[0], -1)
    block_d = triton.next_power_of_2(k_cache.shape[-1])
    _store_quantized_kv_kernel[(k.shape[0], k_cache.shape[1])](
        k,
        v,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        indices,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_scale.stride(0),
        v_scale.stride(0),
        D=k_cache.shape[-1],
        BLOCK_D=block_d,
        IS_INT8=k_cache.dtype is torch.int8,
        num_warps=4,
    )


def _store_tq4_nc_kv_torch(
    *, k_cache: torch.Tensor, v_cache: torch.Tensor, k_scale: torch.Tensor,
    v_scale: torch.Tensor, indices: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    layer_id: int, head_dim: int, inputs_are_transformed: bool,
) -> None:
    num_kv_heads = k_cache.shape[1]
    k_vectors = k.contiguous().view(k.shape[0], num_kv_heads, head_dim)
    v_vectors = v.contiguous().view(v.shape[0], num_kv_heads, head_dim)
    transformed_k = k_vectors if inputs_are_transformed else randomized_hadamard(
        k_vectors, layer_id=layer_id, num_kv_heads=num_kv_heads
    )
    transformed_v = v_vectors if inputs_are_transformed else randomized_hadamard(
        v_vectors, layer_id=layer_id, num_kv_heads=num_kv_heads
    )
    packed_k, scales_k = encode_tq4(transformed_k)
    packed_v, scales_v = encode_tq4(transformed_v)
    rows = indices.to(device=k_cache.device, dtype=torch.long)
    k_cache.index_copy_(0, rows, packed_k)
    v_cache.index_copy_(0, rows, packed_v)
    k_scale.index_copy_(0, rows, scales_k)
    v_scale.index_copy_(0, rows, scales_v)


def store_tq4_nc_kv(
    *, k_cache: torch.Tensor, v_cache: torch.Tensor, k_scale: torch.Tensor,
    v_scale: torch.Tensor, indices: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    layer_id: int, head_dim: int, inputs_are_transformed: bool = False,
) -> None:
    """TQ4-pack and scatter K/V rows; CUDA path is native Triton, CPU is an oracle."""
    assert k_cache.dtype is torch.uint8 and v_cache.dtype is torch.uint8
    assert k_cache.shape == v_cache.shape and k_cache.dim() == 3
    assert head_dim % 2 == 0 and k_cache.shape[-1] == head_dim // 2
    assert k_scale.shape == v_scale.shape == k_cache.shape[:2]
    assert k.shape[0] == v.shape[0] == indices.numel()
    assert k.numel() == v.numel() == indices.numel() * k_cache.shape[1] * head_dim
    if not k_cache.is_cuda:
        _store_tq4_nc_kv_torch(
            k_cache=k_cache, v_cache=v_cache, k_scale=k_scale, v_scale=v_scale,
            indices=indices, k=k, v=v, layer_id=layer_id, head_dim=head_dim,
            inputs_are_transformed=inputs_are_transformed,
        )
        return
    num_heads = k_cache.shape[1]
    k_vectors = k.contiguous().view(k.shape[0], num_heads, head_dim)
    v_vectors = v.contiguous().view(v.shape[0], num_heads, head_dim)
    transformed_k = (k_vectors if inputs_are_transformed else randomized_hadamard(
        k_vectors, layer_id=layer_id, num_kv_heads=num_heads
    )).contiguous().view(k.shape[0], -1)
    transformed_v = (v_vectors if inputs_are_transformed else randomized_hadamard(
        v_vectors, layer_id=layer_id, num_kv_heads=num_heads
    )).contiguous().view(v.shape[0], -1)
    boundaries = _tq4_boundaries(
        k.device.index if k.device.index is not None else torch.cuda.current_device()
    )
    _store_tq4_nc_kv_kernel[(k.shape[0], num_heads)](
        transformed_k, transformed_v, k_cache, v_cache, k_scale, v_scale, indices, boundaries,
        transformed_k.stride(0), transformed_v.stride(0), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1), k_scale.stride(0), v_scale.stride(0),
        D=head_dim, PACKED_D=head_dim // 2, BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
