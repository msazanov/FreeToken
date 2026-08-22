from __future__ import annotations

import torch
import triton
import triton.language as tl


_FP8_E5M2_MAX = 57344.0
_INT8_MAX = 127.0
_SCALE_EPS = 1.0e-8


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
