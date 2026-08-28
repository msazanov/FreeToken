"""Tiled Triton RoPE (provenance: rope<-lightllm).

Adapted from lightllm's rotary_emb kernel (models/llama/triton_kernel/rotary_emb.py):
grid = (cdiv(nnz, BLOCK_SEQ), cdiv(num_heads, BLOCK_HEAD)); each program processes a
BLOCK_SEQ x BLOCK_HEAD x (rotary_dim/2) tile. Changed vs upstream:

  * upstream takes pre-gathered cos/sin (nnz, dim/2); FreeToken passes a
    cos_sin_cache (max_pos, rotary_dim) plus positions -> we gather
    cache[positions[token]] inside the kernel (cos = first half, sin = second
    half over rotary_dim).
  * Q and K are rotated in a single kernel launch (one grid over max head
    count, K stores masked past HEAD_K) to cut launch overhead for tiny decode
    batches.
  * rotation math in fp32 like the vendored/flashinfer kernel; is_neox and
    interleave both supported as constexpr.

Optional pure-triton drop-in for
freetoken.kernel.rope.apply_rope_with_cos_sin_cache_inplace.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["nnz"])
def _rope_tiled(
    Q, K, POS, CACHE,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    nnz,
    HEAD_Q, HEAD_K, rotary_dim, half,
    HAS_K: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DHALF: tl.constexpr,
):
    seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)

    seq_range = seq_pid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)      # (S,)
    head_range = head_pid * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)  # (H,)
    d = tl.arange(0, BLOCK_DHALF)                                  # (D,)

    seq_mask = seq_range < nnz
    dmask = d < half

    pos = tl.load(POS + seq_range, mask=seq_mask, other=0).to(tl.int64)   # (S,)
    cache_row = CACHE + pos[:, None] * rotary_dim
    cs_mask = seq_mask[:, None] & dmask[None, :]
    cos = tl.load(cache_row + d[None, :], mask=cs_mask, other=0.0)         # (S,D) fp32
    sin = tl.load(cache_row + half + d[None, :], mask=cs_mask, other=0.0)  # (S,D) fp32
    cos = cos[:, None, :]   # (S,1,D)
    sin = sin[:, None, :]

    if INTERLEAVE:
        d0 = 2 * d
        d1 = 2 * d + 1
    else:
        d0 = d
        d1 = half + d
    d0 = d0[None, None, :]
    d1 = d1[None, None, :]

    # --- Q ---
    qmask = (seq_mask[:, None, None]
             & (head_range[None, :, None] < HEAD_Q)
             & dmask[None, None, :])
    base_q = seq_range[:, None, None] * stride_qbs + head_range[None, :, None] * stride_qh
    q0 = tl.load(Q + base_q + d0 * stride_qd, mask=qmask, other=0.0).to(tl.float32)
    q1 = tl.load(Q + base_q + d1 * stride_qd, mask=qmask, other=0.0).to(tl.float32)
    o0 = q0 * cos - q1 * sin
    o1 = q1 * cos + q0 * sin
    tl.store(Q + base_q + d0 * stride_qd, o0.to(Q.dtype.element_ty), mask=qmask)
    tl.store(Q + base_q + d1 * stride_qd, o1.to(Q.dtype.element_ty), mask=qmask)

    # --- K ---
    if HAS_K:
        kmask = (seq_mask[:, None, None]
                 & (head_range[None, :, None] < HEAD_K)
                 & dmask[None, None, :])
        base_k = seq_range[:, None, None] * stride_kbs + head_range[None, :, None] * stride_kh
        k0 = tl.load(K + base_k + d0 * stride_kd, mask=kmask, other=0.0).to(tl.float32)
        k1 = tl.load(K + base_k + d1 * stride_kd, mask=kmask, other=0.0).to(tl.float32)
        ok0 = k0 * cos - k1 * sin
        ok1 = k1 * cos + k0 * sin
        tl.store(K + base_k + d0 * stride_kd, ok0.to(K.dtype.element_ty), mask=kmask)
        tl.store(K + base_k + d1 * stride_kd, ok1.to(K.dtype.element_ty), mask=kmask)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Drop-in for the vendored op; rotates query/key in place."""
    if str(cos_sin_cache.dtype) != "torch.float32":
        raise ValueError("cos_sin_cache should be float32")
    assert query.is_cuda and key.is_cuda and positions.is_cuda
    assert cos_sin_cache.is_contiguous()

    nnz = query.shape[0]
    if nnz == 0:
        return
    rotary_dim = cos_sin_cache.shape[1]
    half = rotary_dim // 2
    block_dhalf = triton.next_power_of_2(half)

    head_q = query.shape[1] // head_size
    head_k = key.shape[1] // head_size
    qv = query.view(nnz, head_q, head_size)
    kv = key.view(nnz, head_k, head_size)

    max_head = max(head_q, head_k)

    grid = lambda META: (
        triton.cdiv(nnz, META["BLOCK_SEQ"]),
        triton.cdiv(max_head, META["BLOCK_HEAD"]),
    )
    # Fixed via H100 sweep (27-config grid; 16/1/w4 within 5% of the winner at
    # every nnz 1..4096, faster than tuned on average).
    _rope_tiled[grid](
        qv, kv, positions, cos_sin_cache,
        qv.stride(0), qv.stride(1), qv.stride(2),
        kv.stride(0), kv.stride(1), kv.stride(2),
        nnz, head_q, head_k, rotary_dim, half,
        HAS_K=True,
        INTERLEAVE=not is_neox,
        BLOCK_DHALF=block_dhalf,
        BLOCK_SEQ=16,
        BLOCK_HEAD=1,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _mrope_tiled(
    Q, K, POSITIONS, CACHE,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_pa, stride_pt,
    HEAD_Q: tl.constexpr, HEAD_K: tl.constexpr,
    rotary_dim: tl.constexpr, half: tl.constexpr,
    SECTION_T: tl.constexpr, SECTION_H: tl.constexpr, SECTION_W: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_HEAD: tl.constexpr, BLOCK_DHALF: tl.constexpr,
):
    """One program rotates all Q/K heads for one MRoPE token.

    This follows SGLang's fused MRoPE layout, with the partial-rotary bounds
    fix required by Qwen4-Exp (its rotary width is 64 inside 256-wide heads).
    """
    token = tl.program_id(0)
    d = tl.arange(0, BLOCK_DHALF)
    heads = tl.arange(0, BLOCK_HEAD)
    dmask = d < half

    time = tl.load(POSITIONS + token * stride_pt)
    height = tl.load(POSITIONS + stride_pa + token * stride_pt)
    width = tl.load(POSITIONS + 2 * stride_pa + token * stride_pt)

    # Qwen4 uses interleaved MRoPE axes: t,h,w,t,h,w,... with each axis
    # limited by its configured section.  The explicit dmask on t is crucial:
    # Triton pads the lane count to a power of two, while Qwen4 has partial RoPE.
    hmask = ((d % 3) == 1) & (d <= 3 * SECTION_H) & dmask
    wmask = ((d % 3) == 2) & (d <= 3 * SECTION_W) & dmask
    tmask = ~(hmask | wmask) & dmask

    time_base = CACHE + time * rotary_dim
    height_base = CACHE + height * rotary_dim
    width_base = CACHE + width * rotary_dim
    cos = (
        tl.load(time_base + d, mask=tmask, other=0.0)
        + tl.load(height_base + d, mask=hmask, other=0.0)
        + tl.load(width_base + d, mask=wmask, other=0.0)
    )
    sin = (
        tl.load(time_base + half + d, mask=tmask, other=0.0)
        + tl.load(height_base + half + d, mask=hmask, other=0.0)
        + tl.load(width_base + half + d, mask=wmask, other=0.0)
    )

    if IS_NEOX:
        d0 = d[None, :]
        d1 = (half + d)[None, :]
    else:
        d0 = (2 * d)[None, :]
        d1 = (2 * d + 1)[None, :]
    cos = cos[None, :]
    sin = sin[None, :]

    qmask = (heads[:, None] < HEAD_Q) & dmask[None, :]
    qbase = token * stride_qt + heads[:, None] * stride_qh
    q0 = tl.load(Q + qbase + d0 * stride_qd, mask=qmask, other=0.0).to(tl.float32)
    q1 = tl.load(Q + qbase + d1 * stride_qd, mask=qmask, other=0.0).to(tl.float32)
    tl.store(Q + qbase + d0 * stride_qd, q0 * cos - q1 * sin, mask=qmask)
    tl.store(Q + qbase + d1 * stride_qd, q1 * cos + q0 * sin, mask=qmask)

    kmask = (heads[:, None] < HEAD_K) & dmask[None, :]
    kbase = token * stride_kt + heads[:, None] * stride_kh
    k0 = tl.load(K + kbase + d0 * stride_kd, mask=kmask, other=0.0).to(tl.float32)
    k1 = tl.load(K + kbase + d1 * stride_kd, mask=kmask, other=0.0).to(tl.float32)
    tl.store(K + kbase + d0 * stride_kd, k0 * cos - k1 * sin, mask=kmask)
    tl.store(K + kbase + d1 * stride_kd, k1 * cos + k0 * sin, mask=kmask)


def apply_mrope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    mrope_section: tuple[int, int, int],
    is_neox: bool = True,
) -> None:
    """Apply Qwen4's three-axis, partial MRoPE to Q/K in place.

    `positions` is `[3, tokens]`: temporal, height and width axes. Text-only
    requests conventionally provide identical axes; multimodal callers may use
    different coordinates without a separate code path.
    """
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError(f"MRoPE positions must be [3, tokens], got {tuple(positions.shape)}")
    if str(cos_sin_cache.dtype) != "torch.float32":
        raise ValueError("cos_sin_cache should be float32")
    if not (query.is_cuda and key.is_cuda and positions.is_cuda):
        raise ValueError("MRoPE Triton path requires CUDA query, key and positions")
    if not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous")

    tokens = query.shape[0]
    if tokens == 0:
        return
    rotary_dim = cos_sin_cache.shape[1]
    half = rotary_dim // 2
    if sum(mrope_section) != half:
        raise ValueError(
            f"MRoPE sections {mrope_section} do not cover rotary half-width {half}"
        )
    head_q = query.shape[1] // head_size
    head_k = key.shape[1] // head_size
    qv = query.view(tokens, head_q, head_size)
    kv = key.view(tokens, head_k, head_size)
    _mrope_tiled[(tokens,)](
        qv, kv, positions, cos_sin_cache,
        qv.stride(0), qv.stride(1), qv.stride(2),
        kv.stride(0), kv.stride(1), kv.stride(2),
        positions.stride(0), positions.stride(1),
        HEAD_Q=head_q,
        HEAD_K=head_k,
        rotary_dim=rotary_dim,
        half=half,
        SECTION_T=mrope_section[0],
        SECTION_H=mrope_section[1],
        SECTION_W=mrope_section[2],
        IS_NEOX=is_neox,
        BLOCK_HEAD=triton.next_power_of_2(max(head_q, head_k)),
        BLOCK_DHALF=triton.next_power_of_2(half),
        num_warps=4,
        num_stages=1,
    )


__all__ = [
    "apply_rope_with_cos_sin_cache_inplace",
    "apply_mrope_with_cos_sin_cache_inplace",
]
