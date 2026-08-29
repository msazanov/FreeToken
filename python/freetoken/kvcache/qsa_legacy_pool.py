"""Paged KV storage for Qwen4 compressed sparse attention (QSA)."""

from __future__ import annotations

from typing import Sequence

import torch

from .mha_pool import MHAKVCache


class QSAKVCache(MHAKVCache):
    """Full-resolution paged K/V plus one compressed index-key tier.

    The full K/V slabs retain the runtime-selected cache quantization.  QSA
    index keys deliberately remain in the model compute dtype: their dot
    products select the exact full-resolution rows and cannot use the K/V
    cache's int8/fp8 storage format.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_num_kv_heads: int,
        index_head_dim: int,
        compress_ratio: int,
        layer_ids: Sequence[int],
        kv_cache_dtype: str = "bf16",
    ) -> None:
        if compress_ratio < 2 or page_size % compress_ratio:
            raise ValueError(
                "QSA needs a compression ratio >= 2 that divides the KV page size"
            )
        if dtype.itemsize != 2:
            raise ValueError(f"QSA index keys require a 2-byte compute dtype, got {dtype}")
        self._page_size = int(page_size)
        self._compress_ratio = int(compress_ratio)
        self._compressed_page_size = self._page_size // self._compress_ratio
        self._index_num_kv_heads = int(index_num_kv_heads)
        self._index_head_dim = int(index_head_dim)
        self._num_index_layers = len(layer_ids)
        self._index_dtype = dtype
        self._compressed_k: torch.Tensor | None = None
        self._pending_k: torch.Tensor | None = None
        self._pending_pos: torch.Tensor | None = None
        self._pending_rope: torch.Tensor | None = None
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
            kv_cache_dtype=kv_cache_dtype,
        )
        self._alloc_compressed(num_pages)

    def _alloc_compressed(self, num_pages: int) -> None:
        self._compressed_k = torch.empty(
            self._num_index_layers,
            num_pages,
            self._compressed_page_size,
            self._index_num_kv_heads,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )

    def rebuild(self, num_pages: int) -> None:
        self._compressed_k = None
        super().rebuild(num_pages)
        self._alloc_compressed(num_pages)

    def unit_bytes(self) -> tuple[int, int]:
        kv, swa = super().unit_bytes()
        assert self._compressed_k is not None
        compressed_per_token = (
            self._compressed_k.numel() * self._compressed_k.element_size()
        ) // self._storage_shape[0]
        return kv + compressed_per_token, swa

    @property
    def compress_ratio(self) -> int:
        return self._compress_ratio

    @property
    def compressed_page_size(self) -> int:
        return self._compressed_page_size

    def compressed_k_cache(self, layer_id: int) -> torch.Tensor:
        """Return row-flat index keys: ``[rows, kv_heads, index_dim]``."""
        assert self._compressed_k is not None
        return self._compressed_k[self._dense(layer_id)].view(
            -1, self._index_num_kv_heads, self._index_head_dim
        )

    def store_compressed_k(
        self, keys: torch.Tensor, compressed_rows: torch.Tensor, layer_id: int
    ) -> None:
        self.compressed_k_cache(layer_id)[compressed_rows.long()] = keys

    def ensure_pending_capacity(self, request_rows: int) -> None:
        """Grow the tiny per-request ring for incomplete compression groups."""
        current = 0 if self._pending_k is None else int(self._pending_k.shape[1])
        if current >= request_rows:
            return
        new_rows = max(request_rows, max(16, current * 2))
        shape = (
            self._num_index_layers,
            new_rows,
            self._compress_ratio,
            self._index_num_kv_heads,
            self._index_head_dim,
        )
        pending = torch.empty(shape, dtype=self._index_dtype, device=self._device)
        positions = torch.full(shape[:3], -1, dtype=torch.int64, device=self._device)
        rope = torch.full((*shape[:3], 3), -1, dtype=torch.int64, device=self._device)
        if self._pending_k is not None:
            pending[:, :current].copy_(self._pending_k)
            positions[:, :current].copy_(self._pending_pos)
            rope[:, :current].copy_(self._pending_rope)
        self._pending_k, self._pending_pos, self._pending_rope = pending, positions, rope

    def clear_pending(self, layer_id: int, request_row: int) -> None:
        assert self._pending_pos is not None
        self._pending_pos[self._dense(layer_id), request_row].fill_(-1)

    def pending_group(
        self, layer_id: int, request_row: int, positions: torch.Tensor
    ) -> torch.Tensor:
        assert self._pending_k is not None and self._pending_pos is not None
        dense = self._dense(layer_id)
        slots = torch.remainder(positions, self._compress_ratio).long()
        actual = self._pending_pos[dense, request_row].index_select(0, slots)
        expected = positions.to(device=actual.device, dtype=actual.dtype)
        if not torch.equal(actual, expected):
            raise RuntimeError(
                "QSA pending-key state is missing; use the naive cache and do not "
                "resume a prefix without its QSA state"
            )
        return self._pending_k[dense, request_row].index_select(0, slots)

    def pending_rope_group(
        self, layer_id: int, request_row: int, positions: torch.Tensor
    ) -> torch.Tensor:
        self.pending_group(layer_id, request_row, positions)
        assert self._pending_rope is not None
        slots = torch.remainder(positions, self._compress_ratio).long()
        return self._pending_rope[self._dense(layer_id), request_row].index_select(0, slots)

    def store_pending(
        self,
        layer_id: int,
        request_row: int,
        positions: torch.Tensor,
        keys: torch.Tensor,
        rope_positions: torch.Tensor | None = None,
    ) -> None:
        assert self._pending_k is not None and self._pending_pos is not None
        assert self._pending_rope is not None
        dense = self._dense(layer_id)
        slots = torch.remainder(positions, self._compress_ratio).long()
        self._pending_k[dense, request_row].index_copy_(0, slots, keys)
        self._pending_pos[dense, request_row].index_copy_(
            0, slots, positions.to(device=self._device, dtype=torch.int64)
        )
        if rope_positions is None:
            rope_positions = positions.to(device=self._device, dtype=torch.int64).view(-1, 1).expand(-1, 3)
        if rope_positions.shape != (positions.numel(), 3):
            raise ValueError(
                "QSA pending RoPE positions must have shape [tokens, 3], got "
                f"{tuple(rope_positions.shape)}"
            )
        self._pending_rope[dense, request_row].index_copy_(
            0, slots, rope_positions.to(device=self._device, dtype=torch.int64)
        )


__all__ = ["QSAKVCache"]
