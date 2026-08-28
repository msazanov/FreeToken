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


__all__ = ["QSAKVCache"]
