from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool, KV_CACHE_DTYPES


_KV_DATA_DTYPES = {
    "fp8-e5m2": torch.float8_e5m2,
    "int8": torch.int8,
}


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
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
        layer_ids: Sequence[int] | None = None,
        kv_cache_dtype: str = "bf16",
    ) -> None:
        if kv_cache_dtype not in KV_CACHE_DTYPES:
            raise ValueError(
                f"unsupported kv_cache_dtype {kv_cache_dtype!r}; "
                f"expected one of {sorted(KV_CACHE_DTYPES)}"
            )
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self._num_storage_layers: int
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._num_storage_layers = num_storage_layers
        self._page_size = page_size
        self._local_kv_heads = local_kv_heads
        self._head_dim = head_dim
        self._compute_dtype = dtype
        self._kv_cache_dtype = kv_cache_dtype
        self._data_dtype = _KV_DATA_DTYPES.get(kv_cache_dtype, dtype)
        self._device = device
        self._kv_buffer: torch.Tensor | None = None
        self._k_buffer: torch.Tensor | None = None
        self._v_buffer: torch.Tensor | None = None
        self._k_scale_buffer: torch.Tensor | None = None
        self._v_scale_buffer: torch.Tensor | None = None
        self._allocate(num_pages)

    def _allocate(self, num_pages: int) -> None:
        data_shape = (
            self._num_storage_layers,
            num_pages,
            self._page_size,
            self._local_kv_heads,
            self._head_dim,
        )
        if self._kv_cache_dtype == "bf16":
            self._kv_buffer = torch.empty(
                (2, *data_shape), device=self._device, dtype=self._compute_dtype
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
        else:
            self._k_buffer = torch.empty(data_shape, device=self._device, dtype=self._data_dtype)
            self._v_buffer = torch.empty(data_shape, device=self._device, dtype=self._data_dtype)
            scale_shape = data_shape[:-1]
            self._k_scale_buffer = torch.empty(
                scale_shape, device=self._device, dtype=torch.bfloat16
            )
            self._v_scale_buffer = torch.empty(
                scale_shape, device=self._device, dtype=torch.bfloat16
            )
        self._storage_shape = (num_pages * self._page_size, self._local_kv_heads, self._head_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        device = self._device
        self._kv_buffer = None
        self._k_buffer = None
        self._v_buffer = None
        self._k_scale_buffer = None
        self._v_scale_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._allocate(num_pages)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        assert self._k_buffer is not None and self._v_buffer is not None
        total_bytes = (
            self._k_buffer.numel() * self._k_buffer.element_size()
            + self._v_buffer.numel() * self._v_buffer.element_size()
        )
        if self._k_scale_buffer is not None:
            total_bytes += self._k_scale_buffer.numel() * self._k_scale_buffer.element_size()
        if self._v_scale_buffer is not None:
            total_bytes += self._v_scale_buffer.numel() * self._v_scale_buffer.element_size()
        return total_bytes // self._storage_shape[0], 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        assert self._k_buffer is not None
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        assert self._v_buffer is not None
        return self._v_buffer[self._dense(index)]

    def k_scale(self, index: int) -> torch.Tensor:
        if self._k_scale_buffer is None:
            raise RuntimeError("BF16 KV cache has no quantization scales")
        return self._k_scale_buffer[self._dense(index)]

    def v_scale(self, index: int) -> torch.Tensor:
        if self._v_scale_buffer is None:
            raise RuntimeError("BF16 KV cache has no quantization scales")
        return self._v_scale_buffer[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        assert self._k_buffer is not None and self._v_buffer is not None
        if self.is_quantized:
            from freetoken.kernel.triton.kv_quant import store_quantized_kv

            assert self._k_scale_buffer is not None and self._v_scale_buffer is not None
            store_quantized_kv(
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                k_scale=self._k_scale_buffer[dense].view(self._storage_shape[:2]),
                v_scale=self._v_scale_buffer[dense].view(self._storage_shape[:2]),
                indices=out_loc,
                k=k,
                v=v,
            )
            return

        from freetoken.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._data_dtype

    @property
    def kv_cache_dtype(self) -> str:
        return self._kv_cache_dtype

    @property
    def is_quantized(self) -> bool:
        return self._kv_cache_dtype != "bf16"

    @property
    def num_layers(self) -> int:
        return self._num_layers
