"""Qwen4 compressed sparse-attention backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

_SCORE_WORKSPACE_BYTES = 128 << 20


@dataclass
class QSAMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_q_host: tuple[int, ...]
    logical_positions: torch.Tensor
    last_indices: torch.Tensor
    compressed_rows: tuple[torch.Tensor, ...]

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


def _compact_expanded_selection(
    block_indices: torch.Tensor, query_positions: torch.Tensor, *,
    compress_ratio: int, token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand selected compression groups and append the incomplete-group tail."""
    if block_indices.is_cuda:
        from freetoken.kernel.triton.qsa import compact_qsa_blocks

        return compact_qsa_blocks(
            block_indices, query_positions, compress_ratio=compress_ratio,
            token_budget=token_budget,
        )
    rows, device = block_indices.shape[0], block_indices.device
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = block_indices.long().unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        block_indices.long().unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, -1)[:, :token_budget]
    positions = query_positions.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded <= positions.unsqueeze(1)), expanded,
        torch.full_like(expanded, -1),
    )
    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible = positions + 1
    tail_start = torch.div(visible, compress_ratio, rounding_mode="floor") * compress_ratio
    tail_count = visible - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail = torch.where(
        tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1), tail, torch.full_like(tail, -1)
    )
    result = torch.full(
        (rows, token_budget + compress_ratio - 1), -1, dtype=torch.long, device=device
    )
    result[:, :token_budget].copy_(expanded)
    block_counts = (block_indices >= 0).sum(dim=1)
    tail_columns = block_counts.unsqueeze(1) * compress_ratio + tail_offsets.unsqueeze(0)
    tail_live = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)
    row_ids = torch.arange(rows, device=device).unsqueeze(1).expand_as(tail_columns)
    result[row_ids[tail_live], tail_columns[tail_live]] = tail[tail_live]
    counts = (block_counts * compress_ratio + tail_count).to(torch.int32)
    return result.to(torch.int32), counts


def select_qsa_logical_rows(
    index_q: torch.Tensor, compressed_keys: torch.Tensor, query_positions: torch.Tensor,
    *, compress_ratio: int, token_budget: int, score_workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed keys and return exact logical token selections."""
    rows, heads, dim = index_q.shape
    blocks = compressed_keys.shape[0]
    block_budget = token_budget // compress_ratio
    output_blocks = torch.full(
        (rows, block_budget), -1, dtype=torch.int32, device=index_q.device
    )
    if rows == 0:
        return _compact_expanded_selection(
            output_blocks, query_positions, compress_ratio=compress_ratio, token_budget=token_budget
        )
    if blocks:
        workspace_bytes = _SCORE_WORKSPACE_BYTES if score_workspace_bytes is None else score_workspace_bytes
        if workspace_bytes <= 0:
            raise ValueError("QSA score workspace must be positive")
        bytes_per_row = max(
            blocks * torch.float32.itemsize * (1 if index_q.is_cuda else heads + 1), 1
        )
        row_chunk = max(1, min(rows, workspace_bytes // bytes_per_row))
        keys = compressed_keys[:, 0].transpose(0, 1)
        columns = torch.arange(blocks, device=index_q.device).unsqueeze(0)
        for start in range(0, rows, row_chunk):
            stop = min(start + row_chunk, rows)
            if index_q.is_cuda:
                from freetoken.kernel.triton.qsa import qsa_head_reduced_scores

                logits = qsa_head_reduced_scores(
                    index_q, compressed_keys, row_start=start, row_stop=stop
                )
            else:
                queries = index_q[start:stop].reshape((stop - start) * heads, dim)
                dots = queries.float() @ keys.float()
                logits = torch.relu_(dots.view(stop - start, heads, blocks)).sum(dim=1)
                logits.mul_(dim**-0.5)
            visible_blocks = torch.div(
                query_positions[start:stop].to(torch.long) + 1,
                compress_ratio, rounding_mode="floor",
            )
            logits.masked_fill_(columns >= visible_blocks.unsqueeze(1), -float("inf"))
            width = min(block_budget, blocks)
            if width:
                scores, picks = torch.topk(logits, width, dim=1)
                picks = torch.where(torch.isfinite(scores), picks, torch.full_like(picks, -1))
                output_blocks[start:stop, :width] = picks.to(torch.int32)
    return _compact_expanded_selection(
        output_blocks, query_positions, compress_ratio=compress_ratio, token_budget=token_budget
    )


def qsa_tq4_sparse_gqa(
    q: torch.Tensor, packed_k: torch.Tensor, packed_v: torch.Tensor,
    k_scale: torch.Tensor, v_scale: torch.Tensor, selected_rows: torch.Tensor,
    counts: torch.Tensor, *, layer_id: int,
) -> torch.Tensor:
    """Attend over TQ4 full KV while retaining QSA's unquantized index selection."""
    from freetoken.kernel.triton.qsa import qsa_sparse_gqa
    from freetoken.kvcache.tq4 import randomized_hadamard

    num_kv_heads = packed_k.shape[-2]
    transformed_q = randomized_hadamard(
        q, layer_id=layer_id, num_kv_heads=num_kv_heads
    )
    transformed_output = qsa_sparse_gqa(
        transformed_q, packed_k, packed_v, selected_rows, counts,
        q.shape[-1] ** -0.5, k_scale=k_scale, v_scale=v_scale,
        logical_head_dim=q.shape[-1],
    )
    return randomized_hadamard(
        transformed_output, layer_id=layer_id, num_kv_heads=num_kv_heads, inverse=True
    )


class QSAAttnBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from freetoken.kvcache.qsa_pool import QSAKVCache

        self.config = config
        self.args = config.qwen4_args
        self.kvcache = get_global_ctx().kv_cache
        if not isinstance(self.kvcache, QSAKVCache):
            raise TypeError(f"qsa backend needs QSAKVCache, got {type(self.kvcache).__name__}")
        self.device = self.kvcache.device
        self.compress_ratio = int(self.args.indexer_compress_ratio)
        self.token_budget = int(self.args.indexer_budget)
        if self.token_budget % self.compress_ratio:
            raise ValueError("QSA token budget must divide by its compression ratio")

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        lengths = [int(req.extend_len) for req in reqs]
        cu_host = [0]
        for length in lengths:
            cu_host.append(cu_host[-1] + length)
        cu = torch.tensor(cu_host, dtype=torch.int32, device=self.device)
        logical = (
            torch.cat([
                torch.arange(req.cached_len, req.device_len, device=self.device)
                for req in reqs if req.extend_len
            ], dim=0)
            if sum(lengths) else torch.empty(0, dtype=torch.int64, device=self.device)
        )
        page_table = get_global_ctx().page_table
        compressed_rows: list[torch.Tensor] = []
        for req in reqs:
            complete_blocks = int(req.device_len) // self.compress_ratio
            if complete_blocks:
                starts = torch.arange(complete_blocks, device=self.device, dtype=torch.long).mul_(self.compress_ratio)
                full_rows = page_table[int(req.table_idx)].index_select(0, starts)
                rows = torch.div(full_rows.to(torch.int64), self.compress_ratio, rounding_mode="floor")
            else:
                rows = torch.empty(0, dtype=torch.int64, device=self.device)
            compressed_rows.append(rows)
        batch.attn_metadata = QSAMetadata(
            cu_seqlens_q=cu, cu_seqlens_q_host=tuple(cu_host), logical_positions=logical,
            last_indices=cu[1:] - 1, compressed_rows=tuple(compressed_rows),
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int,
        batch: Batch, attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Qwen4Exp QSA layers call qsa_forward()")

    def _compress_current_keys(self, indexer, index_k: torch.Tensor, layer_id: int, batch: Batch) -> None:
        md = batch.attn_metadata
        assert isinstance(md, QSAMetadata)
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        if reqs:
            self.kvcache.ensure_pending_capacity(max(int(req.table_idx) for req in reqs) + 1)
        page_table = get_global_ctx().page_table
        pooled: list[torch.Tensor] = []
        rope_positions: list[torch.Tensor] = []
        compressed_rows: list[torch.Tensor] = []
        cu, ratio = md.cu_seqlens_q_host, self.compress_ratio
        for req_id, req in enumerate(reqs):
            begin, stop = cu[req_id], cu[req_id + 1]
            if begin == stop:
                continue
            start, end, request_row = int(req.cached_len), int(req.device_len), int(req.table_idx)
            current = index_k[begin:stop]
            batch_rope = getattr(batch, "rope_positions", None)
            current_rope = (
                md.logical_positions[begin:stop].view(-1, 1).expand(-1, 3)
                if batch_rope is None else batch_rope[:, begin:stop].transpose(0, 1)
            )
            if start == 0:
                self.kvcache.clear_pending(layer_id, request_row)
            first_end = ((start + ratio) // ratio) * ratio - 1
            for group_end in range(first_end, end, ratio):
                group_start = group_end - ratio + 1
                if group_start < start:
                    prior_pos = torch.arange(group_start, start, device=self.device)
                    prior = self.kvcache.pending_group(layer_id, request_row, prior_pos)
                    first_rope = self.kvcache.pending_rope_group(layer_id, request_row, prior_pos[:1])[0]
                    members = torch.cat((prior, current[: group_end - start + 1]), dim=0)
                else:
                    lo = group_start - start
                    members, first_rope = current[lo : lo + ratio], current_rope[lo]
                if members.shape[0] != ratio:
                    raise RuntimeError("QSA compression received an incomplete key group")
                pooled.append(members.float().mean(dim=0).to(index_k.dtype))
                rope_positions.append(first_rope)
                full_row = page_table[request_row, group_start].to(torch.int64)
                compressed_rows.append(torch.div(full_row, ratio, rounding_mode="floor"))
            keep_start = max(start, end - ratio)
            keep_positions = torch.arange(keep_start, end, device=self.device)
            self.kvcache.store_pending(
                layer_id, request_row, keep_positions, current[keep_start - start :],
                current_rope[keep_start - start :],
            )
        if pooled:
            positions = torch.stack(rope_positions).transpose(0, 1).contiguous()
            normalized = indexer.normalize_compressed_keys(torch.stack(pooled), positions)
            self.kvcache.store_compressed_k(
                normalized, torch.stack(compressed_rows).to(torch.int64), layer_id
            )

    def _select_physical_rows(
        self, index_q: torch.Tensor, layer_id: int, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        md = batch.attn_metadata
        assert isinstance(md, QSAMetadata)
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        page_table, cu, ratio = get_global_ctx().page_table, md.cu_seqlens_q_host, self.compress_ratio
        selections: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        compressed_pool = self.kvcache.compressed_k_cache(layer_id)
        for req_id, req in enumerate(reqs):
            begin, stop = cu[req_id], cu[req_id + 1]
            if begin == stop:
                continue
            compressed_rows = md.compressed_rows[req_id]
            compressed_keys = (
                compressed_pool.index_select(0, compressed_rows) if compressed_rows.numel() else compressed_pool[:0]
            )
            logical, live = select_qsa_logical_rows(
                index_q[begin:stop], compressed_keys, md.logical_positions[begin:stop],
                compress_ratio=ratio, token_budget=self.token_budget,
            )
            safe = logical.clamp_min(0).long()
            physical = page_table[int(req.table_idx)].index_select(0, safe.reshape(-1)).reshape_as(logical)
            selections.append(torch.where(logical >= 0, physical, torch.full_like(physical, -1)).to(torch.int32))
            counts.append(live)
        if not selections:
            width = self.token_budget + ratio - 1
            return (
                torch.empty((0, width), dtype=torch.int32, device=self.device),
                torch.empty(0, dtype=torch.int32, device=self.device),
            )
        return torch.cat(selections, dim=0), torch.cat(counts, dim=0)

    def qsa_forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, index_q: torch.Tensor,
        index_k: torch.Tensor, indexer, layer_id: int, batch: Batch,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.qsa import qsa_sparse_gqa

        if self.kvcache.is_quantized and self.kvcache.kv_cache_dtype != "tq4-nc":
            raise NotImplementedError(
                "QSA supports packed full KV only with tq4-nc; INT8/FP8 need their own "
                "scale-aware gathered-row attention kernel"
            )
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        self._compress_current_keys(indexer, index_k, layer_id, batch)
        selected, counts = self._select_physical_rows(index_q, layer_id, batch)
        k_raw, v_raw = self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id)
        k_rows = k_raw.view(-1, k_raw.shape[-2], k_raw.shape[-1])
        v_rows = v_raw.view(-1, v_raw.shape[-2], v_raw.shape[-1])
        if self.kvcache.kv_cache_dtype == "tq4-nc":
            k_scale = self.kvcache.k_scale(layer_id).view(-1, k_raw.shape[-2])
            v_scale = self.kvcache.v_scale(layer_id).view(-1, v_raw.shape[-2])
            return qsa_tq4_sparse_gqa(
                q, k_rows, v_rows, k_scale, v_scale, selected, counts, layer_id=layer_id
            )
        return qsa_sparse_gqa(
            q, k_rows, v_rows, selected, counts,
            q.shape[-1] ** -0.5,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        return None  # PLE keeps per-request recurrent state.

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_metadata(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.prepare_metadata(batch)


__all__ = ["QSAAttnBackend", "QSAMetadata", "qsa_tq4_sparse_gqa", "select_qsa_logical_rows"]
