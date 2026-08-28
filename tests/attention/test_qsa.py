"""QSA's CPU reference is the correctness oracle for the Turing kernel path."""

from __future__ import annotations

import math
import os

import pytest
import torch


def test_qsa_selection_expands_complete_blocks_and_visible_tail():
    from freetoken.attention.qsa import select_qsa_logical_rows

    # Two index heads prefer block 0, then block 1. Block 2 is masked because
    # each query has only two fully visible compression groups.
    index_q = torch.tensor([[[2.0, 0.0], [2.0, 0.0]], [[2.0, 0.0], [2.0, 0.0]]])
    keys = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[4.0, 0.0]]])
    selected, counts = select_qsa_logical_rows(
        index_q, keys, torch.tensor([3, 4]), compress_ratio=2, token_budget=4
    )

    assert torch.equal(selected[0], torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32))
    assert torch.equal(selected[1], torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(counts, torch.tensor([4, 5], dtype=torch.int32))


def test_qsa_selection_is_independent_of_score_workspace_size():
    from freetoken.attention.qsa import select_qsa_logical_rows

    torch.manual_seed(7)
    index_q = torch.randn(6, 2, 3)
    compressed_keys = torch.randn(7, 1, 3)
    query_positions = torch.tensor([11, 12, 13, 14, 15, 16])

    small_selected, small_counts = select_qsa_logical_rows(
        index_q, compressed_keys, query_positions, compress_ratio=4, token_budget=8,
        score_workspace_bytes=84,
    )
    large_selected, large_counts = select_qsa_logical_rows(
        index_q, compressed_keys, query_positions, compress_ratio=4, token_budget=8,
        score_workspace_bytes=4096,
    )

    assert torch.equal(small_selected, large_selected)
    assert torch.equal(small_counts, large_counts)


def test_qsa_sparse_gqa_cpu_reference_respects_selected_physical_rows():
    from freetoken.kernel.triton.qsa import qsa_sparse_gqa

    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
    v = torch.tensor([[[10.0, 0.0]], [[0.0, 20.0]], [[30.0, 30.0]]])
    output = qsa_sparse_gqa(
        q, k, v, torch.tensor([[2, 0]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32), 1 / math.sqrt(2),
    )

    keys = k[[2, 0], 0].float()
    values = v[[2, 0], 0]
    expected = torch.softmax(q[0].float() @ keys.T / math.sqrt(2), dim=-1) @ values
    assert torch.allclose(output[0], expected, atol=1e-6)


def test_qsa_tq4_packed_kv_matches_dense_attention_after_inverse_rotation():
    from freetoken.kernel.triton.qsa import qsa_sparse_gqa
    from freetoken.kvcache.tq4 import encode_tq4, randomized_hadamard

    raw_q = torch.tensor([[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]]])
    raw_k = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[0.5, 0.5, 0.0, 0.0]],
    ])
    raw_v = torch.tensor([
        [[10.0, 0.0, 1.0, 0.0]], [[0.0, 20.0, 0.0, 1.0]], [[5.0, 5.0, 5.0, 5.0]],
    ])
    selected = torch.tensor([[2, 0]], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    scale = 1 / math.sqrt(4)
    dense = qsa_sparse_gqa(raw_q, raw_k, raw_v, selected, counts, scale)

    transformed_q = randomized_hadamard(raw_q, layer_id=3, num_kv_heads=1)
    packed_k, k_scale = encode_tq4(randomized_hadamard(raw_k, layer_id=3, num_kv_heads=1))
    packed_v, v_scale = encode_tq4(randomized_hadamard(raw_v, layer_id=3, num_kv_heads=1))
    transformed_output = qsa_sparse_gqa(
        transformed_q, packed_k, packed_v, selected, counts, scale,
        k_scale=k_scale, v_scale=v_scale, logical_head_dim=4,
    )
    restored = randomized_hadamard(
        transformed_output, layer_id=3, num_kv_heads=1, inverse=True
    )

    # TQ4 is lossy, but the packed path must stay close to its dense counterpart.
    assert torch.allclose(restored, dense, atol=1.2, rtol=0.12)


def test_qsa_tq4_adapter_preserves_qsa_row_selection_semantics():
    from freetoken.attention.qsa import qsa_tq4_sparse_gqa
    from freetoken.kernel.triton.qsa import qsa_sparse_gqa
    from freetoken.kvcache.tq4 import encode_tq4, randomized_hadamard

    raw_q = torch.tensor([[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]]])
    raw_k = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[0.5, 0.5, 0.0, 0.0]],
    ])
    raw_v = torch.tensor([
        [[10.0, 0.0, 1.0, 0.0]], [[0.0, 20.0, 0.0, 1.0]], [[5.0, 5.0, 5.0, 5.0]],
    ])
    selected = torch.tensor([[2, 0]], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    scale = 1 / math.sqrt(4)
    expected = qsa_sparse_gqa(raw_q, raw_k, raw_v, selected, counts, scale)

    packed_k, k_scale = encode_tq4(randomized_hadamard(raw_k, layer_id=3, num_kv_heads=1))
    packed_v, v_scale = encode_tq4(randomized_hadamard(raw_v, layer_id=3, num_kv_heads=1))
    actual = qsa_tq4_sparse_gqa(
        raw_q, packed_k, packed_v, k_scale, v_scale, selected, counts, layer_id=3
    )

    assert torch.allclose(actual, expected, atol=1.2, rtol=0.12)


def test_qsa_tq4_rejects_mismatched_packed_value_storage():
    from freetoken.kernel.triton.qsa import qsa_sparse_gqa

    q = torch.ones(1, 2, 4)
    packed_k = torch.zeros(2, 1, 2, dtype=torch.uint8)
    unpacked_v = torch.zeros(2, 1, 2)
    scales = torch.ones(2, 1, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="both packed K and V"):
        qsa_sparse_gqa(
            q, packed_k, unpacked_v, torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32), 0.5,
            k_scale=scales, v_scale=scales, logical_head_dim=4,
        )


@pytest.mark.skipif(
    os.environ.get("FREETOKEN_RUN_QSA_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="explicit GPU validation only; do not contend with a serving runtime",
)
def test_qsa_tq4_packed_kernel_compiles_and_matches_cpu_oracle_on_cuda():
    from freetoken.kernel.triton.qsa import qsa_sparse_gqa
    from freetoken.kvcache.tq4 import encode_tq4

    torch.manual_seed(1)
    # Qwen3.8 Flash Next uses 256-wide full-attention heads.  Cover the actual
    # SM75 Triton specialization, including its register pressure.
    q = torch.randn(2, 4, 256, device="cuda", dtype=torch.float16)
    dense_k = torch.randn(7, 2, 256, device="cuda", dtype=torch.float16)
    dense_v = torch.randn(7, 2, 256, device="cuda", dtype=torch.float16)
    packed_k, k_scale = encode_tq4(dense_k)
    packed_v, v_scale = encode_tq4(dense_v)
    selected = torch.tensor([[6, 3, 0], [5, 2, 1]], device="cuda", dtype=torch.int32)
    counts = torch.tensor([3, 2], device="cuda", dtype=torch.int32)

    actual = qsa_sparse_gqa(
        q, packed_k, packed_v, selected, counts, 256**-0.5,
        k_scale=k_scale, v_scale=v_scale, logical_head_dim=256,
    )
    expected = qsa_sparse_gqa(
        q.cpu(), packed_k.cpu(), packed_v.cpu(), selected.cpu(), counts.cpu(), 256**-0.5,
        k_scale=k_scale.cpu(), v_scale=v_scale.cpu(), logical_head_dim=256,
    ).cuda()

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    os.environ.get("FREETOKEN_RUN_QSA_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="explicit GPU validation only; do not contend with a serving runtime",
)
def test_qsa_head_reduced_scorer_matches_explicit_256_wide_reference_on_cuda():
    from freetoken.attention.qsa import select_qsa_logical_rows
    from freetoken.kernel.triton.qsa import qsa_head_reduced_scores

    torch.manual_seed(8)
    index_q = torch.randn(4, 3, 256, device="cuda", dtype=torch.float16)
    compressed_keys = torch.randn(7, 1, 256, device="cuda", dtype=torch.float16)
    query_positions = torch.tensor([12, 13, 14, 15], device="cuda")

    actual_scores = qsa_head_reduced_scores(
        index_q, compressed_keys, row_start=1, row_stop=4
    )
    expected_scores = torch.relu(
        index_q[1:4].float() @ compressed_keys[:, 0].float().transpose(0, 1)
    ).sum(dim=1) * (256**-0.5)
    assert actual_scores.dtype is torch.float32
    assert torch.allclose(actual_scores, expected_scores, atol=2e-2, rtol=2e-2)

    actual_selected, actual_counts = select_qsa_logical_rows(
        index_q, compressed_keys, query_positions, compress_ratio=4, token_budget=8,
        score_workspace_bytes=84,
    )
    expected_selected, expected_counts = select_qsa_logical_rows(
        index_q.cpu(), compressed_keys.cpu(), query_positions.cpu(),
        compress_ratio=4, token_budget=8, score_workspace_bytes=84,
    )
    assert torch.equal(actual_selected.cpu(), expected_selected)
    assert torch.equal(actual_counts.cpu(), expected_counts)
