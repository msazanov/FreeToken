"""QSA's CPU reference is the correctness oracle for the Turing kernel path."""

from __future__ import annotations

import math

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
