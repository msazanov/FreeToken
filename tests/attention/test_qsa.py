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
