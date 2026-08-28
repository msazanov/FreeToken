import pytest
import torch
import types
import sys


def _activation_and_mul(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu":
        return torch.nn.functional.gelu(gate) * up
    raise AssertionError(f"unsupported test activation {activation}")


def _reference_fused_experts_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    hidden_fp32 = hidden_states.float()
    w1_fp32 = w1.float()
    w2_fp32 = w2.float()
    for token_idx in range(hidden_states.shape[0]):
        for route_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, route_idx])
            route_weight = topk_weights[token_idx, route_idx].float()
            gate_up = torch.matmul(w1_fp32[expert_id], hidden_fp32[token_idx])
            if apply_router_weight_on_input:
                gate_up = gate_up * route_weight
            activated = _activation_and_mul(gate_up, activation)
            contribution = torch.matmul(w2_fp32[expert_id], activated)
            if not apply_router_weight_on_input:
                contribution = contribution * route_weight
            output[token_idx] += contribution
    return output.to(hidden_states.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_accepts_triton_kernel_tuple_output():
    from freetoken.moe.fused import fused_topk

    logits = torch.tensor(
        [[4.0, 1.0, -1.0, 2.0], [0.5, 3.0, 2.0, -2.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_states = torch.zeros((2, 8), device="cuda", dtype=torch.bfloat16)

    weights, ids = fused_topk(hidden_states, logits, topk=2, renormalize=True)

    ref_logits, ref_ids = torch.topk(logits.float(), 2, dim=-1)
    ref_weights = torch.softmax(ref_logits, dim=-1)
    torch.testing.assert_close(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=2e-4, atol=2e-4)


def test_fused_topk_falls_back_when_optional_triton_kernel_rejects_geometry(monkeypatch):
    """An optional router compilation failure must not kill the serving worker."""
    from freetoken.moe.fused import fused_topk
    import freetoken.kernel.backend as backend

    broken = types.ModuleType("triton_kernels.topk")
    broken.topk = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unsupported geometry"))
    monkeypatch.setattr(backend, "is_triton_kernels_installed", lambda: True)
    monkeypatch.setitem(sys.modules, "triton_kernels.topk", broken)

    logits = torch.tensor([[4.0, 1.0, -1.0]], dtype=torch.float32)
    weights, ids = fused_topk(torch.zeros(1, 2), logits, topk=2, renormalize=True)

    assert ids.tolist() == [[0, 1]]
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))


def test_fused_topk_uses_vendored_router_for_non_power_of_two_topk(monkeypatch):
    """Qwen3.8's top-10 geometry must not enter triton_kernels.topk.

    ``triton_kernels.topk`` constructs a ten-wide ``tl.arange`` and fails to
    compile.  The vendored router pads its internal tile and masks the two
    inactive lanes instead.
    """
    from freetoken.moe.fused import fused_topk
    import freetoken.kernel.backend as backend

    called = {}
    vendored = types.ModuleType("freetoken.kernel.triton.moe_router")

    def fake_vendored(logits, topk, renormalize, num_token_non_padded):
        called["args"] = (logits, topk, renormalize, num_token_non_padded)
        return (
            torch.ones((logits.shape[0], topk), dtype=torch.float32),
            torch.arange(topk, dtype=torch.int32).expand(logits.shape[0], -1),
        )

    vendored.fused_topk_softmax = fake_vendored
    monkeypatch.setattr(backend, "is_triton_kernels_installed", lambda: True)
    monkeypatch.setitem(sys.modules, "freetoken.kernel.triton.moe_router", vendored)

    logits = torch.randn((2, 512), dtype=torch.float32)
    weights, ids = fused_topk(torch.zeros((2, 4)), logits, topk=10, renormalize=True)

    assert called["args"][1:] == (10, True, None)
    assert weights.shape == ids.shape == (2, 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vendored_top10_router_matches_unique_reference_on_cuda():
    """The padded top-10 tile preserves Qwen's routed experts on SM75."""
    from freetoken.kernel.triton.moe_router import fused_topk_softmax

    logits = torch.arange(512, device="cuda", dtype=torch.float32).reshape(1, -1) / 1000
    weights, ids = fused_topk_softmax(logits, topk=10, renormalize=True)
    reference_weights, reference_ids = torch.topk(torch.softmax(logits, dim=-1), 10, dim=-1)
    reference_weights = reference_weights / reference_weights.sum(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(ids, reference_ids.to(torch.int32))
    torch.testing.assert_close(weights, reference_weights, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 24])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_experts_decode_matches_reference_for_non_contiguous_slots(batch_size, dtype):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    cache_size = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 8
    torch.manual_seed(42 + batch_size)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    slot_ids = torch.tensor([31, 4, 18, 0, 29, 7, 35, 12], device=device, dtype=torch.int32)
    topk_ids = slot_ids.repeat(batch_size, 1).contiguous()

    output = fused_experts_decode_impl(hidden_states, gate_up, down, topk_weights, topk_ids)
    expected = _reference_fused_experts_decode(hidden_states, gate_up, down, topk_weights, topk_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_matches_grouped_impl(dtype, apply_router_weight_on_input):
    from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

    device = torch.device("cuda")
    batch_size = 4
    num_experts = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    torch.manual_seed(91)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.tensor(
        [[31, 4, 18, 0], [29, 7, 35, 12], [6, 21, 1, 33], [16, 3, 28, 9]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = fused_experts_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_experts_grouped_impl_is_cuda_graph_capturable():
    from freetoken.moe.fused import fused_experts_impl

    device = torch.device("cuda")
    dtype = torch.float16
    batch_size = 1
    num_experts = 8
    hidden_size = 32
    intermediate_size = 24
    torch.manual_seed(17)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    topk_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.float32)
    topk_ids = torch.tensor([[6, 1, 4, 7]], device=device, dtype=torch.int32)
    output = torch.empty_like(hidden_states)

    def run():
        output.copy_(
            fused_experts_impl(
                hidden_states,
                gate_up,
                down,
                topk_weights,
                topk_ids,
            )
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_activation_and_router_weight_modes(
    activation,
    apply_router_weight_on_input,
):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    batch_size = 3
    cache_size = 19
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    dtype = torch.float16
    torch.manual_seed(123)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    topk_weights = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.15, 0.35, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
        device=device,
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[17, 2, 11, 5], [3, 17, 0, 14], [8, 6, 12, 1]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = _reference_fused_experts_decode(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)
