from __future__ import annotations

import inspect


def test_gated_rmsnorm_exposes_centered_weight_mode_for_qwen4():
    from freetoken.kernel.fla.layernorm_gated import rms_norm_gated

    parameter = inspect.signature(rms_norm_gated).parameters["weight_plus_one"]

    assert parameter.default is False
