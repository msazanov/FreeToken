from __future__ import annotations

from freetoken.models.register import get_model_spec


def test_qwen4_text_model_registry_entry():
    spec = get_model_spec("Qwen4ExpForCausalLM")

    assert spec.module == "freetoken.models.qwen4_exp"
    assert spec.model_cls == "Qwen4ExpForCausalLM"
