from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _response(name: str, arguments: object, *, content: str = "") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": [
                        {"type": "function", "function": {"name": name, "arguments": arguments}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def test_textual_tool_imitation_is_rejected():
    from benchmarks.gemma_huggingvoice_tool_superset_acceptance import _validate_response

    body = {
        "choices": [
            {
                "message": {"content": "Хорошо, Марат, я запомнил ваше имя."},
                "finish_reason": "stop",
            }
        ]
    }

    passed, _arguments = _validate_response(
        body, "speaker_memory_remember_name", "sr_test"
    )

    assert passed is False


def test_native_calls_for_all_three_scenarios_are_accepted():
    from benchmarks.gemma_huggingvoice_tool_superset_acceptance import _validate_response

    cases = (
        (
            "speaker_memory_remember_name",
            '{"name":"Марат","speaker_ref":"sr_test"}',
        ),
        ("web_search", '{"query":"актуальная погода в Москве"}'),
        ("camera_snapshot", "{}"),
    )

    for name, arguments in cases:
        passed, parsed = _validate_response(_response(name, arguments), name, "sr_test")
        assert passed is True
        assert isinstance(parsed, dict)


def test_camera_call_requires_an_explicit_json_object_argument_string():
    from benchmarks.gemma_huggingvoice_tool_superset_acceptance import _validate_response

    for raw_arguments in (None, "", "[]", "{bad"):
        passed, _parsed = _validate_response(
            _response("camera_snapshot", raw_arguments),
            "camera_snapshot",
            "sr_test",
        )
        assert passed is False


def test_native_call_requires_function_envelope_type():
    from benchmarks.gemma_huggingvoice_tool_superset_acceptance import _validate_response

    body = _response("camera_snapshot", "{}")
    body["choices"][0]["message"]["tool_calls"][0]["type"] = "not_function"

    passed, _parsed = _validate_response(
        body,
        "camera_snapshot",
        "sr_test",
    )

    assert passed is False


def test_catalog_matches_browser_order_and_allowed_supersets():
    from benchmarks.gemma_huggingvoice_tool_superset_acceptance import SCENARIOS

    memory_names = [
        "speaker_memory_inspect",
        "speaker_memory_remember_name",
        "speaker_memory_confirm",
        "speaker_memory_reject",
        "speaker_memory_block_voice",
        "speaker_memory_unblock_voice",
        "speaker_memory_remember_fact",
        "speaker_memory_recall",
        "speaker_memory_forget",
    ]
    assert [tool["function"]["name"] for tool in SCENARIOS[0]["tools"]] == [
        "web_search",
        *memory_names,
    ]
    assert [tool["function"]["name"] for tool in SCENARIOS[2]["tools"]] == [
        "web_search",
        "camera_snapshot",
        *memory_names,
    ]


def test_acceptance_cli_can_be_launched_by_file_path():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/gemma_huggingvoice_tool_superset_acceptance.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "10/11-tool browser sets" in result.stdout


def test_source_provenance_hashes_each_runtime_source_file():
    from benchmarks.gemma_huggingvoice_tool_superset_acceptance import _source_provenance

    source = _source_provenance()

    assert source["arbiter_proxy_sha256"] == hashlib.sha256(
        (ROOT / "python/freetoken/arbiter/proxy.py").read_bytes()
    ).hexdigest()
    assert source["acceptance_script_sha256"] == hashlib.sha256(
        (ROOT / "benchmarks/gemma_huggingvoice_tool_superset_acceptance.py").read_bytes()
    ).hexdigest()
