from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_registry_entry_preserves_actual_speed_and_metadata(tmp_path):
    from benchmarks.speed_registry import append_artifact

    artifact = tmp_path / "context-1024.json"
    artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-29T10:00:00+00:00",
                "requested_context_tokens": 1024,
                "sampling": {"seed": 20260828, "temperature": 0},
                "metrics": {
                    "prompt_tokens": 1040,
                    "completion_tokens": 153,
                    "prompt_tps": 63.53,
                    "decode_tps": 29.08,
                    "elapsed_s": 21.64,
                },
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "model-context-speed.jsonl"

    appended = append_artifact(
        artifact,
        registry,
        model="Ornith 1.5 35b",
        quantization="Q4_K_M",
        runtime_profile="turing-q4km-int8kv",
        source_kind="end_to_end",
        git_commit="abc1234",
    )

    assert appended is True
    row = json.loads(registry.read_text(encoding="utf-8"))
    assert row == {
        "actual_context_tokens": 1040,
        "artifact": str(artifact),
        "completion_tokens": 153,
        "decode_tps": 29.08,
        "elapsed_s": 21.64,
        "git_commit": "abc1234",
        "model": "Ornith 1.5 35b",
        "prefill_tps": 63.53,
        "quantization": "Q4_K_M",
        "requested_context_tokens": 1024,
        "runtime_profile": "turing-q4km-int8kv",
        "sampling": {"seed": 20260828, "temperature": 0},
        "source_kind": "end_to_end",
        "timestamp_utc": "2026-08-29T10:00:00+00:00",
    }


def test_registry_deduplicates_the_same_raw_artifact(tmp_path):
    from benchmarks.speed_registry import append_artifact

    artifact = tmp_path / "context-1024.json"
    artifact.write_text(
        json.dumps({"metrics": {"prompt_tokens": 1}, "sampling": {}}),
        encoding="utf-8",
    )
    registry = tmp_path / "model-context-speed.jsonl"
    metadata = {
        "model": "test-model",
        "quantization": "test-quant",
        "runtime_profile": "test-profile",
        "source_kind": "unit",
        "git_commit": "deadbee",
    }

    assert append_artifact(artifact, registry, **metadata) is True
    assert append_artifact(artifact, registry, **metadata) is False
    assert len(registry.read_text(encoding="utf-8").splitlines()) == 1


def test_profile_runner_exposes_registry_metadata_for_every_future_run():
    from benchmarks.qwen38_turing_profile import parse_args

    args = parse_args(["--model-path", "/models/test.gguf"])

    assert args.speed_registry.name == "model-context-speed.jsonl"
    assert args.quantization == "unknown"
    assert args.runtime_profile == "unspecified"


def test_profile_runner_can_execute_as_a_script_entrypoint():
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "benchmarks/qwen38_turing_profile.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--speed-registry" in result.stdout
