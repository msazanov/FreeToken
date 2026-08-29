from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_sweep_forwards_forced_long_decode_and_keeps_artifact_identity(tmp_path, monkeypatch):
    from benchmarks import ornith_prefill_sweep as sweep

    repo_root = tmp_path / "repo"
    results_root = repo_root / "benchmarks" / "results"
    ledger = results_root / "benchmark-events.jsonl"
    repo_root.mkdir(parents=True)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        run_name = command[command.index("--run-name") + 1]
        artifact = results_root / run_name / "context-16384.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps(
                {
                    "timestamp_utc": "2026-08-29T00:00:00+00:00",
                    "requested_context_tokens": 16384,
                    "metrics": {"prompt_tokens": 16384, "completion_tokens": 4096, "prompt_tps": 99.0, "decode_tps": 20.0},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    args = sweep.parse_args(
        [
            "--model-path", "/models/ornith.gguf", "--repo-root", str(repo_root),
            "--results-root", str(results_root), "--ledger", str(ledger), "--run-name", "live",
            "--chunks", "1024", "--decode-tokens", "4096", "--ignore-eos", "--trace-stride-events", "64",
        ]
    )

    sweep.run_sweep(args)

    assert "--ignore-eos" in commands[0]
    assert commands[0][commands[0].index("--trace-stride-events") + 1] == "64"
    event = json.loads(ledger.read_text(encoding="utf-8"))
    assert event["event_id"].startswith("artifact:")
    assert event["attempt_id"] == "live-p1024"
    assert event["parameters"]["max_prefill_length"] == 1024


def test_sweep_server_args_keep_dash_prefixed_values_bound_to_server_arg():
    from benchmarks.ornith_prefill_sweep import _ornith_server_args

    values = _ornith_server_args(1280)

    assert "--server-arg=--served-model-name" in values
    assert "--server-arg=--max-prefill-length" in values
    assert "--server-arg=1280" in values
