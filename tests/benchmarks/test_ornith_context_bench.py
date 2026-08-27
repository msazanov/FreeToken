from __future__ import annotations

import json

import pytest


def test_parse_context_tiers_normalizes_k_suffix_and_rejects_invalid_values():
    from benchmarks.ornith_context_bench import parse_context_tiers

    assert parse_context_tiers("1k,16K,64000,112k") == (1024, 16384, 64000, 114688)
    with pytest.raises(ValueError, match="positive"):
        parse_context_tiers("0")


def test_build_repository_dossier_uses_real_sources_and_excludes_git(tmp_path):
    from benchmarks.ornith_context_bench import build_repository_dossier

    (tmp_path / "README.md").write_text("mission: Ornith on RTX 2070\n", encoding="utf-8")
    src = tmp_path / "python" / "freetoken"
    src.mkdir(parents=True)
    (src / "stats.py").write_text("def build_stats(): return 'cache status'\n", encoding="utf-8")
    ignored = tmp_path / ".git"
    ignored.mkdir()
    (ignored / "config").write_text("must not enter dossier\n", encoding="utf-8")

    dossier = build_repository_dossier(tmp_path, max_chars=10_000)

    assert "FILE: README.md" in dossier
    assert "FILE: python/freetoken/stats.py" in dossier
    assert "must not enter dossier" not in dossier


def test_priority_evidence_extracts_anchor_windows_before_general_dossier(tmp_path):
    from benchmarks.ornith_context_bench import build_priority_evidence

    docs = tmp_path / "python" / "freetoken" / "server"
    docs.mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "intro\n## RTX 2070 fork mission\nOrnith target evidence\n", encoding="utf-8"
    )
    (docs / "control_api.py").write_text(
        "padding\n@app.get('/v1/stats')\nasync def stats():\n  return build_stats()\n",
        encoding="utf-8",
    )

    evidence = build_priority_evidence(tmp_path, max_chars=2_000)

    assert "RTX 2070 fork mission" in evidence
    assert "build_stats" in evidence


def test_dossier_budget_leaves_small_fixed_template_margin():
    from benchmarks.ornith_context_bench import dossier_budget

    assert dossier_budget(1024) == 864
    assert dossier_budget(114688) == 114528


def test_parse_sse_event_tracks_first_reasoning_delta_and_usage():
    from benchmarks.ornith_context_bench import parse_sse_event

    role_only = parse_sse_event('data: {"choices":[{"delta":{"role":"assistant"}}]}')
    first_delta = parse_sse_event(
        'data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}'
    )
    usage = parse_sse_event(
        'data: {"choices":[],"usage":{"prompt_tokens":128,"completion_tokens":7}}'
    )

    assert role_only.generated_text == ""
    assert first_delta.generated_text == "plan"
    assert usage.prompt_tokens == 128
    assert usage.completion_tokens == 7


def test_parse_sse_event_tracks_reported_prefix_cache_tokens():
    from benchmarks.ornith_context_bench import parse_sse_event

    usage = parse_sse_event(
        'data: {"choices":[],"usage":{"prompt_tokens":128,"completion_tokens":7,'
        '"prompt_tokens_details":{"cached_tokens":96}}}'
    )

    assert usage.cached_tokens == 96


def test_default_runner_output_directory_is_repository_benchmarks_results():
    from benchmarks.ornith_context_bench import parse_args

    output_dir = parse_args([]).output_dir.resolve()

    assert output_dir.name == "results"
    assert output_dir.parent.name == "benchmarks"
    assert not output_dir.as_posix().startswith("/tmp/")


def test_slice_index_entry_binds_speed_point_to_revision_and_parameters():
    from benchmarks.ornith_context_bench import slice_index_entry

    entry = slice_index_entry(
        {
            "timestamp_utc": "2026-08-27T14:11:21+00:00",
            "artifact": "/repo/benchmarks/results/run/compression-16384.json",
            "requested_context_tokens": 16_384,
            "metrics": {
                "prompt_tokens": 16_373,
                "ttft_s": 134.259,
                "prefill_tps_estimate": 121.95,
                "decode_tps": 20.56,
            },
            "slice": {
                "series": "ornith-rtx2070",
                "label": "tq4-p1024",
                "git": {"commit": "fc9027b", "dirty": False},
                "parameters": {"kv_block": "1024"},
                "runtime_parameters": {"kv_dtype": "tq4-nc"},
                "sampling": {"mode": "greedy-argmax", "temperature": 0.0, "seed": None},
            },
        }
    )

    assert entry["git_commit"] == "fc9027b"
    assert entry["parameters"] == {"kv_block": "1024"}
    assert entry["context_tokens"] == 16_373
    assert entry["decode_tps"] == 20.56
    assert entry["sampling_mode"] == "greedy-argmax"


def test_parse_parameters_requires_key_value_pairs():
    from benchmarks.ornith_context_bench import parse_parameters

    assert parse_parameters(["kv_block=1024", "moe_slots=1429"]) == {
        "kv_block": "1024",
        "moe_slots": "1429",
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_parameters(["bad-parameter"])
