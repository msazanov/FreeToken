from __future__ import annotations

import hashlib
import json
import subprocess

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


def test_parse_sse_event_preserves_unknown_cache_telemetry():
    from benchmarks.ornith_context_bench import parse_sse_event

    usage = parse_sse_event(
        'data: {"choices":[],"usage":{"prompt_tokens":128,"completion_tokens":7}}'
    )

    assert usage.cached_tokens is None


def test_prefill_metrics_never_publish_total_prompt_rate_for_a_cache_hit():
    from benchmarks.ornith_context_bench import prefill_metrics

    cold = prefill_metrics(
        {"prompt_tokens": 1012, "cached_tokens": 0, "ttft_s": 8.0}
    )
    warm = prefill_metrics(
        {"prompt_tokens": 1012, "cached_tokens": 960, "ttft_s": 4.0}
    )
    unknown = prefill_metrics(
        {"prompt_tokens": 1012, "cached_tokens": None, "ttft_s": 4.0}
    )

    assert cold == {
        "new_prompt_tokens": 1012,
        "is_cold_prefill": True,
        "prefill_tps_estimate": 126.5,
        "naive_total_prompt_over_ttft_tps": 126.5,
    }
    assert warm == {
        "new_prompt_tokens": 52,
        "is_cold_prefill": False,
        "prefill_tps_estimate": None,
        "naive_total_prompt_over_ttft_tps": 253.0,
    }
    assert unknown == {
        "new_prompt_tokens": None,
        "is_cold_prefill": None,
        "prefill_tps_estimate": None,
        "naive_total_prompt_over_ttft_tps": 253.0,
    }


def test_default_runner_output_directory_is_repository_benchmarks_results():
    from benchmarks.ornith_context_bench import parse_args

    output_dir = parse_args([]).output_dir.resolve()

    assert output_dir.name == "results"
    assert output_dir.parent.name == "benchmarks"
    assert not output_dir.as_posix().startswith("/tmp/")


def test_model_identity_pins_declared_content_hash_revision_and_file_stat(tmp_path):
    from benchmarks.ornith_context_bench import _model_identity

    model = tmp_path / "ornith.gguf"
    model.write_bytes(b"packed-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()

    identity = _model_identity(
        str(model), content_sha256=digest, revision="hf-revision-123"
    )

    assert identity["content_sha256"] == digest
    assert identity["revision"] == "hf-revision-123"
    assert identity["files"] == [
        {
            "path": str(model.resolve()),
            "bytes": len(b"packed-model"),
            "mtime_ns": model.stat().st_mtime_ns,
        }
    ]


def test_model_identity_rejects_a_non_sha256_label(tmp_path):
    from benchmarks.ornith_context_bench import _model_identity

    model = tmp_path / "ornith.gguf"
    model.write_bytes(b"packed-model")

    with pytest.raises(ValueError, match="64 lowercase hex"):
        _model_identity(str(model), content_sha256="not-a-digest", revision=None)


def test_git_identity_hashes_staged_changes_instead_of_an_empty_worktree_diff(tmp_path):
    from benchmarks.ornith_context_bench import _git_identity

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "bench@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Benchmark Test"],
        check=True,
    )
    source = tmp_path / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "runtime.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "baseline"], check=True
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "runtime.py"], check=True)

    identity = _git_identity(tmp_path)

    assert identity["dirty"] is True
    assert identity["working_tree_diff_sha256"] != hashlib.sha256(b"").hexdigest()


def test_git_identity_hashes_untracked_source_files(tmp_path):
    from benchmarks.ornith_context_bench import _git_identity

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "bench@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Benchmark Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "baseline"], check=True
    )
    untracked = tmp_path / "new_kernel.py"
    untracked.write_text("VALUE = 2\n", encoding="utf-8")

    identity = _git_identity(tmp_path)

    assert identity["untracked_sha256"] == {
        "new_kernel.py": hashlib.sha256(untracked.read_bytes()).hexdigest()
    }


def test_slice_index_entry_binds_speed_point_to_revision_and_parameters():
    from benchmarks.ornith_context_bench import slice_index_entry

    entry = slice_index_entry(
        {
            "timestamp_utc": "2026-08-27T14:11:21+00:00",
            "artifact": "/repo/benchmarks/results/run/compression-16384.json",
            "requested_context_tokens": 16_384,
            "metrics": {
                "prompt_tokens": 16_373,
                "cached_tokens": 0,
                "new_prompt_tokens": 16_373,
                "is_cold_prefill": True,
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
    assert entry["cached_tokens"] == 0
    assert entry["new_prompt_tokens"] == 16_373
    assert entry["is_cold_prefill"] is True
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
