from __future__ import annotations


def test_build_harness_shaped_messages_keeps_only_case_tag_outside_shared_prefix():
    from benchmarks.ornith_harness_cache_bench import build_harness_shaped_messages

    messages = build_harness_shaped_messages(
        repository_prompt="REPOSITORY DOSSIER\ncache-sensitive source text",
        case_tag="cold-001",
    )

    assert messages[0]["role"] == "system"
    assert "Tool calls are protocol" in messages[0]["content"]
    assert "cache-sensitive source text" in messages[1]["content"]
    assert "CACHE_CASE=cold-001" in messages[-1]["content"]
    assert "cold-001" not in messages[0]["content"]
    assert "cold-001" not in messages[1]["content"]


def test_append_turn_preserves_prior_prompt_and_adds_only_a_small_delta():
    from benchmarks.ornith_harness_cache_bench import append_turn

    base = [{"role": "system", "content": "stable"}, {"role": "user", "content": "dossier"}]
    appended = append_turn(base, "prior answer", "append-001")

    assert appended[:-2] == base
    assert appended[-2] == {"role": "assistant", "content": "prior answer"}
    assert appended[-1]["content"] == "CACHE_APPEND=append-001\nContinue with one concise fact."


def test_cache_metrics_uses_server_reported_usage_not_timing_inference():
    from benchmarks.ornith_harness_cache_bench import cache_metrics

    assert cache_metrics({"prompt_tokens": 100, "cached_tokens": 75}) == {
        "prompt_tokens": 100,
        "cached_tokens": 75,
        "new_prompt_tokens": 25,
        "cache_hit_ratio": 0.75,
    }


def test_parse_args_can_force_a_fixed_decode_window_for_runtime_profiling():
    from benchmarks.ornith_harness_cache_bench import parse_args

    assert parse_args(["--ignore-eos"]).ignore_eos is True


def test_result_directory_requires_a_distinct_label_for_nonstandard_run(tmp_path):
    from benchmarks.ornith_harness_cache_bench import result_directory

    assert result_directory(tmp_path, "2026-08-27", None).name == "2026-08-27-ornith-harness-cache"
    assert result_directory(tmp_path, "2026-08-27", "decode-profile").name == (
        "2026-08-27-ornith-harness-cache-decode-profile"
    )
