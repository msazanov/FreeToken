# Task 3 runner preparation report

Date: 2026-08-29

## Scope

Implemented only the exact-output digest preparation for the Qwen profile
runner. No model, benchmark, download, or runtime service was started. No
README, TESTLOG, or CHANGELOG changes were made.

## TDD evidence

### RED

Command:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q -k stream_completion_hashes_text_channels_in_arrival_order
```

Result: `1 failed, 6 deselected`. The failure was the expected
`KeyError: 'response_sha256'` because the stream runner did not yet publish a
digest.

A second RED test for the result-record boundary was then added and produced
the expected `ImportError` because `make_result_record` did not yet exist.

### GREEN

The minimal implementation now:

- creates one SHA-256 object per stream;
- consumes only non-empty string values from `delta.content` and
  `delta.reasoning_content` in stream arrival order, preserving choice order;
- encodes each fragment as UTF-8 bytes directly into the digest, without
  retaining the prompt or response text;
- exposes `response_sha256` in stream metrics and at the top level of the
  result record;
- preserves `server_stats`, including the cache policy, for `lru`, `reap`,
  `control`, and `protected_layer` artifacts.

The tests use a hand-defined stream fixture whose expected byte sequence is
`"думответ!".encode("utf-8")`; the expected hash is independently computed
with `hashlib.sha256`.

Command:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q -k 'stream_completion_hashes_text_channels_in_arrival_order or result_record_promotes_digest_without_retaining_response_body'
```

Result: `2 passed, 6 deselected`.

Final verification command:

```text
PYTHONPATH=python /home/random/freetoken-turing/.venv/bin/python -m pytest tests/benchmarks/test_qwen38_turing_profile.py -q && PYTHONDONTWRITEBYTECODE=1 /home/random/freetoken-turing/.venv/bin/python -m py_compile benchmarks/qwen38_turing_profile.py tests/benchmarks/test_qwen38_turing_profile.py && git diff --check
```

Result: `8 passed`; Python compilation succeeded; `git diff --check` was
clean.

## Not performed by design

- No FreeToken process was launched.
- No model or benchmark was run.
- No model was downloaded.
- No prompt or response body was written to an artifact.
- No README, TESTLOG, or CHANGELOG was modified.
