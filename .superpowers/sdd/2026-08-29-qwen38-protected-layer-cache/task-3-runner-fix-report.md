# Task 3 runner fix report

## Scope

Fix the two Important findings from the review of commit `79a57b8` in
`benchmarks/qwen38_turing_profile.py`. The scope is limited to the stream
parser, its regression tests, and this report. No model, benchmark,
download, or subagent was run.

## Findings addressed

1. `response_sha256` now hashes only UTF-8 bytes from OpenAI
   `choices[].delta.content`. `reasoning_content` is deliberately excluded,
   so changes to hidden thinking do not change the final-answer digest.
2. `_stream_completion` now raises `RuntimeError` for an SSE `error` event,
   for EOF without the `[DONE]` sentinel, and for a stream that has no choice
   with non-null `finish_reason`.

Because `run_context_point` writes its JSON artifact only after
`_stream_completion` returns and obtains `/v1/stats`, any parser failure
propagates before artifact publication. Its existing `finally` block still
terminates only the runner-owned child and closes the log handles.

The runner continues to accept FreeToken's existing one-`data:`-line SSE
format. Multi-line SSE support was intentionally not added.

## TDD evidence

### RED

Added tests for:

- content-only hashing in the presence of reasoning chunks;
- SSE error propagation;
- missing `[DONE]` rejection;
- missing terminal `finish_reason` rejection.

Command:

```text
PYTHONPATH=/home/random/dev/qwen/freetoken/python \
/home/random/freetoken-turing/.venv/bin/python -m pytest \
tests/benchmarks/test_qwen38_turing_profile.py -q
```

Observed result before the production fix: `4 failed, 7 passed`.

### GREEN

After the minimal parser change, the same command reports:

```text
11 passed in 3.91s
```

No runtime model or benchmark process was started.
