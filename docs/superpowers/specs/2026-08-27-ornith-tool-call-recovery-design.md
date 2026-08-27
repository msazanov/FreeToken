# Ornith tool-call recovery across FreeToken and DeepSeek Harness

## Goal

Make long Ornith coding turns terminate promptly and execute tools when the model emits a
complete Qwen3-Coder-style `<function=...>...</function>` call but omits the surrounding
`<tool_call>...</tool_call>` wrapper. Preserve correctly wrapped and parallel tool calls, keep
the OpenAI-compatible stream valid, and make every recovery visible in diagnostics.

The incident this design addresses had three complete `read` calls followed by malformed
markup. FreeToken continued decoding to the 16,384-token output limit while its parser discarded
the bytes it was waiting to classify. DeepSeek Harness then received `finish_reason: length` and
discarded the otherwise complete tool-call blocks. Neither the tools runtime nor the machine had
failed.

## Scope

The change has three coordinated parts:

1. FreeToken recognizes a completed bare invoke, waits briefly for a valid parallel-call
   continuation, and can end generation successfully as `tool_calls`.
2. FreeToken records bounded recovery and parser-stall diagnostics, including a short-lived raw
   tail after a request settles.
3. DeepSeek Harness contributes an Ornith protocol reminder as one section of the existing
   system message and offers a stable, smaller Ornith agent preset for new sessions.

This is not a general XML repair system. It never invents a tool name, parameter, parameter
value, or missing `</function>`. It only synthesizes the outer close after a complete call whose
arguments are already valid JSON in the parser ledger.

## Ownership boundaries

- The Qwen3-Coder detector owns recognition of wrapper and invoke structure.
- The protocol-neutral FreeToken generation loop owns completion-token accounting, successful
  early termination, and conversion to `finish_reason: tool_calls`.
- The scheduler owns cancellation and resource release. Parser recovery must not masquerade as a
  client cancellation or a generation error.
- DeepSeek Harness owns model-facing instructions and stable tool-catalog selection. It does not
  parse or rewrite model output.

The official Ornith Jinja template remains tokenizer-side in FreeToken. Harness receives only a
small behavioral section; copying the complete template into Harness would duplicate role
serialization and can produce invalid multiple-system-message layouts.

## FreeToken parser state machine

### Normal wrapped calls

`<tool_call>` enters the existing block state. Each `<function=NAME>` opens one call and
`</function>` completes it. A real `</tool_call>` marks a terminal tool boundary. The generation
loop closes the current streamed call and stops decoding immediately instead of waiting for EOS.

### Bare completed calls

When `_ps_allow_bare_invoke` admitted a `<function=NAME>` without an outer opener, completing its
`</function>` enters a new `awaiting_bare_continuation` state. The detector exposes this state to
the generation loop; it does not declare success by itself.

While awaiting continuation, only these byte sequences are valid:

- whitespace;
- a partial or complete `</tool_call>`;
- a partial or complete next `<function=...>`;
- a partial or complete `<tool_call>` followed by the next function.

A next function commits the previous call and resumes ordinary parallel-call parsing. A real
outer close commits the call and emits a terminal-boundary signal. Any non-whitespace bytes that
are no longer a prefix of an allowed continuation emit a synthetic-boundary signal with reason
`invalid_continuation`.

The allowed-prefix rule deliberately catches debris such as `<tool\n` quickly. The debris is
retained only in the loopback debug tail; it is neither user-visible content nor part of the tool
arguments.

### Grace and safety limits

The generation loop counts exact `ack.completion_tokens_delta` values, not characters or parser
chunks.

- A bare completed call gets a 32-token continuation grace window.
- Reaching the grace limit emits a synthetic boundary with reason `grace_exhausted`.
- An absolute 128-token no-semantic-progress guard applies when the tool parser is withholding
  bytes after at least one structurally complete call. It emits reason `silent_guard`.

The 128-token guard is a backstop for unforeseen parser states, not a substitute for the
32-token bare-call decision. If no complete call exists, the guard must not execute a partial
call; normal EOS, length, or error handling remains authoritative.

### Parser/generation interface

The streaming parser exposes read-only control state alongside its existing ordered text/call
events:

- whether one or more complete calls await a boundary;
- whether the state is specifically a bare-call grace candidate;
- a one-shot terminal-boundary decision and reason;
- whether raw bytes are currently being withheld.

The interface is detector-neutral, with no generation-loop type checks for
`Qwen3CoderDetector`. Detectors that do not implement recovery report the default inactive
state.

## Successful early stop

On a real or synthetic terminal tool boundary, the generation loop performs this sequence:

1. Drain all parser fragments produced by the current acknowledgement.
2. Close the currently open call from its prefix-stable fragments or valid parser ledger.
3. Emit the final `ToolCallsDelta` exactly once.
4. Send a dedicated successful-stop control message without deleting the frontend
   acknowledgement maps and without classifying the request as a client abort.
5. Ignore further sampled text while waiting for the scheduler's terminal successful-stop
   acknowledgement.
6. Emit `GenDone(finish_reason="tool_calls")` with the actual prompt, cache, and
   completion-token totals observed through that acknowledgement.

The successful-stop acknowledgement is the resource/accounting barrier. The scheduler shares
the existing race-safe request removal and KV/GDN release machinery with client aborts, but sends
a distinct non-error terminal reply. This avoids returning success while an unobserved request
continues decoding and avoids disguising expected control flow as `request aborted`.

This requires dedicated frontend, tokenizer, and backend stop messages plus a dedicated
successful terminal reply. Existing `abort_user()` is unsuitable because it deletes response
queues, marks the request aborted, sleeps, and surfaces scheduler cancellation as an error.

If the stop message cannot be dispatched, FreeToken logs and counts the failure and continues
draining acknowledgements to the model's ordinary terminal result; it must not abandon a live
scheduler request. Duplicate stop messages and late acknowledgements are idempotent.

## Finish-reason rules

- A clean or recovered complete call ends as `tool_calls`.
- `length` wins only when no recovery boundary was committed before the length acknowledgement.
- A partial invoke or invalid JSON argument ledger is never promoted from `length` to
  `tool_calls`.
- Non-streaming generation uses the same structural validity rules when parsing the completed
  text, but has no early-stop optimization because decoding has already ended.

OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses continue to consume the shared
protocol-neutral events; no wire adapter receives Ornith-specific parsing logic.

## Telemetry and diagnostics

`StatsTracker` adds cumulative process counters and per-request fields:

- `recovered_tool_calls_total`;
- `recovery_stop_failures_total`;
- recovery `active`, `reason`, `tool_count`, and `grace_tokens`;
- parser `withholding`, `state`, and `silent_tokens`.

The ordinary `/v1/stats` response exposes only these bounded semantic fields. The loopback-only
`/v1/debug/stream` response also exposes the bounded raw tail and recovery decision. After a
request settles, its debug snapshot remains available for ten minutes or until the next request
starts, whichever comes first. A new request clears the previous settled snapshot, preventing a
stale tail from being presented as current output.

DeepSeek Harness extends its existing native telemetry rows with a stable recovery line such as
`Tool recovery · invalid continuation · 3 calls · stopped at 412 tok`. The raw tail stays behind
the existing disclosure and never enters session history or model context.

## Harness prompt integration

A local `@local/dsh-ornith-compat` host plugin registers one global prompt section named
`ornith:tool-protocol` after the persona and before tool-specific guidance. It states, concisely:

> Emit tool calls only in the catalogued Qwen3-Coder XML form. Close both `</function>` and
> `</tool_call>`. After the final tool call, stop and wait for tool results. Never print partial
> tool tags as prose.

The section is global on this deployment because FreeToken exposes only Ornith. It remains a
separate section name, so the `standard` preset's scoped persona cannot shadow it. The plugin is
installed in both Web and headless profiles from one local package. Request-header/session-log
capture of the assembled prompt remains the audit source.

Tests assemble the real `standard` preset and prove that the reminder occurs exactly once inside
the single system message. No additional system-role message is appended to history.

## Stable Ornith tool presets

Tool-schema count affects both model reliability and prefix-cache identity. New sessions may use
a user-owned `ornith-code` preset copied from the shipped `standard` composition, with a scoped
allow-list containing:

`ask_user_question`, `bash`, `edit`, `glob`, `grep`, `read`, `skill`, `todo_write`, `web_search`,
and `write`.

The preset keeps compaction and agent instructions but omits goal, job, subagent, workflow,
Ralph, and dynamic Cordis tools. It does not change its catalog between turns or modes. This is
the default for newly created local sessions after verification.

Existing sessions retain the preset recorded in their durable session metadata; Harness
correctly refuses to re-scope them. They still receive the global prompt reminder and the
FreeToken parser fix. We do not rewrite session history or force a preset migration merely to
reduce the schema list.

An `ornith-explore` preset may follow later if read-only session creation becomes useful. It is
not required for this incident fix, because changing one live agent's tool list by task phase
would damage request-prefix stability.

## Tests

Implementation follows red-green-refactor.

FreeToken parser tests cover:

- a fully wrapped single call;
- two parallel functions in one wrapper;
- bare call plus immediate next bare function;
- bare call plus delayed valid continuation within 32 token deltas;
- bare call plus invalid `<tool\n` debris;
- bare call plus grace exhaustion;
- an incomplete function and invalid JSON ledger, neither recoverable;
- fragmented markers across arbitrary chunk boundaries;
- no duplicate call fragments or terminal decisions.

Generation tests use a fake acknowledgement stream and prove:

- exact completion-token accounting for the grace window;
- successful internal stop and `finish_reason: tool_calls`;
- no leaked scheduler abort error on any wire adapter;
- `length` is preserved for incomplete calls;
- racing sampled tokens after the stop decision are counted but not emitted;
- duplicate and late successful-stop acknowledgements cannot duplicate or mutate the completed
  stream.

Telemetry tests prove cumulative counters, active and settled snapshots, expiry, next-request
clearing, and loopback-only raw-tail access.

Harness tests boot the actual Web/headless composition and assert one prompt section, one system
message, the exact reduced `ornith-code` tool catalog, and unchanged catalogs across repeated
turn assemblies. The local telemetry package tests render every recovery reason.

## Rollout and rollback

1. Land tests and parser/generation behavior in the FreeToken worktree.
2. Restart FreeToken and run synthetic wrapped, bare, parallel, and malformed-call probes against
   the OpenAI endpoint.
3. Install the Harness compatibility section and telemetry fields; restart Harness once.
4. Run one real repository-read turn with the existing session and inspect the persisted stream.
5. Add and select `ornith-code` only for new sessions after the compatibility path is proven.

FreeToken recovery is guarded by server options for the tool parser: bare-call grace defaults to
32 and silent guard to 128; setting either to zero disables that threshold. Removing the local
Harness plugin and restoring the previous default preset rolls back the Harness side without
touching session data. No model weights, KV-cache format, or persisted conversation record is
migrated.
