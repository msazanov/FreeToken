# Ornith Tool-Call Recovery: Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepSeek Harness guide Ornith toward valid tool XML, show FreeToken recovery diagnostics, and provide a stable reduced tool catalog for new local sessions.

**Architecture:** A persistent local Cordis package contributes one global prompt section without creating a second system message. The existing telemetry package renders additive FreeToken recovery fields. A user-owned preset copies the stable coding stack and restricts schemas at the preset scope.

**Tech Stack:** TypeScript/JavaScript, Cordis, Node test runner, YAML profile patches, DeepSeek Harness agent presets.

**Spec:** `docs/superpowers/specs/2026-08-27-ornith-tool-call-recovery-design.md`

## Global Constraints

- The official Ornith Jinja chat template stays in FreeToken.
- The compatibility reminder appears once in the existing system message.
- Existing sessions retain their recorded preset.
- The new preset's tool catalog is stable across turns and modes.
- Profile-local packages must not shadow Harness singleton dependencies.

---

### Task 1: Ornith Compatibility Prompt Package

**Files:**
- Create: `/home/random/.local/share/dsh-ornith-compat/package.json`
- Create: `/home/random/.local/share/dsh-ornith-compat/index.js`
- Create: `/home/random/.local/share/dsh-ornith-compat/test/compat.test.js`
- Modify: `/home/random/.dsh/profiles/web/package.json`
- Modify: `/home/random/.dsh/profiles/web/cordis.patch.yml`
- Modify: `/home/random/.dsh/profiles/headless/package.json`
- Modify: `/home/random/.dsh/profiles/headless/cordis.patch.yml`

**Interfaces:**
- Produces: Cordis plugin `@local/dsh-ornith-compat` injecting `systemPrompt`.
- Produces: prompt section `ornith:tool-protocol` at order 90.

- [ ] **Step 1: Write failing package behavior test**

Load the plugin against a real Harness `SystemPrompt` context and assemble twice. Assert the rendered prompt contains one protocol reminder, the section name occurs once, and the plugin does not create history messages.

- [ ] **Step 2: Verify RED**

Run: `node --test /home/random/.local/share/dsh-ornith-compat/test/compat.test.js`

Expected: failure because the package/plugin does not exist.

- [ ] **Step 3: Implement the package and persistent profile wiring**

Register a static order-90 section with the approved concise instructions. Add the file dependency and Cordis insert to both profile patch layers; install dependencies without adding local copies of Cordis, scope, LLM, agent-presets, or system-prompt.

- [ ] **Step 4: Verify GREEN and profile isolation**

Run: `node --test /home/random/.local/share/dsh-ornith-compat/test/compat.test.js /home/random/.local/share/dsh-freetoken-telemetry/test/profile-isolation.test.js`

Expected: all tests pass.

### Task 2: Recovery Telemetry Rendering

**Files:**
- Modify: `/home/random/.local/share/dsh-freetoken-telemetry/client.js`
- Modify: `/home/random/.local/share/dsh-freetoken-telemetry/test/telemetry.test.js`

**Interfaces:**
- Consumes: FreeToken public `recovery` and parser fields plus loopback `stream_debug.recovery`.
- Produces: stable visible row `Tool recovery · <reason> · <count> calls · stopped at <tokens> tok`.

- [ ] **Step 1: Write failing render tests**

Add active and settled literal telemetry documents for `explicit_close`, `invalid_continuation`, `grace_exhausted`, and `silent_guard`. Assert every field remains visible in a dedicated row rather than being replaced on the next poll.

- [ ] **Step 2: Verify RED**

Run: `node --test /home/random/.local/share/dsh-freetoken-telemetry/test/telemetry.test.js`

Expected: recovery row assertions fail.

- [ ] **Step 3: Implement additive rendering**

Append recovery as a separate row in `formatTelemetryRows`; keep raw content in the existing disclosure and preserve current prefill/decode, cache, MoE, GPU, RAM, turn, and retry rows.

- [ ] **Step 4: Verify GREEN**

Run: `node --test /home/random/.local/share/dsh-freetoken-telemetry/test/*.test.js`

Expected: all local package tests pass.

### Task 3: Stable `ornith-code` Preset

**Files:**
- Create: `/home/random/.dsh/.agent-presets/ornith-code/agent.cordis.yml`
- Create: `/home/random/.dsh/.agent-presets/ornith-code/preset.json`
- Create: `/home/random/.local/share/dsh-ornith-compat/restrict.js`
- Modify: `/home/random/.dsh/settings.yaml`
- Modify: `/home/random/.local/share/dsh-ornith-compat/test/compat.test.js`

**Interfaces:**
- Produces: scoped plugin face `@local/dsh-ornith-compat/restrict` with configured allow-list.
- Produces: user preset `ornith-code` with exactly ten model-facing tools.
- Consumes: shipped `standard` preset structure and the scoped `ctx.tools.restrict({allow})` API.

- [ ] **Step 1: Write failing real-composition test**

Boot Harness preset discovery against the actual user root, mount `ornith-code`, and assert the assembled schema names are exactly `ask_user_question`, `bash`, `edit`, `glob`, `grep`, `read`, `skill`, `todo_write`, `web_search`, and `write` on two assemblies. Assert the prompt reminder occurs once.

- [ ] **Step 2: Verify RED**

Run: `node --test /home/random/.local/share/dsh-ornith-compat/test/compat.test.js`

Expected: unknown preset or unrestricted tool-catalog failure.

- [ ] **Step 3: Implement preset and restriction plugin**

Copy only the standard rows needed for persona, instructions, shell, filesystem/search, skills, compaction, ask-user, todo, and web. Mount the scoped restriction after registrations. Set `agent-presets.default: ornith-code` in settings without modifying existing session metadata.

- [ ] **Step 4: Verify GREEN and Harness preset e2e tests**

Run: `node --test /home/random/.local/share/dsh-ornith-compat/test/compat.test.js`

Then from the Harness repository run:
`pnpm vitest run apps/cli/tests/web-agent-presets.e2e.ts`.

Expected: the local preset assertions and shipped preset tests pass.

### Task 4: Runtime Rollout and End-to-End Probe

**Files:**
- Runtime/service configuration only; preserve current model files and session records.

**Interfaces:**
- Consumes: verified FreeToken and Harness changes.
- Produces: observed real wrapped/bare/malformed tool-call behavior and visible metrics.

- [ ] **Step 1: Restart FreeToken with explicit thresholds**

Add `--tool-call-bare-grace-tokens 32 --tool-call-silent-guard-tokens 128` to the existing service command, restart it, and confirm `/health`, `/v1/models`, and `/v1/stats` respond.

- [ ] **Step 2: Run deterministic API probes**

Send small tool-enabled streaming requests that instruct Ornith to call `read`. Capture SSE and assert valid tool-call JSON, `finish_reason: tool_calls`, no leaked XML, and generation far below `max_tokens` for recovered cases.

- [ ] **Step 3: Restart Harness and inspect assembled request/session log**

Confirm the prompt has one system message and one `ornith:tool-protocol` contribution. Confirm the existing session still resumes under its old preset and a newly created session selects `ornith-code`.

- [ ] **Step 4: Verify telemetry in API and UI**

Trigger one recovery and confirm `/v1/stats`, `/v1/debug/stream`, the Harness proxy, and the native statistics panel agree on reason, call count, and completion-token position.

- [ ] **Step 5: Run final verification**

Run all focused FreeToken tests, both local package test suites, Harness preset e2e tests, `git diff --check` in both repositories, and inspect service logs for traceback, OOM, CUDA Xid, or unhandled promise rejection.
