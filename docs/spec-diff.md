# Spec vs. reality

SPEC.md §12 says to verify the external interfaces before building. This file records what was
checked, when, and every place where the spec had to bend. Re-check these when upgrading Claude Code
or d1-3B.

## Claude Code hooks — checked 2026-10-11 against code.claude.com/docs/en/hooks (CLI 2.1.296)

| # | Spec says / assumes | Reality | What NOUL-GATE does |
|---|---|---|---|
| H1 | M0 "always allows" with "zero behavior change" | `permissionDecision: "allow"` **skips the permission prompt** the user would otherwise see. Emitting it would *be* a behavior change and a privilege escalation. | Log mode exits 0 with **empty stdout** → Claude Code's normal permission flow runs unchanged. Verified with a real `claude -p` session: a `Write` call not in `--allowedTools` was logged *and* still denied. |
| H2 | Gate outputs allow / ask / deny | Hook outputs are `allow`, `ask`, `deny`, `defer`. Settings deny/ask rules apply regardless of the hook. Hook precedence is deny > defer > ask > allow. `defer` is honored only in `-p` mode. | Gate `deny` → `"deny"`, gate `ask` → `"ask"`, gate `allow` → **no output** (normal flow) unless `--grant-allow` is passed, which emits `"allow"` and lets the gate auto-approve. `defer` is unused. |
| H3 | State includes `last_user_msg`, `recent_tool_outputs` | The payload has only `session_id, transcript_path, cwd, permission_mode, hook_event_name, tool_name, tool_input, tool_use_id` (+ optional `prompt_id, agent_id, agent_type, effort, scratchpad_dir`). | FeatureBuilder reads `transcript_path`. The transcript JSONL is an **undocumented internal format**; the parser is defensive and degrades to empty provenance if it changes. |
| H4 | Exit-code semantics | 0 = success (stdout parsed as JSON if it is one object); 2 = block, stderr is the reason; other = non-blocking error. **A timed-out hook does not block** unless `"onFailure": "block"` (v2.1.295+). | Log mode always exits 0. Enforce mode: installer sets `timeout: 15` and `onFailure: "block"`; if the decision server is unreachable the hook returns `"ask"` (fail to a human, never a silent allow). |
| H5 | A local model behind the hook | Each hook invocation is a fresh process. Loading a 3B model per call costs seconds. | Enforce mode is a thin client to `noulgate serve` (resident model, port 7701). |
| H6 | — | Hook commands get the payload on stdin; `matcher: "*"` matches all tools including `mcp__*`. | Installer adds one `matcher: "*"` group and replaces its own entry on re-install. |

## d1-3B — checked 2026-10-11 against huggingface.co/LiquidAI/d1-3B (README + `api.py`)

| # | Spec says / assumes | Reality | What NOUL-GATE does |
|---|---|---|---|
| D1 | `choice` has `options: [...]` | `choice` takes `criteria: {name: description}` | `questions.py` accepts the spec YAML and translates `options` into `criteria` (with default descriptions for the built-in risk classes). |
| D2 | `score` has `scale: [1,5]` | `score` takes `criteria: [level descriptions]`, 2–10 levels, lowest first. The answer `score` is the expected **0-indexed** level. | `scale: [lo, hi]` becomes `hi-lo+1` level descriptions; answers are mapped back onto `[lo, hi]`. |
| D3 | `system_one`, `system_one_batch` | Confirmed: `system_one(state, questions, images=None)`, `system_one_batch([(state, questions[, images]), ...])`. Returns `{"answers": {name: answer}, "usage": {...}}`. noul → `{"noul": P(yes)}`; choice → `{choice, confidence, probabilities}`; score → `{score, confidence, probabilities, legend}`. | `backends/d1.py` wraps these directly. |
| D4 | "transformers or llama.cpp (GGUF)" | Requires `transformers>=5.14` with `trust_remote_code=True`. **No official GGUF** is published. | transformers only. Loading runs model-repo code, so the backend accepts a pinned `revision`. |
| D5 | License may limit users | **LFM Open License v1.0** (custom, has commercial-use terms). | d1 is an optional extra (`pip install noulgate[d1]`); rules / LLM / Jev backends work without it. |
| D6 | ~8 ms RTX 4090, ~16 ms Jetson AGX Thor | Card confirms these (single question). 3 questions in one pass: 21 ms on a 4090. No CPU figures. | `eval/latency.py` measures our own numbers; see README. |

## Attack benchmarks (spec §8, §12)

No third-party benchmark data is redistributed in this repo. `noulgate dataset import` reads files
the user has downloaded. Check each license yourself before sharing derived datasets.
The fixtures in `data/fixtures/` are **synthetic**, written for this repo, and only exercise the pipeline.
