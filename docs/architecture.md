# Architecture

```
Claude Code ──PreToolUse hook──┐                       ┌── noulgate serve (resident model, 127.0.0.1:7701)
                               │  HookInput (stdin)    │
any MCP client ──mcp-proxy─────┤──────────────────────▶│  FeatureBuilder    features/builder.py
                               │                       │    provenance       features/provenance.py
                               │                       │  DecisionBackend   backends/{d1,jev,llm,rules}.py
                               │                       │  Combiner          policy/combiner.py
                               │                       │  ConformalPolicy   policy/conformal.py
                               ◀── allow / ask / deny ─┘
                               │
                         AuditLog (JSONL)  audit.py, schema.AuditRecord
```

## Round 2 additions at a glance

```
PreToolUse ──▶ hook (stdlib only, ~36 ms) ──▶ serve: FeatureBuilder ─▶ provenance + taint
                                                         │
                                       fast path? ──yes──▶ p = 0, no model call
                                                         │ no
                                                         ▼
                                                 DecisionBackend ─▶ combiner ─▶ conformal policy
PostToolUse / PostToolUseFailure ──▶ hook ──▶ outcome record (same tool_use_id)

audit.jsonl ──▶ noulgate harvest  (ask + outcome/transcript ─▶ human answer ─▶ labeled candidates)
            └─▶ noulgate monitor  (rates per window, ask approval, PSI/KS drift ─▶ exit 1 on alert)
```

## One pending tool call, step by step

1. **Hook** (`integrations/claude_code_hook.py`). Claude Code runs the hook command with the
   PreToolUse payload on stdin. In log mode (M0) the hook writes an `AuditRecord` and prints nothing.
   In enforce mode it POSTs the payload to `noulgate serve`.
2. **FeatureBuilder** reads the session transcript (`transcript_path`) and builds a `GateState`:
   the pending tool and args, the last user message, recent tool outputs, and session counters.
   Every span carries a provenance tag: `user`, `system`, `tool_output:<tool>`, `file:<path>` or
   `web:<domain>`.
3. **Provenance features.** The action's *targets* (URLs, hosts, paths, e-mail addresses, IPs) are
   extracted and looked up in the session's spans. A target that appears only in untrusted spans
   (tool output, files, web pages) and never in what the user typed is flagged in
   `targets_only_in_untrusted_content`. That flag is the indirect-injection signature.
4. **DecisionBackend** answers the question set (`data/questions.yaml`) in one call:
   `user_requested` (noul), `risk_class` (choice) and `severity` (score).
5. **Combiner** maps the answers to one number, p(x) = P(harmful). The untrained default is
   1 − P(benign). After calibration it is a small logistic regression over the answers plus the
   provenance flag.
6. **ConformalPolicy** compares p(x) with the λ pair of the action's cell (its predicted risk
   class, or the pooled cell): allow if p < λ_allow, deny if p > λ_deny, otherwise ask.
7. **Back in the hook**: deny → `permissionDecision: "deny"`; ask → `"ask"`; allow → no output, so
   Claude Code's own permission rules still apply. `--grant-allow` makes the gate emit `"allow"`.
   Every decision is appended to the audit log.

## Session taint (features/taint.py)

Provenance asks "who asked for this?". Taint asks "what has this session already touched?". A session
is tainted once a credential-like file was read (a `file:` span whose path matches the secret pattern)
or secret-looking content appeared in tool output (private keys, cloud and API tokens, `PASSWORD=`
literals). A tainted session followed by an egress call (network commands, web fetches, messaging or
publishing MCP tools, writes outside the project) sets `tainted_egress`. This catches the chain
"read a key, later upload notes to a host the user named", where provenance sees nothing unusual.
Taint reaches the model state (`session_taint`) and the combiner (`secret_in_args`, `tainted_egress`).

## Fast path (policy/fastpath.py)

Reads inside the project and a few read-only shell commands score p = 0 without a model call. The policy
records whether it was calibrated with the fast path, and the gate follows that setting. Calibration then
covers the fast path too: if it ever lets harmful actions through, λ_allow drops to 0 and fast-path calls
are no longer auto-allowed.

## Failure behaviour

| Failure | Log mode | Enforce mode (hook) | MCP proxy |
|---|---|---|---|
| Malformed payload | logged, exit 0, no output | `ask` | n/a |
| Decision server down or slow | n/a | `ask` | `ask` → elicitation or fallback |
| Hook process times out or crashes | fails open (Claude Code default) | blocked (`onFailure: "block"`) | n/a |
| Audit log not writable | warning on stderr, exit 0 | warning; the decision still applies | warning |

## Why a server

A hook is a new process for every tool call. It imports only the standard library: importing pydantic
alone used to cost ~100 ms, and the hook went from 170 ms to 36 ms per call when it stopped doing so. Loading d1-3B takes seconds and 6–12 GB of RAM, so
the model lives in `noulgate serve` and the hook is a ~50 ms HTTP client. `serve` binds to
127.0.0.1 and has no authentication, so do not expose it.

## Batching

`DecisionBackend.decide_batch` exists so that `serve` and evaluation can pack several pending
calls into one d1 pass (`system_one_batch`). The hook sends one call at a time. Claude Code runs
parallel tool calls as separate hook processes, and the server serialises them behind a lock.
Micro-batching concurrent requests is a possible improvement; it is not implemented.
