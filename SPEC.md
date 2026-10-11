# NOUL-GATE — Specification

> Verbatim copy of the project spec this repository implements. Deviations forced by
> the real interfaces of Claude Code and d1-3B are listed in [docs/spec-diff.md](docs/spec-diff.md).

## 1. Thesis

Every tool call an agent makes should pass a typed, calibrated decision gate before execution. A local decision model answers a fixed set of questions about the pending action in one forward pass; thresholds are set with conformal risk control so the rate of harmful actions that get allowed is bounded by a user-chosen α. Uncertain actions go to a human instead of being guessed.

Analogy: an airport metal detector. It is fast and cheap, it waves most people through, and it sends a calibrated fraction to a manual search. The point is not that it never errs, it's that you know its miss rate and can tune it.

## 2. Why now

- Liquid AI released open-weight d1-3B (Oct 2026): typed answers (noul = yes/no with probability, choice, score) in one forward pass, no generated tokens. Reported ~8 ms on an RTX 4090 and ~16 ms on Jetson AGX Thor. Model card: https://huggingface.co/LiquidAI/d1-3B
- TypeSafe positions Jev for exactly this use: scoring, guardrailing and jailbreak detection of prompts, traces and outputs (https://typesafe.ai/blog/introducing-system-one-models-and-jev).
- Tool poisoning and indirect injection succeed at high rates against major agents; LLM-as-judge guards are too slow to run on every tool call.

## 3. Prior art & differentiation

| Existing | What it does | Gap NOUL-GATE fills |
|---|---|---|
| Llama Guard / Prompt Guard style classifiers | Classify text content | Not action-level; no provenance of who asked for the action |
| Static config scanners (e.g., AgentShield) | Scan configs and permissions before runtime | No runtime per-action judgment |
| Rule engines / allowlists | Deterministic allow/deny | Can't judge intent; brittle |
| LLM-as-judge guards | Flexible judgment | 1–10 s latency, uncalibrated confidence |

Differentiator: action-level features + provenance + calibrated probabilities + a stated statistical guarantee + a latency budget.

## 4. Scope

- MVP: Claude Code PreToolUse hook → d1-3B local → allow/ask/deny → JSONL audit log.
- v1: MCP proxy mode (any MCP client), conformal calibration CLI, benchmark report.
- Non-goals: content moderation of chat text; replacing static permission systems (see PERMIT, spec 12).

## 5. Architecture

```
agent harness ──PreToolUse hook / MCP proxy──▶ FeatureBuilder
                                                  │  state = {tool, args, cwd, last_user_msg,
                                                  │           recent_tool_outputs[+provenance], session_stats}
                                                  ▼
                                          DecisionBackend  (d1-3B | Jev adapter | LLM fallback)
                                                  │  answers to the question set
                                                  ▼
                                          ConformalPolicy  (λ_allow, λ_deny per risk class)
                                                  │
                         allow ◀──────────────────┼──────────────▶ deny
                                                  ▼
                                             ask human
                                                  │
                                           AuditLog (JSONL)
```

Provenance is the key feature. Each text span in the state is tagged with where it came from: user, system, tool_output:<tool>, file:<path>, web:<domain>. An action whose justification only appears in tool output or web content is the signature of indirect injection.

## 6. Interfaces

Question set (YAML, user-extensible):

```yaml
questions:
  user_requested: {type: noul, instructions: "Did the user (not tool output or web content) request this action, directly or as an obvious step?"}
  risk_class: {type: choice, options: [benign, data_exfiltration, destructive, privilege_escalation, following_injected_instruction, persistence]}
  severity: {type: score, scale: [1,5], instructions: "Worst plausible impact if this action is malicious"}
```

CLI:

```
noulgate serve --backend d1-3b --port 7701
noulgate calibrate --data cal.jsonl --alpha 0.02 --out policy.json
noulgate eval --data test.jsonl --policy policy.json --report report.html
noulgate install claude-code      # writes the hook config after showing a diff
```

## 7. Core method: conformal risk control on false-allows

Let p(x) be the backend's probability that action x is harmful (combine risk_class ≠ benign with ¬user_requested; start with p = 1 − P(benign) and learn a small logistic combiner on calibration data).

Calibration set: n_h harmful actions with scores p_1…p_{n_h}. Allow an action iff p(x) < λ. Choose
λ̂ = the k-th smallest harmful score, with k = ⌊α(n_h + 1)⌋.
Under exchangeability, for a new harmful action: P(p(x_new) < λ̂) ≤ α. So the false-allow rate among harmful actions is at most α, marginally.

Then pick λ_deny from the benign set the same way to bound false-deny rate β; actions with λ̂ ≤ p < λ_deny go to a human. Report the ask rate — that is the human cost of the guarantee.

Caveats to state in the README: the guarantee is marginal, assumes exchangeability between calibration and deployment, and breaks under adaptive attackers. Re-calibrate on fresh attack data and treat the bound as a monitoring target, not a proof of safety.

## 8. Evaluation

- Data: benign traces from real agent sessions (your own, consented), plus attack traces from public agent-injection benchmarks (check AgentDojo / InjecAgent licenses) and REDLOOP (spec 09) output.
- Metrics: empirical false-allow vs α (does the bound hold on held-out data?), ask rate, false-deny rate, p50/p99 latency, throughput (batch of pending calls).
- Baselines: rules only, Haiku-class LLM judge, uncalibrated d1 at a fixed 0.5 threshold.
- Headline plot: false-allow rate vs ask rate for each backend, with latency annotated.

## 9. Milestones

- M0 Hook stub that logs every pending tool call as JSONL and always allows. Acceptance: a Claude Code session produces a valid log; zero behavior change.
- M1 FeatureBuilder with provenance tags + d1-3B backend. Acceptance: unit tests for provenance tagging; p99 latency measured and reported on available hardware.
- M2 Dataset assembly + labeling tool (simple TUI). Acceptance: ≥500 benign, ≥300 harmful labeled actions, split cal/test.
- M3 Conformal calibration. Acceptance: on test split, empirical false-allow ≤ α + sampling tolerance across 20 random splits.
- M4 MCP proxy mode. Acceptance: works with one non-Claude-Code MCP client.
- M5 Benchmark report + README with honest limits.

## 10. Repo layout & stack

```
noulgate/
  features/ (provenance.py, builder.py)
  backends/ (base.py, d1.py, jev.py, llm.py)
  policy/ (conformal.py, combiner.py)
  integrations/ (claude_code_hook.py, mcp_proxy.py)
  cli.py
data/ eval/ tests/ docs/
```

Python 3.11+, transformers or llama.cpp (GGUF) for d1-3B, FastAPI for serve mode, pytest + hypothesis.

## 11. Risks & honest weaknesses

- The gate is itself an attack target: adversarial text can push the classifier's probability down.
- 3B on CPU is far slower than the GPU figures; document real numbers on laptop hardware.
- Labels for "harmful" are judgment calls; inter-annotator disagreement caps achievable accuracy.
- d1 license terms may limit some users; keep the backend pluggable.

## 12. Verify before building

- Current d1-3B Python interface (system_one, system_one_batch) and question schema from the model card.
- Current Claude Code hook event names, payload schema and exit-code semantics from official docs.
- Licenses of any attack benchmark you redistribute.

## 13. Claude Code kickoff

"Read SPEC.md. Implement M0 only: a PreToolUse hook that logs pending tool calls to JSONL and always allows. First, read the current Claude Code hooks documentation and list any differences from the spec. Write tests for the log schema. Stop after M0 and summarize."
