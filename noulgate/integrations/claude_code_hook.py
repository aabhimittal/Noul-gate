"""Claude Code PreToolUse hook.

Contract (code.claude.com/docs/en/hooks, checked 2026-10-11):
  * stdin: one JSON object (session_id, transcript_path, cwd, permission_mode, hook_event_name,
    tool_name, tool_input, tool_use_id, ...).
  * exit 0 + empty stdout  -> no decision; the call goes through Claude Code's normal permission flow.
  * exit 0 + JSON {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": ...}}
    -> allow (skips the permission prompt), ask (forces a prompt), deny (blocks).
  * exit 2 -> block, stderr shown to Claude. Any other exit -> non-blocking error.

Log mode (M0) never prints a decision: emitting "allow" would *skip the user's permission prompt*,
which is a behavior change, so "always allow" is implemented as "never interfere".

Enforce mode posts the payload to `noulgate serve` and maps the gate's decision:
  deny -> permissionDecision "deny"; ask -> "ask"; allow -> no output (normal permission flow) unless
  grant_allow is set, in which case "allow" (the gate may then skip the user's prompt).
If the server is unreachable, slow, or errors, the hook answers "ask": fail to a human, never to allow.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..audit import AuditLog, truncate_input
from ..schema import AuditRecord, Decision, HookInput

if TYPE_CHECKING:
    import httpx

DEFAULT_SERVER = "http://127.0.0.1:7701"


@dataclass
class HookResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


def _record_from_input(hook: HookInput, mode: str) -> AuditRecord:
    tool_input, cut = truncate_input(hook.tool_input)
    return AuditRecord(
        source="claude_code_hook",
        mode=mode,
        session_id=hook.session_id,
        tool_use_id=hook.tool_use_id,
        tool_name=hook.tool_name,
        tool_input=tool_input,
        tool_input_truncated=cut,
        cwd=hook.cwd,
        permission_mode=hook.permission_mode,
        agent_id=hook.agent_id,
    )


def run_log_mode(stdin_text: str, log: AuditLog) -> HookResult:
    """Observe only. Always exit 0 with empty stdout, even on internal errors (fail open = no change)."""
    try:
        hook = HookInput.model_validate(json.loads(stdin_text))
        record = _record_from_input(hook, "log")
    except (json.JSONDecodeError, ValidationError) as exc:
        record = AuditRecord(tool_name="<unparseable>", mode="log", error=f"bad hook input: {exc}"[:500])
    try:
        log.append(record)
    except OSError as exc:
        return HookResult(stderr=f"noulgate: could not write audit log: {exc}")
    return HookResult()


def _output(decision: str, reason: str) -> str:
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        }
    )


def run_enforce_mode(
    stdin_text: str,
    log: AuditLog,
    server: str = DEFAULT_SERVER,
    timeout: float = 10.0,
    grant_allow: bool = False,
    client: httpx.Client | None = None,
) -> HookResult:
    import httpx  # imported here so log mode stays fast

    t0 = time.perf_counter()
    try:
        hook = HookInput.model_validate(json.loads(stdin_text))
    except (json.JSONDecodeError, ValidationError) as exc:
        reason = "NOUL-GATE could not parse the pending tool call; asking you instead."
        record = AuditRecord(
            tool_name="<unparseable>",
            mode="enforce",
            decision=Decision.ASK,
            emitted="ask",
            reason=reason,
            error=f"bad hook input: {exc}"[:500],
        )
        _safe_append(log, record)
        return HookResult(stdout=_output("ask", reason))
    record = _record_from_input(hook, "enforce")
    try:
        http = client or httpx.Client(timeout=timeout)
        resp = http.post(f"{server.rstrip('/')}/v1/decide", json=hook.model_dump(mode="json"))
        resp.raise_for_status()
        result = resp.json()
        decision = Decision(result["decision"])
        record.p_harm = result.get("p_harm")
        record.risk_class = result.get("risk_class")
        record.answers = result.get("answers")
        record.backend = result.get("backend")
        record.policy_id = result.get("policy_id")
        reason = result.get("reason") or f"NOUL-GATE {decision.value}"
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        decision = Decision.ASK
        reason = f"NOUL-GATE decision server unavailable ({type(exc).__name__}); asking you instead."
        record.error = str(exc)[:500]
    record.decision = decision
    record.reason = reason
    record.latency_ms = (time.perf_counter() - t0) * 1000
    if decision is Decision.ALLOW and not grant_allow:
        record.emitted = "none"
        out = ""
    else:
        record.emitted = decision.value
        out = _output(decision.value, reason)
    _safe_append(log, record)
    return HookResult(stdout=out)


def _safe_append(log: AuditLog, record: AuditRecord) -> None:
    try:
        log.append(record)
    except OSError as exc:
        sys.stderr.write(f"noulgate: could not write audit log: {exc}\n")


def main(
    argv: list[str] | None = None,
    *,
    mode: str = "log",
    log_path: str | None = None,
    server: str | None = None,
    grant_allow: bool = False,
    timeout: float = 10.0,
) -> int:
    stdin_text = sys.stdin.read()
    if mode == "enforce":
        server = server or os.environ.get("NOULGATE_SERVER", DEFAULT_SERVER)
        result = run_enforce_mode(stdin_text, AuditLog(log_path), server, timeout, grant_allow)
    else:
        result = run_log_mode(stdin_text, AuditLog(log_path))
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr + "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
