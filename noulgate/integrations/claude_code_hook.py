"""Claude Code hook: PreToolUse (gate), PostToolUse / PostToolUseFailure (outcome logging).

Contract (code.claude.com/docs/en/hooks, checked 2026-10-11):
  * stdin: one JSON object (session_id, transcript_path, cwd, permission_mode, hook_event_name,
    tool_name, tool_input, tool_use_id, ...).
  * exit 0 + empty stdout  -> no decision; the call goes through Claude Code's normal permission flow.
  * exit 0 + JSON {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": ...}}
    -> allow (skips the permission prompt), ask (forces a prompt), deny (blocks).
  * PostToolUse / PostToolUseFailure carry the same tool_use_id, so a decision can be joined to what
    happened next. No hook event reports the user's answer to an "ask"; `noulgate harvest` infers it.

Log mode (M0) never prints a decision: emitting "allow" would *skip the user's permission prompt*,
which is a behavior change, so "always allow" is implemented as "never interfere".

Enforce mode posts the payload to `noulgate serve` and maps the gate's decision:
  deny -> permissionDecision "deny"; ask -> "ask"; allow -> no output (normal permission flow) unless
  grant_allow is set, in which case "allow" (the gate may then skip the user's prompt).
If the server is unreachable, slow, or errors, the hook answers "ask": fail to a human, never to allow.

This module imports only the standard library: it runs once per tool call, and importing pydantic
alone took ~100 ms of the old 170 ms hook. Records are plain dicts shaped like `schema.AuditRecord`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._jsonl import append_record, default_log_path, new_record, truncate_input

DEFAULT_SERVER = "http://127.0.0.1:7701"
OUTCOME_EVENTS = {"PostToolUse": "ran", "PostToolUseFailure": "failed"}

Post = Callable[[str, dict[str, Any], float], dict[str, Any]]


class HookResult:
    __slots__ = ("exit_code", "stdout", "stderr")

    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = ""):
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr


class BadInput(ValueError):
    pass


def parse_payload(stdin_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdin_text)
    except json.JSONDecodeError as exc:
        raise BadInput(f"not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BadInput("payload is not a JSON object")
    for key in ("session_id", "tool_name"):
        if not isinstance(payload.get(key), str):
            raise BadInput(f"missing or non-string {key!r}")
    if not isinstance(payload.get("tool_input", {}), dict):
        raise BadInput("tool_input is not an object")
    return payload


def _str_or_none(v: Any) -> str | None:
    return v if isinstance(v, str) else None


def _base_record(payload: dict[str, Any], mode: str, kind: str = "decision") -> dict[str, Any]:
    tool_input, cut = truncate_input(payload.get("tool_input") or {})
    return new_record(
        kind=kind,
        hook_event=_str_or_none(payload.get("hook_event_name")) or "PreToolUse",
        mode=mode,
        session_id=payload["session_id"],
        tool_use_id=_str_or_none(payload.get("tool_use_id")),
        tool_name=payload["tool_name"],
        tool_input=tool_input,
        tool_input_truncated=cut,
        cwd=_str_or_none(payload.get("cwd")),
        permission_mode=_str_or_none(payload.get("permission_mode")),
        agent_id=_str_or_none(payload.get("agent_id")),
        transcript_path=_str_or_none(payload.get("transcript_path")),
    )


def _append(log_path: Path | str, record: dict[str, Any]) -> str:
    try:
        append_record(log_path, record)
    except OSError as exc:
        return f"noulgate: could not write audit log: {exc}"
    return ""


def run_outcome(payload: dict[str, Any], log_path: Path | str) -> HookResult:
    """PostToolUse / PostToolUseFailure: record that the call ran (or failed). Never blocks."""
    rec = _base_record(payload, "log", kind="outcome")
    rec["tool_input"], rec["tool_input_truncated"] = {}, False  # already logged with the decision
    rec["outcome"] = OUTCOME_EVENTS[payload["hook_event_name"]]
    if rec["outcome"] == "failed":
        rec["error"] = str(payload.get("error", ""))[:500] or None
    return HookResult(stderr=_append(log_path, rec))


def run_log_mode(stdin_text: str, log_path: Path | str) -> HookResult:
    """Observe only. Always exit 0 with empty stdout, even on internal errors (fail open = no change)."""
    try:
        payload = parse_payload(stdin_text)
    except BadInput as exc:
        return HookResult(
            stderr=_append(
                log_path, new_record(tool_name="<unparseable>", error=f"bad hook input: {exc}"[:500])
            )
        )
    if payload.get("hook_event_name") in OUTCOME_EVENTS:
        return run_outcome(payload, log_path)
    return HookResult(stderr=_append(log_path, _base_record(payload, "log")))


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


def urllib_post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    import urllib.request  # imported here: log mode never needs it (~20 ms)

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local server URL from config
        return json.loads(resp.read())


def run_enforce_mode(
    stdin_text: str,
    log_path: Path | str,
    server: str = DEFAULT_SERVER,
    timeout: float = 10.0,
    grant_allow: bool = False,
    post: Post = urllib_post,
) -> HookResult:
    t0 = time.perf_counter()
    try:
        payload = parse_payload(stdin_text)
    except BadInput as exc:
        reason = "NOUL-GATE could not parse the pending tool call; asking you instead."
        rec = new_record(
            tool_name="<unparseable>",
            mode="enforce",
            decision="ask",
            emitted="ask",
            reason=reason,
            error=f"bad hook input: {exc}"[:500],
        )
        return HookResult(stdout=_output("ask", reason), stderr=_append(log_path, rec))
    if payload.get("hook_event_name") in OUTCOME_EVENTS:
        return run_outcome(payload, log_path)
    rec = _base_record(payload, "enforce")
    try:
        result = post(f"{server.rstrip('/')}/v1/decide", payload, timeout)
        decision = result["decision"]
        if decision not in ("allow", "ask", "deny"):
            raise ValueError(f"unknown decision {decision!r}")
        for key in ("p_harm", "risk_class", "answers", "backend", "policy_id"):
            rec[key] = result.get(key)
        reason = result.get("reason") or f"NOUL-GATE {decision}"
    except (OSError, KeyError, ValueError, TypeError) as exc:  # URLError is an OSError
        decision = "ask"
        reason = f"NOUL-GATE decision server unavailable ({type(exc).__name__}); asking you instead."
        rec["error"] = str(exc)[:500]
    rec.update(decision=decision, reason=reason, latency_ms=(time.perf_counter() - t0) * 1000)
    if decision == "allow" and not grant_allow:
        rec["emitted"], out = "none", ""
    else:
        rec["emitted"], out = decision, _output(decision, reason)
    return HookResult(stdout=out, stderr=_append(log_path, rec))


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
    path = log_path or default_log_path()
    if mode == "enforce":
        server = server or os.environ.get("NOULGATE_SERVER", DEFAULT_SERVER)
        result = run_enforce_mode(stdin_text, path, server, timeout, grant_allow)
    else:
        result = run_log_mode(stdin_text, path)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr + "\n")
    return result.exit_code


_VALUE_FLAGS = {"--mode": "mode", "--log": "log_path", "--server": "server", "--timeout": "timeout"}


def _parse_argv(argv: list[str]) -> dict[str, Any]:
    """Tiny flag parser (argparse alone costs ~8 ms, and this runs on every tool call)."""
    opts: dict[str, Any] = {}
    it = iter(argv)
    for arg in it:
        key, _, val = arg.partition("=")
        if key == "--grant-allow":
            opts["grant_allow"] = True
        elif key in _VALUE_FLAGS:
            val = val or next(it, "")
            opts[_VALUE_FLAGS[key]] = float(val) if key == "--timeout" else val
        else:
            raise SystemExit(f"noulgate hook: unknown argument {arg!r}")
    return opts


if __name__ == "__main__":
    raise SystemExit(main(**_parse_argv(sys.argv[1:])))
