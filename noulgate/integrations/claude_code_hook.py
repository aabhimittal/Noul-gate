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
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pydantic import ValidationError

from ..audit import AuditLog, truncate_input
from ..schema import AuditRecord, HookInput


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


def main(argv: list[str] | None = None, *, mode: str = "log", log_path: str | None = None) -> int:
    result = run_log_mode(sys.stdin.read(), AuditLog(log_path))
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr + "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
