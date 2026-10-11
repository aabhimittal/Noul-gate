"""Data models shared across the gate: hook payloads, audit records, decisions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AUDIT_SCHEMA_VERSION = 2  # v2 added kind, hook_event, outcome, transcript_path


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class HookInput(BaseModel):
    """Claude Code PreToolUse stdin payload (code.claude.com/docs/en/hooks, checked 2026-10-11).

    Unknown fields are kept: Claude Code adds fields over time and the audit log should not lose them.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    transcript_path: str | None = None
    cwd: str | None = None
    permission_mode: str | None = None
    hook_event_name: str = "PreToolUse"
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_use_id: str | None = None
    prompt_id: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None


class AuditRecord(BaseModel):
    """One line of the JSONL audit log. Every pending tool call produces exactly one record."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2] = AUDIT_SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # "decision": a pending call (PreToolUse). "outcome": what happened after it (PostToolUse[Failure]).
    kind: Literal["decision", "outcome"] = "decision"
    source: Literal["claude_code_hook", "mcp_proxy", "serve"] = "claude_code_hook"
    hook_event: str | None = None
    mode: Literal["log", "enforce"] = "log"
    session_id: str | None = None
    tool_use_id: str | None = None
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_input_truncated: bool = False
    cwd: str | None = None
    permission_mode: str | None = None
    agent_id: str | None = None
    transcript_path: str | None = None
    # Gate output. All null in log mode (M0): the gate observed but did not decide.
    decision: Decision | None = None
    emitted: Literal["none", "allow", "ask", "deny"] = "none"
    outcome: Literal["ran", "failed"] | None = None
    p_harm: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_class: str | None = None
    answers: dict[str, Any] | None = None
    reason: str | None = None
    backend: str | None = None
    policy_id: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    error: str | None = None
