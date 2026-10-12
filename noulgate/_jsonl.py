"""Standard-library-only helpers for writing audit records.

The Claude Code hook runs as a new process on every tool call, so it must not import pydantic
(that alone costs ~100 ms). It writes plain dicts shaped like `schema.AuditRecord`; a test checks that
every record it produces validates against the model.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
MAX_FIELD_CHARS = 4000

# Every AuditRecord field and its default, in declaration order (kept in sync by test_schema_sync).
RECORD_DEFAULTS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "event_id": None,
    "ts": None,
    "kind": "decision",
    "source": "claude_code_hook",
    "hook_event": None,
    "mode": "log",
    "session_id": None,
    "tool_use_id": None,
    "tool_name": None,
    "tool_input": {},
    "tool_input_truncated": False,
    "cwd": None,
    "permission_mode": None,
    "agent_id": None,
    "transcript_path": None,
    "decision": None,
    "emitted": "none",
    "outcome": None,
    "p_harm": None,
    "risk_class": None,
    "answers": None,
    "reason": None,
    "backend": None,
    "policy_id": None,
    "latency_ms": None,
    "error": None,
}


def default_log_path() -> Path:
    env = os.environ.get("NOULGATE_LOG")
    return Path(env).expanduser() if env else Path.home() / ".noulgate" / "audit.jsonl"


def truncate_input(value: Any, limit: int = MAX_FIELD_CHARS) -> tuple[Any, bool]:
    """Cap long strings (file contents in Write/Edit calls) so the log stays small."""
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + f"…[+{len(value) - limit} chars]", True
        return value, False
    if isinstance(value, dict):
        out, cut = {}, False
        for k, v in value.items():
            out[k], c = truncate_input(v, limit)
            cut |= c
        return out, cut
    if isinstance(value, list):
        items = [truncate_input(v, limit) for v in value]
        return [v for v, _ in items], any(c for _, c in items)
    return value, False


def new_record(**fields: Any) -> dict[str, Any]:
    unknown = set(fields) - set(RECORD_DEFAULTS)
    if unknown:
        raise KeyError(f"not AuditRecord fields: {sorted(unknown)}")
    rec = dict(RECORD_DEFAULTS)
    rec["event_id"] = uuid.uuid4().hex
    rec["ts"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rec.update(fields)
    return rec


def append_line(path: Path | str, line: str) -> None:
    """Append one line atomically w.r.t. other writers (O_APPEND + exclusive flock)."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode())
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def append_record(path: Path | str, record: dict[str, Any]) -> None:
    append_line(path, json.dumps(record, ensure_ascii=False, default=str) + "\n")
