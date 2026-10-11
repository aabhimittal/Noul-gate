"""Append-only JSONL audit log, safe for concurrent hook processes."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .schema import AuditRecord

DEFAULT_LOG = Path.home() / ".noulgate" / "audit.jsonl"
MAX_FIELD_CHARS = 4000


def default_log_path() -> Path:
    env = os.environ.get("NOULGATE_LOG")
    return Path(env).expanduser() if env else DEFAULT_LOG


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


class AuditLog:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path).expanduser() if path else default_log_path()

    def append(self, record: AuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json() + "\n"
        # O_APPEND + an exclusive lock keeps lines whole when several hook processes write at once.
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def read(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [AuditRecord.model_validate(json.loads(line)) for line in f if line.strip()]
