"""Append-only JSONL audit log, safe for concurrent hook processes."""

from __future__ import annotations

import json
from pathlib import Path

from ._jsonl import MAX_FIELD_CHARS, append_line, default_log_path, truncate_input  # noqa: F401 (re-exported)
from .schema import AuditRecord


class AuditLog:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path).expanduser() if path else default_log_path()

    def append(self, record: AuditRecord) -> None:
        append_line(self.path, record.model_dump_json() + "\n")

    def read(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [AuditRecord.model_validate(json.loads(line)) for line in f if line.strip()]
