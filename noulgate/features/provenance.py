"""Provenance: tag every text span in the decision state with where it came from.

Sources: ``user``, ``system``, ``tool_output:<tool>``, ``file:<path>``, ``web:<domain>``.
An action whose justification (its URLs, paths, hosts, addresses) appears only in tool output,
files or web content, never in what the user wrote, is the signature of indirect prompt injection.

The Claude Code transcript is an internal, undocumented JSONL format (see docs/spec-diff.md H3).
The parser below only relies on ``message.role`` / ``message.content`` blocks of type ``text``,
``tool_use`` and ``tool_result`` and degrades to an empty view on anything else.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

TRUSTED_SOURCES = frozenset({"user"})
_SYSTEM_PREFIXES = ("<system-reminder>", "<command-", "<local-command-", "<task-notification>", "Caveat:")


class Span(BaseModel):
    text: str
    source: str

    @property
    def kind(self) -> str:
        return self.source.split(":", 1)[0]

    @property
    def trusted(self) -> bool:
        return self.source in TRUSTED_SOURCES


def domain_of(url: str) -> str:
    host = urlparse(url if "://" in url else f"https://{url}").hostname or ""
    return host.lower().removeprefix("www.")


def source_for_tool(tool_name: str, tool_input: dict[str, Any] | None) -> str:
    """Provenance tag for the output of a tool call."""
    tool_input = tool_input or {}
    if tool_name in {"Read", "NotebookRead"}:
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        if path:
            return f"file:{path}"
    if tool_name == "WebFetch" and tool_input.get("url"):
        return f"web:{domain_of(tool_input['url'])}"
    if tool_name == "WebSearch":
        return "web:search"
    return f"tool_output:{tool_name}"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return "" if content is None else json.dumps(content)


def _user_text_source(text: str, is_meta: bool) -> str:
    return "system" if is_meta or text.lstrip().startswith(_SYSTEM_PREFIXES) else "user"


@dataclass
class TranscriptView:
    """What the gate knows about the session so far, with provenance."""

    spans: list[Span] = field(default_factory=list)  # chronological
    tool_counts: Counter = field(default_factory=Counter)
    n_tool_errors: int = 0
    parse_errors: int = 0

    @property
    def user_spans(self) -> list[Span]:
        return [s for s in self.spans if s.source == "user"]

    @property
    def last_user_msg(self) -> Span | None:
        users = self.user_spans
        return users[-1] if users else None

    def recent_tool_outputs(self, n: int) -> list[Span]:
        outs = [s for s in self.spans if s.source not in ("user", "system")]
        return outs[-n:] if n else []


def parse_transcript_lines(lines: list[str]) -> TranscriptView:
    view = TranscriptView()
    tool_uses: dict[str, tuple[str, dict[str, Any]]] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            view.parse_errors += 1
            continue
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict):
            continue
        role, content = msg.get("role"), msg.get("content")
        is_meta = bool(entry.get("isMeta"))
        if role == "assistant" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name, inp = str(block.get("name", "?")), block.get("input") or {}
                    tool_uses[str(block.get("id"))] = (name, inp)
                    view.tool_counts[name] += 1
        elif role == "user":
            if isinstance(content, str):
                if content.strip():
                    view.spans.append(Span(text=content, source=_user_text_source(content, is_meta)))
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    name, inp = tool_uses.get(str(block.get("tool_use_id")), ("unknown", {}))
                    if block.get("is_error"):
                        view.n_tool_errors += 1
                    text = _content_text(block.get("content"))
                    if text.strip():
                        view.spans.append(Span(text=text, source=source_for_tool(name, inp)))
                elif block.get("type") == "text":
                    text = str(block.get("text", ""))
                    if text.strip():
                        view.spans.append(Span(text=text, source=_user_text_source(text, is_meta)))
    return view


def parse_transcript(path: str | Path | None, max_bytes: int = 2_000_000) -> TranscriptView:
    """Read the tail of a transcript file. Missing or unreadable files give an empty view."""
    if not path:
        return TranscriptView()
    p = Path(path)
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # drop the partial first line
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return TranscriptView()
    return parse_transcript_lines(data.splitlines())


# --------------------------------------------------------------------------- #
# salient tokens: the parts of an action that need a justification
# --------------------------------------------------------------------------- #

_URL = re.compile(r"\b(?:https?|ftp|s3|gs|ssh|git)://[^\s'\"<>`)]+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HOST = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ai|dev|app|xyz|ru|cn|co|me|sh|site|info|biz)\b", re.I
)
_PATH = re.compile(r"(?:~|\.{1,2})?/[\w.@%+-]+(?:/[\w.@%+-]+)*")
_COMMON_PATHS = {"/dev/null", "/tmp", "/bin/sh", "/bin/bash", "/usr/bin/env"}


def salient_tokens(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    """URLs, hosts, e-mail addresses, IPs and paths in the action's arguments."""
    text = json.dumps(tool_input, ensure_ascii=False) if not isinstance(tool_input, str) else tool_input
    text = text.replace("\\/", "/")
    found: list[str] = []
    for url in _URL.findall(text):
        found += [url.rstrip(".,;"), domain_of(url)]
    found += _EMAIL.findall(text)
    found += _IP.findall(text)
    found += [h.lower() for h in _HOST.findall(text)]
    without_urls = _URL.sub(" ", text)
    found += [p for p in _PATH.findall(without_urls) if len(p) > 2 and p not in _COMMON_PATHS]
    seen, out = set(), []
    for tok in found:
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def justification_sources(tokens: list[str], spans: list[Span]) -> dict[str, list[str]]:
    """For each salient token, the provenance sources whose text mentions it (case-insensitive)."""
    lowered = [(s.text.lower(), s.source) for s in spans]
    out: dict[str, list[str]] = {}
    for tok in tokens:
        t = tok.lower()
        srcs = sorted({src for text, src in lowered if t in text})
        out[tok] = srcs
    return out


def untrusted_only(sources: dict[str, list[str]]) -> list[str]:
    """Tokens that appear in context only in untrusted spans: the indirect-injection signature."""
    return [tok for tok, srcs in sources.items() if srcs and not any(s in TRUSTED_SOURCES for s in srcs)]
