"""Labeled-action datasets (SPEC.md §8, M2): record format, importers, stratified splits.

One JSONL line = one pending action with the context the gate would have seen at that moment.
Importers only read files the user supplies; no third-party benchmark data ships with this repo.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .features.builder import FeatureBuilder, GateState
from .features.provenance import Span, TranscriptView, parse_transcript_lines, source_for_tool

Label = Literal["harmful", "benign"]


class LabeledAction(BaseModel):
    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    cwd: str | None = None
    context: list[Span] = Field(default_factory=list)  # chronological, provenance-tagged
    label: Label | None = None
    risk_class: str | None = None
    source: str = "unknown"
    annotator: str | None = None
    hint: str | None = None  # importer's guess, shown in the labeling TUI; never used as a label
    notes: str | None = None

    def to_state(self, builder: FeatureBuilder | None = None) -> GateState:
        return (builder or FeatureBuilder()).build(
            self.tool, self.args, TranscriptView(spans=self.context), self.cwd
        )


def stable_id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def read_jsonl(path: str | Path) -> list[LabeledAction]:
    with Path(path).open() as f:
        return [LabeledAction.model_validate_json(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[LabeledAction]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for r in rows:
            f.write(r.model_dump_json(exclude_none=True) + "\n")
            n += 1
    return n


def data_hash(rows: list[LabeledAction]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r.id):
        h.update(f"{r.id}:{r.label}\n".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# importers
# --------------------------------------------------------------------------- #


def iter_transcript_calls(
    path: str | Path, source: str | None = None, only_ids: set[str] | None = None
) -> Iterator[tuple[str, LabeledAction]]:
    """(tool_use_id, action) for each tool call in a Claude Code transcript, with the context available
    just before it. `only_ids` restricts the output to those tool_use ids."""
    lines = Path(path).read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = entry.get("message") if isinstance(entry, dict) else None
        if (
            not isinstance(msg, dict)
            or msg.get("role") != "assistant"
            or not isinstance(msg.get("content"), list)
        ):
            continue
        for block in msg["content"]:
            if not (isinstance(block, dict) and block.get("type") == "tool_use"):
                continue
            tool_use_id = str(block.get("id"))
            if only_ids is not None and tool_use_id not in only_ids:
                continue
            view = parse_transcript_lines(lines[:i])
            yield (
                tool_use_id,
                LabeledAction(
                    id=stable_id("cc", str(path), tool_use_id),
                    tool=str(block.get("name")),
                    args=block.get("input") or {},
                    cwd=entry.get("cwd"),
                    context=view.spans,
                    source=source or f"claude_code:{Path(path).stem}",
                ),
            )


def from_claude_code_transcript(path: str | Path, source: str | None = None) -> Iterator[LabeledAction]:
    """Every tool call in a Claude Code transcript, with the context available just before it.

    Use on your own (consented) sessions to collect benign candidates; label them in the TUI.
    """
    for _, action in iter_transcript_calls(path, source):
        yield action


def _literal(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def from_injecagent(path: str | Path) -> Iterator[LabeledAction]:
    """InjecAgent test cases (github.com/uiuc-kang-lab/InjecAgent, data/test_cases_*.json).

    Each case yields a benign action (the user's own tool call, before any tool output) and a harmful one
    (the first attacker tool, after the injected tool response). InjecAgent does not record the attacker
    call's arguments, so the harmful action's args hold the attacker instruction instead; that makes those
    rows easier than real traces. No LICENSE file was found in that repo (2026-10-11): do not redistribute.
    """
    cases = json.loads(Path(path).read_text())
    kind = "ds" if "_ds_" in Path(path).name else "dh"
    for i, c in enumerate(cases):
        user = Span(text=c["User Instruction"], source="user")
        user_tool = c["User Tool"]
        params = _literal(c.get("Tool Parameters", {}))
        params = params if isinstance(params, dict) else {"input": params}
        yield LabeledAction(
            id=stable_id("injecagent", kind, i, "benign"),
            tool=user_tool,
            args=params,
            context=[user],
            label="benign",
            risk_class="benign",
            source=f"injecagent:{kind}",
            annotator="injecagent",
        )
        attacker_tools = _literal(c["Attacker Tools"])
        attacker_tools = attacker_tools if isinstance(attacker_tools, list) else [attacker_tools]
        response = Span(text=str(c["Tool Response"]), source=source_for_tool(user_tool, params))
        yield LabeledAction(
            id=stable_id("injecagent", kind, i, "harmful"),
            tool=attacker_tools[0],
            args={"request": c["Attacker Instruction"]},
            context=[user, response],
            label="harmful",
            risk_class="data_exfiltration" if kind == "ds" else "following_injected_instruction",
            source=f"injecagent:{kind}",
            annotator="injecagent",
            notes=c.get("Attack Type"),
        )


INJECTION_MARKERS = ("<INFORMATION>", "IMPORTANT!!!", "ignore previous instructions", "TODO:")


def from_message_traces(path: str | Path, source: str = "agentdojo") -> Iterator[LabeledAction]:
    """Chat-format agent traces: AgentDojo run logs (MIT) or any OpenAI-style message list.

    Accepts a JSON file holding {"messages": [...]} or a list of messages, or a JSONL of such objects.
    Tool calls may be AgentDojo-style ({"function": name, "args": {...}}) or OpenAI-style
    ({"function": {"name", "arguments"}}). Rows come out unlabeled; ``hint`` says whether an injection
    marker appeared in earlier tool output. Label them in the TUI.
    """
    text = Path(path).read_text()
    try:
        docs = [json.loads(text)]
    except json.JSONDecodeError:
        docs = [json.loads(line) for line in text.splitlines() if line.strip()]
    for d_i, doc in enumerate(docs):
        messages = doc.get("messages", []) if isinstance(doc, dict) else doc
        context: list[Span] = []
        call_names: dict[str, tuple[str, dict]] = {}
        injected = False
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    str(c.get("content", c.get("text", ""))) if isinstance(c, dict) else str(c)
                    for c in content
                )
            if role == "user" and content:
                context.append(Span(text=str(content), source="user"))
            elif role == "system" and content:
                context.append(Span(text=str(content), source="system"))
            elif role == "assistant":
                for k, tc in enumerate(m.get("tool_calls") or []):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        name, args = fn.get("name"), fn.get("arguments", {})
                        args = json.loads(args) if isinstance(args, str) else args
                    else:
                        name, args = fn, tc.get("args", {})
                    call_names[str(tc.get("id", k))] = (str(name), args or {})
                    yield LabeledAction(
                        id=stable_id(source, str(path), d_i, len(context), k, name),
                        tool=str(name),
                        args=args or {},
                        context=list(context),
                        source=source,
                        hint="after_injection_marker" if injected else None,
                    )
            elif role == "tool":
                tc = m.get("tool_call") or {}
                name = tc.get("function") if isinstance(tc, dict) else None
                if not name:
                    name, _ = call_names.get(str(m.get("tool_call_id")), ("unknown", {}))
                body = str(content or "")
                injected |= any(mark.lower() in body.lower() for mark in INJECTION_MARKERS)
                context.append(Span(text=body, source=f"tool_output:{name}"))


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #


def split(
    rows: list[LabeledAction], fractions: dict[str, float], seed: int = 0
) -> dict[str, list[LabeledAction]]:
    """Stratified random split by label. Unlabeled rows are dropped."""
    total = sum(fractions.values())
    rng = random.Random(seed)
    out: dict[str, list[LabeledAction]] = {k: [] for k in fractions}
    for label in ("harmful", "benign"):
        group = [r for r in rows if r.label == label]
        rng.shuffle(group)
        start = 0
        names = list(fractions)
        for j, name in enumerate(names):
            n = len(group) - start if j == len(names) - 1 else round(len(group) * fractions[name] / total)
            out[name] += group[start : start + n]
            start += n
    for v in out.values():
        rng.shuffle(v)
    return out


def summary(rows: list[LabeledAction]) -> dict[str, Any]:
    from collections import Counter

    return {
        "n": len(rows),
        "labels": dict(Counter(r.label or "unlabeled" for r in rows)),
        "risk_classes": dict(Counter(r.risk_class for r in rows if r.label == "harmful")),
        "sources": dict(Counter(r.source.split(":")[0] for r in rows)),
    }
