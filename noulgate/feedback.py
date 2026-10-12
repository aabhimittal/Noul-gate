"""Turn the gate's "ask" decisions into labels (`noulgate harvest`).

Every "ask" puts a human in front of a pending call. Their answer is the most valuable label the system
ever sees, because it is given on exactly the actions the gate was unsure about. Claude Code has no hook
that reports the answer (checked 2026-10-11), so it is inferred from what happened next:

  approved   an outcome record (PostToolUse / PostToolUseFailure) exists for the same tool_use_id:
             the tool ran, so the user said yes
  rejected   the transcript's tool_result for that id is an error matching a user-rejection phrase
  no_human   the tool_result is our own "NOUL-GATE ask" reason: nobody was there to answer
             (non-interactive `claude -p` turns an ask into a denial)
  unknown    anything else (session still running, transcript gone, format changed)

The rejection phrases below are best effort: they could not be captured from an interactive session in
the build environment. "approved" does not depend on them.

Bias warning: these labels only cover the ask region (λ_allow ≤ p ≤ λ_deny). Use them to fit the combiner
and to feed the labeling TUI. Do not pour them straight into the conformal calibration set: they are not
exchangeable with all deployed harmful actions.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .datasets import LabeledAction, iter_transcript_calls

Answer = Literal["approved", "rejected", "no_human", "unknown"]

REJECTION_PHRASES = (
    "the user doesn't want to proceed",
    "user rejected",
    "rejected by the user",
    "permission denied by user",
    "the user denied",
)
OUR_REASON_PREFIX = "noul-gate"


@dataclass
class HarvestedAsk:
    decision: dict[str, Any]
    answer: Answer
    evidence: str = ""


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Raw audit records (schema v1 or v2). Malformed lines are skipped, not fatal."""
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _tool_results(transcript_path: str | None, ids: set[str]) -> dict[str, dict[str, Any]]:
    if not transcript_path or not ids:
        return {}
    try:
        lines = Path(transcript_path).read_text(errors="replace").splitlines()
    except OSError:
        return {}
    found: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = entry.get("message") if isinstance(entry, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") in ids
            ):
                found[block["tool_use_id"]] = block
    return found


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return ""


def classify(
    decision: dict[str, Any], outcome: dict[str, Any] | None, tool_result: dict[str, Any] | None
) -> HarvestedAsk:
    if outcome is not None:
        return HarvestedAsk(decision, "approved", f"tool {outcome.get('outcome', 'ran')} after the prompt")
    if tool_result is not None:
        text = _text(tool_result.get("content")).strip()
        low = text.lower()
        if low.startswith(OUR_REASON_PREFIX):
            return HarvestedAsk(decision, "no_human", "the ask was auto-denied (non-interactive session)")
        if tool_result.get("is_error") and any(p in low for p in REJECTION_PHRASES):
            return HarvestedAsk(decision, "rejected", text[:200])
        return HarvestedAsk(decision, "unknown", text[:200])
    return HarvestedAsk(decision, "unknown", "no outcome and no tool_result found")


def harvest(records: Iterable[dict[str, Any]]) -> list[HarvestedAsk]:
    records = list(records)
    outcomes = {r["tool_use_id"]: r for r in records if r.get("kind") == "outcome" and r.get("tool_use_id")}
    asks = [
        r
        for r in records
        if r.get("kind", "decision") == "decision" and r.get("decision") == "ask" and r.get("tool_use_id")
    ]
    by_transcript: dict[str | None, set[str]] = {}
    for r in asks:
        if r["tool_use_id"] not in outcomes:
            by_transcript.setdefault(r.get("transcript_path"), set()).add(r["tool_use_id"])
    results: dict[str, dict[str, Any]] = {}
    for tpath, ids in by_transcript.items():
        results.update(_tool_results(tpath, ids))
    return [classify(r, outcomes.get(r["tool_use_id"]), results.get(r["tool_use_id"])) for r in asks]


def to_labeled(asks: list[HarvestedAsk], auto_label: bool = False) -> list[LabeledAction]:
    """LabeledActions with the transcript context just before each asked call.

    The human answer goes into `hint`. With auto_label, approved -> benign and rejected -> harmful
    (risk class left empty for the TUI); no_human/unknown rows stay unlabeled.
    """
    by_transcript: dict[str, dict[str, HarvestedAsk]] = {}
    for a in asks:
        tp = a.decision.get("transcript_path")
        if tp:
            by_transcript.setdefault(tp, {})[a.decision["tool_use_id"]] = a
    rows = []
    for tp, by_id in by_transcript.items():
        try:
            calls = list(iter_transcript_calls(tp, source="human_ask", only_ids=set(by_id)))
        except OSError:
            continue
        for tool_use_id, action in calls:
            ask = by_id[tool_use_id]
            hint = f"human_{ask.answer}" if ask.answer in ("approved", "rejected") else ask.answer
            update: dict[str, Any] = {"hint": hint, "notes": f"gate p_harm={ask.decision.get('p_harm')}"}
            if auto_label and ask.answer == "approved":
                update.update(label="benign", risk_class="benign", annotator="human_ask")
            elif auto_label and ask.answer == "rejected":
                update.update(label="harmful", annotator="human_ask")
            rows.append(action.model_copy(update=update))
    return rows


def summary(asks: list[HarvestedAsk]) -> dict[str, int]:
    return dict(Counter(a.answer for a in asks))
