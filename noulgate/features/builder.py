"""FeatureBuilder: turn a pending tool call plus session context into the decision state.

state = {tool, args, cwd, last_user_msg, recent_tool_outputs[+provenance], session_stats}  (SPEC.md §5)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..schema import HookInput
from .provenance import (
    Span,
    TranscriptView,
    justification_sources,
    parse_transcript,
    salient_tokens,
    untrusted_only,
)


class SessionStats(BaseModel):
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    tools_used: dict[str, int] = Field(default_factory=dict)
    n_user_messages: int = 0


class GateState(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    cwd: str | None = None
    last_user_msg: Span | None = None
    earlier_user_msgs: list[Span] = Field(default_factory=list)
    recent_tool_outputs: list[Span] = Field(default_factory=list)
    session_stats: SessionStats = Field(default_factory=SessionStats)
    # Derived provenance features.
    justification: dict[str, list[str]] = Field(default_factory=dict)
    untrusted_only: list[str] = Field(default_factory=list)

    def to_model_state(self, budget_chars: int = 6000) -> dict[str, Any]:
        """The JSON object handed to the decision model, with provenance labels and a size budget.

        Latency grows with state length (d1 card: 3.4k tokens costs ~10x a short state), so tool
        outputs are clipped first, oldest first; the pending action and the user's words are kept.
        """

        def clip(text: str, n: int) -> str:
            return text if len(text) <= n else text[:n] + f"…[+{len(text) - n} chars]"

        state: dict[str, Any] = {
            "pending_action": {"tool": self.tool, "args": _clip_args(self.args, 1500)},
            "cwd": self.cwd,
            "user_request": clip(self.last_user_msg.text, 1500) if self.last_user_msg else None,
            "earlier_user_messages": [clip(s.text, 300) for s in self.earlier_user_msgs[-3:]],
            "provenance_of_action_targets": self.justification or None,
            "targets_only_in_untrusted_content": self.untrusted_only or None,
            "recent_context": [],
            "session": self.session_stats.model_dump(),
        }
        used = len(str(state))
        context: list[dict[str, str]] = []
        for span in reversed(self.recent_tool_outputs):  # newest first gets the budget
            room = budget_chars - used
            if room < 200:
                break
            text = clip(span.text, min(1200, room - 80))
            context.insert(0, {"source": span.source, "text": text})
            used += len(text) + len(span.source) + 30
        state["recent_context"] = context
        return {k: v for k, v in state.items() if v not in (None, [], {})}


def _clip_args(args: dict[str, Any], n: int) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        out[k] = v[:n] + "…" if isinstance(v, str) and len(v) > n else v
    return out


class FeatureBuilder:
    def __init__(self, max_tool_outputs: int = 6, max_span_chars: int = 4000):
        self.max_tool_outputs = max_tool_outputs
        self.max_span_chars = max_span_chars

    def build(
        self,
        tool: str,
        args: dict[str, Any],
        view: TranscriptView | None = None,
        cwd: str | None = None,
    ) -> GateState:
        view = view or TranscriptView()
        users = view.user_spans
        tokens = salient_tokens(tool, args)
        # Provenance is checked against everything we saw this session, not just the clipped window.
        just = justification_sources(tokens, view.spans)
        clipped = [
            Span(text=s.text[: self.max_span_chars], source=s.source)
            for s in view.recent_tool_outputs(self.max_tool_outputs)
        ]
        return GateState(
            tool=tool,
            args=args,
            cwd=cwd,
            last_user_msg=users[-1] if users else None,
            earlier_user_msgs=users[:-1][-3:],
            recent_tool_outputs=clipped,
            session_stats=SessionStats(
                n_tool_calls=sum(view.tool_counts.values()),
                n_tool_errors=view.n_tool_errors,
                tools_used=dict(view.tool_counts),
                n_user_messages=len(users),
            ),
            justification=just,
            untrusted_only=untrusted_only(just),
        )

    def from_hook(self, hook: HookInput) -> GateState:
        return self.build(hook.tool_name, hook.tool_input, parse_transcript(hook.transcript_path), hook.cwd)
