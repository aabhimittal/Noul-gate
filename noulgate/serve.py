"""`noulgate serve`: keep the decision model resident and answer gate requests over local HTTP.

The Claude Code hook is a fresh process per tool call, so it cannot load a 3B model itself; in enforce
mode it posts the hook payload here. Bind to localhost only: the API has no authentication.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pydantic import BaseModel, Field

from ._jsonl import new_record
from .backends.base import DecisionBackend
from .features.builder import FeatureBuilder
from .features.provenance import Span, TranscriptView
from .gate import Gate
from .policy.conformal import ConformalPolicy, ThresholdPolicy
from .questions import QuestionSet
from .schema import HookInput

log = logging.getLogger(__name__)


class StateRequest(BaseModel):
    """Harness-neutral request (used by the MCP proxy): the action plus provenance-tagged context."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    context: list[Span] = Field(default_factory=list)
    cwd: str | None = None


class HookBatch(BaseModel):
    hooks: list[HookInput]


def create_app(
    backend: DecisionBackend,
    questions: QuestionSet,
    policy: ConformalPolicy | ThresholdPolicy | None = None,
    warmup: bool = True,
    store=None,
):
    """`store` (a store.helix.HelixStore) records every decision with its provenance and taint graph."""
    from fastapi import FastAPI

    gate = Gate(backend, questions, policy)
    builder = FeatureBuilder()
    lock = threading.Lock()  # one forward pass at a time; models are not assumed thread-safe
    app = FastAPI(title="noulgate", version="0.1.0")

    if warmup:
        backend.warmup(questions)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "backend": backend.name,
            "policy_id": gate.policy.policy_id,
            "policy_kind": gate.policy.kind,
        }

    def remember(
        result: dict[str, Any],
        state,
        *,
        tool: str,
        args: dict[str, Any],
        session_id=None,
        tool_use_id=None,
        transcript_path=None,
        cwd=None,
    ) -> None:
        if store is None:
            return
        rec = new_record(
            tool_name=tool,
            tool_input=args,
            session_id=session_id,
            tool_use_id=tool_use_id,
            transcript_path=transcript_path,
            cwd=cwd,
            source="serve",
            mode="enforce",
            **{
                k: result.get(k)
                for k in ("decision", "p_harm", "risk_class", "backend", "policy_id", "reason")
            },
        )
        try:
            store.add_decision(rec, state)
        except Exception as exc:  # the store must never break a decision
            log.warning("noulgate: store write failed: %s", exc)

    @app.post("/v1/decide")
    def decide(hook: HookInput) -> dict[str, Any]:
        state = builder.from_hook(hook)
        with lock:
            result = gate.decide(state).to_dict()
            remember(
                result,
                state,
                tool=hook.tool_name,
                args=hook.tool_input,
                session_id=hook.session_id,
                tool_use_id=hook.tool_use_id,
                transcript_path=hook.transcript_path,
                cwd=hook.cwd,
            )
            return result

    @app.post("/v1/decide_batch")
    def decide_batch(req: HookBatch) -> list[dict[str, Any]]:
        states = [builder.from_hook(h) for h in req.hooks]
        with lock:
            return [r.to_dict() for r in gate.decide_batch(states)]

    @app.post("/v1/decide_state")
    def decide_state(req: StateRequest) -> dict[str, Any]:
        state = builder.build(req.tool, req.args, TranscriptView(spans=req.context), req.cwd)
        with lock:
            result = gate.decide(state).to_dict()
            remember(result, state, tool=req.tool, args=req.args, cwd=req.cwd)
            return result

    return app
