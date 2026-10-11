"""`noulgate serve`: keep the decision model resident and answer gate requests over local HTTP.

The Claude Code hook is a fresh process per tool call, so it cannot load a 3B model itself; in enforce
mode it posts the hook payload here. Bind to localhost only: the API has no authentication.
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, Field

from .backends.base import DecisionBackend
from .features.builder import FeatureBuilder
from .features.provenance import Span, TranscriptView
from .gate import Gate
from .policy.conformal import ConformalPolicy, ThresholdPolicy
from .questions import QuestionSet
from .schema import HookInput


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
):
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

    @app.post("/v1/decide")
    def decide(hook: HookInput) -> dict[str, Any]:
        state = builder.from_hook(hook)
        with lock:
            return gate.decide(state).to_dict()

    @app.post("/v1/decide_batch")
    def decide_batch(req: HookBatch) -> list[dict[str, Any]]:
        states = [builder.from_hook(h) for h in req.hooks]
        with lock:
            return [r.to_dict() for r in gate.decide_batch(states)]

    @app.post("/v1/decide_state")
    def decide_state(req: StateRequest) -> dict[str, Any]:
        state = builder.build(req.tool, req.args, TranscriptView(spans=req.context), req.cwd)
        with lock:
            return gate.decide(state).to_dict()

    return app
