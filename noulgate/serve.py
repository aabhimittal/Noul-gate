"""`noulgate serve`: keep the decision model resident and answer gate requests over local HTTP.

The Claude Code hook is a fresh process per tool call, so it cannot load a 3B model itself; in enforce
mode it posts the hook payload here. Without `token` the API has no authentication: bind to localhost.
It then refuses a non-loopback Host (DNS rebinding) or a non-loopback Origin (cross-site pages).
With `token`, every /v1 and /healthz request needs `Authorization: Bearer <token>`.

`GET /` is a small web console (also installable on a phone as a PWA): live recent decisions and a
form to try a call. The page holds no secrets; it asks for the token and sends it as the header.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid
from collections import deque
from importlib.resources import files
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ._jsonl import new_record
from .backends.base import DecisionBackend
from .features.builder import FeatureBuilder
from .features.provenance import Span, TranscriptView
from .gate import Gate
from .policy.conformal import ConformalPolicy, ThresholdPolicy
from .questions import QuestionSet
from .schema import HookInput

LOOPBACK = ("127.0.0.1", "localhost", "::1")
log = logging.getLogger(__name__)


class StateRequest(BaseModel):
    """Harness-neutral request (used by the MCP proxy): the action plus provenance-tagged context."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    context: list[Span] = Field(default_factory=list)
    cwd: str | None = None


class HookBatch(BaseModel):
    hooks: list[HookInput]


class Verdict(BaseModel):
    verdict: Literal["allow", "deny"]


def create_app(
    backend: DecisionBackend,
    questions: QuestionSet,
    policy: ConformalPolicy | ThresholdPolicy | None = None,
    warmup: bool = True,
    store=None,
    token: str | None = None,
    approve_wait: float = 0.0,
):
    """`store` (a store.helix.HelixStore) records every decision with its provenance and taint graph.

    `approve_wait` > 0: a hook call that comes out `ask` waits up to that many seconds for an Allow/Deny
    from the console (`/v1/pending`), then stays `ask`. It grants tool permissions, so it needs `token`.
    """
    if approve_wait and not token:
        raise ValueError("approve_wait grants tool permissions; it needs a token")
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    gate = Gate(backend, questions, policy)
    builder = FeatureBuilder()
    lock = threading.Lock()  # one forward pass at a time; models are not assumed thread-safe
    app = FastAPI(title="noulgate", version="0.1.0")
    pending: dict[str, dict[str, Any]] = {}  # asks waiting for the console; id -> entry with an Event
    recent: deque[dict[str, Any]] = deque(maxlen=200)  # in-memory only; the JSONL log is the record
    web = files("noulgate") / "web"

    @app.middleware("http")
    async def auth(request: Request, call_next):
        # Without a token, only loopback Host headers: blocks DNS-rebinding pages reading /v1/recent.
        if not token and request.url.hostname not in LOOPBACK:
            return JSONResponse({"detail": "non-loopback Host; start with --token"}, status_code=403)
        # ...and no cross-site Origin: blocks other pages posting to 127.0.0.1 (CSRF). The hook sends none.
        origin = request.headers.get("origin")
        if not token and origin is not None and urlsplit(origin).hostname not in LOOPBACK:
            return JSONResponse({"detail": "cross-site Origin; start with --token"}, status_code=403)
        if token and request.url.path.startswith(("/v1/", "/healthz")):
            sent = request.headers.get("authorization", "")
            if not hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
                return JSONResponse({"detail": "bad or missing bearer token"}, status_code=401)
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def console() -> HTMLResponse:
        return HTMLResponse((web / "index.html").read_text())

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def manifest() -> Response:
        return Response((web / "manifest.webmanifest").read_text(), media_type="application/manifest+json")

    @app.get("/v1/pending")
    def pending_asks() -> list[dict[str, Any]]:
        return [{k: v for k, v in e.items() if k != "event"} for e in list(pending.values())]

    @app.post("/v1/pending/{ask_id}")
    def resolve(ask_id: str, body: Verdict) -> dict[str, Any]:
        entry = pending.get(ask_id)
        if entry is None:
            return JSONResponse({"detail": "no such pending ask (answered or timed out)"}, status_code=404)
        entry["verdict"] = body.verdict
        entry["event"].set()
        return {"id": ask_id, "verdict": body.verdict}

    @app.get("/v1/recent")
    def recent_decisions(limit: int = 50) -> list[dict[str, Any]]:
        return list(recent)[-limit:][::-1]

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
        recent.append(
            {
                "ts": time.time(),
                "tool": tool,
                "args": str(args)[:300],
                "session_id": session_id,
                **{k: result.get(k) for k in ("decision", "p_harm", "risk_class", "reason")},
            }
        )
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
        if result["decision"] == "ask" and approve_wait > 0:  # wait outside the lock
            ask_id = uuid.uuid4().hex
            entry = {
                "id": ask_id,
                "ts": time.time(),
                "tool": hook.tool_name,
                "args": str(hook.tool_input)[:300],
                "reason": result.get("reason"),
                "verdict": None,
                "event": threading.Event(),
            }
            pending[ask_id] = entry
            try:
                entry["event"].wait(approve_wait)
            finally:
                pending.pop(ask_id, None)
            if entry["verdict"]:
                result.update(
                    decision=entry["verdict"],
                    reason=f"{entry['verdict']} from the NOUL-GATE console",
                    remote=True,
                )
        with lock:
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
