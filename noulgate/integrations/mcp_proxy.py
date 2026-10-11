"""MCP proxy mode (M4): sit between any MCP client and an upstream stdio MCP server and gate tools/call.

    client  <--stdio-->  noulgate mcp-proxy  <--stdio-->  upstream server

* tools/list is passed through unchanged.
* tools/call is gated: allow -> forwarded; deny -> an error result carrying the reason (the call never
  reaches the upstream server); ask -> an MCP form elicitation to the user if the client supports it,
  otherwise `--ask-fallback` (default deny: fail closed).
* Every tool result that flows back is recorded as a provenance span ``tool_output:<tool>``, so later calls
  whose targets only appear in earlier tool output are flagged.

Limit: MCP does not carry the user's request, so in proxy mode the gate has no ``user`` spans and the
``user_requested`` signal is weaker than in the Claude Code hook. Only tools are proxied (no resources
or prompts).
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import mcp_types as types

from ..audit import AuditLog, truncate_input
from ..features.provenance import Span
from ..schema import AuditRecord, Decision

Decider = Callable[[str, dict[str, Any], list[Span]], Awaitable[dict[str, Any]]]

APPROVAL_SCHEMA = {
    "type": "object",
    "properties": {"approve": {"type": "boolean", "title": "Run this tool call?", "default": False}},
    "required": ["approve"],
}


def in_process_decider(backend_name: str = "rules", policy_path: str | None = None) -> Decider:
    from ..backends import get_backend
    from ..features.builder import FeatureBuilder
    from ..features.provenance import TranscriptView
    from ..gate import Gate
    from ..policy.conformal import load_policy
    from ..questions import load_questions

    gate = Gate(
        get_backend(backend_name), load_questions(), load_policy(policy_path) if policy_path else None
    )
    builder = FeatureBuilder()

    async def decide(tool: str, args: dict[str, Any], context: list[Span]) -> dict[str, Any]:
        state = builder.build(tool, args, TranscriptView(spans=list(context)))
        return (await anyio.to_thread.run_sync(gate.decide, state)).to_dict()

    return decide


def http_decider(server: str, timeout: float = 10.0) -> Decider:
    import httpx

    async def decide(tool: str, args: dict[str, Any], context: list[Span]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                f"{server.rstrip('/')}/v1/decide_state",
                json={"tool": tool, "args": args, "context": [s.model_dump() for s in context]},
            )
            r.raise_for_status()
            return r.json()

    return decide


def _text_of(result: Any) -> str:
    parts = []
    for c in getattr(result, "content", None) or []:
        if getattr(c, "type", None) == "text":
            parts.append(c.text)
    return "\n".join(parts)


def _error(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


class GateProxy:
    """The gating logic, independent of transports so it can be tested in memory."""

    def __init__(
        self,
        upstream,
        decide: Decider,
        log: AuditLog | None = None,
        ask_fallback: str = "deny",
        max_context: int = 20,
    ):
        self.upstream = upstream
        self.decide = decide
        self.log = log
        self.ask_fallback = ask_fallback
        self.context: list[Span] = []
        self.max_context = max_context

    async def list_tools(self, ctx, params) -> types.ListToolsResult:
        return await self.upstream.list_tools(params=params)

    async def _ask_user(self, ctx, name: str, args: dict[str, Any], reason: str) -> bool | None:
        """True/False if the user answered; None if the client can't be asked."""
        session = getattr(ctx, "session", None)
        if session is None:
            return None
        try:
            caps = types.ClientCapabilities(elicitation=types.ElicitationCapability())
            if not session.check_client_capability(caps):
                return None
            res = await session.elicit_form(
                f"NOUL-GATE wants your approval for tool call `{name}` with arguments {args!r}.\n{reason}",
                APPROVAL_SCHEMA,
                related_request_id=getattr(ctx, "request_id", None),
            )
        except Exception:  # no back-channel, client error: treat as "cannot ask"
            return None
        return bool(res.action == "accept" and (res.content or {}).get("approve"))

    async def call_tool(self, ctx, params: types.CallToolRequestParams):
        name, args = params.name, dict(params.arguments or {})
        tool_input, cut = truncate_input(args)
        record = AuditRecord(
            source="mcp_proxy",
            mode="enforce",
            tool_name=name,
            tool_input=tool_input,
            tool_input_truncated=cut,
        )
        try:
            verdict = await self.decide(name, args, self.context)
            decision = Decision(verdict["decision"])
            reason = verdict.get("reason", "")
            record.p_harm, record.risk_class = verdict.get("p_harm"), verdict.get("risk_class")
            record.answers, record.backend = verdict.get("answers"), verdict.get("backend")
            record.policy_id, record.latency_ms = verdict.get("policy_id"), verdict.get("latency_ms")
        except Exception as exc:  # gate failure: ask, never silently allow
            decision, reason = Decision.ASK, f"NOUL-GATE unavailable ({type(exc).__name__}); asking instead."
            record.error = str(exc)[:500]
        record.decision, record.reason = decision, reason
        if decision is Decision.ASK:
            approved = await self._ask_user(ctx, name, args, reason)
            if approved is None:
                approved = self.ask_fallback == "allow"
                record.reason = f"{reason} [client cannot be asked; fallback={self.ask_fallback}]"
            final = Decision.ALLOW if approved else Decision.DENY
        else:
            final = decision
        record.emitted = final.value
        self._log(record)
        if final is Decision.DENY:
            return _error(f"Blocked by NOUL-GATE: {record.reason}")
        result = await self.upstream.call_tool(name, args)
        text = _text_of(result)
        if text.strip():
            self.context.append(Span(text=text[:8000], source=f"tool_output:{name}"))
            del self.context[: -self.max_context]
        return result

    def _log(self, record: AuditRecord) -> None:
        if self.log is None:
            return
        try:
            self.log.append(record)
        except OSError as exc:
            print(f"noulgate: audit log write failed: {exc}", file=sys.stderr)


async def serve_proxy(
    upstream_cmd: list[str], decide: Decider, log: AuditLog | None, ask_fallback: str
) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    params = StdioServerParameters(command=upstream_cmd[0], args=upstream_cmd[1:])
    async with stdio_client(params) as (ur, uw), ClientSession(ur, uw) as upstream:
        await upstream.initialize()
        proxy = GateProxy(upstream, decide, log, ask_fallback)
        server = Server(
            "noulgate-proxy", version="0.1.0", on_list_tools=proxy.list_tools, on_call_tool=proxy.call_tool
        )
        async with stdio_server() as (sr, sw):
            await server.run(sr, sw, server.create_initialization_options())


def run_proxy(
    upstream_cmd: list[str],
    server: str | None = None,
    backend: str = "rules",
    policy: str | None = None,
    log_path: str | None = None,
    ask_fallback: str = "deny",
) -> int:
    decide = http_decider(server) if server else in_process_decider(backend, policy)
    anyio.run(serve_proxy, upstream_cmd, decide, AuditLog(log_path), ask_fallback)
    return 0
