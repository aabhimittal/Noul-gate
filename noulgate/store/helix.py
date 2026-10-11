"""HelixDB-backed store for gate decisions, provenance and taint, kept as a graph.

The JSONL audit log answers "what did the gate decide?". The graph answers questions that span calls and
sessions:

  * which sessions read a secret and then sent data out (tainted egress chains)?
  * which hosts or paths keep showing up only in untrusted content and then become an action's target,
    across sessions? That is cross-session threat intel on injection sources.
  * what happened, in order, in one session?

Graph model:

  (Session {session_id}) -HAS_CALL-> (ToolCall {event_id, tool_use_id, tool, args, ts, decision, emitted,
      p_harm, risk_class, backend, policy_id, outcome, human_answer, untrusted_only, tainted_egress,
      egress_kind, fastpath})
  (ToolCall) -NEXT-> (ToolCall)                     calls in session order
  (ToolCall) -TARGETS {untrusted_only}-> (Target {token})
  (Target) -SEEN_IN-> (Source {name, kind})         where the target appeared: user, web:…, file:…, tool_output:…
  (ToolCall) -AFTER_SECRET-> (Secret {secret})      taint sources the session had touched before this call

Connection URIs:  "memory" (in-process, tests) | "embedded:<dir>" (in-process, persistent) |
                  "http://host:6969" (a HelixDB server; `helix start dev` or the Docker image)

Needs `pip install noulgate[helix]` (helix-db, helix-db-embedded). Writes are not concurrent-safe across
processes on one embedded directory: run one writer, such as `noulgate serve --store …`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..features.builder import GateState

CALL_PROPS = (
    "event_id",
    "tool_use_id",
    "tool",
    "ts",
    "session_id",
    "decision",
    "emitted",
    "p_harm",
    "risk_class",
    "backend",
    "policy_id",
    "mode",
    "reason",
    "outcome",
    "human_answer",
    "untrusted_only",
    "tainted_egress",
    "egress_kind",
    "fastpath",
)
UNIQUE_KEYS = {
    "Session": "session_id",
    "ToolCall": "event_id",
    "Target": "token",
    "Source": "name",
    "Secret": "secret",
}


class StoreUnavailable(RuntimeError):
    pass


def _helix():
    try:
        import helixdb
    except ImportError as exc:  # pragma: no cover - optional extra
        raise StoreUnavailable("the HelixDB store needs `pip install noulgate[helix]`") from exc
    return helixdb


def open_client(uri: str):
    h = _helix()
    if uri == "memory":
        return h.Client.embedded(h.InMemory("noulgate"))
    if uri.startswith("embedded:"):
        root = Path(uri[len("embedded:") :]).expanduser()
        root.mkdir(parents=True, exist_ok=True)  # HelixDB requires an existing root
        return h.Client.embedded(h.Disk(str(root.resolve()), "noulgate"))
    if uri.startswith(("http://", "https://")):
        return h.Client(uri)
    raise ValueError(f"store URI must be 'memory', 'embedded:<dir>' or http(s)://…, got {uri!r}")


def _source_kind(name: str) -> str:
    return name.split(":", 1)[0]


class HelixStore:
    def __init__(self, uri: str = "memory"):
        self.uri = uri
        self.h = _helix()
        self.client = open_client(uri)
        self._ids: dict[tuple[str, str], int] = {}
        self._last_call: dict[str, int] = {}
        self._ensure_indexes()

    # ------------------------------------------------------------------ plumbing

    def _q(self, batch) -> dict[str, Any]:
        return self.client.query(batch.to_query_request())

    def _read(self, name: str, traversal) -> list[dict[str, Any]]:
        return self._q(self.h.read_batch().var_as(name, traversal).returning([name]))[name]

    def _ensure_indexes(self) -> None:
        g, IndexSpec = self.h.g, self.h.IndexSpec
        batch = self.h.write_batch()
        names = []
        for i, (label, key) in enumerate([*UNIQUE_KEYS.items(), ("ToolCall", "tool_use_id")]):
            batch = batch.var_as(
                f"ix{i}", g().create_index_if_not_exists(IndexSpec.node_equality(label, key))
            )
            names.append(f"ix{i}")
        self._q(batch.returning(names))

    def _find(self, label: str, key: str, value: Any) -> int | None:
        cached = self._ids.get((label, str(value)))
        if cached is not None:
            return cached
        rows = self._read(
            "n",
            self.h.g()
            .n_with_label_where(label, self.h.SourcePredicate.eq(key, value))
            .limit(1)
            .value_map(["$id"]),
        )
        if rows:
            self._ids[(label, str(value))] = rows[0]["$id"]
            return rows[0]["$id"]
        return None

    def _node(self, label: str, value: Any, props: dict[str, Any] | None = None) -> tuple[int, bool]:
        """Get or create the node with this unique key. Returns (id, created)."""
        key = UNIQUE_KEYS[label]
        found = self._find(label, key, value)
        if found is not None:
            return found, False
        res = self._q(
            self.h.write_batch()
            .var_as("n", self.h.g().add_n(label, {key: value, **(props or {})}))
            .returning(["n"])
        )
        nid = res["n"][0]["$id"]
        self._ids[(label, str(value))] = nid
        return nid, True

    def _edge(self, src: int, label: str, dst: int, props: dict[str, Any] | None = None) -> None:
        g, NodeId = self.h.g, self.h.NodeId
        self._q(
            self.h.write_batch()
            .var_as("e", g().n(NodeId(src)).add_e(label, NodeId(dst), props or {}).count())
            .returning(["e"])
        )

    def _has_edge(self, src: int, label: str, dst: int) -> bool:
        rows = self._read("e", self.h.g().n(self.h.NodeId(src)).out(label).value_map(["$id"]))
        return any(r["$id"] == dst for r in rows)

    # ------------------------------------------------------------------ writes

    def add_decision(self, record: dict[str, Any], state: GateState | None = None) -> bool:
        """Store one decision record (an AuditRecord dict) and, if given, its provenance and taint.

        Idempotent on event_id: returns False (and changes nothing) if the call is already stored."""
        event_id = record.get("event_id")
        if not event_id or self._find("ToolCall", "event_id", event_id) is not None:
            return False
        props = {k: record.get(k) for k in CALL_PROPS if k in record and k != "event_id"}
        props["tool"] = record.get("tool_name") or record.get("tool")
        props["args"] = json.dumps(record.get("tool_input") or {}, ensure_ascii=False)[:4000]
        props["ts"] = str(record.get("ts") or "")
        if state is not None:
            props.update(
                untrusted_only=bool(state.untrusted_only),
                tainted_egress=state.tainted_egress,
                egress_kind=state.egress_kind,
            )
        props["fastpath"] = record.get("risk_class") == "_fastpath"
        props = {k: v for k, v in props.items() if v is not None}
        call, _ = self._node("ToolCall", event_id, props)

        session_id = record.get("session_id")
        if session_id:
            sid, _ = self._node("Session", session_id)
            self._edge(sid, "HAS_CALL", call)
            prev = self._last_call.get(session_id)
            if prev is None:
                prev = self._latest_call(sid, exclude=call)
            if prev is not None:
                self._edge(prev, "NEXT", call)
            self._last_call[session_id] = call

        if state is not None:
            flagged = set(state.untrusted_only)
            for token, sources in state.justification.items():
                tid, _ = self._node("Target", token)
                self._edge(call, "TARGETS", tid, {"untrusted_only": token in flagged})
                for src in sources:
                    sid_, _ = self._node("Source", src, {"kind": _source_kind(src)})
                    if not self._has_edge(tid, "SEEN_IN", sid_):
                        self._edge(tid, "SEEN_IN", sid_)
            for secret in state.taint_sources:
                xid, _ = self._node("Secret", secret)
                self._edge(call, "AFTER_SECRET", xid)
        return True

    def _latest_call(self, session_node: int, exclude: int) -> int | None:
        rows = self._read(
            "c", self.h.g().n(self.h.NodeId(session_node)).out("HAS_CALL").value_map(["$id", "ts"])
        )
        rows = [r for r in rows if r["$id"] != exclude]
        return max(rows, key=lambda r: (str(r.get("ts", "")), r["$id"]))["$id"] if rows else None

    def _set_on_call(self, tool_use_id: str, name: str, value: Any) -> int:
        g, P = self.h.g, self.h.SourcePredicate
        res = self._q(
            self.h.write_batch()
            .var_as(
                "u",
                g()
                .n_with_label_where("ToolCall", P.eq("tool_use_id", tool_use_id))
                .set_property(name, value)
                .count(),
            )
            .returning(["u"])
        )
        return int(res["u"])

    def record_outcome(self, tool_use_id: str, outcome: str) -> int:
        return self._set_on_call(tool_use_id, "outcome", outcome)

    def record_answer(self, tool_use_id: str, answer: str) -> int:
        return self._set_on_call(tool_use_id, "human_answer", answer)

    def ingest(self, records: Iterable[dict[str, Any]], rebuild_provenance: bool = True) -> dict[str, int]:
        """Backfill from audit records (`noulgate store sync`). Provenance is rebuilt from each decision's
        transcript_path with the same FeatureBuilder the gate uses; missing transcripts just skip it."""
        from ..features.builder import FeatureBuilder
        from ..features.provenance import parse_transcript_lines

        records = list(records)
        builder = FeatureBuilder()
        stats = {"decisions_added": 0, "decisions_skipped": 0, "outcomes": 0}
        transcripts: dict[str, list[str]] = {}
        for rec in records:
            if rec.get("kind", "decision") != "decision" or not rec.get("tool_name"):
                continue
            state = None
            tp = rec.get("transcript_path")
            if rebuild_provenance and tp:
                if tp not in transcripts:
                    try:
                        transcripts[tp] = Path(tp).read_text(errors="replace").splitlines()
                    except OSError:
                        transcripts[tp] = []
                lines = _lines_before(transcripts[tp], rec.get("tool_use_id"))
                state = builder.build(
                    rec["tool_name"],
                    rec.get("tool_input") or {},
                    parse_transcript_lines(lines),
                    rec.get("cwd"),
                )
            added = self.add_decision(rec, state)
            stats["decisions_added" if added else "decisions_skipped"] += 1
        for rec in records:
            if rec.get("kind") == "outcome" and rec.get("tool_use_id") and rec.get("outcome"):
                stats["outcomes"] += self.record_outcome(rec["tool_use_id"], rec["outcome"])
        return stats

    # ------------------------------------------------------------------ reads

    def counts(self) -> dict[str, int]:
        g = self.h.g
        out = {}
        for label in ("Session", "ToolCall", "Target", "Source", "Secret"):
            out[label] = int(
                self._q(self.h.read_batch().var_as("c", g().n_with_label(label).count()).returning(["c"]))[
                    "c"
                ]
            )
        return out

    def calls(self) -> list[dict[str, Any]]:
        return self._read("c", self.h.g().n_with_label("ToolCall").value_map(["$id", "args", *CALL_PROPS]))

    def records(self) -> list[dict[str, Any]]:
        """Audit-record-shaped dicts (decisions plus outcome records), for `monitor` and `harvest`."""
        out = []
        for c in self.calls():
            rec = {k: c.get(k) for k in CALL_PROPS if k != "tool"}
            rec.update(kind="decision", tool_name=c.get("tool"), tool_input=json.loads(c.get("args") or "{}"))
            out.append(rec)
            if c.get("outcome") and c.get("tool_use_id"):
                out.append(
                    {
                        "kind": "outcome",
                        "tool_use_id": c["tool_use_id"],
                        "tool_name": c.get("tool"),
                        "outcome": c["outcome"],
                        "session_id": c.get("session_id"),
                    }
                )
        return out

    def tainted_egress_chains(self) -> list[dict[str, Any]]:
        g, P, NodeId = self.h.g, self.h.SourcePredicate, self.h.NodeId
        calls = self._read(
            "c",
            g()
            .n_with_label_where("ToolCall", P.eq("tainted_egress", True))
            .value_map(["$id", "event_id", "tool", "args", "decision", "egress_kind", "ts"]),
        )
        out = []
        for c in calls:
            secrets = self._read("s", g().n(NodeId(c["$id"])).out("AFTER_SECRET").value_map(["secret"]))
            sessions = self._read("s", g().n(NodeId(c["$id"])).in_("HAS_CALL").value_map(["session_id"]))
            out.append(
                {
                    **{k: v for k, v in c.items() if k != "$id"},
                    "secrets": sorted(s["secret"] for s in secrets),
                    "session_id": sessions[0]["session_id"] if sessions else None,
                }
            )
        return sorted(out, key=lambda r: str(r.get("ts")))

    def injected_targets(self, min_calls: int = 1) -> list[dict[str, Any]]:
        """Targets that never appeared in user text but were acted on, with the untrusted sources they
        came from and the decisions the gate made on them."""
        g, NodeId = self.h.g, self.h.NodeId
        out = []
        for t in self._read("t", g().n_with_label("Target").value_map(["$id", "token"])):
            sources = [
                s["name"] for s in self._read("s", g().n(NodeId(t["$id"])).out("SEEN_IN").value_map(["name"]))
            ]
            if not sources or "user" in sources:
                continue
            calls = self._read(
                "c", g().n(NodeId(t["$id"])).in_("TARGETS").value_map(["decision", "session_id", "tool"])
            )
            if len(calls) < min_calls:
                continue
            out.append(
                {
                    "target": t["token"],
                    "sources": sorted(set(sources)),
                    "calls": len(calls),
                    "sessions": len({c.get("session_id") for c in calls}),
                    "decisions": sorted(str(c.get("decision")) for c in calls),
                }
            )
        return sorted(out, key=lambda r: (-r["sessions"], -r["calls"], r["target"]))

    def session_timeline(self, session_id: str) -> list[dict[str, Any]]:
        g, NodeId = self.h.g, self.h.NodeId
        sid = self._find("Session", "session_id", session_id)
        if sid is None:
            return []
        calls = self._read(
            "c",
            g()
            .n(NodeId(sid))
            .out("HAS_CALL")
            .value_map(
                [
                    "$id",
                    "ts",
                    "tool",
                    "args",
                    "decision",
                    "p_harm",
                    "outcome",
                    "tainted_egress",
                    "untrusted_only",
                    "human_answer",
                ]
            ),
        )
        return sorted(calls, key=lambda c: (str(c.get("ts")), c["$id"]))

    def close(self) -> None:
        self.client.close()


def _lines_before(lines: list[str], tool_use_id: str | None) -> list[str]:
    """Transcript lines up to the assistant message that issued this tool call (its context)."""
    if not tool_use_id:
        return lines
    for i, line in enumerate(lines):
        if f'"{tool_use_id}"' in line and '"tool_use"' in line:
            return lines[:i]
    return lines
