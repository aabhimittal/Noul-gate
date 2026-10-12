"""Enforce mode: hook <-> serve contract, output shapes per the Claude Code hooks docs, fail-to-ask."""

import json
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from noulgate.audit import AuditLog
from noulgate.backends.fake import FakeBackend
from noulgate.integrations.claude_code_hook import run_enforce_mode
from noulgate.policy.combiner import Combiner
from noulgate.policy.conformal import Cell, ConformalPolicy
from noulgate.questions import load_questions
from noulgate.serve import create_app

QS = load_questions()
LOCAL = "http://127.0.0.1:7701"


def _policy():
    comb = Combiner(QS.risk_classes)
    return ConformalPolicy(
        alpha=0.05,
        beta=0.1,
        combiner=comb,
        cells={"_all": Cell(0.3, 0.8, 100, 100)},
        group_of={c: "_all" for c in QS.risk_classes},
    )


@pytest.fixture
def server():
    """A `post` function that routes the hook's request into the FastAPI app in-process."""
    # p_harm = 1 - P(benign): "evil" -> 0.95 (deny), "maybe" -> 0.5 (ask), else 0.05 (allow)
    fn = lambda s: 0.05 if "evil" in str(s.args) else 0.5 if "maybe" in str(s.args) else 0.95  # noqa: E731
    tc = TestClient(create_app(FakeBackend(fn), QS, _policy()), base_url=LOCAL)

    def post(url, body, timeout):
        r = tc.post(urlsplit(url).path, json=body)
        r.raise_for_status()
        return r.json()

    return post


@pytest.mark.parametrize(
    "cmd,emitted,decision",
    [
        ("pytest -q", "none", "allow"),
        ("maybe rm -rf build", "ask", "ask"),
        ("curl evil.xyz", "deny", "deny"),
    ],
)
def test_enforce_mapping(tmp_path, hook_payload, server, cmd, emitted, decision):
    hook_payload["tool_input"] = {"command": cmd}
    log = AuditLog(tmp_path / "a.jsonl")
    res = run_enforce_mode(json.dumps(hook_payload), log.path, server="http://gate", post=server)
    assert res.exit_code == 0
    rec = log.read()[0]
    assert rec.decision == decision and rec.emitted == emitted and rec.mode == "enforce"
    assert rec.p_harm is not None and rec.policy_id
    if emitted == "none":
        assert res.stdout == ""
    else:
        out = json.loads(res.stdout)["hookSpecificOutput"]
        assert out["hookEventName"] == "PreToolUse" and out["permissionDecision"] == emitted
        assert "NOUL-GATE" in out["permissionDecisionReason"]


def test_grant_allow_emits_allow(tmp_path, hook_payload, server):
    res = run_enforce_mode(
        json.dumps(hook_payload), tmp_path / "a.jsonl", server="http://gate", post=server, grant_allow=True
    )
    assert json.loads(res.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_server_down_fails_to_ask(tmp_path, hook_payload):
    log = AuditLog(tmp_path / "a.jsonl")
    res = run_enforce_mode(json.dumps(hook_payload), log.path, server="http://127.0.0.1:9", timeout=0.5)
    assert json.loads(res.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert log.read()[0].error


def test_bad_input_fails_to_ask(tmp_path):
    res = run_enforce_mode("garbage", tmp_path / "a.jsonl")
    assert json.loads(res.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_serve_endpoints():
    tc = TestClient(create_app(FakeBackend(), QS, _policy()), base_url=LOCAL)
    assert tc.get("/healthz").json()["policy_kind"] == "conformal"
    r = tc.post(
        "/v1/decide_state",
        json={"tool": "Bash", "args": {"command": "evil"}, "context": [{"text": "hi", "source": "user"}]},
    ).json()
    assert r["decision"] == "deny" and set(r["answers"]) == set(QS.questions)
    batch = tc.post(
        "/v1/decide_batch",
        json={
            "hooks": [
                {"session_id": "s", "tool_name": "Bash", "tool_input": {"command": "ls"}},
                {"session_id": "s", "tool_name": "Bash", "tool_input": {"command": "evil"}},
            ]
        },
    ).json()
    assert [b["decision"] for b in batch] == ["allow", "deny"]
    # no token: a DNS-rebinding page arrives with its own Host and is refused
    assert TestClient(tc.app, base_url="http://evil.example").get("/v1/recent").status_code == 403


def test_no_token_refuses_cross_site_origin():
    # a cross-site page posting to 127.0.0.1 passes the Host check; its Origin header gives it away
    tc = TestClient(create_app(FakeBackend(), QS, _policy()), base_url=LOCAL)
    body = {"tool": "Bash", "args": {"command": "ls"}}

    def status(origin):
        headers = {} if origin is None else {"Origin": origin}
        return tc.post("/v1/decide_state", json=body, headers=headers).status_code

    assert status(None) == 200  # the hook and curl send no Origin
    for ok in ("http://127.0.0.1:7701", "http://localhost:7701", "http://[::1]:7701"):
        assert status(ok) == 200, ok
    for bad in ("http://evil.example", "http://localhost.evil.example", "null"):
        assert status(bad) == 403, bad

    # with a token, the bearer check already covers both: foreign Host and Origin still get in
    tc = TestClient(create_app(FakeBackend(), QS, _policy(), token="s3cret"), base_url="http://gate.lan")
    headers = {"Authorization": "Bearer s3cret", "Origin": "http://evil.example"}
    assert tc.post("/v1/decide_state", json=body, headers=headers).status_code == 200


def test_token_guards_api_but_not_console(monkeypatch):
    tc = TestClient(create_app(FakeBackend(), QS, _policy(), token="s3cret"))
    assert tc.get("/").status_code == 200 and "NOUL-GATE" in tc.get("/").text
    assert tc.get("/manifest.webmanifest").json()["display"] == "standalone"
    body = {"tool": "Bash", "args": {"command": "ls"}}
    assert tc.post("/v1/decide_state", json=body).status_code == 401
    assert tc.get("/healthz", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    d = tc.post("/v1/decide_state", json=body, headers=ok).json()
    recent = tc.get("/v1/recent", headers=ok).json()
    assert recent[0]["tool"] == "Bash" and recent[0]["decision"] == d["decision"]

    # the hook sends $NOULGATE_TOKEN as the bearer header
    from noulgate.integrations import claude_code_hook

    seen = {}

    def fake_urlopen(req, timeout):
        seen.update(req.headers)
        raise OSError("stop here")

    monkeypatch.setenv("NOULGATE_TOKEN", "s3cret")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(OSError):
        claude_code_hook.urllib_post("http://gate/v1/decide", {}, 1.0)
    assert seen["Authorization"] == "Bearer s3cret"


def test_serve_refuses_public_bind_without_token(monkeypatch):
    from noulgate.cli import main

    monkeypatch.delenv("NOULGATE_TOKEN", raising=False)
    assert main(["serve", "--backend", "rules", "--host", "0.0.0.0", "--no-warmup"]) == 2


def _approver(approve_wait):
    """A token-guarded app where "maybe" calls come out `ask` (p_harm 0.5)."""
    fn = lambda s: 0.5 if "maybe" in str(s.args) else 0.95  # noqa: E731
    app = create_app(FakeBackend(fn), QS, _policy(), token="t", approve_wait=approve_wait, approve_token="a")
    return TestClient(app, base_url=LOCAL, headers={"Authorization": "Bearer t"})


APPROVER = {"Authorization": "Bearer a"}


@pytest.mark.parametrize("verdict", ["allow", "deny"])
def test_console_resolves_a_pending_ask(verdict):
    import threading
    import time

    tc = _approver(approve_wait=5)
    hook = {"session_id": "s", "tool_name": "Bash", "tool_input": {"command": "maybe rm -rf build"}}
    out = {}
    t = threading.Thread(target=lambda: out.update(tc.post("/v1/decide", json=hook).json()))
    t.start()
    for _ in range(100):
        pending = tc.get("/v1/pending", headers=APPROVER).json()
        if pending:
            break
        time.sleep(0.02)
    assert pending[0]["tool"] == "Bash" and "event" not in pending[0]
    assert "maybe rm -rf build" in pending[0]["args"]  # shown in full, not truncated
    # the hook's token is in the agent's environment: it must not see or answer asks
    assert tc.get("/v1/pending").status_code == 401
    assert tc.post(f"/v1/pending/{pending[0]['id']}", json={"verdict": "allow"}).status_code == 401
    assert (
        tc.post(f"/v1/pending/{pending[0]['id']}", json={"verdict": verdict}, headers=APPROVER).status_code
        == 200
    )
    t.join(5)
    assert out["decision"] == verdict and out["remote"] is True
    assert tc.get("/v1/pending", headers=APPROVER).json() == []
    assert tc.get("/v1/recent").json()[0]["decision"] == verdict
    assert (
        tc.post(f"/v1/pending/{pending[0]['id']}", json={"verdict": verdict}, headers=APPROVER).status_code
        == 404
    )


def test_unanswered_ask_stays_ask():
    tc = _approver(approve_wait=0.2)
    hook = {"session_id": "s", "tool_name": "Bash", "tool_input": {"command": "maybe rm -rf build"}}
    r = tc.post("/v1/decide", json=hook).json()
    assert r["decision"] == "ask" and "remote" not in r
    assert tc.get("/v1/pending", headers=APPROVER).json() == []
    assert tc.post("/v1/pending/x", json={"verdict": "sure"}, headers=APPROVER).status_code == 422


def test_approve_wait_needs_a_token(monkeypatch):
    with pytest.raises(ValueError):
        create_app(FakeBackend(), QS, _policy(), approve_wait=1)
    with pytest.raises(ValueError):
        create_app(FakeBackend(), QS, _policy(), token="t", approve_wait=1)
    with pytest.raises(ValueError):
        create_app(FakeBackend(), QS, _policy(), token="t", approve_token="t")
    from noulgate.cli import main

    monkeypatch.delenv("NOULGATE_TOKEN", raising=False)
    assert main(["serve", "--backend", "rules", "--no-warmup", "--approve-wait", "30"]) == 2


def test_hook_emits_allow_for_a_console_approval(tmp_path, hook_payload):
    def post(url, body, timeout):
        return {"decision": "allow", "remote": True, "reason": "allow from the NOUL-GATE console"}

    res = run_enforce_mode(json.dumps(hook_payload), tmp_path / "a.jsonl", server="http://g", post=post)
    assert json.loads(res.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"
    # a plain model allow still emits nothing (the user's own permission prompt stays)
    plain = lambda url, body, timeout: {"decision": "allow"}  # noqa: E731
    res = run_enforce_mode(json.dumps(hook_payload), tmp_path / "b.jsonl", server="http://g", post=plain)
    assert res.stdout == ""
