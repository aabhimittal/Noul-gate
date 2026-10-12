"""M0: the hook logs every pending call as a valid record and never changes Claude Code's behavior."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from noulgate._jsonl import RECORD_DEFAULTS, new_record
from noulgate.audit import AuditLog, truncate_input
from noulgate.integrations.claude_code_hook import run_log_mode
from noulgate.schema import AuditRecord

SCHEMA_FILE = Path(__file__).parent.parent / "docs" / "audit_log.schema.json"


def test_log_mode_records_call_and_emits_nothing(tmp_path, hook_payload):
    log = AuditLog(tmp_path / "a.jsonl")
    result = run_log_mode(json.dumps(hook_payload), log.path)
    assert (result.exit_code, result.stdout) == (0, "")
    [rec] = log.read()
    assert rec.tool_name == "Bash" and rec.tool_input["command"] == "npm test"
    assert rec.session_id == "abc123" and rec.tool_use_id == "toolu_01ABC123"
    assert rec.mode == "log" and rec.decision is None and rec.emitted == "none"


def test_unknown_payload_fields_are_tolerated(tmp_path, hook_payload):
    hook_payload["effort"] = {"level": "high"}
    hook_payload["some_future_field"] = 1
    log = AuditLog(tmp_path / "a.jsonl")
    assert run_log_mode(json.dumps(hook_payload), log.path).exit_code == 0
    assert log.read()[0].error is None


@pytest.mark.parametrize(
    "stdin",
    ["", "not json", "[]", '{"tool_name": "Bash"}', '{"session_id": "s", "tool_name": "B", "tool_input": 3}'],
)
def test_malformed_input_still_exits_zero_and_logs_error(tmp_path, stdin):
    log = AuditLog(tmp_path / "a.jsonl")
    result = run_log_mode(stdin, log.path)
    assert (result.exit_code, result.stdout) == (0, "")
    [rec] = log.read()
    assert rec.error and rec.tool_name == "<unparseable>"


def test_unwritable_log_does_not_block(tmp_path, hook_payload):
    blocker = tmp_path / "file"
    blocker.write_text("")
    result = run_log_mode(json.dumps(hook_payload), blocker / "sub" / "a.jsonl")
    assert result.exit_code == 0 and result.stdout == "" and "could not write" in result.stderr


def test_record_rejects_unknown_fields_and_bad_values():
    with pytest.raises(ValidationError):
        AuditRecord(tool_name="x", bogus=1)
    with pytest.raises(ValidationError):
        AuditRecord(tool_name="x", p_harm=1.5)
    with pytest.raises(ValidationError):
        AuditRecord(tool_name="x", decision="maybe")


def test_long_inputs_are_truncated(tmp_path, hook_payload):
    hook_payload["tool_name"] = "Write"
    hook_payload["tool_input"] = {"file_path": "/x", "content": "a" * 10_000}
    log = AuditLog(tmp_path / "a.jsonl")
    run_log_mode(json.dumps(hook_payload), log.path)
    rec = log.read()[0]
    assert rec.tool_input_truncated and len(rec.tool_input["content"]) < 4100


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(),
        lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(), c, max_size=4),
    )
)
@settings(max_examples=200)
def test_truncation_preserves_structure(value):
    out, _ = truncate_input(value, limit=10)
    assert type(out) is type(value)
    json.dumps(out)


def test_concurrent_appends_keep_lines_whole(tmp_path, hook_payload):
    log = AuditLog(tmp_path / "a.jsonl")
    hook_payload["tool_input"] = {"command": "x" * 3000}
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda _: run_log_mode(json.dumps(hook_payload), log.path), range(200)))
    assert len(log.read()) == 200


def test_published_json_schema_is_current():
    """docs/audit_log.schema.json is generated from the model; regenerate with `noulgate schema`."""
    assert json.loads(SCHEMA_FILE.read_text()) == AuditRecord.model_json_schema()


def test_hook_subprocess_contract(tmp_path, hook_payload):
    log = tmp_path / "a.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "noulgate.cli", "hook", "claude-code", "--log", str(log)],
        input=json.dumps(hook_payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0 and proc.stdout == ""
    AuditRecord.model_validate_json(log.read_text().strip())


def test_lean_record_fields_match_model():
    """The stdlib writer in noulgate/_jsonl.py must stay in sync with schema.AuditRecord."""
    assert set(RECORD_DEFAULTS) == set(AuditRecord.model_fields)
    AuditRecord.model_validate(new_record(tool_name="x"))
    with pytest.raises(KeyError):
        new_record(tool_name="x", bogus=1)


@pytest.mark.parametrize("event,outcome", [("PostToolUse", "ran"), ("PostToolUseFailure", "failed")])
def test_outcome_events_are_logged_and_joinable(tmp_path, hook_payload, event, outcome):
    log = AuditLog(tmp_path / "a.jsonl")
    run_log_mode(json.dumps(hook_payload), log.path)
    post = dict(
        hook_payload, hook_event_name=event, tool_response={"stdout": "x" * 9000}, error="Exit code 1"
    )
    res = run_log_mode(json.dumps(post), log.path)
    assert (res.exit_code, res.stdout) == (0, "")
    decision, out = log.read()
    assert decision.kind == "decision" and out.kind == "outcome" and out.outcome == outcome
    assert out.tool_use_id == decision.tool_use_id == "toolu_01ABC123"
    assert out.tool_input == {}  # not duplicated
    assert (out.error is not None) == (outcome == "failed")


def test_v1_records_still_read(tmp_path):
    old = {
        "schema_version": 1,
        "event_id": "e",
        "ts": "2026-10-11T05:04:02.515130Z",
        "source": "claude_code_hook",
        "mode": "log",
        "tool_name": "Bash",
        "tool_input": {},
        "decision": None,
        "emitted": "none",
    }
    (tmp_path / "a.jsonl").write_text(json.dumps(old) + "\n")
    [rec] = AuditLog(tmp_path / "a.jsonl").read()
    assert rec.schema_version == 1 and rec.kind == "decision"


def test_hook_does_not_import_pydantic():
    code = (
        "import sys, noulgate.cli, noulgate.integrations.claude_code_hook; print('pydantic' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
