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

from noulgate.audit import AuditLog, truncate_input
from noulgate.integrations.claude_code_hook import run_log_mode
from noulgate.schema import AuditRecord

SCHEMA_FILE = Path(__file__).parent.parent / "docs" / "audit_log.schema.json"


def test_log_mode_records_call_and_emits_nothing(tmp_path, hook_payload):
    log = AuditLog(tmp_path / "a.jsonl")
    result = run_log_mode(json.dumps(hook_payload), log)
    assert (result.exit_code, result.stdout) == (0, "")
    [rec] = log.read()
    assert rec.tool_name == "Bash" and rec.tool_input["command"] == "npm test"
    assert rec.session_id == "abc123" and rec.tool_use_id == "toolu_01ABC123"
    assert rec.mode == "log" and rec.decision is None and rec.emitted == "none"


def test_unknown_payload_fields_are_tolerated(tmp_path, hook_payload):
    hook_payload["effort"] = {"level": "high"}
    hook_payload["some_future_field"] = 1
    log = AuditLog(tmp_path / "a.jsonl")
    assert run_log_mode(json.dumps(hook_payload), log).exit_code == 0
    assert log.read()[0].error is None


@pytest.mark.parametrize("stdin", ["", "not json", "[]", '{"tool_name": "Bash"}'])
def test_malformed_input_still_exits_zero_and_logs_error(tmp_path, stdin):
    log = AuditLog(tmp_path / "a.jsonl")
    result = run_log_mode(stdin, log)
    assert (result.exit_code, result.stdout) == (0, "")
    [rec] = log.read()
    assert rec.error and rec.tool_name == "<unparseable>"


def test_unwritable_log_does_not_block(tmp_path, hook_payload):
    blocker = tmp_path / "file"
    blocker.write_text("")
    result = run_log_mode(json.dumps(hook_payload), AuditLog(blocker / "sub" / "a.jsonl"))
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
    run_log_mode(json.dumps(hook_payload), log)
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
        list(ex.map(lambda _: run_log_mode(json.dumps(hook_payload), log), range(200)))
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
