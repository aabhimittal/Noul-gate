import pytest


@pytest.fixture
def hook_payload():
    return {
        "session_id": "abc123",
        "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
        "transcript_path": "/home/user/.claude/projects/x/transcript.jsonl",
        "cwd": "/home/user/my-project",
        "permission_mode": "default",
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "npm test", "description": "Run test suite", "timeout": 120000},
        "tool_use_id": "toolu_01ABC123",
    }
