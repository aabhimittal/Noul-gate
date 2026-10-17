import io
import json

from noulgate.integrations import install


def test_install_shows_diff_and_respects_no(tmp_path):
    path = tmp_path / ".claude" / "settings.json"
    out = io.StringIO()
    wrote = install.install(path, install.hook_entry("log"), yes=False, out=out, inp=io.StringIO("n\n"))
    assert not wrote and not path.exists()
    assert "+++ b/" in out.getvalue() and "PreToolUse" in out.getvalue()


def test_install_preserves_existing_settings_and_is_idempotent(tmp_path):
    path = tmp_path / "settings.json"
    existing = {
        "permissions": {"deny": ["Bash(rm *)"]},
        "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "other"}]}]},
    }
    path.write_text(json.dumps(existing))
    assert install.install(path, install.hook_entry("log"), yes=True, out=io.StringIO())
    data = json.loads(path.read_text())
    assert data["permissions"] == existing["permissions"]
    assert len(data["hooks"]["PreToolUse"]) == 2
    out = io.StringIO()
    assert not install.install(path, install.hook_entry("log"), yes=True, out=out)
    assert "already up to date" in out.getvalue()
    # switching mode replaces our entry rather than adding a second one
    install.install(path, install.hook_entry("enforce"), yes=True, out=io.StringIO())
    groups = json.loads(path.read_text())["hooks"]["PreToolUse"]
    ours = [h for g in groups for h in g["hooks"] if "noulgate" in h["command"]]
    assert len(ours) == 1 and ours[0]["onFailure"] == "block"


def test_uninstall_removes_only_ours(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "other"}]}]
                }
            }
        )
    )
    install.install(path, install.hook_entry("log"), yes=True, out=io.StringIO())
    install.install(path, None, yes=True, out=io.StringIO())
    assert json.loads(path.read_text())["hooks"]["PreToolUse"] == [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": "other"}]}
    ]


def test_installed_command_runs_and_parses_flags(tmp_path):
    import shlex
    import subprocess

    log = tmp_path / "a.jsonl"
    cmd = install.hook_command("log", ["--log", str(log)])
    assert "noulgate.integrations.claude_code_hook" in cmd
    payload = '{"session_id": "s", "tool_name": "Bash", "tool_input": {"command": "ls"}}'
    proc = subprocess.run(shlex.split(cmd), input=payload, capture_output=True, text=True, timeout=30)
    assert (proc.returncode, proc.stdout) == (0, "") and log.exists()
    bad = subprocess.run(
        [*shlex.split(cmd), "--nope"], input=payload, capture_output=True, text=True, timeout=30
    )
    assert bad.returncode != 0


def test_reinstall_replaces_old_cli_form(tmp_path):
    path = tmp_path / "settings.json"
    old = {"type": "command", "command": "/usr/bin/python3 -m noulgate.cli hook claude-code --mode log"}
    path.write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [old]}]}}))
    install.install(path, install.hook_entry("log"), yes=True, out=io.StringIO())
    cmds = [h["command"] for g in json.loads(path.read_text())["hooks"]["PreToolUse"] for h in g["hooks"]]
    assert len(cmds) == 1 and "claude_code_hook" in cmds[0]


def test_outcome_hooks_installed_and_removed(tmp_path):
    path = tmp_path / "settings.json"
    entries = install.hook_entries(
        "enforce", ["--log", "/x/a.jsonl", "--server", "http://h:1", "--grant-allow"]
    )
    install.install(path, entries, yes=True, out=io.StringIO())
    hooks = json.loads(path.read_text())["hooks"]
    assert set(hooks) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    post = hooks["PostToolUse"][0]["hooks"][0]
    assert "--mode log" in post["command"] and "--log /x/a.jsonl" in post["command"]
    assert "--server" not in post["command"] and "--grant-allow" not in post["command"]
    assert "onFailure" not in post  # outcome logging must never block
    assert hooks["PreToolUse"][0]["hooks"][0]["onFailure"] == "block"
    install.install(path, None, yes=True, out=io.StringIO())
    assert json.loads(path.read_text())["hooks"] == {}


def test_install_approve_wait_outlasts_the_server(tmp_path):
    from noulgate.cli import main

    path = tmp_path / "settings.json"
    main(
        ["install", "claude-code", "--settings", str(path), "--mode", "enforce", "--approve-wait", "60", "-y"]
    )
    hooks = json.loads(path.read_text())["hooks"]
    pre = hooks["PreToolUse"][0]["hooks"][0]
    assert pre["timeout"] == 70 and "--timeout 65" in pre["command"] and pre["onFailure"] == "block"
    assert "--timeout" not in hooks["PostToolUse"][0]["hooks"][0]["command"]
