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
