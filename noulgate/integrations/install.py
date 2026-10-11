"""`noulgate install claude-code`: merge the hook into a Claude Code settings.json, diff first."""

from __future__ import annotations

import copy
import difflib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

MARKER = "noulgate"


def hook_command(mode: str, extra: list[str] | None = None) -> str:
    # Absolute interpreter path: the hook runs from Claude Code's environment, not this venv.
    parts = [sys.executable, "-m", "noulgate.cli", "hook", "claude-code", "--mode", mode, *(extra or [])]
    return shlex.join(parts)


def hook_entry(mode: str, extra: list[str] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": "command", "command": hook_command(mode, extra)}
    if mode == "enforce":
        # A timed-out hook fails *open* by default; in enforce mode we want hook failure to block.
        entry["timeout"] = 15
        entry["onFailure"] = "block"
    else:
        entry["timeout"] = 5
    return entry


def settings_path(scope: str, project_dir: Path | None = None) -> Path:
    if scope == "user":
        return Path.home() / ".claude" / "settings.json"
    base = project_dir or Path.cwd()
    return base / ".claude" / ("settings.local.json" if scope == "local" else "settings.json")


def _is_ours(hook: dict[str, Any]) -> bool:
    return MARKER in str(hook.get("command", "")) and "hook claude-code" in str(hook.get("command", ""))


def merge(settings: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with exactly one noulgate PreToolUse hook (replacing any previous one)."""
    out = copy.deepcopy(settings)
    groups = out.setdefault("hooks", {}).setdefault("PreToolUse", [])
    for group in groups:
        group["hooks"] = [h for h in group.get("hooks", []) if not _is_ours(h)]
    groups[:] = [g for g in groups if g.get("hooks")]
    groups.append({"matcher": "*", "hooks": [entry]})
    return out


def uninstall(settings: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(settings)
    groups = out.get("hooks", {}).get("PreToolUse", [])
    for group in groups:
        group["hooks"] = [h for h in group.get("hooks", []) if not _is_ours(h)]
    groups[:] = [g for g in groups if g.get("hooks")]
    return out


def render(settings: dict[str, Any]) -> str:
    return json.dumps(settings, indent=2) + "\n"


def diff(old: str, new: str, path: Path) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))


def install(path: Path, entry: dict[str, Any] | None, *, yes: bool, out=sys.stdout, inp=sys.stdin) -> bool:
    """Show the diff, ask, write. `entry=None` uninstalls. Returns True if the file was written."""
    old_text = path.read_text() if path.exists() else ""
    current = json.loads(old_text) if old_text.strip() else {}
    new = merge(current, entry) if entry else uninstall(current)
    new_text = render(new)
    if old_text and json.loads(old_text) == new:
        out.write(f"{path}: already up to date\n")
        return False
    out.write(diff(old_text, new_text, path) or "(no textual change)\n")
    if not yes:
        out.write(f"Write {path}? [y/N] ")
        out.flush()
        if inp.readline().strip().lower() not in {"y", "yes"}:
            out.write("aborted, nothing written\n")
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text)
    out.write(f"wrote {path}\n")
    return True
