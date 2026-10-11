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
HOOK_MODULE = "noulgate.integrations.claude_code_hook"


def hook_command(mode: str, extra: list[str] | None = None) -> str:
    # Absolute interpreter path: the hook runs from Claude Code's environment, not this venv.
    # The hook module directly, not `noulgate.cli`: skipping argparse saves ~13 ms per tool call.
    parts = [sys.executable, "-m", HOOK_MODULE, "--mode", mode, *(extra or [])]
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
    cmd = str(hook.get("command", ""))
    return MARKER in cmd and ("hook claude-code" in cmd or HOOK_MODULE in cmd)  # old and new command forms


OUTCOME_EVENTS = ("PostToolUse", "PostToolUseFailure")
EVENTS = ("PreToolUse", *OUTCOME_EVENTS)


def hook_entries(
    mode: str, extra: list[str] | None = None, outcomes: bool = True
) -> dict[str, dict[str, Any]]:
    """The hooks to install, per event. Outcome hooks only log (they can't block: the tool already ran),
    so they always run in log mode with a short timeout and fail open."""
    entries = {"PreToolUse": hook_entry(mode, extra)}
    if outcomes:
        log_extra = [a for a in (extra or []) if a != "--grant-allow"]
        log_extra = _drop_flag(log_extra, "--server")
        for event in OUTCOME_EVENTS:
            entries[event] = {"type": "command", "command": hook_command("log", log_extra), "timeout": 5}
    return entries


def _drop_flag(args: list[str], flag: str) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a == flag:
            skip = True
        else:
            out.append(a)
    return out


def _strip_ours(settings: dict[str, Any]) -> None:
    hooks = settings.get("hooks", {})
    for event in list(hooks):
        groups = hooks[event]
        if not isinstance(groups, list):
            continue
        for group in groups:
            group["hooks"] = [h for h in group.get("hooks", []) if not _is_ours(h)]
        groups[:] = [g for g in groups if g.get("hooks")]
        if not groups and event in EVENTS:
            del hooks[event]


def merge(settings: dict[str, Any], entries: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with exactly one noulgate hook per event in `entries` (replacing any previous ones).

    `entries` maps event name -> hook entry; a bare entry (with a "type" key) means PreToolUse only.
    """
    if "type" in entries:
        entries = {"PreToolUse": entries}
    out = copy.deepcopy(settings)
    _strip_ours(out)
    hooks = out.setdefault("hooks", {})
    for event, entry in entries.items():
        hooks.setdefault(event, []).append({"matcher": "*", "hooks": [entry]})
    return out


def uninstall(settings: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(settings)
    _strip_ours(out)
    return out


def render(settings: dict[str, Any]) -> str:
    return json.dumps(settings, indent=2) + "\n"


def diff(old: str, new: str, path: Path) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))


def install(path: Path, entry: dict[str, Any] | None, *, yes: bool, out=sys.stdout, inp=sys.stdin) -> bool:
    """Show the diff, ask, write. `entry` is a hook entry or an {event: entry} map; None uninstalls.
    Returns True if the file was written."""
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
