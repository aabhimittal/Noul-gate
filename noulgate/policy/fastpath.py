"""Fast path: score obviously safe calls without running the decision model.

On a CPU, d1-3B takes ~1.25 s per call, while most tool calls in a coding session are reads inside the
project (Read, Grep, Glob, `git status`). The fast path gives those calls the score p = 0 and skips the
backend entirely.

Why this keeps the guarantee: conformal calibration is valid for *any* fixed score function, as long as
calibration and deployment use the same one. With the fast path on, the score function is

    s(x) = 0             if fast_path(x)
    s(x) = p_model(x)    otherwise

and `calibrate` scores the calibration set the same way. It corrects itself, too: if more than k of the
n_h harmful calibration actions slip through the fast path, λ_allow collapses to 0, and since allow
requires p < λ_allow, fast-path calls stop being auto-allowed. A leaky allowlist costs ask rate, not the
bound.

The allowlist is deliberately narrow. Anything not clearly read-only and inside the project goes to the
model.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import Any

from ..features.builder import GateState
from ..features.taint import SECRET_PATH

READ_TOOLS = {"Read", "Glob", "Grep", "LS", "NotebookRead"}
PATH_KEYS = ("file_path", "path", "notebook_path")
SHELL_META = set(";|&><`$()\n\\*?[]{}~!")
GIT_READ_SUBCOMMANDS = {"status", "diff", "log", "show", "branch"}
# git options that run programs or write files
GIT_UNSAFE_ARGS = (
    "--output",
    "--ext-diff",
    "--textconv",
    "--exec",
    "-c",
    "--upload-pack",
    "--receive-pack",
    "--no-index",  # diff --no-index reads arbitrary files outside the repository
)
FILE_READERS = {"cat", "head", "tail", "wc"}
PLAIN = {"ls", "pwd"}

FASTPATH_CLASS = "_fastpath"


def _inside(path: str, cwd: str) -> bool:
    if not path:
        return False
    p = PurePosixPath(path)
    if ".." in p.parts:
        return False
    if not p.is_absolute():
        return True
    try:
        p.relative_to(PurePosixPath(cwd))
    except ValueError:
        return False
    return True


def _safe_path(path: str, cwd: str) -> bool:
    return _inside(path, cwd) and not SECRET_PATH.search(path)


def _read_tool(state: GateState) -> str | None:
    cwd = state.cwd
    if not cwd:
        return None
    paths = [str(state.args[k]) for k in PATH_KEYS if state.args.get(k)]
    if not all(_safe_path(p, cwd) for p in paths):
        return None
    return f"{state.tool} inside the project"


def _bash(command: str, cwd: str) -> str | None:
    if not command or any(ch in SHELL_META for ch in command):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv:
        return None
    prog, args = argv[0], argv[1:]
    if prog in PLAIN:
        flags_only = all(a.startswith("-") or _safe_path(a, cwd) for a in args)
        return f"{prog}" if flags_only else None
    if prog in FILE_READERS:
        files = [a for a in args if not a.startswith("-") and not a.isdigit()]
        return f"{prog} of project files" if files and all(_safe_path(f, cwd) for f in files) else None
    if prog == "git" and args and args[0] in GIT_READ_SUBCOMMANDS:
        if any(a.startswith(GIT_UNSAFE_ARGS) for a in args[1:]):
            return None
        if args[0] == "branch" and any(
            a in ("-d", "-D", "-m", "-M", "--delete", "--move", "-f") for a in args
        ):
            return None
        return f"git {args[0]}"
    return None


# Claude Code's own bookkeeping tools: they change nothing outside the conversation (a task list, a tool
# search, a question or plan shown to the user). Monitor, Skill and anything MCP are deliberately absent.
META_TOOLS = frozenset(
    {
        "TaskCreate",
        "TaskUpdate",
        "TaskGet",
        "TaskList",
        "TodoWrite",
        "ToolSearch",
        "ExitPlanMode",
        "EnterPlanMode",
        "AskUserQuestion",
        "ReadNotifications",
    }
)


def fast_path(state: GateState) -> str | None:
    """A short reason if this call is trivially safe, else None."""
    if state.secret_in_args or state.tainted_egress:
        return None
    if state.tool in META_TOOLS:
        return f"{state.tool}: conversation-only tool"
    if state.tool in READ_TOOLS:
        return _read_tool(state)
    if state.tool == "Bash" and state.cwd:
        reason = _bash(str(state.args.get("command", "")), state.cwd)
        return f"Bash: {reason}" if reason else None
    return None


def fast_answers(reason: str) -> dict[str, Any]:
    """Placeholder answers recorded for a fast-path call (no model ran)."""
    return {"_fastpath": {"type": "fastpath", "reason": reason}}
