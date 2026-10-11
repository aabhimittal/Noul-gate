"""noulgate command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_hook(args: argparse.Namespace) -> int:
    from .integrations import claude_code_hook

    return claude_code_hook.main(mode=args.mode, log_path=args.log)


def _cmd_install(args: argparse.Namespace) -> int:
    from .integrations import install

    path = Path(args.settings) if args.settings else install.settings_path(args.scope)
    entry = None if args.uninstall else install.hook_entry(args.mode, _hook_extra(args))
    install.install(path, entry, yes=args.yes)
    return 0


def _hook_extra(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.log:
        extra += ["--log", str(Path(args.log).expanduser().resolve())]
    return extra


def _cmd_schema(args: argparse.Namespace) -> int:
    import json

    from .schema import AuditRecord

    text = json.dumps(AuditRecord.model_json_schema(), indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="noulgate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hook", help="run as an agent hook (reads the event on stdin)")
    h.add_argument("harness", choices=["claude-code"])
    h.add_argument("--mode", choices=["log", "enforce"], default="log")
    h.add_argument("--log", help="audit log path (default $NOULGATE_LOG or ~/.noulgate/audit.jsonl)")
    h.set_defaults(func=_cmd_hook)

    i = sub.add_parser("install", help="install the hook into a harness config (shows a diff first)")
    i.add_argument("harness", choices=["claude-code"])
    i.add_argument("--scope", choices=["project", "local", "user"], default="project")
    i.add_argument("--settings", help="explicit settings.json path (overrides --scope)")
    i.add_argument("--mode", choices=["log", "enforce"], default="log")
    i.add_argument("--log", help="audit log path baked into the hook command")
    i.add_argument("--uninstall", action="store_true")
    i.add_argument("-y", "--yes", action="store_true", help="write without asking")
    i.set_defaults(func=_cmd_install)

    sc = sub.add_parser("schema", help="print the audit-log JSON Schema")
    sc.add_argument("--out")
    sc.set_defaults(func=_cmd_schema)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
