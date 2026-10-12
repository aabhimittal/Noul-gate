"""Render docs/demo/e2e_transcript.txt as docs/demo/e2e.html (a terminal-styled, self-contained page).

bash docs/demo/e2e.sh > docs/demo/e2e_transcript.txt 2>&1 && python docs/demo/render_e2e.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>NOUL-GATE End-to-End Run</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: one recorded terminal session, split into the run's numbered steps. */
:root { --bg: #eef1f4; --term: #ffffff; --ink: #16202b; --muted: #586474; --line: #d6dde5; --cmd: #2456c4;
  --allow: #1d7a4b; --deny: #b0271f; --warn: #a86300;
  --ui: "Instrument Sans", ui-sans-serif, system-ui, sans-serif; --mono: "JetBrains Mono", ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #0e1319; --term: #121921; --ink: #dfe6ee;
  --muted: #8f9db0; --line: #263342; --cmd: #85a8ff; --allow: #52c58e; --deny: #ff7166; --warn: #f2a93b; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #0e1319; --term: #121921; --ink: #dfe6ee; --muted: #8f9db0; --line: #263342;
  --cmd: #85a8ff; --allow: #52c58e; --deny: #ff7166; --warn: #f2a93b; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 var(--ui); }
.wrap { max-width: 1040px; margin: 0 auto; padding-inline: 16px; padding-block: 24px 48px; display: grid; gap: 14px; }
h1 { margin: 0; font-size: 1.5rem; text-wrap: balance; }
p { margin: 0; color: var(--muted); max-width: 75ch; }
a { color: var(--cmd); }
section { background: var(--term); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
section h2 { margin: 0; font: 600 0.8rem var(--ui); text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted);
  padding: 10px 14px; border-bottom: 1px solid var(--line); }
pre { margin: 0; padding: 12px 14px; overflow-x: auto; font: 0.8rem/1.55 var(--mono); color: var(--ink); }
.cmd { color: var(--cmd); }
.allow { color: var(--allow); font-weight: 600; }
.deny { color: var(--deny); font-weight: 600; }
.warn { color: var(--warn); font-weight: 600; }
</style></head><body><div class="wrap">
<h1>NOUL-GATE End-to-End Run</h1>
<p>A recorded run of <code>docs/demo/e2e.sh</code>: calibrate a policy, start the decision server with the HelixDB
store, install the Claude Code hook, run a real Claude Code session through it, replay attack payloads, then read
the audit log, monitor, query the graph and attack the gate. Commands are shown as they ran; output is unedited
apart from long lines. See also the <a href="index.html">interactive demo</a>.</p>
__SECTIONS__
</div></body></html>
"""


def colour(line: str) -> str:
    esc = html.escape(line)
    if line.startswith("$ "):
        return f'<span class="cmd">{esc}</span>'
    if line.startswith(("ALERT", "warning")):
        return f'<span class="warn">{esc}</span>'
    esc = re.sub(r"\b(deny|DENY)\b", r'<span class="deny">\1</span>', esc)
    esc = re.sub(r"(-&gt; |\b)(allow)\b(?![_-])", r'\1<span class="allow">\2</span>', esc)
    return esc


def main() -> None:
    text = (HERE / "e2e_transcript.txt").read_text()
    sections, title, body = [], "setup", []
    for line in text.splitlines():
        m = re.match(r"^== (.*) ==$", line)
        if m:
            if body and any(b.strip() for b in body):
                sections.append((title, body))
            title, body = m.group(1), []
        elif not line.startswith("Loading weights"):
            body.append(line if len(line) < 400 else line[:400] + " …")
    if body:
        sections.append((title, body))
    out = "\n".join(
        f"<section><h2>{html.escape(t)}</h2><pre>{chr(10).join(colour(b) for b in lines).strip()}</pre></section>"
        for t, lines in sections
    )
    (HERE / "e2e.html").write_text(PAGE.replace("__SECTIONS__", out))
    print(f"wrote {HERE / 'e2e.html'} ({len(sections)} sections)")


if __name__ == "__main__":
    main()
