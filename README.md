# NOUL-GATE

A typed, calibrated decision gate for agent tool calls. See [SPEC.md](SPEC.md) and
[docs/spec-diff.md](docs/spec-diff.md).

## M0: log every pending Claude Code tool call

```bash
pip install -e .
noulgate install claude-code            # shows a diff of .claude/settings.json, asks before writing
# ...use Claude Code as usual...
cat ~/.noulgate/audit.jsonl
```

The hook only observes. It prints nothing, so Claude Code's permission flow is unchanged.
The record schema is in [docs/audit_log.schema.json](docs/audit_log.schema.json).
