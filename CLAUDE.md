# CLAUDE.md: working on NOUL-GATE

NOUL-GATE is a calibrated decision gate for agent tool calls. Read `SPEC.md` for the intent,
`docs/spec-diff.md` for where reality differs, and `docs/architecture.md` for how it fits together.

## Persistence: use HelixDB

**HelixDB is this project's database.** Anything that needs to persist or query decisions, sessions,
provenance, taint or labels goes through `noulgate/store/helix.py` (`HelixStore`). Don't add SQLite,
Postgres, Redis or ad-hoc JSON files for that.

- Install: the `[helix]` extra (`helix-db==0.3.4`, `helix-db-embedded==0.3.3`). The SessionStart hook
  installs it in cloud sessions.
- Connect: `HelixStore("memory")` for tests, `"embedded:<dir>"` for local persistent use (in-process;
  the directory is created for you), `"http://localhost:6969"` for a HelixDB server (`helix start dev`
  or `docker run -p 6969:8080 ghcr.io/helixdb/helixdb:v0.0.12`). CLI default: `~/.noulgate/helix`.
- Query API: Python SDK operation trees. `write_batch()/read_batch().var_as(name, g()...)
  .returning([...]).to_query_request()`, then `client.query(...)`. Use `NodeId(id)` to address existing
  nodes and `SourcePredicate.eq(key, value)` to filter; `out(label)`, `in_(label)` to traverse.
  Docs: https://docs.helix-db.com/llms.txt (re-check them; the API moved from HelixQL to operation trees).
- Graph model: see the module docstring in `store/helix.py`. To extend it, add a label to
  `UNIQUE_KEYS` (it gets an index and get-or-create semantics), write through `_node`/`_edge`, keep
  ingestion idempotent on a unique key, and add a test in `tests/test_store_helix.py` (embedded,
  in-memory; no server needed).
- One writer per embedded directory (for example `noulgate serve --store …`); readers can open their own.
- The JSONL audit log stays the source of truth written by the hook. The graph is built from it (`noulgate
  store sync`) or written live by `serve --store`.

## Invariants: don't break these

- **The Claude Code hook (`integrations/claude_code_hook.py`, `_jsonl.py`) imports only the standard
  library.** It runs once per tool call. pydantic alone cost about 100 ms; the hook takes 36 ms now.
  `tests/test_m0_log_schema.py::test_hook_does_not_import_pydantic` guards this.
- **Log mode prints nothing.** Emitting `"allow"` would skip the user's permission prompt.
- **Calibration and deployment use the same score function** (`evaluate.score_vector`): same fast-path
  setting, same combiner, same `harden` flag. The policy JSON records them; respect them.
- **Fit the combiner on data disjoint from the conformal calibration split**, or the bound is void.
- **Changing the feature vector** (`policy/combiner.py`) means bumping `FEATURE_VERSION`.
- **Cached backend answers** are keyed by backend name plus a hash of the state. If you change what a
  backend sees, change its name (as `+hardened` does).
- **Don't overclaim.** The guarantee is marginal, assumes exchangeability, and fails against adaptive
  attackers. Synthetic results are labelled SYNTHETIC. Keep the README's honest-limits section current.

## Before pushing

```bash
ruff check . && ruff format --check . && pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same on Python 3.11 and 3.13 with `[serve,mcp,report,dev,helix]`.
The d1-3B model is never needed for tests. To run it, `pip install -e ".[d1]"`; it defaults to bfloat16 on
AVX512-BF16/AMX CPUs (1.25 s/decision) and float32 otherwise (about 29 s).

## Handy commands

```bash
noulgate benchmark --data data/fixtures/synthetic_all.jsonl --backends rules --splits 20
noulgate attack --data data/fixtures/synthetic_all.jsonl --backends rules
noulgate store sync --log ~/.noulgate/audit.jsonl && noulgate store query tainted-egress
python docs/demo/build_demo.py      # regenerate the interactive demo
```
