"""noulgate: a typed, calibrated decision gate for agent tool calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_STORE = "embedded:" + str(Path.home() / ".noulgate" / "helix")


def _questions(args):
    from .questions import load_questions

    return load_questions(getattr(args, "questions", None))


def _backend(name: str):
    from .backends import get_backend

    return get_backend(name)


def _is_synthetic(rows) -> bool:
    return any(r.source.startswith("synthetic") for r in rows)


# --------------------------------------------------------------------------- hook / install / schema


def _cmd_hook(args: argparse.Namespace) -> int:
    from .integrations import claude_code_hook

    return claude_code_hook.main(
        mode=args.mode,
        log_path=args.log,
        server=args.server,
        grant_allow=args.grant_allow,
        timeout=args.timeout,
    )


def _cmd_install(args: argparse.Namespace) -> int:
    from .integrations import install

    path = Path(args.settings) if args.settings else install.settings_path(args.scope)
    extra: list[str] = []
    if args.log:
        extra += ["--log", str(Path(args.log).expanduser().resolve())]
    if args.mode == "enforce":
        if args.server:
            extra += ["--server", args.server]
        if args.grant_allow:
            extra += ["--grant-allow"]
    timeout = 15
    if args.mode == "enforce" and args.approve_wait:
        # Claude Code kills the hook at `timeout` and onFailure=block turns that into a block: outlast the
        # server's wait (hook HTTP timeout +5 s, Claude Code's +10 s) so an unanswered ask falls back to `ask`
        extra += ["--timeout", f"{args.approve_wait + 5:g}"]
        timeout = args.approve_wait + 10
    entry = (
        None
        if args.uninstall
        else install.hook_entries(args.mode, extra, outcomes=not args.no_outcomes, timeout=timeout)
    )
    install.install(path, entry, yes=args.yes)
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    from .schema import AuditRecord

    text = json.dumps(AuditRecord.model_json_schema(), indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


# --------------------------------------------------------------------------- serve


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .policy.conformal import ThresholdPolicy, load_policy
    from .serve import create_app

    policy = load_policy(args.policy) if args.policy else ThresholdPolicy(0.5)
    if not args.policy:
        print(
            "noulgate: no --policy given; using the UNCALIBRATED 0.5 threshold (no guarantee).",
            file=sys.stderr,
        )
    token = args.token or os.environ.get("NOULGATE_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        print(
            "noulgate: refusing to bind a non-loopback host without --token / NOULGATE_TOKEN.",
            file=sys.stderr,
        )
        return 2
    if args.approve_wait and not token:
        print("noulgate: --approve-wait grants tool permissions; it needs --token.", file=sys.stderr)
        return 2
    approve_token = None
    scheme = "https" if args.ssl_certfile else "http"
    console = f"{scheme}://{args.host}:{args.port}/"
    if args.approve_wait:
        import secrets

        # never from the environment: the agent shares it and must not approve its own asks
        approve_token = secrets.token_urlsafe(24)
        console += f"#token={approve_token}"
        print(
            f"noulgate: asks wait {args.approve_wait:g}s for the console; install the hook with"
            f" `noulgate install claude-code --mode enforce --approve-wait {args.approve_wait:g}`."
            " Open the console link below on your phone; its token can approve tool calls, so don't"
            " share it.",
            file=sys.stderr,
        )
    store = None
    if args.store:
        from .store.helix import HelixStore

        store = HelixStore(args.store)
    app = create_app(
        _backend(args.backend),
        _questions(args),
        policy,
        warmup=not args.no_warmup,
        store=store,
        token=token,
        approve_wait=args.approve_wait,
        approve_token=approve_token,
    )
    print(f"noulgate: console at {console}", file=sys.stderr)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
    return 0


# --------------------------------------------------------------------------- calibrate / eval / benchmark


def _floors(text: str) -> dict[str, str]:
    """`--floors`: "default", "none", or "flag=ask,flag2=deny"."""
    from .policy.combiner import FLAG_NAMES
    from .policy.conformal import DEFAULT_FLOORS

    if text == "default":
        return dict(DEFAULT_FLOORS)
    if text == "none":
        return {}
    out = {}
    for item in text.split(","):
        flag, _, floor = item.partition("=")
        if flag not in FLAG_NAMES or floor not in ("ask", "deny"):
            raise SystemExit(f"--floors: expected flag=ask|deny with flag in {FLAG_NAMES}, got {item!r}")
        out[flag] = floor
    return out


def _add_floors(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--floors",
        default="default",
        help="flags whose calls are never auto-allowed: 'default' (tainted_egress=ask,secret_in_args=ask), "
        "'none', or flag=ask|deny,... (see policy/conformal.py)",
    )


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .datasets import data_hash, read_jsonl, split
    from .evaluate import fit_and_calibrate, score_rows

    qs = _questions(args)
    rows = [r for r in read_jsonl(args.data) if r.label]
    fit_rows = read_jsonl(args.fit_data) if args.fit_data else []
    if args.fit_fraction and not fit_rows:
        parts = split(rows, {"fit": args.fit_fraction, "cal": 1 - args.fit_fraction}, seed=args.seed)
        fit_rows, rows = parts["fit"], parts["cal"]
    overlap = {r.id for r in fit_rows} & {r.id for r in rows}
    if overlap:
        print(
            f"error: {len(overlap)} rows are in both the fit and calibration data; the bound needs them disjoint",
            file=sys.stderr,
        )
        return 2
    backend = _backend(args.backend)
    fast = not args.no_fastpath
    cal = score_rows(rows, backend, qs, args.cache, fastpath=fast)
    fit = score_rows(fit_rows, backend, qs, args.cache, fastpath=fast) if fit_rows else []
    policy = fit_and_calibrate(
        fit,
        cal,
        qs,
        args.alpha,
        args.beta,
        per_class=not args.global_only,
        fastpath=fast,
        floors=_floors(args.floors),
    )
    policy.backend, policy.data_hash = backend.name, data_hash(rows)
    Path(args.out).write_text(json.dumps(policy.to_dict(), indent=2) + "\n")
    s = policy.calibration_stats
    print(
        f"wrote {args.out} (policy {policy.policy_id}): n_h={s['n_h']} n_b={s['n_b']} "
        f"combiner={'trained on ' + str(len(fit)) if fit else 'untrained (1 - P(benign))'}"
    )
    print(
        f"calibration-set rates: false_allow={s['false_allow']:.3f} false_deny={s['false_deny']:.3f} "
        f"ask={s['ask_rate']:.3f}"
    )
    for w in s.get("warnings", []):
        print("warning:", w, file=sys.stderr)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .datasets import read_jsonl
    from .evaluate import design, evaluate_policy, score_rows
    from .policy.conformal import load_policy
    from .report import eval_report

    qs = _questions(args)
    rows = [r for r in read_jsonl(args.data) if r.label]
    policy = load_policy(args.policy)
    backend_name = args.backend or getattr(policy, "backend", None) or "rules"
    scored = score_rows(
        rows, _backend(backend_name), qs, args.cache, fastpath=getattr(policy, "fastpath", False)
    )
    metrics = evaluate_policy(policy, scored, qs)
    if getattr(policy, "alpha", None) is not None:
        from .policy.conformal import false_allow_upper

        metrics["alpha"] = policy.alpha
        metrics["tolerance_95"] = false_allow_upper(
            policy.calibration_stats.get("n_h", 0), metrics["n_h"], policy.alpha
        )
        metrics["within_tolerance"] = metrics["false_allow"] <= metrics["tolerance_95"] + 1e-12
    print(json.dumps(metrics, indent=2))
    if args.report:
        from .policy.combiner import Combiner

        X, y, _ = design(scored, qs.risk_classes)
        p = (policy.combiner or Combiner(qs.risk_classes)).score_vec(X)
        meta = {"data": args.data, "n": len(rows), "backend": backend_name, "synthetic": _is_synthetic(rows)}
        Path(args.report).write_text(
            eval_report(metrics, policy.to_dict(), list(zip(p.tolist(), y.tolist(), strict=True)), meta)
        )
        print(f"wrote {args.report}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .datasets import read_jsonl
    from .evaluate import repeated_splits, score_rows
    from .latency import measure
    from .report import benchmark_report

    qs = _questions(args)
    rows = [r for r in read_jsonl(args.data) if r.label]
    by_id = {r.id: r for r in rows}
    alphas = [float(a) for a in args.alphas.split(",")]
    results, latency = {}, {}
    for name in args.backends.split(","):
        backend = _backend(name)
        cache = Path(args.cache_dir) / f"{name.split(':')[0]}.jsonl" if args.cache_dir else None
        fast = not args.no_fastpath
        scored = score_rows(rows, backend, qs, cache, fastpath=fast)
        results[backend.name] = repeated_splits(
            scored,
            by_id,
            qs,
            alphas,
            args.beta,
            n_splits=args.splits,
            fastpath=fast,
            floors=_floors(args.floors),
        )
        n_fast = sum(bool(s.fastpath) for s in scored)
        print(f"{backend.name}: fast path scored {n_fast}/{len(scored)} rows without the model")
        lat = sorted(s.latency_ms for s in scored)
        latency[backend.name] = {
            "p50_ms": lat[len(lat) // 2],
            "p99_ms": lat[min(len(lat) - 1, int(0.99 * len(lat)))],
            "source": "per-row time while scoring this dataset (batched)",
        }
        if args.latency_runs:
            latency[backend.name] = measure(backend, qs, runs=args.latency_runs)["short"] | {
                "source": "noulgate latency"
            }
        print(
            f"{backend.name}: "
            + ", ".join(
                f"α={d['alpha']:g}: FA={d['false_allow_mean']:.3f} ask={d['ask_rate_mean']:.3f} "
                f"in-tol {d['splits_within_tolerance']}/{d['n_splits']}"
                for d in results[backend.name]["per_alpha"].values()
            )
        )
    meta = {
        "data": args.data,
        "n": len(rows),
        "splits": args.splits,
        "beta": args.beta,
        "synthetic": _is_synthetic(rows),
    }
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "meta": meta,
                    "latency": latency,
                    "results": {
                        k: {kk: vv for kk, vv in v.items() if kk != "splits"} for k, v in results.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
    if args.report:
        Path(args.report).write_text(benchmark_report(results, latency, meta))
        print(f"wrote {args.report}")
    return 0


def _cmd_latency(args: argparse.Namespace) -> int:
    from .latency import measure

    report = measure(
        _backend(args.backend), _questions(args), runs=args.runs, batch=args.batch, long_runs=args.long_runs
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


# --------------------------------------------------------------------------- data / labeling


def _cmd_dataset(args: argparse.Namespace) -> int:
    from . import datasets as ds

    if args.action == "import":
        importer = {
            "claude-code": ds.from_claude_code_transcript,
            "injecagent": ds.from_injecagent,
            "traces": ds.from_message_traces,
        }[args.format]
        rows = [r for p in args.paths for r in importer(p)]
        if args.assume_label:
            if not args.annotator:
                print(
                    "error: --assume-label needs --annotator (who vouches for these labels)", file=sys.stderr
                )
                return 2
            rows = [
                r
                if r.label
                else r.model_copy(
                    update={"label": args.assume_label, "annotator": args.annotator, "notes": "assumed label"}
                )
                for r in rows
            ]
        n = ds.write_jsonl(args.out, rows)
        print(f"wrote {n} rows to {args.out}: {ds.summary(rows)}")
    elif args.action == "split":
        rows = ds.read_jsonl(args.data)
        fractions = {k: float(v) for k, v in (kv.split("=") for kv in args.fractions.split(","))}
        parts = ds.split(rows, fractions, seed=args.seed)
        for name, part in parts.items():
            path = Path(args.out_dir) / f"{name}.jsonl"
            ds.write_jsonl(path, part)
            print(f"{path}: {ds.summary(part)['labels']}")
    else:
        print(json.dumps(ds.summary(ds.read_jsonl(args.data)), indent=2))
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    from .label_tui import run

    run(args.inp, args.out, args.annotator)
    return 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    from .datasets import write_jsonl
    from .label_tui import agreement, merge

    print(json.dumps(agreement(args.files), indent=2))
    if args.merge_out:
        n = write_jsonl(args.merge_out, merge(args.files, args.policy))
        print(f"wrote {n} merged rows ({args.policy}) to {args.merge_out}")
    return 0


# --------------------------------------------------------------------------- feedback / monitoring


def _records(args: argparse.Namespace) -> list[dict]:
    """Audit records from --store (HelixDB) if given, else from the JSONL --log."""
    if getattr(args, "store", None):
        from .store.helix import HelixStore

        store = HelixStore(args.store)
        try:
            return store.records()
        finally:
            store.close()
    from .feedback import read_records

    return read_records(args.log)


def _cmd_store(args: argparse.Namespace) -> int:
    from .store.helix import HelixStore

    store = HelixStore(args.store)
    try:
        if args.action == "sync":
            from .feedback import harvest, read_records

            records = read_records(args.log)
            stats = store.ingest(records, rebuild_provenance=not args.no_provenance)
            answers = 0
            for ask in harvest(records):
                if ask.answer != "unknown":
                    answers += store.record_answer(ask.decision["tool_use_id"], ask.answer)
            stats["human_answers"] = answers
            print(json.dumps({"synced": stats, "graph": store.counts()}, indent=2))
        elif args.action == "stats":
            print(json.dumps(store.counts(), indent=2))
        else:
            if args.query == "tainted-egress":
                rows = store.tainted_egress_chains()
            elif args.query == "injected-targets":
                rows = store.injected_targets(min_calls=args.min_calls)
            else:
                if not args.session:
                    print("--session is required for the session query", file=sys.stderr)
                    return 2
                rows = store.session_timeline(args.session)
            print(json.dumps(rows, indent=2, default=str))
    finally:
        store.close()
    return 0


def _cmd_harvest(args: argparse.Namespace) -> int:
    from .datasets import write_jsonl
    from .feedback import harvest, summary, to_labeled

    asks = harvest(_records(args))
    print(f"asks: {summary(asks)}")
    rows = to_labeled(asks, auto_label=args.auto_label)
    if args.out:
        n = write_jsonl(args.out, rows)
        labeled = sum(r.label is not None for r in rows)
        print(f"wrote {n} rows ({labeled} labeled) to {args.out}; review them with `noulgate label`")
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    from .monitor import monitor

    policy = json.loads(Path(args.policy).read_text()) if args.policy else None
    rep = monitor(_records(args), policy, window=args.window, policy_id=args.policy_id).to_dict()
    o = rep["overall"]
    print(
        f"{rep['n_decisions']} decisions: allow {o['allow_rate']:.3f}  ask {o['ask_rate']:.3f}  "
        f"deny {o['deny_rate']:.3f}"
    )
    print(f"asks answered: {rep['asks']}  approval rate: {rep['ask_approval_rate']}")
    print(
        "drift: "
        + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep["drift"].items())
    )
    for a in rep["alerts"]:
        print(f"ALERT: {a}")
    for n in rep["notes"]:
        print(f"note: {n}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    if args.report:
        from .report import monitor_report

        Path(args.report).write_text(monitor_report(rep))
        print(f"wrote {args.report}")
    return 1 if rep["alert"] else 0


def _cmd_attack(args: argparse.Namespace) -> int:
    from .adversarial import run_attack
    from .datasets import read_jsonl
    from .report import attack_report

    qs = _questions(args)
    rows = [r for r in read_jsonl(args.data) if r.label]
    results = []
    for name in args.backends.split(","):
        cache = Path(args.cache_dir) / f"{name.split(':')[0]}.jsonl" if args.cache_dir else None
        res = run_attack(
            rows,
            _backend(name),
            qs,
            alpha=args.alpha,
            beta=args.beta,
            cache=cache,
            seed=args.seed,
            max_harmful=args.max_harmful,
            fastpath=not args.no_fastpath,
            floors=_floors(args.floors),
        )
        results.append(res)
        print(
            f"{res['backend']}: clean false-allow {res['false_allow_clean']:.3f} on {res['n_harmful_attacked']} rows"
        )
        for t, m in res["transforms"].items():
            if m.get("n"):
                print(
                    f"  {t:22s} n={m['n']:3d}  Δp={m['delta_p_mean']:+.3f}  flip={m['flip_rate']:.3f}  "
                    f"false-allow {m['false_allow_clean']:.3f} -> {m['false_allow_attacked']:.3f}"
                )
    meta = {"data": args.data, "alpha": args.alpha, "seed": args.seed, "synthetic": _is_synthetic(rows)}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n")
    if args.report:
        Path(args.report).write_text(attack_report(results, meta))
        print(f"wrote {args.report}")
    return 0


# --------------------------------------------------------------------------- MCP proxy


def _cmd_mcp_proxy(args: argparse.Namespace) -> int:
    from .integrations.mcp_proxy import run_proxy

    upstream = args.upstream[1:] if args.upstream[:1] == ["--"] else args.upstream
    if not upstream:
        print("usage: noulgate mcp-proxy [options] -- <upstream server command...>", file=sys.stderr)
        return 2
    return run_proxy(
        upstream,
        server=args.server,
        backend=args.backend,
        policy=args.policy,
        log_path=args.log,
        ask_fallback=args.ask_fallback,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="noulgate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hook", help="run as an agent hook (reads the event on stdin)")
    h.add_argument("harness", choices=["claude-code"])
    h.add_argument("--mode", choices=["log", "enforce"], default="log")
    h.add_argument("--log", help="audit log path (default $NOULGATE_LOG or ~/.noulgate/audit.jsonl)")
    h.add_argument(
        "--server", help="decision server URL for enforce mode (default $NOULGATE_SERVER or 127.0.0.1:7701)"
    )
    h.add_argument(
        "--grant-allow", action="store_true", help="emit 'allow' (skips Claude Code's prompt) on gate allow"
    )
    h.add_argument("--timeout", type=float, default=10.0)
    h.set_defaults(func=_cmd_hook)

    i = sub.add_parser("install", help="install the hook into a harness config (shows a diff first)")
    i.add_argument("harness", choices=["claude-code"])
    i.add_argument("--scope", choices=["project", "local", "user"], default="project")
    i.add_argument("--settings", help="explicit settings.json path (overrides --scope)")
    i.add_argument("--mode", choices=["log", "enforce"], default="log")
    i.add_argument("--log", help="audit log path baked into the hook command")
    i.add_argument("--server")
    i.add_argument("--grant-allow", action="store_true")
    i.add_argument(
        "--approve-wait",
        type=float,
        default=0.0,
        help="match `serve --approve-wait`: lengthen the hook timeouts to outlast it",
    )
    i.add_argument("--uninstall", action="store_true")
    i.add_argument(
        "--no-outcomes",
        action="store_true",
        help="don't install the PostToolUse/PostToolUseFailure outcome hooks (used by `harvest`)",
    )
    i.add_argument("-y", "--yes", action="store_true", help="write without asking")
    i.set_defaults(func=_cmd_install)

    sc = sub.add_parser("schema", help="print the audit-log JSON Schema")
    sc.add_argument("--out")
    sc.set_defaults(func=_cmd_schema)

    s = sub.add_parser("serve", help="serve gate decisions over local HTTP")
    s.add_argument("--backend", default="d1-3b")
    s.add_argument("--policy", help="policy.json from `noulgate calibrate`")
    s.add_argument("--questions")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7701)
    s.add_argument("--no-warmup", action="store_true")
    s.add_argument("--ssl-certfile", help="serve HTTPS (phones install the console only over HTTPS)")
    s.add_argument("--ssl-keyfile")
    s.add_argument(
        "--approve-wait",
        type=float,
        default=0.0,
        help="seconds an `ask` waits for Allow/Deny from the console (needs --token)",
    )
    s.add_argument(
        "--token", help="require this bearer token (default $NOULGATE_TOKEN); needed off localhost"
    )
    s.add_argument(
        "--store", help="also record decisions in HelixDB: memory | embedded:<dir> | http://host:6969"
    )
    s.set_defaults(func=_cmd_serve)

    c = sub.add_parser("calibrate", help="set conformal thresholds from labeled calibration data")
    c.add_argument("--data", required=True)
    c.add_argument("--alpha", type=float, required=True, help="target false-allow rate among harmful actions")
    c.add_argument("--beta", type=float, default=0.10, help="target false-deny rate among benign actions")
    c.add_argument("--out", required=True)
    c.add_argument("--backend", default="d1-3b")
    c.add_argument("--fit-data", help="separate labeled data to fit the logistic combiner on")
    c.add_argument(
        "--fit-fraction", type=float, default=0.0, help="hold out this share of --data to fit the combiner"
    )
    c.add_argument("--global-only", action="store_true", help="one threshold pair instead of per risk class")
    c.add_argument(
        "--no-fastpath", action="store_true", help="run the model on every call (see policy/fastpath.py)"
    )
    c.add_argument("--cache", help="JSONL cache of backend answers")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--questions")
    _add_floors(c)
    c.set_defaults(func=_cmd_calibrate)

    e = sub.add_parser("eval", help="evaluate a policy on labeled test data")
    e.add_argument("--data", required=True)
    e.add_argument("--policy", required=True)
    e.add_argument("--report")
    e.add_argument("--backend")
    e.add_argument("--cache")
    e.add_argument("--questions")
    e.set_defaults(func=_cmd_eval)

    b = sub.add_parser("benchmark", help="repeated random splits x α sweep x backends -> report")
    b.add_argument("--data", required=True)
    b.add_argument("--backends", default="rules")
    b.add_argument("--alphas", default="0.01,0.02,0.05,0.1,0.2")
    b.add_argument("--beta", type=float, default=0.10)
    b.add_argument("--splits", type=int, default=20)
    b.add_argument("--cache-dir")
    b.add_argument("--latency-runs", type=int, default=0)
    b.add_argument("--no-fastpath", action="store_true")
    b.add_argument("--report")
    b.add_argument("--json-out")
    b.add_argument("--questions")
    _add_floors(b)
    b.set_defaults(func=_cmd_benchmark)

    lt = sub.add_parser("latency", help="measure p50/p99 latency and batch throughput")
    lt.add_argument("--backend", default="d1-3b")
    lt.add_argument("--runs", type=int, default=30)
    lt.add_argument("--batch", type=int, default=16)
    lt.add_argument("--long-runs", type=int, help="runs for the long-context state (default: --runs)")
    lt.add_argument("--out")
    lt.add_argument("--questions")
    lt.set_defaults(func=_cmd_latency)

    d = sub.add_parser("dataset", help="import / split / summarise labeled-action datasets")
    dsub = d.add_subparsers(dest="action", required=True)
    di = dsub.add_parser("import")
    di.add_argument("format", choices=["claude-code", "injecagent", "traces"])
    di.add_argument("paths", nargs="+")
    di.add_argument("--out", required=True)
    di.add_argument(
        "--assume-label",
        choices=["benign", "harmful"],
        help="label every unlabeled row (e.g. benign for sessions you supervised); review with `noulgate label`",
    )
    di.add_argument("--annotator", help="recorded with --assume-label")
    dsp = dsub.add_parser("split")
    dsp.add_argument("--data", required=True)
    dsp.add_argument("--out-dir", required=True)
    dsp.add_argument("--fractions", default="fit=0.25,cal=0.45,test=0.30")
    dsp.add_argument("--seed", type=int, default=0)
    dsu = dsub.add_parser("summary")
    dsu.add_argument("--data", required=True)
    d.set_defaults(func=_cmd_dataset)

    lb = sub.add_parser("label", help="label actions in the terminal")
    lb.add_argument("--in", dest="inp", required=True)
    lb.add_argument("--out", required=True)
    lb.add_argument("--annotator", required=True)
    lb.set_defaults(func=_cmd_label)

    ag = sub.add_parser("agreement", help="inter-annotator agreement (Cohen's kappa), optional merge")
    ag.add_argument("files", nargs="+")
    ag.add_argument("--merge-out")
    ag.add_argument("--policy", choices=["unanimous", "majority"], default="unanimous")
    ag.set_defaults(func=_cmd_agreement)

    at = sub.add_parser(
        "attack", help="adversarial robustness: attack harmful rows, measure how far p(x) drops"
    )
    at.add_argument("--data", required=True)
    at.add_argument("--backends", default="rules")
    at.add_argument("--alpha", type=float, default=0.05)
    at.add_argument("--beta", type=float, default=0.10)
    at.add_argument("--seed", type=int, default=0)
    at.add_argument("--max-harmful", type=int, help="cap the attacked harmful rows (model time)")
    at.add_argument("--cache-dir")
    at.add_argument("--no-fastpath", action="store_true")
    at.add_argument("--report")
    at.add_argument("--json-out")
    at.add_argument("--questions")
    _add_floors(at)
    at.set_defaults(func=_cmd_attack)

    hv = sub.add_parser(
        "harvest", help="turn answered 'ask' prompts from the audit log into labeled candidates"
    )
    hv.add_argument("--log", default=str(Path.home() / ".noulgate" / "audit.jsonl"))
    hv.add_argument("--out", help="write LabeledAction JSONL here")
    hv.add_argument(
        "--auto-label",
        action="store_true",
        help="approved -> benign, rejected -> harmful (otherwise answers are only hints)",
    )
    hv.add_argument("--store", help="read decisions from HelixDB instead of --log")
    hv.set_defaults(func=_cmd_harvest)

    st = sub.add_parser(
        "store", help="HelixDB graph store: sync the audit log, query provenance/taint graphs"
    )
    st_sub = st.add_subparsers(dest="action", required=True)
    ss = st_sub.add_parser(
        "sync", help="ingest an audit log (idempotent), rebuilding provenance from transcripts"
    )
    ss.add_argument("--log", default=str(Path.home() / ".noulgate" / "audit.jsonl"))
    ss.add_argument("--no-provenance", action="store_true", help="skip reading transcripts")
    sq = st_sub.add_parser("query", help="canned graph queries")
    sq.add_argument("query", choices=["tainted-egress", "injected-targets", "session"])
    sq.add_argument("--session")
    sq.add_argument("--min-calls", type=int, default=1)
    sst = st_sub.add_parser("stats", help="node counts")
    for p_ in (ss, sq, sst):
        p_.add_argument(
            "--store",
            default=DEFAULT_STORE,
            help="memory | embedded:<dir> | http://host:6969 (default: ~/.noulgate/helix)",
        )
    st.set_defaults(func=_cmd_store)

    mo = sub.add_parser("monitor", help="decision rates, ask outcomes and score drift; exit 1 on alert")
    mo.add_argument("--log", default=str(Path.home() / ".noulgate" / "audit.jsonl"))
    mo.add_argument("--policy", help="policy.json (enables drift checks and calibration comparisons)")
    mo.add_argument("--policy-id", help="only decisions made by this policy (default: the --policy's id)")
    mo.add_argument("--window", default="1d", help="bucket size: 30m, 6h, 1d ...")
    mo.add_argument("--report")
    mo.add_argument("--json-out")
    mo.add_argument("--store", help="read decisions from HelixDB instead of --log")
    mo.set_defaults(func=_cmd_monitor)

    m = sub.add_parser("mcp-proxy", help="gate every tools/call of an upstream stdio MCP server")
    m.add_argument("--server", help="use a running `noulgate serve` at this URL")
    m.add_argument("--backend", default="rules", help="in-process backend when --server is not given")
    m.add_argument("--policy")
    m.add_argument("--log")
    m.add_argument(
        "--ask-fallback",
        choices=["deny", "allow"],
        default="deny",
        help="what 'ask' becomes when the client cannot show an elicitation prompt",
    )
    m.add_argument("upstream", nargs=argparse.REMAINDER)
    m.set_defaults(func=_cmd_mcp_proxy)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
