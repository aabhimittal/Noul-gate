"""Scoring, calibration and evaluation pipelines (M3, M5).

Backend answers are cached per (backend, questions, row id) so an expensive backend (d1 on CPU, an LLM
judge) is run once per row and reused across splits and α values.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends.base import Answers, DecisionBackend
from .datasets import LabeledAction, data_hash, split
from .latency import percentile
from .policy.combiner import Combiner, features, predicted_class
from .policy.conformal import (
    ConformalPolicy,
    ThresholdPolicy,
    calibrate,
    false_allow_upper,
    rates,
)
from .policy.fastpath import FASTPATH_CLASS, fast_answers, fast_path
from .questions import QuestionSet


@dataclass
class Scored:
    id: str
    label: int  # 1 harmful, 0 benign
    answers: Answers
    flags: dict[str, bool]
    latency_ms: float
    source: str
    fastpath: str | None = None  # reason, if the fast path scored this row (no backend call)


def questions_hash(qs: QuestionSet) -> str:
    return hashlib.sha256(json.dumps(qs.to_d1(), sort_keys=True).encode()).hexdigest()[:12]


def score_rows(
    rows: list[LabeledAction],
    backend: DecisionBackend,
    qs: QuestionSet,
    cache: str | Path | None = None,
    batch: int = 8,
    fastpath: bool = True,
) -> list[Scored]:
    """Backend answers for every labeled row. With `fastpath`, rows the fast path accepts get
    placeholder answers and no backend call (they score p = 0 under a fast-path policy)."""
    rows = [r for r in rows if r.label in ("harmful", "benign")]
    backend.warmup(qs)  # loads lazy models, which also fixes their final name (device, dtype)
    key = f"{backend.name}|{questions_hash(qs)}"
    states = {r.id: r.to_state() for r in rows}
    # Cache by what the backend actually sees, not by row id: regenerated data can reuse ids with new
    # content, and identical states (common in templated data) are scored once.
    fast = {rid: (fast_path(st) if fastpath else None) for rid, st in states.items()}
    digest = {rid: state_hash(st) for rid, st in states.items()}
    cached: dict[str, dict] = {}
    cache_p = Path(cache) if cache else None
    if cache_p and cache_p.exists():
        for line in cache_p.read_text().splitlines():
            d = json.loads(line)
            if d.get("key") == key and "state" in d:
                cached[d["state"]] = d
    todo_hashes = list(dict.fromkeys(h for rid, h in digest.items() if h not in cached and not fast[rid]))
    first_row = {}
    for r in rows:
        first_row.setdefault(digest[r.id], r)
    for i in range(0, len(todo_hashes), batch):
        chunk = [first_row[h] for h in todo_hashes[i : i + batch]]
        t = time.perf_counter()
        answers = backend.decide_batch([states[r.id] for r in chunk], qs)
        ms = (time.perf_counter() - t) * 1000 / len(chunk)
        new = [
            {"key": key, "state": digest[r.id], "answers": a, "latency_ms": ms}
            for r, a in zip(chunk, answers, strict=True)
        ]
        for d in new:
            cached[d["state"]] = d
        if cache_p:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            with cache_p.open("a") as f:
                f.writelines(json.dumps(d) + "\n" for d in new)
    out = []
    for r in rows:
        if fast[r.id]:
            out.append(
                Scored(
                    r.id,
                    int(r.label == "harmful"),
                    fast_answers(fast[r.id]),
                    states[r.id].flags,
                    0.0,
                    r.source,
                    fast[r.id],
                )
            )
        else:
            hit = cached[digest[r.id]]
            out.append(
                Scored(
                    r.id,
                    int(r.label == "harmful"),
                    hit["answers"],
                    states[r.id].flags,
                    float(hit["latency_ms"]),
                    r.source,
                )
            )
    return out


def state_hash(state) -> str:
    return hashlib.sha256(state.model_dump_json().encode()).hexdigest()[:24]


def design(scored: list[Scored], classes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.array([features(s.answers, classes, s.flags) for s in scored], dtype=float)
    y = np.array([s.label for s in scored], dtype=int)
    pred = [predicted_class(s.answers) for s in scored]
    return X, y, pred


def score_vector(
    comb: Combiner, scored: list[Scored], classes: list[str], fastpath: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The score function s(x) used by calibration *and* evaluation: 0 for fast-path rows when the
    policy uses the fast path, the combiner's p(x) otherwise. Returns (scores, labels, predicted)."""
    if not fastpath and any(s.fastpath for s in scored):
        raise ValueError(
            "rows were scored with the fast path; rescore with fastpath=False for a non-fast-path policy"
        )
    X, y, pred = design(scored, classes)
    p = comb.score_vec(X) if len(scored) else np.zeros(0)
    for i, s in enumerate(scored):
        if fastpath and s.fastpath:
            p[i], pred[i] = 0.0, FASTPATH_CLASS
    return p, y, pred


def fit_and_calibrate(
    fit: list[Scored],
    cal: list[Scored],
    qs: QuestionSet,
    alpha: float,
    beta: float,
    per_class: bool = True,
    l2: float = 1.0,
    fastpath: bool = True,
    floors: dict[str, str] | None = None,
) -> ConformalPolicy:
    classes = qs.risk_classes
    comb = Combiner(classes, l2=l2)
    model_rows = [s for s in fit if not (fastpath and s.fastpath)]  # fast-path rows have no model answers
    if model_rows and len({s.label for s in model_rows}) == 2:
        Xf, yf, _ = design(model_rows, classes)
        comb.fit(Xf, yf)
    p, yc, pc = score_vector(comb, cal, classes, fastpath)
    policy = calibrate(
        p, yc, pc, alpha, beta, comb, per_class=per_class, floors=floors, flags=[s.flags for s in cal]
    )
    policy.fastpath = fastpath
    return policy


def evaluate_policy(
    policy: ConformalPolicy | ThresholdPolicy, test: list[Scored], qs: QuestionSet
) -> dict[str, Any]:
    comb = policy.combiner or Combiner(qs.risk_classes)
    p, y, pred = score_vector(comb, test, qs.risk_classes, getattr(policy, "fastpath", False))
    decisions = [policy.decide(float(pi), c, s.flags)[0] for pi, c, s in zip(p, pred, test, strict=True)]
    out = rates(decisions, y)
    out.update({"n_h": int(y.sum()), "n_b": int(len(y) - y.sum())})
    lat = [s.latency_ms for s in test]
    out["latency_p50_ms"] = float(np.median(lat)) if lat else float("nan")
    out["latency_p99_ms"] = percentile(lat, 0.99)
    return out


def repeated_splits(
    scored: list[Scored],
    rows_by_id: dict[str, LabeledAction],
    qs: QuestionSet,
    alphas: list[float],
    beta: float,
    n_splits: int = 20,
    fractions=None,
    per_class: bool = True,
    fit_combiner: bool = True,
    fastpath: bool = True,
    floors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """M3 acceptance: does empirical false-allow on held-out data stay within α + tolerance?"""
    fractions = fractions or {"fit": 0.25, "cal": 0.45, "test": 0.30}
    by_id = {s.id: s for s in scored}
    rows = [rows_by_id[s.id] for s in scored]
    results: dict[str, list[dict]] = {str(a): [] for a in alphas}
    baseline = []
    for seed in range(n_splits):
        parts = split(rows, fractions, seed=seed)
        fit = [by_id[r.id] for r in parts.get("fit", [])] if fit_combiner else []
        cal = [by_id[r.id] for r in parts["cal"]] + (
            [] if fit_combiner else [by_id[r.id] for r in parts.get("fit", [])]
        )
        test = [by_id[r.id] for r in parts["test"]]
        n_cal_h = sum(s.label for s in cal)
        for a in alphas:
            pol = fit_and_calibrate(
                fit, cal, qs, a, beta, per_class=per_class, fastpath=fastpath, floors=floors
            )
            m = evaluate_policy(pol, test, qs)
            # per-cell guarantees hold cell by cell; for the global tolerance use the pooled sizes
            m["tolerance_95"] = false_allow_upper(n_cal_h, m["n_h"], a)
            m["within_tolerance"] = bool(m["false_allow"] <= m["tolerance_95"] + 1e-12)
            results[str(a)].append(m)
        baseline.append(evaluate_policy(ThresholdPolicy(0.5, fastpath=fastpath), test, qs))
    summary = {}
    for a, ms in results.items():
        fa = [m["false_allow"] for m in ms]
        summary[a] = {
            "alpha": float(a),
            "false_allow_mean": float(np.mean(fa)),
            "false_allow_max": float(np.max(fa)),
            "false_allow_sd": float(np.std(fa)),
            "ask_rate_mean": float(np.mean([m["ask_rate"] for m in ms])),
            "ask_rate_benign_mean": float(np.mean([m["ask_rate_benign"] for m in ms])),
            "false_deny_mean": float(np.mean([m["false_deny"] for m in ms])),
            "splits_within_tolerance": int(sum(m["within_tolerance"] for m in ms)),
            "n_splits": len(ms),
            "tolerance_95_mean": float(np.mean([m["tolerance_95"] for m in ms])),
        }
    base = {k: float(np.mean([b[k] for b in baseline])) for k in ("false_allow", "false_deny", "ask_rate")}
    return {
        "per_alpha": summary,
        "splits": results,
        "uncalibrated_threshold_0.5": base,
        "fractions": fractions,
        "beta": beta,
        "n_rows": len(scored),
        "data_hash": data_hash(rows),
    }
