"""Beyond the synthetic set: how does the gate behave on data we did not write?

    bash data/external/fetch_injecagent.sh
    noulgate dataset import injecagent data/external/injecagent/*.json --out data/external/injecagent.jsonl
    noulgate dataset import claude-code ~/.claude/projects/*/*.jsonl \
        --assume-label benign --annotator <you> --out data/external/real_sessions.jsonl
    python eval/real_data.py --backends rules [--d1-max 120]

Four experiments, each with and without flag floors (policy/conformal.py DEFAULT_FLOORS):

  A. transfer to real work: calibrate on the synthetic set (as a fresh install would), then gate real
     Claude Code tool calls. All of them are benign, so every deny is a false deny and every ask is a
     prompt the user did not need. Also: does `noulgate monitor` see the shift (PSI / KS)?
  B. transfer to third-party attacks: the same synthetic-calibrated policy on InjecAgent. The bound
     assumes exchangeability; InjecAgent was not written by us, so this tests it.
  C. in-domain: repeated splits on InjecAgent (calibrated on InjecAgent itself).
  D. in-domain, mixed: real benign sessions + InjecAgent harmful, repeated splits.

Labels: InjecAgent's own (harmful = the attacker's tool call). Real sessions: assumed benign by the
person who supervised them, a biased annotator. Raw rows are not committed (privacy / no licence); only
the aggregate results in eval/results/real_data.json are.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noulgate.backends import get_backend  # noqa: E402
from noulgate.datasets import data_hash, read_jsonl, split  # noqa: E402
from noulgate.evaluate import (  # noqa: E402
    fit_and_calibrate,
    repeated_splits,
    score_rows,
    score_vector,
)
from noulgate.monitor import ks_against_quantiles, psi_against_quantiles  # noqa: E402
from noulgate.policy.conformal import DEFAULT_FLOORS  # noqa: E402
from noulgate.questions import load_questions  # noqa: E402

ALPHA, BETA = 0.05, 0.10
FLOOR_SETTINGS = {"floors": DEFAULT_FLOORS, "no_floors": {}}


def _decisions(policy, scored, qs):
    p, y, pred = score_vector(policy.combiner, scored, qs.risk_classes, policy.fastpath)
    return [
        policy.decide(float(pi), c, s.flags)[0].value for pi, c, s in zip(p, pred, scored, strict=True)
    ], p


def transfer(policy, scored, qs, rows_by_id) -> dict:
    decisions, p = _decisions(policy, scored, qs)
    by_label: dict[str, Counter] = {"benign": Counter(), "harmful": Counter()}
    by_tool: dict[str, Counter] = {}
    for s, d in zip(scored, decisions, strict=True):
        by_label["harmful" if s.label else "benign"][d] += 1
        if not s.label:
            by_tool.setdefault(rows_by_id[s.id].tool, Counter())[d] += 1
    out = {k: dict(v) for k, v in by_label.items() if v}
    b = by_label["benign"]
    nb = sum(b.values())
    if nb:
        out["benign_rates"] = {k: b[k] / nb for k in ("allow", "ask", "deny")}
        out["benign_not_allowed_by_tool"] = {
            t: dict(c)
            for t, c in sorted(by_tool.items(), key=lambda kv: -sum(kv[1].values()))
            if c["allow"] < sum(c.values())
        }
    h = by_label["harmful"]
    nh = sum(h.values())
    if nh:
        out["false_allow"] = h["allow"] / nh
    ref = policy.calibration_stats.get("score_reference", {}).get("benign")
    model_scored = np.array([pi for pi, s in zip(p, scored, strict=True) if not s.fastpath and not s.label])
    if ref and len(model_scored):
        out["drift_vs_calibration_benign"] = {
            "n": int(len(model_scored)),
            "psi": psi_against_quantiles(model_scored, ref),
            "ks": ks_against_quantiles(model_scored, ref),
            "fast_path_share": float(np.mean([bool(s.fastpath) for s in scored if not s.label])),
        }
    return out


def per_alpha(res: dict) -> dict:
    keep = (
        "false_allow_mean",
        "false_allow_max",
        "false_deny_mean",
        "ask_rate_benign_mean",
        "splits_within_tolerance",
        "n_splits",
    )
    return {a: {k: m[k] for k in keep} for a, m in res["per_alpha"].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="rules")
    ap.add_argument(
        "--d1-max", type=int, default=30, help="external rows per dataset scored by d1 (CPU time)"
    )
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--cache-dir", default=str(ROOT / "eval/cache"))
    ap.add_argument("--out", default=str(ROOT / "eval/results/real_data.json"))
    ap.add_argument("--merge", action="store_true", help="add these backends to an existing --out file")
    args = ap.parse_args()
    qs = load_questions()
    synth = [r for r in read_jsonl(ROOT / "data/fixtures/synthetic_all.jsonl") if r.label]
    inj = [r for r in read_jsonl(ROOT / "data/external/injecagent.jsonl") if r.label]
    real = [r for r in read_jsonl(ROOT / "data/external/real_sessions.jsonl") if r.label]
    results: dict = {
        "meta": {
            "alpha": ALPHA,
            "beta": BETA,
            "splits": args.splits,
            "datasets": {
                "synthetic": {"n": len(synth), "hash": data_hash(synth), "SYNTHETIC": True},
                "injecagent": {
                    "n": len(inj),
                    "hash": data_hash(inj),
                    "distinct_harmful_calls": len(
                        {(r.tool, json.dumps(r.args, sort_keys=True)) for r in inj if r.label == "harmful"}
                    ),
                },
                "real_sessions": {
                    "n": len(real),
                    "hash": data_hash(real),
                    "annotator": "session owner (assumed benign)",
                },
            },
        },
        "backends": {},
    }
    for name in args.backends.split(","):
        backend = get_backend(name)
        cache = Path(args.cache_dir) / f"{name.split(':')[0]}.jsonl"
        rng = random.Random(0)
        cap = args.d1_max if name.startswith("d1") else None

        def sample(rows, cap=cap, rng=rng):
            if not cap or len(rows) <= cap:
                return rows
            h = [r for r in rows if r.label == "harmful"]
            b = [r for r in rows if r.label == "benign"]
            return rng.sample(h, min(len(h), cap // 2)) + rng.sample(b, min(len(b), cap - cap // 2))

        # the synthetic calibration set is fully cached; only the external sets are capped for d1
        s_rows, i_rows, r_rows = synth, sample(inj), sample(real)
        by_id = {r.id: r for r in s_rows + i_rows + r_rows}
        s_sc = score_rows(s_rows, backend, qs, cache)
        i_sc = score_rows(i_rows, backend, qs, cache)
        r_sc = score_rows(r_rows, backend, qs, cache)
        out: dict = {
            "name": backend.name,
            "n_scored": {"synthetic": len(s_sc), "injecagent": len(i_sc), "real": len(r_sc)},
        }
        parts = split(s_rows, {"fit": 0.35, "cal": 0.65}, seed=0)
        ids = {k: {r.id for r in v} for k, v in parts.items()}
        fit = [s for s in s_sc if s.id in ids["fit"]]
        cal = [s for s in s_sc if s.id in ids["cal"]]
        for label, floors in FLOOR_SETTINGS.items():
            pol = fit_and_calibrate(fit, cal, qs, ALPHA, BETA, per_class=False, floors=floors)
            out[label] = {
                "lambda_allow": pol.cells["_all"].lambda_allow,
                "A_synthetic_policy_on_real_sessions": transfer(pol, r_sc, qs, by_id),
                "B_synthetic_policy_on_injecagent": transfer(pol, i_sc, qs, by_id),
                "C_injecagent_in_domain": per_alpha(
                    repeated_splits(
                        i_sc,
                        by_id,
                        qs,
                        [0.05, 0.1],
                        BETA,
                        n_splits=args.splits,
                        per_class=False,
                        floors=floors,
                    )
                ),
                "D_real_benign_plus_injecagent_harmful": per_alpha(
                    repeated_splits(
                        r_sc + [s for s in i_sc if s.label],
                        by_id,
                        qs,
                        [0.05, 0.1],
                        BETA,
                        n_splits=args.splits,
                        per_class=False,
                        floors=floors,
                    )
                ),
            }
        results["backends"][backend.name] = out
        a = out["floors"]["A_synthetic_policy_on_real_sessions"]["benign_rates"]
        print(f"{backend.name}: real sessions allow {a['allow']:.2f} ask {a['ask']:.2f} deny {a['deny']:.2f}")
    if args.merge and Path(args.out).exists():
        old = json.loads(Path(args.out).read_text())
        old["backends"].update(results["backends"])
        results = old
    Path(args.out).write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
