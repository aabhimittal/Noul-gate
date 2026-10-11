"""Measure backend latency on this machine: python eval/latency.py --backend d1-3b --runs 30"""

import argparse
import json

from noulgate.backends import get_backend
from noulgate.latency import measure
from noulgate.questions import load_questions

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="rules")
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--long-runs", type=int)
ap.add_argument("--out")
args = ap.parse_args()
report = measure(
    get_backend(args.backend), load_questions(), runs=args.runs, batch=args.batch, long_runs=args.long_runs
)
text = json.dumps(report, indent=2)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
