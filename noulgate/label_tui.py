"""Minimal terminal labeling tool (M2). Plain stdin/stdout so it runs over SSH and in any terminal.

    noulgate label --in candidates.jsonl --out labels/alice.jsonl --annotator alice

Keys: b = benign, h = harmful (then pick a risk class), s = skip, u = undo last, q = quit.
Labels are appended as they are made; re-running resumes where the annotator stopped.
Use one output file per annotator, then `noulgate agreement` to measure inter-annotator agreement.
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import TextIO

from .datasets import LabeledAction, read_jsonl, write_jsonl
from .questions import RISK_CLASS_DESCRIPTIONS

HARM_CLASSES = [c for c in RISK_CLASS_DESCRIPTIONS if c != "benign"]


def _render(row: LabeledAction, i: int, n: int, width: int = 100) -> str:
    out = [
        f"\n{'=' * width}",
        f"[{i + 1}/{n}] {row.id}  source={row.source}" + (f"  hint={row.hint}" if row.hint else ""),
    ]
    out.append(f"PENDING ACTION  {row.tool}")
    args = json.dumps(row.args, ensure_ascii=False, indent=1)
    out += ["  " + line for line in args[:1500].splitlines()]
    out.append("CONTEXT (oldest first, provenance in brackets)")
    for span in row.context[-6:]:
        body = textwrap.shorten(span.text.replace("\n", " "), width=width * 3, placeholder=" …")
        out.append(f"  [{span.source}] {body}")
    if not row.context:
        out.append("  (none)")
    return "\n".join(out)


def run(
    in_path: str | Path,
    out_path: str | Path,
    annotator: str,
    inp: TextIO = sys.stdin,
    out: TextIO = sys.stdout,
) -> Counter:
    rows = read_jsonl(in_path)
    out_p = Path(out_path)
    done: list[LabeledAction] = read_jsonl(out_p) if out_p.exists() else []
    done_ids = {r.id for r in done}
    todo = [r for r in rows if r.id not in done_ids]
    counts: Counter = Counter()
    out.write(f"{len(done)} already labeled by {annotator}, {len(todo)} to go.\n")
    i = 0
    while i < len(todo):
        row = todo[i]
        out.write(_render(row, len(done_ids) + i, len(rows)) + "\n")
        out.write("[b]enign  [h]armful  [s]kip  [u]ndo  [q]uit > ")
        out.flush()
        key = inp.readline()
        if not key:
            break
        key = key.strip().lower()
        if key == "q":
            break
        if key == "s":
            counts["skipped"] += 1
            i += 1
            continue
        if key == "u":
            if done and i > 0:
                undone = done.pop()
                write_jsonl(out_p, done)
                counts[undone.label] -= 1
                i -= 1
                out.write(f"undid {undone.id}\n")
            else:
                out.write("nothing to undo in this session\n")
            continue
        if key not in {"b", "h"}:
            out.write("?\n")
            continue
        label, risk = ("benign", "benign") if key == "b" else ("harmful", None)
        while risk is None:
            out.write(
                "  risk class: " + "  ".join(f"{k + 1}={c}" for k, c in enumerate(HARM_CLASSES)) + " > "
            )
            out.flush()
            ans = inp.readline()
            if not ans:
                return counts
            ans = ans.strip()
            if ans.isdigit() and 1 <= int(ans) <= len(HARM_CLASSES):
                risk = HARM_CLASSES[int(ans) - 1]
        labeled = row.model_copy(update={"label": label, "risk_class": risk, "annotator": annotator})
        done.append(labeled)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("a") as f:
            f.write(labeled.model_dump_json(exclude_none=True) + "\n")
        counts[label] += 1
        i += 1
    out.write(f"\nsession: {dict(counts)}; file: {out_p}\n")
    return counts


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(paths: list[str | Path]) -> dict:
    """Pairwise Cohen's kappa on the harmful/benign label over items both annotators labeled."""
    sets = {str(p): {r.id: r for r in read_jsonl(p)} for p in paths}
    names = list(sets)
    pairs = []
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            common = sorted(set(sets[names[x]]) & set(sets[names[y]]))
            la = [sets[names[x]][i].label for i in common]
            lb = [sets[names[y]][i].label for i in common]
            pairs.append(
                {
                    "a": names[x],
                    "b": names[y],
                    "n_common": len(common),
                    "agree": sum(p == q for p, q in zip(la, lb, strict=True)),
                    "kappa": round(cohen_kappa(la, lb), 4) if common else None,
                }
            )
    return {"pairs": pairs}


def merge(paths: list[str | Path], policy: str = "unanimous") -> list[LabeledAction]:
    """Combine annotator files. 'unanimous' keeps items where all who labeled them agree."""
    by_id: dict[str, list[LabeledAction]] = {}
    for p in paths:
        for r in read_jsonl(p):
            by_id.setdefault(r.id, []).append(r)
    out = []
    for rs in by_id.values():
        labels = {r.label for r in rs}
        if policy == "unanimous" and len(labels) != 1:
            continue
        winner = Counter(r.label for r in rs).most_common(1)[0][0]
        pick = next(r for r in rs if r.label == winner)
        out.append(pick.model_copy(update={"annotator": ",".join(sorted({r.annotator or "?" for r in rs}))}))
    return out
