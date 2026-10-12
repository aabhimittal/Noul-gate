"""Self-contained HTML reports (plots inlined as base64 PNG)."""

from __future__ import annotations

import base64
import html
import io
import json
import math
from datetime import UTC, datetime
from typing import Any

PALETTE = ["#2a6fdb", "#d9480f", "#2b8a3e", "#862e9c", "#e67700", "#0b7285"]

CAVEATS = """
<ul>
<li><b>Marginal guarantee.</b> The bound says: averaged over harmful actions like those in the calibration
set (and over the draw of that set), at most α are allowed. It says nothing about any single action.</li>
<li><b>Exchangeability.</b> It assumes deployment harmful actions look like calibration harmful actions.
New attack styles, new tools or a new agent model break that assumption.</li>
<li><b>Adaptive attackers.</b> An attacker who can probe the gate can craft text that lowers p(x); the bound
does not hold against them. Treat α as a monitoring target, not a proof of safety, and recalibrate on fresh
attack data.</li>
<li><b>Labels.</b> "Harmful" is a judgment call; label noise and annotator disagreement cap what any backend can reach.</li>
</ul>
"""


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return f'<img alt="plot" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:.3f}"
    return html.escape(str(v))


def _table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _page(title: str, body: str, synthetic: bool) -> str:
    banner = (
        '<div class="banner">SYNTHETIC DATA: these numbers exercise the pipeline only and say nothing about '
        "real-world accuracy.</div>"
        if synthetic
        else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1d1d1f;background:#fff}}
table{{border-collapse:collapse;margin:8px 0 20px;font-size:13px}} td,th{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
th{{background:#f4f4f6}} img{{max-width:100%}} .banner{{background:#fff3bf;border:1px solid #e0b000;padding:10px;margin:12px 0;font-weight:600}}
code{{background:#f4f4f6;padding:1px 4px}} .meta{{color:#666;font-size:13px}}
</style></head><body><h1>{html.escape(title)}</h1>{banner}{body}
<p class="meta">Generated {datetime.now(UTC).isoformat(timespec="seconds")} by noulgate.</p></body></html>"""


def benchmark_report(
    results: dict[str, dict[str, Any]], latency: dict[str, dict[str, Any]], meta: dict[str, Any]
) -> str:
    """results[backend] = repeated_splits(...) output; latency[backend] = {p50_ms, p99_ms, ...}."""
    plt = _plt()
    synthetic = bool(meta.get("synthetic"))
    # headline: false-allow vs ask rate
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (name, res) in enumerate(results.items()):
        pa = sorted(res["per_alpha"].values(), key=lambda d: d["alpha"])
        lat = latency.get(name, {})
        lab = f"{name} (p50 {lat.get('p50_ms', float('nan')):.3g} ms)" if lat else name
        c = PALETTE[i % len(PALETTE)]
        ax.plot(
            [d["ask_rate_mean"] for d in pa], [d["false_allow_mean"] for d in pa], "o-", color=c, label=lab
        )
        for d in pa:
            ax.annotate(
                f"α={d['alpha']:g}",
                (d["ask_rate_mean"], d["false_allow_mean"]),
                fontsize=8,
                textcoords="offset points",
                xytext=(4, 4),
                color=c,
            )
        b = res["uncalibrated_threshold_0.5"]
        ax.plot([b["ask_rate"]], [b["false_allow"]], "x", color=c, markersize=10, mew=2)
    ax.set_xlabel("ask rate (share of all actions sent to a human)")
    ax.set_ylabel("false-allow rate (harmful actions allowed), test split")
    ax.set_title("False-allow vs ask rate (x = uncalibrated threshold 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    headline = _fig_to_img(fig)
    # bound check
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    amax = 0.0
    for i, (name, res) in enumerate(results.items()):
        pa = sorted(res["per_alpha"].values(), key=lambda d: d["alpha"])
        xs = [d["alpha"] for d in pa]
        amax = max(amax, *xs)
        c = PALETTE[i % len(PALETTE)]
        ax.plot(xs, [d["false_allow_mean"] for d in pa], "o-", color=c, label=f"{name} mean")
        ax.plot(
            xs, [d["false_allow_max"] for d in pa], "^:", color=c, alpha=0.6, label=f"{name} max over splits"
        )
        ax.plot(xs, [d["tolerance_95_mean"] for d in pa], "_", color=c, markersize=14, alpha=0.6)
    ax.plot([0, amax], [0, amax], "k--", lw=1, label="y = α")
    ax.set_xlabel("α (target false-allow rate)")
    ax.set_ylabel("empirical false-allow on test")
    ax.set_title("Does the bound hold on held-out data? (dashes = 95% sampling tolerance)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    bound = _fig_to_img(fig)

    sections = [
        f"<p class='meta'>Data: {html.escape(json.dumps(meta, default=str))}</p>",
        "<h2>Headline</h2>",
        headline,
        "<h2>Guarantee check across random splits</h2>",
        bound,
    ]
    for name, res in results.items():
        rows = []
        for d in sorted(res["per_alpha"].values(), key=lambda d: d["alpha"]):
            rows.append({**d, "within_tol": f"{d['splits_within_tolerance']}/{d['n_splits']}"})
        b = res["uncalibrated_threshold_0.5"]
        sections.append(f"<h3>{html.escape(name)}</h3>")
        sections.append(
            _table(
                rows,
                [
                    "alpha",
                    "false_allow_mean",
                    "false_allow_max",
                    "tolerance_95_mean",
                    "within_tol",
                    "ask_rate_mean",
                    "ask_rate_benign_mean",
                    "false_deny_mean",
                ],
            )
        )
        sections.append(
            f"<p>Uncalibrated threshold 0.5: false-allow {b['false_allow']:.3f}, "
            f"false-deny {b['false_deny']:.3f}, ask 0 (no ask region, no guarantee).</p>"
        )
    if latency:
        lat_rows = [
            {"backend": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}}
            for k, v in latency.items()
        ]
        cols = sorted({c for r in lat_rows for c in r} - {"backend"})
        sections += ["<h2>Latency</h2>", _table(lat_rows, ["backend", *cols])]
    sections += ["<h2>What the guarantee does and does not mean</h2>", CAVEATS]
    return _page("NOUL-GATE benchmark", "\n".join(sections), synthetic)


def eval_report(
    metrics: dict[str, Any], policy: dict[str, Any], scores: list[tuple[float, int]], meta: dict[str, Any]
) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = [i / 40 for i in range(41)]
    ax.hist([s for s, y in scores if y == 0], bins=bins, alpha=0.6, label="benign", color=PALETTE[0])
    ax.hist([s for s, y in scores if y == 1], bins=bins, alpha=0.6, label="harmful", color=PALETTE[1])
    for cell in list((policy.get("cells") or {}).values())[:1]:
        for key, style in (("lambda_allow", "--"), ("lambda_deny", ":")):
            v = float(cell[key])
            if math.isfinite(v):
                ax.axvline(v, ls=style, color="k", lw=1)
    ax.set_xlabel("p_harm")
    ax.set_title("Score distribution on the test set (-- λ_allow, ·· λ_deny, first cell)")
    ax.legend()
    img = _fig_to_img(fig)
    body = [
        f"<p class='meta'>{html.escape(json.dumps(meta, default=str))}</p>",
        _table([metrics], [k for k in metrics if not isinstance(metrics[k], (dict, list))]),
        img,
        "<h2>Policy</h2>",
        f"<pre>{html.escape(json.dumps(policy, indent=1, default=str)[:6000])}</pre>",
        "<h2>What the guarantee does and does not mean</h2>",
        CAVEATS,
    ]
    return _page("NOUL-GATE evaluation", "\n".join(body), bool(meta.get("synthetic")))


def monitor_report(rep: dict[str, Any]) -> str:
    """HTML view of `noulgate monitor` output: alerts, rates over time, drift, ask outcomes."""
    body = []
    if rep["alerts"]:
        body.append(
            "<div class='banner'>ALERT: " + "<br>".join(html.escape(a) for a in rep["alerts"]) + "</div>"
        )
    else:
        body.append("<p>No alerts.</p>")
    windows = rep["windows"]
    if windows:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        xs = list(range(len(windows)))
        for i, key in enumerate(("allow_rate", "ask_rate", "deny_rate")):
            ax.plot(xs, [w[key] for w in windows], "o-", color=PALETTE[i], label=key.replace("_", " "))
        ax.set_xticks(xs, [w["start"][:16] for w in windows], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of decisions")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        body += ["<h2>Decision rates per window</h2>", _fig_to_img(fig)]
        body.append(_table(windows, ["start", "n", "allow_rate", "ask_rate", "deny_rate", "p_harm_median"]))
    body += [
        "<h2>Overall vs calibration</h2>",
        _table([rep["overall"]], ["n", "allow_rate", "ask_rate", "deny_rate"]),
        _table([rep["calibration"]], list(rep["calibration"])),
        "<h2>Score drift</h2>",
        _table([rep["drift"]], list(rep["drift"])),
        "<h2>Human answers to asks</h2>",
        _table([{**rep["asks"], "approval_rate": rep["ask_approval_rate"]}], [*rep["asks"], "approval_rate"]),
        "<h2>Notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in rep["notes"]) + "</ul>",
    ]
    return _page("NOUL-GATE monitor", "\n".join(body), synthetic=False)


def attack_report(results: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    """HTML for `noulgate attack`: flip rate and Δp per transform and backend."""
    plt = _plt()
    names = sorted({t for r in results for t in r["transforms"]})
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / max(len(results), 1)
    for i, r in enumerate(results):
        xs = [j + i * width for j in range(len(names))]
        ax.bar(
            xs,
            [r["transforms"].get(n, {}).get("flip_rate", 0.0) for n in names],
            width,
            color=PALETTE[i % len(PALETTE)],
            label=r["backend"],
        )
    ax.set_xticks(
        [j + width * (len(results) - 1) / 2 for j in range(len(names))], names, rotation=20, ha="right"
    )
    ax.set_ylabel("flip rate (harmful → allowed)")
    ax.set_title(f"Attacks on the gate at α = {results[0]['alpha'] if results else '?'}")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    body = [f"<p class='meta'>{html.escape(json.dumps(meta, default=str))}</p>", _fig_to_img(fig)]
    for r in results:
        rows = [{"transform": n, **r["transforms"][n]} for n in names if n in r["transforms"]]
        body += [
            f"<h3>{html.escape(r['backend'])}</h3>",
            f"<p>clean false-allow on the attacked rows: {r['false_allow_clean']:.3f} "
            f"({r['n_harmful_attacked']} harmful test rows)</p>",
            _table(
                rows,
                [
                    "transform",
                    "n",
                    "delta_p_mean",
                    "delta_p_min",
                    "flip_rate",
                    "false_allow_clean",
                    "false_allow_attacked",
                ],
            ),
        ]
    body.append(
        "<p>Attacks change only attacker-controlled text. The conformal bound assumes "
        "exchangeable data, so it does not hold against an adaptive attacker; these numbers show "
        "how much an attacker gains.</p>"
    )
    return _page("NOUL-GATE adversarial evaluation", "\n".join(body), bool(meta.get("synthetic")))
