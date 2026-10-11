"""`noulgate monitor`: watch a deployed gate for the failure the guarantee can't see by itself.

The conformal bound holds only while deployed actions look like the calibration actions (SPEC.md §7).
The deployed false-allow rate cannot be observed directly, because an allowed harmful call never
announces itself. So this monitor tracks proxies and says plainly which they are:

* decision rates (allow / ask / deny) per time window, against the rates seen at calibration;
* the ask approval rate (from `noulgate harvest`): if people approve almost every ask, the gate asks
  too much; if they reject many, harmful traffic has risen;
* score drift: PSI and a KS distance between deployed p_harm and the calibration *benign* score
  distribution (deployment traffic is overwhelmingly benign; comparing with the harmful-enriched
  calibration mix would always look like drift).

PSI > 0.2, a KS distance > 0.2, or an ask or deny rate more than 2x its calibration value sets
`alert`. The CLI then exits with status 1, so a cron job or CI step can page someone.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .feedback import harvest

PSI_ALERT = 0.2
KS_ALERT = 0.2
RATE_FACTOR_ALERT = 2.0


def parse_window(text: str) -> timedelta:
    units = {"m": "minutes", "h": "hours", "d": "days"}
    if not text or text[-1] not in units or not text[:-1].isdigit():
        raise ValueError(f"window must look like 30m, 6h or 1d, got {text!r}")
    return timedelta(**{units[text[-1]]: int(text[:-1])})


def _ts(rec: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(str(rec["ts"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None


def psi_against_quantiles(values: np.ndarray, quantiles: list[float], eps: float = 1e-4) -> float:
    """Population stability index of `values` against a reference given by its quantiles.

    Bins are the distinct reference quantiles; each bin's expected mass is the share of quantile steps it
    spans, so ties in the reference (e.g. many scores at exactly 0) are handled.
    """
    q = np.asarray(quantiles, dtype=float)
    steps = len(q) - 1
    edges = np.unique(q)
    if len(edges) < 2 or len(values) == 0:
        return float("nan")
    # expected mass per bin [edges[i], edges[i+1]]: fraction of reference quantile steps inside it
    expected = np.array(
        [np.sum((q[1:] > lo) & (q[1:] <= hi)) for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
        dtype=float,
    )
    expected[0] += np.sum(q[1:] <= edges[0])  # steps of zero width at the minimum
    expected /= steps
    inner = np.clip(values, edges[0], edges[-1])
    observed = np.histogram(inner, bins=edges)[0].astype(float) / len(values)
    e, o = np.maximum(expected, eps), np.maximum(observed, eps)
    return float(np.sum((o - e) * np.log(o / e)))


def ks_against_quantiles(values: np.ndarray, quantiles: list[float]) -> float:
    """Max |F_deployed(x) - F_reference(x)| evaluated at the reference quantile points (an approximation)."""
    if len(values) == 0:
        return float("nan")
    q = np.asarray(quantiles, dtype=float)
    ref_cdf = np.linspace(0, 1, len(q))
    obs_cdf = np.searchsorted(np.sort(values), q, side="right") / len(values)
    return float(np.max(np.abs(obs_cdf - ref_cdf)))


@dataclass
class MonitorReport:
    n_decisions: int
    overall: dict[str, float]
    windows: list[dict[str, Any]]
    asks: dict[str, int]
    ask_approval_rate: float | None
    drift: dict[str, float]
    calibration: dict[str, Any]
    alerts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def alert(self) -> bool:
        return bool(self.alerts)

    def to_dict(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in (
                "n_decisions",
                "overall",
                "windows",
                "asks",
                "ask_approval_rate",
                "drift",
                "calibration",
                "alerts",
                "notes",
            )
        } | {"alert": self.alert}


def _rates(decisions: list[str]) -> dict[str, float]:
    c, n = Counter(decisions), max(len(decisions), 1)
    return {"n": len(decisions), **{f"{k}_rate": c.get(k, 0) / n for k in ("allow", "ask", "deny")}}


def monitor(
    records: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
    window: str = "1d",
    policy_id: str | None = None,
) -> MonitorReport:
    policy_id = policy_id or (policy or {}).get("policy_id")
    decisions = [
        r
        for r in records
        if r.get("kind", "decision") == "decision"
        and r.get("decision")
        and (policy_id is None or r.get("policy_id") in (policy_id, None))
    ]
    notes = [
        "Deployed false-allow rate is not observable: allowed harmful calls never surface. "
        "These are proxies; re-evaluate on fresh labeled attacks to check the bound itself."
    ]
    if policy_id:
        notes.append(f"filtered to policy {policy_id}")
    overall = _rates([r["decision"] for r in decisions])

    windows: list[dict[str, Any]] = []
    width = parse_window(window)
    stamped = sorted((t, r) for r in decisions if (t := _ts(r)) is not None)
    if stamped:
        start = stamped[0][0]
        buckets: dict[int, list[dict[str, Any]]] = {}
        for t, r in stamped:
            buckets.setdefault(int((t - start) / width), []).append(r)
        for i in sorted(buckets):
            w = buckets[i]
            ps = [r["p_harm"] for r in w if isinstance(r.get("p_harm"), (int, float))]
            windows.append(
                {
                    "start": (start + i * width).isoformat(),
                    **_rates([r["decision"] for r in w]),
                    "p_harm_median": float(np.median(ps)) if ps else None,
                }
            )

    asks = harvest(records)
    ask_counts = dict(Counter(a.answer for a in asks))
    answered = ask_counts.get("approved", 0) + ask_counts.get("rejected", 0)
    approval = ask_counts.get("approved", 0) / answered if answered else None

    stats = (policy or {}).get("calibration_stats", {})
    ref = stats.get("score_reference", {})
    scored = np.array(
        [
            r["p_harm"]
            for r in decisions
            if isinstance(r.get("p_harm"), (int, float)) and r.get("risk_class") != "_fastpath"
        ],
        dtype=float,
    )
    drift: dict[str, float] = {"n_scored": int(len(scored))}
    alerts: list[str] = []
    if "benign" in ref and len(scored):
        drift["psi_vs_benign"] = psi_against_quantiles(scored, ref["benign"])
        drift["ks_vs_benign"] = ks_against_quantiles(scored, ref["benign"])
        if drift["psi_vs_benign"] > PSI_ALERT:
            alerts.append(
                f"score drift: PSI {drift['psi_vs_benign']:.2f} > {PSI_ALERT} against calibration benign scores"
            )
        if drift["ks_vs_benign"] > KS_ALERT:
            alerts.append(f"score drift: KS {drift['ks_vs_benign']:.2f} > {KS_ALERT}")
    elif policy is not None and "benign" not in ref:
        notes.append(
            "policy has no score_reference (calibrated before drift monitoring existed); recalibrate to enable it"
        )
    elif policy is not None:
        notes.append("no scored decisions to compare with the calibration scores")
    for key, cal_key in (("ask_rate", "ask_rate_benign"), ("deny_rate", "false_deny")):
        cal = stats.get(cal_key)
        if isinstance(cal, (int, float)) and not math.isnan(cal) and overall["n"] >= 50:
            if overall[key] > RATE_FACTOR_ALERT * max(cal, 0.01):
                alerts.append(
                    f"{key} {overall[key]:.3f} is more than {RATE_FACTOR_ALERT:g}x its calibration value {cal:.3f}"
                )
    if overall["n"] < 50:
        notes.append(f"only {overall['n']} decisions: rate alerts need at least 50")
    calibration = {
        k: stats.get(k) for k in ("ask_rate", "ask_rate_benign", "false_deny", "false_allow", "n_h", "n_b")
    }
    return MonitorReport(
        len(decisions), overall, windows, ask_counts, approval, drift, calibration, alerts, notes
    )
