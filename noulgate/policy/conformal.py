"""Conformal risk control on false-allows (SPEC.md §7).

Calibration: n_h harmful scores. Allow iff p(x) < λ_allow, where λ_allow is the k-th smallest harmful
score and k = ⌊α(n_h + 1)⌋. If calibration and deployment harmful actions are exchangeable,
P(p(x_new) < λ_allow) ≤ k/(n_h + 1) ≤ α: the false-allow rate among harmful actions is at most α,
marginally over the draw of the calibration set. Ties only make the strict inequality rarer.

The deny side mirrors it on benign scores: deny iff p(x) > λ_deny, the k_b-th largest benign score with
k_b = ⌊β(n_b + 1)⌋, so the false-deny rate among benign actions is at most β. Everything else is "ask".
When the two regions overlap (λ_deny < λ_allow) no action needs a human: any single cut-point in
[λ_deny, λ_allow] keeps both bounds. We cut at λ_allow, so α remains the knob the user tunes (cutting
at λ_deny instead pins false-allow near 0 and false-deny near β for every α, which makes α meaningless).
The spec leaves this case open; sending the overlap to "ask" made the ask rate grow with α.

Thresholds can be set per predicted risk class (Mondrian conformal): each cell gets its own λ pair and
its own guarantee conditional on the predicted class. Classes with fewer than ``min_n`` harmful or
benign calibration examples share one pooled cell. ``min_n`` is fixed (default 50) and does not depend
on α: if it did, changing α would change the partition and the ask rate would stop being monotone in α
(observed during development: tiny cells of 9 examples gave noisy thresholds). Choosing cells from
calibration counts is still mildly data-dependent; fix the cells in advance (``groups=``) if you need
the textbook guarantee.

What this is not: conditional on the action, robust to distribution shift, or robust to an attacker
who adapts to the gate. Treat α as a monitoring target and recalibrate on fresh attack data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ..schema import Decision
from .combiner import Combiner

POOLED = "_pooled"
DEFAULT_MIN_CELL = 50  # harmful and benign calibration examples needed before a class gets its own cell


def k_index(alpha: float, n: int) -> int:
    return math.floor(alpha * (n + 1))


def lambda_allow(harm_scores, alpha: float) -> float:
    """Largest threshold with the false-allow guarantee; -inf (allow nothing) if n_h < 1/α - 1."""
    s = np.sort(np.asarray(harm_scores, dtype=float))
    k = k_index(alpha, len(s))
    return float(s[k - 1]) if k >= 1 else -math.inf


def lambda_deny(benign_scores, beta: float) -> float:
    """Smallest threshold with the false-deny guarantee; +inf (deny nothing) if n_b < 1/β - 1."""
    s = np.sort(np.asarray(benign_scores, dtype=float))[::-1]
    k = k_index(beta, len(s))
    return float(s[k - 1]) if k >= 1 else math.inf


def min_n_for(alpha: float) -> int:
    """Smallest calibration size for which k = ⌊α(n+1)⌋ ≥ 1."""
    return max(1, math.ceil(1 / alpha - 1))


def decide(p: float, lam_allow: float, lam_deny: float) -> Decision:
    if lam_deny < lam_allow:
        # Overlap: one cut anywhere in [λ_deny, λ_allow] keeps both bounds (allow ⊆ {p < λ_allow},
        # deny ⊆ {p > λ_deny}), so nobody needs to be asked. Cut at λ_allow so that α stays the
        # operating point: false allows up to the chosen α, and the fewest false denies.
        return Decision.ALLOW if p < lam_allow else Decision.DENY
    if p < lam_allow and p <= lam_deny:
        return Decision.ALLOW
    if p > lam_deny and p >= lam_allow:
        return Decision.DENY
    return Decision.ASK


@dataclass
class Cell:
    lambda_allow: float
    lambda_deny: float
    n_h: int
    n_b: int


@dataclass
class ConformalPolicy:
    alpha: float
    beta: float
    combiner: Combiner
    cells: dict[str, Cell]
    group_of: dict[str, str]  # predicted class -> cell name
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    data_hash: str | None = None
    backend: str | None = None
    calibration_stats: dict[str, Any] = field(default_factory=dict)
    fastpath: bool = False  # score fast-path calls 0 without the model (policy/fastpath.py)
    kind: str = "conformal"

    def cell_for(self, predicted: str) -> str:
        return self.group_of.get(predicted, POOLED if POOLED in self.cells else "_all")

    def decide(self, p: float, predicted: str) -> tuple[Decision, str]:
        name = self.cell_for(predicted)
        c = self.cells[name]
        d = decide(p, c.lambda_allow, c.lambda_deny)
        reason = (
            f"p_harm={p:.3f}, cell={name}, λ_allow={c.lambda_allow:.3f}, λ_deny={c.lambda_deny:.3f}, "
            f"α={self.alpha}, β={self.beta}"
        )
        return d, reason

    @property
    def policy_id(self) -> str:
        blob = json.dumps(self.to_dict(include_id=False), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self, include_id: bool = True) -> dict[str, Any]:
        d = {
            "kind": self.kind,
            "version": 1,
            "alpha": self.alpha,
            "beta": self.beta,
            "created": self.created,
            "backend": self.backend,
            "data_hash": self.data_hash,
            "combiner": self.combiner.to_dict(),
            "cells": {k: _json_cell(v) for k, v in self.cells.items()},
            "group_of": self.group_of,
            "calibration_stats": self.calibration_stats,
            "fastpath": self.fastpath,
        }
        if include_id:
            d["policy_id"] = self.policy_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConformalPolicy:
        cells = {
            k: Cell(_unjson(v["lambda_allow"]), _unjson(v["lambda_deny"]), v["n_h"], v["n_b"])
            for k, v in d["cells"].items()
        }
        return cls(
            alpha=d["alpha"],
            beta=d["beta"],
            combiner=Combiner.from_dict(d["combiner"]),
            cells=cells,
            group_of=d["group_of"],
            created=d.get("created", ""),
            data_hash=d.get("data_hash"),
            backend=d.get("backend"),
            calibration_stats=d.get("calibration_stats", {}),
            fastpath=bool(d.get("fastpath", False)),
        )


@dataclass
class ThresholdPolicy:
    """Uncalibrated baseline (SPEC.md §8): allow iff p < t, else deny. No ask, no guarantee."""

    threshold: float = 0.5
    combiner: Combiner | None = None
    fastpath: bool = False
    kind: str = "threshold"

    def decide(self, p: float, predicted: str) -> tuple[Decision, str]:
        d = Decision.ALLOW if p < self.threshold else Decision.DENY
        return d, f"p_harm={p:.3f} vs fixed threshold {self.threshold} (uncalibrated)"

    @property
    def policy_id(self) -> str:
        return f"threshold-{self.threshold}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "threshold": self.threshold,
            "fastpath": self.fastpath,
            "combiner": self.combiner.to_dict() if self.combiner else None,
            "policy_id": self.policy_id,
        }


def _json_cell(c: Cell) -> dict[str, Any]:
    d = asdict(c)
    for k in ("lambda_allow", "lambda_deny"):
        if math.isinf(d[k]):
            d[k] = "inf" if d[k] > 0 else "-inf"
    return d


def _unjson(v) -> float:
    return float(v)


def load_policy(path) -> ConformalPolicy | ThresholdPolicy:
    with open(path) as f:
        d = json.load(f)
    if d.get("kind") == "threshold":
        comb = Combiner.from_dict(d["combiner"]) if d.get("combiner") else None
        return ThresholdPolicy(d["threshold"], comb, fastpath=bool(d.get("fastpath", False)))
    return ConformalPolicy.from_dict(d)


def calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    predicted: list[str],
    alpha: float,
    beta: float,
    combiner: Combiner,
    per_class: bool = True,
    min_n: int = DEFAULT_MIN_CELL,
    groups: dict[str, str] | None = None,
) -> ConformalPolicy:
    """Compute λ_allow / λ_deny per cell from calibration scores (labels: 1 = harmful, 0 = benign)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predicted = list(predicted)
    min_h = min_b = min_n  # independent of α and β so the partition is the same for every α
    if groups is not None:
        group_of = dict(groups)
    elif per_class:
        group_of = {}
        for cls in sorted(set(predicted)):
            idx = [i for i, c in enumerate(predicted) if c == cls]
            n_h = int(labels[idx].sum())
            n_b = len(idx) - n_h
            group_of[cls] = cls if (n_h >= min_h and n_b >= min_b) else POOLED
    else:
        group_of = {cls: "_all" for cls in set(predicted)}
    cell_names = sorted(set(group_of.values()) | ({POOLED} if per_class and groups is None else set()))
    cells: dict[str, Cell] = {}
    for name in cell_names:
        idx = np.array([i for i, c in enumerate(predicted) if group_of.get(c) == name], dtype=int)
        h = scores[idx][labels[idx] == 1] if len(idx) else np.array([])
        b = scores[idx][labels[idx] == 0] if len(idx) else np.array([])
        cells[name] = Cell(lambda_allow(h, alpha), lambda_deny(b, beta), int(len(h)), int(len(b)))
    policy = ConformalPolicy(alpha=alpha, beta=beta, combiner=combiner, cells=cells, group_of=group_of)
    decisions = [policy.decide(float(p), c)[0] for p, c in zip(scores, predicted, strict=True)]
    policy.calibration_stats = rates(decisions, labels)
    policy.calibration_stats["n_h"] = int(labels.sum())
    policy.calibration_stats["n_b"] = int(len(labels) - labels.sum())
    # Reference score distributions for drift monitoring (`noulgate monitor`).
    policy.calibration_stats["score_reference"] = {
        name: score_quantiles(scores[mask])
        for name, mask in (
            ("all", np.ones(len(scores), bool)),
            ("benign", labels == 0),
            ("harmful", labels == 1),
        )
        if mask.any()
    }
    warnings = [
        f"cell {k}: n_h={c.n_h} < {min_n_for(alpha)} so k=0 and nothing is allowed"
        for k, c in cells.items()
        if c.n_h + c.n_b > 0 and math.isinf(c.lambda_allow)
    ]
    if warnings:
        policy.calibration_stats["warnings"] = warnings
    return policy


REFERENCE_QUANTILES = 21  # 0%, 5%, ..., 100%


def score_quantiles(scores) -> list[float]:
    return [
        float(q) for q in np.quantile(np.asarray(scores, dtype=float), np.linspace(0, 1, REFERENCE_QUANTILES))
    ]


def rates(decisions: list[Decision], labels) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    d = np.array([x.value for x in decisions])
    h, b = labels == 1, labels == 0
    return {
        "false_allow": float(np.mean(d[h] == "allow")) if h.any() else float("nan"),
        "false_deny": float(np.mean(d[b] == "deny")) if b.any() else float("nan"),
        "ask_rate": float(np.mean(d == "ask")) if len(d) else float("nan"),
        "ask_rate_benign": float(np.mean(d[b] == "ask")) if b.any() else float("nan"),
        "allow_rate_benign": float(np.mean(d[b] == "allow")) if b.any() else float("nan"),
        "deny_rate_harmful": float(np.mean(d[h] == "deny")) if h.any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# sampling tolerance for checking the bound on held-out data (M3 acceptance)
# --------------------------------------------------------------------------- #


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_binomial_quantile(n: int, a: float, b: float, q: float) -> int:
    """q-quantile of BetaBinomial(n, a, b): the count of false allows among n test harmful actions."""
    cdf = 0.0
    for x in range(n + 1):
        logp = (
            math.lgamma(n + 1)
            - math.lgamma(x + 1)
            - math.lgamma(n - x + 1)
            + _log_beta(x + a, n - x + b)
            - _log_beta(a, b)
        )
        cdf += math.exp(logp)
        if cdf >= q:
            return x
    return n


def false_allow_upper(n_cal_h: int, n_test_h: int, alpha: float, q: float = 0.95) -> float:
    """Upper q-quantile of the empirical test false-allow rate under exchangeability.

    Given the calibration set, the true false-allow rate is distributed Beta(k, n+1-k) (continuous
    scores); the test count adds binomial noise. Their mixture is beta-binomial.
    """
    k = k_index(alpha, n_cal_h)
    if k < 1 or n_test_h == 0:
        return 0.0
    return beta_binomial_quantile(n_test_h, k, n_cal_h + 1 - k, q) / n_test_h
