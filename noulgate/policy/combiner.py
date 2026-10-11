"""Turn the backend's typed answers into one harm probability p(x) (SPEC.md §7).

Untrained: p = 1 - P(benign). Trained: a small L2-regularised logistic regression over
[1-P(benign), 1-P(user_requested), severity (0..1), P(each harmful class), targets-only-in-untrusted].

Fit it on a split that is disjoint from the conformal calibration split, otherwise the calibration
scores are not exchangeable with deployment scores and the bound in conformal.py does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..backends.base import Answers


def risk_probs(answers: Answers) -> dict[str, float]:
    rc = answers.get("risk_class") or {}
    return dict(rc.get("probabilities") or {})


def predicted_class(answers: Answers) -> str:
    probs = risk_probs(answers)
    return max(probs, key=probs.get) if probs else "unknown"


def feature_names(classes: list[str]) -> list[str]:
    harm = [c for c in classes if c != "benign"]
    return ["not_benign", "not_user_requested", "severity", *[f"p_{c}" for c in harm], "untrusted_only"]


def features(answers: Answers, classes: list[str], untrusted_only: bool) -> list[float]:
    probs = risk_probs(answers)
    not_benign = 1.0 - probs.get("benign", 0.0)
    ur = answers.get("user_requested", {}).get("noul")
    sev = answers.get("severity", {})
    if "probabilities" in sev and sev["probabilities"]:
        n = len(sev["probabilities"])
        sev_norm = float(sev.get("score", 0.0)) / max(n - 1, 1)
    else:
        sev_norm = 0.5
    harm = [c for c in classes if c != "benign"]
    return [
        not_benign,
        1.0 - float(ur) if ur is not None else 0.5,
        sev_norm,
        *[float(probs.get(c, 0.0)) for c in harm],
        1.0 if untrusted_only else 0.0,
    ]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


@dataclass
class Combiner:
    classes: list[str]
    weights: list[float] | None = None  # None -> untrained: p = 1 - P(benign)
    bias: float = 0.0
    l2: float = 1.0
    n_fit: int = 0
    names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.names = feature_names(self.classes)

    @property
    def trained(self) -> bool:
        return self.weights is not None

    def score_vec(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.trained:
            return np.clip(X[:, 0], 0.0, 1.0)
        return _sigmoid(X @ np.asarray(self.weights) + self.bias)

    def score(self, answers: Answers, untrusted_only: bool) -> float:
        return float(self.score_vec(np.array([features(answers, self.classes, untrusted_only)]))[0])

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 50) -> Combiner:
        """IRLS (Newton) for L2-regularised logistic regression; the intercept is not penalised."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if len(set(y.tolist())) < 2:
            raise ValueError("combiner needs both harmful and benign examples")
        A = np.hstack([X, np.ones((n, 1))])
        w = np.zeros(d + 1)
        reg = np.full(d + 1, self.l2)
        reg[-1] = 0.0
        for _ in range(iters):
            p = _sigmoid(A @ w)
            g = A.T @ (p - y) + reg * w
            H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(d + 1)
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.weights, self.bias, self.n_fit = w[:-1].tolist(), float(w[-1]), n
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "feature_names": self.names,
            "weights": self.weights,
            "bias": self.bias,
            "l2": self.l2,
            "n_fit": self.n_fit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Combiner:
        return cls(
            classes=d["classes"],
            weights=d.get("weights"),
            bias=d.get("bias", 0.0),
            l2=d.get("l2", 1.0),
            n_fit=d.get("n_fit", 0),
        )
