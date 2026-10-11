"""M3: conformal thresholds and the false-allow guarantee."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noulgate.policy.combiner import Combiner
from noulgate.policy.conformal import (
    POOLED,
    ConformalPolicy,
    beta_binomial_quantile,
    calibrate,
    decide,
    false_allow_upper,
    k_index,
    lambda_allow,
    lambda_deny,
)
from noulgate.schema import Decision

CLASSES = ["benign", "destructive", "data_exfiltration"]


@given(st.lists(st.floats(0, 1), min_size=1, max_size=300), st.floats(0.005, 0.5))
def test_allow_threshold_bounds_calibration_rank(scores, alpha):
    """At most k of the n calibration harmful scores fall strictly below λ_allow."""
    lam = lambda_allow(scores, alpha)
    k = k_index(alpha, len(scores))
    assert sum(s < lam for s in scores) <= max(k - 1, 0)
    if k == 0:
        assert lam == -math.inf


@given(st.lists(st.floats(0, 1), min_size=1, max_size=300), st.floats(0.005, 0.5))
def test_deny_threshold_mirrors(scores, beta):
    lam = lambda_deny(scores, beta)
    k = k_index(beta, len(scores))
    assert sum(s > lam for s in scores) <= max(k - 1, 0)


@given(st.floats(0, 1), st.floats(0, 1), st.floats(0, 1))
def test_decision_regions_respect_both_thresholds(p, la, ld):
    d = decide(p, la, ld)
    if d is Decision.ALLOW:
        assert p < la
    if d is Decision.DENY:
        assert p > ld


@pytest.mark.parametrize("alpha", [0.02, 0.05, 0.1])
def test_false_allow_guarantee_monte_carlo(alpha):
    """Exchangeable scores (even with many ties): average false-allow over draws is ≤ α."""
    rng = np.random.default_rng(0)
    n_h, trials, fa = 200, 3000, []
    for _ in range(trials):
        cal = np.round(rng.beta(5, 2, n_h), 2)  # rounding creates ties
        new = np.round(rng.beta(5, 2, 50), 2)
        fa.append(np.mean(new < lambda_allow(cal, alpha)))
    assert np.mean(fa) <= alpha + 3 * np.std(fa) / np.sqrt(trials)
    assert np.mean(fa) <= k_index(alpha, n_h) / (n_h + 1) + 0.005


def test_too_little_data_allows_nothing():
    assert lambda_allow([0.1, 0.2], 0.02) == -math.inf
    assert lambda_deny([0.1], 0.02) == math.inf


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.4
    p = np.where(y, rng.beta(6, 2, n), rng.beta(2, 6, n))
    pred = np.where(p > 0.5, np.where(rng.random(n) < 0.5, "destructive", "data_exfiltration"), "benign")
    return p, y.astype(int), pred.tolist()


def test_calibrate_mondrian_and_roundtrip():
    p, y, pred = _toy()
    pol = calibrate(p, y, pred, alpha=0.05, beta=0.1, combiner=Combiner(CLASSES))
    assert set(pol.group_of.values()) <= set(pol.cells)
    assert POOLED in pol.cells
    stats = pol.calibration_stats
    assert stats["false_allow"] <= 0.05 and stats["false_deny"] <= 0.1
    again = ConformalPolicy.from_dict(pol.to_dict())
    assert again.policy_id == pol.policy_id
    for x, c in zip(p[:50], pred[:50], strict=True):
        assert again.decide(float(x), c)[0] == pol.decide(float(x), c)[0]


def test_unknown_class_goes_to_pooled_cell():
    p, y, pred = _toy()
    pol = calibrate(p, y, pred, alpha=0.05, beta=0.1, combiner=Combiner(CLASSES))
    assert pol.cell_for("never_seen") == POOLED


def test_beta_binomial_tolerance():
    assert beta_binomial_quantile(10, 1, 1, 0.5) == 5
    tol = false_allow_upper(300, 150, 0.02)
    assert 0.02 < tol < 0.1
    assert false_allow_upper(10, 100, 0.02) == 0.0


def test_combiner_fit_separates():
    rng = np.random.default_rng(1)
    X = rng.random((400, 8))
    y = (X[:, 0] + 0.5 * X[:, 7] > 0.8).astype(int)
    c = Combiner(["benign", "a", "b", "c", "d"]).fit(X, y)
    acc = np.mean((c.score_vec(X) > 0.5) == y)
    assert acc > 0.9 and c.trained and len(c.weights) == 8
    assert Combiner.from_dict(c.to_dict()).score_vec(X[:3]).tolist() == c.score_vec(X[:3]).tolist()


def test_untrained_combiner_is_one_minus_benign():
    c = Combiner(CLASSES)
    ans = {"risk_class": {"probabilities": {"benign": 0.8, "destructive": 0.2}}}
    assert abs(c.score(ans, False) - 0.2) < 1e-9


@settings(max_examples=50)
@given(st.integers(0, 10_000))
def test_calibrate_never_violates_on_calibration_set(seed):
    p, y, pred = _toy(400, seed)
    pol = calibrate(p, y, pred, alpha=0.1, beta=0.1, combiner=Combiner(CLASSES), per_class=False)
    s = pol.calibration_stats
    assert s["false_allow"] <= 0.1 + 1e-12 and s["false_deny"] <= 0.1 + 1e-12


def test_ask_rate_monotone_in_alpha_with_per_class_cells():
    """Regression: the cell partition must not change with α (it once did, making ask non-monotone)."""
    rng = np.random.default_rng(3)
    n = 3000
    y = (rng.random(n) < 0.4).astype(int)
    p = np.where(y == 1, rng.beta(4, 2, n), rng.beta(2, 4, n))
    pred = rng.choice(["benign", "destructive", "data_exfiltration"], size=n).tolist()
    asks, allows_h, partitions = [], [], []
    for a in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        pol = calibrate(p, y, pred, alpha=a, beta=0.1, combiner=Combiner(CLASSES), min_n=50)
        partitions.append(sorted(pol.group_of.items()))
        asks.append(pol.calibration_stats["ask_rate"])
        allows_h.append(pol.calibration_stats["false_allow"])
    assert all(part == partitions[0] for part in partitions)
    assert all(x >= y_ - 1e-12 for x, y_ in zip(asks, asks[1:], strict=False)), asks
    assert all(x <= y_ + 1e-12 for x, y_ in zip(allows_h, allows_h[1:], strict=False)), allows_h
