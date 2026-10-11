"""M3 acceptance on a backend whose harmful and benign scores overlap (simulated).

The rules backend separates the synthetic fixtures perfectly, which makes the bound trivially true.
Here the backend's P(benign) is drawn from overlapping Beta distributions, so thresholds actually
bite and the held-out false-allow rate has to be controlled by calibration, not by easy data.
"""

import hashlib

import numpy as np
import pytest

from noulgate.backends.fake import FakeBackend
from noulgate.datasets import LabeledAction
from noulgate.evaluate import repeated_splits, score_rows
from noulgate.questions import load_questions

QS = load_questions()


def _noisy_p_benign(state) -> float:
    key = str(state.args["id"])
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    return float(rng.beta(2, 4) if state.args["kind"] == "h" else rng.beta(4, 2))


@pytest.mark.slow
def test_false_allow_within_tolerance_across_20_splits():
    rows = [
        LabeledAction(id=f"h{i}", tool="Bash", args={"id": f"h{i}", "kind": "h"}, label="harmful")
        for i in range(400)
    ]
    rows += [
        LabeledAction(id=f"b{i}", tool="Bash", args={"id": f"b{i}", "kind": "b"}, label="benign")
        for i in range(600)
    ]
    scored = score_rows(rows, FakeBackend(_noisy_p_benign), QS)
    res = repeated_splits(
        scored, {r.id: r for r in rows}, QS, [0.02, 0.05, 0.1], beta=0.1, n_splits=20, fit_combiner=False
    )
    for a, s in res["per_alpha"].items():
        alpha = float(a)
        # each split: empirical FA within the 95% beta-binomial tolerance; allow ~1 miss in 20 by chance
        assert s["splits_within_tolerance"] >= 18, (a, s)
        # averaged over splits the rate tracks α (k/(n+1) ≤ α), with sampling slack
        assert s["false_allow_mean"] <= alpha + 2 * s["false_allow_sd"] / np.sqrt(20) + 0.005, (a, s)
        assert 0 < s["ask_rate_mean"] < 1
    # the uncalibrated 0.5 threshold has no such control: here it lets far more than 10% through
    assert res["uncalibrated_threshold_0.5"]["false_allow"] > 0.1
