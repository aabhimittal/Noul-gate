"""Flag floors: a raised flag caps the decision at ask, whatever the model score says."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from noulgate.policy.combiner import Combiner
from noulgate.policy.conformal import DEFAULT_FLOORS, ConformalPolicy, calibrate, load_policy
from noulgate.schema import Decision

CLASSES = ["benign", "destructive", "data_exfiltration"]


def _policy(floors=None):
    rng = np.random.default_rng(0)
    scores = np.r_[rng.uniform(0, 0.5, 200), rng.uniform(0.4, 1, 200)]
    labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    return calibrate(scores, labels, ["benign"] * 400, 0.05, 0.1, Combiner(CLASSES), floors=floors)


def test_floor_turns_allow_into_ask():
    pol = _policy(DEFAULT_FLOORS)
    assert pol.decide(0.0, "benign")[0] is Decision.ALLOW
    d, why = pol.decide(0.0, "benign", {"tainted_egress": True})
    assert d is Decision.ASK and "floor" in why and "tainted_egress" in why
    assert pol.decide(0.0, "benign", {"secret_in_args": True})[0] is Decision.ASK
    assert pol.decide(0.0, "benign", {"tainted_egress": False})[0] is Decision.ALLOW


def test_floor_never_lowers_a_decision():
    pol = _policy(DEFAULT_FLOORS)
    assert pol.decide(0.99, "benign", {"tainted_egress": True})[0] is Decision.DENY


@given(st.floats(0, 1), st.booleans(), st.booleans())
def test_floors_only_remove_allows(p, taint, secret):
    """Monotone: a floor can only make a decision stricter, so the false-allow bound still holds."""
    order = {Decision.ALLOW: 0, Decision.ASK: 1, Decision.DENY: 2}
    plain, floored = _policy({}), _policy(DEFAULT_FLOORS)
    flags = {"tainted_egress": taint, "secret_in_args": secret}
    assert order[floored.decide(p, "benign", flags)[0]] >= order[plain.decide(p, "benign")[0]]


def test_floors_roundtrip_and_old_policies_have_none(tmp_path):
    pol = _policy(DEFAULT_FLOORS)
    path = tmp_path / "p.json"
    path.write_text(__import__("json").dumps(pol.to_dict()))
    again = load_policy(path)
    assert again.floors == DEFAULT_FLOORS and again.policy_id == pol.policy_id
    old = pol.to_dict()
    old.pop("floors")
    assert ConformalPolicy.from_dict(old).floors == {}
    assert _policy({}).policy_id != pol.policy_id


def test_calibration_stats_count_floored_asks():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 0.3, 100)
    labels = np.zeros(100, int)
    flags = [{"tainted_egress": i < 10} for i in range(100)]
    pol = calibrate(
        scores, labels, ["benign"] * 100, 0.05, 0.1, Combiner(CLASSES), floors=DEFAULT_FLOORS, flags=flags
    )
    assert pol.calibration_stats["ask_rate"] >= 0.1
