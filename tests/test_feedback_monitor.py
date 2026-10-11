"""Feedback loop (asks -> labels) and drift monitoring."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from noulgate._jsonl import new_record
from noulgate.feedback import classify, harvest, read_records, to_labeled
from noulgate.monitor import ks_against_quantiles, monitor, parse_window, psi_against_quantiles
from noulgate.policy.conformal import score_quantiles


def _ask(tid, transcript=None, p=0.5, **kw):
    return new_record(
        tool_name="Bash",
        tool_use_id=tid,
        decision="ask",
        emitted="ask",
        p_harm=p,
        mode="enforce",
        session_id="s",
        transcript_path=transcript,
        **kw,
    )


def _transcript(tmp_path, results):
    """A Claude Code transcript: user message, then one tool_use and tool_result per (id, result) pair."""
    lines = [{"type": "user", "message": {"role": "user", "content": "clean up the build"}}]
    for tid, result in results:
        lines.append(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tid,
                            "name": "Bash",
                            "input": {"command": f"rm -rf build-{tid}"},
                        }
                    ],
                },
            }
        )
        if result is not None:
            lines.append(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tid, "is_error": True, "content": result}
                        ],
                    },
                }
            )
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    return str(p)


def test_harvest_classifies_every_answer(tmp_path):
    tp = _transcript(
        tmp_path,
        [
            ("t_ok", None),
            ("t_no", "The user doesn't want to proceed with this tool use. The tool use was rejected."),
            ("t_auto", "NOUL-GATE ask: p_harm=0.400, cell=_all"),
            ("t_odd", "something else entirely"),
            ("t_gone", None),
        ],
    )
    records = [_ask(t, tp) for t in ("t_ok", "t_no", "t_auto", "t_odd", "t_gone")]
    records.append(
        new_record(
            kind="outcome",
            tool_name="Bash",
            tool_use_id="t_ok",
            outcome="failed",
            session_id="s",
            hook_event="PostToolUseFailure",
        )
    )
    records.append(new_record(tool_name="Bash", tool_use_id="t_allow", decision="allow", session_id="s"))
    answers = {a.decision["tool_use_id"]: a.answer for a in harvest(records)}
    assert answers == {
        "t_ok": "approved",
        "t_no": "rejected",
        "t_auto": "no_human",
        "t_odd": "unknown",
        "t_gone": "unknown",
    }

    rows = {r.args["command"]: r for r in to_labeled(harvest(records), auto_label=True)}
    ok, no = rows["rm -rf build-t_ok"], rows["rm -rf build-t_no"]
    assert (ok.label, ok.hint, ok.annotator) == ("benign", "human_approved", "human_ask")
    assert (no.label, no.hint) == ("harmful", "human_rejected")
    assert rows["rm -rf build-t_auto"].label is None
    assert [s.source for s in ok.context] == ["user"] and ok.context[0].text == "clean up the build"
    # context is cut at the call: the later rejection is not visible to the earlier call
    assert all("rejected" not in s.text for s in ok.context)


def test_harvest_without_auto_label_only_hints(tmp_path):
    tp = _transcript(tmp_path, [("a", None)])
    recs = [
        _ask("a", tp),
        new_record(kind="outcome", tool_name="Bash", tool_use_id="a", outcome="ran", session_id="s"),
    ]
    [row] = to_labeled(harvest(recs))
    assert row.label is None and row.hint == "human_approved"


def test_classify_rejection_requires_error_flag():
    assert classify({}, None, {"is_error": False, "content": "the user rejected nothing"}).answer == "unknown"


def test_read_records_skips_garbage(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"kind": "decision"}\nnot json\n[1]\n')
    assert read_records(p) == [{"kind": "decision"}]


def test_psi_and_ks_detect_shift_and_not_noise():
    rng = np.random.default_rng(0)
    ref = score_quantiles(rng.beta(2, 8, 5000))
    same = rng.beta(2, 8, 2000)
    shifted = rng.beta(5, 3, 2000)
    assert psi_against_quantiles(same, ref) < 0.05 and ks_against_quantiles(same, ref) < 0.05
    assert psi_against_quantiles(shifted, ref) > 1.0 and ks_against_quantiles(shifted, ref) > 0.5


def test_psi_handles_tied_reference():
    ref = score_quantiles(np.r_[np.zeros(600), np.linspace(0.1, 0.9, 400)])  # many exact zeros (fast path)
    same = np.r_[np.zeros(300), np.linspace(0.1, 0.9, 200)]
    assert psi_against_quantiles(same, ref) < 0.1
    assert np.isnan(psi_against_quantiles(np.array([]), ref))


def test_parse_window():
    assert parse_window("6h") == timedelta(hours=6) and parse_window("30m") == timedelta(minutes=30)
    with pytest.raises(ValueError):
        parse_window("1w")


def _policy(rng):
    return {
        "policy_id": "pol1",
        "calibration_stats": {
            "ask_rate_benign": 0.05,
            "false_deny": 0.02,
            "score_reference": {"benign": score_quantiles(rng.beta(2, 8, 3000))},
        },
    }


def _decisions(rng, n, beta_args, ask_share=0.03, start=None):
    start = start or datetime(2026, 10, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        d = "ask" if rng.random() < ask_share else "allow"
        rec = new_record(
            tool_name="Bash", decision=d, p_harm=float(rng.beta(*beta_args)), policy_id="pol1", session_id="s"
        )
        rec["ts"] = (start + timedelta(hours=i)).isoformat()
        out.append(rec)
    return out


def test_monitor_quiet_when_traffic_matches_calibration():
    rng = np.random.default_rng(1)
    rep = monitor(_decisions(rng, 300, (2, 8)), _policy(rng), window="1d")
    assert not rep.alert, rep.alerts
    assert len(rep.windows) == 13 and rep.drift["psi_vs_benign"] < 0.1


def test_monitor_alerts_on_drift_and_ask_surge():
    rng = np.random.default_rng(2)
    rep = monitor(_decisions(rng, 300, (5, 3), ask_share=0.4), _policy(rng))
    text = " ".join(rep.alerts)
    assert rep.alert and "PSI" in text and "ask_rate" in text


def test_monitor_without_policy_reference_notes_it():
    rng = np.random.default_rng(3)
    rep = monitor(_decisions(rng, 10, (2, 8)), {"policy_id": "pol1", "calibration_stats": {}})
    assert not rep.alert and any("score_reference" in n for n in rep.notes)
    assert any("at least 50" in n for n in rep.notes)


def test_monitor_with_reference_but_no_decisions_says_so():
    rng = np.random.default_rng(4)
    rep = monitor([], _policy(rng))
    assert any("no scored decisions" in n for n in rep.notes)
    assert not any("score_reference" in n for n in rep.notes)
