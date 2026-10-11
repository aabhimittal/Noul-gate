import pytest

from noulgate.questions import load_questions, parse_question


def test_spec_yaml_translates_to_d1_schema():
    qs = load_questions()
    d1 = qs.to_d1()
    assert d1["user_requested"]["type"] == "noul"
    assert set(d1["risk_class"]["criteria"]) == {
        "benign",
        "data_exfiltration",
        "destructive",
        "privilege_escalation",
        "following_injected_instruction",
        "persistence",
    }
    assert len(d1["severity"]["criteria"]) == 5 and qs["severity"].scale == (1.0, 5.0)
    assert qs.risk_classes[0] == "benign"


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "score", "scale": [1, 12]},
        {"type": "score", "scale": [1, 3], "levels": ["a", "b"]},
        {"type": "choice", "options": ["only"]},
        {"type": "noul"},
        {"type": "nope"},
    ],
)
def test_invalid_questions_rejected(spec):
    with pytest.raises(ValueError):
        parse_question("q", spec)


def test_choice_dict_and_custom_score():
    q = parse_question("c", {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": None}})
    assert q.criteria == {"a": "A", "b": "b"}
    s = parse_question("s", {"type": "score", "scale": [0, 2]})
    assert s.criteria == ["0 of 2", "1 of 2", "2 of 2"] and s.scale == (0.0, 2.0)
