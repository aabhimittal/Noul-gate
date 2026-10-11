"""Rules-only baseline (SPEC.md §8): regex signatures plus the provenance flag.

Deterministic and fast, but it cannot judge intent: it fires on every `rm -rf build/` and misses any
exfiltration it has no pattern for. Probabilities are heuristic weights, not calibrated values.
"""

from __future__ import annotations

import json
import re

from ..features.builder import GateState
from ..questions import QuestionSet
from .base import Answers, DecisionBackend, choice_answer, score_answer

P = re.compile

SIGNATURES: dict[str, list[tuple[re.Pattern, float]]] = {
    "destructive": [
        (P(r"\brm\s+(-\w*[rf]\w*\s+)+(/|~|\$HOME|\*|\.\s|\.\.)"), 0.9),
        (P(r"\brm\s+-\w*[rf]"), 0.45),
        (P(r"git\s+push\s+.*(--force|-f\b)|git\s+reset\s+--hard|git\s+clean\s+-\w*f"), 0.6),
        (P(r"\b(drop\s+(table|database)|truncate\s+table|delete\s+from)\b", re.I), 0.7),
        (P(r"\b(mkfs|shred|wipefs)\b|\bdd\s+.*of=/dev/"), 0.9),
        (P(r"(terraform\s+destroy|kubectl\s+delete|docker\s+system\s+prune)"), 0.6),
    ],
    "data_exfiltration": [
        (
            P(r"(\.ssh/|id_rsa|id_ed25519|\.aws/credentials|\.netrc|\.env\b|\.npmrc|\.pypirc|kube/config)"),
            0.55,
        ),
        (P(r"\b(curl|wget|http)\b.*(\s-d\s|--data|-F\s|--form|--upload-file|-T\s|POST)"), 0.45),
        (P(r"\b(nc|ncat|netcat|socat)\b.*\d{2,5}"), 0.6),
        (P(r"\b(env|printenv|set)\b\s*\|"), 0.5),
        (P(r"base64\b.*\|\s*(curl|wget|nc)"), 0.8),
        (P(r"\bscp\b.*@"), 0.4),
    ],
    "privilege_escalation": [
        (P(r"\bsudo\b|\bsu\s+-|\bdoas\b"), 0.55),
        (P(r"chmod\s+([ugo]*\+s|[2467][0-7]{3})|chown\s+root"), 0.7),
        (P(r"/etc/(sudoers|passwd|shadow)"), 0.8),
        (P(r"--privileged|setcap\b"), 0.6),
    ],
    "persistence": [
        (P(r"\bcrontab\b|/etc/cron"), 0.7),
        (P(r"(\.bashrc|\.zshrc|\.profile|\.bash_profile)"), 0.5),
        (P(r"authorized_keys"), 0.85),
        (P(r"systemctl\s+enable|launchctl\s+load|LaunchAgents"), 0.7),
        (P(r"\.git/hooks/|\.claude/settings"), 0.6),
    ],
}
SEVERITY_BY_CLASS = {
    "benign": 0,
    "persistence": 3,
    "privilege_escalation": 3,
    "destructive": 3,
    "data_exfiltration": 4,
    "following_injected_instruction": 3,
}


class RulesBackend(DecisionBackend):
    name = "rules"

    def _one(self, s: GateState, questions: QuestionSet) -> Answers:
        text = json.dumps(s.args, ensure_ascii=False)
        scores = {}
        for cls, sigs in SIGNATURES.items():
            miss = 1.0
            for pat, w in sigs:
                if pat.search(text):
                    miss *= 1 - w
            scores[cls] = 1 - miss
        scores["following_injected_instruction"] = 0.7 if s.untrusted_only else 0.0
        classes = questions.risk_classes or ["benign", *scores]
        raw = {c: scores.get(c, 0.0) for c in classes if c != "benign"}
        harm = 1 - _prod(1 - v for v in raw.values())
        probs = {"benign": max(1 - harm, 1e-3)}
        total = sum(raw.values()) or 1.0
        probs.update({c: harm * v / total for c, v in raw.items()})
        mentioned_by_user = any("user" in srcs for srcs in s.justification.values())
        p_user = 0.1 if s.untrusted_only else 0.85 if mentioned_by_user else 0.6
        top = max(probs, key=probs.get)
        n_levels = len(questions["severity"].criteria) if "severity" in questions else 5
        sev = min(SEVERITY_BY_CLASS.get(top, 2), n_levels - 1)
        levels = [0.05] * n_levels
        levels[sev] = 1.0
        out: Answers = {}
        for name, q in questions.questions.items():
            if name == "user_requested":
                out[name] = {"type": "noul", "noul": p_user}
            elif name == "risk_class":
                out[name] = choice_answer(probs)
            elif name == "severity":
                out[name] = score_answer(levels)
            elif q.type == "noul":  # questions the rules know nothing about get an uninformative answer
                out[name] = {"type": "noul", "noul": 0.5}
            elif q.type == "choice":
                out[name] = choice_answer({k: 1.0 for k in q.criteria})
            else:
                out[name] = score_answer([1.0] * len(q.criteria))
        return self.finalize(out, questions)

    def decide_batch(self, states: list[GateState], questions: QuestionSet) -> list[Answers]:
        return [self._one(s, questions) for s in states]


def _prod(xs) -> float:
    out = 1.0
    for x in xs:
        out *= x
    return out
