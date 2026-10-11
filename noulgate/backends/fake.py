"""Deterministic backend for tests and for exercising serve / proxy without a model."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..features.builder import GateState
from ..questions import QuestionSet
from .base import Answers, DecisionBackend, choice_answer, score_answer


class FakeBackend(DecisionBackend):
    """p_benign comes from `fn(state)` (default: 0.05 if the command mentions 'evil', else 0.95)."""

    name = "fake"

    def __init__(self, fn: Callable[[GateState], float] | None = None, delay_s: float = 0.0):
        self.fn = fn or (lambda s: 0.05 if "evil" in str(s.args) else 0.95)
        self.delay_s = delay_s

    def decide_batch(self, states: list[GateState], questions: QuestionSet) -> list[Answers]:
        if self.delay_s:
            time.sleep(self.delay_s)
        out = []
        for s in states:
            pb = float(self.fn(s))
            ans: Answers = {}
            for name, q in questions.questions.items():
                if q.type == "noul":
                    ans[name] = {"type": "noul", "noul": pb}
                elif q.type == "choice":
                    rest = [k for k in q.criteria if k != "benign"]
                    probs = {k: (1 - pb) / max(len(rest), 1) for k in rest}
                    if "benign" in q.criteria:
                        probs["benign"] = pb
                    ans[name] = choice_answer(probs)
                else:
                    n = len(q.criteria)
                    ans[name] = score_answer(
                        [1.0 if i == round((1 - pb) * (n - 1)) else 0.0 for i in range(n)]
                    )
            out.append(self.finalize(ans, questions))
        return out
