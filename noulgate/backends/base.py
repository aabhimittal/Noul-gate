"""Backend interface. Answers use the Decision Index shape returned by d1 and Jev:

noul   -> {"type": "noul", "noul": P(yes)}
choice -> {"type": "choice", "choice": name, "confidence": p, "probabilities": {name: p}}
score  -> {"type": "score", "score": expected 0-indexed level, "probabilities": {...},
           "scaled": expected level mapped onto the question's scale}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..features.builder import GateState
from ..questions import Question, QuestionSet

Answers = dict[str, dict[str, Any]]


class BackendError(RuntimeError):
    pass


def add_scaled(q: Question, answer: dict[str, Any]) -> dict[str, Any]:
    if q.type == "score" and "score" in answer:
        n = len(q.criteria)
        lo, hi = q.scale or (0.0, float(n - 1))
        answer["scaled"] = lo + float(answer["score"]) * (hi - lo) / max(n - 1, 1)
    return answer


def choice_answer(probs: dict[str, float]) -> dict[str, Any]:
    total = sum(probs.values()) or 1.0
    probs = {k: max(0.0, v) / total for k, v in probs.items()}
    best = max(probs, key=probs.get)
    return {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}


def score_answer(level_probs: list[float]) -> dict[str, Any]:
    total = sum(level_probs) or 1.0
    ps = [max(0.0, p) / total for p in level_probs]
    best = max(range(len(ps)), key=ps.__getitem__)
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(ps)),
        "confidence": ps[best],
        "probabilities": {str(i): p for i, p in enumerate(ps)},
    }


class DecisionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def decide_batch(self, states: list[GateState], questions: QuestionSet) -> list[Answers]:
        """Answer every question for every state. Must preserve order."""

    def decide(self, state: GateState, questions: QuestionSet) -> Answers:
        return self.decide_batch([state], questions)[0]

    def warmup(self, questions: QuestionSet) -> None:  # noqa: B027 - optional hook
        pass

    def finalize(self, raw: Answers, questions: QuestionSet) -> Answers:
        missing = [n for n in questions.questions if n not in raw]
        if missing:
            raise BackendError(f"{self.name}: no answer for {missing}")
        return {n: add_scaled(questions[n], dict(raw[n])) for n in questions.questions}
