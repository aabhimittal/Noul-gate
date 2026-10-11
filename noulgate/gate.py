"""The gate: FeatureBuilder -> DecisionBackend -> combiner -> policy -> Decision (SPEC.md §5)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .backends.base import Answers, DecisionBackend
from .features.builder import GateState
from .policy.combiner import Combiner, predicted_class
from .policy.conformal import ConformalPolicy, ThresholdPolicy
from .questions import QuestionSet
from .schema import Decision


@dataclass
class GateResult:
    decision: Decision
    p_harm: float
    risk_class: str
    answers: Answers
    reason: str
    latency_ms: float
    backend: str
    policy_id: str
    untrusted_only: list[str] = field(default_factory=list)

    def explain(self) -> str:
        parts = [f"NOUL-GATE {self.decision.value}: {self.reason}", f"risk_class={self.risk_class}"]
        ur = self.answers.get("user_requested", {}).get("noul")
        if ur is not None:
            parts.append(f"P(user_requested)={ur:.2f}")
        if self.untrusted_only:
            parts.append("targets seen only in untrusted content: " + ", ".join(self.untrusted_only[:5]))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "p_harm": self.p_harm,
            "risk_class": self.risk_class,
            "answers": self.answers,
            "reason": self.explain(),
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "policy_id": self.policy_id,
        }


class Gate:
    def __init__(
        self,
        backend: DecisionBackend,
        questions: QuestionSet,
        policy: ConformalPolicy | ThresholdPolicy | None = None,
    ):
        self.backend = backend
        self.questions = questions
        self.policy = policy or ThresholdPolicy(0.5)
        self.combiner = getattr(self.policy, "combiner", None) or Combiner(questions.risk_classes)

    def decide_batch(self, states: list[GateState]) -> list[GateResult]:
        t = time.perf_counter()
        all_answers = self.backend.decide_batch(states, self.questions)
        per_call_ms = (time.perf_counter() - t) * 1000 / max(len(states), 1)
        out = []
        for state, answers in zip(states, all_answers, strict=True):
            p = self.combiner.score(answers, bool(state.untrusted_only))
            cls = predicted_class(answers)
            decision, reason = self.policy.decide(p, cls)
            out.append(
                GateResult(
                    decision,
                    p,
                    cls,
                    answers,
                    reason,
                    per_call_ms,
                    self.backend.name,
                    self.policy.policy_id,
                    list(state.untrusted_only),
                )
            )
        return out

    def decide(self, state: GateState) -> GateResult:
        return self.decide_batch([state])[0]
