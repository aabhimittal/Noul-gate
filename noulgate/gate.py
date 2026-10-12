"""The gate: FeatureBuilder -> DecisionBackend -> combiner -> policy -> Decision (SPEC.md §5)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .backends.base import Answers, DecisionBackend
from .features.builder import GateState
from .policy.combiner import Combiner, predicted_class
from .policy.conformal import ConformalPolicy, ThresholdPolicy
from .policy.fastpath import FASTPATH_CLASS, fast_answers, fast_path
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
    taint: str = ""

    def explain(self) -> str:
        parts = [f"NOUL-GATE {self.decision.value}: {self.reason}", f"risk_class={self.risk_class}"]
        ur = self.answers.get("user_requested", {}).get("noul")
        if ur is not None:
            parts.append(f"P(user_requested)={ur:.2f}")
        if self.untrusted_only:
            parts.append("targets seen only in untrusted content: " + ", ".join(self.untrusted_only[:5]))
        if self.taint:
            parts.append(self.taint)
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
        use_fast = getattr(self.policy, "fastpath", False)
        fast = [fast_path(s) if use_fast else None for s in states]
        model_states = [s for s, f in zip(states, fast, strict=True) if not f]
        model_answers = iter(self.backend.decide_batch(model_states, self.questions) if model_states else [])
        per_call_ms = (time.perf_counter() - t) * 1000 / max(len(model_states), 1)
        out = []
        for state, reason in zip(states, fast, strict=True):
            if reason:
                answers, p, cls, ms = fast_answers(reason), 0.0, FASTPATH_CLASS, 0.0
            else:
                answers = next(model_answers)
                p, cls, ms = self.combiner.score(answers, state.flags), predicted_class(answers), per_call_ms
            decision, why = self.policy.decide(p, cls, state.flags)
            if reason:
                why = f"fast path ({reason}); {why}"
            out.append(
                GateResult(
                    decision,
                    p,
                    cls,
                    answers,
                    why,
                    ms,
                    "fastpath" if reason else self.backend.name,
                    self.policy.policy_id,
                    list(state.untrusted_only),
                    _taint_note(state),
                )
            )
        return out

    def decide(self, state: GateState) -> GateResult:
        return self.decide_batch([state])[0]


def _taint_note(state: GateState) -> str:
    if state.tainted_egress:
        return f"sends data out ({state.egress_kind}) after this session touched secrets: " + ", ".join(
            state.taint_sources[:3]
        )
    if state.secret_in_args:
        return "call references secrets: " + ", ".join(state.secret_in_args[:3])
    return ""
