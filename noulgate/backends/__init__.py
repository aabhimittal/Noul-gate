"""Decision backends: answer the question set about a pending action."""

from __future__ import annotations

from .base import DecisionBackend


def get_backend(name: str, **kwargs) -> DecisionBackend:
    """Backends are imported lazily so optional heavy dependencies stay optional."""
    if name in {"d1", "d1-3b"}:
        from .d1 import D1Backend

        return D1Backend(**kwargs)
    if name == "jev":
        from .jev import JevBackend

        return JevBackend(**kwargs)
    if name in {"llm", "haiku"}:
        from .llm import LLMJudgeBackend

        return LLMJudgeBackend(**kwargs)
    if name == "rules":
        from .rules import RulesBackend

        return RulesBackend(**kwargs)
    if name == "fake":
        from .fake import FakeBackend

        return FakeBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (d1-3b, jev, llm, rules, fake)")


__all__ = ["DecisionBackend", "get_backend"]
