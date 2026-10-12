"""Decision backends: answer the question set about a pending action."""

from __future__ import annotations

from .base import DecisionBackend


def get_backend(name: str, **kwargs) -> DecisionBackend:
    """Backends are imported lazily so optional heavy dependencies stay optional.

    Suffixes switch off a defence to measure what it buys: ``:raw`` (e.g. ``d1-3b:raw``) turns off
    model-state hardening; ``:joint`` (d1 only) asks user_requested in the same call as the other
    questions instead of from the user's words alone. They combine: ``d1-3b:raw:joint``."""
    base, *suffixes = name.split(":")
    for suffix in suffixes:
        if suffix == "raw" and base in {"d1", "d1-3b", "jev", "llm", "haiku"}:
            kwargs["harden"] = False
        elif suffix == "joint" and base in {"d1", "d1-3b"}:
            kwargs["isolate_user_request"] = False
        else:
            raise ValueError(f"unknown suffix :{suffix} for backend {base!r}")
    name = base
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
