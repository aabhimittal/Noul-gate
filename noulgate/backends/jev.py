"""TypeSafe Jev via its hosted API (docs.typesafe.ai, checked 2026-10-11).

POST https://api.typesafe.ai/v1/systemone   Authorization: Bearer $TYPESAFE_API_KEY
body {"state": ..., "model": "jev-latest", "questions": {...}} -> {"model", "answers", "usage"}
Questions and answers use the same Decision Index schema as d1.

Unlike d1 this is a remote service: every pending action (including file contents and commands) is
sent to a third party, and TypeSafe quotes 70-500 ms end to end. Use it knowingly.
"""

from __future__ import annotations

import os

import httpx

from ..features.builder import GateState
from ..questions import QuestionSet
from .base import Answers, BackendError, DecisionBackend

JEV_URL = "https://api.typesafe.ai/v1/systemone"


class JevBackend(DecisionBackend):
    name = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",
        url: str = JEV_URL,
        timeout: float = 5.0,
        budget_chars: int = 6000,
        harden: bool = True,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.model, self.url, self.budget_chars, self.harden = model, url, budget_chars, harden
        self.name = "jev" + ("+hardened" if harden else "")
        self.client = client or httpx.Client(timeout=timeout)

    def decide_batch(self, states: list[GateState], questions: QuestionSet) -> list[Answers]:
        if not self.api_key:
            raise BackendError("jev backend needs TYPESAFE_API_KEY")
        out = []
        for s in states:  # the public API takes one state per request
            resp = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "state": s.to_model_state(self.budget_chars, harden=self.harden),
                    "model": self.model,
                    "questions": questions.to_d1(),
                },
            )
            if resp.status_code != 200:
                raise BackendError(f"jev HTTP {resp.status_code}: {resp.text[:200]}")
            out.append(self.finalize(resp.json()["answers"], questions))
        return out
