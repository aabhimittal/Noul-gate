"""Liquid AI d1-3B, run locally with transformers (huggingface.co/LiquidAI/d1-3B).

Requires `pip install noulgate[d1]` (transformers>=5.14, torch). The model ships its own code and is
loaded with trust_remote_code=True, i.e. it executes Python from the model repo: pin `revision`.
License: LFM Open License v1.0 — check its commercial-use terms before deploying.
"""

from __future__ import annotations

import os

from ..features.builder import GateState
from ..questions import QuestionSet
from .base import Answers, BackendError, DecisionBackend

MODEL_ID = "LiquidAI/d1-3B"
# Snapshot verified on 2026-10-11. Override with NOULGATE_D1_REVISION after reviewing the new code.
PINNED_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"


class D1Backend(DecisionBackend):
    name = "d1-3b"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        compile: bool = False,
        budget_chars: int = 6000,
    ):
        self.model_id = model_id
        self.revision = revision or os.environ.get("NOULGATE_D1_REVISION", PINNED_REVISION)
        self.device = device
        self.compile = compile
        self.budget_chars = budget_chars
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BackendError("d1 backend needs `pip install noulgate[d1]`") from exc
        device = self.device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, dtype=dtype, revision=self.revision
        ).to(device)
        if self.compile and device == "cuda":
            model.compile(mode="reduce-overhead")
        self.device, self._model = device, model
        return model

    def warmup(self, questions: QuestionSet) -> None:
        model = self._load()
        model.system_one({"pending_action": {"tool": "Bash", "args": {"command": "ls"}}}, questions.to_d1())

    def decide_batch(self, states: list[GateState], questions: QuestionSet) -> list[Answers]:
        model = self._load()
        qs = questions.to_d1()
        requests = [(s.to_model_state(self.budget_chars), qs) for s in states]
        results = model.system_one_batch(requests) if len(requests) > 1 else [model.system_one(*requests[0])]
        return [self.finalize(r["answers"], questions) for r in results]
