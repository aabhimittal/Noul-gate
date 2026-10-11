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


def _cpu_has_bf16() -> bool:
    """AVX512-BF16 / AMX CPUs run d1 in bfloat16 ~20x faster than float32 (measured: 1.25 s vs 29 s)."""
    try:
        flags = open("/proc/cpuinfo").read()
    except OSError:
        return False
    return "avx512_bf16" in flags or "amx_bf16" in flags


class D1Backend(DecisionBackend):
    name = "d1-3b"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        compile: bool = False,
        budget_chars: int = 6000,
        dtype: str | None = None,
    ):
        self.model_id = model_id
        self.revision = revision or os.environ.get("NOULGATE_D1_REVISION", PINNED_REVISION)
        self.device = device
        self.compile = compile
        self.budget_chars = budget_chars
        # The model card uses float32 on CPU. We default to bfloat16 where the CPU supports it natively.
        # Calibrate with the same dtype you deploy: answers shift slightly between precisions.
        self.dtype = dtype or os.environ.get("NOULGATE_D1_DTYPE")
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
        name = self.dtype or ("bfloat16" if device != "cpu" or _cpu_has_bf16() else "float32")
        dtype = getattr(torch, name)
        model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, dtype=dtype, revision=self.revision
        ).to(device)
        if self.compile and device == "cuda":
            model.compile(mode="reduce-overhead")
        self.device, self._model = device, model
        self.name = f"d1-3b[{device},{name}]"
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
