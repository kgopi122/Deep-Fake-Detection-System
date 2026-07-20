from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Signal:
    name: str
    score_0_100: float
    details: dict[str, Any]


@dataclass(frozen=True)
class EvaluatorResult:
    evaluator: str
    score_0_100: float
    signals: list[Signal]


Verdict = Literal["Likely Real", "Uncertain", "Likely AI-Generated"]
