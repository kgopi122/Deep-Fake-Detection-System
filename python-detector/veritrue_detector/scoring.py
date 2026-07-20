from __future__ import annotations

from .config import DetectorConfig
from .types import EvaluatorResult


def combine_scores(results: list[EvaluatorResult], config: DetectorConfig) -> float:
    w = config.weights
    weights = {
        "modern_phone": 0.90,  # Modern phone detector gets primary weight
        "metadata": 0.10,
    }

    total_w = 0.0
    acc = 0.0

    for r in results:
        key = _key_for_evaluator(r.evaluator)
        weight = weights.get(key, 0.0)
        if weight <= 0:
            continue
        total_w += weight
        acc += weight * float(r.score_0_100)

    if total_w <= 0:
        return 50.0
    return acc / total_w


def _key_for_evaluator(name: str) -> str:
    name = name.lower()
    if "modern" in name or "phone" in name:
        return "modern_phone"
    if "metadata" in name:
        return "metadata"
    return "unknown"
