from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import DetectorConfig
from .types import EvaluatorResult, Verdict
from .utils.image_io import load_image_rgb

from .evaluators.metadata_exiftool import MetadataExifToolEvaluator
from .evaluators.modern_phone_detector import ModernPhoneDetector
from .scoring import combine_scores


@dataclass(frozen=True)
class DetectionReport:
    ai_likelihood_0_100: int
    verdict: Verdict
    components: list[dict[str, Any]]
    explanation: list[str]
    disclaimer: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def detect_image(image_path: str, config: DetectorConfig | None = None) -> DetectionReport:
    config = config or DetectorConfig()

    img = load_image_rgb(image_path, max_side=config.max_image_side)

    evaluators = [
        ModernPhoneDetector(config),  # Modern phone camera detector
        MetadataExifToolEvaluator(config.exiftool_path),
    ]

    results: list[EvaluatorResult] = []
    for ev in evaluators:
        try:
            results.append(ev.evaluate(img, image_path))
        except Exception as e:  # keep pipeline resilient
            results.append(
                EvaluatorResult(
                    evaluator=ev.name,
                    score_0_100=50.0,
                    signals=[],
                )
            )

    final_score = combine_scores(results, config)

    verdict: Verdict
    if final_score <= config.thresholds.likely_real_max:
        verdict = "Likely Real"
    elif final_score >= config.thresholds.likely_ai_min:
        verdict = "Likely AI-Generated"
    else:
        verdict = "Uncertain"

    components = [
        {
            "evaluator": r.evaluator,
            "score_0_100": float(np.clip(r.score_0_100, 0, 100)),
            "signals": [asdict(s) for s in r.signals],
        }
        for r in results
    ]

    explanation = _build_explanation(results, final_score)

    return DetectionReport(
        ai_likelihood_0_100=int(round(final_score)),
        verdict=verdict,
        components=components,
        explanation=explanation,
        disclaimer=(
            "This result is probabilistic and may be wrong. "
            "Use it as one input to a broader verification workflow."
        ),
    )


def _build_explanation(results: list[EvaluatorResult], final_score: float) -> list[str]:
    # Top contributing “reasons” by distance from neutral 50.
    ranked = []
    for r in results:
        ranked.append((abs(r.score_0_100 - 50.0), r))
    ranked.sort(key=lambda x: x[0], reverse=True)

    lines: list[str] = [f"Final AI-likelihood: {int(round(final_score))}/100"]
    for _, r in ranked[:4]:
        direction = "higher" if r.score_0_100 >= 50 else "lower"
        lines.append(f"{r.evaluator}: {r.score_0_100:.1f}/100 ({direction} AI risk)")
    return lines
