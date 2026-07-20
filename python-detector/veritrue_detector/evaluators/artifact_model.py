from __future__ import annotations

import numpy as np

from .base import Evaluator
from ..config import DetectorConfig
from ..models.torch_image_classifier import TorchBinaryImageClassifier
from ..types import EvaluatorResult, Signal


class ArtifactModelEvaluator(Evaluator):
    name = "Visual Artifacts (CNN)"

    def __init__(self, config: DetectorConfig):
        self._config = config
        self._clf = TorchBinaryImageClassifier(model_path=config.torch_efficientnet_path, name="efficientnet_artifacts")

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        if not self._clf.is_available:
            return EvaluatorResult(
                evaluator=self.name,
                score_0_100=50.0,
                signals=[
                    Signal(
                        name="cnn_artifacts",
                        score_0_100=50.0,
                        details={"status": "model not configured"},
                    )
                ],
            )

        prob_ai = self._clf.predict_proba_ai(image_rgb)
        score = float(np.clip(prob_ai * 100.0, 0, 100))

        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=score,
            signals=[Signal(name="cnn_artifacts", score_0_100=score, details={"prob_ai": prob_ai})],
        )
