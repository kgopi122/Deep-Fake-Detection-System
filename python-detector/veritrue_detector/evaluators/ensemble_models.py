from __future__ import annotations

import numpy as np

from .base import Evaluator
from ..config import DetectorConfig
from ..models.torch_image_classifier import TorchBinaryImageClassifier
from ..models.stacker import load_stacker
from ..types import EvaluatorResult, Signal


class EnsembleModelsEvaluator(Evaluator):
    name = "Detector Ensemble (EffNet/ViT/CLIP)"

    def __init__(self, config: DetectorConfig):
        self._config = config
        self._eff = TorchBinaryImageClassifier(model_path=config.torch_efficientnet_path, name="efficientnet")
        self._vit = TorchBinaryImageClassifier(model_path=config.torch_vit_path, name="vit")
        self._clip = TorchBinaryImageClassifier(model_path=config.torch_clip_path, name="clip")
        self._stacker = load_stacker(config.stacker_path)

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        probs: dict[str, float] = {}

        if self._eff.is_available:
            probs["efficientnet"] = float(self._eff.predict_proba_ai(image_rgb))
        if self._vit.is_available:
            probs["vit"] = float(self._vit.predict_proba_ai(image_rgb))
        if self._clip.is_available:
            probs["clip"] = float(self._clip.predict_proba_ai(image_rgb))

        if not probs:
            return EvaluatorResult(
                evaluator=self.name,
                score_0_100=50.0,
                signals=[
                    Signal(
                        name="ensemble",
                        score_0_100=50.0,
                        details={"status": "no models configured"},
                    )
                ],
            )

        # Soft voting baseline
        soft = float(np.mean(list(probs.values())))

        # Optional stacking model (logistic regression, etc.)
        if self._stacker is not None:
            stacked = float(self._stacker.predict_proba_ai(probs))
            final = stacked
            mode = "stacking"
        else:
            final = soft
            mode = "soft-vote"

        score = float(np.clip(final * 100.0, 0, 100))
        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=score,
            signals=[
                Signal(
                    name="ensemble",
                    score_0_100=score,
                    details={"mode": mode, "probs": probs},
                )
            ],
        )
