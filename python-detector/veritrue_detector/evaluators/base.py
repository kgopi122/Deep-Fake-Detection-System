from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import EvaluatorResult


class Evaluator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        raise NotImplementedError
