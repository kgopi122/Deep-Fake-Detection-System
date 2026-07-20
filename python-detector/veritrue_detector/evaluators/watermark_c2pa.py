from __future__ import annotations

import json
import subprocess

import numpy as np

from .base import Evaluator
from ..types import EvaluatorResult, Signal


class WatermarkC2paEvaluator(Evaluator):
    name = "Watermark/C2PA"

    def __init__(self, c2patool_path: str | None = None):
        self._c2patool = c2patool_path

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        if not self._c2patool:
            return EvaluatorResult(
                evaluator=self.name,
                score_0_100=50.0,
                signals=[
                    Signal(
                        name="c2pa",
                        score_0_100=50.0,
                        details={"status": "disabled (no c2patool configured)"},
                    )
                ],
            )

        info = self._run_c2pa(image_path)
        if not info:
            return EvaluatorResult(
                evaluator=self.name,
                score_0_100=50.0,
                signals=[
                    Signal(
                        name="c2pa",
                        score_0_100=50.0,
                        details={"status": "no credentials detected"},
                    )
                ],
            )

        # If C2PA credentials exist and are valid, it’s a strong “Likely Real” indicator.
        score = 20.0
        return EvaluatorResult(
            evaluator=self.name,
            score_0_100=score,
            signals=[
                Signal(
                    name="c2pa",
                    score_0_100=score,
                    details=info,
                )
            ],
        )

    def _run_c2pa(self, image_path: str) -> dict | None:
        try:
            proc = subprocess.run(
                [self._c2patool, image_path, "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return None
            return json.loads(proc.stdout)
        except Exception:
            return None
