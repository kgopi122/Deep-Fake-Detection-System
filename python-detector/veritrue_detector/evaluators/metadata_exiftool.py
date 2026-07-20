from __future__ import annotations

import json
import subprocess
from typing import Any

import numpy as np

from .base import Evaluator
from ..types import EvaluatorResult, Signal


_GENERATOR_TAGS = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "comfyui",
    "automatic1111",
    "sdxl",
    "firefly",
    "runway",
    "leonardo",
]

_CAMERA_PARAMS = [
    "Make",
    "Model",
    "LensModel",
    "FNumber",
    "ExposureTime",
    "ISO",
    "FocalLength",
]


class MetadataExifToolEvaluator(Evaluator):
    name = "Metadata (ExifTool)"

    def __init__(self, exiftool_path: str | None = None):
        self._exiftool = exiftool_path or "exiftool"

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        tags = self._read_tags(image_path)

        text_blob = json.dumps(tags, ensure_ascii=False).lower()

        generator_hits = [t for t in _GENERATOR_TAGS if t in text_blob]
        missing_params = [p for p in _CAMERA_PARAMS if p not in tags]

        score = 50.0
        signals: list[Signal] = []

        if generator_hits:
            score = max(score, 85.0)
            signals.append(
                Signal(
                    name="generator_tags",
                    score_0_100=90.0,
                    details={"hits": generator_hits},
                )
            )

        # Missing camera parameters is a *risk* signal, but images can be stripped.
        if len(missing_params) >= 5:
            score = max(score, 70.0)
        elif len(missing_params) >= 3:
            score = max(score, 60.0)

        signals.append(
            Signal(
                name="camera_param_completeness",
                score_0_100=float(100.0 - (len(missing_params) / len(_CAMERA_PARAMS)) * 100.0),
                details={"missing": missing_params},
            )
        )

        return EvaluatorResult(evaluator=self.name, score_0_100=float(score), signals=signals)

    def _read_tags(self, image_path: str) -> dict[str, Any]:
        try:
            # -j: JSON, -n: numeric where possible, -G1: group names, -a: allow duplicates
            proc = subprocess.run(
                [self._exiftool, "-j", "-n", "-G1", "-a", image_path],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return {}
            parsed = json.loads(proc.stdout)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
            return {}
        except Exception:
            return {}
