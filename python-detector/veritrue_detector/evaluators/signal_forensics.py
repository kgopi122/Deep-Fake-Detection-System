from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .base import Evaluator
from ..types import EvaluatorResult, Signal


class SignalForensicsEvaluator(Evaluator):
    name = "Signal/Noise Forensics"

    def evaluate(self, image_rgb: np.ndarray, image_path: str) -> EvaluatorResult:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # Noise residual proxy: high-pass via median blur then subtract.
        med = cv2.medianBlur((gray * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
        residual = gray - med
        residual_std = float(np.std(residual))

        # Frequency domain: radial spectrum stats
        hf_ratio, spectral_flatness = _fft_spectrum_features(gray)

        # Simple “blockiness” proxy (JPEG-like grid artifacts)
        blockiness = _blockiness_proxy(gray)

        score = 50.0
        signals: list[Signal] = []

        # Diffusion images often have unusually smooth residuals + atypical HF distribution.
        # These are heuristics; keep them moderate and explainable.
        if residual_std < 0.012 and hf_ratio < 0.14:
            score = max(score, 78.0)
        elif residual_std < 0.015:
            score = max(score, 65.0)

        if spectral_flatness > 0.62:
            score = max(score, 70.0)

        # Extreme blockiness can mean heavy compression, which confounds analysis.
        # Treat as uncertainty rather than “AI”.
        if blockiness > 0.035:
            score = (score * 0.85) + 7.5

        signals.append(
            Signal(
                name="noise_residual",
                score_0_100=_map_residual_to_risk(residual_std),
                details={"residual_std": residual_std},
            )
        )
        signals.append(
            Signal(
                name="frequency_domain",
                score_0_100=_map_hf_to_risk(hf_ratio, spectral_flatness),
                details={"hf_ratio": hf_ratio, "spectral_flatness": spectral_flatness},
            )
        )
        signals.append(
            Signal(
                name="compression_blockiness",
                score_0_100=float(np.clip(blockiness * 2500.0, 0, 100)),
                details={"blockiness": blockiness},
            )
        )

        return EvaluatorResult(evaluator=self.name, score_0_100=float(np.clip(score, 0, 100)), signals=signals)


def _fft_spectrum_features(gray01: np.ndarray) -> tuple[float, float]:
    h, w = gray01.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    x = (gray01 - float(np.mean(gray01))) * win

    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F) + 1e-12
    logmag = np.log(mag)

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (np.max(r) + 1e-12)

    # High-frequency energy ratio: outer 30% radius / total
    hf_mask = r_norm >= 0.70
    hf_ratio = float(np.sum(mag[hf_mask]) / np.sum(mag))

    # Spectral flatness: exp(mean(log)) / mean
    flatness = float(math.exp(float(np.mean(logmag))) / float(np.mean(mag)))

    return hf_ratio, float(np.clip(flatness, 0.0, 1.5))


def _blockiness_proxy(gray01: np.ndarray) -> float:
    # Measures mean absolute differences across 8px boundaries vs overall.
    x = gray01
    if x.shape[0] < 32 or x.shape[1] < 32:
        return 0.0

    diffs_h = np.abs(x[:, 8::8] - x[:, 7::8])
    diffs_v = np.abs(x[8::8, :] - x[7::8, :])

    boundary = float((np.mean(diffs_h) + np.mean(diffs_v)) * 0.5)
    overall = float(np.mean(np.abs(np.diff(x, axis=1))) + np.mean(np.abs(np.diff(x, axis=0)))) * 0.5

    if overall <= 1e-9:
        return 0.0
    return float(np.clip(boundary / overall, 0.0, 1.0))


def _map_residual_to_risk(residual_std: float) -> float:
    # Lower residual variance => smoother => higher AI risk.
    if residual_std <= 0.010:
        return 85.0
    if residual_std <= 0.013:
        return 70.0
    if residual_std <= 0.017:
        return 55.0
    return 40.0


def _map_hf_to_risk(hf_ratio: float, flatness: float) -> float:
    risk = 50.0
    if hf_ratio < 0.12:
        risk += 18.0
    elif hf_ratio < 0.15:
        risk += 8.0

    if flatness > 0.62:
        risk += 12.0
    return float(np.clip(risk, 0.0, 100.0))
