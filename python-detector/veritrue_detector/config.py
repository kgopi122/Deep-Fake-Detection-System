from __future__ import annotations

from pydantic import BaseModel, Field


class Weights(BaseModel):
    advanced_ensemble: float = 0.40  # Primary detector
    metadata: float = 0.15
    artifact: float = 0.20
    signal: float = 0.15
    ensemble: float = 0.10
    originality: float = 0.00  # Disabled for now
    watermark: float = 0.00


class Thresholds(BaseModel):
    likely_real_max: int = 25  # More strict
    likely_ai_min: int = 75    # More strict


class DetectorConfig(BaseModel):
    # If model evaluators are unavailable, pipeline still runs using other evaluators.
    weights: Weights = Field(default_factory=Weights)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    exiftool_path: str | None = None
    c2patool_path: str | None = None

    # Enhanced model paths
    torch_efficientnet_path: str | None = None
    torch_vit_path: str | None = None
    torch_clip_path: str | None = None
    
    # New ONNX models for advanced detection
    frequency_detector_path: str | None = None
    temporal_consistency_path: str | None = None
    face_geometry_path: str | None = None
    biological_detector_path: str | None = None

    # Optional calibrator/stacker (sklearn joblib)
    stacker_path: str | None = None

    # Originality DB (sqlite) for pHash index
    originality_db_path: str | None = None

    # Operational controls
    max_image_side: int = 2048
