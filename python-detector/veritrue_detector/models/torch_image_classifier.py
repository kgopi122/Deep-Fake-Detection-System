from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TorchBinaryImageClassifier:
    model_path: str | None
    name: str

    def __post_init__(self) -> None:
        self._model = None
        self._torch = None
        self._available = False

        if not self.model_path:
            return

        try:
            import torch
            import torchvision.transforms as T

            self._torch = torch
            self._T = T

            # Expecting a TorchScript model that returns either a single logit/prob or 2-class logits.
            self._model = torch.jit.load(self.model_path, map_location="cpu")
            self._model.eval()
            self._available = True
        except Exception:
            self._available = False

    @property
    def is_available(self) -> bool:
        return bool(self._available)

    def predict_proba_ai(self, image_rgb: np.ndarray) -> float:
        if not self._available or self._torch is None or self._model is None:
            return 0.5

        torch = self._torch
        img = image_rgb

        # Basic preprocessing (customize to match training)
        img = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1)

        # Resize/crop to 224
        t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
        t = t.squeeze(0)

        # Normalize to ImageNet by default
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        t = (t - mean) / std

        with torch.no_grad():
            out = self._model(t.unsqueeze(0))

        if isinstance(out, (tuple, list)):
            out = out[0]

        out = out.squeeze()
        if out.ndim == 0:
            # single logit
            prob = torch.sigmoid(out).item()
            return float(prob)

        if out.numel() == 1:
            prob = torch.sigmoid(out.view(()))
            return float(prob.item())

        # 2-class logits assumed: [real, ai] or [ai, real] unknown. Assume index 1 = AI.
        if out.numel() >= 2:
            logits = out.flatten()[:2]
            prob = torch.softmax(logits, dim=0)[1].item()
            return float(prob)

        return 0.5
